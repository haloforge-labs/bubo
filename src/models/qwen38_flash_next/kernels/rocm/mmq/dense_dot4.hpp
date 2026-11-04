// SPDX-License-Identifier: MIT
// Copyright (c) 2026 gufo contributors
// Copyright (c) 2026 BonkusCheemus (the LDS-staged dot4 GEMM body)
//
// Few-row dense projection, N = 1..8 tokens, on the published ggml block
// layout. The kernel body is SmallBatchDot4LdsGEMMBody from BonkusCheemus/tubo
// (MIT, https://github.com/BonkusCheemus/tubo, HEAD 17232b3ad79e,
// src/models/qwen/hip/kernels/small_batch_dot4.hpp) at the geometry that was
// measured (2 rows per wave, 8 waves, one 16-weight sub-block per lane).
// Changes: the activations are the engine's block_q8_1 rows (one scale per 32
// values) instead of a separate per-16 quantisation, so no extra launch and no
// extra scratch exist; the per-16 sum the minimum term needs is computed while
// staging; weight offsets are 64-bit; the launcher takes the caller's stream
// and returns false for what it does not cover; a non-finite output is stored
// as zero, as the reuse kernel it replaces does.
//
// The weights are decoded by DecodeQuantSub16 (src/models/qwen/hip/
// quant_ops.hpp), included unchanged: this file adds no block decoder.
// Per token, the lane loop and the eight-column butterfly do not depend on N,
// so a row computes to the same bits at every width 1..8.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "dense_dot4.h"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"

namespace gufo::models::qwen38_flash_next::dense_dot4 {

using GgmlType = ::gufo::core::GgmlType;
namespace sub16 = ::gufo::hip;

inline constexpr std::uint32_t kRowsPerWave = 2;
inline constexpr std::uint32_t kWaves = 8;
inline constexpr std::uint32_t kRowsPerBlock = kRowsPerWave * kWaves;
// One staged tile is 32 sub-blocks of 16 K values (one per lane) per token.
inline constexpr std::uint32_t kTileSubs = 32;
// block_q8_1: half2 {d, s} then int8 qs[32].
inline constexpr std::uint32_t kQ8_1Bytes = 36;

template<GgmlType WType, std::size_t Batch>
__launch_bounds__(kWaves * 32) __global__
    void DenseDot4Kernel(const void* __restrict__ w,
                         const std::uint8_t* __restrict__ x,
                         float* __restrict__ y, std::uint32_t m,
                         std::uint32_t k, std::uint32_t x_block_stride) {
  static_assert(Batch >= 1 && Batch <= 8);
  constexpr bool kHasOffset =
      WType == GgmlType::kQ4_K || WType == GgmlType::kQ5_K;
  constexpr std::uint32_t kThreads = kWaves * 32U;
  __shared__ uint4 stage_q[2][Batch * kTileSubs];
  __shared__ float2 stage_ds[2][Batch * kTileSubs];
  const std::uint32_t num_sub = k / 16U;
  // Every wave of the block stages, even one whose rows are past m, so the
  // barriers stay uniform; such a wave just skips the math and the store.
  const auto stage = [&](std::uint32_t buffer, std::uint32_t tile_base) {
    for (std::uint32_t i = threadIdx.x; i < Batch * kTileSubs; i += kThreads) {
      const std::uint32_t t = i / kTileSubs;
      const std::uint32_t sub = tile_base + (i % kTileSubs);
      if (sub < num_sub) {
        // Q8_1 blocks are 36 bytes: the codes are 4-byte aligned words.
        const std::uint8_t* block =
            x + (std::size_t(t) * x_block_stride + sub / 2U) * kQ8_1Bytes;
        const auto* q = reinterpret_cast<const std::uint32_t*>(
            block + 4U + 16U * (sub & 1U));
        const uint4 codes = {q[0], q[1], q[2], q[3]};
        const float d = __half2float(*reinterpret_cast<const __half*>(block));
        float d_sum = 0.0F;
        if constexpr (kHasOffset) {
          int sum = 0;
          sum = __builtin_amdgcn_sudot4(true, int(codes.x), true, 0x01010101,
                                        sum, false);
          sum = __builtin_amdgcn_sudot4(true, int(codes.y), true, 0x01010101,
                                        sum, false);
          sum = __builtin_amdgcn_sudot4(true, int(codes.z), true, 0x01010101,
                                        sum, false);
          sum = __builtin_amdgcn_sudot4(true, int(codes.w), true, 0x01010101,
                                        sum, false);
          d_sum = d * static_cast<float>(sum);
        }
        stage_q[buffer][i] = codes;
        stage_ds[buffer][i] = {d, d_sum};
      } else {
        stage_q[buffer][i] = {0U, 0U, 0U, 0U};
        stage_ds[buffer][i] = {0.0F, 0.0F};
      }
    }
  };
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t row_base =
      ((blockIdx.x * kWaves) + (threadIdx.x >> 5U)) * kRowsPerWave;
  const bool active = row_base < m;
  const std::size_t row_bytes = sub16::QuantRowBytes(WType, k);
  const auto* const wb = static_cast<const std::uint8_t*>(w);
  float sums[kRowsPerWave][Batch] = {};

  stage(0, 0);
  __syncthreads();
  std::uint32_t buffer = 0;
  for (std::uint32_t tile_base = 0; tile_base < num_sub;
       tile_base += kTileSubs) {
    // The other buffer was last read before the previous barrier.
    if (tile_base + kTileSubs < num_sub)
      stage(buffer ^ 1U, tile_base + kTileSubs);
    const std::uint32_t sub = tile_base + lane;
    if (active && sub < num_sub) {
      uint4 qx[Batch];
      float2 ds[Batch];
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        qx[t] = stage_q[buffer][(t * kTileSubs) + lane];
        ds[t] = stage_ds[buffer][(t * kTileSubs) + lane];
      }
#pragma unroll
      for (std::size_t r = 0; r < kRowsPerWave; ++r) {
        // A tail row past m decodes the last row and is never stored.
        const std::uint32_t row = row_base + r < m ? row_base + r : m - 1;
        sub16::QuantSub16 dec;
        sub16::DecodeQuantSub16<true>(WType, wb + std::size_t(row) * row_bytes,
                                      sub, dec);
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          int dot = 0;
          dot = __builtin_amdgcn_sudot4(true, dec.w[0], true, qx[t].x, dot,
                                        false);
          dot = __builtin_amdgcn_sudot4(true, dec.w[1], true, qx[t].y, dot,
                                        false);
          dot = __builtin_amdgcn_sudot4(true, dec.w[2], true, qx[t].z, dot,
                                        false);
          dot = __builtin_amdgcn_sudot4(true, dec.w[3], true, qx[t].w, dot,
                                        false);
          float c = dec.scale * (ds[t].x * static_cast<float>(dot));
          if constexpr (kHasOffset)
            c -= dec.offset * ds[t].y;
          sums[r][t] += c;
        }
      }
    }
    __syncthreads();
    buffer ^= 1U;
  }

  if (!active)
    return;
#pragma unroll
  for (std::size_t r = 0; r < kRowsPerWave; ++r) {
    float values[8] = {};
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      values[t] = sums[r][t];
    // Lanes 4t..4t+3 end up holding token t's total.
    const float total = sub16::ReduceKQuantColumns<8, 16>(values);
    const std::uint32_t t = lane / 4U;
    if (lane % 4U == 0U && t < Batch && row_base + r < m)
      y[(std::size_t(t) * m) + row_base + r] = isfinite(total) ? total : 0.0F;
  }
}

template<GgmlType WType, int Width>
void LaunchType(const void* w, const void* x, float* y, int m, int k,
                int x_block_stride, hipStream_t stream) {
  DenseDot4Kernel<WType, std::size_t(Width)>
      <<<(std::uint32_t(m) + kRowsPerBlock - 1U) / kRowsPerBlock, kWaves * 32U,
         0, stream>>>(w, static_cast<const std::uint8_t*>(x), y,
                      std::uint32_t(m), std::uint32_t(k),
                      std::uint32_t(x_block_stride));
}

// The formats instantiated here; the engine's selection policy (which of
// them it routes) is dense_gemv::Dot4Selected.
template<int Width>
bool Launch(int type, const void* w, const void* x, float* y, int m, int k,
            int x_block_stride, hipStream_t stream) {
  static_assert(Width >= 1 && Width <= 8);
  if (!w || !x || !y || m <= 0 || k <= 0 || k % 256 != 0 ||
      x_block_stride < k / 32)
    return false;
  switch (static_cast<GgmlType>(type)) {
    case GgmlType::kQ4_K:
      LaunchType<GgmlType::kQ4_K, Width>(w, x, y, m, k, x_block_stride, stream);
      return true;
    case GgmlType::kQ5_K:
      LaunchType<GgmlType::kQ5_K, Width>(w, x, y, m, k, x_block_stride, stream);
      return true;
    case GgmlType::kQ6_K:
      LaunchType<GgmlType::kQ6_K, Width>(w, x, y, m, k, x_block_stride, stream);
      return true;
    case GgmlType::kIQ3_S:
      LaunchType<GgmlType::kIQ3_S, Width>(w, x, y, m, k, x_block_stride,
                                          stream);
      return true;
    case GgmlType::kIQ4_XS:
      LaunchType<GgmlType::kIQ4_XS, Width>(w, x, y, m, k, x_block_stride,
                                           stream);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next::dense_dot4
