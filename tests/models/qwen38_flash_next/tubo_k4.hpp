// SPDX-License-Identifier: MIT
// Copyright (c) 2026 gufo contributors
// Copyright (c) 2026 BonkusCheemus (tubo changes)
//
// BENCH-ONLY ablation control, NOT part of the engine and not reachable from
// any dispatch. The kernels below are COPIED from BonkusCheemus/tubo (MIT,
// https://github.com/BonkusCheemus/tubo, HEAD 17232b3ad79e), files
//   src/models/qwen/hip/kernels/sn_q4k_k4.hpp   (SnQ4KK4Kernel and launcher)
//   src/models/qwen/hip/kernels/small_batch_dot4.hpp
//                                (QuantizeActivationsQ8KDot4Kernel only)
//   src/models/qwen/rows32_layout.hpp           (32-row chunk interleave)
// Changes: the includes are replaced by this file's, Iq4NlLookup4 comes from
// this repository's quant_ops.hpp, and the bench entry points at the end are
// new. The kernel text is not edited.
//
// What it is for: the VALU v_dot4 (sudot4) few-row family with int8 (q8_K)
// activations, lane = output row, K split across the waves of a workgroup.
// Layout rule: configs named *_native read the PUBLISHED ggml block layout
// (Q4_K, Q5_K only). Configs named *_rows32 read the tubo 32-row interleave
// that tubo builds at load time; that is a re-layout of a published file and is
// measured here ONLY as a ceiling control on synthetic or copied payload, not
// shippable without an owner decision. Q6_K and IQ4_XS exist only interleaved
// in tubo.
// Not measured by us before: every tubo number is their claim.

#pragma once

#if !defined(ENGINE_ENABLE_HIP)
#define ENGINE_ENABLE_HIP 1
#endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/quant_ops.hpp"

// ---- tubo src/models/qwen/rows32_layout.hpp (verbatim body) ----
namespace gufo::models {

enum class QwenWeightLayout : std::uint8_t {
  kNative,
  kRows32,
  kIq3SRows8,
  kIq3XxsRows4
};

// Native block byte -> Rows32 byte. rows/blocks describe the FULL tensor,
// including when copying only a row range. Q6 d / IQ4 heads are global planes.
constexpr std::size_t Rows32ByteOffset(std::size_t rows, std::size_t blocks,
                                       std::size_t row, std::size_t block,
                                       std::size_t byte,
                                       std::size_t block_bytes) noexcept {
  const bool iq4 = block_bytes == 136;
  const std::size_t meta = iq4 ? 8 : (block_bytes == 210 ? 2 : 0);
  const std::size_t payload = block_bytes - meta;
  const std::size_t tile = row / 32 * blocks + block;
  if ((iq4 && byte < meta) || (!iq4 && byte >= payload))
    return rows * blocks * payload + (tile * 32 + row % 32) * meta +
           (iq4 ? byte : byte - payload);
  const auto offset = byte - (iq4 ? meta : 0);
  return tile * 32 * payload + offset / 16 * 512 + row % 32 * 16 + offset % 16;
}

// Source and destination are non-overlapping full tensor bases. Row ranges
// must cover whole tiles. No tensor-sized scratch; metadata retains full R.
inline void InterleaveKQuantRows32Range(const void* src, void* dst,
                                        std::size_t rows, std::size_t k,
                                        std::size_t block_bytes,
                                        std::size_t begin, std::size_t end) {
  const bool iq4 = block_bytes == 136;
  const std::size_t meta = iq4 ? 8 : (block_bytes == 210 ? 2 : 0);
  const std::size_t payload = block_bytes - meta;
  if (rows % 32 || k % 256 || !block_bytes || payload % 16 || begin % 32 ||
      end % 32 || begin > end || end > rows ||
      (k / 256 && rows > std::numeric_limits<std::size_t>::max() / (k / 256)) ||
      (rows * (k / 256) >
       std::numeric_limits<std::size_t>::max() / block_bytes))
    throw std::invalid_argument("invalid Rows32 tensor or full-tile row range");
  if (begin == end || !k)
    return;
  if (!src || !dst)
    throw std::invalid_argument("null Rows32 source/destination");
  const std::size_t bytes = rows * (k / 256) * block_bytes;
  const auto saddr = reinterpret_cast<std::uintptr_t>(src);
  const auto daddr = reinterpret_cast<std::uintptr_t>(dst);
  if ((saddr <= daddr ? daddr - saddr : saddr - daddr) < bytes)
    throw std::invalid_argument("Rows32 source/destination overlap");
  const auto* s = static_cast<const std::uint8_t*>(src);
  auto* d = static_cast<std::uint8_t*>(dst);
  const std::size_t blocks = k / 256;
  for (std::size_t row = begin; row < end; ++row) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const auto* native = s + (row * blocks + b) * block_bytes;
      for (std::size_t c = 0; c < payload / 16; ++c)
        std::memcpy(d + Rows32ByteOffset(rows, blocks, row, b,
                                         (iq4 ? 8 : 0) + c * 16, block_bytes),
                    native + (iq4 ? 8 : 0) + c * 16, 16);
      if (meta)
        std::memcpy(d + Rows32ByteOffset(rows, blocks, row, b,
                                         iq4 ? 0 : payload, block_bytes),
                    native + (iq4 ? 0 : payload), meta);
    }
  }
}

inline void InterleaveKQuantRows32(const void* src, void* dst, std::size_t rows,
                                   std::size_t k, std::size_t block_bytes) {
  InterleaveKQuantRows32Range(src, dst, rows, k, block_bytes, 0, rows);
}

}  // namespace gufo::models

// ---- tubo small_batch_dot4.hpp: QuantizeActivationsQ8KDot4Kernel (verbatim)
// ----
namespace gufo::hip {
static __global__ void QuantizeActivationsQ8KDot4Kernel(
    const float* __restrict__ x, std::int8_t* __restrict__ xq,
    float2* __restrict__ xds, std::uint32_t subs) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = i < subs;  // every lane reaches the shuffles
  float v[16] = {};
  if (live) {
    const float4* src = reinterpret_cast<const float4*>(x) + (i * 4U);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 f = src[j];
      v[(4 * j) + 0] = f.x;
      v[(4 * j) + 1] = f.y;
      v[(4 * j) + 2] = f.z;
      v[(4 * j) + 3] = f.w;
    }
  }
  float amax = 0.0F;
  int nan = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    amax = fmaxf(amax, fabsf(v[j]));
    nan |= v[j] != v[j] ? 1 : 0;
  }
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    nan |= __shfl_xor(nan, o, 32);
  }
  if (!live)
    return;
  const float d = nan != 0 ? __builtin_nanf("") : amax / 127.0F;
  const float id = amax > 0.0F ? 127.0F / amax : 0.0F;
  std::uint32_t packed[4] = {};
  int sum = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int q =
        static_cast<int>(rintf(fminf(fmaxf(v[j] * id, -127.0F), 127.0F)));
    sum += q;
    packed[j / 4] |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * (j % 4));
  }
  reinterpret_cast<uint4*>(xq)[i] = {packed[0], packed[1], packed[2],
                                     packed[3]};
  xds[i] = {d, __int_as_float(sum)};
}
}  // namespace gufo::hip

// ---- tubo sn_q4k_k4.hpp (verbatim: lines 67-104 helpers, 112-882 kernel) ----
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace gufo::hip {

// Keep the existing bench API; the CPU loader uses the same permutation.
using models::InterleaveKQuantRows32;

namespace sn_q4k_k4_detail {

// Shared with the CPU contract test; only the existing codebook lookup is
// target-specific. sub16 follows natural activation order, 0..15.
#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)
__attribute__((host, device))
#endif
constexpr int Iq4XsScale(std::uint32_t head_x, std::uint32_t head_y,
                         std::uint32_t sub16) noexcept {
  const std::uint32_t j = sub16 >> 1U;
  return static_cast<int>(((head_y >> (4U * j)) & 15U) |
                          (((head_x >> (16U + 2U * j)) & 3U) << 4U)) -
         32;
}

#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)
__attribute__((host, device))
#endif
constexpr std::uint32_t Iq4XsIndices4(std::uint32_t word,
                                      std::uint32_t sub16) noexcept {
  return (word >> (4U * (sub16 & 1U))) & 0x0F0F0F0FU;
}

}  // namespace sn_q4k_k4_detail

}  // namespace gufo::hip
namespace gufo::hip {

namespace sn_q4k_k4_detail {

// ggml get_scale_min_k4(j, scales) on the block head uint4: head.x holds
// {d, dmin} as two fp16, head.y = scales[0..3], head.z = scales[4..7],
// head.w = scales[8..11].
__device__ __forceinline__ void Q4KScaleMin(const uint4& head, std::uint32_t j,
                                            std::uint32_t* sc,
                                            std::uint32_t* mn) {
  const std::uint32_t b = 8U * (j & 3U);
  if (j < 4U) {
    *sc = (head.y >> b) & 63U;
    *mn = (head.z >> b) & 63U;
  } else {
    *sc = ((head.w >> b) & 15U) | ((((head.y >> b) >> 6U) & 3U) << 4U);
    *mn = (((head.w >> b) >> 4U) & 15U) | ((((head.z >> b) >> 6U) & 3U) << 4U);
  }
}

// Cross-WG combine of split-K partials, a copy of SnbSplitKCombine
// (sn_iq3.hip): every split parks its row-group partials in
// scratch[(split*Batch + t)*m + row]; the LAST workgroup of the row group to
// arrive (counter[ticket]) adds the S partials in split order 0..S-1 and the
// caller stores y; the ticket is reset for the next launch. Agent-scope fences
// make the partials visible across CUs; the loads are agent-scope atomics so
// they cannot hit a stale L0 line. Returns true in the WG that must store y.
template<std::size_t Batch>
__device__ __forceinline__ bool SnQ4KSplitKCombine(
    float (&sums)[Batch], std::uint32_t* flag, float* scratch,
    std::uint32_t* counter, std::uint32_t ticket, std::uint32_t m,
    std::uint32_t row0, std::uint32_t split, std::uint32_t splits,
    bool phase0) {
  if (phase0 && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      scratch[(((split * Batch) + t) * m) + row0] = sums[t];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0U)
    *flag = atomicAdd(&counter[ticket], 1U) == splits - 1U;
  __syncthreads();
  const bool last = *flag != 0U;  // uniform across the workgroup
  if (last) {
    __threadfence();
    if (phase0 && row0 < m) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        float acc = 0.0f;
        for (std::uint32_t s = 0; s < splits; ++s) {
          const float part =
              s == split ? sums[t]
                         : __hip_atomic_load(
                               &scratch[(((s * Batch) + t) * m) + row0],
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          acc = s == 0U ? part : acc + part;
        }
        sums[t] = acc;
      }
    }
    if (threadIdx.x == 0U)
      counter[ticket] = 0U;
  }
  return last;
}

}  // namespace sn_q4k_k4_detail

// Packed min-term sums: per (K block b, token t) the 8 sub-block
// activation sums A_j = xds[2j].y + xds[2j+1].y (|A_j| <= 32*127 = 4064, fits
// int16) as 4 dwords {A_2k | A_2k+1 << 16}, then per block the Batch xd
// floats: record b = 5*Batch dwords, [4t..4t+3] = token t's pairs,
// [4*Batch + t] = token t's xd. One wave-uniform record per block, so the
// k4 kernel reads its min-term inputs with one scalar wait per block instead
// of one per token. Same integers, same floats: bit-identical y. This is a
// quantize-side step (fusable into QuantizeActivationsQ8KDot4Kernel).
// Q6 keeps all sixteen A16 sums instead: eight packed dwords per token,
// followed by Batch xd words (9*Batch dwords/block). |A16| <= 2032.
template<std::size_t Batch, bool Q6 = false>
__global__ void SnQ4KPackSumsKernel(const float2* __restrict__ xds,
                                    std::int32_t* __restrict__ xsum,
                                    std::uint32_t blocks) {
  const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= blocks * Batch)
    return;
  const std::uint32_t b = i / Batch;
  const std::uint32_t t = i % Batch;
  const float2* const src = xds + (t * (blocks * 16U)) + (b * 16U);
  constexpr std::uint32_t kPairs = Q6 ? 8U : 4U;
  std::int32_t* const rec = xsum + (b * (kPairs + 1U) * Batch);
#pragma unroll
  for (std::uint32_t q = 0; q < kPairs; ++q) {
    const int lo =
        Q6 ? __float_as_int(src[2U * q].y)
           : __float_as_int(src[4U * q].y) + __float_as_int(src[4U * q + 1U].y);
    const int hi = Q6 ? __float_as_int(src[2U * q + 1U].y)
                      : __float_as_int(src[4U * q + 2U].y) +
                            __float_as_int(src[4U * q + 3U].y);
    rec[(kPairs * t) + q] =
        static_cast<std::int32_t>((static_cast<std::uint32_t>(lo) & 0xFFFFU) |
                                  (static_cast<std::uint32_t>(hi) << 16U));
  }
  rec[(kPairs * Batch) + t] = __float_as_int(src[0].x);
}

// y[token][row] = sum_k W[row][k] * x[token][k], W in ggml Q4_K/Q5_K layout
// or its chunk interleave; x is int8 with one scale per 256 (xq/xds).
// One lane owns one output row; the KPhases waves of a workgroup split K.
// `scratch`/`counter` are only read when SplitK (the split-K combine).
//
// Interleaved: the weights are InterleaveKQuantRows32's
// layout, so the 32 lanes of a wave read one chunk of 32 rows as 512
// contiguous bytes instead of 32 strided 16-byte pieces. Same bytes, same math.
//
// Hoist: load the block's 8 qs chunks together with head/qh at
// the top of the block iteration. Without it the sched_barrier(0) fences of
// the token quads keep each group's qs loads inside its group, so a wave
// waits on DRAM 5 times per block (ISA: vmcnt(0) after 5, 2, 1, 1, 2 loads).
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool SplitK, bool Interleaved = false, bool Q5 = false,
         bool Hoist = false, bool PackedSums = false, int Probe = 0,
         bool Dynamic = false, bool Q6 = false, bool IQ4 = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void SnQ4KK4Kernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, float* __restrict__ scratch,
        std::uint32_t* __restrict__ counter,
        const std::int32_t* __restrict__ xsum = nullptr) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!IQ4 || (Interleaved && Hoist && !PackedSums && !Q5 && !Q6 &&
                         !SplitK && Probe == 0 && !Dynamic),
                "IQ4_XS uses hoisted split metadata, no min term/experiments");
  static_assert(!Q6 || (Interleaved && !Q5 && Probe == 0 && !Dynamic),
                "Q6 uses the split-d interleave, not Probe/Dynamic");
  static_assert(KPhases >= 1U && RowWaves >= 1U);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  constexpr std::uint32_t kPartialFloats =
      (KPhases - 1U) * RowWaves * Batch * 32U;
  __shared__ float lds[kPartialFloats + 4U];
  __shared__ std::uint32_t sk_flag;
  // Dynamic: the KPhases waves of one row-wave take their first
  // block statically (b_begin + phase) and every later one from this LDS
  // counter, so a wave that DRAM serves early takes more blocks and the
  // siblings reach the LDS reduce together (trace: static waits p90 22 us).
  // fp32 sums then depend on which wave took which block: not bitwise
  // run-to-run deterministic (rel error unchanged).
  // 32 slots per row-wave, one per lane: a per-lane (divergent) address keeps
  // LLVM's atomic optimizer from rewriting the claim into a lane scan, which
  // costs ~500 SGPR spills in this loop. Only lane 0's slot counts.
  __shared__ std::uint32_t dyn_next[Dynamic ? RowWaves * 32U : 1];
  if constexpr (!SplitK) {
    (void)scratch;
    (void)counter;
  }

  const std::uint32_t blocks = k >> 8U;
  // SplitK: blockIdx.y = split; each split owns `blocks/splits` contiguous K
  // blocks. Without SplitK these fold to 0 / blocks and the code is the plain
  // k4 path.
  const std::uint32_t splits = SplitK ? gridDim.y : 1U;
  const std::uint32_t split = SplitK ? blockIdx.y : 0U;
  const std::uint32_t b_begin = SplitK ? split * (blocks / splits) : 0U;
  const std::uint32_t b_end = SplitK ? b_begin + (blocks / splits) : blocks;

  const std::uint32_t lane = threadIdx.x & 31U;
  // readfirstlane: the compiler must see the wave index as uniform, or the
  // activation address (which depends on the phase) stops being an s_load.
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 =
      (((blockIdx.x * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t num_sub = k >> 4U;
  // Probe 3/4 trace: lane 0 of each wave writes 8 u64 wall-clock
  // stamps (100 MHz) to scratch: start, end of block iterations 0..3, after
  // the LDS reduce, end, hw id. 3 = full kernel, 4 = loads only.
  constexpr bool kTrace = Probe >= 3;
  constexpr bool kLoadOnly = Probe == 2 || Probe == 4;
  static_assert(!(kTrace && SplitK), "trace stamps live in the split scratch");
  std::uint64_t* const ts =
      kTrace ? reinterpret_cast<std::uint64_t*>(scratch) +
                   (8U * ((static_cast<std::size_t>(blockIdx.x) *
                           (RowWaves * KPhases)) +
                          wave))
             : nullptr;
  std::uint32_t it = 0;
  if constexpr (kTrace) {
    if (lane == 0U)
      ts[0] = wall_clock64();
  }

  // Rows past m are clamped so their loads stay inside the last row; their
  // outputs are simply not stored.
  // Chunk c of block b, row r: (r/32 * blocks + b) * (32*block_bytes)
  // + c*512 + (r%32)*16. Q4_K has 9 chunks (4608 B), Q5_K 11 (5632 B).
  constexpr std::uint32_t kBlockBytes =
      IQ4 ? 136U : (Q6 ? 210U : (Q5 ? 176U : 144U));
  constexpr std::uint32_t kChunkBytes = IQ4 ? 128U : (Q6 ? 208U : kBlockBytes);
  constexpr std::uint32_t kBlockStride =
      Interleaved ? 32U * kChunkBytes : kBlockBytes;
  constexpr std::uint32_t kChunk = Interleaved ? 512U : 16U;
  constexpr std::uint32_t kQsChunk = Q5 ? 3U : 1U;
  const std::uint8_t* const row_base =
      static_cast<const std::uint8_t*>(w) +
      (Interleaved
           ? (static_cast<std::size_t>(row >> 5U) * blocks * kBlockStride) +
                 ((row & 31U) * 16U)
           : static_cast<std::size_t>(row) * (blocks * kBlockBytes));

  float sums[Batch] = {};
  if constexpr (Dynamic) {
    if (phase == 0U)
      dyn_next[(row_wave * 32U) + lane] = KPhases;
    __syncthreads();
  }
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if constexpr (Dynamic) {
      // Branch-free: every lane adds (lane == 0) to its own slot; lane 0's
      // old value is the claimed block.
      const std::uint32_t v =
          atomicAdd(&dyn_next[(row_wave * 32U) + lane], lane == 0U ? 1U : 0U);
      return b_begin + __builtin_amdgcn_readfirstlane(v);
    } else {
      return b + KPhases;
    }
  };
  for (std::uint32_t b = b_begin + phase; b < b_end; b = next_block(b)) {
    const std::uint8_t* const bp =
        row_base + (static_cast<std::size_t>(b) * kBlockStride);

    if constexpr (IQ4) {
      const auto* const meta_plane =
          w + static_cast<std::size_t>(m) * blocks * 128U;
      const std::size_t meta_index =
          ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
          row % 32U;
      const uint2 head =
          *reinterpret_cast<const uint2*>(meta_plane + meta_index * 8U);
      uint4 payload[8];
#pragma unroll
      for (std::uint32_t c = 0; c < 8U; ++c)
        payload[c] = *reinterpret_cast<const uint4*>(bp + c * kChunk);
      __builtin_amdgcn_sched_barrier(0);
      const float d = __half2float(
          __ushort_as_half(static_cast<unsigned short>(head.x & 0xFFFFU)));
      int isum[Batch] = {};
#pragma unroll
      for (std::uint32_t gp = 0; gp < 4U; ++gp) {
        std::uint32_t q[4][4];
        int scale[4];
#pragma unroll
        for (std::uint32_t sub = 0; sub < 4U; ++sub) {
          const std::uint32_t sub16 = 4U * gp + sub;
          const uint4 packed = payload[2U * gp + (sub >> 1U)];
          const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                          packed.w};
          scale[sub] = sn_q4k_k4_detail::Iq4XsScale(head.x, head.y, sub16);
#pragma unroll
          for (std::uint32_t i = 0; i < 4U; ++i)
            q[sub][i] =
                Iq4NlLookup4(sn_q4k_k4_detail::Iq4XsIndices4(words[i], sub16));
        }
#pragma unroll
        for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
          uint4 a[4][4];
#pragma unroll
          for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
            const auto* const p = reinterpret_cast<const uint4*>(
                xq + (t0 + j) * k + b * 256U + gp * 64U);
#pragma unroll
            for (std::uint32_t sub = 0; sub < 4U; ++sub)
              a[j][sub] = p[sub];
          }
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
#pragma unroll
            for (std::uint32_t sub = 0; sub < 4U; ++sub) {
              const uint4 ax = a[j][sub];
              int dot = 0;
              dot = __builtin_amdgcn_sudot4(true, q[sub][0], true, ax.x, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][1], true, ax.y, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][2], true, ax.z, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][3], true, ax.w, dot,
                                            false);
              isum[t0 + j] += __mul24(scale[sub], dot);
            }
          }
          __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const float xd = xds[t * num_sub + b * 16U].x;
        sums[t] += (d * xd) * static_cast<float>(isum[t]);
      }
      continue;
    }
    if constexpr (Q6) {
      if constexpr (Hoist || PackedSums) {
        const std::uint8_t* const d_plane =
            static_cast<const std::uint8_t*>(w) +
            (static_cast<std::size_t>(m) * blocks * kChunkBytes);
        const std::size_t d_index =
            ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
            (row % 32U);
        const unsigned short d_bits =
            *reinterpret_cast<const unsigned short*>(d_plane + (d_index * 2U));
        uint4 payload[13];
#pragma unroll
        for (std::uint32_t c = 0; c < 13U; ++c)
          payload[c] = *reinterpret_cast<const uint4*>(bp + (c * kChunk));
        __builtin_amdgcn_sched_barrier(0);
        const float d = __half2float(__ushort_as_half(d_bits));
        const uint4 scales = payload[12];
        const std::uint32_t sc_words[4] = {scales.x, scales.y, scales.z,
                                           scales.w};
        int isum[Batch] = {};
#pragma unroll
        for (std::uint32_t gp = 0; gp < 4U; ++gp) {
          const std::uint32_t h = gp >> 1U;
          std::uint32_t q[4][4];
#pragma unroll
          for (std::uint32_t sub = 0; sub < 4U; ++sub) {
            const uint4 ql = payload[4U * h + sub];
            const uint4 qh = payload[8U + 2U * h + (sub & 1U)];
            const std::uint32_t lw[4] = {ql.x, ql.y, ql.z, ql.w};
            const std::uint32_t hw[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i)
              q[sub][i] = ((lw[i] >> (4U * (gp & 1U))) & 0x0F0F0F0FU) |
                          (((hw[i] >> (4U * (gp & 1U) + 2U * (sub >> 1U))) &
                            0x03030303U)
                           << 4U);
          }
          // A quad is four tokens x 64 bytes, as in the Q5 winner. The two
          // barriers prevent later quads/sub-blocks from retaining hundreds
          // of activation SGPRs across this quad.
#pragma unroll
          for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
            uint4 a[4][4];
#pragma unroll
            for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
              const uint4* const p = reinterpret_cast<const uint4*>(
                  xq + ((t0 + j) * k) + (b * 256U) + (gp * 64U));
#pragma unroll
              for (std::uint32_t sub = 0; sub < 4U; ++sub)
                a[j][sub] = p[sub];
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
#pragma unroll
              for (std::uint32_t sub = 0; sub < 4U; ++sub) {
                const uint4 ax = a[j][sub];
                int dot = 0;
                dot = __builtin_amdgcn_sudot4(true, q[sub][0], true, ax.x, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][1], true, ax.y, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][2], true, ax.z, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][3], true, ax.w, dot,
                                              false);
                const std::uint32_t s = 4U * gp + sub;
                const int sc = static_cast<std::int8_t>(
                    (sc_words[s >> 2U] >> (8U * (s & 3U))) & 0xFFU);
                isum[t0 + j] += __mul24(sc, dot);
              }
            }
            __builtin_amdgcn_sched_barrier(0);
          }
        }
        if constexpr (PackedSums) {
          const std::int32_t* const rec =
              xsum + (static_cast<std::size_t>(b) * 9U * Batch);
          std::int32_t pk[9U * Batch];
#pragma unroll
          for (std::uint32_t i = 0; i < 9U * Batch; ++i)
            pk[i] = rec[i];
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            int ioff = 0;
#pragma unroll
            for (std::uint32_t sub = 0; sub < 16U; ++sub) {
              const std::int32_t pair = pk[8U * t + (sub >> 1U)];
              const int a_sum =
                  (sub & 1U) ? pair >> 16
                             : static_cast<int>(static_cast<short>(pair));
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
              ioff += __mul24(sc, a_sum);
            }
            const float xd = __int_as_float(pk[8U * Batch + t]);
            sums[t] += (d * xd) * static_cast<float>(isum[t] - 32 * ioff);
          }
        } else {
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            int ioff = 0;
#pragma unroll
            for (std::uint32_t sub = 0; sub < 16U; ++sub) {
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
              ioff += __mul24(
                  sc, __float_as_int(xds[t * num_sub + b * 16U + sub].y));
            }
            sums[t] += (d * xds[t * num_sub + b * 16U].x) *
                       static_cast<float>(isum[t] - 32 * ioff);
          }
        }
      } else {
        const std::uint8_t* const d_plane =
            static_cast<const std::uint8_t*>(w) +
            (static_cast<std::size_t>(m) * blocks * kChunkBytes);
        const std::size_t d_index =
            ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
            (row % 32U);
        const unsigned short d_bits =
            *reinterpret_cast<const unsigned short*>(d_plane + (d_index * 2U));
        const float d = __half2float(__ushort_as_half(d_bits));
        const uint4 scales =
            *reinterpret_cast<const uint4*>(bp + (12U * kChunk));
        const std::uint32_t sc_words[4] = {scales.x, scales.y, scales.z,
                                           scales.w};
        int isum[Batch] = {};
        // In-body loads only. Keep just two ql chunks and one qh chunk live
        // per (128-weight half, 16-byte piece), not all 13 chunks at once.
#pragma unroll
        for (std::uint32_t h = 0; h < 2U; ++h) {
#pragma unroll
          for (std::uint32_t piece = 0; piece < 2U; ++piece) {
            const uint4 ql0 = *reinterpret_cast<const uint4*>(
                bp + ((4U * h + piece) * kChunk));
            const uint4 ql1 = *reinterpret_cast<const uint4*>(
                bp + ((4U * h + 2U + piece) * kChunk));
            const uint4 qh = *reinterpret_cast<const uint4*>(
                bp + ((8U + 2U * h + piece) * kChunk));
            const std::uint32_t hw[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (std::uint32_t g = 0; g < 4U; ++g) {
              const std::uint32_t sub = 8U * h + 2U * g + piece;
              const uint4 ql = (g & 1U) != 0U ? ql1 : ql0;
              const std::uint32_t lw[4] = {ql.x, ql.y, ql.z, ql.w};
              std::uint32_t q[4];
#pragma unroll
              for (std::uint32_t i = 0; i < 4U; ++i)
                q[i] = ((lw[i] >> (4U * (g >> 1U))) & 0x0F0F0F0FU) |
                       (((hw[i] >> (2U * g)) & 0x03030303U) << 4U);
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
#pragma unroll
              for (std::size_t t = 0; t < Batch; ++t) {
                const uint4 ax = *reinterpret_cast<const uint4*>(
                    xq + (static_cast<std::size_t>(t) * k) + (b * 256U) +
                    (sub * 16U));
                int dot = 0;
                dot =
                    __builtin_amdgcn_sudot4(true, q[0], true, ax.x, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[1], true, ax.y, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[2], true, ax.z, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[3], true, ax.w, dot, false);
                const int a_sum =
                    __float_as_int(xds[(t * num_sub) + (b * 16U) + sub].y);
                isum[t] += __mul24(sc, dot - (32 * a_sum));
              }
            }
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          const float xd = xds[(t * num_sub) + (b * 16U)].x;
          sums[t] += (d * xd) * static_cast<float>(isum[t]);
        }
      }
      continue;
    }
    // Shared head: {d, dmin} + 12 scale bytes. Q5_K has two qh chunks
    // before the eight qs chunks; Q4_K has no qh plane.
    const uint4 head = *reinterpret_cast<const uint4*>(bp);
    uint4 qh0{};
    uint4 qh1{};
    if constexpr (Q5) {
      qh0 = *reinterpret_cast<const uint4*>(bp + kChunk);
      qh1 = *reinterpret_cast<const uint4*>(bp + (2U * kChunk));
    }
    const std::uint32_t qhw[8] = {qh0.x, qh0.y, qh0.z, qh0.w,
                                  qh1.x, qh1.y, qh1.z, qh1.w};
    const float d = __half2float(
        __ushort_as_half(static_cast<unsigned short>(head.x & 0xFFFFU)));
    const float dmin = __half2float(
        __ushort_as_half(static_cast<unsigned short>(head.x >> 16U)));
    int sc[8];
    int mn[8];
#pragma unroll
    for (std::uint32_t j = 0; j < 8U; ++j) {
      std::uint32_t s = 0;
      std::uint32_t m8 = 0;
      sn_q4k_k4_detail::Q4KScaleMin(head, j, &s, &m8);
      sc[j] = static_cast<int>(s);
      mn[j] = static_cast<int>(m8);
    }
    // Probe (timing only, WRONG y): 1 = every block reads block 0's
    // activations and sums (always scalar-cache hot); 2 = loads only.
    const std::uint32_t ab = Probe == 1 ? 0U : b;
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(ab) * 256U);
    uint4 qs[Hoist ? 8 : 1];
    if constexpr (Hoist) {
#pragma unroll
      for (std::uint32_t c = 0; c < 8U; ++c)
        qs[c] = *reinterpret_cast<const uint4*>(bp + (kChunk * (kQsChunk + c)));
      // All 11 (Q5) / 9 (Q4) loads of the block issue before any use.
      __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (kLoadOnly) {
      std::uint32_t x = head.x ^ head.y ^ head.z ^ head.w ^ qh0.x ^ qh1.w;
#pragma unroll
      for (std::uint32_t c = 0; c < (Hoist ? 8U : 1U); ++c)
        x ^= qs[c].x ^ qs[c].y ^ qs[c].z ^ qs[c].w;
      sums[0] += static_cast<float>(x & 1U);
      if constexpr (kTrace) {
        __builtin_amdgcn_sched_barrier(0);
        ts[1U + (it & 3U)] =
            wall_clock64();  // all lanes, one address: no branch
        ++it;
      }
      continue;
    }
    int isum[Batch] = {};
#pragma unroll
    for (std::uint32_t gp = 0; gp < 4U; ++gp) {
      // Group gp's 32 qs bytes: low nibbles = sub-block 2gp, high = 2gp+1.
      uint4 c0;
      uint4 c1;
      if constexpr (Hoist) {
        c0 = qs[2U * gp];
        c1 = qs[(2U * gp) + 1U];
      } else {
        c0 = *reinterpret_cast<const uint4*>(bp +
                                             (kChunk * (kQsChunk + (2U * gp))));
        c1 = *reinterpret_cast<const uint4*>(
            bp + (kChunk * (kQsChunk + 1U + (2U * gp))));
      }
      const std::uint32_t gq[8] = {c0.x, c0.y, c0.z, c0.w,
                                   c1.x, c1.y, c1.z, c1.w};
      std::uint32_t lo[8];
      std::uint32_t hi[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8U; ++i) {
        lo[i] = gq[i] & 0x0F0F0F0FU;
        hi[i] = (gq[i] >> 4U) & 0x0F0F0F0FU;
        if constexpr (Q5) {
          lo[i] |= ((qhw[i] >> (2U * gp)) & 0x01010101U) << 4U;
          hi[i] |= ((qhw[i] >> ((2U * gp) + 1U)) & 0x01010101U) << 4U;
        }
      }
      const int sc_lo = sc[2U * gp];
      const int sc_hi = sc[(2U * gp) + 1U];
      // Tokens in quads: a quad = 4 tokens x 64 bytes (both 32-weight
      // sub-blocks of the group) = 4 s_load_b512 = 64 SGPRs, the most that
      // fits beside the rest.
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
        constexpr std::size_t kQuad = 4U;
        uint4 a[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * k) + (gp * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            a[j][c] = p[c];
        }
        // Keep the next quad's 64 SGPRs of activations out of this one.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4 alo = a[j][0];
          const uint4 ami = a[j][1];
          const uint4 blo = a[j][2];
          const uint4 bmi = a[j][3];
          int dot_lo = 0;
          int dot_hi = 0;
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[0], true, alo.x, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[1], true, alo.y, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[2], true, alo.z, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[3], true, alo.w, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[4], true, ami.x, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[5], true, ami.y, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[6], true, ami.z, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[7], true, ami.w, dot_lo, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[0], true, blo.x, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[1], true, blo.y, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[2], true, blo.z, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[3], true, blo.w, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[4], true, bmi.x, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[5], true, bmi.y, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[6], true, bmi.z, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[7], true, bmi.w, dot_hi, false);
          isum[t0 + j] += __mul24(sc_lo, dot_lo) + __mul24(sc_hi, dot_hi);
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    // Per 256-weight block: the min term from the activation sums, then one
    // float fold per (row, token).
    if constexpr (PackedSums) {
      const std::int32_t* const rec =
          xsum + (static_cast<std::size_t>(ab) * 5U * Batch);
      std::int32_t pk[5U * Batch];
#pragma unroll
      for (std::uint32_t i = 0; i < 5U * Batch; ++i)
        pk[i] = rec[i];
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        int imin = 0;
#pragma unroll
        for (std::uint32_t q = 0; q < 4U; ++q) {
          const std::int32_t v = pk[(4U * t) + q];
          imin += __mul24(mn[2U * q], static_cast<int>(static_cast<short>(v)));
          imin += __mul24(mn[(2U * q) + 1U], v >> 16);
        }
        const float xd = __int_as_float(pk[(4U * Batch) + t]);
        sums[t] += xd * ((d * static_cast<float>(isum[t])) -
                         (dmin * static_cast<float>(imin)));
      }
    } else {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const float2* const base = xds + ((t * num_sub) + (b * 16U));
        const uint4* const pairs = reinterpret_cast<const uint4*>(base);
        int imin = 0;
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j) {
          const uint4 pr = pairs[j];
          const int a_sum = static_cast<int>(pr.y) + static_cast<int>(pr.w);
          imin += __mul24(mn[j], a_sum);
        }
        const float xd = base[0].x;
        sums[t] += xd * ((d * static_cast<float>(isum[t])) -
                         (dmin * static_cast<float>(imin)));
      }
    }
    if constexpr (kTrace) {
      __builtin_amdgcn_sched_barrier(0);
      ts[1U + (it & 3U)] = wall_clock64();  // all lanes, one address: no branch
      ++it;
    }
  }

  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1.
    __syncthreads();  // every wave is done reading its weights
    float* const partial = lds;
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  if constexpr (kTrace) {
    if (lane == 0U)
      ts[5] = wall_clock64();
  }
  bool store_y = true;
  if constexpr (SplitK)
    store_y = sn_q4k_k4_detail::SnQ4KSplitKCombine<Batch>(
        sums, &sk_flag, scratch, counter, blockIdx.x, m, row0, split, splits,
        phase == 0U);
  if (store_y && phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
  if constexpr (kTrace) {
    __builtin_amdgcn_sched_barrier(0);
    if (lane == 0U) {
      ts[6] = wall_clock64();
      ts[7] = __smid();
    }
  }
}

// SnQ4KPackSumsKernel over k/256 blocks x Batch tokens; xsum needs
// k/256 * (Q6 ? 9 : 5) * Batch int32.
template<std::size_t Batch, bool Q6 = false>
void LaunchSnQ4KPackSums(const float2* xds, std::int32_t* xsum, std::uint32_t k,
                         hipStream_t stream) {
  const std::uint32_t n = (k / 256U) * static_cast<std::uint32_t>(Batch);
  SnQ4KPackSumsKernel<Batch, Q6>
      <<<(n + 255U) / 256U, 256, 0, stream>>>(xds, xsum, k / 256U);
}

// Quantize the activations to q8_K (one int8 scale per 256), then run the k4
// kernel. k must be a multiple of 256 and `w` 16-byte aligned. quantize=false
// skips the quantize launch (GUFO_DFLASH_PREQUANT=1). For SplitK, `scratch`
// needs splits*Batch*m floats and `counter` one uint32 per row group of
// RowWaves*32 rows (blockIdx.x is the ticket), zeroed; `splits` must divide
// k/256.
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool SplitK = false, bool Interleaved = false, bool Q5 = false,
         bool Hoist = false, bool PackedSums = false, int Probe = 0,
         bool Dynamic = false, bool Q6 = false, bool IQ4 = false>
void LaunchSnQ4KK4(const void* w, const float* x, float* y, std::int8_t* xq,
                   float2* xds, std::uint32_t m, std::uint32_t k,
                   hipStream_t stream, std::uint32_t splits = 1U,
                   float* scratch = nullptr, std::uint32_t* counter = nullptr,
                   bool quantize = true, std::int32_t* xsum = nullptr) {
  static_assert(Batch >= 1 && Batch <= 8);
  if (PackedSums && xsum == nullptr)
    throw std::invalid_argument("PackedSums needs a packed-sum buffer");
  if (quantize) {
    const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
        x, xq, xds, subs);
    if constexpr (PackedSums)
      LaunchSnQ4KPackSums<Batch, Q6>(xds, xsum, k, stream);
  }
  if (Interleaved && m % 32U != 0U)
    throw std::invalid_argument("K-quant interleave needs m % 32 == 0");
  constexpr std::uint32_t kRowsPerBlock = RowWaves * 32U;
  SnQ4KK4Kernel<Batch, RowWaves, KPhases, SplitK, Interleaved, Q5, Hoist,
                PackedSums, Probe, Dynamic, Q6, IQ4>
      <<<dim3((m + kRowsPerBlock - 1U) / kRowsPerBlock, SplitK ? splits : 1U),
         RowWaves * KPhases * 32U, 0, stream>>>(
          static_cast<const std::uint8_t*>(w), xq, xds, y, m, k, scratch,
          counter, xsum);
}

// Legal split count for the k4 grid: the forced S when it is legal (divides
// the K blocks), else 1. Mirrors SnbPickSplitK's legality test.
inline std::uint32_t SnQ4KPickSplit(std::uint32_t blocks,
                                    std::uint32_t forced) noexcept {
  if (forced <= 1U)
    return 1U;
  return blocks % forced == 0U ? forced : 1U;
}

}  // namespace gufo::hip

// tubo's LDS-staged dot4 GEMM (small_batch_dot4.hpp) is no longer copied
// here: its production port reads the engine's Q8_1 activations
// (src/models/qwen38_flash_next/kernels/rocm/mmq/dense_dot4.hpp) and the
// bench reaches it as the dense variant `dot4`, with launch counters.
