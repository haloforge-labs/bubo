#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/device.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"

namespace gufo::models::diffusion_gemma::kernels {
namespace {

constexpr int kCols = 128;          // Weight rows (output columns) per block.
constexpr int kDepth = 64;          // Reduction depth per stage.
constexpr int kPitch = kDepth + 8;  // Padded LDS row: 144 bytes.

using U32x4 = __attribute__((ext_vector_type(4))) unsigned int;

__device__ float GeluTanh(float x) {
  const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

// 16 int8 values to 16 BF16 values (exact: |v| <= 128 needs 8 bits).
__device__ __forceinline__ void Int8ToBf16(U32x4 packed, bf16_t* out) {
  const auto* bytes = reinterpret_cast<const std::int8_t*>(&packed);
  union {
    bf16_t h[16];
    uint4 u[2];
  } converted;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    converted.h[j] = static_cast<bf16_t>(
        __float_as_uint(static_cast<float>(bytes[j])) >> 16);
  reinterpret_cast<uint4*>(out)[0] = converted.u[0];
  reinterpret_cast<uint4*>(out)[1] = converted.u[1];
}

// 16 FP8 E4M3 values to 16 BF16 values (exact).
__device__ __forceinline__ void Fp8ToBf16(U32x4 packed, bf16_t* out) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
  union {
    bf16_t h[16];
    uint4 u[2];
  } converted;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    converted.h[j] = Fp8E4M3ToBf16(bytes[j]);
  reinterpret_cast<uint4*>(out)[0] = converted.u[0];
  reinterpret_cast<uint4*>(out)[1] = converted.u[1];
}

// NVFP4 decode through byte permutes. A block's 16 values share one E4M3
// scale, so its 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} x scale are built
// once as BF16 (exact: an E2M1 magnitude has at most 2 significant bits and
// an E4M3 scale 4) by adding to the BF16 exponent; v_perm_b32 then looks up
// the low and high bytes of four values at a time.
struct Fp4Table {
  unsigned lo[2], hi[2];  // Bytes of entries 0..3 and 4..7.
};

__device__ __forceinline__ Fp4Table MakeFp4Table(unsigned scale_byte) {
  const unsigned s = Fp8E4M3ToBf16(scale_byte & 0x7fU);  // Scales are >= 0.
  const unsigned t = __float_as_uint(ToFloat(s) * 1.5F) >> 16;  // Exact.
  // Entry k: 0, s/2, s, 1.5s, 2s, 3s, 4s, 6s (exponent steps of 0x80).
  unsigned e[8] = {0,         s - 0x80U, s,          t,
                   s + 0x80U, t + 0x80U, s + 0x100U, t + 0x100U};
  if ((s & 0x7fffU) == 0) {
#pragma unroll
    for (int k = 0; k < 8; ++k)
      e[k] = 0;
  }
  Fp4Table table;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    table.lo[half] = 0;
    table.hi[half] = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      table.lo[half] |= (e[4 * half + b] & 0xffU) << (8 * b);
      table.hi[half] |= (e[4 * half + b] >> 8) << (8 * b);
    }
  }
  return table;
}

// 8 values (one word, first value in the low nibble) to 8 BF16 values.
__device__ __forceinline__ void DecodeFp4Word(unsigned word,
                                              const Fp4Table& table,
                                              unsigned out[4]) {
  unsigned lo[2], hi[2];  // [even values, odd values], one byte each.
#pragma unroll
  for (int odd = 0; odd < 2; ++odd) {
    const unsigned codes = (word >> (4 * odd)) & 0x0f0f0f0fU;
    const unsigned select = codes & 0x07070707U;
    lo[odd] = __builtin_amdgcn_perm(table.lo[1], table.lo[0], select);
    hi[odd] = __builtin_amdgcn_perm(table.hi[1], table.hi[0], select) |
              ((codes & 0x08080808U) << 4);  // Sign into bit 15.
  }
  // BF16 words of even values (0, 2 | 4, 6) and odd values (1, 3 | 5, 7).
  const unsigned even01 = __builtin_amdgcn_perm(hi[0], lo[0], 0x05010400U);
  const unsigned even23 = __builtin_amdgcn_perm(hi[0], lo[0], 0x07030602U);
  const unsigned odd01 = __builtin_amdgcn_perm(hi[1], lo[1], 0x05010400U);
  const unsigned odd23 = __builtin_amdgcn_perm(hi[1], lo[1], 0x07030602U);
  out[0] = __builtin_amdgcn_perm(odd01, even01, 0x05040100U);
  out[1] = __builtin_amdgcn_perm(odd01, even01, 0x07060302U);
  out[2] = __builtin_amdgcn_perm(odd23, even23, 0x05040100U);
  out[3] = __builtin_amdgcn_perm(odd23, even23, 0x07060302U);
}

// 32 NVFP4 values (16 bytes) and their two block scales (low byte first) to
// 32 BF16 values in LDS.
__device__ __forceinline__ void Fp4ToBf16(U32x4 packed, unsigned scales,
                                          bf16_t* out) {
  const Fp4Table tables[2] = {MakeFp4Table(scales & 0xffU),
                              MakeFp4Table((scales >> 8) & 0xffU)};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    U32x4 words;
    unsigned decoded[4];
    DecodeFp4Word(packed[w], tables[w / 2], decoded);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      words[q] = decoded[q];
    reinterpret_cast<U32x4*>(out)[w] = words;
  }
}

enum Format { kWeightsBf16, kWeightsQ8, kWeightsFp8, kWeightsFp4 };

// Grouped C = A W^T for one 64-row group of one expert's rows (an entry of
// Dispatch's group list) and one 128-column tile. Spreading an expert's
// groups over workgroups keeps the GPU busy when routing is uneven; the
// groups of one expert read its weights at about the same time, mostly from
// cache. Weight and activation tiles are read with full-line coalesced
// loads, prefetched one stage ahead in registers and staged through
// double-buffered LDS.
//
// kQ8: the weights are Q8_0 (int8 values + one FP16 scale per 32 along the
// reduction). Values are widened to BF16 exactly in LDS; each 32-wide block
// is accumulated separately and scaled by its row's scale in FP32, so the
// product is the exact Q8_0 matrix times the BF16 activations.
//
// kFp8: the weights are FP8 E4M3 with one BF16 scale per weight row. Values
// are widened to BF16 exactly in LDS and each output column is scaled by its
// row's scale after the reduction.
//
// kFp4: the weights are NVFP4: E2M1 values with an E4M3 scale per 16 along
// the reduction (applied, exactly, while widening to BF16 in LDS) and one
// FP32 scale per weight matrix, stored per row and applied after the
// reduction.
//
// kGateUp: A rows gathered from `x` through `source`; tile columns 0..63 are
// gate rows 64*tile.., 64..127 the matching up rows; the epilogue writes
// bf16(gelu(gate) * up). Otherwise A is the contiguous activation and the
// epilogue writes the output as BF16, the checkpoint dtype of the
// reference's expert projection.
template<bool kGateUp, Format kFormat, int kRowsT>
__global__ void __launch_bounds__(kRowsT * 4)
    ExpertKernel(const bf16_t* a_base, const bf16_t* weights,
                 const std::int8_t* q_values, const std::uint16_t* q_scales,
                 const bf16_t* row_scales, const std::uint8_t* block_scales,
                 const float* tensor_scales, ExpertRouting routing, void* out,
                 int depth, int inter, int out_width) {
  constexpr bool kQ8 = kFormat == kWeightsQ8;
  constexpr bool kFp4 = kFormat == kWeightsFp4;
  constexpr bool kBytes = kFormat == kWeightsQ8 || kFormat == kWeightsFp8;
  // Waves: kRowsT / 32 row strips of 32 rows x 4 column groups of 32 columns,
  // so 128-row groups run 16 waves with the 64-row kernel's registers per wave.
  constexpr int kThreads = kRowsT * 4;
  constexpr int kWN = 4, kNT = 2;
  // 64-row groups double-buffer their stages; 128-row groups use one buffer
  // (two would exceed 64 KB) with a second barrier per stage.
  constexpr int kBuffers = kRowsT == 64 ? 2 : 1;
  constexpr int kStage = (kRowsT + kCols) * kPitch;  // bf16 elements.
  constexpr int kBlocks = kDepth / 32;               // Q8_0 blocks per stage.
  constexpr int kStageBytes = kBuffers * kStage * 2;
  constexpr int kEpilogueBytes = 64 * (kCols + 1) * 4;  // 64 rows at a time.
  constexpr int kShared =
      kStageBytes > kEpilogueBytes ? kStageBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) unsigned char shared[kShared];
  __shared__ float stage_scales[kQ8 ? kBuffers : 1][kQ8 ? kCols : 1][kBlocks];
  auto* stages = reinterpret_cast<bf16_t*>(shared);

  const int work = blockIdx.y, tile = blockIdx.x;
  if (work >= *routing.group_count)
    return;
  const int expert = routing.groups[2 * work];
  const int group = routing.groups[2 * work + 1];
  const int live = min(kRowsT, routing.counts[expert] - group);
  const int first = routing.offsets[expert] + group;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int wave_m = wave / kWN, wave_n = wave % kWN;
  const std::size_t matrix_rows = kGateUp ? 2 * inter : out_width;
  const std::size_t expert_base =
      static_cast<std::size_t>(expert) * matrix_rows * depth;
  const auto tile_row = [&](int r) -> std::size_t {
    if constexpr (kGateUp)
      return r < 64 ? 64 * tile + r : inter + 64 * tile + (r - 64);
    else
      return static_cast<std::size_t>(kCols) * tile + r;
  };
  // This thread's weight pieces, 16 bytes each: 128 rows x 64 values is
  // 1024 BF16 pieces (8 per row), 512 8-bit pieces (4 per row) or 256 FP4
  // pieces (2 per row), shared by kThreads threads (with 512 threads, only
  // the first 256 take an FP4 piece). Offsets count values.
  constexpr int kPerRow = kFp4 ? 2 : (kBytes ? 4 : 8);
  constexpr int kValues = kFp4 ? 32 : (kBytes ? 16 : 8);
  constexpr int kAllPieces = kCols * kPerRow;
  constexpr int kPieces = (kAllPieces + kThreads - 1) / kThreads;
  const bool w_owner = kAllPieces >= kThreads || tid < kAllPieces;
  std::size_t w_src[kPieces];
  int w_dst[kPieces];
#pragma unroll
  for (int j = 0; j < kPieces; ++j) {
    const int index = min(j * kThreads + tid, kAllPieces - 1);
    const int r = index / kPerRow, chunk = (index % kPerRow) * kValues;
    w_src[j] = expert_base + tile_row(r) * depth + chunk;
    w_dst[j] = (kRowsT + r) * kPitch + chunk;
  }
  // Q8_0: the first 256 threads stage one block scale each per stage.
  const bool scale_owner = tid < kCols * kBlocks;
  const int scale_row = min(tid / kBlocks, kCols - 1),
            scale_block = tid % kBlocks;
  const std::size_t scale_src =
      kQ8 ? (expert_base + tile_row(scale_row) * depth) / 32 + scale_block : 0;

  const int tiles_m = (live + 15) / 16;
  // Activation pieces: kRowsT rows x 64 values, 8 per piece. Rows past the
  // group's end repeat its last row, whose products are never stored, so
  // the loads need no branch.
  constexpr int kAPieces = kRowsT * kDepth / 8 / kThreads;
  const bf16_t* a_src[kAPieces];
  int a_dst[kAPieces];
#pragma unroll
  for (int j = 0; j < kAPieces; ++j) {
    const int index = j * kThreads + tid;
    const int r = index / 8, chunk = (index % 8) * 8;
    a_dst[j] = r * kPitch + chunk;
    const int position = first + min(r, live - 1);
    const std::size_t row =
        kGateUp ? static_cast<std::size_t>(routing.source[position])
                : static_cast<std::size_t>(position);
    a_src[j] = a_base + row * depth + chunk;
  }
  Accumulator acc[2][kNT];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < kNT; ++n)
      acc[m][n] = Accumulator{};

  // Native vector types: HIP's uint4 kept these registers in scratch.
  U32x4 a_next[kAPieces], w_next[kPieces];
  std::uint16_t scale_next = 0;
  const auto fetch = [&](int k0) {
#pragma unroll
    for (int j = 0; j < kAPieces; ++j)
      a_next[j] = *reinterpret_cast<const U32x4*>(a_src[j] + k0);
#pragma unroll
    for (int j = 0; j < kPieces; ++j) {
      if constexpr (kFp4)
        w_next[j] =
            *reinterpret_cast<const U32x4*>(q_values + (w_src[j] + k0) / 2);
      else if constexpr (kBytes)
        w_next[j] = *reinterpret_cast<const U32x4*>(q_values + w_src[j] + k0);
      else
        w_next[j] = *reinterpret_cast<const U32x4*>(weights + w_src[j] + k0);
    }
    if constexpr (kQ8)
      scale_next = q_scales[scale_src + k0 / 32];
    if constexpr (kFp4)  // The two block scales of this thread's 32 values.
      scale_next = *reinterpret_cast<const std::uint16_t*>(
          block_scales + (w_src[0] + k0) / 16);
  };
  fetch(0);
  int buffer = 0;
  const int col0 = wave_n * kNT * 16;
  for (int k0 = 0; k0 < depth; k0 += kDepth) {
    if (kBuffers == 1 && k0 > 0)
      LdsBarrier();  // The previous stage is consumed.
    bf16_t* stage = stages + buffer * kStage;
#pragma unroll
    for (int j = 0; j < kAPieces; ++j)
      *reinterpret_cast<U32x4*>(stage + a_dst[j]) = a_next[j];
    if (w_owner) {
#pragma unroll
      for (int j = 0; j < kPieces; ++j) {
        if constexpr (kFormat == kWeightsQ8)
          Int8ToBf16(w_next[j], stage + w_dst[j]);
        else if constexpr (kFormat == kWeightsFp8)
          Fp8ToBf16(w_next[j], stage + w_dst[j]);
        else if constexpr (kFp4)
          Fp4ToBf16(w_next[j], scale_next, stage + w_dst[j]);
        else
          *reinterpret_cast<U32x4*>(stage + w_dst[j]) = w_next[j];
      }
    }
    if (kQ8 && scale_owner)
      stage_scales[buffer][scale_row][scale_block] =
          __half2float(__ushort_as_half(scale_next));
    LdsBarrier();
    // The last stage loads itself again: an unconditional prefetch keeps
    // the registers out of scratch.
    fetch(min(k0 + kDepth, depth - kDepth));
    const bf16_t* at = stage;
    const bf16_t* wt = stage + kRowsT * kPitch;
    if (kQ8 && wave_m * 2 < tiles_m) {
      // One 32-wide Q8_0 block at a time, scaled per output column.
#pragma unroll
      for (int blk = 0; blk < kBlocks; ++blk) {
        Accumulator part[2][kNT];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < kNT; ++n)
            part[m][n] = Accumulator{};
#pragma unroll
        for (int s = blk * 2; s < blk * 2 + 2; ++s) {
          Fragment b[kNT];
#pragma unroll
          for (int n = 0; n < kNT; ++n)
            b[n] = LoadFragment(wt + (col0 + n * 16 + lane % 16) * kPitch +
                                s * 16);
#pragma unroll
          for (int m = 0; m < 2; ++m)
            if (wave_m * 2 + m < tiles_m) {
              const Fragment a = LoadFragment(
                  at + ((wave_m * 2 + m) * 16 + lane % 16) * kPitch + s * 16);
#pragma unroll
              for (int n = 0; n < kNT; ++n)
                part[m][n] = Mma(a, b[n], part[m][n]);
            }
        }
#pragma unroll
        for (int n = 0; n < kNT; ++n) {
          const float scale =
              stage_scales[buffer][col0 + n * 16 + lane % 16][blk];
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int i = 0; i < 8; ++i)
              acc[m][n][i] += part[m][n][i] * scale;
        }
      }
    } else if (!kQ8 && wave_m * 2 < tiles_m) {
#pragma unroll
      for (int s = 0; s < kDepth / 16; ++s) {
        Fragment b[kNT];
#pragma unroll
        for (int n = 0; n < kNT; ++n)
          b[n] =
              LoadFragment(wt + (col0 + n * 16 + lane % 16) * kPitch + s * 16);
#pragma unroll
        for (int m = 0; m < 2; ++m)
          if (wave_m * 2 + m < tiles_m) {
            const Fragment a = LoadFragment(
                at + ((wave_m * 2 + m) * 16 + lane % 16) * kPitch + s * 16);
#pragma unroll
            for (int n = 0; n < kNT; ++n)
              acc[m][n] = Mma(a, b[n], acc[m][n]);
          }
      }
    }
    // Double-buffered: the other buffer is written next; its readers
    // finished before the barrier above, so one barrier per stage suffices.
    if constexpr (kBuffers == 2)
      buffer ^= 1;
  }

  if constexpr (kFormat == kWeightsFp8 || kFp4) {
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      const std::size_t row = static_cast<std::size_t>(expert) * matrix_rows +
                              tile_row(col0 + n * 16 + lane % 16);
      const float scale = kFp4 ? tensor_scales[row] : ToFloat(row_scales[row]);
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[m][n][i] *= scale;
    }
  }

  // Stage the tile in LDS, 64 rows at a time (the stages are consumed), so
  // that the outputs are written a row at a time.
  auto* staged = reinterpret_cast<float (*)[kCols + 1]>(shared);
  auto* result = static_cast<bf16_t*>(out);
#pragma unroll
  for (int half = 0; half < kRowsT / 64; ++half) {
    LdsBarrier();
    if (wave_m / 2 == half) {
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < kNT; ++n)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            staged[(wave_m % 2 * 2 + m) * 16 + 2 * i + lane / 16]
                  [col0 + n * 16 + lane % 16] = acc[m][n][i];
    }
    LdsBarrier();
    const int rows = min(64, live - half * 64);
    const int base = first + half * 64;
    if constexpr (kGateUp) {
      for (int index = tid; index < rows * 64; index += kThreads) {
        const int r = index / 64, c = index % 64;
        // The fused gate/up output is a checkpoint-dtype tensor.
        const float g = RoundBf16(staged[r][c]);
        const float u = RoundBf16(staged[r][64 + c]);
        result[static_cast<std::size_t>(base + r) * inter + 64 * tile + c] =
            ToBf16(RoundBf16(GeluTanh(g)) * u);
      }
    } else {
      for (int index = tid; index < rows * kCols; index += kThreads) {
        const int r = index / kCols, c = index % kCols;
        result[static_cast<std::size_t>(base + r) * out_width + kCols * tile +
               c] = ToBf16(staged[r][c]);
      }
    }
  }
}

}  // namespace

int ExpertGroupBound(int rows, int experts, int k, int group_rows) {
  // Each active expert's last group may be partial.
  const int assignments = rows * k;
  return (assignments + group_rows - 1) / group_rows +
         std::min(experts, assignments);
}

namespace {

// One launch of ExpertKernel for whichever weights are set (NVFP4, then FP8,
// then Q8_0, else BF16).
template<bool kGateUp, int kRowsT>
void LaunchExpertsRows(dim3 grid, hipStream_t stream, const bf16_t* a,
                       const bf16_t* bf16_weights, const ExpertRouting& routing,
                       void* out, int depth, int inter, int out_width,
                       ExpertQ8 q8, ExpertFp8 fp8, ExpertFp4 fp4) {
  if (fp4.values != nullptr)
    ExpertKernel<kGateUp, kWeightsFp4, kRowsT><<<grid, kRowsT * 4, 0, stream>>>(
        a, nullptr, reinterpret_cast<const std::int8_t*>(fp4.values), nullptr,
        nullptr, fp4.block_scales, fp4.tensor_scales, routing, out, depth,
        inter, out_width);
  else if (fp8.values != nullptr)
    ExpertKernel<kGateUp, kWeightsFp8, kRowsT><<<grid, kRowsT * 4, 0, stream>>>(
        a, nullptr, reinterpret_cast<const std::int8_t*>(fp8.values), nullptr,
        fp8.scales, nullptr, nullptr, routing, out, depth, inter, out_width);
  else if (q8.values != nullptr)
    ExpertKernel<kGateUp, kWeightsQ8, kRowsT><<<grid, kRowsT * 4, 0, stream>>>(
        a, nullptr, q8.values, q8.scales, nullptr, nullptr, nullptr, routing,
        out, depth, inter, out_width);
  else
    ExpertKernel<kGateUp, kWeightsBf16, kRowsT>
        <<<grid, kRowsT * 4, 0, stream>>>(a, bf16_weights, nullptr, nullptr,
                                          nullptr, nullptr, nullptr, routing,
                                          out, depth, inter, out_width);
}

template<bool kGateUp>
void LaunchExperts(dim3 grid, hipStream_t stream, const bf16_t* a,
                   const bf16_t* bf16_weights, const ExpertRouting& routing,
                   void* out, int depth, int inter, int out_width, ExpertQ8 q8,
                   ExpertFp8 fp8, ExpertFp4 fp4) {
  if (routing.group_rows == 128)
    LaunchExpertsRows<kGateUp, 128>(grid, stream, a, bf16_weights, routing, out,
                                    depth, inter, out_width, q8, fp8, fp4);
  else if (routing.group_rows == 64)
    LaunchExpertsRows<kGateUp, 64>(grid, stream, a, bf16_weights, routing, out,
                                   depth, inter, out_width, q8, fp8, fp4);
  else
    throw std::invalid_argument("expert products: unsupported group rows");
}

}  // namespace

void ExpertGateUp(const bf16_t* x, const bf16_t* gate_up,
                  const ExpertRouting& routing, bf16_t* act, int hidden,
                  int inter, hipStream_t stream, ExpertQ8 q8, ExpertFp8 fp8,
                  ExpertFp4 fp4) {
  if (inter % 64 != 0 || hidden % kDepth != 0)
    throw std::invalid_argument("ExpertGateUp: unsupported shape");
  LaunchExperts<true>(dim3(inter / 64, routing.max_groups), stream, x, gate_up,
                      routing, act, hidden, inter, inter, q8, fp8, fp4);
  CheckLaunch("ExpertGateUp");
}

void ExpertDown(const bf16_t* act, const bf16_t* down,
                const ExpertRouting& routing, bf16_t* out, int hidden,
                int inter, hipStream_t stream, ExpertQ8 q8, ExpertFp8 fp8,
                ExpertFp4 fp4) {
  if (hidden % kCols != 0 || inter % kDepth != 0)
    throw std::invalid_argument("ExpertDown: unsupported shape");
  LaunchExperts<false>(dim3(hidden / kCols, routing.max_groups), stream, act,
                       down, routing, out, inter, inter, hidden, q8, fp8, fp4);
  CheckLaunch("ExpertDown");
}

}  // namespace gufo::models::diffusion_gemma::kernels
