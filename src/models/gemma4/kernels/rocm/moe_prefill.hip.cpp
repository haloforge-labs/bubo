// Routed expert GEMM for prefill over binary16 activations for the K-quant
// expert formats (Q4_K, Q5_K, Q6_K), and for the blockwise ones (Q4_0, Q5_0,
// Q5_1, Q8_0, binary16). It reads the routing layout of Flash-Next's routed
// GEMM: RoutedCompact's 16-padded expert buckets and a tile map of expert |
// tile << 16 entries.
//
// A block computes 128 output rows (eight waves of 16) for one tile of
// kTileTokens bucket rows, skipping the tile's 16-row parts past the bucket.
// A stage is half a super-block (128 K values, four 32-value K blocks): two
// threads per row fetch its code bytes and scale header one stage ahead in
// registers, neighbouring threads on neighbouring bytes, and commit them with
// the tile's activations to LDS, the scales already as binary16. Each lane
// then decodes its own row's K blocks straight into binary16 WMMA fragments.
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and B
// column L%16, each with 16 K values in both half-waves; C lane L holds
// column L%16, rows 2 i + L/16.
//
// Q4_K and Q5_K weights are q * (d sc) - dmin m, evaluated as one binary16
// FMA per pair with the two factors rounded to binary16, as in Flash-Next's
// routed GEMM, so the two produce the same products; Q6_K weights are
// (q - 32) * (d sc).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/hip/half_kquant_gemm.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {

// The K-quant kernel template and the helpers shared with the down kernels
// below (v16h, v8f, kThreads, kWave, kRowsPerBlock, Wmma, CodesToHalves*,
// SaturatedHalf, GeGluValue) live in the shared header.
using namespace gufo::hip::half_kquant;

namespace {

/// K values per stage of the down projection kernel: two 32-value blocks.
constexpr int kDownStageK = 64;
/// Halves per staged token row of the down kernel: 64 values plus padding
/// (36 dwords, conflict-free fragment reads).
constexpr int kDownActStride = kDownStageK + 8;
/// Bytes per staged Q5_1 row: the stage's two raw 24-byte blocks (d, m, qh,
/// 16 code bytes each). 48 bytes (12 dwords) spread the 16 rows a fragment
/// read touches over distinct banks.
constexpr int kQ51Stride = 48;
/// Bytes per staged Q8_0 row: the stage's two raw 34-byte blocks as 17
/// words (d0, codes 0-31, d1, codes 0-31; the first block's codes start two
/// bytes into a word), padded to 80 bytes (20 dwords, conflict-free).
constexpr int kQ80Stride = 80;
constexpr int kQ80Words = 17;
/// Bytes per staged binary16 row: the stage's 64 values, padded to 144 bytes
/// (36 dwords, conflict-free fragment reads).
constexpr int kF16Stride = 144;
/// Bytes per staged Q4_0 row: the stage's two raw 18-byte blocks as nine
/// words (d0, codes 0-15, d1, codes 0-15; the first block's codes start two
/// bytes into a word, the second's on a word), padded like Q5_1 to 48 bytes.
constexpr int kQ40Stride = 48;
constexpr int kQ40Words = 9;
/// Bytes per staged Q5_0 row: the stage's two raw 22-byte blocks as 11
/// words (d0, qh0, codes 0-15, d1, qh1, codes 0-15; the first block's qh and
/// codes start two bytes into a word, the second's on a word), padded like
/// Q5_1 to 48 bytes.
constexpr int kQ50Stride = 48;
constexpr int kQ50Words = 11;

template<ExpertFormat F>
constexpr int kDownStride = F == ExpertFormat::kQ5_1   ? kQ51Stride
                            : F == ExpertFormat::kQ8_0 ? kQ80Stride
                            : F == ExpertFormat::kQ4_0 ? kQ40Stride
                            : F == ExpertFormat::kQ5_0 ? kQ50Stride
                                                       : kF16Stride;
template<ExpertFormat F>
constexpr int kDownBlockBytes = F == ExpertFormat::kQ5_1   ? 24
                                : F == ExpertFormat::kQ8_0 ? 34
                                : F == ExpertFormat::kQ4_0 ? 18
                                : F == ExpertFormat::kQ5_0 ? 22
                                                           : 64;
/// Words a word-staged format (Q8_0, Q4_0, Q5_0) keeps per row and stage.
template<ExpertFormat F>
constexpr int kDownWords = F == ExpertFormat::kQ8_0   ? kQ80Words
                           : F == ExpertFormat::kQ5_0 ? kQ50Words
                                                      : kQ40Words;
/// Floats per token row of the down kernel's output transpose (128 rows plus
/// padding: 132 = 4 mod 64 puts a tile's writes on distinct banks).
constexpr int kOutStride = kRowsPerBlock + 4;

/// Q5_1 K block (32 values) from its staged raw bytes as two fragments:
/// element j < 16 is the low nibble of qs[j] plus bit j of qh as 16, j >= 16
/// the high nibble of qs[j - 16] plus bit j; weight q * d + m, one binary16
/// FMA as in Flash-Next's routed GEMM.
__device__ __forceinline__ void DecodeQ51(const std::uint8_t* block, bool live,
                                          v16h* lo, v16h* hi) {
  const uint2 head = *reinterpret_cast<const uint2*>(block);
  const uint2 q0 = *reinterpret_cast<const uint2*>(block + 8);
  const uint2 q1 = *reinterpret_cast<const uint2*>(block + 16);
  const std::uint32_t qs[4] = {q0.x, q0.y, q1.x, q1.y};
  const std::uint32_t qh = head.y;
  const __half2 dm = __builtin_bit_cast(__half2, live ? head.x : 0U);
  const __half2 scale = __low2half2(dm);
  const __half2 bias = __high2half2(dm);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t bits = qh >> (16 * part + 4 * i);
      // Bit b of `bits` (b < 4) to bit 4 of byte b.
      const std::uint32_t high = ((bits & 1U) | ((bits & 2U) << 7U) |
                                  ((bits & 4U) << 14U) | ((bits & 8U) << 21U))
                                 << 4U;
      const std::uint32_t codes = ((qs[i] >> (4 * part)) & 0x0F0F0F0FU) | high;
      CodesToHalvesAffine(codes, scale, bias, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q8_0 K block s (< 2) of a staged row's 17 words as two fragments: codes
/// flipped to q + 128 and taken back out by the magic, weight q * d, the
/// binary16 FMA of Flash-Next's routed GEMM.
__device__ __forceinline__ void DecodeQ80(const std::uint32_t (&w)[kQ80Words],
                                          int s, bool live, v16h* lo,
                                          v16h* hi) {
  std::uint32_t codes[8];
  std::uint32_t d_bits = 0;
  if (s == 0) {
    d_bits = w[0] & 0xFFFFU;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      codes[i] = __builtin_amdgcn_alignbit(w[i + 1], w[i], 16);
    }
  } else {
    d_bits = w[8] >> 16U;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      codes[i] = w[9 + i];
    }
  }
  const __half2 dm = __builtin_bit_cast(__half2, live ? d_bits : 0U);
  const __half2 scale = __low2half2(dm);
  const __half2 bias = __high2half2(dm);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      CodesToHalvesAffine(codes[4 * part + i] ^ 0x80808080U, scale, bias,
                          &h[2 * i], 128.0F);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q4_0 K block s (< 2) of a staged row's nine words as two fragments: the
/// low nibbles of its 16 code bytes (values 0-15), then the high nibbles
/// (16-31); weight (q - 8) * d, exact before the binary16 rounding.
__device__ __forceinline__ void DecodeQ40(const std::uint32_t (&w)[kQ40Words],
                                          int s, bool live, v16h* lo,
                                          v16h* hi) {
  std::uint32_t codes[4];
  std::uint32_t d_bits = 0;
  if (s == 0) {
    d_bits = w[0] & 0xFFFFU;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = __builtin_amdgcn_alignbit(w[i + 1], w[i], 16);
    }
  } else {
    d_bits = w[4] >> 16U;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = w[5 + i];
    }
  }
  const __half2 scale = __half2half2(__builtin_bit_cast(
      __half, static_cast<std::uint16_t>(live ? d_bits : 0U)));
  const __half2 offset = __float2half2_rn(1032.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      CodesToHalves((codes[i] >> (4 * part)) & 0x0F0F0F0FU, offset, scale,
                    &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q5_0 K block s (< 2) of a staged row's 11 words as two fragments: Q5_1's
/// element order and fifth bits, weight (q - 16) * d, exact before the
/// binary16 rounding.
__device__ __forceinline__ void DecodeQ50(const std::uint32_t (&w)[kQ50Words],
                                          int s, bool live, v16h* lo,
                                          v16h* hi) {
  std::uint32_t codes[4];
  std::uint32_t d_bits = 0;
  std::uint32_t qh = 0;
  if (s == 0) {
    d_bits = w[0] & 0xFFFFU;
    qh = __builtin_amdgcn_alignbit(w[1], w[0], 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = __builtin_amdgcn_alignbit(w[i + 2], w[i + 1], 16);
    }
  } else {
    d_bits = w[5] >> 16U;
    qh = w[6];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = w[7 + i];
    }
  }
  const __half2 scale = __half2half2(__builtin_bit_cast(
      __half, static_cast<std::uint16_t>(live ? d_bits : 0U)));
  const __half2 offset = __float2half2_rn(1040.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t bits = qh >> (16 * part + 4 * i);
      // Bit b of `bits` (b < 4) to bit 4 of byte b, as DecodeQ51.
      const std::uint32_t high = ((bits & 1U) | ((bits & 2U) << 7U) |
                                  ((bits & 4U) << 14U) | ((bits & 8U) << 21U))
                                 << 4U;
      CodesToHalves(((codes[i] >> (4 * part)) & 0x0F0F0F0FU) | high, offset,
                    scale, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Routed Q4_0 / Q5_0 / Q5_1 / Q8_0 / binary16 down projection with FP32
/// outputs: the
/// layout and tiling of RoutedHalfKQuantKernel over 64-value stages (three
/// 16-byte pieces per Q5_1 row, consecutive threads on consecutive pieces).
/// Each token tile's results are transposed through LDS so a token's 128
/// outputs leave as one contiguous 512-byte store. kGeGlu (Q4_0, Q8_0,
/// binary16):
/// the gate/up projection with RoutedHalfKQuantKernel's row split and GeGLU
/// epilogue into `out_half`.
template<ExpertFormat F, int kTileTokens, bool kGeGlu = false,
         bool kDense = false>
__global__ void __launch_bounds__(kThreads)
    RoutedHalfDownKernel(const std::uint8_t* __restrict__ w,
                         const __half* __restrict__ x,
                         const std::int32_t* __restrict__ tiles,
                         const std::int32_t* __restrict__ pad_bounds,
                         const std::int32_t* __restrict__ rows_in,
                         const std::int32_t* __restrict__ rows_out,
                         float* __restrict__ out, __half* __restrict__ out_half,
                         std::uint32_t m, std::uint32_t k,
                         std::uint32_t dense_rows = 0,
                         std::uint32_t dense_group = 0) {
  constexpr bool kQ8 = F == ExpertFormat::kQ8_0;
  // Q8_0, Q4_0 and Q5_0 rows are only 2-byte aligned: staged as words.
  constexpr bool kSmall = F == ExpertFormat::kQ4_0 || F == ExpertFormat::kQ5_0;
  constexpr bool kWordStaged = kQ8 || kSmall;
  constexpr int kRowWords = kDownWords<F>;
  constexpr bool kF16 = F == ExpertFormat::kF16;
  constexpr int kStride = kDownStride<F>;
  constexpr int kStageBytes = 2 * kDownBlockBytes<F>;
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kPieces = kRowsPerBlock * kQ51Stride / 16;  // per stage
  constexpr int kActChunks = kTileTokens * kDownStageK / 8;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  constexpr int kActBytes = kTileTokens * kDownActStride * 2;
  constexpr int kOutBytes = 16 * kOutStride * 4;
  __shared__
      __attribute__((aligned(16))) std::uint8_t s_rows[kRowsPerBlock * kStride];
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_act_bytes[kActBytes > kOutBytes ? kActBytes : kOutBytes];
  auto* s_act = reinterpret_cast<__half*>(s_act_bytes);
  int expert = 0;
  int t_local = 0;
  int bucket_begin = 0;
  int bucket_rows = 0;
  unsigned row_block = blockIdx.x;
  if constexpr (kDense) {
    // RoutedHalfKQuantKernel's dense raster.
    const unsigned token_tiles = (dense_rows + kTileTokens - 1) / kTileTokens;
    const unsigned span = dense_group * (gridDim.x / token_tiles);
    const unsigned first = blockIdx.x / span * dense_group;
    const unsigned width = min(dense_group, token_tiles - first);
    const unsigned within = blockIdx.x % span;
    t_local = static_cast<int>(first + within % width) * kTileTokens;
    row_block = within / width;
    bucket_rows = static_cast<int>(dense_rows);
  } else {
    const std::int32_t tile = tiles[blockIdx.y];
    expert = tile & 0xFFFF;
    t_local = (tile >> 16) * kTileTokens;
    bucket_begin = pad_bounds[expert];
    bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  }
  const auto source_row = [&](int t) {
    return kDense ? t : rows_in[bucket_begin + t];
  };
  const auto dest_row = [&](int t) {
    return kDense ? t : rows_out[bucket_begin + t];
  };
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::uint32_t row0 = row_block * kRowsPerBlock;
  const std::uint32_t half_m = m / 2;
  const std::size_t row_bytes = std::size_t{k} / 32 * kDownBlockBytes<F>;
  const std::uint8_t* w_expert =
      w + std::size_t{static_cast<std::uint32_t>(expert)} * m * row_bytes;
  // Block row l's matrix row: kGeGlu, gate rows then their up rows.
  const auto matrix_row = [&](int l) -> std::uint32_t {
    if constexpr (kGeGlu) {
      return (l >= kRowsPerBlock / 2 ? half_m : 0U) +
             row_block * (kRowsPerBlock / 2) +
             static_cast<std::uint32_t>(l % (kRowsPerBlock / 2));
    } else {
      return row0 + static_cast<std::uint32_t>(l);
    }
  };
  const auto row_live = [&](int l) {
    if constexpr (kGeGlu) {
      return row_block * (kRowsPerBlock / 2) +
                 static_cast<std::uint32_t>(l % (kRowsPerBlock / 2)) <
             half_m;
    } else {
      return matrix_row(l) < m;
    }
  };

  // Weight fetch. Q5_1: piece c (tid, and tid + 256 for tid < 128) is bytes
  // 16 (c % 3) of row c / 3's stage. Q8_0 / Q4_0 / Q5_0: word c = tid +
  // 256 i is word c % 17 (9, 11) of row c / 17 (9, 11)'s stage.
  // Binary16: piece c = tid + 256 i is bytes 16 (c % 8) of row c / 8's stage.
  static_assert(kPieces > kThreads && kPieces <= 2 * kThreads);
  constexpr int kWords = kRowsPerBlock * kRowWords;
  constexpr int kWordsPer =
      kWordStaged ? (kWords + kThreads - 1) / kThreads : 1;
  constexpr int kHalfPieces = kRowsPerBlock * 8;
  constexpr int kHalfPer = kF16 ? kHalfPieces / kThreads : 1;
  const auto row_src = [&](int r_local) {
    return w_expert +
           std::size_t{row_live(r_local) ? matrix_row(r_local) : m - 1} *
               row_bytes;
  };
  const bool f_second = tid + kThreads < kPieces;
  const std::uint8_t* f_src0 = row_src(tid / 3) + (tid % 3) * 16;
  const int c1 = f_second ? tid + kThreads : tid;
  const std::uint8_t* f_src1 = row_src(c1 / 3) + (c1 % 3) * 16;
  uint4 f_data0;
  uint4 f_data1;
  std::uint32_t f_words[kWordsPer];
  // Binary16 stages as plain words: a uint4 array here stayed in scratch.
  std::uint32_t f_pieces[4 * kHalfPer];
  const auto word_of = [&](int i) {
    const int c = tid + i * kThreads;
    return c < kWords ? c : kWords - 1;
  };
  const auto fetch_weights = [&](int stage) {
    if constexpr (kF16) {
#pragma unroll
      for (int i = 0; i < kHalfPer; ++i) {
        const int c = tid + i * kThreads;
        const uint4 piece = *reinterpret_cast<const uint4*>(
            row_src(c / 8) + stage * kStageBytes + (c % 8) * 16);
        f_pieces[4 * i] = piece.x;
        f_pieces[4 * i + 1] = piece.y;
        f_pieces[4 * i + 2] = piece.z;
        f_pieces[4 * i + 3] = piece.w;
      }
    } else if constexpr (kWordStaged) {
#pragma unroll
      for (int i = 0; i < kWordsPer; ++i) {
        const int c = word_of(i);
        f_words[i] = *reinterpret_cast<const std::uint32_t*>(
            row_src(c / kRowWords) + stage * kStageBytes + (c % kRowWords) * 4);
      }
    } else {
      f_data0 = *reinterpret_cast<const uint4*>(f_src0 + stage * kStageBytes);
      f_data1 = *reinterpret_cast<const uint4*>(f_src1 + stage * kStageBytes);
    }
  };
  const auto commit_weights = [&] {
    if constexpr (kF16) {
#pragma unroll
      for (int i = 0; i < kHalfPer; ++i) {
        const int c = tid + i * kThreads;
        *reinterpret_cast<uint4*>(s_rows + (c / 8) * kStride + (c % 8) * 16) =
            make_uint4(f_pieces[4 * i], f_pieces[4 * i + 1],
                       f_pieces[4 * i + 2], f_pieces[4 * i + 3]);
      }
    } else if constexpr (kWordStaged) {
#pragma unroll
      for (int i = 0; i < kWordsPer; ++i) {
        const int c = tid + i * kThreads;
        if (c < kWords) {
          *reinterpret_cast<std::uint32_t*>(s_rows + (c / kRowWords) * kStride +
                                            (c % kRowWords) * 4) = f_words[i];
        }
      }
    } else {
      *reinterpret_cast<uint4*>(s_rows + (tid / 3) * kStride + (tid % 3) * 16) =
          f_data0;
      if (f_second) {
        *reinterpret_cast<uint4*>(s_rows + (c1 / 3) * kStride + (c1 % 3) * 16) =
            f_data1;
      }
    }
  };

  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / 8;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = source_row(t_local + t);
      if (src >= 0) {
        a_src[i] = src * static_cast<std::int32_t>(k) + (chunk % 8) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] = a_src[i] >= 0 ? *reinterpret_cast<const uint4*>(
                                      x + a_src[i] + stage * kDownStageK)
                                : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / 8) * kDownActStride +
                                  (chunk % 8) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const bool live = row_live(wave * 16 + sub);
  const int stages = static_cast<int>(k / kDownStageK);
  fetch_weights(0);
  fetch_act(0);
  for (int stage = 0; stage < stages; ++stage) {
    commit_weights();
    commit_act();
    __syncthreads();
    if (stage + 1 < stages) {
      fetch_weights(stage + 1);
      fetch_act(stage + 1);
    }
    const std::uint8_t* row = s_rows + (wave * 16 + sub) * kStride;
    std::uint32_t words[kQ80Words];
    if constexpr (kQ8) {
      const auto* r4 = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = r4[i];
        words[4 * i] = v.x;
        words[4 * i + 1] = v.y;
        words[4 * i + 2] = v.z;
        words[4 * i + 3] = v.w;
      }
      words[16] = reinterpret_cast<const std::uint32_t*>(row)[16];
    }
    std::uint32_t small_words[kSmall ? kRowWords : 1];
    if constexpr (kSmall) {
      const auto* r4 = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 v = r4[i];
        small_words[4 * i] = v.x;
        small_words[4 * i + 1] = v.y;
        small_words[4 * i + 2] = v.z;
        small_words[4 * i + 3] = v.w;
      }
#pragma unroll
      for (int i = 8; i < kRowWords; ++i) {
        small_words[i] = reinterpret_cast<const std::uint32_t*>(row)[i];
      }
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      v16h a_lo;
      v16h a_hi;
      if constexpr (kF16) {
        // Padding rows (fetched from the last row) contribute zeros.
        using v8u =
            std::uint32_t __attribute__((ext_vector_type(8), aligned(16)));
        const auto* v = reinterpret_cast<const v8u*>(row + 64 * s);
        const v8u zero = {};
        a_lo = __builtin_bit_cast(v16h, live ? v[0] : zero);
        a_hi = __builtin_bit_cast(v16h, live ? v[1] : zero);
      } else if constexpr (kQ8) {
        DecodeQ80(words, s, live, &a_lo, &a_hi);
      } else if constexpr (F == ExpertFormat::kQ4_0) {
        DecodeQ40(small_words, s, live, &a_lo, &a_hi);
      } else if constexpr (F == ExpertFormat::kQ5_0) {
        DecodeQ50(small_words, s, live, &a_lo, &a_hi);
      } else {
        DecodeQ51(row + 24 * s, live, &a_lo, &a_hi);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kDownActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(a_lo, b_lo, acc[j]);
          acc[j] = Wmma(a_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
  }
  if constexpr (kGeGlu) {
    // Up waves hand their binary16 rows to the gate waves through LDS, as in
    // RoutedHalfKQuantKernel.
    static_assert(4 * kTokTiles * 8 * kWave * 2 <=
                  static_cast<int>(sizeof(s_act_bytes)));
    if (wave >= 4) {
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_act[(((wave - 4) * kTokTiles + j) * 8 + i) * kWave + lane] =
              SaturatedHalf(acc[j][i]);
        }
      }
    }
    __syncthreads();
    if (wave >= 4) {
      return;
    }
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      const int t = t_local + j * 16 + sub;
      if (t >= bucket_rows) {
        continue;
      }
      const std::int32_t dst = dest_row(t);
      if (dst < 0) {
        continue;
      }
      const std::size_t base =
          std::size_t{static_cast<std::uint32_t>(dst)} * half_m;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t r =
            row_block * (kRowsPerBlock / 2) +
            static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
        if (r < half_m) {
          const __half up =
              s_act[((wave * kTokTiles + j) * 8 + i) * kWave + lane];
          const float v = GeGluValue(__half2float(SaturatedHalf(acc[j][i])),
                                     __half2float(up));
          out_half[base + r] = SaturatedHalf(v);
        }
      }
    }
    return;
  }
  // Epilogue per token tile: [token][row] through LDS, then 16 threads per
  // token store its 128 rows contiguously.
  auto* s_out = reinterpret_cast<float*>(s_act_bytes);
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tiles) {
      break;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_out[sub * kOutStride + wave * 16 + 2 * i + half] = acc[j][i];
    }
    __syncthreads();
    const int tt = tid / 16;
    const int t = t_local + j * 16 + tt;
    const std::int32_t dst = t < bucket_rows ? dest_row(t) : -1;
    if (dst >= 0) {
      const int r = (tid % 16) * 8;
      const float* src = s_out + tt * kOutStride + r;
      float* o = out + std::size_t{static_cast<std::uint32_t>(dst)} * m + row0 +
                 static_cast<std::uint32_t>(r);
      if (row0 + static_cast<std::uint32_t>(r) + 8 <= m) {
        reinterpret_cast<float4*>(o)[0] = *reinterpret_cast<const float4*>(src);
        reinterpret_cast<float4*>(o)[1] =
            *reinterpret_cast<const float4*>(src + 4);
      } else {
        for (int q = 0; q < 8; ++q) {
          if (row0 + static_cast<std::uint32_t>(r + q) < m) {
            o[q] = src[q];
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

bool LaunchDenseHalfGemm(ExpertFormat format, const void* w, const void* x,
                         float* out, void* out_half, std::uint32_t rows,
                         std::uint32_t m, std::uint32_t k, hipStream_t stream,
                         bool geglu) {
  // Q4_0 and Q5_0 take the down kernel's 64-value stages.
  const bool blockwise =
      format == ExpertFormat::kQ4_0 || format == ExpertFormat::kQ5_0;
  if (rows == 0 || k % (blockwise ? kDownStageK : 256) != 0 ||
      (out == nullptr) == (out_half == nullptr) ||
      (geglu && (out_half == nullptr || m % 2 != 0)) ||
      (blockwise && out_half != nullptr && !geglu)) {
    return false;
  }
  // Tile and sweep width are shared with Flash-Next's dense launch.
  constexpr int kTile = kHalfKQuantDenseTile;
  constexpr std::uint32_t kGroup = kHalfKQuantDenseGroup;
  const std::uint32_t token_tiles = (rows + kTile - 1) / kTile;
  const std::uint32_t row_blocks =
      geglu ? (m / 2 + kRowsPerBlock / 2 - 1) / (kRowsPerBlock / 2)
            : (m + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 grid(token_tiles * row_blocks);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x);
  auto* oh = static_cast<__half*>(out_half);
  const auto launch_blockwise = [&]<ExpertFormat F>() {
    if (geglu) {
      RoutedHalfDownKernel<F, kTile, true, true><<<grid, kThreads, 0, stream>>>(
          wb, xh, nullptr, nullptr, nullptr, nullptr, nullptr, oh, m, k, rows,
          kGroup);
    } else {
      RoutedHalfDownKernel<F, kTile, false, true>
          <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                          nullptr, out, nullptr, m, k, rows,
                                          kGroup);
    }
  };
  const auto launch = [&]<ExpertFormat F>() {
    if (geglu) {
      RoutedHalfKQuantKernel<F, kTile, true, true>
          <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                          nullptr, out, oh, m, k, rows, kGroup);
    } else {
      RoutedHalfKQuantKernel<F, kTile, false, true>
          <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                          nullptr, out, oh, m, k, rows, kGroup);
    }
  };
  switch (format) {
    case ExpertFormat::kQ4_K:
      launch.template operator()<ExpertFormat::kQ4_K>();
      return true;
    case ExpertFormat::kQ5_K:
      launch.template operator()<ExpertFormat::kQ5_K>();
      return true;
    case ExpertFormat::kQ6_K:
      launch.template operator()<ExpertFormat::kQ6_K>();
      return true;
    case ExpertFormat::kQ4_0:
      launch_blockwise.template operator()<ExpertFormat::kQ4_0>();
      return true;
    case ExpertFormat::kQ5_0:
      launch_blockwise.template operator()<ExpertFormat::kQ5_0>();
      return true;
    default:
      return false;
  }
}

bool LaunchRoutedHalfGemm(ExpertFormat format, const void* w, const void* x,
                          const std::int32_t* tiles, std::uint32_t n_tiles,
                          std::uint32_t tile_rows,
                          const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, float* out,
                          void* out_half, std::uint32_t m, std::uint32_t k,
                          hipStream_t stream, bool geglu) {
  if (tile_rows != 96 || n_tiles == 0 ||
      (out == nullptr) == (out_half == nullptr) ||
      (format != ExpertFormat::kQ5_1 && format != ExpertFormat::kQ8_0 &&
       format != ExpertFormat::kF16 && format != ExpertFormat::kQ4_0 &&
       format != ExpertFormat::kQ5_0 && k % 256 != 0)) {
    return false;
  }
  if (geglu &&
      (out_half == nullptr || m % 2 != 0 || format == ExpertFormat::kQ5_1 ||
       format == ExpertFormat::kQ5_0)) {
    return false;
  }
  const dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock, n_tiles);
  const dim3 geglu_grid((m / 2 + kRowsPerBlock / 2 - 1) / (kRowsPerBlock / 2),
                        n_tiles);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x);
  auto* oh = static_cast<__half*>(out_half);
  switch (format) {
    case ExpertFormat::kQ4_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ4_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ4_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ5_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ5_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ5_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ6_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ6_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ6_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ5_1:
      if (out == nullptr || k % kDownStageK != 0) {
        return false;
      }
      RoutedHalfDownKernel<ExpertFormat::kQ5_1, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, nullptr, m, k);
      return true;
    case ExpertFormat::kQ8_0:
      if (k % kDownStageK != 0 || (out == nullptr) != geglu) {
        return false;
      }
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kQ8_0, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds,
                                                  rows_in, rows_out, nullptr,
                                                  oh, m, k);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kQ8_0, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, nullptr, m, k);
      }
      return true;
    case ExpertFormat::kQ4_0:
      if (k % kDownStageK != 0 || (out == nullptr) != geglu) {
        return false;
      }
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kQ4_0, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds,
                                                  rows_in, rows_out, nullptr,
                                                  oh, m, k);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kQ4_0, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, nullptr, m, k);
      }
      return true;
    case ExpertFormat::kQ5_0:
      if (out == nullptr || k % kDownStageK != 0) {
        return false;
      }
      RoutedHalfDownKernel<ExpertFormat::kQ5_0, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, nullptr, m, k);
      return true;
    case ExpertFormat::kF16:
      if (k % kDownStageK != 0 || (out == nullptr) != geglu) {
        return false;
      }
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kF16, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds,
                                                  rows_in, rows_out, nullptr,
                                                  oh, m, k);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kF16, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, nullptr, m, k);
      }
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::models::gemma4::rocm
