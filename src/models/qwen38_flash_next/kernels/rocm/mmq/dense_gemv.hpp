// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
#pragma once

namespace dense = gufo::models::qwen38_flash_next::dense_gemv;

struct DenseGemvMath {
  template<int alignment>
  __device__ static __forceinline__ uint32_t Word(const uint8_t* p) {
    if constexpr (alignment >= 4)
      return uint32_t(get_int_b4(p, 0));
    else
      return uint32_t(get_int_b2(p, 0));
  }
  __device__ static __forceinline__ uint16_t HalfBits(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
  }
  __device__ static __forceinline__ float Half(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const half*>(p));
  }
  __device__ static __forceinline__ dense::Pair Lookup(uint32_t word) {
    const int2 pair = get_int_from_table_16(int(word), kvalues_iq4nl);
    return {uint32_t(pair.x), uint32_t(pair.y)};
  }
  template<int type>
  __device__ static __forceinline__ dense::Pair Grid(
      int index, const uint32_t* staged = nullptr) {
    if (staged) {
      if constexpr (type == 17)
        return {staged[2 * index], staged[2 * index + 1]};
      else
        return {staged[index], 0};
    }
    if constexpr (type == 17) {
      const auto value = reinterpret_cast<const uint2*>(iq2xs_grid)[index];
      return {value.x, value.y};
    } else if constexpr (type == 18)
      return {iq3xxs_grid[index], 0};
    else
      return {iq3s_grid[index], 0};
  }
  __device__ static __forceinline__ int Dot(uint32_t w, uint32_t x, int acc) {
    return ggml_hip_dp4a(int(w), int(x), acc);
  }
  __device__ static __forceinline__ uint32_t Subtract32(uint32_t word) {
    return uint32_t(__vsubss4(int(word), 0x20202020));
  }
  __device__ static __forceinline__ float Mul(float a, float b) {
    return __fmul_rn(a, b);
  }
  __device__ static __forceinline__ float Fma(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
  }
  __device__ static __forceinline__ float Add(float a, float b) {
    return __fadd_rn(a, b);
  }
};

// One wave retains the scalar kernel's lane/block mapping for every input.
// Each packed fragment is loaded and decoded before the exact-N input loop.
template<int type, int tokens, int outputs, bool stage_x = false,
         bool stage_grid = false, int partitions = 1, int row_waves = 4,
         bool loads_only = false>
__launch_bounds__(32 * partitions * row_waves) static __global__
    void mul_mat_vec_dense_reuse(const uint8_t* __restrict__ weights,
                                 const block_q8_1* __restrict__ input,
                                 float* __restrict__ output, int k, int rows,
                                 int input_stride) {
  constexpr int qk = dense::BlockSize(type);
  constexpr int qi = dense::Qi(type);
  constexpr int vdr = dense::Vdr(type);
  constexpr int step = vdr * 32 / qi;
  constexpr int threads = 32 * partitions * row_waves;
  constexpr int lanes_per_row = 32 / partitions;
  constexpr int rows_per_wave = partitions;
  constexpr int cta_rows = row_waves * rows_per_wave;
  constexpr int xblocks = step * (qk / 32);
  constexpr int xwords = xblocks * 9;
  static_assert(partitions == 1 || (partitions == 4 && outputs == 1 &&
                                    dense::IsIqTarget(type) && stage_x));
  // A chunk is 1024 K values for IQ2/3: <=9 KiB at N=8. The 9-word Q8_1
  // stride permutes the LDS banks for the lane's block-strided loads. Reuse
  // the allocation after the last chunk for the exact split-K reduction.
  union Scratch {
    uint32_t x[stage_x ? tokens * xwords : 1];
    float partial[partitions == 4 ? tokens * cta_rows * 2 * 8 : 1];
  };
  __shared__ Scratch shared;
  constexpr int grid_words = type == 17 ? 1024 : type == 18 ? 256 : 512;
  __shared__ uint32_t grid[stage_grid ? grid_words : 1];
  if constexpr (stage_grid) {
    static_assert(dense::IsIqTarget(type));
    for (int i = threadIdx.x; i < grid_words; i += threads) {
      if constexpr (type == 17)
        grid[i] = reinterpret_cast<const uint32_t*>(iq2xs_grid)[i];
      else if constexpr (type == 18)
        grid[i] = iq3xxs_grid[i];
      else
        grid[i] = iq3s_grid[i];
    }
    __syncthreads();
  }
  const int lane = threadIdx.x % 32;
  const int wave = stage_x
                       ? __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 32)
                       : int(threadIdx.x) / 32;
  const int partition = partitions == 1 ? 0 : wave / row_waves;
  const int local_row = partitions == 1 ? wave
                                        : (wave % row_waves) * rows_per_wave +
                                              lane / lanes_per_row;
  const int row0 = (blockIdx.x * cta_rows + local_row) * outputs;
  if constexpr (!stage_x) {
    if (row0 >= rows)
      return;
  }
  const int original_lane =
      partitions == 1 ? lane : partition * lanes_per_row + lane % lanes_per_row;
  const int blocks = k / qk;
  const int iqs = vdr * (original_lane % (qi / vdr));
  const int first_kb = original_lane / (qi / vdr);
  float sums[outputs][tokens]{};
  // Bench-only loads rung (ablation control): the weight loads and the
  // fragment decode, no activations and no products; the output carries the
  // checksum so the loads cannot be removed.
  static_assert(!loads_only || (tokens == 1 && outputs == 1 && partitions == 1 &&
                                !stage_x && !stage_grid));
  uint32_t sink = 0;
  // All staging waves execute every chunk, including invalid output rows and
  // K tails. Without LDS this is the original lane's loop, unchanged.
  for (int pos = stage_x ? 0 : first_kb; pos < blocks; pos += step) {
    const int kb = pos + (stage_x ? first_kb : 0);
    if constexpr (stage_x) {
      const int tile = pos;
      for (int i = threadIdx.x; i < tokens * xwords; i += threads) {
        const int t = i / xwords;
        const int word = i % xwords;
        const int src_block = tile * (qk / 32) + word / 9;
        if (src_block < k / 32)
          shared.x[i] = reinterpret_cast<const uint32_t*>(
              input + t * input_stride + src_block)[word % 9];
      }
      __syncthreads();
    }
    if (kb < blocks && row0 < rows) {
#pragma unroll
      for (int o = 0; o < outputs; ++o) {
        const int row = min(row0 + o, rows - 1);
        const dense::Fragment<type, DenseGemvMath> fragment(
            weights + (size_t(row) * blocks + kb) * dense::BlockBytes(type),
            iqs, stage_grid ? grid : nullptr);
        if constexpr (loads_only) {
          uint32_t bits = __float_as_uint(fragment.scale) ^
                          __float_as_uint(fragment.minimum) ^
                          uint32_t(fragment.subscale) ^
                          uint32_t(fragment.scales[0] ^ fragment.scales[1] ^
                                   fragment.minima[0] ^ fragment.minima[1]);
          for (const auto& v : fragment.values)
            bits ^= v.lo ^ v.hi;
          sink ^= bits;
        }
#pragma unroll
        for (int t = 0; t < (loads_only ? 0 : tokens); ++t) {
          const auto* x =
              stage_x ? reinterpret_cast<const uint8_t*>(
                            shared.x + t * xwords + first_kb * (qk / 32) * 9)
                      : reinterpret_cast<const uint8_t*>(
                            input + t * input_stride + kb * (qk / 32));
          sums[o][t] = fragment.Accumulate(x, iqs, sums[o][t]);
          // Bound activation operand lifetimes instead of hoisting all N
          // rows into VGPRs. This is a compiler scheduling barrier only.
          if constexpr (stage_x)
            __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
    if constexpr (stage_x)
      __syncthreads();
  }
  if constexpr (partitions == 4) {
    // A conventional sum of K partitions would change rounding. Partition g
    // owns ORIGINAL lanes 8*g+[0..7], each with kb=g,g+4,... . Reconstruct
    // butterfly offsets 16 and 8 through LDS, then 4,2,1 within each 8 lanes.
    const int index = (local_row * 2 + (partition & 1)) * 8 + lane % 8;
    constexpr int pitch = cta_rows * 2 * 8;
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (partition >= 2)
        shared.partial[t * pitch + index] = sums[0][t];
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (partition < 2) {
        sums[0][t] =
            DenseGemvMath::Add(sums[0][t], shared.partial[t * pitch + index]);
        if (partition == 1)
          shared.partial[t * pitch + index] = sums[0][t];
      }
    }
    __syncthreads();
    if (partition == 0) {
#pragma unroll
      for (int t = 0; t < tokens; ++t) {
        const float pair = DenseGemvMath::Add(
            sums[0][t], shared.partial[t * pitch + index + 8]);
        const float value = warp_reduce_sum<8>(pair);
        if (lane % 8 == row0 % 2 && row0 < rows)
          output[t * rows + row0] = isfinite(value) ? value : 0.0F;
      }
    }
    return;
  }
  if constexpr (loads_only)
    sums[0][0] = float((sink ^ (sink >> 16)) & 0xFFFFU);  // all 32 bits matter
#pragma unroll
  for (int o = 0; o < outputs; ++o) {
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      const float value = warp_reduce_sum<32>(sums[o][t]);
      if (lane == ((row0 + o) % 2) && row0 + o < rows)
        output[t * rows + row0 + o] = isfinite(value) ? value : 0.0F;
    }
  }
}

template<int type, int outputs, int tokens = 1, bool stage_x = false,
         bool stage_grid = false, int partitions = 1, int row_waves = 4>
static void launch_dense_reuse(const void* w, const block_q8_1* x, float* y,
                               int k, int rows, int n, int stride,
                               hipStream_t stream) {
  if (n == tokens)
    mul_mat_vec_dense_reuse<type, tokens, outputs, stage_x, stage_grid,
                            partitions, row_waves>
        <<<(rows + row_waves * partitions * outputs - 1) /
               (row_waves * partitions * outputs),
           32 * partitions * row_waves, 0, stream>>>(
            static_cast<const uint8_t*>(w), x, y, k, rows, stride);
  else if constexpr (tokens < 8)
    launch_dense_reuse<type, outputs, tokens + 1, stage_x, stage_grid,
                       partitions, row_waves>(w, x, y, k, rows, n, stride,
                                              stream);
}

template<int type>
static void launch_dense_iq_variant(const void* w, const block_q8_1* x,
                                    float* y, int k, int rows, int n,
                                    int stride, hipStream_t stream,
                                    int variant) {
  switch (static_cast<dense::Variant>(variant)) {
    case dense::Variant::kReuse1Cap6: {
      const int first = dense::Cap6FirstWidth(n);
      launch_dense_reuse<type, 1>(w, x, y, k, rows, first, stride, stream);
      if (first < n && hipPeekAtLastError() == hipSuccess)
        launch_dense_reuse<type, 1>(w, x + first * stride, y + first * rows, k,
                                    rows, n - first, stride, stream);
      break;
    }
    case dense::Variant::kReuse1Lds:
      launch_dense_reuse<type, 1, 1, true>(w, x, y, k, rows, n, stride, stream);
      break;
    case dense::Variant::kReuse1LdsGrid:
      launch_dense_reuse<type, 1, 1, false, true>(w, x, y, k, rows, n, stride,
                                                  stream);
      break;
    case dense::Variant::kReuse1SplitKLds:
      launch_dense_reuse<type, 1, 1, true, false, 4, 8>(w, x, y, k, rows, n,
                                                        stride, stream);
      break;
    default:
      GGML_ABORT("invalid dense IQ variant");
  }
}

bool mul_mat_vec_dense_reuse_dispatch(const void* w, int type,
                                      const block_q8_1* x, float* y, int k,
                                      int rows, int n, int stride,
                                      hipStream_t stream, int variant) {
  if (!dense::Covers(type, n, k) || variant < 1 ||
      variant >= dense::kVariantCount)
    return false;
  if (variant >= int(dense::Variant::kReuse1Cap6)) {
    switch (type) {
      case 21:
        launch_dense_iq_variant<21>(w, x, y, k, rows, n, stride, stream,
                                    variant);
        return true;
      case 18:
        launch_dense_iq_variant<18>(w, x, y, k, rows, n, stride, stream,
                                    variant);
        return true;
      case 17:
        launch_dense_iq_variant<17>(w, x, y, k, rows, n, stride, stream,
                                    variant);
        return true;
      default:
        return false;
    }
  }
#define DENSE_REUSE_CASE(T)                                          \
  case T:                                                            \
    if (variant == 1)                                                \
      launch_dense_reuse<T, 1>(w, x, y, k, rows, n, stride, stream); \
    else                                                             \
      launch_dense_reuse<T, 2>(w, x, y, k, rows, n, stride, stream); \
    return true
  switch (type) {
    DENSE_REUSE_CASE(23);
    DENSE_REUSE_CASE(20);
    DENSE_REUSE_CASE(12);
    DENSE_REUSE_CASE(13);
    DENSE_REUSE_CASE(7);
    DENSE_REUSE_CASE(14);
    DENSE_REUSE_CASE(21);
    DENSE_REUSE_CASE(18);
    DENSE_REUSE_CASE(17);
    default:
      return false;
  }
#undef DENSE_REUSE_CASE
}
