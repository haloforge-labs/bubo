// SPDX-License-Identifier: MIT
#pragma once

// Fragment is the same decoder used by reuse1. Decode to signed bytes while
// staging, then consume only full K16 fragments in the matrix-unit loop.
template<int type>
struct KQuantTileSink {
  uint32_t* codes;
  float* scales;
  float* minima;
  int row;
  __device__ __forceinline__ void Word(int element, uint32_t value) {
    codes[dense::KQuantStageWordIndex(row, element)] = value;
  }
  __device__ __forceinline__ void Scale(int group, float value) {
    scales[row * (type == 14 ? 17 : 9) + group] = value;
  }
  __device__ __forceinline__ void Minimum(int group, float value) {
    minima[row * 9 + group] = value;
  }
};

// Ablation stages A and A0 run the same StoreKTile with a register sink, so
// the decode ALU is shared and nothing is written to LDS.
struct KQuantChecksumSink {
  uint32_t value;
  __device__ __forceinline__ void Word(int, uint32_t word) { value ^= word; }
  __device__ __forceinline__ void Scale(int, float scale) {
    value ^= __float_as_uint(scale);
  }
  __device__ __forceinline__ void Minimum(int, float minimum) {
    value ^= __float_as_uint(minimum);
  }
};

// `stage` is bench-only (kquant_stage.hpp). The default is the production
// kernel: every `if constexpr` below keeps the original statement order.
template<int type, int stage = dense::kStageFull>
__launch_bounds__(256) static __global__
    void mul_mat_dense_kquant_wmma(const uint8_t* __restrict__ weights,
                                   const block_q8_1* __restrict__ input,
                                   float* __restrict__ output, int k, int rows,
                                   int input_stride, int tokens) {
  using Int4 = __attribute__((ext_vector_type(4))) int;
  using Int8 = __attribute__((ext_vector_type(8))) int;
  constexpr bool full = stage == dense::kStageFull;
  constexpr bool barriers = stage != dense::kStageLoadsNoBarrier &&
                            stage != dense::kStageLdsNoBarrier;
  constexpr bool to_lds = stage >= dense::kStageLds;
  constexpr bool activations = stage >= dense::kStageActivation;
  constexpr bool products = stage >= dense::kStageWmma;
  constexpr bool fan_out = stage >= dense::kStageShuffle;
  constexpr int pitch = type == 14 ? 17 : 9;
  struct Stage {
    uint32_t codes[1024];
    block_q8_1 x[8][8];
    float scales[16 * pitch];
    float minima[type == 14 ? 1 : 16 * 9];
    int sums[type == 14 ? 1 : 8 * 8];
  };
  union alignas(16) Scratch {
    Stage stage;
    float partial[256 * 8];
  };
  static_assert(sizeof(Stage) <= 8192);
  __shared__ Scratch shared;
  const int lane = threadIdx.x % 32;
  const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 32);
  const int sub = lane % 16, half = lane / 16;
  const int first_row = blockIdx.x * 16;
  const int blocks = k / 256;
  float acc[8]{};
  uint32_t sink = 0;
  for (int kb = 0; kb < blocks; ++kb) {
    constexpr int fragments = type == 14 ? 32 : 16;
    for (int task = threadIdx.x; task < 16 * fragments; task += 256) {
      const int local_row = task / fragments;
      const int iqs = (task % fragments) * (type == 14 ? 1 : 2);
      // Clamp BEFORE forming the global pointer, including M=1/15/17 tails.
      const int row = min(first_row + local_row, rows - 1);
      const dense::Fragment<type, DenseGemvMath> fragment(
          weights + (size_t(row) * blocks + kb) * dense::BlockBytes(type), iqs);
      if constexpr (to_lds) {
        KQuantTileSink<type> sink_lds{shared.stage.codes, shared.stage.scales,
                                      shared.stage.minima, local_row};
        fragment.StoreKTile(iqs, sink_lds);
      } else {
        KQuantChecksumSink sink_reg{sink};
        fragment.StoreKTile(iqs, sink_reg);
        sink = sink_reg.value;
      }
    }
    // One owner per activation block, including its integer correction.
    if constexpr (activations) {
      if (threadIdx.x < tokens * 8) {
        const int token = threadIdx.x / 8, b = threadIdx.x % 8;
        const auto x = input[token * input_stride + kb * 8 + b];
        shared.stage.x[token][b] = x;
        if constexpr (type != 14) {
          int sum = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            sum = ggml_hip_dp4a(0x01010101, get_int_b4(x.qs, j), sum);
          shared.stage.sums[token * 8 + b] = sum;
        }
      }
    }
    if constexpr (barriers)
      __syncthreads();
    if constexpr (to_lds) {
      const int code0 = dense::KQuantStageWordIndex(sub, wave * 32);
      const int code1 = dense::KQuantStageWordIndex(sub, wave * 32 + 16);
      const Int4 low = *reinterpret_cast<const Int4*>(shared.stage.codes + code0);
      const Int4 high =
          *reinterpret_cast<const Int4*>(shared.stage.codes + code1);
      float scales0[8], scales1[8], minima[8];
      const float scale0 =
          shared.stage.scales[sub * pitch + (type == 14 ? 2 * wave : wave)];
      float scale1 = 0, minimum = 0;
      if constexpr (type == 14)
        scale1 = shared.stage.scales[sub * pitch + 2 * wave + 1];
      else
        minimum = shared.stage.minima[sub * 9 + wave];
      if constexpr (!full) {
        sink ^= uint32_t(low[0] ^ low[1] ^ low[2] ^ low[3] ^ high[0] ^ high[1] ^
                         high[2] ^ high[3]) ^
                __float_as_uint(scale0) ^ __float_as_uint(scale1) ^
                __float_as_uint(minimum);
      }
      if constexpr (fan_out || full) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          scales0[i] = __shfl(scale0, 2 * i + half, 32);
          if constexpr (type == 14)
            scales1[i] = __shfl(scale1, 2 * i + half, 32);
          else
            minima[i] = __shfl(minimum, 2 * i + half, 32);
        }
      }
      if constexpr (activations) {
        const int token = min(sub, tokens - 1);
        const auto* x = &shared.stage.x[token][wave];
        const float dx = __low2float(x->ds);
        const Int4 x0 = {get_int_b4(x->qs, 0), get_int_b4(x->qs, 1),
                         get_int_b4(x->qs, 2), get_int_b4(x->qs, 3)};
        const Int4 x1 = {get_int_b4(x->qs, 4), get_int_b4(x->qs, 5),
                         get_int_b4(x->qs, 6), get_int_b4(x->qs, 7)};
        if constexpr (!products) {
          sink ^= uint32_t(x0[0] ^ x0[1] ^ x0[2] ^ x0[3] ^ x1[0] ^ x1[1] ^
                           x1[2] ^ x1[3]) ^
                  __float_as_uint(dx);
          if constexpr (type != 14)
            sink ^= uint32_t(shared.stage.sums[token * 8 + wave]);
        } else {
          Int8 dots0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, low, true, x0, Int8{}, false);
          if constexpr (type == 14) {
            const Int8 dots1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, high, true, x1, Int8{}, false);
            if constexpr (!full) {
              for (int i = 0; i < 8; ++i)
                sink ^= uint32_t(dots0[i] ^ dots1[i]);
              sink ^= __float_as_uint(dx);
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                acc[i] =
                    __fmaf_rn(dx, __fmul_rn(scales0[i], float(dots0[i])), acc[i]);
                acc[i] =
                    __fmaf_rn(dx, __fmul_rn(scales1[i], float(dots1[i])), acc[i]);
              }
            }
          } else {
            dots0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, high, true, x1, dots0, false);
            const float sum = float(shared.stage.sums[token * 8 + wave]);
            if constexpr (!full) {
              for (int i = 0; i < 8; ++i)
                sink ^= uint32_t(dots0[i]);
              sink ^= __float_as_uint(dx) ^ __float_as_uint(sum);
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                acc[i] =
                    __fmaf_rn(dx, __fmul_rn(scales0[i], float(dots0[i])), acc[i]);
                acc[i] = __fmaf_rn(-dx, __fmul_rn(minima[i], sum), acc[i]);
              }
            }
          }
        }
      }
      if constexpr (fan_out && !full) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          sink ^= __float_as_uint(scales0[i]);
          if constexpr (type == 14)
            sink ^= __float_as_uint(scales1[i]);
          else
            sink ^= __float_as_uint(minima[i]);
        }
      }
    }
    // All readers finish before the next K256 tile or output transpose.
    if constexpr (barriers)
      __syncthreads();
  }
  // The sink keeps every earlier rung alive; the epilogue is shared.
  if constexpr (!full)
    acc[0] = float((sink ^ (sink >> 16)) & 0xFFFFU);  // all 32 bits matter
  // New arithmetic is intentionally non-exact. Do not hide NaNs in compares.
  store_dense_matrix_tile<8, false>(acc, shared.partial, output, first_row,
                                    rows, tokens);
}

bool mul_mat_dense_kquant_wmma_dispatch(const void* w, int type,
                                        const block_q8_1* x, float* y, int k,
                                        int rows, int n, int stride,
                                        hipStream_t stream) {
  if (!dense::IsKQuantTarget(type) || n < 4 || n > 8 || k % 256)
    return false;
#define KQUANT_WMMA_CASE(T)                                             \
  case T:                                                               \
    mul_mat_dense_kquant_wmma<T><<<(rows + 15) / 16, 256, 0, stream>>>( \
        static_cast<const uint8_t*>(w), x, y, k, rows, stride, n);      \
    return true
  switch (type) {
    KQUANT_WMMA_CASE(12);
    KQUANT_WMMA_CASE(13);
    KQUANT_WMMA_CASE(14);
    default:
      return false;
  }
#undef KQUANT_WMMA_CASE
}

// ---- Bench-only ablation entries (kquant_stage.hpp). Not reachable from the
// production dispatch; the bench proves every launch by counter. ----
static std::atomic<unsigned long long> g_kquant_stage_launches[3][dense::kStageCount + 1][8]{};

template<int type>
static const void* KQuantStageKernel(int stage) {
  switch (stage) {
#define KQUANT_STAGE_CASE(S) \
  case S:                    \
    return reinterpret_cast<const void*>(mul_mat_dense_kquant_wmma<type, S>)
    KQUANT_STAGE_CASE(dense::kStageLoads);
    KQUANT_STAGE_CASE(dense::kStageLoadsNoBarrier);
    KQUANT_STAGE_CASE(dense::kStageLds);
    KQUANT_STAGE_CASE(dense::kStageLdsNoBarrier);
    KQUANT_STAGE_CASE(dense::kStageActivation);
    KQUANT_STAGE_CASE(dense::kStageWmma);
    KQUANT_STAGE_CASE(dense::kStageShuffle);
    KQUANT_STAGE_CASE(dense::kStageFull);
#undef KQUANT_STAGE_CASE
    default:
      return nullptr;
  }
}

static const void* KQuantStageKernel(int type, int stage) {
  switch (type) {
    case 12:
      return KQuantStageKernel<12>(stage);
    case 13:
      return KQuantStageKernel<13>(stage);
    case 14:
      return KQuantStageKernel<14>(stage);
    default:
      return nullptr;
  }
}

static int KQuantStageIndex(int type) { return type == 12 ? 0 : type == 13 ? 1 : 2; }

extern "C" unsigned long long qfn_mmq_kquant_stage_launch_count(int type,
                                                               int stage,
                                                               int width) {
  if (!dense::IsKQuantTarget(type) || stage < 1 || stage > dense::kStageCount ||
      width < 1 || width > 8)
    return 0;
  return g_kquant_stage_launches[KQuantStageIndex(type)][stage][width - 1].load(
      std::memory_order_relaxed);
}

// Resident CTAs per multiprocessor (WGP on gfx11) the runtime reports for this stage kernel with
// `extra_lds_bytes` of dynamic LDS added; negative on error.
extern "C" int qfn_mmq_kquant_stage_occupancy(int type, int stage,
                                              int extra_lds_bytes) {
  const void* kernel = KQuantStageKernel(type, stage);
  if (!kernel || extra_lds_bytes < 0)
    return -1;
  int blocks = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, kernel, 256, size_t(extra_lds_bytes)) != hipSuccess)
    return -3;
  return blocks;
}

// Launches one ablation stage and returns its kernel id (type * 256 + stage),
// or a negative code: -1 unsupported request, -3 launch error.
extern "C" int qfn_mmq_kquant_stage_launch(int type, int stage, const void* w,
                                           const void* x, float* y, int k,
                                           int rows, int n, int extra_lds_bytes,
                                           hipStream_t stream) {
  const void* kernel = KQuantStageKernel(type, stage);
  if (!kernel || !w || !x || !y || k <= 0 || k % 256 || rows <= 0 || n < 1 ||
      n > 8 || extra_lds_bytes < 0)
    return -1;
  const uint8_t* weights = static_cast<const uint8_t*>(w);
  const block_q8_1* input = static_cast<const block_q8_1*>(x);
  int input_stride = GGML_PAD(k, MATRIX_ROW_PADDING) / QK8_1;
  void* args[]{&weights, &input, &y, &k, &rows, &input_stride, &n};
  if (hipLaunchKernel(kernel, dim3((rows + 15) / 16), dim3(256), args,
                      size_t(extra_lds_bytes), stream) != hipSuccess ||
      hipGetLastError() != hipSuccess)
    return -3;
  g_kquant_stage_launches[KQuantStageIndex(type)][stage][n - 1].fetch_add(
      1, std::memory_order_relaxed);
  return type * 256 + stage;
}
