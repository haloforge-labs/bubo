#include "src/models/qwen36_a3b/decode_kernels.hpp"

#include <cmath>

namespace gufo::models::qwen36_a3b {
namespace {

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34);

__device__ __forceinline__ int LoadB2(const std::int8_t* p) {
  const auto* h = reinterpret_cast<const std::uint16_t*>(p);
  return static_cast<int>(h[0] | (static_cast<std::uint32_t>(h[1]) << 16));
}

__device__ __forceinline__ int LoadB4(const std::int8_t* p) {
  return *reinterpret_cast<const int*>(p);
}

__device__ __forceinline__ int Dp4a(int a, int b, int c) {
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
  return v;
}

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float Bf16Lo(std::uint32_t v) {
  return __uint_as_float(v << 16);
}
__device__ __forceinline__ float Bf16Hi(std::uint32_t v) {
  return __uint_as_float(v & 0xffff0000u);
}

// One lane's share of a Q8_0 row against N Q8_1 inputs: four lanes per
// 32-code block, eight codes each, eight blocks per wave iteration.
template <int N>
__device__ __forceinline__ void RowDotQ8(const Q8_0Block* __restrict__ row,
                                         const Q8_1Block* __restrict__ x,
                                         std::uint32_t x_stride,
                                         std::uint32_t blocks, int lane,
                                         float (&acc)[N]) {
  const int q = 8 * (lane & 3);
  for (std::uint32_t kb = lane >> 2; kb < blocks; kb += 8) {
    const Q8_0Block* w = row + kb;
    const float dw = __half2float(w->d);
    const int w0 = LoadB2(w->qs + q);
    const int w1 = LoadB2(w->qs + q + 4);
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const Q8_1Block* xb = x + t * x_stride + kb;
      int s = Dp4a(w0, LoadB4(xb->qs + q), 0);
      s = Dp4a(w1, LoadB4(xb->qs + q + 4), s);
      acc[t] += dw * __low2float(xb->ds) * static_cast<float>(s);
    }
  }
}

// One lane's share of a BF16 row against N F32 inputs (k % 256 == 0):
// eight weights per lane and iteration.
template <int N>
__device__ __forceinline__ void RowDotBf16(const std::uint16_t* __restrict__ row,
                                           const float* __restrict__ x,
                                           std::uint32_t x_stride,
                                           std::uint32_t k, int lane,
                                           float (&acc)[N]) {
  const auto* w = reinterpret_cast<const uint4*>(row);
  for (std::uint32_t i = lane; i < k / 8; i += 32) {
    const uint4 v = w[i];
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const auto* xv = reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(t) * x_stride + 8 * i);
      const float4 a = xv[0], b = xv[1];
      float s = acc[t];
      s = fmaf(Bf16Lo(v.x), a.x, s);
      s = fmaf(Bf16Hi(v.x), a.y, s);
      s = fmaf(Bf16Lo(v.y), a.z, s);
      s = fmaf(Bf16Hi(v.y), a.w, s);
      s = fmaf(Bf16Lo(v.z), b.x, s);
      s = fmaf(Bf16Hi(v.z), b.y, s);
      s = fmaf(Bf16Lo(v.w), b.z, s);
      s = fmaf(Bf16Hi(v.w), b.w, s);
      acc[t] = s;
    }
  }
}

template <int N>
__device__ __forceinline__ void RowDotF32(const float* __restrict__ row,
                                          const float* __restrict__ x,
                                          std::uint32_t k, int lane,
                                          float (&acc)[N]) {
  const auto* w = reinterpret_cast<const float4*>(row);
  for (std::uint32_t i = lane; i < k / 4; i += 32) {
    const float4 wv = w[i];
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float4 xv = reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(t) * k)[i];
      acc[t] = fmaf(wv.x, xv.x, acc[t]);
      acc[t] = fmaf(wv.y, xv.y, acc[t]);
      acc[t] = fmaf(wv.z, xv.z, acc[t]);
      acc[t] = fmaf(wv.w, xv.w, acc[t]);
    }
  }
}

template <int N>
__device__ __forceinline__ void RowDot(WeightKind type, const void* w,
                                       std::uint32_t row, const Q8_1Block* xq,
                                       const float* xf, std::uint32_t k,
                                       int lane, float (&acc)[N]) {
  switch (type) {
    case WeightKind::kQ8:
      RowDotQ8<N>(static_cast<const Q8_0Block*>(w) +
                      static_cast<std::size_t>(row) * (k / 32),
                  xq, k / 32, k / 32, lane, acc);
      break;
    case WeightKind::kBf16:
      RowDotBf16<N>(static_cast<const std::uint16_t*>(w) +
                        static_cast<std::size_t>(row) * k,
                    xf, k, k, lane, acc);
      break;
    default:
      RowDotF32<N>(static_cast<const float*>(w) +
                       static_cast<std::size_t>(row) * k,
                   xf, k, lane, acc);
      break;
  }
}

// Quantizes one 32-value block held one value per lane.
__device__ __forceinline__ void QuantizeLane(float v, Q8_1Block* out,
                                             int lane) {
  const float amax = WaveMax(fabsf(v));
  const float sum = WaveSum(v);
  const float d = amax / 127.0f;
  out->qs[lane] =
      amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(v / d));
  if (lane == 0) out->ds = __floats2half2_rn(d, sum);
}

// ---------------------------------------------------------------------------

constexpr int kNormThreads = 256;

__global__ __launch_bounds__(kNormThreads) void AddNormQuantKernel(
    float* __restrict__ res, const float* __restrict__ add,
    const float* __restrict__ gamma, float* __restrict__ xn,
    Q8_1Block* __restrict__ xq, std::uint32_t h, float eps,
    std::uint32_t xn_stride, std::uint32_t xq_stride) {
  __shared__ float partial[kNormThreads / 32];
  const std::uint32_t t = blockIdx.x;
  const int tid = threadIdx.x;
  float* r = res + static_cast<std::size_t>(t) * h;
  const std::uint32_t chunks = h / (4 * kNormThreads);  // float4 per thread
  float4 v[4];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t j = 0; j < 4; ++j) {
    if (j >= chunks) break;
    const std::uint32_t i = 4 * (tid + j * kNormThreads);
    float4 x = *reinterpret_cast<const float4*>(r + i);
    if (add != nullptr) {
      const float4 a = *reinterpret_cast<const float4*>(
          add + static_cast<std::size_t>(t) * h + i);
      x.x += a.x;
      x.y += a.y;
      x.z += a.z;
      x.w += a.w;
      *reinterpret_cast<float4*>(r + i) = x;
    }
    v[j] = x;
    ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  }
  ss = WaveSum(ss);
  if ((tid & 31) == 0) partial[tid >> 5] = ss;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kNormThreads / 32; ++w) total += partial[w];
  const float scale = rsqrtf(total / static_cast<float>(h) + eps);
#pragma unroll
  for (std::uint32_t j = 0; j < 4; ++j) {
    if (j >= chunks) break;
    const std::uint32_t i = 4 * (tid + j * kNormThreads);
    const float4 g = *reinterpret_cast<const float4*>(gamma + i);
    float4 y;
    y.x = v[j].x * scale * g.x;
    y.y = v[j].y * scale * g.y;
    y.z = v[j].z * scale * g.z;
    y.w = v[j].w * scale * g.w;
    if (xn != nullptr)
      *reinterpret_cast<float4*>(xn + static_cast<std::size_t>(t) * xn_stride +
                                 i) = y;
    // Eight adjacent threads hold one 32-value block.
    float amax =
        fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
    float sum = y.x + y.y + y.z + y.w;
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, off, 32));
      sum += __shfl_xor(sum, off, 32);
    }
    const float d = amax / 127.0f;
    Q8_1Block* b = xq + static_cast<std::size_t>(t) * xq_stride + i / 32;
    const int o = i % 32;
    const auto q = [&](float x) -> std::uint32_t {
      return static_cast<std::uint32_t>(
          amax == 0.0f ? 0 : static_cast<int>(roundf(x / d)));
    };
    const std::uint32_t packed = (q(y.x) & 0xffu) | ((q(y.y) & 0xffu) << 8) |
                                 ((q(y.z) & 0xffu) << 16) | (q(y.w) << 24);
    *reinterpret_cast<std::uint32_t*>(b->qs + o) = packed;
    if (o == 0) b->ds = __floats2half2_rn(d, sum);
  }
}

__global__ __launch_bounds__(256) void GateQuantKernel(
    const float* __restrict__ x, const float* __restrict__ g,
    std::uint32_t g_stride, GateMode mode, Q8_1Block* __restrict__ xq,
    float* __restrict__ xf, std::uint32_t k) {
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t block = blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (block * 32 >= k) return;
  const std::uint32_t i = block * 32 + lane;
  const float a = x[static_cast<std::size_t>(t) * k + i];
  const float b = g[static_cast<std::size_t>(t) * g_stride + i];
  const float v = mode == GateMode::kMul ? a * b : a * Sigmoid(b);
  if (xf != nullptr) xf[static_cast<std::size_t>(t) * k + i] = v;
  QuantizeLane(v, xq + static_cast<std::size_t>(t) * (k / 32) + block, lane);
}

struct MultiArgs {
  GemvSeg seg[kMaxSegs];
  std::uint32_t first[kMaxSegs + 1];
  std::uint32_t count;
};

template <int N>
__global__ __launch_bounds__(32) void MultiGemvKernel(
    MultiArgs a, const Q8_1Block* __restrict__ xq, const float* __restrict__ xf,
    std::uint32_t k) {
  const std::uint32_t g = blockIdx.x;
  std::uint32_t s = 0;
  while (s + 1 < a.count && g >= a.first[s + 1]) ++s;
  const GemvSeg& seg = a.seg[s];
  const std::uint32_t row = g - a.first[s];
  const int lane = threadIdx.x;
  float acc[N];
#pragma unroll
  for (int t = 0; t < N; ++t) acc[t] = 0.0f;
  RowDot<N>(seg.type, seg.w, row, xq, xf, k, lane, acc);
  float up[N];
  if (seg.mode == GemvMode::kGated) {
#pragma unroll
    for (int t = 0; t < N; ++t) up[t] = 0.0f;
    RowDot<N>(seg.type, seg.w2, row, xq, xf, k, lane, up);
  }
#pragma unroll
  for (int t = 0; t < N; ++t) {
    float v = WaveSum(acc[t]);
    if (seg.mode == GemvMode::kGated) v = Silu(v) * WaveSum(up[t]);
    if (lane == 0) {
      float* o = seg.out + static_cast<std::size_t>(t) * seg.out_stride + row;
      *o = seg.mode == GemvMode::kAccumulate ? *o + v : v;
    }
  }
}

__global__ __launch_bounds__(32) void ExpertsGatedKernel(
    const void* __restrict__ gate, const void* __restrict__ up,
    WeightKind type, const Q8_1Block* __restrict__ xq,
    const float* __restrict__ xf, const std::int32_t* __restrict__ ids,
    float* __restrict__ act, std::uint32_t used, std::uint32_t m,
    std::uint32_t k) {
  const std::uint32_t row = blockIdx.x, slot = blockIdx.y, t = blockIdx.z;
  const int lane = threadIdx.x;
  const std::int32_t e = ids[t * used + slot];
  const std::uint32_t r = static_cast<std::uint32_t>(e) * m + row;
  float g[1] = {0.0f}, u[1] = {0.0f};
  const Q8_1Block* x = xq + static_cast<std::size_t>(t) * (k / 32);
  const float* x32 = xf + static_cast<std::size_t>(t) * k;
  RowDot<1>(type, gate, r, x, x32, k, lane, g);
  RowDot<1>(type, up, r, x, x32, k, lane, u);
  const float gv = WaveSum(g[0]);
  const float uv = WaveSum(u[0]);
  if (lane == 0)
    act[(static_cast<std::size_t>(t) * used + slot) * m + row] = Silu(gv) * uv;
}

constexpr int kDownWaves = 8;

// Q8_1 blocks of every token's routed activations, then its shared one:
// out[(t * (used + 1) + s) * (f / 32) + kb], one 32-value block per wave.
__global__ __launch_bounds__(256) void QuantizeActsKernel(
    const float* __restrict__ act, const float* __restrict__ sh_act,
    Q8_1Block* __restrict__ out, std::uint32_t total, std::uint32_t used,
    std::uint32_t f) {
  const std::uint32_t b = blockIdx.x * 8 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (b >= total) return;
  const std::uint32_t fb = f / 32, slots = used + 1;
  const std::uint32_t t = b / (slots * fb);
  const std::uint32_t s = b / fb % slots;
  const std::uint32_t kb = b % fb;
  const float v =
      s < used
          ? act[(static_cast<std::size_t>(t) * used + s) * f + kb * 32 + lane]
          : sh_act[static_cast<std::size_t>(t) * f + kb * 32 + lane];
  QuantizeLane(v, out + b, lane);
}

template <int N>
__global__ __launch_bounds__(kDownWaves * 32) void ExpertsDownKernel(
    const void* __restrict__ down, WeightKind down_type,
    const void* __restrict__ sh_down, WeightKind sh_type,
    const float* __restrict__ act, const float* __restrict__ sh_act,
    const Q8_1Block* __restrict__ aq, const std::int32_t* __restrict__ ids,
    const float* __restrict__ weights, const float* __restrict__ router,
    std::uint32_t router_stride, std::uint32_t shared_gate,
    float* __restrict__ res, std::uint32_t used, std::uint32_t h,
    std::uint32_t f) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x & 31;
  const std::uint32_t fb = f / 32;
  const std::uint32_t slots = used + 1;
  const std::uint32_t row = blockIdx.x * kDownWaves + wave;
  if (row >= h) return;
#pragma unroll
  for (int t = 0; t < N; ++t) {
    float acc = 0.0f;
    for (std::uint32_t s = 0; s < used; ++s) {
      const std::uint32_t r =
          static_cast<std::uint32_t>(ids[t * used + s]) * h + row;
      float p[1] = {0.0f};
      if (down_type == WeightKind::kQ8) {
        RowDotQ8<1>(static_cast<const Q8_0Block*>(down) +
                        static_cast<std::size_t>(r) * fb,
                    aq + (t * slots + s) * fb, fb, fb, lane, p);
      } else {
        RowDotBf16<1>(static_cast<const std::uint16_t*>(down) +
                          static_cast<std::size_t>(r) * f,
                      act + (static_cast<std::size_t>(t) * used + s) * f, f, f,
                      lane, p);
      }
      acc = fmaf(weights[t * used + s], p[0], acc);
    }
    float p[1] = {0.0f};
    if (sh_type == WeightKind::kQ8) {
      RowDotQ8<1>(static_cast<const Q8_0Block*>(sh_down) +
                      static_cast<std::size_t>(row) * fb,
                  aq + (t * slots + used) * fb, fb, fb, lane, p);
    } else {
      RowDotBf16<1>(static_cast<const std::uint16_t*>(sh_down) +
                        static_cast<std::size_t>(row) * f,
                    sh_act + static_cast<std::size_t>(t) * f, f, f, lane, p);
    }
    acc = fmaf(Sigmoid(router[t * router_stride + shared_gate]), p[0], acc);
    acc = WaveSum(acc);
    if (lane == 0) res[static_cast<std::size_t>(t) * h + row] += acc;
  }
}

template <int N>
void LaunchMulti(const MultiArgs& a, const Q8_1Block* xq, const float* xf,
                 std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(MultiGemvKernel<N>, dim3(a.first[a.count]), dim3(32), 0,
                     stream, a, xq, xf, k);
}

template <int N>
void LaunchDown(const void* down, WeightKind down_type, const void* sh_down,
                WeightKind sh_type, const float* act, const float* sh_act,
                Q8_1Block* act_q, const std::int32_t* ids,
                const float* weights, const float* router,
                std::uint32_t router_stride, std::uint32_t shared_gate,
                float* res, std::uint32_t used, std::uint32_t h,
                std::uint32_t f, hipStream_t stream) {
  const std::uint32_t total = N * (used + 1) * (f / 32);
  hipLaunchKernelGGL(QuantizeActsKernel, dim3((total + 7) / 8), dim3(256), 0,
                     stream, act, sh_act, act_q, total, used, f);
  hipLaunchKernelGGL(ExpertsDownKernel<N>,
                     dim3((h + kDownWaves - 1) / kDownWaves),
                     dim3(kDownWaves * 32), 0, stream, down, down_type,
                     sh_down, sh_type, act, sh_act, act_q, ids, weights, router,
                     router_stride, shared_gate, res, used, h, f);
}

}  // namespace

void AddNormQuant(float* res, const float* add, const float* gamma, float* xn,
                  Q8_1Block* xq, std::uint32_t rows, std::uint32_t h, float eps,
                  hipStream_t stream, std::uint32_t xn_stride,
                  std::uint32_t xq_stride) {
  hipLaunchKernelGGL(AddNormQuantKernel, dim3(rows), dim3(kNormThreads), 0,
                     stream, res, add, gamma, xn, xq, h, eps,
                     xn_stride != 0 ? xn_stride : h,
                     xq_stride != 0 ? xq_stride : h / 32);
}

void GateQuant(const float* x, const float* g, std::uint32_t g_stride,
               GateMode mode, Q8_1Block* xq, float* xf, std::uint32_t rows,
               std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(GateQuantKernel, dim3((k / 32 + 7) / 8, rows), dim3(256),
                     0, stream, x, g, g_stride, mode, xq, xf, k);
}

void MultiGemv(const GemvSeg* segs, int count, const Q8_1Block* xq,
               const float* xf, std::uint32_t rows, std::uint32_t k,
               hipStream_t stream) {
  MultiArgs a{};
  a.count = static_cast<std::uint32_t>(count);
  a.first[0] = 0;
  for (int i = 0; i < count; ++i) {
    a.seg[i] = segs[i];
    a.first[i + 1] = a.first[i] + segs[i].rows;
  }
  switch (rows) {
    case 1: return LaunchMulti<1>(a, xq, xf, k, stream);
    case 2: return LaunchMulti<2>(a, xq, xf, k, stream);
    case 3: return LaunchMulti<3>(a, xq, xf, k, stream);
    case 4: return LaunchMulti<4>(a, xq, xf, k, stream);
    case 5: return LaunchMulti<5>(a, xq, xf, k, stream);
    case 6: return LaunchMulti<6>(a, xq, xf, k, stream);
    case 7: return LaunchMulti<7>(a, xq, xf, k, stream);
    default: return LaunchMulti<8>(a, xq, xf, k, stream);
  }
}

void ExpertsGated(const void* gate, const void* up, WeightKind type,
                  const Q8_1Block* xq, const float* xf,
                  const std::int32_t* ids, float* act, std::uint32_t rows,
                  std::uint32_t used, std::uint32_t m, std::uint32_t k,
                  hipStream_t stream) {
  hipLaunchKernelGGL(ExpertsGatedKernel, dim3(m, used, rows), dim3(32), 0,
                     stream, gate, up, type, xq, xf, ids, act, used, m, k);
}

void ExpertsDown(const void* down, WeightKind down_type, const void* sh_down,
                 WeightKind sh_type, const float* act, const float* sh_act,
                 Q8_1Block* act_q, const std::int32_t* ids,
                 const float* weights,
                 const float* router, std::uint32_t router_stride,
                 std::uint32_t shared_gate, float* res, std::uint32_t rows,
                 std::uint32_t used, std::uint32_t h, std::uint32_t f,
                 hipStream_t stream) {
  switch (rows) {
#define A3B_DOWN(n)                                                          \
  case n:                                                                    \
    return LaunchDown<n>(down, down_type, sh_down, sh_type, act, sh_act,     \
                         act_q, ids, weights, router, router_stride,         \
                         shared_gate, res, used, h, f, stream);
    A3B_DOWN(1)
    A3B_DOWN(2)
    A3B_DOWN(3)
    A3B_DOWN(4)
    A3B_DOWN(5)
    A3B_DOWN(6)
    A3B_DOWN(7)
#undef A3B_DOWN
    default:
      return LaunchDown<8>(down, down_type, sh_down, sh_type, act, sh_act,
                           act_q, ids, weights, router, router_stride,
                           shared_gate, res, used, h, f, stream);
  }
}

}  // namespace gufo::models::qwen36_a3b
