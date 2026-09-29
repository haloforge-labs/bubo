#include "src/models/qwen36_a3b/kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

constexpr unsigned kThreads = 256;

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

__global__ void AddKernel(float* res, const float* x, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) res[i] += x[i];
}

__global__ void MulKernel(float* x, const float* z, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) x[i] *= z[i];
}

__device__ __forceinline__ float Bf16Lo(std::uint32_t v) {
  return __uint_as_float(v << 16);
}
__device__ __forceinline__ float Bf16Hi(std::uint32_t v) {
  return __uint_as_float(v & 0xffff0000u);
}

template <int kTokens>
__global__ void F32GemvKernel(const float* __restrict__ w,
                              const float* __restrict__ x,
                              float* __restrict__ out, std::uint32_t m,
                              std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= m) return;
  const auto* w4 = reinterpret_cast<const float4*>(w + static_cast<std::size_t>(row) * k);
  float acc[kTokens] = {};
  for (std::uint32_t i = lane; i < k / 4; i += 32) {
    const float4 a = w4[i];
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      const float4 b = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(t) * k)[i];
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kTokens; ++t) {
    float v = acc[t];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    if (lane == 0) out[static_cast<std::size_t>(t) * m + row] = v;
  }
}

template <int kTokens>
void LaunchF32Gemv(const float* w, const float* x, float* out, std::uint32_t m,
                   std::uint32_t k, hipStream_t stream) {
  constexpr unsigned kWaves = 2;  // more blocks for a 257-row matrix
  hipLaunchKernelGGL(F32GemvKernel<kTokens>, dim3((m + kWaves - 1) / kWaves),
                     dim3(32 * kWaves), 0, stream, w, x, out, m, k);
}

// One wave per output row; blockIdx.y is the (row, slot) pair.
__global__ void Bf16ExpertsKernel(const std::uint16_t* __restrict__ w,
                                  const float* __restrict__ x,
                                  const std::int32_t* __restrict__ ids,
                                  float* __restrict__ out, std::uint32_t used,
                                  std::uint32_t m, std::uint32_t k) {
  const std::uint32_t pair = blockIdx.y;
  const std::uint32_t r = pair / used;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= m) return;
  const std::int32_t e = ids[pair];
  const auto* wrow = reinterpret_cast<const std::uint32_t*>(
      w + (static_cast<std::size_t>(e) * m + row) * k);
  const float* xr = x + static_cast<std::size_t>(r) * k;
  float acc = 0.0f;
  for (std::uint32_t i = lane; i < k / 2; i += 32) {
    const std::uint32_t v = wrow[i];
    acc = fmaf(Bf16Lo(v), xr[2 * i], acc);
    acc = fmaf(Bf16Hi(v), xr[2 * i + 1], acc);
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
  if (lane == 0) out[static_cast<std::size_t>(pair) * m + row] = acc;
}

}  // namespace

void AddInPlace(float* res, const float* x, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(count)), dim3(kThreads), 0, stream,
                     res, x, count);
}

void MulInPlace(float* x, const float* z, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(MulKernel, dim3(Blocks(count)), dim3(kThreads), 0, stream,
                     x, z, count);
}

void F32Gemv(const float* w, const float* x, float* out, std::uint32_t tokens,
             std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  switch (tokens) {
    case 1: return LaunchF32Gemv<1>(w, x, out, m, k, stream);
    case 2: return LaunchF32Gemv<2>(w, x, out, m, k, stream);
    case 3: return LaunchF32Gemv<3>(w, x, out, m, k, stream);
    case 4: return LaunchF32Gemv<4>(w, x, out, m, k, stream);
    case 5: return LaunchF32Gemv<5>(w, x, out, m, k, stream);
    case 6: return LaunchF32Gemv<6>(w, x, out, m, k, stream);
    case 7: return LaunchF32Gemv<7>(w, x, out, m, k, stream);
    default: return LaunchF32Gemv<8>(w, x, out, m, k, stream);
  }
}

void Bf16Experts(const void* w, const float* x, const std::int32_t* ids,
                 float* out, std::uint32_t rows, std::uint32_t used,
                 std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  constexpr unsigned kWaves = kThreads / 32;
  hipLaunchKernelGGL(Bf16ExpertsKernel, dim3((m + kWaves - 1) / kWaves,
                                             rows * used),
                     dim3(kThreads), 0, stream,
                     static_cast<const std::uint16_t*>(w), x, ids, out, used,
                     m, k);
}

}  // namespace gufo::models::qwen36_a3b
