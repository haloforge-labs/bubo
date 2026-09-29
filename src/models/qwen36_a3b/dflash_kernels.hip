#include "src/models/qwen36_a3b/dflash_kernels.hpp"

#include <cstddef>

namespace gufo::models::qwen36_a3b {
namespace {

/// grid (ceil(width / 256), rows), block 256.
__global__ void TapCopyKernel(const float* __restrict__ src,
                              float* __restrict__ dst, std::uint32_t width,
                              std::uint32_t dst_stride) {
  const std::uint32_t r = blockIdx.y;
  const std::uint32_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= width) return;
  dst[static_cast<std::size_t>(r) * dst_stride + i] =
      src[static_cast<std::size_t>(r) * width + i];
}

/// grid (ceil(width / 256), rows), block 256.
__global__ void RingStoreKernel(const float* __restrict__ k,
                                const float* __restrict__ v,
                                float* __restrict__ ring_k,
                                float* __restrict__ ring_v,
                                const std::uint32_t* __restrict__ pos,
                                std::uint32_t width, std::uint32_t capacity) {
  const std::uint32_t r = blockIdx.y;
  const std::uint32_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= width) return;
  const std::size_t slot = (*pos + r) % capacity;
  const std::size_t src = static_cast<std::size_t>(r) * width + i;
  ring_k[slot * width + i] = k[src];
  ring_v[slot * width + i] = v[src];
}

/// grid (rows), block 256: softmax probability of token tok[r] in row r.
__global__ void TokenProbKernel(const float* __restrict__ logits,
                                const std::uint32_t* __restrict__ tok,
                                float* __restrict__ prob, std::uint32_t vocab) {
  __shared__ float red[256];
  const std::uint32_t r = blockIdx.x;
  const float* row = logits + static_cast<std::size_t>(r) * vocab;
  float m = -INFINITY;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += 256) m = fmaxf(m, row[i]);
  red[threadIdx.x] = m;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += 256)
    sum += __expf(row[i] - m);
  red[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) prob[r] = __expf(row[tok[r]] - m) / red[0];
}

}  // namespace

void TokenProb(const float* logits, const std::uint32_t* tok, float* prob,
               std::uint32_t rows, std::uint32_t vocab, hipStream_t stream) {
  if (rows == 0) return;
  hipLaunchKernelGGL(TokenProbKernel, dim3(rows), dim3(256), 0, stream, logits,
                     tok, prob, vocab);
}

void TapCopy(const float* src, float* dst, std::uint32_t rows,
             std::uint32_t width, std::uint32_t dst_stride, hipStream_t stream) {
  if (rows == 0) return;
  hipLaunchKernelGGL(TapCopyKernel, dim3((width + 255) / 256, rows), dim3(256),
                     0, stream, src, dst, width, dst_stride);
}

void RingStore(const float* k, const float* v, float* ring_k, float* ring_v,
               const std::uint32_t* pos, std::uint32_t rows,
               std::uint32_t width, std::uint32_t capacity,
               hipStream_t stream) {
  if (rows == 0) return;
  hipLaunchKernelGGL(RingStoreKernel, dim3((width + 255) / 256, rows),
                     dim3(256), 0, stream, k, v, ring_k, ring_v, pos, width,
                     capacity);
}

}  // namespace gufo::models::qwen36_a3b
