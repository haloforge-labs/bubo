// Glue kernels for A3B's wide prefill (see prefill_kernels.hpp).
#include "src/models/qwen36_a3b/prefill_kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

constexpr std::uint32_t kThreads = 256;

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

/// One thread per float4 of the destination.
__global__ void PadHeadsKernel(const float4* src, float4* dst,
                               std::uint32_t kv_heads, std::uint32_t group,
                               std::uint32_t padded, std::uint32_t d4,
                               std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= total) return;
  const std::uint32_t e = i % d4;
  const std::size_t head = i / d4;  // t * kv_heads * padded + g * padded + j
  const std::uint32_t j = head % padded;
  const std::size_t tg = head / padded;  // t * kv_heads + g
  if (j >= group) {
    dst[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  dst[i] = src[(tg * group + j) * d4 + e];
}

__global__ void UnpadHeadsKernel(const float4* src, float4* dst,
                                 std::uint32_t group, std::uint32_t padded,
                                 std::uint32_t d4, std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= total) return;
  const std::uint32_t e = i % d4;
  const std::size_t head = i / d4;  // (t * kv_heads + g) * group + j
  const std::uint32_t j = head % group;
  const std::size_t tg = head / group;
  dst[i] = src[(tg * padded + j) * d4 + e];
}

/// hi = BF16(v); lo (when non-null) = BF16(v - hi).
__device__ inline void StoreSplit(float v, __hip_bfloat16* hi,
                                  __hip_bfloat16* lo, std::size_t i) {
  const __hip_bfloat16 h = __float2bfloat16(v);
  hi[i] = h;
  if (lo != nullptr) lo[i] = __float2bfloat16(v - __bfloat162float(h));
}

__global__ void SplitBf16Kernel(const float* x, __hip_bfloat16* hi,
                                __hip_bfloat16* lo, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) StoreSplit(x[i], hi, lo, i);
}

template <bool kHalf>
__global__ void GatherRowsBf16Kernel(const void* src,
                                     const std::int32_t* rows,
                                     std::uint32_t width,
                                     __hip_bfloat16* dst, __hip_bfloat16* lo,
                                     std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= total) return;
  const std::size_t r = i / width;
  const std::uint32_t c = i % width;
  const std::int32_t row = rows[r];
  float v = 0.0f;
  if (row >= 0) {
    const std::size_t at = static_cast<std::size_t>(row) * width + c;
    v = kHalf ? __half2float(static_cast<const __half*>(src)[at])
              : static_cast<const float*>(src)[at];
  }
  StoreSplit(v, dst, lo, i);
}

__global__ void ScatterRowsHalfKernel(const float* src,
                                      const std::int32_t* rows,
                                      std::uint32_t width, __half* dst,
                                      std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= total) return;
  const std::size_t r = i / width;
  const std::int32_t row = rows[r];
  if (row < 0) return;
  dst[static_cast<std::size_t>(row) * width + i % width] =
      __float2half(src[i]);
}

__global__ void SwigluBf16Kernel(const float* gate, const float* up,
                                 __hip_bfloat16* out, __hip_bfloat16* lo,
                                 std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= count) return;
  const float g = gate[i];
  StoreSplit(g / (1.0f + __expf(-g)) * up[i], out, lo, i);
}

// gfx11 WMMA (wave32): A lane l holds row l % 16 (16 K values), B lane l
// column l % 16, both replicated across the two half-waves; D lane l holds
// column l % 16 with accumulator i at row 2i + l / 16.
using Frag = __attribute__((ext_vector_type(16))) unsigned short;
using Acc = __attribute__((ext_vector_type(8))) float;

/// grid (m / 64, tiles), block 256: wave w computes weight rows
/// 16 (w % 4).. of the block's 64 against token rows 32 (w / 4).. of the
/// tile (two 16x16 accumulators), sweeping K from global memory (the four
/// waves of a token half share the activation rows through the caches).
__global__ __launch_bounds__(256) void GroupedBf16Kernel(
    const unsigned short* __restrict__ w, const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ x_lo, const std::int32_t* tiles,
    const std::int32_t* bounds, float* __restrict__ out, std::uint32_t m,
    std::uint32_t k) {
  const std::int32_t tile = tiles[blockIdx.y];
  const std::uint32_t e = tile & 0xFFFF;
  const std::uint32_t end = bounds[e + 1];
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t r = lane & 15u;
  const std::uint32_t m0 = blockIdx.x * 64 + (wave & 3u) * 16;
  const std::uint32_t c0 = bounds[e] + (tile >> 16) * kGroupedTileRows +
                           (wave >> 2) * 32;
  if (c0 >= end) return;
  const std::uint32_t row = min(m0 + r, m - 1);
  const unsigned short* wr = w + (static_cast<std::size_t>(e) * m + row) * k;
  const std::uint32_t ca = c0 + r;
  const std::uint32_t cb = c0 + 16 + r;
  const bool va = ca < end;
  const bool vb = cb < end;
  const std::size_t oa = static_cast<std::size_t>(va ? ca : c0) * k;
  const std::size_t ob = static_cast<std::size_t>(vb ? cb : c0) * k;
  Acc acc0{};
  Acc acc1{};
  const Frag zero{};
  for (std::uint32_t kk = 0; kk < k; kk += 16) {
    const Frag a = *reinterpret_cast<const Frag*>(wr + kk);
    const Frag b0 = va ? *reinterpret_cast<const Frag*>(x + oa + kk) : zero;
    const Frag b1 = vb ? *reinterpret_cast<const Frag*>(x + ob + kk) : zero;
    acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b0, acc0);
    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b1, acc1);
    if (x_lo != nullptr) {
      const Frag l0 =
          va ? *reinterpret_cast<const Frag*>(x_lo + oa + kk) : zero;
      const Frag l1 =
          vb ? *reinterpret_cast<const Frag*>(x_lo + ob + kk) : zero;
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, l0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, l1, acc1);
    }
  }
  const std::uint32_t half = lane >> 4;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t mm = m0 + 2 * i + half;
    if (mm >= m) continue;
    if (va) out[static_cast<std::size_t>(ca) * m + mm] = acc0[i];
    if (vb) out[static_cast<std::size_t>(cb) * m + mm] = acc1[i];
  }
}

/// grid (m / 64, n / 64), block 256: a 64 x 64 output tile, 4 x 4 per
/// thread, K staged 16 at a time in LDS (transposed, padded).
__global__ __launch_bounds__(256) void F32GemmNTKernel(
    const float* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, std::uint32_t m, std::uint32_t n,
    std::uint32_t k) {
  __shared__ float ws[16][68];
  __shared__ float xs[16][68];
  const std::uint32_t j0 = blockIdx.x * 64;
  const std::uint32_t t0 = blockIdx.y * 64;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t tx = tid & 15u;  // output columns j0 + 4 tx..
  const std::uint32_t ty = tid >> 4;   // output rows t0 + 4 ty..
  float acc[4][4] = {};
  for (std::uint32_t kk = 0; kk < k; kk += 16) {
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i) {
      const std::uint32_t idx = tid + i * 256;
      const std::uint32_t row = idx >> 4;
      const std::uint32_t col = idx & 15u;
      ws[col][row] = j0 + row < m && kk + col < k
                         ? w[static_cast<std::size_t>(j0 + row) * k + kk + col]
                         : 0.0f;
      xs[col][row] = t0 + row < n && kk + col < k
                         ? x[static_cast<std::size_t>(t0 + row) * k + kk + col]
                         : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t c = 0; c < 16; ++c) {
      float a[4], b[4];
#pragma unroll
      for (std::uint32_t i = 0; i < 4; ++i) {
        a[i] = ws[c][tx * 4 + i];
        b[i] = xs[c][ty * 4 + i];
      }
#pragma unroll
      for (std::uint32_t r = 0; r < 4; ++r)
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i) acc[r][i] += b[r] * a[i];
    }
    __syncthreads();
  }
#pragma unroll
  for (std::uint32_t r = 0; r < 4; ++r) {
    const std::uint32_t t = t0 + ty * 4 + r;
    if (t >= n) continue;
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i) {
      const std::uint32_t j = j0 + tx * 4 + i;
      if (j < m) out[static_cast<std::size_t>(t) * m + j] = acc[r][i];
    }
  }
}

__global__ void Concat2Kernel(const float4* a, const float4* b, float4* out,
                              std::uint32_t w4, std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i >= total) return;
  const std::size_t t = i / (2 * w4);
  const std::uint32_t c = i % (2 * w4);
  out[i] = c < w4 ? a[t * w4 + c] : b[t * w4 + c - w4];
}

}  // namespace

void PadHeads(const float* src, float* dst, std::uint32_t n_tokens,
              std::uint32_t kv_heads, std::uint32_t group,
              std::uint32_t padded, std::uint32_t d, hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(n_tokens) * kv_heads * padded * (d / 4);
  hipLaunchKernelGGL(PadHeadsKernel, dim3(Blocks(total)), dim3(kThreads), 0,
                     stream, reinterpret_cast<const float4*>(src),
                     reinterpret_cast<float4*>(dst), kv_heads, group, padded,
                     d / 4, total);
}

void UnpadHeads(const float* src, float* dst, std::uint32_t n_tokens,
                std::uint32_t kv_heads, std::uint32_t group,
                std::uint32_t padded, std::uint32_t d, hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(n_tokens) * kv_heads * group * (d / 4);
  hipLaunchKernelGGL(UnpadHeadsKernel, dim3(Blocks(total)), dim3(kThreads), 0,
                     stream, reinterpret_cast<const float4*>(src),
                     reinterpret_cast<float4*>(dst), group, padded, d / 4,
                     total);
}

void GatherRowsBf16(const void* src, bool half_src, const std::int32_t* rows,
                    std::uint32_t n_rows, std::uint32_t width,
                    __hip_bfloat16* dst, __hip_bfloat16* lo,
                    hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_rows) * width;
  if (total == 0) return;
  if (half_src) {
    hipLaunchKernelGGL(GatherRowsBf16Kernel<true>, dim3(Blocks(total)),
                       dim3(kThreads), 0, stream, src, rows, width, dst, lo,
                       total);
  } else {
    hipLaunchKernelGGL(GatherRowsBf16Kernel<false>, dim3(Blocks(total)),
                       dim3(kThreads), 0, stream, src, rows, width, dst, lo,
                       total);
  }
}

void SplitBf16(const float* x, __hip_bfloat16* hi, __hip_bfloat16* lo,
               std::size_t count, hipStream_t stream) {
  if (count == 0) return;
  hipLaunchKernelGGL(SplitBf16Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, hi, lo, count);
}

void ScatterRowsHalf(const float* src, const std::int32_t* rows,
                     std::uint32_t n_rows, std::uint32_t width, __half* dst,
                     hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_rows) * width;
  if (total == 0) return;
  hipLaunchKernelGGL(ScatterRowsHalfKernel, dim3(Blocks(total)),
                     dim3(kThreads), 0, stream, src, rows, width, dst, total);
}

void SwigluBf16(const float* gate, const float* up, __hip_bfloat16* out,
                __hip_bfloat16* lo, std::size_t count, hipStream_t stream) {
  if (count == 0) return;
  hipLaunchKernelGGL(SwigluBf16Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, out, lo, count);
}

void GroupedBf16Gemm(const void* w, const void* x, const void* x_lo,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* bounds, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream) {
  if (n_tiles == 0) return;
  hipLaunchKernelGGL(GroupedBf16Kernel, dim3((m + 63) / 64, n_tiles),
                     dim3(256), 0, stream,
                     static_cast<const unsigned short*>(w),
                     static_cast<const unsigned short*>(x),
                     static_cast<const unsigned short*>(x_lo), tiles, bounds,
                     out, m, k);
}

void F32GemmNT(const float* w, const float* x, float* out, std::uint32_t m,
               std::uint32_t n, std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(F32GemmNTKernel, dim3((m + 63) / 64, (n + 63) / 64),
                     dim3(256), 0, stream, w, x, out, m, n, k);
}

void Concat2(const float* a, const float* b, float* out, std::uint32_t n,
             std::uint32_t width, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n) * 2 * (width / 4);
  hipLaunchKernelGGL(Concat2Kernel, dim3(Blocks(total)), dim3(kThreads), 0,
                     stream, reinterpret_cast<const float4*>(a),
                     reinterpret_cast<const float4*>(b),
                     reinterpret_cast<float4*>(out), width / 4, total);
}

}  // namespace gufo::models::qwen36_a3b
