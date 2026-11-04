#pragma once

// Included after SmallGemmKernel, inside kernels.hip.cpp's private namespace.
// Only BF16 HC down/up: no inject, norms, combine, generic SmallGemm or Q8.
namespace hc_bf16_detail {

template<unsigned tokens, bool down>
__launch_bounds__(128) __global__
    void SmallGemmHcBf16Kernel(const std::uint16_t* __restrict__ w,
                               const float* __restrict__ x,
                               float* __restrict__ out) {
  constexpr unsigned k = down ? 10240 : 320;
  constexpr unsigned m = down ? 320 : 10240;
  constexpr unsigned steps = 4;
  constexpr unsigned split = down ? 4 : 1;
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave =
      __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) >> 5);
  const unsigned row = down ? blockIdx.x : blockIdx.x * 4 + wave;
  const unsigned begin = down ? wave * (k / split) : 0;
  const unsigned end = down ? begin + k / split : k;
  const auto* weight = w + std::size_t(row) * k;
  float acc[tokens]{};
  for (unsigned base = begin + lane * 4; base < end; base += 128 * steps) {
    uint2 packed[steps];
    float4 activation[steps][tokens];
#pragma unroll
    for (unsigned st = 0; st < steps; ++st) {
      const unsigned i = base + st * 128;
      if (i < end) {
        // Four BF16 values arrive as one aligned eight-byte load.
        packed[st] = *reinterpret_cast<const uint2*>(weight + i);
#pragma unroll
        for (unsigned t = 0; t < tokens; ++t)
          activation[st][t] = *reinterpret_cast<const float4*>(x + t * k + i);
      }
    }
#pragma unroll
    for (unsigned st = 0; st < steps; ++st) {
      if (base + st * 128 < end) {
        const uint2 bits = packed[st];
        const float a = __uint_as_float(bits.x << 16);
        const float b = __uint_as_float(bits.x & 0xffff0000U);
        const float c = __uint_as_float(bits.y << 16);
        const float d = __uint_as_float(bits.y & 0xffff0000U);
#pragma unroll
        for (unsigned t = 0; t < tokens; ++t) {
          const float4 v = activation[st][t];
          float sum = __builtin_fmaf(a, v.x, acc[t]);
          sum = __builtin_fmaf(b, v.y, sum);
          sum = __builtin_fmaf(c, v.z, sum);
          acc[t] = __builtin_fmaf(d, v.w, sum);
        }
      }
    }
  }
  if constexpr (down) {
    // Four waves cooperate on one row: 320 CTAs rather than 80. All K
    // partials remain in this CTA, avoiding atomics, scratch and a merge
    // launch.
    __shared__ float partial[tokens][4];
#pragma unroll
    for (unsigned t = 0; t < tokens; ++t) {
      const float sum = WaveSum(acc[t]);
      if (lane == 0)
        partial[t][wave] = sum;
    }
    __syncthreads();
    if (threadIdx.x < tokens) {
      const unsigned t = threadIdx.x;
      out[t * m + row] = __fadd_rn(__fadd_rn(partial[t][0], partial[t][1]),
                                   __fadd_rn(partial[t][2], partial[t][3]));
    }
  } else {
#pragma unroll
    for (unsigned t = 0; t < tokens; ++t) {
      const float sum = WaveSum(acc[t]);
      if (lane == 0)
        out[t * m + row] = sum;
    }
  }
}

template<unsigned tokens>
void Launch(const void* w, const float* x, float* out, bool down,
            hipStream_t stream) {
  if (down)
    hipLaunchKernelGGL((SmallGemmHcBf16Kernel<tokens, true>), dim3(320),
                       dim3(128), 0, stream,
                       static_cast<const std::uint16_t*>(w), x, out);
  else
    hipLaunchKernelGGL((SmallGemmHcBf16Kernel<tokens, false>), dim3(2560),
                       dim3(128), 0, stream,
                       static_cast<const std::uint16_t*>(w), x, out);
}

bool TryLaunch(const void* w, const float* x, float* out, unsigned n,
               unsigned m, unsigned k, hipStream_t stream) {
  if (!hc_bf16_dense::Shape(n, m, k) ||
      !hc_bf16_dense::Aligned(reinterpret_cast<std::uintptr_t>(w),
                              reinterpret_cast<std::uintptr_t>(x)))
    return false;
  const bool down = k == 10240;
  switch (n) {
    case 1:
      Launch<1>(w, x, out, down, stream);
      break;
    case 2:
      Launch<2>(w, x, out, down, stream);
      break;
    case 3:
      Launch<3>(w, x, out, down, stream);
      break;
    case 4:
      Launch<4>(w, x, out, down, stream);
      break;
    case 5:
      Launch<5>(w, x, out, down, stream);
      break;
    case 6:
      Launch<6>(w, x, out, down, stream);
      break;
    case 7:
      Launch<7>(w, x, out, down, stream);
      break;
    case 8:
      Launch<8>(w, x, out, down, stream);
      break;
  }
  return true;
}

}  // namespace hc_bf16_detail
