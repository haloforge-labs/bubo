#ifndef GUFO_MODELS_DIFFUSION_GEMMA_DEVICE_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_DEVICE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::diffusion_gemma::kernels {

using bf16_t = std::uint16_t;

__device__ __forceinline__ float ToFloat(bf16_t x) {
  return __uint_as_float(static_cast<unsigned>(x) << 16);
}

// Round to nearest even, preserving infinities and NaN.
__device__ __forceinline__ bf16_t ToBf16(float value) {
  const unsigned bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U)
    return static_cast<bf16_t>((bits | ((bits & 0xffffU) ? 0x10000U : 0U)) >>
                               16);
  return static_cast<bf16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

__device__ __forceinline__ float RoundBf16(float value) {
  return ToFloat(ToBf16(value));
}

// FP8 E4M3 (finite variant: exponent bias 7, no infinities) to BF16, exact:
// the 3-bit mantissa fits BF16's 7 bits. Normal values rebias the exponent
// (+120); subnormals are m * 2^-9.
__device__ __forceinline__ bf16_t Fp8E4M3ToBf16(unsigned byte) {
  const unsigned sign = (byte & 0x80U) << 8, magnitude = byte & 0x7fU;
  const unsigned normal = (magnitude << 4) + (120U << 7);
  const unsigned subnormal =
      __float_as_uint(static_cast<float>(magnitude) * 0x1p-9F) >> 16;
  return static_cast<bf16_t>(sign | (magnitude >= 8 ? normal : subnormal));
}

__device__ __forceinline__ float WaveSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, 32);
  return value;
}

__device__ __forceinline__ float WaveMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor(value, offset, 32));
  return value;
}

// Workgroup barrier that waits for LDS traffic only. __syncthreads() also
// drains every outstanding global load, which would stall a register
// prefetch of the next stage at each barrier.
__device__ __forceinline__ void LdsBarrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// WMMA tiles (gfx11, wave32). One Mma multiplies a 16x16 A tile by a 16x16
// B tile along k:
//   - A Fragment: lane l holds row l % 16 of A, 16 consecutive k values.
//     Lanes 16..31 repeat lanes 0..15.
//   - B Fragment: lane l holds row l % 16 of B (that is, column l % 16 of
//     the product), 16 consecutive k values, so B is read "transposed":
//     weights stored [n][k] load directly.
//   - Accumulator: element i of lane l is product row 2 * i + l / 16,
//     column l % 16. Kernels write results with exactly this mapping.
// Kernels stage operands in LDS rows padded by 8 elements (16 bytes) so
// that the 16 lanes of a fragment load hit different banks.
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

// 16 consecutive BF16 values (32 bytes, 16-byte aligned).
__device__ __forceinline__ Fragment LoadFragment(const bf16_t* x) {
  union {
    Fragment f;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(x);
  value.u[1] = *reinterpret_cast<const uint4*>(x + 8);
  return value.f;
}

// D = A * B^T + C for a 16x16x16 tile (layout above).
__device__ __forceinline__ Accumulator Mma(Fragment a, Fragment b,
                                           Accumulator c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

}  // namespace gufo::models::diffusion_gemma::kernels

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_DEVICE_HPP_
