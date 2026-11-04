#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_LOAD_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_LOAD_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm::detail {

// Private q8 -> F16 operand decode from eight raw code bytes and stored
// F16 scale bits; not an F32 reconstruction path.
// Return consecutive F16 pairs in x/y/z/w, earlier lane in the low halfword.
// Contract: q in [-127,127], nonnegative F16 scale bits in [0,0x6007].
// No codec/layout changes, saturation, or scale re-rounding are performed.
__device__ __forceinline__ uint4 DecodeQ8Kv8Packed(std::uint64_t code_bits,
                                                   std::uint16_t scale_bits) {
  uint2 bytes = __builtin_bit_cast(uint2, code_bits);
  bytes.x ^= 0x80808080U;
  bytes.y ^= 0x80808080U;

  // XOR maps signed q to q+128. A half with bits 0x6400 | (q+128)
  // represents 1152+q exactly (unit spacing throughout this interval).
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(bytes.x, 0x64646464U, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(bytes.x, 0x64646464U, 0x03070206U);
  const std::uint32_t p2 =
      __builtin_amdgcn_perm(bytes.y, 0x64646464U, 0x01050004U);
  const std::uint32_t p3 =
      __builtin_amdgcn_perm(bytes.y, 0x64646464U, 0x03070206U);
  const __half2 minus_bias = __builtin_bit_cast(__half2, 0xe480e480U);
  std::uint32_t q0 = __builtin_bit_cast(
      std::uint32_t, __hadd2(__builtin_bit_cast(__half2, p0), minus_bias));
  std::uint32_t q1 = __builtin_bit_cast(
      std::uint32_t, __hadd2(__builtin_bit_cast(__half2, p1), minus_bias));
  std::uint32_t q2 = __builtin_bit_cast(
      std::uint32_t, __hadd2(__builtin_bit_cast(__half2, p2), minus_bias));
  std::uint32_t q3 = __builtin_bit_cast(
      std::uint32_t, __hadd2(__builtin_bit_cast(__half2, p3), minus_bias));

  // Subtract the INTEGER bias first, exactly. Keep that boundary opaque to
  // fast-math: an affine half FMA with rounded (-1152*scale) is NOT equivalent.
  // Used outputs keep the boundary opaque; no volatile side effects are needed.
  asm("" : "+v"(q0), "+v"(q1), "+v"(q2), "+v"(q3));
  const __half2 scale2 = __builtin_bit_cast(
      __half2, std::uint32_t{scale_bits} | (std::uint32_t{scale_bits} << 16));

  // q and scale have at most 7 and 11 significant bits, so their product is
  // exact in F32. One RNE16 multiply therefore matches F32-product -> F16,
  // including subnormals when native F16 denormals are preserved. Signed
  // zeros are equivalent under the KV operand contract. The native test
  // qualifies denormal behavior with the actual production compiler flags.
  return uint4{
      __builtin_bit_cast(std::uint32_t,
                         __hmul2(__builtin_bit_cast(__half2, q0), scale2)),
      __builtin_bit_cast(std::uint32_t,
                         __hmul2(__builtin_bit_cast(__half2, q1), scale2)),
      __builtin_bit_cast(std::uint32_t,
                         __hmul2(__builtin_bit_cast(__half2, q2), scale2)),
      __builtin_bit_cast(std::uint32_t,
                         __hmul2(__builtin_bit_cast(__half2, q3), scale2))};
}

// Unchanged load contract: exactly eight codes within one q8 block.
// The source need only be two-byte aligned (34-byte blocks have no padding).
// This is not a load for F16 cache storage.
__device__ __forceinline__ uint4 LoadQ8Kv8Packed(const std::int8_t* codes,
                                                 _Float16 scale) {
  uint2 bytes;
  __builtin_memcpy(&bytes, codes, sizeof(bytes));
  return DecodeQ8Kv8Packed(__builtin_bit_cast(std::uint64_t, bytes),
                           __builtin_bit_cast(std::uint16_t, scale));
}

}  // namespace gufo::models::qwen38_flash_next::rocm::detail

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_LOAD_HPP_
