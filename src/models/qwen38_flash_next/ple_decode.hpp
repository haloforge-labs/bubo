// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PLE_DECODE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PLE_DECODE_HPP_

#include <bit>
#include <cstdint>

#if defined(__HIPCC__)
#define GUFO_PLE_HD __host__ __device__
#else
#define GUFO_PLE_HD
#endif

namespace gufo::models::qwen38_flash_next {

GUFO_PLE_HD inline float PleFloatBits(std::uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_bit_cast(float, bits);
#else
  return std::bit_cast<float>(bits);
#endif
}

/// The conversion and IQ4_NL codebook retain the scalar row path's values.
/// Byte loads permit the 18-byte blocks to start at an unaligned address.
GUFO_PLE_HD inline float PleHalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  std::uint32_t exponent = (h >> 10) & 31U;
  std::uint32_t mantissa = h & 1023U;
  if (exponent == 0) {
    if (mantissa == 0)
      return PleFloatBits(sign);
    // Normalize the half subnormal using integer shifts. Its F32 value is
    // normal, so this conversion does not depend on device denormal modes.
    exponent = 113;
    while ((mantissa & 1024U) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    return PleFloatBits(sign | (exponent << 23) | ((mantissa & 1023U) << 13));
  }
  const std::uint32_t bits = sign |
                             ((exponent == 31 ? 255U : exponent + 112U) << 23) |
                             (mantissa << 13);
  return PleFloatBits(bits);
}

GUFO_PLE_HD inline float PleRowElement(const std::uint8_t* row,
                                       std::uint32_t column, bool iq4) {
  if (!iq4) {
    const std::uint32_t bits = (std::uint32_t{row[2 * column]} |
                                (std::uint32_t{row[2 * column + 1]} << 8))
                               << 16;
    return PleFloatBits(bits);
  }
  constexpr int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                              1,    13,   25,  38,  53,  69,  89,  113};
  const auto* block = row + (column / 32) * 18;
  const auto half = static_cast<std::uint16_t>(std::uint32_t{block[0]} |
                                               (std::uint32_t{block[1]} << 8));
  const auto packed = block[2 + column % 16];
  const auto code = column % 32 < 16 ? packed & 15U : packed >> 4;
  return PleHalfToFloat(half) * static_cast<float>(values[code]);
}

}  // namespace gufo::models::qwen38_flash_next

#undef GUFO_PLE_HD
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PLE_DECODE_HPP_
