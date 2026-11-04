// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_IQ_DECODE_BITS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_IQ_DECODE_BITS_HPP_

#include <cstdint>

#if defined(__HIPCC__)
#define GUFO_IQ_HD __host__ __device__
#else
#define GUFO_IQ_HD
#endif

namespace gufo::models::qwen38_flash_next::iq_decode {

enum class Variant { kBaseline, kPacked2, kPacked4, kPacked8, kLds4 };

GUFO_IQ_HD constexpr bool ActiveExpert(std::int32_t id) {
  return id >= 0;
}

GUFO_IQ_HD constexpr int RowsPerWave(Variant variant) {
  if (variant == Variant::kPacked8)
    return 8;
  if (variant == Variant::kPacked4 || variant == Variant::kLds4)
    return 4;
  return 2;
}

GUFO_IQ_HD constexpr bool Supported(Variant variant, bool iq2_s) {
  return variant >= Variant::kBaseline && variant <= Variant::kLds4 &&
         (iq2_s || variant != Variant::kLds4);
}

// Each two-bit field becomes one signed byte containing c - 1. The high
// bit bias prevents subtraction from borrowing across byte boundaries.
GUFO_IQ_HD constexpr std::uint32_t ExpandQ2Byte(std::uint32_t codes) {
  codes &= 0xFFU;
  codes = (codes | (codes << 12)) & 0x000F000FU;
  codes = (codes | (codes << 6)) & 0x03030303U;
  return ((codes | 0x80808080U) - 0x01010101U) ^ 0x80808080U;
}

// IQ2_S grid bytes are in 1..127. Complementing a negative byte and adding
// one therefore cannot carry into an adjacent byte.
GUFO_IQ_HD constexpr std::uint32_t ApplyIq2Signs(std::uint32_t grid,
                                                 std::uint32_t signs) {
  const std::uint32_t bits = ((signs & 15U) * 0x00204081U) & 0x01010101U;
  return (grid ^ (bits * 255U)) + bits;
}

GUFO_IQ_HD constexpr int ScaleIq2S(int low, int high, std::uint32_t scales) {
  return (low * int(scales & 15U) + high * int(scales >> 4) +
          (low + high) / 2) /
         4;
}

GUFO_IQ_HD constexpr std::uint32_t ParitySigns(std::uint32_t code) {
  code &= 127U;
#if defined(__clang__) || defined(__GNUC__)
  const auto parity = std::uint32_t(__builtin_popcount(code));
#else
  auto parity = code ^ (code >> 4);
  parity ^= parity >> 2;
  parity ^= parity >> 1;
#endif
  return code | ((parity & 1U) << 7);
}
GUFO_IQ_HD constexpr int ScaleIq2Xxs(int sum, std::uint32_t signs_scale) {
  return sum * int((signs_scale >> 27) | 1U) / 8;
}
GUFO_IQ_HD constexpr int ScaleIq3Xxs(int sum, int scale) {
  return (scale * sum + sum / 2) / 2;
}
GUFO_IQ_HD constexpr int ScaleIq3S(int sum, std::uint32_t scale) {
  return sum * (1 + 2 * int(scale & 15U));
}

// Packed grid nibbles are 0..2. The result is eight times the decoded
// ternary value plus its signed one-eighth offset, stored as signed bytes.
GUFO_IQ_HD constexpr std::uint32_t ExpandIq1(std::uint32_t grid,
                                             bool negative_delta) {
  const auto values = (grid & 0x0F0F0F0FU) << 3;
  const auto bias = negative_delta ? 0x09090909U : 0x07070707U;
  return ((values | 0x80808080U) - bias) ^ 0x80808080U;
}

GUFO_IQ_HD constexpr std::uint16_t Iq1Half(const std::uint8_t* scales) {
  return std::uint16_t((scales[1] >> 4) | (scales[3] & 0xF0) |
                       ((scales[5] & 0xF0) << 4) | ((scales[7] & 0xF0) << 8));
}

GUFO_IQ_HD constexpr int Iq1Scale(const std::uint8_t* scales, int chunk,
                                  int half) {
  const int word =
      scales[2 * (chunk / 2)] | (int(scales[2 * (chunk / 2) + 1]) << 8);
  return 1 + 2 * ((word >> (6 * (chunk % 2) + 3 * half)) & 7);
}

}  // namespace gufo::models::qwen38_flash_next::iq_decode

#undef GUFO_IQ_HD
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_IQ_DECODE_BITS_HPP_
