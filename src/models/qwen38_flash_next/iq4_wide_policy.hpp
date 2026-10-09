#pragma once

#include <cstdlib>

namespace gufo::models::qwen38_flash_next {

// Experimental CPU handoff: select a winner with the resident GPU pass gate.
inline constexpr int kIq4WideDefaultVariant = 2;

constexpr int Iq4WideVariant(const char* value) {
  return value && value[0] >= '0' && value[0] <= '3' && value[1] == '\0'
             ? value[0] - '0'
             : kIq4WideDefaultVariant;
}

inline int Iq4WideVariant() {
  // Deliberately not cached: the operator bench changes variants in one
  // process.
  return Iq4WideVariant(std::getenv("GUFO_IQ4_WIDE_VARIANT"));
}

constexpr int Iq4WideTokenTile(int n) {
  return ((n + 7) / 8) * 8;
}

constexpr int Iq4WideRowsPerBlock(int variant, int rows) {
  // At least 80 blocks for the wider geometry on the real 40-CU device.
  return variant >= 2 && rows >= 1280 ? 16 : 8;
}

}  // namespace gufo::models::qwen38_flash_next
