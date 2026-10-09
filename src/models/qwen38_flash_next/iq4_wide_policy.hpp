#pragma once

#include <cstdlib>

namespace gufo::models::qwen38_flash_next {

// Experimental CPU handoff: select a winner with the resident GPU pass gate.
inline constexpr int kIq4WideDefaultVariant = 4;

constexpr int Iq4WideVariant(const char* value) {
  return value && value[0] >= '0' && value[0] <= '4' && value[1] == '\0'
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

constexpr int Iq4WmmaTokenTiles(int n) {
  return (n + 15) / 16;
}

constexpr int Iq4WmmaWaves(const char* value) {
  if (value && value[0] == '4' && value[1] == '\0')
    return 4;
  if (value && value[0] == '1' && value[1] == '6' && value[2] == '\0')
    return 16;
  return 8;
}

inline int Iq4WmmaWaves() {
  // Uncached A/B knob: default eight waves; test NL K640 with four and
  // the 640-row shared-expert gate/up matrices with sixteen.
  return Iq4WmmaWaves(std::getenv("GUFO_IQ4_WMMA_WAVES"));
}

constexpr int Iq4WideRowsPerBlock(int variant, int rows) {
  // WMMA always covers sixteen output rows. Legacy wider geometry requires
  // at least 80 blocks on the real 40-CU device.
  return variant == 4 || (variant >= 2 && rows >= 1280) ? 16 : 8;
}

}  // namespace gufo::models::qwen38_flash_next
