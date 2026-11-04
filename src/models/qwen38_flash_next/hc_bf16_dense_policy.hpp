#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next::hc_bf16_dense {

// Process lifetime, default on (0 restores the generic projection). N1 uses the same kernel as N2..8: the per-row
// reduction order does not depend on the token count, so plain decode and a
// verify pass agree bit for bit when the switch is on.
constexpr int Parse(const char* value) {
  if (value && value[0] == '0' && value[1] == '\0')
    return 0;
  if (!value || (value[0] == '1' && value[1] == '\0'))
    return 1;
  return -1;
}

inline bool Enabled() {
  static const bool enabled = [] {
    const int parsed = Parse(std::getenv("GUFO_HC_BF16_DENSE"));
    if (parsed < 0)
      throw std::invalid_argument(
          "unknown GUFO_HC_BF16_DENSE; expected 0 or 1");
    return parsed == 1;
  }();
  return enabled;
}

constexpr bool Shape(unsigned n, unsigned m, unsigned k) {
  return n >= 1 && n <= 8 &&
         ((m == 320 && k == 10240) || (m == 10240 && k == 320));
}

constexpr bool Aligned(std::uintptr_t w, std::uintptr_t x) {
  return (w & 7) == 0 && (x & 15) == 0;
}

}  // namespace gufo::models::qwen38_flash_next::hc_bf16_dense
