#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROUTED_EXPERT_AUDIT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROUTED_EXPERT_AUDIT_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {
enum class RoutedExpertDroppedRows { kUntouched, kZeroOrUntouched };
struct RoutedExpertRowObservation {
  std::uint32_t executed;
  std::uint32_t invalid;
};

// Bits keep nonfinite checks valid under the device's fast-math flags.
// MMVQ deliberately zero-stores negative-id rows without executing a dot.
#if defined(__HIPCC__)
__host__ __device__
#endif
    constexpr RoutedExpertRowObservation ObserveRoutedExpertRow(
        bool live, bool untouched, std::uint32_t bits,
        RoutedExpertDroppedRows dropped = RoutedExpertDroppedRows::kUntouched) {
  if (!live) {
    const bool allowed =
        untouched || (dropped == RoutedExpertDroppedRows::kZeroOrUntouched &&
                      (bits & 0x7fffffffu) == 0);
    return {allowed ? 0u : 1u, allowed ? 0u : 1u};
  }
  return {untouched ? 0u : 1u,
          untouched || (bits & 0x7f800000u) == 0x7f800000u ? 1u : 0u};
}
}  // namespace gufo::models::qwen38_flash_next
#endif
