#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROUTER_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROUTER_POLICY_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {
enum class PrefillExpertsMode : std::uint32_t { kRenorm, kRef10, kFold, kMass };
enum class PrefillExpertsAudit : std::uint32_t { kLean, kFull };

// Slots are summed on the host from the existing live ExpertCounts download.
// Only full audit writes executed_rows/invalid_rows on device (first scalar).
struct RoutedExpertCounters {
  unsigned long long active_slots{0}, offered_slots{0}, executed_rows{0},
      invalid_rows{0};
};

// Full audit packs two 32-bit totals into one global atomic per workgroup.
// The executor rejects passes offering > UINT32_MAX rows before launching.
struct RoutedExpertAudit {
  unsigned long long packed{0};
  std::uint32_t ExecutedRows() const {
    return static_cast<std::uint32_t>(packed);
  }
  std::uint32_t InvalidRows() const {
    return static_cast<std::uint32_t>(packed >> 32);
  }
};
}  // namespace gufo::models::qwen38_flash_next
#endif
