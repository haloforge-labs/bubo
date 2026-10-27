#ifndef GUFO_CORE_DIAGNOSTICS_COMPATIBILITY_H_
#define GUFO_CORE_DIAGNOSTICS_COMPATIBILITY_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/system_inventory.h"

namespace gufo::diagnostics {

enum class CompatibilityVerdict : std::uint8_t {
  kSupported,
  kUnsupported,
  kUnknown,
  kNotApplicable,  ///< The item is a Linux check and the host is not Linux.
};

[[nodiscard]] std::string_view ToString(CompatibilityVerdict verdict);

struct CompatibilityItem {
  std::string component;
  CompatibilityVerdict verdict{CompatibilityVerdict::kSupported};
  std::string detected_value;
  std::string required_value;
  std::string evidence;
  std::string remediation_hint;
};

struct CompatibilityReport {
  CompatibilityVerdict overall_verdict{CompatibilityVerdict::kSupported};
  std::vector<CompatibilityItem> items;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] CompatibilityReport EvaluateCompatibility(
    const SystemInventory& inventory, std::string_view section = "all");

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_COMPATIBILITY_H_
