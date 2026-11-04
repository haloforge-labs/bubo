#ifndef GUFO_MODELS_QWEN36_A3B_PREFILL_POLICY_HPP_
#define GUFO_MODELS_QWEN36_A3B_PREFILL_POLICY_HPP_

#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace gufo::models::qwen36_a3b {
namespace qfn = qwen38_flash_next;

struct PrefillLayer {
  const qfn::PrefillExpertsSettings* settings{nullptr};
  std::uint32_t layer{0};
  explicit operator bool() const { return settings != nullptr; }
  float Fold() const {
    return settings->mode == qfn::PrefillExpertsMode::kFold
               ? settings->fold.at(layer)
               : 1.0f;
  }
};

inline const qfn::PrefillExpertsSettings& PrefillExperts() {
  return qfn::GetPrefillExperts<8, 40>();
}

// Token-prefix identities (including image prefixes) get the same canonical
// policy. The off path preserves the original bytes, including empty identity.
inline std::vector<std::uint8_t> PrefillCacheIdentity(
    std::span<const std::uint8_t> original,
    const qfn::PrefillExpertsSettings& settings) {
  std::vector<std::uint8_t> identity(original.begin(), original.end());
  if (settings.active) {
    const auto line = settings.IdentityLine();
    identity.insert(identity.end(), line.begin(), line.end());
  }
  return identity;
}
}  // namespace gufo::models::qwen36_a3b
#endif
