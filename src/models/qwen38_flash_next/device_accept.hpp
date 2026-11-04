#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DEVICE_ACCEPT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DEVICE_ACCEPT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

// Independent of the verify (32..38), MTP (40..46), rollback (48) and KV
// (49..50) fields. Low bits of this rollback shape hold WIDTH, not keep.
inline constexpr std::uint64_t kDeviceAcceptKey = std::uint64_t{1} << 39;
inline constexpr std::uint64_t DeviceRollbackKey(std::uint32_t width) {
  return (std::uint64_t{1} << 48) | kDeviceAcceptKey | width;
}

struct GreedyKeepInput {
  std::int32_t drafts[7]{};
  std::int32_t stops[16]{};
  std::uint32_t stop_count{0};
};
enum class GreedyKeepStatus : std::uint32_t { kReady, kStop, kNonFinite };
struct GreedyKeepResult {
  std::uint32_t keep{1};
  GreedyKeepStatus status{GreedyKeepStatus::kReady};
};

enum class DeviceAcceptReason : std::uint32_t {
  kEligible,
  kSampled,
  kConstraint,
  kLogprobs,
  kPenalties,
  kThinkNudge,
  kBehaviour,
  kBatch,
  kHead,
  kStop,
  kNonFinite,
  kCount
};

// Certify that CanSelectArgmax has the same answer for every non-stop ID
// at each visited prefix. Token-dependent grammar/bias/suppression and
// reasoning interventions remain on the existing path. Loop/boundary state
// is advanced through the actual proposed prefix using the existing sampler.
// A mismatch is visited BEFORE Accept, so it has that same certified state.
inline DeviceAcceptReason DeviceAcceptEligibility(
    sampling::SamplerState sampler, std::span<const std::int32_t> chain,
    bool logprobs, bool batch) {
  using R = DeviceAcceptReason;
  const auto& c = sampler.config();
  if (c.temperature != 0.0F || c.uses_random_sampling())
    return R::kSampled;
  if (c.constraint)
    return R::kConstraint;
  if (logprobs)
    return R::kLogprobs;
  if (c.penalties_enabled())
    return R::kPenalties;
  if (c.reasoning_budget || sampler.ForcedToken() ||
      (sampler.BehaviourEnabled() && c.behaviour->think_nudge))
    return R::kThinkNudge;
  if (batch)
    return R::kBatch;
  if (sampler.BehaviourEnabled()) {
    const auto& b = *c.behaviour;
    if (b.logit_bias_enabled || b.model_suppress_tokens || b.marker_bias != 0 ||
        b.marker_entropy_gate)
      return R::kBehaviour;
  }
  auto& preview = sampler;
  for (std::size_t i = 1; i < chain.size(); ++i) {
    if (preview.ForcedToken())
      return R::kThinkNudge;
    if (preview.BehaviourEnabled()) {
      const auto& stops = c.behaviour->eog_ids;
      const auto is_stop = [&](sampling::TokenId id) {
        return std::ranges::find(stops, id) != stops.end();
      };
      if (is_stop(static_cast<sampling::TokenId>(chain[i])))
        return R::kStop;
      // A generic behaviour force can admit exactly the proposed ID while
      // rejecting the verifier's different winner. Two distinct ordinary
      // IDs rule out that singleton case using the authoritative predicate.
      // EOG IDs cannot certify ordinary IDs: deferred EOG treats them
      // differently. All other ID-specific interventions are excluded above.
      sampling::TokenId probe = 0;
      while (probe == static_cast<sampling::TokenId>(chain[i]) ||
             is_stop(probe))
        ++probe;
      if (!preview.CanSelectArgmax(probe, /*penalties_applied=*/true))
        return R::kConstraint;
    }
    if (!preview.CanSelectArgmax(static_cast<sampling::TokenId>(chain[i]),
                                 /*penalties_applied=*/true))
      return R::kConstraint;
    preview.Accept(static_cast<sampling::TokenId>(chain[i]));
  }
  return R::kEligible;
}

struct DeviceAcceptTotals {
  std::uint64_t queued{0}, applied{0}, restored{0};
  std::array<std::uint64_t,
             static_cast<std::size_t>(DeviceAcceptReason::kCount)>
      fallback{};
  void Fallback(DeviceAcceptReason reason) {
    ++fallback[static_cast<std::size_t>(reason)];
  }
};

}  // namespace gufo::models::qwen38_flash_next
#endif
