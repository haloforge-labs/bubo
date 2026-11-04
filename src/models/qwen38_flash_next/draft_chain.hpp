#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_CHAIN_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_CHAIN_HPP_

#include <array>
#include <cstdint>

#include "src/models/qwen38_flash_next/decode_cycle.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint64_t kDraftChainKey = std::uint64_t{1} << 42;
// MTP low 16 bits retain the first (catch-up) row count. Bits 16..18 hold
// proposal depth and 20..26 the per-step sparse mask. The remaining MTP
// fields, rollback tag and KV dtype are unchanged.
inline constexpr std::uint64_t DraftChainKey(std::uint32_t rows,
                                             std::uint32_t depth,
                                             std::uint32_t sparse_mask,
                                             bool full_head) {
  return MtpGraphKey(rows, false, true, false, sparse_mask & 1, false,
                     full_head) |
         kDraftChainKey | (std::uint64_t{depth} << 16) |
         (std::uint64_t{sparse_mask} << 20);
}

// One cache line per publication. GPU release follows the token store;
// the host consumes slots in order and never resets an in-flight slot.
struct alignas(64) DraftChainSlot {
  std::int32_t token{0};
  std::uint32_t ready{0};
};
static_assert(sizeof(DraftChainSlot) == 64);

struct DraftChainPlan {
  bool bounded{true};
  std::uint32_t sparse_mask{0}, end{0}, blocks{0};
};
inline constexpr DraftChainPlan PlanDraftChain(
    std::uint32_t position, std::uint32_t blocks, std::uint32_t rows,
    std::uint32_t depth, std::uint32_t ratio, std::uint32_t budget) {
  DraftChainPlan plan{true, 0, position, blocks};
  if (ratio == 0 || depth == 0 || depth > 7 || rows == 0 || rows > 8) {
    plan.bounded = false;
    return plan;
  }
  for (std::uint32_t i = 0; i < depth; ++i) {
    const auto step_rows = i == 0 ? rows : 1;
    plan.end += step_rows;
    if (plan.end > budget) {
      const auto complete = plan.end / ratio;
      if (complete < plan.blocks ||
          complete - plan.blocks > step_rows / ratio + 1)
        plan.bounded = false;
      plan.blocks = complete;
      plan.sparse_mask |= 1U << i;
    }
  }
  return plan;
}

enum class DraftChainReason : std::uint32_t {
  kEligible,
  kSampled,
  kFalsifier,
  kLookup,
  kBatch,
  kReplay,
  kGeometry,
  kCount
};
inline DraftChainReason DraftChainEligibility(bool sampled,
                                              DraftFalsify falsifier,
                                              bool lookup, bool batch) {
  if (sampled)
    return DraftChainReason::kSampled;
  if (falsifier != DraftFalsify::kNone)
    return DraftChainReason::kFalsifier;
  if (lookup)
    return DraftChainReason::kLookup;
  if (batch)
    return DraftChainReason::kBatch;
  return DraftChainReason::kEligible;
}
struct DraftChainRun {
  DraftChainReason reason{DraftChainReason::kReplay};
  std::uint32_t early_reads{0};
  bool graph_launch{false};
};
struct DraftChainTotals {
  std::uint64_t cycles{0}, drafts{0}, multi_graphs{0}, early_reads{0};
  std::array<std::uint64_t, static_cast<std::size_t>(DraftChainReason::kCount)>
      fallback{};
};

}  // namespace gufo::models::qwen38_flash_next
#endif
