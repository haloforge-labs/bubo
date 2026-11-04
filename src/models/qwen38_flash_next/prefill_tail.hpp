#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_TAIL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_TAIL_HPP_

#include <algorithm>
#include <cstddef>

namespace gufo::models::qwen38_flash_next {
// Positions are in the complete request prompt, including its cached prefix.
// With no tail every row is a cut row, so the boundary is the prompt end.
inline std::size_t PrefillExpertTailBoundary(std::size_t prompt_tokens,
                                             std::size_t tail_tokens) {
  return tail_tokens ? prompt_tokens - std::min(prompt_tokens, tail_tokens)
                     : prompt_tokens;
}

// The expert cut pays for itself only on a pass that reads many new rows: the
// full-precision tail is re-prefilled by the next turn, and a short pass saves
// less than that costs (measured cold: 2.9K tokens +3.5 to +7 %, 7.9K +12 to
// +13 %, 16K and longer +15 to +17 %). A request therefore cuts only when the
// rows it still has to compute, the prompt less the rows restored from the
// cache, reach this count.
inline constexpr std::size_t kPrefillExpertMinNewTokens = 4096;

// Whether a request with `reused_tokens` rows restored cuts its new rows. A
// configuration without a tail cuts every row (it has no full-precision rows
// to re-prefill), so the threshold applies to the tail policy only.
inline bool PrefillExpertCutsRequest(std::size_t prompt_tokens,
                                     std::size_t reused_tokens,
                                     std::size_t tail_tokens) {
  return tail_tokens == 0 ||
         prompt_tokens - std::min(prompt_tokens, reused_tokens) >=
             kPrefillExpertMinNewTokens;
}

// The first full-precision row of a request, in prompt positions. A request
// that does not cut returns zero: every pass runs under the normal arithmetic
// and no hard boundary splits it. Rows restored from the cache keep the
// arithmetic they were computed with; this plan covers the new rows only.
inline std::size_t PrefillExpertRequestBoundary(std::size_t prompt_tokens,
                                                std::size_t reused_tokens,
                                                std::size_t tail_tokens) {
  return PrefillExpertCutsRequest(prompt_tokens, reused_tokens, tail_tokens)
             ? PrefillExpertTailBoundary(prompt_tokens, tail_tokens)
             : 0;
}

// A policy boundary is hard: neither checkpoint capture nor the ordinary
// 128-token end merge may combine its two arithmetic policies in one pass.
inline std::size_t ClipPrefillExpertTail(std::size_t offset, std::size_t budget,
                                         std::size_t boundary) {
  return offset < boundary ? std::min(budget, boundary - offset) : budget;
}

inline bool ApplyPrefillExpertPolicy(std::size_t offset, std::size_t boundary,
                                     std::size_t tail_tokens) {
  return tail_tokens == 0 || offset < boundary;
}
}  // namespace gufo::models::qwen38_flash_next
#endif
