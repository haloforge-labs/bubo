#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::models::qwen38_flash_next {

inline constexpr int kHeadTop = 128;
inline constexpr int kHeadExtras = 256;
inline constexpr int kHeadCapacity = kHeadTop + kHeadExtras;
inline constexpr int kHeadCheckTop = 20;
inline constexpr int kHeadMaxWidth = 8;
inline constexpr int kHeadTile = 1024;

enum class HeadTwoReason {
  kShortlist,
  kTopk0,
  kTopkWide,
  kLogprobs,
  kGrammar,
  kFullNorm,
  kExtraRows,
  kPrefill,
  kFrontier,
  kDisabled,
  kUnavailable,
  kWidth,
  kOther,
  kCount
};

struct HeadTwoPair {
  float value;
  std::int32_t id;
};

struct HeadTwoResult {
  std::uint32_t ids[kHeadCapacity];
  float logits[kHeadCapacity];
  std::uint32_t mismatch, missed;
};

// Mutually exclusive reasons make the full-head traffic count additive.
inline HeadTwoReason HeadTwoConsumer(int top_k, std::size_t min_keep,
                                     bool logprobs, bool fullnorm,
                                     bool constraint, bool other) {
  if (logprobs)
    return HeadTwoReason::kLogprobs;
  if (fullnorm)
    return HeadTwoReason::kFullNorm;
  if (constraint)
    return HeadTwoReason::kGrammar;
  if (other || min_keep > kHeadTop)
    return HeadTwoReason::kOther;
  if (top_k == 0)
    return HeadTwoReason::kTopk0;
  if (top_k < 0 || top_k > kHeadTop)
    return HeadTwoReason::kTopkWide;
  return HeadTwoReason::kShortlist;
}

struct HeadTwoPlan {
  HeadTwoReason reason{HeadTwoReason::kOther};
  std::vector<std::uint32_t> extras;
};

inline bool HeadTwoWidth(int width) {
  return width >= 1 && width <= kHeadMaxWidth;
}

}  // namespace gufo::models::qwen38_flash_next
