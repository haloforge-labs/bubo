#ifndef GUFO_MODELS_QWEN36_A3B_PROMPT_LOOKUP_HPP_
#define GUFO_MODELS_QWEN36_A3B_PROMPT_LOOKUP_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace gufo::models::qwen36_a3b {

/// Prompt lookup: proposals copied from the session's own tokens (the
/// A3B copy of Flash-Next's index, with the match gate as a parameter).
/// The drafter's context (committed tokens plus the step's drafts so far)
/// ends in a key of kKeyTokens; among the most recent earlier occurrences
/// of that key, the one whose preceding tokens match the context furthest
/// back wins, and the tokens that followed it are the proposals. Matches
/// shorter than `min_match` propose nothing: short matches are poor
/// predictions at sampling temperature and would displace MTP drafts.
///
/// Proposals depend only on the prefix, so rejection sampling (with the
/// proposal as a point mass) keeps the target distribution exact.
class PromptLookup {
public:
  static constexpr std::size_t kKeyTokens = 3;
  static constexpr std::size_t kMaxMatch = 64;
  static constexpr std::size_t kMaxCandidates = 16;

  struct Match {
    std::size_t start{0};   ///< index of the first proposal
    std::size_t length{0};  ///< matched context tokens (>= kKeyTokens)
  };

  explicit PromptLookup(std::size_t min_match = 12) : min_match_(min_match) {}

  void Clear() {
    head_.clear();
    prev_.clear();
  }

  /// Indexes tokens appended since the last call. `committed` must extend
  /// what was indexed before (Clear() after anything else).
  void Extend(std::span<const std::int32_t> committed) {
    if (committed.size() < prev_.size()) Clear();
    for (std::size_t end = prev_.size(); end < committed.size(); ++end) {
      // prev_[end] chains occurrences of the key ending just before `end`.
      std::uint32_t previous = kNone;
      if (end >= kKeyTokens) {
        const auto key = Key(committed.subspan(end - kKeyTokens, kKeyTokens));
        auto [it, inserted] =
            head_.try_emplace(key, static_cast<std::uint32_t>(end));
        if (!inserted) {
          previous = it->second;
          it->second = static_cast<std::uint32_t>(end);
        }
      }
      prev_.push_back(previous);
    }
  }

  /// Best match for the context `committed` + `tail`, or a zero-length
  /// match. `committed` must be what Extend() saw.
  [[nodiscard]] Match Find(std::span<const std::int32_t> committed,
                           std::span<const std::int32_t> tail) const {
    const std::size_t context = committed.size() + tail.size();
    if (context < kKeyTokens) return {};
    const auto at = [&](std::size_t i) {
      return i < committed.size() ? committed[i] : tail[i - committed.size()];
    };
    std::int32_t key_tokens[kKeyTokens];
    for (std::size_t i = 0; i < kKeyTokens; ++i)
      key_tokens[i] = at(context - kKeyTokens + i);
    const auto found = head_.find(Key(key_tokens));
    if (found == head_.end()) return {};
    Match best{};
    std::uint32_t end = found->second;
    for (std::size_t n = 0; n < kMaxCandidates && end != kNone; ++n) {
      // An occurrence ending at `end` proposes committed[end...]. The key
      // itself matched; extend backwards against the context.
      std::size_t length = kKeyTokens;
      while (length < kMaxMatch && end > length && context > length &&
             committed[end - length - 1] == at(context - length - 1))
        ++length;
      if (end < committed.size() && length > best.length) best = {end, length};
      end = prev_[end];
    }
    return best.length >= min_match_ ? best : Match{};
  }

private:
  static constexpr std::uint32_t kNone =
      std::numeric_limits<std::uint32_t>::max();

  /// Token ids fit in 21 bits, so a key packs exactly.
  static std::uint64_t Key(std::span<const std::int32_t> tokens) {
    std::uint64_t key = 0;
    for (const auto t : tokens)
      key = (key << 21) |
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(t)) &
             0x1FFFFF);
    return key;
  }

  std::size_t min_match_;
  std::unordered_map<std::uint64_t, std::uint32_t> head_;
  std::vector<std::uint32_t> prev_;
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_PROMPT_LOOKUP_HPP_
