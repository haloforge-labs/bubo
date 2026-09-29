#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::hip::qwen27 {

/// GUFO_QWEN27_LOOKUP=N: opt-in prompt lookup for the DFlash2 backend, with
/// DFlash2's own drafts capped at N (the server's --draft-tokens is then the
/// lookup length limit). 0 or unset disables lookup.
[[nodiscard]] inline std::uint32_t LookupDFlashCap() {
  static const std::uint32_t cap = [] {
    const char* value = std::getenv("GUFO_QWEN27_LOOKUP");
    const int parsed = value != nullptr ? std::atoi(value) : 0;
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0U;
  }();
  return cap;
}

/// GUFO_QWEN27_LOOKUP_MODE=append: copied tokens follow DFlash2's drafts
/// instead of replacing the block (the default).
[[nodiscard]] inline bool LookupAppends() {
  static const bool append = [] {
    const char* value = std::getenv("GUFO_QWEN27_LOOKUP_MODE");
    return value != nullptr && std::strcmp(value, "append") == 0;
  }();
  return append;
}

/// Lookup counters, printed to stderr every 64 lookup proposals when lookup
/// is on. Single-session diagnostics.
class LookupStats {
public:
  static LookupStats& Get() {
    static LookupStats stats;
    return stats;
  }
  void Record(std::size_t proposed, std::size_t accepted) {
    if (proposed == 0)
      return;
    proposed_ += proposed;
    accepted_ += accepted;
    if (++proposals_ % 64 == 0)
      std::fprintf(stderr,
                   "qwen27 lookup: %zu proposals, %zu tokens, %zu accepted "
                   "(%.1f%%)\n",
                   proposals_, proposed_, accepted_,
                   100.0 * static_cast<double>(accepted_) /
                       static_cast<double>(proposed_));
  }

private:
  std::size_t proposals_{0};
  std::size_t proposed_{0};
  std::size_t accepted_{0};
};

/// Prompt lookup: proposals copied from the session's own committed tokens.
/// The context ends in a key of kKeyTokens; among the most recent earlier
/// occurrences of that key, the one whose preceding tokens match the context
/// furthest back wins, and the tokens that followed it are the proposals.
/// Matches shorter than kMinMatch propose nothing (short matches are poor
/// predictions at sampling temperature and would displace DFlash2 drafts).
///
/// Proposals depend only on the prefix, so rejection sampling with the
/// proposal as a point mass keeps the target distribution exact.
class PromptLookup {
public:
  static constexpr std::size_t kKeyTokens = 3;
  static constexpr std::size_t kMinMatch = 12;
  static constexpr std::size_t kMaxMatch = 64;
  static constexpr std::size_t kMaxCandidates = 16;

  struct Match {
    std::size_t start{0};   ///< index of the first proposed token
    std::size_t length{0};  ///< matched context tokens (>= kKeyTokens)
  };

  void Clear() {
    head_.clear();
    prev_.clear();
    indexed_.clear();
  }

  /// Indexes `tokens` beyond what was indexed before. A sequence that does
  /// not extend the indexed one (a new request, a rewind) starts over.
  void Extend(std::span<const tokenization::TokenId> tokens) {
    if (tokens.size() < indexed_.size() ||
        (!indexed_.empty() &&
         std::memcmp(tokens.data(), indexed_.data(),
                     indexed_.size() * sizeof(tokenization::TokenId)) != 0)) {
      Clear();
    }
    for (std::size_t end = prev_.size(); end < tokens.size(); ++end) {
      // prev_[end] chains occurrences of the key ending just before `end`.
      std::uint32_t previous = kNone;
      if (end >= kKeyTokens) {
        const auto key = Key(tokens.subspan(end - kKeyTokens, kKeyTokens));
        auto [it, inserted] =
            head_.try_emplace(key, static_cast<std::uint32_t>(end));
        if (!inserted) {
          previous = it->second;
          it->second = static_cast<std::uint32_t>(end);
        }
      }
      prev_.push_back(previous);
    }
    indexed_.assign(tokens.begin(), tokens.end());
  }

  /// Best match for the whole of `tokens` (what Extend() saw), or a
  /// zero-length match. The newest key has no continuation yet, so it only
  /// matches earlier occurrences.
  [[nodiscard]] Match Find(
      std::span<const tokenization::TokenId> tokens) const {
    const std::size_t context = tokens.size();
    if (context < kKeyTokens + 1)
      return {};
    const auto found = head_.find(Key(tokens.subspan(context - kKeyTokens)));
    if (found == head_.end())
      return {};
    Match best{};
    std::uint32_t end = found->second;
    for (std::size_t n = 0; n < kMaxCandidates && end != kNone; ++n) {
      std::size_t length = kKeyTokens;
      while (length < kMaxMatch && end > length && context > length &&
             tokens[end - length - 1] == tokens[context - length - 1]) {
        ++length;
      }
      if (end < context && length > best.length)
        best = {end, length};
      end = prev_[end];
    }
    return best.length >= kMinMatch ? best : Match{};
  }

private:
  static constexpr std::uint32_t kNone =
      std::numeric_limits<std::uint32_t>::max();

  /// Token ids fit in 21 bits, so a key packs a token triple exactly.
  static std::uint64_t Key(std::span<const tokenization::TokenId> tokens) {
    std::uint64_t key = 0;
    for (const auto t : tokens)
      key = (key << 21) | (static_cast<std::uint64_t>(t) & 0x1FFFFF);
    return key;
  }

  std::unordered_map<std::uint64_t, std::uint32_t> head_;
  std::vector<std::uint32_t> prev_;
  std::vector<tokenization::TokenId> indexed_;
};

}  // namespace gufo::hip::qwen27

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_PROMPT_LOOKUP_27B_HPP_
