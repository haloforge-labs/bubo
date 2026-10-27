#ifndef GUFO_SERVER_PROMPT_SCORING_HPP_
#define GUFO_SERVER_PROMPT_SCORING_HPP_

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

/// Largest prompt (in tokens) whose echo + logprobs response is built. The
/// response is about 1-3 KB per token, so this bounds it near 200 MB.
inline constexpr std::size_t kMaxPromptScoreResponseTokens = 65536;
/// Host bytes budgeted per scored token beyond the session: the stored
/// logprob row plus the JSON response share.
inline constexpr std::uint64_t kPromptScoreHostBytesPerToken = 1024;

/// Message of the 400 for a prompt above --max-score-tokens; both numbers.
[[nodiscard]] inline std::string PromptScoreTooLongMessage(
    std::size_t prompt_tokens, std::size_t limit_tokens) {
  return "prompt has " + std::to_string(prompt_tokens) +
         " tokens, which is above --max-score-tokens " +
         std::to_string(limit_tokens);
}

/// Message of the 400 for a prompt above the model context; both numbers.
[[nodiscard]] inline std::string PromptScoreContextMessage(
    std::size_t prompt_tokens, std::size_t context_tokens) {
  return "prompt has " + std::to_string(prompt_tokens) +
         " tokens, which is above the model context of " +
         std::to_string(context_tokens) + " tokens";
}

/// Throws TextGenerationError (413) when a scoring response would be too big.
inline void CheckPromptScoreResponseSize(std::size_t tokens) {
  if (tokens > kMaxPromptScoreResponseTokens)
    throw TextGenerationError(
        TextGenerationErrorCode::kResponseTooLarge,
        "prompt scoring response for " + std::to_string(tokens) +
            " tokens is above the response limit of " +
            std::to_string(kMaxPromptScoreResponseTokens) + " tokens");
}

/// Throws std::invalid_argument (400) above --max-score-tokens, then
/// TextGenerationError (413) above the response cap.
inline void CheckPromptScoreLimits(std::size_t tokens,
                                   std::size_t max_score_tokens) {
  if (tokens > max_score_tokens)
    throw std::invalid_argument(
        PromptScoreTooLongMessage(tokens, max_score_tokens));
  CheckPromptScoreResponseSize(tokens);
}

/// Estimated host bytes a scoring request needs: the private session, the
/// deferred scratch and a per-token share for rows and the response.
[[nodiscard]] inline std::uint64_t PromptScoreHostBytes(
    std::size_t tokens, std::uint64_t session_bytes,
    std::uint64_t scratch_bytes) {
  return session_bytes + scratch_bytes + tokens * kPromptScoreHostBytesPerToken;
}

/// Empty when the host has the memory; otherwise the error message. On a
/// unified-memory APU the device session draws from the same RAM, so the
/// device free figure alone does not see host pressure.
[[nodiscard]] inline std::optional<std::string> PromptScoreHostShortfall(
    std::size_t tokens, std::uint64_t session_bytes,
    std::uint64_t scratch_bytes, std::uint64_t host_available_bytes) {
  const auto needed = PromptScoreHostBytes(tokens, session_bytes, scratch_bytes);
  if (needed <= host_available_bytes)
    return std::nullopt;
  const auto mib = [](std::uint64_t bytes) {
    return std::to_string(bytes >> 20) + " MiB";
  };
  return "prompt scoring of " + std::to_string(tokens) + " tokens needs about " +
         mib(needed) + " of host memory (session " + mib(session_bytes) +
         ", scratch " + mib(scratch_bytes) + ") but only " +
         mib(host_available_bytes) + " is available";
}

/// The error to throw when std::bad_alloc stops scoring at `token_index`.
[[nodiscard]] inline TextGenerationError PromptScoringOutOfMemory(
    std::string_view stage, std::size_t token_index) {
  return TextGenerationError(
      TextGenerationErrorCode::kScoringOutOfMemory,
      "prompt scoring ran out of host memory during " + std::string(stage) +
          " at token " + std::to_string(token_index));
}

/// Runs `work`; std::bad_alloc becomes PromptScoringOutOfMemory.
template <typename Work>
decltype(auto) GuardPromptScoringStage(std::string_view stage,
                                       std::size_t token_index, Work&& work) {
  try {
    return std::forward<Work>(work)();
  } catch (const std::bad_alloc&) {
    throw PromptScoringOutOfMemory(stage, token_index);
  }
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_PROMPT_SCORING_HPP_
