#ifndef GUFO_SERVER_DEBUG_RESPONSE_HPP_
#define GUFO_SERVER_DEBUG_RESPONSE_HPP_

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

inline void ValidateGeneratedLogprobs(
    const TextGenerationBackend::Result& result) {
  if (result.tokens.size() != result.logprobs.size())
    throw std::logic_error("generated token and logprob counts differ");
  for (std::size_t i = 0; i < result.tokens.size(); ++i)
    if (result.tokens[i] != result.logprobs[i].token)
      throw std::logic_error("generated token and logprob IDs differ");
}

inline void AddDebugTimeline(json::Value& response,
                             const TextGenerationBackend::Result& result) {
  if (result.debug_timeline) {
    response["gufo"]["token_ids"] = json::Value::array();
    for (const auto token : result.tokens)
      response["gufo"]["token_ids"].push_back(static_cast<std::size_t>(token));
    response["gufo"]["timeline"] = result.debug_timeline->Serialize();
    response["gufo"]["timeline_truncated"] = result.debug_timeline->truncated();
  }
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_RESPONSE_HPP_
