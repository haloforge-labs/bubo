#ifndef GUFO_SERVER_LOGPROBS_HPP_
#define GUFO_SERVER_LOGPROBS_HPP_

#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/prompt_scoring.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

using LogprobTokenText = std::function<std::string(sampling::TokenId)>;

inline double JsonLogprob(double value) {
  // A masked token can still be emitted by a post-filter behaviour rewrite.
  if (value == -INFINITY)
    return -9999;
  if (!std::isfinite(value))
    throw std::invalid_argument("invalid nonfinite token logprob");
  return value;
}

inline json::Value LogprobToken(const std::string& text, double value) {
  auto result = json::Value::object();
  result["token"] = core::Utf8Decoder{}.Push(text, true);
  result["logprob"] = JsonLogprob(value);
  result["bytes"] = json::Value::array();
  for (const unsigned char byte : text)
    result["bytes"].push_back(static_cast<int>(byte));
  return result;
}

inline json::Value ChatLogprobs(std::span<const sampling::TokenLogprob> scores,
                               const LogprobTokenText& token_text) {
  auto result = json::Value::object();
  result["content"] = json::Value::array();
  for (const auto& score : scores) {
    auto token = LogprobToken(token_text(score.token), score.logprob);
    token["top_logprobs"] = json::Value::array();
    for (const auto& top : score.top_logprobs)
      token["top_logprobs"].push_back(
          LogprobToken(token_text(top.token), top.value));
    result["content"].push_back(std::move(token));
  }
  return result;
}

inline json::Value LegacyLogprobs(
    std::span<const sampling::TokenId> tokens,
    std::span<const std::optional<sampling::TokenLogprob>> scores,
    const LogprobTokenText& token_text) {
  if (tokens.size() != scores.size())
    throw std::invalid_argument("token and logprob counts differ");
  // Backstop for the response cap that start_score also enforces.
  CheckPromptScoreResponseSize(tokens.size());
  auto result = json::Value::object();
  for (const auto* field : {"tokens", "token_logprobs", "top_logprobs",
                            "text_offset"}) {
    result[field] = json::Value::array();
    result[field].reserve(tokens.size());
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto text = token_text(tokens[i]);
    result["tokens"].push_back(core::Utf8Decoder{}.Push(text, true));
    result["text_offset"].push_back(offset);
    // Offsets count UTF-8 characters; bytes retain the original token bytes.
    for (const unsigned char byte : text)
      offset += (byte & 0xC0) != 0x80;
    if (!scores[i]) {
      result["token_logprobs"].push_back(nullptr);
      result["top_logprobs"].push_back(nullptr);
      continue;
    }
    if (scores[i]->token != tokens[i])
      throw std::invalid_argument("token and logprob IDs differ");
    result["token_logprobs"].push_back(JsonLogprob(scores[i]->logprob));
    auto top_tokens = json::Value::object();
    top_tokens.reserve(scores[i]->top_logprobs.size());
    for (const auto& top : scores[i]->top_logprobs)
      top_tokens[core::Utf8Decoder{}.Push(token_text(top.token), true)] =
          JsonLogprob(top.value);
    result["top_logprobs"].push_back(std::move(top_tokens));
  }
  return result;
}

inline json::Value LegacyLogprobs(
    std::span<const sampling::TokenLogprob> scores,
    const LogprobTokenText& token_text) {
  std::vector<sampling::TokenId> tokens;
  std::vector<std::optional<sampling::TokenLogprob>> nullable;
  tokens.reserve(scores.size());
  nullable.reserve(scores.size());
  for (const auto& score : scores) {
    tokens.push_back(score.token);
    nullable.push_back(score);
  }
  return LegacyLogprobs(tokens, nullable, token_text);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_LOGPROBS_HPP_
