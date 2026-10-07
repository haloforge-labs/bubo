#ifndef GUFO_SERVER_DEBUG_REQUEST_HPP_
#define GUFO_SERVER_DEBUG_REQUEST_HPP_

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/json.hpp"

namespace gufo::server {

// Parse alongside the existing body, so an ordinary request needs no second
// JSON parse and requested diagnostics cannot silently disappear.
inline std::optional<std::string> ParseDebugRequest(const json::Value& body,
                                                   bool enabled,
                                                   bool* timeline) {
  const auto* debug = body.find("debug");
  if (!debug || debug->is_null())
    return std::nullopt;
  if (!debug->is_object())
    return "'debug' must be an object";
  bool probe = false;
  for (const auto& [key, value] : debug->members()) {
    if (key != "timeline" && key != "cache_probe")
      return "unknown debug field: " + key;
    if (!value.is_bool())
      return "'debug." + key + "' must be a boolean";
    if (key == "timeline")
      *timeline = value.as_bool();
    else
      probe = value.as_bool();
  }
  if ((*timeline || probe) && !enabled)
    return "debug hooks are disabled (start the server with --debug-hooks)";
  if (probe)
    return "cache inspection is unavailable: exact hit accounting requires "
           "cache instrumentation";
  return std::nullopt;
}

inline std::optional<std::string> ParseChatLogprobs(const json::Value& body,
                                                   std::int32_t* top) {
  bool enabled = false;
  if (const auto* value = body.find("logprobs"); value && !value->is_null()) {
    if (!value->is_bool())
      return "'logprobs' must be a boolean";
    enabled = value->as_bool();
  }
  std::int32_t count = 0;
  if (const auto* value = body.find("top_logprobs"); value && !value->is_null()) {
    if (!value->is_number() || !std::isfinite(value->as_double()) || value->as_double() < 0 ||
        value->as_double() > 20 ||
        value->as_double() != static_cast<int>(value->as_double()))
      return "'top_logprobs' must be an integer in [0,20]";
    count = static_cast<std::int32_t>(value->as_double());
    if (!enabled && count != 0)
      return "'top_logprobs' requires logprobs=true";
  }
  *top = enabled ? count : -1;
  return std::nullopt;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_REQUEST_HPP_
