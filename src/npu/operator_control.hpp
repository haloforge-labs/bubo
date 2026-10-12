// SPDX-License-Identifier: MIT
#pragma once
#include "session.hpp"
#include "src/core/json.hpp"
namespace gufo::npu {
inline json::Value OperatorControl(NpuSession& session,
                                   std::string_view action) {
  auto result = json::Value::object();
  std::string error;
  if (action == "yield" || action == "resume") {
    const auto changed = session.YieldKeepalive(action == "yield");
    error = changed.error;
  } else if (action != "status") {
    error = "unknown NPU control action";
  }
  const auto status = session.Status();
  result["healthy"] = status.healthy && error.empty();
  result["paused"] = status.paused;
  result["context_retained"] =
      status.healthy ? json::Value(true) : json::Value();
  result["context_retained_source"] =
      "inferred from healthy persistent session; not independently measured";
  result["yield_expired"] = status.yield_expired;
  result["yield_maximum_ms"] = json::Value();
  result["yield_policy"] =
      "pause keepalive and offload; explicit resume; GPU remains available";
  result["error"] = error;
  const auto count = [](std::uint64_t n) {
    return json::Value::parsed_number(static_cast<double>(n),
                                      std::to_string(n));
  };
  result["heartbeat_submitted"] = count(status.heartbeat_submitted);
  result["heartbeat_completed"] = count(status.heartbeat_completed);
  result["prefill_submitted"] = count(status.prefill_submitted);
  result["prefill_completed"] = count(status.prefill_completed);
  return result;
}
}  // namespace gufo::npu
