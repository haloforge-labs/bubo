#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <optional>

#include "src/cli/machine_output.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {
inline json::Value RequestResultMetrics(
    const TextGenerationBackend::Result& result) {
  auto row = json::Value::object();
  const auto counter = [](std::uint64_t value) {
    return value <= 9007199254740991ULL ? json::Value(value) : json::Value();
  };
  const auto number = [](double value) {
    return std::isfinite(value) && value >= 0 ? json::Value(value)
                                              : json::Value();
  };
  row["slot"] = result.slot_id ? counter(*result.slot_id) : json::Value();
  row["prompt_tokens"] = counter(result.prompt_tokens);
  row["completion_tokens"] = counter(result.completion_tokens);
  row["cached_tokens"] = counter(result.cached_prompt_tokens);
  row["ttft_ms"] = result.completion_tokens && result.ttft_ms > 0
                       ? number(result.ttft_ms)
                       : json::Value();
  row["prefill_tps"] =
      result.prefill_ms > 0
          ? number(1000.0 * result.prefill_tokens / result.prefill_ms)
          : json::Value();
  row["decode_tps"] =
      result.decode_ms > 0
          ? number(1000.0 * result.completion_tokens / result.decode_ms)
          : json::Value();
  row["cache_kind"] = result.cache_disk_hit ? "disk"
                      : result.cache_hit    ? "memory"
                                            : "miss";
  row["finish_reason"] =
      result.cancelled ? "cancelled"
      : result.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "length"
      : result.finish_reason ==
              TextGenerationBackend::FinishReason::kStopSequence
          ? "stop_sequence"
          : "stop";
  return row;
}
// Transport-local observer. The scheduler keeps its own metrics; the HTTP
// adapter observes only the aggregate Result returned on this thread.
inline thread_local std::function<void(const TextGenerationBackend::Result&)>
    http_result_observer;
class HttpResultScope {
public:
  explicit HttpResultScope(json::Value& result)
      : previous_(std::move(http_result_observer)) {
    http_result_observer = [&result](const auto& value) {
      result = RequestResultMetrics(value);
    };
  }
  ~HttpResultScope() { http_result_observer = std::move(previous_); }

private:
  std::function<void(const TextGenerationBackend::Result&)> previous_;
};
inline json::Value CompletedRequestRow(std::string id, json::Value metrics,
                                       std::string outcome) {
  if (!metrics.is_object()) {
    metrics = json::Value::object();
    for (const auto* field : {"slot", "prompt_tokens", "completion_tokens",
                              "cached_tokens", "ttft_ms", "prefill_tps",
                              "decode_tps", "cache_kind", "finish_reason"})
      metrics[field] = json::Value();
  }
  metrics["id"] = std::move(id);
  metrics["time"] = bubo::cli::UtcTime();
  metrics["outcome"] = std::move(outcome);
  return metrics;
}
class RequestHistory {
public:
  static constexpr std::size_t capacity = 50;
  void Record(json::Value row) {
    std::lock_guard lock(mutex_);
    if (rows_.size() == capacity)
      rows_.pop_front();
    rows_.push_back(std::move(row));
  }
  json::Value Sample() const {
    std::lock_guard lock(mutex_);
    auto rows = json::Value::array();
    for (const auto& row : rows_)
      rows.push_back(row);
    return rows;
  }

private:
  mutable std::mutex mutex_;
  std::deque<json::Value> rows_;
};
}  // namespace gufo::server
