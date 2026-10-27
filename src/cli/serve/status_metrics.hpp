#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <unistd.h>
#endif

#include "src/cli/serve/generation_metrics.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace status_detail {
// Wire counters have the same exact-integer limit as the machine contract.
inline json::Value Counter(std::uint64_t value) {
  constexpr std::uint64_t limit = 9007199254740991ULL;
  return value <= limit ? json::Value(value) : json::Value();
}

inline json::Value Gauge(std::int64_t value) {
  return value >= 0 ? Counter(static_cast<std::uint64_t>(value))
                    : json::Value();
}

inline json::Value Rate(const std::optional<double>& value) {
  return value && std::isfinite(*value) && *value >= 0.0 ? json::Value(*value)
                                                         : json::Value();
}

inline std::uint64_t MemoryBytes(const char* path, const char* key) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (!line.starts_with(key))
      continue;
    std::istringstream fields(line.substr(std::char_traits<char>::length(key)));
    std::uint64_t kib = 0;
    std::string unit;
    if (fields >> kib >> unit && unit == "kB" &&
        kib <= 9007199254740991ULL / 1024)
      return kib * 1024;
  }
  return 0;
}
}  // namespace status_detail

// Runtime-only: bootstrap status does not include this header or inference
// libraries. The control listener calls it while its backend remains alive.
inline json::Value RuntimeStatusMetrics(const TextGenerationBackend& backend) {
  using status_detail::Counter;
  using status_detail::Gauge;
  auto sample = json::Value::object();
  auto counters = json::Value::object();
  counters["prefill_tokens_total"] =
      Counter(detail::TotalPromptTokens().load(std::memory_order_relaxed));
  counters["decode_tokens_total"] =
      Counter(detail::TotalGenTokens().load(std::memory_order_relaxed));
  counters["cached_tokens_total"] = Counter(
      detail::TotalCachedPromptTokens().load(std::memory_order_relaxed));
  counters["prefill_seconds_total"] =
      detail::TotalPromptSeconds().load(std::memory_order_relaxed);
  counters["decode_seconds_total"] =
      detail::TotalGenSeconds().load(std::memory_order_relaxed);
  counters["max_sequence_tokens"] =
      Counter(detail::MaxSequenceTokens().load(std::memory_order_relaxed));
  counters["draft_rounds_total"] =
      Counter(detail::TotalDraftRounds().load(std::memory_order_relaxed));
  counters["draft_tokens_total"] =
      Counter(detail::TotalDraftTokens().load(std::memory_order_relaxed));
  counters["draft_accepted_tokens_total"] = Counter(
      detail::TotalDraftAcceptedTokens().load(std::memory_order_relaxed));
  counters["device_lost_total"] =
      Counter(detail::DeviceLostTotal().load(std::memory_order_relaxed));
  const auto completed = detail::CompletedRequests().Sample();
  counters["cache_disk_hits_total"] = Counter(completed.disk_hits);
  counters["cache_memory_hits_total"] = Counter(completed.memory_hits);
  counters["cache_misses_total"] = Counter(completed.misses);
  const double successful = static_cast<double>(completed.disk_hits) +
                            static_cast<double>(completed.memory_hits) +
                            static_cast<double>(completed.misses);
  counters["completed_successful_requests_total"] =
      successful <= 9007199254740991.0 ? json::Value(successful)
                                       : json::Value();
  sample["counters"] = std::move(counters);
  sample["cache_hit_ratio"] =
      successful > 0.0 && successful <= 9007199254740991.0
          ? json::Value((static_cast<double>(completed.disk_hits) +
                         static_cast<double>(completed.memory_hits)) /
                        successful)
          : json::Value();

  auto last = json::Value::object();
  last["prefill_tokens_per_second"] =
      status_detail::Rate(completed.prefill_tokens_per_second);
  last["decode_tokens_per_second"] =
      status_detail::Rate(completed.decode_tokens_per_second);
  last["age_ms"] = completed.completed_at
                       ? json::Value(std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() -
                                         *completed.completed_at)
                                         .count())
                       : json::Value();
  last["rate_unit"] = "tokens_per_second";
  last["age_unit"] = "milliseconds";
  sample["last_completed_request"] = std::move(last);
  sample["sessions"] =
      Gauge(detail::RequestsProcessing().load(std::memory_order_relaxed));
  sample["queue"] =
      Gauge(detail::RequestsDeferred().load(std::memory_order_relaxed));

  const auto states = backend.session_states();
  auto slots = states.empty() ? json::Value() : json::Value::array();
  for (std::size_t index = 0; index < states.size(); ++index) {
    const auto& state = states[index];
    auto slot = json::Value::object();
    slot["id"] = Counter(index);
    slot["processing"] = state.processing;
    slot["state"] =
        !state.processing ? "idle"
        : state.processed_prompt_tokens + state.cached_prompt_tokens <
                state.prompt_tokens
            ? "prefilling"
            : "generating";
    slot["tokens_used"] =
        Counter(state.processed_prompt_tokens + state.cached_prompt_tokens +
                state.generated_tokens);
    slot["speculative"] = state.speculative;
    slot["request_id"] =
        state.processing ? Counter(state.request_id) : json::Value();
    slot["context_tokens"] = backend.max_context() > 0
                                 ? Counter(backend.max_context())
                                 : json::Value();
    slot["prompt_tokens"] = Counter(state.prompt_tokens);
    slot["cached_prompt_tokens"] = Counter(state.cached_prompt_tokens);
    slot["prefilled_tokens"] = Counter(state.processed_prompt_tokens);
    slot["generated_tokens"] = Counter(state.generated_tokens);
    slot["remaining_tokens"] =
        state.processing ? Counter(state.remaining_tokens) : json::Value();
    slots.push_back(std::move(slot));
  }
  sample["slots"] = std::move(slots);

#ifdef _WIN32
  const auto rss = gufo_resident_bytes();
  const auto available = gufo_available_physical_bytes();
#else
  const auto rss = status_detail::MemoryBytes("/proc/self/status", "VmRSS:");
  const auto available =
      status_detail::MemoryBytes("/proc/meminfo", "MemAvailable:");
#endif
  auto memory = json::Value::object();
  memory["rss_bytes"] = rss > 0 ? Counter(rss) : json::Value();
  memory["host_available_bytes"] =
      available > 0 ? Counter(available) : json::Value();
  memory["unit"] = "bytes";
  memory["evidence"] = rss > 0 || available > 0 ? "measured" : "unknown";
  sample["memory"] = std::move(memory);
  return sample;
}
}  // namespace gufo::server
