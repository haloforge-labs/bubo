#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/generation_metrics.hpp"

namespace {
using gufo::server::RecordRequestMetrics;
using gufo::server::TextGenerationBackend;
using gufo::server::detail::CompletedRequestMetrics;
using gufo::server::detail::CompletedRequestSample;

int passed = 0;
int failed = 0;

void Check(bool condition, std::string_view name) {
  if (condition) {
    ++passed;
    std::cout << "PASS " << name << '\n';
  } else {
    ++failed;
    std::cerr << "FAIL " << name << '\n';
  }
}

std::uint64_t Total(const CompletedRequestSample& sample) {
  return sample.disk_hits + sample.memory_hits + sample.misses;
}
}  // namespace

int main() {
  constexpr unsigned seeds = 16;
  constexpr unsigned requests_per_seed = 4096;
  std::uint64_t successful_requests = 0;
  const auto global_before = gufo::server::detail::CompletedRequests().Sample();
  for (unsigned seed = 1; seed <= seeds; ++seed) {
    CompletedRequestMetrics metrics;
    std::mt19937 random(seed);
    std::uint64_t disk = 0, memory = 0, miss = 0;
    bool every_prefix_correct = true;
    for (unsigned index = 0; index < requests_per_seed; ++index) {
      TextGenerationBackend::Result result;
      result.prompt_tokens = random() % 512;
      result.cached_prompt_tokens = random() % 512;
      result.prefill_tokens = random() % 512;
      result.completion_tokens = random() % 128;
      result.prefill_ms = 4.0;
      result.decode_ms = 8.0;
      result.cache_disk_hit = (random() % 4) == 0;
      result.cache_hit = (random() % 3) == 0;
      result.cancelled = (random() % 7) == 0;
      result.finish_reason =
          (random() % 11) == 0 ? TextGenerationBackend::FinishReason::kCancelled
                               : TextGenerationBackend::FinishReason::kStop;
      const bool successful =
          !result.cancelled &&
          result.finish_reason !=
              TextGenerationBackend::FinishReason::kCancelled;
      if (successful) {
        ++successful_requests;
        if (result.cache_disk_hit)
          ++disk;
        else if (result.cache_hit)
          ++memory;
        else
          ++miss;
      }
      metrics.Record(result);
      // Exercise the actual scheduler/HTTP completion hook as well as the
      // independent sample object. Each completed result reaches it once.
      RecordRequestMetrics(result);
      const auto sample = metrics.Sample();
      every_prefix_correct &=
          sample.disk_hits == disk && sample.memory_hits == memory &&
          sample.misses == miss && Total(sample) == disk + memory + miss;
    }
    Check(every_prefix_correct,
          "cache_completion_mixed_results_every_prefix_exactly_once");
  }
  const auto global_after = gufo::server::detail::CompletedRequests().Sample();
  Check(Total(global_after) - Total(global_before) == successful_requests,
        "cache_completion_actual_hook_successful_total_exactly_once");

  CompletedRequestMetrics gauges;
  const auto empty = gauges.Sample();
  Check(!empty.completed_at && !empty.prefill_tokens_per_second &&
            !empty.decode_tokens_per_second,
        "last_completed_gauges_unavailable_before_first_completion");
  TextGenerationBackend::Result measured;
  measured.prefill_tokens = 8;
  measured.completion_tokens = 2;
  measured.prefill_ms = 4;
  measured.decode_ms = 8;
  gauges.Record(measured);
  const auto first = gauges.Sample();
  Check(first.prefill_tokens_per_second == 2000.0 &&
            first.decode_tokens_per_second == 250.0 && first.completed_at &&
            *first.completed_at <= std::chrono::steady_clock::now(),
        "last_completed_gauges_have_monotonic_age");
  measured.prefill_ms = 0;
  measured.decode_ms = 0;
  gauges.Record(measured);
  const auto second = gauges.Sample();
  Check(!second.prefill_tokens_per_second && !second.decode_tokens_per_second &&
            second.completed_at && Total(second) == 2,
        "last_completed_gauges_reset_unavailable_durations");
  measured.cancelled = true;
  gauges.Record(measured);
  const auto cancelled = gauges.Sample();
  Check(Total(cancelled) == Total(second) &&
            cancelled.completed_at == second.completed_at,
        "cancelled_completion_changes_neither_cache_counts_nor_last_gauges");

  CompletedRequestMetrics concurrent;
  std::vector<std::thread> workers;
  constexpr unsigned workers_count = 4;
  constexpr unsigned per_worker = 1024;
  for (unsigned worker = 0; worker < workers_count; ++worker) {
    workers.emplace_back([&concurrent, worker] {
      for (unsigned request = 0; request < per_worker; ++request) {
        TextGenerationBackend::Result result;
        result.cache_disk_hit = (request + worker) % 3 == 0;
        result.cache_hit = (request + worker) % 3 == 1;
        concurrent.Record(result);
      }
    });
  }
  for (auto& worker : workers)
    worker.join();
  Check(Total(concurrent.Sample()) == workers_count * per_worker,
        "concurrent_completions_total_exactly_once");
  std::cout << "Cache completion properties: " << passed << " passed, "
            << failed << " failed; " << seeds * requests_per_seed
            << " mixed requests, " << successful_requests << " successful\n";
  return failed == 0 ? 0 : 1;
}
