#pragma once

#ifndef BUBO_SERVE_TESTING
#error "The fake serve backend belongs only to the isolated CPU test target"
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace bubo::cli::testing {
// This backend owns no model, GPU, scheduler or disk cache. The production
// HTTP transport records its completed results through the usual hook.
class FakeServeBackend final : public gufo::server::TextGenerationBackend {
public:
  explicit FakeServeBackend(std::filesystem::path gate = {})
      : gate_(std::move(gate)) {}
  std::string model_id() const override { return "bubo-test-only-fake"; }
  bool ready() const override { return loaded_.load(); }
  std::uint32_t max_context() const override { return 4096; }
  void Load() { loaded_.store(true); }

  std::vector<SessionState> session_states() const override {
    std::unique_lock lock(scheduler_gate_);
    if (!gate_.empty() && processing_ != 0) {
      std::ofstream(gate_.string() + ".sampling").put('\n');
      // Waiting releases the mutex so all four generations can enter even
      // when the sampler arrives first. A queued exclusive shared_mutex lock
      // could otherwise prevent later readers from entering indefinitely.
      generations_finished_.wait(lock, [this] { return processing_ == 0; });
    }
    SessionState slot;
    slot.processing = processing_ != 0;
    return std::vector<SessionState>(gate_.empty() ? 1 : 4, slot);
  }

  Result complete(std::string_view prompt, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck& cancelled = {},
                  const TokenCallback& token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& = {}, bool = true) override {
    struct Processing {
      FakeServeBackend& backend;
      unsigned index;
      explicit Processing(FakeServeBackend& value) : backend(value) {
        std::lock_guard lock(backend.scheduler_gate_);
        index = ++backend.processing_;
        gufo::server::detail::RequestsProcessing().fetch_add(1);
      }
      ~Processing() {
        {
          std::lock_guard lock(backend.scheduler_gate_);
          --backend.processing_;
          gufo::server::detail::RequestsProcessing().fetch_sub(1);
        }
        backend.generations_finished_.notify_all();
      }
    } processing(*this);
    // Test-only scheduler contention: sampling waits until every generation
    // finishes, while arrivals remain independent of the blocked sampler.
    // Production lifecycle/control must keep answering while sampling waits.
    if (!gate_.empty()) {
      std::ofstream(gate_.string() + ".entered-" +
                    std::to_string(processing.index))
          .put('\n');
      while (!std::filesystem::exists(gate_) && !(cancelled && cancelled()))
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Result result;
    result.client_id = client_id;
    result.prompt_tokens = 8;
    result.cache_disk_hit = prompt.starts_with("disk");
    result.cache_hit = result.cache_disk_hit || prompt.starts_with("memory");
    result.cached_prompt_tokens = result.cache_hit ? 4 : 0;
    result.prefill_tokens = result.prompt_tokens - result.cached_prompt_tokens;
    result.prefill_ms = 2;
    result.decode_ms = 4;
    result.text = "fake response";
    for (const std::string_view piece : {"fake ", "response"}) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      if ((cancelled && cancelled()) || (token && !token(piece))) {
        result.cancelled = true;
        result.finish_reason = FinishReason::kCancelled;
        break;
      }
      ++result.completion_tokens;
    }
    return result;
  }

  Result chat(const gufo::server::ChatRequest& request, std::size_t max_tokens,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancelled = {},
              const TokenCallback& token = {}) override {
    const std::string_view prompt =
        request.messages.empty()
            ? std::string_view{}
            : std::string_view(request.messages.back().content);
    return complete(prompt, max_tokens, sampling, cancelled, token,
                    request.client_id);
  }

  std::size_t count_tokens(std::string_view) const override { return 8; }
  std::optional<std::size_t> count_chat_tokens(
      const gufo::server::ChatRequest&) const override {
    return 8;
  }

private:
  std::filesystem::path gate_;
  mutable std::mutex scheduler_gate_;
  mutable std::condition_variable generations_finished_;
  std::atomic<bool> loaded_{false};
  unsigned processing_ = 0;
};
}  // namespace bubo::cli::testing
