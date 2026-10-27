// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_SESSION_HPP_
#define GUFO_NPU_SESSION_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "private_bo_adapter.hpp"
#include "src/core/experimental_authorization.hpp"

namespace gufo::npu {
struct Timing {
  std::chrono::milliseconds timeout{8000};
  std::chrono::milliseconds keepalive_interval{1000};
};
Timing ParseTiming(std::string_view timeout, std::string_view keepalive,
                   std::string_view interval);
bool ProcessFailed() noexcept;
void FailProcess(std::string_view error) noexcept;

struct SessionOptions {
  core::ExperimentalAuthorization authorization;
  std::filesystem::path artifacts;
  Timing timing;
};
struct DispatchInfo {
  int layer = -1, slice = -1, group = -1;
  std::string lane;
  bool heartbeat = false;
};
struct NpuResult {
  bool submitted = false, executed = false, healthy = false;
  bool result_valid = false, dma_uncertain = false;
  std::string error;
  std::vector<std::byte> output;
};

class NpuSession {
public:
  explicit NpuSession(SessionOptions options);
  ~NpuSession();
  NpuSession(const NpuSession&) = delete;
  NpuSession& operator=(const NpuSession&) = delete;
  NpuResult Start();
  bool Healthy() const noexcept;
  std::chrono::milliseconds Timeout() const noexcept;
  bool DispatchBatch(
      std::span<const std::byte> a,
      const std::vector<std::vector<std::uint8_t>>& weights,
      const std::function<void(std::size_t, std::span<const std::byte>)>&
          consume);
  void PrepareWeight(std::span<const std::byte> b);
  // Guard hold: no keepalive heartbeats while paused (session and BOs stay).
  void SetKeepalivePaused(bool paused) noexcept;
  // One synchronous heartbeat through the normal dispatch path. False (with
  // *error) on failure; the normal fault path (FailProcess) has then run.
  bool WarmUpBeat(std::string* error);
  // Sampled buffer-page addresses for the log-only residency trace.
  std::vector<const void*> SamplePages();
  NpuResult Dispatch(std::span<const std::byte> a, std::span<const std::byte> b,
                     DispatchInfo info);
  NpuResult RejectResult(std::string error);
  NpuResult Shutdown();

private:
  struct State;
  std::shared_ptr<State> state_;
};
}  // namespace gufo::npu
#endif
