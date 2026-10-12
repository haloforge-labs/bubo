// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_SESSION_HPP_
#define GUFO_NPU_SESSION_HPP_

#include <chrono>
#include <filesystem>
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
  bool result_valid = false, owner_recovery = false;
  std::string error;
  std::vector<std::byte> output;
};

struct KeepaliveStatus {
  bool healthy = false, paused = false;
  std::uint64_t heartbeat_submitted = 0, heartbeat_completed = 0;
  std::uint64_t prefill_submitted = 0, prefill_completed = 0;
  bool yield_expired = false;
};

class NpuSession {
public:
  explicit NpuSession(SessionOptions options);
  ~NpuSession();
  NpuSession(const NpuSession&) = delete;
  NpuSession& operator=(const NpuSession&) = delete;
  NpuResult Start();
  bool Healthy() const noexcept;
  bool Paused() const noexcept;
  std::chrono::milliseconds Timeout() const noexcept;
  bool DispatchBatch(
      std::span<const std::byte> a,
      const std::vector<std::vector<std::uint8_t>>& weights,
      const std::function<void(std::size_t, std::span<const std::byte>)>&
          consume);
  void PrepareWeight(std::span<const std::byte> b);
  NpuResult Dispatch(std::span<const std::byte> a, std::span<const std::byte> b,
                     DispatchInfo info);
  NpuResult YieldKeepalive(bool pause);
  KeepaliveStatus Status() const noexcept;
  NpuResult RejectResult(std::string error, bool owner_recovery = false);
  NpuResult Shutdown();

private:
  struct State;
  std::shared_ptr<State> state_;
};
}  // namespace gufo::npu
#endif
