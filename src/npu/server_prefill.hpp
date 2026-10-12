// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_SERVER_PREFILL_HPP_
#define GUFO_NPU_SERVER_PREFILL_HPP_
#include <atomic>
#include <utility>

#include "staging.hpp"

namespace gufo::npu {
struct HipLinearZStorage;
class ServerPrefill {
public:
  explicit ServerPrefill(SessionOptions options)
      : session_(std::move(options)) {}
  ServerPrefill(const core::ExperimentalAuthorization& authorization,
                Knobs knobs)
      : knobs_(std::move(knobs)),
        session_(SessionOptions{.authorization = authorization,
                                .artifacts = knobs_.artifacts,
                                .timing = knobs_.timing}) {}
  const Knobs& knobs() const noexcept { return knobs_; }
  ~ServerPrefill();
  std::shared_ptr<HipLinearZStorage>& TransferStorage() {
    return transfer_storage_;
  }
  bool Start();
  NpuSession& session() noexcept { return session_; }
  bool ready() const noexcept {
    return ready_.load() && session_.Healthy();
  }
  void Reject(std::string error);
  void Hold(std::string error) noexcept;
  // Host enqueue thread only; the mapped invalid word is reset by the first
  // transfer and checked after the existing forward completion wait.
  void BeginChunk() noexcept;
  bool BeginTransfer() noexcept {
    chunk_used_ = true;
    return std::exchange(reset_invalid_, false);
  }
  bool ChunkUsed() const noexcept { return chunk_used_; }

private:
  std::atomic<bool> ready_{false};
  const Knobs knobs_;
  NpuSession session_;
  std::shared_ptr<HipLinearZStorage> transfer_storage_;
  bool chunk_used_ = false, reset_invalid_ = true;
};
}  // namespace gufo::npu
#endif
