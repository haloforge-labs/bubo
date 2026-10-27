// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_SERVER_PREFILL_HPP_
#define GUFO_NPU_SERVER_PREFILL_HPP_
#include <atomic>
#include <memory>
#include <utility>

#include "fault_telemetry.hpp"
#include "guard.hpp"
#include "staging.hpp"

namespace gufo::npu {
struct HipLinearZStorage;
class ServerPrefill {
public:
  explicit ServerPrefill(SessionOptions options)
      : session_(std::move(options)), guard_(MakeGuard()) {}
  ServerPrefill(const core::ExperimentalAuthorization& authorization,
                Knobs knobs)
      : knobs_(std::move(knobs)),
        session_(SessionOptions{.authorization = authorization,
                                .artifacts = knobs_.artifacts,
                                .timing = knobs_.timing}),
        guard_(MakeGuard()) {}
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
  // What the NPU lanes check. ready() plus the guard hold, LATCHED once per
  // chunk in BeginChunk (a hold that starts mid-chunk waits for the next
  // chunk, so a chunk never mixes NPU and GPU layers and the transfer
  // constructor cannot fail on a flip). The planner sites keep using ready():
  // a hold must not change the chunk plan.
  bool lane_ready() const noexcept { return ready() && !chunk_hold_; }
  NpuGuard* guard() noexcept { return guard_.get(); }
  // Tests: replace the guard (null = no guard).
  void ReplaceGuardForTest(std::unique_ptr<NpuGuard> guard) {
    guard_ = std::move(guard);
  }
  void Reject(std::string error);
  void Hold(std::string error) noexcept;
  // Host enqueue thread only; the mapped invalid word is reset by the first
  // transfer and checked after the existing forward completion wait.
  // tokens = prompt tokens in this forward pass (failure-line context only).
  void BeginChunk(unsigned tokens = 0) noexcept;
  // Counts BeginChunk calls: a caller that sees it move knows a Forward pass
  // reached its chunk start (ChunkUsed() is only meaningful then). Host
  // enqueue thread only.
  std::uint64_t chunk_serial() const noexcept { return chunk_serial_; }
  // The guard hold will still be on when the next chunk starts, so that chunk
  // cannot use the NPU. Host enqueue thread, before BeginChunk.
  bool HoldWillStay() const { return guard_ && guard_->StaysHeld(); }
  const ChunkStart& chunk() const noexcept { return chunk_; }
  bool BeginTransfer() noexcept {
    // The first pass after a guard re-arm may be GPU-only: the marker waits for
    // the first chunk that really uses the NPU.
    if (!chunk_used_ && guard_ && guard_->ConsumeRearmFlag())
      chunk_.first_after_rearm = true;
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
  bool chunk_hold_ = false;  // latched by BeginChunk, host enqueue thread only
  std::uint64_t chunk_serial_ = 0;
  std::uint64_t logged_bo_generation_ = 0;  // BO address table version logged
  ChunkStart chunk_;
  std::unique_ptr<NpuGuard> guard_;
  std::unique_ptr<NpuGuard> MakeGuard();
};
}  // namespace gufo::npu
#endif
