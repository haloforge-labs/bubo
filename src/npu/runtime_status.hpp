// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_RUNTIME_STATUS_HPP_
#define GUFO_NPU_RUNTIME_STATUS_HPP_
// Producer side of the "npu_prefill_runtime" status contract.
//
// This header has no dependency on the status implementation. The snapshot
// owner installs a publisher, called on state changes and at request end.
// Never per layer; a publisher that throws is swallowed.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "src/core/json.hpp"

namespace gufo::npu {
enum class NpuState { kOn, kOff, kTripped, kFaulted, kUnavailable };
const char* NpuStateName(NpuState state) noexcept;

struct NpuRuntimeStatus {
  NpuState state = NpuState::kOff;
  // null, a startup reason ("NPU startup failed."), "power", "memory", "fault".
  std::optional<std::string> reason;
  std::uint64_t trips = 0;  // G1 holds since start; nothing emits tripped yet
  std::uint64_t hal_faults = 0, whea_events = 0, invalid_chunks = 0;
  // INTERNAL, like Trip: ToJson does not emit it (the contract schema allows no
  // extra keys). Chunks run again on the GPU after an NPU fault.
  std::uint64_t gpu_retries = 0;
  struct ChipPower {
    std::string state = "off";     // ok | unavailable | off
    std::optional<double> w_mean;  // last request; absent = JSON null
    double window_s = 0;           // that request's NPU-active seconds
  } chip_power;
  struct Memory {
    double min_available_mib = 0, max_commit_pct = 0;  // last request
  };
  std::optional<Memory> memory;  // absent until a request used the NPU
  struct Scheme {
    std::string guid, name;
  };
  std::optional<Scheme> scheme;  // read once at NPU start; absent if unreadable
  // While state is tripped: what tripped it. INTERNAL: ToJson does not emit it
  // (the contract schemas allow no extra keys). The contract's own fields carry
  // the numbers: chip_power.w_mean is the tripping mean, memory.* the tripping
  // window.
  struct Trip {
    bool sticky = false;
    std::optional<double> w_mean;  // tripping chunk mean (power)
    std::optional<double> min_available_mib, max_commit_pct;  // memory window
  };
  std::optional<Trip> trip;
};

// Absent optionals: reason and chip_power.w_mean are written as null (the
// contract allows null there); memory and scheme are left out.
json::Value ToJson(const NpuRuntimeStatus& status);

using NpuRuntimeStatusPublisher = std::function<void(const NpuRuntimeStatus&)>;
// Thread-safe; an empty function clears it (default: no publisher).
void SetNpuRuntimeStatusPublisher(NpuRuntimeStatusPublisher publisher);

// ---- called by the NPU code (src/npu) ----
// Snapshot of the current values (no publisher involved).
NpuRuntimeStatus CurrentNpuRuntimeStatus();
// Builds the snapshot and calls the publisher. No-op without a publisher;
// swallows every exception.
void PublishNpuRuntimeStatus() noexcept;
// Start-up result: armed (on) or not armed (off with reason).
void NoteNpuStartup(bool armed, const std::string& reason) noexcept;
// G1 guard hold (timed). A hold never fails a request; faulted always wins.
struct NpuHoldInfo {
  bool held = false;   // false = re-armed
  std::string reason;  // "power" | "memory"
  std::uint64_t trips = 0;
  bool sticky = false;
  std::optional<double> w_mean, min_available_mib, max_commit_pct;
};
void NoteNpuHold(const NpuHoldInfo& info) noexcept;
// Sticky: disabled for the process after an invalid chunk or XRT failure.
void NoteNpuFaulted() noexcept;
void NoteNpuScheme(const std::string& guid, const std::string& name) noexcept;
// Values of the request that just finished (set from the request telemetry).
struct LastRequestSummary {
  std::optional<double> power_w_mean;
  double window_s = 0;
  double min_available_mib = 0, max_commit_pct = 0;
};
void NoteLastRequest(const LastRequestSummary& summary) noexcept;
// Tests: back to the initial state (publisher cleared too).
void ResetNpuRuntimeStatusForTest();
}  // namespace gufo::npu
#endif
