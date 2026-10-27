// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_FAULT_STATE_HPP_
#define GUFO_NPU_FAULT_STATE_HPP_
// Process-wide NPU fault bookkeeping. Header only: the chunk hot path reads
// these atomics and does nothing else (no locks, no event-log calls).
#include <atomic>
#include <chrono>
#include <cstdint>

namespace gufo::npu {
struct FaultState {
  // HAL Id 15 events stored by the subscription thread. WHEA events never
  // move this, so "HAL event during a chunk" cannot be a WHEA event.
  std::atomic<std::uint64_t> hal_seq{0};
  std::atomic<std::uint64_t> hal_faults{0};   // same count as hal_seq
  std::atomic<std::uint64_t> whea_events{0};  // WHEA-Logger events
  // 0 off, 1 on, 2 broken (thread ended abnormally), 3 unavailable.
  std::atomic<int> monitor_state{0};
  // Chip power logging: 0 off, 1 on, 3 unavailable (logged once). No gate.
  std::atomic<int> chip_power_state{0};
  // Memory\Pages/sec: 0 off, 1 on, 3 unavailable (logged once).
  std::atomic<int> pages_state{0};
  // PDH GPU memory counters (sampler thread): 0 off, 1 on, 3 unavailable.
  std::atomic<int> gpu_mem_state{0};
  // Runtime status: invalid-chunk count, and "NPU armed at start-up" (a failure
  // before that is "off", after it "faulted").
  std::atomic<std::uint64_t> invalid_chunks{0};
  // L-REL-2: chunks run again on the GPU after an NPU fault (process total).
  std::atomic<std::uint64_t> gpu_retries{0};
  std::atomic<bool> armed{false};
  // steady_clock ns of the last real (non-heartbeat) NPU dispatch; 0 = none.
  std::atomic<std::int64_t> last_real_dispatch_ns{0};
};
inline FaultState& Faults() {
  static FaultState state;
  return state;
}
inline const char* MonitorStateName() noexcept {
  switch (Faults().monitor_state.load(std::memory_order_relaxed)) {
    case 1:
      return "on";
    case 2:
      return "broken";
    case 3:
      return "unavailable";
    default:
      return "off";
  }
}
inline const char* ChipPowerStateName() noexcept {
  switch (Faults().chip_power_state.load(std::memory_order_relaxed)) {
    case 1:
      return "on";
    case 3:
      return "unavailable";
    default:
      return "off";
  }
}
inline const char* GpuMemStateName() noexcept {
  switch (Faults().gpu_mem_state.load(std::memory_order_relaxed)) {
    case 1:
      return "on";
    case 3:
      return "unavailable";
    default:
      return "off";
  }
}
inline const char* PagesStateName() noexcept {
  switch (Faults().pages_state.load(std::memory_order_relaxed)) {
    case 1:
      return "on";
    case 3:
      return "unavailable";
    default:
      return "off";
  }
}
inline std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
inline void NoteRealDispatch() noexcept {
  Faults().last_real_dispatch_ns.store(SteadyNowNs(),
                                       std::memory_order_relaxed);
}
// Milliseconds since the last real dispatch; -1 when there was none.
inline std::int64_t IdleGapMs() noexcept {
  const auto last =
      Faults().last_real_dispatch_ns.load(std::memory_order_relaxed);
  return last == 0 ? -1 : (SteadyNowNs() - last) / 1000000;
}
}  // namespace gufo::npu
#endif
