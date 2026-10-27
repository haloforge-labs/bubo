// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_HAL_FAULT_MONITOR_HPP_
#define GUFO_NPU_HAL_FAULT_MONITOR_HPP_
// Detection only. A background subscription records Windows "HAL Id 15" IOMMU
// fault events (and WHEA-Logger events) as they arrive. The NPU chunk code
// reads one atomic sequence number (Faults().hal_seq) before and after a chunk;
// it never queries the event log. Nothing here disables the NPU or changes
// results.
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "fault_state.hpp"

namespace gufo::npu {
struct HalFaultEvent {
  std::uint64_t seq = 0;  // assigned by the monitor, 1-based
  std::string kind;       // "HAL" or "WHEA"
  int id = 0;
  std::string provider;
  std::string event_time;    // TimeCreated SystemTime (UTC) from the record
  std::string received_utc;  // when the subscription thread stored it
  std::string reason;        // FaultReason, if the record carries it
  std::string info;          // FaultInformation, if the record carries it
  std::string device;        // device id / BDF, if the record carries it
  std::string data;          // every EventData item as name=value, for the log
  std::string resolves;      // which BO the FaultInformation address falls in
};

// Event source with its own delivery thread. Tests supply a fake.
class IHalFaultSource {
public:
  virtual ~IHalFaultSource() = default;
  // Start delivering events to sink (called from the source's thread). False
  // with *error set when the source cannot start; the monitor then logs once
  // and stays a no-op.
  virtual bool Start(std::function<void(HalFaultEvent)> sink,
                     std::string* error) = 0;
  // Stop delivery and join the thread. Idempotent.
  virtual void Stop() noexcept = 0;
  // Called from the source thread if it ends for any reason except Stop().
  virtual void SetBrokenHandler(std::function<void(const std::string&)>) {}
  virtual bool Running() const noexcept { return false; }
};

// Parse one rendered event XML (EvtRenderEventXml). Returns false when the
// text has no EventID. kind/seq/received_utc are left for the monitor.
bool ParseEventXml(std::string_view xml, HalFaultEvent* event);
// "[npu]" log text for one event, without level or component prefix.
std::string FormatHalEvent(const HalFaultEvent& event);

class HalFaultMonitor {
public:
  using Log = std::function<void(bool warn, const std::string&)>;
  // now_ms: clock for the log rate limit (default steady_clock, in ms).
  HalFaultMonitor(std::unique_ptr<IHalFaultSource> source, FaultState& state,
                  Log log, std::function<std::int64_t()> now_ms = {});
  static constexpr unsigned kMaxLogPerMinute = 20;
  ~HalFaultMonitor();
  HalFaultMonitor(const HalFaultMonitor&) = delete;
  HalFaultMonitor& operator=(const HalFaultMonitor&) = delete;
  // enabled=false: no thread, no source call. Returns true when delivering.
  bool Start(bool enabled);
  void Stop() noexcept;
  bool running() const noexcept { return running_; }
  bool start_failed() const noexcept { return start_failed_; }
  bool broken() const noexcept { return broken_; }
  // Ring sequence at Start(); events before it are never stored.
  std::uint64_t baseline() const noexcept { return baseline_; }
  // Count of HAL Id 15 events. Single atomic load; this is all the chunk code
  // calls. WHEA events do not move it.
  std::uint64_t Sequence() const noexcept {
    return state_.hal_seq.load(std::memory_order_acquire);
  }
  // Ring sequence of the newest stored event (HAL and WHEA).
  std::uint64_t ring_sequence() const;
  // Stored events (HAL and WHEA) with ring seq > after; the ring holds the
  // last kRing. Event.seq is the ring sequence, not Sequence().
  std::vector<HalFaultEvent> EventsSince(std::uint64_t after) const;
  static constexpr std::size_t kRing = 64;

private:
  void Push(HalFaultEvent event);
  std::unique_ptr<IHalFaultSource> source_;
  FaultState& state_;
  Log log_;
  mutable std::mutex mutex_;
  std::array<HalFaultEvent, kRing> ring_;
  bool running_ = false, start_failed_ = false;
  std::atomic<bool> broken_{false};
  std::uint64_t baseline_ = 0, next_seq_ = 0;
  std::function<std::int64_t()> now_ms_;
  std::int64_t window_start_ms_ = 0;
  unsigned window_logged_ = 0;
  std::uint64_t suppressed_ = 0;
};

// Process-wide monitor. Install* replaces any earlier one (stopping it).
HalFaultMonitor* GlobalHalMonitor();
HalFaultMonitor& InstallGlobalHalMonitor(
    std::unique_ptr<IHalFaultSource> source, bool enabled);
// Real Windows event-log subscription (wevtapi loaded dynamically).
std::unique_ptr<IHalFaultSource> MakeWindowsHalFaultSource();
void StopGlobalHalMonitor() noexcept;
}  // namespace gufo::npu
#endif
