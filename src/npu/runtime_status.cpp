// SPDX-License-Identifier: MIT
#include "runtime_status.hpp"

#include <atomic>
#include <memory>
#include <mutex>

#include "fault_state.hpp"

namespace gufo::npu {
namespace {
struct Shared {
  std::mutex mutex;
  std::shared_ptr<const NpuRuntimeStatusPublisher> publisher;
  bool started = false, armed = false, faulted = false;
  bool held = false, hold_sticky = false;
  std::string hold_reason;
  std::uint64_t trips = 0;
  std::optional<double> trip_w, trip_min_avail, trip_commit;
  std::string off_reason;
  std::optional<NpuRuntimeStatus::Scheme> scheme;
  std::optional<LastRequestSummary> last;
};
Shared& S() {
  static Shared shared;
  return shared;
}
std::atomic<bool> has_publisher{false};
// One publisher call at a time and in snapshot order: the snapshot is built and
// delivered under this mutex (never while holding the state mutex).
std::mutex publish_mutex;
}  // namespace

const char* NpuStateName(NpuState state) noexcept {
  switch (state) {
    case NpuState::kOn:
      return "on";
    case NpuState::kOff:
      return "off";
    case NpuState::kTripped:
      return "tripped";
    case NpuState::kFaulted:
      return "faulted";
    case NpuState::kUnavailable:
      return "unavailable";
  }
  return "unavailable";
}

json::Value ToJson(const NpuRuntimeStatus& status) {
  auto out = json::Value::object();
  out["state"] = NpuStateName(status.state);
  out["reason"] = status.reason ? json::Value(*status.reason) : json::Value();
  out["trips"] = status.trips;
  out["hal_faults"] = status.hal_faults;
  out["whea_events"] = status.whea_events;
  out["invalid_chunks"] = status.invalid_chunks;
  auto power = json::Value::object();
  power["state"] = status.chip_power.state;
  power["w_mean"] = status.chip_power.w_mean
                        ? json::Value(*status.chip_power.w_mean)
                        : json::Value();
  power["window_s"] = status.chip_power.window_s;
  out["chip_power"] = std::move(power);
  if (status.memory) {
    auto memory = json::Value::object();
    memory["min_available_mib"] = status.memory->min_available_mib;
    memory["max_commit_pct"] = status.memory->max_commit_pct;
    out["memory"] = std::move(memory);
  }
  if (status.scheme) {
    auto scheme = json::Value::object();
    scheme["guid"] = status.scheme->guid;
    scheme["name"] = status.scheme->name;
    out["scheme"] = std::move(scheme);
  }
  return out;
}

void SetNpuRuntimeStatusPublisher(NpuRuntimeStatusPublisher publisher) {
  auto& s = S();
  std::shared_ptr<const NpuRuntimeStatusPublisher> next;
  if (publisher)
    next =
        std::make_shared<const NpuRuntimeStatusPublisher>(std::move(publisher));
  std::lock_guard lock(s.mutex);
  s.publisher = std::move(next);
  has_publisher.store(static_cast<bool>(s.publisher),
                      std::memory_order_release);
}

NpuRuntimeStatus CurrentNpuRuntimeStatus() {
  auto& s = S();
  auto& faults = Faults();
  NpuRuntimeStatus out;
  {
    std::lock_guard lock(s.mutex);
    if (!s.started) {
      out.state = NpuState::kOff;
    } else if (s.faulted) {
      out.state = NpuState::kFaulted;
      out.reason = "fault";
    } else if (s.held) {
      // Held by the guard: the trip's own numbers, not the last request's.
      out.state = NpuState::kTripped;
      out.reason = s.hold_reason;
      NpuRuntimeStatus::Trip trip;
      trip.sticky = s.hold_sticky;
      trip.w_mean = s.trip_w;
      trip.min_available_mib = s.trip_min_avail;
      trip.max_commit_pct = s.trip_commit;
      out.trip = trip;
    } else if (s.armed) {
      out.state = NpuState::kOn;
    } else {
      out.state = NpuState::kOff;
      out.reason = s.off_reason;
    }
    out.scheme = s.scheme;
    out.trips = s.trips;
    if (s.last) {
      out.chip_power.w_mean = s.last->power_w_mean;
      out.chip_power.window_s = s.last->window_s;
      out.memory = NpuRuntimeStatus::Memory{s.last->min_available_mib,
                                            s.last->max_commit_pct};
    }
    if (out.state == NpuState::kTripped) {
      if (s.trip_w)
        out.chip_power.w_mean = s.trip_w;
      if (s.trip_min_avail && s.trip_commit)
        out.memory =
            NpuRuntimeStatus::Memory{*s.trip_min_avail, *s.trip_commit};
    }
  }
  out.hal_faults = faults.hal_faults.load(std::memory_order_relaxed);
  out.whea_events = faults.whea_events.load(std::memory_order_relaxed);
  out.invalid_chunks = faults.invalid_chunks.load(std::memory_order_relaxed);
  out.gpu_retries = faults.gpu_retries.load(std::memory_order_relaxed);
  switch (faults.chip_power_state.load(std::memory_order_relaxed)) {
    case 1:
      out.chip_power.state = "ok";
      break;
    case 3:
      out.chip_power.state = "unavailable";
      break;
    default:
      out.chip_power.state = "off";
      break;
  }
  return out;
}

void PublishNpuRuntimeStatus() noexcept {
  if (!has_publisher.load(std::memory_order_acquire))
    return;  // cheap: no publisher, nothing to build
  try {
    std::shared_ptr<const NpuRuntimeStatusPublisher> publisher;
    {
      auto& s = S();
      std::lock_guard lock(s.mutex);
      publisher = s.publisher;
    }
    if (!publisher)
      return;
    std::lock_guard order(publish_mutex);
    (*publisher)(CurrentNpuRuntimeStatus());  // outside the state mutex
  } catch (...) {
    // A status consumer must never break the NPU path.
  }
}

void NoteNpuStartup(bool armed, const std::string& reason) noexcept {
  try {
    {
      auto& s = S();
      std::lock_guard lock(s.mutex);
      s.started = true;
      s.armed = armed;
      s.faulted = false;
      s.off_reason = armed ? std::string() : reason;
    }
    Faults().armed.store(armed, std::memory_order_release);
    PublishNpuRuntimeStatus();
  } catch (...) {
  }
}
void NoteNpuHold(const NpuHoldInfo& info) noexcept {
  try {
    {
      auto& s = S();
      std::lock_guard lock(s.mutex);
      s.trips = info.trips;
      if (info.held) {
        s.held = true;
        s.hold_reason = info.reason;
        s.hold_sticky = info.sticky;
        s.trip_w = info.w_mean;
        s.trip_min_avail = info.min_available_mib;
        s.trip_commit = info.max_commit_pct;
      } else {  // re-armed: forget what tripped it (trips keeps counting)
        s.held = false;
        s.hold_sticky = false;
        s.hold_reason.clear();
        s.trip_w.reset();
        s.trip_min_avail.reset();
        s.trip_commit.reset();
      }
    }
    PublishNpuRuntimeStatus();
  } catch (...) {
  }
}
void NoteNpuFaulted() noexcept {
  try {
    // Only after a successful start: a failed start is "off", not "faulted".
    if (!Faults().armed.load(std::memory_order_acquire))
      return;
    {
      auto& s = S();
      std::lock_guard lock(s.mutex);
      if (s.faulted)
        return;
      s.faulted = true;
    }
    PublishNpuRuntimeStatus();
  } catch (...) {
  }
}
void NoteNpuScheme(const std::string& guid, const std::string& name) noexcept {
  try {
    auto& s = S();
    std::lock_guard lock(s.mutex);
    s.scheme = NpuRuntimeStatus::Scheme{guid, name};
  } catch (...) {
  }
}
void NoteLastRequest(const LastRequestSummary& summary) noexcept {
  try {
    auto& s = S();
    std::lock_guard lock(s.mutex);
    s.last = summary;
  } catch (...) {
  }
}
void ResetNpuRuntimeStatusForTest() {
  auto& s = S();
  {
    std::lock_guard lock(s.mutex);
    s.publisher.reset();
    s.started = s.armed = s.faulted = false;
    s.held = s.hold_sticky = false;
    s.hold_reason.clear();
    s.trips = 0;
    s.trip_w.reset();
    s.trip_min_avail.reset();
    s.trip_commit.reset();
    s.off_reason.clear();
    s.scheme.reset();
    s.last.reset();
  }
  has_publisher.store(false, std::memory_order_release);
  Faults().armed.store(false);
  Faults().invalid_chunks.store(0);
  Faults().gpu_retries.store(0);
}
}  // namespace gufo::npu
