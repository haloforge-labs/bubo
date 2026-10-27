// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_GUARD_HPP_
#define GUFO_NPU_GUARD_HPP_
// NPU prefill guard (G1). An envelope guard, not a demonstrated fix. It holds
// the NPU lanes for a timed period when the NPU chunks run above the tested
// power or the machine is under severe memory pressure. A hold NEVER fails a
// request and never touches the process-failed flag (FailProcess/Reject): held
// chunks simply take the existing GPU branch. A fault always wins.
//
// V1: the whole guard runs on the chunk thread (BeginChunk and the chunk-end
// check). There is no sampler observer; the memory rule reads the memory ring
// that the 1 s sampler already fills. Pure logic: every input is injected, so
// the tests drive it with a fake clock.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chip_power.hpp"
#include "fault_telemetry.hpp"

namespace gufo::npu {
enum class GuardMode { kOn, kOff, kLog };
const char* GuardModeName(GuardMode mode) noexcept;
// "on" | "off" | "log"; false for anything else.
bool ParseGuardMode(const std::string& text, GuardMode* mode) noexcept;

// GUFO_NPU_GUARD_TEST (internal): key=value list "power_w=30,hold_s=60".
struct GuardOverrides {
  std::optional<double> power_w, hold_s;
};
// False (and *error set) when the text is malformed. Empty text = no override.
bool ParseGuardOverrides(const std::string& text, GuardOverrides* out,
                         std::string* error);

struct GuardTripInfo {
  bool held = false;
  std::string reason;  // "power" | "memory"
  std::uint64_t trips = 0;
  bool sticky = false;
  std::optional<double> power_w_mean;       // the tripping chunk mean
  std::optional<double> min_available_mib;  // memory window at the trip
  std::optional<double> max_commit_pct;
};

struct GuardHooks {
  std::function<std::int64_t()> now_ns;  // steady_clock ns
  std::function<MemoryWindow(std::int64_t t0, std::int64_t t1, double clamp)>
      memory;  // may be empty
  std::function<PowerWindow(std::int64_t t0, std::int64_t t1)>
      power;                             // may be empty
  std::function<bool()> process_failed;  // faults win
  std::function<void(bool warn, const std::string&)> log;
  std::function<void(const GuardTripInfo&)> publish;  // enforce mode only
  std::function<void(bool paused)> set_keepalive_paused;
  // Synchronous heartbeat before re-arm; false (+ *error) takes the fault path.
  std::function<bool(std::string* error)> warm_up;
};

class NpuGuard {
public:
  NpuGuard(GuardMode mode, GuardOverrides overrides, GuardHooks hooks);
  GuardMode mode() const noexcept { return mode_; }
  // Chunk thread, once per prefill pass, before the lanes run. Returns whether
  // this chunk is held (the caller latches it). Handles the hold timeout and
  // re-arm, and evaluates the memory rule.
  bool OnBeginChunk();
  // Chunk thread, after a chunk that used the NPU and passed its checks.
  void OnChunkEnd(const ChunkStart& start, const ChunkEnd& end);
  // True in on mode while a hold is active. Log mode never holds.
  bool held() const noexcept { return tripped_ && enforce_; }
  // A hold that will still be on when the next chunk starts: sticky, or its
  // time outlasts now (the guard's own clock) by HOLD_SKIP_BACKUP_MARGIN_S.
  // OnBeginChunk re-arms a hold whose time ran out, and it runs a little after
  // this check, so a hold that ends within the margin does not count (that
  // chunk may run on the NPU). Chunk thread.
  bool StaysHeld() const;
  // The internal state machine (log mode runs it without enforcing).
  bool tripped() const noexcept { return tripped_; }
  bool sticky() const noexcept { return sticky_; }
  // True once after a re-arm: marks the first NPU chunk after it (latency and
  // fault evidence for the cold-after-idle path). Chunk thread.
  bool ConsumeRearmFlag() noexcept { return std::exchange(rearmed_, false); }
  std::uint64_t trips() const noexcept { return trips_; }
  double power_threshold_w() const noexcept { return power_w_; }
  double hold_seconds() const noexcept { return hold_s_; }

private:
  void Trip(const char* reason, const std::string& detail,
            std::optional<double> power_w, const MemoryWindow* window);
  void TryRearm(std::int64_t now);
  bool MemoryWindowValid(const MemoryWindow& w) const;
  bool PagesCondition(const MemoryWindow& w) const;
  GuardMode mode_;
  bool enforce_;
  double power_w_, hold_s_;
  GuardHooks hooks_;
  bool tripped_ = false, sticky_ = false, rearmed_ = false;
  std::string reason_;
  std::uint64_t trips_ = 0;
  std::int64_t hold_until_ns_ = 0, first_chunk_ns_ = 0,
               last_still_held_log_ns_ = 0;
  std::int64_t tripped_at_ns_ = 0;
  unsigned power_run_ = 0;
  std::vector<double> run_means_;
};
}  // namespace gufo::npu
#endif
