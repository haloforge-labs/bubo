#ifndef GUFO_CORE_PLATFORM_MEMORY_GUARD_HPP_
#define GUFO_CORE_PLATFORM_MEMORY_GUARD_HPP_

// Memory-pressure guard (v0.1.1): the decisions, with no OS and no GPU calls.
//
// A machine that runs out of physical memory starts paging other programs and
// freezes. The guard keeps Bubo from being the cause, using only numbers the OS
// reports for this machine and Bubo's own byte costs, so a test can imitate a
// small machine with fake readings.
//
//  * Decide(): admission, once per request before prefill. A request is refused
//    only for MANDATORY new bytes (memory it cannot run without). Cache
//    snapshots are optional: below the keep-free floor they are skipped and the
//    request is still served (other programs can keep free RAM under the floor
//    for hours; refusing dialogue there would be a new failure).
//  * PressureMachine: while running. It turns readings into a level and asks
//    for one action at a time (drop snapshots, stop one request). The caller
//    (a thread with its own timer) performs the actions; nothing here frees or
//    cancels anything.
//
// The keep-free floor is the existing GUFO_CACHE_HOST_FLOOR_MB value (default
// 4096 MiB); 0 turns the whole guard off.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace gufo::platform {

struct MemoryReadings {
  bool ram_ok{false};
  std::uint64_t ram_available{0};
  std::uint64_t ram_total{0};
  std::uint64_t commit_available{0};
  /// This process's private RAM (only used to tell "never fits" from "not now").
  std::uint64_t process_ram_private{0};
  bool gpu_ok{false};
  /// WDDM budget minus usage for this process, per segment group.
  std::uint64_t gpu_local_free{0};
  std::uint64_t gpu_nonlocal_free{0};
  std::uint64_t gpu_budget_total{0};
  /// UMA: the LOCAL budget (110.9 GiB here) is the dedicated carve-out (95.8 GiB) PLUS a slice of system RAM, so "local free" overlaps
  /// the RAM reading. `gpu_carve_free` is the free part of the dedicated carve-out alone (min(local free, dedicated - local used)); only
  /// bytes beyond it come from RAM. Unknown (`gpu_carve_known` false): local free is used, which can count the shared slice twice.
  bool gpu_carve_known{false};
  std::uint64_t gpu_carve_free{0};
  /// The Windows low-memory resource notification is signalled.
  bool low_memory_signal{false};
};

class MemoryReadingSource {
public:
  virtual ~MemoryReadingSource() = default;
  /// Must not throw. A failed reading leaves its `_ok` flag false.
  virtual MemoryReadings Read() noexcept = 0;
};

struct MemoryNeed {
  /// Bytes the request cannot run without: lazy vision buffers, deferred
  /// scratch, a prompt-scoring or decision session. Device-private bytes fill
  /// the process GPU budget first; only the rest is charged to system RAM.
  std::uint64_t mandatory_device_bytes{0};
  std::uint64_t mandatory_host_bytes{0};
  /// Cache snapshots. Never a reason to refuse; used for the log line only.
  std::uint64_t optional_bytes{0};
};

enum class AdmitVerdict : std::uint8_t {
  kAdmit,
  kRefuseTransient,  // HTTP 503 insufficient_memory, retry later
  kRefusePermanent,  // HTTP 507 insufficient_memory_permanent
};

struct AdmitResult {
  AdmitVerdict verdict{AdmitVerdict::kAdmit};
  std::uint64_t needed_bytes{0};     // mandatory device + host
  std::uint64_t free_bytes{0};       // what is free for it (RAM above the floor, or GPU)
  std::uint64_t shortfall_bytes{0};  // how much snapshot trimming could still help
  std::string message;               // plain text with the numbers (empty on admit)
};

/// Pure admission rule. `floor_bytes == 0` or unusable readings admit.
[[nodiscard]] AdmitResult Decide(const MemoryReadings& readings,
                                 const MemoryNeed& need,
                                 std::uint64_t floor_bytes);

/// Load time (R2): where session state can live. On a UMA APU the WDDM local budget is the dedicated carve-out PLUS a slice of system RAM, so
/// "GPU free" as reported overlaps the RAM reading. Session state fits the free dedicated carve-out first, then the RAM that is available,
/// and never more than the process's reported free. The load REFUSES only when the need exceeds this physical capacity (the floor is
/// ignored: a configuration that runs, thin on RAM, must still start). Eating into the keep-free floor, or using RAM at all, starts with a
/// warning (PlanStateSpill). Unknown (no usable carve-out reading, or the guard is off): the reported free is returned unchanged.
struct StateFit {
  bool known{false};
  std::uint64_t capacity_bytes{0};         // physical: min(reported free, free carve-out + available RAM)
  std::uint64_t carve_free_bytes{0};
  std::uint64_t ram_available_bytes{0};
  std::uint64_t ram_above_floor_bytes{0};
  std::uint64_t floor_bytes{0};
};
[[nodiscard]] StateFit PlanStateFit(const MemoryReadings& readings, std::uint64_t floor_bytes, std::uint64_t reported_free_bytes);
/// PlanStateFit through the process guard, for a runner's resource claim (reported_free_bytes when there is no guard).
[[nodiscard]] std::uint64_t StateCapacityBytes(std::uint64_t reported_free_bytes);
/// What a need that fits physically costs: `spill_bytes` do not fit the free carve-out and use system RAM; `floor_eaten_bytes` of that spill
/// is below the keep-free floor. A start with either gets the load_spill warning.
struct StateSpill {
  bool known{false};
  std::uint64_t spill_bytes{0};
  std::uint64_t floor_eaten_bytes{0};
  std::uint64_t floor_bytes{0};
  std::uint64_t ram_available_bytes{0};
};
[[nodiscard]] StateSpill PlanStateSpill(const StateFit& fit, std::uint64_t need_bytes);
/// PlanStateSpill through the process guard (unknown when there is no guard).
[[nodiscard]] StateSpill StateSpillFor(std::uint64_t need_bytes);

enum class GuardLevel : std::uint8_t {
  kOk,          // nothing to do
  kSoft,        // RAM available is under the floor: reported only (the cache floor already decides per snapshot, by host bytes)
  kLow,         // Windows low-memory event or GPU budget used up: drop snapshots
  kCancelling,  // still low with nothing left to drop: stop running requests
};

[[nodiscard]] const char* GuardLevelName(GuardLevel level) noexcept;

struct PressureAction {
  enum class Kind : std::uint8_t { kNone, kTrimSnapshots, kCancelOne };
  Kind kind{Kind::kNone};
  std::uint64_t trim_bytes{0};
};

struct PressureStep {
  GuardLevel level{GuardLevel::kOk};
  bool level_changed{false};
  PressureAction action;
  /// Why the level is not kOk: ram_below_floor | low_memory_signal | gpu_budget.
  const char* reason{""};
};

/// Level and action logic. Call Step() on every tick (about 500 ms) and on
/// every wake-up. Single-threaded; the caller owns the synchronisation.
class PressureMachine {
public:
  using Clock = std::chrono::steady_clock;
  struct Timing {
    std::chrono::milliseconds dwell{500};          // between two actions
    std::chrono::milliseconds cancel_after{2000};  // low with nothing to drop
    std::chrono::milliseconds recover_hold{5000};  // all clear before stepping down
  };
  PressureMachine() = default;
  explicit PressureMachine(Timing timing) : timing_(timing) {}

  /// `retained_snapshot_bytes` is what could still be dropped.
  PressureStep Step(const MemoryReadings& readings, std::uint64_t floor_bytes,
                    std::uint64_t retained_snapshot_bytes, Clock::time_point now);

  [[nodiscard]] GuardLevel level() const noexcept { return level_; }
  [[nodiscard]] std::uint64_t level_changes() const noexcept { return changes_; }

private:
  Timing timing_{};
  GuardLevel level_{GuardLevel::kOk};
  std::uint64_t changes_{0};
  unsigned soft_reads_{0}, low_reads_{0};
  bool have_action_time_{false}, have_low_since_{false}, have_clear_since_{false}, have_nolow_since_{false};
  Clock::time_point last_action_{}, low_since_{}, clear_since_{}, nolow_since_{};
};

/// The process-wide guard: the readings source, the floor and the level that
/// the scheduler reads with one relaxed atomic load.
class MemoryGuard {
public:
  struct Counters {
    std::uint64_t refused_transient{0}, refused_permanent{0}, trimmed_bytes{0}, cancelled{0};
  };

  MemoryGuard(MemoryReadingSource* source, std::uint64_t floor_bytes)
      : source_(source), floor_bytes_(floor_bytes) {}

  [[nodiscard]] bool enabled() const noexcept { return floor_bytes_ != 0 && source_ != nullptr; }
  [[nodiscard]] std::uint64_t floor_bytes() const noexcept { return floor_bytes_; }
  [[nodiscard]] GuardLevel level() const noexcept { return level_.load(std::memory_order_relaxed); }
  void SetLevel(GuardLevel level) noexcept { level_.store(level, std::memory_order_relaxed); }
  /// New cache snapshots are allowed. They stop at kLow (the Windows low-memory event or the GPU budget used up); below the floor alone the
  /// cache's own host-bytes floor decides, so device-resident checkpoints stay cached on a machine that lives under the floor. One atomic load.
  [[nodiscard]] bool SnapshotsAllowed() const noexcept { return level() < GuardLevel::kLow; }

  [[nodiscard]] MemoryReadings Read() const noexcept;
  /// Test-only replacement of the OS readings (the --debug-hooks route).
  void SetOverride(const MemoryReadings& readings);
  void ClearOverride();
  [[nodiscard]] bool overridden() const;

  /// Reads once and decides. If a transient shortfall remains and `trim` is
  /// set, calls trim(shortfall) (drop snapshots), reads again and decides again.
  AdmitResult Admit(const MemoryNeed& need,
                    const std::function<std::uint64_t(std::uint64_t)>& trim = {});

  /// Test-only (the --debug-hooks route): every request is charged this many extra mandatory device bytes, so a gate can force an
  /// over-budget admission on any model. Zero (the default) adds nothing.
  void SetDebugNeed(std::uint64_t bytes) noexcept { debug_need_.store(bytes, std::memory_order_relaxed); }
  [[nodiscard]] std::uint64_t debug_need() const noexcept { return debug_need_.load(std::memory_order_relaxed); }

  void NoteTrimmed(std::uint64_t bytes) noexcept { trimmed_.fetch_add(bytes, std::memory_order_relaxed); }
  void NoteCancelled() noexcept { cancelled_.fetch_add(1, std::memory_order_relaxed); }
  [[nodiscard]] Counters counters() const noexcept;

private:
  MemoryReadingSource* source_;
  std::uint64_t floor_bytes_;
  std::atomic<GuardLevel> level_{GuardLevel::kOk};
  mutable std::atomic<std::uint64_t> refused_transient_{0}, refused_permanent_{0};
  std::atomic<std::uint64_t> trimmed_{0}, cancelled_{0};
  std::atomic<bool> override_active_{false};
  std::atomic<std::uint64_t> debug_need_{0};
  mutable std::mutex override_mutex_;
  MemoryReadings override_readings_{};
};

/// One guard per process (null until the server configures it).
[[nodiscard]] MemoryGuard* ProcessMemoryGuard() noexcept;
void SetProcessMemoryGuard(MemoryGuard* guard) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_MEMORY_GUARD_HPP_
