// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_GUARD_CONFIG_HPP_
#define GUFO_NPU_GUARD_CONFIG_HPP_





namespace gufo::npu::guard {
// Power rule: the mean RAPL package power of POWER_CHUNKS consecutive NPU
// chunks above POWER_W trips it.
inline constexpr double POWER_W = 82.0;
inline constexpr unsigned POWER_CHUNKS = 3;
// Re-arm only when the last CLEAR_WINDOW_S seconds average below this.
inline constexpr double POWER_CLEAR_W = 76.0;
// A chunk counts only if its window holds at least this many 1 s intervals.
inline constexpr unsigned POWER_MIN_INTERVALS = 2;
// Idle gap (ms/1000) that starts a new active interval; kept for the log.
inline constexpr double POWER_IDLE_GAP_S = 3.0;
// An idle gap longer than this forgets the consecutive-chunk run.
inline constexpr double POWER_RESET_IDLE_S = 60.0;

// Memory rule (a suspicion about the FREEZE, not the fault): commit charge at
// or above MEM_COMMIT_PCT for the whole MEM_WINDOW_S window, or heavy paging
// with little free memory. 98 % of a system-managed commit limit is close to
// "allocations are about to fail".
inline constexpr double MEM_COMMIT_PCT = 98.0;
inline constexpr double MEM_COMMIT_CLEAR_PCT = 95.0;
inline constexpr double MEM_WINDOW_S = 30.0;
// The window must be covered by real samples: time window, not a count.
inline constexpr double MEM_MIN_SPAN_S = 25.0;
inline constexpr unsigned MEM_MIN_SAMPLES = 15;
inline constexpr double MEM_MAX_GAP_S = 10.0;
inline constexpr double MEM_PAGES_PER_S = 100000.0;
// Each Pages/sec sample is clamped to this factor times the threshold, so one
// spike cannot move the mean over the line.
inline constexpr double MEM_PAGES_CLAMP_FACTOR = 3.0;
inline constexpr double MEM_AVAIL_MIB = 1024.0;
// No memory evaluation for this long after the first prefill chunk (the
// sampler starts before the model upload, when paging peaks).
inline constexpr double MEM_STARTUP_GRACE_S = 60.0;

// Hold.
inline constexpr double HOLD_SECONDS = 600.0;
inline constexpr unsigned STICKY_AFTER_TRIPS = 3;
// Session::Feed skips the retry backup of a chunk only when the hold outlasts
// the chunk start by this margin: the hold can end between Feed's check and
// OnBeginChunk inside Forward, and an extra backup is the safe direction.
inline constexpr double HOLD_SKIP_BACKUP_MARGIN_S = 2.0;
inline constexpr double CLEAR_WINDOW_S = 30.0;
// Heartbeats are NPU dispatches that overlap GPU work: stopped while held.
inline constexpr bool HEARTBEAT_WHILE_HELD = false;
}  // namespace gufo::npu::guard
#endif
