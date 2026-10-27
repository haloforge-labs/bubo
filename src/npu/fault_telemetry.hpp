// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_FAULT_TELEMETRY_HPP_
#define GUFO_NPU_FAULT_TELEMETRY_HPP_
// Context printed with an NPU chunk failure and with the per-request NPU line:
// ms wall-clock stamps, idle gap, host memory, per-chunk counters, HAL events.
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

#include "chip_power.hpp"
#include "fault_state.hpp"
#include "invalid_bits.hpp"

namespace gufo::npu {
// "none", or "|"-joined A-nonfinite, A-saturated, C-exp255, C-exp254-mant128;
// unknown bits print as hex.
std::string DescribeInvalidBits(unsigned bits);

struct WallStamp {
  std::string utc;    // 2026-10-13T22:01:41.123Z
  std::string local;  // 2026-10-13 18:01:41.123 (same clock as the log)
};
WallStamp FormatWallStamp(std::chrono::system_clock::time_point time);
// Test hook: replace the wall clock (empty function restores the real one).
void SetWallClockForTest(
    std::function<std::chrono::system_clock::time_point()>);

struct MemorySample {
  bool ok = false;
  std::uint64_t available_mib = 0;  // physical RAM available
  std::uint64_t commit_mib = 0;  // commit charge (total page file - available)
  std::uint64_t commit_limit_mib = 0;  // commit limit (ullTotalPageFile)
};
class IMemorySource {
public:
  virtual ~IMemorySource() = default;
  virtual MemorySample Sample() = 0;
};
// Default: GlobalMemoryStatusEx (ullAvailPhys,
// ullTotalPageFile-ullAvailPageFile). Returns the previous override; nullptr
// restores the default.
IMemorySource* SetMemorySourceForTest(IMemorySource* source);
// precise_commit: system-wide commit charge/limit from GetPerformanceInfo
// (about 35 us per call, so only the 1 s sampler asks for it). The chunk path
// uses GlobalMemoryStatusEx (about 0.6 us).
MemorySample SampleMemory(bool precise_commit = false);

struct CounterSnapshot {
  std::uint64_t linear_begins = 0, projection_begins = 0, finish_calls = 0,
                failures = 0, hostfunc_lat_ns = 0;
  static CounterSnapshot Take();
};
struct ChunkStart {
  std::chrono::system_clock::time_point wall;
  WallStamp stamp;
  std::int64_t steady_ns = 0;
  std::int64_t idle_gap_ms = -1;
  MemorySample memory;
  CounterSnapshot counters;
  std::uint64_t hal_seq = 0;
  unsigned tokens = 0;
  double cpu_ms = 0;  // process CPU time at chunk start
  MemTraceSample trace;  // G0b: process memory + sampled buffer pages
  bool first_after_rearm = false;  // first NPU chunk after a guard re-arm
};
struct ChunkEnd {
  WallStamp stamp;
  double duration_ms = 0;
  MemorySample memory;
  CounterSnapshot counters;   // delta since chunk start
  std::uint64_t index = 0;    // NPU chunk number inside the request (1-based)
  std::uint64_t hal_new = 0;  // events stored while the chunk ran
  PowerWindow power;          // chip power over the chunk (logging only)
  double cpu_ms = 0;          // process CPU time used during the chunk
  // Extremes over the chunk: chunk start/end samples plus the 1 s memory points
  // taken while it ran (logging only).
  std::uint64_t min_avail_mib = 0;
  double max_commit_pct = 0;
  double max_pages_per_s = -1;  // -1: Pages/sec not available
  MemTraceSample trace;         // G0b, at chunk end
  MemoryWindow gpu_window;      // G0b: GPU memory counters over the chunk
};
ChunkStart BeginChunkTelemetry(unsigned tokens);
ChunkEnd EndChunkTelemetry(const ChunkStart& start);
// One line of "key=value" fields; no newline.
std::string FormatChunkFailure(const ChunkStart& start, const ChunkEnd& end,
                               unsigned cause_bits);
// Per-request accumulation behind the "[npu] request:" line.
void NoteRequestChunk(const ChunkStart& start, const ChunkEnd& end);
// L-REL-2: device-event time of the retry backup copy for the chunk that just
// ended (call after NoteRequestChunk, only for a chunk that used the NPU).
void NoteChunkBackup(double ms);
// Monotonic count of NoteRequestChunk calls (never reset): lets a caller tell
// whether the chunk it just ran was noted.
std::uint64_t RequestChunksNoted() noexcept;
// Fields appended to the request line; resets the accumulation.
std::string TakeRequestTelemetrySuffix();
}  // namespace gufo::npu
#endif
