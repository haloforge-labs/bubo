// SPDX-License-Identifier: MIT
#pragma once
// Cheap per-request NPU prefill counters: relaxed atomics, no I/O. One summary
// line is logged per request by the Qwen text runner (OnRequestFinished).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "fault_telemetry.hpp"
#include "runtime_status.hpp"

namespace gufo::npu {
struct RequestStats {
  std::atomic<std::uint64_t> finish_calls{0};
  std::atomic<std::uint64_t> finish_not_ready{
      0};  // callbacks whose result was not ready
  std::atomic<std::uint64_t> finish_wait_ns{
      0};                                  // stream callback NPU wait time
  std::atomic<std::uint64_t> fence_ns{0};  // host time enqueueing GPU decode
  std::atomic<std::uint64_t> hostfunc_lat_ns{
      0};                                  // Begin enqueue to callback start
  std::atomic<std::uint64_t> failures{0};  // failures reject this chunk
  std::atomic<std::uint64_t> linear_begins{0}, projection_begins{0};
  std::atomic<std::uint64_t> qgate_begins{0}, shared_begins{0};
  std::atomic<std::uint64_t> chunks{
      0};  // distinct offloaded chunks (layer order restarts)
  std::atomic<int> last_layer{-1};
  // Stream-side NPU stall per lane: 0 linear-z, 1 attention q/gate, 2 shared.
  std::atomic<std::uint64_t> lane_wait_ns[3]{{0}, {0}, {0}};
  std::atomic<std::uint64_t> lane_blocked[3]{{0}, {0}, {0}};
};
// Lane of the Finish being enqueued on this thread (read when the stream wait is queued).
inline int& WaitLane() {
  static thread_local int lane = 0;
  return lane;
}
inline RequestStats& Stats() {
  static RequestStats stats;
  return stats;
}
inline std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
// A new chunk starts when the layer index stops increasing (layers are
// dispatched in order inside a chunk).
inline void NoteBegin(int layer, bool linear) {
  auto& s = Stats();
  (linear ? s.linear_begins : s.projection_begins)
      .fetch_add(1, std::memory_order_relaxed);
  const int previous = s.last_layer.exchange(layer, std::memory_order_relaxed);
  if (previous < 0 || layer < previous)
    s.chunks.fetch_add(1, std::memory_order_relaxed);
}
// Returns the delta line since the previous call (empty when the NPU did
// nothing).
inline std::string TakeRequestLine() {
  auto& s = Stats();
  const auto calls = s.finish_calls.exchange(0);
  const auto blocked = s.finish_not_ready.exchange(0);
  const auto wait = s.finish_wait_ns.exchange(0);
  const auto fence = s.fence_ns.exchange(0);
  const auto latency = s.hostfunc_lat_ns.exchange(0);
  const auto failed = s.failures.exchange(0);
  const auto lin = s.linear_begins.exchange(0);
  const auto proj = s.projection_begins.exchange(0);
  const auto qgate = s.qgate_begins.exchange(0);
  const auto shared = s.shared_begins.exchange(0);
  const auto chunks = s.chunks.exchange(0);
  std::uint64_t lane_wait[3], lane_blocked[3];
  for (int i = 0; i < 3; ++i) {
    lane_wait[i] = s.lane_wait_ns[i].exchange(0);
    lane_blocked[i] = s.lane_blocked[i].exchange(0);
  }
  s.last_layer.store(-1);
  if (lin == 0 && proj == 0) {
    PublishNpuRuntimeStatus();  // request end; cheap without a publisher
    return {};
  }
  char text[640];
  std::snprintf(
      text, sizeof(text),
      "request: chunks=%llu linear_begins=%llu projection_begins=%llu "
      "finish_calls=%llu "
      "qgate_begins=%llu shared_begins=%llu finish_blocked=%llu "
      "finish_wait_ms=%.1f fence_ms=%.1f hostfunc_lat_ms=%.1f failures=%llu "
      "stall_ms[linear=%.1f qgate=%.1f shared=%.1f] blocked[linear=%llu qgate=%llu shared=%llu]",
      static_cast<unsigned long long>(chunks),
      static_cast<unsigned long long>(lin),
      static_cast<unsigned long long>(proj),
      static_cast<unsigned long long>(calls),
      static_cast<unsigned long long>(qgate),
      static_cast<unsigned long long>(shared),
      static_cast<unsigned long long>(blocked), static_cast<double>(wait) / 1e6,
      static_cast<double>(fence) / 1e6, static_cast<double>(latency) / 1e6,
      static_cast<unsigned long long>(failed),
      static_cast<double>(lane_wait[0]) / 1e6,
      static_cast<double>(lane_wait[1]) / 1e6,
      static_cast<double>(lane_wait[2]) / 1e6,
      static_cast<unsigned long long>(lane_blocked[0]),
      static_cast<unsigned long long>(lane_blocked[1]),
      static_cast<unsigned long long>(lane_blocked[2]));
  // Fields appended at the end so existing parsers keep working.
  auto line = std::string(text) + TakeRequestTelemetrySuffix();
  PublishNpuRuntimeStatus();  // request end; cheap without a publisher
  return line;
}
}  // namespace gufo::npu
