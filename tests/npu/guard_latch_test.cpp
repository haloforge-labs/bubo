// SPDX-License-Identifier: MIT
// CPU only (fake XRT, fake HIP): the guard hold is latched once per chunk in
// ServerPrefill::BeginChunk and never fails a request.
#include <cassert>
#include <iostream>
#include <vector>

#include "fake_hip.hpp"
#include "fake_xrt.hpp"
#include "src/npu/guard_config.hpp"
#include "src/npu/linear_z_hip.hpp"
#include "src/npu/runtime_status.hpp"
#include "src/npu/server_prefill.hpp"

using namespace gufo::npu;
using namespace std::chrono_literals;

namespace {
constexpr std::int64_t kSec = 1'000'000'000;
struct Feed {
  std::int64_t now = 5000 * kSec;
  double commit_min = 40;
  bool warm_ok = true;
  int warm_calls = 0;
  std::vector<bool> paused;
  GuardHooks Hooks() {
    GuardHooks h;
    h.now_ns = [this] { return now; };
    h.memory = [this](std::int64_t, std::int64_t, double) {
      MemoryWindow w;
      w.ok = true;
      w.samples = 30;
      w.span_s = 29;
      w.max_gap_s = 1;
      w.min_commit_pct = w.max_commit_pct = commit_min;
      w.min_avail_mib = 8000;
      return w;
    };
    h.power = [](std::int64_t, std::int64_t) { return PowerWindow{}; };
    h.process_failed = [] { return ProcessFailed(); };
    h.log = [](bool, const std::string&) {};
    h.publish = [](const GuardTripInfo& i) {
      NpuHoldInfo hold;
      hold.held = i.held;
      hold.reason = i.reason;
      hold.trips = i.trips;
      hold.sticky = i.sticky;
      NoteNpuHold(hold);
    };
    h.set_keepalive_paused = [this](bool p) { paused.push_back(p); };
    h.warm_up = [this](std::string* error) {
      ++warm_calls;
      if (!warm_ok)
        *error = "fake beat failed";
      return warm_ok;
    };
    return h;
  }
};
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  fake::Reset();
  fake::control.expected_a = 0;
  std::vector<NpuRuntimeStatus> statuses;
  SetNpuRuntimeStatusPublisher(
      [&](const NpuRuntimeStatus& s) { statuses.push_back(s); });
  SessionOptions options;
  options.authorization = gufo::core::ExperimentalAuthorization::FakePrefill();
  options.artifacts = argv[1];
  options.timing.keepalive_interval = 0ms;
  auto service = std::make_shared<ServerPrefill>(options);
  assert(service->Start());
  Feed feed;
  service->ReplaceGuardForTest(std::make_unique<NpuGuard>(
      GuardMode::kOn, GuardOverrides{}, feed.Hooks()));
  fake_hip::Stream stream;
  std::vector<std::uint16_t> input(std::size_t{4096} * kK);
  std::vector<float> output(std::size_t{4096} * 16384);
  const auto transfer = [&] {
    return HipLinearZTransfer(service, input.data(), output.data(), 4096,
                              stream.get());
  };

  // A clean chunk: lanes open.
  service->BeginChunk(4096);
  assert(service->ready() && service->lane_ready());
  {
    auto t = transfer();
    (void)t;
  }
  assert(service->ChunkUsed());

  // The hold starts MID-chunk (a power trip from this chunk's own end): this
  // chunk keeps its latched lanes, so a new transfer still constructs.
  ChunkStart start;
  ChunkEnd end;
  end.power = PowerWindow{true, false, 90, 90, 3, 3};
  for (int i = 0; i < 3; ++i)
    service->guard()->OnChunkEnd(start, end);
  assert(service->guard()->held());
  assert(service->lane_ready());
  {
    auto t = transfer();
    (void)t;
  }  // does not throw on a latched open lane
  assert(statuses.back().state == NpuState::kTripped &&
         statuses.back().reason == "power" && statuses.back().trips == 1);
  assert(feed.paused == std::vector<bool>{true});

  // The next chunk latches the hold: lanes closed, planner view unchanged,
  // nothing marked used, no telemetry, nothing rejected or failed.
  service->BeginChunk(4096);
  assert(service->ready());  // PrefillPreferredChunk & co use ready()
  assert(!service->lane_ready());
  assert(!service->ChunkUsed());
  bool threw = false;
  try {
    auto t = transfer();
    (void)t;
  } catch (const std::invalid_argument&) {
    threw = true;  // the executor never gets here: it checks lane_ready() first
  }
  assert(threw);
  std::string error;
  assert(CheckHipPrefillChunk(*service, &error) && error.empty());
  assert(!ProcessFailed() && service->ready());
  assert(service->session().Healthy());

  // Still held a few chunks later (before the hold time).
  feed.now += 100 * kSec;
  service->BeginChunk(4096);
  assert(!service->lane_ready() && service->ready());

  // After the hold time and with clear conditions: warm-up beat, re-arm, and
  // the first chunk after it is marked.
  feed.now += 600 * kSec;
  service->BeginChunk(4096);
  assert(feed.warm_calls == 1);
  assert(service->lane_ready());
  assert(!service->chunk().first_after_rearm);  // marked at the first NPU use
  assert((feed.paused == std::vector<bool>{true, false}));
  assert(statuses.back().state == NpuState::kOn && statuses.back().trips == 1);
  // This first pass after the re-arm stays GPU-only (no transfer): the marker
  // must survive it.
  service->BeginChunk(4096);
  assert(service->lane_ready() && !service->chunk().first_after_rearm);
  {
    auto t = transfer();
    (void)t;
  }
  assert(service->ChunkUsed() && service->chunk().first_after_rearm);
  service->BeginChunk(4096);
  {
    auto t = transfer();
    (void)t;
  }
  assert(!service->chunk().first_after_rearm);  // only the first NPU chunk

  // A failed warm-up beat keeps the hold (the real beat has then run the
  // normal fault path).
  Feed second;
  service->ReplaceGuardForTest(std::make_unique<NpuGuard>(
      GuardMode::kOn, GuardOverrides{}, second.Hooks()));
  service->BeginChunk(4096);
  for (int i = 0; i < 3; ++i)
    service->guard()->OnChunkEnd(start, end);
  service->BeginChunk(4096);
  assert(!service->lane_ready());
  second.warm_ok = false;
  second.now += 700 * kSec;
  service->BeginChunk(4096);
  assert(second.warm_calls == 1 && !service->lane_ready());

  // Off mode: nothing is ever held.
  Feed off_feed;
  service->ReplaceGuardForTest(std::make_unique<NpuGuard>(
      GuardMode::kOff, GuardOverrides{}, off_feed.Hooks()));
  service->BeginChunk(4096);
  assert(service->lane_ready());
  // No guard at all behaves like off.
  service->ReplaceGuardForTest(nullptr);
  service->BeginChunk(4096);
  assert(service->lane_ready());
  // Cost of ServerPrefill::BeginChunk with the guard on a FAKE feed (latch and
  // memory-window logic, not the real sampler) plus the chunk telemetry.
  Feed cost_feed;
  service->ReplaceGuardForTest(std::make_unique<NpuGuard>(
      GuardMode::kOn, GuardOverrides{}, cost_feed.Hooks()));
  service->BeginChunk(4096);
  cost_feed.now += 61 * kSec;
  constexpr int kBegins = 20000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kBegins; ++i)
    service->BeginChunk(4096);
  const auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  kBegins;
  std::cout << "COST: ServerPrefill::BeginChunk with guard (latch + memory "
               "evaluation + chunk telemetry) "
            << us << " us/chunk\n";
  SetNpuRuntimeStatusPublisher({});
  std::cout
      << "PASS guard latch: mid-chunk hold waits for the next chunk; held "
         "chunks leave ChunkUsed false; ready() unchanged; no reject; "
         "re-arm marks the first chunk\n";
}
