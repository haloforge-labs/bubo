// SPDX-License-Identifier: MIT
// CPU only: the NPU prefill guard (G1) state machine with an injected clock and
// fake feeds. No XRT, no PDH, no NPU/GPU.
#include "src/npu/guard.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "src/npu/guard_config.hpp"

using namespace gufo::npu;

namespace {
constexpr std::int64_t kSec = 1'000'000'000;
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

struct Harness {
  std::int64_t now = 1000 * kSec;
  MemoryWindow memory;  // what the fake ring returns
  PowerWindow power;    // clear-window answer
  bool failed = false, warm_ok = true;
  int memory_calls = 0, power_calls = 0, warm_calls = 0;
  std::vector<std::pair<bool, std::string>> log;
  std::vector<GuardTripInfo> published;
  std::vector<bool> paused;
  GuardHooks Hooks() {
    GuardHooks h;
    h.now_ns = [this] { return now; };
    h.memory = [this](std::int64_t, std::int64_t, double) {
      ++memory_calls;
      return memory;
    };
    h.power = [this](std::int64_t, std::int64_t) {
      ++power_calls;
      return power;
    };
    h.process_failed = [this] { return failed; };
    h.log = [this](bool warn, const std::string& t) {
      log.emplace_back(warn, t);
    };
    h.publish = [this](const GuardTripInfo& i) { published.push_back(i); };
    h.set_keepalive_paused = [this](bool p) { paused.push_back(p); };
    h.warm_up = [this](std::string* error) {
      ++warm_calls;
      now += 12'000'000;  // the beat takes 12 ms
      if (!warm_ok)
        *error = "XRT heartbeat did not complete";
      return warm_ok;
    };
    return h;
  }
  bool Logged(const std::string& part) const {
    for (const auto& [warn, t] : log)
      if (Has(t, part))
        return true;
    return false;
  }
};
MemoryWindow Mem(double commit_min, unsigned samples = 30, double span = 29,
                 double gap = 1.0, double pages = -1, double avail = 8000) {
  MemoryWindow w;
  w.ok = samples > 0;
  w.samples = samples;
  w.span_s = span;
  w.max_gap_s = gap;
  w.min_commit_pct = commit_min;
  w.max_commit_pct = commit_min;
  w.min_avail_mib = static_cast<std::uint64_t>(avail);
  if (pages >= 0) {
    w.pages_samples = samples;
    w.pages_mean = pages;
  }
  return w;
}
PowerWindow Pw(double mean, unsigned intervals = 3, bool approx = false) {
  PowerWindow p;
  p.ok = true;
  p.approx = approx;
  p.mean_w = mean;
  p.max_w = mean;
  p.span_s = intervals;
  p.intervals = intervals;
  return p;
}
// One NPU chunk: begin (returns held), then the end with a power window.
bool Chunk(NpuGuard& g, Harness& h, PowerWindow power,
           std::int64_t idle_gap_ms = -1) {
  const bool held = g.OnBeginChunk();
  if (held)
    return true;
  ChunkStart start;
  start.idle_gap_ms = idle_gap_ms;
  ChunkEnd end;
  end.power = power;
  h.now += 4 * kSec;
  g.OnChunkEnd(start, end);
  return false;
}
// Quiet memory so the memory rule stays out of power tests.
void Quiet(Harness& h) {
  h.memory = Mem(40);
}
// Trip the power rule with three hot chunks.
void HotRun(NpuGuard& g, Harness& h, double mean = 84.0) {
  for (int i = 0; i < 3; ++i)
    Chunk(g, h, Pw(mean + i * 0.5));
}

void PowerRule() {
  {  // exactly POWER_CHUNKS - 1 above: no trip; the Nth trips
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    Chunk(g, h, Pw(84));
    Chunk(g, h, Pw(83.5));
    assert(!g.tripped() && !g.held() && g.trips() == 0);
    Chunk(g, h, Pw(85));
    assert(g.tripped() && g.held() && g.trips() == 1);
    // State, publish, keepalive pause, log text.
    assert(h.published.size() == 1 && h.published[0].held &&
           h.published[0].reason == "power" && h.published[0].trips == 1 &&
           h.published[0].power_w_mean && *h.published[0].power_w_mean == 85);
    assert(h.paused == std::vector<bool>{true});
    assert(h.Logged(
        "npu guard: NPU prefill held (power): chunk_means_w=[84.0,83.5,85.0] > "
        "82.0 W x3; hold 600 s; trips=1/3; new chunks use the GPU"));
    for (const auto& [warn, t] : h.log)
      assert(!Has(t, "NPU disabled for process") && warn);
    // The next chunk is held and not counted.
    assert(g.OnBeginChunk());
    // L-REL-2: Session::Feed skips the retry backup only for a hold that
    // outlasts the check by the margin (2 s), measured on the guard's own
    // injected clock; a hold ending within the margin, or already over, gets
    // the backup (OnBeginChunk may re-arm it a moment later).
    const std::int64_t trip = h.now, margin = 2 * kSec;
    const std::int64_t end = trip + 600 * kSec;
    assert(g.StaysHeld());
    h.now = end - margin - 1;  // outlasts the margin by 1 ns
    assert(g.StaysHeld());
    h.now = end - margin;  // exactly the margin: not beyond it
    assert(!g.StaysHeld());
    h.now = end - 1 * kSec;  // expires within the margin
    assert(!g.StaysHeld());
    h.now = end + 1 * kSec;  // already over
    assert(!g.StaysHeld());
    // Feed's check and the chunk start agree on one clock: while the check
    // says "stays held", OnBeginChunk (a moment later) is still held; when the
    // check says "backup", OnBeginChunk may re-arm.
    h.now = end - margin - 1 * kSec;
    assert(g.StaysHeld());
    h.now += 1 * kSec;  // the gap between the check and the chunk start
    assert(g.OnBeginChunk());
    h.now = end + 1 * kSec;
    h.power = Pw(60);  // clear window: the re-arm succeeds
    assert(!g.StaysHeld() && !g.OnBeginChunk());
  }
  {  // log mode and an untripped guard never hold
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kLog, {}, h.Hooks());
    assert(!g.StaysHeld());
    NpuGuard on(GuardMode::kOn, {}, h.Hooks());
    assert(!on.StaysHeld());
  }
  {  // exactly at the threshold is not above it; below resets; flapping never
     // trips
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    for (int i = 0; i < 12; ++i)
      Chunk(g, h, Pw(i % 3 == 2 ? 70 : 90));
    Chunk(g, h, Pw(82.0));
    Chunk(g, h, Pw(82.0));
    Chunk(g, h, Pw(82.0));
    assert(g.trips() == 0);
  }
  {  // approx / short chunks are neutral: neither count nor reset
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(40, 3, /*approx=*/true));
    Chunk(g, h, Pw(40, 1));  // one interval: below POWER_MIN_INTERVALS
    assert(g.trips() == 0);
    Chunk(g, h, Pw(90));
    assert(g.trips() == 1);
  }
  {  // chip power unavailable: no trip and the run is forgotten
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    PowerWindow none;
    Chunk(g, h, none);
    Chunk(g, h, Pw(90));
    assert(g.trips() == 0);
  }
  {  // an idle gap over POWER_RESET_IDLE_S forgets the run
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90), /*idle_gap_ms=*/61'000);
    assert(g.trips() == 0);
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    assert(g.trips() == 1);
  }
}

void MemoryRule() {
  const auto fresh = [](Harness& h) {
    h.memory = Mem(40);
    return NpuGuard(GuardMode::kOn, {}, h.Hooks());
  };
  {  // startup grace: nothing is evaluated in the first 60 s of chunks
    Harness h;
    auto g = fresh(h);
    g.OnBeginChunk();  // first chunk
    h.memory = Mem(99);
    h.now += 30 * kSec;
    assert(!g.OnBeginChunk() && g.trips() == 0);
    h.now += 31 * kSec;
    assert(g.OnBeginChunk() && g.trips() == 1);
    assert(h.Logged(
        "held (memory): commit_pct_min=99.0 >= 98.0 over 29 s (n=30 samples)"));
    assert(h.published.size() == 1 && h.published[0].reason == "memory" &&
           h.published[0].max_commit_pct &&
           *h.published[0].max_commit_pct == 99);
  }
  struct Case {
    const char* name;
    MemoryWindow w;
    bool trips;
  };
  const std::vector<Case> cases = {
      {"97.9 % for the window", Mem(97.9), false},
      {"98.0 % for the window", Mem(98.0), true},
      {"14 samples", Mem(99, 14), false},
      {"15 samples", Mem(99, 15), true},
      {"span 24 s", Mem(99, 30, 24), false},
      {"span 25 s", Mem(99, 30, 25), true},
      {"gap 10.1 s", Mem(99, 30, 29, 10.1), false},
      {"gap 10.0 s", Mem(99, 30, 29, 10.0), true},
      {"pages high, avail 1024", Mem(40, 30, 29, 1, 100000, 1024), false},
      {"pages high, avail 1023", Mem(40, 30, 29, 1, 100000, 1023), true},
      {"pages 99999, avail low", Mem(40, 30, 29, 1, 99999, 500), false},
      {"pages low, avail low", Mem(40, 30, 29, 1, 100, 500), false},
      {"no pages counter, avail low", Mem(40, 30, 29, 1, -1, 500), false},
      {"pages samples 14",
       [] {
         auto w = Mem(40, 30, 29, 1, 100000, 500);
         w.pages_samples = 14;
         return w;
       }(),
       false},
  };
  for (const auto& c : cases) {
    Harness h;
    auto g = fresh(h);
    g.OnBeginChunk();
    h.now += 61 * kSec;
    h.memory = c.w;
    g.OnBeginChunk();
    if (g.tripped() != c.trips) {
      std::cout << "memory case failed: " << c.name << "\n";
      assert(false);
    }
  }
  {  // pages trip message
    Harness h;
    auto g = fresh(h);
    g.OnBeginChunk();
    h.now += 61 * kSec;
    h.memory = Mem(40, 30, 29, 1, 150000, 600);
    g.OnBeginChunk();
    assert(
        h.Logged("held (memory): pages_mean=150000/s >= 100000 and "
                 "avail_min=600 MiB < 1024 over 29 s"));
  }
}

void HoldAndRearm() {
  {  // before expiry stays held; after expiry needs the clear conditions
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    HotRun(g, h);
    const auto tripped_at = h.now;
    assert(g.held());
    h.now = tripped_at + 599 * kSec;
    h.power = Pw(10);  // would be clear, but the time is not over
    assert(g.OnBeginChunk() && h.warm_calls == 0);
    h.now = tripped_at + 601 * kSec;
    h.power = Pw(76.0);  // exactly the clear level: not below it
    assert(g.OnBeginChunk() && h.warm_calls == 0);
    assert(h.Logged("conditions are not clear"));
    h.power = Pw(75.9);
    h.memory = Mem(95.0);
    assert(g.OnBeginChunk() && h.warm_calls == 0);  // commit 95.0 is not < 95.0
    h.memory = Mem(94.9);
    assert(!g.OnBeginChunk());  // clear: re-armed
    assert(h.warm_calls == 1 && !g.tripped() && g.trips() == 1);
    assert(h.published.size() == 2 && !h.published[1].held &&
           h.published[1].trips == 1);
    assert((h.paused == std::vector<bool>{true, false}));
    assert(h.Logged("re-arm warm-up beat ok in 12.0 ms"));
    assert(h.Logged("npu guard: NPU prefill re-armed after "));
    assert(g.ConsumeRearmFlag() && !g.ConsumeRearmFlag());
    // After re-arm the power rule counts from zero.
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    assert(g.trips() == 1);
  }
  {  // chip power unavailable at the clear check counts as clear
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    HotRun(g, h);
    h.now += 601 * kSec;
    h.power = PowerWindow{};
    assert(!g.OnBeginChunk());
  }
  {  // heavy paging keeps a memory hold
    Harness h;
    h.memory = Mem(40);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    g.OnBeginChunk();
    h.now += 61 * kSec;
    h.memory = Mem(99);
    g.OnBeginChunk();
    assert(g.held());
    h.now += 601 * kSec;
    h.memory = Mem(40, 30, 29, 1, 200000, 500);  // pages condition still true
    assert(g.OnBeginChunk());
    h.memory = Mem(40, 30, 29, 1, 200, 5000);
    assert(!g.OnBeginChunk());
  }
  {  // the warm-up beat fails: stays held, nothing re-armed, no publish of
     // clear
    Harness h;
    Quiet(h);
    h.warm_ok = false;
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    HotRun(g, h);
    h.now += 601 * kSec;
    h.power = Pw(60);
    assert(g.OnBeginChunk());
    assert(h.warm_calls == 1 && g.tripped() && h.published.size() == 1);
    assert(
        h.Logged("re-arm warm-up beat FAILED after 12.0 ms: XRT heartbeat did "
                 "not complete"));
  }
  {  // sticky after STICKY_AFTER_TRIPS: no timer, never re-armed
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    for (int trip = 1; trip <= 3; ++trip) {
      HotRun(g, h);
      assert(g.trips() == static_cast<unsigned>(trip));
      if (trip < 3) {
        h.now += 601 * kSec;
        h.power = Pw(60);
        assert(!g.OnBeginChunk());
      }
    }
    assert(g.sticky() && h.published.back().sticky);
    assert(h.Logged("sticky after 3 trips, stays held until restart"));
    assert(g.StaysHeld());  // sticky: always skip, whatever the clock says
    h.now += 100000 * kSec;
    assert(g.StaysHeld());
    h.power = Pw(10);
    assert(g.OnBeginChunk() && g.tripped());
  }
  {  // chunk ends while held are ignored (Trip() itself also refuses to
     // re-enter)
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    HotRun(g, h);
    ChunkStart s;
    ChunkEnd e;
    e.power = Pw(95);
    for (int i = 0; i < 5; ++i)
      g.OnChunkEnd(s, e);
    assert(g.trips() == 1 && h.published.size() == 1);
  }
}

void ModesAndFaults() {
  {  // log mode: the state machine runs, nothing is held or published
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kLog, {}, h.Hooks());
    for (int i = 0; i < 3; ++i)
      assert(!Chunk(g, h, Pw(90)));
    assert(g.tripped() && !g.held() && g.trips() == 1);
    assert(h.published.empty() && h.paused.empty());
    assert(h.Logged("LOG MODE would hold NPU prefill (power)"));
    assert(!h.Logged("held (power)"));
    assert(!g.OnBeginChunk());  // still not held
    h.now += 601 * kSec;
    h.power = Pw(60);
    assert(!g.OnBeginChunk() && !g.tripped());
    assert(h.warm_calls == 0 && h.published.empty());  // no beat, no status
    assert(h.Logged("re-armed"));
  }
  {  // off computes nothing
    Harness h;
    NpuGuard g(GuardMode::kOff, {}, h.Hooks());
    for (int i = 0; i < 10; ++i)
      Chunk(g, h, Pw(99));
    h.now += 1000 * kSec;
    assert(!g.OnBeginChunk());
    assert(h.memory_calls == 0 && h.power_calls == 0 && g.trips() == 0);
  }
  {  // a fault stops everything: no trip, no re-arm
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    Chunk(g, h, Pw(90));
    Chunk(g, h, Pw(90));
    h.failed = true;
    Chunk(g, h, Pw(90));
    assert(g.trips() == 0 && !g.OnBeginChunk());
  }
  {  // trip, then a fault: never re-armed, no warm-up beat
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, {}, h.Hooks());
    HotRun(g, h);
    h.failed = true;
    h.now += 601 * kSec;
    h.power = Pw(60);
    assert(!g.OnBeginChunk() && h.warm_calls == 0 && g.tripped());
  }
  {  // the test override: lower threshold and short hold
    GuardOverrides o;
    std::string error;
    assert(ParseGuardOverrides("power_w=30,hold_s=60", &o, &error));
    assert(*o.power_w == 30 && *o.hold_s == 60);
    Harness h;
    Quiet(h);
    NpuGuard g(GuardMode::kOn, o, h.Hooks());
    HotRun(g, h, 31.0);
    assert(g.held() && g.hold_seconds() == 60 && g.power_threshold_w() == 30);
    assert(h.Logged("hold 60 s"));
    h.now += 59 * kSec;
    h.power = Pw(20);
    assert(g.OnBeginChunk());
    h.now += 2 * kSec;
    assert(!g.OnBeginChunk());
    for (const char* bad :
         {"power_w", "power_w=", "power_w=x", "power_w=-1", "hold_s=0",
          "volts=3", "power_w=30,", ",power_w=30", "power_w=30,,hold_s=60",
          "power_w=0.5", "power_w=1001", "hold_s=86401", "power_w=nan",
          "power_w=inf", "hold_s=1e999", ","})
      assert(!ParseGuardOverrides(bad, &o, &error));
    for (const char* good : {"power_w=1", "power_w=1000", "hold_s=1",
                             "hold_s=86400", "hold_s=90,power_w=30"})
      assert(ParseGuardOverrides(good, &o, &error));
    assert(ParseGuardOverrides("", &o, &error) && !o.power_w && !o.hold_s);
  }
  {  // mode parsing
    GuardMode m;
    assert(ParseGuardMode("on", &m) && m == GuardMode::kOn);
    assert(ParseGuardMode("off", &m) && m == GuardMode::kOff);
    assert(ParseGuardMode("log", &m) && m == GuardMode::kLog);
    assert(!ParseGuardMode("", &m) && !ParseGuardMode("ON", &m) &&
           !ParseGuardMode("1", &m));
  }
}

// Guard logic only, on a fake feed (not the real sampler or telemetry).
void Cost() {
  Harness h;
  h.memory = Mem(40);
  NpuGuard g(GuardMode::kOn, {}, h.Hooks());
  g.OnBeginChunk();
  h.now += 61 * kSec;
  constexpr int kCalls = 2'000'000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i)
    g.OnBeginChunk();
  const auto ns = std::chrono::duration<double, std::nano>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  kCalls;
  std::cout << "COST: OnBeginChunk (latch + memory evaluation, fake feed) "
            << ns << " ns/call\n";
}
}  // namespace

int main() {
  PowerRule();
  MemoryRule();
  HoldAndRearm();
  ModesAndFaults();
  Cost();
  std::cout << "PASS npu guard: power rule, memory rule, hold/re-arm/sticky, "
               "log/off modes, faults win\n";
}
