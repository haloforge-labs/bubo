// SPDX-License-Identifier: MIT
// CPU only: chip power logging (G0) with a fake source. No PDH, no NPU/GPU.
#include "src/npu/chip_power.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "src/npu/fault_telemetry.hpp"
#include "src/npu/request_stats.hpp"

using namespace gufo::npu;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
constexpr std::int64_t kSec = 1'000'000'000;
// Energy (pWh) that a constant `watts` adds over `seconds`.
double Pwh(double watts, double seconds) {
  return watts * seconds / kJoulesPerPwh;
}

struct FakeSource final : IChipPowerSource {
  std::vector<PowerSample> script;
  std::size_t next = 0;
  bool open_ok = true;
  std::string open_error = "RAPL package counters missing";
  bool fail_after_script = true;
  // After the script: this many failing reads, then synthetic good samples.
  int fail_count = 1 << 30, failures_served = 0;
  int opens = 0, reads = 0, closes = 0;
  bool Open(std::string* error) override {
    ++opens;
    if (!open_ok)
      *error = open_error;
    return open_ok;
  }
  bool Read(PowerSample* sample, std::string* error) override {
    ++reads;
    if (next >= script.size()) {
      if (fail_after_script && failures_served < fail_count) {
        ++failures_served;
        *error = "invalid energy counter data";
        return false;
      }
      *sample = script.back();
      sample->ts_ns += static_cast<std::int64_t>(reads) * kSec;
      sample->energy_pwh += Pwh(50, reads);
      return true;
    }
    *sample = script[next++];
    return true;
  }
  void Close() noexcept override { ++closes; }
};

// Samples one second apart: 54 W for 3 s, then 90 W, then 54 W.
std::vector<PowerSample> Script(std::int64_t base) {
  std::vector<PowerSample> s;
  double e = 1e15;
  const double watts[] = {0, 54, 54, 54, 90, 54};
  for (int i = 0; i < 6; ++i) {
    e += Pwh(watts[i], 1);
    s.push_back({base + i * kSec, e});
  }
  return s;
}

struct FakePages final : IPagesSource {
  bool open_ok = true;
  std::vector<double> values;
  std::size_t next = 0;
  bool Open(std::string* error) override {
    if (!open_ok)
      *error = "counter missing";
    return open_ok;
  }
  bool Read(double* value, std::string* error) override {
    if (next >= values.size()) {
      *error = "invalid Pages/sec data";
      return false;
    }
    *value = values[next++];
    return true;
  }
};
struct FakeMemory final : IMemorySource {
  MemorySample sample;
  MemorySample Sample() override { return sample; }
};

struct Captured {
  std::vector<std::pair<bool, std::string>> lines;
  ChipPowerMonitor::Log fn() {
    return [this](bool warn, const std::string& text) {
      lines.emplace_back(warn, text);
    };
  }
};

void Normal() {
  FaultState state;
  Captured log;
  auto owned = std::make_unique<FakeSource>();
  owned->script = Script(10 * kSec);
  auto* source = owned.get();
  ChipPowerMonitor monitor(std::move(owned), state, log.fn());
  assert(monitor.Start(true, /*with_thread=*/false));
  assert(!monitor.thread_running() && state.chip_power_state == 1);
  assert(source->opens == 1 && source->reads == 1);  // priming read
  for (int i = 0; i < 5; ++i)
    assert(monitor.SampleOnce());
  // Window covering the 90 W second and the 54 W seconds around it.
  auto w = monitor.Window(11 * kSec + 1, 15 * kSec);
  assert(w.ok && !w.approx && w.intervals == 4);
  assert(w.mean_w > 62.9 && w.mean_w < 63.1);  // (54+54+90+54)/4
  assert(w.max_w > 89.9 && w.max_w < 90.1);
  // Chunk shorter than the sampling period: nearest second, flagged approx.
  w = monitor.Window(12 * kSec + 100, 12 * kSec + 200);
  assert(w.ok && w.approx && w.intervals == 1);
  assert(w.mean_w > 53.9 && w.mean_w < 54.1);
  // Before any interval exists: not ok.
  assert(!monitor.Window(0, 5 * kSec).ok);
  assert(!log.lines.empty() && !log.lines[0].first);
}

void Unavailable() {
  {  // counter set missing: state unavailable, logged once, gate open
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    owned->open_ok = false;
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    assert(!monitor.Start(true, false) && state.chip_power_state == 3);
    assert(!monitor.SampleOnce() && !monitor.SampleOnce());
    assert(!monitor.Window(0, 100 * kSec).ok);
    assert(log.lines.size() == 1 &&
           Has(log.lines[0].second, "chip power unavailable") &&
           Has(log.lines[0].second, "no gate"));
  }
  {  // read failure after start: one log line, ring cleared
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    owned->script = Script(0);
    owned->script.resize(3);
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    assert(monitor.Start(true, false) && monitor.SampleOnce() &&
           monitor.SampleOnce());
    // Failures are tolerated: the ring is cleared but the source stays
    // available until kFailuresToUnavailable reads fail in a row.
    for (unsigned i = 1; i < ChipPowerMonitor::kFailuresToUnavailable; ++i) {
      assert(!monitor.SampleOnce() && state.chip_power_state == 1);
      assert(!monitor.Window(0, 10 * kSec).ok);
    }
    assert(!monitor.SampleOnce() && state.chip_power_state == 3);
    assert(!monitor.SampleOnce());
    int unavailable = 0;
    for (const auto& [warn, text] : log.lines)
      if (Has(text, "unavailable"))
        ++unavailable;
    assert(unavailable == 1 && !monitor.Window(0, 10 * kSec).ok);
  }
  {  // switch off: the source is never opened
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    auto* source = owned.get();
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    assert(!monitor.Start(false) && source->opens == 0 && source->reads == 0 &&
           state.chip_power_state == 0 && log.lines.empty());
  }
}

void BadData() {
  struct Case {
    const char* name;
    std::vector<PowerSample> samples;
    const char* expect;
  };
  const std::vector<Case> cases = {
      {"energy wraps or resets",
       {{1 * kSec, 1e15}, {2 * kSec, 1e15 + Pwh(50, 1)}, {3 * kSec, 5e14}},
       "went backwards"},
      {"time not increasing",
       {{2 * kSec, 1e15}, {2 * kSec, 1e15 + Pwh(50, 1)}},
       "non-monotonic sample time"},
      {"time goes back",
       {{2 * kSec, 1e15}, {1 * kSec, 1e15 + Pwh(50, 1)}},
       "non-monotonic sample time"},
      {"over 400 W",
       {{1 * kSec, 1e15}, {2 * kSec, 1e15 + Pwh(500, 1)}},
       "outside 0-400 W"},
  };
  for (const auto& c : cases) {
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    owned->script = c.samples;
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    assert(monitor.Start(true, false));
    bool failed = false;
    for (std::size_t i = 1; i < c.samples.size(); ++i)
      if (!monitor.SampleOnce())
        failed = true;
    if (!failed) {
      std::cout << "case did not fail: " << c.name << "\n";
      assert(false);
    }
    // One bad sample is not fatal: the ring is cleared (no interval may span
    // the gap) and the source stays available.
    assert(state.chip_power_state == 1 && !monitor.Window(0, 9 * kSec).ok);
    for (const auto& [warn, text] : log.lines)
      assert(!Has(text, "unavailable"));
    (void)c.expect;
  }
  {  // exactly 400 W and 0 W are accepted
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    owned->script = {{1 * kSec, 1e15},
                     {2 * kSec, 1e15 + Pwh(400, 1)},
                     {3 * kSec, 1e15 + Pwh(400, 1)}};
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    assert(monitor.Start(true, false) && monitor.SampleOnce() &&
           monitor.SampleOnce());
  }
}

void Recovery() {
  // A transient outage (here 5 failed reads, e.g. after sleep/resume) makes the
  // power unavailable; the next good sample brings it back and the window
  // starts from scratch.
  FaultState state;
  Captured log;
  auto owned = std::make_unique<FakeSource>();
  owned->script = Script(10 * kSec);
  owned->fail_count =
      static_cast<int>(ChipPowerMonitor::kFailuresToUnavailable);
  ChipPowerMonitor monitor(std::move(owned), state, log.fn());
  assert(monitor.Start(true, false));
  for (int i = 0; i < 5; ++i)
    assert(monitor.SampleOnce());
  for (unsigned i = 0; i < ChipPowerMonitor::kFailuresToUnavailable; ++i)
    assert(!monitor.SampleOnce());
  assert(state.chip_power_state == 3 && !monitor.available());
  assert(monitor.SampleOnce());  // the source works again
  assert(state.chip_power_state == 1 && monitor.available());
  assert(!monitor.Window(0, 1000 * kSec).ok);  // one sample: no interval yet
  assert(monitor.SampleOnce());
  assert(monitor.Window(0, 1000 * kSec).ok);
  int unavailable = 0, recovered = 0;
  for (const auto& [warn, text] : log.lines) {
    unavailable += Has(text, "chip power unavailable");
    recovered += Has(text, "chip power recovered");
  }
  assert(unavailable == 1 && recovered == 1);
}

void SamplerWithoutRapl() {
  // GUFO_NPU_CHIP_POWER=0 with the guard on: the memory sampler runs, RAPL is
  // never opened, and no "chip power logging on" line appears.
  FaultState state;
  Captured log;
  auto owned = std::make_unique<FakeSource>();
  auto* source = owned.get();
  ChipPowerMonitor monitor(std::move(owned), state, log.fn());
  monitor.Start(true, false, /*power_source=*/false);
  monitor.Tick();
  monitor.Tick();
  assert(source->opens == 0 && source->reads == 0);
  assert(state.chip_power_state == 0 && !monitor.available());
  assert(monitor.MemWindow(0, INT64_MAX).ok);
  for (const auto& [warn, text] : log.lines)
    assert(!Has(text, "chip power logging on"));
}

void MemoryWindowForGuard() {
  struct Fake final : IMemorySource {
    MemorySample s{true, 8000, 40000, 100000};
    MemorySample Sample() override { return s; }
  } memory;
  auto* previous = SetMemorySourceForTest(&memory);
  FaultState state;
  std::int64_t now = 100 * kSec;
  auto owned = std::make_unique<FakeSource>();
  owned->open_ok = false;
  ChipPowerMonitor monitor(std::move(owned), state, {}, std::chrono::seconds(1),
                           [&] { return now; });
  struct Pages final : IPagesSource {
    std::vector<double> v;
    std::size_t i = 0;
    bool Open(std::string*) override { return true; }
    bool Read(double* out, std::string* e) override {
      if (i >= v.size()) {
        *e = "x";
        return false;
      }
      *out = v[i++];
      return true;
    }
  };
  auto pages = std::make_unique<Pages>();
  // Priming point + 5 ticks: 1e8 (a spike), then 200000, 0, 0, 0.
  pages->v = {1e8, 200000, 0, 0, 0};
  monitor.SetPagesSource(std::move(pages));
  monitor.Start(true, false);  // point at 100 s: pages 1e8
  for (int i = 0; i < 5; ++i) {
    now += kSec;
    memory.s.commit_mib = 40000 + i * 10000;  // 40 %, 50 %, ... 80 %
    memory.s.available_mib = 8000 - i * 1000;
    monitor.Tick();
  }
  const auto w = monitor.MemWindow(100 * kSec, 106 * kSec, 300000.0);
  // Points after t0 only (the 100 s point is excluded): 5 samples over 4 s.
  assert(w.ok && w.samples == 5 && w.span_s > 3.9 && w.span_s < 4.1);
  assert(w.max_gap_s > 0.99 && w.max_gap_s < 1.01);
  assert(w.min_commit_pct > 39.9 && w.min_commit_pct < 40.1);
  assert(w.max_commit_pct > 79.9 && w.max_commit_pct < 80.1);
  assert(w.min_avail_mib == 4000);
  assert(w.pages_samples == 4);  // the 5th pages read failed -> n/a
  assert(w.pages_mean > 50000 - 1 && w.pages_mean < 50000 + 1);  // 200000/4
  // The spike is clamped: a window holding the 1e8 sample averages 300000.
  const auto s = monitor.MemWindow(0, 106 * kSec, 300000.0);
  assert(s.pages_samples == 5 && s.pages_mean > 99999 &&
         s.pages_mean < 100001);  // (3e5+2e5)/5
  const auto raw = monitor.MemWindow(0, 106 * kSec);
  assert(raw.pages_mean > 1e7);  // unclamped, for comparison
  // A hole in the sampling shows as a large max gap.
  now += 25 * kSec;
  monitor.Tick();
  const auto holey = monitor.MemWindow(100 * kSec, now);
  assert(holey.max_gap_s > 24.9);
  SetMemorySourceForTest(previous);
}

void Threaded() {
  FaultState state;
  Captured log;
  auto owned = std::make_unique<FakeSource>();
  owned->script = Script(SteadyNowNs());
  owned->script.resize(1);
  owned->fail_after_script = false;
  auto* source = owned.get();
  ChipPowerMonitor monitor(std::move(owned), state, log.fn(),
                           std::chrono::milliseconds(10));
  assert(monitor.Start(true, true) && monitor.thread_running());
  for (int i = 0; i < 200 && source->reads < 4; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  assert(source->reads >= 4);
  const auto t0 = std::chrono::steady_clock::now();
  monitor.Stop();  // joins
  assert(!monitor.thread_running() && source->closes >= 1);
  assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
}

void ChunkTelemetry() {
  // Samples in the recent past; a chunk now falls in the "nearest second" case.
  const auto now = SteadyNowNs();
  FaultState& state = Faults();
  auto owned = std::make_unique<FakeSource>();
  double e = 1e15;
  owned->script = {{now - 3 * kSec, e},
                   {now - 2 * kSec, e += Pwh(54, 1)},
                   {now - 1 * kSec, e += Pwh(80, 1)}};
  InstallGlobalChipPower(std::move(owned), true);
  assert(GlobalChipPower() && state.chip_power_state == 1);
  GlobalChipPower()->SampleOnce();
  GlobalChipPower()->SampleOnce();
  Stats().chunks = 0;
  Stats().last_layer = -1;
  NoteBegin(0, true);
  const auto start = BeginChunkTelemetry(4096);
  volatile double burn = 0;
  for (int i = 0; i < 2'000'000; ++i)
    burn = burn + i * 0.5;
  const auto end = EndChunkTelemetry(start);
  assert(end.power.ok && end.power.approx);
  assert(end.power.mean_w > 79.9 && end.power.mean_w < 80.1);
  assert(end.cpu_ms >= 0);
  NoteRequestChunk(start, end);
  const auto line = FormatChunkFailure(start, end, kInvalidDecoderExp255);
  std::cout << "POWER-FAILURE-FIELDS:" << line.substr(line.find(" monitor="))
            << "\n";
  for (const char* part :
       {"chip_power_w_mean=80.0", "chip_power_w_max=80.0", "chip_power_n=1",
        "chip_power_approx=1", "cpu_ms=", "chip_power=on",
        " commit_limit_mib=", " commit_pct_end=", " min_avail_mib=",
        " max_commit_pct=", " pages_per_s_max="})
    assert(Has(line, part));
  const auto suffix = TakeRequestTelemetrySuffix();
  std::cout << "POWER-REQUEST-SUFFIX:" << suffix << "\n";
  for (const char* part :
       {" chip_power_w_mean=80.0 chip_power_w_max=80.0",
        " cpu_ms=", " chip_power=on", " commit_limit_mib=", " min_avail_mib=",
        " max_commit_pct=", " pages_per_s_max=", " chunk_power_w=[~80.0]",
        " chunk_power_n=1", " first_chunk_after_rearm=0"})
    assert(Has(suffix, part));
  assert(Has(line, " first_chunk_after_rearm=0"));
  // After the source fails: fields print na, state unavailable.
  for (unsigned i = 0; i < ChipPowerMonitor::kFailuresToUnavailable; ++i)
    GlobalChipPower()->SampleOnce();  // script exhausted: reads fail in a row
  assert(state.chip_power_state == 3);
  const auto start2 = BeginChunkTelemetry(256);
  const auto end2 = EndChunkTelemetry(start2);
  assert(!end2.power.ok);
  const auto line2 = FormatChunkFailure(start2, end2, 0);
  assert(Has(line2, "chip_power_w_mean=na") &&
         Has(line2, "chip_power=unavailable"));
  StopGlobalChipPower();
  assert(GlobalChipPower() == nullptr);
  state.chip_power_state = 0;
}

void MemoryAndPages() {
  FakeMemory memory;
  memory.sample = {true, 8000, 40000, 80000};
  auto* previous = SetMemorySourceForTest(&memory);
  {  // power unavailable on purpose: memory points and Pages/sec keep going
    FaultState state;
    Captured log;
    std::int64_t now = 100 * kSec;
    auto owned = std::make_unique<FakeSource>();
    owned->open_ok = false;
    ChipPowerMonitor monitor(std::move(owned), state, log.fn(),
                             std::chrono::seconds(1), [&] { return now; });
    auto pages = std::make_unique<FakePages>();
    pages->values = {10, 5000, 20};
    monitor.SetPagesSource(std::move(pages));
    assert(!monitor.Start(true, false) && state.chip_power_state == 3);
    assert(state.pages_state == 1);  // priming point at 100 s read 10
    now += kSec;
    memory.sample = {true, 3000, 70000, 80000};
    monitor.Tick();  // 5000 pages/s, 87.5 % commit, 3000 MiB available
    now += kSec;
    memory.sample = {true, 6000, 50000, 80000};
    monitor.Tick();  // 20 pages/s
    now += kSec;
    monitor.Tick();  // counter fails: unavailable once, point without pages
    const auto w = monitor.MemWindow(100 * kSec, 103 * kSec);
    assert(w.ok && w.min_avail_mib == 3000 && w.max_pages_per_s == 5000);
    assert(w.max_commit_pct > 87.4 && w.max_commit_pct < 87.6);
    assert(state.pages_state == 3);
    monitor.Tick();
    int pages_lines = 0;
    for (const auto& [warn, text] : log.lines)
      if (Has(text, "Pages/sec unavailable"))
        ++pages_lines;
    assert(pages_lines == 1);
    assert(!monitor.MemWindow(200 * kSec, 300 * kSec).ok);
  }
  {  // Pages/sec counter missing: logged once, power still works
    FaultState state;
    Captured log;
    auto owned = std::make_unique<FakeSource>();
    owned->script = Script(10 * kSec);
    ChipPowerMonitor monitor(std::move(owned), state, log.fn());
    auto pages = std::make_unique<FakePages>();
    pages->open_ok = false;
    monitor.SetPagesSource(std::move(pages));
    assert(monitor.Start(true, false) && state.chip_power_state == 1 &&
           state.pages_state == 3);
    monitor.Tick();
    monitor.Tick();
    int pages_lines = 0;
    for (const auto& [warn, text] : log.lines)
      if (Has(text, "Pages/sec unavailable"))
        ++pages_lines;
    assert(pages_lines == 1);
  }
  {  // switch off: no memory sampling either
    FaultState state;
    auto owned = std::make_unique<FakeSource>();
    ChipPowerMonitor monitor(std::move(owned), state, {});
    assert(!monitor.Start(false) && state.pages_state == 0);
    assert(!monitor.MemWindow(0, 1000 * kSec).ok);
  }
  SetMemorySourceForTest(previous);
}

void Cost() {
  // Window() over a full 512-sample ring: the only power work on the chunk
  // path.
  FaultState state;
  auto owned = std::make_unique<FakeSource>();
  double e = 1e15;
  for (int i = 0; i < 600; ++i) {
    e += Pwh(60, 1);
    owned->script.push_back({static_cast<std::int64_t>(i) * kSec, e});
  }
  ChipPowerMonitor monitor(std::move(owned), state, {});
  assert(monitor.Start(true, false));
  for (int i = 1; i < 600; ++i)
    assert(monitor.SampleOnce());
  constexpr int kCalls = 20000;
  double sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i)
    sink += monitor.Window(560 * kSec, 570 * kSec).mean_w;
  const auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  kCalls;
  {  // the guard's memory input: a 30 s window of the 1 Hz memory ring
    FakeMemory memory;
    memory.sample = {true, 8000, 40000, 100000};
    auto* previous = SetMemorySourceForTest(&memory);
    FaultState mem_state;
    std::int64_t now = 100 * kSec;
    ChipPowerMonitor sampler(std::make_unique<FakeSource>(), mem_state, {},
                             std::chrono::seconds(1), [&] { return now; });
    sampler.Start(true, false, /*power_source=*/false);
    for (int i = 0; i < 600; ++i) {
      now += kSec;
      sampler.Tick();
    }
    const auto m0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kCalls; ++i)
      sink += sampler.MemWindow(now - 30 * kSec, now, 300000.0).samples;
    const auto mem_us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - m0)
                            .count() /
                        kCalls;
    std::cout << "COST: MemWindow (30 s window of a full 512-point ring, guard "
                 "input) "
              << mem_us << " us/call\n";
    SetMemorySourceForTest(previous);
  }
  std::cout << "COST: ChipPowerMonitor::Window over a full ring " << us
            << " us/chunk (cached samples, no PDH); sink=" << (sink > 0)
            << "\n";
  assert(us < 50.0);
}

void ChunkListCap() {
  (void)TakeRequestTelemetrySuffix();
  ChunkStart start;
  for (int i = 0; i < 20; ++i) {
    ChunkEnd end;
    if (i % 5 == 4)
      end.power = {};  // no power data: "na"
    else
      end.power = PowerWindow{true, i == 1, 60.0 + i, 61.0 + i, 3.0, 3};
    start.first_after_rearm = i == 0;
    NoteRequestChunk(start, end);
  }
  const auto suffix = TakeRequestTelemetrySuffix();
  const auto at = suffix.find(" chunk_power_w=[");
  assert(at != std::string::npos);
  const auto close = suffix.find(']', at);
  const std::string list = suffix.substr(at + 16, close - at - 16);
  std::size_t commas = 0;
  for (char c : list)
    commas += c == ',';
  assert(commas == 15);  // first 16 entries only
  assert(list.rfind("60.0,~61.0,62.0,63.0,na,65.0", 0) == 0);
  assert(Has(suffix, "] chunk_power_n=20 first_chunk_after_rearm=1"));
  // The next request starts clean.
  start.first_after_rearm = false;
  ChunkEnd one;
  one.power = PowerWindow{true, false, 70.0, 70.0, 3.0, 3};
  NoteRequestChunk(start, one);
  const auto next = TakeRequestTelemetrySuffix();
  assert(Has(
      next, " chunk_power_w=[70.0] chunk_power_n=1 first_chunk_after_rearm=0"));
}

void Scheme() {
  PowerScheme none;
  assert(FormatPowerScheme(none) == "power scheme unavailable");
  PowerScheme scheme{true, "{6FECC5AE-F350-48A5-B669-B472CB895CCF}",
                     "Performance", "{DED574B5-45A0-4F42-8737-46345C09C238}"};
  assert(FormatPowerScheme(scheme) ==
         "power scheme {6FECC5AE-F350-48A5-B669-B472CB895CCF} name=Performance "
         "overlay={DED574B5-45A0-4F42-8737-46345C09C238}");
}
}  // namespace

int main() {
  Normal();
  Unavailable();
  BadData();
  Recovery();
  SamplerWithoutRapl();
  MemoryWindowForGuard();
  Threaded();
  ChunkTelemetry();
  ChunkListCap();
  Scheme();
  MemoryAndPages();
  Cost();
  std::cout << "PASS chip power: normal, unavailable, wrap/non-monotonic, "
               "out of range, thread-less and threaded, chunk fields\n";
}
