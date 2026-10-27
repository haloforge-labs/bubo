// SPDX-License-Identifier: MIT
// CPU only: G0b memory trace (buffer-page residency + GPU memory counters) with
// fake probe and fake PDH sources. No PDH, no NPU/GPU.
#include "src/npu/mem_trace.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "src/npu/chip_power.hpp"
#include "src/npu/fault_telemetry.hpp"
#include "src/npu/request_stats.hpp"

using namespace gufo::npu;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
struct FakeProbe final : IMemProbe {
  std::vector<MemTraceSample> script;
  std::size_t next = 0;
  bool fail = false;
  int calls = 0;
  bool Sample(MemTraceSample* sample, std::string* error) override {
    ++calls;
    if (fail) {
      *error = "QueryWorkingSetEx failed, error 5";
      return false;
    }
    *sample = script[std::min(next++, script.size() - 1)];
    return true;
  }
};
MemTraceSample Sample(unsigned valid, unsigned locked, std::uint64_t ws,
                      std::uint64_t priv) {
  MemTraceSample s;
  s.ok = true;
  s.pages_sampled = 8;
  s.valid = valid;
  s.shared = 0;
  s.locked = locked;
  s.large = 0;
  s.ws_mib = ws;
  s.private_mib = priv;
  return s;
}
struct FakeGpu final : IGpuMemSource {
  std::vector<GpuMemPoint> script;
  std::size_t next = 0;
  bool open_ok = true;
  std::string read_error;  // non-empty: Read fails with this error
  bool Open(std::string* error) override {
    if (!open_ok)
      *error = "GPU memory counter sets not found";
    return open_ok;
  }
  bool Read(GpuMemPoint* point, std::string* error) override {
    if (!read_error.empty()) {
      *error = read_error;
      return false;
    }
    *point = script[std::min(next++, script.size() - 1)];
    return point->proc_ok ||
           point->adapter_ok;  // no instances yet: false, no error
  }
};
GpuMemPoint Gpu(double local, double dedicated, double shared, double adapter) {
  GpuMemPoint p;
  p.proc_ok = true;
  p.proc_local_mib = local;
  p.proc_dedicated_mib = dedicated;
  p.proc_shared_mib = shared;
  p.adapter_ok = true;
  p.adapter_shared_mib = adapter;
  return p;
}

void ResidencyDropIsReported() {
  std::vector<std::string> log;
  SetMemTraceLogForTest([&](const std::string& t) { log.push_back(t); });
  auto probe = std::make_unique<FakeProbe>();
  probe->script = {Sample(8, 0, 5200, 6100), Sample(5, 0, 4100, 6150)};
  InstallGlobalMemProbe(std::move(probe));
  Stats().chunks = 0;
  Stats().last_layer = -1;
  NoteBegin(0, true);
  const auto start = BeginChunkTelemetry(4096);
  const auto end = EndChunkTelemetry(start);
  NoteRequestChunk(start, end);
  const auto line = FormatChunkFailure(start, end, kInvalidDecoderExp255);
  std::cout << "TRACE-FIELDS:" << line.substr(line.find(" bo_pages_sampled="))
            << "\n";
  for (const char* part :
       {" bo_pages_sampled=8", " bo_pages_valid_begin=8",
        " bo_pages_valid_end=5", " bo_pages_shared_end=0", " bo_pages_locked=0",
        " bo_pages_large=0", " process_ws_mib_begin=5200",
        " process_ws_mib_end=4100", " process_private_mib_end=6150",
        " mem_trace=on", " gpu_proc_local_mib_min=na",
        " gpu_adapter_shared_mib_max=na"})
    assert(Has(line, part));
  const auto suffix = TakeRequestTelemetrySuffix();
  std::cout << "TRACE-REQUEST:"
            << suffix.substr(suffix.find(" bo_pages_sampled=")) << "\n";
  for (const char* part :
       {" bo_pages_sampled=8", " bo_pages_valid_min=5",
        " bo_pages_locked_max=0", " process_ws_mib_first=5200",
        " process_ws_mib_min=4100", " process_ws_mib_last=4100",
        " process_private_mib_max=6150", " mem_trace=on"})
    assert(Has(suffix, part));
  assert(log.empty());
  StopGlobalMemProbe();
  SetMemTraceLogForTest({});
}

void FailOpen() {
  std::vector<std::string> log;
  SetMemTraceLogForTest([&](const std::string& t) { log.push_back(t); });
  auto probe = std::make_unique<FakeProbe>();
  probe->fail = true;
  auto* raw = probe.get();
  InstallGlobalMemProbe(std::move(probe));
  Stats().chunks = 0;
  NoteBegin(0, true);
  for (int i = 0; i < 3; ++i) {
    const auto start = BeginChunkTelemetry(256);
    const auto end = EndChunkTelemetry(start);
    const auto line = FormatChunkFailure(start, end, 0);
    for (const char* part :
         {" bo_pages_sampled=na", " bo_pages_valid_begin=na",
          " bo_pages_valid_end=na", " process_ws_mib_begin=na",
          " process_private_mib_end=na", " mem_trace=na"})
      assert(Has(line, part));
    NoteRequestChunk(start, end);
  }
  assert(raw->calls == 6);  // still tried; failures change nothing else
  assert(log.size() == 1 && Has(log[0], "memory trace unavailable"));
  const auto suffix = TakeRequestTelemetrySuffix();
  assert(Has(suffix, " bo_pages_valid_min=na") && Has(suffix, " mem_trace=na"));
  (void)suffix;
  StopGlobalMemProbe();
  SetMemTraceLogForTest({});
}

void OffAndThrowing() {
  // No probe installed (switch off or test mode): fields read na / off.
  assert(GlobalMemProbe() == nullptr);
  const auto start = BeginChunkTelemetry(256);
  const auto end = EndChunkTelemetry(start);
  const auto line = FormatChunkFailure(start, end, 0);
  assert(Has(line, " bo_pages_sampled=na") && Has(line, " mem_trace=off"));
  // A probe that throws never reaches the caller.
  struct Throwing final : IMemProbe {
    bool Sample(MemTraceSample*, std::string*) override { throw 7; }
  };
  SetMemTraceLogForTest([](const std::string&) {});
  InstallGlobalMemProbe(std::make_unique<Throwing>());
  const auto s2 = BeginChunkTelemetry(256);
  const auto e2 = EndChunkTelemetry(s2);
  assert(Has(FormatChunkFailure(s2, e2, 0), " mem_trace=na"));
  StopGlobalMemProbe();
  SetMemTraceLogForTest({});
}

struct NoPower final : IChipPowerSource {
  bool Open(std::string* error) override {
    *error = "RAPL package counters missing";
    return false;
  }
  bool Read(PowerSample*, std::string*) override { return false; }
};

void GpuCounters() {
  {  // min/max over the chunk from the 1 s points, global path
    auto gpu = std::make_unique<FakeGpu>();
    gpu->script = {Gpu(900, 500, 400, 700), Gpu(1200, 800, 410, 900),
                   Gpu(1100, 700, 390, 800)};
    InstallGlobalChipPower(std::make_unique<NoPower>(), true, nullptr,
                           std::move(gpu), /*with_thread=*/false);
    assert(Faults().gpu_mem_state == 1);
    Stats().chunks = 0;
    NoteBegin(0, true);
    const auto start = BeginChunkTelemetry(4096);
    GlobalChipPower()->Tick();
    GlobalChipPower()->Tick();
    const auto end = EndChunkTelemetry(start);
    NoteRequestChunk(start, end);
    const auto line = FormatChunkFailure(start, end, kInvalidDecoderExp255);
    std::cout << "GPU-FIELDS:"
              << line.substr(line.find(" gpu_proc_local_mib_min=")) << "\n";
    for (const char* part :
         {" gpu_proc_local_mib_min=1100", " gpu_proc_local_mib_max=1200",
          " gpu_proc_dedicated_mib_min=700", " gpu_proc_dedicated_mib_max=800",
          " gpu_proc_shared_mib_min=390", " gpu_proc_shared_mib_max=410",
          " gpu_adapter_shared_mib_max=900", " gpu_mem=on"})
      assert(Has(line, part));
    const auto suffix = TakeRequestTelemetrySuffix();
    assert(Has(suffix, " gpu_proc_local_mib_min=1100") &&
           Has(suffix, " gpu_proc_local_mib_max=1200") &&
           Has(suffix, " gpu_adapter_shared_mib_max=900"));
    StopGlobalChipPower();
    Faults().gpu_mem_state = 0;
    Faults().chip_power_state = 0;
  }
  {  // counter set absent: unavailable, logged once, memory points keep going
    FaultState state;
    std::vector<std::pair<bool, std::string>> log;
    auto gpu = std::make_unique<FakeGpu>();
    gpu->open_ok = false;
    ChipPowerMonitor monitor(
        std::make_unique<NoPower>(), state,
        [&](bool warn, const std::string& t) { log.emplace_back(warn, t); });
    monitor.SetGpuMemSource(std::move(gpu));
    monitor.Start(true, false);
    monitor.Tick();
    monitor.Tick();
    int gpu_lines = 0;
    for (const auto& [warn, text] : log)
      if (Has(text, "GPU memory counters unavailable"))
        ++gpu_lines;
    assert(gpu_lines == 1 && state.gpu_mem_state == 3);
    assert(monitor.MemWindow(0, INT64_MAX).ok &&
           !monitor.MemWindow(0, INT64_MAX).gpu_ok);
  }
  {  // no process instances yet (empty error): stays on, values appear later
    FaultState state;
    auto gpu = std::make_unique<FakeGpu>();
    GpuMemPoint none;  // proc_ok = adapter_ok = false
    gpu->script = {none, none, Gpu(300, 100, 50, 60)};
    ChipPowerMonitor monitor(std::make_unique<NoPower>(), state, {});
    monitor.SetGpuMemSource(std::move(gpu));
    monitor.Start(true, false);  // priming point: none
    monitor.Tick();              // none
    assert(state.gpu_mem_state == 1);
    assert(!monitor.MemWindow(0, INT64_MAX).gpu_ok);
    monitor.Tick();  // the instance appeared
    const auto w = monitor.MemWindow(0, INT64_MAX);
    assert(w.gpu_ok && w.gpu_local_min == 300 &&
           w.gpu_adapter_shared_max == 60);
    assert(state.gpu_mem_state == 1);
  }
  {  // a read error: unavailable once, later ticks do not read again
    FaultState state;
    std::vector<std::pair<bool, std::string>> log;
    auto gpu = std::make_unique<FakeGpu>();
    gpu->read_error = "PdhCollectQueryData failed";
    ChipPowerMonitor monitor(
        std::make_unique<NoPower>(), state,
        [&](bool warn, const std::string& t) { log.emplace_back(warn, t); });
    monitor.SetGpuMemSource(std::move(gpu));
    monitor.Start(true, false);
    monitor.Tick();
    monitor.Tick();
    int gpu_lines = 0;
    for (const auto& [warn, text] : log)
      if (Has(text, "GPU memory counters unavailable"))
        ++gpu_lines;
    assert(gpu_lines == 1 && state.gpu_mem_state == 3);
  }
}
}  // namespace

int main() {
  ResidencyDropIsReported();
  FailOpen();
  OffAndThrowing();
  GpuCounters();
  std::cout
      << "PASS memory trace: residency drop, fail-open, off, GPU counters\n";
}
