// SPDX-License-Identifier: MIT
// CPU only: no GPU, no NPU, no Windows event log. Covers the invalid-word cause
// bits, failure/request line formatting, the HAL event XML parser and the HAL
// fault monitor with a fake source (baseline, event during a chunk, event with
// no Bubo failure, subscription failure, switch off, clean shutdown).
#include "src/npu/hal_fault_monitor.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "fake_hal_source.hpp"
#include "src/npu/bo_address_table.hpp"
#include "src/npu/fault_telemetry.hpp"
#include "src/npu/request_stats.hpp"

using namespace gufo::npu;
using std::chrono::system_clock;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
struct FakeMemory final : IMemorySource {
  MemorySample sample{true, 2801, 41000};
  MemorySample Sample() override { return sample; }
};
struct Captured {
  std::vector<std::pair<bool, std::string>> lines;
  HalFaultMonitor::Log fn() {
    return [this](bool warn, const std::string& text) {
      lines.emplace_back(warn, text);
    };
  }
};

void CauseBits() {
  assert(DescribeInvalidBits(0) == "none");
  assert(DescribeInvalidBits(kInvalidEncoderNonFinite) == "A-nonfinite");
  assert(DescribeInvalidBits(kInvalidEncoderSaturated) == "A-saturated");
  assert(DescribeInvalidBits(kInvalidDecoderExp255) == "C-exp255");
  assert(DescribeInvalidBits(kInvalidDecoderExp254Mant128) ==
         "C-exp254-mant128");
  assert(DescribeInvalidBits(kInvalidEncoderNonFinite |
                             kInvalidDecoderExp255) == "A-nonfinite|C-exp255");
  assert(DescribeInvalidBits(1U << 9) == "unknown-0x200");
  // Non-zero still means invalid: every cause bit is a distinct non-zero bit.
  assert((kInvalidEncoderNonFinite | kInvalidEncoderSaturated |
          kInvalidDecoderExp255 | kInvalidDecoderExp254Mant128) == 0xFU);
}

void Stamps() {
  // 2026-10-13T22:01:41.123Z
  const auto t = system_clock::time_point(std::chrono::seconds(1791928901) +
                                          std::chrono::milliseconds(123));
  const auto stamp = FormatWallStamp(t);
  assert(stamp.utc == "2026-10-13T22:01:41.123Z");
  assert(Has(stamp.local, ".123") && stamp.local.size() == 23);
}

void XmlParser() {
  const std::string hal =
      "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'>"
      "<System><Provider Name='Microsoft-Windows-HAL' "
      "Guid='{de9b2c6b-0000-0000-0000-000000000000}'/><EventID>15</EventID>"
      "<TimeCreated SystemTime='2026-10-13T22:01:41.123456700Z'/>"
      "<Channel>System</Channel></System><EventData>"
      "<Data Name='DeviceId'>0xC501</Data><Data Name='FaultReason'>0x6</Data>"
      "<Data Name='FaultInformation'>0x2E05B000</Data></EventData></Event>";
  HalFaultEvent event;
  assert(ParseEventXml(hal, &event));
  assert(event.id == 15 && event.provider == "Microsoft-Windows-HAL");
  assert(event.event_time == "2026-10-13T22:01:41.123456700Z");
  assert(event.reason == "0x6" && event.info == "0x2E05B000" &&
         event.device == "0xC501");
  assert(Has(event.data, "FaultReason=0x6"));
  // Real WHEA-Logger Id 19 record (this laptop, 2026-10-14 08:14, rendered by
  // EvtSubscribe in the read-only probe; RawData shortened).
  const std::string whea =
      "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'>"
      "<System><Provider Name='Microsoft-Windows-WHEA-Logger' "
      "Guid='{c26c4f3c-3f66-4e99-8f8a-39405cfed220}'/><EventID>19</EventID>"
      "<Version>0</Version><Level>3</Level><Task>0</Task><Opcode>0</Opcode>"
      "<Keywords>0x8000000000000000</Keywords>"
      "<TimeCreated SystemTime='2026-10-14T12:14:08.9608154Z'/>"
      "<EventRecordID>53207</EventRecordID><Channel>System</Channel>"
      "<Computer>bubo</Computer></System><EventData>"
      "<Data Name='ErrorSource'>1</Data><Data Name='MciStat'>0x982010000004080b"
      "</Data><Data Name='ErrorType'>10</Data><Data "
      "Name='RawData'>43504552</Data>"
      "</EventData></Event>";
  HalFaultEvent corrected;
  assert(ParseEventXml(whea, &corrected) && corrected.id == 19);
  assert(corrected.provider == "Microsoft-Windows-WHEA-Logger" &&
         corrected.event_time == "2026-10-14T12:14:08.9608154Z");
  assert(Has(corrected.data, "ErrorType=10") && corrected.reason.empty());
  // Unknown field names: still parsed, kept in data, typed fields stay empty.
  const std::string unnamed =
      "<Event><System><Provider Name=\"Microsoft-Windows-HAL\"/>"
      "<EventID Qualifiers=\"0\">15</EventID>"
      "<TimeCreated SystemTime=\"t\"/></System>"
      "<EventData><Data>0xC501</Data><Data>a &amp; "
      "b</Data></EventData></Event>";
  HalFaultEvent other;
  assert(ParseEventXml(unnamed, &other) && other.id == 15);
  assert(other.reason.empty() && Has(other.data, "item1=a & b"));
  HalFaultEvent bad;
  assert(!ParseEventXml("<Event><System></System></Event>", &bad));
  assert(!ParseEventXml("", &bad));
  const std::string line = FormatHalEvent([&] {
    auto e = event;
    e.kind = "HAL";
    e.received_utc = "r";
    e.seq = 3;
    return e;
  }());
  assert(Has(line,
             "HAL-fault event: 2026-10-13T22:01:41.123456700Z id=15 "
             "reason=0x6 info=0x2E05B000 device=0xC501"));
}

void RateLimitAndBroken() {
  {  // at most 20 log lines per minute; the next window says how many were
     // skipped
    FaultState state;
    Captured log;
    std::int64_t now = 1000;
    auto owned = std::make_unique<fake::HalSource>();
    auto* source = owned.get();
    HalFaultMonitor monitor(std::move(owned), state, log.fn(),
                            [&now] { return now; });
    assert(monitor.Start(true));
    const auto started = log.lines.size();
    for (int i = 0; i < 100; ++i)
      source->EmitHal("0x" + std::to_string(i));
    assert(log.lines.size() - started == HalFaultMonitor::kMaxLogPerMinute);
    assert(state.hal_faults == 100 &&
           monitor.Sequence() == 100);  // all counted
    now += 61000;
    source->EmitHal();
    bool saw = false;
    for (const auto& [warn, text] : log.lines)
      if (Has(text, "suppressed 80 HAL/WHEA events"))
        saw = true;
    assert(saw && log.lines.size() - started == 22);
  }
  {  // the source thread ending abnormally: logged once, monitor=broken
    FaultState state;
    Captured log;
    auto owned = std::make_unique<fake::HalSource>();
    auto* source = owned.get();
    HalFaultMonitor monitor(std::move(owned), state, log.fn());
    assert(monitor.Start(true) && state.monitor_state == 1);
    source->Break("EvtNext failed, error 5");
    source->Break("again");
    assert(monitor.broken() && state.monitor_state == 2);
    int broken_lines = 0;
    for (const auto& [warn, text] : log.lines)
      if (Has(text, "monitor=broken"))
        ++broken_lines;
    assert(broken_lines == 1);
  }
  {  // a throwing source never escapes Start (detection only)
    FaultState state;
    Captured log;
    auto owned = std::make_unique<fake::HalSource>();
    owned->throw_on_start = true;
    HalFaultMonitor monitor(std::move(owned), state, log.fn());
    assert(!monitor.Start(true) && monitor.start_failed() &&
           state.monitor_state == 3);
  }
}

void MonitorBasics() {
  FaultState state;
  Captured log;
  auto owned = std::make_unique<fake::HalSource>();
  auto* source = owned.get();
  HalFaultMonitor monitor(std::move(owned), state, log.fn());
  // No events: nothing stored, sequence stays at the baseline.
  assert(monitor.Start(true) && monitor.running() && source->starts == 1);
  assert(monitor.Sequence() == 0 && monitor.baseline() == 0);
  assert(monitor.EventsSince(0).empty());
  // One HAL event: sequence, counter, ring and immediate log line.
  source->EmitHal();
  assert(monitor.Sequence() == 1 && state.hal_faults == 1 &&
         state.whea_events == 0);
  const auto events = monitor.EventsSince(0);
  assert(events.size() == 1 && events[0].seq == 1 && events[0].reason == "0x6");
  assert(!events[0].received_utc.empty());
  bool saw = false;
  for (const auto& [warn, text] : log.lines)
    if (warn && Has(text,
                    "HAL-fault event: 2026-10-13T22:01:41.123Z id=15 "
                    "reason=0x6 info=0x2E05B000 device=0xC501"))
      saw = true;
  assert(saw);
  // WHEA events are counted and stored, but never move the HAL sequence.
  source->EmitWhea();
  assert(monitor.Sequence() == 1 && state.hal_faults == 1 &&
         state.whea_events == 1 && monitor.ring_sequence() == 2);
  assert(monitor.EventsSince(1).size() == 1 &&
         monitor.EventsSince(1)[0].kind == "WHEA");
  // Ring keeps the last kRing entries; older ones are gone, newer ones intact.
  for (int i = 0; i < 100; ++i)
    source->EmitHal("0x" + std::to_string(i));
  assert(monitor.Sequence() == 101 && monitor.ring_sequence() == 102);
  const auto tail = monitor.EventsSince(0);
  assert(tail.size() == HalFaultMonitor::kRing && tail.front().seq == 39 &&
         tail.back().seq == 102 && tail.back().info == "0x99");
  monitor.Stop();
  assert(source->stops >= 1 && !monitor.running());
}

void MonitorBaseline() {
  FaultState state;
  state.hal_seq = 7;  // HAL count before Start; the monitor only adds to it
  Captured log;
  auto owned = std::make_unique<fake::HalSource>();
  auto* source = owned.get();
  HalFaultMonitor monitor(std::move(owned), state, log.fn());
  assert(monitor.Start(true) && monitor.baseline() == 0);
  source->EmitHal();
  assert(monitor.Sequence() == 8 &&
         monitor.EventsSince(monitor.baseline()).size() == 1);
}

void MonitorFailureAndSwitch() {
  {  // subscription failure: silent no-op, logged once, never throws
    FaultState state;
    Captured log;
    auto owned = std::make_unique<fake::HalSource>();
    owned->fail = true;
    auto* source = owned.get();
    HalFaultMonitor monitor(std::move(owned), state, log.fn());
    assert(!monitor.Start(true) && monitor.start_failed() &&
           !monitor.running());
    assert(log.lines.size() >= 1 && !log.lines[0].first &&
           Has(log.lines[0].second, "unavailable: access denied"));
    assert(monitor.Sequence() == 0 && state.hal_faults == 0);
    (void)source;
  }
  {  // GUFO_NPU_HAL_MONITOR=0: the source is never started, no thread
    FaultState state;
    Captured log;
    auto owned = std::make_unique<fake::HalSource>();
    owned->with_thread = true;
    auto* source = owned.get();
    HalFaultMonitor monitor(std::move(owned), state, log.fn());
    assert(!monitor.Start(false) && !monitor.running());
    assert(source->starts == 0 && !source->thread_running() &&
           log.lines.empty());
  }
  {  // shutdown joins the delivery thread (also from the destructor)
    FaultState state;
    Captured log;
    auto owned = std::make_unique<fake::HalSource>();
    owned->with_thread = true;
    auto* source = owned.get();
    {
      HalFaultMonitor monitor(std::move(owned), state, log.fn());
      assert(monitor.Start(true) && source->thread_running());
      monitor.Stop();
      assert(!source->thread_running() && !monitor.running());
      assert(monitor.Start(true) && source->thread_running());
    }  // destructor stops and joins again; a hang here fails the 30 s timeout
    assert(!source->thread_running());
  }
}

void ChunkContext() {
  FakeMemory memory;
  auto* previous = SetMemorySourceForTest(&memory);
  SetWallClockForTest([] {
    return system_clock::time_point(std::chrono::seconds(1791928901) +
                                    std::chrono::milliseconds(123));
  });
  Captured log;
  auto owned = std::make_unique<fake::HalSource>();
  auto* source = owned.get();
  HalFaultMonitor monitor(std::move(owned), Faults(), log.fn());
  assert(monitor.Start(true));
  const auto before_total = Faults().hal_faults.load();
  Stats().chunks = 0;
  Stats().last_layer = -1;
  NoteBegin(0, true);  // first NPU begin of the request: chunk 1
  NoteRealDispatch();
  const auto start = BeginChunkTelemetry(4096);
  assert(start.tokens == 4096 && start.idle_gap_ms >= 0 && start.memory.ok &&
         start.memory.available_mib == 2801);
  Stats().finish_calls = Stats().finish_calls + 84;
  Stats().linear_begins = Stats().linear_begins + 36;
  Stats().projection_begins = Stats().projection_begins + 48;
  Stats().hostfunc_lat_ns = Stats().hostfunc_lat_ns + 4609400000ULL;
  memory.sample = {true, 2790, 41010};
  source->EmitHal();  // a fault while the chunk runs
  source->EmitHal("0x2DEF0000");
  const auto end = EndChunkTelemetry(start);
  assert(end.hal_new == 2 && end.index >= 1);
  NoteRequestChunk(start, end);
  const auto failure = FormatChunkFailure(start, end, kInvalidDecoderExp255);
  std::cout << "FAILURE-FIELDS: " << failure << "\n";
  for (const char* part :
       {"cause=0x4(C-exp255)", "tokens=4096",
        "start_utc=2026-10-13T22:01:41.123Z",
        "end_utc=2026-10-13T22:01:41.123Z",
        "idle_gap_ms=", "avail_mib_start=2801", "avail_mib_end=2790",
        "commit_mib_start=41000", "commit_mib_end=41010", "linear_begins=36",
        "projection_begins=48", "finish_calls=84", "failures=0",
        "hostfunc_lat_ms=4609.4", "hal_in_chunk=2"})
    assert(Has(failure, part));
  assert(Faults().hal_faults.load() == before_total + 2);
  assert(Has(failure, "hal_faults_total=" + std::to_string(before_total + 2)));
  // Request line: appended fields, hal_faults is the delta since the last line.
  const auto suffix = TakeRequestTelemetrySuffix();
  std::cout << "REQUEST-SUFFIX:" << suffix << "\n";
  for (const char* part :
       {" t_first_utc=2026-10-13T22:01:41.123Z", " t_first_local=",
        " t_last_end_utc=", " idle_gap_ms=", " avail_mib_first=2801",
        " avail_mib_last=2790", " commit_mib_last=41010", " hal_faults=2",
        " whea_events=0"})
    assert(Has(suffix, part));
  const auto again = TakeRequestTelemetrySuffix();
  assert(again == " hal_faults=0 whea_events=0 monitor=on chip_power=off");
  assert(Has(failure, " monitor=on"));
  // A WHEA event during a chunk is NOT a HAL event: hal_new stays 0.
  const auto whea_start = BeginChunkTelemetry(256);
  source->EmitWhea();
  const auto whea_end = EndChunkTelemetry(whea_start);
  assert(whea_end.hal_new == 0);
  NoteRequestChunk(whea_start, whea_end);
  assert(Has(TakeRequestTelemetrySuffix(), " hal_faults=0 whea_events=1"));
  // Event with no Bubo failure: the chunk code sees it from the sequence alone.
  const auto quiet = BeginChunkTelemetry(256);
  source->EmitHal();
  const auto quiet_end = EndChunkTelemetry(quiet);
  assert(quiet_end.hal_new == 1);
  // And no event: the sequence does not move.
  const auto idle = BeginChunkTelemetry(256);
  assert(EndChunkTelemetry(idle).hal_new == 0);
  monitor.Stop();
  SetWallClockForTest({});
  SetMemorySourceForTest(previous);
}

void FaultResolution() {
  // The HAL fault line names the buffer the FaultInformation address hit.
  ResetBoRegistryForTest();
  ResetHalResolutionForTest();
  BoTable bo;
  bo.entries = {{"weight", 0, 0x20000, 0x2DEE0000ULL, true},
                {"weight", 1, 0x20000, 0x2DF00000ULL, true},
                {"output", 0, 0x1000, 0x2E166000ULL, true}};
  PublishBoTable(std::make_shared<const BoTable>(bo));
  FaultState state;
  Captured log;
  auto owned = std::make_unique<fake::HalSource>();
  auto* source = owned.get();
  HalFaultMonitor monitor(std::move(owned), state, log.fn());
  assert(monitor.Start(true));
  const auto start = BeginChunkTelemetry(4096);
  source->EmitHal("0x2DEF0000");  // inside weight#0 (hex, as in the 10-13 log)
  source->EmitHal("773218304");  // 0x2E166000 = output#0 (decimal, as in 10-15)
  source->EmitHal("0x40000000");  // nowhere
  source->EmitHal("");            // no FaultInformation
  const auto events = monitor.EventsSince(0);
  assert(events.size() == 4);
  assert(events[0].resolves == "resolves=weight#0+0x10000");
  assert(events[1].resolves == "resolves=output#0+0x0");
  assert(Has(events[2].resolves, "resolves=none nearest=output#0 dist=0x"));
  assert(events[3].resolves == "resolves=na");
  bool saw = false;
  for (const auto& [warn, text] : log.lines)
    if (Has(text, "HAL-fault event:") &&
        Has(text, "info=0x2DEF0000 device=0xC501 recv=") &&
        Has(text, " resolves=weight#0+0x10000"))
      saw = true;
  assert(saw);
  // The NPU failure line carries the latest resolution when a fault arrived
  // during the chunk (and for 3 s afterwards).
  const auto end = EndChunkTelemetry(start);
  const auto line = FormatChunkFailure(start, end, kInvalidDecoderExp255);
  assert(Has(line, " hal_resolves=na"));  // the last event had no address
  source->EmitHal("0x2DEF0000");
  const auto start2 = BeginChunkTelemetry(256);
  const auto line2 = FormatChunkFailure(start2, EndChunkTelemetry(start2), 0);
  assert(Has(line2, " hal_resolves=weight#0+0x10000"));  // within 3 s
  // A WHEA event has no resolution and does not touch the slot.
  source->EmitWhea();
  assert(
      monitor.EventsSince(monitor.ring_sequence() - 1).back().resolves.empty());
  // No table published: resolves=na (fail-open).
  ResetBoRegistryForTest();
  source->EmitHal("0x2DEF0000");
  assert(monitor.EventsSince(monitor.ring_sequence() - 1).back().resolves ==
         "resolves=na");
  monitor.Stop();
  ResetHalResolutionForTest();
  ResetBoRegistryForTest();
}

void Cost() {
  // The chunk-side HAL cost is one acquire load of Faults().hal_seq.
  constexpr int kIterations = 100'000'000;
  std::uint64_t sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i)
    sink += Faults().hal_seq.load(std::memory_order_acquire) + i;
  const auto ns = std::chrono::duration<double, std::nano>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  kIterations;
  // Total per-chunk telemetry: real clock, GlobalMemoryStatusEx, time
  // formatting and vsnprintf: Begin + End + NoteRequestChunk, then the failure
  // line formatting (only built on failure / HAL event).
  const auto t1 = std::chrono::steady_clock::now();
  constexpr int kChunks = 20000;
  for (int i = 0; i < kChunks; ++i) {
    const auto start = BeginChunkTelemetry(4096);
    const auto end = EndChunkTelemetry(start);
    NoteRequestChunk(start, end);
    sink += end.index;
  }
  const auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t1)
                      .count() /
                  kChunks;
  const auto start = BeginChunkTelemetry(4096);
  const auto end = EndChunkTelemetry(start);
  const auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < kChunks; ++i)
    sink += FormatChunkFailure(start, end, kInvalidDecoderExp255).size();
  const auto format_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - t2)
                             .count() /
                         kChunks;
  (void)TakeRequestTelemetrySuffix();
  std::cout << "COST: hal_seq acquire load " << ns
            << " ns/read (loop incl.); per-chunk telemetry (Begin + End + "
               "NoteRequestChunk, real clock and memory sample) "
            << us << " us/chunk; FormatChunkFailure " << format_us
            << " us (failure path only); sink=" << (sink & 1) << "\n";
  assert(us < 50.0);  // plan limit: more than ~50 us means make it lazy
}
}  // namespace

int main() {
  CauseBits();
  Stamps();
  XmlParser();
  RateLimitAndBroken();
  MonitorBasics();
  MonitorBaseline();
  MonitorFailureAndSwitch();
  ChunkContext();
  FaultResolution();
  Cost();
  std::cout << "PASS cause bits, stamps, XML parser, HAL monitor with fake "
               "source, chunk context\n";
}
