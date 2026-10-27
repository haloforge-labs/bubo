// SPDX-License-Identifier: MIT
// CPU only: the npu_prefill_runtime producer with a fake publisher.
#include "src/npu/runtime_status.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/npu/chip_power.hpp"
#include "src/npu/fault_telemetry.hpp"
#include "src/npu/request_stats.hpp"

using namespace gufo::npu;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
struct Collector {
  std::mutex mutex;
  std::vector<NpuRuntimeStatus> seen;
  void Install() {
    SetNpuRuntimeStatusPublisher([this](const NpuRuntimeStatus& status) {
      std::lock_guard lock(mutex);
      seen.push_back(status);
    });
  }
  std::size_t count() {
    std::lock_guard lock(mutex);
    return seen.size();
  }
  NpuRuntimeStatus last() {
    std::lock_guard lock(mutex);
    assert(!seen.empty());
    return seen.back();
  }
};

void JsonShape() {
  NpuRuntimeStatus status;
  status.state = NpuState::kOff;
  status.reason = "GUFO_ENABLE_NPU=OFF";
  status.chip_power.state = "unavailable";
  auto json = ToJson(status).dump();
  std::cout << "JSON-OFF: " << json << "\n";
  assert(Has(json, "\"state\":\"off\"") &&
         Has(json, "\"reason\":\"GUFO_ENABLE_NPU=OFF\""));
  assert(Has(json, "\"trips\":0") && Has(json, "\"hal_faults\":0") &&
         Has(json, "\"whea_events\":0") && Has(json, "\"invalid_chunks\":0"));
  assert(Has(json,
             "\"chip_power\":{\"state\":\"unavailable\",\"w_mean\":null,"
             "\"window_s\":0}"));
  assert(!Has(json, "memory") && !Has(json, "scheme"));  // absent optionals
  status.state = NpuState::kOn;
  status.reason.reset();
  status.chip_power = {"ok", 61.5, 12.25};
  status.memory = NpuRuntimeStatus::Memory{8123, 41.5};
  status.scheme = NpuRuntimeStatus::Scheme{
      "{6FECC5AE-F350-48A5-B669-B472CB895CCF}", "Performance"};
  status.hal_faults = 2;
  status.whea_events = 1;
  status.invalid_chunks = 1;
  json = ToJson(status).dump();
  std::cout << "JSON-ON: " << json << "\n";
  for (const char* part :
       {"\"state\":\"on\"", "\"reason\":null", "\"hal_faults\":2",
        "\"whea_events\":1", "\"invalid_chunks\":1",
        "\"chip_power\":{\"state\":\"ok\",\"w_mean\":61.5,\"window_s\":12.25}",
        "\"memory\":{\"min_available_mib\":8123,\"max_commit_pct\":41.5}",
        "\"scheme\":{\"guid\":\"{6FECC5AE-F350-48A5-B669-B472CB895CCF}\","
        "\"name\":\"Performance\"}"})
    assert(Has(json, part));
  // The type has all five states; nothing emits tripped yet.
  assert(std::string(NpuStateName(NpuState::kTripped)) == "tripped");
  assert(std::string(NpuStateName(NpuState::kFaulted)) == "faulted");
  assert(std::string(NpuStateName(NpuState::kUnavailable)) == "unavailable");
}

void NoPublisher() {
  ResetNpuRuntimeStatusForTest();
  PublishNpuRuntimeStatus();  // no-op, must not throw
  NoteNpuStartup(true, "");
  NoteNpuFaulted();
  (void)TakeRequestLine();
  Collector collector;
  assert(collector.count() == 0);
}

void Transitions() {
  ResetNpuRuntimeStatusForTest();
  Faults().hal_faults = 0;
  Faults().whea_events = 0;
  Collector c;
  c.Install();
  // Not armed at start-up: off with the reason; a fault before that is not
  // "faulted".
  NoteNpuFaulted();
  assert(c.count() == 0);
  NoteNpuStartup(false, "NPU startup failed.");
  assert(c.count() == 1 && c.last().state == NpuState::kOff &&
         c.last().reason == "NPU startup failed.");
  NoteNpuFaulted();  // never armed: stays off
  assert(c.count() == 1);
  // Armed.
  NoteNpuScheme("{381B4222-F694-41F0-9685-FF5BB260DF2E}", "Balanced");
  NoteNpuStartup(true, "");
  assert(c.count() == 2 && c.last().state == NpuState::kOn &&
         !c.last().reason && !c.last().memory);
  assert(c.last().scheme && c.last().scheme->name == "Balanced");
  // Request end with one NPU chunk: counters, power, memory.
  Faults().hal_faults = 3;
  Faults().whea_events = 1;
  Faults().chip_power_state = 1;
  Stats().chunks = 0;
  Stats().last_layer = -1;
  NoteBegin(0, true);
  const auto start = BeginChunkTelemetry(4096);
  auto end = EndChunkTelemetry(start);
  end.duration_ms = 2500;  // 2.5 s NPU-active
  end.power = {true, false, 61.5, 70, 2.5, 3};
  end.min_avail_mib = 7000;
  end.max_commit_pct = 55.5;
  NoteRequestChunk(start, end);
  const auto line = TakeRequestLine();  // publishes at request end
  assert(!line.empty());
  assert(c.count() == 3);
  auto s = c.last();
  assert(s.state == NpuState::kOn && s.hal_faults == 3 && s.whea_events == 1 &&
         s.invalid_chunks == 0 && s.trips == 0);
  assert(s.chip_power.state == "ok" && s.chip_power.w_mean &&
         *s.chip_power.w_mean > 61.4 && *s.chip_power.w_mean < 61.6);
  assert(s.chip_power.window_s > 2.49 && s.chip_power.window_s < 2.51);
  assert(s.memory && s.memory->min_available_mib == 7000 &&
         s.memory->max_commit_pct > 55.4 && s.memory->max_commit_pct < 55.6);
  // A request that never used the NPU still publishes (cheap, same values).
  (void)TakeRequestLine();
  assert(c.count() == 4);
  // Fault: sticky faulted with reason "fault", published once.
  Faults().invalid_chunks = 1;
  NoteNpuFaulted();
  NoteNpuFaulted();
  assert(c.count() == 5 && c.last().state == NpuState::kFaulted &&
         c.last().reason == "fault" && c.last().invalid_chunks == 1);
  (void)TakeRequestLine();
  assert(c.count() == 6 && c.last().state == NpuState::kFaulted);
  // Chip power off / unavailable are their own field, not NPU states.
  Faults().chip_power_state = 3;
  PublishNpuRuntimeStatus();
  assert(c.last().chip_power.state == "unavailable" &&
         c.last().state == NpuState::kFaulted);
  Faults().chip_power_state = 0;
  Faults().hal_faults = 0;
  Faults().whea_events = 0;
  Faults().invalid_chunks = 0;
}

void AssertContractKeys(const gufo::json::Value& status) {
  const std::set<std::string> top = {
      "state",          "reason",     "trips",  "hal_faults", "whea_events",
      "invalid_chunks", "chip_power", "memory", "scheme"};
  for (const auto& [key, value] : status.members()) {
    assert(top.count(key) && "unknown key in npu_prefill_runtime");
    if (key == "chip_power") {
      for (const auto& [k, v] : value.members())
        assert((k == "state" || k == "w_mean" || k == "window_s") &&
               "chip_power key");
    } else if (key == "memory") {
      for (const auto& [k, v] : value.members())
        assert((k == "min_available_mib" || k == "max_commit_pct") &&
               "memory key");
    } else if (key == "scheme") {
      for (const auto& [k, v] : value.members())
        assert((k == "guid" || k == "name") && "scheme key");
    }
  }
}

void GuardHold() {
  ResetNpuRuntimeStatusForTest();
  Collector c;
  c.Install();
  NoteNpuStartup(true, "");
  assert(c.count() == 1 && c.last().state == NpuState::kOn && !c.last().trip);
  // Power trip: tripped/power, trips, the tripping mean (no request has run:
  // last request == nullopt must not matter), the trip block.
  NpuHoldInfo power;
  power.held = true;
  power.reason = "power";
  power.trips = 1;
  power.w_mean = 84.7;
  NoteNpuHold(power);
  auto s = c.last();
  assert(c.count() == 2 && s.state == NpuState::kTripped &&
         s.reason == "power" && s.trips == 1);
  assert(s.chip_power.w_mean && *s.chip_power.w_mean > 84.6 &&
         *s.chip_power.w_mean < 84.8);
  assert(s.trip && !s.trip->sticky && s.trip->w_mean);
  assert(!s.memory);  // nothing to show for a power trip before any request
  const auto json = ToJson(s).dump();
  std::cout << "JSON-TRIPPED: " << json << "\n";
  assert(Has(json, "\"state\":\"tripped\"") &&
         Has(json, "\"reason\":\"power\"") && Has(json, "\"trips\":1") &&
         Has(json, "\"w_mean\":84.7"));
  // Only the contract's keys: no `trip`, nothing unknown.
  AssertContractKeys(ToJson(s));
  // A request ending while held must not replace the trip's own numbers.
  LastRequestSummary stale;
  stale.power_w_mean = 55.0;
  stale.window_s = 3;
  stale.min_available_mib = 9000;
  stale.max_commit_pct = 30;
  NoteLastRequest(stale);
  (void)TakeRequestLine();  // request end publishes
  s = c.last();
  assert(s.state == NpuState::kTripped && *s.chip_power.w_mean > 84.6);
  // Memory trip: the memory object shows the trip's window.
  NpuHoldInfo mem_off;  // re-arm first
  mem_off.held = false;
  mem_off.trips = 1;
  NoteNpuHold(mem_off);
  s = c.last();
  assert(s.state == NpuState::kOn && !s.trip && !s.reason && s.trips == 1);
  NpuHoldInfo memory;
  memory.held = true;
  memory.reason = "memory";
  memory.trips = 2;
  memory.min_available_mib = 700;
  memory.max_commit_pct = 98.6;
  NoteNpuHold(memory);
  s = c.last();
  assert(s.state == NpuState::kTripped && s.reason == "memory" && s.trips == 2);
  AssertContractKeys(ToJson(s));  // memory trip: still only contract keys
  assert(s.memory && s.memory->min_available_mib == 700 &&
         s.memory->max_commit_pct > 98.5);
  // Sticky is internal now: the status JSON has no trip block, trips counts.
  NpuHoldInfo sticky = memory;
  sticky.trips = 3;
  sticky.sticky = true;
  NoteNpuHold(sticky);
  assert(c.last().trip && c.last().trip->sticky && c.last().trips == 3);
  // Faulted always wins over tripped; trips keeps its count.
  NoteNpuFaulted();
  s = c.last();
  assert(s.state == NpuState::kFaulted && s.reason == "fault" && s.trips == 3);
  assert(!s.trip);
  NoteNpuHold(mem_off);  // a late re-arm cannot undo the fault
  assert(c.last().state == NpuState::kFaulted);
  // Ordering: concurrent state changes publish snapshots in the order they were
  // built, so the last delivered snapshot is the last state set.
  ResetNpuRuntimeStatusForTest();
  Collector d;
  d.Install();
  NoteNpuStartup(true, "");
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
    threads.emplace_back([t] {
      for (int i = 0; i < 300; ++i) {
        NpuHoldInfo h;
        h.held = (i + t) % 2 == 0;
        h.reason = "power";
        h.trips = 1;
        NoteNpuHold(h);
      }
    });
  for (auto& th : threads)
    th.join();
  NpuHoldInfo last;
  last.held = false;
  last.trips = 1;
  NoteNpuHold(last);
  assert(d.last().state == NpuState::kOn);
  // Delivered one at a time.
  ResetNpuRuntimeStatusForTest();
}

void ExceptionsSwallowed() {
  ResetNpuRuntimeStatusForTest();
  std::atomic<int> calls{0};
  SetNpuRuntimeStatusPublisher([&](const NpuRuntimeStatus&) {
    ++calls;
    throw std::runtime_error("consumer bug");
  });
  NoteNpuStartup(true, "");  // must not throw
  NoteNpuFaulted();          // must not throw
  (void)TakeRequestLine();   // must not throw
  PublishNpuRuntimeStatus();
  assert(calls >= 3);
  SetNpuRuntimeStatusPublisher([](const NpuRuntimeStatus&) { throw 42; });
  PublishNpuRuntimeStatus();  // non-std exception too
}

void ThreadSafety() {
  ResetNpuRuntimeStatusForTest();
  NoteNpuStartup(true, "");
  std::atomic<bool> stop{false};
  std::atomic<long> published{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i)
    workers.emplace_back([&] {
      while (!stop)
        PublishNpuRuntimeStatus();
    });
  std::thread toggler([&] {
    for (int i = 0; i < 2000; ++i) {
      SetNpuRuntimeStatusPublisher([&](const NpuRuntimeStatus& s) {
        published.fetch_add(1);
        (void)s.state;
      });
      if (i % 3 == 0)
        SetNpuRuntimeStatusPublisher({});
    }
  });
  toggler.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  stop = true;
  for (auto& w : workers)
    w.join();
  SetNpuRuntimeStatusPublisher({});
  PublishNpuRuntimeStatus();
  assert(published > 0);
}

void Cost() {
  ResetNpuRuntimeStatusForTest();
  NoteNpuStartup(true, "");
  constexpr int kCalls = 1'000'000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i)
    PublishNpuRuntimeStatus();
  const auto none_ns = std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - t0)
                           .count() /
                       kCalls;
  std::atomic<long> n{0};
  SetNpuRuntimeStatusPublisher([&](const NpuRuntimeStatus&) {
    n.fetch_add(1, std::memory_order_relaxed);
  });
  constexpr int kWith = 100000;
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < kWith; ++i)
    PublishNpuRuntimeStatus();
  const auto with_us = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - t1)
                           .count() /
                       kWith;
  std::cout << "COST: publish without publisher " << none_ns
            << " ns; with a trivial publisher " << with_us
            << " us (snapshot build + call), once per request\n";
  SetNpuRuntimeStatusPublisher({});
}
}  // namespace

int main() {
  JsonShape();
  NoPublisher();
  Transitions();
  GuardHold();
  ExceptionsSwallowed();
  ThreadSafety();
  Cost();
  ResetNpuRuntimeStatusForTest();
  std::cout << "PASS npu runtime status: json shape, no publisher, "
               "transitions, request end, exceptions, thread safety\n";
}
