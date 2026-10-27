#include "src/cli/serve/npu_status_publisher.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
unsigned passed = 0;
void Check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
  ++passed;
  std::cout << "PASS " << message << '\n';
}
using namespace std::chrono_literals;
using bubo::cli::NpuStatusPublisher;
using bubo::cli::ServeLifecycle;
using bubo::cli::Value;
namespace npu = gufo::npu;

void Proof() {
  npu::ResetNpuRuntimeStatusForTest();
  ServeLifecycle lifecycle(false);
  Check(!lifecycle.Snapshot().find("npu_prefill_runtime"),
        "absent without publisher (NPU not built)");
  Value final;
  {
    NpuStatusPublisher publisher(lifecycle);
    Check(lifecycle.Snapshot()["npu_prefill_runtime"].dump() ==
              npu::ToJson(npu::CurrentNpuRuntimeStatus()).dump(),
          "current state published at startup");
    for (const auto state :
         {npu::NpuState::kOn, npu::NpuState::kOff, npu::NpuState::kTripped,
          npu::NpuState::kFaulted, npu::NpuState::kUnavailable}) {
      npu::NpuRuntimeStatus sample;
      sample.state = state;
      sample.reason = state == npu::NpuState::kTripped ? "power" : "fixture";
      sample.trips = 1;
      sample.hal_faults = 2;
      sample.whea_events = 3;
      sample.invalid_chunks = 4;
      sample.chip_power = {"ok", 12.5, 0.25};
      sample.memory = npu::NpuRuntimeStatus::Memory{1024, 75};
      sample.scheme = npu::NpuRuntimeStatus::Scheme{"fixture-guid", "fixture"};
      publisher.Publish(sample);
      std::cout << "STATUS "
                << lifecycle.Snapshot()["npu_prefill_runtime"].dump() << '\n';
      Check(lifecycle.Snapshot()["npu_prefill_runtime"].dump() ==
                npu::ToJson(sample).dump(),
            npu::NpuStateName(state));
    }
    npu::NoteNpuStartup(false, "fixture startup denial");
    Check(lifecycle.Snapshot()["npu_prefill_runtime"].member_str("reason") ==
              "fixture startup denial",
          "installed producer callback preserves startup reason");
    npu::NoteNpuStartup(true, "");
    Check(
        lifecycle.Snapshot()["npu_prefill_runtime"].member_str("state") == "on",
        "installed producer callback publishes armed state");
    npu::NoteLastRequest({12.5, 0.25, 1024, 75});
    npu::PublishNpuRuntimeStatus();
    Check(lifecycle.Snapshot()["npu_prefill_runtime"]["memory"].dump() ==
              npu::ToJson(npu::CurrentNpuRuntimeStatus())["memory"].dump(),
          "request-end publication preserves memory");

    // B3's contention boundary: a fake backend observation waits indefinitely
    // while status reads and independent NPU publications must keep answering.
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    auto blocked = std::async(std::launch::async, [&] {
      lifecycle.SetSampler([&] {
        entered.set_value();
        gate.wait();
        return Value::object();
      });
    });
    entered.get_future().wait();
    std::atomic<unsigned> emitted{0};
    std::jthread updates([&](std::stop_token stop) {
      while (!stop.stop_requested()) {
        npu::PublishNpuRuntimeStatus();
        ++emitted;
        std::this_thread::sleep_for(1ms);
      }
    });
    double worst_ms = 0;
    bool complete = true;
    for (unsigned i = 0; i < 100; ++i) {
      const auto start = std::chrono::steady_clock::now();
      auto status = lifecycle.Snapshot();
      worst_ms =
          std::max(worst_ms, std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count());
      complete &= status["npu_prefill_runtime"].member_str("state") == "on";
      std::this_thread::sleep_for(1ms);
    }
    updates.request_stop();
    updates.join();
    release.set_value();
    blocked.get();
    lifecycle.SetSampler({});
    std::cout << "100 status reads, " << emitted.load()
              << " producer publications, maximum status ms=" << worst_ms
              << '\n';
    Check(complete && emitted > 0 && worst_ms < 1000,
          "status stays fast while backend blocked and publisher fires");
    npu::NoteNpuFaulted();
    Check(lifecycle.Snapshot()["npu_prefill_runtime"].member_str("state") ==
              "faulted",
          "installed producer callback publishes fault");
    final = lifecycle.Snapshot()["npu_prefill_runtime"];
  }
  npu::NoteNpuStartup(false, "after shutdown");
  Check(lifecycle.Snapshot()["npu_prefill_runtime"].dump() == final.dump(),
        "publisher cleared at shutdown");
  try {
    NpuStatusPublisher publisher(lifecycle);
    throw std::runtime_error("fixture startup failure");
  } catch (const std::runtime_error&) {
  }
  final = lifecycle.Snapshot()["npu_prefill_runtime"];
  npu::NoteNpuStartup(true, "");
  Check(lifecycle.Snapshot()["npu_prefill_runtime"].dump() == final.dump(),
        "publisher cleared on exception");
}
}  // namespace

int main() {
  try {
    Proof();
    std::cout << passed << " passed, 0 failed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << passed << " passed, 1 failed: " << error.what() << '\n';
    return 1;
  }
}
