// SPDX-License-Identifier: MIT
#pragma once
// Fake HAL fault source: pushes events into the monitor from the test (or from
// an optional delivery thread, to test that Stop() joins). Never touches the
// Windows event log.
#include <atomic>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/npu/hal_fault_monitor.hpp"

namespace gufo::npu::fake {
class HalSource final : public IHalFaultSource {
public:
  bool fail = false, with_thread = false, throw_on_start = false;
  std::string error = "access denied";
  std::atomic<int> starts{0}, stops{0};
  bool Start(std::function<void(HalFaultEvent)> new_sink,
             std::string* out_error) override {
    ++starts;
    if (throw_on_start)
      throw std::runtime_error("planted start failure");
    if (fail) {
      *out_error = error;
      return false;
    }
    sink = std::move(new_sink);
    if (with_thread) {
      stopping = false;
      thread = std::thread([this] {
        while (!stopping)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      });
    }
    return true;
  }
  void Stop() noexcept override {
    ++stops;
    stopping = true;
    if (thread.joinable())
      thread.join();
  }
  bool thread_running() const { return thread.joinable(); }
  void SetBrokenHandler(std::function<void(const std::string&)> h) override {
    broken = std::move(h);
  }
  bool Running() const noexcept override { return thread.joinable(); }
  void Break(const std::string& why) {
    if (broken)
      broken(why);
  }
  std::function<void(const std::string&)> broken;
  // Event shaped like the 10-13 18:01:41 fault: Device 0xC501, FaultReason 0x6.
  void EmitHal(const std::string& info = "0x2E05B000") {
    HalFaultEvent event;
    event.kind = "HAL";
    event.id = 15;
    event.provider = "Microsoft-Windows-HAL";
    event.event_time = "2026-10-13T22:01:41.123Z";
    event.reason = "0x6";
    event.info = info;
    event.device = "0xC501";
    sink(event);
  }
  void EmitWhea() {
    HalFaultEvent event;
    event.kind = "WHEA";
    event.id = 19;
    event.provider = "Microsoft-Windows-WHEA-Logger";
    event.event_time = "2026-10-14T12:14:08.960Z";
    sink(event);
  }
  std::function<void(HalFaultEvent)> sink;

private:
  std::atomic<bool> stopping{false};
  std::thread thread;
};
}  // namespace gufo::npu::fake
