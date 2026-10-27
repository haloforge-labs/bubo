// SPDX-License-Identifier: MIT
// Windows only. Creates the REAL event-log source, starts it, stops it and
// checks the stop is prompt and the thread is gone. The subscription is read
// only (future events of the System channel); no event is needed and none is
// written. Skips (exit 0) when wevtapi cannot subscribe.
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

#include "src/npu/hal_fault_monitor.hpp"

using namespace gufo::npu;

int main() {
  auto source = MakeWindowsHalFaultSource();
  if (!source) {
    std::cout << "SKIP no Windows source on this platform\n";
    return 0;
  }
  std::string error;
  if (!source->Start([](HalFaultEvent) {}, &error)) {
    std::cout << "SKIP wevtapi cannot subscribe: " << error << "\n";
    return 0;
  }
  assert(source->Running());
  const auto t0 = std::chrono::steady_clock::now();
  source->Stop();
  const auto elapsed = std::chrono::steady_clock::now() - t0;
  assert(elapsed < std::chrono::seconds(5));
  assert(!source->Running());
  source->Stop();  // idempotent
  // The monitor around the real source: start, stop, destroy without a hang.
  FaultState state;
  {
    HalFaultMonitor monitor(MakeWindowsHalFaultSource(), state,
                            [](bool, const std::string&) {});
    if (monitor.Start(true))
      assert(state.monitor_state == 1);
    monitor.Stop();
  }
  std::cout
      << "PASS real Windows HAL source: start, prompt stop ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms), thread gone\n";
  return 0;
}
