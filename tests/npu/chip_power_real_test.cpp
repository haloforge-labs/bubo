// SPDX-License-Identifier: MIT
// Windows only, read only: opens the real PDH RAPL package counters, takes
// samples about 1.1 s apart, prints them with the power scheme, and checks
// the values are plausible. Skips (exit 0) when the counters do not exist.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "src/npu/chip_power.hpp"

using namespace gufo::npu;

int main() {
  auto source = MakeWindowsChipPowerSource();
  if (!source) {
    std::cout << "SKIP no Windows PDH source on this platform\n";
    return 0;
  }
  std::string error;
  if (!source->Open(&error)) {
    std::cout << "SKIP RAPL counters unavailable: " << error << "\n";
    return 0;
  }
  PowerSample previous;
  assert(source->Read(&previous, &error));
  double worst_us = 0;
  for (int i = 0; i < 4; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    PowerSample sample;
    const auto t0 = std::chrono::steady_clock::now();
    assert(source->Read(&sample, &error));
    const auto us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    worst_us = std::max(worst_us, us);
    const double seconds = (sample.ts_ns - previous.ts_ns) / 1e9;
    const double watts =
        (sample.energy_pwh - previous.energy_pwh) * kJoulesPerPwh / seconds;
    std::cout << "sample " << i << ": energy_pwh=" << sample.energy_pwh
              << " dt_s=" << seconds << " mean_w=" << watts << " read_us=" << us
              << "\n";
    assert(sample.energy_pwh >= previous.energy_pwh);
    assert(watts >= 0 && watts <= kMaxPlausibleWatts);
    previous = sample;
  }
  source->Close();
  auto pages = MakeWindowsPagesSource();
  if (pages && pages->Open(&error)) {
    double rate = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    const bool ok = pages->Read(&rate, &error);
    std::cout << "Memory Pages/sec: ok=" << ok << " value=" << rate << "\n";
    assert(ok && rate >= 0);
    pages->Close();
  } else {
    std::cout << "Memory Pages/sec unavailable: " << error << "\n";
  }
  const auto scheme = ReadPowerScheme();
  std::cout << FormatPowerScheme(scheme) << "\n";
  std::cout << "process cpu ms=" << ProcessCpuMs() << "\n";
  std::cout << "PASS real PDH chip power source (worst PDH read " << worst_us
            << " us, background thread only)\n";
  return 0;
}
