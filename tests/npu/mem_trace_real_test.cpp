// SPDX-License-Identifier: MIT
// Windows only. Exercises the REAL memory probe on this process's own pages
// (VirtualAlloc, a trim of this process's working set, a VirtualLock on one
// page), measures its cost, and reads the PDH GPU memory counters once. It
// touches nothing outside this process, no NPU, no GPU.
#include <windows.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "src/npu/mem_trace.hpp"

using namespace gufo::npu;

int main() {
  std::cout << std::unitbuf;
  constexpr std::size_t kPage = 4096, kPages = 8;
  char* memory = static_cast<char*>(VirtualAlloc(
      nullptr, kPages * kPage * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
  assert(memory);
  std::vector<const void*> pages;
  for (std::size_t i = 0; i < kPages; ++i)
    pages.push_back(memory + i * 2 * kPage);  // eight distinct pages
  const auto touch = [&] {
    for (const void* p : pages)
      *const_cast<volatile char*>(static_cast<const char*>(p)) = 1;
  };
  touch();
  auto probe = MakeWindowsMemProbe([&] { return pages; });
  MemTraceSample before, locked, trimmed, after;
  std::string error;
  assert(probe->Sample(&before, &error));
  std::cout << "before : sampled=" << before.pages_sampled
            << " valid=" << before.valid << " shared=" << before.shared
            << " locked=" << before.locked << " large=" << before.large
            << " ws_mib=" << before.ws_mib
            << " private_mib=" << before.private_mib << "\n";
  assert(before.pages_sampled == kPages && before.valid == kPages);
  assert(before.ws_mib > 0);  // this tiny process has under 1 MiB private
  // Lock one page (own process only) and see the Locked bit.
  bool lock_ok = VirtualLock(const_cast<void*>(pages[0]), kPage) != 0;
  assert(probe->Sample(&locked, &error));
  std::cout << "locked : lock_ok=" << lock_ok << " valid=" << locked.valid
            << " locked=" << locked.locked << "\n";
  if (lock_ok)
    assert(locked.locked >= 1);
  // Trim this process's working set: the valid bit drops for the other pages.
  using EmptyFn = BOOL(WINAPI*)(HANDLE);
  auto empty = reinterpret_cast<EmptyFn>(
      GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32EmptyWorkingSet"));
  if (empty) {
    empty(GetCurrentProcess());
    assert(probe->Sample(&trimmed, &error));
    std::cout << "trimmed: valid=" << trimmed.valid
              << " locked=" << trimmed.locked << " ws_mib=" << trimmed.ws_mib
              << "\n";
    assert(trimmed.valid < before.valid);  // a drop is reported
    assert(trimmed.ws_mib <= before.ws_mib);
    assert(trimmed.locked <=
           trimmed.valid);  // invalid pages never count as locked
  }
  touch();
  assert(probe->Sample(&after, &error));
  std::cout << "touched: valid=" << after.valid << "\n";
  assert(after.valid == kPages);
  if (lock_ok)
    VirtualUnlock(const_cast<void*>(pages[0]), kPage);

  // Cost of the chunk-path work: one Sample() = GetProcessMemoryInfo +
  // QueryWorkingSetEx on 8 pages. A chunk calls it twice (begin and end).
  constexpr int kCalls = 5000;
  MemTraceSample sample;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i) {
    sample = MemTraceSample{};
    probe->Sample(&sample, &error);
  }
  const auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  kCalls;
  std::cout << "COST: real probe Sample() " << us << " us/call (8 pages), "
            << 2 * us << " us per chunk (begin + end)\n";

  // PDH GPU memory counters: read-only, sampler thread only in the server.
  auto gpu = MakeWindowsGpuMemSource();
  if (gpu && gpu->Open(&error)) {
    for (int i = 0; i < 2; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1100));
      GpuMemPoint point;
      const bool ok = gpu->Read(&point, &error);
      std::cout << "gpu counters: ok=" << ok << " proc_ok=" << point.proc_ok
                << " local=" << point.proc_local_mib
                << " dedicated=" << point.proc_dedicated_mib
                << " shared=" << point.proc_shared_mib
                << " adapter_ok=" << point.adapter_ok
                << " adapter_shared=" << point.adapter_shared_mib
                << " error=" << error << "\n";
    }
    gpu->Close();
  } else {
    std::cout << "gpu counters: Open failed: " << error << "\n";
  }
  std::cout << "PASS real memory probe: valid-bit drop after a trim, locked "
               "bit, cost "
            << 2 * us << " us per chunk\n";
  VirtualFree(memory, 0, MEM_RELEASE);
  return 0;
}
