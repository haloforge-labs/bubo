// Windows implementation of the memory guard's OS readings; see the header.

#include "src/core/platform/memory_readings.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>

#include <algorithm>

#ifdef ENGINE_ENABLE_HIP
#include "src/core/platform/device_memory.hpp"
#endif

namespace gufo::platform {
namespace {

HANDLE LowMemoryHandle() {
  // Created once; the handle lives for the process.
  static HANDLE handle = CreateMemoryResourceNotification(LowMemoryResourceNotification);
  return handle;
}

class SystemMemorySource final : public MemoryReadingSource {
public:
  MemoryReadings Read() noexcept override {
    MemoryReadings r;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
      r.ram_ok = true;
      r.ram_available = status.ullAvailPhys;
      r.ram_total = status.ullTotalPhys;
      r.commit_available = status.ullAvailPageFile;
    }
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
      r.process_ram_private = counters.PrivateUsage;
#ifdef ENGINE_ENABLE_HIP
    const auto split = QueryDeviceMemorySplit();
    if (split.valid) {
      r.gpu_ok = true;
      r.gpu_local_free = split.local_budget > split.local_used ? split.local_budget - split.local_used : 0;
      r.gpu_nonlocal_free = split.nonlocal_budget > split.nonlocal_used ? split.nonlocal_budget - split.nonlocal_used : 0;
      r.gpu_budget_total = static_cast<std::uint64_t>(split.local_budget) + split.nonlocal_budget;
      if (split.dedicated != 0) {
        r.gpu_carve_known = true;
        const std::uint64_t carve_left = split.dedicated > split.local_used ? split.dedicated - split.local_used : 0;
        r.gpu_carve_free = std::min<std::uint64_t>(carve_left, r.gpu_local_free);
      }
    }
#endif
    if (const HANDLE low = LowMemoryHandle()) {
      BOOL state = FALSE;
      if (QueryMemoryResourceNotification(low, &state))
        r.low_memory_signal = state != FALSE;
    }
    return r;
  }
};

}  // namespace

std::unique_ptr<MemoryReadingSource> MakeSystemMemorySource() { return std::make_unique<SystemMemorySource>(); }

bool LowMemoryNotificationAvailable() { return LowMemoryHandle() != nullptr; }

}  // namespace gufo::platform
