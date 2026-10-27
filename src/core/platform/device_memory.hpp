#ifndef GUFO_CORE_PLATFORM_DEVICE_MEMORY_HPP_
#define GUFO_CORE_PLATFORM_DEVICE_MEMORY_HPP_

// hipMemGetInfo with correct numbers on Windows.
//
// On a Windows UMA APU the HIP runtime double-counts every allocation beyond
// the BIOS dedicated carve-out: with 77 GiB allocated on a 128 GB Strix Halo
// (64 GB carve-out) it reports 8.6 GiB free while WDDM's budget still has
// 33.6 GiB, and "free" reaches zero at 82 GiB although allocations succeed
// well past 100 GiB. Gufo sizes its session state from free memory, so on
// Windows the answer comes from the process's WDDM video-memory budget
// (DXGI QueryVideoMemoryInfo, adapter matched by the HIP device LUID).

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::platform {

#ifdef _WIN32
hipError_t DeviceMemoryInfo(std::size_t* free_bytes,
                            std::size_t* total_bytes) noexcept;
#else
inline hipError_t DeviceMemoryInfo(std::size_t* free_bytes,
                                   std::size_t* total_bytes) noexcept {
  return hipMemGetInfo(free_bytes, total_bytes);
}
#endif

/// Per DXGI segment group usage and budget (diagnostics; prefill thread 2026-10-08: the load line sums LOCAL and NON_LOCAL, this splits them).
/// `valid` is false on non-Windows builds or when the adapter cannot be matched.
struct DeviceMemorySplit {
  bool valid{false};
  std::size_t local_used{0}, local_budget{0}, nonlocal_used{0}, nonlocal_budget{0};
  /// DXGI_ADAPTER_DESC.DedicatedVideoMemory: the BIOS carve-out (UMA) or VRAM; the LOCAL budget can exceed it by a slice of system RAM.
  std::size_t dedicated{0};
};
#ifdef _WIN32
DeviceMemorySplit QueryDeviceMemorySplit() noexcept;
#else
inline DeviceMemorySplit QueryDeviceMemorySplit() noexcept { return {}; }
#endif

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_DEVICE_MEMORY_HPP_
