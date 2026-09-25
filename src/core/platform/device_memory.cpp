// Windows implementation of gufo::platform::DeviceMemoryInfo; see the header.

#include "src/core/platform/device_memory.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dxgi1_6.h>
#include <windows.h>

#include <cstring>

namespace gufo::platform {
namespace {

/// The DXGI adapter behind the current HIP device, or null.
IDXGIAdapter3* AdapterForCurrentDevice() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return nullptr;
  }
  LUID luid{};
  static_assert(sizeof(luid) == sizeof(properties.luid));
  std::memcpy(&luid, properties.luid, sizeof(luid));
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),
                                reinterpret_cast<void**>(&factory)))) {
    return nullptr;
  }
  IDXGIAdapter3* adapter = nullptr;
  (void)factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3),
                                   reinterpret_cast<void**>(&adapter));
  factory->Release();
  return adapter;
}

}  // namespace

hipError_t DeviceMemoryInfo(std::size_t* free_bytes,
                            std::size_t* total_bytes) noexcept {
  const hipError_t status = hipMemGetInfo(free_bytes, total_bytes);
  if (status != hipSuccess) {
    return status;
  }
  // Resolved once: the adapter of a process's HIP device does not change.
  static IDXGIAdapter3* const adapter = AdapterForCurrentDevice();
  if (adapter == nullptr) {
    return status;
  }
  DXGI_QUERY_VIDEO_MEMORY_INFO local{};
  DXGI_QUERY_VIDEO_MEMORY_INFO shared{};
  if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                           &local)) ||
      FAILED(adapter->QueryVideoMemoryInfo(
          0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared))) {
    return status;
  }
  const auto available = [](const DXGI_QUERY_VIDEO_MEMORY_INFO& info) {
    return info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage
                                           : 0;
  };
  *free_bytes = static_cast<std::size_t>(available(local) + available(shared));
  *total_bytes = static_cast<std::size_t>(local.Budget + shared.Budget);
  return hipSuccess;
}

}  // namespace gufo::platform
