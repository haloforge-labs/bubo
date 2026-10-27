#ifndef GUFO_CORE_PLATFORM_MEMORY_READINGS_HPP_
#define GUFO_CORE_PLATFORM_MEMORY_READINGS_HPP_

// The OS side of the memory guard (see memory_guard.hpp): one cheap reading of
// free RAM, commit headroom, this process's private RAM, its WDDM video-memory
// budget (local + non-local) and the Windows low-memory notification state.
// Windows only; elsewhere there is no source and the guard stays off.
//
// Cost on the Strix Halo test machine (microbench, 2026-10-20): GlobalMemoryStatusEx
// 0.63 us, QueryVideoMemoryInfo LOCAL+NON_LOCAL pair 0.83 us, the low-memory
// state query 0.16 us. No sampler thread is needed for the per-request reading.

#include <memory>

#include "src/core/platform/memory_guard.hpp"

namespace gufo::platform {

#ifdef _WIN32
[[nodiscard]] std::unique_ptr<MemoryReadingSource> MakeSystemMemorySource();
/// False when CreateMemoryResourceNotification failed (the guard then ignores the Windows signal and says so once).
[[nodiscard]] bool LowMemoryNotificationAvailable();
#else
inline std::unique_ptr<MemoryReadingSource> MakeSystemMemorySource() { return nullptr; }
inline bool LowMemoryNotificationAvailable() { return false; }
#endif

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_MEMORY_READINGS_HPP_
