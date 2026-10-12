// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_TRACE_WINDOWS_HPP_
#define GUFO_NPU_TRACE_WINDOWS_HPP_

#include <windows.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

namespace gufo::npu {
// Also consumed by the independent, XRT-free watchdog. Interlocked operations
// are used across the mapping, not process-local std::atomic/mutex objects.
inline constexpr unsigned kRingEntries = 4096;
inline constexpr unsigned kRingRecordBytes = 2048;
struct WatchState {
  volatile LONG64 sequence = 0, heartbeat = 0, call_started = 0;
  volatile LONG64 dispatch = 0, completed = 0;
  volatile LONG64 write_index = 0, last_driver_seq = 0;
  volatile LONG64 last_durable_seq = 0;
  volatile LONG stop = 0, done = 0, ready = 0, active = 0, dma = 0;
  volatile LONG first_error = 0, queue_depth = 0;
  struct Entry {
    volatile LONG version = 0;
    char json[kRingRecordBytes]{};
  } ring[kRingEntries];
};

std::uint64_t QpcNow() noexcept;
std::uint64_t QpcFrequency() noexcept;
std::string UtcNow();
std::string StateJson(WatchState& state, const std::string& reason);
bool AppendDurable(HANDLE file, const std::string& line) noexcept;
std::uint64_t NextEventSequence(const std::filesystem::path& directory);
}  // namespace gufo::npu
#endif
