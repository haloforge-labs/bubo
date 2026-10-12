// SPDX-License-Identifier: MIT
#include "trace_windows.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"

namespace gufo::npu {
namespace {
json::Value Number(std::uint64_t n) {
  return json::Value::parsed_number(static_cast<double>(n), std::to_string(n));
}
std::uint64_t Read(volatile LONG64& value) {
  return static_cast<std::uint64_t>(InterlockedCompareExchange64(&value, 0, 0));
}
LONG Read(volatile LONG& value) {
  return InterlockedCompareExchange(&value, 0, 0);
}
HANDLE OpenAppend(const std::filesystem::path& path) {
  auto file =
      CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                  OPEN_ALWAYS, FILE_FLAG_WRITE_THROUGH, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("NPU write-through trace open failed: " +
                             std::to_string(GetLastError()));
  return file;
}
}  // namespace

// clang-format off
std::uint64_t QpcNow() noexcept {
  LARGE_INTEGER n{};
  QueryPerformanceCounter(&n);
  return n.QuadPart;
}
std::uint64_t QpcFrequency() noexcept {
  LARGE_INTEGER n{};
  QueryPerformanceFrequency(&n);
  return n.QuadPart;
}
std::string UtcNow() {
  FILETIME ft{};
  GetSystemTimePreciseAsFileTime(&ft);
  SYSTEMTIME st{};
  FileTimeToSystemTime(&ft, &st);
  ULARGE_INTEGER ticks{};
  ticks.LowPart = ft.dwLowDateTime;
  ticks.HighPart = ft.dwHighDateTime;
  char text[48];
  std::snprintf(text, sizeof(text), "%04u-%02u-%02uT%02u:%02u:%02u.%07lluZ",
                st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
                static_cast<unsigned long long>(ticks.QuadPart % 10000000));
  return text;
}
bool AppendDurable(HANDLE file, const std::string& line) noexcept {
  if (file == INVALID_HANDLE_VALUE || line.size() > MAXDWORD)
    return false;
  DWORD written = 0;
  return WriteFile(file, line.data(), static_cast<DWORD>(line.size()), &written,
                   nullptr) &&
         written == line.size() && FlushFileBuffers(file);
}

std::uint64_t NextEventSequence(const std::filesystem::path& directory) {
  // Same byte-range lock and durable 8-byte counter as copied logging.ps1.
  // This is metadata only; no device/DLL access and no environment bypass.
  auto file = CreateFileW((directory / "events.seq").c_str(),
                          GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_ALWAYS, FILE_FLAG_WRITE_THROUGH, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("NPU event sequence file open failed");
  OVERLAPPED lock{};
  bool locked = false;
  try {
    const auto deadline = GetTickCount64() + 2000;
    while (!(locked =
                 LockFileEx(file,
                            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                            0, 1, 0, &lock) != FALSE)) {
      if (GetLastError() != ERROR_LOCK_VIOLATION ||
          GetTickCount64() >= deadline)
        throw std::runtime_error("NPU event sequence lock failed/deadline");
      Sleep(2);
    }
    std::uint64_t sequence = 0;
    DWORD count = 0;
    if (!ReadFile(file, &sequence, sizeof(sequence), &count, nullptr) ||
        (count != 0 && count != sizeof(sequence)) || sequence == UINT64_MAX)
      throw std::runtime_error("NPU event sequence counter invalid");
    ++sequence;
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(file, zero, nullptr, FILE_BEGIN) ||
        !WriteFile(file, &sequence, sizeof(sequence), &count, nullptr) ||
        count != sizeof(sequence) || !FlushFileBuffers(file))
      throw std::runtime_error("NPU event sequence write/flush failed");
    UnlockFileEx(file, 0, 1, 0, &lock);
    CloseHandle(file);
    return sequence;
  } catch (...) {
    if (locked)
      UnlockFileEx(file, 0, 1, 0, &lock);
    CloseHandle(file);
    throw;
  }
}

std::string StateJson(WatchState& s, const std::string& reason) {
  auto state = json::Value::object();
  state["reason"] = reason;
  state["utc"] = UtcNow();
  state["qpc"] = Number(QpcNow());
  state["qpc_freq"] = Number(QpcFrequency());
  state["last_driver_seq"] = Number(Read(s.last_driver_seq));
  state["last_durable_seq"] = Number(Read(s.last_durable_seq));
  state["ring_write_index"] = Number(Read(s.write_index));
  state["dispatch_seq"] = Number(Read(s.dispatch));
  state["last_completed_dispatch"] = Number(Read(s.completed));
  state["outstanding_dispatches"] = json::Value::array();
  if (Read(s.dispatch) > Read(s.completed))
    state["outstanding_dispatches"].push_back(Number(Read(s.dispatch)));
  state["queue_depth"] = Read(s.queue_depth);
  state["active_call"] = Read(s.active) != 0;
  state["call_started_qpc"] = Number(Read(s.call_started));
  state["dma_possible"] = Read(s.dma) != 0;
  state["last_events"] = json::Value::array();
  // Snapshot only stable slots. Torn in-memory entries are explicitly skipped;
  // durable JSONL remains authoritative. Ring truncation never truncates JSONL.
  const auto index = Read(s.write_index);
  const auto start = index > kRingEntries ? index - kRingEntries : 0;
  for (auto i = start; i < index; ++i) {
    auto& entry = s.ring[i % kRingEntries];
    const auto before = Read(entry.version);
    if (before == 0 || (before & 1))
      continue;
    char copy[kRingRecordBytes];
    std::memcpy(copy, entry.json, sizeof(copy));
    MemoryBarrier();
    if (before != Read(entry.version))
      continue;
    copy[sizeof(copy) - 1] = 0;
    state["last_events"].push_back(std::string(copy));
  }
  return state.dump() + "\n";
}

}  // namespace gufo::npu
