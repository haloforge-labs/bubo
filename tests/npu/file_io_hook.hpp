// SPDX-License-Identifier: MIT
#pragma once
#include <windows.h>

#include <atomic>
inline std::atomic<unsigned> npu_test_writes{0}, npu_test_flushes{0};
inline BOOL NpuTestWriteFile(HANDLE h, LPCVOID p, DWORD n, LPDWORD done,
                             LPOVERLAPPED o) {
  ++npu_test_writes;
  return ::WriteFile(h, p, n, done, o);
}
inline BOOL NpuTestFlushFileBuffers(HANDLE h) {
  ++npu_test_flushes;
  return ::FlushFileBuffers(h);
}
#define WriteFile NpuTestWriteFile
#define FlushFileBuffers NpuTestFlushFileBuffers
