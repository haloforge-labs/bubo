// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_MEM_TRACE_HPP_
#define GUFO_NPU_MEM_TRACE_HPP_









#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "fault_state.hpp"

namespace gufo::npu {
struct MemTraceSample {
  bool ok = false;
  std::uint64_t ws_mib = 0, private_mib = 0;
  unsigned pages_sampled = 0, valid = 0, shared = 0, locked = 0, large = 0;
};
class IMemProbe {
public:
  virtual ~IMemProbe() = default;
  virtual bool Sample(MemTraceSample* sample, std::string* error) = 0;
};
// Process-wide probe, read lock-free by the chunk code. nullptr = off.
IMemProbe* GlobalMemProbe() noexcept;
void InstallGlobalMemProbe(std::unique_ptr<IMemProbe> probe);
void StopGlobalMemProbe() noexcept;
// One sample through the global probe: fail-open, logs once on the first
// failure through `log` (default: the server logger). False = "na".
bool SampleMemTrace(MemTraceSample* sample) noexcept;
// Tests: replace the log sink (empty restores the server logger).
void SetMemTraceLogForTest(std::function<void(const std::string&)> log);

// Addresses of the sampled pages (A BO 2, output BO 2, four weight BOs 1 each).
// Called until it returns a non-empty list, then cached: the same addresses are
// sampled for the whole process.
using PageProvider = std::function<std::vector<const void*>()>;
// Windows probe (kernel32 K32GetProcessMemoryInfo / K32QueryWorkingSetEx,
// resolved at run time). Null on other platforms.
std::unique_ptr<IMemProbe> MakeWindowsMemProbe(PageProvider provider);

// One reading of the PDH GPU memory counters, in MiB.
struct GpuMemPoint {
  bool proc_ok = false;
  double proc_local_mib = 0, proc_dedicated_mib = 0, proc_shared_mib = 0;
  bool adapter_ok = false;
  double adapter_shared_mib = 0;
};
class IGpuMemSource {
public:
  virtual ~IGpuMemSource() = default;
  virtual bool Open(std::string* error) = 0;
  virtual bool Read(GpuMemPoint* point, std::string* error) = 0;
  virtual void Close() noexcept {}
};
std::unique_ptr<IGpuMemSource> MakeWindowsGpuMemSource();
}  // namespace gufo::npu
#endif
