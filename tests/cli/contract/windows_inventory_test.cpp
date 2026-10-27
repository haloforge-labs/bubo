#include <iostream>

#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/linux_sysfs.h"
#ifdef _WIN32
#include <windows.h>
#endif

int main() {
  gufo::diagnostics::SystemInventory unsupported;
  unsupported.cpu.architecture = "unsupported";
  unsupported.gpu.architecture = "unsupported";
  const auto platform =
      gufo::diagnostics::EvaluateCompatibility(unsupported, "platform");
  const auto gpu = gufo::diagnostics::EvaluateCompatibility(unsupported, "gpu");
  if (platform.items.size() != 1 || gpu.items.size() != 1 ||
      platform.items[0].remediation_hint !=
          "Bubo only supports Linux or Windows x86-64" ||
      !gpu.items[0].remediation_hint.starts_with(
          "Bubo requires AMD Strix Halo"))
    return 1;
  std::cout << "Compatibility remedies use Bubo: 2 passed, 0 failed\n";
#ifdef _WIN32
  const gufo::diagnostics::LinuxSysfs native;
  const auto cpu = native.QueryCpuInfo();
  const auto memory = native.QueryMemInfo();
  MEMORYSTATUSEX os{};
  os.dwLength = sizeof(os);
  if (!GlobalMemoryStatusEx(&os) || !cpu || cpu->model_name.empty() ||
      cpu->logical_cores != GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) ||
      !cpu->physical_cores || cpu->physical_cores > cpu->logical_cores ||
      !memory || memory->total_bytes != os.ullTotalPhys ||
      !memory->available_bytes)
    return 1;
  const gufo::diagnostics::LinuxSysfs fixture("nonexistent-sys-fixture",
                                              "nonexistent-proc-fixture");
  if (fixture.QueryCpuInfo() || fixture.QueryMemInfo())
    return 1;
  std::cout
      << "Windows native inventory and fixture isolation: 2 passed, 0 failed\n";
#else
  std::cout
      << "Windows inventory: 0 passed, 0 failed, 1 skipped (not Windows)\n";
#endif
}
