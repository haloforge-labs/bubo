#pragma once
#include "src/core/strix_halo.hpp"

// Read-only machine adapter (J5). It reports Windows power state, ADLX GPU
// telemetry, the Variable Graphics Memory carve and OEM identity with the
// source and unit of every value. Nothing here writes power state, VGM,
// firmware or BIOS settings. The probe interface lets CPU tests use fakes.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace bubo::cli {

// One reading. A reading that is not available has a null value and a reason.
struct MachineReading {
  bool available = false;
  gufo::json::Value value;  // null when not available
  std::string source;
  std::string api_version;
  std::string unit;
  std::string sample_time;  // UTC, same format as status sample_time
  std::string reason;       // empty when available
};

MachineReading AvailableReading(gufo::json::Value value, std::string source,
                                std::string api_version, std::string unit,
                                std::string sample_time);
MachineReading UnavailableReading(std::string source, std::string api_version,
                                  std::string unit, std::string sample_time,
                                  std::string reason);

// ADLX reports the carve in "GB" without naming the base. The adapter reads
// the raw double and converts it with the base named below. The raw number
// stays in the output next to the converted bytes.
inline constexpr double kAdlxGbToBytes = 1073741824.0;  // 2^30, assumed

struct MachineGpuReadings {
  // Metric names and support flags come from IADLXGPUMetricsSupport.
  MachineReading power;  // IADLXGPUMetrics::GPUPower, W
  bool power_supported = false;
  MachineReading clock;  // IADLXGPUMetrics::GPUClockSpeed, MHz
  bool clock_supported = false;
  std::string gpu_name;
};

struct MachineVgmReadings {
  MachineReading supported;      // value bool
  MachineReading carved_raw_gb;  // value double, unit GB (ADLX)
  MachineReading remaining_raw_gb;
};

// One DXGI adapter as the real probe or a test fake describes it. DXGI needs
// no vendor SDK. The numbers are the Windows graphics kernel's view.
struct MachineDxgiAdapterInfo {
  std::string name;
  std::uint32_t vendor_id = 0;  // 0x1002 = AMD
  std::uint32_t device_id = 0;
  bool software = false;  // DXGI_ADAPTER_FLAG_SOFTWARE (WARP)
  std::uint64_t dedicated_video_memory = 0;  // DXGI_ADAPTER_DESC1
  std::uint64_t dedicated_system_memory = 0;
  std::uint64_t shared_system_memory = 0;
  bool has_adapter3 = false;  // IDXGIAdapter3 can be queried
  bool query_ok = false;      // QueryVideoMemoryInfo(LOCAL) worked
  std::string query_error;    // reason when query_ok is false
  std::uint64_t budget = 0;
  std::uint64_t current_usage = 0;
  std::uint64_t available_for_reservation = 0;
  std::optional<std::uint64_t> nonlocal_budget;
};

struct MachineDxgiReadings {
  MachineReading pci_id;  // internal identity: vendor << 16 | device
  MachineReading adapter_name;
  MachineReading dedicated_video_memory;     // bytes
  MachineReading shared_system_memory;       // bytes
  MachineReading budget;                     // bytes
  MachineReading total_budget;               // LOCAL + NON_LOCAL bytes
  MachineReading current_usage;              // bytes
  MachineReading available_for_reservation;  // bytes
};

inline constexpr const char* kDxgiDedicatedSource =
    "DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory";
inline constexpr const char* kDxgiSharedSource =
    "DXGI IDXGIAdapter::GetDesc1 SharedSystemMemory";
inline constexpr const char* kDxgiNameSource =
    "DXGI IDXGIAdapter::GetDesc1 Description";
inline constexpr const char* kDxgiBudgetSource =
    "IDXGIAdapter3::QueryVideoMemoryInfo";
inline constexpr std::uint32_t kAmdVendorId = gufo::core::kAmdPciVendorId;
inline constexpr double kDxgiAgreeToleranceBytes = 1048576.0;  // 1 MiB

// Picks the AMD hardware adapter with the largest dedicated video memory (never
// a software adapter, never a fixed index) and builds the six readings.
// `enumeration_error` is the reason when the adapter list could not be read.
MachineDxgiReadings DxgiReadingsFromAdapters(
    const std::vector<MachineDxgiAdapterInfo>& adapters,
    const std::string& enumeration_error, const std::string& sample_time);

class MachineProbe {
public:
  virtual ~MachineProbe() = default;
  virtual MachineReading OverlayScheme() = 0;       // value GUID text
  virtual MachineReading EffectivePowerMode() = 0;  // value raw integer
  virtual MachineReading PowerSource() = 0;         // "ac" / "battery"
  virtual MachineGpuReadings Gpu() = 0;
  virtual MachineVgmReadings Vgm() = 0;
  virtual MachineReading HostPhysicalMemory() = 0;   // value bytes
  virtual MachineReading HostAvailableMemory();      // value bytes, before load
  virtual MachineReading InstalledPhysicalMemory();  // installed DIMMs, bytes
  virtual MachineReading OemManufacturer() = 0;
  virtual MachineReading OemModel() = 0;
  virtual MachineReading BiosVendor() = 0;
  virtual MachineReading BiosVersion() = 0;
  virtual MachineReading BiosDate() = 0;
  // DXGI reads. The default says this probe has no DXGI reader.
  virtual MachineDxgiReadings Dxgi();
};

// The real probe. Missing APIs give unavailable readings, never an error.
std::unique_ptr<MachineProbe> CreateSystemMachineProbe();

struct PlanningMemoryBudgets {
  std::optional<std::uint64_t> gpu_available, host_available;
  std::optional<std::uint64_t> gpu_total, host_total, gpu_dedicated;
  std::string gpu_dedicated_reason =
      "DXGI DedicatedVideoMemory is not reported.";
  bool is_96gb_strix_halo = false;
};
inline bool Is96GiBStrixHalo(const PlanningMemoryBudgets& budgets) {
  return budgets.is_96gb_strix_halo;
}
// Uses the existing DXGI adapter selection. Admission availability is local
// Budget minus CurrentUsage, without shared host memory. The capacity planner
// uses LOCAL + NON_LOCAL total budgets, like DeviceMemoryInfo. The dedicated
// reading is independent of both budgets. Failed readings stay unknown.
PlanningMemoryBudgets ReadPlanningMemoryBudgets(MachineProbe& probe);
PlanningMemoryBudgets ReadPlanningMemoryBudgets();

// Builds the additive `machine` object for status and diagnose output.
gufo::json::Value MachineReport(MachineProbe& probe);

// Name of a raw EFFECTIVE_POWER_MODE value in the Windows API enumeration
// (powersetting.h). Null when the value is not in that enumeration.
std::optional<std::string> WindowsEffectivePowerModeName(std::int64_t raw);

inline constexpr const char* kNoTestedProfileText =
    "No tested profile is available for this memory setting.";

}  // namespace bubo::cli
