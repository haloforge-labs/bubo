#pragma once

// Machine setting warnings (GitHub issues #3 and #5). Two settings of the
// tested machine change whether a large model runs at all: the GPU memory
// reservation and the page file. `bubo diagnose` and `bubo serve` read them
// and say so when one differs. A warning never stops a command and never
// changes a setting. The Windows power mode is only a reading in `bubo
// diagnose`: the Windows value does not reliably follow the profile that the
// docs tell people to set (a game reports GameMode whatever the profile).

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/machine_adapter.hpp"
#include "src/cli/memory_advice.hpp"

namespace bubo::cli {

// The tested machine reserves 96 GB for the GPU. DXGI reports 95.8 GiB for it.
inline constexpr std::uint64_t kExpectedGpuReservationBytes = 96000000000ULL;

struct PageFileSetting {
  std::string path;
  bool system_managed = false;  // Windows chooses the size
  std::uint64_t initial_mib = 0;
  std::uint64_t maximum_mib = 0;
};

// Parses one line of the PagingFiles registry value: "C:\pagefile.sys 65536
// 65536", or "?:\pagefile.sys" (no sizes) for a system-managed page file.
std::optional<PageFileSetting> ParsePagingFilesEntry(std::string_view entry);
// Every configured page file. Null when the setting cannot be read (not
// Windows, no access). An empty list means no page file is configured.
std::optional<std::vector<PageFileSetting>> ReadPageFileSettings();

struct MachineWarningInputs {
  std::optional<std::uint64_t> gpu_reservation_bytes;
  std::optional<std::vector<PageFileSetting>> page_files;
  std::optional<std::int64_t> effective_power_mode;
  bool is_96gb_strix_halo = false;
};

struct MachineWarning {
  std::string check;    // "gpu_reservation" or "page_file"
  std::string message;  // what was read and what was tested
  std::string remedy;
};

// A setting that cannot be read gives no warning.
std::vector<MachineWarning> EvaluateMachineWarnings(
    const MachineWarningInputs& inputs);
MachineWarningInputs ReadMachineWarningInputs(MachineProbe& probe);
// One log line for `serve`. Empty when there is nothing to say.
std::string MachineWarningLogLine(const std::vector<MachineWarning>& warnings);

}  // namespace bubo::cli
