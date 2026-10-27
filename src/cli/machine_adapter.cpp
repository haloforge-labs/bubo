#include "src/cli/machine_adapter.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>

#include "src/cli/machine_output.hpp"

namespace bubo::cli {
namespace {
using gufo::json::Value;

Value ReadingJson(const MachineReading& reading) {
  auto value = Value::object();
  value["value"] = reading.available ? reading.value : Value();
  value["source"] = reading.source;
  value["api_version"] = reading.api_version;
  value["unit"] = reading.unit;
  value["sample_time"] = reading.sample_time;
  value["available"] = reading.available;
  value["reason"] = reading.available || reading.reason.empty()
                        ? Value()
                        : Value(reading.reason);
  return value;
}

bool NumberValue(const MachineReading& reading, double* out) {
  if (!reading.available || !reading.value.is_number())
    return false;
  const double number = reading.value.as_double();
  if (!std::isfinite(number) || number < 0)
    return false;
  *out = number;
  return true;
}

bool StartsWithNoCase(const std::string& text, const std::string& prefix) {
  if (text.size() < prefix.size())
    return false;
  for (std::size_t i = 0; i < prefix.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(text[i])) !=
        std::tolower(static_cast<unsigned char>(prefix[i])))
      return false;
  return true;
}

std::string TextOf(const MachineReading& reading) {
  return reading.available && reading.value.is_string() ? reading.value.str()
                                                        : std::string();
}

// Converts a raw ADLX "GB" reading to bytes. The raw number stays visible.
Value MemoryReadingJson(const MachineReading& raw,
                        const std::string& source_note) {
  double gb = 0;
  Value value;
  if (NumberValue(raw, &gb)) {
    const double bytes = std::round(gb * kAdlxGbToBytes);
    auto reading = raw;
    reading.value = Value(bytes);
    reading.unit = "bytes";
    value = ReadingJson(reading);
    value["raw_value"] = gb;
    value["raw_unit"] = "GB";
    value["gib"] = bytes / 1073741824.0;
    value["conversion"] =
        "bytes = raw_value * 2^30. ADLX documents the unit as GB without a "
        "base. " +
        source_note;
  } else {
    auto reading = raw;
    reading.available = false;
    reading.value = Value();
    reading.unit = "bytes";
    if (reading.reason.empty())
      reading.reason = "The reading is not a valid non-negative number.";
    value = ReadingJson(reading);
    value["raw_value"] = Value();
    value["raw_unit"] = "GB";
    value["gib"] = Value();
    value["conversion"] = Value();
  }
  return value;
}

}  // namespace

MachineReading AvailableReading(Value value, std::string source,
                                std::string api_version, std::string unit,
                                std::string sample_time) {
  MachineReading reading;
  reading.available = true;
  reading.value = std::move(value);
  reading.source = std::move(source);
  reading.api_version = std::move(api_version);
  reading.unit = std::move(unit);
  reading.sample_time = std::move(sample_time);
  return reading;
}

MachineReading UnavailableReading(std::string source, std::string api_version,
                                  std::string unit, std::string sample_time,
                                  std::string reason) {
  MachineReading reading;
  reading.source = std::move(source);
  reading.api_version = std::move(api_version);
  reading.unit = std::move(unit);
  reading.sample_time = std::move(sample_time);
  reading.reason = std::move(reason);
  return reading;
}

MachineDxgiReadings MachineProbe::Dxgi() {
  const auto time = UtcTime();
  const std::string reason = "This probe has no DXGI reader.";
  MachineDxgiReadings out;
  out.adapter_name =
      UnavailableReading(kDxgiNameSource, "DXGI 1.1", "text", time, reason);
  out.dedicated_video_memory = UnavailableReading(
      kDxgiDedicatedSource, "DXGI 1.1", "bytes", time, reason);
  out.shared_system_memory =
      UnavailableReading(kDxgiSharedSource, "DXGI 1.1", "bytes", time, reason);
  out.budget =
      UnavailableReading(kDxgiBudgetSource, "DXGI 1.4", "bytes", time, reason);
  out.total_budget = out.budget;
  out.current_usage = out.budget;
  out.available_for_reservation = out.budget;
  return out;
}

MachineReading MachineProbe::HostAvailableMemory() {
  return UnavailableReading("host available memory", "", "bytes", UtcTime(),
                            "This probe has no available host memory reader.");
}
MachineReading MachineProbe::InstalledPhysicalMemory() {
  return UnavailableReading("installed physical memory", "", "bytes", UtcTime(),
                            "This probe has no installed memory reader.");
}
PlanningMemoryBudgets ReadPlanningMemoryBudgets(MachineProbe& probe) {
  const auto bytes =
      [](const MachineReading& reading) -> std::optional<std::uint64_t> {
    double number = 0;
    if (!NumberValue(reading, &number) || number > 9007199254740991.0 ||
        std::floor(number) != number)
      return std::nullopt;
    return static_cast<std::uint64_t>(number);
  };
  PlanningMemoryBudgets budgets;
  const auto installed = bytes(probe.InstalledPhysicalMemory());
  const auto dxgi = probe.Dxgi();
  const auto pci = bytes(dxgi.pci_id);
  constexpr std::uint64_t gib = 1073741824ULL;
  budgets.is_96gb_strix_halo =
      installed && *installed >= 80 * gib && *installed < 112 * gib && pci &&
      gufo::core::IsStrixHaloPciId(static_cast<std::uint32_t>(*pci >> 16),
                                  static_cast<std::uint32_t>(*pci & 0xffff));
  budgets.host_available = bytes(probe.HostAvailableMemory());
  budgets.host_total = bytes(probe.HostPhysicalMemory());
  budgets.gpu_dedicated = bytes(dxgi.dedicated_video_memory);
  if (budgets.gpu_dedicated == 0)
    budgets.gpu_dedicated.reset();
  if (budgets.gpu_dedicated)
    budgets.gpu_dedicated_reason.clear();
  else if (!dxgi.dedicated_video_memory.reason.empty())
    budgets.gpu_dedicated_reason = dxgi.dedicated_video_memory.reason;
  const auto budget = bytes(dxgi.budget);
  const auto used = bytes(dxgi.current_usage);
  budgets.gpu_total = bytes(dxgi.total_budget);
  // HIP allocates from the local DXGI Budget. DedicatedVideoMemory can be
  // smaller on this UMA device and must not cap the allocatable budget.
  if (budget && used)
    budgets.gpu_available = *budget > *used ? *budget - *used : 0;
  return budgets;
}

PlanningMemoryBudgets ReadPlanningMemoryBudgets() {
  auto probe = CreateSystemMachineProbe();
  return ReadPlanningMemoryBudgets(*probe);
}

MachineDxgiReadings DxgiReadingsFromAdapters(
    const std::vector<MachineDxgiAdapterInfo>& adapters,
    const std::string& enumeration_error, const std::string& time) {
  MachineDxgiReadings out;
  std::string reason = enumeration_error;
  const MachineDxgiAdapterInfo* pick = nullptr;
  if (reason.empty()) {
    for (const auto& adapter : adapters) {
      if (adapter.software || adapter.vendor_id != kAmdVendorId)
        continue;
      if (!pick ||
          adapter.dedicated_video_memory > pick->dedicated_video_memory)
        pick = &adapter;
    }
    if (!pick)
      reason = "DXGI lists no AMD hardware adapter.";
  }
  const auto none = [&](const char* source, const char* version,
                        const char* unit, const std::string& why) {
    return UnavailableReading(source, version, unit, time, why);
  };
  if (!pick) {
    out.pci_id =
        none("DXGI GetDesc1 VendorId/DeviceId", "DXGI 1.1", "id", reason);
    out.adapter_name = none(kDxgiNameSource, "DXGI 1.1", "text", reason);
    out.dedicated_video_memory =
        none(kDxgiDedicatedSource, "DXGI 1.1", "bytes", reason);
    out.shared_system_memory =
        none(kDxgiSharedSource, "DXGI 1.1", "bytes", reason);
    out.budget = none(kDxgiBudgetSource, "DXGI 1.4", "bytes", reason);
    out.total_budget = out.budget;
    out.current_usage = out.budget;
    out.available_for_reservation = out.budget;
    return out;
  }
  const auto bytes = [](std::uint64_t n) {
    return Value(static_cast<double>(n));
  };
  out.pci_id =
      pick->device_id
          ? AvailableReading(
                bytes((std::uint64_t(pick->vendor_id) << 16) | pick->device_id),
                "DXGI GetDesc1 VendorId/DeviceId", "DXGI 1.1", "id", time)
          : none("DXGI GetDesc1 VendorId/DeviceId", "DXGI 1.1", "id",
                 "DXGI device identity is unavailable.");
  out.adapter_name = pick->name.empty()
                         ? none(kDxgiNameSource, "DXGI 1.1", "text",
                                "The adapter has no name.")
                         : AvailableReading(pick->name, kDxgiNameSource,
                                            "DXGI 1.1", "text", time);
  out.dedicated_video_memory =
      pick->dedicated_video_memory == 0
          ? none(kDxgiDedicatedSource, "DXGI 1.1", "bytes",
                 "DXGI reports 0 bytes of dedicated video memory for this "
                 "adapter.")
          : AvailableReading(bytes(pick->dedicated_video_memory),
                             kDxgiDedicatedSource, "DXGI 1.1", "bytes", time);
  out.shared_system_memory =
      AvailableReading(bytes(pick->shared_system_memory), kDxgiSharedSource,
                       "DXGI 1.1", "bytes", time);
  if (!pick->has_adapter3) {
    const std::string why = "IDXGIAdapter3 is not available on this adapter.";
    out.budget = none(kDxgiBudgetSource, "DXGI 1.4", "bytes", why);
    out.total_budget = out.budget;
    out.current_usage = out.budget;
    out.available_for_reservation = out.budget;
  } else if (!pick->query_ok) {
    const std::string why = pick->query_error.empty()
                                ? "QueryVideoMemoryInfo failed."
                                : pick->query_error;
    out.budget = none(kDxgiBudgetSource, "DXGI 1.4", "bytes", why);
    out.total_budget = out.budget;
    out.current_usage = out.budget;
    out.available_for_reservation = out.budget;
  } else {
    out.budget = AvailableReading(bytes(pick->budget), kDxgiBudgetSource,
                                  "DXGI 1.4", "bytes", time);
    out.current_usage =
        AvailableReading(bytes(pick->current_usage), kDxgiBudgetSource,
                         "DXGI 1.4", "bytes", time);
    out.available_for_reservation =
        AvailableReading(bytes(pick->available_for_reservation),
                         kDxgiBudgetSource, "DXGI 1.4", "bytes", time);
  }
  out.total_budget =
      pick->has_adapter3 && pick->query_ok && pick->nonlocal_budget
          ? AvailableReading(
                bytes(pick->budget + *pick->nonlocal_budget),
                "IDXGIAdapter3::QueryVideoMemoryInfo LOCAL + NON_LOCAL Budget",
                "DXGI 1.4", "bytes", time)
          : none("IDXGIAdapter3::QueryVideoMemoryInfo LOCAL + NON_LOCAL Budget",
                 "DXGI 1.4", "bytes",
                 "LOCAL or NON_LOCAL budget is not reported.");
  return out;
}

std::optional<std::string> WindowsEffectivePowerModeName(std::int64_t raw) {
  // Order of EFFECTIVE_POWER_MODE in the Windows SDK powersetting.h.
  switch (raw) {
    case 0:
      return "BatterySaver";
    case 1:
      return "BetterBattery";
    case 2:
      return "Balanced";
    case 3:
      return "HighPerformance";
    case 4:
      return "MaxPerformance";
    case 5:
      return "GameMode";
    case 6:
      return "MixedReality";
    default:
      return std::nullopt;
  }
}

Value MachineReport(MachineProbe& probe) {
  auto machine = Value::object();
  machine["read_only"] = true;
  machine["sample_time"] = UtcTime();

  // Windows policy layer.
  auto windows = Value::object();
  windows["overlay_scheme"] = ReadingJson(probe.OverlayScheme());
  const auto mode = probe.EffectivePowerMode();
  auto mode_json = ReadingJson(mode);
  const bool mode_ok = mode.available && mode.value.is_number();
  const std::int64_t raw =
      mode_ok ? static_cast<std::int64_t>(mode.value.as_double()) : -1;
  mode_json["raw_value"] = mode_ok ? Value(raw) : Value();
  const auto enum_name =
      mode_ok ? WindowsEffectivePowerModeName(raw) : std::nullopt;
  mode_json["windows_enum_name"] = enum_name ? Value(*enum_name) : Value();
  mode_json["windows_enum_source"] =
      "EFFECTIVE_POWER_MODE in powersetting.h (Windows SDK). The name is the "
      "Windows API name only. It is not an ASUS or Armoury Crate mode name.";
  // No ASUS mode label is derived from the Windows value. See the reference
  // observation below for the one measured case.
  mode_json["validated_mapping"] = Value();
  mode_json["validated_mapping_reason"] =
      "No measured mapping from this value to an OEM mode exists. The plan "
      "section 7 measurement gate has not run.";
  const auto manufacturer = probe.OemManufacturer();
  const auto model = probe.OemModel();
  const bool reference_machine =
      StartsWithNoCase(TextOf(manufacturer), "ASUS") &&
      StartsWithNoCase(TextOf(model), "ProArt PX13");
  auto reference = Value::object();
  reference["confidence"] = "harness-observed, ASUS PX13 only";
  reference["source"] =
      "tools/adlx-gpu-monitor field effective_power_mode, operator report "
      "2026-10-13";
  reference["raw_value"] = 4;
  reference["observed_meaning"] = "Armoury Crate Performance (operator report)";
  reference["other_observed"] = Value::object();
  reference["other_observed"]["1"] = "Whisper (operator report)";
  reference["machine"] = "ASUS ProArt PX13";
  reference["validated"] = false;
  reference["applies_to_this_machine"] = reference_machine;
  reference["matches_reference"] =
      reference_machine && mode_ok ? Value(raw == 4) : Value();
  mode_json["reference_observation"] = reference;
  windows["effective_power_mode"] = mode_json;
  windows["power_source"] = ReadingJson(probe.PowerSource());
  machine["windows"] = windows;

  // AMD driver layer: supported GPU metrics only.
  const auto gpu = probe.Gpu();
  auto gpu_json = Value::object();
  auto power = ReadingJson(gpu.power);
  power["label"] = "GPU power";
  power["metric"] = "IADLXGPUMetrics::GPUPower";
  power["support_flag"] = "IADLXGPUMetricsSupport::IsSupportedGPUPower";
  power["supported"] = gpu.power_supported;
  power["validation"] =
      "Not validated against an external meter. On an APU this metric can "
      "include more than the graphics block. Do not show it as package or CPU "
      "power.";
  gpu_json["power"] = power;
  auto clock = ReadingJson(gpu.clock);
  clock["label"] = "GPU clock";
  clock["metric"] = "IADLXGPUMetrics::GPUClockSpeed";
  clock["support_flag"] = "IADLXGPUMetricsSupport::IsSupportedGPUClockSpeed";
  clock["supported"] = gpu.clock_supported;
  gpu_json["clock"] = clock;
  gpu_json["name"] = gpu.gpu_name.empty() ? Value() : Value(gpu.gpu_name);
  machine["gpu"] = gpu_json;

  // Variable Graphics Memory: read only. Reserved memory is not free memory.
  const auto vgm = probe.Vgm();
  auto vgm_json = Value::object();
  vgm_json["supported"] = ReadingJson(vgm.supported);
  const std::string note =
      "The carve is a reservation. It is not free or usable GPU memory.";
  vgm_json["carved"] = MemoryReadingJson(vgm.carved_raw_gb, note);
  vgm_json["remaining"] = MemoryReadingJson(vgm.remaining_raw_gb, note);
  machine["variable_graphics_memory"] = vgm_json;

  // DXGI: the Windows graphics view of the same memory. No vendor SDK needed.
  const auto dxgi = probe.Dxgi();
  auto dxgi_json = Value::object();
  const auto with_gib = [](const MachineReading& reading, const char* metric) {
    auto json = ReadingJson(reading);
    json["metric"] = metric;
    double bytes = 0;
    json["gib"] =
        NumberValue(reading, &bytes) ? Value(bytes / 1073741824.0) : Value();
    return json;
  };
  dxgi_json["adapter_name"] = ReadingJson(dxgi.adapter_name);
  dxgi_json["dedicated_video_memory"] =
      with_gib(dxgi.dedicated_video_memory, "DedicatedVideoMemory");
  dxgi_json["shared_system_memory"] =
      with_gib(dxgi.shared_system_memory, "SharedSystemMemory");
  dxgi_json["budget"] = with_gib(dxgi.budget, "Budget");
  dxgi_json["current_usage"] = with_gib(dxgi.current_usage, "CurrentUsage");
  dxgi_json["available_for_reservation"] =
      with_gib(dxgi.available_for_reservation, "AvailableForReservation");
  // Both readers answered: show whether they give the same carve (within 1
  // MiB).
  double dxgi_bytes = 0;
  double adlx_gb = 0;
  if (NumberValue(dxgi.dedicated_video_memory, &dxgi_bytes) &&
      NumberValue(vgm.carved_raw_gb, &adlx_gb)) {
    const double adlx_bytes = std::round(adlx_gb * kAdlxGbToBytes);
    dxgi_json["agree"] =
        std::fabs(dxgi_bytes - adlx_bytes) <= kDxgiAgreeToleranceBytes;
    dxgi_json["agree_reason"] = Value();
  } else {
    dxgi_json["agree"] = Value();
    dxgi_json["agree_reason"] =
        "Both the DXGI dedicated video memory and the ADLX carve must be "
        "available to compare them.";
  }
  dxgi_json["agree_compares"] =
      "dxgi.dedicated_video_memory and variable_graphics_memory.carved";
  dxgi_json["agree_tolerance_bytes"] = kDxgiAgreeToleranceBytes;
  machine["dxgi"] = dxgi_json;

  // Host memory and OEM identity.
  const auto host = probe.HostPhysicalMemory();
  machine["host_physical_memory"] = ReadingJson(host);
  auto oem = Value::object();
  oem["manufacturer"] = ReadingJson(manufacturer);
  oem["model"] = ReadingJson(model);
  oem["bios_vendor"] = ReadingJson(probe.BiosVendor());
  oem["bios_version"] = ReadingJson(probe.BiosVersion());
  oem["bios_date"] = ReadingJson(probe.BiosDate());
  machine["oem"] = oem;

  // Recipe suggestion: no tested recipe record exists yet.
  auto recipe = Value::object();
  recipe["text"] = kNoTestedProfileText;
  recipe["tested_profile"] = Value();
  recipe["reason"] = "No tested recipe record exists.";
  double carve_bytes = 0;
  double host_bytes = 0;
  auto inputs = Value::object();
  inputs["carved_bytes"] = NumberValue(vgm.carved_raw_gb, &carve_bytes)
                               ? Value(std::round(carve_bytes * kAdlxGbToBytes))
                               : Value();
  inputs["host_physical_bytes"] =
      NumberValue(host, &host_bytes) ? Value(host_bytes) : Value();
  recipe["inputs"] = inputs;
  machine["recipe_suggestion"] = recipe;
  return machine;
}

}  // namespace bubo::cli
