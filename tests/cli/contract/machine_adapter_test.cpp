#include "src/cli/machine_adapter.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {
using bubo::cli::AvailableReading;
using bubo::cli::MachineReading;
using bubo::cli::UnavailableReading;
using gufo::json::Value;

int passed = 0;
int failed = 0;
void Check(bool ok, const char* what) {
  if (ok) {
    ++passed;
  } else {
    ++failed;
    std::cerr << "FAIL: " << what << '\n';
  }
}

const Value& At(const Value& value) {
  return value;
}
template<typename... Rest>
const Value& At(const Value& value, const char* key, Rest... rest) {
  static const Value kNull;
  const Value* child = value.find(key);
  return child ? At(*child, rest...) : kNull;
}

struct Fake final : bubo::cli::MachineProbe {
  MachineReading overlay =
      AvailableReading("ded574b5-45a0-4f42-8737-46345c09c238", "fake overlay",
                       "fake", "guid", "2026-10-14T00:00:00Z");
  MachineReading mode = AvailableReading(4, "fake mode", "fake-v2", "enum",
                                         "2026-10-14T00:00:00Z");
  MachineReading source =
      AvailableReading("ac", "fake ac", "fake", "text", "2026-10-14T00:00:00Z");
  bubo::cli::MachineGpuReadings gpu;
  bubo::cli::MachineVgmReadings vgm;
  MachineReading host =
      AvailableReading(128.0 * 1024 * 1024 * 1024 - 1.0, "fake host", "fake",
                       "bytes", "2026-10-14T00:00:00Z");
  MachineReading manufacturer =
      AvailableReading("ASUS", "fake", "fake", "text", "2026-10-14T00:00:00Z");
  MachineReading model = AvailableReading(
      "ProArt PX13 HN7306EAC", "fake", "fake", "text", "2026-10-14T00:00:00Z");
  MachineReading bios_vendor =
      AvailableReading("AMI", "fake", "fake", "text", "2026-10-14T00:00:00Z");
  MachineReading bios_version = AvailableReading(
      "HN7306EAC.310", "fake", "fake", "text", "2026-10-14T00:00:00Z");
  MachineReading bios_date = AvailableReading("07/08/2026", "fake", "fake",
                                              "text", "2026-10-14T00:00:00Z");
  bubo::cli::MachineDxgiReadings dxgi;
  Fake() {
    const std::string t = "2026-10-14T00:00:00Z";
    gpu.power = AvailableReading(6.5, "fake power", "fake", "W", t);
    gpu.power_supported = true;
    gpu.clock = AvailableReading(800, "fake clock", "fake", "MHz", t);
    gpu.clock_supported = true;
    gpu.gpu_name = "Fake GPU";
    vgm.supported = AvailableReading(true, "fake vgm", "fake", "bool", t);
    vgm.carved_raw_gb = AvailableReading(96.0, "fake carved", "fake", "GB", t);
    vgm.remaining_raw_gb =
        AvailableReading(40.5, "fake remaining", "fake", "GB", t);
    dxgi = bubo::cli::DxgiReadingsFromAdapters(
        {Adapter("Fake AMD GPU", 96ull << 30)}, "", t);
  }
  static bubo::cli::MachineDxgiAdapterInfo Adapter(const char* name,
                                                   std::uint64_t dedicated) {
    bubo::cli::MachineDxgiAdapterInfo a;
    a.name = name;
    a.vendor_id = bubo::cli::kAmdVendorId;
    a.dedicated_video_memory = dedicated;
    a.shared_system_memory = 16ull << 30;
    a.has_adapter3 = true;
    a.query_ok = true;
    a.budget = 120ull << 30;
    a.current_usage = 3ull << 30;
    a.available_for_reservation = 110ull << 30;
    return a;
  }
  bubo::cli::MachineDxgiReadings Dxgi() override { return dxgi; }
  MachineReading OverlayScheme() override { return overlay; }
  MachineReading EffectivePowerMode() override { return mode; }
  MachineReading PowerSource() override { return source; }
  bubo::cli::MachineGpuReadings Gpu() override { return gpu; }
  bubo::cli::MachineVgmReadings Vgm() override { return vgm; }
  MachineReading HostPhysicalMemory() override { return host; }
  MachineReading HostAvailableMemory() override { return host; }
  MachineReading OemManufacturer() override { return manufacturer; }
  MachineReading OemModel() override { return model; }
  MachineReading BiosVendor() override { return bios_vendor; }
  MachineReading BiosVersion() override { return bios_version; }
  MachineReading BiosDate() override { return bios_date; }
};

// Every reading object must keep value and reason consistent with available.
void CheckReadings(const Value& node, const std::string& path) {
  if (!node.is_object())
    return;
  if (const Value* available = node.find("available")) {
    const bool ok = available->as_bool();
    Check(node.find("source") && node.find("api_version") &&
              node.find("unit") && node.find("sample_time") &&
              node.find("value") && node.find("reason"),
          (path + " has all reading fields").c_str());
    Check(ok ? node.find("value")->is_null() == false &&
                   node.find("reason")->is_null()
             : node.find("value")->is_null() &&
                   node.find("reason")->is_string() &&
                   !node.find("reason")->str().empty(),
          (path + " value and reason match available").c_str());
  }
  for (const auto& [key, child] : node.members())
    CheckReadings(child, path + "." + key);
}

constexpr double kGiB = 1073741824.0;
}  // namespace

int main() {
  {
    Fake probe;
    const auto report = bubo::cli::MachineReport(probe);
    Check(At(report, "read_only").as_bool(), "read_only is true");
    CheckReadings(report, "machine");
    const auto& mode = At(report, "windows", "effective_power_mode");
    Check(At(mode, "raw_value").as_double() == 4, "raw mode value kept");
    Check(At(mode, "windows_enum_name").str() == "MaxPerformance",
          "Windows enum name for 4");
    Check(At(mode, "validated_mapping").is_null(),
          "no validated OEM mapping is claimed");
    Check(At(mode, "reference_observation", "confidence").str() ==
              "harness-observed, ASUS PX13 only",
          "confidence text");
    Check(
        At(mode, "reference_observation", "applies_to_this_machine").as_bool(),
        "reference applies on ProArt PX13");
    Check(At(mode, "reference_observation", "matches_reference").as_bool(),
          "raw 4 matches reference");
    Check(At(report, "windows", "overlay_scheme", "value").str() ==
              "ded574b5-45a0-4f42-8737-46345c09c238",
          "overlay GUID text");
    Check(At(report, "windows", "power_source", "value").str() == "ac",
          "AC state");
    Check(At(report, "gpu", "power", "value").as_double() == 6.5 &&
              At(report, "gpu", "power", "unit").str() == "W" &&
              At(report, "gpu", "power", "label").str() == "GPU power",
          "GPU power in W labelled GPU power");
    Check(At(report, "gpu", "clock", "unit").str() == "MHz", "clock unit");
    Check(
        At(report, "gpu", "power", "validation").str().find("Not validated") ==
            0,
        "GPU power validation note");
    const auto& carved = At(report, "variable_graphics_memory", "carved");
    Check(At(carved, "value").as_double() == 96.0 * kGiB,
          "96 GB reading is bytes");
    Check(At(carved, "value").as_double() != 96e9,
          "carve is not converted with a decimal base");
    Check(At(carved, "gib").as_double() == 96.0, "GiB display value");
    Check(At(carved, "raw_value").as_double() == 96.0 &&
              At(carved, "raw_unit").str() == "GB" &&
              At(carved, "unit").str() == "bytes",
          "raw number and units both kept");
    Check(At(report, "variable_graphics_memory", "remaining", "value")
                  .as_double() == std::round(40.5 * kGiB),
          "remaining converted");
    Check(
        At(report, "oem", "model", "value").str() == "ProArt PX13 HN7306EAC" &&
            At(report, "oem", "bios_version", "value").str() == "HN7306EAC.310",
        "OEM identity");
    Check(At(report, "recipe_suggestion", "text").str() ==
              "No tested profile is available for this memory setting.",
          "no tested profile text");
    Check(At(report, "recipe_suggestion", "tested_profile").is_null(),
          "no tested profile record");
    Check(
        At(report, "recipe_suggestion", "inputs", "carved_bytes").as_double() ==
            96.0 * kGiB,
        "recipe inputs carry carve bytes");
    // The JSON text is one line of valid output.
    const auto text = report.dump();
    Check(text.find('\n') == std::string::npos, "single line JSON");
    Check(gufo::json::parse(text).dump() == text, "JSON round trip");
  }
  {  // other mode values and another machine
    Fake probe;
    probe.mode.value = Value(1);
    auto report = bubo::cli::MachineReport(probe);
    auto mode = At(report, "windows", "effective_power_mode");
    Check(At(mode, "windows_enum_name").str() == "BetterBattery",
          "Windows enum name for 1");
    Check(At(mode, "reference_observation", "matches_reference").is_bool() &&
              !At(mode, "reference_observation", "matches_reference").as_bool(),
          "raw 1 is below the reference");
    probe.mode.value = Value(99);
    report = bubo::cli::MachineReport(probe);
    mode = At(report, "windows", "effective_power_mode");
    Check(At(mode, "windows_enum_name").is_null() &&
              At(mode, "raw_value").as_double() == 99,
          "unknown raw value has no name");
    probe.model.value = Value("Other Laptop");
    report = bubo::cli::MachineReport(probe);
    mode = At(report, "windows", "effective_power_mode");
    Check(!At(mode, "reference_observation", "applies_to_this_machine")
                  .as_bool() &&
              At(mode, "reference_observation", "matches_reference").is_null(),
          "reference does not apply to another model");
    probe.model.value = Value("ProArt PX13 HN7306EAC");
    probe.manufacturer.value = Value("Other");
    report = bubo::cli::MachineReport(probe);
    Check(!At(report, "windows", "effective_power_mode",
              "reference_observation", "applies_to_this_machine")
               .as_bool(),
          "reference does not apply to another maker");
  }
  {  // everything unavailable
    Fake probe;
    const std::string t = "2026-10-14T00:00:00Z";
    const auto none = [&](const char* unit) {
      return UnavailableReading("fake", "fake", unit, t, "Not available here.");
    };
    probe.overlay = none("guid");
    probe.mode = none("enum");
    probe.source = none("text");
    probe.gpu.power = none("W");
    probe.gpu.power_supported = false;
    probe.gpu.clock = none("MHz");
    probe.gpu.clock_supported = false;
    probe.gpu.gpu_name.clear();
    probe.vgm.supported = none("bool");
    probe.vgm.carved_raw_gb = none("GB");
    probe.vgm.remaining_raw_gb = none("GB");
    probe.host = none("bytes");
    probe.manufacturer = none("text");
    probe.model = none("text");
    probe.bios_vendor = none("text");
    probe.bios_version = none("text");
    probe.bios_date = none("text");
    const auto report = bubo::cli::MachineReport(probe);
    CheckReadings(report, "unavailable");
    Check(At(report, "windows", "effective_power_mode", "value").is_null() &&
              At(report, "windows", "effective_power_mode", "raw_value")
                  .is_null() &&
              At(report, "windows", "effective_power_mode",
                 "reference_observation", "matches_reference")
                  .is_null(),
          "unavailable mode is null, not zero");
    Check(
        At(report, "variable_graphics_memory", "carved", "value").is_null() &&
            At(report, "variable_graphics_memory", "carved", "gib").is_null() &&
            At(report, "variable_graphics_memory", "carved", "unit").str() ==
                "bytes",
        "unavailable carve is null with a reason");
    Check(At(report, "recipe_suggestion", "inputs", "carved_bytes").is_null() &&
              At(report, "recipe_suggestion", "text").str() ==
                  "No tested profile is available for this memory setting.",
          "recipe text without a carve");
    Check(At(report, "gpu", "name").is_null(), "no GPU name");
  }
  {  // unsupported metric and invalid raw numbers
    Fake probe;
    probe.gpu.power =
        UnavailableReading("fake", "fake", "W", "2026-10-14T00:00:00Z",
                           "ADLX reports GPU power as not supported.");
    probe.gpu.power_supported = false;
    probe.vgm.carved_raw_gb.value = Value(-1.0);
    probe.vgm.remaining_raw_gb.value = Value(std::nan(""));
    const auto report = bubo::cli::MachineReport(probe);
    CheckReadings(report, "unsupported");
    Check(!At(report, "gpu", "power", "supported").as_bool() &&
              !At(report, "gpu", "power", "available").as_bool(),
          "unsupported GPU power");
    Check(At(report, "gpu", "clock", "available").as_bool(),
          "clock stays independent of power");
    Check(!At(report, "variable_graphics_memory", "carved", "available")
                  .as_bool() &&
              !At(report, "variable_graphics_memory", "remaining", "available")
                   .as_bool(),
          "negative and NaN carve readings are rejected");
  }
  {  // fractional GB and byte rounding
    Fake probe;
    probe.vgm.carved_raw_gb.value = Value(0.5);
    const auto report = bubo::cli::MachineReport(probe);
    Check(
        At(report, "variable_graphics_memory", "carved", "value").as_double() ==
            536870912.0,
        "0.5 GB is 536870912 bytes");
    Check(At(report, "variable_graphics_memory", "carved", "gib").as_double() ==
              0.5,
          "0.5 GiB display");
  }
  {  // DXGI: adapter selection and the agree flag
    using bubo::cli::DxgiReadingsFromAdapters;
    const std::string t = "2026-10-14T00:00:00Z";
    auto warp = Fake::Adapter("Microsoft Basic Render Driver", 200ull << 30);
    warp.vendor_id = 0x1414;
    warp.software = true;
    auto soft_amd = Fake::Adapter("AMD flagged software", 300ull << 30);
    soft_amd.software = true;
    auto nvidia = Fake::Adapter("Other vendor GPU", 400ull << 30);
    nvidia.vendor_id = 0x10de;
    // software adapters (even a larger one, even AMD flagged software) are
    // skipped
    auto r = DxgiReadingsFromAdapters(
        {warp, soft_amd, nvidia, Fake::Adapter("Real AMD", 96ull << 30)}, "",
        t);
    Check(r.adapter_name.value.str() == "Real AMD" &&
              r.dedicated_video_memory.value.as_double() == 96.0 * kGiB,
          "software and non-AMD adapters are skipped");
    Check(r.dedicated_video_memory.source ==
                  "DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory" &&
              r.budget.source == "IDXGIAdapter3::QueryVideoMemoryInfo" &&
              r.budget.value.as_double() == 120.0 * kGiB &&
              r.current_usage.value.as_double() == 3.0 * kGiB &&
              r.available_for_reservation.value.as_double() == 110.0 * kGiB &&
              r.shared_system_memory.value.as_double() == 16.0 * kGiB,
          "DXGI readings carry exact source names and values");
    // two AMD adapters: the larger dedicated memory wins, in either order
    for (const bool flip : {false, true}) {
      auto small = Fake::Adapter("Small AMD", 2ull << 30);
      auto large = Fake::Adapter("Large AMD", 8ull << 30);
      r = DxgiReadingsFromAdapters(
          flip ? std::vector{large, small} : std::vector{small, large}, "", t);
      Check(r.adapter_name.value.str() == "Large AMD",
            "two adapters pick the larger regardless of order");
    }
    // only a software adapter: unavailable with a reason
    r = DxgiReadingsFromAdapters({warp}, "", t);
    Check(!r.dedicated_video_memory.available &&
              !r.dedicated_video_memory.reason.empty() && !r.budget.available &&
              !r.adapter_name.available,
          "only a software adapter gives unavailable");
    r = DxgiReadingsFromAdapters({}, "", t);
    Check(!r.dedicated_video_memory.available, "empty adapter list");
    r = DxgiReadingsFromAdapters({}, "CreateDXGIFactory1 returned -1.", t);
    Check(!r.shared_system_memory.available &&
              r.budget.reason == "CreateDXGIFactory1 returned -1.",
          "enumeration error is the reason");
    // missing IDXGIAdapter3: budget fields unavailable, description fields stay
    auto no3 = Fake::Adapter("No Adapter3", 96ull << 30);
    no3.has_adapter3 = false;
    r = DxgiReadingsFromAdapters({no3}, "", t);
    Check(r.dedicated_video_memory.available && !r.budget.available &&
              !r.current_usage.available &&
              !r.available_for_reservation.available &&
              r.budget.reason.find("IDXGIAdapter3") != std::string::npos,
          "missing IDXGIAdapter3 is unavailable with a reason");
    auto failed_query = Fake::Adapter("Query fails", 96ull << 30);
    failed_query.query_ok = false;
    failed_query.query_error = "QueryVideoMemoryInfo returned -2005270527.";
    r = DxgiReadingsFromAdapters({failed_query}, "", t);
    Check(!r.budget.available && r.dedicated_video_memory.available &&
              r.budget.reason.find("-2005270527") != std::string::npos,
          "failed QueryVideoMemoryInfo carries its code");
    // zero dedicated memory
    r = DxgiReadingsFromAdapters({Fake::Adapter("Zero", 0)}, "", t);
    Check(!r.dedicated_video_memory.available &&
              r.dedicated_video_memory.value.is_null() &&
              r.dedicated_video_memory.reason.find("0 bytes") !=
                  std::string::npos &&
              r.budget.available,
          "zero dedicated memory is unavailable, never a 0 carve");
    // report: agree is true within 1 MiB, false beyond it, null when one side
    // is missing
    Fake probe;
    auto report = bubo::cli::MachineReport(probe);
    CheckReadings(report, "dxgi");
    Check(At(report, "dxgi", "agree").is_bool() &&
              At(report, "dxgi", "agree").as_bool() &&
              At(report, "dxgi", "dedicated_video_memory", "gib").as_double() ==
                  96.0,
          "DXGI 96 GiB agrees with ADLX 96 GB");
    probe.dxgi = DxgiReadingsFromAdapters(
        {Fake::Adapter("Within", (96ull << 30) + (1ull << 20))}, "", t);
    report = bubo::cli::MachineReport(probe);
    Check(At(report, "dxgi", "agree").as_bool(),
          "exactly 1 MiB apart still agrees");
    probe.dxgi = DxgiReadingsFromAdapters(
        {Fake::Adapter("Beyond", (96ull << 30) + (1ull << 20) + 1)}, "", t);
    report = bubo::cli::MachineReport(probe);
    Check(At(report, "dxgi", "agree").is_bool() &&
              !At(report, "dxgi", "agree").as_bool(),
          "more than 1 MiB apart is a mismatch");
    probe.dxgi = DxgiReadingsFromAdapters(
        {Fake::Adapter("Mismatch", 16ull << 30)}, "", t);
    report = bubo::cli::MachineReport(probe);
    Check(!At(report, "dxgi", "agree").as_bool(),
          "16 GiB vs 96 GB mismatch is visible");
    probe.vgm.carved_raw_gb =
        UnavailableReading("fake", "fake", "GB", t, "No ADLX.");
    report = bubo::cli::MachineReport(probe);
    Check(
        At(report, "dxgi", "agree").is_null() &&
            At(report, "dxgi", "agree_reason").is_string() &&
            At(report, "dxgi", "dedicated_video_memory", "available").as_bool(),
        "agree is null when ADLX is missing; DXGI still reports");
    Fake base;  // a probe without a DXGI reader
    struct NoDxgi final : bubo::cli::MachineProbe {
      Fake* f = nullptr;
      MachineReading OverlayScheme() override { return f->overlay; }
      MachineReading EffectivePowerMode() override { return f->mode; }
      MachineReading PowerSource() override { return f->source; }
      bubo::cli::MachineGpuReadings Gpu() override { return f->gpu; }
      bubo::cli::MachineVgmReadings Vgm() override { return f->vgm; }
      MachineReading HostPhysicalMemory() override { return f->host; }
      MachineReading OemManufacturer() override { return f->manufacturer; }
      MachineReading OemModel() override { return f->model; }
      MachineReading BiosVendor() override { return f->bios_vendor; }
      MachineReading BiosVersion() override { return f->bios_version; }
      MachineReading BiosDate() override { return f->bios_date; }
    } bare;
    bare.f = &base;
    report = bubo::cli::MachineReport(bare);
    CheckReadings(report, "bare");
    Check(
        !At(report, "dxgi", "dedicated_video_memory", "available").as_bool() &&
            At(report, "dxgi", "agree").is_null(),
        "default probe reports DXGI as unavailable");
  }
  Check(!bubo::cli::WindowsEffectivePowerModeName(-1) &&
            !bubo::cli::WindowsEffectivePowerModeName(7) &&
            *bubo::cli::WindowsEffectivePowerModeName(0) == "BatterySaver" &&
            *bubo::cli::WindowsEffectivePowerModeName(2) == "Balanced",
        "enum name bounds");
  {  // Local budget minus usage, independently of the dedicated-memory label.
    const auto passed_before = passed, failed_before = failed;
    Fake fake;
    fake.dxgi.total_budget = fake.dxgi.budget;
    auto budgets = bubo::cli::ReadPlanningMemoryBudgets(fake);
    Check(budgets.gpu_available == 117ULL * 1073741824ULL,
          "PlanningBudgetLocalBudgetMinusUsage");
    Check(budgets.host_available.has_value(), "PlanningBudgetIndependentHost");
    Check(budgets.gpu_total ==
              static_cast<std::uint64_t>(fake.dxgi.budget.value.as_double()),
          "PlanningTotalUsesBudgetBeforeUsage");
    Check(budgets.gpu_dedicated ==
              static_cast<std::uint64_t>(
                  fake.dxgi.dedicated_video_memory.value.as_double()),
          "PlanningDedicatedMeasuredIndependently");
    Check(budgets.host_total.has_value(), "PlanningHostPhysicalTotal");
    Check(budgets.gpu_dedicated_reason.empty(),
          "PlanningDedicatedAvailableHasNoReason");
    fake.dxgi.budget =
        AvailableReading(110192.0 * 1048576, "fake", "fake", "bytes", "");
    fake.dxgi.current_usage = AvailableReading(0, "fake", "fake", "bytes", "");
    Check(bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available ==
              110192ULL * 1048576,
          "PlanningBudgetAboveDedicatedIsAllocatable");
    fake.dxgi.dedicated_video_memory.available = false;
    const auto unavailable = bubo::cli::ReadPlanningMemoryBudgets(fake);
    Check(
        !unavailable.gpu_dedicated && !unavailable.gpu_dedicated_reason.empty(),
        "PlanningDedicatedUnavailableHasReason");
    Check(bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available ==
              110192ULL * 1048576,
          "PlanningBudgetDoesNotRequireDedicatedLabel");
    fake.dxgi.budget.available = false;
    Check(!bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available,
          "PlanningBudgetMissingBudgetUnknown");
    fake.dxgi.budget =
        AvailableReading(96.0 * kGiB, "fake", "fake", "bytes", "");
    fake.dxgi.current_usage.available = false;
    Check(!bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available,
          "PlanningBudgetMissingUsageUnknown");
    fake.dxgi.current_usage =
        AvailableReading(100.0 * kGiB, "fake", "fake", "bytes", "");
    Check(bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available == 0,
          "PlanningBudgetExhaustedClampsZero");
    fake.dxgi.budget =
        AvailableReading(64.0 * kGiB, "fake", "fake", "bytes", "");
    fake.dxgi.current_usage =
        AvailableReading(kGiB, "fake", "fake", "bytes", "");
    Check(bubo::cli::ReadPlanningMemoryBudgets(fake).gpu_available ==
              63ULL * 1073741824ULL,
          "PlanningBudgetRespectsDxgiBudget");
    std::cout << "PlanningBudgetCases: " << passed - passed_before
              << " passed, " << failed - failed_before << " failed\n";
  }
  {
    auto adapter = Fake::Adapter("Capacity AMD", 96ULL << 30);
    adapter.nonlocal_budget = 2ULL << 30;
    auto readings = bubo::cli::DxgiReadingsFromAdapters({adapter}, "", "");
    Check(
        readings.total_budget.available &&
            readings.total_budget.value.as_double() ==
                static_cast<double>(adapter.budget + *adapter.nonlocal_budget),
        "CapacityTotalIncludesMeasuredNonlocalBudget");
    Check(readings.budget.value.as_double() ==
              static_cast<double>(adapter.budget),
          "AdmissionBudgetRemainsLocalOnly");
    adapter.nonlocal_budget.reset();
    readings = bubo::cli::DxgiReadingsFromAdapters({adapter}, "", "");
    Check(!readings.total_budget.available &&
              !readings.total_budget.reason.empty(),
          "MissingNonlocalTotalIsUnknownWithReason");
    adapter.nonlocal_budget = 0;
    adapter.query_ok = false;
    readings = bubo::cli::DxgiReadingsFromAdapters({adapter}, "", "");
    Check(!readings.total_budget.available &&
              !readings.total_budget.reason.empty(),
          "MissingLocalTotalIsUnknownWithReason");
  }
  {  // the real probe never throws
    auto probe = bubo::cli::CreateSystemMachineProbe();
    const auto report = bubo::cli::MachineReport(*probe);
    CheckReadings(report, "system");
    Check(At(report, "read_only").as_bool(), "system probe report");
  }
  std::cout << "Machine adapter: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
