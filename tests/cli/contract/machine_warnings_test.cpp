#include <iostream>
#include <string>

#include "src/cli/diagnose/diagnose.h"
#include "src/cli/machine_warnings.hpp"
#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/system_inventory.h"

namespace {
using bubo::cli::MachineWarningInputs;
using bubo::cli::PageFileSetting;

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

bool Has(const std::vector<bubo::cli::MachineWarning>& warnings,
         const char* check) {
  for (const auto& warning : warnings)
    if (warning.check == check)
      return true;
  return false;
}

PageFileSetting Fixed() {
  return {"C:\\pagefile.sys", false, 65536, 65536};
}

MachineWarningInputs Good() {
  MachineWarningInputs inputs;
  inputs.gpu_reservation_bytes = 102892924928ULL;  // DXGI value for 96 GB
  inputs.page_files = std::vector<PageFileSetting>{Fixed()};
  inputs.effective_power_mode = 4;
  return inputs;
}

const gufo::diagnostics::DiagnosticCheck* FindCheck(
    const gufo::diagnostics::DiagnosticReport& report, const char* name) {
  for (const auto& check : report.Checks())
    if (check.name == name)
      return &check;
  return nullptr;
}

}  // namespace

int main() {
  // PagingFiles registry lines.
  {
    const auto fixed = bubo::cli::ParsePagingFilesEntry("C:\\pagefile.sys 65536 65536");
    Check(fixed && !fixed->system_managed && fixed->path == "C:\\pagefile.sys" &&
              fixed->initial_mib == 65536 && fixed->maximum_mib == 65536,
          "fixed page file line parses");
    const auto range = bubo::cli::ParsePagingFilesEntry("C:\\pagefile.sys 1024 65536");
    Check(range && !range->system_managed && range->initial_mib == 1024 &&
              range->maximum_mib == 65536,
          "range page file line parses");
    const auto managed = bubo::cli::ParsePagingFilesEntry("?:\\pagefile.sys");
    Check(managed && managed->system_managed && managed->path == "?:\\pagefile.sys",
          "system-managed line has no sizes");
    const auto zero = bubo::cli::ParsePagingFilesEntry("C:\\pagefile.sys 0 0");
    Check(zero && zero->system_managed, "0 0 means system managed");
    const auto spaced =
        bubo::cli::ParsePagingFilesEntry("D:\\My Files\\pagefile.sys 100 100\r\n");
    Check(spaced && spaced->path == "D:\\My Files\\pagefile.sys" &&
              spaced->initial_mib == 100 && !spaced->system_managed,
          "a path with a space keeps its path");
    Check(!bubo::cli::ParsePagingFilesEntry("  "), "a blank line is no entry");
  }

  // Warnings.
  {
    Check(bubo::cli::EvaluateMachineWarnings(Good()).empty(),
          "the tested machine has no warning");
    Check(bubo::cli::EvaluateMachineWarnings({}).empty(),
          "unreadable settings give no warning");

    auto small = Good();
    small.gpu_reservation_bytes = 17000000000ULL;
    const auto small_warnings = bubo::cli::EvaluateMachineWarnings(small);
    Check(small_warnings.size() == 1 && Has(small_warnings, "gpu_reservation") &&
              small_warnings[0].message.find("96 GB") != std::string::npos,
          "a 16 GB reservation warns and names the tested 96 GB");

    auto managed = Good();
    managed.page_files = std::vector<PageFileSetting>{{"?:\\pagefile.sys", true, 0, 0}};
    Check(Has(bubo::cli::EvaluateMachineWarnings(managed), "page_file"),
          "a system-managed page file warns");
    auto range = Good();
    range.page_files = std::vector<PageFileSetting>{{"C:\\pagefile.sys", false, 1024, 65536}};
    Check(Has(bubo::cli::EvaluateMachineWarnings(range), "page_file"),
          "a growing page file warns");
    auto none = Good();
    none.page_files = std::vector<PageFileSetting>{};
    Check(Has(bubo::cli::EvaluateMachineWarnings(none), "page_file"),
          "no page file warns");

    // The Windows power mode is a reading only: it never warns.
    for (const std::int64_t mode : {1, 2, 3, 5}) {
      auto other = Good();
      other.effective_power_mode = mode;
      Check(bubo::cli::EvaluateMachineWarnings(other).empty(),
            "a power mode other than 4 gives no warning");
    }

    MachineWarningInputs bad;
    bad.gpu_reservation_bytes = 17000000000ULL;
    bad.page_files = std::vector<PageFileSetting>{{"?:\\pagefile.sys", true, 0, 0}};
    bad.effective_power_mode = 2;
    const auto bad_warnings = bubo::cli::EvaluateMachineWarnings(bad);
    Check(bad_warnings.size() == 2, "two differing settings give two warnings");
    const auto line = bubo::cli::MachineWarningLogLine(bad_warnings);
    Check(line.find("differs=gpu_reservation,page_file") != std::string::npos &&
              line.find("power_mode") == std::string::npos,
          "the serve line lists the settings and leaves out the power mode");
    Check(bubo::cli::MachineWarningLogLine({}).empty(),
          "no warning gives no serve line");
  }

  // Diagnose rows: WARN rows, PASS rows, and an exit code that stays 0.
  {
    gufo::diagnostics::DiagnosticReport report;
    gufo::cli::AddMachineSettingChecks(report, Good());
    Check(report.Checks().size() == 3 && report.Status() == gufo::diagnostics::DiagnosticStatus::kPass,
          "a matching machine adds three PASS rows");

    MachineWarningInputs bad;
    bad.gpu_reservation_bytes = 17000000000ULL;
    bad.page_files = std::vector<PageFileSetting>{{"?:\\pagefile.sys", true, 0, 0}};
    bad.effective_power_mode = 5;
    gufo::diagnostics::DiagnosticReport warned;
    gufo::cli::AddMachineSettingChecks(warned, bad);
    const auto* reservation = FindCheck(warned, "gpu_reservation");
    const auto* page = FindCheck(warned, "page_file");
    const auto* mode = FindCheck(warned, "power_mode");
    Check(reservation && page &&
              reservation->status == gufo::diagnostics::DiagnosticStatus::kWarn &&
              page->status == gufo::diagnostics::DiagnosticStatus::kWarn,
          "differing settings add WARN rows");
    Check(mode && mode->status == gufo::diagnostics::DiagnosticStatus::kPass &&
              mode->message.find("GameMode (5)") != std::string::npos &&
              mode->details.count("remedy") == 0,
          "the power mode is a reading with its name and no remedy");
    Check(reservation && reservation->details.count("remedy") == 1,
          "a WARN row carries its remedy");
    Check(warned.Status() == gufo::diagnostics::DiagnosticStatus::kWarn &&
              gufo::cli::DiagnosticExitCode(warned) == 0,
          "warnings never fail the command");

    gufo::diagnostics::DiagnosticReport unreadable;
    gufo::cli::AddMachineSettingChecks(unreadable, {});
    Check(unreadable.Checks().empty(), "unreadable settings add no row");
  }

  // The Linux kernel driver item is not applicable on Windows (it failed the
  // whole diagnose report there).
  {
    using gufo::diagnostics::CompatibilityVerdict;
    gufo::diagnostics::SystemInventory inventory;
    inventory.cpu.architecture = "x86_64";
    inventory.gpu.architecture = "gfx1151";
    const auto find_driver = [](const gufo::diagnostics::CompatibilityReport& report) {
      for (const auto& item : report.items)
        if (item.component == "kernel_driver")
          return item;
      return gufo::diagnostics::CompatibilityItem{};
    };

    inventory.toolchain.amdgpu_status = "not applicable (Windows)";
    const auto windows = gufo::diagnostics::EvaluateCompatibility(inventory, "all");
    Check(find_driver(windows).verdict == CompatibilityVerdict::kNotApplicable &&
              find_driver(windows).remediation_hint == "None" &&
              windows.overall_verdict == CompatibilityVerdict::kSupported,
          "the Linux driver item is not applicable on Windows and does not fail");
    gufo::diagnostics::DiagnosticReport report;
    report.SetCompatibility(windows);
    Check(report.Status() == gufo::diagnostics::DiagnosticStatus::kPass &&
              gufo::cli::DiagnosticExitCode(report) == 0,
          "a Windows report with a supported GPU passes");
    Check(gufo::diagnostics::ToString(CompatibilityVerdict::kNotApplicable) ==
              "not_applicable",
          "the verdict has a plain name");

    inventory.toolchain.amdgpu_status = "loaded";
    Check(find_driver(gufo::diagnostics::EvaluateCompatibility(inventory, "all")).verdict ==
              CompatibilityVerdict::kSupported,
          "a loaded Linux driver is supported");
    inventory.toolchain.amdgpu_status = "not loaded";
    const auto missing = gufo::diagnostics::EvaluateCompatibility(inventory, "all");
    Check(find_driver(missing).verdict == CompatibilityVerdict::kUnsupported &&
              missing.overall_verdict == CompatibilityVerdict::kUnsupported,
          "a missing Linux driver still fails");
  }

  std::cout << "machine_warnings_test: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
