#ifndef GUFO_CLI_DIAGNOSE_H_
#define GUFO_CLI_DIAGNOSE_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "src/cli/machine_warnings.hpp"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/report.h"

namespace gufo::cli {
struct DiagnoseOptions {
  bool json_mode = false, fingerprint_mode = false, help = false;
  std::string validate_artifact_path, benchmark_name;
  std::string backends_csv = "cpu,hip";
  std::uint32_t warmup = 3, repetitions = 10, duration_ms = 2000;
  std::string working_set_mib = "64", output_file, section = "all";
};
bool ParseDiagnoseOptions(std::span<const char* const> args,
                          DiagnoseOptions& options, std::string& error);

[[nodiscard]] diagnostics::DiagnosticReport CollectDiagnostics(
    const diagnostics::LinuxSysfs& sysfs = diagnostics::LinuxSysfs(),
    std::string_view section = "all");

/// Adds the GPU reservation and page file rows (WARN with a remedy when the
/// setting differs from the tested one) and the power mode as a PASS reading.
/// A setting that cannot be read adds no row.
void AddMachineSettingChecks(diagnostics::DiagnosticReport& report,
                             const bubo::cli::MachineWarningInputs& inputs);

int RunDiagnose(std::span<const char* const> args);
int DiagnosticExitCode(const diagnostics::DiagnosticReport& report);
int RunProbe(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_DIAGNOSE_H_
