#include "src/cli/diagnose/diagnose.h"

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>

#include "src/cli/machine_output.hpp"

int main(int argc, char** argv) {
  if (argc == 2) {
    std::ofstream output(argv[1], std::ios::binary);
    bubo::cli::MachineWriter writer(output);
    using namespace gufo::diagnostics;
    unsigned passed = 0;
    for (const auto status : {DiagnosticStatus::kPass, DiagnosticStatus::kWarn,
                              DiagnosticStatus::kFail}) {
      DiagnosticReport report;
      report.SetTimestamp("2026-10-13T18:00:00Z");
      report.SetEngineRevision("fixture");
      report.AddCheck({"fixture", status, "Synthetic diagnostic check.", {}});
      auto json = gufo::json::parse(report.ToJson());
      json["schema_version"] = 1;
      if (gufo::cli::DiagnosticExitCode(report) !=
              (status == DiagnosticStatus::kFail ? 1 : 0) ||
          json.member_str("schemaVersion") != "1.0.0")
        return 1;
      if (status == DiagnosticStatus::kFail)
        json["error"] = bubo::cli::Error(
            "diagnostic_failed", "diagnose", "checks", false,
            "Read the failed checks. Apply the reported remedy.");
      writer.Write("diagnose", json, &json);
      ++passed;
    }
    std::cout << "Diagnose pass/warn/fail and exits: " << passed
              << " passed, 0 failed\n";
    return 0;
  }
  return gufo::cli::RunDiagnose(
      std::span<const char* const>(argv, argc).subspan(1));
}
