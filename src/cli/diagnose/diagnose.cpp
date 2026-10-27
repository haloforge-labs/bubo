#include "src/cli/diagnose/diagnose.h"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/machine_warnings.hpp"
#include "src/core/diagnostics/allocation_benchmark.h"
#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/bandwidth.h"
#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::cli {

namespace {

std::uint32_t ParseDiagnosticInteger(std::string_view text) {
  std::uint32_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument(
        "diagnostic count must be an unsigned 32-bit integer");
  return value;
}

void PrintDiagnoseHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " diagnose [OPTIONS]\n\n"
      << "Run non-interactive system, hardware, and benchmark diagnostics for "
         "Strix Halo.\n\n"
      << "Options:\n"
      << "  --json                       Emit structured JSON output "
         "conforming "
         "to diagnostics schema v1.0.0\n"
      << "  --fingerprint                Generate canonical machine "
         "fingerprint "
         "and SHA-256 ID\n"
      << "  --validate-artifact <path>   Validate a diagnostic/benchmark "
         "artifact against schema & fingerprint\n"
      << "  --benchmark <name>           Run benchmark suite: bandwidth, "
         "allocation\n"
      << "  --backends <csv>             Backends to benchmark (default: "
         "cpu,hip)\n"
      << "  --warmup <n>                 Number of warmup iterations (default: "
         "3)\n"
      << "  --repetitions <n>            Minimum benchmark repetitions "
         "(default: 10)\n"
      << "  --duration-ms <ms>           Minimum duration per bandwidth path "
         "in ms "
         "(default: 2000)\n"
      << "  --working-set-mib <csv>      Allocation diagnostic sizes in MiB "
         "(default: 64)\n"
      << "  --output <path>              Write command output to specified "
         "file\n"
      << "  --section <name>             Diagnostic section to run: all, "
         "inventory, platform, gpu\n"
      << "  -h, --help                   Print help information\n";
}

void OutputContent(std::string_view content, std::string_view output_file,
                   bubo::cli::MachineWriter* writer = nullptr,
                   bool failed = false) {
  if (writer) {
    auto document = gufo::json::parse(content);
    document["schema_version"] = 1;
    if (document.contains("checks")) {
      auto probe = bubo::cli::CreateSystemMachineProbe();
      document["machine"] = bubo::cli::MachineReport(*probe);
    }
    if (failed || document.member_str("status") == "FAIL")
      document["error"] = bubo::cli::Error(
          "diagnostic_failed", "diagnose", "checks", false,
          "Read the failed checks. Apply the reported remedy.");
    // Retain the advertised diagnostic keys at the top level. The payload
    // contains the same v1.0.0 report for shared machine clients.
    if (!output_file.empty()) {
      std::ofstream file(std::filesystem::u8path(output_file),
                         std::ios::binary);
      if (!file || !(file << document.dump() << '\n'))
        throw std::runtime_error("cannot write diagnostic output file");
    }
    writer->Write(
        document.contains("checks") ? "diagnose" : "diagnose_artifact",
        document, &document);
    return;
  }
  std::cout << content;
  if (!output_file.empty()) {
    std::ofstream out{std::string(output_file)};
    if (out.is_open()) {
      out << content;
    } else {
      std::cerr << "Warning: could not write output to file: " << output_file
                << "\n";
    }
  }
}

std::vector<std::string> SplitCommaSeparated(std::string_view csv) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start < csv.size()) {
    const auto end = csv.find(',', start);
    if (end == std::string_view::npos) {
      result.emplace_back(csv.substr(start));
      break;
    }
    result.emplace_back(csv.substr(start, end - start));
    start = end + 1;
  }
  return result;
}

}  // namespace

diagnostics::DiagnosticReport CollectDiagnostics(
    const diagnostics::LinuxSysfs& sysfs, std::string_view section) {
  diagnostics::DiagnosticReport report;

  const auto inventory = diagnostics::CollectSystemInventory(sysfs);
  const auto compatibility =
      diagnostics::EvaluateCompatibility(inventory, section);
  const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);

  report.SetInventory(inventory);
  report.SetCompatibility(compatibility);
  report.SetFingerprint(fingerprint);

  // 1. Platform Check
  if (section == "all" || section == "platform" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "platform";
    if (inventory.cpu.architecture == "x86_64") {
      check.status = diagnostics::DiagnosticStatus::kPass;
#ifdef _WIN32
      check.message = "Supported architecture: Windows x86-64";
      check.details["targetArchitecture"] = "x86_64-windows";
#else
      check.message = "Supported architecture: Linux x86-64";
      check.details["targetArchitecture"] = "x86_64-linux";
#endif
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported target platform: only Linux x86-64 is supported";
      report.AddError("Unsupported platform architecture detected");
    }
    report.AddCheck(std::move(check));
  }

  // 2. System, CPU & Memory Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck sys_check;
    sys_check.name = "system";
    sys_check.status = diagnostics::DiagnosticStatus::kPass;
    sys_check.message = "Host operating system and kernel identified";
    sys_check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    report.AddCheck(std::move(sys_check));

    diagnostics::DiagnosticCheck cpu_check;
    cpu_check.name = "cpu";
    cpu_check.status = diagnostics::DiagnosticStatus::kPass;
    cpu_check.message = "CPU processor topology queried";
    cpu_check.details["modelName"] = inventory.cpu.model_name;
    cpu_check.details["logicalCores"] =
        std::to_string(inventory.cpu.logical_cores);
    cpu_check.details["physicalCores"] =
        std::to_string(inventory.cpu.physical_cores);
    report.AddCheck(std::move(cpu_check));

    diagnostics::DiagnosticCheck mem_check;
    mem_check.name = "memory";
    mem_check.status = diagnostics::DiagnosticStatus::kPass;
    mem_check.message = "Host unified memory pool identified";
    mem_check.details["totalBytes"] =
        std::to_string(inventory.memory.total_bytes);
    mem_check.details["availableBytes"] =
        std::to_string(inventory.memory.available_bytes);
    mem_check.details["memoryType"] = inventory.memory.memory_type;
    report.AddCheck(std::move(mem_check));
  }

  // 3. GPU Check
  if (section == "all" || section == "gpu" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "gpu";
    check.details["name"] = inventory.gpu.name;
    check.details["architecture"] = inventory.gpu.architecture;
    check.details["computeUnits"] = std::to_string(inventory.gpu.compute_units);
    check.details["driverName"] = inventory.gpu.driver_name;
    check.details["pciId"] = inventory.gpu.pci_id;
    check.details["powerMode"] = inventory.gpu.power_mode;

    if (inventory.gpu.architecture == "gfx1151") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = "Detected AMD Strix Halo GPU (gfx1151)";
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported GPU architecture: expected gfx1151, detected " +
          inventory.gpu.architecture;
      report.AddError("Unsupported GPU architecture: " +
                      inventory.gpu.architecture);
    }
    report.AddCheck(std::move(check));
  }

  // 4. Toolchain Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "toolchain";
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message = "Toolchain and driver stack verified";
    check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    check.details["amdgpuStatus"] = inventory.toolchain.amdgpu_status;
    check.details["rocmVersion"] = inventory.toolchain.rocm_version;
    report.AddCheck(std::move(check));
  }

  return report;
}

void AddMachineSettingChecks(diagnostics::DiagnosticReport& report,
                             const bubo::cli::MachineWarningInputs& inputs) {
  const auto warnings = bubo::cli::EvaluateMachineWarnings(inputs);
  const auto find = [&](std::string_view name) -> const bubo::cli::MachineWarning* {
    for (const auto& warning : warnings)
      if (warning.check == name)
        return &warning;
    return nullptr;
  };
  // A setting that cannot be read adds no row. A differing setting adds a
  // WARN row; the exit code stays 0 because a warning never fails the command.
  const auto add = [&](std::string name, bool readable, std::string pass_message,
                       std::map<std::string, std::string> details) {
    if (!readable)
      return;
    diagnostics::DiagnosticCheck check;
    check.name = name;
    check.details = std::move(details);
    if (const auto* warning = find(name)) {
      check.status = diagnostics::DiagnosticStatus::kWarn;
      check.message = warning->message;
      check.details["remedy"] = warning->remedy;
      report.AddWarning(warning->message);
    } else {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = std::move(pass_message);
    }
    report.AddCheck(std::move(check));
  };
  std::map<std::string, std::string> reservation_details;
  if (inputs.gpu_reservation_bytes)
    reservation_details["dedicatedVideoMemoryBytes"] =
        std::to_string(*inputs.gpu_reservation_bytes);
  add("gpu_reservation", inputs.gpu_reservation_bytes.has_value(),
      inputs.is_96gb_strix_halo ? bubo::cli::k96GbReservationPass
                               : "GPU memory reservation matches the tested 96 GB setting",
      std::move(reservation_details));
  std::map<std::string, std::string> page_details;
  if (inputs.page_files) {
    std::string files;
    for (const auto& file : *inputs.page_files) {
      if (!files.empty())
        files += "; ";
      files += file.path;
      files += file.system_managed
                   ? " (system managed)"
                   : " (" + std::to_string(file.initial_mib) + "-" +
                         std::to_string(file.maximum_mib) + " MiB)";
    }
    page_details["pageFiles"] = files.empty() ? "none" : files;
  }
  add("page_file", inputs.page_files.has_value(),
      "Page file is a fixed size", std::move(page_details));
  // The power mode is a reading, not a warning: the Windows value does not
  // reliably follow the profile that the docs tell people to set.
  if (inputs.effective_power_mode) {
    const auto raw = *inputs.effective_power_mode;
    const auto name = bubo::cli::WindowsEffectivePowerModeName(raw);
    diagnostics::DiagnosticCheck check;
    check.name = "power_mode";
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message = "Windows effective power mode: " +
                    (name ? *name : std::string("unknown")) + " (" +
                    std::to_string(raw) + ")";
    check.details["effectivePowerMode"] = std::to_string(raw);
    if (name)
      check.details["effectivePowerModeName"] = *name;
    report.AddCheck(std::move(check));
  }
}

bool ParseDiagnoseOptions(std::span<const char* const> args,
                          DiagnoseOptions& options, std::string& error) {
  std::size_t first = 0;
  if (!args.empty()) {
    const std::string_view name = args[0];
    if (name == kProductName || name.ends_with("/" + std::string(kProductName)))
      ++first;
  }
  if (first < args.size() && (std::string_view(args[first]) == "diagnose" ||
                              std::string_view(args[first]) == "probe"))
    ++first;
  ArgParser parser(ProductCommand("diagnose"));
  parser.SetErrorHandler([](auto kind, std::string_view flag) {
    if (kind != ArgParser::ParseError::MissingValue)
      return "unknown diagnose option '" + std::string(flag) + "'";
    std::string hint = "an argument";
    if (flag == "--validate-artifact")
      hint = "a path argument";
    else if (flag == "--warmup" || flag == "--repetitions" ||
             flag == "--duration-ms")
      hint = "an integer argument";
    else if (flag == "--working-set-mib")
      hint = "a comma-separated size list";
    else if (flag == "--output")
      hint = "a file path argument";
    return std::string(flag) + " requires " + hint;
  });
  parser.AddFlag("", "--json", "Machine output", "Output", &options.json_mode);
  parser.AddFlag("", "--jsonl", "Machine stream", "Output", &options.json_mode);
  parser.AddFlag("", "--fingerprint", "Machine fingerprint", "Diagnostics",
                 &options.fingerprint_mode);
  parser.AddOption("", "--validate-artifact", "PATH", "Validate artifact",
                   "Diagnostics", &options.validate_artifact_path);
  parser.AddOption("", "--benchmark", "NAME", "Benchmark", "Diagnostics",
                   &options.benchmark_name);
  parser.AddOption("", "--backends", "CSV", "Backends", "Diagnostics",
                   &options.backends_csv);
  parser.AddOption("", "--working-set-mib", "CSV", "Working set", "Diagnostics",
                   &options.working_set_mib);
  parser.AddOption("", "--output", "PATH", "Output file", "Output",
                   &options.output_file);
  parser.AddOption("", "--section", "NAME", "Diagnostic section", "Diagnostics",
                   &options.section);
  const auto count = [&](std::string_view flag, std::uint32_t* target) {
    parser.AddCustomOption("", flag, "N", "Count", "Diagnostics",
                           [target](auto, auto value, auto*) {
                             *target = ParseDiagnosticInteger(value);
                             return true;
                           });
  };
  count("--warmup", &options.warmup);
  count("--repetitions", &options.repetitions);
  count("--duration-ms", &options.duration_ms);
  const bool parsed = parser.Parse(args.subspan(first), &error);
  options.help = parser.IsHelpRequested();
  return parsed;
}

static int RunDiagnoseImpl(std::span<const char* const> args,
                           bubo::cli::MachineWriter* writer) {
  bool json_mode = writer != nullptr;
  const auto invalid = [&] {
    if (writer)
      writer->Write("error", bubo::cli::Error("invalid_arguments", "diagnose",
                                              "arguments", false,
                                              "Run bubo diagnose --help."));
    return 2;
  };
  DiagnoseOptions options;
  options.json_mode = json_mode;
  std::string parse_error;
  if (!ParseDiagnoseOptions(args, options, parse_error)) {
    std::cerr << "Error: " << parse_error << '\n';
    if (!writer)
      PrintDiagnoseHelp(gufo::cli::kProductName);
    return invalid();
  }
  if (options.help) {
    if (writer)
      writer->Write("capabilities", bubo::cli::Capabilities());
    else
      PrintDiagnoseHelp(gufo::cli::kProductName);
    return 0;
  }
  json_mode = options.json_mode;
  auto& fingerprint_mode = options.fingerprint_mode;
  auto& validate_artifact_path = options.validate_artifact_path;
  auto& benchmark_name = options.benchmark_name;
  auto& backends_csv = options.backends_csv;
  auto& warmup = options.warmup;
  auto& repetitions = options.repetitions;
  auto& duration_ms = options.duration_ms;
  auto& working_set_mib = options.working_set_mib;
  auto& output_file = options.output_file;
  auto& section = options.section;
  if (section != "all" && section != "inventory" && section != "platform" &&
      section != "gpu") {
    std::cerr << "Error: unknown diagnostic section '" << section << "'\n";
    return invalid();
  }

  // Handle artifact validation mode
  if (!validate_artifact_path.empty()) {
    const auto result =
        diagnostics::ValidateArtifactFile(validate_artifact_path);
    const std::string content = json_mode ? result.ToJson() : result.ToHuman();
    OutputContent(content, output_file, writer, !result.is_valid);
    return result.is_valid ? 0 : 1;
  }

  // Handle benchmark execution mode
  if (!benchmark_name.empty()) {
    if (benchmark_name != "bandwidth" && benchmark_name != "allocation" &&
        benchmark_name != "hip-allocation") {
      std::cerr << "Error: unsupported benchmark '" << benchmark_name
                << "'. Supported: bandwidth, allocation\n";
      return invalid();
    }

    const auto inventory =
        diagnostics::CollectSystemInventory(diagnostics::LinuxSysfs());
    const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);

    if (benchmark_name == "bandwidth") {
      diagnostics::BandwidthOptions bw_options;
      bw_options.backends = SplitCommaSeparated(backends_csv);
      if (bw_options.backends.empty()) {
        std::cerr << "Error: --backends requires at least one backend\n";
        return invalid();
      }
      for (const auto& backend : bw_options.backends) {
        if (backend != "cpu" && backend != "hip") {
          std::cerr << "Error: unknown bandwidth backend '" << backend << "'\n";
          return invalid();
        }
      }
      bw_options.warmup = warmup;
      bw_options.repetitions = repetitions;
      bw_options.duration_ms = duration_ms;
      const auto report =
          diagnostics::RunBandwidthBenchmark(bw_options, fingerprint);
      const std::string content =
          json_mode ? report.ToJson() : report.ToHuman();
      OutputContent(content, output_file, writer);
      return 0;
    }

    diagnostics::AllocationBenchmarkOptions allocation_options;
    allocation_options.warmup = warmup;
    allocation_options.repetitions = repetitions;
    if (allocation_options.repetitions == 0) {
      std::cerr << "Error: --repetitions must be at least 1 for the allocation "
                   "diagnostic\n";
      return invalid();
    }
    std::string size_error;
    if (!diagnostics::ParseWorkingSetSizes(
            working_set_mib, &allocation_options.working_set_bytes,
            &size_error)) {
      std::cerr << "Error: " << size_error << "\n";
      return invalid();
    }
    const auto report = diagnostics::RunHipAllocationBenchmark(
        allocation_options, inventory, fingerprint);
    const std::string content = json_mode ? report.ToJson() : report.ToHuman();
    OutputContent(content, output_file, writer, !report.Success());
    return report.Success() ? 0 : 1;
  }

  // Handle fingerprint mode
  if (fingerprint_mode) {
    const auto inventory =
        diagnostics::CollectSystemInventory(diagnostics::LinuxSysfs());
    const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);
    const std::string content =
        json_mode ? fingerprint.ToJson() : fingerprint.ToHuman();
    OutputContent(content, output_file, writer);
    return 0;
  }

  // Handle default diagnostics report
  auto report = CollectDiagnostics(diagnostics::LinuxSysfs(), section);
  if (section == "all" || section == "inventory" || section == "gpu") {
    auto probe = bubo::cli::CreateSystemMachineProbe();
    AddMachineSettingChecks(report, bubo::cli::ReadMachineWarningInputs(*probe));
  }
  const std::string content = json_mode ? report.ToJson() : report.ToHuman();
  OutputContent(content, output_file, writer);

  return DiagnosticExitCode(report);
}

int DiagnosticExitCode(const diagnostics::DiagnosticReport& report) {
  return report.Status() == diagnostics::DiagnosticStatus::kFail ? 1 : 0;
}

int RunDiagnose(std::span<const char* const> args) {
  std::unique_ptr<bubo::cli::MachineWriter> writer;
  if (bubo::cli::MachineRequested(args))
    writer = std::make_unique<bubo::cli::MachineWriter>(std::cout);
  try {
    const auto mode = bubo::cli::ParseMachineMode(args);
    if (writer && mode == bubo::cli::MachineMode::JsonLines)
      writer->Write("capabilities", bubo::cli::Capabilities());
    return RunDiagnoseImpl(args, writer.get());
  } catch (const std::invalid_argument& error) {
    if (writer)
      writer->Write("error", bubo::cli::Error("invalid_arguments", "diagnose",
                                              "arguments", false,
                                              "Run bubo diagnose --help."));
    else
      std::cerr << "Error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    if (writer)
      writer->Write("error",
                    bubo::cli::Error(
                        "unexpected_failure", "diagnose", "diagnostics", false,
                        "Run diagnose again. Check the output folder."));
    else
      std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}

#if defined(ENGINE_ENABLE_HIP)
static bool IsHipRequired() {
  const char* val = std::getenv("GUFO_REQUIRE_HIP");
  if (!val) {
    val = std::getenv("GUFO_REQUIRE_GPU");
  }
  if (!val) {
    val = std::getenv("GUFO_REQUIRE_GFX1151");
  }
  return val && std::string_view(val) != "0" &&
         std::string_view(val) != "false";
}

static void CheckGpu() {
  std::printf("ROCm version: %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
              HIP_VERSION_PATCH);

  int devCount = 0;
  hipError_t err = hipGetDeviceCount(&devCount);
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error on hipGetDeviceCount: %s\n",
                 hipGetErrorName(err));
    if (IsHipRequired()) {
      std::exit(1);
    } else {
      std::printf("SKIP: HIP device discovery failed\n");
      return;
    }
  }

  std::printf("HIP device count: %d\n", devCount);
  if (devCount == 0) {
    std::printf(
        "No HIP device found — is kfd loaded / device in render group?\n");
    if (IsHipRequired()) {
      std::exit(1);
    }
    return;
  }

  for (int i = 0; i < devCount; ++i) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
      continue;
    }
    std::printf("device[%d]: %s\n", i, prop.name);
    std::printf("  arch:        %s\n", prop.gcnArchName);
    std::printf("  computeCap:  %d.%d\n", prop.major, prop.minor);
    std::printf("  totalMem:    %.2f GiB (unified)\n",
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
}
#endif

int RunProbe(std::span<const char* const> args) {
  ArgParser parser(gufo::cli::ProductCommand("probe"),
                   "Probe GPU availability.");
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  std::printf("Hello, Strix Halo!\n");

#if defined(ENGINE_ENABLE_HIP)
  std::printf("Build: CMake, C++20, ROCm/HIP enabled\n");
  CheckGpu();
#else
  std::printf("Build: CMake, C++20, CPU-only (no ROCm)\n");
#endif

  std::printf("OK\n");
  return 0;
}

}  // namespace gufo::cli
