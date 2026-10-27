// Pure argument parsing only: no model, runtime, listener or file writes.
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>

#include "src/cli/diagnose/diagnose.h"
#include "src/cli/models_support.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/serve/serve_command.hpp"
#include "src/cli/status.hpp"
#include "src/cli/stop.hpp"
#include "src/cli/support_bundle.hpp"

int main(int argc, char** argv) {
  using bubo::cli::Value;
  if (argc < 2)
    return 2;
  const std::string command(argv[1]);
  const auto args = std::span<const char* const>(argv, argc).subspan(2);
  const char* local = std::getenv("LOCALAPPDATA");
  const auto runtime = std::filesystem::current_path() / "runtime";
  if (command == "run-prepare")
    return bubo::prepare::Run(args, local ? local : "", std::cout, std::cerr);
  if (command == "run-status")
    return bubo::cli::RunStatus(args, local ? local : "", runtime, std::cout,
                                std::cerr);
  if (command == "run-stop")
    return bubo::cli::RunStop(args, local ? local : "", std::cout, std::cerr);
  if (command == "run-models")
    return bubo::cli::RunModels(args, local ? local : "", std::cout, std::cerr);
  if (command == "run-diagnose")
    return gufo::cli::RunDiagnose(args);
  if (command == "run-bundle")
    return bubo::cli::RunSupportBundle(args, local ? local : "", runtime,
                                       std::cout, std::cerr);
  auto value = Value::object();
  std::string error;
  bool ok = true;
  try {
    if (command == "prepare") {
      const auto result = bubo::prepare::Parse(args, "local");
      ok = static_cast<bool>(result);
      error = result.error;
      const auto& o = result.options;
      value["step"] = static_cast<int>(o.step);
      value["root"] = std::filesystem::u8path(o.root).generic_string();
      value["mtp_model"] = o.mtp_model;
      value["mmproj"] = o.mmproj;
      value["cache_dir"] = o.cache_dir;
      value["runtime_dir"] = o.runtime_dir;
      value["catalogue_id"] = o.catalogue_id;
      value["model_set"] = static_cast<int>(o.model_set);
      value["model_set_chosen"] = o.model_set_chosen;
      value["resume"] = o.resume;
      value["yes"] = o.yes;
      value["verify"] = o.verify;
      value["offline"] = o.offline;
      value["help"] = o.help;
      value["machine"] = o.machine;
      value["plan_only"] = o.plan_only;
    } else if (command == "status") {
      const auto o = bubo::cli::ParseStatusOptions(args);
      value["mode"] = static_cast<int>(o.mode);
      value["watch"] = o.watch;
      value["instance_id"] = o.instance_id;
    } else if (command == "stop") {
      const auto o = bubo::cli::ParseStopOptions(args);
      value["instance_id"] = o.instance_id;
      value["wait_ms"] =
          o.requested_wait
              ? Value(static_cast<std::uint64_t>(*o.requested_wait))
              : Value();
    } else if (command == "models") {
      bubo::cli::ModelsCommandOptions o;
      ok = bubo::cli::ParseModelsCommand(args, o, error);
      value["verb"] = o.verb;
      value["machine"] = o.machine;
      value["help"] = o.help;
      value["model"] = o.model;
      value["context"] = o.context;
      value["sessions"] = o.sessions;
      value["models_root"] = o.models_root.generic_string();
      value["scan_folders"] = Value::array();
      for (const auto& path : o.scan_folders)
        value["scan_folders"].push_back(path.generic_string());
    } else if (command == "diagnose") {
      gufo::cli::DiagnoseOptions o;
      ok = gufo::cli::ParseDiagnoseOptions(args, o, error);
      value["json"] = o.json_mode;
      value["fingerprint"] = o.fingerprint_mode;
      value["help"] = o.help;
      value["validate_artifact"] = o.validate_artifact_path;
      value["benchmark"] = o.benchmark_name;
      value["backends"] = o.backends_csv;
      value["warmup"] = static_cast<std::uint64_t>(o.warmup);
      value["repetitions"] = static_cast<std::uint64_t>(o.repetitions);
      value["duration_ms"] = static_cast<std::uint64_t>(o.duration_ms);
      value["working_set_mib"] = o.working_set_mib;
      value["output"] = o.output_file;
      value["section"] = o.section;
    } else if (command == "bundle") {
      bubo::cli::SupportBundleOptions o;
      ok = bubo::cli::ParseSupportBundleOptions(args, o);
      value["json"] = o.json;
      value["preview"] = o.preview;
      value["bundle_option"] = o.bundle_option;
      value["path"] = o.path_text;
    } else if (command == "serve") {
      const auto o = gufo::cli::ParseServeCommandOptions(args);
      value["filtered"] = Value::array();
      for (const auto* arg : o.filtered)
        value["filtered"].push_back(arg);
      value["log_file"] =
          o.log_path ? Value(o.log_path->filename().generic_string()) : Value();
      value["log_max_bytes"] = o.log_max_bytes;
      value["log_retain"] = static_cast<std::uint64_t>(o.log_retain);
      value["supplied_log_flags"] = o.supplied_log_flags;
    } else if (command == "run-serve") {
      return gufo::cli::RunServeCommand(args, [](auto, auto*, const auto&) {
        // Reaching the runner is a test failure for the rejection-only cases.
        return 99;
      });
    } else {
      return 2;
    }
  } catch (const std::invalid_argument& e) {
    ok = false;
    error = e.what();
  }
  auto result = Value::object();
  result["ok"] = ok;
  result["error"] = error;
  if (ok)
    result["options"] = std::move(value);
  std::cout << result.dump() << '\n';
  return ok ? 0 : 2;
}
