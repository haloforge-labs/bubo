#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/banner.hpp"
#include "src/cli/bench/bench.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/cli_info.hpp"
#include "src/cli/command_table.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/eval/eval.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/model_families.hpp"
#include "src/cli/models_support.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/status.hpp"
#include "src/cli/stop.hpp"
#include "src/cli/support_bundle.hpp"
#include "src/cli/switch_listing.hpp"
#include "src/cli/transcribe/transcribe.hpp"
#include "src/cli/video/video.hpp"

namespace gufo::cli::commands {
int Serve(std::span<const char* const> args, const CommandContext&) {
  return RunServe(args.subspan(1));
}
int Prompt(std::span<const char* const> args, const CommandContext&) {
  return RunPrompt(args.subspan(1));
}
int Chat(std::span<const char* const> args, const CommandContext&) {
  return RunChat(args.subspan(1));
}
int Bench(std::span<const char* const> args, const CommandContext&) {
  return RunBench(args.subspan(1));
}
int Eval(std::span<const char* const> args, const CommandContext&) {
  return RunEval(args.subspan(1));
}
int Video(std::span<const char* const> args, const CommandContext&) {
  return RunVideo(args.subspan(1));
}
int Transcribe(std::span<const char* const> args, const CommandContext&) {
  return RunTranscribe(args.subspan(1));
}
int Probe(std::span<const char* const> args, const CommandContext&) {
  return RunProbe(args.subspan(1));
}
int Diagnose(std::span<const char* const> args, const CommandContext&) {
  return RunDiagnose(args);
}
int Models(std::span<const char* const> args, const CommandContext&) {
  return bubo::cli::RunModels(args.subspan(1),
                              bubo::cli::EnvironmentUtf8("LOCALAPPDATA"),
                              std::cout, std::cerr);
}
}  // namespace gufo::cli::commands

namespace {
int RunHelpTopic(std::span<const char* const> options) {
  if (options.size() <= 1) {
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 0;
  }
  const auto* command =
      gufo::cli::FindCommand<gufo::cli::CommandLayer::Engine>(options[1]);
  if (!command || !command->help_topic) {
    std::cerr << "Error: unknown help topic '" << options[1] << "'\n";
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 2;
  }
  const gufo::cli::CommandContext context;
  if (command->name == "serve" && options.size() > 2) {
    const std::array<const char*, 3> args{options[1], options[2], "--help"};
    return command->run(args, context);
  }
  const std::array<const char*, 2> args{options[1], "--help"};
  return command->run(args, context);
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 2;
  }

  const std::string_view program_name = gufo::cli::kProductName;
  const auto options = args.subspan(1);
  if (!options.empty() && bubo::cli::MachineRequested(options) &&
      (std::string_view(options[0]).starts_with("-") ||
       std::string_view(options[0]) == "help")) {
    bubo::cli::MachineWriter writer(std::cout);
    const std::string_view first(options[0]);
    if (options.size() != 2 || (first != "--help" && first != "-h" &&
                                first != "--version" && first != "-v")) {
      writer.Write("error",
                   bubo::cli::Error("invalid_arguments", "cli", "arguments",
                                    false, "Run bubo status --json."));
      return 2;
    }
    writer.Write("capabilities", bubo::cli::Capabilities());
    return 0;
  }

  if (options.empty()) {
    gufo::cli::PrintGlobalHelp(program_name);
    return 0;
  }

  const std::string_view first_arg = options[0];
  const auto* command =
      gufo::cli::FindCommand<gufo::cli::CommandLayer::Engine>(first_arg);
  const gufo::cli::CommandContext context{.load_engine = {},
                                          .help = RunHelpTopic};
  if (const auto result = gufo::cli::RunCommandVariant(options))
    return *result;
  if (first_arg == "--version" || first_arg == "-v") {
    gufo::cli::PrintGlobalVersion();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    gufo::cli::PrintGlobalHelp(program_name);
    return 0;
  }

  if (command && (command->name == "help" || command->name == "prepare"))
    return command->run(options, context);

  gufo::cli::PrintStartupBanner(gufo::cli::kProductVersion, options);

  if (bubo::cli::MachineRequested(options) &&
      (!command || !command->machine_supported)) {
    bubo::cli::MachineWriter writer(std::cout);
    writer.Write(
        "error",
        bubo::cli::Error("unsupported", "cli", "dispatch", false,
                         "Use a command listed in the status capabilities."));
    return first_arg == "serve" || first_arg == "stop" ? 18 : 2;
  }

  if (command)
    return command->run(options, context);

  // Legacy alias is retained for the optional upstream-branded build only.
  if (gufo::cli::kProductName == "gufo" &&
      (std::string_view(args[0]).ends_with("gufo-server") ||
       std::string_view(args[0]).ends_with("gufo-server.exe"))) {
    return gufo::cli::RunServe(options);
  }

  std::cerr << "Error: unknown command '" << first_arg << "'\n";
  gufo::cli::PrintGlobalHelp(program_name);
  return 2;
}

}  // namespace

#if defined(BUBO_ENGINE_DLL)
extern "C"
    __declspec(dllexport) void BuboSetMachineOutputV1(std::intptr_t handle) {
  bubo::cli::ConfigureMachineOutputHandle(handle);
}
extern "C" __declspec(dllexport) const char* BuboEngineIdentityV1() {
  static const std::string identity = gufo::cli::FormatVersion(
      gufo::cli::kProductName, gufo::cli::kProductVersion, gufo::cli::kRevision,
      gufo::cli::kEngineVersion);
  return identity.c_str();
}
extern "C" __declspec(dllexport) int BuboRunCommandV1(int argc,
                                                      const char* const* argv) {
#else
int main(int argc, char* argv[]) {
#endif
  const std::span<const char* const> args(argv, static_cast<std::size_t>(argc));
#if !defined(BUBO_ENGINE_DLL)
  std::unique_ptr<bubo::cli::MachineWriter> process_output;
  if (bubo::cli::MachineRequested(args))
    process_output =
        std::make_unique<bubo::cli::MachineWriter>(std::cout, true);
#endif
  return run(args);
}
