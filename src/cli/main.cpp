#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/banner.hpp"
#include "src/cli/bench/bench.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/cli_info.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/eval/eval.hpp"
#include "src/cli/model_families.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/transcribe/transcribe.hpp"
#include "src/cli/video/video.hpp"

namespace {

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 2;
  }

  const std::string_view program_name = gufo::cli::kProductName;
  const auto options = args.subspan(1);

  if (options.empty()) {
    gufo::cli::PrintGlobalHelp(program_name);
    return 0;
  }

  const std::string_view first_arg = options[0];
  if (first_arg == "--version" || first_arg == "-v") {
    gufo::cli::PrintGlobalVersion();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    gufo::cli::PrintGlobalHelp(program_name);
    return 0;
  }

  if (first_arg == "help") {
    if (options.size() <= 1) {
      gufo::cli::PrintGlobalHelp(program_name);
      return 0;
    }
    const std::string_view sub = options[1];
    const std::array<const char*, 1> help_flag = {"--help"};
    if (sub == "prepare") {
      return bubo::prepare::Run(help_flag, "", std::cout, std::cerr);
    }
    if (sub == "serve") {
      if (options.size() > 2) {
        const std::array<const char*, 2> serve_help_flags = {options[2],
                                                             "--help"};
        return gufo::cli::RunServe(serve_help_flags);
      }
      return gufo::cli::RunServe(help_flag);
    }
    if (sub == "prompt") {
      return gufo::cli::RunPrompt(help_flag);
    }
    if (sub == "bench") {
      return gufo::cli::RunBench(help_flag);
    }
    if (sub == "eval") {
      return gufo::cli::RunEval(help_flag);
    }
    if (sub == "video") {
      return gufo::cli::RunVideo(help_flag);
    }
    if (sub == "transcribe") {
      return gufo::cli::RunTranscribe(help_flag);
    }
    if (sub == "chat") {
      return gufo::cli::RunChat(help_flag);
    }
    if (sub == "probe") {
      return gufo::cli::RunProbe(help_flag);
    }
    if (sub == "diagnose") {
      return gufo::cli::RunDiagnose(help_flag);
    }
    std::cerr << "Error: unknown help topic '" << sub << "'\n";
    gufo::cli::PrintGlobalHelp(program_name);
    return 2;
  }

  if (first_arg == "prepare") {
    const char* local = std::getenv("LOCALAPPDATA");
    return bubo::prepare::Run(options.subspan(1), local ? local : "", std::cout,
                              std::cerr);
  }

  gufo::cli::PrintStartupBanner(gufo::cli::kProductVersion, options);

  if (first_arg == "probe") {
    return gufo::cli::RunProbe(options.subspan(1));
  }

  if (first_arg == "diagnose") {
    return gufo::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return gufo::cli::RunBench(options.subspan(1));
  }

  if (first_arg == "eval") {
    return gufo::cli::RunEval(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return gufo::cli::RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return gufo::cli::RunVideo(options.subspan(1));
  }

  if (first_arg == "transcribe") {
    return gufo::cli::RunTranscribe(options.subspan(1));
  }

  if (first_arg == "chat") {
    return gufo::cli::RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return gufo::cli::RunServe(options.subspan(1));
  }

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
  return run(args);
}
