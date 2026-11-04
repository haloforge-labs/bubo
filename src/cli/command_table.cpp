#include "src/cli/command_table.hpp"

#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "src/cli/arg_parser.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/cli/status.hpp"
#include "src/cli/stop.hpp"
#include "src/cli/support_bundle.hpp"
#include "src/cli/switch_listing.hpp"

namespace gufo::cli {
namespace commands {
int Prepare(std::span<const char* const> args, const CommandContext&) {
  return bubo::prepare::Run(args.subspan(1),
                            bubo::cli::EnvironmentUtf8("LOCALAPPDATA"),
                            std::cout, std::cerr);
}
int Status(std::span<const char* const> args, const CommandContext&) {
  const auto local = bubo::cli::EnvironmentUtf8("LOCALAPPDATA");
  const int code = bubo::cli::RunStatus(
      args.subspan(1), local,
      bubo::prepare::ResolveProcessRuntime(local,
                                           bubo::cli::ExecutableDirectory())
          .dir,
      std::cout, std::cerr);
#ifdef _WIN32
  // A driver call that never answered is still running on its own thread. It
  // can hold the loader lock, and a normal exit would wait for it.
  if (bubo::cli::MachineProbeHasAbandonedCall()) {
    std::cout.flush();
    std::cerr.flush();
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
  }
#endif
  return code;
}
int Stop(std::span<const char* const> args, const CommandContext&) {
  return bubo::cli::RunStop(args.subspan(1),
                            bubo::cli::EnvironmentUtf8("LOCALAPPDATA"),
                            std::cout, std::cerr);
}
int Help(std::span<const char* const> args, const CommandContext& context) {
  return context.help(args);
}
}  // namespace commands

std::optional<int> RunCommandVariant(std::span<const char* const> options) {
  if (options.empty())
    return std::nullopt;
  const std::string_view command(options[0]);
  const auto args = options.subspan(1);
  if (command == "serve" && bubo::cli::ListSwitchesRequested(args))
    return bubo::cli::RunListSwitches(args, std::cout);
  if (command == "diagnose" && (ArgParser::Contains(args, {"--bundle"}, true) ||
                                ArgParser::Contains(args, {"--preview"})))
    return bubo::cli::RunSupportBundle(
        args, bubo::cli::EnvironmentUtf8("LOCALAPPDATA"),
        bubo::cli::ExecutableDirectory(), std::cout, std::cerr);
  return std::nullopt;
}
}  // namespace gufo::cli
