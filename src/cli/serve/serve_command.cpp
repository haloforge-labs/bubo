#include "src/cli/serve/serve_command.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/release_profile.hpp"

namespace gufo::cli {
ServeCommandOptions ParseServeCommandOptions(
    std::span<const char* const> args) {
  ServeCommandOptions options;
  std::set<std::string> supplied_log_flags;
  if (bubo::cli::MachineRequested(args)) {
    if (bubo::cli::ParseMachineMode(args) != bubo::cli::MachineMode::JsonLines)
      throw std::invalid_argument("Serve requires --jsonl.");
  }
  ArgParser parser("bubo serve");
  parser.SetImplicitHelp(false);
  parser.AddFlag("", "--jsonl", "Machine stream", "Output", nullptr);
  parser.SetOptionInlineValues("--jsonl", false);
  parser.SetUnrecognizedHandler([&](std::string_view arg, auto*) {
    options.filtered.push_back(arg.data());
    return true;
  });
  parser.SetErrorHandler([&](auto, std::string_view name) {
    return supplied_log_flags.contains(std::string(name))
               ? "Repeated log option."
               : "Log option requires a value.";
  });
  const auto log_option = [&](std::string_view name, std::string_view value,
                              auto*) {
    if (!supplied_log_flags.insert(std::string(name)).second)
      throw std::invalid_argument("Repeated log option.");
    if (value.empty() || value.starts_with("-"))
      throw std::invalid_argument("Log option requires a value.");
    if (name == "--log-file") {
      if (value.find_first_of("\r\n\t") != std::string_view::npos)
        throw std::invalid_argument("Invalid log path.");
      options.log_path =
          std::filesystem::absolute(std::filesystem::u8path(value));
    } else {
      std::uint64_t number = 0;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), number);
      if (error != std::errc() || end != value.data() + value.size())
        throw std::invalid_argument("Invalid log rotation number.");
      if (name == "--log-max-bytes")
        options.log_max_bytes = number;
      else {
        if (number < 1 || number > 16)
          throw std::invalid_argument("Invalid retained log count.");
        options.log_retain = static_cast<unsigned>(number);
      }
    }
    return true;
  };
  for (const auto* flag : {"--log-file", "--log-max-bytes", "--log-retain"})
    parser.AddCustomOption("", flag, "VALUE", "Log configuration", "Logging",
                           log_option);
  std::string error;
  if (!parser.Parse(args, &error))
    throw std::invalid_argument(error);
  options.supplied_log_flags = !supplied_log_flags.empty();
  return options;
}

namespace {
// The serve paths write their failure as an "Error..." line to stderr. In
// machine mode the error event repeats the last such line as `detail`. The
// capture lives for the process, so a late writer never meets a dead buffer.
bubo::cli::LastLineCapture* InstallErrorCapture() {
  static bubo::cli::LastLineCapture* capture = [] {
    auto* installed =
        new bubo::cli::LastLineCapture(std::cerr.rdbuf(), "Error");
    std::cerr.rdbuf(installed);
    return installed;
  }();
  return capture;
}
std::string StartupDetail(const bubo::cli::LastLineCapture* capture) {
  if (!capture)
    return {};
  auto line = capture->Last();
  if (line.starts_with("Error: "))
    line.erase(0, 7);
  return line;
}
}  // namespace

int RunServeCommand(std::span<const char* const> args,
                    const ServeRunner& runner) {
  const bool machine = bubo::cli::MachineRequested(args);
  auto* capture = machine ? InstallErrorCapture() : nullptr;
  if (capture)
    capture->Reset();
  std::unique_ptr<bubo::cli::ServeLifecycle> lifecycle;
  try {
    const auto options = ParseServeCommandOptions(args);
    const auto& filtered = options.filtered;
    const auto& log_path = options.log_path;
    const auto log_max_bytes = options.log_max_bytes;
    const auto log_retain = options.log_retain;
    if (machine && ArgParser::Contains(filtered, {"--help", "-h"})) {
      bubo::cli::MachineWriter writer(std::cout);
      writer.Write("capabilities", bubo::cli::Capabilities());
      return 0;
    }
#ifdef _WIN32
    lifecycle = std::make_unique<bubo::cli::ServeLifecycle>(machine);
#else
    if (machine)
      throw bubo::cli::ControlError(18, "Local control requires Windows.");
#endif
    if (!log_path && options.supplied_log_flags)
      throw std::invalid_argument("Log rotation requires --log-file.");
    if (log_path)
      server::Logger::SetLogFile(*log_path, log_max_bytes, log_retain);
    const int code = runner(filtered, lifecycle.get(), log_path);
    if (lifecycle) {
      if (code != 0)
        lifecycle->Transition(
            "error",
            bubo::cli::Error(
                code == 2 ? "invalid_arguments" : "unexpected_failure", "serve",
                "startup", false, "Check the settings and model files.",
                StartupDetail(capture)));
      lifecycle->Finish();
    }
    return code;
  } catch (const bubo::cli::ControlError& error) {
    const auto code = error.ExitCode();
    // stderr is the log channel in machine mode too: the reason is kept.
    if (code != 0)
      std::cerr << "Error: " << error.what() << '\n';
    if (lifecycle) {
      if (code != 0) {
        auto payload = bubo::cli::Error(
            code == 13   ? "port_occupied"
            : code == 12 ? "model_missing"
            : code == 2  ? "invalid_arguments"
            : code == 14 ? "startup_timeout"
            : code == 18 ? "unsupported"
            : code == 19 ? "memory_insufficient"
                         : "control_access",
            "serve", "startup", code == 13,
            code == 13   ? "Select another port."
            : code == 19 ? "Lower --context or --sessions, or free memory."
                         : "Check the local control and model settings.",
            error.what());
        if (code == 19 && error.Details().is_object()) {
          // The numbers behind the refusal, as fields: needed_bytes, available_bytes, sessions, context, per_session_bytes, floor_bytes.
          for (const auto& [key, value] : error.Details().members())
            payload[key] = value;
        }
        lifecycle->Transition("error", std::move(payload));
      }
      lifecycle->Finish();
    }
    return bubo::cli::LifecycleExit(
        machine, static_cast<bubo::cli::MachineExit>(code), code == 13 || code == 19 ? 1 : 2);
  } catch (const std::exception& exception) {
    // stderr is the log channel in machine mode too: the reason is kept.
    std::cerr << "Error: " << exception.what() << '\n';
    const bool invalid =
        dynamic_cast<const std::invalid_argument*>(&exception) != nullptr;
    if (!lifecycle && machine)
      lifecycle = std::make_unique<bubo::cli::ServeLifecycle>(true);
    if (lifecycle) {
      auto payload = bubo::cli::Error(
          invalid ? "invalid_arguments" : "unexpected_failure", "serve",
          "startup", false, "Check the settings and model files.",
          exception.what());
      if (const auto* setting =
              dynamic_cast<const server::ReleaseSettingError*>(&exception))
        payload["switch"] = setting->switch_name;
      lifecycle->Transition("error", std::move(payload));
      lifecycle->Finish();
    }
    return invalid ? 2 : 1;
  }
}

}  // namespace gufo::cli
