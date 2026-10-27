#include "src/cli/stop.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "src/cli/arg_parser.hpp"
#include "src/cli/instance_control.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"

namespace bubo::cli {
StopOptions ParseStopOptions(std::span<const char* const> args) {
  if (ParseMachineMode(args) != MachineMode::Json)
    throw std::invalid_argument("Stop requires --json.");
  StopOptions options;
  gufo::cli::ArgParser parser("bubo stop");
  parser.SetImplicitHelp(false).SetInlineValues(false);
  parser.SetErrorHandler([](auto kind, auto) {
    return kind == gufo::cli::ArgParser::ParseError::MissingValue
               ? "Missing stop option value."
               : "Unknown stop option.";
  });
  parser.AddFlag("", "--json", "Machine output", "Output", nullptr);
  parser.AddCustomOption(
      "", "--instance-id", "ID", "Select instance", "Stop",
      [&](auto, std::string_view value, auto*) {
        if (!options.instance_id.empty() || !ValidInstanceId(value))
          throw std::invalid_argument("Invalid instance id.");
        options.instance_id = value;
        return true;
      });
  parser.AddCustomOption(
      "", "--wait-ms", "MS", "Stop deadline", "Stop",
      [&](auto, std::string_view value, auto*) {
        std::uint32_t number = 0;
        const auto [end, error] =
            std::from_chars(value.data(), value.data() + value.size(), number);
        if (options.requested_wait || number == 0xffffffffU ||
            error != std::errc() || end != value.data() + value.size())
          throw std::invalid_argument("Invalid stop deadline.");
        options.requested_wait = number;
        return true;
      });
  std::string error;
  if (!parser.Parse(args, &error))
    throw std::invalid_argument(error);
  if (options.instance_id.empty())
    throw std::invalid_argument("Select an instance.");
  return options;
}

int RunStop(std::span<const char* const> args, std::string_view local_app_data,
            std::ostream& out, std::ostream& err) {
  if (!MachineRequested(args)) {
    constexpr std::string_view text =
        "Usage: bubo stop --json --instance-id ID [--wait-ms MS]\n\n"
        "To stop a server that you started in a terminal, press Ctrl+C in "
        "that window.\n"
        "The app uses bubo stop --json to stop the server that it started.\n";
    if (args.size() == 1 && (std::string_view(args[0]) == "--help" ||
                             std::string_view(args[0]) == "-h")) {
      out << text;
      return 0;
    }
    err << text;
    return 2;
  }
  MachineWriter writer(out);
  try {
    const auto options = ParseStopOptions(args);
    const auto& instance_id = options.instance_id;
    const auto& requested_wait = options.requested_wait;
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(requested_wait.value_or(45000));
    const auto record = ReadDiscovery(instance_id, local_app_data);
    if (!record)
      throw ControlError(16, "Instance record is unavailable.");
    if (record->phase == "stopped" && RecordedStopped(*record)) {
      auto result = Value::object();
      result["identity"] = IdentityJson(record->identity);
      result["process_exited"] = true;
      result["already_stopped"] = true;
      return writer.Write("stopped", std::move(result)) ? 0 : 1;
    }
    // The verified process handle is retained across both protocol calls and
    // the wait. A reused PID cannot satisfy completion of this operation.
    auto instance = VerifiedConnect(*record);
    auto acknowledgement = instance.RequestStop();
    acknowledgement["identity"] = IdentityJson(instance.Identity());
    acknowledgement["acknowledged"] = true;
    acknowledgement["process_exited"] = false;
    const bool ack_written =
        writer.Write("stopping", std::move(acknowledgement));
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining <= 0 ||
        !instance.WaitForExit(static_cast<std::uint32_t>(remaining))) {
      writer.Write(
          "error",
          Error("stop_timeout", "control", "exit", true,
                "Wait for this instance to exit. Check its log file."));
      return static_cast<int>(MachineExit::StopTimeout);
    }
    auto result = Value::object();
    result["identity"] = IdentityJson(instance.Identity());
    result["process_exited"] = true;
    result["already_stopped"] = false;
    return writer.Write("stopped", std::move(result)) && ack_written ? 0 : 1;
  } catch (const ControlError& error) {
    writer.Write(
        "error",
        Error(error.ExitCode() == 15 ? "protocol_mismatch" : "control_access",
              "control", "identity", false,
              "Select a verified instance owned by this user."));
    return error.ExitCode();
  } catch (const std::invalid_argument&) {
    writer.Write("error",
                 Error("invalid_arguments", "stop", "arguments", false,
                       "Use --json --instance-id ID --wait-ms 45000."));
    return 2;
  } catch (const std::exception&) {
    writer.Write("error",
                 Error("unexpected_failure", "control", "stop", false,
                       "Check the local control record and log file."));
    return 1;
  }
}
}  // namespace bubo::cli
