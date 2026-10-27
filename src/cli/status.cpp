#include "src/cli/status.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/settings_support.hpp"
#include "src/cli/status_throughput.hpp"
#include "src/cli/status_watch.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#else
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepare/runtime_prepare.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/model_notices.hpp"
#include "src/core/prepare/release_catalogue.hpp"
#include "src/core/prepare/runtime_table.hpp"

namespace bubo::cli {
namespace {
namespace fs = std::filesystem;
namespace prep = bubo::prepare;
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
// The package id the selected manifest names, for its prepared-model row.
std::string ManifestPackageId(const fs::path& manifest) {
  try {
    if (!manifest.empty() && fs::is_regular_file(manifest)) {
      const auto id = prep::metadata::Parse(prep::metadata::Read(manifest))
                          .member_str("package_id");
      if (!id.empty())
        return id;
    }
  } catch (const std::exception&) {
  }
  return std::string(prep::kPackageId);
}
bool ModelReady(const fs::path& manifest) {
  if (manifest.empty())
    return false;
  try {
    if (prep::ModelNotices().size() != 3)
      return false;
    const auto document = prep::metadata::Parse(prep::metadata::Read(manifest));
    if (document.member_size("schema_version") != 1)
      return false;
    // Any other catalogue package: its files at their pinned sizes.
    if (document.member_str("package_id") != prep::kPackageId) {
      const auto* entry = FindCatalogueEntry(ShippedCatalogue().entries,
                                             document.member_str("package_id"));
      return entry &&
             CatalogueReadinessAt(*entry, manifest.parent_path()).state ==
                 "prepared";
    }
    const auto state = prep::metadata::Parse(prep::metadata::Read(
        manifest.parent_path() / ".bubo/prepare-state.json"));
    if (state.member_size("schema_version") != 1 ||
        state.member_str("package_id") != prep::kPackageId)
      return false;
    for (const auto& notice : prep::ModelNotices())
      if (!fs::is_regular_file(manifest.parent_path() / "licenses" /
                               fs::u8path(notice.name)))
        return false;
    // The publication marker and compiled relative paths form a cheap status
    // check. Serve keeps the existing leases, fingerprints and hash checks.
    const auto& model = prep::metadata::Required(document, "model");
    const auto& main = prep::metadata::Required(model, "files");
    const auto& artefacts = prep::metadata::Required(document, "artefacts");
    if (!main.is_array() || main.size() != 4 || !artefacts.is_object())
      return false;
    unsigned shard = 0;
    for (const auto& entry : prep::ModelCatalogue()) {
      const auto& file = entry.kind == prep::Kind::MainShard
                             ? main.items()[shard++]
                             : prep::metadata::Required(
                                   artefacts, std::string(entry.logical_name));
      if (file.member_str("file") != entry.destination ||
          file.member_str("sha256") != entry.sha256 ||
          prep::metadata::Integer(prep::metadata::Required(file, "size")) !=
              entry.size)
        return false;
      const auto path = manifest.parent_path() / fs::u8path(entry.destination);
      if (!fs::is_regular_file(path) || fs::file_size(path) != entry.size)
        return false;
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

StatusOptions ParseStatusOptions(std::span<const char* const> args) {
  StatusOptions options;
  options.mode = ParseMachineMode(args);
  bool instance_seen = false;
  gufo::cli::ArgParser parser("bubo status");
  parser.SetImplicitHelp(false);
  parser.SetErrorHandler([&](auto kind, auto) {
    if (kind == gufo::cli::ArgParser::ParseError::MissingValue)
      return instance_seen ? "duplicate instance option"
                           : "missing instance ID";
    return "unknown status option";
  });
  parser.AddFlag("", "--json", "Machine output", "Output", nullptr);
  parser.AddFlag("", "--jsonl", "Machine stream", "Output", nullptr);
  parser.AddCustomFlag(
      "", "--watch", "Watch status", "Status", [&](auto, auto, auto*) {
        if (options.watch)
          throw std::invalid_argument("duplicate watch option");
        options.watch = true;
        return true;
      });
  parser.AddCustomOption(
      "", "--manifest", "PATH", "Report on this package", "Status",
      [&](auto, auto value, auto*) {
        if (!options.manifest.empty() || !options.model.empty() ||
            value.empty())
          throw std::invalid_argument("invalid manifest option");
        options.manifest = fs::u8path(value);
        return true;
      });
  parser.AddCustomOption(
      "", "--model", "PATH", "Report on the package of this model", "Status",
      [&](auto, auto value, auto*) {
        if (!options.manifest.empty() || !options.model.empty() ||
            value.empty())
          throw std::invalid_argument("invalid model option");
        options.model = fs::u8path(value);
        return true;
      });
  parser.AddCustomOption(
      "", "--instance-id", "ID", "Select instance", "Status",
      [&](auto, auto value, auto*) {
        if (instance_seen)
          throw std::invalid_argument("duplicate instance option");
        instance_seen = true;
        options.instance_id = value;
        if (!ValidInstanceId(options.instance_id))
          throw std::invalid_argument("invalid instance ID");
        return true;
      });
  std::string error;
  if (!parser.Parse(args, &error))
    throw std::invalid_argument(error);
  return options;
}

namespace {
// A watch owns its Ctrl+C handler only for its lifetime.
#ifdef _WIN32
std::atomic<bool> watch_interrupted{false};
BOOL WINAPI WatchControl(DWORD event) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT)
    return FALSE;
  return watch_interrupted.exchange(true) ? FALSE : TRUE;
}
#else
volatile std::sig_atomic_t watch_interrupted = 0;
void WatchControl(int) {
  watch_interrupted = 1;
}
#endif

class WatchTerminal {
public:
  explicit WatchTerminal(std::ostream& out) : out_(out) {
    watch_interrupted = false;
#ifdef _WIN32
    SetConsoleCtrlHandler(WatchControl, TRUE);
    handle_ = GetStdHandle(STD_OUTPUT_HANDLE);
    terminal_ =
        &out == &std::cout && GetConsoleMode(handle_, &mode_) &&
        SetConsoleMode(handle_, mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    restore_mode_ = terminal_;
#else
    previous_signal_ = std::signal(SIGINT, WatchControl);
    terminal_ = &out == &std::cout && isatty(STDOUT_FILENO);
#endif
    if (const char* term = std::getenv("TERM");
        term && std::string_view(term) == "dumb")
      terminal_ = false;
  }
  ~WatchTerminal() {
#ifdef _WIN32
    if (restore_mode_)
      SetConsoleMode(handle_, mode_);
    SetConsoleCtrlHandler(WatchControl, FALSE);
#else
    std::signal(SIGINT, previous_signal_);
#endif
  }

  bool Write(const std::vector<std::string>& lines) {
    if (!terminal_) {
      std::ostringstream plain;
      for (std::size_t index = 0; index < lines.size(); ++index)
        plain << (index ? " | " : "") << lines[index];
      const auto text = plain.str();
      if (!previous_plain_ || *previous_plain_ != text) {
        out_ << text << '\n' << std::flush;
        previous_plain_ = text;
      }
      return out_.good();
    }
    if (previous_lines_)
      out_ << "\033[" << previous_lines_ << 'A';
    // Leave the last column unused: terminal wrapping would turn a slot into
    // multiple physical rows and make the next redraw overwrite other slots.
    std::size_t columns = 79;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (GetConsoleScreenBufferInfo(handle_, &info))
      columns = std::max<int>(1, info.srWindow.Right - info.srWindow.Left);
#else
    winsize size{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 1)
      columns = size.ws_col - 1;
#endif
    const auto count = std::max(previous_lines_, lines.size());
    for (std::size_t index = 0; index < count; ++index) {
      out_ << "\r\033[2K";
      if (index < lines.size()) {
        const auto& line = lines[index];
        if (line.size() > columns && columns > 3)
          out_ << line.substr(0, columns - 3) << "...";
        else
          out_ << line.substr(0, columns);
      }
      out_ << '\n';
    }
    if (count > lines.size())
      out_ << "\033[" << count - lines.size() << 'A';
    previous_lines_ = lines.size();
    out_ << std::flush;
    return out_.good();
  }

private:
  std::ostream& out_;
  bool terminal_ = false;
  std::size_t previous_lines_ = 0;
  std::optional<std::string> previous_plain_;
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  DWORD mode_ = 0;
  bool restore_mode_ = false;
#else
  using Handler = void (*)(int);
  Handler previous_signal_ = SIG_DFL;
#endif
};

// Watch has no input messages. Drain available bytes and detect EOF without a
// reader thread that could outlive the command or wait on console input.
bool WatchInputClosed() {
#ifdef _WIN32
  const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  if (input == nullptr || input == INVALID_HANDLE_VALUE)
    return true;
  const DWORD kind = GetFileType(input);
  if (kind == FILE_TYPE_PIPE) {
    DWORD available = 0;
    if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) {
      const DWORD error = GetLastError();
      return error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED ||
             error == ERROR_INVALID_HANDLE;
    }
    if (available != 0) {
      char discard[1024];
      DWORD count = 0;
      if (!ReadFile(input, discard, std::min<DWORD>(available, sizeof(discard)),
                    &count, nullptr))
        return true;
    }
    return false;
  }
  if (kind == FILE_TYPE_DISK) {
    LARGE_INTEGER end{}, position{}, zero{};
    return !GetFileSizeEx(input, &end) ||
           !SetFilePointerEx(input, zero, &position, FILE_CURRENT) ||
           position.QuadPart >= end.QuadPart;
  }
  if (kind == FILE_TYPE_CHAR) {
    DWORD mode = 0;
    // The null device is EOF; an interactive console remains open.
    return GetConsoleMode(input, &mode) == 0;
  }
  return kind == FILE_TYPE_UNKNOWN;
#else
  pollfd input{STDIN_FILENO, POLLIN | POLLHUP, 0};
  const int ready = poll(&input, 1, 0);
  if (ready <= 0)
    return false;
  if (input.revents & (POLLNVAL | POLLERR))
    return true;
  if (input.revents & (POLLIN | POLLHUP)) {
    char discard[1024];
    return read(STDIN_FILENO, discard, sizeof(discard)) <= 0;
  }
  return false;
#endif
}

Value InstanceSummary(const DiscoveryRecord& record, const Value& live = {}) {
  auto summary = DiscoveryJson(record);
  const auto identity = IdentityJson(record.identity);
  for (const auto& [key, value] : identity.members())
    summary[key] = value;
  const auto* readiness = live.find("readiness");
  const auto* ready = readiness ? readiness->find("server") : nullptr;
  summary["ready"] = ready && ready->as_bool();
  const auto* settings = live.find("effective_settings");
  summary["effective_settings"] = settings ? *settings : Value();
  const auto* model = live.find("served_model");
  summary["served_model"] = model ? *model : Value();
  const auto* active = live.find("active_model");
  summary["active_model"] = active ? *active : Value();
  for (const auto* field : {"endpoint", "phase", "log_path"}) {
    if (const auto* value = live.find(field))
      summary[field] = *value;
  }
  return summary;
}

Value RuntimeSnapshot(const DiscoveryRecord& record,
                      std::string_view local_app_data, const fs::path& runtime,
                      VerifiedInstance& instance,
                      const StatusOptions& options = {}) {
  auto snapshot = StatusSnapshot(local_app_data, runtime, false, options.model,
                                 options.manifest);
  const auto live = instance.Snapshot();
  if (!live.is_object())
    throw ControlError(static_cast<int>(MachineExit::ProtocolMismatch),
                       "invalid status snapshot");
  for (const auto& [key, value] : live.members())
    snapshot[key] = value;
  snapshot["identity"] = IdentityJson(instance.Identity());
  snapshot["selection_required"] = false;
  snapshot["instances"] = Value::array();
  snapshot["instances"].push_back(InstanceSummary(record, snapshot));
  return snapshot;
}

Value DiscoverStatus(std::string_view local_app_data, const fs::path& runtime,
                     const StatusOptions& options, bool human_watch = false) {
  if (!options.instance_id.empty()) {
    const auto record = ReadDiscovery(options.instance_id, local_app_data);
    if (!record)
      throw ControlError(static_cast<int>(MachineExit::ControlAccess),
                         "instance record is unavailable");
    if (record->phase == "stopped" && RecordedStopped(*record)) {
      auto snapshot = StatusSnapshot(local_app_data, runtime, true,
                                     options.model, options.manifest);
      snapshot["phase"] = "stopped";
      snapshot["identity"] = IdentityJson(record->identity);
      snapshot["endpoint"] = record->endpoint;
      snapshot["log_path"] =
          record->log_path ? Value(*record->log_path) : Value();
      snapshot["last_error"] = record->last_error;
      snapshot["instances"].push_back(InstanceSummary(*record));
      return snapshot;
    }
    auto instance = VerifiedConnect(*record, options.watch ? 250 : 2000);
    return RuntimeSnapshot(*record, local_app_data, runtime, instance, options);
  }

  std::vector<DiscoveryRecord> records;
  std::vector<VerifiedInstance> instances;
  for (const auto& record : DiscoverRecords(local_app_data)) {
    if (options.watch && WatchInputClosed())
      break;
    if (human_watch && record.phase == "stopped" && RecordedStopped(record))
      continue;
    try {
      auto instance = VerifiedConnect(record, options.watch ? 250 : 2000);
      records.push_back(record);
      instances.push_back(std::move(instance));
    } catch (const ControlError& error) {
      // A discovery file is a candidate, never proof of a running instance.
      // Explicit selection returns the error instead of dropping the record.
      if (human_watch ||
          error.ExitCode() != static_cast<int>(MachineExit::ControlAccess))
        throw;
    }
  }
  if (records.size() == 1)
    return RuntimeSnapshot(records.front(), local_app_data, runtime,
                           instances.front(), options);
  auto snapshot = StatusSnapshot(local_app_data, runtime, true, options.model,
                                 options.manifest);
  if (records.size() > 1) {
    snapshot["phase"] = "multiple_instances";
    snapshot["selection_required"] = true;
    for (std::size_t index = 0; index < records.size(); ++index)
      snapshot["instances"].push_back(
          InstanceSummary(records[index], instances[index].Snapshot()));
  }
  return snapshot;
}
}  // namespace

Value StatusSnapshot(std::string_view local_app_data, const fs::path& runtime,
                     bool sample_machine, const fs::path& model,
                     const fs::path& selected_manifest) {
  auto value = Value::object();
  auto paths = Value::object();
  paths["runtime"] = Utf8(runtime);
  // "shared" (the per-user folder), "explicit" (BUBO_RUNTIME_DIR) or
  // "legacy" (beside bubo.exe), and the shared folder this machine uses.
  paths["runtime_kind"] =
      std::string(prep::ClassifyRuntime(runtime, local_app_data));
  paths["runtime_shared"] = Utf8(prep::SharedRuntimeDirFor(local_app_data));
  paths["local_app_data"] = std::string(local_app_data);
  fs::path manifest;
  Value selection_error;
  try {
    const auto selection = ResolvePreparedDefault(
        model, selected_manifest, EnvironmentUtf8("BUBO_MANIFEST"),
        local_app_data);
    manifest = selection.manifest;
    // --model: the package marker serve finds beside the model (or in the
    // model folder).
    if (manifest.empty() && !model.empty()) {
      std::error_code ec;
      const auto marker =
          (fs::is_directory(model, ec) ? model : model.parent_path()) /
          "bubo-package.json";
      if (fs::is_regular_file(marker, ec))
        manifest = fs::absolute(marker).lexically_normal();
    }
    value["model_selection"] = SelectionJson(selection);
    paths["prepared_default_pointer"] =
        selection.pointer.empty() ? Value() : Value(Utf8(selection.pointer));
  } catch (const std::exception&) {
    selection_error =
        Error("model_missing", "model", "resolver", false,
              "Run bubo prepare model to repair the prepared-default pointer.");
  }
  paths["manifest"] = manifest.empty() ? Value() : Value(Utf8(manifest));
  bool runtime_ready = prep::RuntimeReadyError(runtime).empty();
  for (const auto& member : prep::CompiledRuntimeTable()) {
    std::error_code ec;
    const auto file = runtime / fs::u8path(member.destination);
    if (!fs::is_regular_file(file, ec) ||
        fs::file_size(file, ec) != member.size || ec)
      runtime_ready = false;
  }
  const bool model_ready = ModelReady(manifest);
  value["prepared"] = runtime_ready && model_ready;
  value["phase"] = runtime_ready && model_ready ? "stopped" : "not_prepared";
  value["versions"] = Product();
  value["capabilities"] = Capabilities();
  value["paths"] = paths;
  value["prepared_default"] =
      manifest.empty() ? Value() : Value(Utf8(manifest));
  value["instances"] = Value::array();
  value["selection_required"] = false;
  value["identity"] = Value();
  value["endpoint"] = Value();
  value["endpoints"] = Value();
  value["api_key_env"] = "BUBO_API_KEY";
  value["log_path"] = Value();
  value["effective_settings"] = Value();
  value["inference_defaults"] = Value();
  value["effective_switches"] = Value();
  value["memory_estimate"] = Value();
  value["memory_estimate_reason"] =
      "No active server allocation settings are available.";
  value["pi_snippet"] = Value();
  value["client_snippets"] = Value();
  value["active_model"] = Value();
  value["connection"] = Value();
  value["requests"] = Value();
  value["prepared_models"] =
      PreparedModels(Utf8(manifest), model_ready, ManifestPackageId(manifest));
  value["memory"] = Value();
  value["counters"] = Value();
  value["last_completed_request"] = Value();
  value["sessions"] = Value();
  value["queue"] = Value();
  value["slots"] = Value();
  value["last_transition"] = Value();
  value["sample_time"] = UtcTime();
  if (sample_machine) {
    auto probe = CreateSystemMachineProbe();
    value["machine"] = MachineReport(*probe);
  }
  value["readiness"] = Value::object();
  value["readiness"]["runtime"] = runtime_ready;
  value["readiness"]["model"] = model_ready;
  value["last_error"] =
      !runtime_ready ? Error("runtime_missing", "runtime", "status", false,
                             "Run bubo prepare runtime.")
      : !model_ready ? Error("model_missing", "model", "status", false,
                             "Run bubo prepare model.")
                     : Value();
  if (!selection_error.is_null())
    value["last_error"] = selection_error;
  return value;
}

int RunStatus(std::span<const char* const> args,
              std::string_view local_app_data, const fs::path& runtime,
              std::ostream& out, std::ostream& err) {
  const bool machine = MachineRequested(args);
  if (!machine) {
    if (args.size() == 1 && (std::string_view(args[0]) == "--help" ||
                             std::string_view(args[0]) == "-h")) {
      out << "Usage: bubo status\n"
             "       bubo status --watch [--instance-id ID]\n"
             "       bubo status --json [--watch] [--manifest PATH | --model "
             "PATH]\n\n"
             "Without --json: one line that says what to do next.\n"
             "--watch: live prompt reading and response counts for each "
             "session.\n"
             "Press Ctrl+C to stop watching. Polls once per second; "
             "redirected\n"
             "output prints a plain line only when it changes. Cached tokens\n"
             "count as read; rates count new work "
             "only.\n"
             "A dash means no rate or estimate is available yet. Progress "
             "changes\n"
             "only when the server reports it. NO_COLOR disables terminal "
             "colors.\n"
             "--json: the full status record, for the app. --manifest or "
             "--model\n"
             "reports on that package instead of the saved default.\n";
      return 0;
    }
    const bool watch = std::ranges::any_of(args, [](const char* arg) {
      return std::string_view(arg) == "--watch";
    });
    if (!args.empty() && !watch) {
      err << "status: unknown option '" << args[0]
          << "'; run bubo status --help\n";
      return 2;
    }
    // One line for a person: the next step, or the running server.
    try {
      if (watch) {
        const auto options = ParseStatusOptions(args);
        WatchTerminal terminal(out);
        StatusWatch progress;
        do {
          const auto iteration = std::chrono::steady_clock::now();
          try {
            const auto snapshot =
                DiscoverStatus(local_app_data, runtime, options, true);
            if (!terminal.Write(progress.Lines(snapshot)))
              return 1;
          } catch (const ControlError&) {
            if (!terminal.Write({"Server not answering"}))
              return 1;
          }
          const auto deadline = iteration + std::chrono::seconds(1);
          do {
            if (watch_interrupted || WatchInputClosed())
              return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
          } while (std::chrono::steady_clock::now() < deadline);
        } while (true);
      }
      const auto snapshot = DiscoverStatus(local_app_data, runtime, {});
      const auto phase = snapshot.member_str("phase");
      const auto* endpoint = snapshot.find("endpoint");
      const auto* last_error = snapshot.find("last_error");
      if (phase == "not_prepared" || phase == "error")
        out << "Not ready. "
            << (last_error && last_error->is_object()
                    ? last_error->member_str("remedy")
                    : std::string("Run bubo prepare."))
            << '\n';
      else if (phase == "multiple_instances")
        out << "Several servers are running. See them with bubo status "
               "--json.\n";
      else if (phase == "stopped")
        out << "Ready. Start the server: bubo serve\n";
      else
        out << "A server is " << phase
            << (endpoint && endpoint->is_string() ? " at " + endpoint->str()
                                                  : std::string())
            << ". Press Ctrl+C in its window, or use the app, to stop it.\n";
      return 0;
    } catch (const std::invalid_argument& error) {
      err << "status: " << error.what() << "; run bubo status --help\n";
      return 2;
    } catch (const ControlError& error) {
      err << "status: " << error.what() << '\n';
      return 1;
    } catch (const std::exception& error) {
      err << "status: " << error.what() << '\n';
      return 1;
    }
  }
  MachineWriter writer(out);
  try {
    const auto options = ParseStatusOptions(args);
    if (options.mode == MachineMode::JsonLines)
      writer.Write("capabilities", Capabilities());
    ThroughputBaseline baseline;
    do {
      const auto iteration = std::chrono::steady_clock::now();
      auto snapshot = DiscoverStatus(local_app_data, runtime, options);
      // One-shot polls keep the live instance's baseline across CLI processes.
      // A watch has its own sampling interval; inactive status has no counters.
      if (options.watch || !snapshot.find("live_throughput"))
        baseline.Apply(snapshot);
      if (!writer.Write("status", std::move(snapshot)))
        return static_cast<int>(MachineExit::Unexpected);
      if (!options.watch)
        break;
      const auto deadline = iteration + std::chrono::milliseconds(950);
      do {
        if (WatchInputClosed())
          return 0;
        if (!writer.Connected())
          return static_cast<int>(MachineExit::Unexpected);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      } while (std::chrono::steady_clock::now() < deadline);
    } while (true);
    return 0;
  } catch (const ControlError& error) {
    const bool protocol =
        error.ExitCode() == static_cast<int>(MachineExit::ProtocolMismatch);
    writer.Write(
        "error",
        Error(protocol ? "protocol_mismatch" : "control_access", "status",
              "identity", false,
              protocol ? "Use a CLI with the same control version."
                       : "Select a running instance owned by this user."));
    return error.ExitCode();
  } catch (const std::invalid_argument&) {
    writer.Write("error", Error("invalid_arguments", "status", "arguments",
                                false, "Run bubo status --json."));
    return 2;
  } catch (const std::exception&) {
    writer.Write("error",
                 Error("unexpected_failure", "status", "snapshot", false,
                       "Check the prepared folders. Run status again."));
    return 1;
  }
}
}  // namespace bubo::cli
