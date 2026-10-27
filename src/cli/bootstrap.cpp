#include <windows.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/cli_info.hpp"
#include "src/cli/command_table.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepare/runtime_prepare.hpp"
#include "src/cli/serve/lifecycle.hpp"
#include "src/cli/status.hpp"
#include "src/cli/stop.hpp"
#include "src/cli/support_bundle.hpp"
#include "src/cli/switch_listing.hpp"

namespace {
std::string Utf8(std::wstring_view text) {
  if (text.empty())
    return {};
  const int count = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
      nullptr, 0, nullptr, nullptr);
  if (!count)
    throw std::runtime_error("invalid UTF-16 argument");
  std::string out(count, '\0');
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                           static_cast<int>(text.size()), out.data(), count,
                           nullptr, nullptr))
    throw std::runtime_error("UTF-8 conversion failed");
  return out;
}
std::string LocalAppData() {
  const auto count = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
  if (!count)
    return {};
  std::wstring value(count, L'\0');
  const auto length =
      GetEnvironmentVariableW(L"LOCALAPPDATA", value.data(), count);
  if (!length || length >= count)
    return {};
  value.resize(length);
  return Utf8(value);
}
std::filesystem::path EnginePath() {
  std::wstring name(32768, L'\0');
  const auto length =
      GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
  if (!length || length == name.size())
    throw std::runtime_error("cannot locate bubo.exe");
  name.resize(length);
  return std::filesystem::path(name).parent_path() / L"bubo-engine.dll";
}
// The AMD runtime folder: BUBO_RUNTIME_DIR, else the shared per-user folder
// when installed, else a runtime beside bubo.exe (the v0.1.0 layout).
bubo::prepare::RuntimeLocation Runtime() {
  return bubo::prepare::ResolveProcessRuntime(LocalAppData(),
                                              EnginePath().parent_path());
}
// Lets bubo-engine.dll and the AMD DLLs it imports resolve from the runtime
// folder. A runtime beside bubo.exe is already in the application folder.
// The environment carries the choice to the engine and its children.
// Returns the extra LoadLibraryEx flag: USER_DIRS only when a directory was
// added (the flag alone, with no user directory, fails the load with error 3).
DWORD PrepareRuntimeSearch(const bubo::prepare::RuntimeLocation& runtime) {
  using bubo::prepare::RuntimeKind;
  if (runtime.kind != RuntimeKind::Legacy)
    SetEnvironmentVariableW(L"BUBO_RUNTIME_DIR", runtime.dir.c_str());
  const std::wstring kind = [&] {
    const auto name = bubo::prepare::RuntimeKindName(runtime.kind);
    return std::wstring(name.begin(), name.end());
  }();
  SetEnvironmentVariableW(L"BUBO_RUNTIME_KIND", kind.c_str());
  if (runtime.kind == RuntimeKind::Legacy)
    return 0;
  std::error_code ec;
  auto dir = std::filesystem::weakly_canonical(runtime.dir, ec);
  if (ec)
    dir = runtime.dir;
  // Nothing installed yet: leave the search path alone so the engine load
  // reports the plain missing-file error.
  if (!std::filesystem::is_directory(dir, ec))
    return 0;
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  return AddDllDirectory(dir.c_str()) ? LOAD_LIBRARY_SEARCH_USER_DIRS : 0;
}
// Help of every engine command, printed when the engine cannot load (no AMD
// runtime yet). The text is the engine's own: tools/cli/gen_offline_help.py
// writes the table, and argparse_golden_test.py checks it against the engine.
struct OfflineHelpEntry {
  std::string_view key, text;
};
constexpr OfflineHelpEntry kOfflineHelp[] = {
    {"", ""},
#include "src/cli/offline_help.inc"
};
// The words before the first option, "help" dropped: "serve llm". The longest
// key the table knows wins, so `serve bogus --help` prints the serve help.
std::string_view OfflineHelp(std::span<const char* const> options) {
  std::vector<std::string_view> words;
  for (const char* option : options) {
    const std::string_view word(option);
    if (word.starts_with("-"))
      break;
    if (!(words.empty() && word == "help"))
      words.push_back(word);
  }
  while (!words.empty()) {
    std::string key;
    for (const auto word : words)
      key += (key.empty() ? "" : " ") + std::string(word);
    for (const auto& entry : kOfflineHelp)
      if (entry.key == key)
        return entry.text;
    words.pop_back();
  }
  return {};
}
void BootstrapError(std::string_view command, bubo::cli::Value error) {
  if (command == "serve") {
    bubo::cli::ServeLifecycle lifecycle(true);
    lifecycle.Transition("error", std::move(error));
    lifecycle.Finish();
  } else {
    bubo::cli::MachineWriter writer(std::cout);
    writer.Write("error", std::move(error));
  }
}
int LoadEngine(std::span<const char* const> args) {
  const auto options = args.subspan(1);
  const std::string_view command(options[0]);
  std::unique_ptr<bubo::cli::MachineWriter> loader_output;
  if (bubo::cli::MachineRequested(options))
    loader_output = std::make_unique<bubo::cli::MachineWriter>(std::cout);
  const auto path = EnginePath();
  const bool help_request =
      !bubo::cli::MachineRequested(options) &&
      (command == "help" ||
       gufo::cli::ArgParser::Contains(options.subspan(1), {"--help", "-h"}));
  if (command == "serve" && !help_request) {
    const auto runtime = Runtime();
    // Nothing installed anywhere: continue; the engine load reports it (a CPU
    // or development build has no AMD files to load).
    const auto readiness = runtime.absent ? std::string() : runtime.error;
    if (!readiness.empty()) {
      if (bubo::cli::MachineRequested(options)) {
        bubo::cli::ServeLifecycle lifecycle(true);
        lifecycle.Transition(
            "error", bubo::cli::Error("runtime_missing", "runtime", "bootstrap",
                                      false, "Run bubo prepare runtime."));
        lifecycle.Finish();
        return static_cast<int>(bubo::cli::MachineExit::RuntimeMissing);
      }
      std::cerr << "bubo: " << readiness << '\n';
      return 2;
    }
  }
  const DWORD user_dirs = PrepareRuntimeSearch(Runtime());
  const auto module =
      LoadLibraryExW(path.c_str(), nullptr,
                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                         LOAD_LIBRARY_SEARCH_SYSTEM32 | user_dirs);
  if (!module) {
    const DWORD load_error = GetLastError();
    if (bubo::cli::MachineRequested(options)) {
      const auto error = bubo::cli::Error(
          "runtime_missing", "engine", "bootstrap", false,
          "Run bubo prepare runtime. Reinstall the Bubo package.");
      if (command == "serve") {
        bubo::cli::ServeLifecycle lifecycle(true);
        lifecycle.Transition("error", error);
        lifecycle.Finish();
      } else {
        bubo::cli::MachineWriter writer(std::cout);
        writer.Write("error", error);
      }
      return bubo::cli::LifecycleExit(command == "serve" || command == "models",
                                      bubo::cli::MachineExit::RuntimeMissing,
                                      2);
    }
    if (help_request) {
      const auto text = OfflineHelp(options);
      if (!text.empty()) {
        std::cout << text;
        return 0;
      }
    }
    // A missing runtime is the cause: name it, without the loader's code.
    const auto missing = Runtime().error;
    if (!missing.empty())
      std::cerr << "bubo: " << missing << '\n';
    else
      std::cerr << "bubo: cannot load bubo-engine.dll (Windows error "
                << load_error << "); run bubo prepare runtime\n";
    if (command == "help" ||
        gufo::cli::ArgParser::Contains(options.subspan(1), {"--help", "-h"}))
      std::cerr << "bubo: the help for this command needs the AMD runtime "
                   "files. It works after bubo prepare runtime.\n";
    return 2;
  }
  const auto identity = reinterpret_cast<const char* (*)()>(
      GetProcAddress(module, "BuboEngineIdentityV1"));
  const auto run = reinterpret_cast<int (*)(int, const char* const*)>(
      GetProcAddress(module, "BuboRunCommandV1"));
  const std::string expected = gufo::cli::FormatVersion(
      gufo::cli::kProductName, gufo::cli::kProductVersion, gufo::cli::kRevision,
      gufo::cli::kEngineVersion);
  const char* actual = identity ? identity() : nullptr;
  if (!actual || !run || std::string_view(actual) != expected) {
    if (bubo::cli::MachineRequested(options)) {
      BootstrapError(
          command, bubo::cli::Error("protocol_mismatch", "engine", "bootstrap",
                                    false, "Reinstall the Bubo package."));
      FreeLibrary(module);
      return bubo::cli::LifecycleExit(command == "serve" || command == "models",
                                      bubo::cli::MachineExit::ProtocolMismatch,
                                      2);
    }
    std::cerr << "bubo: engine ABI/version/revision mismatch; reinstall this "
                 "package\n";
    FreeLibrary(module);
    return 2;
  }
  // Keep the engine loaded until process teardown: its workers/static objects
  // have the same process lifetime as before the bootstrap split.
  if (loader_output) {
    const auto configure = reinterpret_cast<void (*)(std::intptr_t)>(
        GetProcAddress(module, "BuboSetMachineOutputV1"));
    if (!configure) {
      BootstrapError(
          command, bubo::cli::Error("protocol_mismatch", "engine", "bootstrap",
                                    false, "Reinstall the Bubo package."));
      return bubo::cli::LifecycleExit(command == "serve" || command == "models",
                                      bubo::cli::MachineExit::ProtocolMismatch,
                                      2);
    }
    configure(loader_output->NativeHandle());
  }
  return run(static_cast<int>(args.size()), args.data());
}

int Run(std::span<const char* const> args) {
  const auto options = args.subspan(1);
  if (bubo::cli::MachineRequested(options)) {
    try {
      bubo::cli::ParseMachineMode(options);
    } catch (const std::invalid_argument&) {
      BootstrapError(
          options.empty() ? "" : options[0],
          bubo::cli::Error("invalid_arguments", "cli", "arguments", false,
                           "Select one machine-output flag."));
      return 2;
    }
  }
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
  if (options.empty() || std::string_view(options[0]) == "--help" ||
      std::string_view(options[0]) == "-h") {
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 0;
  }
  const std::string_view command = options[0];
  const auto* entry =
      gufo::cli::FindCommand<gufo::cli::CommandLayer::Bootstrap>(command);
  gufo::cli::CommandContext context;
  context.load_engine = [&] { return LoadEngine(args); };
  context.help = [&](std::span<const char* const> help) {
    if (help.size() == 1) {
      gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
      return 0;
    }
    if (std::string_view(help[1]) == "prepare") {
      if (help.size() != 2) {
        std::cerr << "prepare: E2 extra help arguments\n";
        return 2;
      }
      bubo::prepare::PrintHelp(std::cout);
      return 0;
    }
    // Engine-free topics (status, stop) answer without the engine DLL.
    const auto* topic =
        gufo::cli::FindCommand<gufo::cli::CommandLayer::Bootstrap>(help[1]);
    if (topic && topic->help_topic && !topic->engine_needed &&
        help.size() == 2) {
      const std::array<const char*, 2> topic_help{help[1], "--help"};
      return topic->run(topic_help, context);
    }
    return context.load_engine();
  };
  if (const auto result = gufo::cli::RunCommandVariant(options))
    return *result;
  if (command == "--version" || command == "-v") {
    gufo::cli::PrintGlobalVersion();
    return 0;
  }
  if (entry && !entry->engine_needed && command != "help")
    return entry->run(options, context);
  if (command == "serve" && bubo::cli::MachineRequested(options)) {
    if (bubo::cli::ParseMachineMode(options) !=
        bubo::cli::MachineMode::JsonLines) {
      BootstrapError(command,
                     bubo::cli::Error("invalid_arguments", "serve", "arguments",
                                      false, "Use bubo serve --jsonl."));
      return 2;
    }
    if (gufo::cli::ArgParser::Contains(options.subspan(1), {"--help", "-h"})) {
      bubo::cli::MachineWriter writer(std::cout);
      writer.Write("capabilities", bubo::cli::Capabilities());
      return 0;
    }
  }
  if (command == "help")
    return entry->run(options, context);
  if (bubo::cli::MachineRequested(options) &&
      (!entry || !entry->machine_supported)) {
    bubo::cli::MachineWriter writer(std::cout);
    writer.Write(
        "error",
        bubo::cli::Error("unsupported", "cli", "dispatch", false,
                         "Use a command listed in the status capabilities."));
    return 2;
  }
  // An unknown command needs no engine: name it and print the help.
  if (!entry && !bubo::cli::MachineRequested(options)) {
    std::cerr << "Error: unknown command '" << command << "'\n";
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 2;
  }
  // No engine import, HIP registration or inference initialization precedes
  // table dispatch. Engine-free variants and commands have already returned.
  return entry ? entry->run(options, context) : context.load_engine();
}
}  // namespace

int wmain(int argc, wchar_t* argv[]) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  std::unique_ptr<bubo::cli::MachineWriter> process_output;
  const bool machine = gufo::cli::ArgParser::Contains(
      std::span<const wchar_t* const>(argv, argc).subspan(1),
      {L"--json", L"--jsonl"});
  try {
    if (machine)
      process_output =
          std::make_unique<bubo::cli::MachineWriter>(std::cout, true);
    std::vector<std::string> owned;
    owned.reserve(argc);
    for (int i = 0; i < argc; ++i)
      owned.push_back(Utf8(argv[i]));
    std::vector<const char*> pointers;
    for (const auto& value : owned)
      pointers.push_back(value.c_str());
    return Run(pointers);
  } catch (const std::exception& error) {
    if (machine) {
      bubo::cli::MachineWriter writer(std::cout);
      writer.Write(
          "error",
          bubo::cli::Error("unexpected_failure", "cli", "bootstrap", false,
                           "Check the command arguments and install folder."));
      return 2;
    }
    std::cerr << "bubo: " << error.what() << '\n';
    return 2;
  }
}
