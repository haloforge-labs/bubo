#include "src/cli/serve/serve.hpp"

#include <arpa/inet.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/machine_warnings.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/cli/models_support.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/cache_disk_limits.hpp"
#include "src/cli/serve/debug_request.hpp"
#include "src/cli/serve/debug_settings.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/lifecycle.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/memory_estimate.hpp"
#include "src/cli/serve/memory_policy.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/cli/serve/serve_command.hpp"
#include "src/cli/serve/status_metrics.hpp"
#include "src/cli/settings_support.hpp"
#include "src/core/prepare/metadata_json.hpp"
#ifdef GUFO_NPU
#include "src/cli/serve/npu_status_publisher.hpp"
#include "src/npu/chunk_retry.hpp"
#include "src/npu/server_prefill.hpp"
#endif
#include "src/cli/serve/sampling_request.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/diagnostics/gpu_queues.h"
#include "src/core/platform/loaded_files.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>

#include "src/cli/serve/diffusion_gemma_backend.hpp"
#include "src/cli/serve/diffusion_gemma_options.hpp"
#include "src/core/platform/device_memory.hpp"
#include "src/models/diffusion_gemma/config.hpp"
#endif
#include "src/cli/serve/trace.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::cli {
namespace {

class ModelLoadLog {
public:
  ModelLoadLog(std::string kind, const std::filesystem::path& artifact)
      : kind_(std::move(kind)), start_(std::chrono::steady_clock::now()) {
    server::Logger::Info("loader", "event=load_started kind=" + kind_ +
                                       " artifact=" + artifact.string() + " " +
                                       server::Logger::MemoryStatus());
  }
  ~ModelLoadLog() {
    if (!completed_) {
      try {
        server::Logger::Error("loader", "event=load_failed kind=" + kind_ +
                                            " " +
                                            server::Logger::MemoryStatus());
      } catch (...) {
      }
    }
  }
  void Complete(std::string_view details, bool gpu_loaded = true) {
    completed_ = true;
    std::ostringstream out;
    out << "event=load_completed kind=" << kind_ << " elapsed_ms="
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start_)
               .count()
        << ' ' << details << ' ' << server::Logger::MemoryStatus();
#if defined(ENGINE_ENABLE_HIP)
    std::size_t free = 0, total = 0;
    if (gpu_loaded &&
        gufo::platform::DeviceMemoryInfo(&free, &total) == hipSuccess) {
      out << " gpu_device_used_mib=" << (total - free) / (1024 * 1024)
          << " gpu_device_total_mib=" << total / (1024 * 1024);
      // DXGI split of the same numbers: LOCAL (the dedicated carve-out) vs
      // NON_LOCAL (shared system memory); the combined figures above sum both.
      if (const auto split = gufo::platform::QueryDeviceMemorySplit();
          split.valid)
        out << " gpu_local_used_mib=" << split.local_used / (1024 * 1024)
            << " gpu_local_budget_mib=" << split.local_budget / (1024 * 1024)
            << " gpu_nonlocal_used_mib=" << split.nonlocal_used / (1024 * 1024)
            << " gpu_nonlocal_budget_mib="
            << split.nonlocal_budget / (1024 * 1024);
    }
#else
    (void)gpu_loaded;
#endif
    server::Logger::Info("loader", out.str());
  }

private:
  std::string kind_;
  std::chrono::steady_clock::time_point start_;
  bool completed_{false};
};

// Reports a fatal signal on the way out. A driver or runtime failure during a
// load can abort the process outright, which runs no destructor and no
// terminate handler, leaving a log that stops at event=load_started. Only
// async-signal-safe calls are allowed here, so the line is assembled by hand
// and written straight to the descriptor.
extern "C" void ReportFatalSignal(int number) {
  static constexpr char kPrefix[] =
      "[ERROR] [server] event=fatal_signal signal=";
  char digits[8];
  std::size_t length = 0;
  int value = number;
  if (value <= 0) {
    digits[length++] = '0';
  } else {
    char reversed[8];
    std::size_t count = 0;
    while (value > 0 && count < sizeof(reversed)) {
      reversed[count++] = static_cast<char>('0' + (value % 10));
      value /= 10;
    }
    while (count > 0) {
      digits[length++] = reversed[--count];
    }
  }
  (void)::write(STDERR_FILENO, kPrefix, sizeof(kPrefix) - 1);
  (void)::write(STDERR_FILENO, digits, length);
  (void)::write(STDERR_FILENO, "\n", 1);
  // Restore the default action and re-raise, so the exit status and any core
  // dump still describe the original fault.
  struct sigaction restore{};
  restore.sa_handler = SIG_DFL;
  (void)::sigemptyset(&restore.sa_mask);
  (void)::sigaction(number, &restore, nullptr);
  (void)::raise(number);
}

void InstallFatalSignalReporter() {
  struct sigaction action{};
  action.sa_handler = ReportFatalSignal;
  (void)::sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  for (const int number : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
    (void)::sigaction(number, &action, nullptr);
  }
}

// Installs a terminate handler that names the reason before the process dies.
// A HIP or driver failure during a load throws, and an exception that reaches
// the top of main terminates without unwinding the stack, so ModelLoadLog's
// destructor never runs and the log ends at event=load_started.
void ReportTerminationReason() {
  static std::terminate_handler previous = nullptr;
  previous = std::set_terminate([] {
    std::string reason = "unknown";
    if (std::current_exception() != nullptr) {
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception& error) {
        reason = error.what();
      } catch (...) {
        reason = "non-standard exception";
      }
    }
    try {
      server::Logger::Error("server", "event=terminated reason=" + reason);
      server::Logger::FlushSync(std::chrono::seconds(2));
    } catch (...) {
      // The handler must not throw on its way out.
    }
    if (previous != nullptr) {
      previous();
    }
    std::abort();
  });
}

// Exit status after the GPU context is lost (EX_TEMPFAIL). Only a new process
// recovers, so a supervisor must restart it.
constexpr int kDeviceLostExitStatus = 75;
// Deep snapshot captures and post-failure device probes legitimately hold
// progress for tens of seconds.
constexpr std::uint64_t kMinStallTimeoutMs = 60000;

// A lost GPU context cannot recover in-process. Leave through the SIGTERM
// shutdown path so a supervisor restarts the server, and bound that teardown:
// joining requests or freeing device memory may block on the dead device.
void ArmDeviceLossWatchdog() {
  constexpr auto kShutdownTimeout = std::chrono::seconds(10);
  // Arm the watchdog before logging or shutdown: neither a blocked log sink
  // nor a dead-device join may prevent the forced exit.
  try {
    std::thread([kShutdownTimeout] {
      // Reserve the last 2 s for logs without extending the 10 s watchdog.
      std::this_thread::sleep_for(kShutdownTimeout - std::chrono::seconds(2));
      server::Logger::FlushSync(std::chrono::seconds(2));
      ::_exit(kDeviceLostExitStatus);
    }).detach();
  } catch (...) {
    server::Logger::FlushSync(std::chrono::seconds(2));
    ::_exit(kDeviceLostExitStatus);
  }
}
void ShutdownAfterDeviceLoss(bubo::cli::ServeLifecycle* lifecycle = nullptr) {
  constexpr auto kShutdownTimeout = std::chrono::seconds(10);
  ArmDeviceLossWatchdog();
  if (lifecycle) {
    lifecycle->Transition(
        "error",
        bubo::cli::Error("device_lost", "engine", "serve", false,
                         "Restart Bubo after the device is available."));
    lifecycle->RecordAbruptExit();
  }
  server::Logger::Error(
      "server", "event=device_lost_shutdown exit_status=" +
                    std::to_string(kDeviceLostExitStatus) +
                    " timeout_s=" + std::to_string(kShutdownTimeout.count()));
  (void)std::raise(SIGTERM);
}

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

// Server-level options are accepted on either side of the modality
// subcommand. Registering them from one place keeps `gufo serve --help`, every
// `gufo serve <modality> --help`, and the parser that actually consumes them
// from drifting apart.

/// `--log-level` and its `-v/--verbose` shorthand share one sink. `level`
/// holds the canonical spelling when given and stays empty otherwise; the two
/// spellings together are a conflict, reported where the level is armed.
struct ServerLogOptions {
  bool verbose = false;
  std::optional<server::LogLevel> level;
};

// Verbosity help text. `gufo serve --help` lists these options by hand while
// every modality help registers them through the parser, so both readings share
// one source for the wording.
constexpr std::string_view kLogLevelHelp =
    "Log verbosity: error, warn, info or debug (default: info)";
constexpr std::string_view kVerboseHelp = "Shorthand for --log-level=debug";
constexpr std::string_view kTraceHelp =
    "Append text request bodies, rendered prompts, generated text and replies "
    "to PATH as JSON Lines; the file holds client content";

void AddServerOptions(gufo::cli::ArgParser& parser, std::string* host,
                      int* port, std::size_t* session_count,
                      std::size_t* max_connections,
                      std::size_t* max_request_body_bytes, std::string* api_key,
                      ServerLogOptions* log, bool* debug_hooks,
                      std::size_t* max_score_tokens) {
  parser.AddOption("-i", "--host", "IP", "Bind address", "Server", host);
  parser.AddOption("-p", "--port", "N", "Port to listen on", "Server", port);
  if (session_count != nullptr)
    parser.AddOption("-j", "--sessions", "N",
                     "Preallocated GPU request sessions", "Server",
                     session_count);
  parser.AddOption("", "--max-connections", "N",
                   "Maximum simultaneous HTTP connections", "Server",
                   max_connections);
  parser.AddOption("", "--max-request-bytes", "N",
                   "Maximum HTTP request body bytes", "Server",
                   max_request_body_bytes);
  parser.AddOption("", "--api-key", "KEY", "API key (or set BUBO_API_KEY)",
                   "Server", api_key);
  parser.AddCustomOption(
      "", "--log-level", "LEVEL", kLogLevelHelp, "Logging",
      [log](std::string_view flag, std::string_view value, std::string* error) {
        const auto level = server::LogLevelFromName(value);
        if (!level.has_value()) {
          *error = std::string(flag) + " must be error, warn, info or debug";
          return false;
        }
        log->level = *level;
        return true;
      });
  parser.AddFlag("-v", "--verbose", kVerboseHelp, "Logging", &log->verbose);
  parser.AddFlag("", "--debug-hooks",
                 "Enable loopback debug diagnostics (default: off)", "Debug",
                 debug_hooks);
  parser.AddOption("", "--max-score-tokens", "N",
                   "Maximum teacher-forced prompt tokens (default: 8192)",
                   "Debug", max_score_tokens);
}

// Help-only sinks for AddServerOptions, so subcommand help can list the
// server options it shares with `gufo serve` without parsing into them.
struct ServerOptionHelpTargets {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  ServerLogOptions log;
  bool debug_hooks{false};
  std::size_t max_score_tokens{8192};
};

void AddServerOptionsForHelp(gufo::cli::ArgParser& parser,
                             ServerOptionHelpTargets* targets,
                             bool include_sessions = true) {
  AddServerOptions(parser, &targets->host, &targets->port,
                   include_sessions ? &targets->session_count : nullptr,
                   &targets->max_connections, &targets->max_request_body_bytes,
                   &targets->api_key, &targets->log, &targets->debug_hooks,
                   &targets->max_score_tokens);
}

// Split a `NAME=VALUE` CLI spec. Returns false when either side is empty.
bool SplitNameValue(std::string_view spec, std::string_view flag,
                    std::string* name, std::string* value, std::string* error) {
  const std::size_t separator = spec.find('=');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 >= spec.size()) {
    *error = std::string(flag) + " expects NAME=VALUE";
    return false;
  }
  *name = std::string(spec.substr(0, separator));
  *value = std::string(spec.substr(separator + 1));
  return true;
}

std::optional<std::string> ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::nullopt;
  }
  std::string text{std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>()};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

// Resolve a `--voice-text` value: an existing file is read for its contents,
// anything else is taken as the transcript itself.
std::string ResolveReferenceText(const std::string& value) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(std::filesystem::path(value), ec)) {
    if (const auto text = ReadTextFile(std::filesystem::path(value))) {
      return *text;
    }
  }
  return value;
}

// Build the registered voices from `--voice NAME=WAV` and the optional
// `--voice-text NAME=<text|path>` overrides. Resolution happens after parsing
// so the two flags may appear in any order.
bool BuildVoicePresets(
    const std::vector<std::pair<std::string, std::string>>& voice_specs,
    const std::map<std::string, std::string>& text_specs,
    const std::map<std::string, std::string>& language_specs,
    std::map<std::string, server::TtsVoicePreset>* presets,
    std::string* error) {
  for (const auto& [name, wav] : voice_specs) {
    if (presets->contains(name)) {
      *error = "duplicate --voice name '" + name + "'";
      return false;
    }
    const std::filesystem::path wav_path(wav);
    std::ifstream wav_file(wav_path, std::ios::binary);
    if (!wav_file) {
      *error = "cannot open voice reference " + wav_path.string();
      return false;
    }
    const std::vector<char> wav_bytes{std::istreambuf_iterator<char>(wav_file),
                                      std::istreambuf_iterator<char>()};

    server::TtsVoicePreset preset;
    std::string decode_error;
    if (!models::qwen3_tts::DecodeWav(
            std::as_bytes(std::span<const char>(wav_bytes)),
            &preset.reference_audio, &decode_error)) {
      *error = wav_path.string() + ": " + decode_error;
      return false;
    }

    // An explicit --voice-text wins; otherwise fall back to a `.txt` sidecar
    // beside the WAV. With neither, the voice clones from the speaker
    // embedding alone, which needs no transcript.
    if (const auto override_text = text_specs.find(name);
        override_text != text_specs.end()) {
      preset.reference_text = ResolveReferenceText(override_text->second);
    } else {
      std::filesystem::path sidecar = wav_path;
      sidecar.replace_extension(".txt");
      if (const auto text = ReadTextFile(sidecar)) {
        preset.reference_text = *text;
      }
    }
    preset.speaker_embedding_only = preset.reference_text.empty();
    if (const auto language = language_specs.find(name);
        language != language_specs.end()) {
      preset.language = language->second;
    }
    presets->emplace(name, std::move(preset));
  }

  for (const auto& [name, unused] : text_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-text names unknown voice '" + name + "'";
      return false;
    }
  }
  for (const auto& [name, unused] : language_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-lang names unknown voice '" + name + "'";
      return false;
    }
  }
  return true;
}

std::optional<ReasoningOptions> ResolveReasoningDefaults(
    std::string_view mode, std::string_view effort, std::string_view preserve,
    std::string* error) {
  ReasoningOptions options;
  if (mode == "on") {
    options.enabled = true;
  } else if (mode == "off") {
    options.enabled = false;
  } else if (mode != "auto") {
    *error = "--think must be on, off, or auto";
    return std::nullopt;
  }

  if (effort != "auto") {
    const auto parsed = ParseReasoningEffort(effort);
    if (!parsed.has_value()) {
      *error =
          "--reasoning-effort must be auto, minimal, low, medium, high, "
          "xhigh, or max";
      return std::nullopt;
    }
    if (options.enabled == false) {
      *error = "--reasoning-effort cannot be set while --think is off";
      return std::nullopt;
    }
    options.enabled = true;
    options.effort = parsed;
  }

  if (preserve == "on") {
    options.preserve_thinking = true;
  } else if (preserve == "off") {
    options.preserve_thinking = false;
  } else if (preserve != "auto") {
    *error = "--preserve-thinking must be on, off, or auto";
    return std::nullopt;
  }
  return options;
}

struct SpeechOptions {
  std::filesystem::path model;
  std::size_t context;
  std::string served_model_name;
  std::vector<std::pair<std::string, std::string>> voice_specs;
  std::map<std::string, std::string> voice_text_specs;
  std::map<std::string, std::string> voice_lang_specs;
};

void AddSpeechOptions(ArgParser& parser, bool tts, SpeechOptions* options) {
  parser.AddOption("-m", "--model", "DIR",
                   tts ? "Qwen3-TTS 12Hz 1.7B model directory"
                       : "Qwen3-ASR 1.7B model directory",
                   "Model", &options->model);
  parser.AddOption("-c", "--context", "N",
                   tts ? "Context capacity (default: 4096)"
                       : "Context capacity per audio chunk (default: 1024)",
                   "Model", &options->context);
  parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                   "Model", &options->served_model_name);
  if (!tts)
    return;
  parser.AddCustomOption(
      "", "--voice", "NAME=PATH",
      "Register a named Qwen3-TTS Base voice from a reference WAV "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string path;
        if (!SplitNameValue(value, "--voice", &name, &path, err)) {
          return false;
        }
        options->voice_specs.emplace_back(std::move(name), std::move(path));
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-lang", "NAME=LANGUAGE",
      "Language a --voice speaks; used when a request omits 'language' "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string language;
        if (!SplitNameValue(value, "--voice-lang", &name, &language, err)) {
          return false;
        }
        if (!options->voice_lang_specs
                 .emplace(std::move(name), std::move(language))
                 .second) {
          *err = "duplicate --voice-lang name";
          return false;
        }
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-text", "NAME=TEXT|PATH",
      "Reference transcript for a --voice, given inline or as a file path; "
      "defaults to a .txt sidecar beside the WAV (repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string text;
        if (!SplitNameValue(value, "--voice-text", &name, &text, err)) {
          return false;
        }
        if (!options->voice_text_specs.emplace(std::move(name), std::move(text))
                 .second) {
          *err = "duplicate --voice-text name";
          return false;
        }
        return true;
      });
}

// The `--model` value wherever it appears in `args`; empty when absent.
std::string ModelArgument(std::span<const char* const> args) {
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if ((arg == "--model" || arg == "-m") && i + 1 < args.size())
      return args[i + 1];
    if (arg.starts_with("--model="))
      return std::string(arg.substr(8));
  }
  return {};
}

bool IsDiffusionCheckpoint(const std::filesystem::path& root) {
  try {
    return bubo::prepare::metadata::Parse(
               bubo::prepare::metadata::Read(root / "config.json"))
               .member_str("model_type") == "diffusion_gemma";
  } catch (const std::exception&) {
    return false;
  }
}

// Folder models have no GGUF header. Resolve their explicit or saved marker
// before choosing the release-profile path, including in a CPU-only build.
std::string DiffusionModelArgument(std::span<const char* const> args) {
  const auto model = ModelArgument(args);
  if (!model.empty())
    return IsDiffusionCheckpoint(std::filesystem::u8path(model)) ? model : "";
  std::filesystem::path manifest;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--manifest" && i + 1 < args.size())
      manifest = std::filesystem::u8path(args[++i]);
    else if (arg.starts_with("--manifest="))
      manifest = std::filesystem::u8path(arg.substr(11));
  }
  const auto selection = bubo::cli::ResolvePreparedDefault(
      {}, manifest, bubo::cli::EnvironmentUtf8("BUBO_MANIFEST"),
      bubo::cli::EnvironmentUtf8("LOCALAPPDATA"));
  if (selection.manifest.empty())
    return {};
  std::error_code ec;
  if (!std::filesystem::is_regular_file(selection.manifest, ec))
    return {};
  const auto marker = bubo::prepare::metadata::Parse(
      bubo::prepare::metadata::Read(selection.manifest));
  const auto* entry = bubo::cli::FindCatalogueEntry(
      bubo::cli::ShippedCatalogue().entries, marker.member_str("package_id"));
  const auto folder = selection.manifest.parent_path();
  if (!entry || entry->packaging != "safetensors_folder" ||
      !IsDiffusionCheckpoint(folder))
    return {};
  // Keep the normal marker schema validation for explicit/saved selection.
  (void)server::LoadModelPackage({}, "default", false, nullptr,
                                 selection.manifest);
  const auto text = folder.u8string();
  return {text.begin(), text.end()};
}

}  // namespace

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand) {
  if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser(std::string(program_name) + " serve image",
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }
  if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve video",
        "Start the MiniMax H3 text-to-video HTTP generation server.");
    parser.AddOption("-m", "--model", "DIR",
                     "Operator-supplied MiniMax H3 directory", "Model",
                     &video_model);
    parser.AddOption(
        "", "--root", "DIR",
        "Storage root for persistent video jobs (default: video-jobs)",
        "Storage", &video_root);
    parser.AddOption("", "--manifest", "PATH", "Pinned H3 manifest override",
                     "Storage", &video_manifest);
    parser.AddOption("", "--ttl", "SEC",
                     "Completed-artifact TTL in seconds (default: 3600)",
                     "Storage", &video_ttl_seconds);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "tts" || subcommand == "asr") {
    const bool tts = subcommand == "tts";
    SpeechOptions options{.context = tts ? 4096U : 1024U};
    ArgParser parser(
        std::string(program_name) + " serve " + std::string(subcommand),
        tts ? "Serve Qwen3-TTS speech synthesis."
            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, tts, &options);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "llm") {
    std::string model;
    std::string served_model_name;
    std::uint32_t max_context = 0;
    std::int64_t max_tokens = -1;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string draft_policy;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::string vision_model_path;
    std::size_t draft_tokens = 7;
    std::size_t min_draft_tokens = 1;
    std::string mtp_policy = "length";
    std::string mtp_draft_vocab = "full";
    bool prompt_lookup = false;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    std::size_t max_pending_requests_per_client = 0;
    std::uint64_t request_timeout_ms = 0;
    std::uint64_t stall_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::size_t cache_ram_bytes = 0;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        server::TextRunnerDiskCacheOptions::kDefaultCapacityBytes;
    std::size_t cache_disk_staging_bytes = 0;
    bool log_progress = false;
    std::string trace_path;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");

    // Model & Context
    parser.AddOption("-m", "--model", "PATH",
                     "GGUF model file or DiffusionGemma checkpoint directory "
                     "(default: the model that 'bubo prepare' installed)",
                     "Model", &model);
    parser.AddOption("", "--mmproj", "PATH",
                     "Matching BF16 vision file (auto-discovered beside model)",
                     "Model", &vision_model_path);
    parser.AddOption("", "--served-model-name", "ID",
                     "Model identifier exposed by the OpenAI API (default: "
                     "package ID or file stem)",
                     "Model", &served_model_name);
    parser.AddOption(
        "-c", "--context", "N",
        "Context tokens per session (default: 0 = model native context)",
        "Model", &max_context);

    // Sampling Defaults
    parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default new-token limit (default: -1 = until EOS or context full)",
        "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(parser, &sampling_config, "Sampling Defaults", true,
                            true);

    // Reasoning Defaults
    parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode: on, off, or auto (default: model)",
        "Reasoning Defaults", &reasoning_mode);
    parser.AddOption(
        "", "--reasoning-effort", "LEVEL",
        "Default effort: auto, minimal, low, medium, high, xhigh, or max",
        "Reasoning Defaults", &reasoning_effort);
    parser.AddOption("", "--preserve-thinking", "MODE",
                     "Replay prior reasoning: on, off, or auto",
                     "Reasoning Defaults", &preserve_thinking);

    // Speculative & Hardware
    parser.AddOption("", "--speculative", "MODE",
                     "HTTP draft backend: dspark, dflash2, mtp, or off",
                     "Speculative", &speculative_backend);
    parser.AddOption("", "--dflash-model", "PATH",
                     "Path to Qwen DFlash2 GGUF file", "Speculative",
                     &dflash_model_path);
    parser.AddOption(
        "", "--draft-policy", "POLICY",
        "Draft length: DFlash2 fixed or adaptive (default: adaptive); Gemma 4 "
        "MTP calibrated, confidence or fixed (default: calibrated)",
        "Speculative", &draft_policy);
    parser.AddOption("", "--dspark-model", "PATH",
                     "Path to DeepSeek V4 Flash DSpark support GGUF file",
                     "Speculative", &dspark_model_path);
    parser.AddOption("", "--mtp-model", "PATH",
                     "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the "
                     "mtp-...-shared-*.gguf file)",
                     "Speculative", &mtp_model_path);
    parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7; "
        "Gemma 4: 15)",
        "Speculative", &draft_tokens);

    parser.AddOption("", "--min-draft-tokens", "N",
                     "Adaptive draft floor (default: 1)", "Speculative",
                     &min_draft_tokens);
    parser.AddOption(
        "", "--mtp-policy", "POLICY",
        "Qwen3.8-Flash-Next sampled draft length: length (acceptance-history "
        "controller) or survival (stop on the draft head's confidence) "
        "(default: length)",
        "Speculative", &mtp_policy);
    parser.AddOption(
        "", "--mtp-draft-vocab", "VOCAB",
        "Qwen3.8-Flash-Next draft head vocabulary: full, latin (special and "
        "ASCII/Latin-script text tokens; half the output head per draft step), "
        "or auto (latin until the output leaves that set, then full) "
        "(default: full)",
        "Speculative", &mtp_draft_vocab);
    parser.AddFlag(
        "", "--prompt-lookup",
        "Qwen3.8-Flash-Next: after an MTP draft, propose the tokens that "
        "followed a 12+ token match earlier in the conversation",
        "Speculative", &prompt_lookup);
    parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512); "
        "DiffusionGemma: prompt tokens per encoder pass (default: 8192)",
        "Scheduling", &prefill_chunk_tokens);
    parser.AddOption("", "--max-pending", "N",
                     "Maximum queued generation requests (default: 16)",
                     "Scheduling", &max_pending_requests);
    parser.AddOption(
        "", "--max-pending-per-client", "N",
        "Maximum queued requests per client IP (default: --max-pending)",
        "Scheduling", &max_pending_requests_per_client);
    parser.AddOption(
        "", "--request-timeout-ms", "MS",
        "Queue plus generation timeout; 0 disables it (default: 0)",
        "Scheduling", &request_timeout_ms);
    parser.AddOption(
        "", "--stall-timeout-ms", "MS",
        "Exit as device_lost when generation makes no progress for this "
        "long; at least 60000, 0 disables it (default: 0)",
        "Scheduling", &stall_timeout_ms);
    parser.AddOption("", "--max-output-bytes", "N",
                     "Maximum generated bytes per request (default: 1048576)",
                     "Scheduling", &max_output_bytes);
    parser.AddOption("", "--max-buffered-output-bytes", "N",
                     "Maximum queued stream bytes per request (default: 65536)",
                     "Scheduling", &max_buffered_output_bytes);
    parser.AddOption(
        "", "--max-buffered-output-total", "N",
        "Maximum queued stream bytes across requests (default: 262144)",
        "Scheduling", &max_buffered_output_bytes_total);
    parser.AddOption(
        "", "--cache-ram-bytes", "N",
        "Retained RAM-cache byte budget (default: 0 = auto: half of free "
        "RAM, at most 32 GiB; explicit values may use free RAM minus 4 GiB)",
        "Cache", &cache_ram_bytes);
    parser.AddOption("", "--cache-disk", "DIR",
                     "Opt-in restart-safe continuation cache directory",
                     "Cache", &cache_disk_directory);
    parser.AddOption("", "--cache-disk-bytes", "N",
                     "Retained disk-cache byte budget (default: " +
                         std::to_string(cache_disk_bytes) + ")",
                     "Cache", &cache_disk_bytes);
    parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "RAM limit for queued snapshots and each disk read "
        "(default: 0 = auto, at most the disk budget and 1/8 available RAM)",
        "Cache", &cache_disk_staging_bytes);
    parser.AddFlag("", "--log-progress",
                   "Log live prefill and decode progress (needs "
                   "--log-level=info or debug)",
                   "Logging", &log_progress);
    parser.AddOption("", "--trace", "PATH", kTraceHelp, "Logging", &trace_path);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    std::cout
        << "\nProfiles:\n"
           "  --profile <default|subagents|exact|vanilla>\n"
           "  --exact                  Alias for --profile exact\n"
           "  --print-effective-settings  Print JSON and exit\n"
           "  --manifest PATH           Select bubo-package.json\n"
           "  --twin-model PATH --draft-head-model PATH\n"
           "  --verify-model            Hash eight manifest files plus "
           "distinct overrides\n"
           "                            (~2-4 minutes on SSD; before loading)\n"
           "  --no-behaviour --no-twin --no-draft-head --no-vision --no-mtp\n"
           "  --no-prompt-lookup --no-cache --no-cache-disk "
           "--allow-unmeasured-memory\n"
           "\nDiffusionGemma checkpoint directories:\n"
           "  Release profiles do not apply. --context, --sessions, "
           "--prefill-chunk, --max-tokens, --mmproj\n"
           "  and --served-model-name apply; the diffusion sampler replaces "
           "temperature, top-k,\n"
           "  top-p and penalties (a request seed still applies). Speculative "
           "decoding,\n"
           "  --cache-disk, --npu-prefill on and constrained JSON output are "
           "rejected.\n";
    return;
  }

  std::cout
      << "Usage: " << program_name
      << " serve [SERVER_OPTIONS] [COMMAND] [OPTIONS]\n\n"
      << "Start the OpenAI-compatible HTTP server for a specific model "
         "modality.\n\n"
      << "Commands:\n"
      << "  llm       Serve text LLM endpoints (/v1/chat/completions, "
         "/v1/completions) [default]\n"
      << "  video     Serve MiniMax H3 video generation endpoint "
         "(/v1/video/generations)\n"
      << "  image     Serve Qwen-Image-2.1 (/v1/images/generations, "
         "/v1/images/edits)\n"
      << "  tts       Serve Qwen3-TTS speech synthesis (/v1/audio/speech)\n"
      << "  asr       Serve Qwen3-ASR transcription "
         "(/v1/audio/transcriptions)\n\n"
      << "Server:\n"
      << "  -i, --host <IP>        Bind address (default: 127.0.0.1)\n"
      << "  -p, --port <N>         Port to listen on (default: 8080)\n"
      << "  -j, --sessions <N>     Preallocated GPU request sessions (default: "
         "1)\n"
      << "      --max-connections <N>\n"
      << "                         Maximum simultaneous HTTP connections "
         "(default: 16)\n"
      << "      --max-request-bytes <N>\n"
      << "                         Maximum HTTP request body bytes (default: "
         "8388608)\n"
      << "      --api-key <KEY>    Require Bearer authorization for requests"
         "; if absent, serve reads the key from BUBO_API_KEY\n"
      << "\nLogging:\n"
      << "      --log-level <LEVEL>\n"
      << "                         " << kLogLevelHelp << "\n"
      << "  -v, --verbose          " << kVerboseHelp << "\n"
      << "\nGeneral:\n"
      << "  -h, --help             Print help\n";
}

static int RunServeImpl(std::span<const char* const> args,
                        bubo::cli::ServeLifecycle* lifecycle,
                        const std::optional<std::filesystem::path>& log_path) {
  // Declared first: the final drain runs after model/server destructors log.
  struct LogShutdown {
    ~LogShutdown() {
      server::Logger::FlushSync(std::chrono::milliseconds(0));
      server::Logger::Shutdown();
    }
  } log_shutdown;
#ifdef GUFO_NPU
  std::unique_ptr<bubo::cli::NpuStatusPublisher> npu_status;
  if (lifecycle)
    npu_status = std::make_unique<bubo::cli::NpuStatusPublisher>(*lifecycle);
#endif
  // Release profiles read a GGUF header; a DiffusionGemma checkpoint directory
  // has none, so it starts without them.
  server::ReleaseStartup release;
  bool release_active = false;
  const auto diffusion_folder = DiffusionModelArgument(args);
  if (diffusion_folder.empty()) {
    release = server::PrepareReleaseSettings(args);
    release_active = true;
  }
  // Filled on the release-settings (LLM) path only; a DiffusionGemma directory
  // has no package or profile to estimate.
  bubo::cli::ReleaseMemoryEstimate memory_estimate;
  // Read before the model loads: the model's own weights are not yet resident.
  const auto memory_budgets = bubo::cli::ReadPlanningMemoryBudgets();
  if (!release.error.empty()) {
    if (lifecycle && lifecycle->MachineEnabled() && release.model_missing)
      throw bubo::cli::ControlError(12, release.error);
    if (lifecycle && lifecycle->MachineEnabled() && release.error_switch)
      throw server::ReleaseSettingError(*release.error_switch, release.error);
    server::Logger::FlushSync(std::chrono::seconds(2));
    std::cerr << "Error: " << release.error << '\n';
    return 2;
  }

#ifdef GUFO_NPU
  // Authorization denials never enter ServerPrefill::Start. Publish the same
  // existing reason once, without asking the status handler to query the NPU.
  if (!release.resolved.prefill_authorization.Allows(
          core::ExperimentalFeature::Prefill))
    npu::NoteNpuStartup(false, release.resolved.prefill_authorization.reason());
#endif

  if (release.active) {
    for (const auto& line : release.resolved.logs) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << line << '\n';
    }
  }
  std::string requested_prefill = "off";
  std::unique_ptr<bubo::cli::EndpointLease> endpoint_lease;
  std::unique_ptr<bubo::cli::PortReservation> reservation;
  auto lifecycle_settings = json::Value::object();
  const auto setting_row = [&](std::string name, json::Value effective,
                               std::string source = "default",
                               json::Value reason = {}) {
    auto row = json::Value::object();
    row["name"] = name;
    row["requested"] = json::Value();
    for (std::size_t i = 0; i < args.size(); ++i) {
      const std::string_view argument(args[i]);
      if (argument == name) {
        const auto& catalog = server::ReleaseSwitchTable();
        const auto entry = std::find_if(
            catalog.begin(), catalog.end(),
            [&](const auto& item) { return item.member_str("name") == name; });
        const bool is_flag = entry != catalog.end() &&
                             entry->find("takes_value") &&
                             !entry->find("takes_value")->as_bool();
        row["requested"] =
            !is_flag && i + 1 < args.size() &&
                    !std::string_view(args[i + 1]).starts_with("--")
                ? json::Value(args[i + 1])
                : json::Value(true);
        source = "user";
      } else if (argument.starts_with(name + "=")) {
        row["requested"] = std::string(argument.substr(name.size() + 1));
        source = "user";
      }
    }
    if (name == "--api-key")
      row["requested"] = json::Value("<redacted>");
    row["effective"] = std::move(effective);
    row["source"] = std::move(source);
    row["reason"] = std::move(reason);
    lifecycle_settings[name] = std::move(row);
  };
  for (const auto& setting : release.resolved.settings) {
    const auto& catalog = server::ReleaseSwitchTable();
    const auto entry =
        std::find_if(catalog.begin(), catalog.end(), [&](const auto& item) {
          return item.member_str("name") == setting.name;
        });
    if (entry != catalog.end() && entry->member_str("kind") == "env" &&
        entry->find("echo") && !entry->find("echo")->as_bool())
      continue;
    const auto value = setting.name == "--api-key" ? json::Value("<redacted>")
                       : setting.value             ? json::Value(*setting.value)
                                                   : json::Value();
    setting_row(setting.name, value, setting.origin,
                setting.skipped_reason.empty()
                    ? json::Value()
                    : json::Value(setting.skipped_reason));
  }
#ifdef GUFO_NPU
  std::shared_ptr<npu::ServerPrefill> persistent_prefill;
  bool prefill_started = false;
#endif
  std::string release_profile_name;
  std::string think_marker_bias;
  std::string loop_breaker;
  bool release_exact = false, print_effective_settings = false;
  bool no_twin = false, no_draft_head = false, no_vision = false;
  bool no_mtp = false, no_behaviour = false, no_prompt_lookup = false;
  bool no_cache = false, no_cache_disk = false, allow_unmeasured_memory = false;
  bool verify_model = false;
  std::string twin_model_path, draft_head_model_path, package_manifest;
  const auto add_release_options = [&](ArgParser& parser) {
    parser.AddOption("", "--manifest", "PATH", "Select bubo-package.json",
                     "Profiles", &package_manifest);
    parser.AddOption("", "--twin-model", "PATH", "Select the dense twin",
                     "Profiles", &twin_model_path);
    parser.AddOption("", "--draft-head-model", "PATH",
                     "Select the MTP draft head", "Profiles",
                     &draft_head_model_path);
    parser.AddFlag(
        "", "--verify-model",
        "Hash eight manifest files plus distinct overrides (~2-4 min on SSD)",
        "Profiles", &verify_model);
    parser.AddOption("", "--npu-prefill", "off|on",
                     "NPU prefill when available (default: off; exact and "
                     "vanilla: always off)",
                     "Profiles", &requested_prefill);
    parser.AddOption("", "--profile", "NAME",
                     "Release profile: default, subagents, exact, or vanilla",
                     "Profiles", &release_profile_name);
    parser.AddOption("", "--think-marker-bias", "BIAS",
                     "Set the thinking marker bias for every effort",
                     "Reasoning Defaults", &think_marker_bias);
    parser.AddOption("", "--loop-breaker", "on|off",
                     "Control the repeated-output loop breaker",
                     "Reasoning Defaults", &loop_breaker);
    parser.AddFlag("", "--exact", "Use the exact profile", "Profiles",
                   &release_exact);
    parser.AddFlag("", "--print-effective-settings",
                   "Print settings as JSON and exit before model loading",
                   "Profiles", &print_effective_settings);
    parser.AddFlag("", "--no-twin", "Suppress the manifest twin", "Profiles",
                   &no_twin);
    parser.AddFlag("", "--no-draft-head", "Suppress the manifest draft head",
                   "Profiles", &no_draft_head);
    parser.AddFlag("", "--no-vision", "Suppress the vision file", "Profiles",
                   &no_vision);
    parser.AddFlag("", "--no-mtp", "Disable MTP defaults", "Profiles", &no_mtp);
    parser.AddFlag("", "--no-behaviour", "Disable the nine behaviour defaults",
                   "Profiles", &no_behaviour);
    parser.AddFlag("", "--no-prompt-lookup", "Disable prompt lookup",
                   "Profiles", &no_prompt_lookup);
    parser.AddFlag("", "--no-cache",
                   "Disable RAM and disk prompt caches for all requests",
                   "Profiles", &no_cache);
    parser.AddFlag("", "--no-cache-disk", "Disable the disk cache", "Profiles",
                   &no_cache_disk);
    parser.AddFlag("", "--allow-unmeasured-memory",
                   "Allow context above the measured subagent limit",
                   "Profiles", &allow_unmeasured_memory);
  };
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  // An exception that escapes a load path terminates without unwinding, so no
  // destructor reports it and the process exits with an empty log. Naming the
  // reason here is the difference between a diagnosable failure and a server
  // that simply vanished after event=load_started.
  ReportTerminationReason();
  InstallFatalSignalReporter();

  std::string host = "127.0.0.1";
  if (const char* env_host = std::getenv("HOST");
      env_host != nullptr && *env_host != '\0') {
    host = env_host;
  } else if (const char* env_strix_host = std::getenv("GUFO_HOST");
             env_strix_host != nullptr && *env_strix_host != '\0') {
    host = env_strix_host;
  }

  int port = 8080;
  if (const char* env_port = std::getenv("PORT");
      env_port != nullptr && *env_port != '\0') {
    const std::string_view sv(env_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  } else if (const char* env_strix_port = std::getenv("GUFO_PORT");
             env_strix_port != nullptr && *env_strix_port != '\0') {
    const std::string_view sv(env_strix_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  }

  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  ServerLogOptions log_options;
  auto active_model = json::Value();
  const char* debug_environment = std::getenv("GUFO_DEBUG_HOOKS");
  bool debug_hooks =
      debug_environment && std::string_view(debug_environment) == "1";
  std::size_t max_score_tokens = 8192;
  std::shared_ptr<const json::Value> debug_settings;

  // Identify the modality only after leading server options. The selected
  // parser consumes every option together, so --served-model-name -v and
  // --model audio cannot be mistaken for server flags or subcommands.
  ArgParser server_parser(gufo::cli::ProductCommand("serve"));
  const auto add_server_options = [&](ArgParser& parser,
                                      bool include_sessions = true) {
    AddServerOptions(parser, &host, &port,
                     include_sessions ? &session_count : nullptr,
                     &max_connections, &max_request_body_bytes, &api_key,
                     &log_options, &debug_hooks, &max_score_tokens);
  };
  add_server_options(server_parser);
  add_release_options(server_parser);
  try {
    if (release.active)
      server::PrefillReleaseFlags(release.resolved, server_parser);
  } catch (const std::exception& error) {
    server::Logger::FlushSync(std::chrono::seconds(2));
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
  std::string subcommand = "llm";
  std::vector<const char*> sub_args(args.begin(), args.end());
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    const auto equals = arg.find('=');
    const auto* option = server_parser.FindOption(arg.substr(0, equals));
    if (option != nullptr) {
      if (!option->is_flag && equals == std::string_view::npos) {
        ++i;
      }
      continue;
    }
    if (arg == "llm" || arg == "video" || (arg == "tts" || arg == "asr") ||
        arg == "image") {
      subcommand = arg;
      sub_args.erase(sub_args.begin() + static_cast<std::ptrdiff_t>(i));
    } else if (arg == "--help" || arg == "-h" || arg == "help") {
      const std::string_view topic =
          arg == "help" && i + 1 < args.size() ? args[i + 1] : "";
      if (!topic.empty() && topic != "llm" && topic != "video" &&
          topic != "tts" && topic != "asr" && topic != "image") {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error: unknown serve command '" << topic << "'\n";
        return 2;
      }
      PrintServeHelp(gufo::cli::kProductName, topic);
      return 0;
    }
    break;
  }
  // Video jobs load weights on demand. They cannot satisfy this command's
  // ready-after-weight-loading contract. Keep the legacy human command.
  if (subcommand == "video" && lifecycle && lifecycle->MachineEnabled())
    throw bubo::cli::ControlError(
        18, "Video lifecycle cannot certify loaded weights.");
  // Hardware queues are claimed on the first HIP dispatch and held for the
  // lifetime of the process, so the budget is decided once per invocation,
  // after the modality parser has armed the log threshold and before any
  // model loads. It lives in `prepare_server_options` so the emitted
  // `queue_budget` line already obeys `--log-level`. Help output loads nothing
  // and returns before that point.

  const auto prepare_server_options = [&] {
    in_addr address{};
    if (::inet_pton(AF_INET, host.c_str(), &address) != 1) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --host must be an IPv4 address\n";
      return false;
    }
    if (port < 0 || port > 65535) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --port must be between 0 and 65535\n";
      return false;
    }
    if (max_score_tokens == 0) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --max-score-tokens must be positive\n";
      return false;
    }
    if (debug_hooks && subcommand != "llm") {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr
          << "GUFO_DEBUG_HOOKS/--debug-hooks ignored for non-text server\n";
      debug_hooks = false;
    }
    if (session_count == 0 || max_connections == 0 ||
        max_request_body_bytes == 0) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: server limits must be positive\n";
      return false;
    }
    // Arm verbosity here: this runs after every server option is parsed and
    // before the first loader, cache or HTTP line is written, so one call sets
    // the level for all modalities. `-v/--verbose` is shorthand for
    // `--log-level=debug`, so the two spellings together are a conflict to
    // report, not a precedence to resolve silently.
    if (log_options.verbose && log_options.level.has_value()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --verbose is shorthand for --log-level=debug, so "
                   "it cannot combine with --log-level\n";
      return false;
    }
    const server::LogLevel resolved =
        log_options.verbose
            ? server::LogLevel::kDebug
            : log_options.level.value_or(server::LogLevel::kInfo);
    server::Logger::SetLevel(resolved);
    // The key can arrive through the environment so that it stays off the
    // process command line. The flag wins; the key itself is never logged.
    if (const auto env_key = bubo::cli::EnvironmentUtf8("BUBO_API_KEY");
        !env_key.empty()) {
      if (api_key.empty())
        api_key = env_key;
      else if (api_key != env_key)
        server::Logger::Warn(
            "server", "event=api_key_conflict used=flag ignored=BUBO_API_KEY");
    }
    if (lifecycle) {
      endpoint_lease = std::make_unique<bubo::cli::EndpointLease>(host, port);
      endpoint_lease->Acquire();
      reservation = std::make_unique<bubo::cli::PortReservation>(host, port);
      reservation->Reserve();
      port = reservation->Port();
      release.resolved.runtime_host = host;
      release.resolved.runtime_port = port;
      setting_row(
          "--host", host,
          std::getenv("HOST") || std::getenv("GUFO_HOST") ? "env" : "default");
      setting_row(
          "--port", port,
          std::getenv("PORT") || std::getenv("GUFO_PORT") ? "env" : "default");
      setting_row("--sessions", session_count);
      setting_row("--max-connections", max_connections);
      setting_row("--max-request-body-bytes", max_request_body_bytes);
      setting_row("--log-level", std::string(server::LogLevelName(resolved)));
      lifecycle->Configure(
          "http://" + host + ":" + std::to_string(port) + "/v1", subcommand,
          lifecycle_settings, log_path);
      lifecycle->OpenControl();
      if (lifecycle->StopRequested())
        throw bubo::cli::ControlError(0, "Start was stopped.");
      lifecycle->Transition("loading");
    }
    // The queue plan is applied only after a valid invocation is established,
    // and the `queue_budget` line is emitted now so an absolute threshold
    // (`--log-level=error`) suppresses this INFO startup diagnostic with the
    // rest of the boot sequence instead of leaking it before SetLevel ran.
    const auto profile = subcommand == "llm" ? diagnostics::QueueProfile::kText
                         : (subcommand == "tts" || subcommand == "asr")
                             ? diagnostics::QueueProfile::kAudio
                             : diagnostics::QueueProfile::kUnmeasured;
    const auto plan =
        diagnostics::PlanQueues(profile, diagnostics::QueryQueueCensus(),
                                std::getenv("GPU_MAX_HW_QUEUES"));
    diagnostics::ApplyQueuePlan(plan);
    server::Logger::Info("gpu",
                         diagnostics::DescribeQueuePlan(subcommand, plan));
    if (subcommand == "llm")
      server::EnableProcessMemoryGuard();
    if (plan.may_exceed_budget) {
      server::Logger::Warn("gpu", diagnostics::DescribeQueuePressure(plan));
      if (lifecycle) {
        auto warning = json::Value::object();
        warning["code"] = "queue_pressure";
        warning["remedy"] = "Reduce active GPU workloads.";
        lifecycle->Transition("warning", std::move(warning));
      }
    }
    // One WARN line when the GPU reservation or page file differs from the
    // tested machine. Never a refusal: a failing probe is logged and startup
    // carries on.
    try {
      auto machine_probe = bubo::cli::CreateSystemMachineProbe();
      const auto machine_line =
          bubo::cli::MachineWarningLogLine(bubo::cli::EvaluateMachineWarnings(
              bubo::cli::ReadMachineWarningInputs(*machine_probe)));
      if (!machine_line.empty())
        server::Logger::Warn("machine", machine_line);
    } catch (...) {
      try {
        server::Logger::Warn("machine",
                             "event=machine_settings status=unreadable");
      } catch (...) {
      }
    }
    // Deliberately emitted before the model opens: a load that fails or hangs
    // never reaches the INFO `event=listening` banner in HttpServer::run, so
    // this is the only record of the resolved options and the armed threshold.
    // The overlap on the listener limits is intentional: the two lines mark
    // different points in the lifecycle at different tiers, so neither can
    // replace the other.
    const std::string options_line =
        "event=options host=" + host + " port=" + std::to_string(port) +
        " max_connections=" + std::to_string(max_connections) +
        " max_request_bytes=" + std::to_string(max_request_body_bytes) +
        " api_key=" + (api_key.empty() ? "unset" : "set") +
        " log_level=" + std::string(server::LogLevelName(resolved));
    server::Logger::Debug("server", options_line);
    return true;
  };
  std::string parse_err;

  std::shared_ptr<server::TextGenerationBackend> backend;
  std::shared_ptr<server::VideoJobService> video_jobs;
  std::shared_ptr<server::TtsService> tts;
  std::shared_ptr<server::AsrService> asr;
  std::shared_ptr<server::ImageService> images;

  if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser(gufo::cli::ProductCommand("serve image"),
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp(gufo::cli::kProductName, "image");
      return 0;
    }
    if (!prepare_server_options())
      return 2;
    if (model.empty()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --model <DIR> is required for image server\n";
      return 2;
    }
    ModelLoadLog load_log("qwen_image_21", model);
    try {
      images = std::make_shared<server::ImageService>(model, name);
    } catch (const std::exception& error) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error loading Qwen-Image-2.1: " << error.what() << '\n';
      return 1;
    }
    load_log.Complete("model=" + name + " weights=mapped upload=on_demand");
  } else if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser video_parser(
        gufo::cli::ProductCommand("serve video"),
        "Start the MiniMax H3 video generation server.");
    video_parser.AddOption("-m", "--model", "DIR",
                           "Operator-supplied MiniMax H3 directory", "Model",
                           &video_model);
    video_parser.AddOption("", "--root", "DIR",
                           "Storage root for persistent video jobs", "Storage",
                           &video_root);
    video_parser.AddOption("", "--manifest", "PATH",
                           "Pinned H3 manifest override", "Storage",
                           &video_manifest);
    video_parser.AddOption("", "--ttl", "SEC",
                           "Completed-artifact TTL in seconds", "Storage",
                           &video_ttl_seconds);

    add_server_options(video_parser);
    if (!video_parser.Parse(sub_args, &parse_err)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp(gufo::cli::kProductName, "video");
      return 2;
    }
    if (video_parser.IsHelpRequested()) {
      PrintServeHelp(gufo::cli::kProductName, "video");
      return 0;
    }
    if (!prepare_server_options()) {
      return 2;
    }

    if (video_ttl_seconds == 0 ||
        video_ttl_seconds >
            static_cast<std::uint64_t>(std::chrono::seconds::max().count())) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --ttl must be a positive duration\n";
      return 2;
    }

    if (video_model.empty()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --model <DIR> is required for video server\n";
      PrintServeHelp(gufo::cli::kProductName, "video");
      return 2;
    }

    ModelLoadLog load_log("video_inventory", video_model);
    video_jobs = std::make_shared<server::VideoJobService>(
        server::VideoJobServiceOptions{
            .model_root = video_model,
            .source_manifest = video_manifest,
            .storage_root = video_root,
            .queue_capacity = 1,
            .artifact_ttl = std::chrono::seconds(
                static_cast<std::chrono::seconds::rep>(video_ttl_seconds)),
            .validate_model_inventory = true,
            .id_factory = {},
            .now = {},
            .runner = {},
        });
    if (!video_jobs->ready()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error enabling MiniMax H3 video service: "
                << video_jobs->initialization_error() << '\n';
      return 1;
    }
    load_log.Complete(
        "model=minimax-h3 sessions=1 queue_capacity=1 weights=lazy", false);
  } else if (subcommand == "tts" || subcommand == "asr") {
    const bool is_tts = subcommand == "tts";
    SpeechOptions options{.context = is_tts ? 4096U : 1024U};
    ArgParser parser(gufo::cli::ProductCommand("serve " + subcommand),
                     is_tts ? "Serve Qwen3-TTS speech synthesis."
                            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, is_tts, &options);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp(gufo::cli::kProductName, subcommand);
      return 0;
    }
    if (!prepare_server_options())
      return 2;
    if (options.model.empty()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --model <DIR> is required for " << subcommand
                << " server\n";
      return 2;
    }
    if (options.context < (is_tts ? 1U : 32U)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --context must be at least " << (is_tts ? 1 : 32)
                << '\n';
      return 2;
    }
    setting_row("--context", static_cast<unsigned long long>(options.context));
    ModelLoadLog load_log(subcommand, options.model);
    if (is_tts) {
      std::map<std::string, server::TtsVoicePreset> voice_presets;
      if (!BuildVoicePresets(options.voice_specs, options.voice_text_specs,
                             options.voice_lang_specs, &voice_presets,
                             &parse_err)) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error: " << parse_err << '\n';
        return 2;
      }
      try {
        tts = std::make_shared<server::TtsService>(server::TtsServiceOptions{
            .model_root = options.model,
            .native_context_tokens = options.context,
            .validate_model = true,
            .model_id = options.served_model_name,
            .voices = {},
            .voice_presets = std::move(voice_presets),
            .runner = {},
        });
      } catch (const std::exception& error) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error loading Qwen3-TTS: " << error.what() << '\n';
        return 1;
      }
      if (!tts->ready()) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error enabling Qwen3-TTS service: "
                  << tts->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + tts->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    } else {
      try {
        asr = std::make_shared<server::AsrService>(server::AsrServiceOptions{
            .model_root = options.model,
            .native_context_tokens = options.context,
            .validate_model = true,
            .model_id = options.served_model_name,
            .runner = {},
        });
      } catch (const std::exception& error) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error loading Qwen3-ASR: " << error.what() << '\n';
        return 1;
      }
      if (!asr->ready()) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error enabling Qwen3-ASR service: "
                  << asr->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + asr->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    }
  } else {
    // Default to LLM server
    std::string model = diffusion_folder;
    std::string served_model_name;
    std::uint32_t max_context = 0;
    std::int64_t max_tokens = -1;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string draft_policy;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::string vision_model_path;
    std::size_t draft_tokens = 7;
    // Without --draft-tokens each model picks its own default (Gemma 4: 15).
    bool draft_tokens_given = false;
    std::size_t min_draft_tokens = 1;
    std::string mtp_policy = "length";
    std::string mtp_draft_vocab = "full";
    bool prompt_lookup = false;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    // Defaults to --max-pending after parsing unless supplied.
    std::size_t max_pending_requests_per_client = 0;
    std::uint64_t request_timeout_ms = 0;
    std::uint64_t stall_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::size_t cache_ram_bytes = 0;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        server::TextRunnerDiskCacheOptions::kDefaultCapacityBytes;
    std::size_t cache_disk_staging_bytes = 0;
    bool log_progress = false;
    std::string trace_path;

    gufo::cli::ArgParser llm_parser(
        gufo::cli::ProductCommand("serve llm"),
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    llm_parser.AddOption("-m", "--model", "PATH",
                         "GGUF model file or DiffusionGemma checkpoint "
                         "directory (default: the model that 'bubo prepare' "
                         "installed)",
                         "Model", &model);
    llm_parser.AddOption(
        "", "--mmproj", "PATH",
        "Matching BF16 vision file (auto-discovered beside model)", "Model",
        &vision_model_path);
    llm_parser.AddOption("", "--served-model-name", "ID",
                         "Model identifier exposed by the OpenAI API (default: "
                         "package ID or file stem)",
                         "Model", &served_model_name);
    llm_parser.AddOption(
        "-c", "--context", "N",
        "Context tokens per session (default: 0 = model native context)",
        "Model", &max_context);
    llm_parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default new-token limit (default: -1 = until EOS or context full)",
        "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(llm_parser, &sampling_config, "Sampling Defaults",
                            true, true);
    llm_parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode: on, off, or auto (default: model)",
        "Reasoning Defaults", &reasoning_mode);
    llm_parser.AddOption(
        "", "--reasoning-effort", "LEVEL",
        "Default effort: auto, minimal, low, medium, high, xhigh, or max",
        "Reasoning Defaults", &reasoning_effort);
    llm_parser.AddOption("", "--preserve-thinking", "MODE",
                         "Replay prior reasoning: on, off, or auto",
                         "Reasoning Defaults", &preserve_thinking);
    llm_parser.AddOption("", "--speculative", "MODE",
                         "HTTP draft backend: dspark, dflash2, mtp, or off",
                         "Speculative", &speculative_backend);
    llm_parser.AddOption("", "--dflash-model", "PATH",
                         "Path to Qwen DFlash2 GGUF file", "Speculative",
                         &dflash_model_path);
    llm_parser.AddOption(
        "", "--draft-policy", "POLICY",
        "Draft length: DFlash2 fixed or adaptive (default: adaptive); Gemma 4 "
        "MTP calibrated, confidence or fixed (default: calibrated)",
        "Speculative", &draft_policy);
    llm_parser.AddOption("", "--dspark-model", "PATH",
                         "Path to DeepSeek V4 Flash DSpark support GGUF file",
                         "Speculative", &dspark_model_path);
    llm_parser.AddOption(
        "", "--mtp-model", "PATH",
        "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the "
        "mtp-...-shared-*.gguf file)",
        "Speculative", &mtp_model_path);
    llm_parser.AddCustomOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7; "
        "Gemma 4: 15)",
        "Speculative",
        [&](std::string_view flag, std::string_view value, std::string* error) {
          std::size_t count = 0;
          const auto [ptr, ec] =
              std::from_chars(value.data(), value.data() + value.size(), count);
          if (ec != std::errc{} || ptr != value.data() + value.size() ||
              count == 0) {
            *error = std::string(flag) + " must be a positive integer";
            return false;
          }
          draft_tokens = count;
          draft_tokens_given = true;
          return true;
        });

    llm_parser.AddOption("", "--min-draft-tokens", "N",
                         "Adaptive draft floor (default: 1)", "Speculative",
                         &min_draft_tokens);
    llm_parser.AddOption(
        "", "--mtp-policy", "POLICY",
        "Qwen3.8-Flash-Next sampled draft length: length (acceptance-history "
        "controller) or survival (stop on the draft head's confidence) "
        "(default: length)",
        "Speculative", &mtp_policy);
    llm_parser.AddOption(
        "", "--mtp-draft-vocab", "VOCAB",
        "Qwen3.8-Flash-Next draft head vocabulary: full, latin (special and "
        "ASCII/Latin-script text tokens; half the output head per draft step), "
        "or auto (latin until the output leaves that set, then full) "
        "(default: full)",
        "Speculative", &mtp_draft_vocab);
    llm_parser.AddFlag(
        "", "--prompt-lookup",
        "Qwen3.8-Flash-Next: after an MTP draft, propose the tokens that "
        "followed a 12+ token match earlier in the conversation",
        "Speculative", &prompt_lookup);
    llm_parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512); "
        "DiffusionGemma: prompt tokens per encoder pass (default: 8192)",
        "Scheduling", &prefill_chunk_tokens);
    llm_parser.AddOption("", "--max-pending", "N",
                         "Maximum queued generation requests (default: 16)",
                         "Scheduling", &max_pending_requests);
    llm_parser.AddOption(
        "", "--max-pending-per-client", "N",
        "Maximum queued requests per client IP (default: --max-pending)",
        "Scheduling", &max_pending_requests_per_client);
    llm_parser.AddOption(
        "", "--request-timeout-ms", "MS",
        "Queue plus generation timeout; 0 disables it (default: 0)",
        "Scheduling", &request_timeout_ms);
    llm_parser.AddOption(
        "", "--stall-timeout-ms", "MS",
        "Exit as device_lost when generation makes no progress for this "
        "long; at least 60000, 0 disables it (default: 0)",
        "Scheduling", &stall_timeout_ms);
    llm_parser.AddOption(
        "", "--max-output-bytes", "N",
        "Maximum generated bytes per request (default: 1048576)", "Scheduling",
        &max_output_bytes);
    llm_parser.AddOption(
        "", "--max-buffered-output-bytes", "N",
        "Maximum queued stream bytes per request (default: 65536)",
        "Scheduling", &max_buffered_output_bytes);
    llm_parser.AddOption(
        "", "--max-buffered-output-total", "N",
        "Maximum queued stream bytes across requests (default: 262144)",
        "Scheduling", &max_buffered_output_bytes_total);
    llm_parser.AddOption(
        "", "--cache-ram-bytes", "N",
        "Retained RAM-cache byte budget (default: 0 = auto: half of free "
        "RAM, at most 32 GiB; explicit values may use free RAM minus 4 GiB)",
        "Cache", &cache_ram_bytes);
    llm_parser.AddOption("", "--cache-disk", "DIR",
                         "Opt-in restart-safe continuation cache directory",
                         "Cache", &cache_disk_directory);
    llm_parser.AddOption("", "--cache-disk-bytes", "N",
                         "Retained disk-cache byte budget (default: " +
                             std::to_string(cache_disk_bytes) + ")",
                         "Cache", &cache_disk_bytes);
    llm_parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "RAM limit for queued snapshots and each disk read "
        "(default: 0 = auto, at most the disk budget and 1/8 available RAM)",
        "Cache", &cache_disk_staging_bytes);
    llm_parser.AddFlag("", "--log-progress",
                       "Log live prefill and decode progress (needs "
                       "--log-level=info or debug)",
                       "Logging", &log_progress);
    llm_parser.AddOption("", "--trace", "PATH", kTraceHelp, "Logging",
                         &trace_path);

    add_server_options(llm_parser);
    add_release_options(llm_parser);
    try {
      if (release.active)
        server::PrefillReleaseFlags(release.resolved, llm_parser);
    } catch (const std::exception& error) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << error.what() << '\n';
      return 2;
    }
    if (!llm_parser.Parse(sub_args, &parse_err)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp(gufo::cli::kProductName, "llm");
      return 2;
    }
    if (llm_parser.IsHelpRequested()) {
      PrintServeHelp(gufo::cli::kProductName, "llm");
      return 0;
    }
    if (!diffusion_folder.empty() && vision_model_path.empty() && !no_vision) {
      const auto* entry = bubo::cli::FindCatalogueEntry(
          bubo::cli::ShippedCatalogue().entries,
          "diffusiongemma-26b-e38-abliterated-nvfp4");
      if (entry)
        for (const auto& file : entry->files)
          if (file.role == "mmproj") {
            const auto candidate = std::filesystem::u8path(model) /
                                   std::filesystem::u8path(file.name);
            std::error_code ec;
            if (std::filesystem::is_regular_file(candidate, ec) &&
                std::filesystem::file_size(candidate, ec) == file.bytes) {
              const auto text = candidate.u8string();
              vision_model_path.assign(text.begin(), text.end());
            }
          }
    }
    const auto memory_policy = bubo::cli::ResolveServeMemoryPolicy(
        release.resolved,
        release.package.manifest_path.empty() ? "" : release.package.package_id,
        std::filesystem::u8path(model), session_count, memory_budgets,
        llm_parser.WasSupplied("--context"), print_effective_settings,
        release_active);
    if (memory_policy.context) {
      max_context = static_cast<std::uint32_t>(*memory_policy.context);
      server::Logger::Info(
          "settings",
          "--context: " + release.resolved.context_resolution_reason);
    }
    if (print_effective_settings) {
      if (IsDiffusionCheckpoint(std::filesystem::u8path(model))) {
        // A DiffusionGemma checkpoint directory starts without release
        // profiles, so there is no resolved state to print.
        auto settings = gufo::json::Value::object();
        settings["profile"] = "none";
        settings["architecture"] = "diffusion-gemma";
        settings["model"] = model;
        settings["mmproj"] = vision_model_path.empty()
                                 ? json::Value()
                                 : json::Value(vision_model_path);
        settings["note"] =
            "release profiles do not apply to a DiffusionGemma checkpoint "
            "directory";
        std::cout << settings.dump() << '\n';
        return 0;
      }
      auto settings = server::ReleaseSettingsJson(release.resolved);
      settings["memory_estimate"] = memory_policy.estimate;
      settings["memory_estimate_reason"] =
          memory_policy.estimate_reason.empty()
              ? json::Value()
              : json::Value(memory_policy.estimate_reason);
      {
        // The AMD runtime folder in use and whether it is the shared
        // per-user folder or a runtime beside bubo.exe.
        const auto runtime = bubo::prepare::ResolveProcessRuntime(
            bubo::cli::EnvironmentUtf8("LOCALAPPDATA"),
            bubo::cli::ExecutableDirectory());
        auto kind = bubo::cli::EnvironmentUtf8(
            std::string(bubo::prepare::kRuntimeKindEnvironment));
        if (kind.empty())
          kind = std::string(bubo::prepare::RuntimeKindName(runtime.kind));
        const auto dir = runtime.dir.u8string();
        auto row = json::Value::object();
        row["dir"] = std::string(dir.begin(), dir.end());
        row["kind"] = kind;
        row["ready"] = runtime.error.empty();
        settings["runtime"] = row;
      }
      std::cout << settings.dump() << '\n';
      return 0;
    }
    if (!release.package.files.empty() &&
        (model.empty() ||
         std::filesystem::is_directory(std::filesystem::u8path(model)))) {
      const auto path = release.package.files.front().path.u8string();
      model.assign(path.begin(), path.end());
    }
    if (no_mtp) {
      speculative_backend = "off";
      mtp_model_path.clear();
    }
    if (no_vision)
      vision_model_path.clear();
    if (no_prompt_lookup)
      prompt_lookup = false;
    if (no_cache || no_cache_disk)
      cache_disk_directory.clear();
    if (!llm_parser.WasSupplied("--max-pending-per-client")) {
      max_pending_requests_per_client = max_pending_requests;
    }
    if (!prepare_server_options()) {
      return 2;
    }
    // Progress lines are INFO-tier. Under a quieter threshold `--log-progress`
    // would be accepted and then silently discarded, so reject the combination
    // instead of ignoring an option the caller asked for.
    if (log_progress && !server::Logger::Enabled(server::LogLevel::kInfo)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --log-progress needs --log-level=info or "
                   "--log-level=debug\n";
      return 2;
    }
    // Opened before the model loads, so an unwritable path fails at once
    // instead of after the weights are resident.
    if (!trace_path.empty()) {
      if (const auto error = server::Trace::Open(trace_path)) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error: cannot open --trace file '" << trace_path
                  << "': " << *error << "\n";
        return 2;
      }
      server::Logger::Warn("trace",
                           "event=trace_enabled path=" + trace_path +
                               " content=prompts,generated_text,replies");
    }
    sampling::SamplingConfig validated_sampling;
    const bool sampling_valid = !server::ParseSamplingConfig(
        json::Value::object(), sampling_config, &validated_sampling);
    if (max_tokens < -1 || max_tokens == 0 ||
        max_tokens > std::numeric_limits<std::uint32_t>::max() ||
        prefill_chunk_tokens == 0 || max_pending_requests == 0 ||
        max_pending_requests_per_client == 0 ||
        max_pending_requests_per_client > max_pending_requests ||
        max_output_bytes == 0 || max_buffered_output_bytes == 0 ||
        max_buffered_output_bytes_total == 0 ||
        !server::CacheDiskLimitsValid(!cache_disk_directory.empty(),
                                      cache_disk_bytes,
                                      cache_disk_staging_bytes) ||
        request_timeout_ms > static_cast<std::uint64_t>(
                                 std::chrono::milliseconds::max().count()) ||
        stall_timeout_ms > static_cast<std::uint64_t>(
                               std::chrono::milliseconds::max().count()) ||
        (stall_timeout_ms != 0 && stall_timeout_ms < kMinStallTimeoutMs) ||
        !sampling_valid) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: sampling and scheduling limits are invalid\n";
      return 2;
    }
    if (draft_tokens == 0 || min_draft_tokens == 0 ||
        min_draft_tokens > draft_tokens ||
        draft_tokens > std::numeric_limits<std::uint32_t>::max()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: speculative draft limits are invalid\n";
      return 2;
    }
    const auto reasoning_defaults = ResolveReasoningDefaults(
        reasoning_mode, reasoning_effort, preserve_thinking, &parse_err);
    if (!reasoning_defaults.has_value()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << parse_err << "\n";
      return 2;
    }

    server::TextSpeculativeConfig speculative_config;
    speculative_config.resolved_artefacts = release.resolved.artefacts.present;
    speculative_config.twin_model_path = release.resolved.artefacts.twin;
    speculative_config.draft_head_model_path =
        release.resolved.artefacts.draft_head;
    speculative_config.vision_disabled =
        release.resolved.artefacts.vision_disabled;
    if (speculative_backend.empty() && !dspark_model_path.empty()) {
      speculative_backend = "dspark";
    }
    if (speculative_backend.empty() || speculative_backend == "off") {
      speculative_config.backend = server::TextSpeculativeBackend::kDisabled;
    } else if (speculative_backend == "dflash2") {
      speculative_config.backend = server::TextSpeculativeBackend::kDFlash;
    } else if (speculative_backend == "dspark") {
      speculative_config.backend = server::TextSpeculativeBackend::kDSpark;
    } else if (speculative_backend == "mtp") {
      speculative_config.backend = server::TextSpeculativeBackend::kMtp;
    } else {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: speculative backend '" << speculative_backend
                << "' is not supported by the HTTP server\n";
      return 2;
    }
    try {
      const bool mtp =
          speculative_config.backend == server::TextSpeculativeBackend::kMtp;
      if (!draft_policy.empty() && !mtp &&
          speculative_config.backend != server::TextSpeculativeBackend::kDFlash)
        throw std::invalid_argument("--draft-policy requires DFlash2 or MTP");
      if (mtp) {
        // The MTP model validates its own policy names at load.
        speculative_config.mtp_draft_policy = draft_policy;
      } else {
        speculative_config.dflash_policy =
            speculative::ParseDFlashDraftPolicy(draft_policy);
      }
    } catch (const std::invalid_argument& exception) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: " << exception.what() << '\n';
      return 2;
    }
    if (speculative_config.backend == server::TextSpeculativeBackend::kDFlash &&
        (dflash_model_path.empty() || min_draft_tokens != 1)) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: DFlash2 requires --dflash-model and "
                   "--min-draft-tokens 1; bound blocks with --draft-tokens\n";
      return 2;
    }
    speculative_config.draft_model_path =
        speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? dspark_model_path
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? mtp_model_path
            : dflash_model_path;
    speculative_config.max_draft_tokens =
        static_cast<std::uint32_t>(draft_tokens);
    speculative_config.max_draft_tokens_given = draft_tokens_given;
    speculative_config.min_draft_tokens =
        static_cast<std::uint32_t>(min_draft_tokens);
    if (model.empty()) {
      server::Logger::FlushSync(std::chrono::seconds(2));
      std::cerr << "Error: --model <PATH> is required\n";
      if (lifecycle && lifecycle->MachineEnabled())
        throw bubo::cli::ControlError(12, "Model is missing.");
      return 2;
    }
    active_model = bubo::cli::ActiveModelIdentity(
        std::filesystem::u8path(model), release.package.manifest_path,
        release.package.manifest_path.empty() ? std::string_view{}
                                              : release.package.package_id);
#if defined(ENGINE_ENABLE_HIP)
    const bool diffusion_gemma =
        models::diffusion_gemma::IsDiffusionGemmaDirectory(
            std::filesystem::u8path(model));
#else
    const bool diffusion_gemma = false;
#endif
    if (diffusion_gemma) {
#if defined(ENGINE_ENABLE_HIP)
      // The diffusion engine has no draft model, continuation
      // snapshot or NPU path: refuse these options before anything loads.
      const char* unsupported = server::UnsupportedDiffusionGemmaOption(
          {.mmproj = !vision_model_path.empty(),
           .speculative = speculative_config.backend !=
                              server::TextSpeculativeBackend::kDisabled ||
                          prompt_lookup,
           .cache_disk = !cache_disk_directory.empty(),
           .npu_prefill = llm_parser.WasSupplied("--npu-prefill") &&
                          requested_prefill != "off",
           .draft_model =
               !twin_model_path.empty() || !draft_head_model_path.empty()});
      if (unsupported != nullptr) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error: " << unsupported
                  << " is not supported for DiffusionGemma\n";
        return 2;
      }
      ModelLoadLog load_log("diffusion_gemma", model);
      const std::string name =
          server::ServedModelName(served_model_name, model, release.package);
      models::diffusion_gemma::EngineOptions engine_options;
      if (max_context != 0)
        engine_options.max_context = static_cast<int>(max_context);
      // The engine's own chunk (8192) stays unless --prefill-chunk is given;
      // the AR default of 512 would shrink its expert row groups.
      if (llm_parser.WasSupplied("--prefill-chunk"))
        engine_options.prefill_chunk = static_cast<int>(std::min<std::size_t>(
            prefill_chunk_tokens, std::numeric_limits<int>::max()));
      engine_options.sessions = static_cast<int>(session_count);
      std::shared_ptr<server::DiffusionGemmaBackend> diffusion;
      try {
        const auto image_token_cap =
            bubo::cli::PlanningImageTokenCap().value_or(0);
        diffusion = std::make_shared<server::DiffusionGemmaBackend>(
            std::filesystem::u8path(model), engine_options, name,
            std::filesystem::u8path(vision_model_path), no_vision);
        if (diffusion->supports_images()) {
          std::size_t free_bytes = 0, total_bytes = 0;
          if (gufo::platform::DeviceMemoryInfo(&free_bytes, &total_bytes) !=
              hipSuccess)
            throw std::runtime_error(
                "cannot check free device memory for DiffusionGemma vision");
          server::ValidateDiffusionGemmaVisionReserve(free_bytes);
        }
        diffusion->set_max_image_tokens(
            static_cast<std::uint32_t>(image_token_cap));
      } catch (const std::exception& error) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error loading DiffusionGemma '" << model
                  << "': " << error.what() << "\n";
        return 1;
      }
      diffusion->set_default_max_tokens(
          max_tokens < 0 ? 0 : static_cast<std::size_t>(max_tokens));
      const double warmup_ms = diffusion->Warmup();
      server::Logger::Info("loader", "event=warmup kind=diffusion_gemma ms=" +
                                         std::to_string(warmup_ms));
      load_log.Complete(
          "model=" + diffusion->model_id() +
          " sessions=" + std::to_string(session_count) +
          " context_tokens=" + std::to_string(diffusion->max_context()) +
          " sampler=block-diffusion vision_reserve_bytes=" +
          std::to_string(diffusion->supports_images()
                             ? server::kDiffusionGemmaVisionReserveBytes
                             : 0));
      backend = diffusion;
#endif
    } else {
      if ((mtp_policy != "length" && mtp_policy != "survival") ||
          (mtp_draft_vocab != "full" && mtp_draft_vocab != "latin" &&
           mtp_draft_vocab != "auto")) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        std::cerr << "Error: --mtp-policy is length or survival, "
                     "--mtp-draft-vocab is full, latin or auto\n";
        return 2;
      }
      speculative_config.mtp_survival = mtp_policy == "survival";
      speculative_config.mtp_latin_draft_vocabulary =
          mtp_draft_vocab == "latin";
      speculative_config.mtp_auto_draft_vocabulary = mtp_draft_vocab == "auto";
      speculative_config.prompt_lookup = prompt_lookup;
      std::string err;
#ifdef GUFO_NPU
      // Before GPU warm-up/upload. DIR is data consumed only after ON, never an
      // enabler. The same boundary is exercised by the CPU counter factory.
      try {
        core::CreateAuthorizedPrefill(
            release.resolved.prefill_authorization,
            [&](const core::ExperimentalAuthorization& authorization) {
              std::map<std::string, std::string> values;
              for (const auto& setting : release.resolved.settings)
                if (npu::IsKnob(setting.name) && setting.value) {
                  // GUFO_NPU_GUARD_TEST is an internal test setting: honoured
                  // only with debug hooks, ignored (and said so) otherwise.
                  if (setting.name == "GUFO_NPU_GUARD_TEST" && !debug_hooks) {
                    server::Logger::Warn(
                        "npu",
                        "GUFO_NPU_GUARD_TEST ignored: it needs debug hooks");
                    continue;
                  }
                  if (setting.name == "GUFO_NPU_TEST_FAIL_CHUNK")
                    if (const auto note =
                            npu::TestFailChunkIgnoredNote(debug_hooks)) {
                      server::Logger::Info("npu", *note);
                      continue;
                    }
                  values.emplace(setting.name, *setting.value);
                }
              auto knobs = npu::ParseKnobs(authorization, values);
              persistent_prefill = std::make_shared<npu::ServerPrefill>(
                  authorization, std::move(knobs));
              prefill_started = persistent_prefill->Start();
            });
      } catch (const std::exception& error) {
        npu::FailProcess(error.what());
        persistent_prefill.reset();
      } catch (...) {
        npu::FailProcess("unknown startup failure");
        persistent_prefill.reset();
      }
#else
      if (release.resolved.prefill_authorization.reason() ==
          "GUFO_ENABLE_NPU=OFF")
        server::Logger::Info(
            "npu", "GUFO_ENABLE_NPU=OFF; NPU disabled for process; GPU path");
#endif
#ifdef GUFO_NPU
      if (prefill_started)
        speculative_config.npu_prefill = persistent_prefill;
      release.resolved.runtime_npu_enabled = prefill_started;
      release.resolved.runtime_npu_reason =
          prefill_started ? "persistent session ready"
          : persistent_prefill
              ? "NPU startup failed."
              : release.resolved.prefill_authorization.reason();
      if (!prefill_started && !persistent_prefill)
        npu::NoteNpuStartup(false, release.resolved.runtime_npu_reason);
      setting_row(
          "--npu-prefill", prefill_started ? "on" : "off", "runtime",
          prefill_started
              ? json::Value()
              : json::Value(
                    persistent_prefill
                        ? "NPU startup failed."
                        : release.resolved.prefill_authorization.reason()));
#else
      release.resolved.runtime_npu_enabled = false;
      release.resolved.runtime_npu_reason =
          release.resolved.prefill_authorization.reason();
      setting_row("--npu-prefill", "off", "build",
                  release.resolved.prefill_authorization.reason());
#endif
      ModelLoadLog load_log("text", model);
      auto inference = std::make_shared<server::InferenceBackend>();
      backend = inference;
      if (!inference->load(
              model, &err, max_context, session_count,
              server::TextPrefillPolicy{
                  .decode_active_tokens = prefill_chunk_tokens,
              },
              server::TextSchedulerPolicy{
                  .max_pending_requests = max_pending_requests,
                  .max_pending_requests_per_client =
                      max_pending_requests_per_client,
                  .max_output_bytes_per_request = max_output_bytes,
                  .max_buffered_output_bytes_per_request =
                      max_buffered_output_bytes,
                  .max_buffered_output_bytes_total =
                      max_buffered_output_bytes_total,
                  .request_timeout =
                      std::chrono::milliseconds{
                          static_cast<std::chrono::milliseconds::rep>(
                              request_timeout_ms)},
                  .stall_timeout =
                      std::chrono::milliseconds{
                          static_cast<std::chrono::milliseconds::rep>(
                              stall_timeout_ms)},
                  .log_progress = log_progress,
              },
              speculative_config,
              server::TextDiskCacheConfig{
                  .directory = cache_disk_directory,
                  .capacity_bytes = cache_disk_bytes,
                  .staging_capacity_bytes = cache_disk_staging_bytes,
                  .model_artifact_fingerprint = release.package.MainIdentity(),
                  .draft_model_artifact_fingerprint =
                      speculative_config.draft_model_path.empty()
                          ? std::string{}
                          : release.package.IdentityFor(std::filesystem::u8path(
                                speculative_config.draft_model_path)),
              },
              vision_model_path,
              server::TextRunnerRamCacheOptions{
                  .capacity_bytes = cache_ram_bytes, .disabled = no_cache})) {
        server::Logger::FlushSync(std::chrono::seconds(2));
        if (const auto fit = server::LastSessionsFitFailure()) {
          // The sessions do not fit this machine: a typed refusal (exit 19,
          // memory_insufficient) with the numbers, not a prose failure.
          auto details = json::Value::object();
          details["needed_bytes"] = fit->needed_bytes;
          details["available_bytes"] = fit->available_bytes;
          details["sessions"] = fit->sessions;
          details["context"] = fit->context;
          details["per_session_bytes"] = fit->per_session_bytes;
          details["floor_bytes"] = fit->floor_bytes;
          throw bubo::cli::ControlError(19, err, std::move(details));
        }
        std::cerr << "Error loading model '" << model << "': " << err << "\n";
        return 1;
      }
      inference->set_model_id(
          server::ServedModelName(served_model_name, model, release.package));
      inference->set_sampling_defaults(
          max_tokens < 0 ? 0 : static_cast<std::size_t>(max_tokens),
          sampling_config, [&] {
            auto supplied = SamplingOptionsSupplied(llm_parser);
            for (const auto& setting : release.resolved.settings) {
              if (!setting.value || setting.origin == "built-in")
                continue;
              if (setting.name == "--temperature")
                supplied.temperature = true;
              else if (setting.name == "--top-k")
                supplied.top_k = true;
              else if (setting.name == "--top-p")
                supplied.top_p = true;
              else if (setting.name == "--min-p")
                supplied.min_p = true;
              else if (setting.name == "--min-keep")
                supplied.min_keep = true;
              else if (setting.name == "--seed")
                supplied.seed = true;
              else if (setting.name == "--repeat-penalty")
                supplied.repeat_penalty = true;
              else if (setting.name == "--repeat-last-n")
                supplied.repeat_last_n = true;
              else if (setting.name == "--frequency-penalty")
                supplied.frequency_penalty = true;
              else if (setting.name == "--presence-penalty")
                supplied.presence_penalty = true;
            }
            return supplied;
          }());
      inference->set_reasoning_defaults(*reasoning_defaults);
      const auto effective_sampling =
          inference->sampling_defaults().Resolve(reasoning_defaults->enabled);
      release.resolved.runtime_sampling = effective_sampling;
      release.resolved.runtime_context = inference->max_context();
      memory_estimate = bubo::cli::BuildReleaseMemoryEstimate(
          release.resolved,
          release.package.manifest_path.empty() ? ""
                                                : release.package.package_id,
          memory_budgets.gpu_available, memory_budgets.host_available, nullptr,
          memory_budgets);
      if (lifecycle) {
        setting_row("--temperature", effective_sampling.temperature, "runtime");
        setting_row("--top-k", effective_sampling.top_k, "runtime");
        setting_row("--top-p", effective_sampling.top_p, "runtime");
        setting_row("--min-p", effective_sampling.min_p, "runtime");
        setting_row("--seed", effective_sampling.seed, "runtime");
        try {
          setting_row("engine", inference->debug_settings(), "engine");
        } catch (const server::DebugSettingsUnsupported&) {
          // This model reports no engine settings: the row is left out.
        }
      }
      server::ChatRequest default_request;
      default_request.reasoning = *reasoning_defaults;
      const auto initial_output =
          inference->initial_output_state(default_request);
      std::ostringstream sampling_log;
      sampling_log << "event=defaults thinking="
                   << (initial_output == server::TextGenerationBackend::
                                             InitialOutputState::kReasoning
                           ? "on"
                       : initial_output == server::TextGenerationBackend::
                                               InitialOutputState::kContent
                           ? "off"
                           : "auto")
                   << " temperature=" << effective_sampling.temperature
                   << " top_k=" << effective_sampling.top_k
                   << " top_p=" << effective_sampling.top_p
                   << " min_p=" << effective_sampling.min_p
                   << " min_keep=" << effective_sampling.min_keep
                   << " seed=" << effective_sampling.seed
                   << " repeat_penalty=" << effective_sampling.repeat_penalty
                   << " repeat_last_n=" << effective_sampling.repeat_last_n
                   << " frequency_penalty="
                   << effective_sampling.frequency_penalty
                   << " presence_penalty="
                   << effective_sampling.presence_penalty;
      server::Logger::Info("sampling", sampling_log.str());
      if (debug_hooks) {
        try {
          auto resolved = json::Value::object();
          try {
            resolved = inference->debug_settings();
          } catch (const server::DebugSettingsUnsupported&) {
            // No engine settings for this model: the rest of the document
            // stands.
          }
          resolved["served_model_name"] = inference->model_id();
          resolved["origin"] = server::ReleaseOrigins(release.resolved);
          resolved["release_profile"] = release.resolved.profile;
#ifdef GUFO_NPU
          resolved["npu_startup_mode"] = prefill_started ? "on" : "off";
          resolved["npu_startup_reason"] =
              prefill_started ? "persistent session ready"
              : persistent_prefill
                  ? "startup failed; disabled for process; no retry"
                  : release.resolved.prefill_authorization.reason();
#else
          resolved["npu_startup_mode"] = "off";
          resolved["npu_startup_reason"] =
              release.resolved.prefill_authorization.reason();
#endif
          resolved["release_profile_settings"] =
              server::ReleaseSettingsJson(release.resolved);
          resolved["sessions"] = session_count;
          resolved["context"] =
              static_cast<unsigned long long>(inference->max_context());
          resolved["speculative_backend"] = speculative_backend;
          resolved["twin_bound"] = release.resolved.artefacts.twin;
          resolved["draft_head_bound"] = release.resolved.artefacts.draft_head;
          resolved["mtp_bound"] = release.resolved.artefacts.mtp;
          resolved["mmproj_bound"] = release.resolved.artefacts.mmproj;
          auto transport = json::Value::object();
          transport["host"] = host;
          transport["port"] = port;
          transport["max_connections"] = max_connections;
          transport["max_request_body_bytes"] = max_request_body_bytes;
          transport["authenticated"] = !api_key.empty();
          transport["debug_hooks"] = true;
          transport["max_score_tokens"] = max_score_tokens;
          resolved["transport"] = std::move(transport);
          auto sampling = json::Value::object();
          sampling["max_tokens"] = max_tokens;
          sampling["temperature"] = effective_sampling.temperature;
          sampling["top_k"] = effective_sampling.top_k;
          sampling["top_p"] = effective_sampling.top_p;
          sampling["min_p"] = effective_sampling.min_p;
          sampling["min_keep"] = effective_sampling.min_keep;
          sampling["seed"] = effective_sampling.seed;
          sampling["repeat_penalty"] = effective_sampling.repeat_penalty;
          sampling["repeat_last_n"] = effective_sampling.repeat_last_n;
          sampling["frequency_penalty"] = effective_sampling.frequency_penalty;
          sampling["presence_penalty"] = effective_sampling.presence_penalty;
          sampling["preset"] =
              static_cast<int>(inference->sampling_defaults().model);
          resolved["sampling_defaults"] = std::move(sampling);
          auto reasoning = json::Value::object();
          reasoning["think"] = reasoning_mode;
          reasoning["effort"] = reasoning_effort;
          reasoning["preserve_thinking"] = preserve_thinking;
          reasoning["initial_output"] = static_cast<int>(initial_output);
          reasoning["enabled"] = reasoning_defaults->enabled
                                     ? json::Value(*reasoning_defaults->enabled)
                                     : json::Value();
          resolved["reasoning_defaults"] = std::move(reasoning);
          resolved["trace_path"] =
              trace_path.empty() ? json::Value() : json::Value(trace_path);
          resolved["log_progress"] = log_progress;
          resolved["log_level"] =
              std::string(server::LogLevelName(server::Logger::Level()));
#if defined(ENGINE_ENABLE_HIP)
          int device = 0;
          hipDeviceProp_t properties{};
          int driver = 0;
          if (hipGetDevice(&device) != hipSuccess ||
              hipGetDeviceProperties(&properties, device) != hipSuccess ||
              hipDriverGetVersion(&driver) != hipSuccess)
            throw std::runtime_error(
                "cannot inspect the loaded GPU and driver");
          auto gpu = json::Value::object();
          gpu["device"] = device;
          gpu["name"] = properties.name;
          gpu["architecture"] = properties.gcnArchName;
          gpu["driver_version"] = driver;
          resolved["gpu"] = std::move(gpu);
#endif
          debug_settings = std::make_shared<const json::Value>(
              server::BuildEffectiveSettings(
                  std::move(resolved), GUFO_RELEASE_VERSION, GUFO_REVISION,
                  crypto::Sha256FileHex(platform::LoadedExecutablePath()),
                  server::ReleaseOrigins(release.resolved)));
          server::Logger::Diagnostic("settings",
                                     "settings " + debug_settings->dump());
        } catch (const std::exception& error) {
          server::Logger::FlushSync(std::chrono::seconds(2));
          std::cerr << "Error enabling debug hooks: " << error.what() << '\n';
          return 2;
        }
      }
      const char* speculation =
          speculative_config.backend == server::TextSpeculativeBackend::kDFlash
              ? "dflash2"
          : speculative_config.backend ==
                  server::TextSpeculativeBackend::kDSpark
              ? "dspark"
          : speculative_config.backend == server::TextSpeculativeBackend::kMtp
              ? "mtp"
              : "off";
      load_log.Complete(
          "model=" + inference->model_id() +
          " sessions=" + std::to_string(session_count) +
          " context_tokens=" + std::to_string(inference->max_context()) +
          " speculative=" + speculation + " draft_limit=" +
          (speculative_config.max_draft_tokens_given
               ? std::to_string(speculative_config.max_draft_tokens)
               : std::string("model")) +
          " disk_cache=" + (cache_disk_directory.empty() ? "off" : "enabled"));
    }
  }

  // run() joins every request thread, so no hook call outlives it.
  std::atomic<bool> device_lost{false};
  const bool authenticated = !api_key.empty();
  server::HttpServer server(
      host, port, backend, video_jobs, tts, asr,
      server::HttpServerOptions{
          .debug_hooks = debug_hooks,
          .max_score_tokens = max_score_tokens,
          .debug_settings = std::move(debug_settings),
          .max_request_body_bytes = max_request_body_bytes,
          .max_connections = max_connections,
          .api_key = std::move(api_key),
          .on_device_lost =
              [&device_lost, lifecycle] {
                device_lost.store(true);
                ShutdownAfterDeviceLoss(lifecycle);
              },
          .on_signal_stop =
              [lifecycle] {
                if (lifecycle)
                  lifecycle->Transition("stopping");
              },
          .on_request_completed =
              [lifecycle](json::Value row) {
                if (lifecycle)
                  lifecycle->Transition("request_completed", std::move(row));
              },
      },
      images);
  std::string err;
  if (reservation)
    reservation->Release();
  if (!server.start(&err)) {
    server::Logger::FlushSync(std::chrono::seconds(2));
    std::cerr << "Error starting HTTP server: " << err << "\n";
    if (lifecycle)
      throw bubo::cli::ControlError(13, "HTTP bind failed.");
    return 1;
  }
  const auto exit_if_device_lost = [&device_lost, &backend, lifecycle] {
    if (device_lost.load() || (backend != nullptr && backend->device_lost())) {
      ArmDeviceLossWatchdog();
      if (lifecycle) {
        lifecycle->Transition(
            "error",
            bubo::cli::Error("device_lost", "engine", "serve", false,
                             "Restart Bubo after the device is available."));
        lifecycle->RecordAbruptExit();
      }
      server::Logger::FlushSync(std::chrono::seconds(2));
      ::_exit(kDeviceLostExitStatus);
    }
  };
  struct ClearCallbacks {
    bubo::cli::ServeLifecycle* lifecycle;
    ~ClearCallbacks() {
      if (lifecycle) {
        lifecycle->SetStop({});
        lifecycle->SetSampler({});
      }
    }
  } clear_callbacks{lifecycle};
  if (lifecycle) {
    const auto served_model = backend  ? backend->model_id()
                              : tts    ? tts->model_id()
                              : asr    ? asr->model_id()
                              : images ? images->model_id()
                                       : subcommand;
    if (backend) {
      setting_row("--context",
                  static_cast<unsigned long long>(backend->max_context()));
    }
    const std::string endpoint = server.bound_endpoint();
    release.resolved.runtime_host = server.bound_host();
    release.resolved.runtime_port = server.port();
    auto contract_settings = server::ReleaseSettingsJson(release.resolved);
    for (const auto& [key, value] : contract_settings.members())
      lifecycle_settings[key] = value;
    for (const auto& row : contract_settings["effective_switches"].items())
      lifecycle_settings[row.member_str("name")] =
          server::ReleaseSettingSummary(row);
    lifecycle_settings["think_marker_entropy_gate"] =
        contract_settings["entropy_gate"];
    lifecycle->Configure(endpoint, served_model, lifecycle_settings, log_path,
                         true);
    auto details = json::Value::object();
    details["memory_estimate"] =
        release_active ? memory_estimate.value : json::Value();
    details["memory_estimate_reason"] =
        memory_estimate.reason.empty() ? json::Value()
                                       : json::Value(memory_estimate.reason);
    details["inference_defaults"] =
        backend ? contract_settings["inference_defaults"] : json::Value();
    details["effective_switches"] = contract_settings["effective_switches"];
    details["think_marker_bias"] = contract_settings["think_marker_bias"];
    details["loop_breaker"] = contract_settings["loop_breaker"];
    details["think_marker_entropy_gate"] = contract_settings["entropy_gate"];
    details["pi_snippet"] =
        backend ? bubo::cli::PiSnippet(endpoint, served_model,
                                       backend->max_context(), authenticated)
                : json::Value();
    details["active_model"] = active_model;
    details["client_snippets"] =
        backend ? bubo::cli::ClientSnippets(served_model, authenticated)
                : json::Value();
    auto connection = json::Value::object();
    connection["endpoint"] = endpoint;
    connection["model_id"] = served_model;
    connection["context_tokens"] =
        backend ? json::Value(
                      static_cast<unsigned long long>(backend->max_context()))
                : json::Value();
    connection["authenticated"] = authenticated;
    details["connection"] = std::move(connection);
    // The row of the loaded package: a Flash Next set or a plain catalogue
    // model, with the main model file being served.
    const auto manifest_text = release.package.manifest_path.u8string();
    const auto model_text = release.package.files.empty()
                                ? std::u8string()
                                : release.package.files.front().path.u8string();
    details["prepared_models"] = bubo::cli::PreparedModels(
        std::string(manifest_text.begin(), manifest_text.end()),
        !release.package.files.empty(), release.package.package_id,
        std::string(model_text.begin(), model_text.end()));
    details["requests"] = backend ? server.recent_requests() : json::Value();
    lifecycle->SetDetails(std::move(details));
    lifecycle->SetStop([&server] { server.request_stop(); });
    if (backend)
      lifecycle->SetSampler([backend, &server] {
        auto metrics = server::RuntimeStatusMetrics(*backend);
        metrics["requests"] = server.recent_requests();
        return metrics;
      });
    if (!lifecycle->StopRequested()) {
      if (!server.ready()) {
        exit_if_device_lost();
        throw bubo::cli::ControlError(14, "Model service is not ready.");
      }
      lifecycle->Transition("ready");
    }
  }
  // Releasing model state on a lost device crashes; the kernel reclaims it.
  // The backend is checked as well: an external SIGTERM can stop the server
  // after the scheduler records the loss but before a request fires the hook.
  try {
    server.run(/*handle_signals=*/true);
  } catch (...) {
    exit_if_device_lost();
    throw;
  }
  exit_if_device_lost();
  return 0;
}

int RunServe(std::span<const char* const> args) {
  return RunServeCommand(args, RunServeImpl);
}
}  // namespace gufo::cli
