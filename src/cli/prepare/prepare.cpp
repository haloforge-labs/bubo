#include "src/cli/prepare/prepare.hpp"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <ostream>

#ifdef _WIN32
#include <windows.h>
#endif

#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
namespace {
std::atomic<bool> gCancelled{false};
#ifdef _WIN32
BOOL WINAPI PrepareControl(DWORD type) {
  if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT)
    return FALSE;
  gCancelled.store(true, std::memory_order_relaxed);
  return TRUE;
}
#else
void PrepareControl(int) {
  gCancelled.store(true, std::memory_order_relaxed);
}
#endif
class CancellationScope {
public:
  CancellationScope() {
    gCancelled.store(false, std::memory_order_relaxed);
#ifdef _WIN32
    SetConsoleCtrlHandler(PrepareControl, TRUE);
#else
    previous_ = std::signal(SIGINT, PrepareControl);
#endif
  }
  ~CancellationScope() {
#ifdef _WIN32
    SetConsoleCtrlHandler(PrepareControl, FALSE);
#else
    std::signal(SIGINT, previous_);
#endif
  }

private:
#ifndef _WIN32
  void (*previous_)(int) = SIG_DFL;
#endif
};
std::string Join(std::string_view root, std::string_view part) {
  const auto path =
      std::filesystem::u8path(root) / std::filesystem::u8path(part);
  const auto utf8 = path.u8string();
  return std::string(utf8.begin(), utf8.end());
}
bool ValidPath(std::string_view path) {
  return !path.empty() && path.front() != '-' &&
         path.find_first_of("\r\n\t") == std::string_view::npos &&
         path.find('\0') == std::string_view::npos;
}
}  // namespace

ParseResult Parse(std::span<const char* const> args,
                  std::string_view local_app_data) {
  ParseResult result;
  bool explicit_output = false;
  std::size_t first = 0;
  if (!args.empty() && std::string_view(args[0]).empty()) {
    result.error = "empty argument";
    return result;
  }
  if (!args.empty() && std::string_view(args[0]).front() != '-') {
    const std::string_view step = args[0];
    if (step == "runtime")
      result.options.step = Step::Runtime;
    else if (step == "model")
      result.options.step = Step::Model;
    else if (step != "all") {
      result.error = "unknown step '" + std::string(step) + "'";
      return result;
    }
    first = 1;
  }
  for (std::size_t i = first; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--yes")
      result.options.yes = true;
    else if (arg == "--verify")
      result.options.verify = true;
    else if (arg == "--offline")
      result.options.offline = true;
    else if (arg == "--resume")
      result.options.resume = true;
    else if (arg == "--no-resume")
      result.options.resume = false;
    else if (arg == "--mtp-model" || arg == "--mmproj" ||
             arg == "--cache-dir" || arg == "--runtime-dir") {
      if (i + 1 == args.size() || !ValidPath(args[i + 1])) {
        result.error = std::string(arg) + " requires one nonempty PATH";
        return result;
      }
      auto* target = arg == "--mtp-model"   ? &result.options.mtp_model
                     : arg == "--mmproj"    ? &result.options.mmproj
                     : arg == "--cache-dir" ? &result.options.cache_dir
                                            : &result.options.runtime_dir;
      if (!target->empty()) {
        result.error = std::string(arg) + " cannot repeat";
        return result;
      }
      *target = args[++i];
    } else if (arg == "--help")
      result.options.help = true;
    else if (arg == "--output") {
      if (explicit_output || i + 1 == args.size() || !ValidPath(args[i + 1])) {
        result.error = "--output requires one nonempty ROOT and cannot repeat";
        return result;
      }
      result.options.root = args[++i];
      explicit_output = true;
    } else {
      result.error = "unknown argument '" + std::string(arg) + "'";
      return result;
    }
  }
  if (!explicit_output && !result.options.help) {
    if (!ValidPath(local_app_data)) {
      result.error = "LOCALAPPDATA is unavailable; use --output ROOT";
      return result;
    }
    result.options.root = Join(local_app_data, "Bubo");
  }
  if (result.options.step == Step::Runtime &&
      (!result.options.mtp_model.empty() || !result.options.mmproj.empty()))
    result.error = "runtime-only rejects --mtp-model and --mmproj";
  if (result.options.step == Step::Model && !result.options.runtime_dir.empty())
    result.error = "model-only rejects --runtime-dir";
  return result;
}

std::string ModelRoot(const Options& options) {
  return Join(Join(options.root, "models"), kPackageId);
}

void PrintHelp(std::ostream& out) {
  out << "Usage: bubo prepare [all|runtime|model] [OPTIONS]\n\n"
         "Prepare pinned runtime and model files. The default step is all.\n"
         "The model step downloads and verifies the eight pinned model files.\n"
         "The runtime step is pending J3 and currently exits 10.\n\n"
         "Options:\n"
         "  --output ROOT  Parent folder (default: %LOCALAPPDATA%\\Bubo)\n"
         "  --yes          Accept the displayed download plan\n"
         "  --verify       Force full SHA-256 verification of model files\n"
         "  --offline      Use local files only; missing files fail before "
         "network\n"
         "  --resume       Resume tool-owned partial downloads (default)\n"
         "  --no-resume    Restart partial downloads\n"
         "  --mtp-model PATH  Copy a hash-matching existing MTP file\n"
         "  --mmproj PATH     Copy a hash-matching existing vision projector\n"
         "  --cache-dir PATH  Runtime archive cache (runtime step pending J3)\n"
         "  --runtime-dir PATH  Runtime destination (runtime step pending J3)\n"
         "  --help         Print help without writes, network or engine "
         "loading\n\n"
         "Model layout: "
         "ROOT\\models\\huihui-qwen38-flash-next-ud-q4-k-xl-v1\\\n"
         "  Four main shards; mtp\\, mmproj\\, twin\\, draft_head\\ "
         "sidecars.\n\n"
         "Exit codes: 0 success/help; 2 syntax/schema/header/profile; 3 "
         "consent/cancel;\n"
         "  4 offline/missing pin; 5 transport/TLS; 6 integrity; 7 "
         "storage/access;\n"
         "  8 prepare lock; 9 archive/internal; 10 runtime step not "
         "implemented.\n";
}

int Run(std::span<const char* const> args, std::string_view local_app_data,
        std::ostream& out, std::ostream& err) {
  const auto parsed = Parse(args, local_app_data);
  if (!parsed) {
    err << "prepare: E2 " << parsed.error << "; run bubo prepare --help\n";
    return static_cast<int>(ExitCode::Syntax);
  }
  if (parsed.options.help) {
    PrintHelp(out);
    return 0;
  }
  if (parsed.options.step != Step::Model) {
    out << "prepare runtime: not implemented in J2 (pending J3)\n";
    return static_cast<int>(ExitCode::NotImplemented);
  }
  CancellationScope cancellation;
  return RunModel(parsed.options, out, err,
                  [] { return gCancelled.load(std::memory_order_relaxed); });
}
}  // namespace bubo::prepare
