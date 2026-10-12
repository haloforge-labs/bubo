#include "src/cli/prepare/prepare.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>

#ifdef _WIN32
#include <windows.h>
#endif

#include "src/cli/prepare/model_prepare.hpp"
#include "src/cli/prepare/runtime_prepare.hpp"
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
         "The runtime step installs AMD ROCm 10.0.0 for gfx1151 beside "
         "bubo.exe.\n\n"
         "Options:\n"
         "  --output ROOT  Parent folder (default: %LOCALAPPDATA%\\Bubo)\n"
         "  --yes          Accept the displayed download plan\n"
         "  --verify       Force full SHA-256 verification of selected files\n"
         "  --offline      Use local files only; missing files fail before "
         "network\n"
         "  --resume       Resume tool-owned partial downloads (default)\n"
         "  --no-resume    Restart partial downloads\n"
         "  --mtp-model PATH  Copy a hash-matching existing MTP file\n"
         "  --mmproj PATH     Copy a hash-matching existing vision projector\n"
         "  --cache-dir PATH  Runtime archive cache\n"
         "  --runtime-dir PATH  Runtime destination (default: beside "
         "bubo.exe)\n"
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
         "  8 prepare lock; 9 archive/internal.\n";
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
  CancellationScope cancellation;
  if (parsed.options.step == Step::All) {
    const auto cancelled = [] {
      return gCancelled.load(std::memory_order_relaxed);
    };
    const PreparePhase model = [&](const Options& options, std::istream&,
                                   std::ostream& output, std::ostream& errors,
                                   PreparationPlan* preview, bool approved) {
      return RunModel(options, output, errors, cancelled, preview, approved);
    };
    const PreparePhase runtime = [&](const Options& options, std::istream&,
                                     std::ostream& output, std::ostream& errors,
                                     PreparationPlan* preview, bool approved) {
      return RunRuntime(options, output, errors, cancelled, preview, approved);
    };
    return RunAll(parsed.options, model, runtime, std::cin, out, err,
                  cancelled);
  }
  if (parsed.options.step == Step::Runtime)
    return RunRuntime(parsed.options, out, err, [] {
      return gCancelled.load(std::memory_order_relaxed);
    });
  return RunModel(parsed.options, out, err,
                  [] { return gCancelled.load(std::memory_order_relaxed); });
}

int RunAll(const Options& options, const PreparePhase& model,
           const PreparePhase& runtime, std::istream& input, std::ostream& out,
           std::ostream& err, const std::function<bool()>& cancelled) {
  PreparationPlan model_plan, runtime_plan;
  // Gate the model's immutable source/legal inputs before even inspecting the
  // runtime. An OPEN model gate must never partially install the runtime.
  if (const auto code = model(options, input, out, err, &model_plan, false))
    return code;
  if (const auto code = runtime(options, input, out, err, &runtime_plan, false))
    return code;
  if (model_plan.ready && runtime_plan.ready) {
    out << "prepare: already prepared; nothing to do\n";
    return 0;
  }
  std::map<std::string, ConsentVolume> volumes;
  constexpr std::uint64_t margin = 2ULL * 1024 * 1024 * 1024;
  for (const auto* plan : {&model_plan, &runtime_plan}) {
    for (const auto& volume : plan->volumes) {
      auto [row, inserted] = volumes.try_emplace(volume.name, volume);
      if (!inserted) {
        row->second.free_bytes =
            std::min(row->second.free_bytes, volume.free_bytes);
        const auto extra =
            volume.required_bytes - std::min(margin, volume.required_bytes);
        if (extra > std::numeric_limits<std::uint64_t>::max() -
                        row->second.required_bytes) {
          err << "prepare: E7 combined space calculation overflow\n";
          return 7;
        }
        row->second.required_bytes += extra;
      }
    }
  }
  for (const auto& [name, volume] : volumes) {
    out << "Combined volume " << name << ": free " << volume.free_bytes
        << " bytes; required " << volume.required_bytes << " bytes\n";
    if (volume.free_bytes < volume.required_bytes) {
      err << "prepare: E7 insufficient combined free space on " << name << '\n';
      return 7;
    }
  }
  out << "Combined preparation plan. One approval covers both disclosures "
         "below.\n"
      << runtime_plan.text << model_plan.text
      << "Approve the complete runtime and model plan? [y/N]\n";
  if (options.yes)
    out << "approval: --yes (displayed plan accepted)\n";
  else {
    std::string answer;
    if (!std::getline(input, answer) || (answer != "y" && answer != "Y")) {
      err << "prepare: E3 plan declined\n";
      return 3;
    }
  }
  if (cancelled && cancelled()) {
    err << "prepare: E3 cancelled\n";
    return 3;
  }
  if (!runtime_plan.ready) {
    if (const auto code = runtime(options, input, out, err, nullptr, true))
      return code;
  }
  if (!model_plan.ready) {
    if (const auto code = model(options, input, out, err, nullptr, true))
      return code;
  }
  // A phase that was ready in the preview may change while the other phase
  // downloads. Revalidate a skipped model phase read-only; newly acquired
  // model files were protected through the final manifest publication.
  PreparationPlan final_model, final_runtime;
  if (model_plan.ready) {
    if (const auto code = model(options, input, out, err, &final_model, false))
      return code;
  } else
    final_model.ready = true;
  if (const auto code =
          runtime(options, input, out, err, &final_runtime, false))
    return code;
  if (!final_model.ready || !final_runtime.ready) {
    err << "prepare: E7 prepared files changed during the combined "
           "preparation; run bubo prepare again\n";
    return 7;
  }
  return 0;
}
}  // namespace bubo::prepare
