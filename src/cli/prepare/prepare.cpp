#include "src/cli/prepare/prepare.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>

#include "src/cli/arg_parser.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/cli/prepare/vc_runtime.hpp"
#include "src/cli/prepared_default.hpp"

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
  result.options.local_app_data = local_app_data;
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

  gufo::cli::ArgParser parser("bubo prepare");
  parser.SetImplicitHelp(false).SetInlineValues(false);
  const auto missing = [](std::string_view flag) {
    if (flag == "--output")
      return std::string(
          "--output requires one nonempty ROOT and cannot repeat");
    if (flag == "--catalogue-id")
      return std::string(
          "--catalogue-id requires one catalogue id and cannot repeat");
    if (flag == "--model-set")
      return std::string(
          "--model-set requires huihui, original or lowmemory and cannot "
          "repeat");
    return std::string(flag) + " requires one nonempty PATH";
  };
  parser.SetErrorHandler([&](auto kind, std::string_view arg) {
    return kind == gufo::cli::ArgParser::ParseError::MissingValue
               ? missing(arg)
               : "unknown argument '" + std::string(arg) + "'";
  });
  const auto machine = [&](auto, auto, std::string* error) {
    if (result.options.machine) {
      *error = "select one machine-output flag";
      return false;
    }
    result.options.machine = true;
    return true;
  };
  parser.AddCustomFlag("", "--json", "Machine output", "Output", machine);
  parser.AddCustomFlag("", "--jsonl", "Machine stream", "Output", machine);
  parser.AddFlag("", "--yes", "Accept plan", "Prepare", &result.options.yes);
  parser.AddFlag("", "--verify", "Verify files", "Prepare",
                 &result.options.verify);
  parser.AddFlag("", "--offline", "Use local files", "Prepare",
                 &result.options.offline);
  parser.AddFlag("", "--resume", "Resume downloads", "Prepare",
                 &result.options.resume);
  parser.AddInverseFlag("", "--no-resume", "Restart downloads", "Prepare",
                        &result.options.resume);
  parser.AddFlag("", "--plan", "Show plan", "Prepare",
                 &result.options.plan_only);
  parser.AddFlag("", "--help", "Print help", "General", &result.options.help);
  parser.AddCustomOption(
      "", "--catalogue-id", "ID", "Catalogue entry", "Model",
      [&](auto flag, auto value, std::string* error) {
        if (!ValidPath(value) || !result.options.catalogue_id.empty()) {
          *error = missing(flag);
          return false;
        }
        result.options.catalogue_id = value;
        return true;
      });
  parser.AddCustomOption("", "--model-set", "SET", "Model set", "Model",
                         [&](auto flag, auto value, std::string* error) {
                           const auto set = ModelSetFromName(value);
                           if (!set || result.options.model_set_chosen) {
                             *error = missing(flag);
                             return false;
                           }
                           result.options.model_set = *set;
                           result.options.model_set_chosen = true;
                           return true;
                         });
  const auto path = [&](std::string_view flag, std::string* target) {
    parser.AddCustomOption(
        "", flag, "PATH", "Path", "Prepare",
        [&, target](auto name, auto value, std::string* error) {
          if (!ValidPath(value)) {
            *error = missing(name);
            return false;
          }
          if (!target->empty()) {
            *error = std::string(name) + " cannot repeat";
            return false;
          }
          *target = value;
          return true;
        });
  };
  path("--mtp-model", &result.options.mtp_model);
  path("--mmproj", &result.options.mmproj);
  path("--cache-dir", &result.options.cache_dir);
  path("--runtime-dir", &result.options.runtime_dir);
  parser.AddCustomOption("", "--output", "ROOT", "Parent folder", "Prepare",
                         [&](auto flag, auto value, std::string* error) {
                           if (explicit_output || !ValidPath(value)) {
                             *error = missing(flag);
                             return false;
                           }
                           result.options.root = value;
                           explicit_output = true;
                           return true;
                         });
  if (!parser.Parse(args.subspan(first), &result.error))
    return result;

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
  if (result.options.step == Step::Runtime && result.options.model_set_chosen)
    result.error = "runtime-only rejects --model-set";
  if (result.options.step == Step::Runtime &&
      !result.options.catalogue_id.empty())
    result.error = "runtime-only rejects --catalogue-id";
  if (result.options.step != Step::Model && result.options.plan_only)
    result.error = "--plan requires the model step: bubo prepare model --plan";
  if (!result.options.catalogue_id.empty() && result.error.empty()) {
    if (result.options.model_set_chosen)
      result.error = "--catalogue-id cannot be used with --model-set";
    else {
      const auto& shipped = bubo::cli::ShippedCatalogue();
      const auto* entry = bubo::cli::FindCatalogueEntry(
          shipped.entries, result.options.catalogue_id);
      if (!shipped.error.empty())
        result.error = "the shipped model catalogue is invalid";
      else if (!entry) {
        // bubo models list needs the engine; name the ids here instead.
        result.error = "unknown catalogue id '" + result.options.catalogue_id +
                       "'. Catalogue ids:";
        for (const auto& known : shipped.entries)
          result.error +=
              (&known == &shipped.entries.front() ? " " : ", ") + known.id;
      } else if ((!result.options.mtp_model.empty() ||
                  !result.options.mmproj.empty()) &&
                 !entry->flash_next_set)
        result.error = "--mtp-model and --mmproj apply to Flash Next only";
      if (result.error.empty() && entry && entry->flash_next_set) {
        result.options.model_set = *entry->flash_next_set;
        result.options.model_set_chosen = true;
      }
    }
  }
  if (result.options.step == Step::Model && !result.options.runtime_dir.empty())
    result.error = "model-only rejects --runtime-dir";
  return result;
}

std::string ModelRoot(const Options& options) {
  return Join(Join(options.root, "models"),
              options.catalogue_id.empty()
                  ? PackageId(options.model_set)
                  : std::string_view(options.catalogue_id));
}

std::string CatalogueServeCommand(const Options& options) {
  // serve --manifest accepts the folder of a GGUF package. A DiffusionGemma
  // safetensors folder is served by its path.
  const auto* entry = bubo::cli::FindCatalogueEntry(
      bubo::cli::ShippedCatalogue().entries, options.catalogue_id);
  const bool folder = entry && entry->packaging == "safetensors_folder";
  return std::string("bubo serve llm ") + (folder ? "--model" : "--manifest") +
         " \"" + ModelRoot(options) + "\"";
}

void PrintHelp(std::ostream& out) {
  out << "Usage: bubo prepare [all|runtime|model] [OPTIONS]\n\n"
         "Prepare pinned runtime and model files. The default step is all.\n"
         "The model step downloads and verifies the files of one model:\n"
         "the default Flash Next package, or --catalogue-id ID.\n"
         "The runtime step installs AMD ROCm 10.0.0 for gfx1151 in a "
         "shared per-user folder.\n\n"
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
         "  --model-set SET   Model files: huihui (default), original or "
         "lowmemory\n"
         "  --catalogue-id ID Model from the catalogue (bubo models list)\n"
         "  --plan            Model step: show files, sizes, free space and\n"
         "                    licences, then stop (no download, no write)\n"
         "  --cache-dir PATH  Runtime archive cache\n"
         "  --runtime-dir PATH  Runtime destination (default: "
         "%LOCALAPPDATA%\\Bubo\\runtime\\rocm-10.0.0-gfx1151)\n"
         "  --help         Print help without writes, network or engine "
         "loading\n\n"
         "Model layout: "
         "ROOT\\models\\huihui-qwen38-flash-next-ud-q4-k-xl-v1\\\n"
         "  Four main shards; mtp\\, mmproj\\, twin\\, draft_head\\ "
         "files.\n"
         "  --model-set original uses "
         "ROOT\\models\\qwen38-flash-next-base-ud-q4-k-xl-v1\\\n\n"
         "Exit codes: 0 success/help; 2 syntax/schema/header/profile; 3 "
         "consent/cancel;\n"
         "  4 offline/missing pin; 5 transport/TLS; 6 integrity; 7 "
         "storage/access;\n"
         "  8 prepare lock; 9 archive/internal.\n";
}

bool ConfigureFirstRunModel(Options& options,
                            const bubo::cli::PlanningMemoryBudgets& budgets) {
  if (!bubo::cli::Is96GiBStrixHalo(budgets) || options.step == Step::Runtime)
    return false;
  // The GUI names its preselected package explicitly. Remember that same
  // lower-memory choice on a fresh 96 GB installation without replacing an
  // explicit choice of another model.
  const bool selected_lower =
      options.model_set == ModelSet::LowMemory &&
      (options.catalogue_id.empty() ||
       options.catalogue_id == PackageId(ModelSet::LowMemory));
  if ((options.model_set_chosen || !options.catalogue_id.empty()) &&
      !selected_lower)
    return false;
  const auto environment = bubo::cli::EnvironmentUtf8("BUBO_MANIFEST");
  const auto selection = bubo::cli::ResolvePreparedDefault(
      {}, {}, environment, options.local_app_data);
  if (selection.source != "standard_location" ||
      std::filesystem::exists(selection.manifest))
    return false;
  options.model_set = ModelSet::LowMemory;
  const auto local = options.local_app_data;
  options.publish_default = [local](const std::string& manifest) {
    const auto current = bubo::cli::ResolvePreparedDefault(
        {}, {}, bubo::cli::EnvironmentUtf8("BUBO_MANIFEST"), local);
    const auto path =
        std::filesystem::absolute(std::filesystem::u8path(manifest))
            .lexically_normal();
    // A selection made while preparation was in progress takes priority.
    if (current.source != "standard_location" && current.manifest != path)
      return 0;
    std::string error;
    if (bubo::cli::RecordPreparedDefault(local, path, error))
      return 0;
    std::cerr << "prepare: E7 " << error << '\n';
    return 7;
  };
  return true;
}

int CompleteModelPreparation(
    const Options& options, bool automatic_model, std::ostream& out,
    const std::function<int(const std::string&)>& remember) {
  if (!automatic_model) {
    if (options.model_set != ModelSet::Huihui)
      out << "prepare: the saved default model is unchanged; start the "
          << ModelSetName(options.model_set) << " set with --manifest \""
          << ModelRoot(options) << "\\bubo-package.json\"\n";
    else if (!options.catalogue_id.empty() &&
             options.catalogue_id != kPackageId)
      out << "prepare: the saved default model is unchanged; start this model "
             "with: "
          << CatalogueServeCommand(options) << "\n";
  }
  return remember(ModelRoot(options) + "/bubo-package.json");
}

int Run(std::span<const char* const> args, std::string_view local_app_data,
        std::ostream& out, std::ostream& err,
        const VcRuntimeProbe* prerequisite_probe) {
  std::unique_ptr<bubo::cli::MachineWriter> writer;
  if (bubo::cli::MachineRequested(args))
    writer = std::make_unique<bubo::cli::MachineWriter>(out);
  auto parsed = Parse(args, local_app_data);
  if (!parsed) {
    err << "prepare: E2 " << parsed.error << "; run bubo prepare --help\n";
    if (writer) {
      writer->Write("error",
                    bubo::cli::PrepareError(2, "arguments", parsed.error));
      return 2;
    }
    return static_cast<int>(ExitCode::Syntax);
  }
  if (parsed.options.help) {
    if (writer)
      writer->Write("capabilities", bubo::cli::Capabilities());
    else
      PrintHelp(out);
    return 0;
  }
  if (const auto code = CheckVcRuntime(
          prerequisite_probe ? *prerequisite_probe : NativeVcRuntimeProbe(),
          err)) {
    if (writer)
      writer->Write(
          "error",
          bubo::cli::Error("runtime_missing", "prepare", "prerequisite", false,
                           "Install the Microsoft Visual C++ runtime."));
    return code;
  }
  CancellationScope cancellation;
  const bool automatic_model = ConfigureFirstRunModel(
      parsed.options, bubo::cli::ReadPlanningMemoryBudgets());
  const auto automatic_publish = parsed.options.publish_default;
  const auto remember = [&](const std::string& manifest) {
    if (automatic_model)
      return automatic_publish(manifest);
    // Outside the fresh 96 GB path the saved default stays Huihui. Other sets
    // are started with an explicit --manifest PATH (the result record names
    // that path).
    if (parsed.options.model_set != ModelSet::Huihui ||
        (!parsed.options.catalogue_id.empty() &&
         parsed.options.catalogue_id != kPackageId))
      return 0;
    std::string error;
    if (bubo::cli::RecordPreparedDefault(
            local_app_data, std::filesystem::u8path(manifest), error))
      return 0;
    err << "prepare: E7 " << error << '\n';
    return 7;
  };
  parsed.options.publish_default = remember;
  const auto finish = [&](int code) {
    if (!code && parsed.options.step != Step::Runtime)
      return CompleteModelPreparation(parsed.options, automatic_model, out, remember);
    return code;
  };
  if (writer) {
    const auto cancelled = [] {
      return gCancelled.load(std::memory_order_relaxed);
    };
    const auto observe = [&](const Options& options) {
      if (options.cancelled && options.cancelled())
        gCancelled.store(true, std::memory_order_relaxed);
      return cancelled();
    };
    const PreparePhase model = [&](const Options& options, std::istream&,
                                   std::ostream& output, std::ostream& errors,
                                   PreparationPlan* preview, bool approved) {
      return RunModel(
          options, output, errors, [&] { return observe(options); }, preview,
          approved);
    };
    const PreparePhase runtime = [&](const Options& options, std::istream&,
                                     std::ostream& output, std::ostream& errors,
                                     PreparationPlan* preview, bool approved) {
      return RunRuntime(
          options, output, errors, [&] { return observe(options); }, preview,
          approved);
    };
    return RunMachine(parsed.options, model, runtime, std::cin, *writer, err,
                      cancelled);
  }
  if (parsed.options.plan_only) {
    PreparationPlan plan;
    const auto code =
        RunModel(parsed.options, out, err, [] { return false; }, &plan, false);
    if (code)
      return code;
    out << plan.text;
    for (const auto& volume : plan.volumes)
      if (volume.free_bytes < volume.required_bytes) {
        err << "prepare: E7 " << NotEnoughSpaceText(volume) << '\n';
        return 7;
      }
    return 0;
  }
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
    return finish(
        RunAll(parsed.options, model, runtime, std::cin, out, err, cancelled));
  }
  if (parsed.options.step == Step::Runtime)
    return RunRuntime(parsed.options, out, err, [] {
      return gCancelled.load(std::memory_order_relaxed);
    });
  return finish(RunModel(parsed.options, out, err, [] {
    return gCancelled.load(std::memory_order_relaxed);
  }));
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
      err << "prepare: E7 " << NotEnoughSpaceText(volume) << '\n';
      return 7;
    }
  }
  // One answer is read below, so print one question: drop the last line of
  // each component plan when it is that component's own [y/N] question.
  const auto without_question = [](std::string text) {
    if (text.ends_with("[y/N]\n")) {
      const auto line = text.rfind('\n', text.size() - 2);
      text.erase(line == std::string::npos ? 0 : line + 1);
    }
    return text;
  };
  out << "Combined preparation plan. One approval covers both disclosures "
         "below.\n"
      << without_question(runtime_plan.text)
      << without_question(model_plan.text)
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
