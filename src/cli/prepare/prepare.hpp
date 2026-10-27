#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::cli {
class MachineWriter;
struct PlanningMemoryBudgets;
}  // namespace bubo::cli

namespace bubo::prepare {
struct VcRuntimeProbe;
enum class Step { All, Runtime, Model };
enum class ExitCode {
  Success = 0,
  Syntax = 2,
  Cancel = 3,
  MissingSource = 4,
  Transport = 5,
  Integrity = 6,
  Storage = 7,
  Locked = 8,
  Internal = 9,
  NotImplemented = 10
};
struct Options {
  Step step = Step::All;
  // The process data location used by the shared runtime resolver. --output
  // controls root (model/archive storage) independently.
  std::string local_app_data;
  std::string root;
  std::string mtp_model;
  std::string mmproj;
  std::string cache_dir;
  std::string runtime_dir;
  bool resume = true;
  bool yes = false;
  bool verify = false;
  bool offline = false;
  bool help = false;
  bool machine = false;
  // --model-set. The default set is Huihui; the model_set_chosen flag records
  // an explicit option (runtime-only rejects it).
  ModelSet model_set = ModelSet::Huihui;
  bool model_set_chosen = false;
  // --catalogue-id: a model catalogue entry (bubo models list). Flash Next ids
  // select their model set; every other id is prepared by the catalogue path.
  std::string catalogue_id;
  // --plan: show the plan (files, sizes, free space, licences) and stop. No
  // download, no write.
  bool plan_only = false;
  // Typed observers have no source, hash or destination authority.
  std::function<void(gufo::json::Value)> progress;
  std::function<bool()> cancelled;
  std::function<int(const std::string&)> publish_default;
  std::function<bool(const gufo::json::Value&)> authorize_plan;
};
struct ParseResult {
  Options options;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};
// Fresh 96 GB Strix Halo installations prepare and remember the lower-memory
// package. Explicit choices and authoritative saved selections take priority.
bool ConfigureFirstRunModel(Options&, const bubo::cli::PlanningMemoryBudgets&);
int CompleteModelPreparation(
    const Options&, bool automatic_model, std::ostream&,
    const std::function<int(const std::string&)>& remember);
struct ConsentVolume {
  std::string name;
  std::uint64_t free_bytes = 0, required_bytes = 0;
};
// required_bytes includes the preparation safety margin, not just writes.
gufo::json::Value PlanVolume(const ConsentVolume&);
// The free-space refusal in plain words, with the drive letter and GiB, e.g.
// "Not enough free space on C:. Needs 110.6 GiB; 110.4 GiB free. ...".
std::string NotEnoughSpaceText(const ConsentVolume&,
                               std::string_view option = "--output");
// Read-only preflight output. Internal orchestration displays both steps before
// one approval; this does not provide a CLI or catalogue override.
struct PreparationPlan {
  bool ready = false;
  std::string text;
  std::vector<ConsentVolume> volumes;
  gufo::json::Value machine = gufo::json::Value::object();
  // Runtime phase only: the folder the run resolved, for the result record.
  std::string runtime_dir;
};
using PreparePhase =
    std::function<int(const Options&, std::istream&, std::ostream&,
                      std::ostream&, PreparationPlan*, bool)>;
int RunAll(const Options&, const PreparePhase& model,
           const PreparePhase& runtime, std::istream&, std::ostream&,
           std::ostream&, const std::function<bool()>& cancelled = {});
ParseResult Parse(std::span<const char* const> args,
                  std::string_view local_app_data);
int RunMachine(const Options&, const PreparePhase& model,
               const PreparePhase& runtime, std::istream&,
               bubo::cli::MachineWriter&, std::ostream&,
               const std::function<bool()>& cancelled = {},
               const std::function<int(std::string&)>& read_line = {});
std::string PlanDigest(const gufo::json::Value&);
gufo::json::Value PlanSource(std::string_view id, std::string_view url,
                             std::uint64_t size, std::string_view sha256,
                             std::string_view repository,
                             std::string_view revision, std::string_view path,
                             std::string_view license);
void ReportProgress(const Options&, std::string_view component,
                    std::string_view phase, std::string_view file,
                    std::uint64_t done, std::uint64_t total,
                    std::uint64_t resume = 0,
                    std::string_view method = "sha256");
std::string ModelRoot(const Options& options);
// The command that starts a model prepared with --catalogue-id: serve llm
// with --manifest <folder> (GGUF entries) or --model <folder> (DiffusionGemma
// safetensors folder).
std::string CatalogueServeCommand(const Options& options);
void PrintHelp(std::ostream& out);
int Run(std::span<const char* const> args, std::string_view local_app_data,
        std::ostream& out, std::ostream& err,
        const VcRuntimeProbe* prerequisite_probe = nullptr);
}  // namespace bubo::prepare
