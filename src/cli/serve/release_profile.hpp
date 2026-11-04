#ifndef GUFO_SERVER_RELEASE_PROFILE_HPP_
#define GUFO_SERVER_RELEASE_PROFILE_HPP_

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/serve/model_manifest.hpp"
#include "src/cli/serve/served_model_alias.hpp"
#include "src/core/experimental_authorization.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

// The stem of a split GGUF shard names the whole model: drop a trailing
// "-00001-of-00004" (five digits, "-of-", five digits). Any other stem stays.
inline std::string StripShardSuffix(std::string stem) {
  constexpr std::string_view kMiddle = "-of-";
  constexpr std::size_t kDigits = 5;
  constexpr std::size_t kSuffix = 1 + kDigits + kMiddle.size() + kDigits;
  if (stem.size() <= kSuffix)
    return stem;
  const std::size_t at = stem.size() - kSuffix;
  const auto digits = [&](std::size_t from) {
    for (std::size_t i = from; i < from + kDigits; ++i)
      if (stem[i] < '0' || stem[i] > '9')
        return false;
    return true;
  };
  if (stem[at] != '-' || !digits(at + 1) ||
      stem.compare(at + 1 + kDigits, kMiddle.size(), kMiddle) != 0 ||
      !digits(at + 1 + kDigits + kMiddle.size()))
    return stem;
  stem.resize(at);
  return stem;
}

// Clients see the package id, or the stem of an imported GGUF file without a
// shard suffix. An explicit name wins; the API also accepts the compatibility
// alias bubo.
inline std::string ServedModelName(std::string_view supplied,
                                   const std::filesystem::path& model_path,
                                   const ModelPackage& package = {}) {
  if (!supplied.empty())
    return std::string(supplied);
  if (!package.manifest_path.empty() || !package.files.empty())
    return package.package_id;
  const auto name = model_path.stem().u8string();
  return StripShardSuffix(std::string(name.begin(), name.end()));
}

inline bool AcceptsServedModel(std::string_view requested,
                               std::string_view served) {
  return requested == served || requested == ServedModelAlias();
}

inline constexpr std::string_view kPrefillFoldFileName = "prefill-fold-L3.txt";
// The per-layer fold scalars were fitted on the 512-expert Flash-Next models.
inline constexpr std::uint32_t kPrefillFoldExperts = 512;
// Why `--prefill-mode fast` cannot run on a model. The one place that decides
// it: serve refuses with it and `bubo models estimate` reports it as
// capabilities.prefill_mode_fast, so the hint cannot drift from the refusal.
// An empty family or a zero expert count means "not known yet" and does not
// refuse (the first startup pass runs before the model header is read).
enum class PrefillFastRefusal { kNone, kFamily, kExperts };
PrefillFastRefusal PrefillFastRefusalFor(std::string_view model_family,
                                         std::uint32_t expert_count);
// Why `--kv-cache-type q8_0` cannot run on a model: `bubo models estimate`
// reports it as capabilities.kv_cache_q8 and serve falls back to f16 by the
// same rule. Q8 K/V storage exists on Flash-Next only (the 512-expert models
// and the Coder); an empty family means "not known yet".
enum class KvCacheQ8Refusal { kNone, kFamily };
KvCacheQ8Refusal KvCacheQ8RefusalFor(std::string_view model_family);
std::string_view KvCacheQ8RefusalText(KvCacheQ8Refusal);
// serve does not refuse such a request: the effective cache type is f16 and
// this line is logged once.
inline constexpr std::string_view kKvCacheQ8FallbackLine =
    "KV cache: Q8 is not supported for this model yet; using f16";
// Why `--refusal-direction on` cannot run on a model, with the same contract
// as PrefillFastRefusal: serve refuses with it and `bubo models estimate`
// reports it as capabilities.refusal_direction. The packaged direction was
// derived on the 512-expert Flash-Next models; an already abliterated model
// (the Huihui builds) would get a second projection on top of its own.
enum class RefusalDirectionRefusal { kNone, kFamily, kExperts, kAbliterated };
RefusalDirectionRefusal RefusalDirectionRefusalFor(std::string_view model_family,
                                                   std::uint32_t expert_count,
                                                   bool already_abliterated);
// The user-facing sentence for a refusal (empty for kNone).
std::string RefusalDirectionRefusalText(RefusalDirectionRefusal refusal);
// A Huihui package id or "abliterated" in the GGUF general.name.
bool ModelIsAbliterated(std::string_view package_id,
                        std::string_view general_name);
using ReleaseEnvironment = std::map<std::string, std::optional<std::string>>;
class ReleaseSettingError : public std::invalid_argument {
public:
  ReleaseSettingError(std::string name, std::string message)
      : std::invalid_argument(std::move(message)),
        switch_name(std::move(name)) {}
  const std::string switch_name;
};
struct ResolvedModelArtefacts {
  // present prevents engine getenv fallback even when a path is disabled.
  bool present = false;
  bool vision_disabled = false;
  std::string twin, draft_head, mtp, mmproj;
};
struct ReleaseInputs {
  std::string profile = std::string(cli::kDefaultProfile);
  std::map<std::string, std::string> explicit_flags;
  std::filesystem::path model_path;
  // Fold scalars for --prefill-mode fast; installed beside the executable.
  std::filesystem::path prefill_fold_file;
  ReleaseEnvironment environment;
  std::function<const char*(const char*)> real_environment;
  std::string model_family;
  std::string architecture;
  std::optional<std::uint64_t> native_context;
  bool manifest_present = false;
  // The model GGUF carries its own MTP block (nextn_predict_layers > 0), so
  // `--speculative mtp` needs no separate --mtp-model file.
  bool model_has_mtp = false;
  // Routed experts per layer from the GGUF header (0 = unknown). The fast
  // prefill fold table is calibrated for kPrefillFoldExperts of them.
  std::uint32_t expert_count = 0;
  // The model is already abliterated (ModelIsAbliterated).
  bool model_abliterated = false;
  // Validated package paths; tests may supply fixture paths directly.
  std::map<std::string, std::string> manifest_defaults;
};
struct ReleaseSetting {
  std::string name;
  std::optional<std::string> value;
  std::string origin = "built-in";
  std::string release_class;
  std::string skipped_reason;
  std::optional<std::string> requested;
};
struct ReleaseResolution {
  core::ExperimentalAuthorization prefill_authorization;
  std::string profile;
  std::vector<ReleaseSetting> settings;
  std::vector<ReleaseSetting> injections;
  std::map<std::string, std::string> effective_flags;
  std::vector<std::string> logs;
  std::vector<std::string> turned_off;
  std::string architecture;
  std::string model_family;
  std::optional<std::uint64_t> native_context;
  ResolvedModelArtefacts artefacts;
  json::Value model_selection;
  std::optional<sampling::SamplingConfig> runtime_sampling;
  std::optional<bool> runtime_npu_enabled;
  std::string runtime_npu_reason;
  std::optional<std::string> runtime_host;
  std::optional<int> runtime_port;
  std::optional<std::uint64_t> runtime_context;
  std::string context_resolution_reason;
};
struct ReleaseStartup {
  ReleaseResolution resolved;
  std::string error;
  std::optional<std::string> error_switch;
  bool print_effective = false;
  bool active = false;
  bool model_missing = false;
  ModelPackage package;
};
const std::vector<json::Value>& ReleaseSwitchTable();
std::vector<json::Value> NormalizeSwitchTable(
    std::span<const json::Value> rows);
void ValidateSwitchTable(std::span<const json::Value> rows);
// Uses the same row-driven validator as profile and explicit-value resolution.
void ValidateSwitchValue(const json::Value& row, const std::string& value);
ReleaseEnvironment SnapshotReleaseEnvironment();
// The table parameter lets contract tests exercise extension rows through the
// production resolver and emitter, without replacing the compiled catalog.
ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs);
ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs,
                                         std::span<const json::Value> rows);
// Named status entries keep their contract fields; registry metadata is
// already carried by effective_switches and need not be repeated per key.
json::Value ReleaseSettingSummary(const json::Value& row);
// Selects the estimator's row; never recomputes a memory/context formula.
// An explicit --context always retains its value. The detected 96 GB path may
// adapt one session too, keeping its default when already recommended.
std::optional<std::uint64_t> ResolveDefaultSessionContext(
    ReleaseResolution& resolved, std::uint64_t sessions,
    const json::Value& memory_estimate, bool allow_single_session_adaptation = false);
void ApplyInjections(const ReleaseResolution& resolved);
// The explicit flags serve records for these command-line tokens (the
// spelling serve takes: `--name value` or `--name=value`). `models estimate`
// resolves its switches through this, so the two read a flag the same way.
std::map<std::string, std::string> ExplicitFlagsFromServeArgs(
    const std::vector<std::string>& tokens);
// Called as the first statement of RunServe, before any defaults or threads.
ReleaseStartup PrepareReleaseSettings(
    std::span<const char* const> args,
    const std::map<std::string, std::string>& manifest_defaults = {},
    bool inspect_only = false);
void PrefillReleaseFlags(const ReleaseResolution& resolved,
                         const cli::ArgParser& parser);
json::Value ReleaseSettingsJson(
    const ReleaseResolution& resolved,
    std::span<const json::Value> rows = ReleaseSwitchTable());
json::Value ReleaseOrigins(const ReleaseResolution& resolved);
std::string RenderReleaseDocs(std::span<const json::Value> rows);
std::string RenderReleaseInclude(std::span<const json::Value> rows);
std::string RenderArtefactBindingTable();
}  // namespace gufo::server
#endif
