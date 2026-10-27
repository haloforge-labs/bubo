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
#include "src/core/experimental_authorization.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

// Clients see the package id, or the stem of an imported GGUF file. An
// explicit name wins; the API also accepts the compatibility alias bubo.
inline std::string ServedModelName(std::string_view supplied,
                                   const std::filesystem::path& model_path,
                                   const ModelPackage& package = {}) {
  if (!supplied.empty())
    return std::string(supplied);
  if (!package.manifest_path.empty() || !package.files.empty())
    return package.package_id;
  const auto name = model_path.stem().u8string();
  return {name.begin(), name.end()};
}

inline constexpr std::string_view ServedModelAlias() {
  return "bubo";
}
inline bool AcceptsServedModel(std::string_view requested,
                               std::string_view served) {
  return requested == served || requested == ServedModelAlias();
}

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
  ReleaseEnvironment environment;
  std::function<const char*(const char*)> real_environment;
  std::string model_family;
  std::string architecture;
  std::optional<std::uint64_t> native_context;
  bool manifest_present = false;
  // The model GGUF carries its own MTP block (nextn_predict_layers > 0), so
  // `--speculative mtp` needs no separate --mtp-model file.
  bool model_has_mtp = false;
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
