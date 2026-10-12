#ifndef GUFO_SERVER_RELEASE_PROFILE_HPP_
#define GUFO_SERVER_RELEASE_PROFILE_HPP_

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/brand.hpp"
#include "src/cli/serve/model_manifest.hpp"
#include "src/core/experimental_authorization.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

using ReleaseEnvironment = std::map<std::string, std::optional<std::string>>;
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
  // Validated package paths; tests may supply fixture paths directly.
  std::map<std::string, std::string> manifest_defaults;
};
struct ReleaseSetting {
  std::string name;
  std::optional<std::string> value;
  std::string origin = "built-in";
  std::string release_class;
  std::string skipped_reason;
};
struct ReleaseResolution {
  core::ExperimentalAuthorization prefill_authorization;
  core::ExperimentalAuthorization fabric_authorization;
  std::string profile;
  std::vector<ReleaseSetting> settings;
  std::vector<ReleaseSetting> injections;
  std::map<std::string, std::string> effective_flags;
  std::vector<std::string> logs;
  std::vector<std::string> turned_off;
  std::string architecture;
  std::optional<std::uint64_t> native_context;
  ResolvedModelArtefacts artefacts;
};
struct ReleaseStartup {
  ReleaseResolution resolved;
  std::string error;
  bool print_effective = false;
  bool active = false;
  ModelPackage package;
};
const std::vector<json::Value>& ReleaseSwitchTable();
void ValidateSwitchTable(std::span<const json::Value> rows);
ReleaseEnvironment SnapshotReleaseEnvironment();
ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs);
void ApplyInjections(const ReleaseResolution& resolved);
// Called as the first statement of RunServe, before any defaults or threads.
ReleaseStartup PrepareReleaseSettings(
    std::span<const char* const> args,
    const std::map<std::string, std::string>& manifest_defaults = {});
void PrefillReleaseFlags(const ReleaseResolution& resolved,
                         const cli::ArgParser& parser);
json::Value ReleaseSettingsJson(const ReleaseResolution& resolved);
json::Value ReleaseOrigins(const ReleaseResolution& resolved);
std::string RenderReleaseDocs(std::span<const json::Value> rows);
std::string RenderReleaseInclude(std::span<const json::Value> rows);
std::string RenderArtefactBindingTable();
}  // namespace gufo::server
#endif
