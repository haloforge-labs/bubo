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
#include "src/core/json.hpp"

namespace gufo::server {

using ReleaseEnvironment = std::map<std::string, std::optional<std::string>>;
// P1 extension point: production leaves this empty. No discovery or file checks
// are implemented here. Tests can simulate an already reviewed manifest.
struct ManifestArtefacts {};
struct ReleaseInputs {
  std::string profile = std::string(cli::kDefaultProfile);
  std::map<std::string, std::string> explicit_flags;
  std::filesystem::path model_path;
  ReleaseEnvironment environment;
  std::function<const char*(const char*)> real_environment;
  std::string model_family;
  std::string architecture;
  std::optional<std::uint64_t> native_context;
  ManifestArtefacts manifest;
  // Simulate a future manifest's resolved defaults without a file loader.
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
  std::string profile;
  std::vector<ReleaseSetting> settings;
  std::vector<ReleaseSetting> injections;
  std::map<std::string, std::string> effective_flags;
  std::vector<std::string> logs;
  std::vector<std::string> turned_off;
  std::string architecture;
  std::optional<std::uint64_t> native_context;
};
struct ReleaseStartup {
  ReleaseResolution resolved;
  std::string error;
  bool print_effective = false;
  bool active = false;
};
const std::vector<json::Value>& ReleaseSwitchTable();
void ValidateSwitchTable(std::span<const json::Value> rows);
ReleaseEnvironment SnapshotReleaseEnvironment();
ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs);
void ApplyInjections(const ReleaseResolution& resolved);
// Called as the first statement of RunServe, before any defaults or threads.
ReleaseStartup PrepareReleaseSettings(std::span<const char* const> args);
void PrefillReleaseFlags(const ReleaseResolution& resolved,
                         const cli::ArgParser& parser);
json::Value ReleaseSettingsJson(const ReleaseResolution& resolved);
json::Value ReleaseOrigins(const ReleaseResolution& resolved);
std::string RenderReleaseDocs(std::span<const json::Value> rows);
std::string RenderReleaseInclude(std::span<const json::Value> rows);
}  // namespace gufo::server
#endif
