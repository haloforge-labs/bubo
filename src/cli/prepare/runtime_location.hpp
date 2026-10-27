#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace bubo::prepare {
// The AMD ROCm runtime lives in one shared per-user folder:
//   %LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151
// so a new Bubo package folder does not lose it. A runtime beside bubo.exe
// (the v0.1.0 layout) keeps working.
inline constexpr std::string_view kRuntimeId = "rocm-10.0.0-gfx1151";
inline constexpr std::string_view kRuntimeDirEnvironment = "BUBO_RUNTIME_DIR";
inline constexpr std::string_view kRuntimeKindEnvironment = "BUBO_RUNTIME_KIND";

enum class RuntimeKind { Explicit, Shared, Legacy };
std::string_view RuntimeKindName(RuntimeKind kind);

struct RuntimeLocation {
  std::filesystem::path dir;
  RuntimeKind kind = RuntimeKind::Shared;
  // Empty when the runtime in `dir` can be used. Otherwise one plain sentence
  // for the user that ends with the remedy.
  std::string error;
  // True when no runtime was found anywhere (no marker, no unfinished
  // transaction, no explicit choice). Only then may a caller continue and let
  // the engine load report the missing files (CPU and development builds have
  // no AMD runtime). A damaged runtime is never "absent".
  bool absent = false;
  // Where the choice came from: option, environment, shared, legacy, default.
  std::string source;
};

// ROOT\runtime\<runtime id>. ROOT is the Bubo data folder
// (%LOCALAPPDATA%\Bubo).
std::filesystem::path SharedRuntimeDir(const std::filesystem::path& data_root);
// LOCALAPPDATA\Bubo\runtime\<runtime id>; empty when LOCALAPPDATA is unknown.
std::filesystem::path SharedRuntimeDirFor(std::string_view local_app_data);

// Empty when `text` can name a runtime folder: absolute and local.
std::string CheckRuntimeDirOption(std::string_view text);

// Returns an error text, or empty when `dir` holds a usable runtime. A marker
// or an unfinished transaction is authoritative. Without a marker a manually
// installed runtime (amdhip64_7.dll present) is accepted, as before.
std::string DefaultRuntimeCheck(const std::filesystem::path& dir);
bool HasRuntimeMarker(const std::filesystem::path& dir);

struct RuntimeSearch {
  std::string option;       // --runtime-dir
  std::string environment;  // BUBO_RUNTIME_DIR
  std::string local_app_data;
  std::filesystem::path exe_dir;
  // Test hook. Default: DefaultRuntimeCheck.
  std::function<std::string(const std::filesystem::path&)> check;
};
// Order: option > environment > shared folder when its marker is valid >
// legacy folder beside bubo.exe when usable > the shared folder (not installed,
// or damaged, with the error text).
RuntimeLocation ResolveRuntime(const RuntimeSearch& search);
// Process inputs shared by preparation and the server. The model/archive
// output root is deliberately separate from runtime selection.
RuntimeSearch ProcessRuntimeSearch(std::string_view local_app_data,
                                   const std::filesystem::path& exe_dir);
// Same, with BUBO_RUNTIME_DIR read from the process environment.
RuntimeLocation ResolveProcessRuntime(std::string_view local_app_data,
                                      const std::filesystem::path& exe_dir);
// BUBO_RUNTIME_DIR from the process environment; empty when unset.
std::string RuntimeDirFromEnvironment();
// "shared", "explicit" or "legacy" for a runtime folder already chosen.
std::string_view ClassifyRuntime(const std::filesystem::path& runtime,
                                 std::string_view local_app_data);
}  // namespace bubo::prepare
