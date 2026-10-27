#pragma once

// bubo models list / bubo models estimate.
//
// Nothing here starts a server, loads weights or touches the network. A GGUF
// file is read by its header only. The engine's own loaders decide whether a
// file is runnable and how large its per-session state is.

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/brand.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace bubo::cli {
struct ModelsCommandOptions {
  std::string verb;
  bool machine = false;
  std::vector<std::filesystem::path> scan_folders;
  std::filesystem::path models_root;
  std::string model;
  std::string profile = std::string(gufo::cli::kDefaultProfile);
  std::uint64_t context = 0, sessions = 1;
  bool help = false;
};
bool ParseModelsCommand(std::span<const char* const> args,
                        ModelsCommandOptions& options, std::string& error);

inline constexpr int kScanMaxDepth = 3;  // folders below a scan folder
inline constexpr std::size_t kScanMaxFiles = 2000;
inline constexpr std::size_t kScanMaxFolders = 20000;

struct GgufFacts {
  std::string path, file_name, model_name, architecture, error;
  bool header_ok = false;
  std::uint64_t size_bytes = 0;   // this file
  std::uint64_t total_bytes = 0;  // all shards found beside it
  unsigned shards_found = 1, shards_expected = 1;
  bool split_follower = false;  // shard 2 or later: listed with shard 1
  std::optional<std::uint64_t> native_context;
};

struct ArchitectureSupport {
  bool runnable = false;
  bool sidecar = false;  // vision projector or draft file, not a model
  std::string route;     // engine route when runnable
  std::string reason;    // empty when runnable
};

struct Inspection {
  GgufFacts facts;
  ArchitectureSupport support;
  std::unique_ptr<gufo::core::GgufReader> reader;  // null when unreadable
};
// Header only. Never reads tensor payloads.
Inspection InspectGguf(const std::filesystem::path& path);
ArchitectureSupport CheckArchitectureSupport(const gufo::core::GgufReader&);

struct ScanOptions {
  std::vector<std::filesystem::path> folders;
  int max_depth = kScanMaxDepth;
  std::size_t max_files = kScanMaxFiles;
  std::size_t max_folders = kScanMaxFolders;
};
struct ScanWarning {
  std::string folder, reason;
};
struct ScanResult {
  std::vector<std::filesystem::path> files;  // .gguf files found, sorted
  // Folders that hold a bubo-package.json, sorted. Such a folder is one
  // prepared package: its files are not in `files` and it is not entered.
  std::vector<std::filesystem::path> packages;
  std::vector<ScanWarning> warnings;
  std::vector<std::string> scanned_folders;
};
// A folder is refused when it is a drive root or a network share root, does
// not exist, or is not a folder. Links, junctions and other reparse points are
// not followed. A visited-folder set also stops any loop that remains.
ScanResult ScanModelFolders(const ScanOptions&);
bool IsFilesystemRoot(const std::filesystem::path&);
// Windows reparse tags that make a folder a link the scan must not follow:
// IO_REPARSE_TAG_SYMLINK and IO_REPARSE_TAG_MOUNT_POINT (a junction). Any
// other reparse point is an ordinary folder or file, for example a OneDrive
// placeholder (IO_REPARSE_TAG_CLOUD*).
bool IsLinkReparseTag(std::uint32_t tag);

struct EstimateRequest {
  std::string profile = std::string(gufo::cli::kDefaultProfile);
  std::uint64_t context =
      0;  // 0: serve release-profile context (native if unset)
  std::uint64_t sessions = 1;
  std::optional<std::uint64_t> available_bytes, host_available_bytes;
  PlanningMemoryBudgets capacity_budgets;
  // Sum of the files the user selected. Unset: all shards found for a path.
  std::optional<std::uint64_t> weights_bytes;
};
struct EstimateResult {
  std::string state;                  // "estimated" | "none_yet"
  std::string reason;                 // set when none_yet
  gufo::json::Value memory_estimate;  // null when none_yet
  std::string evidence_label;
  std::vector<std::string> notes;
  gufo::json::Value fit_inputs = gufo::json::Value::object();
};
EstimateResult EstimateForGguf(const Inspection&, const EstimateRequest&);
EstimateResult EstimateForCatalogueEntry(
    const CatalogueEntry&, const std::filesystem::path& models_root,
    const EstimateRequest&);
EstimateResult EstimateForFolder(const std::filesystem::path&,
                                 const EstimateRequest&);
gufo::json::Value EstimateJson(const EstimateResult&);
// The 96 GB path refuses a definite non-fit before loading model weights.
// Unknown readings and all other hardware retain the existing start policy.
std::string StartupMemoryRefusal(const PlanningMemoryBudgets&,
                                 const gufo::json::Value&, bool full_size);

// Memory the server can use, read the way serve reads it. nullopt when the
// machine does not report it.
std::optional<std::uint64_t> HostAvailableBytes();

struct ModelsListOptions {
  std::filesystem::path models_root;  // default %LOCALAPPDATA%\Bubo\models
  std::vector<std::filesystem::path> scan_folders;
  std::optional<std::uint64_t> available_bytes, host_available_bytes;
  PlanningMemoryBudgets capacity_budgets;
};
gufo::json::Value BuildModelsList(const ModelsListOptions&,
                                  const std::vector<CatalogueEntry>&);
// "prepared" | "partial" | "not_prepared": file sizes only.
std::string CataloguePreparedState(const CatalogueEntry&,
                                   const std::filesystem::path& models_root);
std::filesystem::path CatalogueEntryFolder(
    const CatalogueEntry&, const std::filesystem::path& models_root);

int RunModels(std::span<const char* const> args,
              std::string_view local_app_data, std::ostream& out,
              std::ostream& err);

}  // namespace bubo::cli
