#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace bubo::cli {
struct SupportBundleOptions {
  bool json = false, preview = false, bundle_option = false;
  std::string path_text;
};
bool ParseSupportBundleOptions(std::span<const char* const> args,
                               SupportBundleOptions& options);

struct SupportBundleFile {
  std::string name;
  std::string text;
  std::string sha256;
};

struct SupportBundle {
  std::vector<SupportBundleFile> files;
  std::string zip;
  std::string zip_sha256;
};
struct SupportBinaryHashes {
  std::optional<std::string> bootstrap;
  std::optional<std::string> engine;
};

// This builder has no filesystem, clock, process, or network inputs. Every
// serialized value is reconstructed from an explicit, typed allow-list.
[[nodiscard]] SupportBundle BuildSupportBundle(
    const gufo::json::Value& status,
    const std::vector<gufo::json::Value>& history,
    const std::vector<std::string>& logs,
    const SupportBinaryHashes& binaries = {});

// Preview and saved events expose the same exact ZIP member text and hashes.
// Only "saved" differs. No output path or local identity enters either event.
[[nodiscard]] gufo::json::Value SupportBundleEvent(const SupportBundle& bundle,
                                                   bool saved);

// The target must be a new regular file. This function writes only that path.
void SaveSupportBundle(const SupportBundle& bundle,
                       const std::filesystem::path& path);

// args are the options after "diagnose". --preview may stand alone or accompany
// --bundle PATH. --json is required; duplicate and unknown options are errors.
int RunSupportBundle(std::span<const char* const> args,
                     std::string_view local_app_data,
                     const std::filesystem::path& runtime, std::ostream& out,
                     std::ostream& err);

}  // namespace bubo::cli
