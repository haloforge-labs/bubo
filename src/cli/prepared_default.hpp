#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace bubo::cli {
struct ModelSelection {
  std::filesystem::path model, manifest, pointer;
  std::string source;
};
std::string EnvironmentUtf8(std::string_view name);
std::filesystem::path ExecutableDirectory();
ModelSelection ResolvePreparedDefault(
    const std::filesystem::path& explicit_model,
    const std::filesystem::path& explicit_manifest,
    std::string_view environment_manifest, std::string_view local_app_data);
gufo::json::Value SelectionJson(const ModelSelection&);
bool RecordPreparedDefault(std::string_view local_app_data,
                           const std::filesystem::path& manifest,
                           std::string& error);
}  // namespace bubo::cli
