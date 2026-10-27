#pragma once

#include <filesystem>
#include <iosfwd>
#include <span>
#include <string_view>

#include "src/cli/machine_output.hpp"
#include "src/core/json.hpp"

namespace bubo::cli {
struct StatusOptions {
  MachineMode mode{MachineMode::Json};
  bool watch{false};
  std::string instance_id;
  // The package to report on, as serve takes it: --manifest or --model, not
  // both. Empty: the usual resolution (BUBO_MANIFEST, pointer, standard folder).
  std::filesystem::path manifest, model;
};
StatusOptions ParseStatusOptions(std::span<const char* const> args);

gufo::json::Value StatusSnapshot(std::string_view local_app_data,
                                 const std::filesystem::path& runtime,
                                 bool sample_machine = true,
                                 const std::filesystem::path& model = {},
                                 const std::filesystem::path& manifest = {});
int RunStatus(std::span<const char* const> args,
              std::string_view local_app_data,
              const std::filesystem::path& runtime, std::ostream& out,
              std::ostream& err);
}  // namespace bubo::cli
