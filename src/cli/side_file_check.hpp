#pragma once

// Does a helper file fit a model? `bubo models check-side-file` and `serve`
// ask this one function, so the answer and the reason are the same in both.
//
// Only file headers are read: metadata and tensor names, shapes and types. No
// tensor data is read, and nothing here touches the GPU or the engine library.
// The checks reuse the engine's own rules (Config::MtpMatches, the Qwen vision
// projector constants, BindDraftOutput's shape and type rule).

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::server {
struct ReleaseResolution;
}

namespace bubo::cli {
enum class SideRole { kMtp, kMmproj, kDraftHead };
// "mtp" | "mmproj" | "draft_head"
std::string_view SideRoleName(SideRole role);
std::optional<SideRole> ParseSideRole(std::string_view name);

struct SideFileCheck {
  SideRole role = SideRole::kMtp;
  bool fits = false;
  // Each flag is false when an earlier one is false: a later check is not
  // meaningful then. kind_ok: a readable GGUF of the expected kind. family_ok:
  // the model family the engine expects. dims_ok: the sizes and tensors match.
  bool kind_ok = false, family_ok = false, dims_ok = false;
  std::string reason;  // one plain sentence; empty when the file fits
  std::string note;    // set only when it fits and something cannot be checked
};

// Checks `side_file` for `role` against the model whose first (or only) file is
// `model`. A file that is not a GGUF, or is not of the expected kind, is an
// answer (fits false). nullopt, with `error` set, means the check could not
// run: a file is missing or the model header cannot be read.
std::optional<SideFileCheck> CheckSideFile(
    const std::filesystem::path& model, SideRole role,
    const std::filesystem::path& side_file, std::string* error);
// The payload of the "side_file_check" event.
gufo::json::Value SideFileCheckJson(const SideFileCheck& check);

struct SideFileRefusal {
  std::string switch_name;  // the serve option that named the file
  std::string reason;       // the reason CheckSideFile gave
};
// Serve: the first side file the user named (flag or environment) that does not
// fit the model, or nullopt. A file that a package manifest pins is not checked
// here (the manifest hashes pin it). Runs before the model loads.
std::optional<SideFileRefusal> ExplicitSideFileRefusal(
    const gufo::server::ReleaseResolution& resolved,
    const std::filesystem::path& model);
}  // namespace bubo::cli
