#include "src/cli/side_file_check.hpp"

#include <array>
#include <exception>
#include <memory>
#include <system_error>

#include "src/cli/serve/release_profile.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/vision/projector_header.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/native_dispatch.hpp"

namespace bubo::cli {
namespace fs = std::filesystem;
namespace core = gufo::core;
namespace flash = gufo::models::qwen38_flash_next;
namespace vision = gufo::models::qwen::vision;
using gufo::json::Value;

namespace {
constexpr std::string_view kFlashNextFamily = "qwen3.8-flash-next";
constexpr const char* kNotGguf = "This file is not a valid GGUF file.";
constexpr const char* kOriginNote =
    "The dimensions match. The file's origin cannot be checked from its "
    "headers.";

std::string_view Architecture(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("");
}

SideFileCheck Refused(SideRole role, bool kind_ok, bool family_ok, bool dims_ok,
                      std::string reason) {
  SideFileCheck check;
  check.role = role;
  check.kind_ok = kind_ok;
  check.family_ok = family_ok;
  check.dims_ok = dims_ok;
  check.reason = std::move(reason);
  return check;
}

SideFileCheck Fits(SideRole role, std::string note) {
  SideFileCheck check;
  check.role = role;
  check.fits = check.kind_ok = check.family_ok = check.dims_ok = true;
  check.note = std::move(note);
  return check;
}

// Draft (MTP) file: Config::MtpMatches decides, as MtpWeights::Bind does.
SideFileCheck CheckMtp(const core::GgufReader& trunk,
                       const core::GgufReader& side) {
  constexpr auto role = SideRole::kMtp;
  const auto trunk_architecture = Architecture(trunk);
  const auto side_architecture = Architecture(side);
  const bool flash_draft =
      side_architecture == "qwen4exp" &&
      side.GetMetadataUint32("qwen4exp.nextn_predict_layers").value_or(0) > 0;
  const bool gemma_draft = side_architecture == "gemma4-assistant";
  if (!flash_draft && !gemma_draft)
    return Refused(role, false, false, false, "This file is not a draft file.");
  if (trunk_architecture == "gemma4") {
    // The Gemma 4 draft file is bound by the Gemma 4 loader; its header rules
    // are not repeated here.
    if (gemma_draft)
      return Fits(role,
                  "This draft file cannot be checked against this model from "
                  "its headers.");
    return Refused(role, true, false, false,
                   "This draft file is built for a different model.");
  }
  std::string error;
  std::optional<flash::Config> trunk_config;
  if (trunk_architecture == "qwen4exp")
    trunk_config = flash::Config::FromGguf(trunk, true, &error);
  if (!trunk_config)
    return Refused(role, true, false, false,
                   "This draft file is for Qwen3.8 Flash Next models only.");
  if (!flash_draft)
    return Refused(role, true, false, false,
                   "This draft file is built for a different model.");
  const auto side_config = flash::Config::FromGguf(side, false, &error);
  if (!side_config || !side_config->MtpMatches(*trunk_config))
    return Refused(role, true, true, false,
                   "This draft file is built for a different model.");
  return Fits(role, kOriginNote);
}

// Image projector: the constants the Qwen vision encoder hard-codes.
SideFileCheck CheckMmproj(const core::GgufReader& trunk,
                          const core::GgufReader& side) {
  constexpr auto role = SideRole::kMmproj;
  const std::string trunk_architecture(Architecture(trunk));
  const auto width =
      trunk.GetMetadataUint32(trunk_architecture + ".embedding_length")
          .value_or(0);
  const auto header = vision::CheckProjectorHeader(side, width);
  if (!header.kind_ok)
    return Refused(role, false, false, false,
                   "This file is not an image projector file.");
  if (!trunk_architecture.starts_with("qwen")) {
    // Another family brings its own vision encoder; only a Qwen projector can
    // be named wrong here.
    if (side.GetMetadataString("clip.projector_type") == vision::kProjectorType)
      return Refused(role, true, false, false,
                     "This image file is built for a different model.");
    return Fits(role,
                "This image file cannot be checked against this model from "
                "its headers.");
  }
  if (!header.family_ok)
    return Refused(role, true, false, false,
                   "This image file is not a Qwen3.8 image projector.");
  if (!header.dims_ok)
    return Refused(role, true, true, false,
                   "This image file is built for a different model.");
  return Fits(role, kOriginNote);
}

// Draft head: BindDraftOutput wants output.weight with the trunk head's shape
// in Q4_K or Q5_K.
SideFileCheck CheckDraftHead(const core::GgufReader& trunk,
                             const core::GgufReader& side) {
  constexpr auto role = SideRole::kDraftHead;
  const auto* head = side.FindTensor("output.weight");
  if (head == nullptr)
    return Refused(role, false, false, false,
                   "This file is not a draft head file.");
  std::string error;
  if (Architecture(trunk) != "qwen4exp" ||
      !flash::Config::FromGguf(trunk, true, &error))
    return Refused(
        role, true, false, false,
        "This draft head file is for Qwen3.8 Flash Next models only.");
  // The trunk head is output.weight, or the embedding table when the model
  // shares it (ModelWeights::Bind).
  const auto* trunk_head = trunk.FindTensor("output.weight");
  if (trunk_head == nullptr)
    trunk_head = trunk.FindTensor("token_embd.weight");
  const auto shape = [](const core::GgufTensorInfo& tensor) {
    const auto& d = tensor.dimensions;
    return std::array<std::uint64_t, 3>{d.size() > 0 ? d[0] : 1,
                                        d.size() > 1 ? d[1] : 1,
                                        d.size() > 2 ? d[2] : 1};
  };
  const bool fits =
      trunk_head != nullptr && trunk_head->dimensions.size() <= 3 &&
      flash::NativeDenseKernel(trunk_head->type, false) != nullptr &&
      head->dimensions.size() <= 3 && shape(*head) == shape(*trunk_head) &&
      shape(*head)[2] == 1 &&
      (head->type == core::GgmlType::kQ4_K ||
       head->type == core::GgmlType::kQ5_K);
  if (!fits)
    return Refused(
        role, true, true, false,
        "This draft head file does not match this model's output layer.");
  return Fits(role, kOriginNote);
}
}  // namespace

std::string_view SideRoleName(SideRole role) {
  switch (role) {
    case SideRole::kMtp:
      return "mtp";
    case SideRole::kMmproj:
      return "mmproj";
    case SideRole::kDraftHead:
      return "draft_head";
  }
  return "";
}

std::optional<SideRole> ParseSideRole(std::string_view name) {
  for (const auto role :
       {SideRole::kMtp, SideRole::kMmproj, SideRole::kDraftHead})
    if (SideRoleName(role) == name)
      return role;
  return std::nullopt;
}

std::optional<SideFileCheck> CheckSideFile(const fs::path& model, SideRole role,
                                           const fs::path& side_file,
                                           std::string* error) {
  const auto fail = [&](std::string text) {
    if (error)
      *error = std::move(text);
    return std::nullopt;
  };
  std::error_code ec;
  if (!fs::is_regular_file(model, ec))
    return fail("The model file is not on this computer.");
  if (!fs::is_regular_file(side_file, ec))
    return fail("The side file is not on this computer.");
  std::string open_error;
  // Only the draft head reads a tensor of the model, and the output layer can
  // sit in a later shard of a split model; the other checks read metadata.
  const auto trunk = role == SideRole::kDraftHead
                         ? core::GgufReader::OpenFile(model, &open_error)
                         : core::GgufReader::OpenHeaderFile(model, &open_error);
  if (!trunk)
    return fail("The model header cannot be read: " + open_error);
  const auto side = core::GgufReader::OpenHeaderFile(side_file, &open_error);
  if (!side)
    return Refused(role, false, false, false, kNotGguf);
  switch (role) {
    case SideRole::kMtp:
      return CheckMtp(*trunk, *side);
    case SideRole::kMmproj:
      return CheckMmproj(*trunk, *side);
    case SideRole::kDraftHead:
      return CheckDraftHead(*trunk, *side);
  }
  return fail("Unknown side file role.");
}

Value SideFileCheckJson(const SideFileCheck& check) {
  auto value = Value::object();
  value["role"] = std::string(SideRoleName(check.role));
  value["fits"] = check.fits;
  value["kind_ok"] = check.kind_ok;
  value["family_ok"] = check.family_ok;
  value["dims_ok"] = check.dims_ok;
  value["reason"] = check.reason;
  if (!check.note.empty())
    value["note"] = check.note;
  return value;
}

std::optional<SideFileRefusal> ExplicitSideFileRefusal(
    const gufo::server::ReleaseResolution& resolved, const fs::path& model) {
  std::error_code ec;
  if (!fs::is_regular_file(model, ec))
    return std::nullopt;  // a folder model has no GGUF header to check against
  struct Row {
    const char* setting;  // the release-settings row
    const char* flag;     // the serve option, for the error
    SideRole role;
  };
  const bool flash_next = resolved.model_family == kFlashNextFamily;
  // The draft file only matters when speculative decoding with it is selected.
  bool mtp_selected = false;
  for (const auto& candidate : resolved.settings)
    if (candidate.name == "--speculative" && candidate.value)
      mtp_selected = *candidate.value == "mtp";
  for (const Row& row : {Row{"--mtp-model", "--mtp-model", SideRole::kMtp},
                         Row{"--mmproj", "--mmproj", SideRole::kMmproj},
                         Row{"GUFO_DRAFT_OUT_MODEL", "--draft-head-model",
                             SideRole::kDraftHead}}) {
    // The families that bind a typed draft file; any other family reads its
    // own files and keeps its own checks.
    if (row.role == SideRole::kMtp &&
        (!mtp_selected || !(flash_next || resolved.model_family == "gemma4")))
      continue;
    if (row.role == SideRole::kDraftHead && !flash_next)
      continue;
    const gufo::server::ReleaseSetting* setting = nullptr;
    for (const auto& candidate : resolved.settings)
      if (candidate.name == row.setting)
        setting = &candidate;
    // Only a file the user named: a manifest pins its own files by hash.
    if (!setting || !setting->value || setting->value->empty() ||
        (setting->origin != "flag" && setting->origin != "env"))
      continue;
    const auto path = fs::u8path(*setting->value);
    if (!fs::is_regular_file(path, ec))
      continue;  // the loader names a missing file in its own words
    try {
      std::string error;
      const auto check = CheckSideFile(model, row.role, path, &error);
      if (check && !check->fits)
        return SideFileRefusal{row.flag, check->reason};
    } catch (const std::exception&) {
      // A check that cannot run never stops the server; the loader decides.
    }
  }
  return std::nullopt;
}
}  // namespace bubo::cli
