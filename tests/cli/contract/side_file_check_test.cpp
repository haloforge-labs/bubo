// bubo models check-side-file: the shared side-file check (draft file, image
// projector, draft head), the command line around it, and the capabilities that
// `bubo models estimate` reports. Synthetic GGUF headers only: tiny files whose
// tensor payloads are never written (the file is only extended). No real model,
// no GPU, no network.
#include "src/cli/side_file_check.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/model_catalogue.hpp"
#include "src/cli/models_support.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "tests/support/temporary_directory.hpp"

namespace {
namespace fs = std::filesystem;
using namespace bubo::cli;
using gufo::core::GgmlType;
using gufo::json::Value;

constexpr std::uint64_t kGiB = 1073741824ULL;
int checks = 0;
void Require(bool condition, const std::string& message) {
  ++checks;
  if (std::getenv("BUBO_TEST_TRACE"))
    std::cerr << "  check " << checks << ": " << message.substr(0, 70)
              << (condition ? "" : "  <-- FAILED") << std::endl;
  if (!condition)
    throw std::runtime_error(message);
}
bool Contains(const std::string& text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

// ---- synthetic GGUF writer -------------------------------------------------
class Gguf {
public:
  void Str(const std::string& key, const std::string& value) {
    Key(key, 8);
    String(value);
  }
  void U32(const std::string& key, std::uint32_t value) {
    Key(key, 4);
    Raw(value);
  }
  void F32(const std::string& key, float value) {
    Key(key, 6);
    Raw(value);
  }
  void Bool(const std::string& key, bool value) {
    Key(key, 7);
    Raw(std::uint8_t{value});
  }
  void U32Array(const std::string& key, const std::vector<std::uint32_t>& v) {
    Key(key, 9);
    Raw(std::uint32_t{4});
    Raw(std::uint64_t{v.size()});
    for (auto item : v)
      Raw(item);
  }
  void F32Array(const std::string& key, const std::vector<float>& v) {
    Key(key, 9);
    Raw(std::uint32_t{6});
    Raw(std::uint64_t{v.size()});
    for (auto item : v)
      Raw(item);
  }
  void BoolArray(const std::string& key, std::size_t count, bool value) {
    Key(key, 9);
    Raw(std::uint32_t{7});
    Raw(std::uint64_t{count});
    for (std::size_t i = 0; i < count; ++i)
      Raw(std::uint8_t{value});
  }
  // A tensor whose payload is never written; dims are in GGUF order.
  void Tensor(const std::string& name, std::vector<std::uint64_t> dims,
              GgmlType type) {
    tensors_.push_back({name, std::move(dims), type});
  }
  void Write(const fs::path& path) const {
    std::string head;
    head += "GGUF";
    Append(head, std::uint32_t{3});
    Append(head, std::uint64_t{tensors_.size()});
    Append(head, std::uint64_t{count_});
    head += kv_;
    std::uint64_t offset = 0;
    for (const auto& tensor : tensors_) {
      Append(head, std::uint64_t{tensor.name.size()});
      head += tensor.name;
      Append(head, static_cast<std::uint32_t>(tensor.dims.size()));
      std::uint64_t elements = 1;
      for (const auto dim : tensor.dims) {
        Append(head, dim);
        elements *= dim;
      }
      Append(head, static_cast<std::uint32_t>(tensor.type));
      Append(head, offset);
      offset += (Bytes(tensor.type, elements) + 31) / 32 * 32;
    }
    head.resize((head.size() + 31) / 32 * 32, '\0');
    {
      std::ofstream file(path, std::ios::binary | std::ios::trunc);
      file.write(head.data(), static_cast<std::streamsize>(head.size()));
      if (!file)
        throw std::runtime_error("cannot write test GGUF");
    }
    // The payload is zero bytes that nobody reads: extend the file.
    fs::resize_file(path, head.size() + offset);
  }

private:
  struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    GgmlType type;
  };
  static std::uint64_t Bytes(GgmlType type, std::uint64_t elements) {
    switch (type) {
      case GgmlType::kF32:
        return elements * 4;
      case GgmlType::kF16:
      case GgmlType::kBF16:
        return elements * 2;
      case GgmlType::kQ8_0:
        return elements / 32 * 34;
      case GgmlType::kQ4_K:
        return elements / 256 * 144;
      case GgmlType::kQ5_K:
        return elements / 256 * 176;
      default:
        throw std::runtime_error("test writer: unsupported tensor type");
    }
  }
  template<class T>
  static void Append(std::string& out, T value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
  }
  template<class T>
  void Raw(T value) {
    Append(kv_, value);
  }
  void String(const std::string& value) {
    Raw(std::uint64_t{value.size()});
    kv_ += value;
  }
  void Key(const std::string& key, std::uint32_t type) {
    String(key);
    Raw(type);
    ++count_;
  }
  std::string kv_;
  std::uint64_t count_ = 0;
  std::vector<TensorSpec> tensors_;
};

// ---- the synthetic files ---------------------------------------------------
constexpr std::uint64_t kVocab = 16;  // rows of the output layer

struct FlashShape {
  std::uint32_t hidden = 2560, experts = 512;
  float rope_theta = 10000000.0F;
  bool draft = false;        // the MTP sidecar: one more block, no head tensor
  bool head_tensor = true;   // trunk: output.weight
  bool shared_head = false;  // trunk: token_embd.weight instead
};
// A Flash Next header that Config::FromGguf accepts (trunk, or draft-only
// sidecar when `draft`).
void WriteFlashNext(const fs::path& path, const FlashShape& s = {}) {
  const std::uint32_t layers = s.draft ? 49 : 48;
  Gguf g;
  g.Str("general.architecture", "qwen4exp");
  g.U32("qwen4exp.block_count", layers);
  if (s.draft)
    g.U32("qwen4exp.nextn_predict_layers", 1);
  g.U32("qwen4exp.embedding_length", s.hidden);
  g.U32("qwen4exp.context_length", 262144);
  g.U32("qwen4exp.hyper_connection.count", 4);
  g.U32("qwen4exp.hyper_connection.low_rank", 320);
  g.U32("qwen4exp.full_attention_interval", 4);
  g.U32("qwen4exp.attention.head_count", 24);
  g.U32("qwen4exp.attention.head_count_kv", 2);
  g.U32("qwen4exp.attention.key_length", 256);
  g.U32("qwen4exp.rope.dimension_count", 64);
  g.F32("qwen4exp.rope.freq_base", s.rope_theta);
  g.F32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F);
  g.U32Array("qwen4exp.rope.dimension_sections", {11, 11, 10, 0});
  g.U32("qwen4exp.attention.indexer.head_count", 4);
  g.U32("qwen4exp.attention.indexer.key_length", 128);
  g.U32("qwen4exp.attention.indexer.top_k", 2048);
  std::vector<std::uint32_t> ratios;
  for (std::uint32_t i = 0; i < layers; ++i)
    ratios.push_back(i < 48 && (i + 1) % 4 == 0 ? 4 : 0);
  g.U32Array("qwen4exp.attention.compress_ratios", ratios);
  g.U32("qwen4exp.ssm.conv_kernel", 4);
  g.U32("qwen4exp.ssm.state_size", 128);
  g.U32("qwen4exp.ssm.group_count", 16);
  g.U32("qwen4exp.ssm.time_step_rank", 48);
  g.U32("qwen4exp.ssm.inner_size", 6144);
  g.U32("qwen4exp.expert_count", s.experts);
  g.U32("qwen4exp.expert_used_count", 10);
  g.U32("qwen4exp.expert_feed_forward_length", 640);
  g.U32("qwen4exp.expert_shared_feed_forward_length", 640);
  if (!s.draft && s.head_tensor)
    g.Tensor(s.shared_head ? "token_embd.weight" : "output.weight",
             {2560, kVocab}, s.shared_head ? GgmlType::kF16 : GgmlType::kQ8_0);
  g.Write(path);
}

// A header with only the architecture and the width key, for the other
// families a side file can be offered for.
void WriteArch(const fs::path& path, const std::string& arch,
               std::uint32_t hidden = 0) {
  Gguf g;
  g.Str("general.architecture", arch);
  if (hidden)
    g.U32(arch + ".embedding_length", hidden);
  g.Write(path);
}

void WriteDraftHead(const fs::path& path, std::uint64_t rows, GgmlType type,
                    bool with_tensor = true) {
  Gguf g;
  g.Str("general.name", "Synthetic draft head");
  if (with_tensor)
    g.Tensor("output.weight", {2560, rows}, type);
  g.Write(path);
}

struct ProjectorShape {
  std::uint32_t width = 2560, patch = 16;
  std::string type = "qwen3vl_merger";
  bool all_tensors = true;
};
// The header of a Qwen image projector: the metadata and the four tensors that
// depend on the text-model width, plus small fillers up to the 334 tensors the
// encoder counts.
void WriteProjector(const fs::path& path, const ProjectorShape& s = {}) {
  Gguf g;
  g.Str("general.architecture", "clip");
  g.Str("general.type", "mmproj");
  g.Bool("clip.has_vision_encoder", true);
  g.Bool("clip.use_gelu", true);
  g.Str("clip.projector_type", s.type);
  g.U32("clip.vision.projection_dim", s.width);
  g.U32("clip.vision.patch_size", s.patch);
  g.U32("clip.vision.embedding_length", 1152);
  g.U32("clip.vision.feed_forward_length", 4304);
  g.U32("clip.vision.block_count", 27);
  g.U32("clip.vision.attention.head_count", 16);
  g.U32("clip.vision.spatial_merge_size", 2);
  g.U32("clip.vision.image_size", 768);
  g.F32("clip.vision.attention.layer_norm_epsilon", 1e-6F);
  g.F32Array("clip.vision.image_mean", {0.5F, 0.5F, 0.5F});
  g.F32Array("clip.vision.image_std", {0.5F, 0.5F, 0.5F});
  g.BoolArray("clip.vision.is_deepstack_layers", 27, false);
  g.Tensor("mm.0.weight", {4608, 4608}, GgmlType::kBF16);
  g.Tensor("mm.0.bias", {4608}, GgmlType::kF32);
  g.Tensor("mm.2.weight", {4608, s.width}, GgmlType::kBF16);
  g.Tensor("mm.2.bias", {s.width}, GgmlType::kF32);
  if (s.all_tensors)
    for (int i = 0; i < 330; ++i)
      g.Tensor("v.filler." + std::to_string(i), {1}, GgmlType::kF32);
  g.Write(path);
}

// ---- the checks ------------------------------------------------------------
SideFileCheck Check(const fs::path& model, SideRole role,
                    const fs::path& file) {
  std::string error;
  const auto check = CheckSideFile(model, role, file, &error);
  Require(check.has_value(), "the check ran: " + error);
  return *check;
}

void Draft(const fs::path& dir) {
  const auto trunk = dir / "trunk.gguf";
  WriteFlashNext(trunk);
  const auto fits = dir / "mtp.gguf";
  WriteFlashNext(fits, {.draft = true});
  auto check = Check(trunk, SideRole::kMtp, fits);
  Require(check.fits && check.kind_ok && check.family_ok && check.dims_ok &&
              check.reason.empty() && Contains(check.note, "origin cannot be"),
          "mtp fits: all flags true, empty reason, origin note");
  Require(check.role == SideRole::kMtp, "the check names its role");

  // The sidecar may carry another expert count than a pruned trunk.
  const auto pruned = dir / "trunk-256.gguf";
  WriteFlashNext(pruned, {.experts = 256});
  Require(Check(pruned, SideRole::kMtp, fits).fits,
          "512-expert draft file fits a 256-expert trunk");

  const auto wide = dir / "mtp-hidden.gguf";
  WriteFlashNext(wide, {.hidden = 2048, .draft = true});
  check = Check(trunk, SideRole::kMtp, wide);
  Require(
      !check.fits && check.kind_ok && check.family_ok && !check.dims_ok &&
          check.reason == "This draft file is built for a different model." &&
          check.note.empty(),
      "mtp with another hidden size: dims_ok false, plain reason");

  const auto theta = dir / "mtp-theta.gguf";
  WriteFlashNext(theta, {.rope_theta = 1000000.0F, .draft = true});
  check = Check(trunk, SideRole::kMtp, theta);
  Require(!check.fits && check.kind_ok && check.family_ok && !check.dims_ok,
          "MtpMatches: another rope base is another model");

  check = Check(trunk, SideRole::kMtp, trunk);
  Require(!check.fits && !check.kind_ok && !check.family_ok && !check.dims_ok &&
              check.reason == "This file is not a draft file.",
          "a full model is not a draft file: kind_ok false");

  const auto dense = dir / "dense.gguf";
  WriteArch(dense, "qwen35", 5120);
  check = Check(dense, SideRole::kMtp, fits);
  Require(!check.fits && check.kind_ok && !check.family_ok && !check.dims_ok &&
              check.reason ==
                  "This draft file is for Qwen3.8 Flash Next models only.",
          "mtp for another architecture: family_ok false");

  const auto gemma = dir / "gemma.gguf";
  const auto assistant = dir / "gemma-assistant.gguf";
  WriteArch(gemma, "gemma4", 256);
  WriteArch(assistant, "gemma4-assistant", 256);
  check = Check(gemma, SideRole::kMtp, assistant);
  Require(check.fits && !check.note.empty(),
          "a Gemma 4 draft file for a Gemma 4 model fits, with a note");
  check = Check(trunk, SideRole::kMtp, assistant);
  Require(!check.fits && check.kind_ok && !check.family_ok,
          "a Gemma 4 draft file for Flash Next: family_ok false");
  check = Check(gemma, SideRole::kMtp, fits);
  Require(!check.fits && check.kind_ok && !check.family_ok,
          "a Flash Next draft file for Gemma 4: family_ok false");

  const auto junk = dir / "junk.gguf";
  std::ofstream(junk, std::ios::binary) << std::string(128, 'x');
  check = Check(trunk, SideRole::kMtp, junk);
  Require(!check.fits && !check.kind_ok && !check.family_ok && !check.dims_ok &&
              check.reason == "This file is not a valid GGUF file.",
          "not a GGUF: kind_ok false");

  std::string error;
  Require(!CheckSideFile(trunk, SideRole::kMtp, dir / "absent.gguf", &error) &&
              !error.empty(),
          "a missing side file cannot be checked");
  error.clear();
  Require(!CheckSideFile(dir / "absent.gguf", SideRole::kMtp, fits, &error) &&
              !error.empty(),
          "a missing model cannot be checked");
  error.clear();
  Require(!CheckSideFile(junk, SideRole::kMtp, fits, &error) &&
              Contains(error, "model header"),
          "a model that is not a GGUF cannot be checked");
}

void Projector(const fs::path& dir) {
  const auto flash = dir / "trunk.gguf";  // written by Draft()
  const auto fits = dir / "mmproj.gguf";
  WriteProjector(fits);
  auto check = Check(flash, SideRole::kMmproj, fits);
  Require(check.fits && check.kind_ok && check.family_ok && check.dims_ok &&
              check.reason.empty() && Contains(check.note, "origin cannot be"),
          "mmproj fits a Flash Next model");

  const auto other = dir / "mmproj-other.gguf";
  WriteProjector(other, {.type = "gemma4v"});
  check = Check(flash, SideRole::kMmproj, other);
  Require(
      !check.fits && check.kind_ok && !check.family_ok && !check.dims_ok &&
          check.reason == "This image file is not a Qwen3.8 image projector.",
      "mmproj with another projector type: family_ok false");

  const auto small = dir / "mmproj-2048.gguf";
  WriteProjector(small, {.width = 2048});
  check = Check(flash, SideRole::kMmproj, small);
  Require(!check.fits && check.kind_ok && check.family_ok && !check.dims_ok &&
              check.reason == "This image file is built for a different model.",
          "mmproj with another projection_dim: dims_ok false");

  const auto patch = dir / "mmproj-patch.gguf";
  WriteProjector(patch, {.patch = 14});
  Require(!Check(flash, SideRole::kMmproj, patch).dims_ok,
          "mmproj with another patch size: dims_ok false");

  const auto bare = dir / "mmproj-bare.gguf";
  WriteProjector(bare, {.all_tensors = false});
  Require(!Check(flash, SideRole::kMmproj, bare).dims_ok,
          "mmproj with another tensor count: dims_ok false");

  const auto dense = dir / "dense.gguf";  // written by Draft(), width 5120
  const auto wide = dir / "mmproj-5120.gguf";
  WriteProjector(wide, {.width = 5120});
  Require(Check(dense, SideRole::kMmproj, wide).fits,
          "mmproj of width 5120 fits a 5120-wide dense Qwen model");
  Require(!Check(flash, SideRole::kMmproj, wide).fits,
          "the same file does not fit the 2560-wide Flash Next model");

  check = Check(flash, SideRole::kMmproj, flash);
  Require(!check.fits && !check.kind_ok &&
              check.reason == "This file is not an image projector file.",
          "a model is not an image projector: kind_ok false");

  const auto gemma = dir / "gemma.gguf";
  Require(Check(gemma, SideRole::kMmproj, other).fits,
          "another family's projector is not judged by Qwen's constants");
  check = Check(gemma, SideRole::kMmproj, fits);
  Require(!check.fits && check.kind_ok && !check.family_ok,
          "a Qwen projector for a Gemma 4 model: family_ok false");
}

void DraftHead(const fs::path& dir) {
  const auto trunk = dir / "trunk.gguf";  // output.weight 2560 x 16 Q8_0
  const auto fits = dir / "head-q4.gguf";
  WriteDraftHead(fits, kVocab, GgmlType::kQ4_K);
  auto check = Check(trunk, SideRole::kDraftHead, fits);
  Require(check.fits && check.kind_ok && check.family_ok && check.dims_ok &&
              check.reason.empty() && Contains(check.note, "origin cannot be"),
          "draft head fits: Q4_K with the trunk head's shape");
  const auto q5 = dir / "head-q5.gguf";
  WriteDraftHead(q5, kVocab, GgmlType::kQ5_K);
  Require(Check(trunk, SideRole::kDraftHead, q5).fits, "draft head Q5_K fits");

  const auto q8 = dir / "head-q8.gguf";
  WriteDraftHead(q8, kVocab, GgmlType::kQ8_0);
  check = Check(trunk, SideRole::kDraftHead, q8);
  Require(!check.fits && check.kind_ok && check.family_ok && !check.dims_ok &&
              check.reason ==
                  "This draft head file does not match this "
                  "model's output layer.",
          "draft head with the wrong type: dims_ok false");
  const auto rows = dir / "head-rows.gguf";
  WriteDraftHead(rows, kVocab * 2, GgmlType::kQ4_K);
  Require(!Check(trunk, SideRole::kDraftHead, rows).dims_ok,
          "draft head with the wrong shape: dims_ok false");

  const auto none = dir / "head-none.gguf";
  WriteDraftHead(none, kVocab, GgmlType::kQ4_K, false);
  check = Check(trunk, SideRole::kDraftHead, none);
  Require(!check.fits && !check.kind_ok,
          "a file without output.weight is not a draft head");

  const auto shared = dir / "trunk-shared.gguf";
  WriteFlashNext(shared, {.shared_head = true});
  Require(Check(shared, SideRole::kDraftHead, fits).fits,
          "a model that shares its embedding table as the head is checked "
          "against that table");
  const auto dense = dir / "dense.gguf";
  check = Check(dense, SideRole::kDraftHead, fits);
  Require(!check.fits && check.kind_ok && !check.family_ok &&
              Contains(check.reason, "Qwen3.8 Flash Next models only"),
          "draft head for another architecture: family_ok false");
}

void Command(const fs::path& dir) {
  const auto run = [&](std::vector<const char*> args, std::string& out_text,
                       std::string& err_text) {
    std::ostringstream out, err;
    const auto code = RunModels(args, dir.string(), out, err);
    out_text = out.str();
    err_text = err.str();
    return code;
  };
  const auto trunk = (dir / "trunk.gguf").string();
  const auto mtp = (dir / "mtp.gguf").string();
  const auto wide = (dir / "mtp-hidden.gguf").string();
  std::string out, err;
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp",
               "--file", mtp.c_str(), "--json"},
              out, err) == 0,
          "fitting file: exit 0");
  auto event = gufo::json::parse(out);
  Require(event.member_str("type") == "side_file_check", "event type");
  const auto* payload = event.find("payload");
  Require(payload && payload->member_str("role") == "mtp" &&
              payload->find("fits")->as_bool() &&
              payload->find("kind_ok")->as_bool() &&
              payload->find("family_ok")->as_bool() &&
              payload->find("dims_ok")->as_bool() &&
              payload->member_str("reason").empty() &&
              Contains(payload->member_str("note"), "origin cannot be"),
          "fitting file: payload");
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp",
               "--file", wide.c_str(), "--json"},
              out, err) == 0,
          "file that does not fit: exit 0 as well");
  event = gufo::json::parse(out);
  payload = event.find("payload");
  Require(payload && !payload->find("fits")->as_bool() &&
              !payload->find("dims_ok")->as_bool() &&
              payload->find("note") == nullptr &&
              payload->member_str("reason") ==
                  "This draft file is built for a different model.",
          "file that does not fit: payload has a reason and no note");
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp",
               "--file", mtp.c_str(), "--jsonl"},
              out, err) == 0 &&
              Contains(out, "\"type\":\"capabilities\"") &&
              Contains(out, "\"type\":\"side_file_check\""),
          "--jsonl: capabilities, then the answer");
  const auto folder = dir / "model-folder";
  fs::create_directories(folder);
  fs::copy_file(trunk, folder / "only.gguf",
                fs::copy_options::overwrite_existing);
  Require(run({"check-side-file", "--model", folder.string().c_str(), "--role",
               "mtp", "--file", mtp.c_str(), "--json"},
              out, err) == 0 &&
              Contains(out, "\"fits\":true"),
          "a folder with one GGUF file is the model");
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp",
               "--file", mtp.c_str()},
              out, err) == 0 &&
              Contains(out, "fits this model"),
          "human output");

  const auto absent = (dir / "absent.gguf").string();
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp",
               "--file", absent.c_str(), "--json"},
              out, err) == 12 &&
              Contains(out, "\"type\":\"error\"") &&
              Contains(out, "model_missing"),
          "missing side file: typed error, code 12");
  Require(run({"check-side-file", "--model", absent.c_str(), "--role", "mtp",
               "--file", mtp.c_str(), "--json"},
              out, err) == 12 &&
              Contains(out, "model_missing"),
          "missing model: typed error, code 12");
  const auto junk = (dir / "junk.gguf").string();
  Require(run({"check-side-file", "--model", junk.c_str(), "--role", "mtp",
               "--file", mtp.c_str(), "--json"},
              out, err) == 2 &&
              Contains(out, "invalid_arguments"),
          "a model that is not a GGUF: typed error, code 2");
  Require(
      run({"check-side-file", "--role", "mtp", "--file", mtp.c_str(), "--json"},
          out, err) == 2,
      "--model is required");
  Require(
      run({"check-side-file", "--model", trunk.c_str(), "--file", mtp.c_str()},
          out, err) == 2,
      "--role is required");
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "mtp"},
              out, err) == 2,
          "--file is required");
  Require(run({"check-side-file", "--model", trunk.c_str(), "--role", "twin",
               "--file", mtp.c_str()},
              out, err) == 2,
          "--role takes mtp, mmproj or draft_head");
  Require(run({"check-side-file", "--help"}, out, err) == 0 &&
              Contains(out, "check-side-file --model"),
          "help names the command");
  Require(run({"estimate", "--model", trunk.c_str(), "--role", "mtp"}, out,
              err) == 2,
          "--role belongs to check-side-file only");
}

void CapabilityChecks(const fs::path& dir) {
  using gufo::server::PrefillFastRefusal;
  using gufo::server::PrefillFastRefusalFor;
  Require(
      PrefillFastRefusalFor("qwen3.8-flash-next", 512) ==
              PrefillFastRefusal::kNone &&
          PrefillFastRefusalFor("qwen3.8-flash-next", 256) ==
              PrefillFastRefusal::kExperts &&
          PrefillFastRefusalFor("qwen35", 0) == PrefillFastRefusal::kFamily &&
          PrefillFastRefusalFor("gemma4", 128) == PrefillFastRefusal::kFamily &&
          PrefillFastRefusalFor("", 0) == PrefillFastRefusal::kNone,
      "the shared rule: family first, then expert count, unknown allowed");

  const auto available = [](const Value& capabilities) {
    return capabilities.find("prefill_mode_fast")->find("available")->as_bool();
  };
  const auto reason = [](const Value& capabilities) {
    return capabilities.find("prefill_mode_fast")->member_str("reason");
  };
  const std::string unavailable =
      "Fast mode is for Qwen3.8 Flash Next with 512 experts only.";
  const auto flash = ModelCapabilities("qwen4exp", 512);
  Require(available(flash) && reason(flash).empty() &&
              flash.find("routed_experts")->as_size() == 512,
          "512-expert Flash Next: available, empty reason, experts reported");
  const auto pruned = ModelCapabilities("qwen4exp", 256);
  Require(!available(pruned) && reason(pruned) == unavailable &&
              pruned.find("routed_experts")->as_size() == 256,
          "256-expert Flash Next: not available");
  const auto dense = ModelCapabilities("qwen35", 0);
  Require(!available(dense) && reason(dense) == unavailable &&
              dense.find("routed_experts") == nullptr,
          "a dense model: not available, no expert count");
  Require(!available(ModelCapabilities("gemma4", 128)) &&
              !available(ModelCapabilities("qwen35moe", 256)) &&
              !available(ModelCapabilities("", 0)),
          "other MoE families and unknown models: not available");

  // Q8 K/V storage: Flash Next of any expert count (the 512-expert models and
  // the Coder), nothing else; serve's rule is KvCacheQ8RefusalFor.
  using gufo::server::KvCacheQ8Refusal;
  using gufo::server::KvCacheQ8RefusalFor;
  Require(KvCacheQ8RefusalFor("qwen3.8-flash-next") == KvCacheQ8Refusal::kNone &&
              KvCacheQ8RefusalFor("") == KvCacheQ8Refusal::kNone &&
              KvCacheQ8RefusalFor("qwen35") == KvCacheQ8Refusal::kFamily &&
              KvCacheQ8RefusalFor("gemma4") == KvCacheQ8Refusal::kFamily,
          "the Q8 rule: Flash Next only, unknown allowed");
  const auto q8 = [](const Value& capabilities) {
    return capabilities.find("kv_cache_q8")->find("available")->as_bool();
  };
  Require(q8(ModelCapabilities("qwen4exp", 512)) &&
              q8(ModelCapabilities("qwen4exp", 256)) &&
              !q8(ModelCapabilities("qwen35", 0)) &&
              !q8(ModelCapabilities("gemma4", 128)) &&
              !q8(ModelCapabilities("", 0)) &&
              ModelCapabilities("qwen35", 0)
                      .find("kv_cache_q8")
                      ->member_str("reason")
                      .find("not supported for this model yet; it uses f16") !=
                  std::string::npos,
          "kv_cache_q8 capability: Flash Next (both expert counts) yes, dense, "
          "Gemma and unknown no, with a reason");

  // In-engine abliteration: 512-expert Flash Next that is not already
  // abliterated (owner 11-02: not the Huihui builds, not the 256-expert Coder).
  using gufo::server::ModelIsAbliterated;
  using gufo::server::RefusalDirectionRefusal;
  using gufo::server::RefusalDirectionRefusalFor;
  Require(RefusalDirectionRefusalFor("qwen3.8-flash-next", 512, false) ==
                  RefusalDirectionRefusal::kNone &&
              RefusalDirectionRefusalFor("qwen3.8-flash-next", 512, true) ==
                  RefusalDirectionRefusal::kAbliterated &&
              RefusalDirectionRefusalFor("qwen3.8-flash-next", 256, false) ==
                  RefusalDirectionRefusal::kExperts &&
              RefusalDirectionRefusalFor("gemma4", 128, false) ==
                  RefusalDirectionRefusal::kFamily &&
              RefusalDirectionRefusalFor("", 0, false) ==
                  RefusalDirectionRefusal::kNone,
          "abliteration rule: family, then experts, then already abliterated");
  Require(ModelIsAbliterated("huihui-qwen38-flash-next-ud-q4-k-xl-v1", "") &&
              ModelIsAbliterated("huihui-qwen38-flash-next-lowmemory-v1", "") &&
              ModelIsAbliterated("", "Qwen3.8 Flash Next (Huihui abliterated "
                                     "non-expert + ISTA GSQ-RCO IQ3_S experts)") &&
              !ModelIsAbliterated("", "Qwen3.8-Flash-Next GSQ-RCO") &&
              !ModelIsAbliterated("qwen38-flash-next-base-ud-q4-k-xl-v1",
                                  "Qwen3.8 Flash Next"),
          "Huihui packages and abliterated names count; ISTA and Unsloth do not");
  const auto direction = [](const Value& capabilities) {
    return capabilities.find("refusal_direction")->find("available")->as_bool();
  };
  const auto direction_reason = [](const Value& capabilities) {
    return capabilities.find("refusal_direction")->member_str("reason");
  };
  Require(direction(flash) && direction_reason(flash).empty(),
          "512-expert Flash Next: abliteration available");
  const auto huihui = ModelCapabilities("qwen4exp", 512, true);
  Require(!direction(huihui) &&
              direction_reason(huihui) == "This model is already abliterated." &&
              available(huihui),
          "already abliterated: abliteration off, fast prefill still on");
  Require(!direction(pruned) &&
              direction_reason(pruned) ==
                  "In-engine abliteration is not tested for this model.",
          "256-expert Coder: abliteration not available");
  Require(!direction(dense) && !direction(ModelCapabilities("", 0)),
          "dense and unknown models: abliteration not available");

  // The estimate of synthetic headers carries the same object.
  EstimateRequest request;
  request.available_bytes = 128 * kGiB;
  request.host_available_bytes = 128 * kGiB;
  request.weights_bytes = 82500000000ULL;
  const auto full = dir / "trunk.gguf";
  const auto cut = dir / "trunk-256.gguf";
  const auto estimate_of = [&](const fs::path& path) {
    return EstimateJson(EstimateForGguf(InspectGguf(path), request));
  };
  auto json = estimate_of(full);
  const auto* capabilities = json.find("capabilities");
  Require(capabilities && available(*capabilities) &&
              capabilities->find("routed_experts")->as_size() == 512,
          "estimate of a 512-expert header: fast prefill available");
  json = estimate_of(cut);
  capabilities = json.find("capabilities");
  Require(capabilities && !available(*capabilities) &&
              capabilities->find("routed_experts")->as_size() == 256,
          "estimate of a 256-expert header: fast prefill not available");
  json = estimate_of(dir / "dense.gguf");
  capabilities = json.find("capabilities");
  Require(capabilities && !available(*capabilities),
          "estimate of a dense header: not available");
  json = estimate_of(dir / "junk.gguf");
  Require(json.find("capabilities") == nullptr,
          "an unreadable file has no capabilities");
  json = estimate_of(dir / "mmproj.gguf");
  Require(json.find("capabilities") == nullptr,
          "a side file has no capabilities");

  // The catalogue: fast prefill is available for exactly the Flash Next
  // entries with 512 experts.
  const auto& shipped = ShippedCatalogue();
  Require(shipped.error.empty(), "the shipped catalogue is valid");
  const std::set<std::string> flash_ids = {
      std::string(bubo::prepare::kPackageId),
      std::string(bubo::prepare::kOriginalPackageId),
      std::string(bubo::prepare::kLowMemoryPackageId)};
  std::set<std::string> fast_ids;
  for (const auto& entry : shipped.entries) {
    const auto caps = CatalogueCapabilities(entry);
    if (available(caps))
      fast_ids.insert(entry.id);
    else
      Require(reason(caps) == unavailable,
              entry.id + ": the reason is the one plain sentence");
  }
  Require(fast_ids == flash_ids,
          "fast prefill is available for exactly the three 512-expert Flash "
          "Next entries");
  const auto* coder = FindCatalogueEntry(
      shipped.entries, "qwen38-flash-next-coder-ista-gsq-rco-256");
  Require(coder && coder->routed_experts == 256 &&
              CatalogueCapabilities(*coder).find("routed_experts")->as_size() ==
                  256,
          "the 256-expert Coder states its expert count");
  // The same object comes out of a catalogue estimate with no file present.
  for (const char* id : {"qwen38-flash-next-coder-ista-gsq-rco-256",
                         "gemma4-26b-a4b-unsloth-qat-ud-q4-k-xl"}) {
    json = EstimateJson(EstimateForCatalogueEntry(
        *FindCatalogueEntry(shipped.entries, id), dir / "no-models", request));
    capabilities = json.find("capabilities");
    Require(capabilities && !available(*capabilities),
            std::string(id) + ": catalogue estimate reports the capabilities");
  }
}
}  // namespace

int main() {
  try {
    TemporaryDirectory directory("bubo-side-file-test");
    const auto dir = directory.path();
    Draft(dir);
    Projector(dir);
    DraftHead(dir);
    Command(dir);
    CapabilityChecks(dir);
    std::cout << "side_file_check_test: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "side_file_check_test FAILED: " << error.what() << '\n';
    return 1;
  }
}
