// J16: bubo models list / estimate. Synthetic GGUF files only: tiny headers
// (and tiny F32 tensors for the dense Qwen loader) written by this test. No
// real model, no GPU, no network. The engine's own loaders judge each file.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/machine_output.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/cli/models_support.hpp"
#include "src/cli/serve/memory_estimate.hpp"
#include "src/cli/settings_support.hpp"
#include "src/cli/status.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/session_state_bytes.hpp"
#include "src/models/qwen36_a3b/config.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace bubo::cli;
namespace prepare = bubo::prepare;
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
  void U32Array(const std::string& key, const std::vector<std::uint32_t>& v) {
    Key(key, 9);
    Raw(std::uint32_t{4});
    Raw(std::uint64_t{v.size()});
    for (auto item : v)
      Raw(item);
  }
  // A zero-filled F32 tensor with `elements` values.
  void Tensor(const std::string& name, std::uint64_t elements) {
    tensors_.push_back({name, elements});
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
      Append(head, std::uint32_t{1});
      Append(head, std::uint64_t{tensor.elements});
      Append(head, std::uint32_t{0});  // F32
      Append(head, offset);
      offset += (tensor.elements * 4 + 31) / 32 * 32;
    }
    head.resize((head.size() + 31) / 32 * 32, '\0');
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(head.data(), static_cast<std::streamsize>(head.size()));
    const std::string zeros(4096, '\0');
    for (std::uint64_t left = offset; left > 0;) {
      const auto chunk = std::min<std::uint64_t>(left, zeros.size());
      file.write(zeros.data(), static_cast<std::streamsize>(chunk));
      left -= chunk;
    }
    if (!file)
      throw std::runtime_error("cannot write test GGUF");
  }

private:
  struct TensorSpec {
    std::string name;
    std::uint64_t elements;
  };
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

// A tiny dense qwen35 model: 4 layers (3 recurrent, 1 attention). The head
// geometry is large enough that the per-token cache is measurable.
struct DenseShape {
  std::uint32_t kv_heads = 8, heads = 8, head_dim = 256;
  std::uint32_t context = 262144;
  bool drop_tensor = false;
};
void WriteDense(const fs::path& path, const DenseShape& s = {}) {
  constexpr std::uint64_t hidden = 64, inter = 64, vocab = 16;
  constexpr std::uint64_t rank = 2, inner = 256, groups = 1, state = 128;
  const std::uint64_t qkv = 2 * groups * state + inner;
  Gguf g;
  g.Str("general.architecture", "qwen35");
  g.Str("general.name", "Synthetic dense");
  g.U32("qwen35.block_count", 4);
  g.U32("qwen35.embedding_length", hidden);
  g.U32("qwen35.feed_forward_length", inter);
  g.U32("qwen35.attention.head_count", s.heads);
  g.U32("qwen35.attention.head_count_kv", s.kv_heads);
  g.U32("qwen35.attention.key_length", s.head_dim);
  g.U32("qwen35.context_length", s.context);
  g.U32("qwen35.full_attention_interval", 4);
  g.U32("qwen35.ssm.conv_kernel", 4);
  g.U32("qwen35.ssm.state_size", state);
  g.U32("qwen35.ssm.group_count", groups);
  g.U32("qwen35.ssm.time_step_rank", rank);
  g.U32("qwen35.ssm.inner_size", inner);
  g.F32("qwen35.rope.freq_base", 10000000.0F);
  g.U32("qwen35.rope.dimension_count", std::min<std::uint32_t>(64, s.head_dim));
  g.Tensor("token_embd.weight", vocab * hidden);
  g.Tensor("output_norm.weight", hidden);
  for (std::uint32_t i = 0; i < 4; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    g.Tensor(p + "attn_norm.weight", hidden);
    g.Tensor(p + "post_attention_norm.weight", hidden);
    g.Tensor(p + "ffn_gate.weight", inter * hidden);
    g.Tensor(p + "ffn_up.weight", inter * hidden);
    g.Tensor(p + "ffn_down.weight", hidden * inter);
    if (i == 3) {
      const std::uint64_t attention = std::uint64_t{s.heads} * s.head_dim;
      const std::uint64_t kv = std::uint64_t{s.kv_heads} * s.head_dim;
      if (!s.drop_tensor)
        g.Tensor(p + "attn_q.weight", 2 * attention * hidden);
      g.Tensor(p + "attn_k.weight", kv * hidden);
      g.Tensor(p + "attn_v.weight", kv * hidden);
      g.Tensor(p + "attn_output.weight", hidden * attention);
      g.Tensor(p + "attn_q_norm.weight", s.head_dim);
      g.Tensor(p + "attn_k_norm.weight", s.head_dim);
    } else {
      g.Tensor(p + "attn_qkv.weight", qkv * hidden);
      g.Tensor(p + "attn_gate.weight", inner * hidden);
      g.Tensor(p + "ssm_a", rank);
      g.Tensor(p + "ssm_conv1d.weight", qkv * 4);
      g.Tensor(p + "ssm_dt.bias", rank);
      g.Tensor(p + "ssm_alpha.weight", rank * hidden);
      g.Tensor(p + "ssm_beta.weight", rank * hidden);
      g.Tensor(p + "ssm_norm.weight", 128);
      g.Tensor(p + "ssm_out.weight", hidden * inner);
    }
  }
  g.Write(path);
}
void WriteA3b(const fs::path& path, std::uint32_t nextn = 0) {
  Gguf g;
  g.Str("general.architecture", "qwen35moe");
  g.U32("qwen35moe.block_count", 8 + nextn);
  g.U32("qwen35moe.embedding_length", 2048);
  g.U32("qwen35moe.attention.head_count", 16);
  g.U32("qwen35moe.attention.head_count_kv", 2);
  g.U32("qwen35moe.attention.key_length", 256);
  g.U32("qwen35moe.rope.dimension_count", 64);
  g.F32("qwen35moe.rope.freq_base", 10000000.0F);
  g.F32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F);
  g.U32("qwen35moe.full_attention_interval", 4);
  g.U32("qwen35moe.ssm.conv_kernel", 4);
  g.U32("qwen35moe.ssm.state_size", 128);
  g.U32("qwen35moe.ssm.group_count", 16);
  g.U32("qwen35moe.ssm.time_step_rank", 32);
  g.U32("qwen35moe.expert_count", 256);
  g.U32("qwen35moe.expert_used_count", 8);
  g.U32("qwen35moe.expert_feed_forward_length", 512);
  g.U32("qwen35moe.expert_shared_feed_forward_length", 512);
  g.U32("qwen35moe.context_length", 262144);
  if (nextn)
    g.U32("qwen35moe.nextn_predict_layers", nextn);
  g.Write(path);
}
void WriteFlashNext(const fs::path& path, bool broken = false) {
  Gguf g;
  g.Str("general.architecture", "qwen4exp");
  g.U32("qwen4exp.block_count", 48);
  g.U32("qwen4exp.embedding_length", 2560);
  g.U32("qwen4exp.context_length", 262144);
  g.U32("qwen4exp.hyper_connection.count", 4);
  g.U32("qwen4exp.hyper_connection.low_rank", 320);
  g.U32("qwen4exp.full_attention_interval", 4);
  g.U32("qwen4exp.attention.head_count", 24);
  g.U32("qwen4exp.attention.head_count_kv", 2);
  g.U32("qwen4exp.attention.key_length", broken ? 128 : 256);
  g.U32("qwen4exp.rope.dimension_count", 64);
  g.F32("qwen4exp.rope.freq_base", 10000000.0F);
  g.F32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F);
  g.U32Array("qwen4exp.rope.dimension_sections", {11, 11, 10, 0});
  g.U32("qwen4exp.attention.indexer.head_count", 4);
  g.U32("qwen4exp.attention.indexer.key_length", 128);
  g.U32("qwen4exp.attention.indexer.top_k", 2048);
  std::vector<std::uint32_t> ratios;
  for (std::uint32_t i = 0; i < 48; ++i)
    ratios.push_back((i + 1) % 4 == 0 ? 4 : 0);
  g.U32Array("qwen4exp.attention.compress_ratios", ratios);
  g.U32("qwen4exp.ssm.conv_kernel", 4);
  g.U32("qwen4exp.ssm.state_size", 128);
  g.U32("qwen4exp.ssm.group_count", 16);
  g.U32("qwen4exp.ssm.time_step_rank", 48);
  g.U32("qwen4exp.ssm.inner_size", 6144);
  g.U32("qwen4exp.expert_count", 512);
  g.U32("qwen4exp.expert_used_count", 10);
  g.U32("qwen4exp.expert_feed_forward_length", 640);
  g.U32("qwen4exp.expert_shared_feed_forward_length", 640);
  g.Write(path);
}
// A two-layer dense gemma4 header that gemma4::Config::FromGguf accepts:
// one sliding and one global layer.
void WriteGemma4(const fs::path& path) {
  Gguf g;
  g.Str("general.architecture", "gemma4");
  g.Str("general.name", "Synthetic gemma4");
  g.U32("gemma4.block_count", 2);
  g.U32("gemma4.embedding_length", 256);
  g.U32("gemma4.context_length", 131072);
  g.F32("gemma4.attention.layer_norm_rms_epsilon", 1e-6F);
  g.U32("gemma4.attention.head_count", 4);
  g.U32("gemma4.feed_forward_length", 512);
  g.U32("gemma4.attention.head_count_kv", 2);
  g.U32Array("gemma4.attention.sliding_window_pattern", {1, 0});
  g.U32("gemma4.attention.key_length", 256);
  g.U32("gemma4.attention.key_length_swa", 128);
  g.U32("gemma4.attention.value_length", 256);
  g.U32("gemma4.attention.value_length_swa", 128);
  g.U32("gemma4.rope.dimension_count", 256);
  g.U32("gemma4.rope.dimension_count_swa", 128);
  g.F32("gemma4.rope.freq_base", 1000000.0F);
  g.F32("gemma4.rope.freq_base_swa", 10000.0F);
  g.U32("gemma4.attention.sliding_window", 512);
  g.Write(path);
}
void WriteBare(const fs::path& path, const std::string& arch,
               bool with_arch = true) {
  Gguf g;
  if (with_arch) {
    g.Str("general.architecture", arch);
    g.U32(arch + ".block_count", 30);
    g.U32(arch + ".context_length", 131072);
  }
  g.Str("general.name", "Synthetic " + arch);
  g.Write(path);
}

fs::path MakeBase() {
  const auto base =
      fs::temp_directory_path() /
      ("bubo-models-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(base);
  return base;
}
void Touch(const fs::path& path, const std::string& text = "x") {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

bool SamePath(const std::string& a, const fs::path& b) {
  return !a.empty() && fs::absolute(fs::u8path(a)).lexically_normal() ==
                           fs::absolute(b).lexically_normal();
}
// An imported package whose id the catalogue does not know: a tiny dense model
// and a clip file, both listed in the marker with their real sizes. Returns
// the marker path.
fs::path WriteCustomPackage(const fs::path& folder,
                            bool with_projector = true) {
  fs::create_directories(folder);
  WriteDense(folder / "custom.gguf");
  auto files = Value::array();
  auto artefacts = Value::object();
  const auto record = [&](const std::string& name) {
    auto row = Value::object();
    row["file"] = name;
    row["size"] = static_cast<std::uint64_t>(fs::file_size(folder / name));
    row["sha256"] = std::string(64, 'b');
    return row;
  };
  files.push_back(record("custom.gguf"));
  if (with_projector) {
    WriteBare(folder / "mmproj-BF16.gguf", "clip");
    artefacts["mmproj-BF16.gguf"] = record("mmproj-BF16.gguf");
  }
  auto source = Value::object();
  source["repository"] = "fixture/custom";
  source["revision"] = std::string(40, '2');
  auto model = Value::object();
  model["family"] = "qwen35";
  model["source"] = std::move(source);
  model["files"] = std::move(files);
  auto marker = Value::object();
  marker["schema_version"] = 1;
  marker["package_id"] = "custom-pkg";
  marker["engine_min"] = "0.1.0";
  marker["model"] = std::move(model);
  marker["artefacts"] = std::move(artefacts);
  std::ofstream(folder / "bubo-package.json", std::ios::binary)
      << marker.dump() << "\n";
  return folder / "bubo-package.json";
}

// ---- scanner ---------------------------------------------------------------
bool HasFile(const ScanResult& result, const fs::path& path) {
  return std::find(result.files.begin(), result.files.end(), path) !=
         result.files.end();
}
bool HasWarning(const ScanResult& result, std::string_view needle) {
  return std::any_of(result.warnings.begin(), result.warnings.end(),
                     [&](const ScanWarning& w) {
                       return Contains(w.reason, needle) ||
                              Contains(w.folder, needle);
                     });
}
void Scanner(const fs::path& base) {
  const auto root = base / "scan";
  Touch(root / "a.gguf");
  Touch(root / "d1" / "b.gguf");
  Touch(root / "d1" / "d2" / "c.gguf");
  Touch(root / "d1" / "d2" / "d3" / "d.gguf");         // depth 3: found
  Touch(root / "d1" / "d2" / "d3" / "d4" / "e.gguf");  // depth 4: not found
  Touch(root / "half.gguf.partial");
  Touch(root / "notes.txt");
  Touch(root / "UPPER.GGUF");
  ScanOptions options;
  options.folders = {root};
  const auto result = ScanModelFolders(options);
  Require(HasFile(result, root / "a.gguf") &&
              HasFile(result, root / "d1" / "b.gguf") &&
              HasFile(result, root / "d1" / "d2" / "c.gguf") &&
              HasFile(result, root / "d1" / "d2" / "d3" / "d.gguf") &&
              HasFile(result, root / "UPPER.GGUF"),
          "files down to depth 3 are found (case-insensitive extension)");
  Require(!HasFile(result, root / "d1" / "d2" / "d3" / "d4" / "e.gguf"),
          "depth 4 is not scanned");
  Require(HasWarning(result, "deeper than 3 levels"),
          "the depth limit is stated in a warning");
  Require(result.files.size() == 5, "partial downloads and text are skipped");
  Require(result.scanned_folders.size() == 1, "one scanned folder");

  // Depth limit is a parameter, not a constant of the loop.
  ScanOptions shallow = options;
  shallow.max_depth = 1;
  const auto one = ScanModelFolders(shallow);
  Require(HasFile(one, root / "d1" / "b.gguf") &&
              !HasFile(one, root / "d1" / "d2" / "c.gguf"),
          "max_depth 1");
  // File limit.
  ScanOptions limited = options;
  limited.max_files = 2;
  const auto few = ScanModelFolders(limited);
  Require(few.files.size() == 2 && HasWarning(few, "stopped after 2 files"),
          "file limit");

  // A folder named twice is scanned once.
  ScanOptions twice;
  twice.folders = {root, root, root / "d1"};
  const auto dup = ScanModelFolders(twice);
  Require(std::count(dup.files.begin(), dup.files.end(),
                     root / "d1" / "b.gguf") == 1,
          "overlapping folders list a file once");

  // Missing folder and empty name.
  ScanOptions missing;
  missing.folders = {base / "does-not-exist", fs::path{}};
  const auto none = ScanModelFolders(missing);
  Require(none.files.empty() && none.warnings.size() == 2 &&
              none.scanned_folders.empty(),
          "missing and empty folders give warnings");

  // Never a drive or share root.
  Require(IsFilesystemRoot("C:\\") && IsFilesystemRoot("C:") &&
              IsFilesystemRoot("D:/") &&
              IsFilesystemRoot("\\\\server\\share") &&
              IsFilesystemRoot("\\\\server\\share\\"),
          "roots are detected");
  Require(!IsFilesystemRoot("C:\\Users") && !IsFilesystemRoot(root) &&
              !IsFilesystemRoot("\\\\server\\share\\models") &&
              !IsFilesystemRoot("") && !IsFilesystemRoot("D:\\models\\"),
          "non-roots are accepted");
  // Only a symbolic link or a junction is a link. Other reparse points are
  // ordinary folders and files: a OneDrive placeholder (cloud files), a
  // compressed or deduplicated file, an app-execution alias.
  Require(IsLinkReparseTag(0xA000000CU) && IsLinkReparseTag(0xA0000003U),
          "a symbolic link and a mount point are links");
  Require(!IsLinkReparseTag(0x9000001AU) && !IsLinkReparseTag(0x9000701AU) &&
              !IsLinkReparseTag(0x9000F01AU) &&
              !IsLinkReparseTag(0x80000017U) &&
              !IsLinkReparseTag(0x80000013U) &&
              !IsLinkReparseTag(0x8000001BU) && !IsLinkReparseTag(0),
          "cloud-file, WOF, dedup, app-exec and no tag are not links");
  ScanOptions drives;
  drives.folders = {"C:\\", "C:", "D:\\", "\\\\server\\share"};
  const auto refused = ScanModelFolders(drives);
  Require(refused.files.empty() && refused.scanned_folders.empty() &&
              refused.warnings.size() == 4 &&
              HasWarning(refused, "never scanned"),
          "a drive or share root is refused and nothing is read");

  // Junction loop: root\loop\back -> root. The scan must finish, list each
  // file once and not follow the link.
  const auto loop_root = base / "loops";
  Touch(loop_root / "m.gguf");
  Touch(loop_root / "sub" / "n.gguf");
  fs::create_directories(loop_root / "sub");
  const auto link = loop_root / "sub" / "back";
  const std::string command = "cmd /c mklink /J \"" + link.string() + "\" \"" +
                              loop_root.string() + "\" > NUL 2>&1";
  Require(std::system(command.c_str()) == 0 && fs::exists(link),
          "test junction created");
  ScanOptions looped;
  looped.folders = {loop_root};
  const auto cycle = ScanModelFolders(looped);
  Require(cycle.files.size() == 2 && HasFile(cycle, loop_root / "m.gguf") &&
              HasFile(cycle, loop_root / "sub" / "n.gguf") &&
              HasWarning(cycle, "not followed"),
          "a junction loop is not followed");
  // A scan folder that is itself a junction is refused.
  ScanOptions via_link;
  via_link.folders = {link};
  const auto linked = ScanModelFolders(via_link);
  Require(linked.files.empty() && HasWarning(linked, "link or junction"),
          "a scan folder that is a junction is refused");
  // Remove the junction itself, never its target.
  std::error_code ec;
  fs::remove(link, ec);
  Require(fs::exists(loop_root / "m.gguf"), "target survives link removal");
}

// ---- header inspection and architecture support ----------------------------
void Support(const fs::path& base) {
  const auto dir = base / "gguf";
  fs::create_directories(dir);
  WriteBare(dir / "gemma.gguf", "gemma4");
  auto gemma = InspectGguf(dir / "gemma.gguf");
  Require(gemma.facts.header_ok && gemma.facts.architecture == "gemma4" &&
              gemma.facts.model_name == "Synthetic gemma4" &&
              gemma.facts.native_context == 131072 &&
              gemma.facts.size_bytes == fs::file_size(dir / "gemma.gguf"),
          "header facts: architecture, name, size, context");
  // Serve runs gemma4 files; the engine's config check judges this one.
  Require(!gemma.support.runnable && !gemma.support.sidecar &&
              Contains(gemma.support.reason, "cannot load this gemma4 file"),
          "a gemma4 header the engine refuses names the reason: " +
              gemma.support.reason);
  WriteGemma4(dir / "gemma-ok.gguf");
  const auto gemma_ok = InspectGguf(dir / "gemma-ok.gguf");
  Require(gemma_ok.support.runnable && gemma_ok.support.route == "gemma4" &&
              gemma_ok.support.reason.empty(),
          "a complete gemma4 header is runnable: " + gemma_ok.support.reason);
  {
    EstimateRequest request;
    const auto estimate = EstimateForGguf(gemma_ok, request);
    Require(estimate.state == "none_yet" &&
                Contains(estimate.reason, "no memory estimate") &&
                !Contains(estimate.reason, "no loader"),
            "Gemma 4: runnable, no estimate yet: " + estimate.reason);
  }
  WriteBare(dir / "diffusion.gguf", "diffusion_gemma");
  Require(!InspectGguf(dir / "diffusion.gguf").support.runnable &&
              Contains(InspectGguf(dir / "diffusion.gguf").support.reason,
                       "safetensors folder"),
          "DiffusionGemma is served from its folder, not a GGUF");
  WriteBare(dir / "noarch.gguf", "", false);
  const auto noarch = InspectGguf(dir / "noarch.gguf");
  Require(noarch.facts.header_ok && !noarch.support.runnable &&
              Contains(noarch.support.reason, "general.architecture"),
          "a file with no architecture key");
  Touch(dir / "junk.gguf", "this is not a gguf file at all, just text");
  const auto junk = InspectGguf(dir / "junk.gguf");
  Require(!junk.facts.header_ok && !junk.support.runnable &&
              Contains(junk.support.reason, "header cannot be read"),
          "garbage is not a readable header");
  WriteBare(dir / "clip.gguf", "clip");
  const auto clip = InspectGguf(dir / "clip.gguf");
  Require(clip.support.sidecar && !clip.support.runnable &&
              Contains(clip.support.reason, "vision projector"),
          "a projector is a side file");
  WriteDense(dir / "dense.gguf");
  const auto dense = InspectGguf(dir / "dense.gguf");
  Require(dense.support.runnable && dense.support.route == "qwen" &&
              dense.support.reason.empty(),
          "tiny dense Qwen is runnable by the engine's loader: " +
              dense.support.reason);
  DenseShape no_q;
  no_q.drop_tensor = true;
  WriteDense(dir / "dense-bad.gguf", no_q);
  const auto bad = InspectGguf(dir / "dense-bad.gguf");
  Require(
      !bad.support.runnable &&
          Contains(bad.support.reason, "Unexpected Qwen layer kind"),
      "a Qwen file with a missing tensor is refused: " + bad.support.reason);
  WriteA3b(dir / "a3b.gguf");
  const auto a3b = InspectGguf(dir / "a3b.gguf");
  if (a3b.support.runnable)
    Require(a3b.support.route == "qwen36_a3b", "A3B route");
  else
    Require(Contains(a3b.support.reason, "without"), "A3B off in this build");
  WriteFlashNext(dir / "fn.gguf");
  const auto fn = InspectGguf(dir / "fn.gguf");
  Require(fn.support.runnable && fn.support.route == "qwen38_flash_next",
          "Flash Next trunk header: " + fn.support.reason);
  WriteFlashNext(dir / "fn-bad.gguf", true);
  const auto fn_bad = InspectGguf(dir / "fn-bad.gguf");
  Require(!fn_bad.support.runnable &&
              Contains(fn_bad.support.reason, "cannot load"),
          "a Flash Next file the loader rejects");
  WriteBare(dir / "ds.gguf", "deepseek4");
  Require(InspectGguf(dir / "ds.gguf").support.route == "deepseek_v4_flash",
          "DeepSeek route");

  // Split files: shard 2 is listed with shard 1; sizes add up.
  for (unsigned i = 0; i < 2; ++i) {
    Gguf g;
    g.Str("general.architecture", "gemma4");
    g.U32("split.count", 2);
    g.U32("split.no", i);
    char name[64];
    std::snprintf(name, sizeof(name), "big-%05u-of-00002.gguf", i + 1);
    g.Write(dir / name);
  }
  const auto first = InspectGguf(dir / "big-00001-of-00002.gguf");
  const auto second = InspectGguf(dir / "big-00002-of-00002.gguf");
  Require(!first.facts.split_follower && second.facts.split_follower &&
              first.facts.shards_found == 2 &&
              first.facts.shards_expected == 2 &&
              first.facts.total_bytes ==
                  fs::file_size(dir / "big-00001-of-00002.gguf") +
                      fs::file_size(dir / "big-00002-of-00002.gguf"),
          "split shards");
  fs::remove(dir / "big-00002-of-00002.gguf");
  Require(InspectGguf(dir / "big-00001-of-00002.gguf").facts.shards_found == 1,
          "a missing shard is counted");
}

// ---- estimate --------------------------------------------------------------
std::uint64_t TotalMin(const Value& estimate) {
  return estimate.find("total_range")->find("min")->member_size("bytes");
}
std::uint64_t PickFor(const Value& estimate, std::uint64_t sessions) {
  for (const auto& row :
       estimate.find("recommended_context")->find("by_sessions")->items())
    if (row.member_size("sessions") == sessions)
      return row.member_size("context_per_session");
  return 0;
}

gufo::server::ReleaseStartup InspectSettings(std::vector<std::string> args) {
  std::vector<const char*> raw;
  for (const auto& arg : args)
    raw.push_back(arg.c_str());
  auto startup = gufo::server::PrepareReleaseSettings(raw, {}, true);
  Require(startup.error.empty(),
          "inspect-only release settings: " + startup.error);
  return startup;
}

void EstimateConsistency(const fs::path& base) {
  using namespace gufo::server;
  const auto start = checks;
  // Independent fixed-budget expectations for the tiny header fixtures. These
  // pin selection as well as parity, so common estimator bugs cannot pass by
  // comparing the two callers to one another.
  constexpr std::uint64_t flash_picks[3][4] = {
      {4096, 4096, 4096, 4096},
      {262144, 139264, 90112, 65536},
      {262144, 262144, 245760, 180224}};
  constexpr std::uint64_t dense_picks[3][4] = {
      {262144, 262144, 262144, 221184},
      {262144, 262144, 262144, 262144},
      {262144, 262144, 262144, 262144}};
  for (const auto* file : {"fn.gguf", "dense.gguf"}) {
    const auto path = base / "gguf" / file;
    const auto inspection = InspectGguf(path);
    for (const auto budget : {12ULL, 24ULL, 36ULL}) {
      for (const auto sessions : {1ULL, 2ULL, 3ULL, 4ULL}) {
        EstimateRequest request;
        request.sessions = sessions;
        request.available_bytes = budget * kGiB;
        request.host_available_bytes = 128 * kGiB;
        request.capacity_budgets.gpu_total = budget * kGiB;
        request.capacity_budgets.host_total = 128 * kGiB;
        const auto listed =
            EstimateForGguf(inspection, request).memory_estimate;
        auto startup = InspectSettings(
            {"--model", path.string(), "--sessions", std::to_string(sessions)});
        auto table = *MemoryTableFor(prepare::kPackageId);
        table.main_weights_bytes = inspection.facts.total_bytes;
        // This is serve's package path for Flash Next and its metadata-driven
        // fallback for dense Qwen. Both feed the actual context resolver.
        const auto served =
            std::string_view(file) == "fn.gguf"
                ? BuildReleaseMemoryEstimate(
                      startup.resolved, std::string(prepare::kPackageId),
                      request.available_bytes, request.host_available_bytes,
                      &table, request.capacity_budgets)
                      .value
                : EstimateForGguf(inspection, request).memory_estimate;
        Require(
            listed.find("recommended_context")->find("by_sessions")->dump() ==
                served.find("recommended_context")->find("by_sessions")->dump(),
            "models estimate and serve resolver use identical rows: " +
                std::string(file) + " budget=" + std::to_string(budget) +
                " sessions=" + std::to_string(sessions) + " listed=" +
                listed.find("recommended_context")
                    ->find("by_sessions")
                    ->items()
                    .front()
                    .dump() +
                " served=" +
                served.find("recommended_context")
                    ->find("by_sessions")
                    ->items()
                    .front()
                    .dump());
        const auto pick = PickFor(listed, sessions);
        if (std::getenv("MODELS_TEST_PRINT_EXPECTED")) {
          std::cout << "PICK file=" << file << " budget=" << budget
                    << " sessions=" << sessions << " actual=" << pick << "\n";
          continue;
        }
        Require(pick == (std::string_view(file) == "fn.gguf"
                             ? flash_picks
                             : dense_picks)[budget / 12 - 1][sessions - 1],
                "fixed-budget context matches independent expected row: " +
                    std::string(file) + " budget=" + std::to_string(budget) +
                    " sessions=" + std::to_string(sessions) +
                    " actual=" + std::to_string(pick) + " expected=" +
                    std::to_string(
                        (std::string_view(file) == "fn.gguf"
                             ? flash_picks
                             : dense_picks)[budget / 12 - 1][sessions - 1]));
        const auto& row = served.find("recommended_context")
                              ->find("by_sessions")
                              ->items()
                              .at(sessions - 1);
        if (sessions == 1) {
          Require(
              !ResolveDefaultSessionContext(startup.resolved, sessions, served),
              "single-session native default is preserved");
        } else if (row.find("recommended")->as_bool()) {
          Require(ResolveDefaultSessionContext(startup.resolved, sessions,
                                               served) == pick,
                  "serve selects the listed recommended row");
          const auto settings = ReleaseSettingsJson(startup.resolved);
          for (const auto& setting :
               settings.find("effective_switches")->items())
            if (setting.member_str("name") == "--context")
              Require(setting.member_str("source") == "default",
                      "recommended_context keeps the contract source default");
        } else {
          bool rejected = false;
          try {
            (void)ResolveDefaultSessionContext(startup.resolved, sessions,
                                               served);
          } catch (const std::exception&) {
            rejected = true;
          }
          Require(rejected,
                  "unrecommended rows cannot silently select a context");
        }
      }
    }
    EstimateRequest request;
    request.context = 8192;
    request.sessions = 4;
    request.available_bytes = 36 * kGiB;
    request.host_available_bytes = 128 * kGiB;
    const auto listed = EstimateForGguf(inspection, request).memory_estimate;
    auto startup = InspectSettings(
        {"--model", path.string(), "--sessions", "4", "--context", "8192"});
    Require(!ResolveDefaultSessionContext(startup.resolved, 4, listed) &&
                startup.resolved.effective_flags.at("--context") == "8192",
            "explicit context is preserved");
  }
  std::cout << "EstimateConsistency: " << checks - start << " checks passed\n";
}

void MeasuredAdmission(const fs::path& data_path,
                       const fs::path& manifest = {}) {
  using namespace gufo::server;
  const auto start = checks;
  const auto data =
      prepare::metadata::Parse(prepare::metadata::Read(data_path));
  for (const auto& record : data.find("snapshot_capacity_records")->items()) {
    const auto plan = PlanSnapshotCapacity(
        {.host_total_bytes = record.member_size("host_total_bytes"),
         .host_resident_bytes = record.member_size("host_resident_bytes"),
         .model_footprint_bytes = record.member_size("model_footprint_bytes"),
         .reserve_bytes = record.member_size("reserve_bytes"),
         .device_total_bytes = record.member_size("device_total_bytes"),
         .explicit_bytes = record.member_size("explicit_bytes")});
    Require(plan.ceiling_bytes == record.member_size("ceiling_bytes"),
            "byte-exact logged engine ceiling");
    Require(plan.automatic_bytes == record.member_size("automatic_bytes"),
            "byte-exact logged engine automatic capacity");
    Require(plan.capacity_bytes == record.member_size("capacity_bytes"),
            "byte-exact logged engine explicit capacity");
    Require(plan.limit == record.member_str("limit"),
            "logged engine limiting side");
    Require((plan.clamp.empty() ? "none" : plan.clamp) ==
                record.member_str("clamp"),
            "logged engine clamp label");
    std::cout << "ByteExactCapacity " << record.member_str("source_path")
              << ": ceiling=" << plan.ceiling_bytes
              << " automatic=" << plan.automatic_bytes
              << " capacity=" << plan.capacity_bytes
              << " clamp=" << record.member_str("clamp") << "\n";
  }
  for (const auto& point : data.find("points")->items()) {
    const auto context = point.member_size("context");
    const auto sessions = point.member_size("sessions");



    struct ImageCap {
      explicit ImageCap(std::uint64_t tokens) {
#ifdef _WIN32
        _putenv_s("GUFO_MAX_IMAGE_TOKENS", std::to_string(tokens).c_str());
#else
        setenv("GUFO_MAX_IMAGE_TOKENS", std::to_string(tokens).c_str(), 1);
#endif
      }
      ~ImageCap() {
#ifdef _WIN32
        _putenv_s("GUFO_MAX_IMAGE_TOKENS", "");
#else
        unsetenv("GUFO_MAX_IMAGE_TOKENS");
#endif
      }
    } image_cap(context);
    ReleaseResolution resolved;
    if (!manifest.empty()) {
      auto startup =
          InspectSettings({"--manifest", manifest.string(), "--profile",
                           "default", "--sessions", std::to_string(sessions),
                           "--context", std::to_string(context)});
      Require(startup.package.package_id == data.member_str("package_id"),
              "measured package identity matches the retained observations");
      Require(
          startup.resolved.native_context ==
                  data.member_size("native_context") &&
              startup.resolved.architecture == data.member_str("architecture"),
          "live header matches recorded estimate inputs");
      resolved = std::move(startup.resolved);
    } else {
      // Independent recorded facts from the real package/header. No file-size
      // approximation, engine load, GPU operation or runtime is required.
      ReleaseInputs inputs;
      inputs.architecture = data.member_str("architecture");
      inputs.model_family = data.member_str("model_family");
      inputs.native_context = data.member_size("native_context");
      inputs.manifest_present = true;
      inputs.explicit_flags = {{"--sessions", std::to_string(sessions)},
                               {"--context", std::to_string(context)}};
      inputs.environment["GUFO_MAX_IMAGE_TOKENS"] = std::to_string(context);
      const auto& artefacts = *data.find("artefacts");
      inputs.manifest_defaults = {
          {"--mtp-model", artefacts.member_str("mtp")},
          {"--mmproj", artefacts.member_str("mmproj")},
          {"GUFO_DENSE_TWIN_MODEL", artefacts.member_str("twin")},
          {"GUFO_DRAFT_OUT_MODEL", artefacts.member_str("draft_head")}};
      resolved = ResolveReleaseSettings(inputs);
    }
    PlanningMemoryBudgets budgets;
    budgets.gpu_total = data.member_size("gpu_total_bytes");
    budgets.host_total = data.member_size("host_total_bytes");
    budgets.gpu_dedicated = data.member_size("dedicated_video_memory_bytes");
    const auto planning = BuildReleaseMemoryEstimate(
        resolved, data.member_str("package_id"),
        data.member_size("gpu_available_bytes"),
        data.member_size("host_available_bytes"), nullptr, budgets);
    Require(!planning.value.is_null(), "measured-point estimate is available");
    const auto actual = TotalMin(planning.value);
    const auto measured =
        (point.member_size("device_used_mib") +
         sessions * point.member_size("vision_reserved_mib_per_session")) *
        1048576ULL;
    Require(actual == point.member_size("expected_admission_bytes"),
            "independent retained admission total");
    Require(
        actual >= measured &&
            actual - measured <= data.member_size("maximum_overestimate_bytes"),
        "admission stays on the safe side within 3 GiB of measured use");
    Require(
        planning.inputs.ram_cache_bytes == 8589934592ULL &&
            planning.inputs.cache_retention_lazy &&
            planning.value.find("device_cache_snapshot_allowance")
                    ->member_size("bytes") == 0,
        "explicit profile ceiling is retained without a startup allocation");
    // R1 (the engine's startup admission) is still what the retained estimate
    // matches; R2 adds the spill rule to `fits`: the part of the need beyond
    // the dedicated carve-out must fit the RAM above its floor together with
    // the host need. A point that runs on the machine but only on that margin
    // is not recommended and says why; serve still starts it (load_spill
    // warning).
    {
      const bool expected_fits = point.find("expected_fits") == nullptr ||
                                 point.find("expected_fits")->as_bool();
      Require(planning.value.find("fits")->as_bool() == expected_fits,
              "R1/R2 planner fits verdict for the measured point");
      if (const auto* needle = point.find("expected_fit_reason_contains")) {
        const auto* reason = planning.value.find("fit_reason");
        Require(reason != nullptr && reason->is_string() &&
                    reason->str().find(needle->str()) != std::string::npos,
                "a not-fitting point carries the spill reason");
      }
    }
    if (std::getenv("MODELS_TEST_PRINT_EXPECTED")) {
      std::cout << "ADMISSION_POINT sessions=" << sessions
                << " context=" << context << " cache="
                << planning.value.member_size("cache_at_clamp_bytes")
                << " spill=" << planning.value.member_size("host_spill_bytes")
                << " recommended="
                << planning.value.find("recommended")->as_bool() << " failures="
                << planning.value.find("recommendation_failures")->dump()
                << " fits=" << planning.value.find("fits")->as_bool() << "\n";
      continue;
    }
    Require(
        planning.value.member_size("cache_at_clamp_bytes") ==
            point.member_size("expected_cache_at_clamp_bytes"),
        "R2 independent cache-at-clamp bytes: actual " +
            std::to_string(planning.value.member_size("cache_at_clamp_bytes")) +
            " expected " +
            std::to_string(point.member_size("expected_cache_at_clamp_bytes")) +
            " (sessions " + std::to_string(sessions) + " context " +
            std::to_string(context) + ")");
    Require(planning.value.member_size("host_spill_bytes") ==
                point.member_size("expected_host_spill_bytes"),
            "R3 independent host spill bytes: actual " +
                std::to_string(planning.value.member_size("host_spill_bytes")) +
                " expected " +
                std::to_string(point.member_size("expected_host_spill_bytes")) +
                " (sessions " + std::to_string(sessions) + " context " +
                std::to_string(context) + ")");
    Require(planning.value.find("recommended")->as_bool() ==
                point.find("expected_recommended")->as_bool(),
            "R2/R3 measured setup verdict");
    Require(planning.value.find("recommendation_failures")->dump() ==
                point.find("expected_recommendation_failures")->dump(),
            "R2/R3 independent measured setup failure terms");
    Require(PickFor(planning.value, 2) < 262144,
            "lazy growth lowers the two-session recommendation");
    std::cout << "Recommendation " << sessions << " x " << context
              << " fits=" << planning.value.find("fits")->as_bool()
              << " recommended="
              << planning.value.find("recommended")->as_bool() << " failures="
              << planning.value.find("recommendation_failures")->dump()
              << " cache=" << planning.value.member_size("cache_at_clamp_bytes")
              << " spill=" << planning.value.member_size("host_spill_bytes")
              << " reason="
              << planning.value.member_str("recommendation_reason") << "\n";
    std::cout << "MeasuredAdmission " << sessions << " x " << context
              << ": estimate=" << BytesToGiB(actual)
              << " GiB measured=" << BytesToGiB(measured)
              << " GiB excess=" << BytesToGiB(actual - measured) << " GiB\n";
  }
  std::cout << "MeasuredAdmission: " << checks - start << " checks passed\n";
}
void Estimates(const fs::path& base) {
  const auto dir = base / "gguf";
  EstimateRequest request;
  request.available_bytes = 12 * kGiB;
  request.host_available_bytes = 128 * kGiB;

  // Dense Qwen, from the header, through the engine's state-size function.
  const auto dense = InspectGguf(dir / "dense.gguf");
  auto result = EstimateForGguf(dense, request);
  Require(result.state == "estimated" && result.reason.empty() &&
              result.evidence_label == "estimated from the file header",
          "dense estimate: " + result.reason);
  auto estimate = result.memory_estimate;
  Require(estimate.member_str("evidence") == "estimated_from_header" &&
              !estimate.find("fit_verified")->as_bool() &&
              result.state == "estimated",
          "dense evidence label and fit_verified false");
  std::string error;
  const auto config = dense.reader->ExtractModelConfig(&error);
  Require(config.has_value(), "config " + error);
  const auto expected_state = [&](std::uint64_t context) {
    return gufo::core::QwenDenseSessionStateBytes(*config, context, 2, 4);
  };
  // Defaults match serve: native context, one session.
  Require(estimate.member_size("context_tokens") == 262144 &&
              estimate.member_size("sessions") == 1,
          "estimate defaults");
  const auto state = estimate.find("per_session_state")->member_size("bytes");
  Require(state >= expected_state(262144) &&
              state <= expected_state(262144) + 2 * 1048576,
          "per-session state equals the engine function (within MiB rounding)");
  Require(
      estimate.find("weights")->member_size("bytes") == dense.facts.total_bytes,
      "weights are the file size");
  Require(estimate.find("scratch_phase_allowance")
                  ->find("min")
                  ->member_str("evidence") == "unknown",
          "scratch is not claimed");
  // Monotonic in context and in sessions.
  std::uint64_t previous = 0;
  for (std::uint64_t context : {2048ULL, 8192ULL, 32768ULL, 131072ULL}) {
    auto r = request;
    r.context = context;
    const auto e = EstimateForGguf(dense, r).memory_estimate;
    Require(TotalMin(e) > previous, "total grows with context");
    previous = TotalMin(e);
  }
  previous = 0;
  for (std::uint64_t sessions : {1ULL, 2ULL, 4ULL}) {
    auto r = request;
    r.sessions = sessions;
    const auto e = EstimateForGguf(dense, r).memory_estimate;
    Require(TotalMin(e) > previous, "total grows with sessions");
    previous = TotalMin(e);
  }
  // Recommended context: at most the model's context, never grows with more
  // sessions, grows with available memory.
  const auto rec = [&](std::uint64_t available) {
    auto r = request;
    r.available_bytes = available;
    return EstimateForGguf(dense, r).memory_estimate;
  };
  const auto small = rec(8 * kGiB), large = rec(64 * kGiB);
  Require(PickFor(large, 1) == 262144 && PickFor(small, 1) == 262144,
          "one session fits the full context");
  Require(
      PickFor(small, 2) < PickFor(small, 1) && PickFor(small, 2) % 8192 == 0,
      "two sessions get a smaller context on a small machine");
  for (std::uint64_t s = 2; s <= 8; ++s)
    Require(PickFor(small, s) <= PickFor(small, s - 1),
            "context per session never grows with sessions");
  for (std::uint64_t s = 1; s <= 8; ++s)
    Require(PickFor(large, s) >= PickFor(small, s),
            "more memory never lowers the context");
  Require(
      estimate.find("recommended_context")->member_size("max_sessions") == 8,
      "dense runs up to 8 sessions in the estimate");
  // No available memory: null plus reason, still an estimate.
  {
    auto r = request;
    r.available_bytes.reset();
    const auto e = EstimateForGguf(dense, r);
    Require(
        e.state == "estimated" &&
            e.memory_estimate.find("recommended_context")->is_null() &&
            !e.memory_estimate.find("recommended_context_reason")->is_null(),
        "no available memory gives a null recommendation with a reason");
  }
  // Out-of-range context and sessions.
  {
    auto r = request;
    r.context = 999999999;
    Require(EstimateForGguf(dense, r).state == "none_yet", "context too large");
  }

  // A different shape changes the answer: this is per-architecture, not a
  // fixed number. Two layers' worth of KV vs eight kv heads.
  DenseShape narrow;
  narrow.kv_heads = 1;
  narrow.heads = 2;
  narrow.head_dim = 32;
  WriteDense(dir / "dense-narrow.gguf", narrow);
  const auto narrow_result =
      EstimateForGguf(InspectGguf(dir / "dense-narrow.gguf"), request);
  Require(narrow_result.state == "estimated" &&
              narrow_result.memory_estimate.find("per_session_state")
                      ->member_size("bytes") < state,
          "a narrower cache needs less state");

  // A3B: recurrent layers hold fixed state, attention layers grow with context.
  const auto a3b = InspectGguf(dir / "a3b.gguf");
  if (a3b.support.runnable) {
    gufo::models::qwen36_a3b::Config c;
    std::string a3b_error;
    Require(gufo::models::qwen36_a3b::ReadConfig(*a3b.reader, c, &a3b_error),
            "A3B config " + a3b_error);
    const auto at = [&](std::uint64_t context) {
      return gufo::core::A3bSessionStateBytes(
          c.layers, c.full_attention_interval, c.KvDim(), c.ssm_conv_kernel,
          c.ConvChannels(), c.ssm_v_heads, c.ssm_head_dim, context);
    };
    auto r = request;
    r.context = 65536;
    const auto e = EstimateForGguf(a3b, r);
    Require(e.state == "estimated", "A3B estimate: " + e.reason);
    const auto got =
        e.memory_estimate.find("per_session_state")->member_size("bytes");
    Require(got >= at(65536) && got <= at(65536) + 2 * 1048576,
            "A3B per-session state equals the engine function");
    // 8 layers: 6 recurrent + 2 attention. Slope = 2 layers x 3 planes x
    // kv_dim x 2 bytes per token.
    Require(at(2000) - at(1000) == 2ULL * 3 * (2 * 256) * 2 * 1000,
            "A3B cache slope: only the 2 attention layers grow");
    Require(at(0) > 0, "A3B recurrent layers hold fixed state");
    r.sessions = 2;
    const auto two = EstimateForGguf(a3b, r);
    Require(two.state == "none_yet" && Contains(two.reason, "at most 1"),
            "A3B runs one session");
    Require(std::any_of(e.notes.begin(), e.notes.end(),
                        [](const std::string& n) {
                          return Contains(n, "RAM cache: load-dependent");
                        }),
            "A3B note says the RAM cache is not in the estimate");
    // Nextn blocks are not part of the trunk.
    WriteA3b(dir / "a3b-mtp.gguf", 1);
    const auto mtp = InspectGguf(dir / "a3b-mtp.gguf");
    gufo::models::qwen36_a3b::Config cm;
    Require(gufo::models::qwen36_a3b::ReadConfig(*mtp.reader, cm, &a3b_error) &&
                cm.layers == 8 && cm.nextn == 1,
            "MTP block excluded from the layer count");
  }

  // Flash Next file: the measured coefficient table, weights = file size.
  const auto fn = InspectGguf(dir / "fn.gguf");
  auto fn_request = request;
  fn_request.available_bytes = 128 * kGiB;
  fn_request.weights_bytes = 82500000000ULL;
  const auto fn_result = EstimateForGguf(fn, fn_request);
  Require(fn_result.state == "estimated" &&
              fn_result.memory_estimate.member_str("evidence") == "estimated" &&
              fn_result.memory_estimate.find("weights")->member_size("bytes") ==
                  82500000000ULL &&
              !fn_result.memory_estimate.find("recommended_context")->is_null(),
          "Flash Next header uses the measured table: " + fn_result.reason);

  // No estimate: DeepSeek route, unsupported file, sidecar, garbage.
  const auto none = [&](const char* name, std::string_view needle) {
    const auto r = EstimateForGguf(InspectGguf(dir / name), request);
    Require(r.state == "none_yet" && r.memory_estimate.is_null() &&
                Contains(r.reason, needle),
            std::string(name) + " has no estimate yet: " + r.reason);
    const auto json = EstimateJson(r);
    Require(json.member_str("estimate_state") == "none_yet" &&
                !json.find("reason")->is_null() &&
                json.find("memory_estimate")->is_null() &&
                !json.find("fit_verified")->as_bool(),
            "none_yet JSON shape");
  };
  none("ds.gguf", "Bubo has no memory estimate for this model yet.");
  none("gemma.gguf", "cannot load this gemma4 file");
  none("gemma-ok.gguf", "Bubo has no memory estimate for this model yet.");
  none("clip.gguf", "vision projector");
  none("junk.gguf", "header cannot be read");

  // Folder: DiffusionGemma safetensors.
  const auto folder = base / "diffusion";
  Touch(folder / "config.json", "{\"model_type\": \"diffusion_gemma\"}");
  const auto d = EstimateForFolder(folder, request);
  Require(d.state == "none_yet" && Contains(d.reason, "DiffusionGemma"),
          "DiffusionGemma folder: no estimate yet");
  Touch(base / "other" / "config.json", "{\"model_type\": \"llama\"}");
  Require(Contains(EstimateForFolder(base / "other", request).reason, "llama"),
          "another folder type");
  fs::create_directories(base / "empty");
  Require(EstimateForFolder(base / "empty", request).state == "none_yet",
          "a folder with no config");

  // Catalogue entries.
  const auto& shipped = ShippedCatalogue();
  const auto* lowmemory = FindCatalogueEntry(
      shipped.entries, "huihui-qwen38-flash-next-lowmemory-v1");
  Require(lowmemory != nullptr, "lowmemory catalogue estimate exists");
  const auto lowmemory_result =
      EstimateForCatalogueEntry(*lowmemory, base / "models", request);
  Require(lowmemory_result.state == "estimated" &&
              !lowmemory_result.memory_estimate.is_null(),
          "lowmemory has a calibrated estimate on every machine");
  Require(!lowmemory->measured_main_resident_bytes,
          "lowmemory resident coefficient data slot is unmeasured");
  Require(lowmemory_result.memory_estimate.find("fits")->is_bool() &&
              lowmemory_result.memory_estimate.find("weights")->member_size(
                  "bytes") == 61064419200ULL,
          "lowmemory uses the calibrated main coefficient plus selected side "
          "files while its measured resident slot remains unavailable");
  const auto* flash = FindCatalogueEntry(shipped.entries, prepare::kPackageId);
  auto catalogue_request = request;
  catalogue_request.available_bytes = 100 * kGiB;
  // This case verifies native-context admission, independently of the release
  // default (200,000 for Flash Next). Keep the retained native total exact.
  catalogue_request.context = 262144;
  const auto flash_result =
      EstimateForCatalogueEntry(*flash, base / "models", catalogue_request);
  Require(flash_result.state == "estimated" &&
              !flash_result.memory_estimate.find("recommended_context")
                   ->is_null() &&
              Contains(flash_result.fit_inputs.dump(), "weights_bytes"),
          "Flash Next catalogue estimate needs no file");
  // Keep the explicit native-context case and the release-default case.
  // Vision is reserved at the default 65536-token cap: 674234432 bytes, not
  // the 2687500352 of a cap equal to the 262144-token context.
  Require(
      TotalMin(flash_result.memory_estimate) == 99448815808ULL &&
          flash_result.memory_estimate.member_size("context_tokens") == 262144,
      "catalogue native-context admission excludes the retention ceiling");
  auto default_request = catalogue_request;
  default_request.context = 0;
  const auto default_result =
      EstimateForCatalogueEntry(*flash, base / "models", default_request);
  Require(TotalMin(default_result.memory_estimate) == 97728669888ULL &&
              default_result.memory_estimate.member_size("context_tokens") ==
                  200000,
          "catalogue default-context admission matches the real CLI capture");
  const auto* original =
      FindCatalogueEntry(shipped.entries, prepare::kOriginalPackageId);
  Require(
      EstimateForCatalogueEntry(*original, base / "models", catalogue_request)
              .state == "estimated",
      "Flash Next original set");
  const auto* gemma_entry = FindCatalogueEntry(
      shipped.entries, "gemma4-26b-a4b-huihui-abliterated-qat-q4-k");
  const auto g = EstimateForCatalogueEntry(*gemma_entry, base / "models",
                                           catalogue_request);
  // The engine runs Gemma 4; only the CPU-side estimate is missing.
  Require(g.state == "none_yet" && g.memory_estimate.is_null() &&
              g.reason ==
                  "The model file is not on this computer, so its header "
                  "cannot be read.",
          "an absent Gemma package has no memory verdict and explains the "
          "missing header");
  const auto* diff = FindCatalogueEntry(
      shipped.entries, "diffusiongemma-26b-e38-abliterated-nvfp4");
  Require(EstimateForCatalogueEntry(*diff, base / "models", catalogue_request)
                  .state == "none_yet",
          "DiffusionGemma entry: no estimate yet");
  // A GGUF entry whose file is not on this computer cannot be read yet.
  const auto* q38 =
      FindCatalogueEntry(shipped.entries, "qwen38-27b-ud-q4-k-xl");
  const auto missing =
      EstimateForCatalogueEntry(*q38, base / "models", catalogue_request);
  Require(missing.state == "none_yet" &&
              Contains(missing.reason, "not on this computer"),
          "undownloaded GGUF entry: header cannot be read");
}

// ---- list ------------------------------------------------------------------
Value List(const fs::path& base) {
  // A synthetic catalogue entry whose file is the tiny dense model.
  const auto models = base / "list-models";
  const auto folder = models / "tiny-dense";
  fs::create_directories(folder);
  WriteDense(folder / "tiny.gguf");
  const auto size = fs::file_size(folder / "tiny.gguf");
  const std::string text =
      "{\"schema_version\":1,\"catalogue_version\":\"t\",\"entries\":[{"
      "\"id\":\"tiny-dense\",\"display_name\":\"Tiny dense\","
      "\"role\":\"supported\",\"architecture\":\"qwen35\","
      "\"repository\":\"fixture/tiny\",\"commit\":\"" +
      std::string(40, '1') +
      "\",\"files\":[{\"name\":\"tiny.gguf\","
      "\"repo_path\":\"tiny.gguf\",\"url\":\"https://huggingface.co/fixture/"
      "tiny/resolve/" +
      std::string(40, '1') + "/tiny.gguf\",\"bytes\":" + std::to_string(size) +
      ",\"sha256\":\"" + std::string(64, 'a') +
      "\",\"role\":\"main\"}],\"licence\":{\"id\":\"MIT\",\"name\":\"MIT\","
      "\"url\":\"https://opensource.org/license/mit\",\"prompt_line\":\"MIT.\","
      "\"verified_upstream\":true},\"download_enabled\":true},"
      "{\"id\":\"absent\",\"display_name\":\"Absent\",\"role\":\"supported\","
      "\"architecture\":\"qwen35\",\"download_enabled\":false,"
      "\"disabled_reason\":\"not pinned "
      "yet\",\"files\":[{\"name\":\"absent.gguf\","
      "\"repo_path\":\"absent.gguf\",\"url\":\"\",\"bytes\":0,\"sha256\":\"\","
      "\"role\":\"main\"}],\"licence\":{\"id\":\"MIT\",\"name\":\"MIT\","
      "\"url\":\"https://opensource.org/license/mit\",\"prompt_line\":\"MIT.\","
      "\"verified_upstream\":true}}]}";
  const auto load = ParseCatalogue(text);
  Require(load.error.empty(), "list catalogue: " + load.error);
  Require(CataloguePreparedState(load.entries[0], models) == "prepared" &&
              CataloguePreparedState(load.entries[1], models) == "not_prepared",
          "prepared state by file size");
  {
    std::ofstream(folder / "tiny.gguf", std::ios::binary | std::ios::app)
        << "z";
    Require(CataloguePreparedState(load.entries[0], models) == "not_prepared",
            "wrong size is not prepared");
    WriteDense(folder / "tiny.gguf");
  }
  // User folder with a runnable file, an unsupported file and a sidecar.
  const auto mine = base / "my-models";
  fs::create_directories(mine);
  WriteDense(mine / "mine.gguf");
  WriteBare(mine / "other.gguf", "gemma4");
  WriteBare(mine / "proj.gguf", "clip");
  // An imported package whose id the catalogue does not know (E20).
  const auto imported = base / "imported-models";
  const auto imported_marker = WriteCustomPackage(imported / "custom-pkg");
  ModelsListOptions options;
  options.models_root = models;
  options.scan_folders = {mine, "C:\\", imported};
  options.available_bytes = 64 * kGiB;
  options.host_available_bytes = 128 * kGiB;
  auto list = BuildModelsList(options, load.entries);
  for (const char* key :
       {"listing", "prepared_models", "catalogue", "scanned", "scan_warnings"})
    Require(list.find(key) != nullptr, std::string("list key ") + key);
  Require(list.find("prepared_models")->size() == 2 &&
              list.find("prepared_models")->items()[0].member_str("id") ==
                  "tiny-dense" &&
              list.find("prepared_models")
                      ->items()[0]
                      .member_str("prepared_state") == "prepared",
          "prepared_models lists the prepared entry, then the imported one");
  {
    // E20: every row says whether the catalogue knows it. The models-root
    // row keeps its fields; the imported package comes from its marker.
    const auto& root_row = list.find("prepared_models")->items()[0];
    const auto& custom = list.find("prepared_models")->items()[1];
    Require(root_row.find("catalogue") != nullptr &&
                root_row.find("catalogue")->as_bool() &&
                custom.find("catalogue") != nullptr &&
                !custom.find("catalogue")->as_bool() &&
                custom.member_str("id") == "custom-pkg" &&
                custom.member_str("display_name") == "custom.gguf" &&
                custom.member_str("prepared_state") == "prepared" &&
                SamePath(custom.member_str("manifest_path"), imported_marker) &&
                SamePath(custom.member_str("model_path"),
                         imported / "custom-pkg" / "custom.gguf") &&
                custom.find("capabilities")->find("mmproj")->as_bool() &&
                !custom.find("capabilities")->find("mtp")->as_bool(),
            "an unknown marker lists as catalogue false, from the marker");
    // Neither folder is in prepare's models/<id> layout: nothing to repair
    // with --output.
    Require(root_row.find("repair_output") != nullptr &&
                root_row.find("repair_output")->is_null() &&
                custom.find("repair_output") != nullptr &&
                custom.find("repair_output")->is_null(),
            "repair_output is null outside prepare's layout");
  }
  Require(list.find("catalogue")->size() == 2, "catalogue rows");
  const auto& tiny = list.find("catalogue")->items()[0];
  Require(
      tiny.member_str("prepared_state") == "prepared" &&
          tiny.find("estimate")->member_str("estimate_state") == "estimated" &&
          !tiny.find("estimate")->find("memory_estimate")->is_null(),
      "prepared catalogue entry has an estimate from its header");
  Require(list.find("catalogue")
                  ->items()[1]
                  .find("estimate")
                  ->member_str("estimate_state") == "none_yet",
          "disabled entry has no estimate");
  // Files inside a catalogue folder are not repeated under scanned.
  std::set<std::string> names;
  for (const auto& row : list.find("scanned")->items())
    names.insert(row.member_str("file_name"));
  Require(
      names == std::set<std::string>{"mine.gguf", "other.gguf", "proj.gguf"},
      "scanned = the user's files only");
  Require(list.find("listing")->member_size("package_files_skipped") == 1,
          "the package file is counted, not listed");
  for (const auto& row : list.find("scanned")->items()) {
    const auto name = row.member_str("file_name");
    if (name == "mine.gguf")
      Require(row.find("runnable")->as_bool() &&
                  row.member_str("kind") == "model" &&
                  row.find("reason")->is_null() &&
                  row.find("estimate")->member_str("estimate_state") ==
                      "estimated" &&
                  row.member_size("size_bytes") ==
                      fs::file_size(mine / "mine.gguf"),
              "runnable file carries size and an estimate");
    if (name == "other.gguf")
      Require(
          !row.find("runnable")->as_bool() &&
              Contains(row.member_str("reason"), "gemma4") &&
              row.find("estimate")->member_str("estimate_state") == "none_yet",
          "unsupported file carries the reason");
    if (name == "proj.gguf")
      Require(row.member_str("kind") == "sidecar", "sidecar kind");
  }
  Require(
      !list.find("scan_warnings")->items().empty() &&
          Contains(list.find("scan_warnings")->items()[0].member_str("reason"),
                   "never scanned"),
      "the drive root passed by the user is refused with a warning");
  Require(list.find("listing")->member_size("scan_depth_limit") == 3,
          "depth limit is stated");
  // Installed Flash Next paths differ from their upstream repository paths.
  // A hard link is a regular installed file, and one missing shard must not
  // turn the entire listing into an error.
  auto linked = load.entries[0];
  linked.id = "linked-flash";
  linked.flash_next_set = prepare::ModelSet::Huihui;
  linked.files[0].name = "first.gguf";
  linked.files[0].repo_path = "remote/first.gguf";
  auto second = linked.files[0];
  second.name = "mmproj-BF16.gguf";
  second.repo_path = "remote/mmproj-BF16.gguf";
  second.role = "mmproj";
  linked.files.push_back(second);
  const auto linked_root = base / "linked-models";
  const auto linked_folder = linked_root / linked.id;
  fs::create_directories(linked_folder);
  fs::create_hard_link(folder / "tiny.gguf", linked_folder / "first.gguf");
  fs::create_directories(linked_folder / "mmproj");
  fs::create_hard_link(
      folder / "tiny.gguf",
      linked_folder / "mmproj" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf");
  std::ofstream(linked_folder / "bubo-package.json") << "{}";
  Require(CataloguePreparedState(linked, linked_root) == "prepared",
          "hard-linked installed destinations are prepared");
  fs::remove(linked_folder / "first.gguf");
  ModelsListOptions incomplete;
  incomplete.models_root = linked_root;
  incomplete.available_bytes = 64 * kGiB;
  incomplete.host_available_bytes = 128 * kGiB;
  const auto partial = BuildModelsList(incomplete, {linked});
  const auto& partial_row = partial.find("catalogue")->items()[0];
  bool names_missing_shard = false;
  for (const auto& note : partial_row.find("estimate")->find("notes")->items())
    names_missing_shard |= Contains(note.str(), "first.gguf");
  Require(partial_row.member_str("prepared_state") == "partial" &&
              names_missing_shard &&
              partial.find("prepared_models")->items().empty(),
          "missing shard is named and not listed as prepared");
  // Machine output is one JSON object per event and parses back.
  Require(!list.dump().empty() && list.dump().find('\n') == std::string::npos,
          "single-line JSON");
  return list;
}

// ---- vision projectors of plain catalogue models ----------------------------
// The file at `path` grows to exactly `size` bytes. The added part is a sparse
// hole: no data is written and no disk space is allocated.
void ExtendSparse(const fs::path& path, std::uint64_t size) {
#ifdef _WIN32
  const HANDLE file =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  Require(file != INVALID_HANDLE_VALUE, "open for sparse: " + path.string());
  DWORD returned = 0;
  const bool sparse = DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0,
                                      nullptr, 0, &returned, nullptr) != 0;
  CloseHandle(file);
  Require(sparse, "sparse flag: " + path.string());
#endif
  fs::resize_file(path, size);
}
// A BF16-projector stand-in: a clip header with three F32 tensors.
constexpr std::uint64_t kClipUploadBytes = 1000 * 4 + 1152ULL * 1536 * 2;
void WriteClip(const fs::path& path, std::uint64_t size) {
  Gguf g;
  g.Str("general.architecture", "clip");
  g.U32("clip.vision.embedding_length", 1152);
  g.Tensor("v.patch_embd.weight", 4);
  g.Tensor("v.patch_embd.weight.1", 4);
  g.Tensor("v.blk.0.attn_q.weight", 1000);
  g.Write(path);
  ExtendSparse(path, size);
}
const CatalogueFile* DownloadableFile(const CatalogueEntry& entry,
                                      std::string_view role) {
  for (const auto& file : entry.files)
    if (file.role == role && file.downloadable)
      return &file;
  return nullptr;
}
std::string MmprojSetting(const gufo::server::ReleaseStartup& startup) {
  for (const auto& setting : startup.resolved.settings)
    if (setting.name == "--mmproj")
      return setting.value.value_or("") + "|" + setting.origin;
  return "|";
}
bool AnyContains(const std::vector<std::string>& lines,
                 std::string_view needle) {
  return std::any_of(lines.begin(), lines.end(), [&](const std::string& line) {
    return Contains(line, needle);
  });
}
// The package marker that `prepare --catalogue-id` writes last.
void WriteCatalogueMarker(const CatalogueEntry& entry, const fs::path& folder) {
  auto files = Value::array();
  auto artefacts = Value::object();
  for (const auto& file : entry.files) {
    if (!file.downloadable)
      continue;
    auto row = Value::object();
    row["file"] = file.name;
    row["size"] = file.bytes;
    row["sha256"] = file.sha256;
    if (file.role == "main")
      files.push_back(std::move(row));
    else
      artefacts[file.name] = std::move(row);
  }
  auto source = Value::object();
  source["repository"] = entry.repository;
  source["revision"] = entry.commit;
  auto model = Value::object();
  model["family"] = entry.architecture;
  model["source"] = std::move(source);
  model["files"] = std::move(files);
  auto marker = Value::object();
  marker["schema_version"] = 1;
  marker["package_id"] = entry.id;
  marker["engine_min"] = "0.1.0";
  marker["model"] = std::move(model);
  marker["artefacts"] = std::move(artefacts);
  auto profiles = Value::array();
  for (const auto* profile : {"default", "subagents", "exact", "stock"})
    profiles.push_back(profile);
  marker["profiles_supported"] = std::move(profiles);
  std::ofstream(folder / "bubo-package.json", std::ios::binary)
      << marker.dump() << "\n";
}

// E11: a plain catalogue model (not Flash Next) lists its BF16 projector,
// counts it for readiness, binds it as --mmproj when it is prepared, and
// adds the engine's vision term to the estimate. Real catalogue entries; each
// file is a tiny header grown to its pinned size as a sparse file.
void CatalogueVision(const fs::path& base) {
  const auto& shipped = ShippedCatalogue();
  Require(shipped.error.empty(), "shipped catalogue: " + shipped.error);
  const auto entry_of = [&](std::string_view id) {
    const auto* entry = FindCatalogueEntry(shipped.entries, id);
    Require(entry != nullptr, "catalogue entry " + std::string(id));
    return entry;
  };
  const auto root = base / "vision-models";
  ModelsListOptions options;
  options.models_root = root;
  options.available_bytes = 96 * kGiB;
  options.host_available_bytes = 128 * kGiB;
  EstimateRequest request;
  request.context = 8192;
  request.available_bytes = 96 * kGiB;
  request.host_available_bytes = 128 * kGiB;

  // Qwen3.8 27B (dense qwen35).
  const auto* entry = entry_of("qwen38-27b-ud-q4-k-xl");
  const auto* main = DownloadableFile(*entry, "main");
  const auto* projector = DownloadableFile(*entry, "mmproj");
  Require(main != nullptr && projector != nullptr &&
              projector->name == "mmproj-BF16.gguf" &&
              projector->bytes == 931146432ULL,
          "Qwen3.8 27B lists its BF16 projector as a downloadable file");
  const auto folder = CatalogueEntryFolder(*entry, root);
  fs::create_directories(folder);
  const auto main_path = folder / fs::u8path(main->name);
  const auto projector_path = folder / fs::u8path(projector->name);
  WriteDense(main_path);
  ExtendSparse(main_path, main->bytes);

  // 1. Model file only: partial, the projector is named; serve is text only.
  Require(CataloguePreparedState(*entry, root) == "partial",
          "a model without its projector is partial");
  {
    const auto list = BuildModelsList(options, {*entry});
    const auto& row = list.find("catalogue")->items()[0];
    bool named = false;
    for (const auto& note : row.find("estimate")->find("notes")->items())
      named |= Contains(note.str(),
                        "Missing or wrong-sized file: "
                        "mmproj-BF16.gguf");
    Require(row.member_str("prepared_state") == "partial" && named &&
                list.find("prepared_models")->items().empty(),
            "models list names the missing projector");
    const auto startup = InspectSettings({"--model", main_path.string()});
    Require(startup.resolved.artefacts.mmproj.empty() &&
                MmprojSetting(startup) == "|built-in" &&
                AnyContains(startup.resolved.logs, "mmproj-BF16.gguf"),
            "without the projector serve starts text only and says why");
  }
  const auto text_only = EstimateForCatalogueEntry(*entry, root, request);
  Require(text_only.state == "estimated" &&
              text_only.memory_estimate.find("weights")->member_size("bytes") ==
                  main->bytes,
          "text-only estimate: the weights are the model file: " +
              text_only.reason);

  // 2. Projector present: prepared, bound as --mmproj, counted.
  WriteClip(projector_path, projector->bytes);
  Require(CataloguePreparedState(*entry, root) == "prepared",
          "model plus projector is prepared");
  {
    const auto list = BuildModelsList(options, {*entry});
    const auto& prepared = list.find("prepared_models")->items();
    Require(prepared.size() == 1 &&
                prepared[0].find("capabilities")->find("mmproj")->as_bool() &&
                prepared[0].member_size("size_bytes") ==
                    main->bytes + projector->bytes &&
                SamePath(prepared[0].member_str("model_path"), main_path) &&
                prepared[0].find("manifest_path")->is_null(),
            "prepared row carries the projector and the model path");
  }
  {
    const auto startup = InspectSettings({"--model", main_path.string()});
    Require(SamePath(startup.resolved.artefacts.mmproj, projector_path) &&
                Contains(MmprojSetting(startup), "|manifest") &&
                SamePath(gufo::server::ReleaseSettingsJson(startup.resolved)
                             .find("engine")
                             ->member_str("mmproj_bound"),
                         projector_path),
            "serve binds the prepared projector: " + MmprojSetting(startup));
    const auto off =
        InspectSettings({"--model", main_path.string(), "--no-vision"});
    Require(off.resolved.artefacts.mmproj.empty() &&
                off.resolved.artefacts.vision_disabled,
            "--no-vision still turns the projector off");
    const auto other = base / "gguf" / "clip.gguf";
    const auto chosen = InspectSettings(
        {"--model", main_path.string(), "--mmproj", other.string()});
    Require(SamePath(chosen.resolved.artefacts.mmproj, other),
            "an explicit --mmproj still wins");
  }
  const auto with_vision = EstimateForCatalogueEntry(*entry, root, request);
  Require(with_vision.state == "estimated" &&
              with_vision.memory_estimate.find("weights")->member_size(
                  "bytes") == main->bytes + kClipUploadBytes &&
              AnyContains(with_vision.notes, "mmproj-BF16.gguf"),
          "the estimate adds the encoder's uploaded projector bytes: " +
              with_vision.reason);

  // 3. The package marker that prepare writes: --model and --manifest bind.
  WriteCatalogueMarker(*entry, folder);
  {
    const auto list = BuildModelsList(options, {*entry});
    const auto& prepared = list.find("prepared_models")->items();
    Require(list.find("catalogue")
                        ->items()[0]
                        .find("estimate")
                        ->member_str("estimate_state") == "estimated" &&
                prepared.size() == 1 &&
                SamePath(prepared[0].member_str("manifest_path"),
                         folder / "bubo-package.json") &&
                SamePath(prepared[0].member_str("model_path"), main_path),
            "a prepared folder with its marker lists and estimates");
    const auto by_model = InspectSettings({"--model", main_path.string()});
    Require(by_model.package.package_id == entry->id &&
                SamePath(by_model.resolved.artefacts.mmproj, projector_path),
            "a prepared folder serves with its marker and its projector");
    const auto by_manifest = InspectSettings(
        {"--manifest", (folder / "bubo-package.json").string()});
    Require(!by_manifest.package.files.empty() &&
                by_manifest.package.files.front().path ==
                    fs::absolute(main_path).lexically_normal() &&
                SamePath(by_manifest.resolved.artefacts.mmproj, projector_path),
            "--manifest selects the model and binds the projector");
  }
  // A projector that went missing after prepare: text only, no failure.
  fs::remove(projector_path);
  {
    const auto startup = InspectSettings(
        {"--manifest", (folder / "bubo-package.json").string()});
    Require(startup.resolved.artefacts.mmproj.empty() &&
                AnyContains(startup.resolved.logs, "mmproj-BF16.gguf"),
            "a missing projector leaves serve text only");
  }

  // Cyber-Tiel (qwen35moe) and Gemma 4 31B (gemma4): the same rule.
  for (const auto* id : {"cyber-tiel-coder-35b-a3b-mtp-q4-k-m",
                         "gemma4-31b-huihui-abliterated-qat-q4-k"}) {
    const auto* other = entry_of(id);
    const auto* other_main = DownloadableFile(*other, "main");
    const auto* other_projector = DownloadableFile(*other, "mmproj");
    Require(other_main != nullptr && other_projector != nullptr,
            std::string(id) + " lists its projector");
    const auto other_folder = CatalogueEntryFolder(*other, root);
    fs::create_directories(other_folder);
    const auto path = other_folder / fs::u8path(other_main->name);
    if (other->architecture == "qwen35moe")
      WriteA3b(path);
    else if (other->architecture == "gemma4")
      WriteGemma4(path);
    else
      WriteBare(path, other->architecture);
    ExtendSparse(path, other_main->bytes);
    WriteClip(other_folder / fs::u8path(other_projector->name),
              other_projector->bytes);
    if (const auto* head = DownloadableFile(*other, "mtp")) {
      const auto head_path = other_folder / fs::u8path(head->name);
      WriteBare(head_path, "gemma4-assistant");
      ExtendSparse(head_path, head->bytes);
    }
    Require(CataloguePreparedState(*other, root) == "prepared",
            std::string(id) + " prepared");
    const auto startup = InspectSettings({"--model", path.string()});
    Require(SamePath(startup.resolved.artefacts.mmproj,
                     other_folder / fs::u8path(other_projector->name)),
            std::string(id) + " binds its projector");
    const auto estimate = EstimateForCatalogueEntry(*other, root, request);
    if (other->architecture == "qwen35moe") {
      if (InspectGguf(path).support.runnable)
        Require(estimate.state == "estimated" &&
                    estimate.memory_estimate.find("weights")->member_size(
                        "bytes") == other_main->bytes + kClipUploadBytes,
                "A3B estimate adds the projector: " + estimate.reason);
      else
        Require(estimate.state == "none_yet" &&
                    estimate.memory_estimate.is_null() &&
                    estimate.reason ==
                        "This build was made without Qwen3.6-35B-A3B "
                        "(qwen35moe) support.",
                "a build without A3B gives no unsupported projector estimate");
    } else
      Require(estimate.state == "none_yet" &&
                  estimate.memory_estimate.is_null() &&
                  estimate.reason ==
                      "Bubo has no memory estimate for this model yet.",
              "Gemma 4 has no verdict and uses the plain no-estimate sentence");
  }
  // One damaged package marker gives its row no estimate; the list stands.
  {
    const auto* broken = entry_of("qwen36-27b-ud-q4-k-xl");
    const auto broken_folder = CatalogueEntryFolder(*broken, root);
    fs::create_directories(broken_folder);
    const auto path =
        broken_folder / fs::u8path(DownloadableFile(*broken, "main")->name);
    WriteDense(path);
    ExtendSparse(path, DownloadableFile(*broken, "main")->bytes);
    std::ofstream(broken_folder / "bubo-package.json") << "{}";
    const auto list = BuildModelsList(options, {*broken, *entry});
    const auto& rows = list.find("catalogue")->items();
    Require(rows.size() == 2 &&
                rows[0].find("estimate")->member_str("estimate_state") ==
                    "none_yet" &&
                Contains(rows[0].find("estimate")->member_str("reason"),
                         "The estimate failed") &&
                rows[1].member_str("prepared_state") == "partial",
            "a damaged marker fails its own row only");
  }
  // HauhauCS ships its own projector in its own repository; stock keeps the
  // Unsloth one.
  {
    const auto* hauhau = DownloadableFile(
        *entry_of("gemma4-26b-a4b-huihui-abliterated-qat-q4-k"), "mmproj");
    const auto* stock = DownloadableFile(
        *entry_of("gemma4-26b-a4b-unsloth-qat-ud-q4-k-xl"), "mmproj");
    Require(hauhau != nullptr && stock != nullptr &&
                hauhau->repository.empty() && hauhau->sha256 != stock->sha256,
            "HauhauCS lists its own 26B projector");
  }
}

// ---- E20: prepared packages in any folder -----------------------------------
// Where prepare puts a catalogue file inside a package folder.
fs::path InstallPath(const CatalogueEntry& entry, const CatalogueFile& file,
                     const fs::path& folder) {
  if (entry.flash_next_set)
    for (const auto& pinned : prepare::ModelCatalogue(*entry.flash_next_set))
      if (pinned.file_name == file.name)
        return folder / fs::u8path(pinned.destination);
  return folder / fs::u8path(file.name);
}
// A prepared catalogue package: every downloadable file at its pinned size
// (sparse, no disk space) and the marker last, as prepare writes it. Plain
// models get real headers so the header estimate can read them.
void InstallEntry(const CatalogueEntry& entry, const fs::path& folder,
                  std::string_view skip_role = {}) {
  for (const auto& file : entry.files) {
    if (!file.downloadable || file.role == skip_role)
      continue;
    const auto path = InstallPath(entry, file, folder);
    fs::create_directories(path.parent_path());
    if (!entry.flash_next_set && file.role == "main") {
      WriteDense(path);
      ExtendSparse(path, file.bytes);
    } else if (!entry.flash_next_set && file.role == "mmproj") {
      WriteClip(path, file.bytes);
    } else {
      Touch(path);
      ExtendSparse(path, file.bytes);
    }
  }
  fs::create_directories(folder);
  WriteCatalogueMarker(entry, folder);
}
const Value* RowByManifest(const Value& list, const fs::path& manifest) {
  for (const auto& row : list.find("prepared_models")->items()) {
    const auto* path = row.find("manifest_path");
    if (path && !path->is_null() && SamePath(path->str(), manifest))
      return &row;
  }
  return nullptr;
}
bool Flag(const Value& row, const char* key) {
  const auto* value = row.find(key);
  return value && value->as_bool();
}

void Packages(const fs::path& base) {
  const auto& shipped = ShippedCatalogue();
  const auto* flash = FindCatalogueEntry(shipped.entries, prepare::kPackageId);
  const auto* dense =
      FindCatalogueEntry(shipped.entries, "qwen38-27b-ud-q4-k-xl");
  Require(flash != nullptr && dense != nullptr, "catalogue entries for E20");
  const std::vector<CatalogueEntry> catalogue{*flash, *dense};

  // One scan folder holds packages at several depths and one loose file.
  const auto scan = base / "pkg-scan";
  const auto flash_full = scan / "download-a" / "models" / flash->id;
  const auto flash_partial = scan / "partial-flash" / flash->id;
  const auto dense_b = scan / "download-b" / dense->id;
  const auto dense_copy = scan / "copy" / dense->id;
  const auto dense_partial = scan / "old" / dense->id;
  const auto renamed = scan / "renamed" / "my-copy-of-27b";
  InstallEntry(*flash, flash_full);
  fs::create_directories(flash_partial);
  WriteCatalogueMarker(*flash, flash_partial);  // marker only: nothing else
  InstallEntry(*dense, dense_b);
  InstallEntry(*dense, dense_copy);
  InstallEntry(*dense, dense_partial, "mmproj");
  InstallEntry(*dense, renamed);
  const auto custom_marker = WriteCustomPackage(scan / "imported" / "custom");
  // A GGUF file inside a package folder is the package's, never a loose row.
  fs::create_directories(dense_b / "extra");
  WriteDense(dense_b / "extra" / "nested.gguf");
  fs::create_directories(scan / "loose");
  WriteDense(scan / "loose" / "loose.gguf");

  ModelsListOptions options;
  options.models_root = base / "pkg-no-models-root";
  options.scan_folders = {scan};
  options.available_bytes = 96 * kGiB;
  options.host_available_bytes = 128 * kGiB;
  const auto list = BuildModelsList(options, catalogue);
  std::set<std::string> names;
  for (const auto& row : list.find("scanned")->items())
    names.insert(row.member_str("file_name"));
  Require(names == std::set<std::string>{"loose.gguf"},
          "package files are never loose scanned rows");
  Require(list.find("prepared_models")->size() == 7,
          "one prepared_models row per package folder: " +
              std::to_string(list.find("prepared_models")->size()));
  Require(list.find("scan_warnings")->items().empty(),
          "a package folder is not a scan problem");

  const auto* flash_row = RowByManifest(list, flash_full / "bubo-package.json");
  Require(
      flash_row != nullptr && flash_row->member_str("id") == flash->id &&
          flash_row->member_str("display_name") == flash->display_name &&
          Flag(*flash_row, "catalogue") &&
          flash_row->member_str("prepared_state") == "prepared" &&
          flash_row->member_size("size_bytes") ==
              CatalogueDownloadBytes(*flash) &&
          Flag(*flash_row->find("capabilities"), "mtp") &&
          Flag(*flash_row->find("capabilities"), "twin") &&
          Flag(*flash_row->find("capabilities"), "mmproj") &&
          flash_row->find("capabilities")->member_size("context_max") == 262144,
      "a Flash Next package lists as the full package from its entry");
  const auto* flash_partial_row =
      RowByManifest(list, flash_partial / "bubo-package.json");
  Require(flash_partial_row != nullptr &&
              flash_partial_row->member_str("prepared_state") == "partial",
          "a Flash Next folder with only its marker says partial");

  const auto* b_row = RowByManifest(list, dense_b / "bubo-package.json");
  const auto* copy_row = RowByManifest(list, dense_copy / "bubo-package.json");
  Require(b_row != nullptr && copy_row != nullptr &&
              b_row->member_str("id") == dense->id &&
              copy_row->member_str("id") == dense->id &&
              b_row->member_str("prepared_state") == "prepared" &&
              copy_row->member_str("prepared_state") == "prepared" &&
              Flag(*b_row, "catalogue") && Flag(*copy_row, "catalogue") &&
              Flag(*b_row->find("capabilities"), "mmproj") &&
              SamePath(b_row->member_str("model_path"),
                       InstallPath(*dense, *DownloadableFile(*dense, "main"),
                                   dense_b)),
          "the same id in two folders gives two rows, each with its path");
  const auto* old_row =
      RowByManifest(list, dense_partial / "bubo-package.json");
  Require(old_row != nullptr &&
              old_row->member_str("prepared_state") == "partial" &&
              Flag(*old_row, "catalogue"),
          "a package missing its projector says partial");
  const auto* renamed_row = RowByManifest(list, renamed / "bubo-package.json");
  Require(renamed_row != nullptr &&
              renamed_row->member_str("id") == dense->id &&
              renamed_row->member_str("prepared_state") == "prepared" &&
              Flag(*renamed_row, "catalogue"),
          "a renamed package folder is checked at its own folder");
  const auto* custom_row = RowByManifest(list, custom_marker);
  Require(custom_row != nullptr &&
              custom_row->member_str("id") == "custom-pkg" &&
              !Flag(*custom_row, "catalogue") &&
              custom_row->member_str("prepared_state") == "prepared",
          "an unknown marker lists with catalogue false");
  if (std::getenv("BUBO_TEST_TRACE"))
    std::cerr << "E20 example row: " << flash_row->dump() << "\n"
              << "E20 unknown row: " << custom_row->dump() << std::endl;

  // The unknown package says partial when a recorded file is gone.
  fs::remove(scan / "imported" / "custom" / "mmproj-BF16.gguf");
  {
    const auto after = BuildModelsList(options, catalogue);
    const auto* row = RowByManifest(after, custom_marker);
    Require(row != nullptr && row->member_str("prepared_state") == "partial",
            "an unknown package with a recorded file missing says partial");
  }
  // A marker nobody can read is named in the warnings, not hidden.
  Touch(scan / "damaged" / "bubo-package.json", "not json");
  Touch(scan / "damaged" / "d.gguf");
  {
    const auto after = BuildModelsList(options, catalogue);
    bool named = false;
    for (const auto& warning : after.find("scan_warnings")->items())
      named |= Contains(warning.member_str("folder"), "damaged") &&
               Contains(warning.member_str("reason"), "bubo-package.json");
    std::set<std::string> loose;
    for (const auto& row : after.find("scanned")->items())
      loose.insert(row.member_str("file_name"));
    Require(
        named && loose == names && after.find("prepared_models")->size() == 7,
        "an unreadable marker gives a warning and no rows");
  }

  // Under --models-root nothing changes: one row per prepared catalogue
  // folder, even when the same folder is scanned again by name.
  const auto root = base / "pkg-models";
  InstallEntry(*dense, root / dense->id);
  ModelsListOptions inside;
  inside.models_root = root;
  inside.scan_folders = {root, root};
  inside.available_bytes = 96 * kGiB;
  inside.host_available_bytes = 128 * kGiB;
  {
    const auto again = BuildModelsList(inside, catalogue);
    Require(again.find("prepared_models")->size() == 1 &&
                Flag(again.find("prepared_models")->items()[0], "catalogue") &&
                again.find("scanned")->items().empty(),
            "a models-root package is listed once and not repeated");
  }
  // A package folder that is not one of the catalogue's own folders but sits
  // under the models root is listed too.
  WriteCustomPackage(root / "imported-here");
  {
    const auto again = BuildModelsList(inside, catalogue);
    Require(again.find("prepared_models")->size() == 2 &&
                !Flag(again.find("prepared_models")->items()[1], "catalogue"),
            "an unknown package under the models root is listed after it");
  }

  // repair_output: the --output folder that prepare accepts for the package,
  // by prepare's own layout rule <output>/models/<id>; null elsewhere.
  {
    const auto repair = [&](const fs::path& marker) {
      const auto* row = RowByManifest(list, marker);
      Require(row != nullptr && row->find("repair_output") != nullptr,
              "every package row carries repair_output");
      return row->find("repair_output")->is_null()
                 ? std::string()
                 : row->member_str("repair_output");
    };
    Require(
        SamePath(repair(flash_full / "bubo-package.json"), scan / "download-a"),
        "a package in <output>/models/<id> names <output>");
    Require(repair(dense_b / "bubo-package.json").empty() &&
                repair(renamed / "bubo-package.json").empty() &&
                repair(custom_marker).empty(),
            "a package outside that layout has no repair folder");
    const auto bubo_root = base / "bubo-root";
    InstallEntry(*dense, bubo_root / "models" / dense->id);
    ModelsListOptions layout;
    layout.models_root = bubo_root / "models";
    layout.available_bytes = 96 * kGiB;
    layout.host_available_bytes = 128 * kGiB;
    const auto under_root = BuildModelsList(layout, catalogue);
    const auto& row = under_root.find("prepared_models")->items()[0];
    Require(under_root.find("prepared_models")->size() == 1 &&
                SamePath(row.member_str("repair_output"), bubo_root),
            "a models-root package names the output root above models");
  }

  // Item 12: `bubo status --manifest PATH` / `--model PATH` answers for that
  // package, with the resolution it already does for BUBO_MANIFEST.
  {
    const auto local = (base / "status-local").string();
    const auto runtime = base / "status-runtime";
    const auto ready_of = [](const Value& snapshot) {
      return snapshot.find("readiness")->find("model")->as_bool();
    };
    const auto row_of = [](const Value& snapshot) {
      return snapshot.find("prepared_models")->items()[0];
    };
    const auto none = StatusSnapshot(local, runtime, false);
    Require(!ready_of(none), "no selection: nothing is ready");
    const auto by_manifest = StatusSnapshot(local, runtime, false, {},
                                            dense_b / "bubo-package.json");
    Require(
        ready_of(by_manifest) &&
            row_of(by_manifest).member_str("id") == dense->id &&
            row_of(by_manifest).member_str("prepared_state") == "prepared" &&
            by_manifest.find("model_selection")->member_str("source") ==
                "explicit_manifest" &&
            SamePath(by_manifest.find("paths")->member_str("manifest"),
                     dense_b / "bubo-package.json"),
        "status --manifest reports a ready catalogue package");
    const auto partial = StatusSnapshot(local, runtime, false, {},
                                        dense_partial / "bubo-package.json");
    Require(!ready_of(partial) &&
                row_of(partial).member_str("prepared_state") == "not_prepared",
            "status --manifest reports a partial package as not ready");
    const auto by_model = StatusSnapshot(
        local, runtime, false,
        InstallPath(*dense, *DownloadableFile(*dense, "main"), dense_b), {});
    Require(ready_of(by_model) &&
                by_model.find("model_selection")->member_str("source") ==
                    "explicit_model" &&
                SamePath(by_model.find("paths")->member_str("manifest"),
                         dense_b / "bubo-package.json"),
            "status --model finds the package marker beside the model");
    const std::vector<const char*> good{"--json", "--manifest", "x.json"};
    Require(ParseStatusOptions(good).manifest == fs::path("x.json"),
            "status parses --manifest");
    const std::vector<const char*> both{"--json", "--manifest", "a", "--model",
                                        "b"};
    bool refused = false;
    try {
      (void)ParseStatusOptions(both);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Require(refused, "status refuses --manifest with --model");
  }

  // The estimate by folder: the table for Flash Next, the header for GGUF.
  EstimateRequest request;
  request.context = 8192;
  request.available_bytes = 96 * kGiB;
  request.host_available_bytes = 128 * kGiB;
  {
    EstimateResult table;
    std::string thrown;
    try {
      table = EstimateForFolder(flash_partial, request);
    } catch (const std::exception& error) {
      thrown = error.what();
    }
    const auto direct =
        EstimateForCatalogueEntry(*flash, flash_partial.parent_path(), request);
    Require(thrown.empty() && table.state == "estimated" &&
                table.memory_estimate.member_str("fit_scope") ==
                    "conservative_startup_estimate" &&
                Contains(table.evidence_label, "measured Flash Next") &&
                table.memory_estimate.dump() == direct.memory_estimate.dump(),
            "a Flash Next folder keeps its measured table: " + thrown +
                table.reason);
  }
  {
    EstimateResult header;
    std::string thrown;
    try {
      header = EstimateForFolder(dense_b, request);
    } catch (const std::exception& error) {
      thrown = error.what();
    }
    const auto direct =
        EstimateForCatalogueEntry(*dense, dense_b.parent_path(), request);
    Require(thrown.empty() && header.state == "estimated" &&
                header.evidence_label == "estimated from the file header" &&
                header.memory_estimate.dump() == direct.memory_estimate.dump(),
            "a GGUF package folder gets the header estimate: " + thrown +
                header.reason);
    const auto moved = EstimateForFolder(renamed, request);
    Require(
        moved.state == "estimated" &&
            moved.memory_estimate.dump() == direct.memory_estimate.dump(),
        "a renamed GGUF package folder estimates the same: " + moved.reason);
  }
  {
    // The same two through the command, by path.
    std::ostringstream out, err;
    const auto folder_arg = flash_partial.string();
    std::vector<const char*> args{"estimate", "--model", folder_arg.c_str(),
                                  "--json"};
    Require(RunModels(args, base.string(), out, err) == 0 &&
                Contains(out.str(), "\"subject_kind\":\"path\"") &&
                Contains(out.str(), "conservative_startup_estimate"),
            "estimate --model FOLDER gives the Flash Next table: " + out.str() +
                err.str());
    std::ostringstream list_out, list_err;
    const auto root_arg = (base / "pkg-no-models-root").string();
    const auto scan_arg = scan.string();
    std::vector<const char*> list_args{"list",          "--json",
                                       "--models-root", root_arg.c_str(),
                                       "--scan-folder", scan_arg.c_str()};
    Require(RunModels(list_args, base.string(), list_out, list_err) == 0 &&
                Contains(list_out.str(), "\"catalogue\":false") &&
                Contains(list_out.str(), "\"prepared_state\":\"partial\""),
            "list --scan-folder prints package rows: " + list_err.str());
  }
}

// ---- item 7: the mtp head of a Gemma 4 package
// --------------------------------- HauhauCS ships a 252 MB MTP head in its
// package. Serve binds it and decodes with MTP by default; --no-mtp turns it
// off; without the head the package serves as before. Cyber-Tiel does not claim
// MTP.
std::string SettingValue(const gufo::server::ReleaseStartup& startup,
                         const std::string& name) {
  for (const auto& setting : startup.resolved.settings)
    if (setting.name == name)
      return setting.value.value_or("");
  return "<missing row>";
}
void GemmaMtp(const fs::path& base) {
  const auto& shipped = ShippedCatalogue();
  const auto* hauhau = FindCatalogueEntry(
      shipped.entries, "gemma4-26b-a4b-huihui-abliterated-qat-q4-k");
  const auto* stock = FindCatalogueEntry(
      shipped.entries, "gemma4-26b-a4b-unsloth-qat-ud-q4-k-xl");
  const auto* cyber = FindCatalogueEntry(shipped.entries,
                                         "cyber-tiel-coder-35b-a3b-mtp-q4-k-m");
  Require(hauhau && stock && cyber, "catalogue entries for item 7");
  // Cyber-Tiel: the row claims nothing, and neither does the name.
  Require(
      !Contains(cyber->display_name, "MTP") &&
          !Flag(*PreparedModelRow(*cyber, "", "", true).find("capabilities"),
                "mtp"),
      "Cyber-Tiel does not claim MTP");
  const auto* big = FindCatalogueEntry(
      shipped.entries, "gemma4-31b-huihui-abliterated-qat-q4-k");
  Require(big != nullptr, "the 31B entry");
  Require(Flag(*PreparedModelRow(*hauhau, "", "", true).find("capabilities"),
               "mtp") &&
              Flag(*PreparedModelRow(*stock, "", "", true).find("capabilities"),
                   "mtp") &&
              Flag(*PreparedModelRow(*big, "", "", true).find("capabilities"),
                   "mtp"),
          "the 26B and 31B QAT entries claim their MTP heads");

  const auto install = [&](const CatalogueEntry& entry, const fs::path& folder,
                           bool with_head) {
    for (const auto& file : entry.files) {
      if (!file.downloadable || (!with_head && file.role == "mtp"))
        continue;
      const auto path = folder / fs::u8path(file.name);
      fs::create_directories(path.parent_path());
      if (file.role == "main")
        WriteGemma4(path);
      else if (file.role == "mmproj")
        WriteClip(path, file.bytes);
      else
        WriteBare(path, "gemma4-assistant");
      if (file.role != "mmproj")
        ExtendSparse(path, file.bytes);
    }
    WriteCatalogueMarker(entry, folder);
    return folder / "bubo-package.json";
  };
  const auto head_name = DownloadableFile(*hauhau, "mtp")->name;
  const auto root = base / "gemma-mtp";
  const auto with_head = root / "with-head" / hauhau->id;
  const auto marker = install(*hauhau, with_head, true);
  {
    const auto startup = InspectSettings({"--manifest", marker.string()});
    Require(SettingValue(startup, "--speculative") == "mtp" &&
                SamePath(SettingValue(startup, "--mtp-model"),
                         with_head / fs::u8path(head_name)) &&
                SamePath(startup.resolved.artefacts.mtp,
                         with_head / fs::u8path(head_name)),
            "HauhauCS binds its head and decodes with MTP by default: " +
                SettingValue(startup, "--speculative") + " " +
                SettingValue(startup, "--mtp-model"));
    const auto muted =
        InspectSettings({"--manifest", marker.string(), "--no-mtp"});
    Require(SettingValue(muted, "--speculative") == "off" &&
                SettingValue(muted, "--mtp-model").empty(),
            "--no-mtp wins over the head");
    const auto stock_profile =
        InspectSettings({"--manifest", marker.string(), "--profile", "stock"});
    Require(SettingValue(stock_profile, "--speculative") != "mtp",
            "profile stock keeps MTP off");
  }
  // The head is not there (or the wrong size): the package serves as before.
  const auto without = root / "without-head" / hauhau->id;
  install(*hauhau, without, false);
  {
    const auto startup = InspectSettings(
        {"--manifest", (without / "bubo-package.json").string()});
    const auto speculative = SettingValue(startup, "--speculative");
    Require((speculative == "off" || speculative.empty()) &&
                SettingValue(startup, "--mtp-model").empty(),
            "without its head the package serves with MTP off: " + speculative);
  }
  // Stock Gemma 26B lists the same head (a file of the HauhauCS repository)
  // and takes the same path.
  const auto plain = root / "stock" / stock->id;
  const auto plain_marker = install(*stock, plain, true);
  {
    const auto startup = InspectSettings({"--manifest", plain_marker.string()});
    Require(
        SettingValue(startup, "--speculative") == "mtp" &&
            SamePath(SettingValue(startup, "--mtp-model"),
                     plain / fs::u8path(DownloadableFile(*stock, "mtp")->name)),
        "stock Gemma 26B binds the same head and decodes with MTP");
  }
  // A Gemma 4 entry with no mtp role is unchanged.
  const auto big_folder = root / "big" / big->id;
  const auto big_marker = install(*big, big_folder, true);
  {
    const auto startup = InspectSettings({"--manifest", big_marker.string()});
    const auto speculative = SettingValue(startup, "--speculative");
    Require(speculative == "mtp" &&
                SamePath(SettingValue(startup, "--mtp-model"),
                         big_folder /
                             fs::u8path(DownloadableFile(*big, "mtp")->name)),
            "Gemma 4 31B QAT binds its shipped head and keeps MTP on");
  }
}

// ---- command line ----------------------------------------------------------
void Command(const fs::path& base) {
  const auto run = [&](std::vector<const char*> args, std::string& out_text,
                       std::string& err_text) {
    std::ostringstream out, err;
    const auto code = RunModels(args, base.string(), out, err);
    out_text = out.str();
    err_text = err.str();
    return code;
  };
  std::string out, err;
  Require(run({"--help"}, out, err) == 0 && Contains(out, "Usage: bubo models"),
          "help");
  Require(run({}, out, err) == 2, "no verb");
  Require(run({"list", "--bogus"}, out, err) == 2, "unknown option");
  Require(run({"estimate"}, out, err) == 2, "estimate needs --model");
  Require(run({"estimate", "--model", "x", "--context", "0"}, out, err) == 2,
          "context must be positive");
  Require(run({"estimate", "--model",
               (base / "no-such-file.gguf").string().c_str()},
              out, err) == 12,
          "missing model path is code 12");
  const auto dir = base / "gguf";
  // Human-mode estimate of a real synthetic file.
  const auto dense = (dir / "dense.gguf").string();
  Require(run({"estimate", "--model", dense.c_str(), "--context", "4096"}, out,
              err) == 0 &&
              Contains(out, "Evidence: estimated from the file header") &&
              Contains(out, "Fit is not verified."),
          "human estimate: " + out + err);
  // Machine-mode list is one event with the envelope.
  const auto root = (base / "cmd-root").string();
  const auto mine = (base / "my-models").string();
  Require(run({"list", "--json", "--models-root", root.c_str(), "--scan-folder",
               mine.c_str()},
              out, err) == 0,
          "list --json exit");
  Require(std::count(out.begin(), out.end(), '\n') == 1 &&
              Contains(out, "\"type\":\"models\"") &&
              Contains(out, "\"schema_version\":1") &&
              Contains(out, "\"catalogue\"") && Contains(out, "mine.gguf"),
          "list --json: one models event");
  {
    const auto event = gufo::json::parse(out);
    const auto& catalogue = *event.find("payload")->find("catalogue");
    Require(catalogue.size() == 11, "models list --json ships eleven entries");
    unsigned lowmemory_rows = 0;
    for (const auto& row : catalogue.items()) {
      Require(row.member_str("id") != "huihui-qwen38-flash-next-ista-hybrid-v1",
              "retired catalogue id is absent from models list");
      if (row.member_str("id") != "huihui-qwen38-flash-next-lowmemory-v1")
        continue;
      ++lowmemory_rows;
      Require(
          row.member_str("role") == "supported" &&
              row.member_str("default_status") == "none" &&
              row.find("speed")->is_null() &&
              row.find("estimate")->member_str("estimate_state") == "estimated",
          "lowmemory JSON row has a calibrated estimate but no global default "
          "or speed");
    }
    Require(lowmemory_rows == 1, "models list --json has one lowmemory entry");
    Require(run({"estimate", "--model", "huihui-qwen38-flash-next-lowmemory-v1",
                 "--json"},
                out, err) == 0,
            "models estimate lowmemory --json exit");
    const auto estimate = gufo::json::parse(out);
    const auto& payload = *estimate.find("payload");
    Require(payload.member_str("estimate_state") == "estimated" &&
                !payload.find("memory_estimate")->is_null(),
            "models estimate lowmemory --json returns the calibrated estimate");
  }
  Require(
      run({"estimate", "--model", dense.c_str(), "--json"}, out, err) == 0 &&
          Contains(out, "\"type\":\"model_estimate\"") &&
          Contains(out, "\"estimate_state\":\"estimated\""),
      "estimate --json: one model_estimate event");
  Require(run({"estimate", "--model", "gemma4-26b-a4b-unsloth-qat-ud-q4-k-xl",
               "--json"},
              out, err) == 0 &&
              Contains(out, "\"estimate_state\":\"none_yet\"") &&
              Contains(out, "\"subject_kind\":\"catalogue\""),
          "catalogue id estimate --json");
  Require(
      run({"estimate", "--model", "nonexistent", "--json"}, out, err) == 12 &&
          Contains(out, "\"type\":\"error\"") && Contains(out, "model_missing"),
      "unknown model: typed error, code 12");
}
}  // namespace

// --write-fixtures DIR: writes the three example events (one JSON line each)
// from the synthetic data above. The checked-in copies are these files with
// the temp folder name and the run-time envelope values replaced.
void WriteFixtures(const fs::path& base, const Value& list,
                   const fs::path& out) {
  fs::create_directories(out);
  const auto write = [&](const char* name, const char* type, Value payload) {
    std::ostringstream text;
    {
      MachineWriter writer(text);
      writer.Write(type, std::move(payload));
    }
    std::ofstream(out / name, std::ios::binary) << text.str();
  };
  write("models.json", "models", list);
  EstimateRequest request;
  request.available_bytes = 64 * kGiB;
  request.host_available_bytes = 128 * kGiB;
  auto estimated = EstimateJson(
      EstimateForGguf(InspectGguf(base / "gguf" / "dense.gguf"), request));
  estimated["subject"] = (base / "gguf" / "dense.gguf").string();
  estimated["subject_kind"] = "path";
  write("model-estimate.json", "model_estimate", estimated);
  const auto& shipped = ShippedCatalogue();
  const auto* gemma = FindCatalogueEntry(
      shipped.entries, "gemma4-26b-a4b-huihui-abliterated-qat-q4-k");
  auto none =
      EstimateJson(EstimateForCatalogueEntry(*gemma, base / "models", request));
  none["subject"] = gemma->id;
  none["subject_kind"] = "catalogue";
  write("model-estimate-none.json", "model_estimate", none);
  const auto* flash = FindCatalogueEntry(shipped.entries, prepare::kPackageId);
  auto table =
      EstimateJson(EstimateForCatalogueEntry(*flash, base / "models", request));
  table["subject"] = flash->id;
  table["subject_kind"] = "catalogue";
  write("model-estimate-flash-next.json", "model_estimate", table);
  // The full shipped catalogue, as `bubo models list` prints it (no files
  // scanned): used by the GUI team and by the schema check.
  ModelsListOptions options;
  options.models_root = base / "no-models";
  options.available_bytes = 64 * kGiB;
  options.host_available_bytes = 128 * kGiB;
  write("models-shipped-catalogue.json", "models",
        BuildModelsList(options, shipped.entries));
}

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--measured-data") {
      MeasuredAdmission(argv[2]);
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--measured-manifest") {
      MeasuredAdmission(argv[3], argv[2]);
      return 0;
    }
    const auto base = MakeBase();
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage Scanner\n";
    Scanner(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage Support\n";
    Support(base);
    EstimateConsistency(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage Estimates\n";
    Estimates(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage List\n";
    const auto list = List(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage CatalogueVision\n";
    CatalogueVision(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage Packages\n";
    Packages(base);
    GemmaMtp(base);
    if (std::getenv("BUBO_TEST_TRACE"))
      std::cerr << "stage Command\n";
    Command(base);
    if (argc == 3 && std::string_view(argv[1]) == "--write-fixtures")
      WriteFixtures(base, list, argv[2]);
    std::error_code ec;
    fs::remove_all(base, ec);
    std::cout << "models_test: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "models_test FAILED: " << error.what() << '\n';
    return 1;
  }
}
