#include "src/models/diffusion_gemma/weights.hpp"

#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/upload.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

/// One contiguous source range copied to an arena offset.
struct Copy {
  std::filesystem::path file;
  std::uint64_t source{0};
  std::uint64_t bytes{0};
  std::size_t destination{0};
};

class Planner {
public:
  Planner(const std::unordered_map<std::string, TensorEntry>& index)
      : index_(index) {}

  /// Reserves one arena slot holding `names` back to back; returns its offset.
  std::size_t Place(const std::vector<std::string>& names,
                    const std::vector<std::int64_t>& expected_rows,
                    std::int64_t expected_cols) {
    const std::size_t offset = Align(size_);
    std::size_t cursor = offset;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto& t = Find(names[i]);
      if (t.dtype != "BF16")
        throw std::runtime_error(names[i] + ": expected BF16, found " +
                                 t.dtype);
      std::int64_t elements = 1;
      for (const auto d : t.shape)
        elements *= d;
      if (i < expected_rows.size() &&
          elements != expected_rows[i] * expected_cols)
        throw std::runtime_error(names[i] + ": unexpected shape");
      copies_.push_back(Copy{t.file, t.offset, t.bytes, cursor});
      cursor += t.bytes;
    }
    size_ = cursor;
    return offset;
  }
  const TensorEntry& Find(const std::string& name) const {
    const auto found = index_.find(name);
    if (found == index_.end())
      throw std::runtime_error("checkpoint is missing tensor " + name);
    return found->second;
  }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::vector<Copy>& copies() { return copies_; }

private:
  static std::size_t Align(std::size_t x) {
    return (x + 255) & ~std::size_t{255};
  }
  const std::unordered_map<std::string, TensorEntry>& index_;
  std::vector<Copy> copies_;
  std::size_t size_{0};
};

}  // namespace

float ReadF32Scalar(const TensorEntry& t) {
  const int fd = ::open(t.file.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + t.file.string());
  float value = 0;
  const auto got = ::pread(fd, &value, 4, static_cast<off_t>(t.offset));
  ::close(fd);
  if (got != 4)
    throw std::runtime_error("short read of a tensor scale");
  return value;
}

float ReadBf16Scalar(const TensorEntry& t) {
  const int fd = ::open(t.file.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + t.file.string());
  std::uint16_t raw = 0;
  const auto got = ::pread(fd, &raw, 2, static_cast<off_t>(t.offset));
  ::close(fd);
  if (got != 2)
    throw std::runtime_error("short read of a layer scalar");
  const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

std::unordered_map<std::string, TensorEntry> ReadSafetensorsIndex(
    const std::filesystem::path& root) {
  std::unordered_map<std::string, TensorEntry> index;
  std::vector<std::string> shards;
  const auto index_path = root / "model.safetensors.index.json";
  if (std::filesystem::exists(index_path)) {
    const auto doc = json::parse(ReadFile(index_path));
    std::map<std::string, bool> unique;
    for (const auto& [name, file] : doc.find("weight_map")->members())
      unique[file.get_str()] = true;
    for (const auto& [file, unused] : unique)
      shards.push_back(file);
  } else {
    shards.push_back("model.safetensors");
  }
  for (const auto& shard : shards) {
    const auto path = root / shard;
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot open " + path.string());
    std::uint64_t header_bytes = 0;
    in.read(reinterpret_cast<char*>(&header_bytes), 8);
    std::string header(header_bytes, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!in)
      throw std::runtime_error("truncated safetensors header in " + shard);
    const auto doc = json::parse(header);
    for (const auto& [name, value] : doc.members()) {
      if (name == "__metadata__")
        continue;
      TensorEntry t;
      t.file = path;
      t.dtype = value.find("dtype")->get_str();
      for (const auto& d : value.find("shape")->items())
        t.shape.push_back(static_cast<std::int64_t>(d.as_size()));
      const auto& range = value.find("data_offsets")->items();
      const auto begin = static_cast<std::uint64_t>(range[0].as_size());
      const auto end = static_cast<std::uint64_t>(range[1].as_size());
      t.offset = 8 + header_bytes + begin;
      t.bytes = end - begin;
      index.emplace(name, std::move(t));
    }
  }
  return index;
}

DeviceWeights::DeviceWeights(const std::filesystem::path& root,
                             const Config& c) {
  if (const auto gguf = FindGgufWeights(root); !gguf.empty()) {
    LoadGguf(gguf, c);
  } else {
    const auto index = ReadSafetensorsIndex(root);
    if (index.contains(
            "model.decoder.layers.0.experts.0.gate_proj.weight_scale_2"))
      LoadNvfp4(index, c);
    else if (index.contains(
                 "model.decoder.layers.0.experts.0.gate_proj.weight"))
      LoadFp8(index, c);
    else
      LoadBf16(index, c);
  }
  try {
    BlockDenseMatrices(c);
  } catch (...) {
    (void)hipFree(arena_);
    arena_ = nullptr;
    throw;
  }
}

void DeviceWeights::BlockDenseMatrices(const Config& c) {
  struct Matrix {
    const bf16_t* data;
    int rows, cols;
    bool gate_up;  // For kernels::LinearGeluMul.
  };
  std::vector<Matrix> matrices = {
      {weights_.sc_gate_up, 2 * c.intermediate, c.hidden, true},
      {weights_.sc_down, c.hidden, c.intermediate, false}};
  for (int l = 0; l < c.layers; ++l) {
    const auto& w = weights_.layers[l];
    const int hd = c.head_dim(l), kv = c.kv_heads(l);
    const int qkv_rows = (c.heads + (c.is_global[l] ? 1 : 2) * kv) * hd;
    matrices.push_back({w.qkv, qkv_rows, c.hidden, false});
    matrices.push_back({w.o, c.hidden, c.heads * hd, false});
    matrices.push_back({w.mlp_gate_up, 2 * c.intermediate, c.hidden, true});
    matrices.push_back({w.mlp_down, c.hidden, c.intermediate, false});
    matrices.push_back({w.router_proj, c.experts, c.hidden, false});
  }
  std::size_t largest = 0;
  for (const auto& m : matrices)
    largest = std::max(largest, std::size_t(m.rows) * m.cols * 2);
  void* scratch = nullptr;
  HipCheck(hipMalloc(&scratch, largest), "allocating blocking scratch");
  try {
    for (const auto& m : matrices) {
      // The arena is this object's own allocation; the view is const only
      // for the engine.
      auto* data = const_cast<bf16_t*>(m.data);
      HipCheck(hipMemcpy(scratch, data, std::size_t(m.rows) * m.cols * 2,
                         hipMemcpyDeviceToDevice),
               "copying a matrix to block");
      if (m.gate_up)
        kernels::BlockGateUpWeights(static_cast<const bf16_t*>(scratch), data,
                                    m.rows / 2, m.cols, nullptr);
      else
        kernels::BlockLinearWeights(static_cast<const bf16_t*>(scratch), data,
                                    m.rows, m.cols, nullptr);
      HipCheck(hipStreamSynchronize(nullptr), "blocking a matrix");
    }
  } catch (...) {
    (void)hipFree(scratch);
    throw;
  }
  HipCheck(hipFree(scratch), "freeing blocking scratch");
}

void DeviceWeights::LoadBf16(
    const std::unordered_map<std::string, TensorEntry>& index,
    const Config& c) {
  Planner plan(index);
  const std::string p = "model.decoder.";
  const std::int64_t h = c.hidden;

  struct LayerOffsets {
    std::size_t qkv, o, q_norm, k_norm, input_norm, post_attention_norm,
        pre_ff_norm, pre_ff_norm_2, post_ff_norm, post_ff_norm_1,
        post_ff_norm_2, mlp_gate_up, mlp_down, router_proj, router_scale,
        per_expert_scale, expert_gate_up, expert_down;
  };
  const std::size_t embed =
      plan.Place({p + "embed_tokens.weight"}, {c.vocab}, h);
  const std::size_t final_norm = plan.Place({p + "norm.weight"}, {1}, h);
  const std::string sc = p + "self_conditioning.";
  const std::size_t sc_pre = plan.Place({sc + "pre_norm.weight"}, {1}, h);
  const std::size_t sc_gate_up =
      plan.Place({sc + "gate_proj.weight", sc + "up_proj.weight"},
                 {c.intermediate, c.intermediate}, h);
  const std::size_t sc_down =
      plan.Place({sc + "down_proj.weight"}, {h}, c.intermediate);

  std::vector<LayerOffsets> offsets(c.layers);
  weights_.layers.resize(c.layers);
  for (int l = 0; l < c.layers; ++l) {
    const std::string L = p + "layers." + std::to_string(l) + ".";
    const std::int64_t hd = c.head_dim(l), kv = c.kv_heads(l);
    auto& o = offsets[l];
    if (c.is_global[l]) {
      o.qkv = plan.Place(
          {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight"},
          {c.heads * hd, kv * hd}, h);
    } else {
      o.qkv = plan.Place(
          {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight",
           L + "self_attn.v_proj.weight"},
          {c.heads * hd, kv * hd, kv * hd}, h);
    }
    o.o = plan.Place({L + "self_attn.o_proj.weight"}, {h}, c.heads * hd);
    o.q_norm = plan.Place({L + "self_attn.q_norm.weight"}, {1}, hd);
    o.k_norm = plan.Place({L + "self_attn.k_norm.weight"}, {1}, hd);
    o.input_norm = plan.Place({L + "input_layernorm.weight"}, {1}, h);
    o.post_attention_norm =
        plan.Place({L + "post_attention_layernorm.weight"}, {1}, h);
    o.pre_ff_norm =
        plan.Place({L + "pre_feedforward_layernorm.weight"}, {1}, h);
    o.pre_ff_norm_2 =
        plan.Place({L + "pre_feedforward_layernorm_2.weight"}, {1}, h);
    o.post_ff_norm =
        plan.Place({L + "post_feedforward_layernorm.weight"}, {1}, h);
    o.post_ff_norm_1 =
        plan.Place({L + "post_feedforward_layernorm_1.weight"}, {1}, h);
    o.post_ff_norm_2 =
        plan.Place({L + "post_feedforward_layernorm_2.weight"}, {1}, h);
    o.mlp_gate_up =
        plan.Place({L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"},
                   {c.intermediate, c.intermediate}, h);
    o.mlp_down = plan.Place({L + "mlp.down_proj.weight"}, {h}, c.intermediate);
    o.router_proj = plan.Place({L + "router.proj.weight"}, {c.experts}, h);
    o.router_scale = plan.Place({L + "router.scale"}, {1}, h);
    o.per_expert_scale =
        plan.Place({L + "router.per_expert_scale"}, {1}, c.experts);
    o.expert_gate_up =
        plan.Place({L + "experts.gate_up_proj"},
                   {std::int64_t{c.experts} * 2 * c.expert_intermediate}, h);
    o.expert_down =
        plan.Place({L + "experts.down_proj"}, {std::int64_t{c.experts} * h},
                   c.expert_intermediate);
    weights_.layers[l].layer_scalar =
        ReadBf16Scalar(plan.Find(L + "layer_scalar"));
  }

  const std::size_t total = plan.size();
  HipCheck(hipMalloc(&arena_, total), "allocating the weight arena");
  weights_.device_bytes = total;
  auto* base = static_cast<std::uint8_t*>(arena_);
  const auto at = [base](std::size_t offset) {
    return reinterpret_cast<const bf16_t*>(base + offset);
  };
  weights_.embed = at(embed);
  weights_.final_norm = at(final_norm);
  weights_.sc_pre_norm = at(sc_pre);
  weights_.sc_gate_up = at(sc_gate_up);
  weights_.sc_down = at(sc_down);
  for (int l = 0; l < c.layers; ++l) {
    const auto& o = offsets[l];
    auto& w = weights_.layers[l];
    w.qkv = at(o.qkv);
    w.o = at(o.o);
    w.q_norm = at(o.q_norm);
    w.k_norm = at(o.k_norm);
    w.input_norm = at(o.input_norm);
    w.post_attention_norm = at(o.post_attention_norm);
    w.pre_ff_norm = at(o.pre_ff_norm);
    w.pre_ff_norm_2 = at(o.pre_ff_norm_2);
    w.post_ff_norm = at(o.post_ff_norm);
    w.post_ff_norm_1 = at(o.post_ff_norm_1);
    w.post_ff_norm_2 = at(o.post_ff_norm_2);
    w.mlp_gate_up = at(o.mlp_gate_up);
    w.mlp_down = at(o.mlp_down);
    w.router_proj = at(o.router_proj);
    w.router_scale = at(o.router_scale);
    w.per_expert_scale = at(o.per_expert_scale);
    w.expert_gate_up = at(o.expert_gate_up);
    w.expert_down = at(o.expert_down);
  }

  std::vector<FileRange> ranges;
  std::vector<void*> destinations;
  for (const auto& copy : plan.copies()) {
    ranges.push_back({copy.file, copy.source, copy.bytes});
    destinations.push_back(base + copy.destination);
  }
  try {
    UploadFileRanges(ranges, destinations);
  } catch (...) {
    (void)hipFree(arena_);
    arena_ = nullptr;
    throw;
  }
}

DeviceWeights::~DeviceWeights() {
  if (arena_ != nullptr)
    (void)hipFree(arena_);
}

}  // namespace gufo::models::diffusion_gemma
