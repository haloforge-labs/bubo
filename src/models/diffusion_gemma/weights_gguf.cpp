// DiffusionGemma weights from a llama.cpp Q8_0 GGUF ("diffusion-gemma"
// architecture). Tensor layouts match the safetensors checkpoint (GGUF
// dimensions are listed innermost first). Expert matrices stay 8-bit on the
// device, split into int8 values and FP16 block scales; every other Q8_0
// matrix is dequantized to BF16 for the dense products, and F32 vectors are
// rounded to BF16 like the checkpoint's own.
#include <hip/hip_runtime_api.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/upload.hpp"
#include "src/models/diffusion_gemma/weights.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

constexpr std::size_t kQ8Block = 34;  // FP16 scale + 32 int8.
/// Device staging for tensors awaiting conversion (raised to the largest).
constexpr std::size_t kConversionStaging = 512U << 20;

std::size_t Align(std::size_t x) {
  return (x + 255) & ~std::size_t{255};
}

enum class Op { kDequant, kRepack, kToBf16 };

struct Step {
  const core::GgufTensorInfo* tensor;
  Op op;
  std::size_t destination;  ///< BF16 or int8 values.
  std::size_t scales{0};    ///< kRepack only.
};

}  // namespace

std::filesystem::path FindGgufWeights(const std::filesystem::path& root) {
  std::error_code ec;
  if (std::filesystem::exists(root / "model.safetensors.index.json", ec) ||
      std::filesystem::exists(root / "model.safetensors", ec))
    return {};
  std::filesystem::path found;
  for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
    if (entry.path().extension() == ".gguf") {
      if (!found.empty())
        throw std::runtime_error(root.string() +
                                 " holds more than one .gguf file");
      found = entry.path();
    }
  }
  return found;
}

void DeviceWeights::LoadGguf(const std::filesystem::path& file,
                             const Config& c) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(file, &error);
  if (!reader)
    throw std::runtime_error("cannot open " + file.string() + ": " + error);
  if (reader->GetMetadataString("general.architecture").value_or("") !=
      "diffusion-gemma")
    throw std::runtime_error(file.string() + " is not a diffusion-gemma GGUF");
  const std::size_t H = c.hidden, I = c.intermediate, E = c.experts,
                    EI = c.expert_intermediate;

  std::vector<Step> steps;
  std::size_t size = 0;
  const auto find = [&](const std::string& name,
                        std::uint64_t elements) -> const core::GgufTensorInfo* {
    const auto* t = reader->FindTensor(name);
    if (t == nullptr)
      throw std::runtime_error("GGUF is missing tensor " + name);
    if (t->ElementCount() != elements)
      throw std::runtime_error(
          "GGUF tensor " + name + " has " + std::to_string(t->ElementCount()) +
          " elements, expected " + std::to_string(elements));
    return t;
  };
  // Dense: Q8_0 dequantized (or F32 rounded) to BF16, back to back.
  const auto place =
      [&](const std::vector<std::pair<std::string, std::uint64_t>>& parts) {
        const std::size_t offset = Align(size);
        std::size_t cursor = offset;
        for (const auto& [name, elements] : parts) {
          const auto* t = find(name, elements);
          Op op;
          if (t->type == core::GgmlType::kQ8_0)
            op = Op::kDequant;
          else if (t->type == core::GgmlType::kF32)
            op = Op::kToBf16;
          else
            throw std::runtime_error("GGUF tensor " + name +
                                     ": unsupported type (want Q8_0 or F32)");
          steps.push_back(Step{t, op, cursor});
          cursor += elements * 2;
        }
        size = cursor;
        return offset;
      };
  // Experts: int8 values and FP16 scales.
  const auto place_q8 = [&](const std::string& name, std::uint64_t elements,
                            std::size_t* scales) {
    const auto* t = find(name, elements);
    if (t->type != core::GgmlType::kQ8_0)
      throw std::runtime_error("GGUF tensor " + name + " is not Q8_0");
    const std::size_t values = Align(size);
    *scales = Align(values + elements);
    size = *scales + elements / 32 * 2;
    steps.push_back(Step{t, Op::kRepack, values, *scales});
    return values;
  };

  const std::size_t embed = place({{"token_embd.weight", 1ULL * c.vocab * H}});
  const std::size_t final_norm = place({{"output_norm.weight", H}});
  const std::size_t sc_pre = place({{"self_cond_pre_norm.weight", H}});
  const std::size_t sc_gate_up =
      place({{"self_cond_gate.weight", I * H}, {"self_cond_up.weight", I * H}});
  const std::size_t sc_down = place({{"self_cond_down.weight", H * I}});

  struct Offsets {
    std::size_t qkv, o, q_norm, k_norm, input_norm, post_attention_norm,
        pre_ff_norm, pre_ff_norm_2, post_ff_norm, post_ff_norm_1,
        post_ff_norm_2, mlp_gate_up, mlp_down, router_proj, router_scale,
        per_expert_scale, gate_up_q8, gate_up_scale, down_q8, down_scale;
  };
  std::vector<Offsets> offsets(c.layers);
  weights_.layers.assign(c.layers, LayerWeights{});
  for (int l = 0; l < c.layers; ++l) {
    const std::string L = "blk." + std::to_string(l) + ".";
    const std::uint64_t hd = c.head_dim(l), kv = c.kv_heads(l);
    auto& o = offsets[l];
    std::vector<std::pair<std::string, std::uint64_t>> qkv = {
        {L + "attn_q.weight", c.heads * hd * H},
        {L + "attn_k.weight", kv * hd * H}};
    if (!c.is_global[l])
      qkv.push_back({L + "attn_v.weight", kv * hd * H});
    o.qkv = place(qkv);
    o.o = place({{L + "attn_output.weight", H * c.heads * hd}});
    o.q_norm = place({{L + "attn_q_norm.weight", hd}});
    o.k_norm = place({{L + "attn_k_norm.weight", hd}});
    o.input_norm = place({{L + "attn_norm.weight", H}});
    o.post_attention_norm = place({{L + "post_attention_norm.weight", H}});
    o.pre_ff_norm = place({{L + "ffn_norm.weight", H}});
    o.pre_ff_norm_2 = place({{L + "pre_ffw_norm_2.weight", H}});
    o.post_ff_norm = place({{L + "post_ffw_norm.weight", H}});
    o.post_ff_norm_1 = place({{L + "post_ffw_norm_1.weight", H}});
    o.post_ff_norm_2 = place({{L + "post_ffw_norm_2.weight", H}});
    o.mlp_gate_up =
        place({{L + "ffn_gate.weight", I * H}, {L + "ffn_up.weight", I * H}});
    o.mlp_down = place({{L + "ffn_down.weight", H * I}});
    o.router_proj = place({{L + "ffn_gate_inp.weight", E * H}});
    o.router_scale = place({{L + "ffn_gate_inp.scale", H}});
    o.per_expert_scale = place({{L + "ffn_down_exps.scale", E}});
    o.gate_up_q8 = place_q8(L + "ffn_gate_up_exps.weight", E * 2 * EI * H,
                            &o.gate_up_scale);
    o.down_q8 = place_q8(L + "ffn_down_exps.weight", E * H * EI, &o.down_scale);
    const auto* scalar = find(L + "layer_output_scale.weight", 1);
    if (scalar->type != core::GgmlType::kF32)
      throw std::runtime_error("layer_output_scale is not F32");
    std::memcpy(&weights_.layers[l].layer_scalar, scalar->data, 4);
  }

  HipCheck(hipMalloc(&arena_, size), "allocating the weight arena");
  weights_.device_bytes = size;
  weights_.format = "Q8_0 experts";
  // Every tensor is converted on the device: Q8_0 to BF16 (dense) or to
  // separate int8 values and FP16 scales (experts), F32 to BF16.
  const auto* mapped =
      static_cast<const std::uint8_t*>(reader->GetMappedRegions().front().data);
  hipStream_t stream{};
  try {
    HipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
             "GGUF weight stream");
    StagedConversion conversion(kConversionStaging, stream);
    auto* base = static_cast<std::uint8_t*>(arena_);
    for (const auto& step : steps) {
      const auto* t = step.tensor;
      const std::size_t elements = t->ElementCount();
      if (step.op != Op::kToBf16 && t->size_bytes != elements / 32 * kQ8Block)
        throw std::runtime_error("unexpected Q8_0 size");
      const FileRange range{
          file,
          static_cast<std::uint64_t>(static_cast<const std::uint8_t*>(t->data) -
                                     mapped),
          t->size_bytes};
      conversion.Add({range}, [&, step, elements](
                                  std::span<const std::uint8_t* const> staged) {
        switch (step.op) {
          case Op::kDequant:
            kernels::DequantQ8(
                staged[0], reinterpret_cast<bf16_t*>(base + step.destination),
                elements, stream);
            break;
          case Op::kRepack:
            kernels::RepackQ8(
                staged[0],
                reinterpret_cast<std::int8_t*>(base + step.destination),
                reinterpret_cast<std::uint16_t*>(base + step.scales),
                elements / 32, stream);
            break;
          case Op::kToBf16:
            kernels::F32ToBf16(
                reinterpret_cast<const float*>(staged[0]),
                reinterpret_cast<bf16_t*>(base + step.destination), elements,
                stream);
            break;
        }
      });
    }
    conversion.Flush();
  } catch (...) {
    if (stream) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    (void)hipFree(arena_);
    arena_ = nullptr;
    throw;
  }
  (void)hipStreamDestroy(stream);

  auto* base = static_cast<std::uint8_t*>(arena_);
  const auto bf = [base](std::size_t off) {
    return reinterpret_cast<const bf16_t*>(base + off);
  };
  weights_.embed = bf(embed);
  weights_.final_norm = bf(final_norm);
  weights_.sc_pre_norm = bf(sc_pre);
  weights_.sc_gate_up = bf(sc_gate_up);
  weights_.sc_down = bf(sc_down);
  for (int l = 0; l < c.layers; ++l) {
    const auto& o = offsets[l];
    auto& w = weights_.layers[l];
    w.qkv = bf(o.qkv);
    w.o = bf(o.o);
    w.q_norm = bf(o.q_norm);
    w.k_norm = bf(o.k_norm);
    w.input_norm = bf(o.input_norm);
    w.post_attention_norm = bf(o.post_attention_norm);
    w.pre_ff_norm = bf(o.pre_ff_norm);
    w.pre_ff_norm_2 = bf(o.pre_ff_norm_2);
    w.post_ff_norm = bf(o.post_ff_norm);
    w.post_ff_norm_1 = bf(o.post_ff_norm_1);
    w.post_ff_norm_2 = bf(o.post_ff_norm_2);
    w.mlp_gate_up = bf(o.mlp_gate_up);
    w.mlp_down = bf(o.mlp_down);
    w.router_proj = bf(o.router_proj);
    w.router_scale = bf(o.router_scale);
    w.per_expert_scale = bf(o.per_expert_scale);
    w.expert_gate_up_q8 =
        reinterpret_cast<const std::int8_t*>(base + o.gate_up_q8);
    w.expert_gate_up_scale =
        reinterpret_cast<const std::uint16_t*>(base + o.gate_up_scale);
    w.expert_down_q8 = reinterpret_cast<const std::int8_t*>(base + o.down_q8);
    w.expert_down_scale =
        reinterpret_cast<const std::uint16_t*>(base + o.down_scale);
  }
}

}  // namespace gufo::models::diffusion_gemma
