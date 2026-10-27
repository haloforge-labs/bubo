// DiffusionGemma weights from a compressed-tensors FP8 checkpoint
// (llm-compressor FP8_DYNAMIC: E4M3 weights with one BF16 scale per output
// channel; activation quantization is not applied here). Experts stay FP8 on
// the device in the BF16 checkpoint's fused layout; every other FP8 matrix is
// dequantized to BF16 for the dense products; BF16 tensors are copied as
// they are.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/upload.hpp"
#include "src/models/diffusion_gemma/weights.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

std::size_t Align(std::size_t x) {
  return (x + 255) & ~std::size_t{255};
}

/// Device staging for FP8 matrices awaiting dequantization.
constexpr std::size_t kConversionStaging = 512U << 20;

struct Step {
  const TensorEntry* tensor;
  std::size_t destination;
  const TensorEntry* scale{nullptr};  ///< Set: dequantize FP8 rows to BF16.
  std::size_t rows{0}, cols{0};
};

}  // namespace

void DeviceWeights::LoadFp8(
    const std::unordered_map<std::string, TensorEntry>& index,
    const Config& c) {
  const std::size_t H = c.hidden, I = c.intermediate, E = c.experts,
                    EI = c.expert_intermediate;
  const auto find = [&](const std::string& name, const char* dtype,
                        std::size_t elements) -> const TensorEntry& {
    const auto found = index.find(name);
    if (found == index.end())
      throw std::runtime_error("checkpoint is missing tensor " + name);
    const auto& t = found->second;
    std::size_t count = 1;
    for (const auto d : t.shape)
      count *= static_cast<std::size_t>(d);
    if (t.dtype != dtype || count != elements)
      throw std::runtime_error(name + ": expected " + dtype + " with " +
                               std::to_string(elements) + " elements");
    return t;
  };

  std::vector<Step> steps;
  std::size_t size = 0;
  // BF16 tensors back to back.
  const auto copy =
      [&](const std::vector<std::pair<std::string, std::size_t>>& parts) {
        const std::size_t offset = Align(size);
        std::size_t cursor = offset;
        for (const auto& [name, elements] : parts) {
          steps.push_back(Step{&find(name, "BF16", elements), cursor});
          cursor += elements * 2;
        }
        size = cursor;
        return offset;
      };
  // FP8 matrices (rows x cols, one scale per row) dequantized to BF16, back
  // to back.
  const auto dequant =
      [&](const std::vector<std::pair<std::string, std::size_t>>& parts,
          std::size_t cols) {
        const std::size_t offset = Align(size);
        std::size_t cursor = offset;
        for (const auto& [name, rows] : parts) {
          const auto& values = find(name + ".weight", "F8_E4M3", rows * cols);
          const auto& scale = find(name + ".weight_scale", "BF16", rows);
          steps.push_back(Step{&values, cursor, &scale, rows, cols});
          cursor += rows * cols * 2;
        }
        size = cursor;
        return offset;
      };

  const std::string p = "model.decoder.";
  const std::size_t embed = copy({{p + "embed_tokens.weight", c.vocab * H}});
  const std::size_t final_norm = copy({{p + "norm.weight", H}});
  const std::string sc = p + "self_conditioning.";
  const std::size_t sc_pre = copy({{sc + "pre_norm.weight", H}});
  const std::size_t sc_gate_up =
      copy({{sc + "gate_proj.weight", I * H}, {sc + "up_proj.weight", I * H}});
  const std::size_t sc_down = copy({{sc + "down_proj.weight", H * I}});

  struct Offsets {
    std::size_t qkv, o, q_norm, k_norm, input_norm, post_attention_norm,
        pre_ff_norm, pre_ff_norm_2, post_ff_norm, post_ff_norm_1,
        post_ff_norm_2, mlp_gate_up, mlp_down, router_proj, router_scale,
        per_expert_scale, gate_up, gate_up_scale, down, down_scale;
  };
  std::vector<Offsets> offsets(c.layers);
  weights_.layers.assign(c.layers, LayerWeights{});
  for (int l = 0; l < c.layers; ++l) {
    const std::string L = p + "layers." + std::to_string(l) + ".";
    const std::size_t hd = c.head_dim(l), kv = c.kv_heads(l);
    auto& o = offsets[l];
    std::vector<std::pair<std::string, std::size_t>> qkv = {
        {L + "self_attn.q_proj", c.heads * hd},
        {L + "self_attn.k_proj", kv * hd}};
    if (!c.is_global[l])
      qkv.push_back({L + "self_attn.v_proj", kv * hd});
    o.qkv = dequant(qkv, H);
    o.o = dequant({{L + "self_attn.o_proj", H}}, c.heads * hd);
    o.q_norm = copy({{L + "self_attn.q_norm.weight", hd}});
    o.k_norm = copy({{L + "self_attn.k_norm.weight", hd}});
    o.input_norm = copy({{L + "input_layernorm.weight", H}});
    o.post_attention_norm = copy({{L + "post_attention_layernorm.weight", H}});
    o.pre_ff_norm = copy({{L + "pre_feedforward_layernorm.weight", H}});
    o.pre_ff_norm_2 = copy({{L + "pre_feedforward_layernorm_2.weight", H}});
    o.post_ff_norm = copy({{L + "post_feedforward_layernorm.weight", H}});
    o.post_ff_norm_1 = copy({{L + "post_feedforward_layernorm_1.weight", H}});
    o.post_ff_norm_2 = copy({{L + "post_feedforward_layernorm_2.weight", H}});
    o.mlp_gate_up =
        dequant({{L + "mlp.gate_proj", I}, {L + "mlp.up_proj", I}}, H);
    o.mlp_down = dequant({{L + "mlp.down_proj", H}}, I);
    o.router_proj = copy({{L + "router.proj.weight", E * H}});
    o.router_scale = copy({{L + "router.scale", H}});
    o.per_expert_scale = copy({{L + "router.per_expert_scale", E}});
    // Experts: FP8 values and BF16 row scales in the fused layout, expert
    // by expert (gate rows, then up rows).
    o.gate_up = Align(size);
    o.gate_up_scale = Align(o.gate_up + E * 2 * EI * H);
    o.down = Align(o.gate_up_scale + E * 2 * EI * 2);
    o.down_scale = Align(o.down + E * H * EI);
    size = o.down_scale + E * H * 2;
    for (std::size_t e = 0; e < E; ++e) {
      const std::string X = L + "experts." + std::to_string(e) + ".";
      const std::size_t gate_row = e * 2 * EI, up_row = gate_row + EI;
      steps.push_back(Step{&find(X + "gate_proj.weight", "F8_E4M3", EI * H),
                           o.gate_up + gate_row * H});
      steps.push_back(Step{&find(X + "up_proj.weight", "F8_E4M3", EI * H),
                           o.gate_up + up_row * H});
      steps.push_back(Step{&find(X + "gate_proj.weight_scale", "BF16", EI),
                           o.gate_up_scale + gate_row * 2});
      steps.push_back(Step{&find(X + "up_proj.weight_scale", "BF16", EI),
                           o.gate_up_scale + up_row * 2});
      steps.push_back(Step{&find(X + "down_proj.weight", "F8_E4M3", H * EI),
                           o.down + e * H * EI});
      steps.push_back(Step{&find(X + "down_proj.weight_scale", "BF16", H),
                           o.down_scale + e * H * 2});
    }
    weights_.layers[l].layer_scalar =
        ReadBf16Scalar(find(L + "layer_scalar", "BF16", 1));
  }

  HipCheck(hipMalloc(&arena_, size), "allocating the weight arena");
  weights_.device_bytes = size;
  weights_.format = "FP8 experts";
  auto* base = static_cast<std::uint8_t*>(arena_);

  // BF16 tensors and FP8 experts are copied as stored; dense FP8 matrices
  // are dequantized to BF16 through device staging.
  std::vector<FileRange> ranges;
  std::vector<void*> destinations;
  hipStream_t stream{};
  try {
    HipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
             "FP8 weight stream");
    StagedConversion conversion(kConversionStaging, stream);
    for (const auto& step : steps) {
      const auto& t = *step.tensor;
      if (step.scale == nullptr) {
        ranges.push_back({t.file, t.offset, t.bytes});
        destinations.push_back(base + step.destination);
        continue;
      }
      conversion.Add(
          {{t.file, t.offset, t.bytes},
           {step.scale->file, step.scale->offset, step.scale->bytes}},
          [&, step](std::span<const std::uint8_t* const> staged) {
            kernels::DequantFp8Rows(
                staged[0], reinterpret_cast<const bf16_t*>(staged[1]),
                reinterpret_cast<bf16_t*>(base + step.destination), step.rows,
                step.cols, stream);
          });
    }
    UploadFileRanges(ranges, destinations);
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
    w.expert_gate_up_fp8 = base + o.gate_up;
    w.expert_gate_up_fp8_scale = bf(o.gate_up_scale);
    w.expert_down_fp8 = base + o.down;
    w.expert_down_fp8_scale = bf(o.down_scale);
  }
}

}  // namespace gufo::models::diffusion_gemma
