// DiffusionGemma weights from a ModelOpt NVFP4 checkpoint
// (nvidia/diffusiongemma-26B-A4B-it-NVFP4). Only the experts are quantized:
// E2M1 values packed two per byte with one E4M3 scale per 16 values and one
// FP32 scale per matrix (`weight_scale_2`). They stay 4-bit on the device
// and are widened to BF16 in the expert kernels; activations stay BF16 (the
// checkpoint's FP4 activation scales, `input_scale`, are not applied). Every
// other tensor is BF16 and copied as it is.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/upload.hpp"
#include "src/models/diffusion_gemma/weights.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

std::size_t Align(std::size_t x) {
  return (x + 255) & ~std::size_t{255};
}

}  // namespace

void DeviceWeights::LoadNvfp4(
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

  // Tensors copied as stored, to arena offsets.
  std::vector<std::pair<const TensorEntry*, std::size_t>> copies;
  std::size_t size = 0;
  const auto copy =
      [&](const std::vector<std::pair<std::string, std::size_t>>& parts) {
        const std::size_t offset = Align(size);
        std::size_t cursor = offset;
        for (const auto& [name, elements] : parts) {
          copies.push_back({&find(name, "BF16", elements), cursor});
          cursor += elements * 2;
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
        per_expert_scale, gate_up, gate_up_blocks, gate_up_scale, down,
        down_blocks, down_scale;
  };
  std::vector<Offsets> offsets(c.layers);
  // Per-row FP32 matrix scales, uploaded after the arena is allocated.
  std::vector<std::pair<std::size_t, std::vector<float>>> tensor_scales;
  weights_.layers.assign(c.layers, LayerWeights{});
  for (int l = 0; l < c.layers; ++l) {
    const std::string L = p + "layers." + std::to_string(l) + ".";
    const std::size_t hd = c.head_dim(l), kv = c.kv_heads(l);
    auto& o = offsets[l];
    std::vector<std::pair<std::string, std::size_t>> qkv = {
        {L + "self_attn.q_proj.weight", c.heads * hd * H},
        {L + "self_attn.k_proj.weight", kv * hd * H}};
    if (!c.is_global[l])
      qkv.push_back({L + "self_attn.v_proj.weight", kv * hd * H});
    o.qkv = copy(qkv);
    o.o = copy({{L + "self_attn.o_proj.weight", H * c.heads * hd}});
    o.q_norm = copy({{L + "self_attn.q_norm.weight", hd}});
    o.k_norm = copy({{L + "self_attn.k_norm.weight", hd}});
    o.input_norm = copy({{L + "input_layernorm.weight", H}});
    o.post_attention_norm = copy({{L + "post_attention_layernorm.weight", H}});
    o.pre_ff_norm = copy({{L + "pre_feedforward_layernorm.weight", H}});
    o.pre_ff_norm_2 = copy({{L + "pre_feedforward_layernorm_2.weight", H}});
    o.post_ff_norm = copy({{L + "post_feedforward_layernorm.weight", H}});
    o.post_ff_norm_1 = copy({{L + "post_feedforward_layernorm_1.weight", H}});
    o.post_ff_norm_2 = copy({{L + "post_feedforward_layernorm_2.weight", H}});
    o.mlp_gate_up = copy({{L + "mlp.gate_proj.weight", I * H},
                          {L + "mlp.up_proj.weight", I * H}});
    o.mlp_down = copy({{L + "mlp.down_proj.weight", H * I}});
    o.router_proj = copy({{L + "router.proj.weight", E * H}});
    o.router_scale = copy({{L + "router.scale", H}});
    o.per_expert_scale = copy({{L + "router.per_expert_scale", E}});
    // Experts in the fused layout, expert by expert (gate rows, then up
    // rows): packed values, block scales and per-row matrix scales.
    o.gate_up = Align(size);
    o.gate_up_blocks = Align(o.gate_up + E * 2 * EI * H / 2);
    o.gate_up_scale = Align(o.gate_up_blocks + E * 2 * EI * H / 16);
    o.down = Align(o.gate_up_scale + E * 2 * EI * 4);
    o.down_blocks = Align(o.down + E * H * EI / 2);
    o.down_scale = Align(o.down_blocks + E * H * EI / 16);
    size = o.down_scale + E * H * 4;
    std::vector<float> gate_up_scale(E * 2 * EI), down_scale(E * H);
    for (std::size_t e = 0; e < E; ++e) {
      const std::string X = L + "experts." + std::to_string(e) + ".";
      const std::size_t gate_row = e * 2 * EI, up_row = gate_row + EI;
      const auto expert = [&](const std::string& name, std::size_t row,
                              std::size_t rows, std::size_t cols,
                              std::size_t values, std::size_t blocks,
                              std::vector<float>& scales) {
        copies.push_back({&find(X + name + ".weight", "U8", rows * cols / 2),
                          values + row * cols / 2});
        copies.push_back(
            {&find(X + name + ".weight_scale", "F8_E4M3", rows * cols / 16),
             blocks + row * cols / 16});
        const float scale =
            ReadF32Scalar(find(X + name + ".weight_scale_2", "F32", 1));
        std::fill_n(scales.begin() + static_cast<std::ptrdiff_t>(row), rows,
                    scale);
      };
      expert("gate_proj", gate_row, EI, H, o.gate_up, o.gate_up_blocks,
             gate_up_scale);
      expert("up_proj", up_row, EI, H, o.gate_up, o.gate_up_blocks,
             gate_up_scale);
      expert("down_proj", e * H, H, EI, o.down, o.down_blocks, down_scale);
    }
    tensor_scales.push_back({o.gate_up_scale, std::move(gate_up_scale)});
    tensor_scales.push_back({o.down_scale, std::move(down_scale)});
    weights_.layers[l].layer_scalar =
        ReadBf16Scalar(find(L + "layer_scalar", "BF16", 1));
  }

  HipCheck(hipMalloc(&arena_, size), "allocating the weight arena");
  weights_.device_bytes = size;
  weights_.format = "NVFP4 experts";
  auto* base = static_cast<std::uint8_t*>(arena_);
  try {
    std::vector<FileRange> ranges;
    std::vector<void*> destinations;
    for (const auto& [tensor, offset] : copies) {
      ranges.push_back({tensor->file, tensor->offset, tensor->bytes});
      destinations.push_back(base + offset);
    }
    UploadFileRanges(ranges, destinations);
    for (const auto& [offset, values] : tensor_scales)
      HipCheck(hipMemcpy(base + offset, values.data(), values.size() * 4,
                         hipMemcpyHostToDevice),
               "uploading NVFP4 matrix scales");
  } catch (...) {
    (void)hipFree(arena_);
    arena_ = nullptr;
    throw;
  }

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
    w.expert_gate_up_fp4 = base + o.gate_up;
    w.expert_gate_up_fp4_block_scale = base + o.gate_up_blocks;
    w.expert_gate_up_fp4_tensor_scale =
        reinterpret_cast<const float*>(base + o.gate_up_scale);
    w.expert_down_fp4 = base + o.down;
    w.expert_down_fp4_block_scale = base + o.down_blocks;
    w.expert_down_fp4_tensor_scale =
        reinterpret_cast<const float*>(base + o.down_scale);
  }
}

}  // namespace gufo::models::diffusion_gemma
