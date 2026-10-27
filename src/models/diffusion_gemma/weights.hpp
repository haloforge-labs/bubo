#ifndef GUFO_MODELS_DIFFUSION_GEMMA_WEIGHTS_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/models/diffusion_gemma/config.hpp"

namespace gufo::models::diffusion_gemma {

using bf16_t = std::uint16_t;

/// One tensor in a sharded safetensors checkpoint.
struct TensorEntry {
  std::filesystem::path file;
  std::uint64_t offset{0};  ///< Absolute byte offset in `file`.
  std::uint64_t bytes{0};
  std::string dtype;
  std::vector<std::int64_t> shape;
};

/// Parses model.safetensors.index.json and every shard header.
std::unordered_map<std::string, TensorEntry> ReadSafetensorsIndex(
    const std::filesystem::path& root);

/// Reads a one-element BF16 tensor (a layer scalar) as float.
float ReadBf16Scalar(const TensorEntry& tensor);
/// Reads a one-element FP32 tensor (a quantization scale).
float ReadF32Scalar(const TensorEntry& tensor);

// The dense matrices (QKV, O, MLP, router and self-conditioning products)
// are held in kernels::Linear's blocked layout (BlockLinearWeights; the
// gate/up matrices by BlockGateUpWeights, for kernels::LinearGeluMul); the
// shapes below are their logical [rows][cols]. Embeddings, norms and experts
// keep the checkpoint's row-major layout.
struct LayerWeights {
  // Attention. Fused projection rows: Q (heads*hd), K (kv*hd) and, for
  // sliding layers, V (kv*hd). Global layers reuse K's projection for V.
  const bf16_t* qkv{};
  const bf16_t* o{};
  const bf16_t* q_norm{};
  const bf16_t* k_norm{};
  // Norms (hidden).
  const bf16_t* input_norm{};
  const bf16_t* post_attention_norm{};
  const bf16_t* pre_ff_norm{};
  const bf16_t* pre_ff_norm_2{};
  const bf16_t* post_ff_norm{};
  const bf16_t* post_ff_norm_1{};
  const bf16_t* post_ff_norm_2{};
  // Dense MLP: gate rows then up rows (2*intermediate x hidden); down.
  const bf16_t* mlp_gate_up{};
  const bf16_t* mlp_down{};
  // Router.
  const bf16_t* router_proj{};       ///< experts x hidden
  const bf16_t* router_scale{};      ///< hidden
  const bf16_t* per_expert_scale{};  ///< experts
  // Experts: [E][2*ei][hidden] (gate rows then up rows), [E][hidden][ei].
  const bf16_t* expert_gate_up{};
  const bf16_t* expert_down{};
  // Q8_0 experts (GGUF): int8 values in the same layout, and one FP16 scale
  // per 32 consecutive values along the reduction dimension. When set, the
  // BF16 expert pointers above are null.
  const std::int8_t* expert_gate_up_q8{};
  const std::uint16_t* expert_gate_up_scale{};
  const std::int8_t* expert_down_q8{};
  const std::uint16_t* expert_down_scale{};
  // FP8 E4M3 experts (compressed-tensors FP8 checkpoint): values in the BF16
  // layout and one BF16 scale per weight row, [E][2*ei] and [E][hidden].
  const std::uint8_t* expert_gate_up_fp8{};
  const bf16_t* expert_gate_up_fp8_scale{};
  const std::uint8_t* expert_down_fp8{};
  const bf16_t* expert_down_fp8_scale{};
  // NVFP4 experts (ModelOpt checkpoint): E2M1 values packed two per byte in
  // the BF16 layout, one E4M3 scale per 16 values along the reduction, and
  // each matrix's FP32 scale repeated per row, [E][2*ei] and [E][hidden].
  const std::uint8_t* expert_gate_up_fp4{};
  const std::uint8_t* expert_gate_up_fp4_block_scale{};
  const float* expert_gate_up_fp4_tensor_scale{};
  const std::uint8_t* expert_down_fp4{};
  const std::uint8_t* expert_down_fp4_block_scale{};
  const float* expert_down_fp4_tensor_scale{};
  float layer_scalar{1.0F};
};

struct Weights {
  const bf16_t* embed{};  ///< vocab x hidden, also the tied LM head.
  const bf16_t* final_norm{};
  const bf16_t* sc_pre_norm{};
  const bf16_t* sc_gate_up{};  ///< 2*intermediate x hidden
  const bf16_t* sc_down{};     ///< hidden x intermediate
  std::vector<LayerWeights> layers;
  std::size_t device_bytes{0};
  /// "BF16" for the safetensors checkpoint, "Q8_0 experts" for a GGUF,
  /// "FP8 experts" for a compressed-tensors FP8 checkpoint and "NVFP4
  /// experts" for a ModelOpt NVFP4 checkpoint.
  std::string format{"BF16"};
};

/// The Q8_0 GGUF in `root` when it holds one and no safetensors index
/// (config.json and tokenizer.json still come from `root`); empty otherwise.
std::filesystem::path FindGgufWeights(const std::filesystem::path& root);

/// Owns the device allocation backing a Weights view.
class DeviceWeights {
public:
  DeviceWeights(const std::filesystem::path& root, const Config& config);
  ~DeviceWeights();
  DeviceWeights(const DeviceWeights&) = delete;
  DeviceWeights& operator=(const DeviceWeights&) = delete;

  [[nodiscard]] const Weights& view() const { return weights_; }

private:
  void LoadBf16(const std::unordered_map<std::string, TensorEntry>& index,
                const Config& config);
  void LoadGguf(const std::filesystem::path& file, const Config& config);
  void LoadFp8(const std::unordered_map<std::string, TensorEntry>& index,
               const Config& config);
  void LoadNvfp4(const std::unordered_map<std::string, TensorEntry>& index,
                 const Config& config);
  /// Rewrites the dense matrices in kernels::Linear's blocked layout.
  void BlockDenseMatrices(const Config& config);

  void* arena_{nullptr};
  Weights weights_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_WEIGHTS_HPP_
