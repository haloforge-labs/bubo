#ifndef GUFO_MODELS_DIFFUSION_GEMMA_CONFIG_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace gufo::models::diffusion_gemma {

/// Text architecture of google/diffusiongemma-26B-A4B-it. One set of layer
/// weights serves both the causal encoder (prompt and committed canvases) and
/// the bidirectional denoising decoder. Every value is read from config.json
/// and checked against the shapes the kernels are written for.
struct Config {
  int hidden{2816};
  int layers{30};
  int heads{16};
  int sliding_head_dim{256};
  int sliding_kv_heads{8};
  int global_head_dim{512};
  int global_kv_heads{2};
  int intermediate{2112};
  int experts{128};
  int top_k{8};
  int expert_intermediate{704};
  int vocab{262144};
  int sliding_window{1024};
  int canvas{256};
  int max_positions{262144};
  float rms_eps{1e-6F};
  float final_softcap{30.0F};
  float sliding_theta{10000.0F};
  float global_theta{1000000.0F};
  float global_rotary_fraction{0.25F};
  std::vector<bool> is_global;  ///< Per layer: full attention.
  std::vector<int> eos_ids;
  int pad_id{0};
  int bos_id{2};

  [[nodiscard]] int head_dim(int layer) const {
    return is_global[layer] ? global_head_dim : sliding_head_dim;
  }
  [[nodiscard]] int kv_heads(int layer) const {
    return is_global[layer] ? global_kv_heads : sliding_kv_heads;
  }
};

/// Diffusion sampling defaults from generation_config.json.
struct DiffusionDefaults {
  int max_denoising_steps{48};
  float t_min{0.4F};
  float t_max{0.8F};
  float entropy_bound{0.1F};
  int stability_threshold{1};
  float confidence_threshold{0.005F};
};

/// Throws std::runtime_error on a missing file or an unsupported shape.
Config LoadConfig(const std::filesystem::path& root);
DiffusionDefaults LoadDiffusionDefaults(const std::filesystem::path& root);

/// True when `root` is a DiffusionGemma checkpoint directory.
bool IsDiffusionGemmaDirectory(const std::filesystem::path& root);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_CONFIG_HPP_
