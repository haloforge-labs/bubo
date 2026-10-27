#ifndef GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_
#define GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_

// Qwen3.6-35B-A3B (qwen35moe) architecture parameters and the GGUF key reader.
// The engine and the CPU-side memory estimate (bubo models estimate) read the
// file through this one function.

#include <cstdint>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen36_a3b {

inline constexpr std::string_view kArch = "qwen35moe";

struct Config {
  std::uint32_t layers{0};
  std::uint32_t hidden{0};
  std::uint32_t vocab{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t rotary_dim{0};
  float rope_theta{0};
  float eps{0};
  std::uint32_t full_attention_interval{0};
  std::uint32_t ssm_conv_kernel{0};
  std::uint32_t ssm_head_dim{0};
  std::uint32_t ssm_k_heads{0};
  std::uint32_t ssm_v_heads{0};
  std::uint32_t experts{0};
  std::uint32_t experts_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t nextn{0};  // MTP blocks after the trunk (not in `layers`)

  bool IsLinear(std::uint32_t layer) const {
    return (layer + 1) % full_attention_interval != 0;
  }
  std::uint32_t ConvChannels() const {
    return (2 * ssm_k_heads + ssm_v_heads) * ssm_head_dim;
  }
  std::uint32_t SsmValueDim() const { return ssm_v_heads * ssm_head_dim; }
  std::uint32_t QDim() const { return heads * head_dim; }
  std::uint32_t KvDim() const { return kv_heads * head_dim; }
};

inline void ConfigFail(std::string* error, std::string message) {
  if (error != nullptr && error->empty())
    *error = std::move(message);
}

/// Reads and checks the qwen35moe metadata. Returns false with *error set.
inline bool ReadConfig(const core::GgufReader& g, Config& c_,
                       std::string* error) {
  if (g.GetMetadataString("general.architecture") != kArch) {
    ConfigFail(error, "GGUF architecture is not qwen35moe");
    return false;
  }
  bool ok = true;
  const auto u32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataUint32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      ConfigFail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0);
  };
  const auto f32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataFloat32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      ConfigFail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0.0F);
  };
  c_.layers = u32("block_count");
  c_.hidden = u32("embedding_length");
  c_.heads = u32("attention.head_count");
  c_.kv_heads = u32("attention.head_count_kv");
  c_.head_dim = u32("attention.key_length");
  c_.rotary_dim = u32("rope.dimension_count");
  c_.rope_theta = f32("rope.freq_base");
  c_.eps = f32("attention.layer_norm_rms_epsilon");
  c_.full_attention_interval = u32("full_attention_interval");
  c_.ssm_conv_kernel = u32("ssm.conv_kernel");
  c_.ssm_head_dim = u32("ssm.state_size");
  c_.ssm_k_heads = u32("ssm.group_count");
  c_.ssm_v_heads = u32("ssm.time_step_rank");
  c_.experts = u32("expert_count");
  c_.experts_used = u32("expert_used_count");
  c_.expert_ff = u32("expert_feed_forward_length");
  c_.shared_ff = u32("expert_shared_feed_forward_length");
  if (!ok)
    return false;
  // MTP files append their nextn block(s) after the trunk.
  c_.nextn = g.GetMetadataUint32(std::string(kArch) + ".nextn_predict_layers")
                 .value_or(0);
  if (c_.nextn >= c_.layers) {
    ConfigFail(error, "bad nextn_predict_layers");
    return false;
  }
  c_.layers -= c_.nextn;
  // The FN kernels this runtime borrows fix these geometry points.
  if (c_.head_dim != 256 || c_.ssm_head_dim != 128 || c_.ssm_conv_kernel != 4 ||
      c_.heads % c_.kv_heads != 0 || c_.ssm_v_heads % c_.ssm_k_heads != 0 ||
      c_.experts > 512 || c_.full_attention_interval == 0 ||
      c_.hidden % 32 != 0) {
    ConfigFail(error, "unsupported qwen35moe geometry");
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_
