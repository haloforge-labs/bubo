#ifndef GUFO_CORE_SESSION_STATE_BYTES_HPP_
#define GUFO_CORE_SESSION_STATE_BYTES_HPP_

// Per-session model state sizes (KV cache, convolution state, recurrent
// state). The GPU allocators and the CPU-side memory estimate (bubo models
// estimate) both call these functions, so the estimate cannot drift from what
// the engine allocates. Pure arithmetic; no HIP types.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/core/model_config.hpp"

namespace gufo::core {

namespace detail {
inline std::size_t StateMul(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error("session state size overflows");
  return left * right;
}
inline std::size_t StateAdd(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("session state total overflows");
  return left + right;
}
}  // namespace detail

/// Dense Qwen (qwen35) request state: attention K and V planes for every
/// full-attention layer, plus convolution and DeltaNet state for every
/// recurrent layer. Same arithmetic as QwenGpuArena::EstimateMemoryUsage.
[[nodiscard]] inline std::size_t QwenDenseSessionStateBytes(
    const ModelConfig& config, std::size_t context,
    std::size_t kv_element_bytes, std::size_t recurrent_element_bytes) {
  using detail::StateAdd;
  using detail::StateMul;
  const std::size_t kv = StateMul(config.num_key_value_heads, config.head_dim);
  const std::size_t total_kv =
      StateMul(StateMul(config.FullAttentionLayerCount(), kv), context);
  std::size_t total = StateMul(StateMul(total_kv, 2), kv_element_bytes);
  const std::size_t total_conv =
      StateMul(StateMul(config.SsmLayerCount(), config.SsmQkvSize()),
               config.ssm_conv_kernel);
  total = StateAdd(total, StateMul(total_conv, sizeof(float)));
  const std::size_t total_deltanet =
      StateMul(StateMul(StateMul(config.SsmLayerCount(),
                                 config.ssm_time_step_rank),
                        config.ssm_state_size),
               config.SsmValueSize());
  return StateAdd(total, StateMul(total_deltanet, recurrent_element_bytes));
}

/// Qwen3.6-35B-A3B (qwen35moe) request state, per layer. A recurrent layer
/// holds a convolution window and a DeltaNet matrix in float; an attention
/// layer holds K, V and transposed-V planes in half precision.
[[nodiscard]] inline std::size_t A3bConvElements(
    std::size_t ssm_conv_kernel, std::size_t conv_channels) {
  return detail::StateMul(ssm_conv_kernel - 1, conv_channels);
}
[[nodiscard]] inline std::size_t A3bRecurrentElements(
    std::size_t ssm_v_heads, std::size_t ssm_head_dim) {
  return detail::StateMul(detail::StateMul(ssm_v_heads, ssm_head_dim),
                          ssm_head_dim);
}
[[nodiscard]] inline std::size_t A3bKvPlaneElements(std::size_t context,
                                                    std::size_t kv_dim) {
  return detail::StateMul(context, kv_dim);
}
inline constexpr std::size_t kA3bKvPlanes = 3;  // K, V, transposed V
[[nodiscard]] inline std::size_t A3bSessionStateBytes(
    std::size_t layers, std::size_t full_attention_interval,
    std::size_t kv_dim, std::size_t ssm_conv_kernel,
    std::size_t conv_channels, std::size_t ssm_v_heads,
    std::size_t ssm_head_dim, std::size_t context) {
  using detail::StateAdd;
  using detail::StateMul;
  if (full_attention_interval == 0)
    throw std::invalid_argument("full_attention_interval is zero");
  std::size_t total = 0;
  for (std::size_t i = 0; i < layers; ++i) {
    if ((i + 1) % full_attention_interval != 0)  // same test as IsLinear
      total = StateAdd(
          total,
          StateMul(StateAdd(A3bConvElements(ssm_conv_kernel, conv_channels),
                            A3bRecurrentElements(ssm_v_heads, ssm_head_dim)),
                   sizeof(float)));
    else
      total = StateAdd(
          total, StateMul(StateMul(kA3bKvPlanes,
                                   A3bKvPlaneElements(context, kv_dim)),
                          2));  // sizeof(__half)
  }
  return total;
}

}  // namespace gufo::core

#endif  // GUFO_CORE_SESSION_STATE_BYTES_HPP_
