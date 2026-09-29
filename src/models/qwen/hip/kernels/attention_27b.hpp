#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_27B_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::hip {

/// Qwen3.8-27B verification attention over the FP16 KV cache (2..4 rows, six
/// query heads per KV head, head_dim 256, context >= 512). Each (row, head)
/// visits the same split-K partitions, tokens and operations in the same order
/// as LaunchCausalDecodeAttention, so the output is bit identical; the lane
/// reduction uses DPP row shifts instead of LDS permutes, two tokens are
/// scored before their in-order softmax updates, and the next KV tile is
/// fetched while the current one is consumed. Four rows share one KV tile per
/// block. Returns false (nothing launched) outside that shape.
bool TryLaunchVerifyAttention27(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream, std::span<float> scratch);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_27B_HPP_
