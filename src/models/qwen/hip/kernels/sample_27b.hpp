#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SAMPLE_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SAMPLE_27B_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {

/// LaunchGPUSpeculativeSampling / LaunchGPUSampling with an exact top-k
/// selection instead of sorting the whole vocabulary when top_k is set. The
/// sorted kernels only read the first max(top_k, min_keep) entries, and the
/// selection reproduces exactly that prefix of the stable descending sort, so
/// results are bit-identical. Other configurations use the original launchers.
void LaunchGPUSpeculativeSampling27(
    const float* logits, std::uint32_t* out_token, std::uint32_t* out_accepted,
    std::size_t vocab_size, const GpuSamplingParameters& parameters,
    std::uint32_t draft_token, float draft_token_probability,
    const std::uint32_t* draft_candidate_ids,
    const float* draft_candidate_probabilities,
    std::size_t draft_candidate_count, double acceptance_uniform,
    double residual_uniform, const sampling::TokenPenalty* penalties,
    std::size_t penalty_count, GpuSamplingWorkspace* workspace,
    hipStream_t stream);

void LaunchGPUSampling27(const float* logits, std::uint32_t* out_token,
                         std::size_t vocab_size,
                         const GpuSamplingParameters& parameters,
                         const sampling::TokenPenalty* penalties,
                         std::size_t penalty_count,
                         GpuSamplingWorkspace* workspace, hipStream_t stream);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SAMPLE_27B_HPP_
