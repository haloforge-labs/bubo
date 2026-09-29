// Qwen3.8-27B sampling with an exact top-k selection in place of the full
// vocabulary sort (Windows). The kernels between the markers are verbatim
// copies from sample.hip (which Flash-Next also links); only the launchers
// below differ, and they fall back to the sample.hip launchers whenever the
// selection does not apply.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/sample_27b.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {
namespace {

// ---- verbatim from sample.hip ----
constexpr std::size_t kSamplingThreads = 256;

__global__ void ScatterDraftProbabilitiesKernel(
    const std::uint32_t* __restrict__ draft_candidate_ids,
    const float* __restrict__ draft_candidate_probabilities,
    std::size_t draft_candidate_count, float* __restrict__ dense,
    std::size_t vocab_size) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= draft_candidate_count)
    return;
  const auto token = draft_candidate_ids[index];
  if (token >= vocab_size)
    return;
  // Only the first occurrence writes. Sum duplicates in their original order
  // so the lookup retains the sparse verifier's exact F32 probability.
  for (std::size_t previous = 0; previous < index; ++previous) {
    if (draft_candidate_ids[previous] == token)
      return;
  }
  float probability = 0.0F;
  for (std::size_t next = index; next < draft_candidate_count; ++next) {
    if (draft_candidate_ids[next] == token)
      probability += draft_candidate_probabilities[next];
  }
  dense[token] = probability;
}

__global__ void PrepareSamplingKernel(const float* __restrict__ logits,
                                      float* __restrict__ adjusted_logits,
                                      std::uint32_t* __restrict__ token_ids,
                                      std::size_t vocab_size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vocab_size) {
    return;
  }
  adjusted_logits[index] = isfinite(logits[index]) ? logits[index] : -INFINITY;
  token_ids[index] = static_cast<std::uint32_t>(index);
}

__global__ void PrepareCandidateLogitsKernel(const float* logits, float* scores,
                                             std::uint32_t* ids,
                                             std::size_t size) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    scores[i] = isfinite(logits[i]) ? logits[i] : -INFINITY;
    ids[i] = static_cast<std::uint32_t>(i);
  }
}

__global__ void ApplySparsePenaltiesKernel(
    float* __restrict__ logits,
    const sampling::TokenPenalty* __restrict__ penalties,
    std::size_t penalty_count, std::size_t vocab_size, float repeat_penalty,
    float frequency_penalty, float presence_penalty) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= penalty_count) {
    return;
  }
  const auto penalty = penalties[index];
  const std::uint32_t token = penalty.token;
  if (token >= vocab_size) {
    return;
  }
  float adjusted = logits[token];
  if (penalty.repeated && repeat_penalty != 1.0F) {
    adjusted = adjusted <= 0.0F ? adjusted * repeat_penalty
                                : adjusted / repeat_penalty;
  }
  adjusted -= frequency_penalty * static_cast<float>(penalty.generated_count);
  if (penalty.generated_count != 0)
    adjusted -= presence_penalty;
  logits[token] = adjusted;
}


__global__ void SortedSamplingKernel(
    const float* __restrict__ sorted_logits,
    const std::uint32_t* __restrict__ sorted_token_ids,
    std::uint32_t* __restrict__ out_token, std::size_t vocab_size,
    float temperature, std::int32_t top_k, float top_p, float min_p,
    std::size_t min_keep, double uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t minimum = min(vocab_size, max(min_keep, std::size_t{1}));
  const std::size_t initial_count =
      top_k > 0 ? min(vocab_size, max(static_cast<std::size_t>(top_k), minimum))
                : vocab_size;

  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ std::size_t top_p_keep;
  __shared__ std::size_t final_keep;
  __shared__ std::uint32_t selected_segment;
  __shared__ double selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
    }
    return;
  }
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
    local_sum += expf((sorted_logits[index] - maximum) * inverse_temperature);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    top_p_keep = initial_count;
    if (top_p < 1.0F) {
      const float target = top_p * total;
      selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        if (target <= segment_prefix[segment] + segment_sums[segment]) {
          selected_segment = static_cast<std::uint32_t>(segment);
          break;
        }
      }
      selected_threshold = target - segment_prefix[selected_segment];
      top_p_keep = 0;
    }
  }
  __syncthreads();

  if (top_p < 1.0F && tid == selected_segment) {
    double threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      if (threshold <= 0.0F) {
        keep = index + 1U;
        break;
      }
    }
    top_p_keep = max(keep, minimum);
  }
  __syncthreads();

  if (tid == 0) {
    final_keep = top_p_keep;
    if (min_p > 0.0F && final_keep > 1U) {
      const float threshold = maximum + temperature * logf(min_p);
      std::size_t low = 0;
      std::size_t high = final_keep;
      while (low < high) {
        const std::size_t middle = low + ((high - low) / 2U);
        if (sorted_logits[middle] >= threshold) {
          low = middle + 1U;
        } else {
          high = middle;
        }
      }
      final_keep = min(final_keep, max(low, minimum));
    }
    if (temperature <= 0.0F || final_keep == 1U) {
      *out_token = sorted_token_ids[0];
    }
  }
  __syncthreads();
  if (temperature <= 0.0F || final_keep == 1U) {
    return;
  }

  const std::size_t sample_begin = (final_keep * tid) / blockDim.x;
  const std::size_t sample_end = (final_keep * (tid + 1U)) / blockDim.x;
  local_sum = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    local_sum += expf((sorted_logits[index] - maximum) * inverse_temperature);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    const double threshold = uniform * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    double threshold = selected_threshold;
    std::uint32_t selected = sorted_token_ids[sample_end - 1U];
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      selected = sorted_token_ids[index];
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}


__global__ void SortedSpeculativeSamplingKernel(
    const float* __restrict__ sorted_logits,
    const std::uint32_t* __restrict__ sorted_token_ids,
    std::uint32_t* __restrict__ out_token,
    std::uint32_t* __restrict__ out_accepted, std::size_t vocab_size,
    float temperature, std::int32_t top_k, float top_p, float min_p,
    std::size_t min_keep, std::uint32_t draft_token,
    float draft_token_probability,
    const float* __restrict__ draft_probabilities, double acceptance_uniform,
    double residual_uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t minimum = min(vocab_size, max(min_keep, std::size_t{1}));
  const std::size_t initial_count =
      top_k > 0 ? min(vocab_size, max(static_cast<std::size_t>(top_k), minimum))
                : vocab_size;

  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ float draft_weights[kSamplingThreads];
  __shared__ std::size_t top_p_keep;
  __shared__ std::size_t final_keep;
  __shared__ float target_total;
  __shared__ bool accepted;
  __shared__ bool use_residual;
  __shared__ std::uint32_t selected_segment;
  __shared__ double selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
      *out_accepted = 2;
    }
    return;
  }
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
    local_sum += expf((sorted_logits[index] - maximum) * inverse_temperature);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    top_p_keep = initial_count;
    if (top_p < 1.0F) {
      const float target = top_p * total;
      selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        if (target <= segment_prefix[segment] + segment_sums[segment]) {
          selected_segment = static_cast<std::uint32_t>(segment);
          break;
        }
      }
      selected_threshold = target - segment_prefix[selected_segment];
      top_p_keep = 0;
    }
  }
  __syncthreads();

  if (top_p < 1.0F && tid == selected_segment) {
    double threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      if (threshold <= 0.0F) {
        keep = index + 1U;
        break;
      }
    }
    top_p_keep = max(keep, minimum);
  }
  __syncthreads();

  if (tid == 0) {
    final_keep = top_p_keep;
    if (min_p > 0.0F && final_keep > 1U) {
      const float threshold = maximum + temperature * logf(min_p);
      std::size_t low = 0;
      std::size_t high = final_keep;
      while (low < high) {
        const std::size_t middle = low + ((high - low) / 2U);
        if (sorted_logits[middle] >= threshold) {
          low = middle + 1U;
        } else {
          high = middle;
        }
      }
      final_keep = min(final_keep, max(low, minimum));
    }
  }
  __syncthreads();

  const std::size_t sample_begin = (final_keep * tid) / blockDim.x;
  const std::size_t sample_end = (final_keep * (tid + 1U)) / blockDim.x;
  local_sum = 0.0F;
  float local_draft_weight = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    const float weight =
        expf((sorted_logits[index] - maximum) * inverse_temperature);
    local_sum += weight;
    if (sorted_token_ids[index] == draft_token) {
      local_draft_weight = weight;
    }
  }
  segment_sums[tid] = local_sum;
  draft_weights[tid] = local_draft_weight;
  __syncthreads();

  if (tid == 0) {
    target_total = 0.0F;
    float draft_target_weight = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      target_total += segment_sums[segment];
      draft_target_weight += draft_weights[segment];
    }
    const float draft_target_probability = draft_target_weight / target_total;
    accepted =
        acceptance_uniform * draft_token_probability <
        draft_target_probability;
    *out_accepted = accepted ? 1U : 0U;
    if (accepted) {
      *out_token = draft_token;
    }
  }
  __syncthreads();
  if (accepted) {
    return;
  }

  local_sum = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    const float target_weight =
        expf((sorted_logits[index] - maximum) * inverse_temperature);
    const float draft_probability =
        draft_probabilities[sorted_token_ids[index]];
    local_sum += fmaxf(target_weight - draft_probability * target_total, 0.0F);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    use_residual = total > 0.0F && isfinite(total);
  }
  __syncthreads();

  if (!use_residual) {
    segment_sums[tid] = 0.0F;
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      segment_sums[tid] +=
          expf((sorted_logits[index] - maximum) * inverse_temperature);
    }
    __syncthreads();
    if (tid == 0) {
      float total = 0.0F;
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        segment_prefix[segment] = total;
        total += segment_sums[segment];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float total =
        segment_prefix[blockDim.x - 1U] + segment_sums[blockDim.x - 1U];
    const double threshold = residual_uniform * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (segment_sums[segment] > 0.0F &&
          threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    double threshold = selected_threshold;
    std::uint32_t selected = sorted_token_ids[sample_end - 1U];
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      const float target_weight =
          expf((sorted_logits[index] - maximum) * inverse_temperature);
      float weight = target_weight;
      if (use_residual) {
        const float draft_probability =
            draft_probabilities[sorted_token_ids[index]];
        weight =
            fmaxf(target_weight - draft_probability * target_total, 0.0F);
      }
      if (weight <= 0.0F) {
        continue;
      }
      threshold -= weight;
      selected = sorted_token_ids[index];
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}

// ---- end of verbatim copies ----

// Exact top-k selection (Qwen3.8-27B, Windows). The sorted sampling kernels
// read only the first max(top_k, min_keep) entries of the descending sort. A
// stable descending radix sort orders by the float's radix key and keeps
// ascending token ids among equal keys; selecting the largest
// (key, ~token) pairs reproduces exactly that prefix without sorting the
// whole vocabulary.
constexpr std::size_t kSelectBlock = 1024;
constexpr std::size_t kSelectThreads = 256;
constexpr std::size_t kMaxSelect = 64;
#ifdef _WIN32
constexpr bool kUseSelect = true;
#else
constexpr bool kUseSelect = false;  // Linux keeps the upstream launchers.
#endif

__device__ __forceinline__ std::uint64_t SelectKey(float value,
                                                   std::uint32_t index) {
  const std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t key =
      (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
  return (static_cast<std::uint64_t>(key) << 32) |
         static_cast<std::uint64_t>(0xFFFFFFFFU - index);
}

// Sorts up to kSelectBlock keys per block (descending) and writes the block's
// first `keep` keys; missing entries are 0, which orders below every value.
__global__ void SelectTopKeysKernel(const float* __restrict__ values,
                                    const std::uint64_t* __restrict__ keys_in,
                                    std::size_t count, std::size_t keep,
                                    std::uint64_t* __restrict__ keys_out) {
  __shared__ std::uint64_t tile[kSelectBlock];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kSelectBlock;
  for (std::size_t i = threadIdx.x; i < kSelectBlock; i += blockDim.x) {
    const std::size_t index = base + i;
    std::uint64_t key = 0;
    if (index < count)
      key = keys_in != nullptr
                ? keys_in[index]
                : SelectKey(values[index], static_cast<std::uint32_t>(index));
    tile[i] = key;
  }
  __syncthreads();
  for (std::size_t size = 2; size <= kSelectBlock; size <<= 1) {
    for (std::size_t stride = size >> 1; stride > 0; stride >>= 1) {
      for (std::size_t i = threadIdx.x; i < kSelectBlock; i += blockDim.x) {
        const std::size_t partner = i ^ stride;
        if (partner > i) {
          const bool descending = (i & size) == 0;
          const std::uint64_t a = tile[i];
          const std::uint64_t b = tile[partner];
          if (descending ? a < b : a > b) {
            tile[i] = b;
            tile[partner] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (std::size_t i = threadIdx.x; i < keep; i += blockDim.x)
    keys_out[static_cast<std::size_t>(blockIdx.x) * keep + i] = tile[i];
}

__global__ void WriteSelectedKernel(const std::uint64_t* __restrict__ keys,
                                    const float* __restrict__ values,
                                    std::size_t keep,
                                    float* __restrict__ sorted_values,
                                    std::uint32_t* __restrict__ sorted_ids) {
  const std::size_t i = threadIdx.x;
  if (i >= keep)
    return;
  const auto index = static_cast<std::uint32_t>(
      0xFFFFFFFFU - static_cast<std::uint32_t>(keys[i] & 0xFFFFFFFFU));
  sorted_values[i] = values[index];
  sorted_ids[i] = index;
}

// Writes the first `keep` entries of the stable descending sort of `values`
// into sorted_values / sorted_ids. `scratch` holds 2 * ceil(count / 1024) *
// keep keys.
void SelectTopK(const float* values, std::size_t count, std::size_t keep,
                std::uint64_t* scratch, float* sorted_values,
                std::uint32_t* sorted_ids, hipStream_t stream) {
  std::size_t blocks = (count + kSelectBlock - 1) / kSelectBlock;
  std::uint64_t* current = scratch;
  std::uint64_t* next = scratch + blocks * keep;
  SelectTopKeysKernel<<<static_cast<unsigned>(blocks), kSelectThreads, 0,
                        stream>>>(values, nullptr, count, keep, current);
  std::size_t remaining = blocks * keep;
  while (remaining > keep) {
    blocks = (remaining + kSelectBlock - 1) / kSelectBlock;
    SelectTopKeysKernel<<<static_cast<unsigned>(blocks), kSelectThreads, 0,
                          stream>>>(nullptr, current, remaining, keep, next);
    std::swap(current, next);
    remaining = blocks * keep;
  }
  WriteSelectedKernel<<<1, static_cast<unsigned>(kMaxSelect), 0, stream>>>(
      current, values, keep, sorted_values, sorted_ids);
}

std::size_t SelectCount(const GpuSamplingParameters& parameters,
                        std::size_t vocab_size) {
  // Same prefix length as the sorted kernels' initial_count.
  const std::size_t minimum =
      std::min(vocab_size, std::max(parameters.min_keep, std::size_t{1}));
  if (parameters.top_k <= 0)
    return vocab_size;
  return std::min(vocab_size,
                  std::max(static_cast<std::size_t>(parameters.top_k), minimum));
}

bool SelectFits(const GpuSamplingWorkspace& workspace, std::size_t keep) {
  const std::size_t blocks =
      (workspace.vocab_size + kSelectBlock - 1) / kSelectBlock;
  return keep <= kMaxSelect &&
         2 * blocks * keep * sizeof(std::uint64_t) <=
             workspace.sort_temp_storage_bytes;
}

}  // namespace

void LaunchGPUSpeculativeSampling27(
    const float* logits, std::uint32_t* out_token, std::uint32_t* out_accepted,
    std::size_t vocab_size, const GpuSamplingParameters& parameters,
    std::uint32_t draft_token, float draft_token_probability,
    const std::uint32_t* draft_candidate_ids,
    const float* draft_candidate_probabilities,
    std::size_t draft_candidate_count, double acceptance_uniform,
    double residual_uniform, const sampling::TokenPenalty* penalties,
    std::size_t penalty_count, GpuSamplingWorkspace* workspace,
    hipStream_t stream) {
  const bool needs_ordering =
      parameters.top_k > 0 || parameters.top_p < 1.0F ||
      (parameters.min_p > 0.0F && parameters.min_keep > 1);
  const std::size_t keep = SelectCount(parameters, vocab_size);
  if (!kUseSelect || !needs_ordering || parameters.top_k <= 0 ||
      workspace == nullptr ||
      workspace->vocab_size != vocab_size || !SelectFits(*workspace, keep) ||
      !std::isfinite(acceptance_uniform) || acceptance_uniform < 0 ||
      acceptance_uniform >= 1 || !std::isfinite(residual_uniform) ||
      residual_uniform < 0 || residual_uniform >= 1 || logits == nullptr ||
      out_token == nullptr || out_accepted == nullptr ||
      draft_candidate_count == 0 ||
      draft_candidate_count > workspace->penalty_capacity ||
      penalty_count > workspace->penalty_capacity ||
      draft_candidate_ids == nullptr ||
      draft_candidate_probabilities == nullptr ||
      !std::isfinite(draft_token_probability) ||
      draft_token_probability <= 0.0F || parameters.temperature <= 0.0F ||
      (parameters.top_k == 1 && parameters.min_keep <= 1)) {
    // Everything else (and every validation error) takes the original path.
    LaunchGPUSpeculativeSampling(
        logits, out_token, out_accepted, vocab_size, parameters, draft_token,
        draft_token_probability, draft_candidate_ids,
        draft_candidate_probabilities, draft_candidate_count,
        acceptance_uniform, residual_uniform, penalties, penalty_count,
        workspace, stream);
    return;
  }

  HIP_CHECK(hipMemcpyAsync(
      workspace->draft_candidate_ids, draft_candidate_ids,
      draft_candidate_count * sizeof(std::uint32_t), hipMemcpyHostToDevice,
      stream));
  HIP_CHECK(hipMemcpyAsync(
      workspace->draft_candidate_probabilities,
      draft_candidate_probabilities, draft_candidate_count * sizeof(float),
      hipMemcpyHostToDevice, stream));

  const std::size_t blocks =
      (vocab_size + kSamplingThreads - 1U) / kSamplingThreads;
  PrepareSamplingKernel<<<static_cast<unsigned int>(blocks), kSamplingThreads,
                          0, stream>>>(logits, workspace->adjusted_logits,
                                       workspace->token_ids, vocab_size);
  if (penalty_count > 0) {
    HIP_CHECK(hipMemcpyAsync(workspace->penalties, penalties,
                             penalty_count * sizeof(sampling::TokenPenalty),
                             hipMemcpyHostToDevice, stream));
    const std::size_t penalty_blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(penalty_blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalties, penalty_count,
        vocab_size, parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }
  SelectTopK(workspace->adjusted_logits, vocab_size, keep,
             static_cast<std::uint64_t*>(workspace->sort_temp_storage),
             workspace->sorted_logits, workspace->sorted_token_ids, stream);

  auto* draft_probabilities = reinterpret_cast<float*>(workspace->token_ids);
  HIP_CHECK(hipMemsetAsync(draft_probabilities, 0, vocab_size * sizeof(float),
                           stream));
  ScatterDraftProbabilitiesKernel<<<
      static_cast<unsigned>((draft_candidate_count + kSamplingThreads - 1) /
                            kSamplingThreads),
      kSamplingThreads, 0, stream>>>(
      workspace->draft_candidate_ids, workspace->draft_candidate_probabilities,
      draft_candidate_count, draft_probabilities, vocab_size);
  SortedSpeculativeSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      out_accepted, vocab_size, parameters.temperature, parameters.top_k,
      parameters.top_p, parameters.min_p, parameters.min_keep, draft_token,
      draft_token_probability, draft_probabilities, acceptance_uniform,
      residual_uniform);
}

void LaunchGPUSampling27(const float* logits, std::uint32_t* out_token,
                         std::size_t vocab_size,
                         const GpuSamplingParameters& parameters,
                         const sampling::TokenPenalty* penalties,
                         std::size_t penalty_count,
                         GpuSamplingWorkspace* workspace, hipStream_t stream) {
  const std::size_t keep = SelectCount(parameters, vocab_size);
  if (!kUseSelect || parameters.top_k <= 0 || parameters.temperature <= 0.0F ||
      (parameters.top_k == 1 && parameters.min_keep <= 1) ||
      workspace == nullptr || workspace->vocab_size != vocab_size ||
      !SelectFits(*workspace, keep) || !std::isfinite(parameters.uniform) ||
      parameters.uniform < 0 || parameters.uniform >= 1 || logits == nullptr ||
      out_token == nullptr || penalty_count > workspace->penalty_capacity) {
    LaunchGPUSampling(logits, out_token, vocab_size, parameters, penalties,
                      penalty_count, workspace, stream);
    return;
  }
  const std::size_t blocks =
      (vocab_size + kSamplingThreads - 1U) / kSamplingThreads;
  PrepareSamplingKernel<<<static_cast<unsigned int>(blocks), kSamplingThreads,
                          0, stream>>>(logits, workspace->adjusted_logits,
                                       workspace->token_ids, vocab_size);
  if (penalty_count > 0) {
    HIP_CHECK(hipMemcpyAsync(workspace->penalties, penalties,
                             penalty_count * sizeof(sampling::TokenPenalty),
                             hipMemcpyHostToDevice, stream));
    const std::size_t penalty_blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(penalty_blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalties, penalty_count,
        vocab_size, parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }
  SelectTopK(workspace->adjusted_logits, vocab_size, keep,
             static_cast<std::uint64_t*>(workspace->sort_temp_storage),
             workspace->sorted_logits, workspace->sorted_token_ids, stream);
  SortedSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      vocab_size, parameters.temperature, parameters.top_k, parameters.top_p,
      parameters.min_p, parameters.min_keep, parameters.uniform);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
