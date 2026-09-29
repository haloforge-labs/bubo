#include "src/models/qwen/hip/kernels/attention_27b.hpp"

#include <hip/hip_fp16.h>

#include <cmath>

#include "src/models/qwen/hip/detail/attention_policy.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGroup = 6;
constexpr std::uint32_t kMaxSplits = detail::kSplitKDecodeAttentionMaxSplits;

__device__ __forceinline__ float4 LoadHalf4(const __half* source) {
  const auto* pairs = reinterpret_cast<const __half2*>(source);
  const float2 lower = __half22float2(pairs[0]);
  const float2 upper = __half22float2(pairs[1]);
  return make_float4(lower.x, lower.y, upper.x, upper.y);
}

// DPP row shift: lane i of each 16-lane row reads lane i + Offset. Lane 0's
// butterfly only reads in-row lanes, so its sum equals
// __shfl_down(score, Offset, 16) bit for bit.
template<int Offset>
__device__ __forceinline__ float RowShiftDown(float value) {
  return __int_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_int(value), 0x100 | Offset, 0xf, 0xf, true));
}

// Split-K partials with the arithmetic of production's
// QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16>: 16 lanes per query
// head, six heads per KV head. PerRow: grid (kv_heads, rows, splits), one row
// per block. Otherwise: grid (kv_heads, splits, row groups), Rows rows per
// block sharing every KV tile. Rows occupy whole waves (96 lanes), so token
// bounds are wave-uniform.
template<unsigned Rows, unsigned Pair, bool PerRow>
__launch_bounds__(Rows * 96) __global__
    void VerifyAttentionPartials27(const float* __restrict__ q,
                                   const __half* __restrict__ k_cache,
                                   const __half* __restrict__ v_cache,
                                   float* __restrict__ scratch,
                                   std::uint32_t layer_idx, std::uint32_t pos,
                                   std::uint32_t max_context,
                                   std::uint32_t num_heads,
                                   std::uint32_t num_kv_heads,
                                   std::uint32_t split_count,
                                   std::uint32_t row_base) {
  constexpr unsigned Lanes = 16;
  constexpr std::uint32_t head_dim = kHeadDim;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / Lanes;
  constexpr unsigned kThreads = Rows * 96;
  // One uint4 of K and one of V per thread per tile.
  constexpr unsigned kTile = kThreads / 32;

  const unsigned first_row =
      row_base + (PerRow ? blockIdx.y : blockIdx.z * Rows);
  const unsigned row = PerRow ? first_row : first_row + threadIdx.x / 96;
  const unsigned local = threadIdx.x % 96;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = PerRow ? blockIdx.z : blockIdx.y;
  const std::uint32_t head = kv_head * kGroup + local / Lanes;
  const std::uint32_t lane = local % Lanes;

  q += std::size_t{row} * num_heads * head_dim;
  float* const row_scratch =
      scratch + std::size_t{row} * num_heads * kMaxSplits * (head_dim + 2);

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  // Each row's own causal length and partition, as in production.
  const auto range = [&](unsigned r, std::size_t& begin, std::size_t& end) {
    const std::size_t sequence_length = static_cast<std::size_t>(pos) + r + 1;
    const std::size_t partition_length =
        (sequence_length + split_count - 1) / split_count;
    begin = static_cast<std::size_t>(split) * partition_length;
    end = min(begin + partition_length, sequence_length);
  };
  std::size_t my_begin, my_end, union_begin, unused, union_end;
  range(row, my_begin, my_end);
  range(first_row, union_begin, unused);
  range(PerRow ? row : first_row + Rows - 1, unused, union_end);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] =
        query[lane + (vector / 2) * Lanes + (vector % 2) * 32];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const auto score_of = [&](const __half* key) {
    float partial[2] = {};
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
      for (unsigned vector = 0; vector < 2; ++vector) {
        const float4 query_value = query_values[half * 2 + vector];
        const float4 key_value =
            LoadHalf4(key + ((lane + half * 16 + vector * 32) * 4));
        partial[half] +=
            (query_value.x * key_value.x) + (query_value.y * key_value.y) +
            (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
    }
    float score = partial[0] + partial[1];
    score += RowShiftDown<8>(score);
    score += RowShiftDown<4>(score);
    score += RowShiftDown<2>(score);
    score += RowShiftDown<1>(score);
    return __shfl(score, 0, Lanes) * scale;
  };
  const auto update = [&](float score, const __half* value) {
    // One of expf(running_max - new_max) and expf(score - new_max) is
    // expf(0) = 1; the other is expf(-|running_max - score|), bit for bit
    // (negating a difference is exact, including running_max = -inf).
    const float difference = running_max - score;
    const float ratio = expf(-fabsf(difference));
    const bool new_maximum = difference < 0.0F;
    const float previous_scale = new_maximum ? ratio : 1.0F;
    const float value_scale = new_maximum ? 1.0F : ratio;
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_maximum ? score : running_max;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = LoadHalf4(
          value + ((lane + (vector / 2) * Lanes + (vector % 2) * 32) * 4));
      float4& output = output_values[vector];
      // Preserve the scalar kernel's rounded product and FMA operands.
      output.x =
          fmaf(output.x, previous_scale, __fmul_rn(value_data.x, value_scale));
      output.y =
          fmaf(output.y, previous_scale, __fmul_rn(value_data.y, value_scale));
      output.z =
          fmaf(value_data.z, value_scale, __fmul_rn(output.z, previous_scale));
      output.w =
          fmaf(value_data.w, value_scale, __fmul_rn(output.w, previous_scale));
    }
  };

  __shared__ __align__(16) __half keys[kTile][head_dim];
  __shared__ __align__(16) __half values[kTile][head_dim];
  const unsigned stage_token = threadIdx.x / 32;
  const unsigned stage_dim = (threadIdx.x % 32) * 8;
  uint4 next_key{}, next_value{};
  const auto fetch = [&](std::size_t begin) {
    if (begin + stage_token < union_end) {
      const std::size_t offset =
          cache_head_offset + (begin + stage_token) * kv_width + stage_dim;
      next_key = *reinterpret_cast<const uint4*>(k_cache + offset);
      next_value = *reinterpret_cast<const uint4*>(v_cache + offset);
    }
  };

  if (union_begin < union_end) {
    fetch(union_begin);
    for (std::size_t begin = union_begin; begin < union_end; begin += kTile) {
      const std::size_t tile_end = min(begin + kTile, union_end);
      *reinterpret_cast<uint4*>(keys[stage_token] + stage_dim) = next_key;
      *reinterpret_cast<uint4*>(values[stage_token] + stage_dim) = next_value;
      __syncthreads();
      if (tile_end < union_end)
        fetch(tile_end);
      const unsigned first = static_cast<unsigned>(
          (my_begin > begin ? min(my_begin, tile_end) : begin) - begin);
      const unsigned last = static_cast<unsigned>(
          (my_end < tile_end ? max(my_end, begin) : tile_end) - begin);
      unsigned token = first;
      if constexpr (Pair > 1) {
#pragma unroll 1
        for (; token + Pair <= last; token += Pair) {
          float scores[Pair];
#pragma unroll
          for (unsigned p = 0; p < Pair; ++p)
            scores[p] = score_of(keys[token + p]);
#pragma unroll
          for (unsigned p = 0; p < Pair; ++p)
            update(scores[p], values[token + p]);
        }
      }
#pragma unroll 1
      for (; token < last; ++token)
        update(score_of(keys[token]), values[token]);
      __syncthreads();
    }
  }

  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * kMaxSplits) + split;
  if (lane == 0) {
    row_scratch[(partition_index * 2) + 0] = running_max;
    row_scratch[(partition_index * 2) + 1] = running_sum;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * kMaxSplits * 2;
  auto* partial_outputs =
      reinterpret_cast<float4*>(row_scratch + stats_elements);
  auto* partial_output = partial_outputs + (partition_index * vector_width);
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    partial_output[lane + (vector / 2) * Lanes + (vector % 2) * 32] =
        output_values[vector];
  }
}

// Verbatim copy of QwenDecodeSplitKAttentionReduceKernel (attention_decode.hip).
__launch_bounds__(32) __global__ void VerifyAttentionReduce27(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;
  constexpr std::size_t max_splits = kMaxSplits;

  const std::size_t row = blockIdx.y;
  if (gate != nullptr)
    gate += row * num_heads * head_dim;
  out_context += row * num_heads * head_dim;
  scratch += row * num_heads * max_splits * (head_dim + 2);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max =
        fmaxf(global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  float global_sum = 0.0F;
  float4 results[vectors_per_lane]{};
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partition_index = head_partition_offset + split;
    const float partition_scale =
        expf(scratch[partition_index * 2] - global_max);
    global_sum += scratch[partition_index * 2 + 1] * partition_scale;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 partial = partial_outputs[(partition_index * vector_width) +
                                             lane + (vector * 32)];
      auto& result = results[vector];
      result.x += partial.x * partition_scale;
      result.y += partial.y * partition_scale;
      result.z += partial.z * partition_scale;
      result.w += partial.w * partition_scale;
    }
  }
  const float inverse_sum = 1.0F / global_sum;

#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    auto result = results[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

}  // namespace

bool TryLaunchVerifyAttention27(const float* q, const float* gate,
                                const void* k_cache_f16,
                                const void* v_cache_f16, float* out_context,
                                std::uint32_t layer_idx, std::uint32_t start_pos,
                                std::size_t batch_size,
                                std::uint32_t max_context,
                                std::uint32_t num_heads,
                                std::uint32_t num_kv_heads,
                                std::uint32_t head_dim, hipStream_t stream,
                                std::span<float> scratch) {
  // Same route conditions as production's shared-KV split-K verification
  // kernel (LaunchCausalDecodeAttention + LaunchSplitKAttentionHalfPartials).
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(num_heads, head_dim);
  if (batch_size < 2 || batch_size > 8 || k_cache_f16 == nullptr ||
      v_cache_f16 == nullptr || head_dim != kHeadDim || num_kv_heads == 0 ||
      num_heads != kGroup * num_kv_heads ||
      static_cast<std::size_t>(start_pos) + 1 < 512 ||
      scratch.size() / batch_size < row_scratch ||
      !detail::IsSplitKDecodeAttentionSupported(
          static_cast<std::size_t>(start_pos) + 1, num_heads, num_kv_heads,
          head_dim))
    return false;
  const auto splits = detail::SelectDecodeAttentionSplitCount(
      static_cast<std::size_t>(start_pos) + 1);
  const auto* k = static_cast<const __half*>(k_cache_f16);
  const auto* v = static_cast<const __half*>(v_cache_f16);
  const auto rows = static_cast<unsigned>(batch_size);
  // Measured on gfx1151 at 6K-48K context (kernel-bench/kb27_attn, alternating
  // medians): per-row blocks are fastest at two, three, five and six rows,
  // shared four-row blocks at four and eight, one four-row block plus three
  // per-row blocks at seven. Every row keeps its own partitions, so the
  // grouping never changes the result.
  const unsigned groups = rows == 4 || rows == 7 ? 1 : rows == 8 ? 2 : 0;
  if (groups != 0) {
    VerifyAttentionPartials27<4, 2, false>
        <<<dim3(num_kv_heads, splits, groups), 4 * 96, 0, stream>>>(
            q, k, v, scratch.data(), layer_idx, start_pos, max_context,
            num_heads, num_kv_heads, splits, 0);
  }
  if (rows > groups * 4) {
    VerifyAttentionPartials27<1, 1, true>
        <<<dim3(num_kv_heads, rows - groups * 4, splits), 96, 0, stream>>>(
            q, k, v, scratch.data(), layer_idx, start_pos, max_context,
            num_heads, num_kv_heads, splits, groups * 4);
  }
  VerifyAttentionReduce27<<<dim3(num_heads, rows), 32, 0, stream>>>(
      gate, scratch.data(), out_context, num_heads, splits);
  return true;
}

}  // namespace gufo::hip
