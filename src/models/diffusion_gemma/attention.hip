#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/device.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"

namespace gufo::models::diffusion_gemma::kernels {
namespace {

constexpr int kQueries = 64;  // Per block: 4 groups of 16.

/// Shape of one attention block for a head dimension: the head dimension is
/// split into kParts slices (one wave per query group and slice), and keys
/// are consumed kKeys at a time. kWaves is the target waves per SIMD: for
/// 512, two 16-wave blocks share a WGP (192 registers, a small spill) so
/// that one block's barriers are covered by the other's work.
template<int HD>
struct Tile;
template<>
struct Tile<256> {
  static constexpr int kParts = 2;
  static constexpr int kKeys = 32;
  static constexpr int kWaves = 4;
};
template<>
struct Tile<512> {
  static constexpr int kParts = 4;
  static constexpr int kKeys = 16;
  static constexpr int kWaves = 8;
};

// One block: 64 queries of one head (optionally one slice of the keys).
// Wave w handles query group w%4 and head-dimension slice w/4: its 16x128
// query fragment stays in registers for the whole pass. Each key tile is
// staged once in LDS (K row-major, V transposed), every wave computes its
// slice's share of QK, the slices are summed in LDS for an online softmax,
// and each wave accumulates P V for its 128 output dimensions.
template<int HD>
__global__ void __launch_bounds__(128 * Tile<HD>::kParts)
    __attribute__((amdgpu_waves_per_eu(Tile<HD>::kWaves)))
    AttentionKernel(const bf16_t* q, const bf16_t* k_cache,
                    const bf16_t* v_cache, bf16_t* out, int tokens, int q_heads,
                    int group, int capacity, int kv_len, int q_pos0, int k_pos0,
                    bool causal, int window, int split_keys, float* partial_out,
                    float* partial_stats) {
  constexpr int kParts = Tile<HD>::kParts;
  constexpr int kKeys = Tile<HD>::kKeys;
  constexpr int kThreads = 128 * kParts;
  constexpr int kSlice = HD / kParts;  // 128 dimensions per wave.
  constexpr int kFrags = kSlice / 16;  // 8 fragments.
  constexpr int kKPitch = HD + 8;      // Padded K row.
  constexpr int kVPitch = kKeys + 8;   // Padded transposed V row.
  constexpr int kRowThreads = kThreads / kQueries;
  constexpr int kPerThread = kKeys / kRowThreads;
  __shared__ __attribute__((aligned(16))) bf16_t kt[kKeys][kKPitch];
  __shared__ __attribute__((aligned(16))) bf16_t vt[HD][kVPitch];
  __shared__ float partial[kParts][kQueries][kKeys + 1];
  __shared__ __attribute__((aligned(16))) bf16_t p[kQueries][kKeys + 8];
  __shared__ float row_max[kQueries], row_sum[kQueries], correction[kQueries];

  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int group_q = wave % 4, part = wave / 4;
  const int head = blockIdx.y, kv_head = head / group, split = blockIdx.z;
  const int t0 = blockIdx.x * kQueries;
  const bf16_t* keys =
      k_cache + static_cast<std::size_t>(kv_head) * capacity * HD;
  const bf16_t* values =
      v_cache + static_cast<std::size_t>(kv_head) * capacity * HD;

  if (tid < kQueries) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0;
  }

  const int last_t = min(t0 + kQueries, tokens) - 1;
  const int q_lo = q_pos0 + t0, q_hi = q_pos0 + last_t;
  int slot_end = kv_len;
  if (causal)
    slot_end = min(slot_end, q_hi - k_pos0 + 1);
  int slot_begin = 0;
  if (window > 0)
    slot_begin = max(0, q_lo - window + 1 - k_pos0);
  slot_begin = (slot_begin / kKeys) * kKeys;
  if (partial_out != nullptr) {
    // This block covers one split_keys-wide slice of the key range.
    slot_begin = max(slot_begin, split * split_keys);
    slot_end = min(slot_end, (split + 1) * split_keys);
  }

  // This wave's query fragments, resident for the whole pass.
  const int my_t = t0 + group_q * 16 + lane % 16;
  Fragment qf[kFrags];
  {
    const bf16_t* my_q =
        q +
        (static_cast<std::size_t>(min(my_t, tokens - 1)) * q_heads + head) *
            HD +
        part * kSlice;
#pragma unroll
    for (int f = 0; f < kFrags; ++f)
      qf[f] = my_t < tokens ? LoadFragment(my_q + f * 16) : Fragment{};
  }
  Accumulator acc[kFrags];
#pragma unroll
  for (int c = 0; c < kFrags; ++c)
    acc[c] = Accumulator{};

  // K and V pieces of the next tile are loaded into registers while the
  // current tile is computed.
  constexpr int kPieces = kKeys * HD / 8;
  constexpr int kLoads = kPieces / kThreads;
  uint4 k_next[kLoads], v_next[kLoads];
  const auto fetch = [&](int tile) {
#pragma unroll
    for (int it = 0; it < kLoads; ++it) {
      const int index = it * kThreads + tid;
      const int key = index / (HD / 8), d0 = (index % (HD / 8)) * 8;
      const int slot = tile + key;
      k_next[it] = slot < kv_len
                       ? *reinterpret_cast<const uint4*>(
                             keys + static_cast<std::size_t>(slot) * HD + d0)
                       : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int it = 0; it < kLoads; ++it) {
      // Consecutive threads take consecutive keys: conflict-free LDS writes.
      const int index = it * kThreads + tid;
      const int key = index % kKeys, d0 = (index / kKeys) * 8;
      const int slot = tile + key;
      v_next[it] = slot < kv_len
                       ? *reinterpret_cast<const uint4*>(
                             values + static_cast<std::size_t>(slot) * HD + d0)
                       : uint4{0, 0, 0, 0};
    }
  };
  if (slot_begin < slot_end)
    fetch(slot_begin);
  for (int base = slot_begin; base < slot_end; base += kKeys) {
    // Stage K (row-major) and V (transposed) once the previous tile's LDS
    // is consumed.
    __syncthreads();
#pragma unroll
    for (int it = 0; it < kLoads; ++it) {
      const int index = it * kThreads + tid;
      const int key = index / (HD / 8), d0 = (index % (HD / 8)) * 8;
      *reinterpret_cast<uint4*>(&kt[key][d0]) = k_next[it];
    }
#pragma unroll
    for (int it = 0; it < kLoads; ++it) {
      const int index = it * kThreads + tid;
      const int key = index % kKeys, d0 = (index / kKeys) * 8;
      const bf16_t* v8 = reinterpret_cast<const bf16_t*>(&v_next[it]);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        vt[d0 + j][key] = v8[j];
    }
    __syncthreads();
    if (base + kKeys < slot_end)
      fetch(base + kKeys);
    // This slice's share of QK for the wave's 16 queries.
#pragma unroll
    for (int kc = 0; kc < kKeys / 16; ++kc) {
      Accumulator dot{};
#pragma unroll
      for (int f = 0; f < kFrags; ++f)
        dot =
            Mma(qf[f],
                LoadFragment(&kt[kc * 16 + lane % 16][part * kSlice + f * 16]),
                dot);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        partial[part][group_q * 16 + 2 * i + lane / 16][kc * 16 + lane % 16] =
            dot[i];
    }
    __syncthreads();
    // Online softmax: kRowThreads threads per query row.
    {
      const int r = tid / kRowThreads, sub = tid % kRowThreads;
      const int t = t0 + r, q_position = q_pos0 + t;
      float s[kPerThread];
      float tile_max = -INFINITY;
#pragma unroll
      for (int j = 0; j < kPerThread; ++j) {
        const int key = sub * kPerThread + j, slot = base + key;
        const int k_position = k_pos0 + slot;
        bool visible = slot < kv_len && slot < slot_end;
        if (causal)
          visible = visible && k_position <= q_position;
        if (window > 0)
          visible = visible && k_position > q_position - window;
        float sum = 0;
#pragma unroll
        for (int pp = 0; pp < kParts; ++pp)
          sum += partial[pp][r][key];
        s[j] = visible ? sum : -INFINITY;
        tile_max = fmaxf(tile_max, s[j]);
      }
#pragma unroll
      for (int offset = 1; offset < kRowThreads; offset *= 2)
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 32));
      const float previous = row_max[r];
      const float next = fmaxf(previous, tile_max);
      float scale = 1.0F, sum = 0;
      if (next == -INFINITY) {
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
          p[r][sub * kPerThread + j] = 0;
      } else {
        scale = expf(previous - next);
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
          const bf16_t value = ToBf16(expf(s[j] - next));
          p[r][sub * kPerThread + j] = value;
          sum += ToFloat(value);
        }
      }
#pragma unroll
      for (int offset = 1; offset < kRowThreads; offset *= 2)
        sum += __shfl_xor(sum, offset, 32);
      if (sub == 0) {
        row_max[r] = next;
        row_sum[r] = row_sum[r] * scale + sum;
        correction[r] = scale;
      }
    }
    __syncthreads();
    // O = O * correction + P V for this wave's rows and slice.
    float rescale[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      rescale[i] = correction[group_q * 16 + 2 * i + lane / 16];
    Fragment pf[kKeys / 16];
#pragma unroll
    for (int kc = 0; kc < kKeys / 16; ++kc)
      pf[kc] = LoadFragment(&p[group_q * 16 + lane % 16][kc * 16]);
#pragma unroll
    for (int c = 0; c < kFrags; ++c) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[c][i] *= rescale[i];
      const int n = part * kSlice + c * 16 + lane % 16;
#pragma unroll
      for (int kc = 0; kc < kKeys / 16; ++kc)
        acc[c] = Mma(pf[kc], LoadFragment(&vt[n][kc * 16]), acc[c]);
    }
  }
  __syncthreads();
  if (partial_out != nullptr) {
    // Unnormalised output and the row's running max and sum, merged later.
    const std::size_t rows = static_cast<std::size_t>(tokens) * q_heads;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = group_q * 16 + 2 * i + lane / 16;
      const int t = t0 + r;
      if (t >= tokens)
        continue;
      const std::size_t row = static_cast<std::size_t>(t) * q_heads + head;
      float* dst =
          partial_out + (split * rows + row) * HD + part * kSlice + lane % 16;
#pragma unroll
      for (int c = 0; c < kFrags; ++c)
        dst[c * 16] = acc[c][i];
      if (part == 0 && lane % 16 == 0) {
        partial_stats[(split * rows + row) * 2] = row_max[r];
        partial_stats[(split * rows + row) * 2 + 1] = row_sum[r];
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = group_q * 16 + 2 * i + lane / 16;
    const int t = t0 + r;
    if (t >= tokens)
      continue;
    const float denominator = row_sum[r];
    const float inverse = denominator > 0 ? 1.0F / denominator : 0.0F;
    bf16_t* dst = out + (static_cast<std::size_t>(t) * q_heads + head) * HD +
                  part * kSlice + lane % 16;
#pragma unroll
    for (int c = 0; c < kFrags; ++c)
      dst[c * 16] = ToBf16(acc[c][i] * inverse);
  }
}

// Merges split partials: out = sum_s e^(m_s - M) O_s / sum_s e^(m_s - M) l_s.
template<int HD>
__global__ void __launch_bounds__(256)
    MergeKernel(const float* partial_out, const float* partial_stats,
                bf16_t* out, std::size_t rows, int splits) {
  const std::size_t row = blockIdx.x;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s)
    m = fmaxf(m, partial_stats[(s * rows + row) * 2]);
  float denominator = 0;
  for (int s = 0; s < splits; ++s) {
    const float ms = partial_stats[(s * rows + row) * 2];
    if (ms != -INFINITY)
      denominator += expf(ms - m) * partial_stats[(s * rows + row) * 2 + 1];
  }
  const float inverse = denominator > 0 ? 1.0F / denominator : 0.0F;
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float sum = 0;
    for (int s = 0; s < splits; ++s) {
      const float ms = partial_stats[(s * rows + row) * 2];
      if (ms != -INFINITY)
        sum += expf(ms - m) * partial_out[(s * rows + row) * HD + d];
    }
    out[row * HD + d] = ToBf16(sum * inverse);
  }
}

template<int HD>
void Launch(const bf16_t* q, const bf16_t* k_cache, const bf16_t* v_cache,
            bf16_t* out, int tokens, int q_heads, int kv_heads, int capacity,
            int kv_len, int q_pos0, int k_pos0, bool causal, int window,
            hipStream_t stream, const SplitWorkspace& workspace) {
  constexpr int kKeys = Tile<HD>::kKeys;
  const int query_tiles = (tokens + kQueries - 1) / kQueries;
  const int group = q_heads / kv_heads;
  // Too few blocks to fill the GPU over a long key range: split the keys
  // across blocks and merge. Aim for about 512 blocks, 512+ keys each.
  int splits = 1;
  const int blocks = query_tiles * q_heads;
  if (workspace.out != nullptr && !causal && blocks < 512)
    splits = std::max(1, std::min({(512 + blocks - 1) / blocks, kv_len / 512,
                                   workspace.max_splits}));
  const std::size_t rows = static_cast<std::size_t>(tokens) * q_heads;
  if (splits > 1 && rows * splits * HD > workspace.capacity)
    splits = 1;
  int split_keys = kv_len;
  if (splits > 1)
    split_keys = ((kv_len + splits - 1) / splits + kKeys - 1) / kKeys * kKeys;
  const dim3 grid(query_tiles, q_heads, splits);
  float* partial_out = splits > 1 ? workspace.out : nullptr;
  float* partial_stats = splits > 1 ? workspace.stats : nullptr;
  AttentionKernel<HD><<<grid, 128 * Tile<HD>::kParts, 0, stream>>>(
      q, k_cache, v_cache, out, tokens, q_heads, group, capacity, kv_len,
      q_pos0, k_pos0, causal, window, split_keys, partial_out, partial_stats);
  if (splits > 1)
    MergeKernel<HD><<<rows, 256, 0, stream>>>(partial_out, partial_stats, out,
                                              rows, splits);
}

}  // namespace

void Attention(const bf16_t* q, const bf16_t* k_cache, const bf16_t* v_cache,
               bf16_t* out, int tokens, int q_heads, int kv_heads, int head_dim,
               int capacity, int kv_len, int q_pos0, int k_pos0, bool causal,
               int window, hipStream_t stream,
               const SplitWorkspace& workspace) {
  if (head_dim == 256)
    Launch<256>(q, k_cache, v_cache, out, tokens, q_heads, kv_heads, capacity,
                kv_len, q_pos0, k_pos0, causal, window, stream, workspace);
  else if (head_dim == 512)
    Launch<512>(q, k_cache, v_cache, out, tokens, q_heads, kv_heads, capacity,
                kv_len, q_pos0, k_pos0, causal, window, stream, workspace);
  else
    throw std::invalid_argument("Attention: unsupported head dimension");
  CheckLaunch("Attention");
}

}  // namespace gufo::models::diffusion_gemma::kernels
