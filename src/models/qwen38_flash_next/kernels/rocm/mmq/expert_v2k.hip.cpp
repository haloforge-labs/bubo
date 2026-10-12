// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Adapted from https://github.com/Niko1221/Strata
// V2K/S26/S27: src/kernels/cuda/iq_kernels.cu, measured base 82f46a8
// (introduced by 04e1e11afb45c040960e90c834e0fe85a475a4ab).
// Group planning: src/kernels/cuda/verify_kernels.cu.
// Gufo adaptation: separate tensor bases and gufo vec_dot arithmetic.
// Full MIT permission and warranty notice: VENDOR.md in this directory.
#include "expert_v2k.hpp"

#include "qfn_mmq.h"
#include "qfn_mmq_prelude.h"
#include "s26_tsum.hpp"

namespace qfn_mmq {
#include "vecdotq.hpp"
namespace v2k {

static_assert(sizeof(block_q8_1) == kQ8BlockBytes);
constexpr int kChunk = 4;
constexpr int kRows = 4;
constexpr int kWaves = 8;

struct Plan {
  int32_t *expert, *start, *count, *token, *destination;
  block_q8_1 *xq, *hq;
  Plan(void* scratch, int tokens) {
    const Layout l(tokens);
    auto* p = static_cast<char*>(scratch);
    expert = reinterpret_cast<int32_t*>(p + l.expert);
    start = reinterpret_cast<int32_t*>(p + l.start);
    count = reinterpret_cast<int32_t*>(p + l.count);
    token = reinterpret_cast<int32_t*>(p + l.token);
    destination = reinterpret_cast<int32_t*>(p + l.destination);
    xq = reinterpret_cast<block_q8_1*>(p + l.xq);
    hq = reinterpret_cast<block_q8_1*>(p + l.hq);
  }
};

// Measured-base resident_plan_par_kernel's first/rank/size prefix plan.
// All entries, including negative ids, receive a compact entry/destination.
__global__ void expert_v2k_group_kernel(const int32_t* __restrict__ ids,
                                        int entries, int32_t* expert,
                                        int32_t* start, int32_t* count,
                                        int32_t* token, int32_t* destination) {
  __shared__ int s_id[128], s_first[128], s_size[128];
  const int i = threadIdx.x;
  int id = -1;
  if (i < entries) {
    id = ids[i] < 0 ? -1 : ids[i];
    s_id[i] = id;
  }
  __syncthreads();
  int first = i, rank = 0, size = 0;
  if (i < entries) {
    for (int j = 0; j < entries; ++j) {
      const bool same = s_id[j] == id;
      if (same && j < first)
        first = j;
      if (same && j < i)
        ++rank;
      size += same;
    }
    s_first[i] = first;
    s_size[i] = size;
  }
  __syncthreads();
  if (i < entries) {
    int group = 0, begin = 0;
    for (int j = 0; j < first; ++j) {
      if (s_first[j] == j) {
        ++group;
        begin += s_size[j];
      }
    }
    if (i == first) {
      expert[group] = id;
      start[group] = begin;
    }
    destination[begin + rank] = i;
    token[begin + rank] = i / kUsed;
    if (i == 0) {
      int groups = 0;
      for (int j = 0; j < entries; ++j)
        groups += s_first[j] == j;
      start[groups] = entries;
      *count = groups;
    }
  }
}

template<ggml_type Type>
struct Fragment;
template<>
struct Fragment<GGML_TYPE_Q4_K> {
  using W = Q4MoeFragment;
  static constexpr int qk = QK_K, ipb = QI4_K / 2, step = 2;
  __device__ static W Load(const void* row, int block, int iqs) {
    return W(row, block, iqs);
  }
  __device__ static float Apply(const W& w, const block_q8_1* x, int iqs) {
    return w.Dot(x, iqs);
  }
};
template<>
struct Fragment<GGML_TYPE_Q5_1> {
  struct W {
    int vl[2], vh[2];
    half2 dm;
  };
  static constexpr int qk = QK5_1, ipb = QI5_1 / 2, step = 2;
  __device__ static W Load(const void* row, int block, int iqs) {
    const auto* b = static_cast<const block_q5_1*>(row) + block;
    W w;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      w.vl[i] = get_int_b4(b->qs, iqs + i);
      w.vh[i] = get_int_b4(b->qh, 0) >> (4 * (iqs + i));
    }
    w.dm = b->dm;
    return w;
  }
  __device__ static float Apply(const W& w, const block_q8_1* x, int iqs) {
    int u[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      u[2 * i] = get_int_b4(x->qs, iqs + i);
      u[2 * i + 1] = get_int_b4(x->qs, iqs + i + QI5_1);
    }
    return vec_dot_q5_1_q8_1_impl<2>(w.vl, w.vh, u, w.dm, x->ds);
  }
};
template<>
struct Fragment<GGML_TYPE_Q8_0> {
  struct W {
    int v[2];
    float d;
  };
  static constexpr int qk = QK8_0, ipb = QI8_0 / 2, step = 2;
  __device__ static W Load(const void* row, int block, int iqs) {
    const auto* b = static_cast<const block_q8_0*>(row) + block;
    W w;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      w.v[i] = get_int_b2(b->qs, iqs + i);
    w.d = __half2float(b->d);
    return w;
  }
  __device__ static float Apply(const W& w, const block_q8_1* x, int iqs) {
    int u[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      u[i] = get_int_b4(x->qs, iqs + i);
    const float d = __low2float(x->ds);
    return vec_dot_q8_0_q8_1_impl<float, 2>(w.v, u, w.d, d);
  }
};

template<ggml_type TG>
__global__ void __launch_bounds__(256)
    expert_v2k_gu_kernel(const void* __restrict__ gate_w,
                         const void* __restrict__ up_w,
                         const int32_t* __restrict__ expert,
                         const int32_t* __restrict__ start,
                         const int32_t* __restrict__ count,
                         const int32_t* __restrict__ token,
                         const block_q8_1* __restrict__ xq,
                         float* __restrict__ gate, float* __restrict__ up) {
  using F = Fragment<TG>;
  __shared__ block_q8_1 s_x[kChunk * 80];
  const int g = blockIdx.y;
  if (g >= *count)
    return;  // Uniform before the chunk's LDS barriers.
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * kWaves * kRows + wave;
  const int e0 = start[g], e1 = start[g + 1], id = expert[g];
  const bool is_up = row0 >= kFf;
  const int rr = is_up ? row0 - kFf : row0;
  if (id < 0) {
    for (int e = e0; e < e1; ++e)
#pragma unroll
      for (int q = 0; q < kRows; ++q)
        if (lane == 0)
          (is_up ? up : gate)[std::size_t(e) * kFf + rr + 8 * q] = 0.0f;
    return;
  }
  constexpr std::size_t row_bytes = (kHidden / QK_K) * sizeof(block_q4_K);
  const auto* base = static_cast<const char*>(is_up ? up_w : gate_w) +
                     std::size_t(id) * kFf * row_bytes;
  const void* wr[kRows];
#pragma unroll
  for (int q = 0; q < kRows; ++q)
    wr[q] = base + std::size_t(rr + 8 * q) * row_bytes;
  for (int e = e0; e < e1; e += kChunk) {
    const int n = min(kChunk, e1 - e);
    __syncthreads();
    for (int i = threadIdx.x; i < n * 80; i += 256) {
      const int c = i / 80, j = i % 80;
      memcpy(&s_x[i], &xq[std::size_t(token[e + c]) * 80 + j],
             sizeof(block_q8_1));
    }
    __syncthreads();
    float sums[kRows][kChunk] = {};
    for (int k = lane; k < (kHidden / F::qk) * F::ipb; k += 32) {
      const int kb = k / F::ipb, iqs = F::step * (k % F::ipb);
      const typename F::W w[kRows] = {
          F::Load(wr[0], kb, iqs), F::Load(wr[1], kb, iqs),
          F::Load(wr[2], kb, iqs), F::Load(wr[3], kb, iqs)};
#pragma unroll
      for (int q = 0; q < kRows; ++q)
#pragma unroll
        for (int c = 0; c < kChunk; ++c)
          if (c < n)
            sums[q][c] += F::Apply(w[q], s_x + c * 80 + kb * (F::qk / 32), iqs);
    }
    float v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      v[j] = sums[j / kChunk][j % kChunk];
    const float sum = s26ts::tsum<16>(v, lane);
    const int j = s26ts::tsum_token<16>(lane), c = j % kChunk;
    if (lane == s26ts::tsum_lane<16>(j) && c < n) {
      const int q = j / kChunk;
      (is_up ? up : gate)[std::size_t(e + c) * kFf + rr + 8 * q] =
          isfinite(sum) ? sum : 0.0f;
    }
  }
}

__global__ void expert_v2k_swiglu_q8_1_kernel(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              block_q8_1* __restrict__ hq,
                                              int values) {
#pragma clang fp contract(off)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= values)
    return;  // Complete 32-value quantization waves.
  const float g = gate[i];
  const float xi = (g * (1.0f / (1.0f + __expf(-g)))) * up[i];
  float amax = fabsf(xi), sum = xi;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }
  const float d = amax / 127.0f;
  hq[i / 32].qs[i % 32] = amax == 0.0f ? 0 : roundf(xi / d);
  if (i % 32 == 0)
    hq[i / 32].ds = make_half2(d, sum);
}

template<ggml_type TD>
__global__ void __launch_bounds__(256)
    expert_v2k_down_kernel(const void* __restrict__ weights,
                           const int32_t* __restrict__ expert,
                           const int32_t* __restrict__ start,
                           const int32_t* __restrict__ count,
                           const int32_t* __restrict__ destination,
                           const block_q8_1* __restrict__ hq,
                           float* __restrict__ out) {
  using F = Fragment<TD>;
  __shared__ block_q8_1 s_h[kChunk * 20];
  const int g = blockIdx.y;
  if (g >= *count)
    return;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r0 = blockIdx.x * kWaves * kRows + wave;
  const int e0 = start[g], e1 = start[g + 1], id = expert[g];
  if (id < 0) {
    for (int e = e0; e < e1; ++e)
#pragma unroll
      for (int q = 0; q < kRows; ++q)
        if (lane == 0)
          out[std::size_t(destination[e]) * kHidden + r0 + 8 * q] = 0.0f;
    return;
  }
  constexpr std::size_t block_bytes =
      TD == GGML_TYPE_Q5_1 ? sizeof(block_q5_1) : sizeof(block_q8_0);
  constexpr std::size_t row_bytes = (kFf / F::qk) * block_bytes;
  const auto* base =
      static_cast<const char*>(weights) + std::size_t(id) * kHidden * row_bytes;
  const void* wr[kRows];
#pragma unroll
  for (int q = 0; q < kRows; ++q)
    wr[q] = base + std::size_t(r0 + 8 * q) * row_bytes;
  for (int e = e0; e < e1; e += kChunk) {
    const int n = min(kChunk, e1 - e);
    __syncthreads();
    for (int i = threadIdx.x; i < n * 20; i += 256) {
      const int c = i / 20, j = i % 20;
      memcpy(&s_h[i], &hq[std::size_t(e + c) * 20 + j], sizeof(block_q8_1));
    }
    __syncthreads();
    float sums[kRows][kChunk] = {};
    for (int k = lane; k < (kFf / F::qk) * F::ipb; k += 32) {
      const int kb = k / F::ipb, iqs = F::step * (k % F::ipb);
      const typename F::W w[kRows] = {
          F::Load(wr[0], kb, iqs), F::Load(wr[1], kb, iqs),
          F::Load(wr[2], kb, iqs), F::Load(wr[3], kb, iqs)};
#pragma unroll
      for (int q = 0; q < kRows; ++q)
#pragma unroll
        for (int c = 0; c < kChunk; ++c)
          if (c < n)
            sums[q][c] += F::Apply(w[q], s_h + c * 20 + kb, iqs);
    }
    float v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      v[j] = sums[j / kChunk][j % kChunk];
    const float sum = s26ts::tsum<16>(v, lane);
    const int j = s26ts::tsum_token<16>(lane), c = j % kChunk;
    if (lane == s26ts::tsum_lane<16>(j) && c < n)
      out[std::size_t(destination[e + c]) * kHidden + r0 + 8 * (j / kChunk)] =
          isfinite(sum) ? sum : 0.0f;
  }
}

bool ValidScratch(int tokens, const void* scratch, std::size_t bytes) {
  return tokens >= 2 && tokens <= kMaxTokens && scratch &&
         bytes >= Layout(tokens).bytes;
}
}  // namespace v2k
}  // namespace qfn_mmq

extern "C" size_t qfn_mmq_expert_v2k_scratch_bytes(int max_tokens) {
  return max_tokens >= 2 && max_tokens <= qfn_mmq::v2k::kMaxTokens
             ? qfn_mmq::v2k::Layout(max_tokens).bytes
             : 0;
}

extern "C" int qfn_mmq_expert_v2k_gu(const void* gate_w, const void* up_w,
                                     const float* x, const int32_t* ids,
                                     float* gate, float* up, int n_tokens,
                                     void* scratch, size_t bytes,
                                     hipStream_t stream) {
  using namespace qfn_mmq;
  if (!gate_w || !up_w || !x || !ids || !gate || !up ||
      !v2k::ValidScratch(n_tokens, scratch, bytes))
    return -1;
  const v2k::Plan p(scratch, n_tokens);
  const int rc = qfn_mmq_quantize_q8_1(x, p.xq, n_tokens, v2k::kHidden, stream);
  if (rc != 0)
    return rc;
  v2k::expert_v2k_group_kernel<<<1, 128, 0, stream>>>(
      ids, n_tokens * v2k::kUsed, p.expert, p.start, p.count, p.token,
      p.destination);
  if (hipGetLastError() != hipSuccess)
    return -2;
  v2k::expert_v2k_gu_kernel<GGML_TYPE_Q4_K>
      <<<dim3(40, n_tokens * v2k::kUsed), 256, 0, stream>>>(
          gate_w, up_w, p.expert, p.start, p.count, p.token, p.xq, gate, up);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int qfn_mmq_expert_v2k_bridge(const float* gate, const float* up,
                                         void* hq, int entries,
                                         hipStream_t stream) {
  using namespace qfn_mmq;
  if (!gate || !up || !hq || entries < 1 ||
      entries > v2k::kMaxTokens * v2k::kUsed)
    return -1;
  const int values = entries * v2k::kFf;
  v2k::expert_v2k_swiglu_q8_1_kernel<<<(values + 255) / 256, 256, 0, stream>>>(
      gate, up, static_cast<block_q8_1*>(hq), values);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int qfn_mmq_expert_v2k_down(int down_type, const void* weights,
                                       float* out, int n_tokens, void* scratch,
                                       size_t bytes, hipStream_t stream) {
  using namespace qfn_mmq;
  if (!weights || !out || !v2k::ValidScratch(n_tokens, scratch, bytes) ||
      (down_type != GGML_TYPE_Q5_1 && down_type != GGML_TYPE_Q8_0))
    return -1;
  const v2k::Plan p(scratch, n_tokens);
  const dim3 grid(80, n_tokens * v2k::kUsed);
  if (down_type == GGML_TYPE_Q5_1)
    v2k::expert_v2k_down_kernel<GGML_TYPE_Q5_1><<<grid, 256, 0, stream>>>(
        weights, p.expert, p.start, p.count, p.destination, p.hq, out);
  else
    v2k::expert_v2k_down_kernel<GGML_TYPE_Q8_0><<<grid, 256, 0, stream>>>(
        weights, p.expert, p.start, p.count, p.destination, p.hq, out);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int qfn_mmq_expert_v2k(int gu_type, int down_type,
                                  const void* gate_w, const void* up_w,
                                  const void* down_w, const float* x,
                                  const int32_t* ids, float* gate, float* up,
                                  float* out, int n_tokens, void* scratch,
                                  size_t bytes, hipStream_t stream) {
  using namespace qfn_mmq;
  if (gu_type != GGML_TYPE_Q4_K ||
      (down_type != GGML_TYPE_Q5_1 && down_type != GGML_TYPE_Q8_0) || !down_w ||
      !out)
    return -1;
  int rc = qfn_mmq_expert_v2k_gu(gate_w, up_w, x, ids, gate, up, n_tokens,
                                 scratch, bytes, stream);
  if (rc != 0)
    return rc;
  const v2k::Plan p(scratch, n_tokens);
  rc = qfn_mmq_expert_v2k_bridge(gate, up, p.hq, n_tokens * v2k::kUsed, stream);
  if (rc != 0)
    return rc;
  return qfn_mmq_expert_v2k_down(down_type, down_w, out, n_tokens, scratch,
                                 bytes, stream);
}
