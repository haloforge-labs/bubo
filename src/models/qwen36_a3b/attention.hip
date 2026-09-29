// A3B decode/verify attention: each KV head is read once per step (per row
// block, the rows of one slice sharing it through L2) for all 8 query heads
// of its group (see attention.hpp).
#include "src/models/qwen36_a3b/attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gufo::models::qwen36_a3b {
namespace {

constexpr std::uint32_t kGroup = 8;   // query heads per KV head
constexpr std::uint32_t kDim = 256;   // head dim
constexpr std::uint32_t kKeys = 8;    // keys scored per softmax update
constexpr std::uint32_t kQ = 4;       // queries per thread
constexpr std::uint32_t kWaves = 4;   // sub-slices per slice
constexpr std::uint32_t kSub = kAttnChunk / kWaves;
constexpr std::uint32_t kThreads = 32 * kWaves;
constexpr std::uint32_t kPart = kDim + 2;  // m, l, acc[256]

int g_decode_mode = -1;  // SetDecodeAttentionMode; -1 = environment

/// Sum over the 16 lanes of a DPP row (row_xmask 1, 2, 4, 8).
template <int Mask>
__device__ inline float XorRow(float v) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                         0x160 | Mask, 0xF, 0xF, false));
}
__device__ inline float RowSum16(float v) {
  v += XorRow<1>(v);
  v += XorRow<2>(v);
  v += XorRow<4>(v);
  v += XorRow<8>(v);
  return v;
}

__device__ inline std::uint32_t ActiveSlices(std::uint32_t pos0,
                                             std::uint32_t rows) {
  return (pos0 + rows + kAttnChunk - 1) / kAttnChunk;
}

/// grid (rows, kv_heads, slices), block 128: one row's 8 queries of one KV
/// group against one kAttnChunk-key slice. Wave w takes keys
/// [64w, 64w + 64) of the slice; lane 16c + d owns the four queries (heads
/// 4c..4c+3) and dims {128i + 8d + e}, so a K/V load of 16 halves feeds four
/// queries and a score is a 16-lane DPP row sum. The waves merge in LDS in a
/// fixed order, and slices are fixed from position 0: a query's arithmetic
/// depends only on its position, never on the batch width, so speculative
/// output stays identical to plain decoding. Slices past the row exit.
__global__ __launch_bounds__(kThreads) void DecodeAttentionKernel(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* partials, const std::uint32_t* start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t splits) {
  __shared__ float lds[kWaves - 1][kGroup][kPart];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t g = blockIdx.y;
  const std::uint32_t s = blockIdx.z;
  const std::uint32_t rows = gridDim.x;
  const std::uint32_t pos0 = *start_pos;
  if (s >= ActiveSlices(pos0, rows)) return;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t dl = lane & 15u;
  const std::uint32_t hq = (lane >> 4) * kQ;  // first head of the four
  const std::uint32_t h0 = g * kGroup + hq;
  const std::uint32_t k0 = s * kAttnChunk + wave * kSub;
  // This row's visible keys in the sub-slice (the four queries share it).
  const std::uint32_t end = min(k0 + kSub, pos0 + t + 1);

  const float scale = rsqrtf(static_cast<float>(kDim));
  float qv[kQ][16];
#pragma unroll
  for (std::uint32_t a = 0; a < kQ; ++a) {
    const float* qrow =
        q + (static_cast<std::size_t>(t) * heads + h0 + a) * kDim + dl * 8;
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i) {
      const float4 x = *reinterpret_cast<const float4*>(qrow + i * 128);
      const float4 y = *reinterpret_cast<const float4*>(qrow + i * 128 + 4);
      qv[a][i * 8 + 0] = x.x * scale;
      qv[a][i * 8 + 1] = x.y * scale;
      qv[a][i * 8 + 2] = x.z * scale;
      qv[a][i * 8 + 3] = x.w * scale;
      qv[a][i * 8 + 4] = y.x * scale;
      qv[a][i * 8 + 5] = y.y * scale;
      qv[a][i * 8 + 6] = y.z * scale;
      qv[a][i * 8 + 7] = y.w * scale;
    }
  }
  const std::size_t stride = static_cast<std::size_t>(kv_heads) * kDim;
  const __half* kb = k_cache + g * kDim + dl * 8;
  const __half* vb = v_cache + g * kDim + dl * 8;
  const auto load16 = [&](const __half* row, float* out) {
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i) {
      const uint4 pk = *reinterpret_cast<const uint4*>(row + i * 128);
      const auto* h2 = reinterpret_cast<const __half2*>(&pk);
#pragma unroll
      for (std::uint32_t e = 0; e < 4; ++e) {
        const float2 f = __half22float2(h2[e]);
        out[i * 8 + 2 * e] = f.x;
        out[i * 8 + 2 * e + 1] = f.y;
      }
    }
  };

  float m[kQ], l[kQ], acc[kQ][16];
#pragma unroll
  for (std::uint32_t a = 0; a < kQ; ++a) {
    m[a] = -INFINITY;
    l[a] = 0.0f;
#pragma unroll
    for (std::uint32_t x = 0; x < 16; ++x) acc[a][x] = 0.0f;
  }

  // The whole wave shares `end`, so DPP rows never see an inactive lane.
  for (std::uint32_t j0 = k0; j0 < end; j0 += kKeys) {
    float sc[kQ][kKeys];
#pragma unroll
    for (std::uint32_t u = 0; u < kKeys; ++u) {
      const std::uint32_t j = j0 + u;
      float dot[kQ] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (j < end) {
        float kf[16];
        load16(kb + static_cast<std::size_t>(j) * stride, kf);
#pragma unroll
        for (std::uint32_t a = 0; a < kQ; ++a)
#pragma unroll
          for (std::uint32_t x = 0; x < 16; ++x) dot[a] += qv[a][x] * kf[x];
      }
#pragma unroll
      for (std::uint32_t a = 0; a < kQ; ++a)
        sc[a][u] = j < end ? RowSum16(dot[a]) : -INFINITY;
    }
#pragma unroll
    for (std::uint32_t a = 0; a < kQ; ++a) {
      float mx = m[a];
#pragma unroll
      for (std::uint32_t u = 0; u < kKeys; ++u) mx = fmaxf(mx, sc[a][u]);
      const float r = __expf(m[a] - mx);  // key j0 is visible: mx is finite
      if (r != 1.0f) {
        l[a] *= r;
#pragma unroll
        for (std::uint32_t x = 0; x < 16; ++x) acc[a][x] *= r;
      }
      m[a] = mx;
#pragma unroll
      for (std::uint32_t u = 0; u < kKeys; ++u) {
        sc[a][u] = __expf(sc[a][u] - mx);  // 0 for masked keys
        l[a] += sc[a][u];
      }
    }
#pragma unroll
    for (std::uint32_t u = 0; u < kKeys; ++u) {
      if (j0 + u >= end) break;
      float vf[16];
      load16(vb + static_cast<std::size_t>(j0 + u) * stride, vf);
#pragma unroll
      for (std::uint32_t a = 0; a < kQ; ++a)
#pragma unroll
        for (std::uint32_t x = 0; x < 16; ++x) acc[a][x] += sc[a][u] * vf[x];
    }
  }

  // Waves 1..3 hand their state to wave 0, which folds them in order.
  if (wave > 0) {
#pragma unroll
    for (std::uint32_t a = 0; a < kQ; ++a) {
      float* dst = lds[wave - 1][hq + a];
      if (dl == 0) {
        dst[0] = m[a];
        dst[1] = l[a];
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 2; ++i)
#pragma unroll
        for (std::uint32_t e = 0; e < 8; ++e)
          dst[2 + i * 128 + dl * 8 + e] = acc[a][i * 8 + e];
    }
  }
  __syncthreads();
  if (wave > 0) return;
  const std::uint32_t queries = kGroup * rows;
#pragma unroll
  for (std::uint32_t a = 0; a < kQ; ++a) {
    float mx = m[a];
    for (std::uint32_t w = 0; w + 1 < kWaves; ++w)
      mx = fmaxf(mx, lds[w][hq + a][0]);
    const auto weight = [&](float mw) {
      return mw == -INFINITY ? 0.0f : __expf(mw - mx);
    };
    const float r0 = weight(m[a]);
    float sum = l[a] * r0;
    float out[16];
#pragma unroll
    for (std::uint32_t x = 0; x < 16; ++x) out[x] = acc[a][x] * r0;
    for (std::uint32_t w = 0; w + 1 < kWaves; ++w) {
      const float* src = lds[w][hq + a];
      const float rw = weight(src[0]);
      sum += src[1] * rw;
#pragma unroll
      for (std::uint32_t i = 0; i < 2; ++i)
#pragma unroll
        for (std::uint32_t e = 0; e < 8; ++e)
          out[i * 8 + e] += src[2 + i * 128 + dl * 8 + e] * rw;
    }
    float* part =
        partials + ((static_cast<std::size_t>(g) * splits + s) * queries +
                    t * kGroup + hq + a) *
                       kPart;
    if (dl == 0) {
      part[0] = mx;
      part[1] = sum;
    }
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i)
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e)
        part[2 + i * 128 + dl * 8 + e] = out[i * 8 + e];
  }
}

/// grid (heads, rows), block 256 = head dim: merges the active slices in
/// order.
__global__ void DecodeAttentionMergeKernel(const float* partials, float* out,
                                           const std::uint32_t* start_pos,
                                           std::uint32_t rows,
                                           std::uint32_t heads,
                                           std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t g = h / kGroup;
  const std::uint32_t qi = t * kGroup + h % kGroup;
  const std::uint32_t queries = kGroup * rows;
  const std::uint32_t active = ActiveSlices(*start_pos, rows);
  const auto at = [&](std::uint32_t s) {
    return partials +
           ((static_cast<std::size_t>(g) * splits + s) * queries + qi) * kPart;
  };
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < active; ++s) m = fmaxf(m, at(s)[0]);
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < active; ++s) {
    const float* part = at(s);
    const float scale = part[0] == -INFINITY ? 0.0f : __expf(part[0] - m);
    l += part[1] * scale;
    acc += part[2 + i] * scale;
  }
  out[(static_cast<std::size_t>(t) * heads + h) * kDim + i] = acc / l;
}

// --- Prefill attention (F16 WMMA). ---
// gfx11 WMMA (wave32): A lane l holds row l % 16 (16 K values), B lane l
// column l % 16, replicated across the half-waves; D lane l holds column
// l % 16 with accumulator i at row 2i + l / 16.
using HFrag = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Acc = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kPQ = 16;       // queries per block
constexpr std::uint32_t kPH = 4;        // heads per block (one wave each)
constexpr std::uint32_t kPK = 16;       // keys per tile
constexpr std::uint32_t kRowS = 256 + 8;  // q/k LDS row stride (halves)
constexpr std::uint32_t kVtS = kPK + 8;   // transposed V / P row stride

template <int Mask>
__device__ inline float XorRowMax(float v) {
  return fmaxf(v, __builtin_bit_cast(
                      float, __builtin_amdgcn_update_dpp(
                                 0, __builtin_bit_cast(int, v), 0x160 | Mask,
                                 0xF, 0xF, false)));
}
__device__ inline float RowMax16(float v) {
  v = XorRowMax<1>(v);
  v = XorRowMax<2>(v);
  v = XorRowMax<4>(v);
  return XorRowMax<8>(v);
}

/// grid (ceil(n / 16), kv_heads * 2), block 256: queries t0..t0+15 of heads
/// hb..hb+3 (head hh = wave pair 2hh, 2hh + 1; wave part p = dims 128p..).
/// Each wave keeps its head's Q for its 128 dims as 8 A fragments. Per
/// 32-key tile (K and V^T staged once in LDS for all four heads):
///  - S (16 queries x 32 keys, two 16 x 16 tiles): each wave of the pair
///    sums its 128 dims, partials are exchanged through LDS (a + b == b + a,
///    so both hold the same S);
///  - both run the online softmax on it; the even wave writes P (F16) to LDS
///    in the A layout;
///  - O: wave 2hh + p accumulates head hh's dims 128p.. (8 tiles).
/// kPrecise bit 1: P is scaled by 2^15 (no F16 underflow of small weights
/// over long contexts) and split into F16 hi + lo parts (two P V products,
/// ~22 bits); bit 2: Q is split the same way (two Q K^T products).
template <unsigned kPrecise>
__global__ __launch_bounds__(256) void PrefillAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out, std::uint32_t start, std::uint32_t n,
    std::uint32_t heads, std::uint32_t kv_heads) {
  constexpr std::uint32_t kKeys = 32;              // keys per tile
  constexpr std::uint32_t kVt = kKeys + 8;         // V^T / P row stride
  constexpr std::uint32_t kSteps = kDim / 32;      // K steps per wave (128 dims)
  constexpr std::uint32_t kTiles = kDim / 16 / 2;  // O tiles per wave
  __shared__ __attribute__((aligned(16))) _Float16 k_s[kKeys][kRowS];
  __shared__ __attribute__((aligned(16))) _Float16 vt_s[kDim][kVt];
  // Partial S per (head, half) in accumulator layout (two tiles); P (F16,
  // A layout) reuses the odd half's slot once the even wave has read it.
  __shared__ __attribute__((aligned(16))) float s_part[kPH][2][16][32];
  static_assert(sizeof(float) * 32 * 16 >= sizeof(_Float16) * kPQ * kVt);
  constexpr bool kSplitP = (kPrecise & 1u) != 0;
  constexpr bool kSplitQ = (kPrecise & 2u) != 0;
  constexpr float kPScale = kSplitP ? 32768.0f : 1.0f;
  __shared__ __attribute__((aligned(16)))
  _Float16 p_lo_all[kSplitP ? kPH : 1][kPQ][kVt];
  const std::uint32_t t0 = blockIdx.x * kPQ;
  const std::uint32_t g = blockIdx.y / (kGroup / kPH);
  const std::uint32_t hb = g * kGroup + (blockIdx.y % (kGroup / kPH)) * kPH;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t r = lane & 15u;
  const std::uint32_t half = lane >> 4;
  const std::uint32_t hh = wave >> 1;
  const std::uint32_t part = wave & 1u;
  const float scale = rsqrtf(static_cast<float>(kDim));
  auto* p_s = reinterpret_cast<_Float16(*)[kVt]>(&s_part[hh][1][0][0]);
  auto* p_lo = p_lo_all[kSplitP ? hh : 0];

  // This wave's Q fragments: query row r of head hb + hh, dims 128 part +
  // 16 ks.., scaled, F16.
  HFrag qa[kSteps];
  HFrag qa_lo[kSplitQ ? kSteps : 1];
  {
    const std::uint32_t t = min(t0 + r, n - 1);
    const float* qrow = q + (static_cast<std::size_t>(t) * heads + hb + hh) *
                                kDim + part * (kDim / 2);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
#pragma unroll
      for (std::uint32_t e4 = 0; e4 < 4; ++e4) {
        const float4 v =
            *reinterpret_cast<const float4*>(qrow + ks * 16 + e4 * 4);
        const float xs[4] = {v.x * scale, v.y * scale, v.z * scale,
                             v.w * scale};
#pragma unroll
        for (std::uint32_t e = 0; e < 4; ++e) {
          const _Float16 hi = static_cast<_Float16>(xs[e]);
          qa[ks][e4 * 4 + e] = hi;
          if constexpr (kSplitQ)
            qa_lo[ks][e4 * 4 + e] =
                static_cast<_Float16>(xs[e] - static_cast<float>(hi));
        }
      }
    }
  }
  const std::uint32_t last = min(t0 + kPQ, n) - 1;
  const std::uint32_t kend = start + last + 1;  // keys [0, kend)
  const std::size_t kvs = static_cast<std::size_t>(kv_heads) * kDim;

  Acc o[kTiles];
#pragma unroll
  for (std::uint32_t b = 0; b < kTiles; ++b) o[b] = Acc{};
  float m[8], l[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f;
  }

  // K/V tiles are fetched one tile ahead into registers.
  constexpr std::uint32_t kFetch = kKeys * (kDim / 8) / 256;
  uint4 nk[kFetch], nv[kFetch];
  // K: lanes over the 16-byte chunks of one key row (row-major LDS writes);
  // V: lanes over 32 consecutive keys of one chunk, so the transposed LDS
  // writes of a wave land on consecutive halves (no bank conflicts).
  const auto fetch = [&](std::uint32_t j0) {
#pragma unroll
    for (std::uint32_t f = 0; f < kFetch; ++f) {
      const std::uint32_t i = tid + f * 256;
      const std::uint32_t jk = j0 + i / (kDim / 8);
      const std::uint32_t ck = i % (kDim / 8);
      const std::uint32_t jv = j0 + i % kKeys;
      const std::uint32_t cv = i / kKeys;
      nk[f] = make_uint4(0u, 0u, 0u, 0u);
      nv[f] = make_uint4(0u, 0u, 0u, 0u);
      if (jk < kend)
        nk[f] = *reinterpret_cast<const uint4*>(
            k_cache + static_cast<std::size_t>(jk) * kvs + g * kDim + ck * 8);
      if (jv < kend)
        nv[f] = *reinterpret_cast<const uint4*>(
            v_cache + static_cast<std::size_t>(jv) * kvs + g * kDim + cv * 8);
    }
  };
  fetch(0);
  for (std::uint32_t j0 = 0; j0 < kend; j0 += kKeys) {
    __syncthreads();  // the previous tile is consumed
#pragma unroll
    for (std::uint32_t f = 0; f < kFetch; ++f) {
      const std::uint32_t i = tid + f * 256;
      *reinterpret_cast<uint4*>(&k_s[i / (kDim / 8)][(i % (kDim / 8)) * 8]) =
          nk[f];
      const std::uint32_t key = i % kKeys;
      const std::uint32_t c = i / kKeys;
      const auto* hv = reinterpret_cast<const _Float16*>(&nv[f]);
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e) vt_s[c * 8 + e][key] = hv[e];
    }
    if (j0 + kKeys < kend) fetch(j0 + kKeys);
    __syncthreads();
    Acc s[2] = {Acc{}, Acc{}};
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
      const std::uint32_t kk = part * (kDim / 2) + ks * 16;
#pragma unroll
      for (std::uint32_t sb = 0; sb < 2; ++sb) {
        const HFrag b =
            *reinterpret_cast<const HFrag*>(&k_s[sb * 16 + r][kk]);
        s[sb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], b, s[sb]);
        if constexpr (kSplitQ)
          s[sb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa_lo[ks], b,
                                                             s[sb]);
      }
    }
#pragma unroll
    for (std::uint32_t sb = 0; sb < 2; ++sb)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_part[hh][part][sb * 8 + i][lane] = s[sb][i];
    __syncthreads();
#pragma unroll
    for (std::uint32_t sb = 0; sb < 2; ++sb)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s[sb][i] += s_part[hh][part ^ 1u][sb * 8 + i][lane];
    // s[sb][i]: query row 2i + half, key j0 + 16 sb + r (same in both waves).
    float rescale[8];
    float p[2][8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t row = 2 * i + half;
      const std::uint32_t limit = start + t0 + row;  // last visible key
      float v[2];
#pragma unroll
      for (std::uint32_t sb = 0; sb < 2; ++sb) {
        const std::uint32_t j = j0 + sb * 16 + r;
        v[sb] = j < kend && j <= limit ? s[sb][i] : -INFINITY;
      }
      const float mn = fmaxf(m[i], fmaxf(RowMax16(v[0]), RowMax16(v[1])));
      // Key 0 is visible to every row, so mn is finite from the first tile.
      rescale[i] = __expf(m[i] - mn);
      float sum = 0.0f;
#pragma unroll
      for (std::uint32_t sb = 0; sb < 2; ++sb) {
        p[sb][i] = v[sb] == -INFINITY ? 0.0f : __expf(v[sb] - mn);
        sum += RowSum16(p[sb][i]);
      }
      l[i] = l[i] * rescale[i] + sum;
      m[i] = mn;
    }
    if (part == 0) {
      // The odd wave read slot [0]; slot [1] (read above by this wave) now
      // takes P.
#pragma unroll
      for (std::uint32_t sb = 0; sb < 2; ++sb)
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const float ps = p[sb][i] * kPScale;
          const _Float16 hi = static_cast<_Float16>(ps);
          p_s[2 * i + half][sb * 16 + r] = hi;
          if constexpr (kSplitP)
            p_lo[2 * i + half][sb * 16 + r] =
                static_cast<_Float16>(ps - static_cast<float>(hi));
        }
    }
#pragma unroll
    for (std::uint32_t b = 0; b < kTiles; ++b)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) o[b][i] *= rescale[i];
    __syncthreads();  // P complete
#pragma unroll
    for (std::uint32_t kb = 0; kb < 2; ++kb) {
      const HFrag pa = *reinterpret_cast<const HFrag*>(&p_s[r][kb * 16]);
      HFrag pl{};
      if constexpr (kSplitP)
        pl = *reinterpret_cast<const HFrag*>(&p_lo[r][kb * 16]);
#pragma unroll
      for (std::uint32_t b = 0; b < kTiles; ++b) {
        const HFrag vb = *reinterpret_cast<const HFrag*>(
            &vt_s[(part * kTiles + b) * 16 + r][kb * 16]);
        o[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o[b]);
        if constexpr (kSplitP)
          o[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pl, vb, o[b]);
      }
    }
  }
  const std::uint32_t h = hb + hh;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t t = t0 + 2 * i + half;
    if (t >= n) continue;
    const float inv = 1.0f / (l[i] * kPScale);
    const std::size_t base = (static_cast<std::size_t>(t) * heads + h) * kDim;
#pragma unroll
    for (std::uint32_t b = 0; b < kTiles; ++b) {
      const std::uint32_t d = (part * kTiles + b) * 16 + r;
      const float gt = gate[base + d];
      out[base + d] = o[b][i] * inv / (1.0f + __expf(-gt));
    }
  }
}

// --- Decode attention on the WMMA cores. ---
constexpr std::uint32_t kQs = kDim + 8;  // Q LDS row stride (halves)
constexpr std::uint32_t kPs = 16 + 8;    // P LDS row stride (halves)
constexpr std::uint32_t kCSlice = 256;   // keys per block

/// grid (kv_heads, slices of kCSlice keys), block 32 x kW (kW = 4 for up to
/// 2 M tiles, 8 above: halves each wave's O). The batch's queries of KV
/// group g: query 8t + h (row t, head h of the group), M tile mt = rows 2mt,
/// 2mt + 1. V comes from the tiled V^T cache (StoreVt), whose 16-key x 16-dim
/// fragments are contiguous 32-byte runs.
/// Per 16-key tile, one barrier:
///  - S phase: wave w < kMT scores M tile w (A = Q from LDS, B = K rows
///    straight from memory; the waves' identical loads meet in the CU
///    cache), runs its online softmax (m, l stay in that wave) and publishes
///    P (F16, A layout) and the per-row rescale in LDS (double-buffered);
///  - PV phase: wave w owns dims (256 / kW) w.. of every M tile: its V^T
///    fragments are loaded once (issued before the S phase) and used for all
///    kMT tiles.
/// Batch-width invariance: slices and tiles are fixed from position 0 and a
/// WMMA output element depends only on its own row, so a query's arithmetic
/// is the same at every width -- except that the WMMA unit is not zero-safe:
/// B values still move an output whose A entry is 0 (measured by the bench's
/// probe). A tile holding keys past pos0 has keys masked for some rows yet
/// real in this batch (zero in a 1-row call), so there each half of an M
/// tile (one row each) gets V zeroed past its own limit, exactly the V a
/// 1-row call sees; keys past the batch read as 0. Fully masked tiles
/// rescale by exactly 1 and add 0 x 0.
/// kPrecise bit 1: P scaled by 2^15 and split hi + lo (two P V products).
template <unsigned kPrecise, std::uint32_t kMT, std::uint32_t kW>
__global__ __launch_bounds__(32 * kW) void DecodeAttentionCoopKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ vt_cache, float* __restrict__ partials,
    const std::uint32_t* __restrict__ start_pos, std::uint32_t rows,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t splits,
    std::uint32_t max_context) {
  constexpr bool kSplitP = (kPrecise & 1u) != 0;
  constexpr float kPScale = kSplitP ? 32768.0f : 1.0f;
  constexpr std::uint32_t kQT = 16 / kW;  // O tiles per wave
  static_assert(kMT <= kW);
  __shared__ __attribute__((aligned(16))) _Float16 q_s[kMT][16][kQs];
  __shared__ __attribute__((aligned(16)))
  _Float16 p_s[2][kMT][kSplitP ? 2 : 1][16][kPs];
  __shared__ float resc_s[2][kMT][16];

  const std::uint32_t g = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  const std::uint32_t pos0 = *start_pos;
  const std::uint32_t kend = pos0 + rows;
  const std::uint32_t j_begin = s * kCSlice;
  if (j_begin >= kend) return;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t r = lane & 15u;
  const std::uint32_t half = lane >> 4;
  const float scale = rsqrtf(static_cast<float>(kDim));

  for (std::uint32_t idx = tid; idx < kMT * 16 * (kDim / 4); idx += 32 * kW) {
    const std::uint32_t tile = idx / (16 * (kDim / 4));
    const std::uint32_t rr = idx / (kDim / 4) % 16;
    const std::uint32_t c = (idx % (kDim / 4)) * 4;
    const std::uint32_t t = min(2 * tile + rr / 8, rows - 1);
    const float4 v = *reinterpret_cast<const float4*>(
        q + (static_cast<std::size_t>(t) * heads + g * kGroup + rr % 8) * kDim +
        c);
    q_s[tile][rr][c + 0] = static_cast<_Float16>(v.x * scale);
    q_s[tile][rr][c + 1] = static_cast<_Float16>(v.y * scale);
    q_s[tile][rr][c + 2] = static_cast<_Float16>(v.z * scale);
    q_s[tile][rr][c + 3] = static_cast<_Float16>(v.w * scale);
  }
  __syncthreads();

  const std::uint32_t kstop = min(j_begin + kCSlice, kend);
  const std::size_t kvs = static_cast<std::size_t>(kv_heads) * kDim;
  const __half* kbase = k_cache + g * kDim;
  const __half* vtb = vt_cache +
                      (static_cast<std::size_t>(g) * (max_context / 16) * kDim +
                       wave * (kDim / kW)) * 16;
  const bool s_wave = wave < kMT;  // uniform per wave
  const std::uint32_t my_lim_a = pos0 + 2 * wave;

  float m[8], l[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f;
  }
  Acc o[kMT][kQT];
#pragma unroll
  for (std::uint32_t t = 0; t < kMT; ++t)
#pragma unroll
    for (std::uint32_t b = 0; b < kQT; ++b) o[t][b] = Acc{};

  std::uint32_t buf = 0;
  for (std::uint32_t j0 = j_begin; j0 < kstop; j0 += 16, buf ^= 1u) {
    HFrag vb[kQT];
#pragma unroll
    for (std::uint32_t b = 0; b < kQT; ++b)
      vb[b] = *reinterpret_cast<const HFrag*>(
          vtb + (static_cast<std::size_t>(j0 / 16) * kDim + b * 16 + r) * 16);
    if (s_wave) {
      Acc sc{};
      const std::uint32_t jr = j0 + r;
      const bool valid = jr < kend;
      const auto* krow = reinterpret_cast<const HFrag*>(
          kbase + static_cast<std::size_t>(valid ? jr : 0) * kvs);
#pragma unroll
      for (std::uint32_t ks = 0; ks < kDim / 16; ++ks) {
        const HFrag b = valid ? krow[ks] : HFrag{};
        const HFrag a =
            *reinterpret_cast<const HFrag*>(&q_s[wave][r][ks * 16]);
        sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t limit = my_lim_a + (i < 4 ? 0u : 1u);
        const float v = jr <= limit ? sc[i] : -INFINITY;
        const float mn = fmaxf(m[i], RowMax16(v));
        const float rs = mn == -INFINITY ? 1.0f : __expf(m[i] - mn);
        const float p = v == -INFINITY ? 0.0f : __expf(v - mn);
        l[i] = l[i] * rs + RowSum16(p);
        m[i] = mn;
        if (r == 0) resc_s[buf][wave][2 * i + half] = rs;
        const float ps = p * kPScale;
        const _Float16 hi = static_cast<_Float16>(ps);
        p_s[buf][wave][0][2 * i + half][r] = hi;
        if constexpr (kSplitP)
          p_s[buf][wave][kSplitP ? 1 : 0][2 * i + half][r] =
              static_cast<_Float16>(ps - static_cast<float>(hi));
      }
    }
    __syncthreads();
    const bool boundary = j0 + 16 > pos0 + 1;
#pragma unroll
    for (std::uint32_t t = 0; t < kMT; ++t) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const float rs = resc_s[buf][t][2 * i + half];
#pragma unroll
        for (std::uint32_t b = 0; b < kQT; ++b) o[t][b][i] *= rs;
      }
      const HFrag pa =
          *reinterpret_cast<const HFrag*>(&p_s[buf][t][0][r][0]);
      HFrag pl{};
      if constexpr (kSplitP)
        pl = *reinterpret_cast<const HFrag*>(
            &p_s[buf][t][kSplitP ? 1 : 0][r][0]);
      const std::uint32_t lim_a = pos0 + 2 * t;
      const std::uint32_t lim_b = lim_a + 1;
#pragma unroll
      for (std::uint32_t b = 0; b < kQT; ++b) {
        if (!boundary) {
          o[t][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb[b], o[t][b]);
          if constexpr (kSplitP)
            o[t][b] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pl, vb[b], o[t][b]);
        } else {
          HFrag va = vb[b], vc = vb[b];
#pragma unroll
          for (std::uint32_t k = 0; k < 16; ++k) {
            const std::uint32_t key = j0 + k;
            if (key > lim_a || key >= kend) va[k] = 0;
            if (key > lim_b || key >= kend) vc[k] = 0;
          }
          Acc ta = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, va, o[t][b]);
          Acc tb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vc, o[t][b]);
          if constexpr (kSplitP) {
            ta = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pl, va, ta);
            tb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pl, vc, tb);
          }
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            o[t][b][i] = i < 4 ? ta[i] : tb[i];
        }
      }
    }
  }

  const std::uint32_t queries = kGroup * rows;
#pragma unroll
  for (std::uint32_t t = 0; t < kMT; ++t) {
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t row = 2 * i + half;
      const std::uint32_t tr = 2 * t + row / 8;
      if (tr >= rows) continue;
      float* dst = partials + ((static_cast<std::size_t>(g) * splits + s) *
                                   queries + tr * kGroup + row % 8) *
                                  kPart;
      if (wave == t && r == 0) {
        dst[0] = m[i];
        dst[1] = l[i];
      }
#pragma unroll
      for (std::uint32_t b = 0; b < kQT; ++b)
        dst[2 + wave * (kDim / kW) + b * 16 + r] =
            o[t][b][i] * (1.0f / kPScale);
    }
  }
}

/// grid (heads, rows), block 256: merges the active `chunk`-key slices in
/// order (DecodeAttentionMergeKernel with a slice size).
__global__ void DecodeAttentionMergeChunkKernel(
    const float* partials, float* out, const std::uint32_t* start_pos,
    std::uint32_t rows, std::uint32_t heads, std::uint32_t splits,
    std::uint32_t chunk) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t g = h / kGroup;
  const std::uint32_t qi = t * kGroup + h % kGroup;
  const std::uint32_t queries = kGroup * rows;
  const std::uint32_t active = (*start_pos + rows + chunk - 1) / chunk;
  const auto at = [&](std::uint32_t s) {
    return partials +
           ((static_cast<std::size_t>(g) * splits + s) * queries + qi) * kPart;
  };
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < active; ++s) m = fmaxf(m, at(s)[0]);
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < active; ++s) {
    const float* part = at(s);
    const float scale = part[0] == -INFINITY ? 0.0f : __expf(part[0] - m);
    l += part[1] * scale;
    acc += part[2 + i] * scale;
  }
  out[(static_cast<std::size_t>(t) * heads + h) * kDim + i] = acc / l;
}

}  // namespace

bool PrefillAttention(const float* q, const float* gate, const __half* k_cache,
                      const __half* v_cache, float* out,
                      std::uint32_t start_pos, std::uint32_t n_tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, hipStream_t stream) {
  if (head_dim != kDim || heads != kv_heads * kGroup || n_tokens == 0)
    return false;
  // A3B_ATTN_PRECISE (diagnostic): bit 1 splits P, bit 2 splits Q.
  static const unsigned precise = [] {
    const char* v = std::getenv("A3B_ATTN_PRECISE");
    return v != nullptr ? static_cast<unsigned>(std::atoi(v)) & 3u : 0u;
  }();
  const dim3 grid((n_tokens + kPQ - 1) / kPQ, kv_heads * (kGroup / kPH));
  switch (precise) {
    case 1:
      hipLaunchKernelGGL(PrefillAttentionKernel<1>, grid, dim3(256), 0, stream,
                         q, gate, k_cache, v_cache, out, start_pos, n_tokens,
                         heads, kv_heads);
      break;
    case 2:
      hipLaunchKernelGGL(PrefillAttentionKernel<2>, grid, dim3(256), 0, stream,
                         q, gate, k_cache, v_cache, out, start_pos, n_tokens,
                         heads, kv_heads);
      break;
    case 3:
      hipLaunchKernelGGL(PrefillAttentionKernel<3>, grid, dim3(256), 0, stream,
                         q, gate, k_cache, v_cache, out, start_pos, n_tokens,
                         heads, kv_heads);
      break;
    default:
      hipLaunchKernelGGL(PrefillAttentionKernel<0>, grid, dim3(256), 0, stream,
                         q, gate, k_cache, v_cache, out, start_pos, n_tokens,
                         heads, kv_heads);
  }
  return true;
}

std::uint32_t AttentionSlices(std::uint32_t max_context) {
  return (max_context + kAttnChunk - 1) / kAttnChunk;
}

std::size_t AttentionPartialFloats(std::uint32_t kv_heads,
                                   std::uint32_t max_context) {
  return static_cast<std::size_t>(kv_heads) * AttentionSlices(max_context) *
         kGroup * 8 * kPart;
}

bool DecodeAttention(const float* q, const __half* k_cache,
                     const __half* v_cache, const __half* vt_cache, float* out,
                     float* partials, const std::uint32_t* start_pos,
                     std::uint32_t rows, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim,
                     std::uint32_t max_context, hipStream_t stream) {
  if (head_dim != kDim || heads != kv_heads * kGroup || rows == 0 ||
      rows > 8 || max_context == 0)
    return false;
  // A3B_DECODE_ATTN overrides the kernel (see SetDecodeAttentionMode).
  static const int env_mode = [] {
    const char* v = std::getenv("A3B_DECODE_ATTN");
    return v != nullptr ? std::atoi(v) : 1;
  }();
  const int mode = g_decode_mode >= 0 ? g_decode_mode : env_mode;
  if (mode != 0 && vt_cache != nullptr) {
    const std::uint32_t csplits = (max_context + kCSlice - 1) / kCSlice;
    const dim3 cgrid(kv_heads, csplits);
    const bool split_p = mode == 2;
#define A3B_COOP(P, MT)                                                     \
  hipLaunchKernelGGL((DecodeAttentionCoopKernel<P, MT, (MT > 2 ? 8 : 4)>),  \
                     cgrid, dim3(MT > 2 ? 256 : 128), 0, stream, q, k_cache, \
                     vt_cache, partials, start_pos, rows, heads, kv_heads,   \
                     csplits, max_context)
    switch ((rows + 1) / 2) {
      case 1: if (split_p) { A3B_COOP(1, 1); } else { A3B_COOP(0, 1); } break;
      case 2: if (split_p) { A3B_COOP(1, 2); } else { A3B_COOP(0, 2); } break;
      case 3: if (split_p) { A3B_COOP(1, 3); } else { A3B_COOP(0, 3); } break;
      default: if (split_p) { A3B_COOP(1, 4); } else { A3B_COOP(0, 4); }
    }
#undef A3B_COOP
    hipLaunchKernelGGL(DecodeAttentionMergeChunkKernel, dim3(heads, rows),
                       dim3(kDim), 0, stream, partials, out, start_pos, rows,
                       heads, csplits, kCSlice);
    return true;
  }
  const std::uint32_t splits = AttentionSlices(max_context);
  hipLaunchKernelGGL(DecodeAttentionKernel, dim3(rows, kv_heads, splits),
                     dim3(kThreads), 0, stream, q, k_cache, v_cache, partials,
                     start_pos, heads, kv_heads, splits);
  hipLaunchKernelGGL(DecodeAttentionMergeKernel, dim3(heads, rows),
                     dim3(kDim), 0, stream, partials, out, start_pos, rows,
                     heads, splits);
  return true;
}

void SetDecodeAttentionMode(int mode) { g_decode_mode = mode; }

namespace {
/// grid (n, kv_heads), block 256 (dims).
__global__ void StoreVtKernel(const __half* v, __half* vt,
                              const std::uint32_t* pos, std::uint32_t kv_heads,
                              std::uint32_t max_context) {
  const std::uint32_t p = *pos + blockIdx.x;
  const std::uint32_t g = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  vt[((static_cast<std::size_t>(g) * (max_context / 16) + p / 16) * kDim + d) *
         16 +
     p % 16] = v[(static_cast<std::size_t>(p) * kv_heads + g) * kDim + d];
}
}  // namespace

void StoreVt(const __half* v_cache, __half* vt_cache,
             const std::uint32_t* pos, std::uint32_t n, std::uint32_t kv_heads,
             std::uint32_t max_context, hipStream_t stream) {
  if (vt_cache == nullptr || n == 0) return;
  hipLaunchKernelGGL(StoreVtKernel, dim3(n, kv_heads), dim3(kDim), 0, stream,
                     v_cache, vt_cache, pos, kv_heads, max_context);
}

namespace {
/// Probe: one 16x16x16 WMMA of a (16 x 16 F16) by b (16 x 16, column-major
/// per lane) into d (row-major floats).
__global__ void WmmaProbeKernel(const _Float16* a, const _Float16* b,
                                float* d) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t r = lane & 15u;
  HFrag fa, fb;
  for (std::uint32_t k = 0; k < 16; ++k) {
    fa[k] = a[r * 16 + k];
    fb[k] = b[r * 16 + k];  // column r
  }
  Acc c{};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fa, fb, c);
  for (std::uint32_t i = 0; i < 8; ++i) d[(2 * i + (lane >> 4)) * 16 + r] = c[i];
}
}  // namespace

namespace {
/// Bandwidth floor: sum n uint4s (grid-stride).
__global__ void StreamReadKernel(const uint4* a, std::size_t n, unsigned* out) {
  unsigned acc = 0;
  for (std::size_t i = blockIdx.x * 256 + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * 256) {
    const uint4 v = a[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *out = acc;
}
}  // namespace

static void WmmaRowProbe() {
  std::vector<_Float16> a(256), b(256);
  std::uint64_t st = 12345;
  const auto rnd = [&] {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>(st >> 40) / static_cast<float>(1u << 24);
  };
  _Float16 *da, *db;
  float* dd;
  (void)hipMalloc(&da, 512);
  (void)hipMalloc(&db, 512);
  (void)hipMalloc(&dd, 1024);
  int diffs_lo = 0, diffs_hi = 0;
  for (int trial = 0; trial < 200; ++trial) {
    // P-like rows (0..1, some tiny, some zero) and V-like columns.
    for (int i = 0; i < 256; ++i) {
      const float u = rnd();
      a[i] = static_cast<_Float16>(u < 0.3f ? 0.0f : std::exp(-20.0f * rnd()));
      b[i] = static_cast<_Float16>(4.0f * rnd() - 2.0f);
    }
    std::vector<float> d0(256), d1(256);
    (void)hipMemcpy(da, a.data(), 512, hipMemcpyHostToDevice);
    (void)hipMemcpy(db, b.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d0.data(), dd, 1024, hipMemcpyDeviceToHost);
    // Change rows 8..15 only, then rows 0..7 only.
    auto a2 = a;
    for (int i = 128; i < 256; ++i) a2[i] = static_cast<_Float16>(rnd() * 3);
    (void)hipMemcpy(da, a2.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d1.data(), dd, 1024, hipMemcpyDeviceToHost);
    if (std::memcmp(d0.data(), d1.data(), 512) != 0) ++diffs_lo;
    auto a3 = a;
    for (int i = 0; i < 128; ++i) a3[i] = static_cast<_Float16>(rnd() * 3);
    (void)hipMemcpy(da, a3.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d1.data(), dd, 1024, hipMemcpyDeviceToHost);
    if (std::memcmp(d0.data() + 128, d1.data() + 128, 512) != 0) ++diffs_hi;
  }
  std::printf("wmma probe: rows 0-7 changed by rows 8-15 in %d/200, rows 8-15 "
              "changed by rows 0-7 in %d/200\n",
              diffs_lo, diffs_hi);
  // Zero-safety: A[0..7][k] = 0; does B[k][*] still move rows 0..7?
  int zero_leaks = 0, zero_leaks_all = 0;
  for (int trial = 0; trial < 200; ++trial) {
    for (int i = 0; i < 256; ++i) {
      a[i] = static_cast<_Float16>(std::exp(-12.0f * rnd()));
      b[i] = static_cast<_Float16>(4.0f * rnd() - 2.0f);
    }
    const int k = trial % 16;
    for (int row = 0; row < 8; ++row) a[row * 16 + k] = 0;
    std::vector<float> d0(256), d1(256);
    (void)hipMemcpy(da, a.data(), 512, hipMemcpyHostToDevice);
    (void)hipMemcpy(db, b.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d0.data(), dd, 1024, hipMemcpyDeviceToHost);
    auto b2 = b;
    for (int col = 0; col < 16; ++col) b2[col * 16 + k] = 0;  // B[k][col]
    (void)hipMemcpy(db, b2.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d1.data(), dd, 1024, hipMemcpyDeviceToHost);
    if (std::memcmp(d0.data(), d1.data(), 512) != 0) ++zero_leaks;
    // Same with the whole column k of A zero.
    for (int row = 8; row < 16; ++row) a[row * 16 + k] = 0;
    (void)hipMemcpy(da, a.data(), 512, hipMemcpyHostToDevice);
    (void)hipMemcpy(db, b.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d0.data(), dd, 1024, hipMemcpyDeviceToHost);
    (void)hipMemcpy(db, b2.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(WmmaProbeKernel, dim3(1), dim3(32), 0, 0, da, db, dd);
    (void)hipMemcpy(d1.data(), dd, 1024, hipMemcpyDeviceToHost);
    if (std::memcmp(d0.data(), d1.data(), 512) != 0) ++zero_leaks_all;
  }
  std::printf("wmma probe: B[k] moves rows whose A[.][k] = 0 in %d/200 (only "
              "rows 0-7 zero), %d/200 (whole column zero)\n",
              zero_leaks, zero_leaks_all);
  (void)hipFree(da);
  (void)hipFree(db);
  (void)hipFree(dd);
}

int RunDecodeAttentionBench() {
  WmmaRowProbe();
  constexpr std::uint32_t kKv = 2, kHeads = 16, kRowsMax = 8;
  constexpr std::uint32_t kCtx = 176 * 1024;
  const std::size_t kv_elems = static_cast<std::size_t>(kCtx) * kKv * kDim;
  const std::size_t q_elems = static_cast<std::size_t>(kRowsMax) * kHeads * kDim;
  // Random K/V in [-2, 2] and Q in [-4, 4]: peaked enough softmax rows that
  // the F16 probabilities matter, like real attention.
  std::uint64_t st = 0x9E3779B97F4A7C15ull;
  const auto rnd = [&] {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return static_cast<float>(st >> 40) / static_cast<float>(1u << 24) * 2.0f -
           1.0f;
  };
  std::vector<__half> hk(kv_elems), hv(kv_elems);
  for (std::size_t i = 0; i < kv_elems; ++i) {
    hk[i] = __float2half(2.0f * rnd());
    hv[i] = __float2half(2.0f * rnd());
  }
  std::vector<float> hq(q_elems);
  for (auto& x : hq) x = 4.0f * rnd();
  __half *k = nullptr, *v = nullptr, *vt = nullptr;
  float *qd = nullptr, *out = nullptr, *out1 = nullptr, *parts = nullptr;
  std::uint32_t* pos = nullptr;
  const std::size_t part_floats = AttentionPartialFloats(kKv, kCtx);
  if (hipMalloc(&k, kv_elems * 2) != hipSuccess ||
      hipMalloc(&v, kv_elems * 2) != hipSuccess ||
      hipMalloc(&vt, kv_elems * 2) != hipSuccess ||
      hipMalloc(&qd, q_elems * 4) != hipSuccess ||
      hipMalloc(&out, q_elems * 4) != hipSuccess ||
      hipMalloc(&out1, q_elems * 4) != hipSuccess ||
      hipMalloc(&parts, part_floats * 4) != hipSuccess ||
      hipMalloc(&pos, 4) != hipSuccess) {
    std::fprintf(stderr, "attn bench: out of memory\n");
    return 1;
  }
  (void)hipMemcpy(k, hk.data(), kv_elems * 2, hipMemcpyHostToDevice);
  (void)hipMemcpy(v, hv.data(), kv_elems * 2, hipMemcpyHostToDevice);
  {
    // V^T tiles [kv_head][kCtx / 16][dim][16].
    std::vector<__half> hvt(kv_elems);
    for (std::size_t j = 0; j < kCtx; ++j)
      for (std::uint32_t g = 0; g < kKv; ++g)
        for (std::uint32_t d = 0; d < kDim; ++d)
          hvt[((static_cast<std::size_t>(g) * (kCtx / 16) + j / 16) * kDim + d) *
                  16 +
              j % 16] = hv[(j * kKv + g) * kDim + d];
    (void)hipMemcpy(vt, hvt.data(), kv_elems * 2, hipMemcpyHostToDevice);
  }
  (void)hipMemcpy(qd, hq.data(), q_elems * 4, hipMemcpyHostToDevice);
  hipStream_t stream;
  (void)hipStreamCreate(&stream);
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0);
  (void)hipEventCreate(&e1);
  const auto run = [&](int mode, std::uint32_t p0, std::uint32_t rows,
                       const float* qp, float* o) {
    SetDecodeAttentionMode(mode);
    (void)hipMemcpyAsync(pos, &p0, 4, hipMemcpyHostToDevice, stream);
    (void)DecodeAttention(qp, k, v, vt, o, parts, pos, rows, kHeads, kKv, kDim,
                          kCtx, stream);
  };
  const auto fetch = [&](const float* d, std::size_t n) {
    std::vector<float> h(n);
    (void)hipStreamSynchronize(stream);
    (void)hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost);
    return h;
  };
  {
    // Isolation at depth 4096, WMMA: row 0 of a 2-row batch vs a 1-row call.
    for (const int imode : {0, 1}) {
    const std::uint32_t p0 = 4094;
    const std::size_t hn = static_cast<std::size_t>(kHeads) * kDim;
    float* q2 = nullptr;
    (void)hipMalloc(&q2, 2 * hn * 4);
    (void)hipMemcpy(q2, qd, hn * 4, hipMemcpyDeviceToDevice);
    (void)hipMemcpy(q2 + hn, qd, hn * 4, hipMemcpyDeviceToDevice);
    run(imode, p0, 1, qd, out1);
    const auto single = fetch(out1, hn);
    run(imode, p0, 2, qd, out);
    const auto mixed = fetch(out, hn);
    run(imode, p0, 2, q2, out);
    const auto dup = fetch(out, hn);
    std::printf("isolate mode %d: [q0,q1] vs single %s, [q0,q0] vs single %s, "
                "[q0,q1] vs [q0,q0] %s\n", imode,
                std::memcmp(mixed.data(), single.data(), hn * 4) ? "DIFF" : "same",
                std::memcmp(dup.data(), single.data(), hn * 4) ? "DIFF" : "same",
                std::memcmp(mixed.data(), dup.data(), hn * 4) ? "DIFF" : "same");
    (void)hipFree(q2);
    }
    if (std::getenv("A3B_PROBE_ONLY") != nullptr) return 0;
  }
  for (const std::uint32_t depth : {16384u, 65536u, 172032u}) {
    // K + V of `depth` keys for both KV heads, as one layer reads them.
    const std::size_t n = static_cast<std::size_t>(depth) * kKv * kDim * 2 / 8;
    float best = 1e9f;
    for (const std::uint32_t blocks : {640u, 1280u, 2560u, 5120u}) {
      for (int w = 0; w < 2; ++w)
        hipLaunchKernelGGL(StreamReadKernel, dim3(blocks), dim3(256), 0, stream,
                           reinterpret_cast<const uint4*>(k), n,
                           reinterpret_cast<unsigned*>(pos));
      (void)hipEventRecord(e0, stream);
      for (int i = 0; i < 10; ++i) {
        hipLaunchKernelGGL(StreamReadKernel, dim3(blocks), dim3(256), 0, stream,
                           reinterpret_cast<const uint4*>(k), n / 2,
                           reinterpret_cast<unsigned*>(pos));
        hipLaunchKernelGGL(StreamReadKernel, dim3(blocks), dim3(256), 0, stream,
                           reinterpret_cast<const uint4*>(v), n / 2,
                           reinterpret_cast<unsigned*>(pos));
      }
      (void)hipEventRecord(e1, stream);
      (void)hipEventSynchronize(e1);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, e0, e1);
      best = std::min(best, ms * 100.0f);
    }
    std::printf("stream floor depth %u: %.1f us (%.0f GB/s)\n", depth, best,
                n * 16.0 / (best * 1e3));
  }
  const int modes[] = {0, 1, 2};
  const char* names[] = {"alu", "wmma", "wmma+P"};
  std::printf("decode attention, one layer (x10 for a step), us per call\n");
  std::printf("%7s %4s", "depth", "rows");
  for (const char* n : names) std::printf(" %9s", n);
  std::printf("   max rel diff vs alu   invariant\n");
  for (const std::uint32_t depth :
       {4096u, 16384u, 32768u, 65536u, 131072u, 172032u}) {
    for (const std::uint32_t rows : {1u, 4u, 8u}) {
      const std::uint32_t p0 = depth - rows;
      const std::size_t n = static_cast<std::size_t>(rows) * kHeads * kDim;
      std::printf("%7u %4u", depth, rows);
      std::vector<std::vector<float>> res;
      for (const int mode : modes) {
        for (int w = 0; w < 3; ++w) run(mode, p0, rows, qd, out);
        const int reps = 20;
        (void)hipEventRecord(e0, stream);
        for (int i = 0; i < reps; ++i) run(mode, p0, rows, qd, out);
        (void)hipEventRecord(e1, stream);
        (void)hipEventSynchronize(e1);
        float ms = 0;
        (void)hipEventElapsedTime(&ms, e0, e1);
        std::printf(" %9.1f", ms * 1000.0f / reps);
        res.push_back(fetch(out, n));
      }
      std::printf("   ");
      for (std::size_t m = 1; m < res.size(); ++m) {
        double worst = 0, big = 0;
        for (std::size_t i = 0; i < n; ++i) {
          big = std::max(big, static_cast<double>(std::fabs(res[0][i])));
          worst = std::max(worst, static_cast<double>(
                                      std::fabs(res[m][i] - res[0][i])));
        }
        std::printf(" %.2e", worst / big);
      }
      // Row t of this batch vs a 1-row call at position p0 + t, bitwise.
      std::printf("  ");
      for (std::size_t m = 0; m < res.size(); ++m) {
        bool same = true;
        double worst = 0;
        std::uint32_t bad_rows = 0;
        for (std::uint32_t t = 0; t < rows; ++t) {
          const std::size_t off = static_cast<std::size_t>(t) * kHeads * kDim;
          run(modes[m], p0 + t, 1, qd + off, out1);
          const auto one = fetch(out1, kHeads * kDim);
          const bool eq = std::memcmp(one.data(), res[m].data() + off,
                                      kHeads * kDim * 4) == 0;
          if (!eq) bad_rows |= 1u << t;
          same = same && eq;
          for (std::size_t i = 0; i < kHeads * kDim; ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(
                                        one[i] - res[m][off + i])));
        }
        if (same)
          std::printf(" yes");
        else
          std::printf(" NO(rows %x, %.1e)", bad_rows, worst);
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  }
  SetDecodeAttentionMode(-1);
  (void)hipFree(k);
  (void)hipFree(v);
  (void)hipFree(vt);
  (void)hipFree(qd);
  (void)hipFree(out);
  (void)hipFree(out1);
  (void)hipFree(parts);
  (void)hipFree(pos);
  (void)hipStreamDestroy(stream);
  return 0;
}

}  // namespace gufo::models::qwen36_a3b
