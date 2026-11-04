// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2026 Victor Shaw
// SPDX-License-Identifier: MIT AND Apache-2.0
// Packed weights are decoded once per channel and reused across input rows.
#include <hip/hip_fp16.h>

#include <climits>
#include <cmath>

#include "head_two_kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

struct Q81 {
  __half d, sum;
  signed char q[32];
};
struct Q4K {
  __half d, min;
  unsigned char scales[12], q[128];
};
static_assert(sizeof(Q81) == 36 && sizeof(Q4K) == 144);

__device__ float WaveSum(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor(x, offset, 32);
  return x;
}

__device__ int Dot4(int a, int b, int sum = 0) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, a, true, b, sum, false);
#else
  return sum;
#endif
}

template<int N>
__global__ void PrescoreQ4(const Q4K* weights, const Q81* x, float* scores,
                           int rows, int cols, int stride, int rotation) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int source = (row + rotation) % rows;
  const int group = lane / 4, channel = (lane & 3) * 8;
  float accum[N] = {};
  for (int sb = 0; sb < cols / 256; ++sb) {
    const auto& w = weights[source * (cols / 256) + sb];
    const int scale = group < 4 ? w.scales[group] & 63
                                : (w.scales[group + 4] & 15) |
                                      ((w.scales[group - 4] >> 6) << 4);
    const int minimum =
        group < 4 ? w.scales[group + 4] & 63
                  : (w.scales[group + 4] >> 4) | ((w.scales[group] >> 6) << 4);
    const float d = __half2float(w.d) * float(scale);
    const float m = __half2float(w.min) * float(minimum);
    const auto* packed =
        reinterpret_cast<const int*>(w.q + (group / 2) * 32 + channel);
    const int v0 = (packed[0] >> ((group & 1) * 4)) & 0x0f0f0f0f;
    const int v1 = (packed[1] >> ((group & 1) * 4)) & 0x0f0f0f0f;
#pragma unroll
    for (int r = 0; r < N; ++r) {
      const auto& a = x[r * stride + sb * 8 + group];
      const auto* q = reinterpret_cast<const int*>(a.q + channel);
      const int dot = Dot4(v1, q[1], Dot4(v0, q[0]));
      const int sumq = Dot4(0x01010101, q[1], Dot4(0x01010101, q[0]));
      const float local = __builtin_fmaf(d, float(dot), -(m * float(sumq)));
      accum[r] = __builtin_fmaf(local, __half2float(a.d), accum[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < N; ++r) {
    const float value = WaveSum(accum[r]);
    if (lane == 0)
      scores[r * rows + row] = value;
  }
}

__device__ bool Better(HeadTwoPair a, HeadTwoPair b) {
  return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ HeadTwoPair Empty() {
  return {-INFINITY, INT_MAX};
}

// Each tile exports its sorted top K. Pairwise merges discard only entries
// whose rank in their combined tile is at least K.
__global__ void TopTiles(const float* values, HeadTwoPair* output, int rows,
                         int k) {
  __shared__ HeadTwoPair tile[kHeadTile];
  const int r = blockIdx.y, t = threadIdx.x;
  for (int i = t; i < kHeadTile; i += blockDim.x) {
    const int id = blockIdx.x * kHeadTile + i;
    const float v = id < rows ? values[r * rows + id] : -INFINITY;
    tile[i] = id < rows && isfinite(v) ? HeadTwoPair{v, id} : Empty();
  }
  __syncthreads();
  for (int size = 2; size <= kHeadTile; size *= 2) {
    for (int stride = size / 2; stride; stride /= 2) {
      for (int i = t; i < kHeadTile; i += blockDim.x) {
        const int peer = i ^ stride;
        if (peer > i && (((i & size) == 0) ? Better(tile[peer], tile[i])
                                           : Better(tile[i], tile[peer]))) {
          const auto a = tile[i];
          tile[i] = tile[peer];
          tile[peer] = a;
        }
      }
      __syncthreads();
    }
  }
  for (int i = t; i < k; i += blockDim.x)
    output[(r * gridDim.x + blockIdx.x) * k + i] = tile[i];
}

__global__ void Merge(const HeadTwoPair* input, HeadTwoPair* output, int groups,
                      int k) {
  const int out_groups = (groups + 1) / 2;
  const int r = blockIdx.y, g = blockIdx.x, t = threadIdx.x;
  auto* out = output + (r * out_groups + g) * k;
  for (int i = t; i < k; i += blockDim.x)
    out[i] = Empty();
  __syncthreads();
  const auto* pair = input + (r * groups + 2 * g) * k;
  if (2 * g + 1 == groups) {
    for (int i = t; i < k; i += blockDim.x)
      out[i] = pair[i];
    return;
  }
  for (int i = t; i < 2 * k; i += blockDim.x) {
    const auto value = pair[i];
    if (value.id == INT_MAX)
      continue;
    const auto* other = pair + (i < k ? k : 0);
    int lo = 0, hi = k;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (Better(other[mid], value))
        lo = mid + 1;
      else
        hi = mid;
    }
    const int rank = i % k + lo;
    if (rank < k)
      out[rank] = value;
  }
}

HeadTwoPair* Top(const float* scores, int width, int rows, int k,
                 HeadTwoBuffers b, hipStream_t stream) {
  int groups = (rows + kHeadTile - 1) / kHeadTile, slot = 0;
  TopTiles<<<dim3(groups, width), 256, 0, stream>>>(scores, b.sort[slot], rows,
                                                    k);
  while (groups > 1) {
    Merge<<<dim3((groups + 1) / 2, width), 256, 0, stream>>>(
        b.sort[slot], b.sort[slot ^ 1], groups, k);
    groups = (groups + 1) / 2;
    slot ^= 1;
  }
  return b.sort[slot];
}

__global__ void PublishIds(const HeadTwoPair* top, int* ids, const int* extras,
                           int k, int rescore, int width, int rows,
                           int rotation) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * rescore)
    return;
  const int r = i / rescore, j = i % rescore;
  int id = j < k ? top[r * k + j].id : extras[j - k];
  if (j < k && id != INT_MAX)
    id = (id + rotation) % rows;
  if (j >= k && id != INT_MAX)
    for (int a = 0; a < k; ++a)
      if (top[r * k + a].id != INT_MAX &&
          (top[r * k + a].id + rotation) % rows == id) {
        id = INT_MAX;
        break;
      }
  ids[i] = id;
}

__global__ void Extract(int k, const int* ids, const float* exact,
                        HeadTwoResult* results) {
  __shared__ HeadTwoPair tile[512];
  const int r = blockIdx.x, t = threadIdx.x;
  tile[t] = t < k && isfinite(exact[r * k + t])
                ? HeadTwoPair{exact[r * k + t], ids[r * k + t]}
                : Empty();
  __syncthreads();
  for (int size = 2; size <= 512; size *= 2) {
    for (int stride = size / 2; stride; stride /= 2) {
      const int peer = t ^ stride;
      if (peer > t && (((t & size) == 0) ? Better(tile[peer], tile[t])
                                         : Better(tile[t], tile[peer]))) {
        const auto a = tile[t];
        tile[t] = tile[peer];
        tile[peer] = a;
      }
      __syncthreads();
    }
  }
  auto& result = results[r];
  if (t < kHeadCapacity) {
    result.ids[t] = tile[t].id;
    result.logits[t] = tile[t].value;
  }
  if (t == 0)
    result.mismatch = result.missed = 0;
}

__global__ void Sparse(float* full, int rows, const HeadTwoResult* result) {
  const int r = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += gridDim.x * blockDim.x)
    full[r * rows + i] = -INFINITY;
}

__global__ void Scatter(float* full, int rows, const HeadTwoResult* result) {
  const int r = blockIdx.x, t = threadIdx.x;
  if (t < kHeadCapacity && result[r].ids[t] < rows)
    full[r * rows + result[r].ids[t]] = result[r].logits[t];
}

__global__ void CheckRows(const HeadTwoPair* true_top, const float* full,
                          int rows, int k, const int* ids, const float* exact,
                          HeadTwoResult* results) {
  const int r = blockIdx.x, t = threadIdx.x;
  for (int j = t; j < k; j += blockDim.x) {
    const int id = ids[r * k + j];
    if (id >= 0 && id < rows &&
        __float_as_uint(exact[r * k + j]) !=
            __float_as_uint(full[r * rows + id]))
      atomicOr(&results[r].mismatch, 1u);
  }
  if (t < kHeadCheckTop) {
    const int truth = true_top[r * 32 + t].id;
    bool covered = false;
    for (int j = 0; j < k; ++j)
      covered |= ids[r * k + j] == truth;
    if (!covered)
      atomicOr(&results[r].missed, 1u);
  }
}
}  // namespace

bool HeadTwoPre(const void* cheap, const void* q8, int width, int rows,
                int cols, int shortlist, HeadTwoBuffers b, hipStream_t stream,
                int row_rotation, int rescore) {
  if (!HeadTwoWidth(width) || rows < shortlist || cols % 256 ||
      (shortlist != 64 && shortlist != 128 && shortlist != 256 &&
       shortlist != 512))
    return false;
  if (!rescore)
    rescore = shortlist;
  const int stride = ((cols + 511) / 512) * 16;
  const auto* x = static_cast<const Q81*>(q8);
#define HEAD_WIDTH(N)                                                    \
  case N:                                                                \
    PrescoreQ4<N><<<((rows + 3) / 4), 128, 0, stream>>>(                 \
        static_cast<const Q4K*>(cheap), x, b.scores, rows, cols, stride, \
        row_rotation);                                                   \
    break
  switch (width) {
    HEAD_WIDTH(1);
    HEAD_WIDTH(2);
    HEAD_WIDTH(3);
    HEAD_WIDTH(4);
    HEAD_WIDTH(5);
    HEAD_WIDTH(6);
    HEAD_WIDTH(7);
    HEAD_WIDTH(8);
  }
#undef HEAD_WIDTH
  const auto* top = Top(b.scores, width, rows, shortlist, b, stream);
  PublishIds<<<(width * rescore + 255) / 256, 256, 0, stream>>>(
      top, b.ids, b.extras, shortlist, rescore, width, rows, row_rotation);
  return hipGetLastError() == hipSuccess;
}

bool HeadTwoExtract(float* full, int width, int rows, int shortlist,
                    HeadTwoBuffers b, hipStream_t stream, bool extract) {
  if (extract)
    Extract<<<width, 512, 0, stream>>>(shortlist, b.ids, b.exact, b.result);
  Sparse<<<dim3(128, width), 256, 0, stream>>>(full, rows, b.result);
  Scatter<<<width, 512, 0, stream>>>(full, rows, b.result);
  return hipGetLastError() == hipSuccess;
}

bool HeadTwoCheck(const float* full, int width, int rows, int shortlist,
                  HeadTwoBuffers b, hipStream_t stream) {
  const auto* top = Top(full, width, rows, 32, b, stream);
  CheckRows<<<width, 128, 0, stream>>>(top, full, rows, shortlist, b.ids,
                                       b.exact, b.result);
  return hipGetLastError() == hipSuccess;
}
}  // namespace gufo::models::qwen38_flash_next::rocm
