#include "src/models/qwen38_flash_next/kernels/rocm/stageb.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
constexpr unsigned kThreads = 128;

// Separate translation unit: no fast-math or reassociation. Expert products,
// prefix sums, dot products and device accumulation are all FP64. No residual,
// shared-expert row, norm output or model scratch is a writable argument.
template<typename T>
__global__ void PrefillStageBKernel(const T* expert_out, const float* weights,
                                    const std::int32_t* ids,
                                    const float* inject, unsigned parts,
                                    unsigned hidden, unsigned row_start,
                                    unsigned layer, unsigned stride,
                                    unsigned every, unsigned capacity,
                                    StageBTotal* totals, StageBRow* samples) {
  const unsigned row = blockIdx.x;
  const unsigned token = row_start + row;
  const bool accumulate = token % stride == 0;
  const bool sample = token % every == 0;
  if (!accumulate && !sample)
    return;  // Uniform over the block.
  __shared__ unsigned order[10];
  __shared__ double w[10], omitted[5], top4;
  __shared__ float gates[4], gate_l2, gate_max;
  __shared__ bool invalid;
  __shared__ double reduction[13][kThreads];
  if (threadIdx.x == 0) {
    invalid = false;
    double mass = 0;
    for (unsigned s = 0; s < 10; ++s) {
      w[s] = weights[std::size_t{row} * 10 + s];
      order[s] = s;
      mass += w[s];
      if (!isfinite(w[s]) || w[s] < 0 || ids[std::size_t{row} * 10 + s] < 0)
        invalid = true;
      for (unsigned p = 0; p < s; ++p)
        if (ids[std::size_t{row} * 10 + p] == ids[std::size_t{row} * 10 + s])
          invalid = true;
    }
    if (fabs(mass - 1.) > 1e-5)
      invalid = true;
    // Sort unequal weights, retaining the supplied router rank on equality.
    // Normalization can round distinct probabilities to equal stored weights.
    for (unsigned s = 1; s < 10; ++s) {
      const unsigned item = order[s];
      unsigned p = s;
      while (p && w[item] > w[order[p - 1]]) {
        order[p] = order[p - 1];
        --p;
      }
      order[p] = item;
    }
    top4 = 0;
    double prefix = 0;
    unsigned candidate = 0;
    for (unsigned s = 0; s < 10; ++s) {
      prefix += w[order[s]];
      if (s < 4)
        top4 += w[order[s]];
      if (s + 1 == (candidate == 0   ? 2U
                    : candidate == 1 ? 3U
                    : candidate == 2 ? 4U
                    : candidate == 3 ? 6U
                                     : 10U))
        omitted[candidate++] = s == 9 ? 0. : 1. - prefix;
    }
    float l2 = 0;
    gate_max = 0;
    for (unsigned s = 0; s < 4; ++s) {
      float logit = 0;
      for (unsigned p = 0; p < parts; ++p)
        logit += inject[(std::size_t{row} * 4 + s) * parts + p];
      gates[s] = 2.f * (1.f / (1.f + __expf(-logit / 4.f)));
      if (!isfinite(gates[s]) || !isfinite(logit))
        invalid = true;
      l2 += gates[s] * gates[s];
      gate_max = fmaxf(gate_max, gates[s]);
    }
    gate_l2 = sqrtf(l2);
  }
  __syncthreads();
  if (invalid) {
    if (threadIdx.x == 0)
      atomicAdd(&totals[layer].invalid, 1ULL);
    return;
  }
  double local[13]{};  // four RF, four RR, four squared errors, one FF
  const T* rows = expert_out + std::size_t{row} * 10 * hidden;
  for (unsigned h = threadIdx.x; h < hidden; h += kThreads) {
    double prefix = 0, kept[4];
    unsigned candidate = 0;
    for (unsigned s = 0; s < 10; ++s) {
      const unsigned slot = order[s];
      prefix += w[slot] * static_cast<double>(static_cast<float>(
                              rows[std::size_t{slot} * hidden + h]));
      if (candidate < 4 && s + 1 == (candidate == 0   ? 2U
                                     : candidate == 1 ? 3U
                                     : candidate == 2 ? 4U
                                                      : 6U))
        kept[candidate++] = prefix;
    }
    const double full = prefix;
    for (unsigned k = 0; k < 4; ++k) {
      local[k] += kept[k] * full;
      local[4 + k] += kept[k] * kept[k];
      const double error = full - kept[k];
      local[8 + k] += error * error;
    }
    local[12] += full * full;
  }
  for (unsigned f = 0; f < 13; ++f)
    reduction[f][threadIdx.x] = local[f];
  __syncthreads();
  for (unsigned offset = kThreads / 2; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      for (unsigned f = 0; f < 13; ++f)
        reduction[f][threadIdx.x] += reduction[f][threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double ff = reduction[12][0];
    if (accumulate) {
      auto& t = totals[layer];
      for (unsigned k = 0; k < 5; ++k) {
        atomicAdd(&t.rf[k], k == 4 ? ff : reduction[k][0]);
        atomicAdd(&t.rr[k], k == 4 ? ff : reduction[4 + k][0]);
        atomicAdd(&t.error2[k], k == 4 ? 0. : reduction[8 + k][0]);
        atomicAdd(&t.omitted[k], omitted[k]);
      }
      atomicAdd(&t.ff, ff);
      atomicAdd(&t.top1_mass, w[order[0]]);
      atomicAdd(&t.top4_mass, top4);
      atomicAdd(&t.gate_l2, static_cast<double>(gate_l2));
      atomicAdd(&t.rows, 1ULL);
    }
    if (sample) {
      const auto first = (std::uint64_t{row_start} + every - 1) / every;
      const auto index = token / every - first;
      if (index >= capacity) {
        atomicAdd(&totals[layer].invalid, 1ULL);
        return;
      }
      auto& out = samples[std::size_t{layer} * capacity + index];
      out.token = token;
      out.layer = layer;
      out.gate_l2 = gate_l2;
      out.gate_max = gate_max;
      out.top1_mass = static_cast<float>(w[order[0]]);
      out.top4_mass = static_cast<float>(top4);
      out.f_norm = static_cast<float>(sqrt(ff));
      for (unsigned k = 0; k < 4; ++k) {
        out.gates[k] = gates[k];
        out.error_norm[k] = static_cast<float>(sqrt(reduction[8 + k][0]));
        out.omitted[k] = static_cast<float>(omitted[k]);
      }
    }
  }
}
}  // namespace

void PrefillStageB(const void* expert_out, bool half_rows, const float* weights,
                   const std::int32_t* ids, const float* inject,
                   std::uint32_t parts, std::uint32_t tokens,
                   std::uint32_t hidden, std::uint32_t row_start,
                   std::uint32_t layer, std::uint32_t stride,
                   std::uint32_t every, std::uint32_t capacity,
                   StageBTotal* totals, StageBRow* samples,
                   hipStream_t stream) {
  if (!expert_out || !weights || !ids || !inject || !parts || !tokens ||
      !hidden || !stride || !every || !capacity || !totals || !samples ||
      layer >= 48 || std::uint64_t{row_start} + tokens > UINT32_MAX)
    throw std::invalid_argument("invalid stage B diagnostic geometry");
  if (half_rows)
    PrefillStageBKernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __half*>(expert_out), weights, ids, inject, parts,
        hidden, row_start, layer, stride, every, capacity, totals, samples);
  else
    PrefillStageBKernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const float*>(expert_out), weights, ids, inject, parts,
        hidden, row_start, layer, stride, every, capacity, totals, samples);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
