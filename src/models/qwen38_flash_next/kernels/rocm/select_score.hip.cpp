// SelectScoreKernel (the indexer scorer of SelectBlocks) in its own translation unit.
//
// Why a separate file: with the engine's compiler (AMD clang 24.0.0git) the SLP vectorizer merges this kernel's 128
// scalar query loads per head into <8 x float> loads and hoists them ahead of the FMAs. The merged values do not
// fit the 106 SGPRs, 304 of them are spilled into vector lanes (v_writelane / v_readlane), and the kernel issues
// 1213 VALU instructions where 595 are needed. CMake compiles this file with -fno-slp-vectorize (see
// src/models/qwen38_flash_next/CMakeLists.txt). The source below is the kernel as it was in kernels.hip.cpp, byte
// for byte; only its location and the launcher changed, so scores and masks are bit-identical (routed bench
// 10-21: SelectBlocks 1.6-1.8x faster at depths 1K-258K, hashes of all scores and masks equal).
// Do not move it back into kernels.hip.cpp or drop the flag without re-reading the ISA (-Rpass-analysis=
// kernel-resource-usage must show SGPRs Spill: 0 for SelectScoreKernel).
#include "src/models/qwen38_flash_next/kernels/rocm/select_score.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all
// four heads. Preserve the original wave32 F32 accumulation and reduction
// tree: even tiny changes can exchange blocks at the top-k boundary.
__global__ void SelectScoreKernel(const float* q, const __half* blocks,
                                  float* scores, std::uint32_t n_tokens,
                                  const std::uint32_t* start_pos,
                                  std::uint32_t first_token,
                                  std::uint32_t ratio, std::uint32_t budget,
                                  std::uint32_t max_blocks) {
#pragma clang fp reassociate(off)
  const auto t0 = blockIdx.x;
  const auto complete = (*start_pos + first_token + t0 + 1) / ratio;
  const auto b = blockIdx.y * blockDim.x + threadIdx.x;
  if (t0 >= n_tokens || complete <= budget || b >= complete)
    return;
  // Load both stored halves together. Unpacking preserves their exact F32
  // values while avoiding a separate narrow memory load for each component.
  std::uint32_t key[kSelectDim / 2];
#pragma unroll
  for (unsigned i = 0; i < kSelectDim / 2; ++i)
    key[i] = reinterpret_cast<const std::uint32_t*>(
        blocks)[std::size_t{b} * kSelectDim / 2 + i];
  float total = 0.0F;
#pragma unroll
  for (unsigned h = 0; h < kSelectHeads; ++h) {
    __builtin_amdgcn_sched_barrier(0);
    const auto* query = q + (std::size_t{t0} * kSelectHeads + h) * kSelectDim;
    float partial[32];
#pragma unroll
    for (unsigned lane = 0; lane < 32; ++lane) {
      float dot = 0.0F;
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        const auto bits = static_cast<unsigned short>(
            key[(lane + i * 32) / 2] >> ((lane & 1) * 16));
        dot = fmaf(query[lane + i * 32], __half2float(__ushort_as_half(bits)),
                   dot);
      }
      partial[lane] = dot;
      // Bound query-load hoisting: keeping all 128 scalar values live spills
      // registers on gfx1151. Scheduling groups retain every FMA and sum.
      if ((lane + 1) % 16 == 0)
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (unsigned delta = 16; delta; delta >>= 1) {
#pragma unroll
      for (unsigned lane = 0; lane < delta; ++lane)
        partial[lane] += partial[lane + delta];
    }
    total += fmaxf(partial[0], 0.0F);
  }
  scores[std::size_t{t0} * max_blocks + b] = total;
}

}  // namespace

void LaunchSelectScore(const float* q, const __half* blocks, float* scores,
                       std::uint32_t n_tokens, const std::uint32_t* start_pos,
                       std::uint32_t first_token, std::uint32_t ratio,
                       std::uint32_t budget, std::uint32_t max_blocks,
                       std::uint32_t grid_y, hipStream_t stream) {
  // One block per (query row, 256 keys); the grid depth stays 1.
  const dim3 grid(n_tokens, grid_y);
  hipLaunchKernelGGL(SelectScoreKernel, grid, dim3(kSelectThreads), 0, stream,
                     q, blocks, scores, n_tokens, start_pos, first_token, ratio,
                     budget, max_blocks);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
