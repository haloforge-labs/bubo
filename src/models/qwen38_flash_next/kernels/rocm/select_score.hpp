#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SELECT_SCORE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SELECT_SCORE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {

/// The indexer of the Qwen3.8 Flash-Next sparse attention: 4 heads of 128, scored per key block.
constexpr std::uint32_t kSelectHeads = 4;
constexpr std::uint32_t kSelectDim = 128;
/// Keys scored per block (one thread each); SelectBlocks sizes its grid with this.
constexpr std::uint32_t kSelectThreads = 256;

/// Launches SelectScoreKernel on a (n_tokens, grid_y) grid of kSelectThreads-thread blocks.
/// Defined in select_score.hip.cpp, which is compiled with -fno-slp-vectorize.
void LaunchSelectScore(const float* q, const __half* blocks, float* scores,
                       std::uint32_t n_tokens, const std::uint32_t* start_pos,
                       std::uint32_t first_token, std::uint32_t ratio,
                       std::uint32_t budget, std::uint32_t max_blocks,
                       std::uint32_t grid_y, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SELECT_SCORE_HPP_
