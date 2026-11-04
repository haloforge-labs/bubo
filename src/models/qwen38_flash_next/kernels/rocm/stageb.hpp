#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_STAGEB_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_STAGEB_HPP_
#include <hip/hip_runtime.h>

#include "src/models/qwen38_flash_next/prefill_stageb_types.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
// Exactly one diagnostic launch. Only totals and samples are writable, and
// neither is passed to any model operator. IDs resolve ties when slots reorder.
void PrefillStageB(const void* expert_out, bool half_rows, const float* weights,
                   const std::int32_t* ids, const float* inject,
                   std::uint32_t parts, std::uint32_t tokens,
                   std::uint32_t hidden, std::uint32_t row_start,
                   std::uint32_t layer, std::uint32_t stride,
                   std::uint32_t sample_every, std::uint32_t sample_capacity,
                   StageBTotal* totals, StageBRow* samples, hipStream_t stream);
}  // namespace gufo::models::qwen38_flash_next::rocm
#endif
