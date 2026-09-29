#ifndef GUFO_MODELS_QWEN36_A3B_DFLASH_KERNELS_HPP_
#define GUFO_MODELS_QWEN36_A3B_DFLASH_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

/// Small helpers for the A3B DFlash-2 drafter (dflash.cpp); the draft's
/// convolution, attention and selector come from the 27B DFlash kernels in
/// gufo_core, used read-only.
namespace gufo::models::qwen36_a3b {

/// dst[r * dst_stride + i] = src[r * width + i] for r < rows, i < width: one
/// trunk layer's output rows into their slot of the concatenated features.
void TapCopy(const float* src, float* dst, std::uint32_t rows,
             std::uint32_t width, std::uint32_t dst_stride, hipStream_t stream);

/// prob[r] = softmax(logits[r])[tok[r]] over `vocab` logits per row.
void TokenProb(const float* logits, const std::uint32_t* tok, float* prob,
               std::uint32_t rows, std::uint32_t vocab, hipStream_t stream);

/// Stores K and V rows (F32, `width` each) at ring slots (*pos + r) %
/// capacity.
void RingStore(const float* k, const float* v, float* ring_k, float* ring_v,
               const std::uint32_t* pos, std::uint32_t rows,
               std::uint32_t width, std::uint32_t capacity,
               hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_DFLASH_KERNELS_HPP_
