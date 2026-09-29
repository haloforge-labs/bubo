#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_VERIFY_GEMM_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_VERIFY_GEMM_27B_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// Qwen3.8-27B verification projections at 5..8 rows: launch geometries
/// measured faster on Windows (kernel-bench/kb27_sweep) for the UD-Q4_K_XL
/// shapes. Every geometry is the production exact small-batch kernel with
/// another block shape; outputs are bit-identical to
/// LaunchBatchedQuantGEMMFp32. Returns false when no tuned geometry applies.
[[nodiscard]] bool TryLaunchVerifyGemm27(core::GgmlType type, const void* w,
                                         const float* x, float* y,
                                         std::size_t batch, std::size_t m,
                                         std::size_t k, hipStream_t stream);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_VERIFY_GEMM_27B_HPP_
