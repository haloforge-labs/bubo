#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_HALF_KQUANT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_HALF_KQUANT_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {

/// Weight formats of DenseHalfKQuantGemm. 32 bits wide: the value crosses from
/// GCC-built host code into clang-built HIP code (see gemma4 ExpertFormat).
enum class DenseHalfFormat : std::uint32_t { kQ4_K, kQ5_K, kQ6_K };

/// out[t][r] = sum_k W[r][k] x[t][k] for t < rows, r < m, over binary16
/// activation rows `x_f16` (row stride exactly k halves) and K-quant weights
/// `w` ([m][k], k a multiple of 256), accumulated in FP32 on the matrix cores.
/// The kernel is the Gemma 4 lane's dense binary16 K-quant GEMM
/// (src/core/hip/half_kquant_gemm.hpp, one source for both lanes). Any
/// rows >= 1 works; a token's result does not depend on its tile position or
/// on rows. Returns false without launching for rows == 0 or k % 256 != 0.
[[nodiscard]] bool DenseHalfKQuantGemm(DenseHalfFormat format, const void* w,
                                       const void* x_f16, float* out_f32,
                                       std::uint32_t rows, std::uint32_t m,
                                       std::uint32_t k, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_HALF_KQUANT_HPP_
