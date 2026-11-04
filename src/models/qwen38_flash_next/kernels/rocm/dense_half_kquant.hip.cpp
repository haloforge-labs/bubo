// Dense binary16 GEMM over Q4_K / Q5_K / Q6_K weights
// (GUFO_PREFILL_NATIVE_GEMM=2).
//
// The kernel is not written here: it is the Gemma 4 lane's
// RoutedHalfKQuantKernel (dense mode), which both lanes include from
// src/core/hip/half_kquant_gemm.hpp. This file only instantiates it for the
// three formats and launches it with the grid of Gemma 4's
// LaunchDenseHalfGemm (moe_prefill.hip.cpp). Compiled with the target's
// default flags, as the Gemma 4 lane is.
#include "src/models/qwen38_flash_next/kernels/rocm/dense_half_kquant.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/hip/half_kquant_gemm.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

bool DenseHalfKQuantGemm(DenseHalfFormat format, const void* w,
                         const void* x_f16, float* out_f32, std::uint32_t rows,
                         std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  namespace hk = ::gufo::hip::half_kquant;
  if (rows == 0 || k % 256 != 0) {
    return false;
  }
  constexpr int kTile = hk::kHalfKQuantDenseTile;
  constexpr std::uint32_t kGroup = hk::kHalfKQuantDenseGroup;
  const std::uint32_t token_tiles = (rows + kTile - 1) / kTile;
  const std::uint32_t row_blocks =
      (m + hk::kRowsPerBlock - 1) / hk::kRowsPerBlock;
  const dim3 grid(token_tiles * row_blocks);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x_f16);
  const auto launch = [&]<hk::ExpertFormat F>() {
    hk::RoutedHalfKQuantKernel<F, kTile, false, true>
        <<<grid, hk::kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                            nullptr, out_f32, nullptr, m, k,
                                            rows, kGroup);
  };
  switch (format) {
    case DenseHalfFormat::kQ4_K:
      launch.template operator()<hk::ExpertFormat::kQ4_K>();
      return true;
    case DenseHalfFormat::kQ5_K:
      launch.template operator()<hk::ExpertFormat::kQ5_K>();
      return true;
    case DenseHalfFormat::kQ6_K:
      launch.template operator()<hk::ExpertFormat::kQ6_K>();
      return true;
  }
  return false;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
