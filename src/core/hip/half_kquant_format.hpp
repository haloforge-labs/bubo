#ifndef GUFO_CORE_HIP_HALF_KQUANT_FORMAT_HPP_
#define GUFO_CORE_HIP_HALF_KQUANT_FORMAT_HPP_

#include <cstdint>

namespace gufo::hip::half_kquant {

/// Weight formats of the binary16 K-quant GEMM (half_kquant_gemm.hpp) and of
/// the Gemma 4 routed expert kernels, which share it.
/// 32 bits wide: the value crosses from GCC-built host code into clang-built
/// HIP code, and clang assumes a narrower argument arrives zero-extended where
/// GCC leaves the upper bits undefined.
enum class ExpertFormat : std::uint32_t {
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_0,
  kQ5_1,
  kF16,
  kQ4_0,
  kQ5_0,
};

}  // namespace gufo::hip::half_kquant

#endif  // GUFO_CORE_HIP_HALF_KQUANT_FORMAT_HPP_
