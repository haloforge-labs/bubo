#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_DISPATCH_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_DISPATCH_HPP_

#include <cstdint>
#include <initializer_list>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

constexpr core::GgmlType StackedProjectionType(core::GgmlType type,
                                             bool preserve_types) {
  return !preserve_types && type == core::GgmlType::kF32
             ? core::GgmlType::kF16
             : type;
}

inline constexpr std::initializer_list<core::GgmlType> kNativeDenseTypes = {
    core::GgmlType::kQ8_0,   core::GgmlType::kBF16,   core::GgmlType::kF16,
    core::GgmlType::kF32,    core::GgmlType::kQ4_K,   core::GgmlType::kQ5_K,
    core::GgmlType::kQ6_K,   core::GgmlType::kQ5_1,   core::GgmlType::kIQ3_S,
    core::GgmlType::kIQ4_XS, core::GgmlType::kIQ4_NL, core::GgmlType::kQ2_0,
    core::GgmlType::kQ3_K};

constexpr bool NativeQuantDense(core::GgmlType type) {
  using T = core::GgmlType;
  switch (type) {
    case T::kQ3_K:
    case T::kQ4_K:
    case T::kQ5_K:
    case T::kQ6_K:
    case T::kQ5_1:
    case T::kIQ3_S:
    case T::kIQ4_XS:
    case T::kIQ4_NL:
    case T::kQ2_0:
      return true;
    default:
      return false;
  }
}

constexpr const char* NativeDenseKernel(core::GgmlType type, bool matrix) {
  if (NativeQuantDense(type))
    return matrix ? "mmq" : "mmvq";
  switch (type) {
    case core::GgmlType::kQ8_0:
      return matrix ? "q8-w8a8-or-f16" : "q8-mmvq";
    case core::GgmlType::kBF16:
    case core::GgmlType::kF16:
      return matrix ? "half-blaslt" : "small-gemm";
    case core::GgmlType::kF32:
      return matrix ? "f32-blas" : "small-gemm";
    default:
      return nullptr;
  }
}

/// Shapes where the decode-once int8 WMMA GEMM (hip::LaunchBatchedQuantGEMMPreQuantized)
/// beats the vendored mmq in prefill, measured on gfx1151 at 4096 tokens
/// (pf-native-gemm microbench). A pure function of tensor type, rows (m),
/// columns (k) and the pass kind: the token count never enters, so a tensor
/// takes the same route in every chunk of a prompt. Q4_K, Q5_K and Q6_K need
/// the launcher's wave64 kernels (m >= 1024, k >= 1024, k % 256 == 0); the
/// wave32 128 x 128 tile spills for Q6_K (0.05-0.09 of Q8_0). The 640 x 2560
/// gate/up shape stays on mmq for every type.
constexpr bool NativeWmmaGemm(core::GgmlType type, std::uint64_t rows,
                              std::uint64_t cols, bool prefill_phase) {
  using T = core::GgmlType;
  if (!prefill_phase)
    return false;
  switch (type) {
    case T::kQ4_K:
    case T::kQ5_K:
    case T::kQ6_K:
    case T::kIQ4_XS:
    case T::kIQ3_S:
      return cols % 256 == 0 && rows >= 1024 && cols >= 1024;
    case T::kIQ4_NL:
      return cols % 32 == 0 && rows >= 1024 && cols >= 512;
    default:
      return false;
  }
}

/// GUFO_PREFILL_NATIVE_GEMM: 0 off, 1 the int8 decode-once GEMM of
/// NativeWmmaGemm, 2 (unset) "half": 1, and Q4_K / Q5_K / Q6_K trunk tensors
/// take the binary16 decode-once GEMM of NativeHalfGemm instead. Returns -1
/// for any other text, "" included.
inline constexpr int kNativeGemmModeInvalid = -1;
inline constexpr const char* kNativeGemmModeHelp =
    "unknown GUFO_PREFILL_NATIVE_GEMM; expected 0, 1 or 2";
constexpr int ParseNativeGemmMode(const char* value) {
  if (!value)
    return 2;
  if (value[0] >= '0' && value[0] <= '2' && value[1] == '\0')
    return value[0] - '0';
  return kNativeGemmModeInvalid;
}

/// Value 2: shapes where the dense binary16 K-quant GEMM (the Gemma 4 lane's
/// kernel, half_kquant_gemm.hpp) beats the int8 route and mmq in prefill,
/// measured on gfx1151 at 4096 tokens (pf-half-route bench, candidate d). A
/// superset of NativeWmmaGemm for the three types (it adds the 640 x 2560
/// gate/up shape), so none falls through to value 1. A pure function of tensor
/// type, rows (m), columns (k) and the pass kind; the token count never enters.
constexpr bool NativeHalfGemm(core::GgmlType type, std::uint64_t rows,
                              std::uint64_t cols, bool prefill_phase) {
  using T = core::GgmlType;
  if (!prefill_phase)
    return false;
  switch (type) {
    case T::kQ4_K:
    case T::kQ5_K:
    case T::kQ6_K:
      return cols % 256 == 0 && cols >= 1024 && rows >= 512;
    default:
      return false;
  }
}

/// The launcher picks a different kernel family below 96 rows. A pass under
/// 96 tokens therefore runs as 96 rows (the extra rows are zero and dropped),
/// so a token is computed by the same kernel in every chunk of a prompt.
inline constexpr std::uint32_t kNativeGemmWideRows = 96;
constexpr std::uint32_t NativeGemmLaunchRows(std::uint32_t n_tokens) {
  return n_tokens < kNativeGemmWideRows ? kNativeGemmWideRows : n_tokens;
}
/// Any pass of 1..max_batch rows takes the GEMM (the activation staging holds
/// max_batch rows, or 96 when that is smaller).
constexpr bool NativeWmmaGemmTokens(std::uint32_t n_tokens,
                                    std::uint32_t max_batch) {
  return n_tokens >= 1 && n_tokens <= max_batch;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
