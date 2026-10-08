#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_BATCH_TWIN_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_BATCH_TWIN_POLICY_HPP_

#include <cstdint>
#include <cstdlib>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {
// These are the decode loop's session and chain bounds, not kernel tile sizes.
inline constexpr std::uint32_t kBatchMaxSessions = 8;
inline constexpr std::uint32_t kBatchMaxChain = 8;
inline constexpr std::uint32_t kTwinBatchMax =
    kBatchMaxSessions * kBatchMaxChain;

inline bool TwinBatchEnabled() {
  const char* e = std::getenv("GUFO_DENSE_TWIN_BATCH");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}
inline bool BatchUsesTwin(bool armed, bool enabled, bool prefill) {
  return armed && enabled && !prefill;
}
inline bool TwinBatchCapacityValid(std::uint32_t sessions,
                                   std::uint32_t chain) {
  return sessions > 0 && sessions <= kBatchMaxSessions && chain > 0 &&
         chain <= kBatchMaxChain &&
         std::uint64_t{sessions} * chain <= kTwinBatchMax;
}
inline const char* TwinBatchLoadError(bool armed, bool enabled,
                                      std::uint32_t sessions,
                                      std::uint32_t chain, long max_n) {
  if (!armed)
    return nullptr;
  if (!enabled && sessions > 1)
    return "GUFO_DENSE_TWIN_MODEL supports one decode session (the batched "
           "decode loop is not twinned)";
  if (max_n < 8)
    return "decode twin requires GUFO_DENSE_TWIN_MAX_N >= 8 (no "
           "precision mixing)";
  if (enabled && !TwinBatchCapacityValid(sessions, chain))
    return "batched decode twin exceeds kTwinBatchMax / decode chain capacity";
  return nullptr;
}
inline bool TwinDenseType(core::GgmlType t) {
  using T = core::GgmlType;
  return t == T::kIQ4_XS || t == T::kIQ4_NL || t == T::kQ4_K || t == T::kQ5_K ||
         t == T::kQ5_1;
}
enum class BatchDensePath { kDense, kTwinSlices, kQ8, kSmallGemm };
inline BatchDensePath BatchDenseDispatch(core::GgmlType t, std::uint32_t rows,
                                         bool armed) {
  if (rows <= kBatchMaxChain)
    return BatchDensePath::kDense;
  if (armed && TwinDenseType(t))
    return BatchDensePath::kTwinSlices;
  return t == core::GgmlType::kQ8_0 ? BatchDensePath::kQ8
                                    : BatchDensePath::kSmallGemm;
}
}  // namespace gufo::models::qwen38_flash_next
#endif
