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

// Same default, strtol interpretation and clamp as the single-session cutoff.
// Load and the cached serving cutoff share this parser.
inline std::uint32_t TwinMaxNValue(const char* e) {
  if (e == nullptr || *e == '\0')
    return 8;
  const long x = std::strtol(e, nullptr, 10);
  return static_cast<std::uint32_t>(x < 1 ? 1 : (x > 8 ? 8 : x));
}

inline bool TwinBatchEnabled() {
  const char* e = std::getenv("GUFO_DENSE_TWIN_BATCH");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}
inline bool BatchUsesTwin(bool armed, bool enabled, bool prefill) {
  return armed && enabled && !prefill;
}
inline bool ConfiguredMultiSessionTwin(bool armed, bool enabled,
                                       std::uint32_t sessions) {
  return armed && enabled && sessions >= 2;
}
inline bool TwinBatchCapacityValid(std::uint32_t sessions,
                                   std::uint32_t chain) {
  return sessions > 0 && sessions <= kBatchMaxSessions && chain > 0 &&
         chain <= kBatchMaxChain &&
         std::uint64_t{sessions} * chain <= kTwinBatchMax;
}
inline const char* TwinBatchLoadError(bool armed, bool enabled,
                                      std::uint32_t sessions,
                                      std::uint64_t chain,
                                      std::uint32_t max_n) {
  if (!armed)
    return nullptr;
  if (!enabled && sessions > 1)
    return "GUFO_DENSE_TWIN_MODEL supports one decode session (the batched "
           "decode loop is not twinned)";
  // Single-session serving keeps the legacy cutoff, even with batch opt-in.
  // The all-twin/full-chain contract applies to the multi-session loop.
  if (!enabled || sessions == 1)
    return nullptr;
  if (max_n < kBatchMaxChain)
    return "decode twin requires GUFO_DENSE_TWIN_MAX_N >= 8 (no "
           "precision mixing)";
  if (chain > kBatchMaxChain)
    return "batched decode twin exceeds decode chain capacity";
  if (!TwinBatchCapacityValid(sessions, static_cast<std::uint32_t>(chain)))
    return "batched decode twin exceeds kTwinBatchMax / decode chain capacity";
  return nullptr;
}
inline bool TwinDenseType(core::GgmlType t) {
  using T = core::GgmlType;
  return t == T::kIQ4_XS || t == T::kIQ4_NL || t == T::kQ4_K || t == T::kQ5_K ||
         t == T::kQ5_1;
}
enum class BatchDensePath {
  kDense,
  kTwinSlices,
  kTwinMultiRow,
  kQ8,
  kSmallGemm
};
inline BatchDensePath BatchDenseDispatch(core::GgmlType t, std::uint32_t rows,
                                         bool armed,
                                         bool multi_session = false) {
  if (armed && multi_session &&
      (t == core::GgmlType::kIQ4_XS || t == core::GgmlType::kIQ4_NL))
    return BatchDensePath::kTwinMultiRow;
  if (rows <= kBatchMaxChain)
    return BatchDensePath::kDense;
  if (armed && TwinDenseType(t))
    return BatchDensePath::kTwinSlices;
  return t == core::GgmlType::kQ8_0 ? BatchDensePath::kQ8
                                    : BatchDensePath::kSmallGemm;
}
}  // namespace gufo::models::qwen38_flash_next
#endif
