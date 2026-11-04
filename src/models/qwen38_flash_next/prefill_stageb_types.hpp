#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_STAGEB_TYPES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_STAGEB_TYPES_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {
inline constexpr unsigned kStageBKeep[5]{2, 3, 4, 6, 10};

// Diagnostic-owned PODs shared by host, device and the independent GPU checker.
// No pointer to these buffers is ever supplied to a model kernel.
struct StageBTotal {
  double rf[5]{}, rr[5]{}, error2[5]{}, omitted[5]{};
  double ff{}, top1_mass{}, top4_mass{}, gate_l2{};
  unsigned long long rows{}, invalid{};
};
static_assert(sizeof(StageBTotal) == 208);

struct StageBRow {
  std::uint32_t token{}, layer{};
  float gates[4]{}, gate_l2{}, gate_max{}, top1_mass{}, top4_mass{}, f_norm{};
  float error_norm[4]{}, omitted[4]{};
};
static_assert(sizeof(StageBRow) == 76);
}  // namespace gufo::models::qwen38_flash_next
#endif
