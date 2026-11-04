#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MOE_FUSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MOE_FUSION_HPP_
#include <cstdint>
namespace gufo::models::qwen38_flash_next {
inline constexpr std::uint64_t kMoeFusionKey = std::uint64_t{1} << 47;
inline constexpr bool DecodeMoeFusionShape(std::uint32_t rows,
                                           std::uint32_t slots,
                                           std::uint32_t hidden,
                                           std::uint32_t streams, bool half,
                                           bool prefill) {
  return !half && !prefill && rows > 0 && rows <= 8 && slots > 0 &&
         slots <= 16 && hidden > 0 && hidden <= 2560 && hidden % 32 == 0 &&
         streams == 4;
}
struct MoeFusionAudit {
  std::uint64_t launches{0}, checks{0}, mismatches{0};
};
struct MoeFusionInput {
  const float* expert{nullptr};
  const float* weights{nullptr};
  const float* shared{nullptr};
  const float* gate{nullptr};
  std::uint32_t slots{0}, gate_stride{0};
  MoeFusionAudit* audit{nullptr};
};
}  // namespace gufo::models::qwen38_flash_next
#endif
