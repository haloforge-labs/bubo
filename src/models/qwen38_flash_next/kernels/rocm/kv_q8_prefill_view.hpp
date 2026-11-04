#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_PREFILL_VIEW_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_PREFILL_VIEW_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm::detail {

// Scratch sizing only: never changes the executor/session context limit.
constexpr std::uint32_t Q8PrefillViewCapacity(
    std::uint32_t configured, std::uint32_t fallback,
    std::uint32_t wmma_limit) noexcept {
  const auto requested = configured != 0 ? configured : fallback;
  return requested < wmma_limit ? requested : wmma_limit;
}

// Larger supported sessions use the canonical reader beyond this scratch.
// Subtraction avoids overflow when checking an untrusted frontier.
constexpr bool Q8PrefillViewFits(std::uint32_t capacity, std::uint32_t start,
                                 std::uint32_t count) noexcept {
  return capacity != 0 && start <= capacity && count <= capacity - start;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::detail

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_Q8_PREFILL_VIEW_HPP_
