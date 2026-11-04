#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_CHECKPOINT_MEMORY_TEST_SUPPORT_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_CHECKPOINT_MEMORY_TEST_SUPPORT_HPP_

#include <cstdint>
#include <stdexcept>

#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace gufo::test {

// Preserve the original F16 minimum exactly. Layer count, K/V copies and
// checkpoint depth cancel in the selected/F16 ordinary-KV byte ratio.
constexpr std::uint64_t QfnDeepCheckpointMinimumBytes(
    std::uint32_t kv_elements,
    models::qwen38_flash_next::KvCacheType cache_type) {
  constexpr std::uint64_t f16_minimum = 80ULL << 20;
  using models::qwen38_flash_next::KvCacheType;
  using models::qwen38_flash_next::KvRowBytes;
  const auto f16_row_bytes = KvRowBytes(kv_elements, KvCacheType::kF16);
  const auto selected_row_bytes = KvRowBytes(kv_elements, cache_type);
  if (f16_row_bytes == 0)
    throw std::invalid_argument(
        "checkpoint memory bound needs a nonempty KV row");
  return f16_minimum * selected_row_bytes / f16_row_bytes;
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_CHECKPOINT_MEMORY_TEST_SUPPORT_HPP_
