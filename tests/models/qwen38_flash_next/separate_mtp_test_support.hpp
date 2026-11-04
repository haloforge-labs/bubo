#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_SEPARATE_MTP_TEST_SUPPORT_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_SEPARATE_MTP_TEST_SUPPORT_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::test {
// Test-only views of the supported stacked Q8 MTP allocation. Exercise the
// separate producer without adding weight formats, changing loader contracts,
// copying weights or relying on a private low-bit model. The non-const owner
// outlives these aliases; DeviceModel retains the original allocation.
inline void UseSeparateMtpViews(
    models::qwen38_flash_next::rocm::DeviceModel& model) {
  namespace qfn = models::qwen38_flash_next;
  auto& layer = const_cast<qfn::rocm::DeviceLayer&>(model.mtp());
  const auto stacked = layer.attn_qkv;
  const auto& c = model.config();
  if (stacked.empty() || stacked.type != core::GgmlType::kQ8_0 ||
      stacked.cols % 32 != 0 ||
      stacked.rows != 2 * c.AttentionQDim() + 2 * c.AttentionKvDim())
    throw std::invalid_argument(
        "separate producer fixture needs stacked Q8 MTP");
  const std::size_t stride = std::size_t{stacked.cols / 32} * 34;
  const auto view = [&](std::uint32_t first, std::uint32_t rows) {
    auto result = stacked;
    result.data = static_cast<std::uint8_t*>(stacked.data) + first * stride;
    result.rows = rows;
    return result;
  };
  layer.attn_q = view(0, 2 * c.AttentionQDim());
  layer.attn_k = view(2 * c.AttentionQDim(), c.AttentionKvDim());
  layer.attn_v =
      view(2 * c.AttentionQDim() + c.AttentionKvDim(), c.AttentionKvDim());
  layer.attn_qkv = {};
}
}  // namespace gufo::test
#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_SEPARATE_MTP_TEST_SUPPORT_HPP_
