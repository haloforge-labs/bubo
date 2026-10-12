// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_BFP_HPP_
#define GUFO_NPU_BFP_HPP_
#include <array>
#include <span>
#include <vector>

#include "bfp_layout.hpp"

namespace gufo::npu::bfp {
using Group = std::array<std::uint8_t, 9>;
// Shared raw FP32 exponent, nearest-even mantissas clamped to [-128,127].
// Like the fork, exponent <7 flushes mantissas to zero. Nonfinite inputs
// and exponent 255 or FP32-overflowing streams are refused. This is reduced
// precision.
Group EncodeGroup(const std::array<float, 8>& values);
std::array<float, 8> DecodeGroup(const Group& group);
float Half(std::uint16_t bits);
// Checked storage bounds, shared by CPU and HIP launch boundaries.
std::size_t MatrixExtent(std::size_t rows, std::size_t pitch,
                         std::size_t offset, std::size_t columns);
// Fixed qualified geometry only. CPU FP16 activations; owned Q8_0
// staging, already padded to N rows. Results own their bytes.
std::vector<std::uint8_t> EncodeA(std::span<const std::uint16_t> input,
                                  std::size_t pitch = K);
std::vector<std::uint8_t> EncodeB(std::span<const std::byte> staged_q8);
// C to pitched FP32 with element-level masking of partial groups. Validate
// the entire stream before writing output; a bad exponent is transactional.
void DecodeC(std::span<const std::uint8_t> input, std::span<float> output,
             std::size_t pitch, std::size_t col0 = 0,
             std::size_t valid_cols = N, bool accumulate = false);
}  // namespace gufo::npu::bfp
#endif
