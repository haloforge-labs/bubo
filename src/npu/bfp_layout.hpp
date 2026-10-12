// SPDX-License-Identifier: MIT
// Selectively adapted from mikealanni/gufo 8e0fb1de, kernels.hip.cpp.
#ifndef GUFO_NPU_BFP_LAYOUT_HPP_
#define GUFO_NPU_BFP_LAYOUT_HPP_
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define BFP_HD __host__ __device__
#else
#define BFP_HD
#endif
namespace gufo::npu::bfp {
inline constexpr std::size_t M = 4096, K = 2560, N = 1024;
inline constexpr std::size_t ABytes = M * K / 8 * 9;
inline constexpr std::size_t BBytes = K * N / 8 * 9;
inline constexpr std::size_t CBytes = M * N / 8 * 9;
inline constexpr std::size_t Q8Bytes = N * K / 32 * 34;
enum class Layout { A, B, C };
struct Coordinate {
  std::size_t row, col;
};
// Internal: caller proves g is below the fixed operand's group count.
// A: 128x64 tiles, 2x1 superblocks. B: 64x128 tiles, 1x2
// superblocks, tile columns outermost. C: 512x128, 2x2 superblocks.
BFP_HD constexpr Coordinate Position(Layout layout, std::size_t g) {
  const auto tile = g / (layout == Layout::C ? 8192 : 1024);
  const auto i = g % (layout == Layout::C ? 8192 : 1024);
  if (layout == Layout::A)
    return {(tile / 40) * 128 + (i >> 7) * 16 + ((i >> 3) & 1) * 8 + (i & 7),
            (tile % 40) * 64 + ((i >> 4) & 7) * 8};
  if (layout == Layout::B)
    return {(tile % 40) * 64 + ((i >> 4) & 7) * 8,
            (tile / 40) * 128 + (i >> 7) * 16 + ((i >> 3) & 1) * 8 + (i & 7)};
  return {(tile / 8) * 512 + (i >> 8) * 16 + ((i >> 4) & 1) * 8 + (i & 7),
          (tile % 8) * 128 + ((i >> 5) & 7) * 16 + ((i >> 3) & 1) * 8};
}
}  // namespace gufo::npu::bfp
#undef BFP_HD
#endif
