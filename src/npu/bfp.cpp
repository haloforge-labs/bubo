// SPDX-License-Identifier: MIT
// Selective BFP math/layout adaptation from mikealanni/gufo 8e0fb1de.
#include "bfp.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gufo::npu::bfp {
namespace {
[[noreturn]] void Invalid() {
  throw std::invalid_argument("NPU BFP storage or values invalid");
}
float Even(float x) {
  const float lo = std::floor(x), fraction = x - lo;
  return lo +
         (fraction > 0.5F || (fraction == 0.5F && std::fmod(lo, 2.0F) != 0));
}
int Signed(std::uint8_t x) {
  return x < 128 ? x : int(x) - 256;
}
void Validate(const Group& group) {
  if (group[0] == 255 ||
      (group[0] == 254 &&
       std::find(group.begin() + 1, group.end(), 128) != group.end()))
    Invalid();
}
}  // namespace
float Half(std::uint16_t bits) {
  const int e = (bits >> 10) & 31, m = bits & 1023;
  if (e == 31)
    Invalid();
  const float value =
      e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(1024 + m), e - 25);
  return bits & 0x8000 ? -value : value;
}
Group EncodeGroup(const std::array<float, 8>& values) {
  unsigned exponent = 0;
  for (float value : values) {
    if (!std::isfinite(value))
      Invalid();
    exponent =
        std::max(exponent, (std::bit_cast<std::uint32_t>(value) >> 23) & 255U);
  }
  const float inv =
      std::bit_cast<float>(exponent >= 7 ? (260U - exponent) << 23 : 0U);
  Group out{};
  out[0] = static_cast<std::uint8_t>(exponent);
  for (unsigned i = 0; i < 8; ++i) {
    const auto q =
        static_cast<int>(std::clamp(Even(values[i] * inv), -128.0F, 127.0F));
    out[i + 1] = static_cast<std::uint8_t>(q & 255);
  }
  Validate(out);
  return out;
}
std::array<float, 8> DecodeGroup(const Group& group) {
  Validate(group);
  const float mult = std::ldexp(1.0F, int(group[0]) - 127) / 64.0F;
  std::array<float, 8> out{};
  for (unsigned i = 0; i < 8; ++i)
    out[i] = float(Signed(group[i + 1])) * mult;
  return out;
}
std::size_t MatrixExtent(std::size_t rows, std::size_t pitch,
                         std::size_t offset, std::size_t columns) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (!rows || !columns || offset > pitch || columns > pitch - offset ||
      (rows - 1) > (max - offset - columns) / pitch)
    Invalid();
  return (rows - 1) * pitch + offset + columns;
}
std::vector<std::uint8_t> EncodeA(std::span<const std::uint16_t> input,
                                  std::size_t pitch) {
  if (input.size() < MatrixExtent(M, pitch, 0, K) || !input.data())
    Invalid();
  std::vector<std::uint8_t> out(ABytes);
  for (std::size_t g = 0; g < ABytes / 9; ++g) {
    const auto p = Position(Layout::A, g);
    std::array<float, 8> v;
    for (unsigned i = 0; i < 8; ++i)
      v[i] = Half(input[p.row * pitch + p.col + i]);
    const auto bytes = EncodeGroup(v);
    std::copy(bytes.begin(), bytes.end(), out.begin() + g * 9);
  }
  return out;
}
std::vector<std::uint8_t> EncodeB(std::span<const std::byte> input) {
  if (input.size() != Q8Bytes || !input.data())
    Invalid();
  std::vector<std::uint8_t> out(BBytes);
  for (std::size_t g = 0; g < BBytes / 9; ++g) {
    const auto p = Position(Layout::B, g);
    const auto block = (p.col * K + p.row) / 32 * 34;
    const auto d = std::to_integer<unsigned>(input[block]) |
                   (std::to_integer<unsigned>(input[block + 1]) << 8);
    const auto scale = Half(static_cast<std::uint16_t>(d));
    std::array<float, 8> v;
    for (unsigned i = 0; i < 8; ++i)
      v[i] = scale * Signed(std::to_integer<std::uint8_t>(
                         input[block + 2 + p.row % 32 + i]));
    const auto bytes = EncodeGroup(v);
    std::copy(bytes.begin(), bytes.end(), out.begin() + g * 9);
  }
  return out;
}
void DecodeC(std::span<const std::uint8_t> input, std::span<float> output,
             std::size_t pitch, std::size_t col0, std::size_t valid_cols,
             bool accumulate) {
  if (valid_cols == 0 || valid_cols > N || input.size() != CBytes ||
      !input.data() ||
      output.size() < MatrixExtent(M, pitch, col0, valid_cols) ||
      !output.data())
    Invalid();
  for (std::size_t g = 0; g < CBytes / 9; ++g) {
    Group bytes;
    std::copy_n(input.begin() + g * 9, 9, bytes.begin());
    Validate(bytes);
  }
  for (std::size_t g = 0; g < CBytes / 9; ++g) {
    const auto p = Position(Layout::C, g);
    Group bytes;
    std::copy_n(input.begin() + g * 9, 9, bytes.begin());
    const auto values = DecodeGroup(bytes);
    for (unsigned i = 0; i < 8 && p.col + i < valid_cols; ++i) {
      auto& dst = output[p.row * pitch + col0 + p.col + i];
      dst = accumulate ? dst + values[i] : values[i];
    }
  }
}
}  // namespace gufo::npu::bfp
