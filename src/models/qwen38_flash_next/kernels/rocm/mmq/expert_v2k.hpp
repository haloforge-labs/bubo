// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace qfn_mmq::v2k {
constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr int kUsed = 10;
constexpr int kMaxTokens = 8;
constexpr std::size_t kQ8BlockBytes = 36;
constexpr std::size_t Align(std::size_t bytes) {
  return (bytes + 255) & ~std::size_t{255};
}

// Byte offsets in caller-owned, HIP-aligned scratch. Gate/up use the
// executor's existing float buffers; no float h image is allocated.
struct Layout {
  std::size_t xq, hq, expert, start, count, token, destination, bytes;
  constexpr explicit Layout(int tokens)
      : xq(0),
        hq(Align(std::size_t(tokens) * (kHidden / 32) * kQ8BlockBytes)),
        expert(Align(hq +
                     std::size_t(tokens) * kUsed * (kFf / 32) * kQ8BlockBytes)),
        start(expert + std::size_t(tokens) * kUsed * sizeof(std::int32_t)),
        count(start + (std::size_t(tokens) * kUsed + 1) * sizeof(std::int32_t)),
        token(count + sizeof(std::int32_t)),
        destination(token + std::size_t(tokens) * kUsed * sizeof(std::int32_t)),
        bytes(Align(destination +
                    std::size_t(tokens) * kUsed * sizeof(std::int32_t))) {}
};

constexpr bool ValidLayouts() {
  for (int n = 2; n <= kMaxTokens; ++n) {
    const Layout l(n);
    if (l.hq < std::size_t(n) * 80 * kQ8BlockBytes ||
        l.expert < l.hq + std::size_t(n) * 10 * 20 * kQ8BlockBytes ||
        l.start < l.expert + std::size_t(n) * 10 * 4 ||
        l.count < l.start + (std::size_t(n) * 10 + 1) * 4 ||
        l.token < l.count + 4 ||
        l.destination < l.token + std::size_t(n) * 10 * 4 ||
        l.bytes < l.destination + std::size_t(n) * 10 * 4 || l.hq % 256 ||
        l.expert % 256 || l.bytes % 256)
      return false;
  }
  return true;
}
static_assert(ValidLayouts(), "V2K scratch regions must not overlap");
static_assert(Layout(8).hq == 23040);
static_assert(Layout(8).expert - Layout(8).hq == 57600);
static_assert(Layout(8).bytes == 82176);
static_assert(std::size_t(80) * kFf * sizeof(float) == 204800);
static_assert(std::size_t(80) * kHidden * sizeof(float) == 819200);
static_assert(4 * 80 * kQ8BlockBytes == 11520);
static_assert(4 * 20 * kQ8BlockBytes == 2880);
}  // namespace qfn_mmq::v2k
