// SPDX-License-Identifier: MIT
#include "src/npu/bfp.hpp"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace gufo::npu::bfp;
namespace {
void Check(bool value) {
  if (!value)
    throw std::runtime_error("BFP oracle failed");
}
template<class F>
void Reject(F call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected);
}
// Independent inverse oracle, specified as nested tiles/superblocks/8x8
// blocks instead of the production flattened stream-to-coordinate mapping.
std::size_t Index(Layout l, std::size_t r, std::size_t c) {
  if (l == Layout::A)
    return ((r / 128 * 40 + c / 64) * 1024 + r % 128 / 16 * 128 +
            c % 64 / 8 * 16 + r % 16 / 8 * 8 + r % 8) *
           9;
  if (l == Layout::B)
    return ((c / 128 * 40 + r / 64) * 1024 + c % 128 / 16 * 128 +
            r % 64 / 8 * 16 + c % 16 / 8 * 8 + c % 8) *
           9;
  return ((r / 512 * 8 + c / 128) * 8192 + r % 512 / 16 * 256 +
          c % 128 / 16 * 32 + r % 16 / 8 * 16 + c % 16 / 8 * 8 + r % 8) *
         9;
}
int Pattern(std::size_t r, std::size_t c) {
  constexpr int values[] = {1, -1, 2, -2};
  return values[(r * 17 + c * 7) % 4];
}
void Layouts() {
  for (auto l : {Layout::A, Layout::B, Layout::C}) {
    const auto rows = l == Layout::B ? K : M, cols = l == Layout::A ? K : N;
    std::vector<bool> seen(rows * cols / 8);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c) {
        if ((l == Layout::B ? r : c) % 8)
          continue;
        const auto g = Index(l, r, c) / 9;
        Check(g < seen.size() && !seen[g]);
        seen[g] = true;
        const auto p = Position(l, g);
        Check(p.row == r && p.col == c);
      }
    Check(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }));
  }
  std::cout << "PASS: full A/B/C bijections, independent inverse oracle, all "
               "tile/superblock boundaries\n";
}
void Groups() {
  Check(EncodeGroup({1.9921875F, -1.9921875F, 0.5F / 64, 1.5F / 64, 2.5F / 64,
                     -0.5F / 64, -1.5F / 64, -2.5F / 64}) ==
        Group{127, 127, 128, 0, 2, 2, 0, 254, 254});
  const auto base = EncodeGroup({1, -1, 1.5F, -1.5F, 2, -2, 0, -0.0F});
  Check(base == Group{128, 32, 224, 48, 208, 64, 192, 0, 0});
  Check(DecodeGroup(base) ==
        std::array<float, 8>{1, -1, 1.5F, -1.5F, 2, -2, 0, 0});
  Check(EncodeGroup({}) == Group{});
  for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
    Check(std::fesetround(mode) == 0);
    Check(EncodeGroup({1.9921875F, -1.9921875F, 0.5F / 64, 1.5F / 64, 2.5F / 64,
                       -0.5F / 64, -1.5F / 64, -2.5F / 64}) ==
          Group{127, 127, 128, 0, 2, 2, 0, 254, 254});
  }
  Check(std::fesetround(FE_TONEAREST) == 0);
  for (unsigned e = 0; e < 255; ++e) {
    const auto value = std::bit_cast<float>(e << 23 | (e ? 0U : 1U));
    std::array<float, 8> v;
    v.fill(value);
    const auto b = EncodeGroup(v);
    Check(b[0] == e && b[1] == (e < 7 ? 0 : 64));
    if (e < 7)
      Check(std::all_of(b.begin() + 1, b.end(), [](auto q) { return q == 0; }));
  }
  for (float bad : {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    Reject([&] { EncodeGroup({bad, 0, 0, 0, 0, 0, 0, 0}); });
  Reject([] { DecodeGroup(Group{255}); });
  Reject([] { DecodeGroup(Group{254, 128}); });
  Reject([] {
    EncodeGroup({-std::numeric_limits<float>::max(), 0, 0, 0, 0, 0, 0, 0});
  });
  for (unsigned h = 0; h < 65536; ++h) {
    const auto e = (h >> 10) & 31, m = h & 1023;
    if (e == 31) {
      Reject([&] { Half(h); });
      continue;
    }
    const long double magnitude =
        e ? (1.0L + m / 1024.0L) * std::pow(2.0L, int(e) - 15)
          : m * std::pow(2.0L, -24);
    const float expected = float(h & 0x8000 ? -magnitude : magnitude);
    Check(std::bit_cast<std::uint32_t>(Half(h)) ==
          std::bit_cast<std::uint32_t>(expected));
  }
  std::cout
      << "PASS: golden groups, signed mantissas, ties-even/clamp, rounding "
         "modes, exponents 0..254, exhaustive FP16 domain, nonfinite refusal\n";
}
void Operands() {
  const std::size_t pitch = K + 3;
  std::vector<std::uint16_t> a(MatrixExtent(M, pitch, 0, K), 0x7c00);
  for (std::size_t r = 0; r < M; ++r)
    for (std::size_t c = 0; c < K; ++c) {
      int p = Pattern(r, c);
      a[r * pitch + c] =
          (std::abs(p) == 2 ? 0x4000 : 0x3c00) | (p < 0 ? 0x8000 : 0);
    }
  auto encoded = EncodeA(a, pitch);
  Check(encoded.size() == ABytes);
  for (std::size_t r = 0; r < M; ++r)
    for (std::size_t c = 0; c < K; ++c) {
      const auto pos = Index(Layout::A, r, c / 8 * 8);
      Check(encoded[pos] == 128 &&
            encoded[pos + 1 + c % 8] == std::uint8_t(Pattern(r, c) * 32 & 255));
    }
  a.back() = 0x7c00;
  Reject([&] { EncodeA(a, pitch); });
  a.back() = 0x3c00;
  Reject([&] { EncodeA(std::span(a).first(a.size() - 1), pitch); });
  Reject([&] { EncodeA(a, K - 1); });
  std::vector<std::byte> b(Q8Bytes);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t k = 0; k < K; k += 32) {
      const auto block = (n * K + k) / 32 * 34;
      b[block + 1] = std::byte{0x3c};
      for (unsigned i = 0; i < 32; ++i)
        b[block + 2 + i] = std::byte(Pattern(k + i, n) & 255);
    }
  auto weights = EncodeB(b);
  Check(weights.size() == BBytes);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t k = 0; k < K; ++k) {
      const auto pos = Index(Layout::B, k / 8 * 8, n);
      Check(weights[pos] == 128 &&
            weights[pos + 1 + k % 8] == std::uint8_t(Pattern(k, n) * 32 & 255));
    }
  b.back() = std::byte{0};
  Check(EncodeB(b) != weights);
  Reject([&] { EncodeB(std::span(b).first(b.size() - 1)); });
  b[1] = std::byte{0x7c};
  Reject([&] { EncodeB(b); });
  std::cout << "PASS: full FP16 A/pitch and Q8 B bytes, signed dequantization, "
               "own storage, nonfinite/pitch/size refusal\n";
}
void Outputs() {
  std::vector<std::uint8_t> stream(CBytes);
  for (std::size_t r = 0; r < M; ++r)
    for (std::size_t c = 0; c < N; ++c) {
      const auto pos = Index(Layout::C, r, c / 8 * 8);
      stream[pos] = 127;
      stream[pos + 1 + c % 8] =
          std::uint8_t((int((r * 17 + c * 7) % 251) - 125) & 255);
    }
  for (auto cols : {1U, 7U, 9U, 767U, 768U, 1023U, 1024U}) {
    const std::size_t pitch = cols + 5, off = 1;
    const auto length = MatrixExtent(M, pitch, off, cols);
    std::vector<float> out(length + 9, 777.0F);
    DecodeC(stream, std::span(out).first(length), pitch, off, cols);
    for (std::size_t r = 0; r < M; ++r)
      for (std::size_t c = 0; c < cols; ++c)
        Check(out[r * pitch + off + c] ==
              float(int((r * 17 + c * 7) % 251) - 125) / 64);
    for (std::size_t i = 0; i < out.size(); ++i)
      if (i % pitch < off || i % pitch >= off + cols || i >= length)
        Check(out[i] == 777.0F);
    DecodeC(stream, std::span(out).first(length), pitch, off, cols, true);
    Check(out[off] == -250.0F / 64);
  }
  std::vector<float> out(M * N, 777.0F);
  stream.back() = 0;
  stream[CBytes - 9] = 255;
  Reject([&] { DecodeC(stream, out, N); });
  Check(
      std::all_of(out.begin(), out.end(), [](float v) { return v == 777.0F; }));
  stream[CBytes - 9] = 127;
  Reject([&] { DecodeC(stream, std::span(out).first(out.size() - 1), N); });
  Reject([&] { DecodeC(std::span(stream).first(CBytes - 1), out, N); });
  Reject([&] { DecodeC(stream, out, N, 0, 0); });
  Reject([&] { DecodeC(stream, out, N, 0, N + 1); });
  std::vector<float> padded(M * (N + 3));
  Reject([&] { DecodeC(stream, padded, N + 3, 0, N + 1); });
  Reject([&] { DecodeC(stream, out, N, 1, N); });
  Reject(
      [] { MatrixExtent(M, std::numeric_limits<std::size_t>::max(), 0, N); });
  std::cout << "PASS: full C independent row/column pattern, partial groups "
               "1/7/9/767/768/1023/1024, canaries/accumulate, transactional "
               "bad stream, exact-end bounds/overflow\n";
}
}  // namespace
int main() {
  Groups();
  Layouts();
  Operands();
  Outputs();
}
