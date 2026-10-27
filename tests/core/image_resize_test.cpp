// core::ResizeBicubic must equal Pillow's Image.resize(..., BICUBIC) byte for byte (fixtures from tools/vision/make_resize_fixtures.py).
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "tests/core/image_resize_fixtures.hpp"

namespace {

std::vector<std::uint8_t> Pixels(std::uint32_t width, std::uint32_t height, std::uint32_t seed) {
  std::uint32_t state = (0x9E3779B9U * seed) & 0xFFFFFFFFU;
  if (state == 0) {
    state = 1;
  }
  std::vector<std::uint8_t> out(std::size_t{width} * height * 3);
  for (auto& byte : out) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    byte = static_cast<std::uint8_t>((state >> 8) & 0xFF);
  }
  return out;
}

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

}  // namespace

int main() {
  for (const auto& c : resize_fixtures::Cases()) {
    const std::string label = std::to_string(c.in_w) + "x" + std::to_string(c.in_h) + " -> " + std::to_string(c.out_w) + "x" + std::to_string(c.out_h);
    const gufo::core::Image input{c.in_w, c.in_h, Pixels(c.in_w, c.in_h, c.seed)};
    const auto out = gufo::core::ResizeBicubic(input, c.out_w, c.out_h);
    Expect(out.width == c.out_w && out.height == c.out_h, label + ": size");
    Expect(out.pixels.size() == c.expected.size(), label + ": byte count");
    if (out.pixels.size() == c.expected.size()) {
      std::size_t different = 0, first = 0;
      for (std::size_t i = 0; i < out.pixels.size(); ++i) {
        if (out.pixels[i] != c.expected[i]) {
          if (different++ == 0) {
            first = i;
          }
        }
      }
      Expect(different == 0, label + ": " + std::to_string(different) + " bytes differ from Pillow (first at " + std::to_string(first) + ")");
    }
  }
  // Invalid arguments are refused, not resized.
  const gufo::core::Image tiny{2, 2, std::vector<std::uint8_t>(12, 7)};
  bool threw = false;
  try {
    (void)gufo::core::ResizeBicubic(tiny, 0, 4);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "zero output width is refused");
  threw = false;
  try {
    (void)gufo::core::ResizeBicubic(gufo::core::Image{2, 2, std::vector<std::uint8_t>(5)}, 4, 4);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "pixel buffer of the wrong size is refused");
  std::cout << (failures == 0 ? "image_resize_test: OK" : "image_resize_test: FAILED") << "\n";
  return failures == 0 ? 0 : 1;
}
