// Copyright (c) 2023-2026 The ggml authors
// CPU oracle tests for the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and Q2_0 dequantizers: the expected values are produced by the gguf-py
// reference (Q2_0: the llama.cpp bec4772f formula) in gen_iq_dequant_vectors.py.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/iq_grids.hpp"

namespace {

#include "tests/core/ggml_iq_dequant_vectors.inc"

// Fixed 56-byte block and independent binary32 decoder output. Both delta
// signs, grid endpoints and packed per-16 scales are represented.
constexpr std::uint8_t kIq1MBytes[] = {
    0,   255, 85, 122, 159, 196, 233, 14,  51,  88,  125, 162, 199, 236,
    17,  54,  91, 128, 165, 202, 239, 20,  57,  94,  131, 168, 205, 242,
    23,  60,  97, 134, 240, 33,  82,  131, 180, 229, 22,  71,  120, 169,
    218, 11,  60, 109, 158, 207, 35,  1,   86,  4,   137, 167, 188, 58,
};
constexpr float kIq1MExpected[] = {
    -4.59375F,  -4.59375F, -4.59375F, -4.59375F,  -4.59375F,  -4.59375F,
    -4.59375F,  -4.59375F, 4.59375F,  4.59375F,   4.59375F,   4.59375F,
    4.59375F,   4.59375F,  4.59375F,  4.59375F,   0.84375F,   0.84375F,
    7.59375F,   7.59375F,  0.84375F,  0.84375F,   0.84375F,   -5.90625F,
    7.59375F,   -5.90625F, 0.84375F,  0.84375F,   -5.90625F,  -5.90625F,
    -5.90625F,  0.84375F,  -5.90625F, 0.84375F,   -5.90625F,  7.59375F,
    0.84375F,   -5.90625F, -5.90625F, 0.84375F,   -5.90625F,  -5.90625F,
    -5.90625F,  -5.90625F, 7.59375F,  -5.90625F,  -5.90625F,  7.59375F,
    0.09375F,   0.09375F,  -0.65625F, -0.65625F,  0.09375F,   0.09375F,
    0.09375F,   0.09375F,  0.65625F,  -0.09375F,  -0.09375F,  -0.09375F,
    -0.84375F,  -0.84375F, -0.84375F, -0.84375F,  1.21875F,   1.21875F,
    10.96875F,  -8.53125F, 10.96875F, 1.21875F,   1.21875F,   1.21875F,
    -1.21875F,  -1.21875F, 8.53125F,  8.53125F,   8.53125F,   8.53125F,
    -10.96875F, -1.21875F, 0.46875F,  0.46875F,   0.46875F,   4.21875F,
    0.46875F,   4.21875F,  4.21875F,  0.46875F,   -4.21875F,  -4.21875F,
    -0.46875F,  -4.21875F, -4.21875F, -0.46875F,  -0.46875F,  3.28125F,
    2.53125F,   2.53125F,  2.53125F,  -1.96875F,  0.28125F,   0.28125F,
    0.28125F,   2.53125F,  -1.96875F, 2.53125F,   2.53125F,   -1.96875F,
    2.53125F,   -1.96875F, 2.53125F,  -1.96875F,  -3.28125F,  0.46875F,
    4.21875F,   0.46875F,  -3.28125F, 4.21875F,   0.46875F,   4.21875F,
    0.46875F,   4.21875F,  4.21875F,  -3.28125F,  4.21875F,   0.46875F,
    0.46875F,   0.46875F,  -2.53125F, 1.96875F,   -0.28125F,  -2.53125F,
    -0.28125F,  -0.28125F, -2.53125F, -2.53125F,  0.28125F,   0.28125F,
    2.53125F,   0.28125F,  2.53125F,  -1.96875F,  2.53125F,   2.53125F,
    -0.28125F,  1.96875F,  1.96875F,  1.96875F,   -0.28125F,  1.96875F,
    -0.28125F,  -2.53125F, -0.28125F, -0.28125F,  -0.28125F,  1.96875F,
    -2.53125F,  -0.28125F, -2.53125F, -0.28125F,  -1.21875F,  -10.96875F,
    -1.21875F,  8.53125F,  -1.21875F, -1.21875F,  -10.96875F, -1.21875F,
    -10.96875F, -1.21875F, 8.53125F,  8.53125F,   -10.96875F, -1.21875F,
    8.53125F,   -1.21875F, -0.65625F, -5.90625F,  -0.65625F,  -0.65625F,
    -0.65625F,  4.59375F,  -5.90625F, -0.65625F,  -4.59375F,  -4.59375F,
    -4.59375F,  0.65625F,  0.65625F,  0.65625F,   -4.59375F,  -4.59375F,
    5.90625F,   -0.84375F, -7.59375F, -0.84375F,  -0.84375F,  5.90625F,
    -0.84375F,  -0.84375F, -5.90625F, 0.84375F,   7.59375F,   -5.90625F,
    7.59375F,   -5.90625F, 0.84375F,  0.84375F,   -1.40625F,  -12.65625F,
    -1.40625F,  -1.40625F, 9.84375F,  -12.65625F, -12.65625F, 9.84375F,
    1.40625F,   1.40625F,  -9.84375F, 1.40625F,   12.65625F,  1.40625F,
    1.40625F,   12.65625F, -4.21875F, -4.21875F,  -4.21875F,  -4.21875F,
    -4.21875F,  3.28125F,  -4.21875F, 3.28125F,   -4.21875F,  -0.46875F,
    -0.46875F,  -0.46875F, -0.46875F, -0.46875F,  -0.46875F,  -4.21875F,
    -1.03125F,  7.21875F,  -9.28125F, -9.28125F,  -1.03125F,  -9.28125F,
    7.21875F,   7.21875F,  -1.03125F, -9.28125F,  -1.03125F,  -1.03125F,
    -1.03125F,  7.21875F,  -1.03125F, -1.03125F,
};

void Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

using DequantFn = void (*)(const void*, float*, std::size_t);
using DotFn = float (*)(const void*, std::span<const float>, std::size_t);

void CheckType(std::string_view name, gufo::core::GgmlType type, std::size_t block_bytes, std::size_t block_elements,
               std::span<const std::uint8_t> bytes, std::span<const float> expected, DequantFn dequantize, DotFn dot, bool exact = false) {
  using gufo::quant::EncodedSizeBytes;
  using gufo::quant::QuantizedBlockElements;
  const std::size_t blocks = bytes.size() / block_bytes;
  const std::size_t elements = blocks * block_elements;
  Require(bytes.size() == blocks * block_bytes, std::string(name) + ": vector size is a whole number of blocks");
  Require(expected.size() == elements, std::string(name) + ": expected size");
  Require(QuantizedBlockElements(type) == block_elements, std::string(name) + ": block elements");
  Require(EncodedSizeBytes(type, elements) == bytes.size(), std::string(name) + ": encoded size");
  Require(EncodedSizeBytes(type, elements + 1) == 0, std::string(name) + ": misaligned element count is rejected");
  std::vector<float> actual(elements);
  dequantize(bytes.data(), actual.data(), elements);
  float max_abs = 0.0F;
  for (const float v : expected)
    max_abs = std::fmax(max_abs, std::fabs(v));
  Require(max_abs > 0.0F, std::string(name) + ": the vector is not all zero");
  for (std::size_t i = 0; i < elements; ++i) {
    if (exact ? std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0 : std::fabs(actual[i] - expected[i]) > 1e-6F * max_abs) {
      std::cerr << "FAIL: " << name << " element " << i << ": " << actual[i] << " vs " << expected[i] << '\n';
      std::exit(1);
    }
  }
  std::vector<float> input(elements);
  double reference = 0.0;
  for (std::size_t i = 0; i < elements; ++i) {
    input[i] = 0.25F + 0.01F * static_cast<float>(i % 17);
    reference += static_cast<double>(expected[i]) * static_cast<double>(input[i]);
  }
  const float got = dot(bytes.data(), input, elements);
  Require(std::fabs(static_cast<double>(got) - reference) <= 1e-4 * (std::fabs(reference) + max_abs), std::string(name) + ": dot product");
}

void TestTables() {
  using namespace gufo::quant;
  // Spot values of the verbatim ggml tables (first and last entries).
  Require(SignsIq2xs()[0] == 0 && SignsIq2xs()[1] == 129 && SignsIq2xs()[127] == 255, "ksigns_iq2xs");
  Require(Iq2xxsGrid()[0] == 0x0808080808080808ULL, "iq2xxs_grid first entry");
  Require(Iq3xxsGrid()[0] == 0x04040404U, "iq3xxs_grid first entry");
  // Every sign pattern's eighth bit is the parity of the low seven bits.
  for (int i = 0; i < 128; ++i) {
    const std::uint8_t s = SignsIq2xs()[i];
    int parity = 0;
    for (int b = 0; b < 7; ++b)
      parity ^= (s >> b) & 1;
    Require(((s & 0x7F) == i) && (((s >> 7) & 1) == parity), "ksigns_iq2xs parity layout");
  }
}

void TestTypeNames() {
  using gufo::core::GgmlType;
  using gufo::core::ToString;
  Require(static_cast<int>(GgmlType::kIQ2_XS) == 17 && static_cast<int>(GgmlType::kIQ3_XXS) == 18 && static_cast<int>(GgmlType::kIQ2_S) == 22 &&
              static_cast<int>(GgmlType::kQ2_0) == 42,
          "GGUF type ids");
  Require(ToString(GgmlType::kQ2_0) == "Q2_0" && ToString(GgmlType::kIQ3_XXS) == "IQ3_XXS" && ToString(GgmlType::kIQ2_S) == "IQ2_S" &&
              ToString(GgmlType::kIQ2_XS) == "IQ2_XS",
          "type names");
}

}  // namespace

int main() {
  using gufo::core::GgmlType;
  using namespace gufo::quant;
  TestTypeNames();
  TestTables();
  CheckType("IQ2_XXS", GgmlType::kIQ2_XXS, 66, 256, kIq2XxsBytes, kIq2XxsExpected, DequantizeIQ2_XXS, DotProductIQ2_XXS);
  CheckType("IQ2_XS", GgmlType::kIQ2_XS, 74, 256, kIq2XsBytes, kIq2XsExpected, DequantizeIQ2_XS, DotProductIQ2_XS);
  CheckType("IQ2_S", GgmlType::kIQ2_S, 82, 256, kIq2SBytes, kIq2SExpected, DequantizeIQ2_S, DotProductIQ2_S);
  CheckType("IQ1_M", GgmlType::kIQ1_M, 56, 256, kIq1MBytes, kIq1MExpected, DequantizeIQ1_M, DotProductIQ1_M, true);
  CheckType("IQ3_XXS", GgmlType::kIQ3_XXS, 98, 256, kIq3XxsBytes, kIq3XxsExpected, DequantizeIQ3_XXS, DotProductIQ3_XXS);
  CheckType("Q2_0", GgmlType::kQ2_0, 18, 64, kQ20Bytes, kQ20Expected, DequantizeQ2_0, DotProductQ2_0);
  std::cout << "ggml_iq_dequant_test passed\n";
  return 0;
}
