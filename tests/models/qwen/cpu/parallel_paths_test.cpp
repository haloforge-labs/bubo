#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/ssm.hpp"

namespace {
using gufo::core::GgmlType;
using gufo::models::QwenTensorRef;
void Check(bool condition) {
  if (!condition)
    throw std::runtime_error(
        "Qwen parallel CPU result differs from independent expectation");
}
QwenTensorRef Tensor(const std::vector<float>& values) {
  return {.data = values.data(),
          .type = GgmlType::kF32,
          .num_elements = values.size()};
}

void TestGemm() {
  constexpr std::size_t rows = 257, columns = 32;
  std::mt19937 rng(81);
  std::uniform_int_distribution<int> dist(-8, 8);
  std::vector<float> input(columns), weights(rows * columns), expected(rows),
      output(rows);
  for (auto& value : input)
    value = static_cast<float>(dist(rng)) / 4;
  for (auto& value : weights)
    value = static_cast<float>(dist(rng)) / 4;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < columns; ++col)
      expected[row] += weights[row * columns + col] * input[col];
  gufo::models::TensorGEMV(Tensor(weights), input, rows, columns, output);
  Check(output == expected);

  for (const auto type : {GgmlType::kBF16, GgmlType::kF16}) {
    const std::uint16_t one = type == GgmlType::kBF16 ? 0x3f80 : 0x3c00;
    std::vector<std::uint16_t> encoded(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
      expected[row] = 0;
      for (std::size_t col = 0; col < columns; ++col) {
        const bool negative = (row + col) % 3 == 0;
        encoded[row * columns + col] = one | (negative ? 0x8000 : 0);
        expected[row] += (negative ? -1.0F : 1.0F) * input[col];
      }
    }
    gufo::models::TensorGEMV(
        {.data = encoded.data(), .type = type, .num_elements = encoded.size()},
        input, rows, columns, output);
    Check(output == expected);
  }
  std::vector<gufo::quant::block_q8_0> encoded(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    encoded[row].d = 0x3c00;  // Independent fixture: FP16 1.0.
    expected[row] = 0;
    for (std::size_t col = 0; col < columns; ++col) {
      const auto value = static_cast<std::int8_t>(dist(rng));
      encoded[row].qs[col] = value;
      expected[row] += static_cast<float>(value) * input[col];
    }
  }
  gufo::models::TensorGEMV({.data = encoded.data(),
                            .type = GgmlType::kQ8_0,
                            .num_elements = rows * columns},
                           input, rows, columns, output);
  Check(output == expected);
}

void TestSsm() {
  // Scalar head equations give an independent analytic expectation, with 32
  // distinct heads and 96 channels to exercise all three parallel loops.
  constexpr std::uint32_t heads = 32;
  std::vector<float> qkv(3 * heads, 0.5F), gate(heads, 1.0F);
  std::vector<float> conv(6 * heads), projection(heads, 1.0F);
  for (std::size_t c = 0; c < 3 * heads; ++c) {
    conv[2 * c] = 1.0F;
    conv[2 * c + 1] = 2.0F;
  }
  gufo::models::QwenSsmParameters parameters{.qkv = Tensor(qkv),
                                             .gate = Tensor(gate),
                                             .a = {},
                                             .dt = {},
                                             .alpha = {},
                                             .beta = {},
                                             .norm = {},
                                             .output = Tensor(projection),
                                             .conv1d = Tensor(conv),
                                             .key_head_count = heads,
                                             .value_head_count = heads,
                                             .key_dim = 1,
                                             .val_dim = 1,
                                             .conv_kernel = 2};
  gufo::models::QwenSsmCache cache(1, 3 * heads, 2, heads, 1, 1);
  auto state = cache.GetConvState(0);
  for (std::size_t c = 0; c < 3 * heads; ++c)
    state[2 * c + 1] = 0.2F;
  std::vector<float> head_state(heads);
  for (std::uint32_t h = 0; h < heads; ++h) {
    head_state[h] = 0.01F * static_cast<float>(h + 1);
    cache.GetDeltaNetState(0, h)[0] = head_state[h];
  }
  std::vector<float> input{1.0F}, qkv_scratch(3 * heads), gate_scratch(heads),
      scratch(heads), output(1);
  for (unsigned step = 0; step < 2; ++step) {
    const float convolution = step == 0 ? 1.2F : 1.5F;
    const float value = convolution / (1.0F + std::exp(-convolution));
    const float normalized = value / std::sqrt(value * value + 1e-6F);
    const float decay = std::exp(std::log(2.0F) * -0.05F);
    const float gating = 1.0F / (1.0F + std::exp(-1.0F));
    float expected_sum = 0;
    for (auto& s : head_state) {
      const float decayed = s * decay;
      s = decayed + 0.5F * (value - decayed * normalized) * normalized;
      expected_sum += s * normalized * gating;
    }
    gufo::models::ForwardSSM(input, parameters, cache, 0, qkv_scratch,
                             gate_scratch, scratch, output);
    Check(std::abs(output[0] - expected_sum) < 2e-5F);
    for (std::uint32_t h = 0; h < heads; ++h)
      Check(std::abs(cache.GetDeltaNetState(0, h)[0] - head_state[h]) < 1e-6F);
    for (std::size_t c = 0; c < 3 * heads; ++c) {
      Check(state[2 * c] == (step == 0 ? 0.2F : 0.5F));
      Check(state[2 * c + 1] == 0.5F);
    }
  }
}
}  // namespace

int main() {
  TestGemm();
  TestSsm();
  std::cout << "PASS: independent F32/BF16/F16/Q8 GEMV and two-step "
               "convolution/DeltaNet/gating expectations\n";
}
