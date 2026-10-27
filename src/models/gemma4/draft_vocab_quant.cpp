#include "src/models/gemma4/draft_vocab_quant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace gufo::models::gemma4 {

bool QuantizeDraftVocabQ8_0(const void* bf16, std::size_t rows,
                            std::size_t cols,
                            std::span<quant::block_q8_0> output,
                            std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = "BF16 draft vocabulary Q8_0: " + message;
    }
    return false;
  };
  if (bf16 == nullptr || rows == 0 || cols == 0 || cols % 32 != 0 ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) /
                 cols ||
      output.size() != rows * cols / 32) {
    return fail("requires nonempty rows aligned to 32 and exact output size");
  }
  const auto* source = static_cast<const std::uint8_t*>(bf16);
  for (std::size_t b = 0; b < output.size(); ++b) {
    float values[32];
    float amax = 0.0F;
    for (std::size_t i = 0; i < 32; ++i) {
      std::uint16_t bits;
      std::memcpy(&bits, source + (b * 32 + i) * 2, 2);
      values[i] = std::bit_cast<float>(std::uint32_t{bits} << 16);
      if (!std::isfinite(values[i])) {
        return fail("nonfinite weight in block " + std::to_string(b));
      }
      amax = std::max(amax, std::abs(values[i]));
    }
    // Same GGML Q8_0 convention as the host DFlash quantizer. Do not round
    // the scale before choosing codes; the encoded scale rounds separately.
    const float d = amax / 127.0F;
    auto& block = output[b];
    block.d = std::bit_cast<std::uint16_t>(static_cast<_Float16>(d));
    if ((block.d & 0x7c00U) == 0x7c00U) {
      return fail("binary16 scale overflow in block " + std::to_string(b));
    }
    const float inverse = d == 0.0F ? 0.0F : 1.0F / d;
    for (std::size_t i = 0; i < 32; ++i) {
      // BF16 subnormals can have a finite scale with an infinite reciprocal.
      const float level =
          std::isfinite(inverse) ? values[i] * inverse : values[i] / d;
      block.qs[i] =
          static_cast<std::int8_t>(std::clamp(std::lround(level), -127L, 127L));
    }
  }
  return true;
}

bool ValidateDraftVocabQ4KScales(std::span<const quant::block_q4_K> blocks,
                                 std::string* error,
                                 std::string_view tensor_name,
                                 std::size_t first_block) {
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const bool bad_d = (blocks[b].d & 0x7c00U) == 0x7c00U;
    const bool bad_dmin = (blocks[b].dmin & 0x7c00U) == 0x7c00U;
    if (bad_d || bad_dmin) {
      if (error != nullptr) {
        *error = std::string("BF16 draft vocabulary Q4_K: nonfinite ") +
                 (bad_d ? "d" : "dmin") + " in super-block " +
                 std::to_string(first_block + b);
        if (!tensor_name.empty()) {
          *error = "tensor " + std::string(tensor_name) + ": " + *error;
        }
      }
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::gemma4
