#include "src/models/gemma4/draft_vocab_quant.hpp"

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "src/core/quant/bf16_q8_0.hpp"

namespace gufo::models::gemma4 {

bool QuantizeDraftVocabQ8_0(const void* bf16, std::size_t rows,
                            std::size_t cols,
                            std::span<quant::block_q8_0> output,
                            std::string* error) {
  // The quantiser lives in src/core/quant so the Qwen hyper-connection
  // narrowing shares it; only the error context stays here.
  if (quant::QuantizeBf16Q8_0(bf16, rows, cols, output, error)) {
    return true;
  }
  if (error != nullptr) {
    *error = "draft vocabulary " + *error;
  }
  return false;
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
