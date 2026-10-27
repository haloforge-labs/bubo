#ifndef GUFO_MODELS_GEMMA4_DRAFT_VOCAB_QUANT_HPP_
#define GUFO_MODELS_GEMMA4_DRAFT_VOCAB_QUANT_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::gemma4 {

/// Row-major BF16 -> canonical GGML Q8_0, 32 values per block. Uses only a
/// block's F32 values on the stack; callers own the bounded output buffer.
/// d = amax / 127 rounded to binary16 (nearest even); codes round x / d
/// (using the unrounded d) to nearest, halfway away from zero. Rejects invalid
/// shapes, nonfinite weights and binary16 scale overflow without fallback.
[[nodiscard]] bool QuantizeDraftVocabQ8_0(const void* bf16, std::size_t rows,
                                          std::size_t cols,
                                          std::span<quant::block_q8_0> output,
                                          std::string* error);

/// Checks the actual binary16 d/dmin after Q4_K repacking. Callers can scan
/// bounded chunks and supply the tensor name and first super-block index.
[[nodiscard]] bool ValidateDraftVocabQ4KScales(
    std::span<const quant::block_q4_K> blocks, std::string* error,
    std::string_view tensor_name = {}, std::size_t first_block = 0);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_DRAFT_VOCAB_QUANT_HPP_
