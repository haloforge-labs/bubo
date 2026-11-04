#ifndef GUFO_CORE_QUANT_BF16_Q8_0_HPP_
#define GUFO_CORE_QUANT_BF16_Q8_0_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::quant {

/// Row-major BF16 -> canonical GGML Q8_0, 32 values per block. Uses only a
/// block's F32 values on the stack; callers own the bounded output buffer.
/// d = amax / 127 rounded to binary16 (nearest even); codes round x / d
/// (using the unrounded d) to nearest, halfway away from zero. Rejects invalid
/// shapes, nonfinite weights and binary16 scale overflow without fallback.
/// The one BF16 -> Q8_0 quantiser: the Gemma 4 draft vocabulary and the
/// Qwen hyper-connection load-time narrowing both call it.
[[nodiscard]] bool QuantizeBf16Q8_0(const void* bf16, std::size_t rows,
                                    std::size_t cols,
                                    std::span<block_q8_0> output,
                                    std::string* error);

/// The same quantiser over row ranges on up to `threads` threads (block
/// results do not depend on the split, so the output equals the serial one).
/// The first failing range's error is reported.
[[nodiscard]] bool QuantizeBf16Q8_0Threads(const void* bf16, std::size_t rows,
                                           std::size_t cols,
                                           std::span<block_q8_0> output,
                                           std::string* error,
                                           unsigned threads);

/// The complete device image of a BF16 matrix as Q8_0: the quantised blocks
/// followed by `tail_bytes` zero bytes, so ONE synchronous copy puts the tensor
/// and its zero tail on the device (no separate asynchronous memset whose
/// completion order against later allocations has to be reasoned about).
/// `image` is resized to blocks * 34 + tail_bytes; every byte of it is written
/// (a previous content is neither read nor kept), which the stress test checks
/// with a poisoned buffer.
[[nodiscard]] bool BuildBf16Q8_0Image(const void* bf16, std::size_t rows,
                                      std::size_t cols, std::size_t tail_bytes,
                                      unsigned threads,
                                      std::vector<std::uint8_t>& image,
                                      std::string* error);

/// The same for an exact BF16 -> F32 widening (count values, then the zero tail).
[[nodiscard]] bool BuildBf16F32Image(const void* bf16, std::size_t count,
                                     std::size_t tail_bytes,
                                     std::vector<std::uint8_t>& image,
                                     std::string* error);

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_BF16_Q8_0_HPP_
