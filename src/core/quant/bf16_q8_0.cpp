#include "src/core/quant/bf16_q8_0.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace gufo::quant {

bool QuantizeBf16Q8_0(const void* bf16, std::size_t rows, std::size_t cols,
                      std::span<block_q8_0> output, std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = "BF16 Q8_0: " + message;
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

bool QuantizeBf16Q8_0Threads(const void* bf16, std::size_t rows,
                             std::size_t cols, std::span<block_q8_0> output,
                             std::string* error, unsigned threads) {
  threads = static_cast<unsigned>(std::min<std::size_t>(
      std::max(threads, 1U), std::max<std::size_t>(rows, 1)));
  if (threads == 1 || bf16 == nullptr || cols % 32 != 0 ||
      output.size() != rows * cols / 32) {
    return QuantizeBf16Q8_0(bf16, rows, cols, output, error);
  }
  const std::size_t per_thread = (rows + threads - 1) / threads;
  std::vector<std::string> errors(threads);
  std::vector<char> failed(threads, 0);
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < threads; ++i) {
    const std::size_t first = std::size_t{i} * per_thread;
    if (first >= rows) {
      break;
    }
    const std::size_t count = std::min(per_thread, rows - first);
    pool.emplace_back([&, i, first, count] {
      failed[i] = !QuantizeBf16Q8_0(
          static_cast<const std::uint8_t*>(bf16) + first * cols * 2, count, cols,
          output.subspan(first * cols / 32, count * cols / 32), &errors[i]);
      if (failed[i]) {
        // The block number above counts from the start of this row range.
        errors[i] += " (row range starts at row " + std::to_string(first) + ")";
      }
    });
  }
  for (auto& thread : pool) {
    thread.join();
  }
  for (unsigned i = 0; i < threads; ++i) {
    if (failed[i]) {
      if (error != nullptr) {
        *error = errors[i];
      }
      return false;
    }
  }
  return true;
}

bool BuildBf16Q8_0Image(const void* bf16, std::size_t rows, std::size_t cols,
                        std::size_t tail_bytes, unsigned threads,
                        std::vector<std::uint8_t>& image, std::string* error) {
  if (bf16 == nullptr || rows == 0 || cols == 0 || cols % 32 != 0 ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) /
                 cols) {
    if (error != nullptr) {
      *error = "BF16 Q8_0 image: requires nonempty rows aligned to 32";
    }
    return false;
  }
  const std::size_t blocks = rows * cols / 32;
  image.resize(blocks * sizeof(block_q8_0) + tail_bytes);
  // block_q8_0 is a 34-byte trivially copyable struct with alignment 2; the
  // vector storage is at least 16-byte aligned.
  const std::span<block_q8_0> output(
      reinterpret_cast<block_q8_0*>(image.data()), blocks);
  if (!QuantizeBf16Q8_0Threads(bf16, rows, cols, output, error, threads)) {
    return false;
  }
  if (tail_bytes != 0) {
    std::memset(image.data() + blocks * sizeof(block_q8_0), 0, tail_bytes);
  }
  return true;
}

bool BuildBf16F32Image(const void* bf16, std::size_t count,
                       std::size_t tail_bytes, std::vector<std::uint8_t>& image,
                       std::string* error) {
  if (bf16 == nullptr || count == 0 ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) / 2) {
    if (error != nullptr) {
      *error = "BF16 F32 image: requires a nonempty source";
    }
    return false;
  }
  image.resize(count * sizeof(float) + tail_bytes);
  const auto* source = static_cast<const std::uint8_t*>(bf16);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t bits;
    std::memcpy(&bits, source + i * 2, 2);
    const float value = std::bit_cast<float>(std::uint32_t{bits} << 16);
    std::memcpy(image.data() + i * sizeof(float), &value, sizeof(float));
  }
  if (tail_bytes != 0) {
    std::memset(image.data() + count * sizeof(float), 0, tail_bytes);
  }
  return true;
}

}  // namespace gufo::quant
