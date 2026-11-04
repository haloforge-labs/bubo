#include "src/models/qwen38_flash_next/kv_cache.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace gufo::models::qwen38_flash_next {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

bool IsFiniteHalf(std::uint16_t bits) {
  return (bits & 0x7c00) != 0x7c00;
}

std::uint16_t ReadHalf(const std::byte* bytes) {
  return std::to_integer<std::uint16_t>(bytes[0]) |
         (std::to_integer<std::uint16_t>(bytes[1]) << 8);
}

void WriteHalf(std::byte* bytes, std::uint16_t bits) {
  bytes[0] = std::byte(bits & 0xff);
  bytes[1] = std::byte(bits >> 8);
}

std::uint32_t CheckedElements(std::size_t elements) {
  if (elements > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Flash-Next KV row is too wide");
  return static_cast<std::uint32_t>(elements);
}

void RejectOverlap(std::span<const std::byte> input,
                   std::span<const std::byte> output) {
  if (input.empty() || output.empty())
    return;
  const auto a = reinterpret_cast<std::uintptr_t>(input.data());
  const auto b = reinterpret_cast<std::uintptr_t>(output.data());
  if (a <= b ? b - a < input.size() : a - b < output.size())
    throw std::invalid_argument("overlapping Flash-Next KV buffers");
}

KvBlockQ8_0 QuantizeBlock(const float* input) {
  float amax = 0;
  for (std::size_t i = 0; i < kKvQ8BlockElements; ++i) {
    if (!std::isfinite(input[i]))
      throw std::invalid_argument("nonfinite Flash-Next KV input");
    amax = std::max(amax, std::abs(input[i]));
  }
  const float d = amax / 127.0f;
  KvBlockQ8_0 block{};
  if (d == 0)
    return block;
  block.scale = KvFloatToHalfBits(d);
  if (!IsFiniteHalf(block.scale))
    throw std::invalid_argument("Flash-Next KV scale overflows F16");
  const float stored_scale = KvHalfBitsToFloat(block.scale);
  for (std::size_t i = 0; i < kKvQ8BlockElements; ++i) {
    // Division by the unrounded F32 scale is intentional: multiplying by an
    // approximate reciprocal or dividing by stored_scale changes tie bytes.
    const auto q = RoundKvQ8Code(input[i] / d);
    block.values[i] = q;
    if (!IsFiniteHalf(KvFloatToHalfBits(q * stored_scale)))
      throw std::invalid_argument("Flash-Next KV reconstruction overflows F16");
  }
  return block;
}
}  // namespace

std::int8_t RoundKvQ8Code(float scaled) {
  if (!std::isfinite(scaled))
    throw std::invalid_argument("nonfinite Flash-Next scaled KV value");
  return static_cast<std::int8_t>(
      std::clamp(std::round(scaled), -127.0f, 127.0f));
}

std::uint16_t KvFloatToHalfBits(float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
  const auto exponent = (bits >> 23) & 0xff;
  const auto fraction = bits & 0x7fffff;
  if (exponent == 0xff)
    return sign | (fraction == 0 ? 0x7c00 : 0x7e00);
  if (exponent < 102)
    return sign;
  if (exponent > 142)
    return sign | 0x7c00;

  if (exponent < 113) {
    // Include the hidden bit, then round once into the half subnormal field.
    // exponent=102 includes exactly the tie at half the smallest subnormal.
    const auto significand = fraction | 0x800000;
    const auto shift = 126 - exponent;
    const auto truncated = significand >> shift;
    const auto remainder = significand & ((std::uint32_t{1} << shift) - 1);
    const auto halfway = std::uint32_t{1} << (shift - 1);
    return sign | static_cast<std::uint16_t>(
                      truncated + (remainder > halfway ||
                                   (remainder == halfway && (truncated & 1))));
  }
  auto magnitude = ((exponent - 112) << 10) | (fraction >> 13);
  const auto remainder = fraction & 0x1fff;
  magnitude += remainder > 0x1000 || (remainder == 0x1000 && (magnitude & 1));
  return sign | static_cast<std::uint16_t>(magnitude);
}

float KvHalfBitsToFloat(std::uint16_t bits) noexcept {
  const auto sign = std::uint32_t{bits & 0x8000U} << 16;
  const auto exponent = (bits >> 10) & 0x1f;
  auto fraction = std::uint32_t{bits & 0x3ffU};
  std::uint32_t magnitude = 0;
  if (exponent == 0) {
    if (fraction != 0) {
      std::uint32_t e = 113;
      while ((fraction & 0x400) == 0) {
        fraction <<= 1;
        --e;
      }
      magnitude = (e << 23) | ((fraction & 0x3ff) << 13);
    }
  } else if (exponent == 31) {
    magnitude = 0x7f800000 | (fraction == 0 ? 0 : 0x400000);
  } else {
    magnitude = ((exponent + 112) << 23) | (fraction << 13);
  }
  return std::bit_cast<float>(sign | magnitude);
}

void EncodeKvRow(std::span<const float> input, KvCacheType type,
                 std::span<std::byte> output) {
  const auto bytes = KvRowBytes(CheckedElements(input.size()), type);
  if (output.size() != bytes)
    throw std::invalid_argument("incorrect Flash-Next KV output size");
  RejectOverlap(std::as_bytes(input), output);
  // Two passes give fail-closed, whole-row validation without a row-sized
  // temporary allocation. No other block or companion can affect a scale.
  if (type == KvCacheType::kF16) {
    for (const auto x : input) {
      if (!std::isfinite(x) || !IsFiniteHalf(KvFloatToHalfBits(x)))
        throw std::invalid_argument("Flash-Next KV input is not finite F16");
    }
    for (std::size_t i = 0; i < input.size(); ++i)
      WriteHalf(output.data() + i * 2, KvFloatToHalfBits(input[i]));
  } else {
    for (std::size_t i = 0; i < input.size(); i += kKvQ8BlockElements)
      (void)QuantizeBlock(input.data() + i);
    for (std::size_t i = 0; i < input.size(); i += kKvQ8BlockElements) {
      const auto block = QuantizeBlock(input.data() + i);
      auto* dst =
          output.data() + (i / kKvQ8BlockElements) * sizeof(KvBlockQ8_0);
      WriteHalf(dst, block.scale);
      for (std::size_t j = 0; j < kKvQ8BlockElements; ++j)
        dst[2 + j] = std::byte(static_cast<std::uint8_t>(block.values[j]));
    }
  }
}

void ValidateKvRow(std::span<const std::byte> input, std::uint32_t elements,
                   KvCacheType type) {
  if (input.size() != KvRowBytes(elements, type))
    throw std::invalid_argument("incorrect Flash-Next KV input size");
  if (type == KvCacheType::kF16) {
    for (std::size_t i = 0; i < input.size(); i += 2) {
      if (!IsFiniteHalf(ReadHalf(input.data() + i)))
        throw std::invalid_argument("nonfinite Flash-Next F16 KV payload");
    }
  } else {
    for (std::size_t i = 0; i < input.size(); i += sizeof(KvBlockQ8_0)) {
      const auto scale_bits = ReadHalf(input.data() + i);
      if ((scale_bits & 0x8000) != 0 || !IsFiniteHalf(scale_bits))
        throw std::invalid_argument("invalid Flash-Next KV scale");
      const float scale = KvHalfBitsToFloat(scale_bits);
      int largest = 0;
      for (std::size_t j = 0; j < kKvQ8BlockElements; ++j) {
        const auto q = std::bit_cast<std::int8_t>(
            std::to_integer<std::uint8_t>(input[i + 2 + j]));
        if (q == -128)
          throw std::invalid_argument("invalid Flash-Next KV reconstruction");
        largest = std::max(largest, std::abs(static_cast<int>(q)));
      }
      // One overflow check per block; finite nonnegative scales make the
      // maximum code magnitude sufficient for every reconstructed operand.
      if (!IsFiniteHalf(KvFloatToHalfBits(largest * scale)))
        throw std::invalid_argument("invalid Flash-Next KV reconstruction");
    }
  }
}

void DecodeKvRow(std::span<const std::byte> input, KvCacheType type,
                 std::span<float> output) {
  ValidateKvRow(input, CheckedElements(output.size()), type);
  RejectOverlap(input, std::as_bytes(output));
  if (type == KvCacheType::kF16) {
    for (std::size_t i = 0; i < output.size(); ++i)
      output[i] = KvHalfBitsToFloat(ReadHalf(input.data() + i * 2));
  } else {
    for (std::size_t i = 0; i < output.size(); i += kKvQ8BlockElements) {
      const auto* src =
          input.data() + (i / kKvQ8BlockElements) * sizeof(KvBlockQ8_0);
      const float scale = KvHalfBitsToFloat(ReadHalf(src));
      for (std::size_t j = 0; j < kKvQ8BlockElements; ++j) {
        const auto q = std::bit_cast<std::int8_t>(
            std::to_integer<std::uint8_t>(src[2 + j]));
        output[i + j] = KvHalfBitsToFloat(KvFloatToHalfBits(q * scale));
      }
    }
  }
}

}  // namespace gufo::models::qwen38_flash_next
