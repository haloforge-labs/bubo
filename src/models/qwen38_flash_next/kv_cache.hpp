#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace gufo::models::qwen38_flash_next {

enum class KvCacheType : std::uint32_t {
  kF16 = 0,
  kQ8_0 = 1,
  kQ8_0Rotated = 2
};

// Combined snapshot and persistence ABI: integration's version 20 includes
// draft cooldown and prefill-tail history; packed KV adds an explicit dtype.
// Reject both historical integration and PR layouts rather than reusing them.
inline constexpr std::uint32_t kKvCacheNumericsEpoch = 21;
inline constexpr std::uint32_t kKvQ8BlockElements = 32;

// Equal row widths do not make rotated and unrotated state interchangeable.
inline constexpr std::uint32_t kKvRotatedNumericsEpoch = 22;
static_assert(kKvRotatedNumericsEpoch > kKvCacheNumericsEpoch);
[[nodiscard]] constexpr bool IsKnownKvCacheType(KvCacheType type) noexcept {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool KvUsesQ8Codec(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return false;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      return true;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}

[[nodiscard]] constexpr bool KvIsRotated(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
      return false;
    case KvCacheType::kQ8_0Rotated:
      return true;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}

[[nodiscard]] constexpr std::uint32_t ExpectedKvPayloadVersion(
    KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
      return kKvCacheNumericsEpoch;
    case KvCacheType::kQ8_0Rotated:
      return kKvRotatedNumericsEpoch;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}

// Standard q8_0: IEEE F16 scale followed by 32 signed quantized values.
// There is no inter-block padding. Serialized scales (also F16 rows) are
// little-endian; byte-buffer APIs do not require aligned storage.
struct KvBlockQ8_0 {
  std::uint16_t scale;
  std::int8_t values[32];
};
static_assert(sizeof(KvBlockQ8_0) == 34);
static_assert(offsetof(KvBlockQ8_0, scale) == 0);
static_assert(offsetof(KvBlockQ8_0, values) == 2);
static_assert(std::is_standard_layout_v<KvBlockQ8_0>);
static_assert(std::is_trivially_copyable_v<KvBlockQ8_0>);

// Empty rows are valid. Unknown types, partial q8 blocks and size_t overflow
// are invalid. KvRowBytes throws invalid_argument for invalid geometry.
[[nodiscard]] constexpr bool IsValidKvRow(std::uint32_t elements,
                                          KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return elements <= std::numeric_limits<std::size_t>::max() / 2;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      return elements % kKvQ8BlockElements == 0 &&
             elements / kKvQ8BlockElements <=
                 std::numeric_limits<std::size_t>::max() / sizeof(KvBlockQ8_0);
  }
  return false;
}

[[nodiscard]] constexpr std::size_t KvRowBytes(std::uint32_t elements,
                                               KvCacheType type) {
  if (!IsValidKvRow(elements, type))
    throw std::invalid_argument("invalid Flash-Next KV row geometry");
  return type == KvCacheType::kF16
             ? std::size_t{elements} * 2
             : std::size_t{elements / kKvQ8BlockElements} * sizeof(KvBlockQ8_0);
}

[[nodiscard]] constexpr const char* KvCacheTypeName(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return "f16";
    case KvCacheType::kQ8_0:
      return "q8_0";
    case KvCacheType::kQ8_0Rotated:
      return "q8_0-rotated";
  }
  return "unknown";
}

// Portable IEEE binary32/binary16 conversions, with round-to-nearest-even
// independent of the host rounding mode. Preserve signed zero and infinity;
// NaNs become signed quiet NaNs. Finite overflow rounds to infinity.
[[nodiscard]] std::uint16_t KvFloatToHalfBits(float value) noexcept;
[[nodiscard]] float KvHalfBitsToFloat(std::uint16_t bits) noexcept;

// Canonical ggml q8_0 rounding primitive: roundf ties away from zero,
// clamp to [-127,127], rejecting nonfinite scaled values.
[[nodiscard]] std::int8_t RoundKvQ8Code(float scaled);

// CPU oracle for canonical live-row insertion, NOT snapshot compression.
// K input is F32 post-RoPE; V input is F32. Each q8 block independently uses
// d = max(abs(x))/127 in F32, q = clamp(roundf(x/d), -127, 127), where roundf
// ties go away from zero. Only the STORED scale is rounded to F16 RNE.
// All-zero blocks have scale=0/q=0. If F32 d itself underflows to zero, the
// block also becomes zero. If only the stored F16 scale underflows, preserve
// q computed from F32 d (no rescaling). CPU decode preserves zero signs.
// Decode rounds q * stored_scale to F16 before promoting to float. GPU scalar
// attention/WMMA operands require IEEE value equality, with +0/-0 equivalent;
// nonzero reconstructed values and packed bytes are not relaxed.
//
// Exact buffer lengths are required. All APIs throw invalid_argument for
// invalid geometry, nonfinite input/payload, negative scales (including -0),
// -128 quants, or scale/reconstruction overflow to F16 infinity. Encode and
// Decode reject overlapping buffers. Validation finishes before any output
// writes, so a rejected call leaves the entire destination unchanged. Empty
// buffers are accepted only for an empty valid row. No HIP dependency.
// With rotation enabled these storage-only APIs accept kQ8_0Rotated as
// ALREADY-ROTATED coordinates. They never rotate, inverse-rotate or migrate
// bytes; the caller must choose the K/Q or V/output transform separately.
void EncodeKvRow(std::span<const float> input, KvCacheType type,
                 std::span<std::byte> output);
void DecodeKvRow(std::span<const std::byte> input, KvCacheType type,
                 std::span<float> output);
void ValidateKvRow(std::span<const std::byte> input, std::uint32_t elements,
                   KvCacheType type);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
