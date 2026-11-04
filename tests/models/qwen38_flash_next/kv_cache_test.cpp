#include "src/models/qwen38_flash_next/kv_cache.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kv_q8_prefill_view.hpp"
#include "tests/models/qwen38_flash_next/checkpoint_memory_test_support.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using qfn::KvCacheType;

namespace {
void CheckCheckpointMemoryBound() {
  using gufo::test::QfnDeepCheckpointMinimumBytes;
  constexpr std::uint64_t f16_minimum = 80ULL << 20;
  static_assert(QfnDeepCheckpointMinimumBytes(512, KvCacheType::kF16) ==
                f16_minimum);
  static_assert(QfnDeepCheckpointMinimumBytes(512, KvCacheType::kQ8_0) ==
                f16_minimum * 544 / 1024);
  static_assert(QfnDeepCheckpointMinimumBytes(512, KvCacheType::kQ8_0Rotated) ==
                f16_minimum * 544 / 1024);
  static_assert(QfnDeepCheckpointMinimumBytes(256, KvCacheType::kF16) ==
                f16_minimum);
  static_assert(QfnDeepCheckpointMinimumBytes(1024, KvCacheType::kQ8_0) ==
                f16_minimum * 1088 / 2048);
}

void CheckPrefillViewPolicy() {
  using qfn::rocm::detail::Q8PrefillViewCapacity;
  using qfn::rocm::detail::Q8PrefillViewFits;
  static_assert(Q8PrefillViewCapacity(200000, 262144, 262144) == 200000,
                "q8 view must use configured capacity, not native capacity");
  static_assert(Q8PrefillViewCapacity(0, 262144, 262144) == 262144,
                "default q8 view must retain native capacity");
  static_assert(Q8PrefillViewCapacity(400000, 262144, 262144) == 262144,
                "q8 view must not exceed the WMMA context limit");
  static_assert(Q8PrefillViewCapacity(200000, 0, 262144) == 200000 &&
                    Q8PrefillViewCapacity(0, 0, 262144) == 0,
                "q8 view zero fallback policy");
  static_assert(Q8PrefillViewFits(200000, 0, 200000) &&
                    Q8PrefillViewFits(200000, 199744, 256) &&
                    Q8PrefillViewFits(200000, 199999, 1) &&
                    Q8PrefillViewFits(200000, 200000, 0),
                "q8 view must fit a frontier exactly at capacity");
  // These are view eligibility checks, NOT session/context rejection rules:
  // a larger direct-API frontier must keep the canonical packed fallback.
  static_assert(!Q8PrefillViewFits(200000, 199744, 257) &&
                    !Q8PrefillViewFits(200000, 200000, 1) &&
                    !Q8PrefillViewFits(200000, 200001, 0) &&
                    !Q8PrefillViewFits(0, 0, 0) && !Q8PrefillViewFits(0, 0, 1),
                "q8 view accepted an absent or insufficient allocation");
  static_assert(Q8PrefillViewCapacity(200000, 262144, 0) == 0 &&
                    Q8PrefillViewCapacity(200000, 262144, 131072) == 131072,
                "unsupported or smaller WMMA geometry must cap the view");
  constexpr auto max = std::numeric_limits<std::uint32_t>::max();
  static_assert(Q8PrefillViewCapacity(max, 262144, 262144) == 262144 &&
                    Q8PrefillViewFits(max, 0, max) &&
                    Q8PrefillViewFits(max, max - 3, 3) &&
                    Q8PrefillViewFits(max, max, 0) &&
                    !Q8PrefillViewFits(max, max - 3, 4) &&
                    !Q8PrefillViewFits(max, max, 1) &&
                    !Q8PrefillViewFits(200000, max - 127, 256) &&
                    !Q8PrefillViewFits(200000, 1, max),
                "q8 view range check wrapped uint32_t");
}

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

template<class F>
void Reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid KV input accepted");
}

std::vector<std::byte> Encode(std::span<const float> input, KvCacheType type) {
  std::vector<std::byte> bytes(
      qfn::KvRowBytes(static_cast<std::uint32_t>(input.size()), type));
  qfn::EncodeKvRow(input, type, bytes);
  return bytes;
}

bool SameFloats(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](float x, float y) {
           return std::bit_cast<std::uint32_t>(x) ==
                  std::bit_cast<std::uint32_t>(y);
         });
}

void CheckLayout() {
  static_assert(static_cast<std::uint32_t>(KvCacheType::kF16) == 0);
  static_assert(static_cast<std::uint32_t>(KvCacheType::kQ8_0) == 1);
  static_assert(qfn::KvRowBytes(512, KvCacheType::kF16) == 1024);
  static_assert(qfn::KvRowBytes(512, KvCacheType::kQ8_0) == 544);
  static_assert(qfn::KvRowBytes(32, KvCacheType::kQ8_0) == 34);
  static_assert(qfn::KvRowBytes(0, KvCacheType::kQ8_0) == 0);
  static_assert(!qfn::IsValidKvRow(31, KvCacheType::kQ8_0));
  static_assert(!qfn::IsValidKvRow(32, static_cast<KvCacheType>(99)));
  Require(std::string_view(qfn::KvCacheTypeName(KvCacheType::kF16)) == "f16" &&
              std::string_view(qfn::KvCacheTypeName(KvCacheType::kQ8_0)) ==
                  "q8_0" &&
              std::string_view(qfn::KvCacheTypeName(
                  static_cast<KvCacheType>(99))) == "unknown",
          "KV type names changed");
  Reject([] { (void)qfn::KvRowBytes(33, KvCacheType::kQ8_0); });
  Reject([] { (void)qfn::KvRowBytes(0, static_cast<KvCacheType>(99)); });
  // Independent production geometry: only ordinary K/V shrink, not pooled
  // indexer keys, token IDs, or the 133032096-byte fixed private reservation.
  Require(13 * 2 * qfn::KvRowBytes(512, KvCacheType::kQ8_0) + 832 + 4 == 14980,
          "q8 checkpoint slope changed");
  Require(4ULL * 200000 * 13 * 2 *
                  (qfn::KvRowBytes(512, KvCacheType::kF16) -
                   qfn::KvRowBytes(512, KvCacheType::kQ8_0)) ==
              9984000000ULL,
          "q8 live allocation delta changed");
}

void CheckHalfConversions() {
  // Raw binary32 fixtures avoid asking the converter to generate its oracle.
  struct Fixture {
    std::uint32_t f32;
    std::uint16_t f16;
  };
  constexpr Fixture fixtures[] = {
      {0x00000000, 0x0000}, {0x80000000, 0x8000}, {0x3f800000, 0x3c00},
      {0xbf800000, 0xbc00}, {0x33800000, 0x0001}, {0x33000000, 0x0000},
      {0x33000001, 0x0001}, {0x33c00000, 0x0002}, {0x387fc000, 0x03ff},
      {0x38800000, 0x0400}, {0x387fe000, 0x0400}, {0x3f801000, 0x3c00},
      {0x3f803000, 0x3c02}, {0xbf801000, 0xbc00}, {0x477fe000, 0x7bff},
      {0x477fefff, 0x7bff}, {0x477ff000, 0x7c00}, {0x7f7fffff, 0x7c00},
      {0x7f800000, 0x7c00}, {0xff800000, 0xfc00}, {0x7f800001, 0x7e00},
      {0xffc12345, 0xfe00}};
  const int original_rounding = std::fegetround();
  for (const auto rounding :
       {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    Require(std::fesetround(rounding) == 0, "cannot set host rounding mode");
    for (const auto fixture : fixtures)
      Require(qfn::KvFloatToHalfBits(std::bit_cast<float>(fixture.f32)) ==
                  fixture.f16,
              "F32 to F16 fixture/rounding mode mismatch");
  }
  Require(std::fesetround(original_rounding) == 0,
          "cannot restore rounding mode");

  // Exhaustive half decoding against an independent arithmetic definition.
  for (std::uint32_t h = 0; h <= 0xffff; ++h) {
    const auto bits = static_cast<std::uint16_t>(h);
    const int exponent = (h >> 10) & 31;
    const int fraction = h & 1023;
    const float actual = qfn::KvHalfBitsToFloat(bits);
    if (exponent == 31 && fraction != 0) {
      Require(std::isnan(actual), "half NaN lost");
      Require(qfn::KvFloatToHalfBits(actual) == ((h & 0x8000) | 0x7e00),
              "NaN canonicalization changed");
      continue;
    }
    double magnitude = exponent == 0
                           ? std::ldexp(double(fraction), -24)
                           : std::ldexp(double(1024 + fraction), exponent - 25);
    if (exponent == 31)
      magnitude = std::numeric_limits<double>::infinity();
    const auto expected =
        static_cast<float>((h & 0x8000) ? -magnitude : magnitude);
    Require(SameFloats(std::span(&actual, 1), std::span(&expected, 1)),
            "half decode disagrees with independent arithmetic");
    Require(qfn::KvFloatToHalfBits(actual) == bits,
            "finite half roundtrip changed");
  }
  // Every adjacent positive finite half pair has an exactly representable
  // F32 midpoint. Check RNE and its immediate neighbours, including subnormals.
  for (std::uint16_t h = 0; h < 0x7bff; ++h) {
    const float low = qfn::KvHalfBitsToFloat(h);
    const float high = qfn::KvHalfBitsToFloat(h + 1);
    const float midpoint = (low + high) * 0.5f;
    Require(qfn::KvFloatToHalfBits(midpoint) == ((h & 1) ? h + 1 : h),
            "half midpoint did not round to even");
    Require(qfn::KvFloatToHalfBits(std::nextafter(midpoint, low)) == h &&
                qfn::KvFloatToHalfBits(std::nextafter(midpoint, high)) == h + 1,
            "half midpoint neighbours rounded incorrectly");
  }
}

void CheckKnownBytes() {
  std::array<float, 32> input = {127,    -127,    0,    -0.0f, 0.5f,   -0.5f,
                                 1.5f,   -1.5f,   2.5f, -2.5f, 126.5f, -126.5f,
                                 0.499f, -0.499f, 42,   -42};
  constexpr std::array<std::uint8_t, 34> expected = {
      0x00, 0x3c, 0x7f, 0x81, 0x00, 0x00, 0x01, 0xff, 0x02,
      0xfe, 0x03, 0xfd, 0x7f, 0x81, 0x00, 0x00, 0x2a, 0xd6};
  const auto bytes = Encode(input, KvCacheType::kQ8_0);
  for (std::size_t i = 0; i < expected.size(); ++i)
    Require(bytes[i] == std::byte(expected[i]),
            "known q8 packed bytes changed");
  constexpr std::array<float, 32> decoded = {
      127, -127, 0, 0, 1, -1, 2, -2, 3, -3, 127, -127, 0, 0, 42, -42};
  std::array<float, 32> output{};
  qfn::DecodeKvRow(bytes, KvCacheType::kQ8_0, output);
  Require(SameFloats(output, decoded), "known q8 reconstruction changed");

  // F32 d = 1 + 2^-12 rounds to stored F16 1. Using the stored scale
  // for quantization would incorrectly round element 1 up to 1.
  input.fill(0);
  input[0] = 127.031005859375f;
  input[1] = 0.50006103515625f;
  input[2] = 0.5001220703125f;
  input[3] = -0.5001220703125f;
  const auto unrounded = Encode(input, KvCacheType::kQ8_0);
  Require(unrounded[0] == std::byte{0} && unrounded[1] == std::byte{0x3c} &&
              unrounded[2] == std::byte{127} && unrounded[3] == std::byte{0} &&
              unrounded[4] == std::byte{1} && unrounded[5] == std::byte{0xff},
          "q8 quantized using rounded F16 scale or wrong tie rule");
  for (const auto& [d, expected_low] :
       {std::pair{1.00048828125f, 0}, std::pair{1.00146484375f, 2}}) {
    input.fill(0);
    input[0] = 127 * d;
    const auto tie = Encode(input, KvCacheType::kQ8_0);
    Require(tie[0] == std::byte(expected_low) && tie[1] == std::byte{0x3c},
            "stored q8 scale did not round to nearest even");
  }

  // Independent operand oracle: F16 scale 0x3555 = 0.333251953125.
  // These products must be rounded to half, NOT returned as raw F32 products.
  std::array<std::byte, 34> operand{};
  operand[0] = std::byte{0x55};
  operand[1] = std::byte{0x35};
  operand[2] = std::byte{3};
  operand[3] = std::byte{7};
  operand[4] = std::byte{11};
  operand[5] = std::byte{127};
  operand[6] = std::byte{0xfd};
  qfn::DecodeKvRow(operand, KvCacheType::kQ8_0, output);
  Require(output[0] == 1.0f && output[1] == 2.33203125f &&
              output[2] == 3.666015625f && output[3] == 42.3125f &&
              output[4] == -1.0f,
          "q8 reconstruction is not the canonical F16 attention operand");
}

void CheckNonPowerOfTwoScales() {
  // Expected signed codes and stored scale bits are independent of Decode.
  const auto check = [](float amax, float center, std::uint16_t scale,
                        std::array<int, 3> codes) {
    std::array<float, 32> input{};
    std::array<int, 32> expected{};
    input[0] = amax;
    input[1] = -amax;
    expected[0] = 127;
    expected[1] = -127;
    const std::array<float, 3> neighbours = {
        std::nextafter(center, 0.0F), center,
        std::nextafter(center, std::numeric_limits<float>::infinity())};
    for (std::size_t j = 0; j < neighbours.size(); ++j) {
      input[2 + j] = neighbours[j];
      input[5 + j] = -neighbours[j];
      expected[2 + j] = codes[j];
      expected[5 + j] = -codes[j];
    }
    const auto bytes = Encode(input, KvCacheType::kQ8_0);
    Require(
        bytes[0] == std::byte(scale & 255) && bytes[1] == std::byte(scale >> 8),
        "non-power-of-two stored scale changed");
    for (std::size_t j = 0; j < expected.size(); ++j)
      Require(bytes[2 + j] == std::byte(static_cast<std::uint8_t>(expected[j])),
              "non-power-of-two signed code fixture changed");
  };
  // Exact d=381/127=3: every representable half-integer tie, both signs,
  // and +/-1 F32 ULP in the INPUT. Ties must round away from zero.
  for (int k = 0; k < 127; ++k)
    check(381.0F, (k + 0.5F) * 3.0F, 0x4200, {k, k + 1, k + 1});

  // Nonexact F32 d=amax/127. Raw inputs and expected codes were derived
  // with exact rational arithmetic and binary32 RNE, not an encode/decode
  // pair. Some input neighbours divide to the SAME half-integer in F32.
  // 0x3f80003c distinguishes amax/127 from amax*(1/127); the 10 and 1000
  // fixtures distinguish x/d from x*(1/d), including negative sign mirrors.
  struct Fixture {
    std::uint32_t amax, center;
    std::uint16_t scale;
    std::array<int, 3> codes;
  };
  constexpr Fixture fixtures[] = {
      {0x3f80003c, 0x3b810241, 0x2008, {0, 1, 1}},
      {0x3f80003c, 0x3f00003c, 0x2008, {63, 63, 64}},
      {0x41200000, 0x3f53a74f, 0x2d0a, {11, 11, 11}},
      {0x41200000, 0x411f5ebd, 0x2d0a, {126, 126, 127}},
      {0x447a0000, 0x407bf7f0, 0x47e0, {0, 1, 1}},
      {0x447a0000, 0x42a55ab6, 0x47e0, {10, 11, 11}},
      {0x447a0000, 0x43fa0000, 0x47e0, {63, 64, 64}},
      {0x3dcccccd, 0x3c077551, 0x1273, {10, 11, 11}},
      // RNE(x/d)==1.5 although the exact ratio is below it: code must be 2.
      {0x43be8003, 0x40900002, 0x4200, {1, 2, 2}},
  };
  for (const auto& fixture : fixtures)
    check(std::bit_cast<float>(fixture.amax),
          std::bit_cast<float>(fixture.center), fixture.scale, fixture.codes);
}

void CheckRoundAwayAndSaturation() {
  // The named ggml q8_0 convention is roundf (away), NOT rint/nearbyint (even).
  for (const auto center : {0.5f, 1.5f, 2.5f}) {
    const auto lower_code = static_cast<int>(center);
    for (const auto sign : {1.0f, -1.0f}) {
      const auto toward_zero = std::nextafter(sign * center, 0.0f);
      const auto away = std::nextafter(sign * center, sign * INFINITY);
      Require(qfn::RoundKvQ8Code(toward_zero) == sign * lower_code,
              "near tie toward zero was rounded onto the tie");
      Require(qfn::RoundKvQ8Code(sign * center) == sign * (lower_code + 1),
              "exact half tie was not rounded away from zero");
      Require(qfn::RoundKvQ8Code(away) == sign * (lower_code + 1),
              "near tie away from zero changed canonical rounding");
    }
  }
  for (const auto sign : {1.0f, -1.0f}) {
    const auto center = sign * 127.5f;
    for (const auto value : {std::nextafter(center, 0.0f), center,
                             std::nextafter(center, sign * INFINITY),
                             sign * std::numeric_limits<float>::max()})
      Require(qfn::RoundKvQ8Code(value) == sign * 127,
              "round-away saturation did not clamp to signed 127");
  }
  Require(qfn::RoundKvQ8Code(0.0f) == 0 && qfn::RoundKvQ8Code(-0.0f) == 0,
          "signed zero did not produce code zero");
  for (const auto value :
       {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()})
    Reject([&] { (void)qfn::RoundKvQ8Code(value); });
}

void CheckZerosAndUnderflow() {
  std::array<float, 32> input{};
  input[1] = -0.0f;
  auto bytes = Encode(input, KvCacheType::kQ8_0);
  Require(std::all_of(bytes.begin(), bytes.end(),
                      [](auto b) { return b == std::byte{0}; }),
          "allzero q8 block is not canonical zero");
  input[0] = std::numeric_limits<float>::denorm_min();
  Require(Encode(input, KvCacheType::kQ8_0) == bytes,
          "F32 scale underflow did not yield canonical zero");
  for (const auto amax : {std::numeric_limits<float>::min(),
                          std::numeric_limits<float>::min() / 8}) {
    input[0] = amax;
    input[1] = -amax;
    const auto tiny = Encode(input, KvCacheType::kQ8_0);
    std::array<std::byte, 34> expected{};
    expected[2] = std::byte{127};
    expected[3] = std::byte{0x81};
    Require(std::equal(tiny.begin(), tiny.end(), expected.begin()),
            "F32 subnormal scale flushed before computing canonical codes");
  }
  input[0] = 127 * std::ldexp(1.0f, -26);
  input[1] = -input[0];
  bytes = Encode(input, KvCacheType::kQ8_0);
  Require(bytes[0] == std::byte{0} && bytes[1] == std::byte{0} &&
              bytes[2] == std::byte{127} && bytes[3] == std::byte{0x81},
          "F16 scale underflow changed F32-scale quant bytes");
  // A byte snapshot must retain these nonzero quants even though all
  // reconstructed operands are zero. Decode/re-encode is observably lossy.
  std::array<std::byte, 34> expected_snapshot{};
  expected_snapshot[2] = std::byte{127};
  expected_snapshot[3] = std::byte{0x81};
  const auto snapshot = bytes;
  Require(
      std::equal(snapshot.begin(), snapshot.end(), expected_snapshot.begin()),
      "known underflow snapshot bytes changed");
  std::array<float, 32> output{};
  qfn::DecodeKvRow(snapshot, KvCacheType::kQ8_0, output);
  Require(
      Encode(output, KvCacheType::kQ8_0) != snapshot && bytes == snapshot,
      "snapshot reconstruction must not be used to regenerate packed bytes");
  Require(output[0] == 0 && !std::signbit(output[0]) && output[1] == 0 &&
              std::signbit(output[1]),
          "underflowed reconstruction lost signed zero");
  input[0] = 127 * std::ldexp(1.0f, -24);
  bytes = Encode(input, KvCacheType::kQ8_0);
  Require(bytes[0] == std::byte{1} && bytes[1] == std::byte{0},
          "smallest nonzero F16 scale flushed to zero");
  qfn::DecodeKvRow(bytes, KvCacheType::kQ8_0, output);
  Require(output[0] == input[0], "subnormal operand reconstruction changed");
}

std::vector<float> Values(std::size_t count, std::uint32_t seed) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    seed = seed * 1664525U + 1013904223U;
    values[i] =
        static_cast<float>(static_cast<int>(seed % 20001) - 10000) / 1024;
  }
  return values;
}

void CheckIndependentBlocksAndRoundtrip() {
  for (const auto type : {KvCacheType::kF16, KvCacheType::kQ8_0}) {
    auto values = Values(512, 193);
    const auto original_values = values;
    const auto packed = Encode(values, type);
    Require(Encode(values, type) == packed,
            "repeated encoding is nondeterministic");
    const auto block_bytes = qfn::KvRowBytes(32, type);
    for (std::size_t block = 0; block < 16; ++block) {
      const auto alone =
          Encode(std::span(values).subspan(block * 32, 32), type);
      Require(std::equal(alone.begin(), alone.end(),
                         packed.begin() + block * block_bytes),
              "a companion block changed packed bytes");
    }
    auto companions = Values(512, 42);
    companions.insert(companions.end(), values.begin(), values.end());
    const auto with_companion = Encode(companions, type);
    Require(std::equal(packed.begin(), packed.end(),
                       with_companion.begin() + packed.size()),
            "unrelated row/launch companion changed packed bytes");
    std::reverse(companions.begin(), companions.begin() + 512);
    const auto changed_companion = Encode(companions, type);
    Require(std::equal(packed.begin(), packed.end(),
                       changed_companion.begin() + packed.size()),
            "changed companion affected independent row");
    values[35] = 1100;
    const auto changed_block = Encode(values, type);
    Require(std::equal(packed.begin(), packed.begin() + block_bytes,
                       changed_block.begin()) &&
                std::equal(packed.begin() + 2 * block_bytes, packed.end(),
                           changed_block.begin() + 2 * block_bytes),
            "one changed block affected other block scales");

    // Snapshot transport copies canonical packed bytes verbatim. Deliberately
    // unaligned storage catches native half/block pointer assumptions.
    std::vector<std::byte> snapshot(packed.size() + 2, std::byte{0xa5});
    std::copy(packed.begin(), packed.end(), snapshot.begin() + 1);
    const auto payload = std::span(snapshot).subspan(1, packed.size());
    std::vector<float> live(512), restored(512);
    qfn::DecodeKvRow(packed, type, live);
    qfn::ValidateKvRow(payload, 512, type);
    qfn::DecodeKvRow(payload, type, restored);
    Require(SameFloats(live, restored), "packed roundtrip changed operands");
    Require(snapshot.front() == std::byte{0xa5} &&
                snapshot.back() == std::byte{0xa5} &&
                std::equal(packed.begin(), packed.end(), payload.begin()),
            "snapshot transport/decode mutated packed bytes");
    for (std::size_t i = 0; i < live.size(); ++i)
      Require(std::abs(live[i] - original_values[i]) <
                  (type == KvCacheType::kF16 ? 0.004f : 0.05f),
              "roundtrip error exceeds analytic fixture bound");
  }
  const std::array<float, 4> input = {1, -2, std::ldexp(1.0f, -24), -0.0f};
  const auto bytes = Encode(input, KvCacheType::kF16);
  const std::array<std::byte, 8> expected = {
      std::byte{0}, std::byte{0x3c}, std::byte{0}, std::byte{0xc0},
      std::byte{1}, std::byte{0},    std::byte{0}, std::byte{0x80}};
  Require(std::equal(bytes.begin(), bytes.end(), expected.begin()),
          "known little-endian F16 bytes changed");
}

// Small CPU attention reference, intentionally not a production engine or a
// claim of cross-geometry scalar/WMMA equality. Both paths use the identical
// causal reduction order and forced append boundaries with canonical operands.
std::vector<float> Attention(std::span<const std::byte> keys,
                             std::span<const std::byte> values,
                             KvCacheType type, std::uint32_t rows) {
  constexpr std::uint32_t width = 64;
  const auto stride = qfn::KvRowBytes(width, type);
  const auto query = Values(width, 79 + rows);
  std::vector<float> scores(rows), output(width, 0);
  std::array<float, width> row{};
  for (std::uint32_t r = 0; r < rows; ++r) {
    qfn::DecodeKvRow(keys.subspan(r * stride, stride), type, row);
    float dot = 0;
    for (std::uint32_t i = 0; i < width; ++i)
      dot += query[i] * row[i];
    scores[r] = dot / 8;
  }
  const float maximum = *std::max_element(scores.begin(), scores.end());
  float denominator = 0;
  for (auto& score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    qfn::DecodeKvRow(values.subspan(r * stride, stride), type, row);
    const float weight = scores[r] / denominator;
    for (std::uint32_t i = 0; i < width; ++i)
      output[i] += weight * row[i];
  }
  return output;
}

void CheckSameBoundaryWarmColdAttention() {
  constexpr std::uint32_t width = 64;
  constexpr std::uint32_t total = 263;
  constexpr std::array<std::uint32_t, 8> plan = {1,   2,   3,   4,
                                                 255, 256, 257, total};
  const auto keys = Values(total * width, 1001);
  const auto values = Values(total * width, 2001);
  for (const auto type : {KvCacheType::kF16, KvCacheType::kQ8_0}) {
    const auto stride = qfn::KvRowBytes(width, type);
    const auto append = [&](std::vector<std::byte>& dst,
                            const std::vector<float>& src, std::uint32_t begin,
                            std::uint32_t end) {
      dst.resize(end * stride);
      for (auto r = begin; r < end; ++r)
        qfn::EncodeKvRow(std::span(src).subspan(r * width, width), type,
                         std::span(dst).subspan(r * stride, stride));
    };
    std::vector<std::byte> cold_k, cold_v;
    std::vector<std::vector<float>> cold_outputs;
    std::uint32_t previous = 0;
    for (const auto end : plan) {
      append(cold_k, keys, previous, end);
      append(cold_v, values, previous, end);
      cold_outputs.push_back(Attention(cold_k, cold_v, type, end));
      previous = end;
    }
    for (std::size_t cut_index = 0; cut_index + 1 < plan.size(); ++cut_index) {
      const auto cut = plan[cut_index];
      std::vector<std::byte> live_k, live_v;
      previous = 0;
      for (std::size_t i = 0; i <= cut_index; ++i) {
        append(live_k, keys, previous, plan[i]);
        append(live_v, values, previous, plan[i]);
        previous = plan[i];
      }
      const auto snapshot_k = live_k;
      const auto snapshot_v = live_v;
      Require(
          std::equal(snapshot_k.begin(), snapshot_k.end(), cold_k.begin()) &&
              std::equal(snapshot_v.begin(), snapshot_v.end(), cold_v.begin()),
          "capture boundary changed canonical packed prefix");
      // Destroy/reuse the original slot, then restore elsewhere without
      // decoding/re-encoding. Snapshot bytes must not acquire cache history.
      std::fill(live_k.begin(), live_k.end(), std::byte{0xa5});
      std::fill(live_v.begin(), live_v.end(), std::byte{0x5a});
      auto warm_k = snapshot_k;
      auto warm_v = snapshot_v;
      Require(SameFloats(Attention(warm_k, warm_v, type, cut),
                         cold_outputs[cut_index]),
              "restored attention differs at matched boundary");
      previous = cut;
      for (std::size_t i = cut_index + 1; i < plan.size(); ++i) {
        append(warm_k, keys, previous, plan[i]);
        append(warm_v, values, previous, plan[i]);
        Require(SameFloats(Attention(warm_k, warm_v, type, plan[i]),
                           cold_outputs[i]),
                "warm/cold attention differs under identical boundary plan");
        previous = plan[i];
      }
      Require(
          warm_k == cold_k && warm_v == cold_v &&
              std::equal(snapshot_k.begin(), snapshot_k.end(),
                         warm_k.begin()) &&
              std::equal(snapshot_v.begin(), snapshot_v.end(), warm_v.begin()),
          "restore/append requantized or mutated packed snapshot bytes");
    }
  }
}

void CheckInvalidBuffers() {
  const auto invalid_type = static_cast<KvCacheType>(42);
  for (const auto type : {KvCacheType::kF16, KvCacheType::kQ8_0}) {
    std::vector<float> input(64, 1), output(64, 777);
    auto bytes = Encode(input, type);
    const auto valid_bytes = bytes;
    const auto original_output = output;
    Reject([&] {
      qfn::EncodeKvRow(input, type, std::span(bytes).first(bytes.size() - 1));
    });
    bytes.push_back(std::byte{0});
    Reject([&] { qfn::EncodeKvRow(input, type, bytes); });
    Reject([&] { qfn::DecodeKvRow(bytes, type, output); });
    Reject([&] { qfn::ValidateKvRow(bytes, 64, type); });
    bytes = valid_bytes;
    Reject([&] {
      qfn::DecodeKvRow(std::span(bytes).first(bytes.size() - 1), type, output);
    });
    Reject([&] { qfn::DecodeKvRow(bytes, type, std::span(output).first(32)); });
    Require(SameFloats(output, original_output),
            "invalid decode partially wrote output");
    for (const auto bad : {std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::max(), 70000.0f}) {
      input.back() = bad;
      Reject([&] { qfn::EncodeKvRow(input, type, bytes); });
      Require(bytes == valid_bytes, "late invalid input partially encoded row");
    }
    // Last half / last block corruptions must be caught before output writes.
    auto corrupt = valid_bytes;
    const auto offset = type == KvCacheType::kF16 ? corrupt.size() - 2
                                                  : sizeof(qfn::KvBlockQ8_0);
    for (const auto scale : {0x7c00, 0xfc00, 0x7e00}) {
      corrupt = valid_bytes;
      corrupt[offset] = std::byte(scale & 0xff);
      corrupt[offset + 1] = std::byte(scale >> 8);
      Reject([&] { qfn::ValidateKvRow(corrupt, 64, type); });
      Reject([&] { qfn::DecodeKvRow(corrupt, type, output); });
      Require(SameFloats(output, original_output),
              "corrupt payload partially decoded");
    }
    qfn::EncodeKvRow({}, type, {});
    qfn::DecodeKvRow({}, type, {});
    qfn::ValidateKvRow({}, 0, type);
    Reject([&] { qfn::EncodeKvRow(input, type, {}); });
    Reject([&] { qfn::DecodeKvRow({}, type, output); });

    // Valid overlapping spans must not silently destroy later source data.
    std::array<float, 64> shared{};
    auto shared_bytes = std::as_writable_bytes(std::span(shared));
    Reject([&] {
      qfn::EncodeKvRow(shared, type, shared_bytes.first(bytes.size()));
    });
    Reject([&] {
      qfn::DecodeKvRow(shared_bytes.first(bytes.size()), type, shared);
    });
    Require(std::all_of(shared.begin(), shared.end(),
                        [](float x) { return x == 0; }),
            "overlap rejection changed storage");
  }
  std::array<float, 32> input{};
  std::array<std::byte, 34> bytes{};
  Reject([&] { qfn::EncodeKvRow(input, invalid_type, bytes); });
  Reject([&] { qfn::DecodeKvRow(bytes, invalid_type, input); });
  Reject([&] { qfn::ValidateKvRow(bytes, 32, invalid_type); });
  Reject([&] {
    qfn::EncodeKvRow(std::span(input).first(31), KvCacheType::kQ8_0, bytes);
  });
  Reject([&] {
    qfn::DecodeKvRow(bytes, KvCacheType::kQ8_0, std::span(input).first(31));
  });
  Reject([&] { qfn::ValidateKvRow(bytes, 31, KvCacheType::kQ8_0); });
  for (const auto scale : {0x8000, 0xbc00, 0x7bff}) {
    bytes.fill(std::byte{0});
    bytes[0] = std::byte(scale & 0xff);
    bytes[1] = std::byte(scale >> 8);
    bytes[2] = std::byte{127};
    Reject([&] { qfn::DecodeKvRow(bytes, KvCacheType::kQ8_0, input); });
  }
  bytes.fill(std::byte{0});
  bytes[2] = std::byte{0x80};
  Reject([&] { qfn::ValidateKvRow(bytes, 32, KvCacheType::kQ8_0); });
  input.fill(0);
  input.back() = 65504;
  Reject([&] { qfn::EncodeKvRow(input, KvCacheType::kQ8_0, bytes); });
}
}  // namespace

int main() {
  try {
    CheckCheckpointMemoryBound();
    CheckPrefillViewPolicy();
    CheckLayout();
    CheckHalfConversions();
    CheckKnownBytes();
    CheckNonPowerOfTwoScales();
    CheckRoundAwayAndSaturation();
    CheckZerosAndUnderflow();
    CheckIndependentBlocksAndRoundtrip();
    CheckSameBoundaryWarmColdAttention();
    CheckInvalidBuffers();
    std::cout << "Flash-Next portable KV codec checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Flash-Next KV codec: " << e.what() << '\n';
    return 1;
  }
}
