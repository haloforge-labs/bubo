#include "src/models/qwen38_flash_next/kv_rotation.hpp"

#include <xmmintrin.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen38_flash_next/continuation_layout.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kv_producer.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using qfn::KvCacheType;
using qfn::KvRotationKind;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid rotation or payload accepted");
}
std::uint32_t Bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}
void Transform() {
  for (auto kind : {KvRotationKind::kKeyQuery, KvRotationKind::kValueOutput}) {
    const unsigned width = kind == KvRotationKind::kKeyQuery ? 256 : 64;
    const float scale = width == 256 ? 0.0625F : 0.125F;
    // Frozen independent Python struct.pack/unpack binary32 oracle, xorshift32
    // seed 0x6d2b79f5, exponents 0..137; normalization rounded before every
    // ascending butterfly. Includes subnormals and widely separated magnitudes.
    std::array<float, 256> mixed;
    std::uint32_t state = 0x6d2b79f5;
    for (unsigned i = 0; i < mixed.size(); ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      mixed[i] =
          std::bit_cast<float>((state & 0x807fffffU) | ((i % 138) << 23));
    }
    qfn::RotateKvHead(mixed, kind);
    const auto digest = gufo::crypto::Sha256Hex(
        {reinterpret_cast<const std::uint8_t*>(mixed.data()), sizeof(mixed)});
    Require(digest == (width == 256 ? "4a6dfb921ada7004d42989c934bedc1e1aab8989"
                                      "5c9a6c1ff383909f0eb2aa6a"
                                    : "9b171eec14d1fe1b7da51dab9b1f825fa39e8823"
                                      "cce2d0fb187a72e0dd19e3b6"),
            "normalization/butterfly F32 rounding golden changed");
    // Independent Sylvester matrix oracle on every basis, including all V
    // blocks. These exact powers of two have no summation ambiguity.
    for (unsigned basis = 0; basis < 256; ++basis) {
      std::array<float, 256> head{};
      head[basis] = 1;
      qfn::RotateKvHead(head, kind);
      for (unsigned i = 0; i < 256; ++i) {
        const auto expected = i / width != basis / width ? 0.0F
                              : std::popcount((i % width) & (basis % width)) % 2
                                  ? -scale
                                  : scale;
        Require(Bits(head[i]) == Bits(expected), "basis/ordering changed");
      }
      qfn::RotateKvHead(head, kind);
      for (unsigned i = 0; i < 256; ++i)
        Require(Bits(head[i]) == Bits(i == basis ? 1.0F : 0.0F),
                "exact basis inverse changed");
    }
    std::array<float, 256> zeros;
    zeros.fill(-0.0F);
    qfn::RotateKvHead(zeros, kind);
    for (unsigned i = 0; i < 256; ++i)
      Require(Bits(zeros[i]) == (i % width == 0 ? 0x80000000U : 0U),
              "strict CPU zero-sign contract changed");
    std::array<float, 256> tiny{};
    tiny[0] = std::numeric_limits<float>::min();
    qfn::RotateKvHead(tiny, kind);
    for (unsigned i = 0; i < 256; ++i)
      Require(Bits(tiny[i]) ==
                  (i < width ? (width == 256 ? 0x80000U : 0x100000U) : 0U),
              "normalization-first gradual underflow changed");
    tiny.fill(0);
    tiny[0] = -std::numeric_limits<float>::denorm_min();
    qfn::RotateKvHead(tiny, kind);
    Require(Bits(tiny[0]) == 0U, "subnormal tie normalization changed");
    std::array<float, 256> extreme;
    extreme.fill(std::numeric_limits<float>::max());
    const auto before = extreme;
    Reject([&] { qfn::RotateKvHead(extreme, kind); });
    Require(extreme == before, "overflow changed input before rejection");
    // Coherent amplification: plain q8 is valid but rotated reconstruction
    // overflows F16, despite finite transformed F32. No implicit codec
    // rotation.
    extreme.fill(width == 256 ? 5000.0F : 10000.0F);
    std::array<std::byte, 272> packed;
    qfn::EncodeKvRow(extreme, KvCacheType::kQ8_0, packed);
    qfn::RotateKvHead(extreme, kind);
    Reject(
        [&] { qfn::EncodeKvRow(extreme, KvCacheType::kQ8_0Rotated, packed); });
  }
  std::array<float, 256> head{};
  Reject([&] {
    qfn::RotateKvHead(std::span(head).first(255), KvRotationKind::kKeyQuery);
  });
  Reject([&] { qfn::RotateKvHead(head, static_cast<KvRotationKind>(99)); });
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    head[127] = invalid;
    Reject([&] { qfn::RotateKvHead(head, KvRotationKind::kKeyQuery); });
    Require(Bits(head[127]) == Bits(invalid),
            "nonfinite rejection changed input");
  }
  head.fill(0);
  Require(std::fesetround(FE_UPWARD) == 0, "cannot set test rounding mode");
  Reject([&] { qfn::RotateKvHead(head, KvRotationKind::kKeyQuery); });
  Require(std::fesetround(FE_TONEAREST) == 0,
          "cannot restore test rounding mode");
}
void ProducerSequence() {
  // Execute the actual HIP-free sequencing helper used by both executor
  // cache-only paths. Recording calls prove ordering/dispatch, NOT numerics,
  // GPU launches, separate-weight reachability or packed-byte equivalence.
  enum Call { kRms, kRope, kRotateKey, kRotateValue, kStoreKey, kStoreValue };
  for (auto type :
       {KvCacheType::kF16, KvCacheType::kQ8_0, KvCacheType::kQ8_0Rotated}) {
    for (int fail : {-1, 0, 1}) {
      std::vector<Call> calls;
      const bool ok = qfn::rocm::detail::ProduceKvRows(
          type,
          [&] {
            calls.push_back(kRms);
            calls.push_back(kRope);
          },
          [&](bool value, KvRotationKind kind) {
            Require(kind == (value ? KvRotationKind::kValueOutput
                                   : KvRotationKind::kKeyQuery),
                    "cache-only K/V rotation kinds exchanged");
            calls.push_back(value ? kRotateValue : kRotateKey);
            return fail != int(value);
          },
          [&](bool value) {
            calls.push_back(value ? kStoreValue : kStoreKey);
          });
      std::vector<Call> expected{kRms, kRope};
      const bool rotated = qfn::KvIsRotated(type);
      if (rotated) {
        expected.push_back(kRotateKey);
        if (fail != 0)
          expected.push_back(kRotateValue);
      }
      const bool success = !rotated || fail == -1;
      if (success) {
        expected.push_back(kStoreKey);
        expected.push_back(kStoreValue);
      }
      Require(ok == success && calls == expected,
              "cache-only missing/late/double rotation or store after refusal");
    }
  }
}
void Representation() {
  constexpr auto rotated = KvCacheType::kQ8_0Rotated;
  static_assert(static_cast<unsigned>(rotated) == 2);
  static_assert(qfn::KvRowBytes(512, rotated) == 544);
  static_assert(qfn::kKvRotatedNumericsEpoch == 22);
  static_assert(qfn::ExpectedKvPayloadVersion(rotated) ==
                qfn::kKvRotatedNumericsEpoch);
  static_assert(qfn::Session::kSnapshotPayloadVersion == 21);
  for (auto type : {KvCacheType::kF16, KvCacheType::kQ8_0, rotated}) {
    Require(qfn::IsKnownKvCacheType(type), "known type rejected");
    Require(qfn::KvIsRotated(type) == (type == rotated), "basis dispatch");
    Require(qfn::KvUsesQ8Codec(type) == (type != KvCacheType::kF16),
            "codec dispatch");
    for (auto payload : {KvCacheType::kF16, KvCacheType::kQ8_0, rotated})
      for (unsigned version : {0U, 17U, 18U, 19U, 20U, 21U, 22U, 23U})
        Require(
            qfn::Session::IsCompatibleSnapshotPayload(version, payload, type) ==
                (type == payload &&
                 version == qfn::ExpectedKvPayloadVersion(type)),
            "forged type/version pair accepted");
  }
  const auto unknown = static_cast<KvCacheType>(99);
  Require(!qfn::IsKnownKvCacheType(unknown), "unknown type known");
  Reject([&] { (void)qfn::KvUsesQ8Codec(unknown); });
  Reject([&] { (void)qfn::KvIsRotated(unknown); });
  Reject([&] { (void)qfn::ExpectedKvPayloadVersion(unknown); });
  Require(!qfn::Session::IsCompatibleSnapshotPayload(20, unknown, unknown),
          "unknown pair");
  const auto identity = qfn::KvCacheCompatibilityIdentity(rotated);
  Require(
      identity.find("ordinary_kv_dtype=q8_0-rotated\n") != std::string::npos &&
          identity.find("numerical_kernel_epoch=" +
                        std::to_string(qfn::kKvRotatedNumericsEpoch) + "\n") !=
              std::string::npos &&
          identity.find("4-contiguous-h64") != std::string::npos &&
          identity.find("before-sigmoid-gate") != std::string::npos,
      "rotated identity omitted basis/version/placement");
  // Storage APIs consume already-rotated coordinates and return the same
  // coordinates, not an inverse transform. Structural labels are not proof
  // against deliberate relabeling of plain codes.
  std::array<float, 32> input{};
  input[0] = -1;
  std::array<std::byte, 34> plain{}, packed{};
  qfn::EncodeKvRow(input, KvCacheType::kQ8_0, plain);
  qfn::EncodeKvRow(input, rotated, packed);
  Require(plain == packed, "codec implicitly rotated input");
  std::array<float, 32> decoded{};
  qfn::DecodeKvRow(packed, rotated, decoded);
  Require(decoded[0] < 0 && decoded[1] == 0,
          "decode implicitly inverted basis");
  for (unsigned fault = 0; fault < 6; ++fault) {
    auto bad = packed;
    if (fault == 0)
      bad[2] = std::byte{0x80};
    if (fault == 1) {
      bad[0] = std::byte{0};
      bad[1] = std::byte{0x80};
    }
    if (fault == 2)
      bad[1] |= std::byte{0x80};
    if (fault == 3) {
      bad[0] = std::byte{0};
      bad[1] = std::byte{0x7c};
    }
    if (fault == 4) {
      bad[0] = std::byte{1};
      bad[1] = std::byte{0x7c};
    }
    if (fault == 5) {
      bad[0] = std::byte{0xff};
      bad[1] = std::byte{0x7b};
      bad[2] = std::byte{127};
    }
    Reject([&] { qfn::ValidateKvRow(bad, 32, rotated); });
    decoded.fill(42);
    Reject([&] { qfn::DecodeKvRow(bad, rotated, decoded); });
    for (auto value : decoded)
      Require(value == 42, "malformed decode wrote output");
  }
}
}  // namespace
int main() {
  try {
    Require((_mm_getcsr() & ((1U << 15) | (1U << 6))) == 0,
            "CPU oracle requires FTZ/DAZ clear");
    Require(std::fegetround() == FE_TONEAREST, "CPU oracle requires RNE");
    Transform();
    ProducerSequence();
    Representation();
    std::cout << "KV rotation CPU checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
