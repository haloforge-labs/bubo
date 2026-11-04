#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_KV_ROTATION_RESTORE_TEST_SUPPORT_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_KV_ROTATION_RESTORE_TEST_SUPPORT_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace gufo::test::kv_rotation_restore {
namespace qfn = gufo::models::qwen38_flash_next;
inline void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}

// Test-only named wire probes, matching engine.cpp's SessionSnapshotHeader
// and the prefix of executor.cpp's SnapshotHeader. No guessed numeric offsets,
// private-member access, synthetic snapshot or production hook. Before
// mutation, bind these probes to an ACTUAL captured frame's geometry, size and
// epochs. A layout drift must fail qualification, not become a skipped fault
// case.
struct PublicHeader {
  std::array<char, 8> magic;
  std::uint32_t version, vocab_size, token_count, hidden_rows;
  std::uint64_t executor_bytes;
  qfn::MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes, policy_concurrency;
  std::uint32_t falsify_rounds, reserved;
  qfn::KvCacheType kv_cache_type;
  std::uint32_t storage_reserved;
};
struct NativePrefix {
  std::array<char, 8> magic;
  std::uint32_t num_layers, full_attention_interval, conv_elems, state_elems;
  std::uint32_t kv_row, kv_cache_type, numerics_epoch;
};
static_assert(std::is_trivially_copyable_v<PublicHeader>);
static_assert(std::is_trivially_copyable_v<NativePrefix>);
struct Headers {
  PublicHeader outer;
  NativePrefix native;
  std::size_t native_offset;
};
inline Headers ReadHeaders(std::span<const std::uint8_t> bytes,
                           const qfn::Model& model) {
  Require(bytes.size() >= sizeof(PublicHeader), "short captured public frame");
  PublicHeader outer{};
  std::memcpy(&outer, bytes.data(), sizeof(outer));
  Require(
      outer.magic ==
              std::array<char, 8>{'Q', 'F', 'N', 'S', 'E', 'S', 'S', '1'} &&
          outer.version == qfn::ExpectedKvPayloadVersion(model.CacheType()) &&
          outer.kv_cache_type == model.CacheType() &&
          outer.vocab_size == model.VocabSize() && outer.token_count != 0 &&
          outer.policy_concurrency == model.DecodeConcurrency() &&
          (outer.image_identity_bytes == 0 || outer.image_identity_bytes == 32),
      "captured public header does not match named wire probe");
  const std::uint64_t offset =
      sizeof(PublicHeader) + std::uint64_t{outer.image_identity_bytes} +
      std::uint64_t{outer.token_count} * sizeof(std::int32_t) +
      std::uint64_t{outer.vocab_size} * sizeof(float);
  Require(offset <= bytes.size() &&
              outer.executor_bytes == bytes.size() - offset &&
              outer.executor_bytes >= sizeof(NativePrefix),
          "captured public/native boundary does not match metadata");
  NativePrefix native{};
  std::memcpy(&native, bytes.data() + offset, sizeof(native));
  const auto& config = model.config();
  Require(
      native.magic ==
              std::array<char, 8>{'Q', 'F', 'N', 'S', 'N', 'A', 'P', '8'} &&
          native.num_layers == config.num_layers &&
          native.full_attention_interval == config.full_attention_interval &&
          native.conv_elems ==
              (config.ssm_conv_kernel - 1) * config.SsmConvChannels() &&
          native.state_elems == config.ssm_num_v_heads * config.ssm_head_dim *
                                    config.ssm_head_dim &&
          native.kv_row == config.AttentionKvDim() &&
          native.kv_cache_type ==
              static_cast<std::uint32_t>(model.CacheType()) &&
          native.numerics_epoch ==
              qfn::ExpectedKvPayloadVersion(model.CacheType()),
      "captured native header does not match named wire probe");
  return {outer, native, static_cast<std::size_t>(offset)};
}
template<class T>
inline void Put(std::vector<std::uint8_t>& bytes, std::size_t offset, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  Require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
          "wire field outside captured frame");
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
inline std::vector<std::uint8_t> Payload(qfn::Session& session) {
  std::string error;
  auto snapshot =
      session.SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
  Require(bool(snapshot), error);
  std::vector<std::uint8_t> bytes(snapshot->SizeBytes());
  Require(snapshot->CopyTo(bytes), "materialize restore-fault control");
  return bytes;
}
inline void RejectWithoutMutation(qfn::Session& session,
                                  std::span<const std::uint8_t> bad,
                                  const std::vector<std::uint8_t>& good,
                                  const std::string& expected_error = {}) {
  // Callers install the exact good control first. Compare one temporary at
  // a time instead of retaining another model-sized baseline allocation.
  Require(Payload(session) == good,
          "restore-fault baseline is not the valid control");
  std::string error;
  Require(!session.RestoreSnapshot(bad, &error), "malformed snapshot accepted");
  Require(expected_error.empty() || error == expected_error,
          "snapshot rejected for another reason: " + error);
  Require(!error.empty() && session.IsValid() && Payload(session) == good,
          "snapshot preflight rejection changed public/native state");
  Require(session.RestoreSnapshot(good, &error), error);
  Require(session.IsValid() && Payload(session) == good,
          "valid restore after rejection did not recover exact bytes");
}
inline void CheckHeaderPairs(qfn::Session& session, const qfn::Model& model,
                             const std::vector<std::uint8_t>& good) {
  static_assert(qfn::Session::kSnapshotPayloadVersion == 21);
  const auto headers = ReadHeaders(good, model);
  const auto saved = Payload(session);
  std::string error;
  Require(session.RestoreSnapshot(good, &error), error);
  const auto check = [&](std::uint32_t type, std::uint32_t version) {
    if (type == static_cast<std::uint32_t>(model.CacheType()) &&
        version == qfn::ExpectedKvPayloadVersion(model.CacheType()))
      return;
    auto bad = good;
    Put(bad, offsetof(PublicHeader, kv_cache_type),
        static_cast<qfn::KvCacheType>(type));
    Put(bad, offsetof(PublicHeader, version), version);
    RejectWithoutMutation(session, bad, good);
    bad = good;  // Valid outer frame MUST NOT hide a mismatched native pair.
    Put(bad, headers.native_offset + offsetof(NativePrefix, kv_cache_type),
        type);
    Put(bad, headers.native_offset + offsetof(NativePrefix, numerics_epoch),
        version);
    RejectWithoutMutation(session, bad, good);
  };
  for (auto type : {0U, 1U, 2U, 0xffffffffU})
    for (auto version : {0U, 18U, 19U, 20U, 21U, 22U, 23U})
      check(type, version);
  Require(session.RestoreSnapshot(saved, &error), error);
  Require(Payload(session) == saved,
          "header negatives disturbed original session");
}

enum class Fault {
  kMinus128,
  kNegativeScale,
  kNegativeZeroScale,
  kInfinityScale,
  kNanScale,
  kHalfOverflow
};
inline constexpr std::array kFaults = {
    Fault::kMinus128,      Fault::kNegativeScale, Fault::kNegativeZeroScale,
    Fault::kInfinityScale, Fault::kNanScale,      Fault::kHalfOverflow};
inline void CorruptLastBlock(std::span<std::byte> bytes, Fault fault) {
  Require(bytes.size() >= sizeof(qfn::KvBlockQ8_0), "short packed fault piece");
  auto block = bytes.last(sizeof(qfn::KvBlockQ8_0));
  const auto code = offsetof(qfn::KvBlockQ8_0, values);
  if (fault == Fault::kMinus128) {
    block[code] = std::byte{0x80};
    return;
  }
  const std::uint16_t scale = fault == Fault::kNegativeScale       ? 0xbc00
                              : fault == Fault::kNegativeZeroScale ? 0x8000
                              : fault == Fault::kInfinityScale     ? 0x7c00
                              : fault == Fault::kNanScale          ? 0x7e00
                                                                   : 0x7bff;
  const auto field = offsetof(qfn::KvBlockQ8_0, scale);
  block[field] = std::byte(scale & 255);
  block[field + 1] = std::byte(scale >> 8);
  std::fill(block.begin() + code, block.end(), std::byte{0});
  if (fault == Fault::kHalfOverflow)
    block[code] = std::byte{127};
}
inline std::size_t UniqueRegion(std::span<const std::uint8_t> frame,
                                std::span<const std::byte> captured) {
  Require(!captured.empty(), "empty captured KV region");
  const auto bytes = std::as_bytes(frame);
  const auto first =
      std::search(bytes.begin(), bytes.end(), captured.begin(), captured.end());
  Require(first != bytes.end(), "captured component absent from snapshot");
  Require(std::search(first + 1, bytes.end(), captured.begin(),
                      captured.end()) == bytes.end(),
          "captured KV region is not unique; no guessed offset permitted");
  return static_cast<std::size_t>(first - bytes.begin());
}
}  // namespace gufo::test::kv_rotation_restore
#endif
