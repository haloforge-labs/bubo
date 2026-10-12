#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
inline constexpr std::uint64_t kFingerprintBytes = 1024 * 1024;
struct TensorFingerprint {
  std::string tensor_name;
  // Absolute file offset of the tensor payload, not its GGUF-relative offset.
  std::uint64_t offset;
  std::uint64_t tensor_bytes;
  std::uint64_t checked_length;  // min(1 MiB, tensor_bytes)
  std::string sha256;
  std::uint32_t type = 0;
  std::vector<std::uint64_t> dimensions;
  std::uint64_t relative_offset = 0;
};
struct FingerprintTable {
  std::string logical_name;
  std::string file_sha256;
  std::vector<TensorFingerprint> tensors;
  std::uint32_t schema_version = 1;
  std::string generator_version;
  std::uint64_t header_bytes = 0;
  std::string header_sha256;
};
struct FingerprintResult {
  FingerprintTable table;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};
// Format loader only. Does not open/hash model bytes or claim full coverage.
// J2 generates complete tables and implements the production check policy.
FingerprintResult LoadFingerprintTable(std::string_view json,
                                       const Entry& shard);
// Compiled immutable production pins; empty/error is never a verified table.
FingerprintResult ProductionFingerprintTable(const Entry& shard);
std::string SerializeFingerprintTable(const FingerprintTable& table);
}  // namespace bubo::prepare
