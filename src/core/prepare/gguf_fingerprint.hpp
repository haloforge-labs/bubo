#pragma once

#include <filesystem>
#include <string>

#include "src/core/prepare/fingerprint_table.hpp"

namespace bubo::prepare {
inline constexpr std::string_view kFingerprintGeneratorVersion =
    "bubo-gguf-fingerprint-v2";
// Bounded read of the header/table and min(1 MiB, tensor bytes) for EVERY
// tensor. Caller must first establish whole-file source provenance before
// generating pins.
FingerprintResult GenerateGgufFingerprint(const std::filesystem::path& path,
                                          const Entry& shard);
// Returns empty on success. Schema 1 format-only fixtures cannot verify files.
// Full SHA is deliberately opt-in and uses the existing OpenSSL hasher.
std::string CheckGgufFingerprint(const std::filesystem::path& path,
                                 const Entry& shard,
                                 const FingerprintTable& pin,
                                 bool full_verify = false);
}  // namespace bubo::prepare
