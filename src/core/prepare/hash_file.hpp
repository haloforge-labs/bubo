#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

#include "src/core/crypto/sha256.hpp"

namespace bubo::prepare {
using HashProgress = std::function<void(std::uint64_t, std::uint64_t)>;
// One reader and one EVP consumer, two fixed 4 MiB buffers. A live context can
// absorb a saved prefix once and continue across in-process transport retries.
std::uint64_t HashInto(const std::filesystem::path& path,
                       gufo::crypto::Sha256Hasher& hasher, std::uint64_t limit,
                       const HashProgress& progress = {},
                       std::intptr_t protected_handle = -1);
std::string HashFile(const std::filesystem::path& path,
                     const HashProgress& progress = {});
}  // namespace bubo::prepare
