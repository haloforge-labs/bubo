#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
struct TreeFile {
  std::string path;
  std::uint64_t size;
  std::string lfs_oid;
};
struct SourceEvidence {
  std::string repository;
  std::string commit;
  std::vector<TreeFile> files;
};
struct EvidenceResult {
  SourceEvidence evidence;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};
EvidenceResult LoadSourceEvidence(std::string_view json);
// Offline, no alternate trust pins or runtime URL/catalogue overrides.
std::string ValidateReleaseEvidence(std::span<const Entry> entries,
                                    std::span<const SourceEvidence> evidence);
// Local composition evidence is separate from immutable HF source trees.
// Hosting placeholders never qualify as published provenance.
std::string ValidateLowMemoryLocalEvidence(std::string_view json);
}  // namespace bubo::prepare
