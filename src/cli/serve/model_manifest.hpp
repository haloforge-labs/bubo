#pragma once

#include <filesystem>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/fingerprint_table.hpp"

namespace gufo::server {
struct PackageFile {
  std::string logical_name, sha256;
  std::filesystem::path path;
  std::uint64_t size = 0;
  bool observed_verified = false;
  std::shared_ptr<bubo::prepare::ReadLease> lease;
};
struct ModelPackage {
  std::filesystem::path manifest_path;
  std::map<std::string, std::string> defaults;
  std::vector<PackageFile> files;
  std::uint64_t hashed_files = 0, hashed_bytes = 0;
  std::string MainIdentity() const;
  std::string IdentityFor(const std::filesystem::path&) const;
};
// Only bubo-package.json is discovered, beside the supplied main shard.
// Explicit selection must use the same filename. A selected broken manifest
// never falls back. Junctions are permitted; containment is lexical.
ModelPackage LoadModelPackage(const std::filesystem::path& model_path,
                              const std::string& profile, bool verify = false,
                              std::ostream* progress = nullptr,
                              const std::filesystem::path& selected = {});
// Validate selected overrides using the corresponding compiled package pin.
// Held read leases keep identity stable until the engine has opened its maps.
void CheckPackageOverrides(ModelPackage& package,
                           const std::map<std::string, std::string>& paths,
                           bool verify, std::ostream* progress = nullptr);
#ifdef BUBO_PREPARE_TESTING
ModelPackage LoadModelPackageForTesting(
    const std::filesystem::path& model_path, const std::string& profile,
    std::span<const bubo::prepare::Entry> entries, bool verify = false,
    std::ostream* progress = nullptr,
    const std::function<bubo::prepare::FingerprintResult(
        const bubo::prepare::Entry&)>& fingerprints = {},
    const std::filesystem::path& selected = {});
#endif
}  // namespace gufo::server
