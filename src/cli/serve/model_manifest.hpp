#pragma once

#include <filesystem>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/fingerprint_table.hpp"

namespace gufo::server {
class ModelMissingError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};
struct PackageFile {
  std::string logical_name, sha256;
  std::filesystem::path path;
  std::uint64_t size = 0;
  bool observed_verified = false;
  std::shared_ptr<bubo::prepare::ReadLease> lease;
  bubo::prepare::FileStamp verified_stamp;
  bool freshly_verified = false;
};
struct ModelPackage {
  std::filesystem::path manifest_path;
  std::map<std::string, std::string> defaults;
  // Package id of the manifest that was loaded (the Huihui id until a
  // manifest names another approved set).
  std::string package_id{bubo::prepare::kPackageId};
  std::vector<PackageFile> files;
  std::uint64_t hashed_files = 0, hashed_bytes = 0;
  bool stamp_save_failed = false;
  // Lines for the serve log, for example a plain catalogue model whose vision
  // projector is not prepared (serve then starts text only).
  std::vector<std::string> notes;
  std::string MainIdentity() const;
  std::string IdentityFor(const std::filesystem::path&) const;
};
// Shared by validated manifests and catalogue planning before files exist.
std::string_view ModelArtefactSetting(bubo::prepare::Kind kind);
// Only bubo-package.json is discovered, beside the supplied main shard.
// Explicit selection must use the same filename. A selected broken manifest
// never falls back. Junctions are permitted; containment is lexical.
// A plain catalogue model (not Flash Next) lives in <root>\<id>\. Its
// package marker (package_id = the catalogue id) is checked against the
// compiled catalogue pins. With or without the marker, a vision projector
// that is present beside the model with its pinned size and a readable
// header becomes the "--mmproj" default; a missing one adds a note.
ModelPackage LoadModelPackage(const std::filesystem::path& model_path,
                              const std::string& profile, bool verify = false,
                              std::ostream* progress = nullptr,
                              const std::filesystem::path& selected = {});
// Validate selected overrides using the corresponding compiled package pin.
// Held read leases keep identity stable until the engine has opened its maps.
void CheckPackageOverrides(ModelPackage& package,
                           const std::map<std::string, std::string>& paths,
                           bool verify, std::ostream* progress = nullptr);
// Verify only missing identities for the main shards and selected artefacts.
// Empty packages require preparation or an explicit pinned manifest.
void EnsureDiskCacheIdentity(ModelPackage& package,
                             std::span<const std::filesystem::path> selected,
                             std::ostream* progress = nullptr);
#ifdef BUBO_PREPARE_TESTING
ModelPackage LoadModelPackageForTesting(
    const std::filesystem::path& model_path, const std::string& profile,
    std::span<const bubo::prepare::Entry> entries, bool verify = false,
    std::ostream* progress = nullptr,
    const std::function<bubo::prepare::FingerprintResult(
        const bubo::prepare::Entry&)>& fingerprints = {},
    const std::filesystem::path& selected = {}, bool select_set = false);
#endif
}  // namespace gufo::server
