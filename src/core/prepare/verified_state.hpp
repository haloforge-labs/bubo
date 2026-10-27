#pragma once

#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <utility>

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
using VerifiedStamps = std::map<std::pair<std::string, std::string>, FileStamp>;

inline VerifiedStamps ReadVerifiedStamps(
    const std::filesystem::path& root,
    std::string_view package_id = kPackageId) {
  try {
    const auto state = metadata::Parse(metadata::Read(
        NativePath(root / ".bubo" / "verified-state.json")));
    metadata::Keys(state, {"schema_version", "package_id", "files"});
    if (metadata::Integer(metadata::Required(state, "schema_version")) != 1 ||
        metadata::String(metadata::Required(state, "package_id")) != package_id)
      return {};
    const auto& files = metadata::Required(state, "files");
    if (!files.is_array())
      return {};
    VerifiedStamps stamps;
    for (const auto& row : files.items()) {
      metadata::Keys(row, {"file", "sha256", "size", "modified", "identity"});
      const auto name = metadata::String(metadata::Required(row, "file"));
      const auto hash = metadata::String(metadata::Required(row, "sha256"));
      FileStamp stamp{metadata::Integer(metadata::Required(row, "size")),
                      metadata::String(metadata::Required(row, "modified")),
                      metadata::String(metadata::Required(row, "identity"))};
      if (name.empty() || !IsLowerHex(hash, 64) || stamp.modified.empty() ||
          stamp.identity.empty() ||
          !stamps.emplace(std::make_pair(name, hash), std::move(stamp)).second)
        return {};
    }
    return stamps;
  } catch (...) {
    return {};  // Missing or invalid metadata authorizes no payload identity.
  }
}

// Callers supply only snapshots held continuously since a matching full hash.
// Merge under a separate lease so concurrent verifiers preserve each other's
// rows without blocking model reads or the preparation ownership lock.
inline bool SaveVerifiedStamps(const std::filesystem::path& root,
                               const VerifiedStamps& verified,
                               std::string& error,
                               std::string_view package_id = kPackageId) {
  if (verified.empty())
    return true;
  try {
    const auto directory = NativePath(root / ".bubo");
    std::filesystem::create_directories(directory);
    PrepareLock lock;
    if (!lock.Acquire(directory / "verified-state.lock", error))
      return false;
    auto stamps = ReadVerifiedStamps(root, package_id);
    for (const auto& [key, stamp] : verified)
      stamps[key] = stamp;
    auto state = metadata::Value::object(), files = metadata::Value::array();
    state["schema_version"] = 1;
    state["package_id"] = std::string(package_id);
    for (const auto& [key, stamp] : stamps) {
      auto row = metadata::Value::object();
      row["file"] = key.first;
      row["sha256"] = key.second;
      row["size"] = static_cast<unsigned long long>(stamp.size);
      row["modified"] = stamp.modified;
      row["identity"] = stamp.identity;
      files.push_back(std::move(row));
    }
    state["files"] = std::move(files);
    return AtomicWrite(directory / "verified-state.json", state.dump() + "\n",
                       error);
  } catch (const std::exception& failure) {
    error = failure.what();
    return false;
  }
}
}  // namespace bubo::prepare
