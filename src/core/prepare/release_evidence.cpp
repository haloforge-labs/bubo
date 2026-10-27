#include "src/core/prepare/release_evidence.hpp"

#include <set>
#include <stdexcept>

#include "src/core/prepare/lowmemory_data.hpp"
#include "src/core/prepare/metadata_json.hpp"

namespace bubo::prepare {
EvidenceResult LoadSourceEvidence(std::string_view text) {
  EvidenceResult result;
  try {
    using namespace metadata;
    const auto root = Parse(text);
    Keys(root, {"schema_version", "repository", "commit", "tree"});
    if (Integer(Required(root, "schema_version")) != 1)
      throw std::invalid_argument("unsupported source evidence schema");
    result.evidence.repository = String(Required(root, "repository"));
    result.evidence.commit = String(Required(root, "commit"));
    if (!IsLowerHex(result.evidence.commit, 40))
      throw std::invalid_argument("evidence commit must be 40 lowercase hex");
    const auto& tree = Required(root, "tree");
    if (!tree.is_array() || tree.empty() || tree.size() > 4096)
      throw std::invalid_argument("expected 1..4096 HF tree entries");
    std::set<std::string> paths;
    for (const auto& value : tree.items()) {
      const auto& path = String(Required(value, "path"));
      if (!IsRelativePath(path) || !paths.insert(path).second)
        throw std::invalid_argument("invalid/duplicate HF tree path");
      const auto& type = String(Required(value, "type"));
      if (type == "directory")
        continue;
      if (type != "file")
        throw std::invalid_argument("unknown HF tree entry type");
      const auto size = Integer(Required(value, "size"));
      const auto* lfs = value.find("lfs");
      // Cards/notices can be ordinary Git files; payload rows must have LFS.
      std::string oid;
      if (lfs && !lfs->is_null()) {
        oid = String(Required(*lfs, "oid"));
        if (!IsLowerHex(oid, 64))
          throw std::invalid_argument("LFS oid must be 64 lowercase hex");
        if (const auto* length = lfs->find("size"))
          if (Integer(*length) != size)
            throw std::invalid_argument("HF size differs from LFS size");
      }
      result.evidence.files.push_back({path, size, std::move(oid)});
    }
  } catch (const std::exception& error) {
    result.evidence = {};
    result.error = error.what();
  }
  return result;
}

std::string ValidateReleaseEvidence(std::span<const Entry> entries,
                                    std::span<const SourceEvidence> evidence) {
  if (const auto fields = ValidateFields(entries, true); !fields.empty())
    return fields;
  std::set<std::string_view> repos;
  for (const auto& source : evidence)
    if (!repos.insert(source.repository).second)
      return "duplicate source evidence repository";
  for (const auto& entry : entries) {
    const SourceEvidence* source = nullptr;
    for (const auto& candidate : evidence)
      if (candidate.repository == entry.source_repo)
        source = &candidate;
    const auto fail = [&](std::string_view reason) {
      return std::string(entry.logical_name) + ": " + std::string(reason);
    };
    if (!source)
      return fail("missing evidence for source repository");
    if (source->commit != entry.commit)
      return fail("source evidence commit differs from pin");
    const TreeFile* file = nullptr;
    for (const auto& candidate : source->files)
      if (candidate.path == entry.repo_path) {
        if (file)
          return fail("duplicate evidence path");
        file = &candidate;
      }
    if (!file)
      return fail("pinned path absent from HF tree evidence");
    if (file->size != entry.size)
      return fail("HF tree size differs from pin");
    if (file->lfs_oid != entry.sha256)
      return fail("HF tree lfs.oid differs from SHA-256 pin");
  }
  return {};
}
std::string ValidateLowMemoryLocalEvidence(std::string_view text) {
  try {
    using namespace metadata;
    const auto proof = Parse(text);
    Keys(proof, {"schema_version", "evidence_kind", "compose_manifest_sha256",
                 "huihui_revision", "ista_revision", "files"});
    if (Integer(Required(proof, "schema_version")) != 1 ||
        String(Required(proof, "evidence_kind")) != "local-composed-gguf" ||
        String(Required(proof, "compose_manifest_sha256")) !=
            kLowMemoryComposeManifestSha256 ||
        String(Required(proof, "huihui_revision")) != kHuihuiCommit ||
        String(Required(proof, "ista_revision")) != kLowMemoryIstaRevision)
      throw std::invalid_argument(
          "local composition provenance differs from its pins");
    const auto& files = Required(proof, "files");
    if (!files.is_array() || files.size() != kLowMemoryShards.size())
      throw std::invalid_argument("local composition requires four shard rows");
    for (std::size_t i = 0; i < kLowMemoryShards.size(); ++i) {
      const auto& row = files.items()[i];
      const auto& pin = kLowMemoryShards[i];
      Keys(row, {"name", "local_source_name", "size", "sha256"});
      if (String(Required(row, "name")) != pin.file_name ||
          String(Required(row, "local_source_name")) != pin.local_source_name ||
          Integer(Required(row, "size")) != pin.size ||
          String(Required(row, "sha256")) != pin.sha256)
        throw std::invalid_argument(
            "local composition shard identity differs from its pin");
    }
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}
}  // namespace bubo::prepare
