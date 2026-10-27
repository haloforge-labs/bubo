#include <cassert>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/core/prepare/lowmemory_data.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/release_evidence.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

int main(int argc, char* argv[]) {
  using namespace bubo::prepare;
  const auto root =
      std::filesystem::u8path(argc == 3 ? argv[2] : BUBO_PREPARE_FIXTURES);
  std::vector<SourceEvidence> evidence;
  try {
    for (auto name : {"huihui.json", "unsloth.json", "bubo.json"}) {
      const auto loaded = LoadSourceEvidence(metadata::Read(root / name));
      if (!loaded)
        throw std::invalid_argument(loaded.error);
      evidence.push_back(loaded.evidence);
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: fixture evidence: " << error.what() << '\n';
    return 1;
  }
  const auto baseline = ValidateReleaseEvidence(fixture::kCatalogue, evidence);
  if (argc == 3 && std::string_view(argv[1]) == "--validate-fixture") {
    std::cout << (baseline.empty()
                      ? "PASS: fixture release gate\n"
                      : "FAIL: fixture release gate: " + baseline + "\n");
    return baseline.empty() ? 0 : 1;
  }
  assert(argc == 1 && baseline.empty());
  // Each of the three independent source repositories must have evidence;
  // no other repository's valid metadata can stand in for the missing one.
  for (std::size_t repo = 0; repo < evidence.size(); ++repo) {
    auto missing = evidence;
    missing.erase(missing.begin() + repo);
    assert(ValidateReleaseEvidence(fixture::kCatalogue, missing)
               .find("missing evidence for source repository") !=
           std::string::npos);
  }
  for (std::size_t row = 0; row < fixture::kCatalogue.size(); ++row) {
    auto empty = fixture::kCatalogue;
    empty[row].commit = "";
    assert(ValidateReleaseEvidence(empty, evidence).find("empty PIN SLOT") !=
           std::string::npos);
  }
  for (int mutation = 0; mutation < 11; ++mutation) {
    auto entries = fixture::kCatalogue;
    auto broken = evidence;
    switch (mutation) {
      case 0:
        entries[0].commit = "";
        break;
      case 1:
        entries[2].source_repo = "PIN_SLOT_BUBO_REPO_ID";
        break;
      case 2:
        broken[0].files[0].lfs_oid = std::string(64, '0');
        break;
      case 3:
        broken.erase(broken.begin());
        break;
      case 4:
        broken[1].files[0].size = 4;
        break;
      case 5:
        broken[2].commit = "4444444444444444444444444444444444444444";
        break;
      case 6:
        broken[0].files[0].path = "other.gguf";
        break;
      case 7:
        entries[1].commit = "main";
        break;
      case 8:
        broken.push_back(broken[0]);
        break;
      case 9:
        broken[0].files.push_back(broken[0].files[0]);
        break;
      case 10:
        entries[0].licence_id = "MIT";
        break;
    }
    assert(!ValidateReleaseEvidence(entries, broken).empty());
  }
  const auto valid = metadata::Read(root / "huihui.json");
  for (auto invalid :
       {"{}", "[]", "{\"schema_version\":1,\"schema_version\":1}",
        "{\"schema_version\":2}"})
    assert(!LoadSourceEvidence(invalid));
  auto malformed = valid;
  auto replace = [&](const std::string& from, const std::string& to) {
    const auto pos = malformed.find(from);
    assert(pos != malformed.npos);
    malformed.replace(pos, from.size(), to);
  };
  replace("\"size\": 3", "\"size\": 3.5");
  assert(!LoadSourceEvidence(malformed));
  malformed = valid;
  replace("\"size\": 3", "\"size\": 9007199254740992");
  assert(!LoadSourceEvidence(malformed));
  assert(!LoadSourceEvidence(std::string(metadata::kMaxJsonBytes + 1, ' ')));
  const auto local = metadata::Read(
      root / "../../../tools/release/prepare-evidence/lowmemory-local.json");
  assert(ValidateLowMemoryLocalEvidence(local).empty());
  const auto proof = metadata::Parse(local);
  for (std::size_t i = 0; i < kLowMemoryShards.size(); ++i) {
    const auto& row = metadata::Required(proof, "files").items()[i];
    assert(row.member_str("name") == kLowMemoryShards[i].file_name);
    assert(row.member_str("local_source_name") ==
           kLowMemoryShards[i].local_source_name);
  }
  for (const auto* key : {"compose_manifest_sha256", "huihui_revision",
                          "ista_revision", "evidence_kind"}) {
    auto changed = metadata::Parse(local);
    changed[key] = "wrong";
    assert(!ValidateLowMemoryLocalEvidence(changed.dump()).empty());
  }
  for (const auto* key : {"name", "local_source_name", "sha256", "size"}) {
    auto changed = metadata::Parse(local);
    auto rows = gufo::json::Value::array();
    for (std::size_t i = 0; i < changed["files"].size(); ++i) {
      auto row = changed["files"].items()[i];
      if (i == 0)
        row[key] = key == std::string_view("size") ? gufo::json::Value(1)
                                                   : gufo::json::Value("wrong");
      rows.push_back(std::move(row));
    }
    changed["files"] = std::move(rows);
    assert(!ValidateLowMemoryLocalEvidence(changed.dump()).empty());
  }
  auto duplicate = metadata::Parse(local);
  auto rows = gufo::json::Value::array();
  for (std::size_t i = 0; i < duplicate["files"].size(); ++i)
    rows.push_back(duplicate["files"].items()[i == 1 ? 0 : i]);
  duplicate["files"] = std::move(rows);
  assert(!ValidateLowMemoryLocalEvidence(duplicate.dump()).empty());
  std::cout
      << "PASS: lowmemory local composition evidence and nine corruptions\n";
  std::cout << "PASS: release gate fixture for all three source repos; empty "
               "slot, sentinel, oid, size, path, commit, mapping and malformed "
               "metadata rejected\n";
}
