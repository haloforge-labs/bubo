#include "src/core/prepare/release_catalogue.hpp"

#include <algorithm>
#include <set>

#include "src/core/prepare/lowmemory_data.hpp"

namespace bubo::prepare {
namespace {
constexpr std::string_view kHuihui =
    "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF";
constexpr std::string_view kUnsloth = "unsloth/Qwen3.8-Flash-Next-GGUF";
constexpr std::string_view kLicence = "qwen-community-1.0";
// All eight rows copied from ADDENDUM-20261011-1750.md section 2.
// Bytes/hashes: tools/release/prepare-evidence (HF API listings at the pinned
// revisions) section 1 (main_1..4, twin), section 2 (draft_head), section 3
// (mtp, mmproj). Repos/commits/paths/destinations: addendum section 2; package
// ID: section 3.
constexpr std::array<Entry, 8> kModels = {
    {{"main_1", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      10946752ULL,
      "290a31ca20b74ed2c21b257d378a5678d821ad9fdd8feb1fc52c719761668903",
      kHuihui, kHuihuiCommit,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      "fingerprints/main_1.json"},
     {"main_2", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      49859583296ULL,
      "0ebf70d076e0e98f74dc414a90b92c67d7267876ae524bc7799cf463e27b754b",
      kHuihui, kHuihuiCommit,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      "fingerprints/main_2.json"},
     {"main_3", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      49376141664ULL,
      "e9b12f44e13e60d6ef9396449f7b4356b2e2911222b8183a596830c2418db19a",
      kHuihui, kHuihuiCommit,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      "fingerprints/main_3.json"},
     {"main_4", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      12087983680ULL,
      "983901b3e47686f56f5fcec87ba8c51e7e2c6996277dd77bc3fc804a1d563da3",
      kHuihui, kHuihuiCommit,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      "fingerprints/main_4.json"},
     {"mtp", "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256ULL,
      "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", Kind::Mtp, kLicence,
      "mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"},
     {"mmproj", "mmproj-BF16.gguf", 907542944ULL,
      "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT, "mmproj-BF16.gguf", Kind::Mmproj,
      kLicence, "mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf"},
     {"twin", "iq4-nofullattn-nohc.gguf", 1230788768ULL,
      "18e0831234551a4d14a0407c24598fb02fdb1fe83a20a4fcff23f410399c66c2",
      BUBO_HF_REPO_ID, PIN_SLOT_BUBO_COMMIT, "twin/iq4-nofullattn-nohc.gguf",
      Kind::Twin, kLicence, "twin/iq4-nofullattn-nohc.gguf"},
     {"draft_head", "draft-out-q4k.gguf", 357581248ULL,
      "6867948dc9f5c467818daa4281c4878dc64b53cd1bba9d16970fd77cd1656777",
      BUBO_HF_REPO_ID, PIN_SLOT_BUBO_COMMIT, "draft_head/draft-out-q4k.gguf",
      Kind::DraftHead, kLicence, "draft_head/draft-out-q4k.gguf"}}};
// Original set (not abliterated). Bytes/hashes: tools/release/prepare-evidence
// unsloth.json @ 766911a6 and bubo-0a3c343b.json @ 0a3c343b; prepare_gate
// --release checks every row against those trees. mtp, mmproj and draft_head
// are the same files as in the Huihui set. The twin is the base-model twin.
constexpr std::array<Entry, 8> kOriginalModels = {
    {{"main_1", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      10946624ULL,
      "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      "fingerprints/base_main_1.json"},
     {"main_2", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      49859583136ULL,
      "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      "fingerprints/base_main_2.json"},
     {"main_3", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      49376141504ULL,
      "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      "fingerprints/base_main_3.json"},
     {"main_4", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      12087983520ULL,
      "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      Kind::MainShard, kLicence,
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      "fingerprints/base_main_4.json"},
     {"mtp", "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256ULL,
      "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT,
      "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", Kind::Mtp, kLicence,
      "mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"},
     {"mmproj", "mmproj-BF16.gguf", 907542944ULL,
      "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355",
      kUnsloth, PIN_SLOT_UNSLOTH_COMMIT, "mmproj-BF16.gguf", Kind::Mmproj,
      kLicence, "mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf"},
     {"twin", "iq4-nofullattn-nohc.gguf", 1230788768ULL,
      "d6aa3c0425247aca2ae37680a96a401ba09431addc552a1ee60af3a51927e13e",
      BUBO_HF_REPO_ID, kOriginalBuboCommit,
      "twin-base/iq4-nofullattn-nohc.gguf", Kind::Twin, kLicence,
      "twin/iq4-nofullattn-nohc.gguf"},
     {"draft_head", "draft-out-q4k.gguf", 357581248ULL,
      "6867948dc9f5c467818daa4281c4878dc64b53cd1bba9d16970fd77cd1656777",
      BUBO_HF_REPO_ID, kOriginalBuboCommit, "draft_head/draft-out-q4k.gguf",
      Kind::DraftHead, kLicence, "draft_head/draft-out-q4k.gguf"}}};
// Only the four main shards differ. Copy the Huihui auxiliary rows verbatim,
// including its twin: there is no independently composed auxiliary file.
constexpr auto kLowMemoryModels = [] {
  auto entries = kModels;
  for (std::size_t i = 0; i < kLowMemoryShards.size(); ++i) {
    const auto& pin = kLowMemoryShards[i];
    entries[i].file_name = pin.file_name;
    entries[i].size = pin.size;
    entries[i].sha256 = pin.sha256;
    entries[i].source_repo = kLowMemoryRepository;
    entries[i].commit = kLowMemoryCommit;
    entries[i].repo_path = pin.file_name;
    entries[i].destination = pin.file_name;
    entries[i].fingerprint_table_slot = pin.fingerprint_slot;
  }
  return entries;
}();
}  // namespace
bool IsLowerHex(std::string_view text, std::size_t length) {
  return text.size() == length && std::ranges::all_of(text, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool IsRelativePath(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.back() == '/' ||
      path.find_first_of("\\:\r\n\t") != std::string_view::npos)
    return false;
  std::size_t pos = 0;
  do {
    const auto end = path.find('/', pos);
    const auto part = path.substr(pos, end == path.npos ? end : end - pos);
    if (part.empty() || part == "." || part == ".." || part.back() == '.' ||
        part.back() == ' ' || std::ranges::any_of(part, [](unsigned char c) {
          return c < 32 || c == 127 || c == '<' || c == '>' || c == '"' ||
                 c == '|' || c == '?' || c == '*';
        }))
      return false;
    auto stem = std::string(part.substr(0, part.find('.')));
    std::ranges::transform(stem, stem.begin(), [](unsigned char c) {
      return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A')
                                  : static_cast<char>(c);
    });
    if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" ||
        (stem.size() == 4 &&
         (stem.starts_with("COM") || stem.starts_with("LPT")) &&
         stem[3] >= '1' && stem[3] <= '9'))
      return false;
    if (end == path.npos)
      break;
    pos = end + 1;
  } while (true);
  return true;
}
namespace {
std::string FoldAscii(std::string_view value) {
  std::string folded(value);
  std::ranges::transform(folded, folded.begin(), [](unsigned char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a')
                                : static_cast<char>(c);
  });
  return folded;
}
bool Repo(std::string_view repo) {
  const auto slash = repo.find('/');
  return slash != repo.npos && repo.find('/', slash + 1) == repo.npos &&
         IsRelativePath(repo) && std::ranges::all_of(repo, [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' ||
                  c == '.';
         });
}
}  // namespace
const std::array<Entry, 8>& ModelCatalogue() {
  return kModels;
}
const std::array<Entry, 8>& ModelCatalogue(ModelSet set) {
  if (set == ModelSet::LowMemory)
    return kLowMemoryModels;
  return set == ModelSet::Original ? kOriginalModels : kModels;
}
std::string_view PackageId(ModelSet set) {
  if (set == ModelSet::LowMemory)
    return kLowMemoryPackageId;
  return set == ModelSet::Original ? kOriginalPackageId : kPackageId;
}
std::string_view RecipeId(ModelSet set) {
  if (set == ModelSet::LowMemory)
    return kLowMemoryRecipeId;
  return set == ModelSet::Original ? kOriginalRecipeId : kRecipeId;
}
std::string_view ModelSetName(ModelSet set) {
  if (set == ModelSet::LowMemory)
    return "lowmemory";
  return set == ModelSet::Original ? "original" : "huihui";
}
std::optional<ModelSet> ModelSetFromName(std::string_view name) {
  if (name == "huihui")
    return ModelSet::Huihui;
  if (name == "original")
    return ModelSet::Original;
  if (name == "lowmemory")
    return ModelSet::LowMemory;
  return std::nullopt;
}
std::optional<ModelSet> ModelSetFromPackageId(std::string_view package_id) {
  if (package_id == kPackageId)
    return ModelSet::Huihui;
  if (package_id == kOriginalPackageId)
    return ModelSet::Original;
  if (package_id == kLowMemoryPackageId)
    return ModelSet::LowMemory;
  return std::nullopt;
}

std::string ValidateFields(std::span<const Entry> entries, bool release) {
  if (entries.empty())
    return "empty catalogue";
  std::set<std::string> names, destinations, fingerprint_slots;
  for (const auto& entry : entries) {
    const auto fail = [&](std::string_view why) {
      return std::string(entry.logical_name) + ": " + std::string(why);
    };
    if (!entry.direct_url.empty())
      return fail("model catalogue cannot set an AMD origin");
    if (!IsRelativePath(entry.logical_name) ||
        entry.logical_name.find('/') != entry.logical_name.npos ||
        !names.insert(FoldAscii(entry.logical_name)).second)
      return fail("empty/duplicate logical name");
    if (!IsRelativePath(entry.file_name) ||
        entry.file_name.find('/') != entry.file_name.npos ||
        !IsRelativePath(entry.repo_path) ||
        !IsRelativePath(entry.destination) ||
        !destinations.insert(FoldAscii(entry.destination)).second)
      return fail("invalid/duplicate path");
    if (entry.repo_path.substr(entry.repo_path.find_last_of('/') + 1) !=
        entry.file_name)
      return fail("repo basename differs from file name");
    if (!entry.size || entry.size > 9007199254740991ULL ||
        !IsLowerHex(entry.sha256, 64))
      return fail("invalid size/sha256");
    if (entry.licence_id != kLicence)
      return fail("missing/unapproved licence mapping");
    if (!entry.commit.empty() && !IsLowerHex(entry.commit, 40))
      return fail("commit must be 40 lowercase hex");
    if (release && entry.commit.empty())
      return fail("empty PIN SLOT commit");
    if (entry.source_repo == "PIN_SLOT_BUBO_REPO_ID") {
      if (release)
        return fail("PIN_SLOT_BUBO_REPO_ID sentinel");
    } else if (!Repo(entry.source_repo))
      return fail("invalid repository ID");
    if (entry.kind < Kind::MainShard || entry.kind > Kind::DraftHead)
      return fail("invalid kind");
    if (entry.kind == Kind::MainShard) {
      if (!IsRelativePath(entry.fingerprint_table_slot) ||
          !fingerprint_slots.insert(FoldAscii(entry.fingerprint_table_slot))
               .second)
        return fail("missing/duplicate fingerprint table slot");
    } else if (!entry.fingerprint_table_slot.empty())
      return fail("fingerprint table belongs to a main shard");
  }
  return {};
}
}  // namespace bubo::prepare
