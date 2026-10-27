// Selectable model sets (huihui, original, lowmemory) through one prepare path.
// Covers the catalogue pins, per-set fingerprint tables and legal bundles,
// the generated consent text (golden bytes for BOTH sets; the Huihui golden is
// the pre-change output), the --model-set option, and one end-to-end prepare of
// the original set with fixture files (package id, root, state and manifest).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/prepare/model_prepare.hpp"
#include "src/cli/serve/model_manifest.hpp"
#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/lowmemory_data.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/model_notices.hpp"
#include "src/core/prepare/transport.hpp"
#include "src/core/prepare/verified_state.hpp"

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
constexpr std::string_view kAbc =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
int checks = 0;
bool print_plans = false;  // --print-plans: show each set's preview
void Require(bool condition, std::string_view message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(std::string(message));
}
std::string Normalize(std::string text) {
  std::replace(text.begin(), text.end(), '\\', '/');
  text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
  return text;
}
std::string ReadAll(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  Require(bool(file), "cannot read " + path.string());
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}
std::vector<std::string> Lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);)
    lines.push_back(line);
  return lines;
}
std::string Consent(ModelSet set) {
  const auto& entries = ModelCatalogue(set);
  std::uint64_t total = 0;
  for (const auto& entry : entries)
    total += entry.size;
  const std::vector<std::string> status(entries.size(), "download");
  const std::vector<ConsentVolume> volumes = {
      {"fixture", 400000000000ULL, 116619283648ULL}};
  return Normalize(ModelConsent(entries, "M", status, total, volumes, true));
}
class Fake final : public Transport {
public:
  int requests = 0;
  TransferResult Perform(const Request&, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    ++requests;
    const Response response{200,
                            {{"content-length", "3"},
                             {"etag", "\"abc\""},
                             {"x-linked-etag", std::string(kAbc)},
                             {"x-linked-size", "3"}}};
    if (!headers(response))
      return {};
    const std::array<std::uint8_t, 3> data = {'a', 'b', 'c'};
    body(data);
    return {};
  }
};
// Synthetic rows in the layout of the original set (same kinds and paths as
// the production rows, tiny 3-byte payloads). Never a production pin.
constexpr std::array<Entry, 8> kFixture = {
    {{"main_1", "one.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/one.gguf",
      Kind::MainShard, "qwen-community-1.0", "one.gguf",
      "fingerprints/base_one.json"},
     {"main_2", "two.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/two.gguf",
      Kind::MainShard, "qwen-community-1.0", "two.gguf",
      "fingerprints/base_two.json"},
     {"main_3", "three.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/three.gguf",
      Kind::MainShard, "qwen-community-1.0", "three.gguf",
      "fingerprints/base_three.json"},
     {"main_4", "four.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/four.gguf",
      Kind::MainShard, "qwen-community-1.0", "four.gguf",
      "fingerprints/base_four.json"},
     {"mtp", "mtp.gguf", 3, kAbc, "fixture/side",
      "2222222222222222222222222222222222222222", "MTP/mtp.gguf", Kind::Mtp,
      "qwen-community-1.0", "mtp/mtp.gguf"},
     {"mmproj", "mmproj.gguf", 3, kAbc, "fixture/side",
      "2222222222222222222222222222222222222222", "mmproj.gguf", Kind::Mmproj,
      "qwen-community-1.0", "mmproj/mmproj.gguf"},
     {"twin", "twin.gguf", 3, kAbc, "fixture/bubo",
      "3333333333333333333333333333333333333333", "twin-base/twin.gguf",
      Kind::Twin, "qwen-community-1.0", "twin/twin.gguf"},
     {"draft_head", "head.gguf", 3, kAbc, "fixture/bubo",
      "3333333333333333333333333333333333333333", "draft_head/head.gguf",
      Kind::DraftHead, "qwen-community-1.0", "draft_head/head.gguf"}}};

void CatalogueAndPins() {
  Require(!ModelSetFromName("hybrid") &&
              !ModelSetFromPackageId("huihui-qwen38-flash-next-ista-hybrid-v1"),
          "retired set and package names do not resolve");
  const auto lowmemory = ModelSetFromName("lowmemory");
  Require(lowmemory.has_value(), "lowmemory set exists");
  Require(ModelSetFromPackageId("huihui-qwen38-flash-next-lowmemory-v1") ==
                  lowmemory &&
              ModelSetName(*lowmemory) == "lowmemory",
          "lowmemory identity round trips");
  const auto& lowmemory_rows = ModelCatalogue(*lowmemory);
  Require(kLowMemoryRepository ==
                  "HaloForgeLabs/bubo-qwen3.8-flash-next-lower-memory" &&
              kLowMemoryCommit == "2f4f068840c7d85c7b0d8ac044db9e410687d636",
          "release hosting pins the final revision with downloads on");
  for (std::size_t i = 0; i < kLowMemoryShards.size(); ++i) {
    const auto suffix = "0000" + std::to_string(i + 1) + "-of-00004.gguf";
    const auto remote = "Qwen3.8-Flash-Next-Huihui-LowMemory-" + suffix;
    const auto local = "Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-" + suffix;
    const auto& pin = kLowMemoryShards[i];
    const auto& row = lowmemory_rows[i];
    Require(pin.file_name == remote && pin.local_source_name == local,
            "data maps each uploaded LowMemory shard to its local source");
    Require(row.file_name == remote && row.repo_path == remote &&
                row.destination == remote &&
                row.source_repo == kLowMemoryRepository &&
                row.commit == kLowMemoryCommit,
            "catalogue uses uploaded names for source and destination");
  }
  Require(lowmemory_rows.size() == 8 &&
              RecipeId(*lowmemory) ==
                  "huihui-lowmemory-iq4xs-nofullattn-nohc-q4k-v1",
          "lowmemory keeps the eight roles and has its own recipe");
  for (std::size_t i = 0; i < lowmemory_rows.size(); ++i) {
    const auto& row = lowmemory_rows[i];
    const auto& standard = ModelCatalogue()[i];
    Require(
        row.kind == standard.kind && row.logical_name == standard.logical_name,
        "lowmemory roles match Huihui");
    if (i >= 4)
      Require(
          row.size == standard.size && row.sha256 == standard.sha256 &&
              row.source_repo == standard.source_repo &&
              row.commit == standard.commit &&
              row.repo_path == standard.repo_path &&
              row.destination == standard.destination,
          "lowmemory auxiliary pins are exactly Huihui, including the twin");
    else
      Require(row.sha256 != standard.sha256 &&
                  row.fingerprint_table_slot ==
                      std::string("fingerprints/lowmemory_") +
                          std::string(row.logical_name) + ".json" &&
                  ProductionFingerprintTable(row),
              "lowmemory replacement shards have independent generated "
              "fingerprints");
  }
  const auto& huihui = ModelCatalogue(ModelSet::Huihui);
  const auto& original = ModelCatalogue(ModelSet::Original);
  // The default overload is the Huihui set, byte for byte.
  Require(&ModelCatalogue() == &huihui, "default catalogue is the Huihui set");
  for (const auto set : {ModelSet::Huihui, ModelSet::Original}) {
    Require(ValidateFields(ModelCatalogue(set), true).empty(),
            "release fields");
    Require(ValidateFields(ModelCatalogue(set), false).empty(),
            "development fields");
  }
  std::uint64_t all = 0, shards = 0;
  for (const auto& entry : huihui) {
    all += entry.size;
    if (entry.kind == Kind::MainShard)
      shards += entry.size;
  }
  Require(all == 116617136608ULL && shards == 111334655392ULL,
          "Huihui totals unchanged");
  all = shards = 0;
  for (const auto& entry : original) {
    all += entry.size;
    if (entry.kind == Kind::MainShard) {
      shards += entry.size;
      Require(entry.fingerprint_table_slot ==
                  std::string("fingerprints/base_") +
                      std::string(entry.logical_name) + ".json",
              "original fingerprint slot");
      Require(entry.source_repo == "unsloth/Qwen3.8-Flash-Next-GGUF" &&
                  entry.commit == PIN_SLOT_UNSLOTH_COMMIT,
              "original shards come from the pinned unsloth revision");
    }
  }
  Require(original.size() == 8 && all == 116617136000ULL &&
              shards == 111334654784ULL,
          "original totals");
  Require(original[6].commit == kOriginalBuboCommit &&
              original[7].commit == kOriginalBuboCommit &&
              original[6].repo_path == "twin-base/iq4-nofullattn-nohc.gguf" &&
              original[6].destination == huihui[6].destination &&
              original[6].sha256 ==
                  "d6aa3c0425247aca2ae37680a96a401ba09431addc552a1ee60af3a51927"
                  "e13e",
          "original twin row");
  for (std::size_t row : {4u, 5u, 7u})
    Require(original[row].sha256 == huihui[row].sha256 &&
                original[row].size == huihui[row].size,
            "mtp, mmproj and draft head are the same files in both sets");
  for (std::size_t row : {0u, 1u, 2u, 3u, 6u})
    Require(original[row].sha256 != huihui[row].sha256,
            "shard and twin hashes differ between sets");
  Require(PackageId(ModelSet::Huihui) == kPackageId &&
              PackageId(ModelSet::Original) ==
                  "qwen38-flash-next-base-ud-q4-k-xl-v1" &&
              RecipeId(ModelSet::Huihui) == kRecipeId,
          "package ids");
  Require(ModelSetFromName("huihui") == ModelSet::Huihui &&
              ModelSetFromName("original") == ModelSet::Original &&
              !ModelSetFromName("Original") && !ModelSetFromName("") &&
              !ModelSetFromName("base") &&
              ModelSetFromPackageId(kOriginalPackageId) == ModelSet::Original &&
              !ModelSetFromPackageId("other"),
          "set names");
  Require(ModelSetName(ModelSet::Original) == "original" &&
              ModelSetName(ModelSet::Huihui) == "huihui",
          "set name text");
  // Mutated rows of the original set stay rejected by the same field checks.
  auto copy = original;
  copy[0].sha256 = "bad";
  Require(!ValidateFields(copy, true).empty(), "bad sha rejected");
  copy = original;
  copy[6].commit = "";
  Require(!ValidateFields(copy, true).empty(), "empty commit rejected");
  copy = original;
  copy[1].fingerprint_table_slot = copy[0].fingerprint_table_slot;
  Require(!ValidateFields(copy, true).empty(), "duplicate slot rejected");
}

void FingerprintTables() {
  const std::array<std::size_t, 4> tensors = {0, 297, 752, 175};
  for (const auto set : {ModelSet::Huihui, ModelSet::Original}) {
    const auto& entries = ModelCatalogue(set);
    for (std::size_t i = 0; i < 4; ++i) {
      const auto table = ProductionFingerprintTable(entries[i]);
      Require(bool(table) && table.table.schema_version == 2,
              "every production shard has a complete v2 table");
      Require(table.table.file_sha256 == entries[i].sha256,
              "the table named for the set belongs to the set's shard");
      Require(table.table.tensors.size() == tensors[i], "tensor counts");
      Require(table.table.generator_version == "bubo-gguf-fingerprint-v2",
              "generator version");
    }
  }
  // A table of one set never verifies a shard of the other set.
  const auto huihui = ProductionFingerprintTable(ModelCatalogue()[1]);
  const auto original =
      ProductionFingerprintTable(ModelCatalogue(ModelSet::Original)[1]);
  Require(huihui && original &&
              huihui.table.header_sha256 != original.table.header_sha256 &&
              huihui.table.header_bytes != original.table.header_bytes,
          "the two sets have different tables");
  auto wrong = ModelCatalogue(ModelSet::Original)[1];
  wrong.sha256 = ModelCatalogue()[1].sha256;
  Require(!ProductionFingerprintTable(wrong),
          "a shard with the other set's name and hash mix is refused");
  auto unknown = ModelCatalogue()[1];
  unknown.logical_name = "main_9";
  Require(!ProductionFingerprintTable(unknown), "unknown shard has no table");
}

void Notices() {
  const auto& huihui = ModelNotices(ModelSet::Huihui);
  const auto& original = ModelNotices(ModelSet::Original);
  const auto& lowmemory = ModelNotices(*ModelSetFromName("lowmemory"));
  Require(
      lowmemory.size() == 3 && lowmemory[0].text == huihui[0].text &&
          lowmemory[1].name == "NOTICE" &&
          lowmemory[1].sha256 ==
              "76ca2a43af1ec76d88a0369d5385309b8bc03cf276c11df9fd42915a5acc897"
              "b" &&
          lowmemory[0].sha256 ==
              "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c0"
              "8" &&
          lowmemory[1].text.find("ISTA-DASLab/") != std::string_view::npos &&
          lowmemory[1].text.find("PLACEHOLDER") == std::string_view::npos,
      "release notice and licence bytes are pinned");
  Require(&ModelNotices() == &huihui, "default notices are the Huihui set");
  Require(huihui.size() == 3 && original.size() == 3, "three texts per set");
  constexpr std::string_view names[] = {"LICENSE", "NOTICE", "PROVENANCE.md"};
  // Frozen SHA-256 of the Huihui bundle (unchanged by J7) and the original
  // bundle (release draft, byte copy).
  constexpr std::string_view huihui_sha[] = {
      "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08",
      "1fb7de96edde06a2961da785037840e5ece18f977f92fcd7addc587d0e43f364",
      "658e26a9b4ac3de25b1d5feae5a4bc33e827ec6c06ebd38208bc9e0bbde807c1"};
  constexpr std::string_view original_sha[] = {
      "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08",
      "85c386c7fc866973beb8acdac1097572beb3acffdd2671b40f7ea3e01dcf7d9f",
      "1c8113b70fe77dcbc023a8a20498d48ed8785968f2d44fbda56621bf48c586dd"};
  for (std::size_t i = 0; i < 3; ++i) {
    Require(huihui[i].name == names[i] && original[i].name == names[i],
            "notice names");
    Require(huihui[i].sha256 == huihui_sha[i], "Huihui notice bytes frozen");
    Require(original[i].sha256 == original_sha[i], "original notice pins");
    Require(!original[i].text.empty(), "notice text");
  }
  Require(huihui[0].text == original[0].text, "same Qwen licence text");
  const auto contains = [](std::string_view text, std::string_view needle) {
    return text.find(needle) != text.npos;
  };
  Require(contains(huihui[1].text, "abliterat"),
          "Huihui NOTICE keeps its abliteration warning");
  Require(!contains(original[1].text, "abliteration") &&
              !contains(original[1].text, "Huihui") &&
              !contains(original[2].text, "Huihui") &&
              contains(original[1].text,
                       "766911a6b7369840a91dbcd95f9f997acaab6"
                       "cd6") &&
              contains(original[1].text,
                       "0a3c343bec2095f43f2906371c83e4faa8a47"
                       "da1"),
          "original NOTICE names Unsloth and the new revision, not Huihui");
  // Every size and hash in the original PROVENANCE table equals the catalogue.
  for (const auto& entry : ModelCatalogue(ModelSet::Original))
    Require(contains(original[2].text, entry.sha256),
            "original PROVENANCE lists every catalogue SHA-256");
}

void ConsentText() {
  const fs::path fixtures = BUBO_PREPARE_FIXTURES;
  const auto huihui = Consent(ModelSet::Huihui);
  const auto original = Consent(ModelSet::Original);
  Require(huihui == Normalize(ReadAll(fixtures / "model-consent-huihui.txt")),
          "Huihui consent bytes equal the pre-change golden");
  Require(
      original == Normalize(ReadAll(fixtures / "model-consent-original.txt")),
      "original consent bytes equal the golden");
  // No hand-written wording: the text is generated, so apart from the file
  // rows (up to the first total line) and the three total lines the two
  // texts are identical, line for line.
  const auto a = Lines(huihui), b = Lines(original);
  Require(a.size() == b.size(), "same line count");
  std::size_t rows_end = 0;
  while (rows_end < a.size() && !a[rows_end].starts_with("Total file size:"))
    ++rows_end;
  Require(rows_end > 8 && rows_end < a.size(), "file rows found");
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool generated_data =
        i >= 1 && (i < rows_end || a[i].starts_with("Total file size:") ||
                   a[i].starts_with("Total download size:") ||
                   a[i].starts_with("The main shards total"));
    if (!generated_data)
      Require(a[i] == b[i], "fixed sentences are identical: " + a[i]);
  }
  Require(a != b, "the sets show different files");
  // Long rows wrap at 80 columns: search the text without line breaks.
  std::string joined = original;
  joined.erase(std::remove(joined.begin(), joined.end(), '\n'), joined.end());
  Require(joined.find("unsloth/Qwen3.8-Flash-Next-GGUF/resolve/"
                      "766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/") !=
                  std::string::npos &&
              joined.find("huihui-ai") == std::string::npos &&
              joined.find("0a3c343bec2095f43f2906371c83e4faa8a47da1/"
                          "twin-base/") != std::string::npos,
          "original rows name the unsloth shards and the twin-base path");
  Require(huihui.find("huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF") !=
              std::string::npos,
          "Huihui rows unchanged");
}

void OptionParsing() {
  const auto parse = [](std::vector<const char*> args) {
    return Parse(args, "C:\\Users\\fixture\\AppData\\Local");
  };
  auto defaults = parse({"model", "--yes"});
  Require(defaults && defaults.options.model_set == ModelSet::Huihui &&
              !defaults.options.model_set_chosen,
          "default set is huihui");
  Require(Normalize(ModelRoot(defaults.options))
              .ends_with("/models/huihui-qwen38-flash-next-ud-q4-k-xl-v1"),
          "default model root unchanged");
  auto original = parse({"model", "--model-set", "original", "--yes"});
  Require(original && original.options.model_set == ModelSet::Original &&
              original.options.model_set_chosen,
          "original set parsed");
  Require(Normalize(ModelRoot(original.options))
              .ends_with("/models/qwen38-flash-next-base-ud-q4-k-xl-v1"),
          "original model root");
  auto huihui = parse({"--model-set", "huihui"});
  Require(huihui && huihui.options.model_set == ModelSet::Huihui &&
              huihui.options.model_set_chosen,
          "explicit huihui");
  for (auto bad : {std::vector<const char*>{"--model-set"},
                   {"--model-set", "base"},
                   {"--model-set", "Original"},
                   {"--model-set", ""},
                   {"--model-set", "original", "--model-set", "huihui"},
                   {"runtime", "--model-set", "original"}})
    Require(!parse(bad), "invalid --model-set use rejected");
  Require(bool(parse({"all", "--model-set", "original"})),
          "all step accepts it");
}

// End to end with fixture files: the original set is prepared into its own
// package folder with its own package id in the manifest, state and verified
// stamps. The Huihui folder is not touched.
void PrepareOriginalFixture(const fs::path& base) {
  Options options;
  options.step = Step::Model;
  options.yes = true;
  options.model_set = ModelSet::Original;
  options.model_set_chosen = true;
  options.root = (base / "root").string();
  Fake transport;
  ModelTestEnvironment env;
  env.entries = kFixture;
  env.package_id = kOriginalPackageId;
  env.recipe_id = kOriginalRecipeId;
  env.transport = &transport;
  env.notices = {{"LICENSE", "abc", std::string(kAbc)},
                 {"NOTICE", "abc", std::string(kAbc)},
                 {"PROVENANCE.md", "abc", std::string(kAbc)}};
  env.volume = [](const fs::path&) {
    return VolumeInfo{
        "fixture-volume", 16ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
  };
  env.fingerprint = [](const Entry& entry, const fs::path& path) {
    return HashFile(path) == entry.sha256 ? std::string{}
                                          : "synthetic fingerprint mismatch";
  };
  env.download_options.wait = [](std::chrono::seconds) {};
  std::istringstream input;
  std::ostringstream out, err;
  const auto code = RunModelForTesting(options, env, input, out, err);
  Require(code == 0,
          "fixture prepare of the original set: " + err.str() + out.str());
  const auto model = fs::u8path(ModelRoot(options));
  Require(model.filename() == "qwen38-flash-next-base-ud-q4-k-xl-v1",
          "original folder name");
  Require(
      !fs::exists(fs::path(options.root) / "models" / std::string(kPackageId)),
      "the Huihui package folder is not created");
  using namespace metadata;
  const auto manifest = Parse(Read(model / "bubo-package.json"));
  Require(String(Required(manifest, "package_id")) == kOriginalPackageId,
          "manifest package id");
  const auto& twin = Required(Required(manifest, "artefacts"), "twin");
  Require(String(Required(twin, "recipe_id")) == kOriginalRecipeId &&
              String(Required(twin, "file")) == "twin/twin.gguf",
          "manifest recipe id and twin row");
  const auto state = Parse(Read(model / ".bubo" / "prepare-state.json"));
  Require(String(Required(state, "package_id")) == kOriginalPackageId,
          "state package id");
  const auto verified = Parse(Read(model / ".bubo" / "verified-state.json"));
  Require(String(Required(verified, "package_id")) == kOriginalPackageId,
          "verified-state package id");
  Require(out.str().find("prepare: model prepared") != std::string::npos &&
              transport.requests == 8,
          "eight pinned downloads");
  // A second run is a no-op that needs no network.
  transport.requests = 0;
  std::ostringstream out2, err2;
  std::istringstream input2;
  Require(RunModelForTesting(options, env, input2, out2, err2) == 0 &&
              transport.requests == 0 &&
              out2.str().find("already prepared") != std::string::npos,
          "prepared original set is a no-op");
  // The same folder is not accepted as the Huihui package: its state names
  // another package id, so the Huihui run does not treat it as prepared.
  auto as_huihui = env;
  as_huihui.package_id = kPackageId;
  as_huihui.recipe_id = kRecipeId;
  std::ostringstream out3, err3;
  std::istringstream input3;
  Require(RunModelForTesting(options, as_huihui, input3, out3, err3) != 0 ||
              out3.str().find("already prepared") == std::string::npos,
          "state of one package id never marks another as prepared");
}

// The production catalogue and legal bundle of each set pass the real
// pre-flight (field checks, notice hashes, plan, consent) and produce a plan
// without any request or write. A fake volume gives enough space.
void ProductionPreview(const fs::path& base) {
  for (const auto set : {ModelSet::Huihui, ModelSet::Original}) {
    Options options;
    options.step = Step::Model;
    options.machine = true;
    options.model_set = set;
    options.model_set_chosen = true;
    options.root =
        (base / ("preview-" + std::string(ModelSetName(set)))).string();
    ModelTestEnvironment env;
    env.entries = ModelCatalogue(set);
    env.package_id = PackageId(set);
    env.recipe_id = RecipeId(set);
    for (const auto& notice : ModelNotices(set))
      env.notices.push_back({std::string(notice.name), std::string(notice.text),
                             std::string(notice.sha256)});
    env.volume = [](const fs::path&) {
      return VolumeInfo{"fixture-volume", 400000000000ULL, "NTFS", false, {}};
    };
    env.fingerprint = [](const Entry&, const fs::path&) {
      return std::string("no model file exists in this test");
    };
    PreparationPlan plan;
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunModelForTesting(options, env, input, out, err, &plan) == 0,
            "production preview: " + err.str());
    Require(!plan.ready && !fs::exists(options.root),
            "a preview writes nothing");
    if (print_plans)
      std::cout << "=== preview " << ModelSetName(set) << " ===\n"
                << plan.text << "--- machine plan ---\n"
                << plan.machine.dump() << "\n";
    std::uint64_t total = 0;
    for (const auto& entry : env.entries)
      total += entry.size;
    const std::vector<std::string> status(env.entries.size(), "download");
    Require(Normalize(plan.text) ==
                Normalize(ModelConsent(env.entries, ModelRoot(options), status,
                                       total, plan.volumes, true)),
            "the plan text is the generated consent text");
    const auto& sources = plan.machine.find("sources")->items();
    Require(sources.size() == 8, "eight sources in the plan");
    for (std::size_t i = 0; i < 8; ++i) {
      Require(
          sources[i].member_str("repository") == env.entries[i].source_repo &&
              sources[i].member_str("revision") == env.entries[i].commit &&
              sources[i].member_str("sha256") == env.entries[i].sha256 &&
              sources[i].member_str("url").find(
                  std::string(env.entries[i].commit)) != std::string::npos,
          "plan source row matches the catalogue");
    }
    const auto& licences = plan.machine.find("licenses")->items();
    Require(licences.size() == 3 &&
                licences[1].member_str("path") == "NOTICE" &&
                licences[1].member_str("text") == ModelNotices(set)[1].text,
            "the plan carries this set's legal texts");
    Require(Normalize(plan.machine.find("paths")->member_str("model"))
                .ends_with("/models/" + std::string(PackageId(set))),
            "plan model path uses the set's package folder");
  }
}

// The production loader chooses the approved set from the manifest's package
// id. Bogus bodies stop at later checks; the first failing check shows which
// catalogue was applied.
void LoaderSelectsSet(const fs::path& base) {
  using Value = gufo::json::Value;
  const auto try_load = [&](const std::string& package_id, bool select,
                            std::string repo, std::string revision) {
    const auto directory = base / ("loader-" + package_id);
    fs::create_directories(directory);
    auto manifest = Value::object();
    manifest["schema_version"] = 1;
    manifest["package_id"] = package_id;
    manifest["engine_min"] = "0.1.0";
    auto model = Value::object(), source = Value::object();
    source["repository"] = repo;
    source["revision"] = revision;
    model["source"] = source;
    model["family"] = "qwen3.8-flash-next";
    model["files"] = Value::array();
    manifest["model"] = model;
    manifest["artefacts"] = Value::object();
    auto profiles = Value::array();
    profiles.push_back("default");
    manifest["profiles_supported"] = profiles;
    std::ofstream(directory / "bubo-package.json", std::ios::binary)
        << manifest.dump();
    try {
      gufo::server::LoadModelPackageForTesting(directory / "bubo-package.json",
                                               "default", ModelCatalogue(),
                                               false, nullptr, {}, {}, select);
    } catch (const std::exception& error) {
      return std::string(error.what());
    }
    return std::string("no error");
  };
  const std::string unsloth = "unsloth/Qwen3.8-Flash-Next-GGUF";
  const std::string unsloth_commit(PIN_SLOT_UNSLOTH_COMMIT);
  const auto contains = [](const std::string& text, std::string_view needle) {
    return text.find(needle) != std::string::npos;
  };
  // Original id + selection: the original catalogue applies, so the unsloth
  // source passes and the loader reaches the shard-count check.
  Require(contains(try_load(std::string(kOriginalPackageId), true, unsloth,
                            unsloth_commit),
                   "exactly four shards"),
          "original manifest uses the original catalogue");
  // Huihui id: the Huihui catalogue applies and the unsloth source is refused.
  Require(
      contains(try_load(std::string(kPackageId), true, unsloth, unsloth_commit),
               "model source entry mismatch"),
      "Huihui manifest still uses the Huihui catalogue");
  // Without selection (tests, other callers) the original id is refused.
  Require(contains(try_load(std::string(kOriginalPackageId), false, unsloth,
                            unsloth_commit),
                   "package_id entry mismatch"),
          "no set selection keeps the Huihui id rule");
  // Without an explicit manifest or verification, unknown markers are imports.
  Require(
      try_load("other-package", true, unsloth, unsloth_commit) == "no error",
      "unknown adjacent marker leaves a plain import");
  Require(contains(try_load("other-package", false, unsloth, unsloth_commit),
                   "package_id entry mismatch"),
          "non-catalogue testing mode keeps its explicit package pin");
  Require(contains(try_load(std::string(PackageId(ModelSet::LowMemory)), true,
                            std::string(kLowMemoryRepository),
                            std::string(kLowMemoryCommit)),
                   "exactly four shards"),
          "lowmemory manifest selects its compiled source and catalogue");
}

// Optional real-file CPU integration. The caller supplies an owned scratch
// root with all eight pinned files. No engine or transport is started.
void PrepareAndLoadLowMemory() {
  const char* root = std::getenv("BUBO_TEST_LOWMEMORY_ROOT");
  if (!root)
    return;
  const char* args[] = {"model",
                        "--catalogue-id",
                        "huihui-qwen38-flash-next-lowmemory-v1",
                        "--output",
                        root,
                        "--offline",
                        "--yes",
                        "--verify"};
  const auto parsed = Parse(args, root);
  Require(bool(parsed), "real lowmemory options parse");
  std::ostringstream out, err;
  Require(RunModel(parsed.options, out, err) == 0,
          "real local lowmemory prepare: " + err.str());
  const auto directory = fs::u8path(ModelRoot(parsed.options));
  const auto document =
      metadata::Parse(metadata::Read(directory / "bubo-package.json"));
  Require(document.member_str("engine_min") == "0.1.0",
          "lowmemory uses the unchanged Huihui engine_min policy");
  Require(
      ReadVerifiedStamps(directory, PackageId(ModelSet::LowMemory)).size() == 8,
      "full local verification writes eight genuine SHA-256 stamps");
  const auto loaded = gufo::server::LoadModelPackage(
      directory / ModelCatalogue(ModelSet::LowMemory).front().destination,
      "default", false, nullptr, directory / "bubo-package.json");
  Require(
      loaded.package_id == PackageId(ModelSet::LowMemory) &&
          loaded.files.size() == 8 && loaded.hashed_bytes == 0 &&
          std::all_of(loaded.files.begin(), loaded.files.end(),
                      [](const auto& file) { return file.observed_verified; }),
      "serve's engine-free loader accepts the prepared lowmemory package");
  std::cout
      << "PASS: local lowmemory prepare -> production serve manifest loader; "
         "8 files, engine_min 0.1.0, no engine start\n";
}
}  // namespace

int main(int argc, char** argv) {
  print_plans = argc == 2 && std::string_view(argv[1]) == "--print-plans";
  try {
    CatalogueAndPins();
    FingerprintTables();
    Notices();
    ConsentText();
    OptionParsing();
    const auto base =
        fs::temp_directory_path() /
        ("bubo-model-set-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(!fs::exists(base), "fixture root must not exist");
    fs::create_directories(base);
    try {
      PrepareOriginalFixture(base);
      ProductionPreview(base);
      LoaderSelectsSet(base);
      PrepareAndLoadLowMemory();
    } catch (...) {
      fs::remove_all(base);  // Only the directory this test created above.
      throw;
    }
    fs::remove_all(base);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS: prepare_model_set_test " << checks << " checks\n";
  return 0;
}
