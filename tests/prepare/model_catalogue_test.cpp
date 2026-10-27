// J16: the model catalogue (loader, pin rules, shipped data) and
// `bubo prepare model --catalogue-id` (plan, licence text before download,
// verify, repair). Downloads come from an in-process fake transport: no
// network, no real file, no model.
#include "src/cli/model_catalogue.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/transport.hpp"

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
using bubo::cli::CatalogueEntry;
using bubo::cli::ParseCatalogue;
using bubo::cli::ShippedCatalogue;
using bubo::cli::ValidateCatalogue;
using gufo::json::Value;

constexpr std::string_view kAbc =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr std::string_view kCommit = "1111111111111111111111111111111111111111";
int checks = 0;
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
bool Contains(const std::string& text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

// One synthetic catalogue entry as JSON text. Each mutation below edits the
// parsed text before it is loaded again.
std::string EntryJson(bool enabled = true, bool verified = true,
                      const std::string& note = "",
                      const std::string& speed = "null",
                      const std::string& id = "fixture-model") {
  const auto file = [&](const std::string& name, const std::string& role) {
    return std::string("{\"name\":\"") + name + "\",\"repo_path\":\"" + name +
           "\",\"url\":\"https://huggingface.co/fixture/model/resolve/" +
           std::string(kCommit) + "/" + name + "\",\"bytes\":3,\"sha256\":\"" +
           std::string(kAbc) + "\",\"role\":\"" + role + "\"}";
  };
  return std::string(
             "{\"schema_version\":1,\"catalogue_version\":\"test\","
             "\"entries\":[{\"id\":\"") +
         id +
         "\",\"display_name\":\"Fixture model\",\"role\":\"supported\","
         "\"architecture\":\"qwen35\",\"set\":\"original\","
         "\"repository\":\"fixture/model\",\"commit\":\"" +
         std::string(kCommit) + "\",\"files\":[" + file("one.gguf", "main") +
         "," + file("config.json", "support") +
         "],\"licence\":{\"id\":\"Apache-2.0\",\"name\":\"Apache License "
         "2.0\",\"url\":\"https://www.apache.org/licenses/LICENSE-2.0\","
         "\"prompt_line\":\"License: Apache-2.0. Fixture model.\",\"note\":\"" +
         note + "\",\"verified_upstream\":" + (verified ? "true" : "false") +
         "},\"download_enabled\":" + (enabled ? "true" : "false") +
         (enabled ? "" : ",\"disabled_reason\":\"not pinned yet\"") +
         ",\"speed\":" + speed + "}]}";
}
std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto at = text.find(from);
  if (at == std::string::npos)
    throw std::runtime_error("test text not found: " + from);
  text.replace(at, from.size(), to);
  return text;
}

void CataloguePinRules() {
  const auto good = ParseCatalogue(EntryJson());
  Require(good.error.empty() && good.entries.size() == 1, "good entry loads");
  Require(bubo::cli::PinProblem(good.entries[0]).empty(), "good entry pins");

  struct Mutation {
    const char* name;
    std::string from, to;
  };
  const std::vector<Mutation> broken = {
      {"no sha256", std::string(kAbc), ""},
      {"short sha256", std::string(kAbc), "abc123"},
      {"upper-case sha256", std::string(kAbc),
       "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"},
      {"no bytes", "\"bytes\":3", "\"bytes\":0"},
      {"url from another commit", std::string(kCommit) + "/one.gguf",
       "2222222222222222222222222222222222222222/one.gguf"},
      {"url not https", "https://huggingface.co/fixture", "http://x/fixture"},
      {"no url",
       "\"url\":\"https://huggingface.co/fixture/model/resolve/" +
           std::string(kCommit) + "/one.gguf\"",
       "\"url\":\"\""},
      {"no commit", "\"commit\":\"" + std::string(kCommit) + "\"",
       "\"commit\":\"\""},
      {"short commit", "\"commit\":\"" + std::string(kCommit) + "\"",
       "\"commit\":\"abc\""},
      {"no licence id", "\"id\":\"Apache-2.0\"", "\"id\":\"\""},
      {"no licence link",
       "\"url\":\"https://www.apache.org/licenses/LICENSE-2.0\"",
       "\"url\":\"\""},
      {"licence link not https",
       "\"url\":\"https://www.apache.org/licenses/LICENSE-2.0\"",
       "\"url\":\"http://www.apache.org/licenses/LICENSE-2.0\""},
      {"no prompt line",
       "\"prompt_line\":\"License: Apache-2.0. Fixture "
       "model.\"",
       "\"prompt_line\":\"\""},
      {"unknown role", "\"role\":\"main\"", "\"role\":\"weights\""},
  };
  for (const auto& mutation : broken) {
    const auto text = Replace(EntryJson(), mutation.from, mutation.to);
    const auto load = ParseCatalogue(text);
    Require(!load.error.empty() && load.entries.empty(),
            std::string("download_enabled with a broken pin is refused: ") +
                mutation.name);
  }
  // The same broken pin is fine while the entry stays switched off with a
  // reason: such an entry can never be downloaded.
  auto disabled = Replace(EntryJson(false), std::string(kAbc), "");
  const auto off = ParseCatalogue(disabled);
  Require(off.error.empty() && off.entries.size() == 1 &&
              !off.entries[0].download_enabled &&
              off.entries[0].disabled_reason == "not pinned yet",
          "a disabled entry may be incomplete");
  Require(!bubo::cli::PinProblem(off.entries[0]).empty(),
          "the incomplete entry still reports its pin problem");
  // A disabled entry needs a reason; an enabled entry has none.
  Require(
      !ParseCatalogue(Replace(EntryJson(false),
                              ",\"disabled_reason\":\"not pinned yet\"", ""))
           .error.empty(),
      "disabled without a reason is refused");
  Require(!ParseCatalogue(Replace(EntryJson(true), "\"download_enabled\":true",
                                  "\"download_enabled\":true,"
                                  "\"disabled_reason\":\"x\""))
               .error.empty(),
          "enabled with a reason is refused");
  // A missing flag defaults to off, which then needs a reason.
  Require(!ParseCatalogue(
               Replace(EntryJson(true), "\"download_enabled\":true,", ""))
               .error.empty(),
          "a missing download_enabled flag means off and needs a reason");
  // Licence flag: not read upstream needs a note.
  Require(!ParseCatalogue(EntryJson(true, false)).error.empty(),
          "unverified licence without a note is refused");
  const auto noted = ParseCatalogue(
      EntryJson(true, false, "MIT, as stated by the model card"));
  Require(
      noted.error.empty() && noted.entries[0].download_enabled &&
          noted.entries[0].licence.note == "MIT, as stated by the model card",
      "unverified licence with a note is allowed");
  // Structure.
  Require(
      !ParseCatalogue("{\"schema_version\":2,\"entries\":[]}").error.empty(),
      "unknown schema version");
  Require(!ParseCatalogue(Replace(EntryJson(), "\"role\":\"supported\"",
                                  "\"role\":\"best\""))
               .error.empty(),
          "unknown entry role");
  Require(!ParseCatalogue(Replace(EntryJson(), "\"bytes\":3,",
                                  "\"bytes\":3,"
                                  "\"extra\":1,"))
               .error.empty(),
          "unknown file key");
  Require(!ParseCatalogue("not json").error.empty(), "not JSON");
  // Duplicate ids.
  {
    auto text = EntryJson();
    const auto inner = text.substr(text.find("{\"id\""));
    const auto entry = inner.substr(0, inner.rfind("]}"));
    Require(!ParseCatalogue(
                 Replace(text, "\"entries\":[", "\"entries\":[" + entry + ","))
                 .error.empty(),
            "repeated id");
  }
  // A file that is not downloadable does not need a download pin match.
  auto optional = Replace(EntryJson(), "\"role\":\"support\"",
                          "\"role\":\"support\",\"optional\":true,"
                          "\"downloadable\":false,"
                          "\"licence_text\":\"not yet read\"");
  const auto side = ParseCatalogue(optional);
  Require(side.error.empty() && !side.entries[0].files[1].downloadable &&
              side.entries[0].files[1].licence_text == "not yet read" &&
              bubo::cli::CatalogueDownloadBytes(side.entries[0]) == 3,
          "an optional side file with an unread licence is not downloaded");
  // A downloaded side file from another repository needs its own licence
  // line, and "not yet read" is not one.
  const auto foreign = [&](const std::string& licence) {
    const std::string other(40, '2');
    return Replace(
        EntryJson(),
        "\"url\":\"https://huggingface.co/fixture/model/resolve/" +
            std::string(kCommit) + "/config.json\"",
        "\"url\":\"https://huggingface.co/other/projector/resolve/" + other +
            "/config.json\",\"repository\":\"other/projector\",\"commit\":\"" +
            other + "\"" + licence);
  };
  Require(!ParseCatalogue(foreign("")).error.empty(),
          "a side file from another repository without a licence is refused");
  Require(!ParseCatalogue(foreign(",\"licence_text\":\"not yet read\""))
               .error.empty(),
          "an unread licence cannot be downloaded");
  Require(ParseCatalogue(foreign(",\"licence_text\":\"License: MIT. Other.\""))
              .error.empty(),
          "a side file with its own licence line is allowed");
}

void SpeedRules() {
  const std::string label = "measured on Strix Halo 128 GB, smoke-grade";
  const auto with = [&](const std::string& speed) {
    return ParseCatalogue(EntryJson(true, true, "", speed));
  };
  const auto complete =
      "{\"writing_tok_s\":50.5,\"reading_tok_s\":1000,\"prompt_tokens\":8192,"
      "\"source\":\"test\",\"label\":\"" +
      label + "\"}";
  const auto ok = with(complete);
  Require(ok.error.empty() && ok.entries[0].speed.present &&
              *ok.entries[0].speed.writing_tok_s == 50.5 &&
              *ok.entries[0].speed.reading_tok_s == 1000 &&
              *ok.entries[0].speed.prompt_tokens == 8192 &&
              ok.entries[0].speed.label == label,
          "complete speed loads");
  // Null for the whole object is fine and stays null in the JSON.
  const auto none = with("null");
  Require(none.error.empty() && !none.entries[0].speed.present &&
              bubo::cli::CatalogueEntryJson(none.entries[0])
                  .find("speed")
                  ->is_null(),
          "null speed");
  // A field that was not measured is null; the object is still valid.
  const auto partial = with(
      "{\"writing_tok_s\":67,\"reading_tok_s\":null,\"prompt_tokens\":null,"
      "\"source\":\"s\",\"label\":\"l\"}");
  Require(partial.error.empty() && !partial.entries[0].speed.reading_tok_s &&
              bubo::cli::CatalogueEntryJson(partial.entries[0])
                  .find("speed")
                  ->find("reading_tok_s")
                  ->is_null(),
          "not-measured field stays null");
  // Half-filled objects are refused: no source, no label, no number at all,
  // a zero or negative number, a string where a number belongs.
  for (const char* broken :
       {"{\"writing_tok_s\":5,\"source\":\"\",\"label\":\"l\"}",
        "{\"writing_tok_s\":5,\"source\":\"s\"}",
        "{\"source\":\"s\",\"label\":\"l\"}",
        "{\"writing_tok_s\":null,\"reading_tok_s\":null,\"source\":\"s\","
        "\"label\":\"l\"}",
        "{\"writing_tok_s\":0,\"source\":\"s\",\"label\":\"l\"}",
        "{\"writing_tok_s\":-3,\"source\":\"s\",\"label\":\"l\"}",
        "{\"writing_tok_s\":\"fast\",\"source\":\"s\",\"label\":\"l\"}",
        "{\"writing_tok_s\":5,\"prompt_tokens\":0,\"source\":\"s\","
        "\"label\":\"l\"}",
        "{\"writing_tok_s\":5,\"source\":\"s\",\"label\":\"l\",\"x\":1}"})
    Require(!with(broken).error.empty(),
            std::string("incomplete speed refused: ") + broken);
}

void StockComparisonRules() {
  const auto with = [&](const std::string& stock) {
    return ParseCatalogue(
        Replace(EntryJson(), "\"speed\":null",
                "\"speed\":null,\"stock_comparison\":" + stock));
  };
  Require(with("null").error.empty() &&
              !with("null").entries[0].stock_comparison.present,
          "null stock comparison");
  Require(with("{\"engine\":\"e\",\"prefill_pp8192\":5,\"decode_tg128\":6,"
               "\"source\":\"s\"}")
              .error.empty(),
          "complete stock comparison");
  for (const char* broken :
       {"{\"engine\":\"e\",\"prefill_pp8192\":5,\"source\":\"s\"}",
        "{\"engine\":\"\",\"prefill_pp8192\":5,\"decode_tg128\":6,\"source\":"
        "\"s\"}",
        "{\"engine\":\"e\",\"prefill_pp8192\":0,\"decode_tg128\":6,\"source\":"
        "\"s\"}",
        "{\"engine\":\"e\",\"prefill_pp8192\":5,\"decode_tg128\":6,\"source\":"
        "\"s\",\"x\":1}"})
    Require(!with(broken).error.empty(),
            std::string("incomplete stock comparison refused: ") + broken);
}

void ShippedData() {
  const auto& shipped = ShippedCatalogue();
  Require(shipped.error.empty(),
          "shipped catalogue is valid: " + shipped.error);
  Require(ValidateCatalogue(shipped.entries).empty(), "revalidates");
  Require(shipped.entries.size() == 11, "eleven shipped entries");
  const auto find = [&](std::string_view id) {
    const auto* entry = bubo::cli::FindCatalogueEntry(shipped.entries, id);
    Require(entry != nullptr, std::string("entry ") + std::string(id));
    return entry;
  };
  const auto* lowmemory = find("huihui-qwen38-flash-next-lowmemory-v1");
  Require(lowmemory->display_name ==
                  "Qwen3.8 Flash Next · HaloForge Labs lower-memory quant "
                  "(GSQ-RCO)" &&
              lowmemory->role == "supported" &&
              lowmemory->default_status == "none" &&
              !lowmemory->speed.present &&
              !lowmemory->stock_comparison.present &&
              lowmemory->default_note.empty() && lowmemory->download_enabled,
          "lowmemory is a published non-default download option");
  // Flash Next rows are the compiled pins, not a copy.
  for (const auto set : {ModelSet::Huihui, ModelSet::Original}) {
    const auto* entry = find(PackageId(set));
    const auto& rows = ModelCatalogue(set);
    Require(entry->role == "recommended" && entry->flash_next_set == set &&
                entry->download_enabled && entry->files.size() == rows.size(),
            "Flash Next entry shape");
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      total += rows[i].size;
      Require(entry->files[i].bytes == rows[i].size &&
                  entry->files[i].sha256 == rows[i].sha256 &&
                  entry->files[i].url == OriginUrl(rows[i]),
              "Flash Next file equals the compiled pin");
    }
    Require(bubo::cli::CatalogueDownloadBytes(*entry) == total,
            "Flash Next total");
  }
  Require(find(kPackageId)->default_status == "default" &&
              find(kOriginalPackageId)->default_status == "none" &&
              find(kPackageId)->set == "uncensored" &&
              find(kOriginalPackageId)->set == "original",
          "Flash Next defaults and sets");
  // Every release package is fully pinned and available for preparation.
  for (const auto& entry : shipped.entries) {
    Require(bubo::cli::PinProblem(entry).empty(), "pinned: " + entry.id);
    Require(entry.download_enabled && entry.disabled_reason.empty(),
            "downloadable: " + entry.id);
    Require(!entry.licence.prompt_line.empty(), "licence line: " + entry.id);
  }
  // Items the coordinator named.
  const auto* cyber = find("cyber-tiel-coder-35b-a3b-mtp-q4-k-m");
  Require(cyber->licence.prompt_line ==
                  "MIT, as stated by the model card; upstream LICENSE file "
                  "not found" &&
              cyber->licence.note == cyber->licence.prompt_line &&
              !cyber->licence.verified_upstream && cyber->download_enabled,
          "Cyber-Tiel carries the exact licence line");
  // V12-7: four makers' QAT builds, each complete in one repository.
  for (const auto id : {"gemma4-26b-a4b-unsloth-qat-ud-q4-k-xl",
                        "gemma4-31b-unsloth-qat-ud-q4-k-xl",
                        "gemma4-26b-a4b-huihui-abliterated-qat-q4-k",
                        "gemma4-31b-huihui-abliterated-qat-q4-k"}) {
    const auto* entry = find(id);
    Require(entry->files.size() == 3 && entry->download_enabled,
            "Gemma QAT model, vision and MTP");
    std::set<std::string> roles;
    for (const auto& file : entry->files) {
      roles.insert(file.role);
      Require(file.repository.empty() && file.commit.empty() &&
                  file.downloadable && !file.optional &&
                  file.url.starts_with("https://huggingface.co/" +
                                       entry->repository + "/resolve/" +
                                       entry->commit + "/"),
              "Gemma files use their own pinned repository");
    }
    Require(roles == std::set<std::string>{"main", "mmproj", "mtp"},
            "Gemma QAT capabilities");
    Require(entry->licence.id == "Apache-2.0" && !entry->estimate_note.empty(),
            "Gemma licence and unavailable estimate");
  }
  for (const auto& entry : shipped.entries) {
    Require(entry.id.find("hauhaucs") == std::string::npos &&
                entry.id.find("abliterated-v2") == std::string::npos,
            "removed Gemma rows stay import-only");
    const auto json = bubo::cli::CatalogueEntryJson(entry);
    Require(json.find("about") && !json.find("about")->str().empty() &&
                json.find("technical_name") &&
                !json.find("technical_name")->str().empty(),
            "every row has a CLI-owned info card");
    Require(json.find("gpu_bytes") && json.find("system_memory_bytes"),
            "unmeasured placement is explicitly null");
  }
  const auto* diffusion = find("diffusiongemma-26b-e38-abliterated-nvfp4");
  Require(diffusion->packaging == "safetensors_folder" &&
              diffusion->download_enabled,
          "NVFP4 folder includes the published converted vision file");
  Require(
      std::count_if(diffusion->files.begin(), diffusion->files.end(),
                    [](const auto& f) { return f.role == "main"; }) == 2 &&
          std::count_if(diffusion->files.begin(), diffusion->files.end(),
                        [](const auto& f) { return f.role == "mmproj"; }) == 1,
      "DiffusionGemma has two NVFP4 shards and a vision file");
  const auto* q38 = find("qwen38-27b-ud-q4-k-xl");
  Require(q38->files.size() == 3 && q38->files[1].optional &&
              !q38->files[1].downloadable &&
              q38->files[1].licence_text == "not yet read" &&
              bubo::cli::CatalogueDownloadBytes(*q38) ==
                  17559178144ULL + 931146432ULL,
          "DFlash2 draft is optional, unread and not downloaded");
  // E11: every plain GGUF model that reads images lists its BF16 projector
  // (HF LFS sha256 at the pinned revision; local copies checked 10-20).
  struct Projector {
    std::string_view id, repository, commit;
    std::uint64_t bytes;
    std::string_view sha256;
    std::string_view name = "mmproj-BF16.gguf";
    std::string_view licence = {};  // a same-repository line, if any
  };
  const Projector projectors[] = {
      {"qwen38-27b-ud-q4-k-xl", "", "", 931146432ULL,
       "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53"},
      {"qwen36-27b-ud-q4-k-xl", "", "", 931146304ULL,
       "05353347512982ee62317b9d8c89372bc815f4b4043580e7ef3ad411ec1a1cd3"},
      {"cyber-tiel-coder-35b-a3b-mtp-q4-k-m", "", "", 902822016ULL,
       "d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837"},
  };
  for (const auto& want : projectors) {
    const auto* entry = find(want.id);
    const bubo::cli::CatalogueFile* file = nullptr;
    for (const auto& row : entry->files)
      if (row.role == "mmproj")
        file = &row;
    Require(file != nullptr, "projector listed: " + entry->id);
    const auto repository = want.repository.empty()
                                ? entry->repository
                                : std::string(want.repository);
    const auto commit =
        want.commit.empty() ? entry->commit : std::string(want.commit);
    Require(file->name == want.name && file->downloadable && !file->optional &&
                file->bytes == want.bytes && file->sha256 == want.sha256 &&
                file->url == "https://huggingface.co/" + repository +
                                 "/resolve/" + commit + "/" +
                                 std::string(want.name),
            "projector pin: " + entry->id);
    // A projector from another repository carries its own licence line.
    Require(want.repository.empty()
                ? file->repository.empty() &&
                      (want.licence.empty()
                           ? file->licence_text.empty()
                           : Contains(file->licence_text, want.licence) &&
                                 Contains(file->licence_text, "Apache-2.0"))
                : file->repository == want.repository &&
                      file->commit == want.commit &&
                      Contains(file->licence_text, "Apache-2.0") &&
                      Contains(file->licence_text, "Unsloth"),
            "projector source and licence: " + entry->id);
  }
  const auto speed = [&](const CatalogueEntry* entry) {
    Require(entry->speed.present, "speed present: " + entry->id);
    Require(entry->speed.label == "measured on Strix Halo 128 GB, smoke-grade",
            "speed label");
    return entry->speed;
  };
  Require(
      *speed(cyber).writing_tok_s == 59 && *speed(cyber).reading_tok_s == 1655,
      "Cyber-Tiel speed");
  Require(*speed(q38).writing_tok_s == 11.4 && *speed(q38).reading_tok_s == 475,
          "Qwen3.8-27B speed");
  Require(*speed(find("qwen36-27b-ud-q4-k-xl")).writing_tok_s == 11.45 &&
              *speed(find("qwen36-27b-ud-q4-k-xl")).reading_tok_s == 389.5 &&
              *speed(find("qwen36-27b-ud-q4-k-xl")).prompt_tokens == 8329 &&
              find("qwen36-27b-ud-q4-k-xl")->estimate_note.empty(),
          "Qwen3.6-27B speed from the merged-runtime smoke, no smoke caveat");
  Require(*speed(find(kPackageId)).writing_tok_s == 54.30 &&
              *speed(find(kPackageId)).reading_tok_s == 1201,
          "Flash Next speed");
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

ModelTestEnvironment MakeEnvironment(Transport& transport,
                                     std::uint64_t free_bytes) {
  ModelTestEnvironment env;
  env.transport = &transport;
  env.volume = [free_bytes](const fs::path&) {
    return VolumeInfo{"fixture-volume", free_bytes, "NTFS", false, {}};
  };
  env.download_options.wait = [](std::chrono::seconds) {};
  return env;
}

void ParseOptions() {
  const auto parse = [](std::vector<const char*> args) {
    return Parse(args, "C:\\Users\\fixture\\AppData\\Local");
  };
  const auto flash = parse(
      {"model", "--catalogue-id", std::string(kOriginalPackageId).c_str()});
  Require(flash && flash.options.model_set == ModelSet::Original &&
              flash.options.model_set_chosen &&
              Normalize(ModelRoot(flash.options))
                  .ends_with("/models/qwen38-flash-next-base-ud-q4-k-xl-v1"),
          "a Flash Next id selects its model set and folder");
  const auto other =
      parse({"model", "--catalogue-id", "qwen36-27b-ud-q4-k-xl", "--plan"});
  Require(other && other.options.plan_only && !other.options.model_set_chosen &&
              Normalize(ModelRoot(other.options))
                  .ends_with("/models/qwen36-27b-ud-q4-k-xl"),
          "another id uses its own folder; --plan is parsed");
  Require(!parse({"model", "--catalogue-id", "no-such-model"}), "unknown id");
  Require(!parse({"model", "--catalogue-id"}), "missing id");
  Require(!parse({"model", "--catalogue-id", "qwen36-27b-ud-q4-k-xl",
                  "--catalogue-id", "qwen36-27b-ud-q4-k-xl"}),
          "id cannot repeat");
  Require(!parse({"model", "--catalogue-id", "qwen36-27b-ud-q4-k-xl",
                  "--model-set", "original"}),
          "--catalogue-id and --model-set exclude each other");
  Require(!parse({"runtime", "--catalogue-id", "qwen36-27b-ud-q4-k-xl"}),
          "runtime step rejects an id");
  Require(!parse({"runtime", "--plan"}) && !parse({"--plan"}),
          "--plan needs the model step");
  Require(!parse({"model", "--catalogue-id", "qwen36-27b-ud-q4-k-xl",
                  "--mmproj", "C:\\x.gguf"}),
          "side-file options apply to Flash Next only");
  // Existing parse results do not change.
  Require(parse({"model", "--yes"}) &&
              parse({"model", "--yes"}).options.catalogue_id.empty(),
          "plain model step unchanged");
}

void PrepareById(const fs::path& base) {
  const auto load = ParseCatalogue(EntryJson());
  Require(load.error.empty(), "fixture entry");
  const auto& entry = load.entries[0];
  Options options;
  options.step = Step::Model;
  options.yes = true;
  options.catalogue_id = entry.id;
  options.root = (base / "root").string();

  // 1. Plan only: size, free space and the licence line, and nothing written.
  {
    Fake transport;
    auto env = MakeEnvironment(transport, 123456789012ULL);
    PreparationPlan plan;
    std::istringstream input;
    std::ostringstream out, err;
    Options plan_options = options;
    plan_options.plan_only = true;
    plan_options.machine = true;
    plan_options.root = "C:\\golden-root";  // fixed length: stable wrapping
    const auto code = RunCatalogueModelForTesting(plan_options, entry, env,
                                                  input, out, err, &plan);
    Require(code == 0, "plan: " + err.str());
    Require(!plan.ready && transport.requests == 0 && !fs::exists(options.root),
            "a plan writes nothing and uses no network");
    auto text = Normalize(plan.text);
    {
      // Golden: the plan text for the fixture entry under a fixed root (the
      // 80-column wrap depends on the path length). BUBO_WRITE_GOLDEN=1
      // rewrites the file.
      const fs::path golden = fs::path(BUBO_PREPARE_FIXTURES) /
                              "model-consent-catalogue-fixture.txt";
      if (std::getenv("BUBO_WRITE_GOLDEN"))
        std::ofstream(golden, std::ios::binary) << text;
      std::ifstream file(golden, std::ios::binary);
      Require(bool(file), "golden file exists");
      std::ostringstream expected;
      expected << file.rdbuf();
      Require(Normalize(expected.str()) == text,
              "plan text equals the golden:\n" + text);
    }
    Require(Contains(text, "one.gguf") && Contains(text, "config.json") &&
                Contains(text, "Total file size: 6 bytes") &&
                Contains(text, "Free space: 123456789012 bytes") &&
                Contains(text, "License: Apache-2.0. Fixture model.") &&
                Contains(text,
                         "Licence link: "
                         "https://www.apache.org/licenses/LICENSE-2.0") &&
                !Contains(text, "Download and prepare all files in this plan?"),
            "plan text carries size, free space and the licence block, and "
            "no question (--plan asks nothing)");
    Require(!Contains(text, "Qwen Community License"),
            "no Qwen text in another model's plan");
    const auto& licences = plan.machine.find("licenses")->items();
    Require(licences.size() == 1 &&
                licences[0].member_str("path") == "LICENSE-INFO.txt" &&
                Contains(licences[0].member_str("text"),
                         "License: Apache-2.0. Fixture model."),
            "machine plan carries the licence notice");
    const auto& sources = plan.machine.find("sources")->items();
    Require(sources.size() == 2 &&
                sources[0].member_str("url") ==
                    "https://huggingface.co/fixture/model/resolve/" +
                        std::string(kCommit) + "/one.gguf" &&
                sources[0].member_str("sha256") == kAbc,
            "machine plan sources come from the catalogue pin");
    Require(plan.volumes.size() == 1 &&
                plan.volumes[0].free_bytes == 123456789012ULL,
            "free space of the target drive is in the plan");
    const auto& volume = plan.machine["volumes"].items().front();
    Require(
        volume.member_size("safety_margin_bytes") == 2147483648ULL &&
            volume.member_size("bytes_to_write") > 6 &&
            volume.member_size("free_after_download_bytes") ==
                123456789012ULL - volume.member_size("bytes_to_write"),
        "catalogue plan publishes remaining space without charging its margin");
  }
  // 2. Not enough space stops with the storage code before any download.
  {
    Fake transport;
    auto env = MakeEnvironment(transport, 1000);
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, entry, env, input, out, err) ==
                    7 &&
                transport.requests == 0 &&
                Contains(err.str(), "Not enough free space on"),
            "insufficient space is code 7, no request, plain message: " +
                err.str());
  }
  // 2b. --plan still reports the plan when the drive is too small, so a
  // client can say how much space is missing.
  {
    Fake transport;
    auto env = MakeEnvironment(transport, 1000);
    PreparationPlan plan;
    std::istringstream input;
    std::ostringstream out, err;
    auto plan_options = options;
    plan_options.plan_only = true;
    plan_options.machine = true;
    Require(RunCatalogueModelForTesting(plan_options, entry, env, input, out,
                                        err, &plan) == 0 &&
                plan.volumes.size() == 1 &&
                plan.volumes[0].free_bytes == 1000 &&
                plan.volumes[0].required_bytes > 1000 &&
                plan.machine["volumes"].items().front().member_size(
                    "free_after_download_bytes") == 0 &&
                Contains(Normalize(plan.text), "Free space: 1000 bytes") &&
                transport.requests == 0,
            "plan with too little free space is reported, not refused");
  }
  // 3. Prepare: licence text is shown before the first request.
  Fake transport;
  auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
  {
    std::istringstream input;
    std::ostringstream out, err;
    const auto code =
        RunCatalogueModelForTesting(options, entry, env, input, out, err);
    Require(code == 0, "prepare: " + err.str() + out.str());
    const auto text = out.str();
    const auto licence_at = text.find("License: Apache-2.0. Fixture model.");
    const auto download_at = text.find("prepare download:");
    Require(licence_at != std::string::npos &&
                download_at != std::string::npos && licence_at < download_at,
            "licence text appears before the first download line");
    Require(transport.requests == 2, "two downloads");
    const auto folder = fs::path(options.root) / "models" / entry.id;
    Require(fs::exists(folder / "one.gguf") &&
                fs::exists(folder / "config.json") &&
                fs::exists(folder / "licenses" / "LICENSE-INFO.txt") &&
                fs::exists(folder / "bubo-package.json") &&
                fs::exists(folder / ".bubo" / "prepare-state.json"),
            "files, licence, state and manifest are written");
    Require(!fs::exists(fs::path(options.root) / "models" /
                        std::string(kPackageId)),
            "the Flash Next folder is not created");
    using namespace metadata;
    const auto manifest = Parse(Read(folder / "bubo-package.json"));
    Require(
        String(Required(manifest, "package_id")) == entry.id &&
            String(Required(Required(manifest, "model"), "family")) == "qwen35",
        "manifest names the catalogue id and the architecture");
  }
  // 4. Verify: a second run is a no-op with no request.
  transport.requests = 0;
  {
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, entry, env, input, out, err) ==
                    0 &&
                transport.requests == 0 &&
                Contains(out.str(), "already prepared"),
            "prepared entry is a no-op");
  }
  // --verify hashes the files again and still needs no network.
  {
    auto verify = options;
    verify.verify = true;
    std::istringstream input;
    std::ostringstream out, err;
    Require(
        RunCatalogueModelForTesting(verify, entry, env, input, out, err) == 0 &&
            transport.requests == 0,
        "--verify passes on intact files");
  }
  // 5. Repair: a deleted file is fetched again, the intact one is kept.
  const auto folder = fs::path(options.root) / "models" / entry.id;
  fs::remove(folder / "config.json");
  {
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, entry, env, input, out, err) ==
                    0 &&
                transport.requests == 1 && fs::exists(folder / "config.json"),
            "repair downloads only the missing file");
  }
  // 6. A damaged file of the right size is caught by --verify and replaced
  // (this case asserted exit 6 before the repair path existed): the full hash
  // finds it, one download fetches it again, no damaged copy is left.
  {
    std::ofstream(folder / "one.gguf", std::ios::binary | std::ios::trunc)
        << "xyz";
    auto verify = options;
    verify.verify = true;
    transport.requests = 0;
    std::istringstream input;
    std::ostringstream out, err;
    Require(
        RunCatalogueModelForTesting(verify, entry, env, input, out, err) == 0 &&
            transport.requests == 1 &&
            Contains(out.str(), "SHA-256 differs from the pin") &&
            HashFile(folder / "one.gguf") == kAbc,
        "--verify finds a changed file and downloads it again: " + err.str());
    for (const auto& item : fs::directory_iterator(folder))
      Require(item.path().filename().string().find(".damaged-") ==
                  std::string::npos,
              "no damaged copy is left");
  }
  // 7. A disabled entry never reaches the network.
  {
    auto disabled = ParseCatalogue(EntryJson(false)).entries[0];
    Fake none;
    auto env2 = MakeEnvironment(none, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, disabled, env2, input, out,
                                        err) == 4 &&
                none.requests == 0 && Contains(err.str(), "not pinned yet"),
            "disabled entry: code 4, no request");
  }
  // 8. Offline mode with files missing fails before the network (code 4).
  {
    auto offline = options;
    offline.root = (base / "root-offline").string();
    offline.offline = true;
    Fake none;
    auto env2 = MakeEnvironment(none, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(offline, entry, env2, input, out,
                                        err) == 4 &&
                none.requests == 0,
            "offline with missing files");
  }
  // 9. A declined plan downloads nothing.
  {
    auto ask = options;
    ask.yes = false;
    ask.root = (base / "root-decline").string();
    Fake none;
    auto env2 = MakeEnvironment(none, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input("n\n");
    std::ostringstream out, err;
    Require(
        RunCatalogueModelForTesting(ask, entry, env2, input, out, err) == 3 &&
            none.requests == 0,
        "declined plan");
  }
}

// E11: prepare fetches the projector next to the model, shows its licence,
// and replaces an older package marker instead of refusing it.
void ProjectorPrepare(const fs::path& base) {
  using namespace metadata;
  const auto projector_of = [](const CatalogueEntry& entry) {
    const bubo::cli::CatalogueFile* found = nullptr;
    for (const auto& file : entry.files)
      if (file.role == "mmproj" && file.downloadable)
        found = &file;
    Require(found != nullptr, "projector of " + entry.id);
    return found;
  };
  // 1. Real entries, plan only: no network, the projector row and licence.
  for (const auto* id :
       {"qwen38-27b-ud-q4-k-xl", "gemma4-31b-unsloth-qat-ud-q4-k-xl"}) {
    const auto* entry =
        bubo::cli::FindCatalogueEntry(ShippedCatalogue().entries, id);
    Require(entry != nullptr, std::string("entry ") + id);
    const auto* projector = projector_of(*entry);
    Options options;
    options.step = Step::Model;
    options.plan_only = true;
    options.machine = true;
    options.catalogue_id = entry->id;
    options.root = (base / "root-projector-plan").string();
    Fake transport;
    auto env = MakeEnvironment(transport, 123456789012ULL);
    PreparationPlan plan;
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, *entry, env, input, out, err,
                                        &plan) == 0,
            "projector plan: " + err.str());
    const auto text = Normalize(plan.text);
    Require(Contains(text, "mmproj-BF16.gguf | huggingface.co |") &&
                Contains(text, std::to_string(projector->bytes)) &&
                Contains(text, projector->sha256) && transport.requests == 0 &&
                !fs::exists(options.root),
            "the plan shows the projector with its size and hash:\n" + text);
    bool source = false;
    for (const auto& row : plan.machine.find("sources")->items())
      source |= row.member_str("url") == projector->url &&
                row.member_str("sha256") == projector->sha256;
    Require(source, "the machine plan sources include the projector pin");
    const auto& licences = plan.machine.find("licenses")->items();
    if (projector->licence_text.empty()) {
      Require(licences.size() == 1,
              "a projector of the model's own repository shares its licence");
    } else {
      // The plan wraps at 80 columns on spaces.
      auto joined = text;
      std::replace(joined.begin(), joined.end(), '\n', ' ');
      const auto at = joined.find(projector->licence_text);
      Require(licences.size() == 2 &&
                  licences[1].member_str("path") ==
                      "LICENSE-INFO-mmproj-BF16.gguf.txt" &&
                  Contains(licences[1].member_str("text"),
                           projector->licence_text) &&
                  Contains(licences[1].member_str("text"),
                           "unsloth/gemma-4-31B-it-GGUF") &&
                  at != std::string::npos &&
                  at < joined.find("Download and prepare all files"),
              "a projector from another repository shows its own licence "
              "before approval:\n" +
                  text);
    }
  }
  // 2. A model prepared before its projector was listed: the older package
  //    marker is replaced and only the projector is fetched.
  {
    const auto before = ParseCatalogue(EntryJson()).entries[0];
    auto after = before;
    auto projector = after.files[0];
    projector.name = projector.repo_path = "mmproj-BF16.gguf";
    projector.url = "https://huggingface.co/fixture/model/resolve/" +
                    std::string(kCommit) + "/mmproj-BF16.gguf";
    projector.role = "mmproj";
    after.files.push_back(projector);
    Require(bubo::cli::PinProblem(after).empty(), "projector fixture pins");
    Options options;
    options.step = Step::Model;
    options.yes = true;
    options.catalogue_id = before.id;
    options.root = (base / "root-projector").string();
    Fake transport;
    auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, before, env, input, out,
                                        err) == 0 &&
                transport.requests == 2,
            "first prepare: " + err.str());
    transport.requests = 0;
    std::ostringstream out2, err2;
    Require(RunCatalogueModelForTesting(options, after, env, input, out2,
                                        err2) == 0,
            "prepare after the projector was added: " + err2.str());
    const auto folder = fs::path(options.root) / "models" / before.id;
    const auto manifest = Parse(Read(folder / "bubo-package.json"));
    const auto& row =
        Required(Required(manifest, "artefacts"), "mmproj-BF16.gguf");
    Require(transport.requests == 1 &&
                fs::exists(folder / "mmproj-BF16.gguf") &&
                String(Required(row, "file")) == "mmproj-BF16.gguf" &&
                Integer(Required(row, "size")) == 3 &&
                String(Required(row, "sha256")) == kAbc,
            "only the projector is fetched; the marker names it");
    transport.requests = 0;
    std::ostringstream out3, err3;
    Require(
        RunCatalogueModelForTesting(options, after, env, input, out3, err3) ==
                0 &&
            transport.requests == 0 && Contains(out3.str(), "already prepared"),
        "the new marker is valid");
  }
  // 3. A projector from another repository: its licence notice is written
  //    beside the model's, which keeps its text.
  {
    auto entry = ParseCatalogue(EntryJson()).entries[0];
    entry.id = "fixture-foreign";
    auto projector = entry.files[0];
    const std::string other(40, '2');
    projector.name = projector.repo_path = "mmproj-BF16.gguf";
    projector.repository = "other/projector";
    projector.commit = other;
    projector.url = "https://huggingface.co/other/projector/resolve/" + other +
                    "/mmproj-BF16.gguf";
    projector.role = "mmproj";
    projector.licence_text = "License: MIT. Other projector.";
    entry.files.push_back(projector);
    Require(bubo::cli::PinProblem(entry).empty(), "foreign fixture pins");
    Options options;
    options.step = Step::Model;
    options.yes = true;
    options.catalogue_id = entry.id;
    options.root = (base / "root-foreign").string();
    Fake transport;
    auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(RunCatalogueModelForTesting(options, entry, env, input, out, err) ==
                    0 &&
                transport.requests == 3,
            "prepare with a foreign projector: " + err.str());
    const auto shown = out.str().find("License: MIT. Other projector.");
    Require(shown != std::string::npos &&
                shown < out.str().find("prepare download:"),
            "the projector licence is shown before the first download");
    const auto folder = fs::path(options.root) / "models" / entry.id;
    const auto notice =
        Read(folder / "licenses" / "LICENSE-INFO-mmproj-BF16.gguf.txt");
    const auto model_notice = Read(folder / "licenses" / "LICENSE-INFO.txt");
    Require(Contains(notice, "License: MIT. Other projector.") &&
                Contains(notice, "other/projector") &&
                Contains(model_notice, "License: Apache-2.0. Fixture model.") &&
                !Contains(model_notice, "Other projector"),
            "two notices: the model's and the projector's");
    transport.requests = 0;
    std::ostringstream out2, err2;
    Require(
        RunCatalogueModelForTesting(options, entry, env, input, out2, err2) ==
                0 &&
            transport.requests == 0 && Contains(out2.str(), "already prepared"),
        "the second notice is checked like the first: " + err2.str());
  }
}

// A transport that serves `size` bytes of 'a' in 64 KiB pieces.
class BigFake final : public Transport {
public:
  explicit BigFake(std::size_t size) : data_(size, 'a') {
    sha256_ = gufo::crypto::Sha256Hex(std::span(
        reinterpret_cast<const std::uint8_t*>(data_.data()), data_.size()));
  }
  const std::string& sha256() const { return sha256_; }
  TransferResult Perform(const Request&, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    const auto size = std::to_string(data_.size());
    const Response response{200,
                            {{"content-length", size},
                             {"etag", "\"big\""},
                             {"x-linked-etag", sha256_},
                             {"x-linked-size", size}}};
    if (!headers(response))
      return {};
    for (std::size_t at = 0; at < data_.size(); at += 65536) {
      const auto count = std::min<std::size_t>(65536, data_.size() - at);
      body(std::span(reinterpret_cast<const std::uint8_t*>(data_.data() + at),
                     count));
    }
    return {};
  }

private:
  std::string data_, sha256_;
};

// E13 handover: progress while an existing file is hashed, one human
// progress line per GiB, and the exact serve command after a catalogue
// prepare.
void PrepareProgressAndCommand(const fs::path& base) {
  // 1. A 3 MiB download prints one progress line (at the end), not one per
  //    MiB.
  {
    BigFake transport(3 * 1024 * 1024);
    auto entry = ParseCatalogue(EntryJson()).entries[0];
    entry.files.resize(1);
    entry.files[0].name = entry.files[0].repo_path = "big.gguf";
    entry.files[0].bytes = 3 * 1024 * 1024;
    entry.files[0].sha256 = transport.sha256();
    entry.files[0].url = "https://huggingface.co/fixture/model/resolve/" +
                         std::string(kCommit) + "/big.gguf";
    Require(bubo::cli::PinProblem(entry).empty(), "big fixture pins");
    Options options;
    options.step = Step::Model;
    options.yes = true;
    options.catalogue_id = entry.id;
    options.root = (base / "root-big").string();
    auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(
        RunCatalogueModelForTesting(options, entry, env, input, out, err) == 0,
        "3 MiB prepare: " + err.str());
    std::istringstream lines(out.str());
    int progress_lines = 0;
    for (std::string line; std::getline(lines, line);)
      progress_lines += line.starts_with("prepare download: big.gguf ") &&
                        line.ends_with(" bytes");
    Require(progress_lines == 1,
            "one human progress line per GiB or at the end; got " +
                std::to_string(progress_lines) + ":\n" + out.str());
  }
  // 2. A catalogue file already on disk without a valid stamp is hashed in
  //    full, and that hash reports verify progress.
  {
    const auto entry = ParseCatalogue(EntryJson()).entries[0];
    Options options;
    options.step = Step::Model;
    options.yes = true;
    options.catalogue_id = entry.id;
    options.root = (base / "root-rehash").string();
    Fake transport;
    auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
    std::istringstream input;
    std::ostringstream out, err;
    Require(
        RunCatalogueModelForTesting(options, entry, env, input, out, err) == 0,
        "first prepare: " + err.str());
    fs::remove(fs::path(options.root) / "models" / entry.id / ".bubo" /
               "prepare-state.json");
    std::vector<gufo::json::Value> events;
    options.progress = [&](gufo::json::Value value) {
      events.push_back(std::move(value));
    };
    std::ostringstream out2, err2;
    Require(RunCatalogueModelForTesting(options, entry, env, input, out2,
                                        err2) == 0,
            "rehash prepare: " + err2.str());
    bool main_verify = false;
    for (const auto& event : events)
      main_verify |= event.member_str("phase") == "verify" &&
                     event.member_str("file_id") == "one.gguf" &&
                     event.member_size("bytes_done") == 3;
    Require(main_verify, "the full hash of one.gguf reports verify progress");
  }
  // 3. After a catalogue prepare the CLI names the exact serve command.
  {
    Options options;
    options.step = Step::Model;
    options.root = "C:\\fixture-root";
    options.catalogue_id = "qwen38-27b-ud-q4-k-xl";
    const auto gguf = Normalize(CatalogueServeCommand(options));
    Require(gguf ==
                "bubo serve llm --manifest \"C:/fixture-root/models/"
                "qwen38-27b-ud-q4-k-xl\"",
            "GGUF entry: serve --manifest <folder>: " + gguf);
    options.catalogue_id = "diffusiongemma-26b-e38-abliterated-nvfp4";
    const auto folder = Normalize(CatalogueServeCommand(options));
    Require(folder ==
                "bubo serve llm --model \"C:/fixture-root/models/"
                "diffusiongemma-26b-e38-abliterated-nvfp4\"",
            "safetensors folder: serve --model <folder>: " + folder);
  }
}

// Machine mode: --plan writes the plan records and one result, no approval.
void MachinePlanOnly(const fs::path& base) {
  const auto load = ParseCatalogue(EntryJson());
  const auto& entry = load.entries[0];
  Options options;
  options.step = Step::Model;
  options.machine = true;
  options.plan_only = true;
  options.catalogue_id = entry.id;
  options.root = (base / "root-machine-plan").string();
  Fake transport;
  auto env = MakeEnvironment(transport, 16ULL * 1024 * 1024 * 1024);
  const PreparePhase phase = [&](const Options& o, std::istream& in,
                                 std::ostream& out, std::ostream& err,
                                 PreparationPlan* preview, bool approved) {
    return RunCatalogueModelForTesting(o, entry, env, in, out, err, preview,
                                       approved);
  };
  std::ostringstream wire, errors;
  std::istringstream none;
  int code = -1;
  {
    bubo::cli::MachineWriter writer(wire);
    code = RunMachine(options, phase, {}, none, writer, errors, {},
                      [](std::string&) { return -1; });
  }
  const auto text = wire.str();
  Require(code == 0, "machine plan exit: " + errors.str() + text);
  Require(Contains(text, "\"type\":\"plan\"") &&
              Contains(text, "LICENSE-INFO.txt") &&
              Contains(text, "License: Apache-2.0. Fixture model.") &&
              Contains(text, "\"plan_only\":true") &&
              Contains(text, "\"prepared\":false") &&
              Contains(text, "\"type\":\"result\""),
          "machine plan carries the plan, the licence and a plan_only result");
  Require(!Contains(text, "approval_required") && transport.requests == 0 &&
              !fs::exists(options.root),
          "machine plan asks for no approval and writes nothing");
}
}  // namespace

int main() {
  try {
    const auto base =
        fs::temp_directory_path() /
        ("bubo-catalogue-test-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(base);
    CataloguePinRules();
    SpeedRules();
    StockComparisonRules();
    ShippedData();
    ParseOptions();
    PrepareById(base);
    ProjectorPrepare(base);
    PrepareProgressAndCommand(base);
    MachinePlanOnly(base);
    std::error_code ec;
    fs::remove_all(base, ec);
    std::cout << "model_catalogue_test: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "model_catalogue_test FAILED: " << error.what() << '\n';
    return 1;
  }
}
