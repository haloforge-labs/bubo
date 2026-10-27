#include "src/cli/model_catalogue.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/lowmemory_data.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/transport.hpp"

namespace bubo::cli {
namespace {
namespace meta = bubo::prepare::metadata;
using gufo::json::Value;

const char kEmbedded[] =
#include "model_catalogue_data.inc"
    ;

bool Hex(std::string_view text, std::size_t length) {
  return text.size() == length &&
         std::all_of(text.begin(), text.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool RepoName(std::string_view repo) {
  const auto slash = repo.find('/');
  return slash != repo.npos && slash > 0 && slash + 1 < repo.size() &&
         repo.find('/', slash + 1) == repo.npos &&
         std::all_of(repo.begin(), repo.end(), [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' ||
                  c == '.';
         });
}
bool Role(std::string_view role) {
  return role == "main" || role == "mtp" || role == "draft_head" ||
         role == "twin" || role == "mmproj" || role == "support";
}
std::string OptString(const Value& object, const char* key) {
  const auto* value = object.find(key);
  if (!value || value->is_null())
    return {};
  return meta::String(*value);
}
bool OptBool(const Value& object, const char* key, bool fallback) {
  const auto* value = object.find(key);
  if (!value || value->is_null())
    return fallback;
  if (!value->is_bool())
    throw std::invalid_argument(std::string(key) + " must be true or false");
  return value->as_bool();
}

void ReadInfo(CatalogueEntry& entry, const Value& row) {
  entry.technical_name = OptString(row, "technical_name");
  entry.about = OptString(row, "about");
  entry.selection_label = OptString(row, "selection_label");
  entry.system_memory_note = OptString(row, "system_memory_note");
  entry.size_split_source = OptString(row, "size_split_source");
  for (const auto* key : {"gpu_bytes", "system_memory_bytes"}) {
    const auto* value = row.find(key);
    if (value && !value->is_null())
      (std::string_view(key) == "gpu_bytes" ? entry.gpu_bytes
                                            : entry.system_memory_bytes) =
          meta::Integer(*value);
  }
  if (const auto* links = row.find("about_links")) {
    if (!links->is_array())
      throw std::invalid_argument("about_links must be an array");
    for (const auto& link : links->items()) {
      const auto url = meta::String(link);
      if (!url.starts_with("https://"))
        throw std::invalid_argument("about link must use https");
      entry.about_links.push_back(url);
    }
  }
  if (entry.gpu_bytes.has_value() != entry.system_memory_bytes.has_value() ||
      (entry.gpu_bytes &&
       (entry.size_split_source.empty() || entry.system_memory_note.empty())))
    throw std::invalid_argument(
        "placement needs both sizes, source and part name");
}

CatalogueEntry FlashNext(prepare::ModelSet set) {
  CatalogueEntry entry;
  const auto& rows = prepare::ModelCatalogue(set);
  entry.id = std::string(prepare::PackageId(set));
  entry.display_name = set == prepare::ModelSet::Huihui
                           ? prepare::kPackageDisplayName
                       : set == prepare::ModelSet::LowMemory
                           ? prepare::kLowMemoryPackageDisplayName
                           : prepare::kOriginalPackageDisplayName;
  entry.role =
      set == prepare::ModelSet::LowMemory ? "supported" : "recommended";
  entry.architecture = prepare::kFlashNextArchitecture;
  entry.set = set == prepare::ModelSet::Original ? "original" : "uncensored";
  entry.default_status = set == prepare::ModelSet::Huihui ? "default" : "none";
  entry.download_enabled = true;
  entry.flash_next_set = set;
  const auto root = meta::Parse(kEmbedded);
  const auto& info =
      meta::Required(meta::Required(root, "flash_next"),
                     set == prepare::ModelSet::Huihui     ? "huihui"
                     : set == prepare::ModelSet::Original ? "original"
                                                          : "lowmemory");
  ReadInfo(entry, info);
  if (set == prepare::ModelSet::LowMemory) {
    entry.measured_main_resident_bytes =
        prepare::kLowMemoryMeasuredMainResidentBytes;
    // Local-only availability still needs compiled evidence for every main
    // shard. An incomplete development build must not advertise local reuse.
    for (const auto& row : rows)
      if (row.kind == prepare::Kind::MainShard &&
          !prepare::ProductionFingerprintTable(row)) {
        entry.download_enabled = false;
        entry.disabled_reason =
            "Local verification evidence is not available for this lowmemory "
            "set.";
        break;
      }
  }
  entry.repository = std::string(rows[0].source_repo);
  entry.commit = std::string(rows[0].commit);
  for (const auto& row : rows) {
    CatalogueFile file;
    file.name = std::string(row.file_name);
    file.repo_path = std::string(row.repo_path);
    file.url = prepare::OriginUrl(row);
    file.bytes = row.size;
    file.sha256 = std::string(row.sha256);
    file.repository = std::string(row.source_repo);
    file.commit = std::string(row.commit);
    switch (row.kind) {
      case prepare::Kind::MainShard:
        file.role = "main";
        break;
      case prepare::Kind::Mtp:
        file.role = "mtp";
        break;
      case prepare::Kind::Mmproj:
        file.role = "mmproj";
        break;
      case prepare::Kind::Twin:
        file.role = "twin";
        break;
      case prepare::Kind::DraftHead:
        file.role = "draft_head";
        break;
      case prepare::Kind::Support:
        file.role = "support";
        break;
    }
    entry.files.push_back(std::move(file));
  }
  entry.licence.id = "qwen-community-1.0";
  entry.licence.name = "Qwen Community License 1.0";
  // The lowmemory retains the approved Huihui licence text; its hosting
  // placeholder must not masquerade as the verified upstream licence source.
  const auto& licence_source =
      prepare::ModelCatalogue(prepare::ModelSet::Huihui)[0];
  entry.licence.url =
      "https://huggingface.co/" +
      (set == prepare::ModelSet::LowMemory
           ? std::string(licence_source.source_repo)
           : entry.repository) +
      "/tree/" +
      (set == prepare::ModelSet::LowMemory ? std::string(licence_source.commit)
                                           : entry.commit);
  entry.licence.prompt_line =
      "License: Qwen Community License 1.0. It is not MIT. Prepare shows the "
      "full text before it downloads.";
  entry.licence.verified_upstream = true;
  if (set == prepare::ModelSet::Huihui) {



    entry.speed.present = true;
    entry.speed.writing_tok_s = 54.30;
    entry.speed.reading_tok_s = 1201.0;
    entry.speed.prompt_tokens = 8192;
    entry.speed.source =
        "Bubo release measurements: decode = mean of 6 greedy probes; "
        "prefill = 8K natural-text prompt, median of 3";
    entry.speed.label = "measured on Strix Halo 128 GB, smoke-grade";
  }
  return entry;
}

CatalogueEntry ParseEntry(const Value& row) {
  meta::Keys(row, {"id",
                   "display_name",
                   "role",
                   "architecture",
                   "set",
                   "repository",
                   "commit",
                   "files",
                   "licence",
                   "default_status",
                   "default_note",
                   "family_group",
                   "download_enabled",
                   "disabled_reason",
                   "packaging",
                   "estimate_note",
                   "estimate_pointers",
                   "speed",
                   "stock_comparison",
                   "technical_name",
                   "about",
                   "selection_label",
                   "about_links",
                   "gpu_bytes",
                   "system_memory_bytes",
                   "system_memory_note",
                   "size_split_source"});
  CatalogueEntry entry;
  ReadInfo(entry, row);
  entry.id = meta::String(meta::Required(row, "id"));
  entry.display_name = meta::String(meta::Required(row, "display_name"));
  entry.role = meta::String(meta::Required(row, "role"));
  entry.architecture = meta::String(meta::Required(row, "architecture"));
  entry.set = OptString(row, "set");
  entry.repository = OptString(row, "repository");
  entry.commit = OptString(row, "commit");
  entry.default_status = OptString(row, "default_status");
  if (entry.default_status.empty())
    entry.default_status = "none";
  entry.default_note = OptString(row, "default_note");
  entry.family_group = OptString(row, "family_group");
  entry.download_enabled = OptBool(row, "download_enabled", false);
  entry.disabled_reason = OptString(row, "disabled_reason");
  entry.packaging = OptString(row, "packaging");
  entry.estimate_note = OptString(row, "estimate_note");
  if (const auto* pointers = row.find("estimate_pointers");
      pointers && pointers->is_array())
    for (const auto& item : pointers->items())
      entry.estimate_pointers.push_back(meta::String(item));
  if (const auto* speed = row.find("speed"); speed && !speed->is_null()) {
    meta::Keys(*speed, {"writing_tok_s", "reading_tok_s", "prompt_tokens",
                        "source", "label"});
    entry.speed.present = true;
    const auto number = [&](const char* key) -> std::optional<double> {
      const auto* v = speed->find(key);
      if (!v || v->is_null())
        return std::nullopt;
      if (!v->is_number())
        throw std::invalid_argument(std::string(key) + " must be a number");
      return v->as_double();
    };
    entry.speed.writing_tok_s = number("writing_tok_s");
    entry.speed.reading_tok_s = number("reading_tok_s");
    if (const auto* v = speed->find("prompt_tokens"); v && !v->is_null())
      entry.speed.prompt_tokens = meta::Integer(*v);
    entry.speed.source = OptString(*speed, "source");
    entry.speed.label = OptString(*speed, "label");
  }
  if (const auto* stock = row.find("stock_comparison");
      stock && !stock->is_null()) {
    meta::Keys(*stock, {"engine", "prefill_pp8192", "decode_tg128", "source"});
    entry.stock_comparison.present = true;
    entry.stock_comparison.engine =
        meta::String(meta::Required(*stock, "engine"));
    entry.stock_comparison.source =
        meta::String(meta::Required(*stock, "source"));
    entry.stock_comparison.prefill_pp8192 =
        meta::Required(*stock, "prefill_pp8192").as_double();
    entry.stock_comparison.decode_tg128 =
        meta::Required(*stock, "decode_tg128").as_double();
  }
  const auto& licence = meta::Required(row, "licence");
  meta::Keys(licence,
             {"id", "name", "url", "prompt_line", "note", "verified_upstream"});
  entry.licence.id = OptString(licence, "id");
  entry.licence.name = OptString(licence, "name");
  entry.licence.url = OptString(licence, "url");
  entry.licence.prompt_line = OptString(licence, "prompt_line");
  entry.licence.note = OptString(licence, "note");
  entry.licence.verified_upstream =
      OptBool(licence, "verified_upstream", false);
  const auto& files = meta::Required(row, "files");
  if (!files.is_array())
    throw std::invalid_argument("files must be an array");
  for (const auto& item : files.items()) {
    meta::Keys(item, {"name", "repo_path", "url", "bytes", "sha256", "role",
                      "repository", "commit", "optional", "downloadable",
                      "licence_text"});
    CatalogueFile file;
    file.name = OptString(item, "name");
    file.repo_path = OptString(item, "repo_path");
    file.url = OptString(item, "url");
    if (const auto* bytes = item.find("bytes"); bytes && !bytes->is_null())
      file.bytes = meta::Integer(*bytes);
    file.sha256 = OptString(item, "sha256");
    file.role = OptString(item, "role");
    file.repository = OptString(item, "repository");
    file.commit = OptString(item, "commit");
    file.optional = OptBool(item, "optional", false);
    file.downloadable = OptBool(item, "downloadable", true);
    file.licence_text = OptString(item, "licence_text");
    entry.files.push_back(std::move(file));
  }
  return entry;
}
}  // namespace

std::string PinProblem(const CatalogueEntry& entry) {
  const auto missing = [&](const std::string& what) {
    return entry.id + ": " + what;
  };
  if (entry.id.empty())
    return "entry without an id";
  if (entry.files.empty())
    return missing("no files");
  const auto& licence = entry.licence;
  if (licence.id.empty() || licence.name.empty() || licence.url.empty() ||
      licence.prompt_line.empty())
    return missing("licence is incomplete (id, name, link and prompt line)");
  if (!licence.url.starts_with("https://"))
    return missing("licence link is not https");
  if (!licence.verified_upstream && licence.note.empty())
    return missing("licence not read upstream and no licence note");
  bool main_file = false;
  std::set<std::string> names;
  for (const auto& file : entry.files) {
    const auto& repository =
        file.repository.empty() ? entry.repository : file.repository;
    const auto& commit = file.commit.empty() ? entry.commit : file.commit;
    if (file.name.empty() || file.repo_path.empty() ||
        !names.insert(file.name).second)
      return missing("file without a name or with a repeated name");
    if (!Role(file.role))
      return missing(file.name + ": unknown role");
    if (!RepoName(repository) || !Hex(commit, 40))
      return missing(file.name + ": no pinned repository and commit");
    if (file.bytes == 0 || file.bytes > 9007199254740991ULL)
      return missing(file.name + ": no byte count");
    if (!Hex(file.sha256, 64))
      return missing(file.name + ": no SHA-256");
    if (file.url != "https://huggingface.co/" + repository + "/resolve/" +
                        commit + "/" + file.repo_path)
      return missing(file.name + ": URL is not the pinned resolve URL");
    // The entry licence covers its own repository. A downloaded file from
    // another repository carries its own licence line for the plan. Flash
    // Next sets have their own approved legal bundle.
    if (!entry.flash_next_set && file.downloadable &&
        repository != entry.repository &&
        (file.licence_text.empty() || file.licence_text == "not yet read"))
      return missing(file.name +
                     ": file from another repository has no "
                     "licence line");
    if (file.downloadable && file.role == "main")
      main_file = true;
  }
  if (!main_file)
    return missing("no downloadable main file");
  return {};
}

std::string ValidateCatalogue(const std::vector<CatalogueEntry>& entries) {
  std::set<std::string> ids;
  for (const auto& entry : entries) {
    if (entry.id.empty() || !ids.insert(entry.id).second)
      return "empty or repeated catalogue id";
    if (entry.role != "recommended" && entry.role != "supported")
      return entry.id + ": role must be recommended or supported";
    if (entry.default_status != "none" &&
        entry.default_status != "to be decided" &&
        entry.default_status != "default")
      return entry.id + ": unknown default_status";
    if (entry.speed.present) {
      const auto& speed = entry.speed;
      if (speed.source.empty() || speed.label.empty())
        return entry.id + ": speed needs a source and a label";
      if (!speed.writing_tok_s && !speed.reading_tok_s)
        return entry.id +
               ": speed has no number (use null for the whole object)";
      if ((speed.writing_tok_s && *speed.writing_tok_s <= 0) ||
          (speed.reading_tok_s && *speed.reading_tok_s <= 0))
        return entry.id + ": speed numbers must be positive";
      if (speed.prompt_tokens && *speed.prompt_tokens == 0)
        return entry.id + ": prompt_tokens must be positive";
    }
    if (entry.stock_comparison.present) {
      const auto& stock = entry.stock_comparison;
      if (stock.engine.empty() || stock.source.empty() ||
          stock.prefill_pp8192 <= 0 || stock.decode_tg128 <= 0)
        return entry.id +
               ": stock_comparison needs an engine, a source and two positive "
               "numbers (use null for the whole object)";
    }
    if (entry.download_enabled) {
      if (const auto problem = PinProblem(entry); !problem.empty())
        return problem + " (download_enabled needs a full pin)";
      if (!entry.disabled_reason.empty())
        return entry.id + ": download_enabled with a disabled_reason";
    } else if (entry.disabled_reason.empty())
      return entry.id + ": disabled entry needs a disabled_reason";
  }
  return {};
}

CatalogueLoad ParseCatalogue(std::string_view json_text) {
  CatalogueLoad load;
  try {
    const auto root = meta::Parse(json_text);
    meta::Keys(
        root, {"schema_version", "catalogue_version", "flash_next", "entries"});
    if (meta::Integer(meta::Required(root, "schema_version")) != 1)
      throw std::invalid_argument("unsupported catalogue schema_version");
    const auto& rows = meta::Required(root, "entries");
    if (!rows.is_array())
      throw std::invalid_argument("entries must be an array");
    for (const auto& row : rows.items())
      load.entries.push_back(ParseEntry(row));
    load.error = ValidateCatalogue(load.entries);
  } catch (const std::exception& error) {
    load.error = error.what();
  }
  if (!load.error.empty())
    load.entries.clear();
  return load;
}

const CatalogueLoad& ShippedCatalogue() {
  static const CatalogueLoad shipped = [] {
    auto load = ParseCatalogue(kEmbedded);
    if (!load.error.empty())
      return load;
    std::vector<CatalogueEntry> all;
    all.push_back(FlashNext(prepare::ModelSet::Huihui));
    all.push_back(FlashNext(prepare::ModelSet::Original));
    all.push_back(FlashNext(prepare::ModelSet::LowMemory));
    for (auto& entry : load.entries)
      all.push_back(std::move(entry));
    load.entries = std::move(all);
    load.error = ValidateCatalogue(load.entries);
    if (!load.error.empty())
      load.entries.clear();
    return load;
  }();
  return shipped;
}

const CatalogueEntry* FindCatalogueEntry(
    const std::vector<CatalogueEntry>& entries, std::string_view id) {
  for (const auto& entry : entries)
    if (entry.id == id)
      return &entry;
  return nullptr;
}

std::uint64_t CatalogueDownloadBytes(const CatalogueEntry& entry) {
  std::uint64_t total = 0;
  for (const auto& file : entry.files)
    if (file.downloadable)
      total += file.bytes;
  return total;
}

Value PreparedModelRow(const CatalogueEntry& entry,
                       const std::string& manifest_path,
                       const std::string& model_path, bool prepared) {
  auto model = Value::object();
  model["id"] = entry.id;
  model["display_name"] = entry.display_name;
  model["manifest_path"] =
      manifest_path.empty() ? Value() : Value(manifest_path);
  model["model_path"] = model_path.empty() ? Value() : Value(model_path);
  bool mtp = false, twin = false, mmproj = false;
  for (const auto& file : entry.files) {
    if (!file.downloadable)
      continue;
    mtp |= file.role == "mtp";
    twin |= file.role == "twin";
    mmproj |= file.role == "mmproj";
  }
  model["size_bytes"] = CatalogueDownloadBytes(entry);
  model["prepared_state"] = prepared ? "prepared" : "not_prepared";
  auto capabilities = Value::object();
  capabilities["context_max"] = entry.flash_next_set ? Value(262144) : Value();
  capabilities["mtp"] = mtp;
  capabilities["twin"] = twin;
  capabilities["mmproj"] = mmproj;
  model["capabilities"] = std::move(capabilities);
  return model;
}

Value CatalogueEntryJson(const CatalogueEntry& entry) {
  auto value = Value::object();
  value["id"] = entry.id;
  value["display_name"] = entry.display_name;
  value["technical_name"] = entry.technical_name;
  value["about"] = entry.about;
  value["selection_label"] = entry.selection_label;
  auto links = Value::array();
  for (const auto& link : entry.about_links)
    links.push_back(link);
  value["about_links"] = std::move(links);
  value["gpu_bytes"] = entry.gpu_bytes ? Value(*entry.gpu_bytes) : Value();
  value["system_memory_bytes"] =
      entry.system_memory_bytes ? Value(*entry.system_memory_bytes) : Value();
  value["system_memory_note"] = entry.system_memory_note.empty()
                                    ? Value()
                                    : Value(entry.system_memory_note);
  value["size_split_source"] = entry.size_split_source.empty()
                                   ? Value()
                                   : Value(entry.size_split_source);
  value["role"] = entry.role;
  value["architecture"] = entry.architecture;
  value["set"] = entry.set.empty() ? Value() : Value(entry.set);
  value["default_status"] = entry.default_status;
  value["family_group"] =
      entry.family_group.empty() ? Value() : Value(entry.family_group);
  value["download_enabled"] = entry.download_enabled;
  value["disabled_reason"] =
      entry.disabled_reason.empty() ? Value() : Value(entry.disabled_reason);
  value["packaging"] = entry.packaging.empty() ? "gguf" : entry.packaging;
  auto licence = Value::object();
  licence["id"] = entry.licence.id;
  licence["name"] = entry.licence.name;
  licence["url"] = entry.licence.url;
  licence["prompt_line"] = entry.licence.prompt_line;
  licence["note"] =
      entry.licence.note.empty() ? Value() : Value(entry.licence.note);
  licence["verified_upstream"] = entry.licence.verified_upstream;
  value["licence"] = std::move(licence);
  auto files = Value::array();
  for (const auto& file : entry.files) {
    auto row = Value::object();
    row["name"] = file.name;
    row["role"] = file.role;
    row["bytes"] = file.bytes;
    row["sha256"] = file.sha256;
    row["url"] = file.url;
    row["optional"] = file.optional;
    row["downloadable"] = file.downloadable;
    row["licence_text"] =
        file.licence_text.empty() ? Value() : Value(file.licence_text);
    files.push_back(std::move(row));
  }
  value["files"] = std::move(files);
  value["download_bytes"] = CatalogueDownloadBytes(entry);
  value["default_note"] =
      entry.default_note.empty() ? Value() : Value(entry.default_note);
  value["estimate_note"] =
      entry.estimate_note.empty() ? Value() : Value(entry.estimate_note);
  if (entry.speed.present) {
    auto speed = Value::object();
    const auto& v = entry.speed;
    speed["writing_tok_s"] =
        v.writing_tok_s ? Value(*v.writing_tok_s) : Value();
    speed["reading_tok_s"] =
        v.reading_tok_s ? Value(*v.reading_tok_s) : Value();
    speed["prompt_tokens"] =
        v.prompt_tokens ? Value(*v.prompt_tokens) : Value();
    speed["source"] = v.source;
    speed["label"] = v.label;
    value["speed"] = std::move(speed);
  } else
    value["speed"] = Value();
  if (entry.stock_comparison.present) {
    auto stock = Value::object();
    stock["engine"] = entry.stock_comparison.engine;
    stock["prefill_pp8192"] = entry.stock_comparison.prefill_pp8192;
    stock["decode_tg128"] = entry.stock_comparison.decode_tg128;
    stock["source"] = entry.stock_comparison.source;
    value["stock_comparison"] = std::move(stock);
  } else
    value["stock_comparison"] = Value();
  auto pointers = Value::array();
  for (const auto& pointer : entry.estimate_pointers)
    pointers.push_back(pointer);
  value["estimate_pointers"] = std::move(pointers);
  return value;
}

// ---- readiness of an installed package ------------------------------------

std::filesystem::path CatalogueFilePath(const CatalogueEntry& entry,
                                        const CatalogueFile& file,
                                        const std::filesystem::path& folder) {
  if (entry.flash_next_set)
    for (const auto& pinned : prepare::ModelCatalogue(*entry.flash_next_set))
      if (pinned.file_name == file.name)
        return folder / std::filesystem::u8path(pinned.destination);
  return folder / std::filesystem::u8path(file.name);
}

CatalogueReadiness CatalogueReadinessAt(const CatalogueEntry& entry,
                                        const std::filesystem::path& folder) {
  std::size_t wanted = 0, present = 0;
  CatalogueReadiness readiness;
  for (const auto& file : entry.files) {
    if (!file.downloadable)
      continue;
    ++wanted;
    std::error_code ec;
    const auto path = CatalogueFilePath(entry, file, folder);
    const bool ok = std::filesystem::is_regular_file(path, ec) &&
                    std::filesystem::file_size(path, ec) == file.bytes;
    if (ok) {
      ++present;
    } else {
      // Forward slashes, as the catalogue and the package manifest write them.
      const auto relative = path.lexically_relative(folder).generic_u8string();
      readiness.missing_files.emplace_back(relative.begin(), relative.end());
    }
  }
  // Serve and status need the package manifest, which prepare writes last.
  bool published = false;
  if (entry.flash_next_set) {
    std::error_code manifest_ec;
    published = std::filesystem::is_regular_file(folder / "bubo-package.json",
                                                 manifest_ec);
    if (!published)
      readiness.missing_files.push_back("bubo-package.json");
  }
  if (wanted == 0 || (present == 0 && !published))
    readiness.state = "not_prepared";
  else
    readiness.state = readiness.missing_files.empty() ? "prepared" : "partial";
  return readiness;
}

}  // namespace bubo::cli
