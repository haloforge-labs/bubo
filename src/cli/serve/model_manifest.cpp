#include "src/cli/serve/model_manifest.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <iostream>
#include <set>
#include <stdexcept>

#include "src/cli/brand.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/verified_state.hpp"

namespace gufo::server {
namespace {
namespace fs = std::filesystem;
namespace prep = bubo::prepare;
namespace meta = prep::metadata;
using json::Value;
std::string Display(const fs::path& path) {
  const auto value = path.u8string();
  std::string result(value.begin(), value.end());
  std::replace_if(
      result.begin(), result.end(),
      [](unsigned char c) { return c < 32 || c == 127; }, '?');
  return result;
}
[[noreturn]] void Fail(const fs::path& path, std::string why) {
  std::replace(why.begin(), why.end(), '\n', ' ');
  std::replace(why.begin(), why.end(), '\r', ' ');
  throw std::invalid_argument(Display(path) + ": " + why);
}
fs::path Absolute(const fs::path& path) {
  return fs::absolute(path).lexically_normal();
}
bool SafeRelative(std::string_view value) {
  if (value.empty() || value.front() == '/' || value.front() == '\\' ||
      value.find(':') != value.npos ||
      std::ranges::any_of(value,
                          [](unsigned char c) { return c < 32 || c == 127; }))
    return false;
  std::string text(value);
  std::replace(text.begin(), text.end(), '\\', '/');
  for (std::size_t first = 0; first <= text.size();) {
    const auto end = text.find('/', first);
    const auto part = text.substr(first, end == text.npos ? end : end - first);
    if (part.empty() || part == "." || part == "..")
      return false;
    if (end == text.npos)
      break;
    first = end + 1;
  }
  return true;
}
void CheckHeader(const fs::path& path) {
  std::string error;
  if (!core::GgufReader::OpenHeaderFile(prep::NativePath(path), &error))
    Fail(path, "invalid GGUF header/tensor table: " + error);
}
void Verify(PackageFile& file, ModelPackage& package, std::ostream* output) {
  const auto stamp = file.lease->Snapshot();
  if (output)
    *output << "verify-model: " << Display(file.path) << " (" << file.size
            << " bytes)\n";
  std::uint64_t printed = 0;
  const auto digest = prep::HashFile(
      prep::NativePath(file.path),
      [&](std::uint64_t done, std::uint64_t total) {
        if (output && (done == total || done - printed >= 64 * 1024 * 1024)) {
          *output << "verify-model: " << Display(file.path) << ' ' << done
                  << '/' << total << " bytes\n";
          printed = done;
        }
      });
  ++package.hashed_files;
  package.hashed_bytes += file.size;
  if (digest != file.sha256)
    Fail(file.path, "SHA-256 mismatch");
  if (file.lease->Snapshot() != stamp)
    Fail(file.path, "file identity changed during verification");
  file.verified_stamp = stamp;
  file.freshly_verified = true;
  file.observed_verified = true;
}
void SaveVerification(ModelPackage& package, std::ostream* output) {
  prep::VerifiedStamps rows;
  for (const auto& file : package.files)
    if (file.freshly_verified)
      rows[{file.logical_name, file.sha256}] = file.verified_stamp;
  std::string error;
  if (!prep::SaveVerifiedStamps(package.manifest_path.parent_path(), rows,
                                error, package.package_id)) {
    if (output && !package.stamp_save_failed) {
      std::replace_if(
          error.begin(), error.end(),
          [](unsigned char c) { return c < 32 || c == 127; }, ' ');
      *output << "verification stamp not saved: " << error << '\n';
    }
    package.stamp_save_failed = true;
  }
  for (auto& file : package.files)
    file.freshly_verified = false;
}
std::array<unsigned, 3> Version(std::string_view value) {
  std::array<unsigned, 3> result{};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto dot = value.find('.');
    const auto part = value.substr(0, dot);
    const auto [end, ec] =
        std::from_chars(part.data(), part.data() + part.size(), result[i]);
    if (part.empty() || ec != std::errc{} || end != part.data() + part.size() ||
        (i < 2 && dot == value.npos) || (i == 2 && dot != value.npos))
      throw std::invalid_argument("invalid engine_min version");
    value.remove_prefix(i < 2 ? dot + 1 : value.size());
  }
  return result;
}
void CheckEngineAndProfiles(const Value& document, const fs::path& manifest,
                            const std::string& profile) {
  if (Version(meta::String(meta::Required(document, "engine_min"))) >
      Version(cli::kProductVersion))
    Fail(manifest, "engine_min requires a newer Bubo engine");
  const auto& profiles = meta::Required(document, "profiles_supported");
  if (!profiles.is_array())
    Fail(manifest, "profiles_supported must be an array");
  std::set<std::string> supported;
  for (const auto& item : profiles.items()) {
    auto name = meta::String(item);
    if (name == "stock")
      name = "vanilla";
    if ((name != "default" && name != "vanilla" && name != "exact" &&
         name != "subagents") ||
        !supported.insert(name).second)
      Fail(manifest, "invalid profiles_supported");
  }
  if (!supported.contains(profile == "stock" ? "vanilla" : profile))
    Fail(manifest, "profile is not supported: " + profile);
}

// ---- plain catalogue models (not Flash Next) -------------------------------
// `prepare --catalogue-id ID` puts an entry's files in <root>\<id>\ under
// their catalogue names and writes bubo-package.json (the package marker)
// last. Pins come from the compiled catalogue.
using bubo::cli::CatalogueEntry;
using bubo::cli::CatalogueFile;
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
const CatalogueEntry* PlainCatalogueEntry(const std::string& id) {
  const auto* entry =
      bubo::cli::FindCatalogueEntry(bubo::cli::ShippedCatalogue().entries, id);
  return entry && !entry->flash_next_set && entry->packaging.empty() ? entry
                                                                     : nullptr;
}
std::vector<const CatalogueFile*> MainFiles(const CatalogueEntry& entry) {
  std::vector<const CatalogueFile*> files;
  for (const auto& file : entry.files)
    if (file.downloadable && file.role == "main")
      files.push_back(&file);
  return files;
}
// The entry whose catalogue folder and first main file `model` is.
const CatalogueEntry* CatalogueFolderEntry(const fs::path& model) {
  const auto* entry =
      PlainCatalogueEntry(Utf8(Absolute(model).parent_path().filename()));
  if (!entry)
    return nullptr;
  const auto mains = MainFiles(*entry);
  std::error_code ec;
  const auto size = fs::file_size(model, ec);
  return !ec && !mains.empty() && Utf8(model.filename()) == mains[0]->name &&
                 size == mains[0]->bytes
             ? entry
             : nullptr;
}
void CheckPinnedRow(const Value& row, const CatalogueFile& pin,
                    const fs::path& root) {
  meta::Keys(row, {"file", "size", "sha256"});
  if (meta::String(meta::Required(row, "file")) != pin.name ||
      meta::Integer(meta::Required(row, "size")) != pin.bytes ||
      meta::String(meta::Required(row, "sha256")) != pin.sha256)
    Fail(root / fs::u8path(pin.name), "manifest entry mismatch");
}
// `marker` is null for a catalogue folder without its marker: then only the
// projector default is set and the package stays empty, as for any GGUF.
void LoadCataloguePackage(const CatalogueEntry& entry, const fs::path& root,
                          const Value* marker, const fs::path& model,
                          bool verify, std::ostream* output,
                          ModelPackage& result) {
  const auto mains = MainFiles(entry);
  const auto stamps = prep::ReadVerifiedStamps(root, entry.id);
  const auto open = [&](const CatalogueFile& pin) {
    PackageFile file{
        pin.name,  pin.sha256, Absolute(root / fs::u8path(pin.name)),
        pin.bytes, false,      {},
        {},        false};
    file.lease = std::make_shared<prep::ReadLease>(prep::NativePath(file.path));
    const auto stamp = file.lease->Snapshot();
    if (stamp.size != pin.bytes)
      Fail(file.path, "size mismatch");
    CheckHeader(file.path);
    const auto stored = stamps.find({file.logical_name, file.sha256});
    file.observed_verified = stored != stamps.end() && stored->second == stamp;
    if (verify)
      Verify(file, result, output);
    if (file.lease->Snapshot() != stamp)
      Fail(file.path, "file identity changed during validation");
    return file;
  };
  if (marker) {
    result.package_id = entry.id;
    const auto& source_model = meta::Required(*marker, "model");
    meta::Keys(source_model, {"family", "source", "files"});
    if (meta::String(meta::Required(source_model, "family")) !=
        entry.architecture)
      Fail(result.manifest_path, "model family entry mismatch");
    const auto& source = meta::Required(source_model, "source");
    meta::Keys(source, {"repository", "revision"});
    if (meta::String(meta::Required(source, "repository")) !=
            entry.repository ||
        meta::String(meta::Required(source, "revision")) != entry.commit)
      Fail(result.manifest_path, "model source entry mismatch");
    const auto& files = meta::Required(source_model, "files");
    if (!files.is_array() || files.size() != mains.size())
      Fail(result.manifest_path, "model files entry mismatch");
    for (std::size_t i = 0; i < mains.size(); ++i)
      CheckPinnedRow(files.items()[i], *mains[i], root);
    // Side files the marker names. A marker from an older catalogue can name
    // fewer; it never names a file the entry does not pin.
    const auto& artefacts = meta::Required(*marker, "artefacts");
    if (!artefacts.is_object())
      Fail(result.manifest_path, "artefacts must be an object");
    for (const auto& [name, row] : artefacts.members()) {
      const auto pin = std::find_if(
          entry.files.begin(), entry.files.end(), [&](const auto& file) {
            return file.name == name && file.downloadable &&
                   file.role != "main";
          });
      if (pin == entry.files.end())
        Fail(root / fs::u8path(name), "manifest entry mismatch");
      CheckPinnedRow(row, *pin, root);
    }
    bool model_named = model.empty() || fs::is_directory(model);
    for (const auto* pin : mains) {
      const auto path = Absolute(root / fs::u8path(pin->name));
      try {
        result.files.push_back(open(*pin));
      } catch (const std::exception& error) {
        std::error_code availability_error;
        if (!fs::exists(path, availability_error) && !availability_error)
          throw ModelMissingError(Display(result.manifest_path) + ": " +
                                  Display(path) + ": " + error.what());
        throw;
      }
      if (!model.empty() && Absolute(model) == path)
        model_named = true;
    }
    if (!model_named)
      Fail(model, "main model is not named by the selected manifest");
  }
  // The vision projector is bound when it is present and sound. Without it
  // serve starts text only and says why.
  for (const auto& pin : entry.files) {
    if (!pin.downloadable || pin.role != "mmproj")
      continue;
    const auto path = Absolute(root / fs::u8path(pin.name));
    std::string why;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec))
      why = " is not prepared";
    else if (fs::file_size(path, ec) != pin.bytes || ec)
      why = " has the wrong size";
    else
      try {
        auto file = open(pin);
        result.defaults[std::string(ModelArtefactSetting(prep::Kind::Mmproj))] =
            Display(path);
        if (marker)
          result.files.push_back(std::move(file));
      } catch (const std::exception& error) {
        if (verify)
          throw;
        why = std::string(": ") + error.what();
      }
    if (!why.empty())
      result.notes.push_back("vision: " + Display(path) + why +
                             "; image input is off. Run bubo prepare model "
                             "--catalogue-id " +
                             entry.id + ".");
  }
  // The MTP head (role mtp) is bound the same way when it is present and
  // sound. Whether the family decodes with it is the release rules' decision
  // (release_profile.cpp); without the file the package serves with MTP off.
  for (const auto& pin : entry.files) {
    if (!pin.downloadable || pin.role != "mtp")
      continue;
    const auto path = Absolute(root / fs::u8path(pin.name));
    std::error_code ec;
    std::string why;
    if (!fs::is_regular_file(path, ec))
      why = " is not prepared";
    else if (fs::file_size(path, ec) != pin.bytes || ec)
      why = " has the wrong size";
    else
      try {
        auto file = open(pin);
        result.defaults[std::string(ModelArtefactSetting(prep::Kind::Mtp))] =
            Display(path);
        if (marker)
          result.files.push_back(std::move(file));
      } catch (const std::exception& error) {
        if (verify)
          throw;
        why = std::string(": ") + error.what();
      }
    if (!why.empty())
      result.notes.push_back("mtp: " + Display(path) + why +
                             "; MTP decoding is off. Run bubo prepare model "
                             "--catalogue-id " +
                             entry.id + ".");
  }
}
ModelPackage Load(
    const fs::path& model, const std::string& profile,
    std::span<const prep::Entry> entries, bool verify, std::ostream* output,
    const std::function<prep::FingerprintResult(const prep::Entry&)>&
        fingerprints,
    const fs::path& selected, bool select_set = false) {
  ModelPackage result;
  if (model.empty() && selected.empty()) {
    if (verify)
      Fail("bubo-package.json",
           "prepared model manifest required; run bubo prepare or provide "
           "--manifest bubo-package.json");
    return result;
  }
  const bool model_directory = !model.empty() && fs::is_directory(model);
  fs::path manifest;
  if (!selected.empty())
    manifest =
        fs::is_directory(selected) ? selected / "bubo-package.json" : selected;
  else if (model_directory) {
    manifest = model / "bubo-package.json";
    if (!fs::exists(manifest))
      manifest = model / "models" / prep::kPackageId / "bubo-package.json";
  } else
    manifest = model.parent_path() / "bubo-package.json";
  if (manifest.filename() != "bubo-package.json")
    Fail(manifest, "manifest filename must be bubo-package.json");
  if (!fs::exists(manifest)) {
    if (!selected.empty() || verify)
      throw ModelMissingError(
          Display(manifest) + ": " +
          "prepared model manifest is missing; run bubo prepare or provide "
          "--manifest bubo-package.json");
    // A catalogue folder without its marker still binds its projector.
    if (select_set && !model.empty() && !model_directory)
      if (const auto* entry = CatalogueFolderEntry(model))
        LoadCataloguePackage(*entry, Absolute(model).parent_path(), nullptr,
                             model, false, output, result);
    return result;
  }
  result.manifest_path = Absolute(manifest);
  const auto root = result.manifest_path.parent_path();
  try {
    prep::ReadLease manifest_lease(prep::NativePath(result.manifest_path));
    const auto document = meta::Parse(meta::Read(result.manifest_path));
    meta::Keys(document, {"schema_version", "package_id", "engine_min", "model",
                          "artefacts", "profiles_supported", "recipe_inputs"});
    if (meta::Integer(meta::Required(document, "schema_version")) != 1)
      Fail(manifest, "unsupported schema_version");
    if (select_set) {
      const auto& id = meta::String(meta::Required(document, "package_id"));
      // Unknown adjacent markers cannot supply catalogue pins or side files.
      // Explicit manifest selection and verification still require pins.
      if (selected.empty() && !verify &&
          !bubo::cli::FindCatalogueEntry(bubo::cli::ShippedCatalogue().entries,
                                         id)) {
        result.manifest_path.clear();
        return result;
      }
      if (const auto* entry = PlainCatalogueEntry(id)) {
        CheckEngineAndProfiles(document, manifest, profile);
        LoadCataloguePackage(*entry, root, &document, model, verify, output,
                             result);
        if (verify)
          SaveVerification(result, output);
        return result;
      }
      // A catalogue folder that is not GGUF (DiffusionGemma) has its own
      // loader; its marker describes no GGUF package.
      const auto* folder_entry = bubo::cli::FindCatalogueEntry(
          bubo::cli::ShippedCatalogue().entries, id);
      if (folder_entry && !folder_entry->flash_next_set) {
        result.manifest_path.clear();
        return result;
      }
    }
    // The production loader accepts each approved set; the manifest names it.
    // Every pin below then comes from that set's compiled catalogue. A caller
    // with its own entries (tests) always keeps the Huihui package id.
    std::string_view package_id = prep::kPackageId;
    std::string_view recipe_id = prep::kRecipeId;
    if (select_set)
      if (const auto set = prep::ModelSetFromPackageId(
              meta::String(meta::Required(document, "package_id")))) {
        entries = prep::ModelCatalogue(*set);
        package_id = prep::PackageId(*set);
        recipe_id = prep::RecipeId(*set);
      }
    if (meta::String(meta::Required(document, "package_id")) != package_id)
      Fail(manifest, "package_id entry mismatch");
    result.package_id = std::string(package_id);
    CheckEngineAndProfiles(document, manifest, profile);
    const auto& main = meta::Required(document, "model");
    meta::Keys(main, {"family", "source", "files"});
    if (meta::String(meta::Required(main, "family")) != "qwen3.8-flash-next")
      Fail(manifest, "model family entry mismatch");
    const auto& source = meta::Required(main, "source");
    meta::Keys(source, {"repository", "revision"});
    if (entries.size() != 8 ||
        meta::String(meta::Required(source, "repository")) !=
            entries.front().source_repo ||
        meta::String(meta::Required(source, "revision")) !=
            entries.front().commit)
      Fail(manifest, "model source entry mismatch");
    const auto& shards = meta::Required(main, "files");
    const auto& artefacts = meta::Required(document, "artefacts");
    if (!shards.is_array() || shards.size() != 4)
      Fail(manifest, "model files must name exactly four shards");
    meta::Keys(artefacts, {"mtp", "mmproj", "twin", "draft_head"});
    const auto stamps = prep::ReadVerifiedStamps(root, package_id);
    std::size_t shard_index = 0;
    std::optional<std::uint64_t> total_tensors;
    bool model_named = model.empty() || model_directory;
    std::uint64_t shard_tensors = 0;
    for (const auto& entry : entries) {
      const auto& row =
          entry.kind == prep::Kind::MainShard
              ? shards.items().at(shard_index)
              : meta::Required(artefacts, std::string(entry.logical_name));
      std::string name;
      try {
        meta::Keys(
            row, {"file", "size", "sha256", "recipe_id", "generator_version"});
        name = meta::String(meta::Required(row, "file"));
        if (!SafeRelative(name))
          Fail(fs::u8path(name), "unsafe manifest relative path");
        if (name != entry.destination ||
            meta::Integer(meta::Required(row, "size")) != entry.size ||
            meta::String(meta::Required(row, "sha256")) != entry.sha256)
          Fail(root / fs::u8path(name), "manifest entry mismatch");
        const bool derived = entry.kind == prep::Kind::Twin ||
                             entry.kind == prep::Kind::DraftHead;
        if (entry.kind != prep::Kind::MainShard &&
            meta::String(meta::Required(row, "recipe_id")) != recipe_id)
          Fail(root / fs::u8path(name), "recipe entry mismatch");
        if (derived &&
            (meta::String(meta::Required(row, "recipe_id")) != recipe_id ||
             meta::String(meta::Required(row, "generator_version")) != "1.0.0"))
          Fail(root / fs::u8path(name), "recipe entry mismatch");
        if ((!derived && row.contains("generator_version")) ||
            (entry.kind == prep::Kind::MainShard && row.contains("recipe_id")))
          Fail(root / fs::u8path(name), "unexpected recipe entry");
      } catch (const std::exception& error) {
        Fail(name.empty() ? manifest : root / fs::u8path(name), error.what());
      }
      PackageFile file{std::string(entry.logical_name),
                       std::string(entry.sha256),
                       Absolute(root / fs::u8path(name)),
                       entry.size,
                       false,
                       {},
                       {},
                       false};
      try {
        file.lease =
            std::make_shared<prep::ReadLease>(prep::NativePath(file.path));
        const auto stamp = file.lease->Snapshot();
        if (stamp.size != entry.size)
          Fail(file.path, "size mismatch");
        CheckHeader(file.path);
        if (entry.kind == prep::Kind::MainShard) {
          if (!model.empty() && Absolute(model) == file.path)
            model_named = true;
          std::string error;
          const auto reader = core::GgufReader::OpenHeaderFile(
              prep::NativePath(file.path), &error);
          if (reader->GetMetadataString("general.architecture") !=
                  std::optional<std::string_view>("qwen4exp") ||
              reader->GetMetadataUint32("split.count") !=
                  std::optional<std::uint32_t>(4) ||
              reader->GetMetadataUint32("split.no") !=
                  std::optional<std::uint32_t>(
                      static_cast<std::uint32_t>(shard_index)))
            Fail(file.path, "model family/split header mismatch");
          const auto count = reader->GetMetadataUint64("split.tensors.count");
          if (!count || (total_tensors && count != total_tensors))
            Fail(file.path, "split.tensors.count mismatch");
          total_tensors = count;
          shard_tensors += reader->GetTensorCount();
          if (fingerprints) {
            const auto pin = fingerprints(entry);
            if (!pin)
              Fail(file.path, pin.error);
            if (const auto error = prep::CheckGgufHeader(
                    prep::NativePath(file.path), entry, pin.table);
                !error.empty())
              Fail(file.path, error);
          }
          ++shard_index;
        }
        const auto stored = stamps.find({file.logical_name, file.sha256});
        // Fingerprint-only preparation records target digests. Only a full
        // hash stamp matching this held read lease authorizes cache identity.
        file.observed_verified =
            stored != stamps.end() && stored->second == stamp;
        if (verify)
          Verify(file, result, output);
        if (file.lease->Snapshot() != stamp)
          Fail(file.path, "file identity changed during validation");
      } catch (const std::exception& error) {
        std::error_code availability_error;
        if (!fs::exists(file.path, availability_error) && !availability_error)
          throw ModelMissingError(Display(result.manifest_path) + ": " +
                                  Display(file.path) + ": " + error.what());
        Fail(file.path, error.what());
      }
      if (entry.kind != prep::Kind::MainShard)
        result.defaults[std::string(ModelArtefactSetting(entry.kind))] =
            Display(file.path);
      result.files.push_back(std::move(file));
    }
    if (!model_named)
      Fail(model, "main model is not named by the selected manifest");
    if (!total_tensors || shard_tensors != *total_tensors)
      Fail(manifest,
           "split.tensors.count differs from the combined tensor table");
    if (output && verify)
      *output << "verify-model: checked " << result.hashed_files << " files, "
              << result.hashed_bytes << " bytes\n";
  } catch (const ModelMissingError&) {
    throw;
  } catch (const std::exception& error) {
    Fail(result.manifest_path, error.what());
  }
  if (verify)
    SaveVerification(result, output);
  return result;
}
std::string Aggregate(std::span<const PackageFile* const> files) {
  std::string value = "gguf-sha256-v1";
  for (const auto* file : files) {
    if (!file->observed_verified)
      return {};
    value += ":" + std::to_string(file->size) + ":" + file->sha256;
  }
  return crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}
}  // namespace
std::string_view ModelArtefactSetting(bubo::prepare::Kind kind) {
  switch (kind) {
    case bubo::prepare::Kind::Mtp:
      return "--mtp-model";
    case bubo::prepare::Kind::Mmproj:
      return "--mmproj";
    case bubo::prepare::Kind::Twin:
      return "GUFO_DENSE_TWIN_MODEL";
    case bubo::prepare::Kind::DraftHead:
      return "GUFO_DRAFT_OUT_MODEL";
    default:
      throw std::invalid_argument("not a model artefact");
  }
}
ModelPackage LoadModelPackage(const fs::path& model, const std::string& profile,
                              bool verify, std::ostream* output,
                              const fs::path& selected) {
  const auto selection = bubo::cli::ResolvePreparedDefault(
      model, selected, bubo::cli::EnvironmentUtf8("BUBO_MANIFEST"),
      bubo::cli::EnvironmentUtf8("LOCALAPPDATA"));
  const auto manifest =
      selection.source == "standard_location" && !fs::exists(selection.manifest)
          ? fs::path{}
          : selection.manifest;
  return Load(selection.model, profile, prep::ModelCatalogue(), verify, output,
              prep::ProductionFingerprintTable, manifest, true);
}
#ifdef BUBO_PREPARE_TESTING
ModelPackage LoadModelPackageForTesting(
    const fs::path& model, const std::string& profile,
    std::span<const prep::Entry> entries, bool verify, std::ostream* output,
    const std::function<prep::FingerprintResult(const prep::Entry&)>&
        fingerprints,
    const fs::path& selected, bool select_set) {
  return Load(model, profile, entries, verify, output, fingerprints, selected,
              select_set);
}
#endif
void CheckPackageOverrides(ModelPackage& package,
                           const std::map<std::string, std::string>& paths,
                           bool verify, std::ostream* output) {
  if (package.manifest_path.empty())
    return;
  const std::map<std::string, std::string> names = {
      {"--mtp-model", "mtp"},
      {"--mmproj", "mmproj"},
      {"GUFO_DENSE_TWIN_MODEL", "twin"},
      {"GUFO_DRAFT_OUT_MODEL", "draft_head"}};
  const auto stamps = prep::ReadVerifiedStamps(
      package.manifest_path.parent_path(), package.package_id);
  for (const auto& [key, path] : paths) {
    if (!names.contains(key) || path.empty())
      continue;
    const auto original = std::find_if(
        package.files.begin(), package.files.end(),
        [&](const auto& file) { return file.logical_name == names.at(key); });
    if (original == package.files.end())
      continue;  // Manifest absent preserves upstream unpinned path semantics.
    const auto absolute = Absolute(fs::u8path(path));
    const auto already_named =
        std::find_if(package.files.begin(), package.files.end(),
                     [&](const auto& file) { return file.path == absolute; });
    if (already_named != package.files.end()) {
      if (already_named->logical_name != original->logical_name ||
          already_named->size != original->size ||
          already_named->sha256 != original->sha256)
        Fail(absolute, "override manifest entry mismatch");
      continue;
    }
    PackageFile file{original->logical_name,
                     original->sha256,
                     absolute,
                     original->size,
                     false,
                     {},
                     {},
                     false};
    try {
      file.lease =
          std::make_shared<prep::ReadLease>(prep::NativePath(file.path));
      const auto stamp = file.lease->Snapshot();
      if (stamp.size != file.size)
        Fail(file.path, "override size mismatch");
      CheckHeader(file.path);
      const auto stored = stamps.find({file.logical_name, file.sha256});
      file.observed_verified =
          stored != stamps.end() && stored->second == stamp;
      if (verify)
        Verify(file, package, output);
      if (file.lease->Snapshot() != stamp)
        Fail(file.path, "file identity changed during validation");
    } catch (const std::exception& error) {
      Fail(file.path, error.what());
    }
    package.files.push_back(std::move(file));
  }
  if (verify)
    SaveVerification(package, output);
}
void EnsureDiskCacheIdentity(ModelPackage& package,
                             std::span<const fs::path> selected,
                             std::ostream* output) {
  if (package.manifest_path.empty())
    throw std::invalid_argument(
        "--cache-disk requires a prepared model package; run bubo prepare "
        "or provide --manifest bubo-package.json");
  std::vector<PackageFile*> needed;
  for (auto& file : package.files)
    if (file.logical_name.starts_with("main_") ||
        std::ranges::any_of(selected, [&](const auto& path) {
          return !path.empty() && file.path == Absolute(path);
        }))
      needed.push_back(&file);
  for (const auto& path : selected)
    if (!path.empty() && std::ranges::none_of(needed, [&](const auto* file) {
          return file->path == Absolute(path);
        }))
      Fail(path, "selected file is not named by the pinned manifest");
  const bool missing = std::ranges::any_of(
      needed, [](const auto* file) { return !file->observed_verified; });
  if (output) {
    if (missing)
      *output << "disk cache: verifying model files once (about 2-4 minutes); "
                 "later starts skip this\n";
    else
      *output << "disk cache: model identity from verification stamp\n";
  }
  for (auto* file : needed)
    if (!file->observed_verified)
      Verify(*file, package, output);
  if (missing) {
    if (output)
      *output << "verify-model: checked " << package.hashed_files << " files, "
              << package.hashed_bytes << " bytes\n";
    SaveVerification(package, output);
  }
}
std::string ModelPackage::MainIdentity() const {
  std::vector<const PackageFile*> main;
  for (const auto& file : files)
    if (file.logical_name.starts_with("main_"))
      main.push_back(&file);
  return main.size() == 4 ? Aggregate(main) : std::string{};
}
std::string ModelPackage::IdentityFor(const fs::path& path) const {
  for (const auto& file : files)
    if (file.path == Absolute(path)) {
      const std::array<const PackageFile*, 1> selected{&file};
      return Aggregate(selected);
    }
  return {};
}
}  // namespace gufo::server
