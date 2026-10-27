#pragma once

// Model catalogue shipped with the CLI. The three Flash Next entries are built
// from the compiled package pins (release_catalogue), with names and identities
// generated from model_catalogue.json; every other entry comes from that JSON.
// There is no runtime override. ParseCatalogue exists so tests can load a
// synthetic catalogue.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::cli {

struct CatalogueLicence {
  std::string id, name, url;
  // The exact line the download prompt shows.
  std::string prompt_line;
  // Extra caution shown with the line (for example "upstream file not found").
  std::string note;
  // true: the upstream licence file was read. false: the card states the
  // licence only. An entry with false needs a note to be downloadable.
  bool verified_upstream = false;
};

// Speed measured on our engine, one source for the GUI. Never guessed: the
// whole object is absent (null in JSON) when nothing was measured; a field
// that was not measured is null. `source` and `label` are always set.
struct CatalogueSpeed {
  bool present = false;
  std::optional<double> writing_tok_s;  // decode
  std::optional<double> reading_tok_s;  // prefill
  std::optional<std::uint64_t> prompt_tokens;
  std::string source, label;
};

// The same model on stock llama.cpp, from a user llama-bench run. Data
// only. Absent (null in JSON) when no comparison exists.
struct CatalogueStockComparison {
  bool present = false;
  std::string engine, source;
  double prefill_pp8192 = 0, decode_tg128 = 0;
};

struct CatalogueFile {
  std::string name, repo_path, url, sha256;
  std::uint64_t bytes = 0;
  // main, mtp, draft_head, twin, mmproj, or support (config and tokenizer
  // files of a safetensors folder).
  std::string role;
  // A file that is not part of the pin the entry owns (optional side file).
  std::string repository, commit;
  bool optional = false;
  // false: never downloaded by prepare (licence not read).
  bool downloadable = true;
  // "not yet read" for an unread side-file licence. A downloaded file from
  // another repository has its own licence line here; prepare writes it to
  // licenses/LICENSE-INFO-<name>.txt and shows it in the plan.
  std::string licence_text;
};

struct CatalogueEntry {
  std::string id, display_name;
  std::string technical_name, about, selection_label;
  std::vector<std::string> about_links;
  // Weight placement only, excluding conversation caches and work buffers.
  // Absent unless exact tensor arithmetic or a cited measurement is available.
  std::optional<std::uint64_t> gpu_bytes, system_memory_bytes;
  std::string system_memory_note, size_split_source;
  std::string role;          // recommended | supported
  std::string architecture;  // general.architecture or model_type
  std::string set;           // uncensored | original | ""
  std::string repository, commit;
  std::vector<CatalogueFile> files;
  CatalogueLicence licence;
  std::string default_status;  // none | to be decided | default
  std::string family_group;    // entries that are alternatives of one model
  std::string
      default_note;  // one measured line on why this entry is the default
  bool download_enabled = false;
  std::string disabled_reason;
  std::string packaging;  // "" (gguf) | safetensors_folder
  std::string estimate_note;
  std::vector<std::string> estimate_pointers;
  CatalogueSpeed speed;
  CatalogueStockComparison stock_comparison;
  // Main resident storage measured by decode; not inferred from download size.
  // The lowmemory has an independent, initially absent coefficient data slot.
  std::optional<std::uint64_t> measured_main_resident_bytes;
  // Set for the three Flash Next entries: prepare uses its fixed package path.
  std::optional<prepare::ModelSet> flash_next_set;
};

struct CatalogueLoad {
  std::vector<CatalogueEntry> entries;
  std::string error;  // empty when the catalogue is valid
};

// Empty when the entry carries a full pin (https URL on the pinned repository
// and commit, byte count, SHA-256 on every downloadable file; a licence id,
// name, URL and prompt line; a note when the upstream licence was not read).
std::string PinProblem(const CatalogueEntry&);
// Pin rules plus structural rules. A download_enabled entry with a pin
// problem is an error. A disabled entry needs a reason.
std::string ValidateCatalogue(const std::vector<CatalogueEntry>&);
CatalogueLoad ParseCatalogue(std::string_view json_text);
// The catalogue compiled into this program (validated once).
const CatalogueLoad& ShippedCatalogue();
const CatalogueEntry* FindCatalogueEntry(const std::vector<CatalogueEntry>&,
                                         std::string_view id);
std::uint64_t CatalogueDownloadBytes(const CatalogueEntry&);
gufo::json::Value CatalogueEntryJson(const CatalogueEntry&);

// Where prepare puts a catalogue file inside the package folder: the pinned
// destination for Flash Next, the catalogue name for any other entry.
std::filesystem::path CatalogueFilePath(const CatalogueEntry&,
                                        const CatalogueFile&,
                                        const std::filesystem::path& folder);
struct CatalogueReadiness {
  std::string state;  // prepared | partial | not_prepared
  std::vector<std::string> missing_files;
};
// The size check of a catalogue package whose files are in `folder`: every
// downloadable file at its pinned size, and for Flash Next the package marker.
// File sizes only; nothing is read or hashed.
CatalogueReadiness CatalogueReadinessAt(const CatalogueEntry&,
                                        const std::filesystem::path& folder);
// One `prepared_models` row (models list, status, serve): id, display_name,
// manifest_path and model_path (null when empty), size_bytes, prepared_state
// and capabilities. model_path is the absolute main model file.
gufo::json::Value PreparedModelRow(const CatalogueEntry&,
                                   const std::string& manifest_path,
                                   const std::string& model_path,
                                   bool prepared);

}  // namespace bubo::cli
