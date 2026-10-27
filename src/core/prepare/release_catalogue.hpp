#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "model_identity.hpp"

namespace bubo::prepare {
// Source: ADDENDUM-20261011-1750.md section 2; owner evidence closes these
// slots.
inline constexpr std::string_view PIN_SLOT_UNSLOTH_COMMIT =
    "766911a6b7369840a91dbcd95f9f997acaab6cd6";
inline constexpr std::string_view PIN_SLOT_BUBO_COMMIT =
    "bb72204d4f9bdd1568c225cce3bf878673009780";
inline constexpr std::string_view BUBO_HF_REPO_ID =
    "HaloForgeLabs/bubo-qwen3.8-flash-next";
inline constexpr std::string_view kHuihuiCommit =
    "73e9eb7c69fdbf667def63e233e70845e7d5ac0b";
// Second model set: the original (not abliterated) unsloth shards with the
// HaloForgeLabs twin-base and draft head. Same eight-row layout.
inline constexpr std::string_view kOriginalBuboCommit =
    "0a3c343bec2095f43f2906371c83e4faa8a47da1";
inline constexpr std::string_view kRecipeId =
    "huihui-ud-q4-k-xl-iq4xs-nofullattn-nohc-q4k-v1";
inline constexpr std::string_view kOriginalRecipeId =
    "base-ud-q4-k-xl-iq4xs-nofullattn-nohc-q4k-v1";
enum class ModelSet { Huihui, Original, LowMemory };
// Support: config and tokenizer files of a catalogue download (model
// catalogue entries only; the fixed Flash Next rows never use it).
enum class Kind { MainShard, Mtp, Mmproj, Twin, DraftHead, Support };
struct Entry {
  std::string_view logical_name;
  std::string_view file_name;
  std::uint64_t size;
  std::string_view sha256;
  std::string_view source_repo;
  std::string_view commit;
  std::string_view repo_path;
  Kind kind;
  std::string_view licence_id;
  std::string_view destination;
  // Relative evidence slot. J2 generates a table from each production shard;
  // an absent table is OPEN, never an empty table treated as verified.
  std::string_view fingerprint_table_slot = {};
  // Internal compiled AMD wheel origin; model entries always leave this empty.
  std::string_view direct_url = {};
};
// The default set (Huihui). Existing callers keep this overload unchanged.
const std::array<Entry, 8>& ModelCatalogue();
const std::array<Entry, 8>& ModelCatalogue(ModelSet set);
std::string_view PackageId(ModelSet set);
std::string_view RecipeId(ModelSet set);
// CLI name of a set: "huihui", "original" or "lowmemory".
std::string_view ModelSetName(ModelSet set);
std::optional<ModelSet> ModelSetFromName(std::string_view name);
std::optional<ModelSet> ModelSetFromPackageId(std::string_view package_id);
std::string ValidateFields(std::span<const Entry> entries, bool release);
bool IsLowerHex(std::string_view text, std::size_t length);
bool IsRelativePath(std::string_view path);
}  // namespace bubo::prepare
