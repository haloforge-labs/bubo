#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace bubo::prepare {
// Source: ADDENDUM-20261011-1750.md section 2; owner evidence closes these
// slots.
inline constexpr std::string_view PIN_SLOT_UNSLOTH_COMMIT = "";
inline constexpr std::string_view PIN_SLOT_BUBO_COMMIT = "";
inline constexpr std::string_view BUBO_HF_REPO_ID = "PIN_SLOT_BUBO_REPO_ID";
inline constexpr std::string_view kHuihuiCommit =
    "73e9eb7c69fdbf667def63e233e70845e7d5ac0b";
inline constexpr std::string_view kPackageId =
    "huihui-qwen38-flash-next-ud-q4-k-xl-v1";
enum class Kind { MainShard, Mtp, Mmproj, Twin, DraftHead };
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
const std::array<Entry, 8>& ModelCatalogue();
std::string ValidateFields(std::span<const Entry> entries, bool release);
bool IsLowerHex(std::string_view text, std::size_t length);
bool IsRelativePath(std::string_view path);
}  // namespace bubo::prepare
