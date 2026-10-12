#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace bubo::prepare {
struct ZipMember {
  std::string name;
  std::uint64_t compressed = 0, size = 0, offset = 0, data_offset = 0;
  std::uint32_t crc = 0;
  std::uint16_t method = 0, flags = 0;
};
// Strict Windows-portable relative names; used for archive and destination
// validation, including unselected archive members.
bool SafeZipPath(std::string_view name, bool directory = false);
class ZipReader {
public:
  explicit ZipReader(const std::filesystem::path& path);
  const std::vector<ZipMember>& members() const { return members_; }
  // Streams only this member. Output cannot exceed the trusted limit. CRC,
  // exact stream termination and length are checked before returning a digest.
  std::string Extract(
      const ZipMember& member, std::uint64_t limit,
      const std::function<void(std::span<const std::uint8_t>)>& sink,
      std::string_view expected_sha256 = {},
      const std::function<bool()>& cancelled = {}) const;

private:
  std::filesystem::path path_;
  std::vector<ZipMember> members_;
};
}  // namespace bubo::prepare
