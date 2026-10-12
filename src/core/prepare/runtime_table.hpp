#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace bubo::prepare {
struct WheelPin {
  std::string_view name, file, url;
  std::uint64_t size;
  std::string_view sha256;
};
const std::array<WheelPin, 3>& RuntimeWheels();
struct RuntimeMember {
  unsigned wheel = 0;
  std::string member, destination;
  std::uint64_t size = 0;
  std::string sha256;
};
std::vector<RuntimeMember> LoadRuntimeTable(std::string_view text,
                                            bool production = true);
std::string SerializeRuntimeTable(const std::vector<RuntimeMember>& rows);
std::vector<RuntimeMember> CompiledRuntimeTable();
// Release tooling only. No network; production mode first verifies each whole
// archive against Appendix B, then requires all 263 rows and Appendix A DLLs.
std::vector<RuntimeMember> GenerateRuntimeTable(
    const std::array<std::filesystem::path, 3>& wheels, bool production = true);
}  // namespace bubo::prepare
