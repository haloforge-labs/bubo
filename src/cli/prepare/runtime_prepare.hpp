#pragma once
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <vector>

#include "src/cli/prepare/prepare.hpp"
#include "src/core/prepare/runtime_notices.hpp"
#include "src/core/prepare/runtime_table.hpp"
#include "src/core/prepare/transport.hpp"

namespace bubo::prepare {
std::string RuntimeConsent(const std::filesystem::path& runtime,
                           const std::filesystem::path& licenses);
std::vector<std::string> RuntimeHolders(
    const std::filesystem::path& root,
    const std::vector<RuntimeMember>& members);
int RunRuntime(const Options&, std::ostream&, std::ostream&,
               const std::function<bool()>& cancelled = {});
#ifdef BUBO_PREPARE_TESTING
struct RuntimeTestEnvironment {
  std::array<WheelPin, 3> wheels;
  std::vector<RuntimeMember> members;
  std::vector<RuntimeNotice> notices;
  Transport* transport = nullptr;
  DownloadOptions download_options;
  std::array<std::string, 3> origins;
  std::function<VolumeInfo(const std::filesystem::path&)> volume;
  std::function<std::vector<std::string>()> holders;
  // Return false to fail after a namespace move. Same transaction/rollback.
  std::function<bool(std::size_t)> after_install;
  std::function<bool()> cancelled;
};
int RunRuntimeForTesting(const Options&, const RuntimeTestEnvironment&,
                         std::istream&, std::ostream&, std::ostream&);
#endif
}  // namespace bubo::prepare
