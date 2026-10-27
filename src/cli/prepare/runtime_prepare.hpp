#pragma once
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <vector>

#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/core/prepare/runtime_notices.hpp"
#include "src/core/prepare/runtime_table.hpp"
#include "src/core/prepare/transport.hpp"

namespace bubo::prepare {
std::string RuntimeConsent(const std::filesystem::path& runtime,
                           const std::filesystem::path& licenses);
std::vector<std::string> RuntimeHolders(
    const std::filesystem::path& root,
    const std::vector<RuntimeMember>& members);
// Legacy manually installed runtimes have no marker. A present tool marker or
// journal is authoritative: cheap metadata/length checks precede engine load.
std::string RuntimeReadyError(const std::filesystem::path& root);
int RunRuntime(const Options&, std::ostream&, std::ostream&,
               const std::function<bool()>& cancelled = {},
               PreparationPlan* preview = nullptr, bool approved = false);
#ifdef BUBO_PREPARE_TESTING
struct RuntimeTestEnvironment {
  // Isolated process inputs for the production runtime resolver. In
  // particular, test output roots must never select a real user's runtime.
  RuntimeSearch runtime_search;
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
  // End-to-end fake-wheel plans can disclose their synthetic pins explicitly.
  // Existing binding-text fixtures continue to render the production block.
  bool synthetic_disclosure = false;
};
int RunRuntimeForTesting(const Options&, const RuntimeTestEnvironment&,
                         std::istream&, std::ostream&, std::ostream&,
                         PreparationPlan* preview = nullptr,
                         bool approved = false);
std::string RuntimeReadyErrorForTesting(const std::filesystem::path& root,
                                        const RuntimeTestEnvironment&);
#endif
}  // namespace bubo::prepare
