#pragma once

#include <filesystem>
#include <functional>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

#include "src/cli/prepare/prepare.hpp"
#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/release_catalogue.hpp"
#ifdef BUBO_PREPARE_TESTING
#include "src/core/prepare/transport.hpp"
#endif

namespace bubo::prepare {
std::string ModelConsent(std::span<const Entry> entries,
                         const std::filesystem::path& model_root,
                         std::span<const std::string> status,
                         std::uint64_t remaining,
                         std::span<const ConsentVolume> volumes,
                         bool model_selected = true);
int RunModel(const Options&, std::ostream& out, std::ostream& err,
             const std::function<bool()>& cancelled = {},
             PreparationPlan* preview = nullptr, bool approved = false);
#ifdef BUBO_PREPARE_TESTING
struct TestNotice {
  std::string name, text, sha256;
};
// This type and entry point are absent from the production library. There is
// no environment variable, CLI switch or runtime catalogue override.
struct ModelTestEnvironment {
  std::span<const Entry> entries;
  std::vector<TestNotice> notices;
  Transport* transport = nullptr;
  DownloadOptions download_options;
  std::function<VolumeInfo(const std::filesystem::path&)> volume;
  std::function<std::string(const Entry&, const std::filesystem::path&)>
      fingerprint;
  std::function<bool()> cancelled;
  // Crash injection before readiness publication, after all other writes.
  std::function<bool()> before_manifest;
};
int RunModelForTesting(const Options&, const ModelTestEnvironment&,
                       std::istream& input, std::ostream& out,
                       std::ostream& err, PreparationPlan* preview = nullptr,
                       bool approved = false);
#endif
}  // namespace bubo::prepare
