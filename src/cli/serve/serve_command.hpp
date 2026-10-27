#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "src/cli/serve/lifecycle.hpp"

namespace gufo::cli {
struct ServeCommandOptions {
  std::vector<const char*> filtered;
  std::optional<std::filesystem::path> log_path;
  std::uint64_t log_max_bytes = 8 * 1024 * 1024;
  unsigned log_retain = 3;
  bool supplied_log_flags = false;
};
ServeCommandOptions ParseServeCommandOptions(std::span<const char* const> args);

using ServeRunner =
    std::function<int(std::span<const char* const>, bubo::cli::ServeLifecycle*,
                      const std::optional<std::filesystem::path>&)>;
// The production runner loads models. An isolated CPU target injects a fake
// runner; the command parser, sinks, lifecycle and errors remain production.
int RunServeCommand(std::span<const char* const> args,
                    const ServeRunner& runner);
}  // namespace gufo::cli
