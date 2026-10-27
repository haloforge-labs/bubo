#pragma once

#include "src/cli/instance_control.hpp"
#include "src/cli/serve/memory_estimate.hpp"

namespace bubo::cli {
struct ServeMemoryPolicy {
  std::optional<std::uint64_t> context;
  gufo::json::Value estimate;
  std::string estimate_reason;
};
ServeMemoryPolicy ResolveServeMemoryPolicy(
    gufo::server::ReleaseResolution&, const std::string& package_id,
    const std::filesystem::path& model, std::uint64_t sessions,
    const PlanningMemoryBudgets&, bool explicit_context,
    bool inspect_only = false, bool release_active = true);
}  // namespace bubo::cli
