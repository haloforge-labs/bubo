#pragma once

#include <charconv>

#include "src/cli/serve/release_profile.hpp"
#include "src/cli/settings_support.hpp"

namespace bubo::cli {
struct ReleaseMemoryEstimate {
  MemoryEstimateInputs inputs;
  Value value;
  std::string reason;
};
// All planning paths consume the same resolved release settings, including
// registry built-ins when no profile/user value was injected. Runtime context
// and NPU results refine this same input after loading; budgets stay pre-load.
inline ReleaseMemoryEstimate BuildReleaseMemoryEstimate(
    const gufo::server::ReleaseResolution& resolved, std::string package_id,
    std::optional<std::uint64_t> gpu_available,
    std::optional<std::uint64_t> host_available,
    const MemoryModelTable* table = nullptr,
    const PlanningMemoryBudgets& capacity_budgets = {}) {
  const auto value = [&](std::string_view name) -> std::optional<std::string> {
    for (const auto& setting : resolved.settings)
      if (setting.name == name && setting.value)
        return setting.value;
    for (const auto& row : gufo::server::ReleaseSwitchTable())
      if (row.member_str("name") == name) {
        const auto* builtin = row.find("built_in");
        if (builtin && builtin->is_string())
          return builtin->str();
        return std::nullopt;
      }
    throw std::invalid_argument("unknown estimate setting: " +
                                std::string(name));
  };
  const auto number = [&](std::string_view name) {
    const auto text = value(name);
    std::uint64_t out = 0;
    if (text) {
      const auto [end, ec] =
          std::from_chars(text->data(), text->data() + text->size(), out);
      if (ec != std::errc{} || end != text->data() + text->size())
        throw std::invalid_argument("invalid estimate setting: " +
                                    std::string(name));
    }
    return out;
  };
  MemoryEstimateInputs inputs;
  inputs.package_id = std::move(package_id);
  inputs.profile = resolved.profile;
  inputs.context = resolved.runtime_context.value_or(number("--context"));
  if (inputs.context == 0)
    inputs.context = resolved.native_context.value_or(0);
  inputs.sessions = number("--sessions");
  // --cache-ram-bytes off (the word the release rule accepts next to a number)
  // is no RAM cache, like --no-cache.
  const bool cache_off =
      value("--no-cache") == std::optional<std::string>("1") ||
      value("--cache-ram-bytes") == std::optional<std::string>("off");
  inputs.ram_cache_bytes = cache_off ? 0 : number("--cache-ram-bytes");
  inputs.cache_automatic = !cache_off && inputs.ram_cache_bytes == 0;





  inputs.cache_retention_lazy =
      !cache_off && resolved.model_family == "qwen3.8-flash-next";
  inputs.mtp = !resolved.artefacts.mtp.empty() &&
               value("--speculative") == std::optional<std::string>("mtp");
  inputs.kv_q8 = value("--kv-cache-type") == std::optional<std::string>("q8_0");
  inputs.twin = !resolved.artefacts.twin.empty();
  inputs.draft_head = !resolved.artefacts.draft_head.empty();
  inputs.vision =
      !resolved.artefacts.mmproj.empty() && !resolved.artefacts.vision_disabled;
  inputs.npu_prefill = resolved.runtime_npu_enabled.value_or(
      resolved.prefill_authorization.Allows(
          gufo::core::ExperimentalFeature::Prefill));
  inputs.available_bytes = gpu_available;
  inputs.host_available_bytes = host_available;
  inputs.capacity_budgets = capacity_budgets;
  const auto floor = value("GUFO_CACHE_HOST_FLOOR_MB");
  inputs.host_floor_bytes =
      PlanningHostFloorBytes(floor ? floor->c_str() : nullptr);
  const auto cap = value("GUFO_MAX_IMAGE_TOKENS");
  inputs.image_token_cap = PlanningImageTokenCap(cap ? cap->c_str() : nullptr);
  ReleaseMemoryEstimate result;
  result.inputs = std::move(inputs);
  result.value =
      table ? MemoryEstimateForTable(*table, result.inputs, &result.reason)
            : MemoryEstimate(result.inputs, &result.reason);
  return result;
}
// Catalogue and local package estimates resolve the same default context as
// serve, for several sessions on every package.
inline ReleaseMemoryEstimate BuildPlanningReleaseMemoryEstimate(
    gufo::server::ReleaseResolution& resolved, const std::string& package_id,
    std::optional<std::uint64_t> gpu_available,
    std::optional<std::uint64_t> host_available, const MemoryModelTable* table,
    const PlanningMemoryBudgets& budgets, bool explicit_context) {
  auto result = BuildReleaseMemoryEstimate(resolved, package_id, gpu_available,
                                           host_available, table, budgets);
  const bool memory96 = Is96GiBStrixHalo(budgets);
  // serve resolves the recommended context for several sessions on every
  // package (ResolveServeMemoryPolicy); the estimate must judge the same
  // context, or a start the recommender calls a fit looks tight to a caller
  // that omits --context.
  if (explicit_context || (!memory96 && result.inputs.sessions <= 1))
    return result;
  const auto* recommendation = result.value.find("recommended_context");
  const auto* rows =
      recommendation ? recommendation->find("by_sessions") : nullptr;
  if (rows && rows->is_array())
    for (const auto& row : rows->items()) {
      const auto* fits = row.find("fits");
      const auto* recommended = row.find("recommended");
      if (row.member_size("sessions") != result.inputs.sessions || !fits ||
          !fits->is_bool() || !fits->as_bool() || !recommended ||
          !recommended->is_bool() || !recommended->as_bool())
        continue;
      if (gufo::server::ResolveDefaultSessionContext(
              resolved, result.inputs.sessions, result.value, memory96))
        result = BuildReleaseMemoryEstimate(resolved, package_id, gpu_available,
                                            host_available, table, budgets);
      break;
    }
  return result;
}
}  // namespace bubo::cli
