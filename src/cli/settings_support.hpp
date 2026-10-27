#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/cli/client_snippets.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/model_catalogue.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/image.hpp"
#include "src/cli/memory_coefficients.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::cli {
inline Value SettingsProfiles() {
  auto profiles = Value::array();
  for (const auto* name : {"default", "exact", "vanilla", "subagents"}) {
    auto row = Value::object();
    row["name"] = name;
    std::string label(name);
    label[0] = static_cast<char>(label[0] - 'a' + 'A');
    if (std::string_view(name) == "vanilla")
      label = "Vanilla (no Bubo tuning)";
    row["label"] = label;
    row["gui"] = std::string_view(name) != "subagents";
    profiles.push_back(std::move(row));
  }
  return profiles;
}





struct MemoryCoefficientsV1 {
  static constexpr unsigned version = 1;
  static constexpr std::uint64_t context = 262144;
  static constexpr std::uint64_t session_mib = 7176;
  static constexpr std::uint64_t fixed_mib = 259;
  static constexpr std::uint64_t main_weights_bytes = 82500000000ULL;
  static constexpr std::uint64_t mtp_bytes = 2790000000ULL;
  static constexpr std::uint64_t twin_bytes = 1230000000ULL;
  static constexpr std::uint64_t draft_head_bytes = 360000000ULL;
  static constexpr std::uint64_t vision_bytes = 910000000ULL;
  // Decode-approved: load runtime overhead + deferred scratch, once per model.
  static constexpr std::uint64_t scratch_min_bytes = 3460000000ULL;
  static constexpr std::uint64_t scratch_max_bytes = 3460000000ULL;
  static constexpr std::uint64_t reserve_bytes = 4ULL * 1024 * 1024 * 1024;
};
inline double BytesToGiB(std::uint64_t bytes) {
  return static_cast<double>(bytes) / 1073741824.0;
}
inline double BytesToGB(std::uint64_t bytes) {
  return static_cast<double>(bytes) / 1000000000.0;
}
inline std::uint64_t MiBToBytes(std::uint64_t mib) {
  return mib * 1048576ULL;
}
inline Value MemoryAmount(std::uint64_t bytes, std::string_view evidence) {
  auto value = Value::object();
  value["bytes"] = bytes;
  value["gib"] = BytesToGiB(bytes);
  value["evidence"] = std::string(evidence);
  return value;
}
// One coefficient table per model package. The estimator and the recommended
// context read only this struct, so a new model adds one table and one row in
// MemoryTableFor. Nothing below assumes the shipped package.
struct MemoryModelTable {
  std::string_view package_id;
  unsigned version = 0;
  std::uint64_t reference_context = 0;  // context of the measured coefficient
  std::uint64_t session_mib = 0, fixed_mib = 0;
  std::uint64_t main_weights_bytes = 0, mtp_bytes = 0, twin_bytes = 0,
                draft_head_bytes = 0, vision_bytes = 0;
  std::uint64_t scratch_min_bytes = 0, scratch_max_bytes = 0, reserve_bytes = 0;
  std::uint64_t estimate_context_limit = 0;  // estimator input range
  std::uint64_t context_max = 0;             // the model's own context limit
  std::uint64_t context_min = 0;   // smallest context ever recommended
  std::uint64_t max_sessions = 0;  // estimator input range
  std::string_view reference_source_path;
  // A3B and Gemma caches are host vectors. Flash Next cache and copy-on-write
  // snapshot storage use the device carve; borrowed snapshots do not allocate
  // an unconditional second copy of the full session.
  bool cache_on_host = false;
  std::uint64_t vision_hidden_size = 0;
};
inline const MemoryModelTable* MemoryTableFor(std::string_view package_id) {
  using C = MemoryCoefficientsV1;
  static const MemoryModelTable flash_next{
      prepare::kPackageId,
      C::version,
      C::context,
      C::session_mib,
      C::fixed_mib,
      C::main_weights_bytes,
      C::mtp_bytes,
      C::twin_bytes,
      C::draft_head_bytes,
      C::vision_bytes,
      C::scratch_min_bytes,
      C::scratch_max_bytes,
      C::reserve_bytes,
      1048576,
      262144,
      4096,
      64,
      "Bubo memory model, calibrated on Strix Halo 128 GB",
      false,
      2560};
  static const MemoryModelTable lowmemory = [&] {
    auto table = flash_next;
    table.package_id = prepare::PackageId(prepare::ModelSet::LowMemory);
    table.main_weights_bytes = kLowMemoryCalibratedMainResidentBytes;
    return table;
  }();
  if (package_id == lowmemory.package_id)
    return &lowmemory;
  if (prepare::ModelSetFromPackageId(package_id))
    return &flash_next;
  return nullptr;
}
// The planner keeps the same minimum OS headroom as explicit cache limits, even
// when the optional snapshot-admission floor is disabled. Larger overrides
// apply here too; malformed/negative overrides retain the planning minimum.
inline std::uint64_t PlanningHostFloorBytes(
    const char* raw = std::getenv("GUFO_CACHE_HOST_FLOOR_MB")) {
  auto floor = gufo::server::kHostSnapshotHeadroomBytes;
  if (raw) {
    const std::string_view text(raw);
    const auto first = text.find_first_not_of(" \t\r\n\v\f");
    if (first == std::string_view::npos || text[first] == '-')
      return floor;
    // Match snapshot admission's strtoull grammar, including leading ASCII
    // whitespace and '+'. Never let overflow wrap or become a small floor.
    char* end = nullptr;
    errno = 0;
    const std::uint64_t mib = std::strtoull(raw, &end, 10);
    if (end != raw && *end == '\0') {
      constexpr std::uint64_t kMiB = 1048576;
      constexpr std::uint64_t kMaxBytes = 9007199254740991;
      if (errno == ERANGE || mib > kMaxBytes / kMiB)
        return kMaxBytes;
      floor = std::max(floor, mib * kMiB);
    }
  }
  return floor;
}
inline std::optional<std::uint64_t> PlanningImageTokenCap(
    const char* raw = std::getenv("GUFO_MAX_IMAGE_TOKENS")) {
  if (!raw || !*raw)
    return std::nullopt;
  // Match Model::Load's strtoul grammar and uint32 bound before any GPU work.
  char* end = nullptr;
  errno = 0;
  const auto value = std::strtoul(raw, &end, 10);
  if (end == raw || *end || errno == ERANGE || value == 0 ||
      value > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument(
        "GUFO_MAX_IMAGE_TOKENS must be a positive integer");
  return value;
}
struct MemoryEstimateInputs {
  std::string package_id;
  std::string profile;
  std::uint64_t context = 0, sessions = 0, ram_cache_bytes = 0;
  bool mtp = false, twin = false, draft_head = false, vision = false;
  std::optional<std::uint64_t> carve_budget_bytes, host_budget_bytes;
  // GPU and host availability are sampled before model loading. Legacy budget
  // inputs above remain accepted for callers that supply explicit observations.
  std::optional<std::uint64_t> available_bytes, host_available_bytes;
  std::uint64_t snapshot_pool_bytes = 0;
  bool npu_prefill = false;
  bool cache_automatic = false;
  // A retention ceiling is not an allocation at startup. Flash Next borrows
  // live pages and allocates additional snapshot storage on demand. Explicit
  // snapshot_pool_bytes still charges any separately supplied pool allowance.
  bool cache_retention_lazy = false;
  PlanningMemoryBudgets capacity_budgets;
  // Planning floor, independent of whether snapshot admission is enabled.
  std::uint64_t host_floor_bytes = PlanningHostFloorBytes();
  std::optional<std::uint64_t> image_token_cap = PlanningImageTokenCap();
};
struct MemoryTotals {
  std::uint64_t weights = 0, per_session = 0, session_total = 0,
                vision_per_session = 0, vision_total = 0, host_allowance = 0,
                device_cache = 0, host_need = 0, total_min = 0, total_max = 0;
};
// The one estimator. Context and sessions are explicit so the recommended
// context uses the same arithmetic as the estimate.
inline MemoryTotals ComputeMemoryTotals(const MemoryModelTable& table,
                                        const MemoryEstimateInputs& inputs,
                                        std::uint64_t context,
                                        std::uint64_t sessions) {
  MemoryTotals t;
  t.weights = table.main_weights_bytes + (inputs.mtp ? table.mtp_bytes : 0) +
              (inputs.twin ? table.twin_bytes : 0) +
              (inputs.draft_head ? table.draft_head_bytes : 0) +
              (inputs.vision ? table.vision_bytes : 0);
  // The fixed/context split is an estimate. Only the aggregate reference
  // coefficient has measurement evidence. Keep it separate in the output.
  t.per_session = MiBToBytes(table.fixed_mib) +
                  (MiBToBytes(table.session_mib - table.fixed_mib) * context +
                   table.reference_context - 1) /
                      table.reference_context;




  if (inputs.vision && table.vision_hidden_size)
    t.vision_per_session =
        12 * context +
        4 * table.vision_hidden_size *
            std::min<std::uint64_t>(
                context, inputs.image_token_cap.value_or(
                             gufo::core::kDefaultRequestImageTokens)) +
        64;
  t.vision_total = t.vision_per_session * sessions;
  t.session_total = t.per_session * sessions + t.vision_total;
  t.host_allowance = table.cache_on_host ? inputs.ram_cache_bytes : 0;
  t.device_cache = (table.cache_on_host || inputs.cache_retention_lazy
                        ? 0
                        : inputs.ram_cache_bytes) +
                   inputs.snapshot_pool_bytes;


  t.host_need = 85ULL * 1073741824ULL / 100 +
                (inputs.npu_prefill ? 7ULL * 1073741824ULL / 2 : 0) +
                t.host_allowance;
  t.total_min =
      t.weights + t.session_total + table.scratch_min_bytes + t.device_cache;
  t.total_max =
      t.weights + t.session_total + table.scratch_max_bytes + t.device_cache;
  return t;
}
inline constexpr std::uint64_t kRecommendedContextStep = 8192;
// Admission excludes the snapshot-pool reserve. Recommendations retain UX
// headroom, independently of the engine's ability to admit the sessions.
inline std::uint64_t RecommendedMargin(std::uint64_t available) {
  return std::max<std::uint64_t>(5ULL * 1073741824, available / 20);
}
inline std::optional<std::uint64_t> GpuAvailable(
    const MemoryEstimateInputs& in) {
  return in.available_bytes ? in.available_bytes : in.carve_budget_bytes;
}
inline std::optional<std::uint64_t> HostAvailable(
    const MemoryEstimateInputs& in) {
  return in.host_available_bytes ? in.host_available_bytes
                                 : in.host_budget_bytes;
}
struct MemoryRecommendationFailures {
  bool gpu_headroom = false, host_floor = false, unconfirmed = false;
};
inline void WriteMemoryRecommendationFailures(
    Value& value, const MemoryRecommendationFailures& failures) {
  auto terms = Value::array();
  if (failures.gpu_headroom)
    terms.push_back("gpu_headroom");
  if (failures.host_floor)
    terms.push_back("host_floor");
  if (failures.unconfirmed)
    terms.push_back("unconfirmed");
  value["recommendation_failures"] = std::move(terms);
}
inline std::string GiBText(double gib);
inline std::string MemoryFitReason(
    const MemoryTotals& totals, const MemoryEstimateInputs& inputs,
    std::uint64_t gpu_available,
    MemoryRecommendationFailures* failures = nullptr) {
  std::string reason;
  // Admission shortfalls already prove the corresponding recommendation term
  // fails, including when growth/spill readings are unavailable. Capture these
  // existing comparisons rather than re-evaluating them for the wire tags.
  if (totals.total_min > gpu_available) {
    reason = "GPU dedicated memory is short.";
    if (failures)
      failures->gpu_headroom = true;
  }
  const auto host = HostAvailable(inputs);
  if (!host || totals.host_need + inputs.host_floor_bytes > *host) {
    if (failures) {
      failures->host_floor = host.has_value();
      failures->unconfirmed = !host;
    }
    if (!reason.empty())
      reason += " ";
    reason += host ? "Host RAM is short after its floor."
                   : "Host available RAM is not reported.";
  }
  // R2, the same arithmetic as the engine's start check (memory_guard.hpp PlanStateFit): the GPU budget (dedicated carve-out plus a slice of
  // system RAM) overlaps the RAM reading, so what does not fit the dedicated carve-out lands in RAM and RAM has to keep its floor for it. The
  // two checks above each look at one side; this one looks at both.
  const auto dedicated = inputs.capacity_budgets.gpu_dedicated;
  if (dedicated && host && totals.total_min > *dedicated) {
    const auto spill = totals.total_min - *dedicated;
    if (totals.host_need + spill + inputs.host_floor_bytes > *host) {
      if (failures) {
        failures->gpu_headroom = true;
        failures->host_floor = true;
      }
      if (!reason.empty())
        reason += " ";
      reason += "Memory is short: " + GiBText(BytesToGiB(spill)) +
                " of the GPU need would use system RAM, and RAM would fall under its floor.";
    }
  }
  return reason;
}
struct MemoryRecommendation {
  bool recommended = false;
  MemoryRecommendationFailures failures;
  std::optional<std::uint64_t> cache_at_clamp = 0, host_spill;
  std::optional<double> free_after_growth_gib, host_free_after_spill_gib;
  std::string reason;
};
inline std::string GiBText(double gib) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << gib << " GiB";
  return out.str();
}
inline MemoryRecommendation ComputeMemoryRecommendation(
    const MemoryModelTable& table, const MemoryTotals& totals,
    const MemoryEstimateInputs& inputs, std::uint64_t available) {
  MemoryRecommendation result;
  result.reason = MemoryFitReason(totals, inputs, available, &result.failures);
  result.recommended = result.reason.empty();
  const auto append = [&](const std::string& text) {
    if (!result.reason.empty())
      result.reason += " ";
    result.reason += text;
  };
  if (inputs.cache_retention_lazy) {
    const auto& budgets = inputs.capacity_budgets;
    if (!budgets.gpu_total || !budgets.host_total) {
      result.cache_at_clamp.reset();
      result.recommended = false;
      result.failures.unconfirmed = true;
      append(
          "Cache growth ceiling is unavailable: GPU or host total memory "
          "is not reported.");
    } else {
      // Same deterministic engine planner; executor + separate vision claims
      // form the footprint. Shared phase scratch stays in startup admission.
      const auto plan = gufo::server::PlanSnapshotCapacity(
          {.host_total_bytes = *budgets.host_total,
           .host_resident_bytes = totals.host_need - totals.host_allowance,
           .model_footprint_bytes = totals.weights + totals.session_total,
           .reserve_bytes = table.reserve_bytes,
           .device_total_bytes = budgets.gpu_total,
           .explicit_bytes = inputs.ram_cache_bytes});
      result.cache_at_clamp = plan.capacity_bytes;
    }
  }
  if (result.cache_at_clamp) {
    const auto growth = *result.cache_at_clamp;
    const auto need = totals.total_min + growth;
    const auto free = (static_cast<double>(available) - need) / 1073741824.0;
    result.free_after_growth_gib = free;
    const auto margin = RecommendedMargin(available);
    const bool margin_ok = need <= available && available - need >= margin;
    result.recommended = result.recommended && margin_ok;
    if (!margin_ok)
      result.failures.gpu_headroom = true;
    append("Cache can grow to " + GiBText(BytesToGiB(growth)) + "; " +
           GiBText(free) +
           (margin_ok ? " left >= margin " : " left < margin ") +
           GiBText(BytesToGiB(margin)) + " GPU headroom.");
    const auto dedicated = inputs.capacity_budgets.gpu_dedicated;
    if (dedicated) {
      result.host_spill = need > *dedicated ? need - *dedicated : 0;
      if (const auto host = HostAvailable(inputs)) {
        const auto host_need = totals.host_need + *result.host_spill;
        const auto host_free =
            (static_cast<double>(*host) - host_need) / 1073741824.0;
        result.host_free_after_spill_gib = host_free;
        const bool host_ok =
            host_need <= *host && *host - host_need >= inputs.host_floor_bytes;
        result.recommended = result.recommended && host_ok;
        if (!host_ok)
          result.failures.host_floor = true;
        append("Host spill " + GiBText(BytesToGiB(*result.host_spill)) + "; " +
               GiBText(host_free) +
               (host_ok ? " left >= floor " : " left < floor ") +
               GiBText(BytesToGiB(inputs.host_floor_bytes)) + ".");
      }
    } else {
      append("Host spill is not checked: " +
             inputs.capacity_budgets.gpu_dedicated_reason +
             " Recommendation uses the cache-growth margin only.");
    }
  }
  return result;
}
inline void WriteMemoryRecommendation(Value& value,
                                      const MemoryRecommendation& result) {
  WriteMemoryRecommendationFailures(value, result.failures);
  value["cache_at_clamp_bytes"] =
      result.cache_at_clamp ? Value(*result.cache_at_clamp) : Value();
  value["host_spill_bytes"] =
      result.host_spill ? Value(*result.host_spill) : Value();
  value["free_after_growth_gib"] = result.free_after_growth_gib
                                       ? Value(*result.free_after_growth_gib)
                                       : Value();
  value["host_free_after_spill_gib"] =
      result.host_free_after_spill_gib
          ? Value(*result.host_free_after_spill_gib)
          : Value();
}
// Largest context per session (a multiple of the step, capped at the model's
// context_max, never below context_min) for each session count 1..max, such
// that the MINIMUM total estimate leaves at least the margin free.
inline Value RecommendedContextTable(const MemoryModelTable& table,
                                     const MemoryEstimateInputs& inputs,
                                     std::uint64_t available) {
  const auto margin = RecommendedMargin(available);
  const auto step = kRecommendedContextStep;
  const auto floor_context = std::min(table.context_min, table.context_max);
  const auto fits = [&](std::uint64_t context, std::uint64_t sessions) {
    const auto total = ComputeMemoryTotals(table, inputs, context, sessions);
    return ComputeMemoryRecommendation(table, total, inputs, available)
        .recommended;
  };
  auto out = Value::object();
  out["available_bytes"] = available;
  out["available_gib"] = BytesToGiB(available);
  out["available_evidence"] = "measured";
  out["margin_bytes"] = margin;
  out["margin_gib"] = BytesToGiB(margin);
  out["margin_rule"] =
      "max(5 GiB, 5% of GPU available) after cache growth at the engine "
      "clamp; host free after spill must retain its floor. Snapshot reserve "
      "is not admission memory.";
  out["range_basis"] = "gpu.need";
  out["step_tokens"] = step;
  out["minimum_context_tokens"] = floor_context;
  out["context_max_tokens"] = table.context_max;
  out["max_sessions"] = table.max_sessions;
  auto rows = Value::array();
  for (std::uint64_t sessions = 1; sessions <= table.max_sessions; ++sessions) {
    std::uint64_t pick = 0;
    for (std::uint64_t c = table.context_max / step * step; c >= step;
         c -= step)
      if (fits(c, sessions)) {
        pick = c;
        break;
      }
    if (pick == 0 && fits(floor_context, sessions))
      pick = floor_context;
    const bool ok = pick != 0;
    if (!ok)
      pick = floor_context;
    const auto total = ComputeMemoryTotals(table, inputs, pick, sessions);
    auto row = Value::object();
    row["sessions"] = sessions;
    row["context_per_session"] = pick;
    row["fits"] = MemoryFitReason(total, inputs, available).empty();
    row["recommended"] = ok;
    const auto recommendation =
        ComputeMemoryRecommendation(table, total, inputs, available);
    row["reason"] = recommendation.reason;
    WriteMemoryRecommendation(row, recommendation);
    row["total_min_bytes"] = total.total_min;
    row["free_after_gib"] = (static_cast<double>(available) -
                             static_cast<double>(total.total_min)) /
                            1073741824.0;
    rows.push_back(std::move(row));
  }
  out["by_sessions"] = std::move(rows);
  return out;
}
// The estimator for an explicit table. MemoryEstimate looks the table up by
// package id; the header-based estimate (models_support) builds its own.
inline Value MemoryEstimateForTable(const MemoryModelTable& C,
                                    const MemoryEstimateInputs& inputs,
                                    std::string* reason) {
  const auto unavailable = [&](const char* text) {
    if (reason)
      *reason = text;
    return Value();
  };
  if (inputs.profile != "default" && inputs.profile != "subagents" &&
      inputs.profile != "exact" && inputs.profile != "vanilla" &&
      inputs.profile != "stock")
    return unavailable("No coefficient table for this profile.");
  if (inputs.context == 0 || inputs.context > C.estimate_context_limit ||
      inputs.sessions == 0 || inputs.sessions > C.max_sessions ||
      inputs.ram_cache_bytes > 1024ULL * 1024 * 1024 * 1024 ||
      inputs.snapshot_pool_bytes > 1024ULL * 1024 * 1024 * 1024)
    return unavailable(
        "Context, session count or cache size is unavailable or outside the "
        "estimate range.");
  if (reason)
    reason->clear();
  auto value = Value::object();
  value["table_version"] = static_cast<int>(C.version);
  value["evidence"] = "estimated";
  value["byte_unit"] = "bytes";
  value["display_unit"] = "GiB";
  value["fit_verified"] = false;
  value["reason"] =
      inputs.cache_retention_lazy
          ? "Conservative startup estimate. The configured cache retention "
            "ceiling is not allocated at startup; on-demand cache and snapshot "
            "growth are not included. Full-context retention is not verified."
      : inputs.cache_automatic
          ? "Conservative startup estimate. Automatic cache and snapshot pool "
            "growth are "
            "load-dependent and not included; full-context cache retention is "
            "not verified."
          : "Planning estimate. Allocation and placement are not verified.";
  value["cache_mode"] = inputs.cache_automatic ? "automatic" : "explicit";
  value["fit_scope"] =
      inputs.cache_automatic || inputs.cache_retention_lazy
          ? (C.vision_hidden_size ? "conservative_startup_estimate"
                                  : "startup_lower_bound")
          : "planning";
  const auto totals =
      ComputeMemoryTotals(C, inputs, inputs.context, inputs.sessions);
  const auto weights = totals.weights;
  const auto per_session = totals.per_session;
  const auto session_total = totals.session_total;
  const auto host_allowance = totals.host_allowance;
  const auto total_min = totals.total_min;
  const auto total_max = totals.total_max;
  value["weights"] = MemoryAmount(weights, "estimated");
  value["per_session_state"] = MemoryAmount(per_session, "estimated");
  value["per_session_vision"] =
      MemoryAmount(totals.vision_per_session, "estimated");
  value["vision_session_total"] =
      MemoryAmount(totals.vision_total, "estimated");
  value["session_state_total"] = MemoryAmount(session_total, "estimated");
  value["sessions"] = inputs.sessions;
  value["context_tokens"] = inputs.context;
  value["scratch_phase_allowance"] = Value::object();
  value["scratch_phase_allowance"]["min"] =
      MemoryAmount(C.scratch_min_bytes, "estimated");
  value["scratch_phase_allowance"]["max"] =
      MemoryAmount(C.scratch_max_bytes, "estimated");
  value["host_cache_snapshot_allowance"] =
      MemoryAmount(host_allowance, "estimated");
  value["reserve"] = MemoryAmount(C.reserve_bytes, "estimated");
  value["total_range"] = Value::object();
  value["total_range"]["min"] = MemoryAmount(total_min, "estimated");
  value["total_range"]["max"] = MemoryAmount(total_max, "estimated");
  value["carve_budget"] =
      inputs.carve_budget_bytes
          ? MemoryAmount(*inputs.carve_budget_bytes, "measured")
          : Value();
  value["host_budget"] =
      inputs.host_budget_bytes
          ? MemoryAmount(*inputs.host_budget_bytes, "measured")
          : Value();
  value["reference_coefficient"] =
      MemoryAmount(MiBToBytes(C.session_mib), "measured");
  value["reference_coefficient"]["context_tokens"] = C.reference_context;
  value["reference_coefficient"]["conditions"] =
      "Executor only, one 262144-token session; vision is a separate term.";
  value["reference_coefficient"]["source_path"] =
      std::string(C.reference_source_path);
  value["device_cache_snapshot_allowance"] =
      MemoryAmount(totals.device_cache, "estimated");
  const auto gpu_available = GpuAvailable(inputs);
  const auto host_available = HostAvailable(inputs);
  value["gpu"] = Value::object();
  value["gpu"]["available"] =
      gpu_available ? MemoryAmount(*gpu_available, "measured") : Value();
  const auto& budgets = inputs.capacity_budgets;
  value["gpu"]["total"] = budgets.gpu_total
                              ? MemoryAmount(*budgets.gpu_total, "measured")
                              : Value();
  value["gpu"]["dedicated_video_memory"] =
      budgets.gpu_dedicated ? MemoryAmount(*budgets.gpu_dedicated, "measured")
                            : Value();
  value["gpu"]["dedicated_video_memory_reason"] =
      budgets.gpu_dedicated ? Value() : Value(budgets.gpu_dedicated_reason);
  value["gpu"]["dedicated_video_memory_source"] = kDxgiDedicatedSource;
  value["gpu"]["need"] = MemoryAmount(total_min, "estimated");
  value["gpu"]["need_max"] = MemoryAmount(total_max, "estimated");
  value["gpu"]["reserve"] = MemoryAmount(C.reserve_bytes, "estimated");
  value["host"] = Value::object();
  value["host"]["available"] =
      host_available ? MemoryAmount(*host_available, "measured") : Value();
  value["host"]["total"] = budgets.host_total
                               ? MemoryAmount(*budgets.host_total, "measured")
                               : Value();
  value["host"]["need"] = MemoryAmount(totals.host_need, "estimated");
  value["host"]["floor"] = MemoryAmount(inputs.host_floor_bytes, "estimated");
  const auto fit_reason =
      gpu_available
          ? MemoryFitReason(totals, inputs, *gpu_available)
          : std::string("GPU available dedicated memory is not reported.");
  value["fits"] =
      gpu_available && host_available ? Value(fit_reason.empty()) : Value();
  value["fit_reason"] = fit_reason.empty() ? Value() : Value(fit_reason);
  if (gpu_available && host_available) {
    const auto recommendation =
        ComputeMemoryRecommendation(C, totals, inputs, *gpu_available);
    value["recommended"] = recommendation.recommended;
    value["recommendation_reason"] = recommendation.reason;
    WriteMemoryRecommendation(value, recommendation);
  } else {
    value["recommended"] = Value();
    value["recommendation_reason"] = fit_reason;
    WriteMemoryRecommendationFailures(value, {.unconfirmed = true});
  }
  value["formula"] =
      "GPU: weights + session_state_total + scratch_phase_allowance + "
      "device_cache_snapshot_allowance; session_state_total includes executor "
      "and separate vision state; reserve sizes snapshot pool only; host: "
      "process + NPU staging "
      "+ "
      "host_cache_snapshot_allowance + floor";
  // Additive in schema_version 1. Null with a reason when the machine's
  // available memory is not known.
  if (gpu_available && host_available) {
    value["recommended_context"] =
        RecommendedContextTable(C, inputs, *gpu_available);
    value["recommended_context_reason"] = Value();
  } else {
    value["recommended_context"] = Value();
    value["recommended_context_reason"] =
        gpu_available
            ? "Host available RAM is not reported on this machine."
            : "GPU available dedicated memory is not reported on this machine.";
  }
  return value;
}
inline Value MemoryEstimate(const MemoryEstimateInputs& inputs,
                            std::string* reason) {
  const auto* table_ptr = MemoryTableFor(inputs.package_id);
  if (!table_ptr) {
    if (reason)
      *reason = "No coefficient table for this model package.";
    return Value();
  }
  return MemoryEstimateForTable(*table_ptr, inputs, reason);
}
inline Value PiSnippet(std::string_view endpoint, std::string_view model,
                       std::uint64_t context, bool authenticated) {
  if (endpoint.empty() || model.empty() || context == 0)
    return Value();
  if (!endpoint.starts_with("http://127.0.0.1:") &&
      !endpoint.starts_with("http://[::1]:"))
    return Value();
  auto provider = Value::object();
  provider["baseUrl"] = std::string(endpoint);
  provider["api"] = "openai-completions";
  provider["apiKey"] =
      authenticated ? "BUBO_API_KEY" : std::string(kLoopbackDummyKey);
  auto entry = Value::object();
  entry["id"] = std::string(model);
  entry["name"] = std::string(model);
  entry["contextWindow"] = context;
  provider["models"] = Value::array();
  provider["models"].push_back(std::move(entry));
  auto snippet = Value::object();
  snippet["providers"] = Value::object();
  snippet["providers"]["bubo"] = std::move(provider);
  return snippet;
}
inline Value ActiveModelIdentity(const std::filesystem::path& model,
                                 const std::filesystem::path& manifest,
                                 std::string_view catalogue_id) {
  const auto path_value = [](const std::filesystem::path& path) {
    if (path.empty())
      return Value();
    const auto text =
        std::filesystem::absolute(path).lexically_normal().u8string();
    return Value(std::string(text.begin(), text.end()));
  };
  auto identity = Value::object();
  identity["catalogue_id"] =
      catalogue_id.empty() ? Value() : Value(std::string(catalogue_id));
  identity["model_path"] = path_value(model);
  identity["manifest_path"] = path_value(manifest);
  return identity;
}
// The prepared-model row of the package that was selected or loaded: its
// catalogue id and name (a Flash Next set or a plain catalogue model). An id
// the catalogue does not know reports the Huihui package, as before.
inline Value PreparedModels(const std::string& manifest, bool ready,
                            std::string_view package_id = prepare::kPackageId,
                            const std::string& model_path = {}) {
  const auto& catalogue = ShippedCatalogue().entries;
  const auto* known = FindCatalogueEntry(catalogue, package_id);
  if (!known)
    known = FindCatalogueEntry(catalogue, prepare::kPackageId);
  if (known) {
    auto rows = Value::array();
    rows.push_back(PreparedModelRow(*known, manifest, model_path, ready));
    return rows;
  }
  auto entries = Value::array();
  auto model = Value::object();
  model["id"] = std::string(prepare::kPackageId);
  model["display_name"] = std::string(prepare::kPackageDisplayName);
  model["manifest_path"] = manifest.empty() ? Value() : Value(manifest);
  std::uint64_t bytes = 0;
  for (const auto& entry : prepare::ModelCatalogue())
    bytes += entry.size;
  model["size_bytes"] = bytes;
  model["prepared_state"] = ready ? "prepared" : "not_prepared";
  model["capabilities"] = Value::object();
  model["capabilities"]["context_max"] = 262144;
  for (const auto* name : {"mtp", "twin", "mmproj"})
    model["capabilities"][name] = true;
  entries.push_back(std::move(model));
  return entries;
}
}  // namespace bubo::cli
