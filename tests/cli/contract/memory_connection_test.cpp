#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "src/cli/serve/request_history.hpp"
#include "src/cli/settings_support.hpp"

namespace {
class ScopedEnvironment {
public:
  explicit ScopedEnvironment(const char* name = "GUFO_CACHE_HOST_FLOOR_MB")
      : name_(name) {
    if (const char* value = std::getenv(name_))
      previous = value;
  }
  ~ScopedEnvironment() { Set(previous ? previous->c_str() : nullptr); }
  bool Set(const char* value) const {
#ifdef _WIN32
    return _putenv_s(name_, value ? value : "") == 0;
#else
    return (value ? setenv(name_, value, 1) : unsetenv(name_)) == 0;
#endif
  }
  std::optional<std::string> previous;

private:
  const char* name_;
};
}  // namespace

int main() {
  using namespace bubo::cli;
  unsigned passed = 0;
  const auto check = [&](bool good) {
    if (!good) {
      std::cerr << "Memory/connection assertion " << passed + 1 << " failed\n";
      throw std::runtime_error("Memory/connection assertion failed");
    }
    ++passed;
  };
  const auto check_failures = [&](const Value& result,
                                  std::string_view expected) {
    const auto* failures = result.find("recommendation_failures");
    check(failures && failures->is_array() && failures->dump() == expected);
  };
  check(MiBToBytes(7176) == 7524581376ULL);
  check(MemoryAmount(1024, "estimated").member_size("bytes") == 1024);
  check(BytesToGiB(1073741824ULL) == 1.0);
  check(BytesToGB(1000000000ULL) == 1.0);
  check(BytesToGiB(1000000000ULL) != BytesToGB(1000000000ULL));
  MemoryEstimateInputs inputs;
  std::string reason;
  check(MemoryEstimate(inputs, &reason).is_null() && !reason.empty());
  inputs.package_id = std::string(bubo::prepare::kPackageId);
  inputs.profile = "bad";
  check(MemoryEstimate(inputs, &reason).is_null() && !reason.empty());
  inputs.profile = "default";
  inputs.context = 262144;
  inputs.sessions = 1;
  inputs.host_available_bytes = 128ULL * 1073741824ULL;
  inputs.mtp = inputs.twin = inputs.draft_head = inputs.vision = true;
  auto estimate = MemoryEstimate(inputs, &reason);
  check(reason.empty() && estimate.member_str("evidence") == "estimated");
  check(estimate.find("per_session_state")->member_size("bytes") ==
        MiBToBytes(7176));
  check(estimate.find("reference_coefficient")->member_str("evidence") ==
        "measured");
  check(estimate.find("carve_budget")->is_null() &&
        estimate.find("host_budget")->is_null());
  check(!estimate.find("fit_verified")->as_bool());
  check(estimate.find("total_range")->find("min")->member_size("bytes") >
        64ULL * 1073741824ULL);
  // Without available memory the recommendation is null with a reason.
  check(estimate.find("recommended_context")->is_null() &&
        !estimate.find("recommended_context_reason")->is_null());
  check_failures(estimate, "[\"unconfirmed\"]");

  // Recommended context per session count.
  constexpr std::uint64_t kGiB = 1073741824ULL;
  const auto* table = MemoryTableFor(bubo::prepare::kPackageId);
  check(table != nullptr && MemoryTableFor("no-such-package") == nullptr);
  const auto recommend = [&](std::uint64_t available_gib, std::uint64_t cache) {
    auto in = inputs;
    in.ram_cache_bytes = cache;
    in.available_bytes = available_gib * kGiB;
    return MemoryEstimate(in, &reason);
  };
  const auto rows_of = [](const gufo::json::Value& est) {
    return est.find("recommended_context")->find("by_sessions")->items();
  };
  const auto verify_table = [&](std::uint64_t available_gib,
                                std::uint64_t cache) {
    auto in = inputs;
    in.ram_cache_bytes = cache;
    const auto est = recommend(available_gib, cache);
    const auto* rec = est.find("recommended_context");
    check(rec && !rec->is_null());
    const std::uint64_t available = available_gib * kGiB;
    const std::uint64_t margin = rec->member_size("margin_bytes");
    check(margin == std::max<std::uint64_t>(5 * kGiB, available / 20));
    check(rec->member_size("available_bytes") == available);
    const auto rows = rows_of(est);
    check(rows.size() == table->max_sessions);
    std::uint64_t previous = table->context_max;
    for (std::size_t n = 0; n != rows.size(); ++n) {
      const auto& row = rows[n];
      const std::uint64_t sessions = row.member_size("sessions");
      const std::uint64_t context = row.member_size("context_per_session");
      check(sessions == n + 1);
      check(context <= previous);  // more sessions never give a larger context
      previous = context;
      check(context <= table->context_max && context >= table->context_min);
      const auto fits_with_margin = [&](std::uint64_t c) {
        return ComputeMemoryTotals(*table, in, c, sessions).total_min +
                   margin <=
               available;
      };
      if (row.find("recommended")->as_bool()) {
        check_failures(row, "[]");
        check(row.find("fits")->as_bool());
        check(fits_with_margin(context));
        check(context == table->context_min ||
              context % kRecommendedContextStep == 0);
        const auto next = context < kRecommendedContextStep
                              ? kRecommendedContextStep
                              : context + kRecommendedContextStep;
        check(context == table->context_max || !fits_with_margin(next));
        check(row.member_double("free_after_gib") >=
              static_cast<double>(margin) / kGiB - 1e-9);
      } else {
        check_failures(row, "[\"gpu_headroom\"]");
        check(context == table->context_min);
        check(!fits_with_margin(table->context_min));
        check(row.member_double("free_after_gib") <
              static_cast<double>(margin) / kGiB);
      }
    }
    return est;
  };
  // Shipped profiles on a 128 GiB machine: 262144 x 1 and 131072 x 4 fit.
  const auto est128 = verify_table(128, 0);
  const auto rows128 = rows_of(est128);
  check(rows128[0].find("fits")->as_bool() &&
        rows128[0].member_size("context_per_session") >= 262144);
  check(rows128[3].find("fits")->as_bool() &&
        rows128[3].member_size("context_per_session") >= 131072);
  // 96 and 110 GiB: the table is valid, and the 96 GiB table is never larger.
  const auto est96 = verify_table(96, 0);
  const auto est110 = verify_table(110, 0);
  const auto rows96 = rows_of(est96);
  const auto rows110 = rows_of(est110);
  for (std::size_t n = 0; n != rows128.size(); ++n) {
    check(rows96[n].member_size("context_per_session") <=
          rows110[n].member_size("context_per_session"));
    check(rows110[n].member_size("context_per_session") <=
          rows128[n].member_size("context_per_session"));
  }
  // A device RAM cache lowers the numbers and keeps every invariant.
  const auto est128_cache = verify_table(128, 8 * kGiB);
  check(rows_of(est128_cache)[3].member_size("context_per_session") <=
        rows128[3].member_size("context_per_session"));
  // Small memory: nothing fits, the row says so and lists the minimum.
  const auto est32 = verify_table(32, 0);
  for (const auto& row : rows_of(est32))
    check(!row.find("fits")->as_bool());
  // 112 GiB: one session at the full context fits with the margin.
  const auto est112 = verify_table(112, 0);
  check(rows_of(est112)[0].find("fits")->as_bool());
  // Unsupported model: no estimate and no recommendation.
  auto other = inputs;
  other.package_id = "some-other-model";
  other.available_bytes = 128 * kGiB;
  check(MemoryEstimate(other, &reason).is_null() && !reason.empty());
  // Unknown available memory: the recommendation is null with a reason.
  auto unknown = inputs;
  unknown.available_bytes.reset();
  const auto est_unknown = MemoryEstimate(unknown, &reason);
  check(est_unknown.find("recommended_context")->is_null() &&
        !est_unknown.find("recommended_context_reason")->is_null());
  // The estimate itself did not change.
  check(est128.find("total_range")->find("min")->member_size("bytes") ==
        estimate.find("total_range")->find("min")->member_size("bytes"));
  estimate = est128;

  // Lazy retention is admitted empty, but recommendations must survive growth.
  // Fixed byte observations are independent of the engine capacity formula.
  {
    const auto count_start = passed;
    auto lazy = inputs;



    lazy.image_token_cap = 262144;
    lazy.cache_retention_lazy = true;
    lazy.ram_cache_bytes = 6 * kGiB;
    lazy.available_bytes = 119078647808ULL;
    lazy.capacity_budgets.gpu_total = 119078647808ULL;
    lazy.capacity_budgets.host_total = 33982058496ULL;
    lazy.capacity_budgets.gpu_dedicated = 128 * kGiB;
    lazy.host_available_bytes = 13919617024ULL;
    lazy.sessions = 2;
    const auto check_setup_failures = [&](std::string_view expected) {
      check_failures(MemoryEstimate(lazy, &reason), expected);
    };
    auto result = MemoryEstimate(lazy, &reason);
    check(result.find("fits")->as_bool());
    check(!result.find("recommended")->as_bool());
    check(result.member_size("cache_at_clamp_bytes") == 5602469888ULL);
    check(result.member_str("recommendation_reason").find("left < margin") !=
          std::string::npos);
    check_setup_failures("[\"gpu_headroom\"]");
    lazy.capacity_budgets.gpu_dedicated = 102892924928ULL;
    check_setup_failures("[\"gpu_headroom\",\"host_floor\"]");
    lazy.capacity_budgets.gpu_dedicated = 128 * kGiB;
    const auto original_need =
        result.find("gpu")->find("need")->member_size("bytes");
    lazy.sessions = 1;
    result = MemoryEstimate(lazy, &reason);
    check(result.find("recommended")->as_bool());
    check_setup_failures("[]");
    check(result.member_size("cache_at_clamp_bytes") == 6 * kGiB);
    lazy.capacity_budgets.gpu_dedicated = 102892924928ULL;
    result = MemoryEstimate(lazy, &reason);
    const auto spill = result.member_size("host_spill_bytes");
    check(spill == 5011607744ULL);
    const auto host_need =
        result.find("host")->find("need")->member_size("bytes");
    lazy.host_available_bytes = host_need + spill + lazy.host_floor_bytes;
    check(MemoryEstimate(lazy, &reason).find("recommended")->as_bool());
    check_setup_failures("[]");
    --*lazy.host_available_bytes;
    result = MemoryEstimate(lazy, &reason);
    check(!result.find("recommended")->as_bool());
    check(result.find("fits")->as_bool());
    check(result.member_str("recommendation_reason").find("Host spill") !=
          std::string::npos);
    check_setup_failures("[\"host_floor\"]");
    lazy.capacity_budgets.gpu_dedicated.reset();
    result = MemoryEstimate(lazy, &reason);
    check(result.find("recommended")->as_bool());
    check(result.find("host_spill_bytes")->is_null());
    check(result.find("gpu")->find("dedicated_video_memory")->is_null());
    check(result.member_str("recommendation_reason").find("not checked") !=
          std::string::npos);
    check_setup_failures("[]");
    lazy.capacity_budgets.gpu_total.reset();
    result = MemoryEstimate(lazy, &reason);
    check(!result.find("recommended")->as_bool());
    check(result.find("cache_at_clamp_bytes")->is_null());
    check(result.member_str("recommendation_reason")
              .find("ceiling is unavailable") != std::string::npos);
    check_setup_failures("[\"unconfirmed\"]");
    for (const auto& row : rows_of(result))
      check_failures(row, "[\"unconfirmed\"]");
    lazy.sessions = 2;
    result = MemoryEstimate(lazy, &reason);
    check(result.find("gpu")->find("need")->member_size("bytes") ==
          original_need);
    std::cout << "LazyGrowthHostSpillAndUnknownReadings: "
              << passed - count_start << " passed, 0 failed\n";
  }

  {
    const auto count_start = passed;
    auto row_inputs = inputs;
    row_inputs.available_bytes = 32 * kGiB;
    row_inputs.capacity_budgets.gpu_dedicated = 128 * kGiB;
    const auto gpu_only = MemoryEstimate(row_inputs, &reason);
    for (const auto& row : rows_of(gpu_only))
      check_failures(row, "[\"gpu_headroom\"]");
    row_inputs.host_available_bytes = kGiB;
    const auto both = MemoryEstimate(row_inputs, &reason);
    for (const auto& row : rows_of(both))
      check_failures(row, "[\"gpu_headroom\",\"host_floor\"]");
    row_inputs.available_bytes = 128 * kGiB;
    const auto host_only = MemoryEstimate(row_inputs, &reason);
    for (const auto& row : rows_of(host_only))
      check_failures(row, "[\"host_floor\"]");
    std::cout << "ContextRowFailureTerms: " << passed - count_start
              << " passed, 0 failed\n";
  }

  // OwnerCarve262144: the allocatable local budget, not the host reading,
  // funds weights, state and scratch. No unconditional full snapshot mirror.
  auto owner = inputs;
  owner.available_bytes = MiBToBytes(110192);
  owner.host_available_bytes = 11900000000ULL;
  owner.npu_prefill = true;
  const auto owner_est = MemoryEstimate(owner, &reason);
  check(owner_est.find("fits")->as_bool());
  check(rows_of(owner_est)[0].find("fits")->as_bool());
  check(rows_of(owner_est)[0].member_size("context_per_session") == 262144);
  check(owner_est.find("gpu")->find("need")->member_size("bytes") <=
        *owner.available_bytes);
  std::cout << "OwnerCarve262144: 4 passed, 0 failed\n";
  auto host_short = owner;
  host_short.host_available_bytes = kGiB;
  const auto host_est = MemoryEstimate(host_short, &reason);
  check(!host_est.find("fits")->as_bool());
  check(host_est.member_str("fit_reason").find("Host RAM") !=
        std::string::npos);
  check(!rows_of(host_est)[0].find("fits")->as_bool());
  check_failures(host_est, "[\"host_floor\"]");
  check_failures(rows_of(host_est)[0], "[\"host_floor\"]");
  std::cout << "HostBudgetBlocksFit: 5 passed, 0 failed\n";
  auto gpu_short = owner;
  gpu_short.available_bytes = 32 * kGiB;
  const auto gpu_est = MemoryEstimate(gpu_short, &reason);
  check(!gpu_est.find("fits")->as_bool());
  check(gpu_est.member_str("fit_reason").find("GPU") != std::string::npos);
  auto zero = owner;
  zero.available_bytes = 0;
  check(!MemoryEstimate(zero, &reason).find("fits")->as_bool());
  auto missing_host = owner;
  missing_host.host_available_bytes.reset();
  const auto missing_host_est = MemoryEstimate(missing_host, &reason);
  check(missing_host_est.find("fits")->is_null());
  check(missing_host_est.find("recommended_context")->is_null());
  check_failures(missing_host_est, "[\"unconfirmed\"]");
  check(
      missing_host_est.member_str("recommended_context_reason").find("Host") !=
      std::string::npos);
  std::cout << "GpuShortZeroAndUnknownHost: 7 passed, 0 failed\n";
  auto cached = owner;
  cached.ram_cache_bytes = 8 * kGiB;
  cached.snapshot_pool_bytes = 8 * kGiB;
  const auto cache_est = MemoryEstimate(cached, &reason);
  check(
      cache_est.find("device_cache_snapshot_allowance")->member_size("bytes") ==
      16 * kGiB);
  check(cache_est.find("host")->find("need")->member_size("bytes") ==
        owner_est.find("host")->find("need")->member_size("bytes"));
  check(!cache_est.find("fits")->as_bool());
  auto host_table = *table;
  host_table.cache_on_host = true;
  const auto host_cache_est =
      MemoryEstimateForTable(host_table, cached, &reason);
  check(host_cache_est.find("host_cache_snapshot_allowance")
            ->member_size("bytes") == 8 * kGiB);
  check(host_cache_est.find("device_cache_snapshot_allowance")
            ->member_size("bytes") == 8 * kGiB);
  std::cout << "CacheAndSnapshotPlacement: 5 passed, 0 failed\n";
  const auto lazy_start = passed;
  cached.cache_retention_lazy = true;
  const auto lazy_est = MemoryEstimate(cached, &reason);
  check(
      lazy_est.find("device_cache_snapshot_allowance")->member_size("bytes") ==
      8 * kGiB);
  check(lazy_est.member_str("cache_mode") == "explicit");
  check(lazy_est.member_str("fit_scope") == "conservative_startup_estimate");
  check(lazy_est.member_str("reason").find("ceiling is not allocated") !=
        std::string::npos);
  const auto lazy_host_est =
      MemoryEstimateForTable(host_table, cached, &reason);
  check(lazy_host_est.find("host_cache_snapshot_allowance")
            ->member_size("bytes") == 8 * kGiB);
  check(lazy_host_est.find("device_cache_snapshot_allowance")
            ->member_size("bytes") == 8 * kGiB);
  std::cout << "LazyCeilingPreservesExplicitPoolAndHostCache: "
            << passed - lazy_start << " passed, 0 failed\n";

  auto automatic = owner;
  automatic.cache_automatic = true;
  const auto automatic_est = MemoryEstimate(automatic, &reason);
  check(automatic_est.member_str("cache_mode") == "automatic");
  check(automatic_est.member_str("fit_scope") ==
        "conservative_startup_estimate");
  check(automatic_est.member_str("reason").find("not included") !=
        std::string::npos);
  auto high_floor = owner;
  high_floor.host_floor_bytes = 16 * kGiB;
  check(!MemoryEstimate(high_floor, &reason).find("fits")->as_bool());
  auto no_npu = owner;
  no_npu.npu_prefill = false;
  const auto no_npu_est = MemoryEstimate(no_npu, &reason);
  check(owner_est.find("host")->find("need")->member_size("bytes") -
            no_npu_est.find("host")->find("need")->member_size("bytes") ==
        7 * kGiB / 2);
  std::cout << "AutomaticCacheScopeFloorAndNpu: 5 passed, 0 failed\n";

  auto explicit_budgets = owner;
  explicit_budgets.available_bytes.reset();
  explicit_budgets.host_available_bytes.reset();
  explicit_budgets.carve_budget_bytes = owner.available_bytes;
  explicit_budgets.host_budget_bytes = owner.host_available_bytes;
  const auto explicit_est = MemoryEstimate(explicit_budgets, &reason);
  check(explicit_est.find("fits")->as_bool());
  check(rows_of(explicit_est)[0].member_size("context_per_session") == 262144);
  auto huge_pool = owner;
  huge_pool.snapshot_pool_bytes = ~std::uint64_t{0};
  check(MemoryEstimate(huge_pool, &reason).is_null() && !reason.empty());
  std::cout << "ExplicitBudgetsAndPoolRange: 3 passed, 0 failed\n";

  // Independent constants from Model::VisionSessionBytes and the owner's
  // ledger; executor rounding stays 7176 MiB and is not a vision allocation.
  auto case_start = passed;
  for (const auto sessions : {1ULL, 2ULL, 4ULL}) {
    for (const auto vision : {false, true}) {
      for (const auto cap : {0ULL, 8192ULL, 524288ULL}) {
        auto in = owner;
        in.sessions = sessions;
        in.vision = vision;
        in.image_token_cap =
            cap ? std::optional<std::uint64_t>(cap) : std::nullopt;
        const auto result = MemoryEstimate(in, &reason);
        // Unset (cap 0) takes the default of 65536 image tokens:
        // 12 * 262144 + 10240 * 65536 + 64.
        const std::uint64_t expected_vision = !vision       ? 0
                                              : cap == 8192 ? 87031872ULL
                                              : cap == 0    ? 674234432ULL
                                                            : 2687500352ULL;
        check(result.find("per_session_state")->member_size("bytes") ==
              7524581376ULL);
        check(result.find("per_session_vision")->member_size("bytes") ==
              expected_vision);
        check(result.find("vision_session_total")->member_size("bytes") ==
              expected_vision * sessions);
        check(result.find("session_state_total")->member_size("bytes") ==
              (7524581376ULL + expected_vision) * sessions);
        // Neither the snapshot sizing reserve nor per-session scratch is
        // charged to admission. Explicit cache allowances are tested above.
        check(result.find("gpu")->find("need")->member_size("bytes") ==
              (vision ? 87790000000ULL : 86880000000ULL) + 3460000000ULL +
                  (7524581376ULL + expected_vision) * sessions);
      }
    }
  }
  std::cout << "VisionOnOffCappedUncapped1x2x4: " << passed - case_start
            << " passed, 0 failed\n";

  case_start = passed;
  {
    ScopedEnvironment environment("GUFO_MAX_IMAGE_TOKENS");
    for (const auto* raw : {"8192", "+8192", " 8192"}) {
      check(environment.Set(raw));
      check(MemoryEstimateInputs().image_token_cap == 8192);
    }
    for (const auto* raw : {"", static_cast<const char*>(nullptr)}) {
      check(environment.Set(raw));
      check(!MemoryEstimateInputs().image_token_cap);
    }
    for (const auto* raw : {"0", "invalid", "8192x", "4294967296"}) {
      check(environment.Set(raw));
      bool rejected = false;
      try {
        (void)MemoryEstimateInputs();
      } catch (const std::invalid_argument& error) {
        rejected = std::string_view(error.what()) ==
                   "GUFO_MAX_IMAGE_TOKENS must be a positive integer";
      }
      check(rejected);
    }
  }
  std::cout << "VisionCapEnvironment: " << passed - case_start
            << " passed, 0 failed\n";

  case_start = passed;
  for (const auto sessions : {1ULL, 2ULL}) {
    auto in = owner;
    in.sessions = sessions;
    in.cache_automatic = true;
    in.image_token_cap =
        262144;                                                  
    const auto result = MemoryEstimate(in, &reason);
    check(result.find("fits")->as_bool());
    check(result.find("recommended")->as_bool() == (sessions == 1));
    check(result.member_str("fit_scope") == "conservative_startup_estimate");
    check(result.find("gpu")->find("need")->member_size("bytes") ==
          (sessions == 1 ? 101462081728ULL : 111674163456ULL));
    const std::uint64_t ledger_need = 82728ULL * 1048576 + 2902ULL * 1048576 +
                                      417962394ULL + 10211815764ULL * sessions;
    check(ledger_need < *in.available_bytes);
    if (sessions == 2)
      check(result.member_str("recommendation_reason").find("headroom") !=
            std::string::npos);
    std::cout << "OwnerLedger sessions=" << sessions << " " << result.dump()
              << '\n';
    // Old GPU source (102.9 GB): one fits; two falsely appear too large.
    in.available_bytes = 102892924928ULL;
    const auto old_budget = MemoryEstimate(in, &reason);
    check(old_budget.find("fits")->as_bool() == (sessions == 1));
  }
  std::cout << "OwnerAdmissionAndRecommendation: " << passed - case_start
            << " passed, 0 failed\n";

  std::optional<std::string> previous_floor;
  {
    ScopedEnvironment environment;
    previous_floor = environment.previous;
    auto count_start = passed;
    for (const auto* raw : {"+16384", " 16384", "\t+16384", "8192"}) {
      check(environment.Set(raw));
      const auto expected =
          std::string_view(raw) == "8192" ? 8 * kGiB : 16 * kGiB;
      check(PlanningHostFloorBytes() == expected);
      auto configured = owner;
      configured.host_floor_bytes = PlanningHostFloorBytes();
      check(!MemoryEstimate(configured, &reason).find("fits")->as_bool());
    }
    std::cout << "PlanningHostFloorStrtoullGrammar: " << passed - count_start
              << " passed, 0 failed\n";
    count_start = passed;
    for (const auto* raw :
         {"invalid", "8192x", "8192 ", "-16384", " -16384", "0", "1", ""}) {
      check(environment.Set(raw));
      check(PlanningHostFloorBytes() ==
            gufo::server::kHostSnapshotHeadroomBytes);
    }
    check(environment.Set(nullptr));
    check(PlanningHostFloorBytes() == gufo::server::kHostSnapshotHeadroomBytes);
    std::cout << "PlanningHostFloorInvalidAndMinimum: " << passed - count_start
              << " passed, 0 failed\n";
    count_start = passed;
    for (const auto* raw :
         {"8589934592", "18446744073709551615", "18446744073709551616"}) {
      check(environment.Set(raw));
      check(PlanningHostFloorBytes() == 9007199254740991ULL);
      auto configured = owner;
      configured.host_floor_bytes = PlanningHostFloorBytes();
      check(!MemoryEstimate(configured, &reason).find("fits")->as_bool());
    }
    std::cout << "PlanningHostFloorOverflowClampsHigh: " << passed - count_start
              << " passed, 0 failed\n";
  }
  const char* restored_floor = std::getenv("GUFO_CACHE_HOST_FLOOR_MB");
  check((restored_floor ? std::optional<std::string>(restored_floor)
                        : std::nullopt) == previous_floor);
  std::cout << "PlanningHostFloorEnvironmentRestored: 1 passed, 0 failed\n";

  auto snippet =
      PiSnippet("http://127.0.0.1:18110/v1", "fixture-model", 262144, true);
  check(snippet.dump().find("BUBO_API_KEY") != std::string::npos);
  check(snippet.dump().find("synthetic-auth-secret") == std::string::npos);
  const std::string real_model = "huihui-qwen38-flash-next-ud-q4-k-xl-v1";
  check(PiSnippet("http://127.0.0.1:18110/v1", real_model, 200000, false)
            .dump()
            .find(real_model) != std::string::npos);
  for (const auto& row : ClientSnippets(real_model, false).items())
    check(row.member_str("template").find(real_model) != std::string::npos);
  check(PiSnippet("", "", 0, false).is_null());
  {
    const auto start = passed;
    for (const bool authenticated : {false, true}) {
      const auto templates =
          ClientSnippets("model'\"\n$(`test`)", authenticated);
      check(templates.size() == 3);
      for (const auto& row : templates.items()) {
        const auto text = row.member_str("template");
        check(!row.member_str("kind").empty() &&
              !row.member_str("title").empty());
        check(text.find("{api_key}") != std::string::npos);
        check(text.find(row.member_str("kind") == "anthropic_python"
                            ? "{anthropic_base_url}"
                            : "{openai_base_url}") != std::string::npos);
        check(row.find("requires_key")->as_bool() == authenticated);
        check(authenticated
                  ? row.find("default_api_key")->is_null()
                  : row.member_str("default_api_key") == "bubo-loopback");
        check(text.find("synthetic-auth-secret") == std::string::npos);
      }
    }
    // Each template must work as pasted on the Windows product:
    // - the server reads only "Authorization: Bearer"; the Anthropic SDK sends
    //   that header only for auth_token= (api_key= sends X-Api-Key: a 401);
    // - Windows PowerShell 5.1 sends a string body as ISO-8859-1 unless the
    //   content type names UTF-8;
    // - a POSIX-quoted curl line fails in both cmd and PowerShell.
    for (const auto& row : ClientSnippets("fake", true).items()) {
      const auto kind = row.member_str("kind");
      const auto text = row.member_str("template");
      check(kind.find("curl") == std::string::npos &&
            text.find("curl ") == std::string::npos);
      if (kind == "anthropic_python")
        check(text.find("auth_token=\"{api_key}\"") != std::string::npos &&
              text.find("api_key=") == std::string::npos);
      if (kind == "powershell_openai_chat")
        check(text.find("application/json; charset=utf-8") !=
              std::string::npos);
    }
    check(ClientSnippets("", false).is_null());
    // The new templates retain placeholders even where legacy Pi is null.
    check(PiSnippet("http://0.0.0.0:32145/v1", "fake", 4096, true).is_null());
    check(ClientSnippets("fake", true).size() == 3);
    std::cout << "StructuredClientTemplates: " << passed - start
              << " passed, 0 failed\n";
  }
  {
    const auto start = passed;
    for (const auto set :
         {bubo::prepare::ModelSet::Huihui, bubo::prepare::ModelSet::Original}) {
      const auto id = bubo::prepare::PackageId(set);
      const auto selected = ActiveModelIdentity(
          "models/é/one.gguf", "models/é/bubo-package.json", id);
      check(selected.member_str("catalogue_id") == id);
      check(std::filesystem::u8path(selected.member_str("model_path"))
                .is_absolute());
      check(std::filesystem::u8path(selected.member_str("manifest_path"))
                .is_absolute());
    }
    const auto imported = ActiveModelIdentity("import.gguf", {}, {});
    check(imported.find("catalogue_id")->is_null());
    check(imported.find("manifest_path")->is_null());
    check(!imported.member_str("model_path").empty());
    std::cout << "ActiveModelIdentity: " << passed - start
              << " passed, 0 failed\n";
  }
  gufo::server::RequestHistory history;
  gufo::server::TextGenerationBackend::Result result;
  result.prompt_tokens = 100;
  result.completion_tokens = 10;
  result.prefill_ms = 50;
  result.prefill_tokens = 100;
  result.decode_ms = 200;
  result.text = "synthetic-response-secret";
  result.client_id = "synthetic-client-secret";
  for (unsigned n = 0; n != 75; ++n)
    history.Record(gufo::server::CompletedRequestRow(
        "r" + std::to_string(n), gufo::server::RequestResultMetrics(result),
        "completed"));
  const auto ring = history.Sample();
  check(ring.size() == 50 && ring.items().front().member_str("id") == "r25");
  check(ring.dump().find("synthetic-") == std::string::npos);
  check(ring.items().back().member_double("decode_tps") == 50.0);
  check(ring.items().back().find("slot")->is_null());
  // R2: the GPU budget overlaps the RAM reading. What does not fit the dedicated carve-out lands in RAM, and RAM keeps its floor for it.
  {
    constexpr std::uint64_t kG = 1073741824ULL;
    bubo::cli::MemoryTotals totals;
    totals.total_min = 100 * kG;   // 4 GiB beyond the 96 GiB carve-out
    totals.host_need = 1 * kG;
    bubo::cli::MemoryEstimateInputs in;
    in.host_floor_bytes = 4 * kG;
    in.capacity_budgets.gpu_dedicated = 96 * kG;
    in.host_available_bytes = 13 * kG;
    check(bubo::cli::MemoryFitReason(totals, in, 110 * kG).empty());   // 1 + 4 + 4 = 9 GiB <= 13 GiB
    in.host_available_bytes = 8 * kG;                                  // 9 GiB > 8 GiB
    const auto reason = bubo::cli::MemoryFitReason(totals, in, 110 * kG);
    check(reason.find("Memory is short: 4.000 GiB of the GPU need would use system RAM") != std::string::npos);
    bubo::cli::MemoryRecommendationFailures failures;
    (void)bubo::cli::MemoryFitReason(totals, in, 110 * kG, &failures);
    check(failures.gpu_headroom && failures.host_floor);
    in.capacity_budgets.gpu_dedicated.reset();                         // no carve-out reading: the old rule only
    check(bubo::cli::MemoryFitReason(totals, in, 110 * kG).empty());
    totals.total_min = 90 * kG;                                        // fits the carve-out: nothing spills
    in.capacity_budgets.gpu_dedicated = 96 * kG;
    check(bubo::cli::MemoryFitReason(totals, in, 110 * kG).empty());
  }
  std::cout << "Memory, units, connection and request ring: " << passed
            << " passed, 0 failed\n";
  std::cout << snippet.dump() << '\n';
  std::cout << estimate.dump() << '\n';
}
