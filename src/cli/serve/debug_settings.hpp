#ifndef GUFO_SERVER_DEBUG_SETTINGS_HPP_
#define GUFO_SERVER_DEBUG_SETTINGS_HPP_

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli/serve/behaviour_options.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::server {

// Keep this catalog explicit: the source-completeness test rejects newly read
// variables until their diagnostic meaning has been reviewed.
inline constexpr const char* kDebugSettingsEnvironment[] = {
    "GUFO_DEBUG_HOOKS",
    "GUFO_THINK_EOG_GUARD",
    "GUFO_EOG_BOUNDARY",
    "GUFO_EOG_MIN_PROB_EXACT",
    "GUFO_EOG_MIN_PROB",
    "GUFO_EOG_DEFER_P",
    "GUFO_SPEC_EOG_STOP",
    "GUFO_LOOP_BREAKER",
    "GUFO_LOOP_BREAKER_EOG_CLOSE",
    "GUFO_LOOP_BREAKER_TRANSITION",
    "GUFO_LOOP_BREAKER_TRANSITION_TEXT",
    "GUFO_REASONING_BUDGET",
    "GUFO_LOGIT_BIAS",
    "GUFO_MODEL_SUPPRESS_TOKENS",
    "GUFO_THINK_MARKER_BIAS",
    "GUFO_THINK_MARKER_BIAS_BY_EFFORT",
    "GUFO_THINK_MARKER_LIST",
    "GUFO_PRODUCTION_SAMPLING",
    "GUFO_LENIENT",
    "GUFO_LENIENT_LAMBDA",
    "GUFO_LENIENT_THINK_ONLY",
    "GUFO_MTP_TRACE",
    "GUFO_PLATFORM_TUNING",
    "GUFO_DISK_BOUNDARY_ALIGN",
    "GUFO_DISK_INPASS",
    "GUFO_DISK_DEFER",
    "GUFO_DISK_DEFER_MS",
    "GUFO_PREFILL_DIAG",
    "GUFO_MOE_BENCH",
    "GUFO_DENSE_TWIN_MODEL",
    "GUFO_DENSE_TWIN_MAX_N",
    "GUFO_DENSE_TWIN_SHADOW",
    "GUFO_DENSE_TWIN_ALLOW_Q8",
    "GUFO_DENSE_TWIN_TRACE",
    "GUFO_DRAFT_OUT_MODEL",
    "GUFO_PREFILL_FORWARD_TOKENS",
    "GUFO_DISPATCH_TELEMETRY",
    "GUFO_TRACE",
    "GPU_MAX_HW_QUEUES",
    "HOST",
    "GUFO_HOST",
    "PORT",
    "GUFO_PORT",
    "XDG_CACHE_HOME",
    "LOCALAPPDATA",
    "HOME",
};

struct DebugSettingsIrrelevantEnvironment {
  const char* name;
  const char* reason;
};

inline constexpr DebugSettingsIrrelevantEnvironment
    kDebugSettingsIrrelevantEnvironment[] = {
        {"NO_COLOR", "Controls ANSI terminal colors only."},
};

[[nodiscard]] inline bool DebugSettingsListsEnvironment(std::string_view name) {
  for (const char* candidate : kDebugSettingsEnvironment) {
    if (name == candidate)
      return true;
  }
  for (const auto& candidate : kDebugSettingsIrrelevantEnvironment) {
    if (name == candidate.name)
      return true;
  }
  return false;
}

// Resolved model, runner, device, CLI and cache values are supplied by their
// owners after construction; reparsing flags here would conceal auto budgets.
[[nodiscard]] inline json::Value BuildEffectiveSettings(
    json::Value resolved, std::string_view version, std::string_view revision,
    std::string_view build_id) {
  if (!resolved.is_object())
    throw std::invalid_argument("resolved debug settings must be an object");
  auto settings = json::Value::object();
  settings["version"] = std::string(version);
  settings["revision"] = std::string(revision);
  settings["exe_build_id"] = std::string(build_id);
  auto environment = json::Value::object();
  for (const char* name : kDebugSettingsEnvironment) {
    const char* value = std::getenv(name);
    environment[name] = value ? json::Value(value) : json::Value();
  }
  settings["environment"] = std::move(environment);
  auto irrelevant = json::Value::object();
  for (const auto& entry : kDebugSettingsIrrelevantEnvironment)
    irrelevant[entry.name] = entry.reason;
  settings["irrelevant_environment"] = std::move(irrelevant);

  const auto parsed = ParseBehaviourOptions(json::Value::object());
  auto behaviour = json::Value::object();
  behaviour["enabled"] = parsed != nullptr;
  behaviour["think_eog_guard"] = BehaviourSwitch("GUFO_THINK_EOG_GUARD");
  behaviour["eog_boundary"] = BehaviourSwitch("GUFO_EOG_BOUNDARY");
  behaviour["eog_min_prob_exact"] = BehaviourSwitch("GUFO_EOG_MIN_PROB_EXACT");
  behaviour["eog_min_prob"] = BehaviourNumber("GUFO_EOG_MIN_PROB", -1.0);
  behaviour["eog_defer_p"] = BehaviourNumber("GUFO_EOG_DEFER_P", 0.0);
  behaviour["spec_eog_stop"] = BehaviourSwitch("GUFO_SPEC_EOG_STOP");
  behaviour["loop_breaker"] = BehaviourSwitch("GUFO_LOOP_BREAKER");
  behaviour["loop_eog_close"] = BehaviourSwitch("GUFO_LOOP_BREAKER_EOG_CLOSE");
  behaviour["loop_transition"] = BehaviourNumber("GUFO_LOOP_BREAKER_TRANSITION", 2.0);
  behaviour["reasoning_budget_enabled"] = BehaviourSwitch("GUFO_REASONING_BUDGET");
  behaviour["logit_bias_enabled"] = BehaviourSwitch("GUFO_LOGIT_BIAS");
  behaviour["model_suppress_tokens"] = BehaviourSwitch("GUFO_MODEL_SUPPRESS_TOKENS");
  behaviour["production_sampling"] = BehaviourSwitch("GUFO_PRODUCTION_SAMPLING");
  behaviour["marker_bias"] = BehaviourNumber("GUFO_THINK_MARKER_BIAS", 0.0);
  auto efforts = json::Value::object();
  sampling::BehaviourConfig inactive;
  const auto& cfg = parsed ? *parsed : inactive;
  for (const char* effort : {"minimal", "low", "medium", "high", "xhigh", "max"})
    efforts[effort] = ResolveMarkerBias(cfg, effort);
  behaviour["marker_bias_by_effort"] = std::move(efforts);
  resolved["behaviour"] = std::move(behaviour);

  auto tuning = json::Value::object();
  const auto& platform_tuning = platform::PlatformTuning();
  for (const auto& field : platform::detail::kTuningFields)
    tuning[std::string(field.name)] = platform_tuning.*field.member;
  resolved["platform_tuning"] = std::move(tuning);
  const auto& lenient = models::qwen38_flash_next::MtpLenient();
  auto lenient_settings = json::Value::object();
  lenient_settings["enabled"] = lenient.enabled;
  lenient_settings["lambda"] = lenient.lambda;
  lenient_settings["think_only"] = lenient.think_only;
  resolved["lenient"] = std::move(lenient_settings);
  const auto* trace = settings["environment"].find("GUFO_MTP_TRACE");
  resolved["mtp_trace"] = trace && trace->is_string() && trace->str() == "1";
  const auto* align = settings["environment"].find("GUFO_DISK_BOUNDARY_ALIGN");
  resolved["disk_boundary_align"] =
      align && align->is_string() && !align->str().empty()
          ? static_cast<unsigned long long>(std::strtoull(align->str().c_str(), nullptr, 10))
          : 256ULL;
  settings["resolved"] = std::move(resolved);
  return settings;
}

// Resolve per-request switches in the hook 1 builder, using the sampler's
// configured values rather than reparsing request fields or environment text.
[[nodiscard]] inline json::Value BuildEffectiveSettings(
    json::Value settings, const sampling::SamplingConfig& request_sampling,
    bool cache_prompt) {
  settings["resolved"]["cache_prompt"] = cache_prompt;
  auto& behaviour = settings["resolved"]["behaviour"];
  sampling::BehaviourConfig inactive;
  const auto& policy = request_sampling.behaviour ? *request_sampling.behaviour : inactive;
  behaviour["enabled"] = request_sampling.behaviour != nullptr;
  behaviour["think_eog_guard"] = policy.think_eog_guard;
  behaviour["eog_boundary"] = policy.eog_boundary;
  behaviour["eog_min_prob_exact"] = policy.eog_min_prob_exact;
  behaviour["eog_min_prob"] = policy.eog_min_prob;
  behaviour["eog_defer_p"] = policy.eog_defer_p;
  behaviour["spec_eog_stop"] = policy.spec_eog_stop;
  behaviour["loop_breaker"] = policy.loop_breaker;
  behaviour["loop_eog_close"] = policy.loop_eog_close;
  behaviour["loop_transition"] = policy.loop_transition;
  behaviour["reasoning_budget_enabled"] = policy.reasoning_budget_enabled;
  behaviour["logit_bias_enabled"] = policy.logit_bias_enabled;
  behaviour["model_suppress_tokens"] = policy.model_suppress_tokens;
  behaviour["production_sampling"] = policy.production_sampling;
  behaviour["marker_bias"] = policy.marker_bias;
  auto efforts = json::Value::object();
  for (const char* effort : {"minimal", "low", "medium", "high", "xhigh", "max"})
    efforts[effort] = ResolveMarkerBias(policy, effort);
  behaviour["marker_bias_by_effort"] = std::move(efforts);
  if (auto* ram = settings["resolved"].find("ram_cache"))
    settings["resolved"]["ram_cache"]["enabled"] =
        ram->member_size("capacity_bytes") != 0;
  return settings;
}

// This is a projection of hook 1's builder, never a second environment parser.
[[nodiscard]] inline json::Value BuildTimelineBundle(
    json::Value cached_settings, const sampling::SamplingConfig& sampling, bool cache_prompt) {
  const auto settings = BuildEffectiveSettings(std::move(cached_settings),
                                               sampling, cache_prompt);
  const auto& effective = *settings.find("resolved");
  auto bundle = json::Value::object();
  for (const char* name : {"behaviour", "lenient", "mtp_policy", "cache_prompt", "ram_cache",
                           "disk_cache", "disk_boundary_align"}) {
    if (const auto* value = effective.find(name))
      bundle[name] = *value;
  }
  if (bundle.contains("mtp_policy")) {
    auto* policy = &bundle["mtp_policy"];
    (*policy)["name"] =
        !policy->find("enabled") || !policy->find("enabled")->as_bool()
            ? "none"
            : policy->find("survival") && policy->find("survival")->as_bool()
                  ? "survival" : "length";
    // The engine installs the latin table only when MTP weights exist (engine.cpp:219-221): report the property only for an enabled policy.
    if (const auto* vocab = policy->find("draft_vocabulary");
        vocab && policy->find("enabled") && policy->find("enabled")->as_bool())
      (*policy)["latin_vocab"] = vocab->str() == "latin_text" ? "latin" : "full";
    if (const auto* cap = policy->find("runner_max_draft_tokens"))
      (*policy)["cap"] = *cap;
  }
  return bundle;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_SETTINGS_HPP_
