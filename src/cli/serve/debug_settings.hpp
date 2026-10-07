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

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_SETTINGS_HPP_
