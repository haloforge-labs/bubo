#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "src/cli/serve/release_profile.hpp"

namespace {
using gufo::json::Value;
using gufo::server::ReleaseInputs;
using gufo::server::ReleaseResolution;

int checks = 0;
void Check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
  ++checks;
}
const Value& Member(const Value& object, const char* name) {
  const auto* value = object.find(name);
  if (!value)
    throw std::runtime_error(std::string("missing ") + name);
  return *value;
}
const Value& Switch(const Value& settings, const std::string& name) {
  for (const auto& row : Member(settings, "effective_switches").items())
    if (row.member_str("name") == name)
      return row;
  throw std::runtime_error("missing switch " + name);
}
ReleaseInputs Inputs(const std::string& profile = "default") {
  ReleaseInputs inputs;
  inputs.profile = profile;
  inputs.model_family = "qwen3.8-flash-next";
  inputs.architecture = "qwen4exp";
  return inputs;
}
const gufo::server::ReleaseSetting& Setting(const ReleaseResolution& resolved,
                                            const std::string& name) {
  for (const auto& setting : resolved.settings)
    if (setting.name == name)
      return setting;
  throw std::runtime_error("missing resolver setting " + name);
}
void Profiles() {
  for (const auto* profile : {"default", "exact", "vanilla", "subagents"}) {
    auto resolved = gufo::server::ResolveReleaseSettings(Inputs(profile));
    auto settings = gufo::server::ReleaseSettingsJson(resolved);
    Check(settings.member_str("profile") == profile, "profile matches");
    Check(Switch(settings, "--profile").member_str("effective") == profile,
          "profile switch reports the selected profile");
    const auto& switches = Member(settings, "effective_switches");
    Check(switches.is_array() &&
              switches.size() == gufo::server::ReleaseSwitchTable().size(),
          "every registry row has an effective value");
    std::set<std::string> names;
    for (const auto& row : switches.items()) {
      Check(names.insert(row.member_str("name")).second,
            "each switch is listed once");
      Check(Member(row, "group").is_string() &&
                Member(row, "audience").is_string() &&
                Member(row, "allowed").is_object(),
            "typed switch metadata is present");
      Check(row.member_str("source") == "default" ||
                row.member_str("source") == "profile" ||
                row.member_str("source") == "user" ||
                row.member_str("source") == "env",
            "effective switch source is public vocabulary");
    }
    const auto& inference = Member(settings, "inference_defaults");
    for (const auto* name : {"temperature", "top_p", "top_k", "min_p",
                             "repeat_penalty", "presence_penalty", "seed",
                             "reasoning_effort", "think", "max_tokens"}) {
      const auto& entry = Member(inference, name);
      Check(entry.is_object() && Member(entry, "source").is_string() &&
                !Member(entry, "value").is_null(),
            "typed inference default is present");
    }
    Check(Member(inference, "temperature").find("value")->is_number(),
          "temperature is numeric");
    Check(Member(inference, "top_k").find("value")->is_number(),
          "top k is numeric");
    Check(Member(inference, "max_tokens").find("value")->as_double() == 0,
          "unlimited output uses backend value zero");
    const auto& npu = Member(settings, "npu_prefill");
    if (std::string(profile) == "exact" || std::string(profile) == "vanilla") {
      Check(
          npu.member_str("effective") == "off" &&
              Member(npu, "forced").as_bool() &&
              npu.member_str("reason").find("forces OFF") != std::string::npos,
          "restricted profiles force NPU off with reason");
    }
#ifndef GUFO_NPU
    Check(npu.member_str("effective") == "off", "non-NPU build reports off");
#else
    // v0.1.1: the NPU is off by default in every profile; only an explicit
    // --npu-prefill on turns it on (release_profile_test covers the opt-in).
    Check(npu.member_str("effective") == "off",
          "NPU-capable build keeps the NPU off by default");
#endif
    Check(!Member(settings, "entropy_gate").find("available")->as_bool(),
          "entropy gate is unavailable");
  }
  const auto default_settings = gufo::server::ReleaseSettingsJson(
      gufo::server::ResolveReleaseSettings(Inputs()));
  Check(
      Member(Member(default_settings, "inference_defaults"), "reasoning_effort")
              .member_str("source") == "profile",
      "default effort comes from profile");
  const auto stock_settings = gufo::server::ReleaseSettingsJson(
      gufo::server::ResolveReleaseSettings(Inputs("vanilla")));
  Check(Member(Member(stock_settings, "inference_defaults"), "reasoning_effort")
                .member_str("source") == "default",
        "vanilla effort comes from model default");
  Check(!Member(Member(stock_settings, "loop_breaker"), "value").as_bool() &&
            Switch(stock_settings, "--loop-breaker").member_str("effective") ==
                "off",
        "vanilla reports the disabled runtime loop breaker");
  Check(Switch(stock_settings, "--no-twin").find("effective")->is_bool() &&
            !Switch(stock_settings, "--no-twin").find("effective")->as_bool(),
        "unset boolean flags report a known false value");
  const auto& profiles =
      Member(Member(default_settings, "capabilities"), "profiles").items();
  Check(profiles.size() == 4 && !Member(profiles.back(), "gui").as_bool() &&
            profiles.back().member_str("name") == "subagents",
        "subagents stays CLI only");
}
void Precedence() {
  auto inputs = Inputs();
  Check(Setting(gufo::server::ResolveReleaseSettings(Inputs("vanilla")),
                "GUFO_EOG_MIN_PROB")
                .origin == "built-in",
        "vanilla starts at built-in behaviour default");
  Check(
      Setting(gufo::server::ResolveReleaseSettings(inputs), "GUFO_EOG_MIN_PROB")
              .origin == "profile:default",
      "profile beats built-in behaviour default");
  inputs.manifest_defaults["GUFO_EOG_MIN_PROB"] = "0.3";
  Check(
      Setting(gufo::server::ResolveReleaseSettings(inputs), "GUFO_EOG_MIN_PROB")
              .origin == "manifest",
      "manifest beats profile behaviour default");
  inputs.environment["GUFO_EOG_MIN_PROB"] = "0.4";
  Check(
      Setting(gufo::server::ResolveReleaseSettings(inputs), "GUFO_EOG_MIN_PROB")
              .origin == "env",
      "environment beats manifest behaviour default");
  inputs.explicit_flags["GUFO_EOG_MIN_PROB"] = "0.5";
  Check(
      Setting(gufo::server::ResolveReleaseSettings(inputs), "GUFO_EOG_MIN_PROB")
              .origin == "flag",
      "explicit override beats environment behaviour default");
  inputs.explicit_flags.erase("GUFO_EOG_MIN_PROB");
  inputs.manifest_defaults["--temperature"] = "0.6";
  auto manifest = gufo::server::ReleaseSettingsJson(
      gufo::server::ResolveReleaseSettings(inputs));
  Check(std::abs(Member(Member(manifest, "inference_defaults"), "temperature")
                     .find("value")
                     ->as_double() -
                 0.6) < 0.001 &&
            Member(Member(manifest, "inference_defaults"), "temperature")
                    .member_str("source") == "profile",
        "manifest sampling default is typed and sourced");
  inputs.explicit_flags["--temperature"] = "0.4";
  inputs.environment["HOST"] = "127.0.0.2";
  inputs.environment["PORT"] = "8123";
  inputs.explicit_flags["--host"] = "127.0.0.1";
  inputs.explicit_flags["--port"] = "18110";
  auto resolved = gufo::server::ResolveReleaseSettings(inputs);
  auto settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(
      Member(Member(settings, "inference_defaults"), "temperature")
                  .member_str("source") == "flag" &&
          std::abs(Member(Member(settings, "inference_defaults"), "temperature")
                       .find("value")
                       ->as_double() -
                   0.4) < 0.001,
      "flag beats manifest sampling default");
  Check(Switch(settings, "--host").member_str("effective") == "127.0.0.1" &&
            Switch(settings, "--host").member_str("source") == "user" &&
            Switch(settings, "--port").member_str("effective") == "18110",
        "explicit endpoint beats environment");
  inputs.explicit_flags.erase("--host");
  inputs.explicit_flags.erase("--port");
  resolved = gufo::server::ResolveReleaseSettings(inputs);
  settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(Switch(settings, "--host").member_str("effective") == "127.0.0.2" &&
            Switch(settings, "--port").member_str("effective") == "8123" &&
            Switch(settings, "--host").member_str("source") == "env",
        "environment endpoint fallback is visible");
  resolved.runtime_host = "127.0.0.2";
  resolved.runtime_port = 18201;
  resolved.runtime_context = 131072;
  settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(Member(Switch(settings, "--port"), "effective").as_double() == 18201 &&
            Member(Switch(settings, "--context"), "effective").as_double() ==
                131072,
        "bound port and loaded context replace requested values");
  inputs.environment["HOME"] = "synthetic-home-secret";
  settings = gufo::server::ReleaseSettingsJson(
      gufo::server::ResolveReleaseSettings(inputs));
  Check(settings.dump().find("synthetic-home-secret") == std::string::npos,
        "arbitrary environment text is omitted");
}
void Behaviour() {
  auto inputs = Inputs();
  inputs.environment["GUFO_LOOP_BREAKER"] = "0";
  auto resolved = gufo::server::ResolveReleaseSettings(inputs);
  auto settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(!Member(Member(settings, "loop_breaker"), "value").as_bool() &&
            Member(settings, "loop_breaker").member_str("source") == "env",
        "environment breaker overrides profile");
  inputs.explicit_flags["--loop-breaker"] = "on";
  inputs.explicit_flags["--think-marker-bias"] = "0.75";
  inputs.environment["GUFO_THINK_MARKER_BIAS"] = "-1";
  resolved = gufo::server::ResolveReleaseSettings(inputs);
  settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(Member(Member(settings, "loop_breaker"), "value").as_bool() &&
            Member(settings, "loop_breaker").member_str("source") == "user",
        "breaker flag beats environment");
  const auto& bias = Member(settings, "think_marker_bias");
  Check(std::abs(Member(bias, "value").as_double() - 0.75) < 0.001 &&
            bias.member_str("source") == "user" &&
            std::abs(Member(Member(Member(bias, "by_effort"), "low"), "value")
                         .as_double() -
                     0.75) < 0.001,
        "bias flag overrides environment and effort map");
  Check(!Setting(resolved, "GUFO_THINK_MARKER_BIAS_BY_EFFORT").value,
        "explicit bias clears the effort map");
  bool injected_breaker = false, injected_bias = false, cleared_map = false;
  for (const auto& injection : resolved.injections) {
    injected_breaker |= injection.name == "GUFO_LOOP_BREAKER" &&
                        injection.origin == "flag" &&
                        injection.value == std::optional<std::string>("1");
    injected_bias |= injection.name == "GUFO_THINK_MARKER_BIAS" &&
                     injection.origin == "flag" &&
                     injection.value == std::optional<std::string>("0.75");
    cleared_map |= injection.name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT" &&
                   injection.origin == "flag" && !injection.value;
  }
  Check(injected_breaker && injected_bias && cleared_map,
        "flags reach the request environment");
  resolved.runtime_npu_enabled = false;
  resolved.runtime_npu_reason = "NPU startup failed.";
  settings = gufo::server::ReleaseSettingsJson(resolved);
  Check(Member(settings, "npu_prefill").member_str("effective") == "off" &&
            Member(settings, "npu_prefill").member_str("reason") ==
                "NPU startup failed." &&
            Switch(settings, "--npu-prefill").member_str("effective") == "off",
        "runtime NPU fallback replaces authorized request");
  auto exact = Inputs("exact");
  exact.environment["GUFO_LOOP_BREAKER"] = "1";
  exact.environment["GUFO_THINK_MARKER_BIAS"] = "0.5";
  exact.environment["GUFO_THINK_MARKER_BIAS_BY_EFFORT"] = "low:0.5";
  exact.explicit_flags["--loop-breaker"] = "off";
  exact.explicit_flags["--think-marker-bias"] = "0";
  settings = gufo::server::ReleaseSettingsJson(
      gufo::server::ResolveReleaseSettings(exact));
  Check(
      !Member(Member(settings, "loop_breaker"), "value").as_bool() &&
          Member(settings, "think_marker_bias").find("value")->as_double() == 0,
      "exact off flags override inherited behaviour environment");
}
}  // namespace

int main() {
  try {
    Profiles();
    Precedence();
    Behaviour();
    std::cout << "Settings contract: " << checks << " passed, 0 failed\n";
  } catch (const std::exception& error) {
    std::cerr << "Settings contract: " << checks
              << " passed, 1 failed: " << error.what() << '\n';
    return 1;
  }
}
