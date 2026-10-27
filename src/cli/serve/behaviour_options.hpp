#ifndef GUFO_SERVER_BEHAVIOUR_OPTIONS_HPP_
#define GUFO_SERVER_BEHAVIOUR_OPTIONS_HPP_

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <set>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/reasoning.hpp"

namespace gufo::server {

// Production anchors: common/sampling.cpp:1096-1167, server-context.cpp:58-156,
// server-common.cpp:77-105 and server-schema.cpp:394-503. All new policies are
// opt-in; no environment variables means the original Gufo sampler.
inline bool BehaviourSwitch(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return false;
  const std::string_view text(value);
  if (text == "0") return false;
  if (text == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

inline double BehaviourNumber(const char* name, double disabled) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return disabled;
  std::size_t end = 0;
  double number = std::stod(value, &end);
  if (end != std::string_view(value).size() || !std::isfinite(number))
    throw std::invalid_argument(std::string(name) + " must be finite");
  return number;
}

inline int ThinkGuardEscalate() {
  const char* value = std::getenv("GUFO_THINK_GUARD_ESCALATE");
  if (value == nullptr)
    return 16;
  const std::string_view text(value);
  int result = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || text.front() < '0' || text.front() > '9' ||
      error != std::errc{} || end != text.data() + text.size() || result < 0)
    throw std::invalid_argument(
        "GUFO_THINK_GUARD_ESCALATE must be an integer in [0,2147483647]");
  return result;
}

inline std::shared_ptr<sampling::BehaviourConfig> ParseBehaviourOptions(
    const json::Value& body,
    std::shared_ptr<const sampling::BehaviourConfig> previous = {}) {
  auto cfg = previous ? std::make_shared<sampling::BehaviourConfig>(*previous)
                      : std::make_shared<sampling::BehaviourConfig>();
  cfg->think_eog_guard = BehaviourSwitch("GUFO_THINK_EOG_GUARD");
  cfg->think_guard_escalate = ThinkGuardEscalate();
  cfg->eog_boundary = BehaviourSwitch("GUFO_EOG_BOUNDARY");
  cfg->eog_min_prob_exact = BehaviourSwitch("GUFO_EOG_MIN_PROB_EXACT");
  cfg->eog_min_prob = BehaviourNumber("GUFO_EOG_MIN_PROB", -1.0);
  cfg->eog_defer_p = BehaviourNumber("GUFO_EOG_DEFER_P", 0.0);
  if (cfg->eog_min_prob < -1.0 || cfg->eog_min_prob > 1.0 ||
      cfg->eog_defer_p < 0.0 || cfg->eog_defer_p > 1.0)
    throw std::invalid_argument("EOG probabilities must be in [0,1] (-1 disables minimum)");
  cfg->spec_eog_stop = BehaviourSwitch("GUFO_SPEC_EOG_STOP");
  cfg->loop_breaker = BehaviourSwitch("GUFO_LOOP_BREAKER");
  const char* tool_args = std::getenv("GUFO_LOOP_BREAKER_TOOL_ARGS");
  cfg->loop_tool_args = !tool_args || !*tool_args || BehaviourSwitch("GUFO_LOOP_BREAKER_TOOL_ARGS");
  cfg->loop_eog_close = BehaviourSwitch("GUFO_LOOP_BREAKER_EOG_CLOSE");
  const double transition = BehaviourNumber("GUFO_LOOP_BREAKER_TRANSITION", 2.0);
  if (transition != std::floor(transition) || transition < 0 || transition > 2)
    throw std::invalid_argument("GUFO_LOOP_BREAKER_TRANSITION must be 0, 1 or 2");
  cfg->loop_transition = static_cast<int>(transition);
  cfg->reasoning_budget_enabled = BehaviourSwitch("GUFO_REASONING_BUDGET");
  cfg->logit_bias_enabled = BehaviourSwitch("GUFO_LOGIT_BIAS");
  cfg->model_suppress_tokens = BehaviourSwitch("GUFO_MODEL_SUPPRESS_TOKENS");
  cfg->marker_bias = BehaviourNumber("GUFO_THINK_MARKER_BIAS", 0.0);
  cfg->marker_entropy_gate.reset();
  if (const char* gate = std::getenv("GUFO_THINK_MARKER_ENTROPY_GATE");
      gate && *gate) {
    const double tau = BehaviourNumber("GUFO_THINK_MARKER_ENTROPY_GATE", 0.0);
    if (tau < 0 || tau > 12)
      throw std::invalid_argument(
          "GUFO_THINK_MARKER_ENTROPY_GATE must be in [0,12]");
    cfg->marker_entropy_gate = tau;
  }
  if (const char* map = std::getenv("GUFO_THINK_MARKER_BIAS_BY_EFFORT"))
    cfg->marker_bias_by_effort = map;
  cfg->production_sampling = BehaviourSwitch("GUFO_PRODUCTION_SAMPLING");

  if (const auto* budget = body.find("reasoning_budget_tokens");
      cfg->reasoning_budget_enabled && budget && !budget->is_null()) {
    const double n = budget->is_number() ? budget->as_double() : -2.0;
    if (!std::isfinite(n) || n != std::floor(n) || n < -1 || n > INT32_MAX)
      throw std::invalid_argument("reasoning_budget_tokens must be an integer in [-1,2147483647]");
    cfg->reasoning_budget = static_cast<std::int64_t>(n);
  }
  if (const auto* message = body.find("reasoning_budget_message");
      cfg->reasoning_budget_enabled && message && !message->is_null()) {
    if (!cfg->reasoning_budget_enabled || !message->is_string())
      throw std::invalid_argument("reasoning_budget_message requires budget switch and a string");
    cfg->reasoning_budget_message = message->str();
  }
  if (cfg->reasoning_budget_enabled) {
    if (const auto* start=body.find("reasoning_budget_start_tag")) {
      if (!start->is_string() || start->str().empty())
        throw std::invalid_argument("reasoning_budget_start_tag must be a nonempty string");
      cfg->reasoning_start_tag=start->str();
    }
    const auto* tags=body.find("reasoning_budget_end_tags");
    if (tags) {
      if (!tags->is_array() || tags->empty())
        throw std::invalid_argument("reasoning_budget_end_tags must be a nonempty array");
      cfg->reasoning_end_tags.clear();
      for (const auto& tag : tags->items()) {
        if (!tag.is_string() || tag.str().empty())
          throw std::invalid_argument("reasoning end tags must be nonempty strings");
        cfg->reasoning_end_tags.push_back(tag.str());
      }
    } else if (const auto* end=body.find("reasoning_budget_end_tag")) {
      if (!end->is_string() || end->str().empty())
        throw std::invalid_argument("reasoning_budget_end_tag must be a nonempty string");
      cfg->reasoning_end_tags={end->str()};
    }
  }
  if (const auto* bias = body.find("think_marker_bias");
      bias && !bias->is_null() && (!cfg->marker_bias_by_effort.empty() || cfg->marker_bias != 0.0)) {
    if (!bias->is_number() || !std::isfinite(bias->as_double()))
      throw std::invalid_argument("think_marker_bias must be finite");
    cfg->request_marker_bias = bias->as_double();
  }
  const auto add_bias = [&cfg](std::string_view key, const json::Value& value) {
    float bias;
    if (value.is_bool() && !value.as_bool())
      bias = -std::numeric_limits<float>::infinity();
    else if (value.is_number() && std::isfinite(value.as_double()) &&
             std::abs(value.as_double()) <= std::numeric_limits<float>::max())
      bias = static_cast<float>(value.as_double());
    else throw std::invalid_argument("logit_bias values must be finite numbers or false");
    std::uint32_t id = 0;
    auto [end, error] = std::from_chars(key.data(), key.data()+key.size(), id);
    if (error == std::errc{} && end == key.data()+key.size())
      cfg->logit_bias.emplace_back(id, bias);
    else if (!key.empty() && key.front() != '-' && key.front() != '+' &&
             !std::all_of(key.begin(),key.end(),[](unsigned char ch){return ch>='0' && ch<='9';}))
      cfg->text_logit_bias.emplace_back(std::string(key), bias);
    else throw std::invalid_argument("logit_bias token ids must be non-negative integers");
  };
  if (const auto* biases = body.find("logit_bias");
      cfg->logit_bias_enabled && biases && !biases->is_null()) {
    cfg->logit_bias.clear(); cfg->text_logit_bias.clear();
    if (biases->is_object()) {
      for (const auto& [key,value] : biases->members()) add_bias(key,value);
    } else if (biases->is_array()) {
      for (const auto& pair : biases->items()) {
        if (!pair.is_array() || pair.size()!=2)
          throw std::invalid_argument("logit_bias array entries must be [token,bias]");
        const auto& token = pair.items()[0];
        if (token.is_string()) add_bias(token.str(),pair.items()[1]);
        else if (token.is_number() && token.as_double() >= 0 &&
                 token.as_double() <= UINT32_MAX && token.as_double()==std::floor(token.as_double()))
          add_bias(std::to_string(static_cast<std::uint32_t>(token.as_double())),pair.items()[1]);
        else throw std::invalid_argument("logit_bias token must be an id or string");
      }
    } else throw std::invalid_argument("logit_bias must be an object or array");
  }
  if (!(cfg->think_eog_guard || cfg->eog_boundary || cfg->eog_min_prob >= 0 ||
        cfg->eog_defer_p > 0 || cfg->spec_eog_stop || cfg->loop_breaker ||
        cfg->reasoning_budget_enabled || cfg->logit_bias_enabled ||
        cfg->model_suppress_tokens || cfg->marker_bias != 0 ||
        !cfg->marker_bias_by_effort.empty() || cfg->production_sampling))
    return nullptr;
  return cfg;
}

// Explicit request > resolved effort mapping > process-wide bias > zero.
inline double ResolveMarkerBias(const sampling::BehaviourConfig& cfg,
                                std::string_view effort) {
  if (cfg.request_marker_bias) return *cfg.request_marker_bias;
  if (effort == "high" || effort == "max") effort = "xhigh";
  if (effort == "minimal") effort = "low";
  double result = cfg.marker_bias;
  std::string_view map(cfg.marker_bias_by_effort);
  const auto trim=[](std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
  };
  map=trim(map);
  std::set<std::string> seen;
  if (!map.empty() && map.back()==',')
    throw std::invalid_argument("marker effort map must not have a trailing comma");
  while (!map.empty()) {
    const auto comma = map.find(',');
    const auto entry = trim(map.substr(0,comma));
    const auto colon = entry.find(':');
    if (colon == std::string_view::npos)
      throw std::invalid_argument("marker effort map entries must be effort:bias");
    const auto name=trim(entry.substr(0,colon));
    const auto number=trim(entry.substr(colon+1));
    if (name.empty() || !seen.insert(std::string(name)).second)
      throw std::invalid_argument("marker effort map labels must be nonempty and unique");
    std::size_t parsed = 0;
    double value;
    try { value=std::stod(std::string(number),&parsed); }
    catch (const std::exception&) { throw std::invalid_argument("marker effort map bias must be finite"); }
    if (parsed != number.size() || !std::isfinite(value) || std::abs(value)>std::numeric_limits<float>::max())
      throw std::invalid_argument("marker effort map bias must be finite");
    if (name == effort) result = value;
    if (comma == std::string_view::npos) break;
    map.remove_prefix(comma+1);
  }
  return result;
}

inline std::string_view BehaviourEffortName(std::optional<ReasoningEffort> effort) {
  if (!effort) return {};
  switch (*effort) {
    case ReasoningEffort::kMinimal: return "minimal";
    case ReasoningEffort::kLow: return "low";
    case ReasoningEffort::kMedium: return "medium";
    case ReasoningEffort::kHigh: return "high";
    case ReasoningEffort::kXHigh: return "xhigh";
    case ReasoningEffort::kMax: return "max";
  }
  return {};
}
} // namespace gufo::server
#endif
