#include "src/cli/serve/release_profile.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/cli/serve/artefact_binding_cases.hpp"
#include "src/cli/serve/behaviour_options.hpp"
#include "src/cli/settings_support.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/text_sampling_defaults.hpp"
#include "src/models/qwen38_flash_next/batch_twin_policy.hpp"

namespace gufo::server {
namespace {
std::optional<std::string> CopyEnvironmentValue(const char* value) {
  return value ? std::optional<std::string>(value) : std::nullopt;
}
bool Has(std::string_view value,
         std::initializer_list<std::string_view> choices) {
  return std::find(choices.begin(), choices.end(), value) != choices.end();
}
std::optional<std::string> Text(const json::Value& row,
                                const std::string& key) {
  const auto* value = row.find(key);
  return value && value->is_string() ? std::optional(value->str())
                                     : std::nullopt;
}
std::optional<std::string> ProfileValue(const json::Value& row,
                                        const std::string& profile) {
  if (profile == "vanilla")
    return {};
  const auto& values = *row.find("values");
  if (auto value = Text(values, profile))
    return value;
  return profile == "exact" ? Text(values, "default") : std::nullopt;
}
void ValidateValue(const json::Value& row, const std::string& value) {
  const auto rule = row.member_str("value_rule");
  const auto fail = [&] {
    throw ReleaseSettingError(
        row.member_str("name"),
        row.member_str("name") + ": invalid value '" +
            (row.find("secret")->as_bool() ? "<redacted>" : value) + "'");
  };
  const auto& allowed = *row.find("allowed");
  if (const auto* values = allowed.find("values")) {
    const bool listed =
        std::ranges::any_of(values->items(), [&](const auto& item) {
          return value == (item.is_string() ? item.str()
                           : item.is_bool() ? (item.as_bool() ? "1" : "0")
                                            : item.dump());
        });
    if (!listed)
      fail();
    return;
  }
  if (rule == "uint" || rule == "int31") {
    const auto parse = [&](const std::string& text) {
      std::uint64_t number = 0;
      const auto parsed =
          std::from_chars(text.data(), text.data() + text.size(), number);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        fail();
      return number;
    };
    const auto number = parse(value);
    // Use exact JSON number text: uint64 maxima cannot round through double.
    for (const auto* key : {"minimum", "exclusive_minimum", "maximum"}) {
      if (const auto* bound = allowed.find(key)) {
        const auto limit = parse(bound->dump());
        if ((std::string_view(key) == "minimum" && number < limit) ||
            (std::string_view(key) == "exclusive_minimum" && number <= limit) ||
            (std::string_view(key) == "maximum" && number > limit))
          fail();
      }
    }
    return;
  }
  // Effort maps and platform tuning have reader-owned structured grammars;
  // allowed describes them as formatted strings rather than duplicating
  // parsers.
  if (rule == "effort_map") {
    sampling::BehaviourConfig config;
    config.marker_bias_by_effort = value;
    (void)ResolveMarkerBias(config, "medium");
    return;
  }
  if (rule == "tuning") {
    std::istringstream stream(value);
    for (std::string item; std::getline(stream, item, ',');) {
      const auto first = item.find_first_not_of(' ');
      if (first == std::string::npos)
        continue;
      item = item.substr(first, item.find_last_not_of(' ') - first + 1);
      if (item == "all" || item == "none")
        continue;
      if (item.front() == '+' || item.front() == '-')
        item.erase(0, 1);
      bool known = false;
      for (const auto& field : platform::detail::kTuningFields)
        known |= item == field.name;
      if (!known)
        fail();
    }
    return;
  }
  if (allowed.member_str("type") == "string")
    return;
  std::size_t end = 0;
  double number = 0;
  if (rule == "forward") {
    // PrefillChunkTokens uses strtol's signed decimal prefix grammar, which
    // allowed cannot express. Only that grammar stays here; bounds are row
    // data.
    number = std::strtol(value.c_str(), nullptr, 10);
  } else {
    try {
      number = std::stod(value, &end);
    } catch (...) {
      fail();
    }
    if (end != value.size() || !std::isfinite(number))
      fail();
  }
  if (allowed.member_str("type") == "integer" && number != std::floor(number))
    fail();
  if (const auto* alternatives = allowed.find("also"))
    for (const auto& alternative : alternatives->items())
      if (number == alternative.as_double())
        return;
  if (const auto* minimum = allowed.find("minimum"))
    if (number < minimum->as_double())
      fail();
  if (const auto* minimum = allowed.find("exclusive_minimum"))
    if (number <= minimum->as_double())
      fail();
  if (const auto* maximum = allowed.find("maximum"))
    if (number > maximum->as_double())
      fail();
}

// The two rows that bind an MTP head and decode with it are scoped to Flash
// Next in the registry. A Gemma 4 package that carries an mtp head shares
// them. Every other row and every other family is unchanged.
bool SharedWithGemma4(const json::Value& row, const ReleaseInputs& inputs) {
  const auto name = row.member_str("name");
  return inputs.model_family == "gemma4" &&
         row.member_str("model_family") == "qwen3.8-flash-next" &&
         (name == "--speculative" || name == "--mtp-model");
}

bool Matches(const json::Value& row, const ReleaseInputs& inputs) {
  const auto family = row.member_str("model_family");
  if (family != "any" && family != inputs.model_family &&
      !SharedWithGemma4(row, inputs))
    return false;
#ifndef _WIN32
  if (row.member_str("platform") == "windows")
    return false;
#endif
  return true;
}
std::string EscapeCell(std::string text) {
  for (std::size_t at = 0; (at = text.find('|', at)) != std::string::npos;
       at += 2)
    text.insert(at, "\\");
  std::replace(text.begin(), text.end(), '\n', ' ');
  std::replace(text.begin(), text.end(), '\r', ' ');
  for (std::size_t at = 0; (at = text.find("TODO:", at)) != std::string::npos;)
    text.replace(at, 5, "Follow-up:");
  return text.empty() ? "None" : text;
}
std::string AllowedText(const json::Value& allowed) {
  if (const auto* values = allowed.find("values")) {
    std::string result;
    for (const auto& value : values->items()) {
      if (!result.empty())
        result += ", ";
      result += value.is_string() ? value.str() : value.dump();
    }
    return result;
  }
  const auto* minimum = allowed.find("minimum");
  const auto* exclusive = allowed.find("exclusive_minimum");
  if (minimum || exclusive) {
    std::string result =
        allowed.member_str("type") + " " + (exclusive ? "(" : "[") +
        (exclusive ? exclusive->dump() : minimum->dump()) + ", ";
    const auto* maximum = allowed.find("maximum");
    result += maximum ? maximum->dump() : "\xE2\x88\x9E";
    result += "]";
    if (const auto* also = allowed.find("also")) {
      result += " or ";
      bool first = true;
      for (const auto& value : also->items()) {
        if (!first)
          result += ", ";
        result += value.dump();
        first = false;
      }
    }
    return result;
  }
  std::string result = allowed.member_str("type");
  if (const auto* format = allowed.find("format"))
    result += ": " + format->str();
  return result;
}
std::string ProfileDefaultsText(const json::Value& row) {
  const auto& values = *row.find("values");
  const auto value = [&](std::string_view name) {
    const auto* selected = values.find(std::string(name));
    return selected && selected->is_string()
               ? selected->str()
               : row.member_str("default_text") + " (built-in)";
  };
  const auto* exact = values.find("exact");
  return "default=" + value("default") + ". subagents=" + value("subagents") +
         ". exact=" +
         (exact && exact->is_string() ? exact->str() : value("default"));
}
const json::Value& SwitchNormalization() {
  static const json::Value contract =
#define GUFO_RELEASE_NORMALIZATION
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_NORMALIZATION
      ;
  return contract;
}
}  // namespace

void ValidateSwitchValue(const json::Value& row, const std::string& value) {
  ValidateValue(row, value);
}

std::vector<json::Value> NormalizeSwitchTable(
    std::span<const json::Value> rows) {
  std::vector<json::Value> normalized;
  normalized.reserve(rows.size());
  for (const auto& source : rows) {
    if (!source.is_object())
      throw std::invalid_argument("invalid switch row");
    auto row = source;
    for (const auto& [key, value] :
         SwitchNormalization().find("defaults")->members())
      if (!row.contains(key))
        row[key] = value;
    if (!source.contains("restart_required") &&
        row.member_str("class") == "REMOVED")
      row["restart_required"] =
          *SwitchNormalization().find("removed_restart_required");
    normalized.push_back(std::move(row));
  }
  std::stable_sort(normalized.begin(), normalized.end(),
                   [](const auto& a, const auto& b) {
                     return a.member_str("name") < b.member_str("name");
                   });
  return normalized;
}
const std::vector<json::Value>& ReleaseSwitchTable() {
  static const auto rows = [] {
    const std::vector<json::Value> source = {
#define GUFO_RELEASE_ROWS
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_ROWS
    };
    return NormalizeSwitchTable(source);
  }();
  return rows;
}
ReleaseEnvironment SnapshotReleaseEnvironment() {
  ReleaseEnvironment environment;
#define GUFO_RELEASE_SNAPSHOT
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_SNAPSHOT
  return environment;
}
void ValidateSwitchTable(std::span<const json::Value> source_rows) {
  const auto rows = NormalizeSwitchTable(source_rows);
  const std::set<std::string> fields = {
      "name",           "kind",          "class",        "model_family",
      "platform",       "requires",      "values",       "exact_off",
      "inject_with",    "suppressed_by", "purpose",      "default_text",
      "hardware_scope", "note",          "built_in",     "value_rule",
      "aliases",        "takes_value",   "echo",         "secret",
      "group",          "audience",      "experimental", "restart_required",
      "gui_control",    "allowed"};
  std::set<std::string> names;
  for (const auto& row : rows) {
    const auto name = row.member_str("name");
    const auto fail = [&] {
      throw std::invalid_argument("invalid switch row: " + name);
    };
    if (row.size() != fields.size() + (row.contains("label") ? 1 : 0) ||
        name.empty() || !names.insert(name).second)
      fail();
    for (const auto& key : fields)
      if (!row.contains(key))
        fail();
    if (const auto* label = row.find("label"))
      if (!label->is_string() ||
          label->str().find_first_not_of(" \t\r\n\v\f") == std::string::npos)
        fail();
    if (!Has(row.member_str("kind"), {"env", "flag"}) ||
        !Has(row.member_str("class"), {"DEFAULT", "OPTION", "DEBUG", "INTERNAL",
                                       "REMOVED", "OTHER_MODEL"}) ||
        !Has(row.member_str("model_family"),
             {"any", "qwen3.8-flash-next", "qwen3.8-27b", "deepseek-v4-flash",
              "qwen3.6-a3b", "minimax-h3", "qwen-image-2.1", "qwen3-tts",
              "qwen3-asr"}) ||
        !Has(row.member_str("platform"), {"any", "windows"}) ||
        !Has(row.member_str("hardware_scope"),
             {"generic", "strix-halo-tuned"}) ||
        !Has(row.member_str("value_rule"),
             {"text", "bool", "min_prob", "prob", "lambda", "transition",
              "number", "effort_map", "lenient", "forward", "twin_width",
              "uint", "int31", "readers", "tuning", "on_off"}) ||
        !std::ranges::any_of(SwitchNormalization().find("groups")->items(),
                             [&](const auto& group) {
                               return row.member_str("group") == group.str();
                             }) ||
        !Has(row.member_str("audience"), {"user", "advanced", "debug"}) ||
        !row.find("experimental")->is_bool() ||
        !row.find("restart_required")->is_bool() ||
        !row.find("gui_control")->is_bool() ||
        !row.find("allowed")->is_object())
      fail();
    const bool flag_name =
        name.starts_with("--") && name.size() > 2 &&
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-", 2) ==
            std::string::npos;
    const bool env_name =
        !name.empty() && name.front() >= 'A' && name.front() <= 'Z' &&
        name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") ==
            std::string::npos;
    if ((row.member_str("kind") == "flag" && !flag_name) ||
        (row.member_str("kind") == "env" && !env_name))
      fail();
    for (const auto* key : {"purpose", "default_text"})
      if (row.member_str(key).find_first_not_of(" \t\r\n") == std::string::npos)
        fail();
    std::istringstream purpose(row.member_str("purpose"));
    std::size_t words = 0;
    for (std::string word; purpose >> word;) {
      ++words;
      std::transform(
          word.begin(), word.end(), word.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      while (!word.empty() &&
             std::ispunct(static_cast<unsigned char>(word.back())))
        word.pop_back();
      if (Has(word, {"i", "we", "our", "ours", "us", "unfortunately", "sadly",
                     "mistake", "bug"}))
        fail();
    }
    if (words == 0 || words > 20)
      fail();
    for (const auto* key : {"requires", "suppressed_by", "aliases"}) {
      const auto* list = row.find(key);
      if (!list->is_array())
        fail();
      for (const auto& entry : list->items())
        if (!entry.is_string())
          fail();
    }
    for (const auto& requirement : row.find("requires")->items())
      if (!Has(requirement.str(),
               {"sessions==1 or GUFO_DENSE_TWIN_BATCH==1", "speculative==mtp",
                "artefact:twin", "artefact:draft_head", "artefact:mtp",
                "artefact:mmproj", "think!=off"}))
        fail();
    if (!row.find("secret")->is_bool() ||
        (row.find("secret")->as_bool() &&
         (row.find("echo")->as_bool() || row.find("gui_control")->as_bool())) ||
        !row.find("echo")->is_bool() || !row.find("exact_off")->is_bool() ||
        !row.find("takes_value")->is_bool() || !row.find("note")->is_string())
      fail();
    for (const auto* key : {"inject_with", "built_in"})
      if (!row.find(key)->is_null() &&
          (!row.find(key)->is_string() || row.find(key)->str().empty()))
        fail();
    const auto& values = *row.find("values");
    if (!values.is_object() || values.size() != 3)
      fail();
    for (const auto* profile : {"default", "subagents", "exact"}) {
      const auto* value = values.find(profile);
      if (!value ||
          (!value->is_null() && (!value->is_string() || value->str().empty())))
        fail();
      if (value->is_string())
        ValidateValue(row, value->str());
    }
  }
  for (const auto& row : rows) {
    auto references = row.find("suppressed_by")->items();
    if (auto name = Text(row, "inject_with"))
      references.emplace_back(*name);
    for (const auto& name : references)
      if (!names.contains(name.str()) || !name.str().starts_with("--"))
        throw std::invalid_argument("unknown switch reference: " + name.str());
  }
}

json::Value ReleaseSettingSummary(const json::Value& row) {
  auto summary = json::Value::object();
  for (const auto* key :
       {"requested", "effective", "source", "reason", "allowed"})
    if (const auto* value = row.find(key))
      summary[key] = *value;
  return summary;
}

ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs) {
  return ResolveReleaseSettings(inputs, ReleaseSwitchTable());
}

ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& requested,
                                         std::span<const json::Value> rows) {
  auto inputs = requested;
  if (inputs.profile == "stock")
    inputs.profile = "vanilla";
  if (!Has(inputs.profile, {"default", "subagents", "exact", "vanilla"}))
    throw ReleaseSettingError("--profile",
                              "unknown profile: " + inputs.profile);
  ValidateSwitchTable(rows);
  ReleaseResolution result;
  result.profile = inputs.profile;
  result.architecture = inputs.architecture;
  result.model_family = inputs.model_family;
  result.native_context = inputs.native_context;
  // Resolve NPU policy before model metadata or configuration injection.
  const auto authorize = [&] {
    const auto found = inputs.explicit_flags.find("--npu-prefill");
    const bool present = found != inputs.explicit_flags.end();
    const std::string value = present ? found->second : "off";
    if (value != "off" && value != "on")
      throw ReleaseSettingError("--npu-prefill",
                                "--npu-prefill: invalid value '" + value + "'");
    const bool restricted =
        inputs.profile == "exact" || inputs.profile == "vanilla";
    if (restricted && present && value == "on")
      throw ReleaseSettingError("--npu-prefill",
                                "--npu-prefill effective=off: profile=" +
                                    inputs.profile + " forces OFF");
    bool allowed = !restricted && value == "on";
    std::string reason = restricted
                             ? "profile=" + inputs.profile + " forces OFF"
                         : present && value == "off" ? "explicit CLI OFF"
                         : present                   ? "explicit CLI ON"
                                                     : "profile default OFF";
#ifndef GUFO_NPU
    if (allowed) {
      allowed = false;
      reason = "GUFO_ENABLE_NPU=OFF";
    }
#endif
    return core::ExperimentalAuthorization(core::ExperimentalFeature::Prefill,
                                           allowed, std::move(reason));
  };
  result.prefill_authorization = authorize();
  const bool exact_controls =
      inputs.profile == "exact" && inputs.model_family == "qwen3.8-flash-next";
  std::map<std::string, ReleaseSetting> settings;
  const auto enabled = [&](const std::string& name) {
    auto it = inputs.explicit_flags.find(name);
    return it != inputs.explicit_flags.end() && it->second != "0";
  };
  const std::map<std::string, std::pair<std::string, std::string>>
      path_controls = {
          {"GUFO_DENSE_TWIN_MODEL", {"--twin-model", "--no-twin"}},
          {"GUFO_DRAFT_OUT_MODEL", {"--draft-head-model", "--no-draft-head"}},
          {"--mtp-model", {"--mtp-model", "--no-mtp"}},
          {"--mmproj", {"--mmproj", "--no-vision"}}};
  for (const auto& [key, control] : path_controls) {
    if (inputs.explicit_flags.contains(control.first) &&
        enabled(control.second))
      throw ReleaseSettingError(
          control.first, control.first + " conflicts with " + control.second);
    if (exact_controls && key != "--mmproj" &&
        inputs.explicit_flags.contains(control.first))
      throw ReleaseSettingError(
          control.first,
          "profile=exact: explicit value re-enables " + control.first);
  }
  if (enabled("--no-mtp") && inputs.explicit_flags.contains("--speculative") &&
      inputs.explicit_flags.at("--speculative") == "mtp")
    throw ReleaseSettingError("--speculative",
                              "--speculative mtp conflicts with --no-mtp");
  // Resolve every candidate before evaluating dependencies; row order cannot
  // affect sessions, backend, or twin-batch requirements.
  for (const auto& row : rows) {
    const auto name = row.member_str("name");
    ReleaseSetting setting{
        name, Text(row, "built_in"), "built-in", row.member_str("class"), {}};
    if (const auto requested = inputs.explicit_flags.find(name);
        requested != inputs.explicit_flags.end() &&
        row.member_str("kind") == "flag" && !row.find("secret")->as_bool())
      setting.requested = requested->second;
    // This policy is resolved before configuration for every model family. A
    // manifest or environment entry cannot override the command-line flag.
    if (name == "--npu-prefill") {
      const auto& authorization = result.prefill_authorization;
      const bool on = authorization.Allows(core::ExperimentalFeature::Prefill);
      setting.value = on ? "on" : "off";
      setting.origin = inputs.explicit_flags.contains(name)
                           ? "flag"
                           : "profile:" + inputs.profile;
      if (!on)
        setting.skipped_reason = authorization.reason();
      settings.emplace(name, std::move(setting));
      continue;
    }
    if (name.find("_NPU_") == 4 && !result.prefill_authorization.Allows(
                                       core::ExperimentalFeature::Prefill)) {
      settings.emplace(name, std::move(setting));
      continue;
    }
    const bool is_env = row.member_str("kind") == "env";
    std::optional<std::string> real;
    const std::string environment_name = name == "--mtp-model"
                                             ? "BUBO_MTP_MODEL"
                                         : name == "--mmproj" ? "BUBO_MMPROJ"
                                                              : name;
    if ((is_env || inputs.environment.contains(name) ||
         inputs.environment.contains(environment_name)) &&
        setting.release_class != "REMOVED") {
      if (inputs.real_environment)
        real = CopyEnvironmentValue(
            inputs.real_environment(environment_name.c_str()));
      else if (auto found = inputs.environment.find(name);
               found != inputs.environment.end())
        real = found->second;
      else if (auto found = inputs.environment.find(environment_name);
               found != inputs.environment.end())
        real = found->second;
      if (real && real->empty()) {
        result.logs.push_back("warning: empty environment value: " + name +
                              " (treated as unset)");
        real.reset();
      }
      if (((name == "GUFO_THINK_MARKER_BIAS" ||
            name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT") &&
           inputs.explicit_flags.contains("--think-marker-bias")) ||
          (name == "GUFO_LOOP_BREAKER" &&
           inputs.explicit_flags.contains("--loop-breaker")))
        real.reset();
    }
    const bool match = Matches(row, inputs);
    const auto path_control = path_controls.find(name);
    const bool restricted_path =
        match && path_control != path_controls.end() && name != "--mmproj" &&
        (inputs.profile == "vanilla" || exact_controls);
    if (restricted_path && real) {
      result.logs.push_back(
          name + ": inherited env suppressed by profile=" + inputs.profile);
      real.reset();
    }
    const bool exact_off = exact_controls && row.find("exact_off")->as_bool();
    auto profile = match ? ProfileValue(row, inputs.profile) : std::nullopt;
    if (inputs.profile == "exact" && !exact_controls &&
        row.find("exact_off")->as_bool())
      profile.reset();
    if (name == "--context" && profile && inputs.native_context &&
        *profile == std::to_string(*inputs.native_context)) {
      profile.reset();
      result.logs.push_back(
          "--context: model native context equals the profile default; keep "
          "native context");
    }
    bool suppressed = false;
    for (const auto& flag : row.find("suppressed_by")->items())
      suppressed |= enabled(flag.str());
    if (auto trigger = Text(row, "inject_with");
        trigger && (!inputs.explicit_flags.contains(*trigger) ||
                    inputs.explicit_flags.at(*trigger).empty()))
      profile.reset();
    if (exact_off && match) {
      result.turned_off.push_back(name);
      profile = Text(*row.find("values"), "exact");
      if (const auto flag = inputs.explicit_flags.find(name);
          flag != inputs.explicit_flags.end() &&
          row.member_str("kind") == "flag" &&
          std::optional<std::string>(flag->second) != profile)
        throw ReleaseSettingError(
            name, "profile=exact: explicit value re-enables " + name);
      if (real && ((name == "GUFO_LENIENT" && *real == "bounded") ||
                   (name != "GUFO_LENIENT" && real != profile)))
        throw ReleaseSettingError(
            name, "profile=exact: explicit value re-enables " + name);
    }
    if (suppressed && match && inputs.profile != "vanilla") {
      if (enabled("--no-behaviour") &&
          std::find_if(row.find("suppressed_by")->items().begin(),
                       row.find("suppressed_by")->items().end(),
                       [](const auto& flag) {
                         return flag.str() == "--no-behaviour";
                       }) != row.find("suppressed_by")->items().end()) {
        profile = name == "GUFO_EOG_MIN_PROB" ? std::optional<std::string>("-1")
                  : name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT"
                      ? std::nullopt
                      : std::optional<std::string>("0");
      } else {
        profile.reset();
      }
    }
    if (profile || (exact_off && match)) {
      setting.value = profile;
      setting.origin = suppressed ? "flag" : "profile:" + inputs.profile;
    }
    auto manifest = inputs.manifest_defaults.find(name);
    // Every engine reads --mmproj, so a packaged projector applies to every
    // model family (Flash Next packages and plain catalogue models).
    if ((inputs.profile != "vanilla" || name == "--mmproj") &&
        (match || name == "--mmproj") && !suppressed &&
        manifest != inputs.manifest_defaults.end() && !exact_off) {
      setting.value = manifest->second;
      setting.origin = "manifest";
    }
    if (real) {
      setting.value = real;
      setting.origin = "env";
      if (row.find("echo")->as_bool() && !row.find("secret")->as_bool())
        setting.requested = real;
      if (suppressed && !profile && inputs.profile != "vanilla" &&
          !row.find("secret")->as_bool())
        result.logs.push_back("env override: " + name + "=" + *real +
                              " (profile default says unset)");
      if (profile && real != profile && !row.find("secret")->as_bool())
        result.logs.push_back("env override: " + name + "=" + *real +
                              " (profile default says " + *profile + ")");
    }
    if (auto explicit_flag = inputs.explicit_flags.find(name);
        explicit_flag != inputs.explicit_flags.end()) {
      setting.value = explicit_flag->second;
      setting.origin = "flag";
    }
    if (path_control != path_controls.end() && match) {
      if (restricted_path) {
        setting.value.reset();
        setting.origin = "profile:" + inputs.profile;
      }
      if (auto selected =
              inputs.explicit_flags.find(path_control->second.first);
          selected != inputs.explicit_flags.end()) {
        setting.value = selected->second;
        setting.origin = "flag";
      }
      if (enabled(path_control->second.second)) {
        setting.value.reset();
        setting.origin = "flag";
      }
    }
    if (name == "--speculative" && enabled("--no-mtp")) {
      setting.value = "off";
      setting.origin = "flag";
    }
    if (name == "--prompt-lookup" && enabled("--no-prompt-lookup")) {
      setting.value = "0";
      setting.origin = "flag";
    }
    if ((name == "--cache-disk" &&
         (enabled("--no-cache-disk") || enabled("--no-cache") ||
          (exact_controls && setting.value))) ||
        (name == "--mmproj" && enabled("--no-vision")) ||
        (name == "--mtp-model" && enabled("--no-mtp"))) {
      setting.value.reset();
      setting.origin = "flag";
    }
    // Explicit CLI values only exist for flag rows. Env rows are handled
    // above; resolver tests may also simulate higher-precedence values.
    if (exact_off && match && setting.value &&
        ((name == "GUFO_LENIENT" && *setting.value == "bounded") ||
         (name != "GUFO_LENIENT" &&
          setting.value != Text(*row.find("values"), "exact"))))
      throw ReleaseSettingError(
          name, "profile=exact: explicit value re-enables " + name);
    if (inputs.profile == "exact" && name == "--seed" && setting.value) {
      std::int64_t seed = -1;
      const auto parsed =
          std::from_chars(setting.value->data(),
                          setting.value->data() + setting.value->size(), seed);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != setting.value->data() + setting.value->size() ||
          seed < 0)
        throw ReleaseSettingError("--seed",
                                  "profile=exact: --seed must be 0 or more");
    }
    if (exact_controls && match && real &&
        row.member_str("class") == "INTERNAL" &&
        !row.find("secret")->as_bool() && !row.find("exact_off")->as_bool())
      result.logs.push_back("profile=exact: " + name +
                            " is set (internal control)");
    if (setting.value) {
      try {
        ValidateValue(row, *setting.value);
      } catch (const std::invalid_argument& error) {
        if (Has(name, {"--think-marker-bias", "--loop-breaker"}) ||
            (match && inputs.profile != "vanilla" &&
             inputs.model_family == "qwen3.8-flash-next"))
          throw;
        result.logs.push_back("warning: " + std::string(error.what()));
      }
    }
    if (row.member_str("kind") == "flag" &&
        row.member_str("value_rule") == "uint" && setting.value) {
      std::uint64_t number = 0;
      const auto parsed = std::from_chars(
          setting.value->data(), setting.value->data() + setting.value->size(),
          number);
      if (parsed.ec == std::errc{} &&
          parsed.ptr == setting.value->data() + setting.value->size())
        setting.value = std::to_string(number);
    }
    if (name == "--cache-disk" && setting.value && setting.value->empty())
      setting.value.reset();
    // Exact is byte-exact only when every prefill launch starts at a multiple
    // of 4 (the sparse-attention query group). Forward size 4096 guarantees it.
    if (exact_controls && match && name == "GUFO_PREFILL_FORWARD_TOKENS" &&
        setting.value &&
        std::strtol(setting.value->c_str(), nullptr, 10) != 4096)
      throw ReleaseSettingError(
          name, "profile=exact: incompatible " + name + "=" + *setting.value +
                    "; exact requires 4096 (a multiple of 4: exact is "
                    "byte-exact only for prefill starts aligned to 4)");
    if (exact_controls && match &&
        ((name == "--sessions" &&
          setting.value != std::optional<std::string>("1")) ||
         (name == "--no-cache" &&
          setting.value != std::optional<std::string>("1"))))
      throw ReleaseSettingError(name, "profile=exact: incompatible " + name);
    if (match && !real && !inputs.explicit_flags.contains(name) &&
        ((suppressed && !setting.value) || (exact_off && !setting.value)))
      setting.skipped_reason = exact_off ? "suppressed by profile=exact"
                                         : "suppressed by explicit flag";
    settings.emplace(name, std::move(setting));
  }
  const auto environment_value =
      [&](const char* name) -> std::optional<std::string> {
    std::optional<std::string> value;
    if (inputs.real_environment)
      value = CopyEnvironmentValue(inputs.real_environment(name));
    else if (const auto found = inputs.environment.find(name);
             found != inputs.environment.end())
      value = found->second;
    return value && !value->empty() ? value : std::nullopt;
  };
  // These two environment fallbacks are read by serve before parsing flags.
  // Mirror that order here so the settings contract reports the same endpoint.
  if (auto& host = settings.at("--host"); host.origin == "built-in") {
    auto value = environment_value("HOST");
    if (!value)
      value = environment_value("GUFO_HOST");
    if (value) {
      host.value = value;
      host.origin = "env";
      host.requested = value;
    }
  }
  if (auto& port = settings.at("--port"); port.origin == "built-in") {
    auto value = environment_value("PORT");
    if (!value)
      value = environment_value("GUFO_PORT");
    if (value) {
      int parsed_port = 0;
      const auto parsed = std::from_chars(
          value->data(), value->data() + value->size(), parsed_port);
      if (parsed.ec == std::errc{} &&
          parsed.ptr == value->data() + value->size()) {
        port.value = std::to_string(parsed_port);
        port.origin = "env";
        port.requested = value;
      }
    }
  }
  // BehaviourOptions reads these environment names at request time. Resolve
  // the CLI controls into those same rows, then inject their effective values.
  if (const auto flag = inputs.explicit_flags.find("--think-marker-bias");
      flag != inputs.explicit_flags.end()) {
    if (enabled("--no-behaviour"))
      throw ReleaseSettingError(
          "--think-marker-bias",
          "--think-marker-bias conflicts with --no-behaviour");
    if (exact_controls && std::stod(flag->second) != 0.0)
      throw ReleaseSettingError(
          "--think-marker-bias",
          "profile=exact: --think-marker-bias must be zero");
    auto& bias = settings.at("GUFO_THINK_MARKER_BIAS");
    bias.value = flag->second;
    bias.origin = "flag";
    bias.requested = flag->second;
    bias.skipped_reason.clear();
    auto& effort_map = settings.at("GUFO_THINK_MARKER_BIAS_BY_EFFORT");
    effort_map.value.reset();
    effort_map.origin = "flag";
    effort_map.skipped_reason = "overridden by --think-marker-bias";
  }
  if (const auto flag = inputs.explicit_flags.find("--loop-breaker");
      flag != inputs.explicit_flags.end()) {
    if (enabled("--no-behaviour"))
      throw ReleaseSettingError("--loop-breaker",
                                "--loop-breaker conflicts with --no-behaviour");
    if (exact_controls && flag->second == "on")
      throw ReleaseSettingError("--loop-breaker",
                                "profile=exact: --loop-breaker must be off");
    auto& breaker = settings.at("GUFO_LOOP_BREAKER");
    breaker.value = flag->second == "on" ? "1" : "0";
    breaker.origin = "flag";
    breaker.requested = flag->second;
    breaker.skipped_reason.clear();
  }
  // The public flag rows show the setting read by the request path, including
  // vanilla mode and direct environment overrides.
  if (settings.contains("--loop-breaker")) {
    const auto& effective = settings.at("GUFO_LOOP_BREAKER");
    auto& flag = settings.at("--loop-breaker");
    flag.value =
        effective.value == std::optional<std::string>("1") ? "on" : "off";
    flag.origin = effective.origin;
    flag.skipped_reason = effective.skipped_reason;
  }
  if (settings.contains("--think-marker-bias")) {
    const auto& effective = settings.at("GUFO_THINK_MARKER_BIAS");
    auto& flag = settings.at("--think-marker-bias");
    flag.value = effective.value.value_or("0");
    flag.origin = effective.origin;
    flag.skipped_reason = effective.skipped_reason;
  }
  if ((inputs.profile == "vanilla" || exact_controls) &&
      inputs.model_family == "qwen3.8-flash-next" &&
      !(inputs.profile == "vanilla" &&
        inputs.explicit_flags.contains("--mtp-model") &&
        inputs.explicit_flags.contains("--speculative") &&
        inputs.explicit_flags.at("--speculative") == "mtp")) {
    if (inputs.explicit_flags.contains("--mtp-model"))
      throw ReleaseSettingError(
          "--mtp-model",
          "profile=vanilla: --mtp-model requires explicit --speculative mtp");
    if (inputs.profile == "vanilla" &&
        inputs.explicit_flags.contains("--speculative") &&
        inputs.explicit_flags.at("--speculative") == "mtp")
      throw ReleaseSettingError(
          "--speculative",
          "profile=vanilla: --speculative mtp requires explicit --mtp-model");
    settings.at("--mtp-model").value.reset();
    if (!inputs.explicit_flags.contains("--speculative") || exact_controls) {
      settings.at("--speculative").value = "off";
      settings.at("--speculative").origin = "profile:" + inputs.profile;
    }
  }
  // The dense decode twin with more than one session needs the batched twin
  // (decode gate passed 2026-10-14, w4c4-onpass-twin-nonexact-batch). Set it
  // here, as the subagents profile does, so every profile sees one rule. An
  // explicit env or flag value keeps its place and the requirement below then
  // decides, as before.
  std::optional<ReleaseSetting> auto_batch_before;
  if (inputs.model_family == "qwen3.8-flash-next" &&
      inputs.profile != "vanilla" && !exact_controls) {
    auto& batch = settings.at("GUFO_DENSE_TWIN_BATCH");
    const auto& twin = settings.at("GUFO_DENSE_TWIN_MODEL");
    const auto& sessions_value = settings.at("--sessions").value;
    std::uint64_t sessions = 1;
    if (sessions_value)
      std::from_chars(sessions_value->data(),
                      sessions_value->data() + sessions_value->size(),
                      sessions);
    if (batch.origin == "built-in" && sessions > 1 && twin.value &&
        !twin.value->empty() && twin.skipped_reason.empty()) {
      auto_batch_before = batch;
      batch.value = "1";
      batch.origin = "profile:" + inputs.profile;
    }
  }
  // Defaults can depend on defaults removed later in sorted row order.
  // Deletion is monotone, so at most one pass per row reaches a fixed point.
  bool changed;
  do {
    changed = false;
    for (const auto& row : rows) {
      auto& setting = settings.at(row.member_str("name"));
      if (!setting.skipped_reason.empty())
        continue;
      const bool active =
          setting.value &&
          !(row.member_str("value_rule") == "bool" && *setting.value == "0") &&
          !(setting.name == "--speculative" && *setting.value == "off") &&
          !(setting.name == "--reasoning-effort" && *setting.value == "auto");
      const bool candidate = active && setting.origin != "built-in";
      if (!candidate && row.member_str("class") != "DEFAULT")
        continue;
      if (setting.value && !active)
        continue;
      for (const auto& req : row.find("requires")->items()) {
        if (setting.name == "GUFO_LENIENT" && setting.value &&
            *setting.value != "bounded")
          continue;
        bool met = true;
        if (req.str() == "think!=off")
          met =
              settings.at("--think").value != std::optional<std::string>("off");
        else if (req.str() == "speculative==mtp")
          met = settings.at("--speculative").value ==
                std::optional<std::string>("mtp");
        else if (req.str() == "sessions==1 or GUFO_DENSE_TWIN_BATCH==1")
          met = settings.at("--sessions").value ==
                    std::optional<std::string>("1") ||
                settings.at("GUFO_DENSE_TWIN_BATCH").value ==
                    std::optional<std::string>("1");
        else if (req.str().starts_with("artefact:")) {
          const std::map<std::string, std::string> artefacts = {
              {"artefact:mtp", "--mtp-model"},
              {"artefact:mmproj", "--mmproj"},
              {"artefact:twin", "GUFO_DENSE_TWIN_MODEL"},
              {"artefact:draft_head", "GUFO_DRAFT_OUT_MODEL"}};
          const auto& value = settings.at(artefacts.at(req.str())).value;
          met = value && !value->empty();
          // An MTP head inside the model file meets the MTP artefact for the
          // other families. Flash-Next keeps requiring its own sidecar.
          if (req.str() == "artefact:mtp" && inputs.model_has_mtp &&
              inputs.model_family != "qwen3.8-flash-next" &&
              inputs.model_family != "gemma4")
            met = true;
          // The MTP artefact is required only by the MTP backend.
          if (setting.name == "--speculative" && setting.value &&
              *setting.value != "mtp")
            met = true;
        }
        if (met)
          continue;
        const auto path = path_controls.find(setting.name);
        const bool positive_path =
            path != path_controls.end() &&
            inputs.explicit_flags.contains(path->second.first);
        if (positive_path && Matches(row, inputs) &&
            inputs.model_family == "qwen3.8-flash-next")
          throw ReleaseSettingError(
              path->second.first, path->second.first + ": needs " + req.str());
        if (!Matches(row, inputs) || inputs.profile == "vanilla" ||
            (inputs.model_family != "qwen3.8-flash-next" &&
             !SharedWithGemma4(row, inputs))) {
          if (candidate)
            result.logs.push_back("warning: " + setting.name + ": needs " +
                                  req.str());
          continue;
        }
        const bool explicit_value =
            setting.origin == "env" ||
            (setting.origin == "flag" &&
             inputs.explicit_flags.contains(setting.name));
        if (explicit_value)
          throw ReleaseSettingError(setting.name,
                                    setting.name + ": needs " + req.str());
        setting.skipped_reason = "skipped: needs " + req.str();
        result.logs.push_back(setting.name + ": " + setting.skipped_reason);
        setting.value.reset();
        changed = true;
        break;
      }
    }
  } while (changed);
  // The batched setting only follows a twin that stays on.
  if (auto_batch_before &&
      settings.at("GUFO_DENSE_TWIN_MODEL").value.value_or("").empty())
    settings.at("GUFO_DENSE_TWIN_BATCH") = *auto_batch_before;
  for (const auto& row : rows) {
    auto& setting = settings.at(row.member_str("name"));
    if (Matches(row, inputs) && inputs.profile != "vanilla" &&
        setting.name == "GUFO_DENSE_TWIN_MAX_N" && setting.value &&
        settings.at("GUFO_DENSE_TWIN_BATCH").value ==
            std::optional<std::string>("1") &&
        settings.at("--sessions").value != std::optional<std::string>("1") &&
        models::qwen38_flash_next::TwinMaxNValue(setting.value->c_str()) < 8)
      throw ReleaseSettingError(
          "GUFO_DENSE_TWIN_MAX_N",
          "GUFO_DENSE_TWIN_MAX_N must be at least 8 for batched twin");
    if (row.member_str("kind") == "flag" && setting.value)
      result.effective_flags[setting.name] = *setting.value;
    if (row.member_str("kind") == "env" &&
        !path_controls.contains(setting.name) && setting.value &&
        (setting.origin == "manifest" ||
         setting.origin.starts_with("profile:") || setting.origin == "flag") &&
        (inputs.profile != "vanilla" || setting.origin == "flag"))
      result.injections.push_back(setting);
    if (setting.name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT" &&
        setting.origin == "flag" && !setting.value)
      result.injections.push_back(setting);
    result.settings.push_back(setting);
  }
  const auto path = [&](const std::string& key) {
    return settings.at(key).value.value_or("");
  };
  // The vision projector is read by every engine; the other typed artefacts
  // belong to Flash Next.
  result.artefacts.mmproj = path("--mmproj");
  result.artefacts.vision_disabled = enabled("--no-vision");
  if (inputs.model_family == "qwen3.8-flash-next") {
    result.artefacts.present = true;
    result.artefacts.twin = path("GUFO_DENSE_TWIN_MODEL");
    result.artefacts.draft_head = path("GUFO_DRAFT_OUT_MODEL");
    result.artefacts.mtp = path("--mtp-model");
  } else if (inputs.model_family == "gemma4") {
    // The MTP head is the one typed artefact Gemma 4 shares with Flash Next.
    result.artefacts.mtp = path("--mtp-model");
  }
  if (inputs.profile == "subagents" &&
      inputs.model_family == "qwen3.8-flash-next" &&
      !enabled("--allow-unmeasured-memory")) {
    const auto sessions = std::stoull(result.effective_flags.at("--sessions"));
    auto context = std::stoull(result.effective_flags.at("--context"));
    if (context == 0) {
      if (!inputs.native_context)
        throw ReleaseSettingError(
            "--context",
            "profile=subagents: native context requires a model header or "
            "--allow-unmeasured-memory");
      context = *inputs.native_context;
    }
    if (context > 524288 ||
        sessions > 524288 / std::max<std::uint64_t>(context, 1))
      throw ReleaseSettingError(
          "--context",
          "profile=subagents: sessions x context exceeds measured 524288 "
          "tokens; lower context or pass --allow-unmeasured-memory");
  }
  if (enabled("--no-cache") && inputs.explicit_flags.contains("--cache-disk") &&
      !inputs.explicit_flags.at("--cache-disk").empty())
    result.logs.push_back("cache: --no-cache overrides --cache-disk");
  std::size_t env = 0, skipped = 0;
  for (const auto& setting : result.settings) {
    env += setting.origin == "env";
    skipped += !setting.skipped_reason.empty();
  }
  result.logs.insert(result.logs.begin(),
                     "profile=" + inputs.profile + ": injected " +
                         std::to_string(result.injections.size()) +
                         " defaults, " + std::to_string(env) + " from env, " +
                         std::to_string(skipped) + " skipped (details below)");
  if (inputs.profile == "exact" && result.turned_off.empty())
    result.logs.push_back(
        "profile=exact: byte-exact controls apply to Qwen3.8 Flash Next only; "
        "seed=" +
        result.effective_flags.at("--seed") + " applied");
  if (inputs.profile == "exact" && !result.turned_off.empty()) {
    std::string line = "profile=exact: turned off: ";
    for (const auto& name : result.turned_off) {
      if (line.back() != ' ')
        line += ", ";
      line += name;
    }
    line += "; pinned: seed=" + result.effective_flags.at("--seed") +
            ", forward=" +
            settings.at("GUFO_PREFILL_FORWARD_TOKENS").value.value_or("unset") +
            ", sessions=1, cache=off";
    result.logs.push_back(line);
  }
  return result;
}
void ApplyInjections(const ReleaseResolution& resolved) {
  // Snapshot again using generated literal reads. Never overwrite a value set
  // between resolution and application (startup is single-threaded).
  for (const auto& injection : resolved.injections) {
    const auto& rows = ReleaseSwitchTable();
    const auto row =
        std::find_if(rows.begin(), rows.end(), [&](const auto& item) {
          return item.member_str("name") == injection.name;
        });
    if (row == rows.end() || row->member_str("kind") != "env" ||
        row->member_str("class") == "REMOVED" ||
        (!injection.value &&
         !(injection.name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT" &&
           injection.origin == "flag")))
      throw std::invalid_argument("invalid injection: " + injection.name);
    if (injection.value)
      ValidateValue(*row, *injection.value);
  }
  const auto real = SnapshotReleaseEnvironment();
  // Empty CRT values must not reach strict readers, even in vanilla or another
  // family where this resolver injects no defaults.
  for (const auto& [name, value] : real) {
    if (!value || !value->empty())
      continue;
#ifdef _WIN32
    const int status = _putenv_s(name.c_str(), "");
#else
    const int status = ::unsetenv(name.c_str());
#endif
    if (status != 0)
      throw std::runtime_error("cannot unset empty environment value: " + name);
  }
  for (const auto& injection : resolved.injections) {
    const auto found = real.find(injection.name);
    if (injection.origin != "flag" && found != real.end() && found->second &&
        !found->second->empty())
      continue;
#ifdef _WIN32
    const int status =
        _putenv_s(injection.name.c_str(),
                  injection.value ? injection.value->c_str() : "");
#else
    const int status = injection.value ? ::setenv(injection.name.c_str(),
                                                  injection.value->c_str(), 1)
                                       : ::unsetenv(injection.name.c_str());
#endif
    if (status != 0)
      throw std::runtime_error("cannot inject " + injection.name);
  }
}

ReleaseStartup PrepareReleaseSettings(
    std::span<const char* const> args,
    const std::map<std::string, std::string>& manifest_defaults,
    bool inspect_only) {
  ReleaseStartup startup;
  try {
    ReleaseInputs inputs;
    inputs.manifest_defaults = manifest_defaults;
    std::string modality = "llm";
    bool help = false;
    const std::set<std::string> server_options = {"--host",
                                                  "--port",
                                                  "--sessions",
                                                  "--max-connections",
                                                  "--max-request-bytes",
                                                  "--log-level",
                                                  "--verbose",
                                                  "--debug-hooks",
                                                  "--max-score-tokens"};
    // Prefix observation skips registry flags without validating their inline
    // values, before any GGUF-header work. ArgParser validates later; using it
    // here would change malformed-value/modality/help preflight ordering.
    for (std::size_t i = 0; i < args.size(); ++i) {
      const std::string arg(args[i]);
      const auto equals = arg.find('=');
      const auto name = arg.substr(0, equals);
      const json::Value* option = nullptr;
      for (const auto& row : ReleaseSwitchTable()) {
        bool named = row.member_str("name") == name;
        for (const auto& alias : row.find("aliases")->items())
          named |= alias.str() == name;
        if (row.member_str("kind") == "flag" && named &&
            (row.find("secret")->as_bool() ||
             server_options.contains(row.member_str("name")) ||
             row.member_str("name") == "--profile" ||
             row.member_str("name").starts_with("--no-") ||
             Has(row.member_str("name"),
                 {"--exact", "--print-effective-settings",
                  "--allow-unmeasured-memory", "--npu-prefill"}))) {
          option = &row;
          break;
        }
      }
      if (option) {
        if (option->find("takes_value")->as_bool() &&
            equals == std::string::npos)
          ++i;
        continue;
      }
      if (Has(arg, {"llm", "video", "image", "tts", "asr"}))
        modality = arg;
      help = Has(arg, {"help", "--help", "-h"});
      break;
    }
    if (help)
      return startup;
    // Registry provenance observation, not command validation: unknown engine
    // flags are ignored token by token and raw help ends header preflight.
    // ArgParser has no non-validating registry observer with these semantics;
    // keep this bounded pass until it does, rather than add a second parser.
    for (std::size_t i = 0; i < args.size(); ++i) {
      std::string arg(args[i]);
      if (arg == "help" || arg == "--help" || arg == "-h") {
        help = true;
        break;
      }
      if (Has(arg, {"llm", "video", "image", "tts", "asr"}))
        continue;
      const auto equals = arg.find('=');
      const auto name = arg.substr(0, equals);
      if (Has(name, {"--twin-model", "--draft-head-model", "--verify-model"})) {
        std::string value = "1";
        if (name != "--verify-model") {
          if (equals != std::string::npos)
            value = arg.substr(equals + 1);
          else if (++i < args.size())
            value = args[i];
          else
            throw std::invalid_argument("missing value for " + name);
        } else if (equals != std::string::npos)
          throw std::invalid_argument(name + " does not take a value");
        inputs.explicit_flags[name] = value;
        continue;
      }
      const json::Value* option = nullptr;
      for (const auto& row : ReleaseSwitchTable()) {
        if (row.member_str("kind") != "flag")
          continue;
        bool match = row.member_str("name") == name;
        for (const auto& alias : row.find("aliases")->items())
          match |= alias.str() == name;
        if (match) {
          option = &row;
          break;
        }
      }
      if (!option)
        continue;  // The existing parser reports unknown options.
      std::string value = "1";
      if (option->find("takes_value")->as_bool()) {
        if (equals != std::string::npos)
          value = arg.substr(equals + 1);
        else if (++i < args.size())
          value = args[i];
        else
          throw std::invalid_argument("missing value for " + name);
      } else if (equals != std::string::npos)
        throw std::invalid_argument(name + " does not take a value");
      inputs.explicit_flags[option->member_str("name")] = value;
    }
    if (help)
      return startup;
    if (modality != "llm") {
      if ((inputs.explicit_flags.contains("--npu-prefill") &&
           inputs.explicit_flags.at("--npu-prefill") != "off"))
        throw std::invalid_argument(
            "experimental ON requires serve llm; effective=off");
      return startup;
    }
    startup.active = true;
    if (auto found = inputs.explicit_flags.find("--profile");
        found != inputs.explicit_flags.end())
      inputs.profile = found->second;
    if (inputs.profile == "stock")
      inputs.profile = "vanilla";
    if (inputs.explicit_flags.contains("--exact")) {
      if (inputs.explicit_flags.contains("--profile") &&
          inputs.profile != "exact")
        throw std::invalid_argument("--exact conflicts with --profile");
      inputs.profile = "exact";
    }
    startup.print_effective =
        inputs.explicit_flags.contains("--print-effective-settings");
    for (const auto* name : {"--model", "--manifest"})
      if (inputs.explicit_flags.contains(name) &&
          inputs.explicit_flags.at(name).empty())
        throw std::invalid_argument(std::string(name) +
                                    " requires a nonempty path");
    if (auto found = inputs.explicit_flags.find("--model");
        found != inputs.explicit_flags.end())
      inputs.model_path = std::filesystem::u8path(found->second);
    // Deny experimental ON before even opening the model header.
    (void)ResolveReleaseSettings(inputs);
    inputs.environment = SnapshotReleaseEnvironment();
    const auto selector = [&](const std::string& env) -> std::string {
      if (auto found = inputs.environment.find(env);
          found != inputs.environment.end() && found->second &&
          !found->second->empty())
        return *found->second;
      return {};
    };
    const auto selection = bubo::cli::ResolvePreparedDefault(
        inputs.model_path,
        inputs.explicit_flags.contains("--manifest")
            ? std::filesystem::u8path(inputs.explicit_flags.at("--manifest"))
            : std::filesystem::path{},
        bubo::cli::EnvironmentUtf8("BUBO_MANIFEST"),
        bubo::cli::EnvironmentUtf8("LOCALAPPDATA"));
    const bool verify = inputs.explicit_flags.contains("--verify-model");
    startup.model_missing =
        (!selection.model.empty() &&
         !std::filesystem::is_regular_file(selection.model) &&
         !std::filesystem::is_directory(selection.model)) ||
        (!selection.manifest.empty() &&
         !std::filesystem::is_regular_file(selection.manifest));
    startup.package =
        LoadModelPackage(inputs.model_path, inputs.profile, verify, &std::cerr,
                         selection.source == "standard_location" &&
                                 !std::filesystem::exists(selection.manifest)
                             ? std::filesystem::path{}
                             : selection.manifest);
    inputs.manifest_present = !startup.package.manifest_path.empty();
    if (!startup.package.files.empty() &&
        (inputs.model_path.empty() ||
         std::filesystem::is_directory(inputs.model_path)))
      inputs.model_path = startup.package.files.front().path;
    inputs.manifest_defaults = startup.package.defaults;
    for (const auto& [env, key] : std::map<std::string, std::string>{
             {"BUBO_MTP_MODEL", "--mtp-model"}, {"BUBO_MMPROJ", "--mmproj"}}) {
      const auto value = selector(env);
      if (!value.empty())
        inputs.environment[key] = value;
    }
    if (!inputs.model_path.empty()) {
      std::string error;
      const auto reader =
          core::GgufReader::OpenHeaderFile(inputs.model_path, &error);
      if (!reader)
        throw std::invalid_argument("cannot read model header: " + error);
      const auto architecture =
          reader->GetMetadataString("general.architecture");
      if (!architecture)
        throw std::invalid_argument("model header has no general.architecture");
      inputs.architecture = std::string(*architecture);
      inputs.model_family =
          std::string(bubo::prepare::ModelFamilyForArchitecture(*architecture));
      inputs.model_has_mtp =
          reader->GetMetadataUint32(inputs.architecture +
                                    ".nextn_predict_layers")
              .value_or(0) > 0;
      inputs.native_context =
          reader->GetMetadataUint64(inputs.architecture + ".context_length");
      if (!inputs.native_context)
        if (auto value = reader->GetMetadataUint32(inputs.architecture +
                                                   ".context_length"))
          inputs.native_context = *value;
    } else if (startup.print_effective)
      inputs.model_family = "qwen3.8-flash-next";
    startup.resolved = ResolveReleaseSettings(inputs);
    startup.resolved.model_selection = bubo::cli::SelectionJson(selection);
    // For example a plain catalogue model whose projector is not prepared.
    for (const auto& note : startup.package.notes)
      if (note.starts_with("mtp:") ||
          (startup.resolved.artefacts.mmproj.empty() &&
           !startup.resolved.artefacts.vision_disabled))
        startup.resolved.logs.push_back(note);
    std::map<std::string, std::string> selected;
    for (const auto& setting : startup.resolved.settings)
      if (setting.value &&
          (setting.origin == "flag" || setting.origin == "env"))
        selected[setting.name] = *setting.value;
    CheckPackageOverrides(startup.package, selected, verify, &std::cerr);
    if (verify)
      startup.resolved.logs.push_back(
          "verify-model: checked " +
          std::to_string(startup.package.hashed_files) + " files, " +
          std::to_string(startup.package.hashed_bytes) +
          " bytes (including distinct overrides)");
    if (!startup.package.manifest_path.empty()) {
      const auto manifest_path = startup.package.manifest_path.u8string();
      startup.resolved.logs.push_back(
          "manifest: " +
          std::string(manifest_path.begin(), manifest_path.end()) +
          "; size + pinned header/tensor table checked; payload hashes are "
          "opt-in");
    }
    if (const auto cache =
            startup.resolved.effective_flags.find("--cache-disk");
        !inspect_only && cache != startup.resolved.effective_flags.end() &&
        !cache->second.empty()) {
      const auto& artefacts = startup.resolved.artefacts;
      const std::array<std::filesystem::path, 4> paths{
          std::filesystem::u8path(artefacts.mtp),
          std::filesystem::u8path(artefacts.mmproj),
          std::filesystem::u8path(artefacts.twin),
          std::filesystem::u8path(artefacts.draft_head)};
      EnsureDiskCacheIdentity(startup.package, paths, &std::cerr);
    }
    if (!inspect_only)
      ApplyInjections(startup.resolved);
  } catch (const std::exception& error) {
    startup.error = error.what();
    if (const auto* setting = dynamic_cast<const ReleaseSettingError*>(&error))
      startup.error_switch = setting->switch_name;
    startup.model_missing =
        startup.model_missing ||
        dynamic_cast<const ModelMissingError*>(&error) != nullptr;
    std::replace_if(
        startup.error.begin(), startup.error.end(),
        [](unsigned char c) { return c < 32 || c == 127; }, ' ');
  }
  return startup;
}
std::optional<std::uint64_t> ResolveDefaultSessionContext(
    ReleaseResolution& resolved, std::uint64_t sessions,
    const json::Value& memory_estimate, bool allow_single_session_adaptation) {
  const bool adapt_one_session = allow_single_session_adaptation && sessions == 1;
  const auto setting =
      std::find_if(resolved.settings.begin(), resolved.settings.end(),
                   [](const auto& row) { return row.name == "--context"; });
  if ((sessions <= 1 && !adapt_one_session) ||
      setting == resolved.settings.end() || setting->origin == "flag")
    return std::nullopt;
  if (adapt_one_session) {
    const auto* recommended = memory_estimate.find("recommended");
    if (recommended && recommended->is_bool() && recommended->as_bool())
      return std::nullopt;
  }
  const auto* recommendation = memory_estimate.find("recommended_context");
  const auto* rows =
      recommendation ? recommendation->find("by_sessions") : nullptr;
  if (rows && rows->is_array())
    for (const auto& row : rows->items()) {
      if (row.member_size("sessions") != sessions)
        continue;
      auto context = row.member_size("context_per_session");
      const auto* fits = row.find("fits");
      const auto* recommended = row.find("recommended");
      if (!fits || !fits->is_bool() || !fits->as_bool() || context == 0 ||
          (recommended && !recommended->as_bool()) ||
          context > std::numeric_limits<std::uint32_t>::max()) {
        if (adapt_one_session)
          return std::nullopt;
        else
          throw std::invalid_argument(
              "--context: recommended_context for --sessions " +
              std::to_string(sessions) +
              " does not fit: " + row.member_str("reason"));
      }
      if (adapt_one_session)
        context =
            std::min(context, memory_estimate.member_size("context_tokens"));
      setting->value = std::to_string(context);
      setting->origin = "recommended_context";
      resolved.effective_flags["--context"] = *setting->value;
      resolved.context_resolution_reason =
          adapt_one_session ? "Using " + std::to_string(context) +
                                     " tokens per session to leave room for "
                                     "the model and its cache."
                               : "Resolved " + std::to_string(context) +
                                     " tokens per session from CLI "
                                     "recommended_context.by_sessions for " +
                                     std::to_string(sessions) +
                                     " sessions (no --context override).";
      resolved.logs.push_back("--context: " +
                              resolved.context_resolution_reason);
      return context;
    }
  const auto reason = memory_estimate.member_str("recommended_context_reason");
  if (adapt_one_session)
    return std::nullopt;
  throw bubo::cli::ControlError(
      static_cast<int>(bubo::cli::MachineExit::Unsupported),
      "--context: CLI recommended_context is unavailable for --sessions " +
          std::to_string(sessions) +
          (reason.empty() ? "; supply an explicit --context." : ": " + reason));
}

void PrefillReleaseFlags(const ReleaseResolution& resolved,
                         const cli::ArgParser& parser) {
  for (const auto& setting : resolved.settings) {
    if (!setting.value || setting.origin == "flag" ||
        setting.origin == "built-in")
      continue;
    const auto* option = parser.FindOption(setting.name);
    if (!option)
      continue;
    if (option->is_flag && *setting.value == "0")
      continue;
    std::string error;
    if (!option->parse_fn(setting.name, *setting.value, &error))
      throw std::invalid_argument(error);
  }
  // Suppression flags set false through the existing inverse flag sinks.
}
json::Value ReleaseOrigins(const ReleaseResolution& resolved) {
  auto origins = json::Value::object();
  for (const auto& setting : resolved.settings) {
    if (setting.name == "GUFO_THINK_MARKER_ENTROPY_GATE" && !setting.value)
      continue;
    origins[setting.name] = setting.origin;
  }
  return origins;
}
json::Value ReleaseSettingsJson(const ReleaseResolution& resolved,
                                std::span<const json::Value> catalog) {
  const auto setting_for = [&](std::string_view name) -> const ReleaseSetting* {
    const auto found =
        std::find_if(resolved.settings.begin(), resolved.settings.end(),
                     [&](const auto& setting) { return setting.name == name; });
    return found == resolved.settings.end() ? nullptr : &*found;
  };
  const auto source_for = [](const ReleaseSetting* setting,
                             bool inference_default = false) {
    if (!setting || setting->origin == "built-in" ||
        setting->origin == "recommended_context")
      return std::string("default");
    if (setting->origin == "env")
      return std::string("env");
    if (setting->origin == "flag")
      return std::string(inference_default ? "flag" : "user");
    return std::string("profile");
  };
  // Text environment variables can contain paths, prompt text, or credentials.
  // Only validated scalar environment controls are shown in machine output.
  const auto visible = [](const json::Value& row,
                          const ReleaseSetting* setting) {
    if (row.member_str("kind") != "env")
      return true;
    if (row.find("secret")->as_bool())
      return false;
    if (!row.find("echo")->as_bool())
      return false;
    // Static built-in wording contains no user paths or credentials. A null
    // value here means the engine default applies, not that it defaults to 0.
    if (setting && !setting->value && setting->origin == "built-in" &&
        setting->skipped_reason.empty())
      return true;
    if (!Has(row.member_str("value_rule"),
             {"bool", "min_prob", "prob", "lambda", "transition", "number",
              "twin_width", "uint", "int31", "readers", "forward"}))
      return false;
    if (setting && setting->value) {
      try {
        ValidateValue(row, *setting->value);
      } catch (const std::invalid_argument&) {
        return false;
      }
    }
    return true;
  };
  auto output = json::Value::object();
  output["profile"] = resolved.profile;
  output["model_selection"] = resolved.model_selection;
  const bool npu_on = resolved.runtime_npu_enabled.value_or(
      resolved.prefill_authorization.Allows(
          core::ExperimentalFeature::Prefill));
  const std::string npu_reason = resolved.runtime_npu_reason.empty()
                                     ? resolved.prefill_authorization.reason()
                                     : resolved.runtime_npu_reason;
  auto experimental = json::Value::object();
  experimental["npu_prefill"] = npu_on ? "on" : "off";
  experimental["npu_reason"] = npu_reason;
  experimental["npu_deny_reason"] =
      npu_on ? json::Value() : json::Value(npu_reason);
  output["experimental"] = std::move(experimental);
  output["architecture"] = resolved.architecture.empty()
                               ? json::Value()
                               : json::Value(resolved.architecture);
  output["native_context"] = resolved.native_context
                                 ? json::Value(static_cast<unsigned long long>(
                                       *resolved.native_context))
                                 : json::Value();
  auto table = json::Value::array();
  for (const auto& setting : resolved.settings) {
    if (setting.name == "GUFO_THINK_MARKER_ENTROPY_GATE" && !setting.value)
      continue;
    const auto found =
        std::find_if(catalog.begin(), catalog.end(), [&](const auto& item) {
          return item.member_str("name") == setting.name;
        });
    if (found == catalog.end() || !visible(*found, &setting))
      continue;
    auto row = json::Value::object();
    row["name"] = setting.name;
    row["value"] = found->find("secret")->as_bool() && setting.value
                       ? json::Value("<redacted>")
                   : setting.value ? json::Value(*setting.value)
                                   : json::Value();
    if (!setting.value && setting.origin == "built-in" &&
        setting.skipped_reason.empty())
      row["value"] = found->member_str("default_text");
    if (setting.name == "--npu-prefill")
      row["value"] = npu_on ? "on" : "off";
    else if (setting.name == "--host" && resolved.runtime_host)
      row["value"] = *resolved.runtime_host;
    else if (setting.name == "--port" && resolved.runtime_port)
      row["value"] = std::to_string(*resolved.runtime_port);
    row["origin"] = setting.origin;
    row["class"] = setting.release_class;
    if (!setting.skipped_reason.empty())
      row["skipped_reason"] = setting.skipped_reason;
    if (setting.name == "--npu-prefill" && !npu_on)
      row["skipped_reason"] = npu_reason;
    table.push_back(std::move(row));
  }
  output["settings"] = std::move(table);
  output["origin"] = ReleaseOrigins(resolved);
  auto off = json::Value::array();
  for (const auto& name : resolved.turned_off)
    off.push_back(name);
  output["turned_off"] = std::move(off);
  if (resolved.profile == "exact" && !resolved.turned_off.empty()) {
    auto pins = json::Value::object();
    pins["seed"] = resolved.effective_flags.at("--seed");
    for (const auto& setting : resolved.settings)
      if (setting.name == "GUFO_PREFILL_FORWARD_TOKENS")
        pins["forward"] =
            setting.value ? json::Value(*setting.value) : json::Value();
    pins["sessions"] = "1";
    pins["disk"] = "off";
    pins["cache"] = "off";
    output["pinned"] = std::move(pins);
  }
  auto engine = json::Value::object();
  for (const auto* name : {"--sessions", "--context", "--speculative"}) {
    const auto found = resolved.effective_flags.find(name);
    engine[std::string(name).substr(2)] =
        found == resolved.effective_flags.end() ? json::Value()
                                                : json::Value(found->second);
  }
  engine["twin_bound"] = resolved.artefacts.twin;
  // True when more than one session decodes with the batched 4-bit twin.
  {
    const auto* batch = setting_for("GUFO_DENSE_TWIN_BATCH");
    const auto sessions = resolved.effective_flags.find("--sessions");
    engine["twin_batched"] = !resolved.artefacts.twin.empty() && batch &&
                             batch->value == std::optional<std::string>("1") &&
                             sessions != resolved.effective_flags.end() &&
                             sessions->second != "1";
  }
  engine["draft_head_bound"] = resolved.artefacts.draft_head;
  engine["mtp_bound"] = resolved.artefacts.mtp;
  engine["mmproj_bound"] = resolved.artefacts.mmproj;
  engine["vision_disabled"] = resolved.artefacts.vision_disabled;
  engine["typed_artefacts"] = resolved.artefacts.present;
  if (resolved.native_context && engine.member_str("context") == "0")
    engine["context"] = std::to_string(*resolved.native_context);
  if (resolved.runtime_context)
    engine["context"] = std::to_string(*resolved.runtime_context);
  output["engine"] = std::move(engine);

  auto capabilities = json::Value::object();
  capabilities["profiles"] = bubo::cli::SettingsProfiles();
#ifdef GUFO_NPU
  capabilities["npu_prefill"] = true;
#else
  capabilities["npu_prefill"] = false;
#endif
  output["capabilities"] = std::move(capabilities);

  const auto requested = [](const ReleaseSetting* setting) -> json::Value {
    return setting && setting->requested ? json::Value(*setting->requested)
                                         : json::Value();
  };
  const auto field = [](json::Value value, std::string source) {
    auto entry = json::Value::object();
    entry["value"] = std::move(value);
    entry["source"] = std::move(source);
    return entry;
  };
  const auto has_override = [&](std::string_view name) {
    const auto* setting = setting_for(name);
    return setting && setting->value && setting->origin != "built-in";
  };
  const auto number = [&](std::string_view name, double fallback) {
    const auto* setting = setting_for(name);
    return setting && setting->value ? std::stod(*setting->value) : fallback;
  };
  sampling::SamplingConfig configured;
  sampling::SamplingOverrides supplied;
  supplied.temperature = has_override("--temperature");
  supplied.top_k = has_override("--top-k");
  supplied.top_p = has_override("--top-p");
  supplied.min_p = has_override("--min-p");
  supplied.seed = has_override("--seed");
  supplied.repeat_penalty = has_override("--repeat-penalty");
  supplied.presence_penalty = has_override("--presence-penalty");
  if (supplied.temperature)
    configured.temperature = static_cast<float>(number("--temperature", 0));
  if (supplied.top_k)
    configured.top_k = static_cast<std::int32_t>(number("--top-k", 0));
  if (supplied.top_p)
    configured.top_p = static_cast<float>(number("--top-p", 1));
  if (supplied.min_p)
    configured.min_p = static_cast<float>(number("--min-p", 0));
  if (supplied.seed)
    configured.seed = static_cast<std::int64_t>(number("--seed", -1));
  if (supplied.repeat_penalty)
    configured.repeat_penalty =
        static_cast<float>(number("--repeat-penalty", 1));
  if (supplied.presence_penalty)
    configured.presence_penalty =
        static_cast<float>(number("--presence-penalty", 0));
  const auto* think_setting = setting_for("--think");
  const auto* effort_setting = setting_for("--reasoning-effort");
  std::string think =
      think_setting && think_setting->value ? *think_setting->value : "auto";
  const std::string effort =
      effort_setting && effort_setting->value ? *effort_setting->value : "auto";
  std::string think_source = source_for(think_setting, true);
  if (think == "auto" && effort != "auto") {
    think = "on";
    think_source = source_for(effort_setting, true);
  }
  const std::optional<bool> thinking = think == "on" ? std::optional<bool>(true)
                                       : think == "off"
                                           ? std::optional<bool>(false)
                                           : std::nullopt;
  const auto model = resolved.architecture == "qwen4exp" ||
                             resolved.model_family == "qwen3.8-flash-next"
                         ? sampling::TextModelPreset::kQwen38
                     : resolved.architecture == "deepseek4" ||
                             resolved.model_family == "deepseek-v4-flash"
                         ? sampling::TextModelPreset::kDeepSeekV4Flash
                         : sampling::TextModelPreset::kUnspecified;
  const auto* production_setting = setting_for("GUFO_PRODUCTION_SAMPLING");
  const bool production = production_setting && production_setting->value &&
                          *production_setting->value == "1";
  const auto native = sampling::ResolveTextSampling(model, thinking, configured,
                                                    supplied, false);
  const auto effective =
      resolved.runtime_sampling.value_or(sampling::ResolveTextSampling(
          model, thinking, configured, supplied, production));
  const auto sampling_source = [&](std::string_view name, bool changed) {
    if (has_override(name))
      return source_for(setting_for(name), true);
    if (production && changed)
      return source_for(production_setting, true);
    return std::string("default");
  };
  auto inference = json::Value::object();
  inference["temperature"] =
      field(effective.temperature,
            sampling_source("--temperature",
                            effective.temperature != native.temperature));
  inference["top_p"] =
      field(effective.top_p,
            sampling_source("--top-p", effective.top_p != native.top_p));
  inference["top_k"] =
      field(effective.top_k,
            sampling_source("--top-k", effective.top_k != native.top_k));
  inference["min_p"] =
      field(effective.min_p,
            sampling_source("--min-p", effective.min_p != native.min_p));
  inference["repeat_penalty"] =
      field(effective.repeat_penalty,
            sampling_source("--repeat-penalty",
                            effective.repeat_penalty != native.repeat_penalty));
  inference["presence_penalty"] =
      field(effective.presence_penalty,
            sampling_source("--presence-penalty", effective.presence_penalty !=
                                                      native.presence_penalty));
  inference["seed"] =
      field(effective.seed, source_for(setting_for("--seed"), true));
  inference["reasoning_effort"] =
      field(effort, source_for(effort_setting, true));
  inference["think"] = field(think, think_source);
  const auto* max_tokens = setting_for("--max-tokens");
  const auto token_limit =
      max_tokens && max_tokens->value ? std::stoll(*max_tokens->value) : -1;
  inference["max_tokens"] =
      field(token_limit < 0 ? 0 : token_limit, source_for(max_tokens, true));
  output["inference_defaults"] = std::move(inference);

  const auto* bias_setting = setting_for("GUFO_THINK_MARKER_BIAS");
  const auto* map_setting = setting_for("GUFO_THINK_MARKER_BIAS_BY_EFFORT");
  sampling::BehaviourConfig behaviour;
  behaviour.marker_bias = number("GUFO_THINK_MARKER_BIAS", 0);
  if (map_setting && map_setting->value)
    behaviour.marker_bias_by_effort = *map_setting->value;
  auto bias = json::Value::object();
  bias["value"] = behaviour.marker_bias;
  bias["source"] = source_for(bias_setting);
  bias["requested"] = requested(bias_setting);
  auto effort_biases = json::Value::object();
  for (const auto* name : {"low", "medium", "high", "xhigh"}) {
    const auto mapped = [&] {
      std::string_view map(behaviour.marker_bias_by_effort);
      const std::string_view effective_name =
          std::string_view(name) == "high" ? "xhigh" : name;
      while (!map.empty()) {
        const auto comma = map.find(',');
        auto entry = map.substr(0, comma);
        const auto colon = entry.find(':');
        if (colon != std::string_view::npos) {
          auto label = entry.substr(0, colon);
          while (!label.empty() &&
                 std::isspace(static_cast<unsigned char>(label.front())))
            label.remove_prefix(1);
          while (!label.empty() &&
                 std::isspace(static_cast<unsigned char>(label.back())))
            label.remove_suffix(1);
          if (label == effective_name)
            return true;
        }
        if (comma == std::string_view::npos)
          break;
        map.remove_prefix(comma + 1);
      }
      return false;
    }();
    effort_biases[name] =
        field(ResolveMarkerBias(behaviour, name),
              source_for(mapped ? map_setting : bias_setting));
  }
  bias["by_effort"] = std::move(effort_biases);
  output["think_marker_bias"] = std::move(bias);
  const auto* breaker = setting_for("GUFO_LOOP_BREAKER");
  auto loop = json::Value::object();
  loop["value"] = breaker && breaker->value && *breaker->value == "1";
  loop["source"] = source_for(breaker);
  loop["requested"] = requested(breaker);
  output["loop_breaker"] = std::move(loop);
  auto entropy = json::Value::object();
  entropy["available"] = false;
  output["entropy_gate"] = std::move(entropy);
  output["think_marker_entropy_gate"] = output["entropy_gate"];

  const auto* npu_setting = setting_for("--npu-prefill");
  auto npu = json::Value::object();
  npu["requested"] = npu_setting && npu_setting->requested
                         ? json::Value(*npu_setting->requested)
                         : json::Value("off");
  npu["effective"] = npu_on ? "on" : "off";
  npu["source"] =
      npu_reason == "GUFO_ENABLE_NPU=OFF" ? "default" : source_for(npu_setting);
  npu["reason"] = npu_reason;
  npu["forced"] = !npu_on && (!npu_setting || !npu_setting->requested ||
                              *npu_setting->requested != "off");
  auto npu_allowed = json::Value::array();
  npu_allowed.push_back("off");
  if (resolved.prefill_authorization.Allows(core::ExperimentalFeature::Prefill))
    npu_allowed.push_back("on");
  npu["allowed"] = std::move(npu_allowed);
  output["npu_prefill"] = npu;

  auto switches = json::Value::array();
  auto environment_overrides = json::Value::array();
  const auto& inference_defaults = *output.find("inference_defaults");
  for (const auto& catalog_row : catalog) {
    const auto name = catalog_row.member_str("name");
    const auto* setting = setting_for(name);
    const bool show_value =
        visible(catalog_row, setting) && !catalog_row.find("secret")->as_bool();
    auto effective_value = setting && setting->value && show_value
                               ? json::Value(*setting->value)
                               : json::Value();
    if (show_value && setting && !setting->value &&
        setting->origin == "built-in" && setting->skipped_reason.empty())
      effective_value = catalog_row.member_str("default_text");
    if (catalog_row.member_str("kind") == "flag" &&
        !catalog_row.find("takes_value")->as_bool())
      effective_value = setting && setting->value && *setting->value != "0";
    std::string effective_source = source_for(setting);
    if (name == "--profile") {
      effective_value = resolved.profile;
      const auto* exact_alias = setting_for("--exact");
      if (exact_alias && exact_alias->origin == "flag" &&
          exact_alias->value == std::optional<std::string>("1"))
        effective_source = "user";
    }
    if (name.starts_with("--")) {
      std::string key = name.substr(2);
      std::replace(key.begin(), key.end(), '-', '_');
      if (const auto* default_entry = inference_defaults.find(key)) {
        effective_value = *default_entry->find("value");
        effective_source = default_entry->member_str("source");
        if (effective_source == "flag")
          effective_source = "user";
      }
    }
    if (name == "--npu-prefill")
      effective_value = npu_on ? "on" : "off";
    else if (name == "--host" && resolved.runtime_host)
      effective_value = *resolved.runtime_host;
    else if (name == "--port" && resolved.runtime_port)
      effective_value = *resolved.runtime_port;
    else if (name == "--context" && resolved.runtime_context)
      effective_value =
          static_cast<unsigned long long>(*resolved.runtime_context);
    else if (name == "--context" && resolved.native_context && setting &&
             setting->value == std::optional<std::string>("0"))
      effective_value =
          static_cast<unsigned long long>(*resolved.native_context);
    if (name.find("_NPU_") == 4 && !npu_on)
      effective_value = json::Value();
    auto switch_row = json::Value::object();
    switch_row["name"] = name;
    if (const auto* label = catalog_row.find("label"))
      switch_row["label"] = *label;
    switch_row["kind"] = catalog_row.member_str("kind");
    switch_row["group"] = *catalog_row.find("group");
    switch_row["audience"] = *catalog_row.find("audience");
    switch_row["allowed"] = *catalog_row.find("allowed");
    if (name == "--npu-prefill" && !resolved.prefill_authorization.Allows(
                                       core::ExperimentalFeature::Prefill)) {
      auto allowed = json::Value::object();
      allowed["type"] = "enum";
      allowed["values"] = *npu.find("allowed");
      switch_row["allowed"] = std::move(allowed);
    }
    switch_row["experimental"] = *catalog_row.find("experimental");
    switch_row["restart_required"] = *catalog_row.find("restart_required");
    switch_row["gui_control"] = *catalog_row.find("gui_control");
    switch_row["requested"] = show_value ? requested(setting) : json::Value();
    switch_row["effective"] = effective_value;
    switch_row["source"] =
        name == "--npu-prefill" ? npu.member_str("source") : effective_source;
    std::string reason =
        name == "--context" && !resolved.context_resolution_reason.empty()
            ? resolved.context_resolution_reason
        : setting ? setting->skipped_reason
                  : "";
    if (name == "--npu-prefill" && !npu_on)
      reason = npu_reason;
    else if (name.find("_NPU_") == 4 && !npu_on)
      reason = npu_reason;
    else if (resolved.profile == "exact" && setting &&
             std::find(resolved.turned_off.begin(), resolved.turned_off.end(),
                       name) != resolved.turned_off.end())
      reason = "profile=exact forces this switch off";
    else if (!show_value && setting && setting->value)
      reason = "Value omitted from settings output.";
    else if (name == "--port" && resolved.runtime_port && setting &&
             setting->value == std::optional<std::string>("0"))
      reason = "The server assigned the listening port.";
    switch_row["reason"] = reason.empty() ? json::Value() : json::Value(reason);
    switches.push_back(std::move(switch_row));
    if (setting && setting->origin == "env" && show_value && setting->value) {
      auto override_row = json::Value::object();
      override_row["name"] = name;
      override_row["value"] = effective_value;
      environment_overrides.push_back(std::move(override_row));
    }
  }
  output["effective_switches"] = std::move(switches);
  output["environment_overrides"] = std::move(environment_overrides);
  return output;
}

std::string RenderReleaseInclude(std::span<const json::Value> source_rows) {
  const auto rows = NormalizeSwitchTable(source_rows);
  ValidateSwitchTable(rows);
  std::string out =
      "// Generated by tools/release/gen_switches.py. Do not edit.\n#if "
      "defined(GUFO_RELEASE_ROWS)\n";
  for (const auto& row : rows)
    out += "json::parse(R\"switch(" + row.dump() + ")switch\"),\n";
  out += "#elif defined(GUFO_RELEASE_SNAPSHOT)\n";
  for (const auto& row : rows) {
    if (row.member_str("kind") != "env" || row.member_str("class") == "REMOVED")
      continue;
    const auto name = json::Value(row.member_str("name")).dump();
    out += "environment[" + name +
           "] = CopyEnvironmentValue(std::get"
           "env(" +
           name + "));\n";
  }
  out += "#elif defined(GUFO_RELEASE_CATALOG)\n";
  for (const auto& row : rows) {
    if (row.member_str("kind") == "env" &&
        row.member_str("class") != "REMOVED" && row.find("echo")->as_bool())
      out += json::Value(row.member_str("name")).dump() + ",\n";
  }
  out += "#elif defined(GUFO_RELEASE_NORMALIZATION)\n";
  out +=
      "json::parse(R\"switch(" + SwitchNormalization().dump() + ")switch\")\n";
  return out + "#endif\n";
}
std::string RenderArtefactBindingTable() {
  std::string out =
      "| Profile | Artefact | Source | Binding |\n| --- | --- | --- | --- |\n";
  for (const auto& row : kArtefactBindingCases) {
    const std::string result =
        row.expected == BindingExpectation::bound      ? "Bound"
        : row.expected == BindingExpectation::disabled ? "Disabled"
                                                       : "Rejected (exit 2)";
    out += "| " + std::string(row.profile) + " | " + std::string(row.artefact) +
           " | " + std::string(row.source) + " | " + result + " |\n";
  }
  return out;
}
std::string RenderReleaseDocs(std::span<const json::Value> source_rows) {
  const auto rows = NormalizeSwitchTable(source_rows);
  ValidateSwitchTable(rows);
  std::string out =
      "# Release switches\n\nGenerated from "
      "`src/cli/serve/release_switches.json`.\n\nProfile defaults apply to "
      "Qwen3.8 Flash Next. A Gemma 4 package with an MTP file shares the MTP "
      "rows (`--speculative`, `--mtp-model`).\nOther model families keep their "
      "existing "
      "defaults except for the exact seed and NPU profile policy.\nExplicit "
      "flags "
      "and real "
      "environment values take "
      "precedence.\nEmpty environment values count as unset and produce a "
      "warning.\n\n"
      "The optional `label` field supplies short display wording.\n"
      "The `group`, `audience`, `experimental`, `restart_required`, and "
      "`allowed` fields are typed registry metadata.\n"
      "Each row owns its group and allowed values; suggested values document "
      "reader recommendations, not enforced limits.\n"
      "The default audience is debug, with no GUI control.\n"
      "Switches default to non-experimental and require restart, except "
      "removed switches.\n"
      "A null profile value means no value is injected: the engine built-in in "
      "default_text applies. The exact profile inherits a non-null default "
      "profile value where shown.\n"
      "Changing a setting marked Restart requires a new server start.\n"
      "The `debug` audience appears in the collapsed Debug section of the "
      "Advanced panel.\n\n";
  const auto join = [](const json::Value& array) {
    std::string result;
    for (const auto& value : array.items()) {
      if (!result.empty())
        result += ", ";
      result += value.str();
    }
    return result;
  };
  const auto table = [&](std::initializer_list<std::string_view> classes) {
    out +=
        "| Name | Group | Audience | What it does | Profile defaults | "
        "Built-in | Allowed | Restart | Experimental | Platform and "
        "hardware | Requires | Suppressed by | When to change |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |\n";
    for (const auto& row : rows) {
      if (!Has(row.member_str("class"), classes))
        continue;
      const auto scope = row.member_str("platform") + "; " +
                         row.member_str("hardware_scope") + "; " +
                         row.member_str("model_family");
      out +=
          "| `" + row.member_str("name") + "` | " + row.member_str("group") +
          " | " + row.member_str("audience") + " | " +
          EscapeCell(row.member_str("purpose")) + " | " +
          EscapeCell(ProfileDefaultsText(row)) + " | " +
          EscapeCell(
              Text(row, "built_in").value_or(row.member_str("default_text"))) +
          " | " + EscapeCell(AllowedText(*row.find("allowed"))) + " | " +
          (row.find("restart_required")->as_bool() ? "yes" : "no") + " | " +
          (row.find("experimental")->as_bool() ? "yes" : "no") + " | " + scope +
          " | " + EscapeCell(join(*row.find("requires"))) + " | " +
          EscapeCell(join(*row.find("suppressed_by"))) + " | " +
          EscapeCell(row.member_str("note")) + " |\n";
    }
    out += "\n";
  };
  out += "## Options and defaults\n\n";
  table({"DEFAULT", "OPTION"});
  out +=
      "## Profiles\n"
      "\n"
      "| Profile | Effect |\n"
      "| --- | --- |\n"
      "| default | Use fast upstream practice for one session and medium "
      "reasoning effort. Output is not byte-exact. |\n"
      "| subagents | Use four sessions with 131072 context tokens each and a 4 "
      "GiB RAM cache. |\n"
      "| exact | Use reproducible plain autoregressive decoding with one "
      "session and no prompt cache. |\n"
      "| vanilla | Keep upstream defaults. Inject no environment values or "
      "flag "
      "defaults. |\n"
      "\n"
      "Subagents verifies up to 524288 total context tokens.\n"
      "Use `--allow-unmeasured-memory` above this measured limit.\n"
      "Four concurrent deep sessions can reach the host RAM guard near 50K "
      "tokens per session.\n"
      "The default profile follows upstream gufo practice: fast output that is "
      "not byte-exact.\n"
      "Exact is a reproducibility preset.\n"
      "Output is byte-identical across runs of the same build and seed.\n"
      "Output is identical to plain autoregressive decoding of that build with "
      "the same settings.\n"
      "Output is not identical across builds or to other engines.\n"
      "Exact disables speculation, behaviour policies, lenient acceptance, the "
      "dense twin, the draft head, and RAM and disk prompt caches.\n"
      "Exact uses one session, forward size 4096, and seed 1 unless a seed is "
      "supplied.\n"
      "Exact refuses values that enable disabled settings or change its "
      "session count or prefill geometry.\n"
      "It runs at about half the decode speed, and every request prefills its "
      "full prompt.\n"
      "Sidecar files (MTP, vision, dense twin, draft head) are set with their "
      "own options.\n"
      "Reference: `bubo serve --profile vanilla --speculative off --no-cache "
      "--sessions 1 --seed 1 --think on --reasoning-effort medium`.\n"
      "Use `--print-effective-settings` to inspect settings without loading "
      "tensors or starting a server.\n"
      "\n";
  out += "## Diagnostics\n\n";
  table({"DEBUG", "INTERNAL"});
  out +=
      "## Removed and other models\n\nOther model switches remain available "
      "for their model "
      "families.\n\n";
  table({"REMOVED", "OTHER_MODEL"});
  out +=
      "## Engine-managed queue cap\n\n"
      "| Name | What it does | Default | When to change | Platform and "
      "hardware |\n"
      "| --- | --- | --- | --- | --- |\n"
      "| `GPU_MAX_HW_QUEUES` | Set the hardware queue cap. | engine sets | The "
      "engine manages this queue cap; a real environment value wins. | any; "
      "generic; any |\n\n";
  out.pop_back();  // Keep one final newline, matching the Python renderer.
  return out;
}
}  // namespace gufo::server
