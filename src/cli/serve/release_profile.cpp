#include "src/cli/serve/release_profile.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/cli/serve/behaviour_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/tuning.hpp"
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
  if (profile == "stock")
    return {};
  const auto& values = *row.find("values");
  if (auto value = Text(values, profile))
    return value;
  return profile == "exact" ? Text(values, "default") : std::nullopt;
}
void ValidateValue(const json::Value& row, const std::string& value) {
  const auto rule = row.member_str("value_rule");
  const auto fail = [&] {
    throw std::invalid_argument(row.member_str("name") + ": invalid value '" +
                                value + "'");
  };
  const auto name = row.member_str("name");
  if (row.member_str("kind") == "flag") {
    if (name == "--speculative" &&
        !Has(value, {"mtp", "off", "dspark", "dflash2"}))
      fail();
    if (name == "--mtp-policy" && !Has(value, {"survival", "length"}))
      fail();
    if (name == "--mtp-draft-vocab" && !Has(value, {"auto", "latin", "full"}))
      fail();
    if (name == "--think" && !Has(value, {"auto", "on", "off"}))
      fail();
    if (name == "--reasoning-effort" &&
        !Has(value,
             {"auto", "minimal", "low", "medium", "high", "xhigh", "max"}))
      fail();
  }
  if (rule == "uint" || rule == "int31") {
    std::uint64_t number = 0;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
      fail();
    if (rule == "int31" && number > 2147483647)
      fail();
    if ((name == "--sessions" || name == "--max-request-bytes" ||
         name == "--max-connections") &&
        number == 0)
      fail();
    if (name == "GUFO_DISK_EVICT_SKIP_TOKENS" && number > 1000000)
      fail();
    if ((name == "GUFO_DISK_IDLE_MS" || name == "GUFO_DISK_DEFER_MS") &&
        number > 600000)
      fail();
    return;
  }
  if (rule == "forward") {
    // Match PrefillChunkTokens' base-10 prefix grammar before checking its
    // range.
    const long number = std::strtol(value.c_str(), nullptr, 10);
    if (number < 64 || number > 16384)
      fail();
    return;
  }
  if (rule == "text" || rule == "twin_width")
    return;
  if (rule == "lenient") {
    // The reader enables only bounded; every other literal means off.
    return;
  }
  if (rule == "bool") {
    if (value != "0" && value != "1")
      fail();
    return;
  }
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
  std::size_t end = 0;
  double number = 0;
  try {
    number = std::stod(value, &end);
  } catch (...) {
    fail();
  }
  if (end != value.size() || !std::isfinite(number))
    fail();
  if (rule == "number")
    return;
  if (rule == "min_prob") {
    if (number != -1 && (number < 0 || number > 1))
      fail();
  } else if (rule == "prob") {
    if (number < 0 || number > 1)
      fail();
  } else if (rule == "lambda") {
    if (number <= 0 || number > 2)
      fail();
  } else {
    if (number != std::floor(number))
      fail();
    if (rule == "transition" && (number < 0 || number > 2))
      fail();
    if (rule == "twin_width" && (number < 1 || number > 8))
      fail();
    if (rule == "readers" && (number < 1 || number > 16))
      fail();
    if (rule == "uint" && (number < 0 || number > 9007199254740991.0))
      fail();
  }
}
bool Matches(const json::Value& row, const ReleaseInputs& inputs) {
  const auto family = row.member_str("model_family");
  if (family != "any" && family != inputs.model_family)
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
}  // namespace

const std::vector<json::Value>& ReleaseSwitchTable() {
  static const std::vector<json::Value> rows = {
#define GUFO_RELEASE_ROWS
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_ROWS
  };
  return rows;
}
ReleaseEnvironment SnapshotReleaseEnvironment() {
  ReleaseEnvironment environment;
#define GUFO_RELEASE_SNAPSHOT
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_SNAPSHOT
  return environment;
}
void ValidateSwitchTable(std::span<const json::Value> rows) {
  const std::set<std::string> fields = {
      "name",           "kind",          "class",    "model_family",
      "platform",       "requires",      "values",   "exact_off",
      "inject_with",    "suppressed_by", "purpose",  "default_text",
      "hardware_scope", "note",          "built_in", "value_rule",
      "aliases",        "takes_value",   "echo"};
  std::set<std::string> names;
  for (const auto& row : rows) {
    const auto name = row.member_str("name");
    const auto fail = [&] {
      throw std::invalid_argument("invalid switch row: " + name);
    };
    if (!row.is_object() || row.size() != fields.size() || name.empty() ||
        !names.insert(name).second)
      fail();
    for (const auto& key : fields)
      if (!row.contains(key))
        fail();
    if (!Has(row.member_str("kind"), {"env", "flag"}) ||
        !Has(row.member_str("class"), {"DEFAULT", "OPTION", "DEBUG", "INTERNAL",
                                       "REMOVED", "OTHER_MODEL"}) ||
        !Has(row.member_str("model_family"),
             {"any", "qwen3.8-flash-next", "qwen3.8-27b", "deepseek-v4-flash",
              "minimax-h3", "qwen-image-2.1", "qwen3-tts", "qwen3-asr"}) ||
        !Has(row.member_str("platform"), {"any", "windows"}) ||
        !Has(row.member_str("hardware_scope"),
             {"generic", "strix-halo-tuned"}) ||
        !Has(row.member_str("value_rule"),
             {"text", "bool", "min_prob", "prob", "lambda", "transition",
              "number", "effort_map", "lenient", "forward", "twin_width",
              "uint", "int31", "readers", "tuning"}))
      fail();
    if (Has(name, {"GUFO_EVAL_API_KEY", "PATH"}) && row.find("echo")->as_bool())
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
    if (!row.find("echo")->is_bool() || !row.find("exact_off")->is_bool() ||
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

ReleaseResolution ResolveReleaseSettings(const ReleaseInputs& inputs) {
  if (!Has(inputs.profile, {"default", "subagents", "exact", "stock"}))
    throw std::invalid_argument("unknown profile: " + inputs.profile);
  const auto& rows = ReleaseSwitchTable();
  ValidateSwitchTable(rows);
  ReleaseResolution result;
  result.profile = inputs.profile;
  result.architecture = inputs.architecture;
  result.native_context = inputs.native_context;
  std::map<std::string, ReleaseSetting> settings;
  const auto enabled = [&](const std::string& name) {
    auto it = inputs.explicit_flags.find(name);
    return it != inputs.explicit_flags.end() && it->second != "0";
  };
  // Resolve every candidate before evaluating dependencies; row order cannot
  // affect sessions, backend, or twin-batch requirements.
  for (const auto& row : rows) {
    const auto name = row.member_str("name");
    ReleaseSetting setting{
        name, Text(row, "built_in"), "built-in", row.member_str("class"), {}};
    const bool is_env = row.member_str("kind") == "env";
    std::optional<std::string> real;
    if ((is_env || inputs.environment.contains(name)) &&
        setting.release_class != "REMOVED") {
      if (inputs.real_environment)
        real = CopyEnvironmentValue(inputs.real_environment(name.c_str()));
      else if (auto found = inputs.environment.find(name);
               found != inputs.environment.end())
        real = found->second;
      if (real && real->empty()) {
        result.logs.push_back("warning: empty environment value: " + name +
                              " (treated as unset)");
        real.reset();
      }
    }
    const bool match = Matches(row, inputs);
    auto profile = match ? ProfileValue(row, inputs.profile) : std::nullopt;
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
    if (inputs.profile == "exact" && match &&
        row.find("exact_off")->as_bool()) {
      result.turned_off.push_back(name);
      profile = Text(*row.find("values"), "exact");
      if (const auto flag = inputs.explicit_flags.find(name);
          flag != inputs.explicit_flags.end() &&
          row.member_str("kind") == "flag" &&
          std::optional<std::string>(flag->second) != profile)
        throw std::invalid_argument(
            "profile=exact: explicit value re-enables " + name);
      if (real && ((name == "GUFO_LENIENT" && *real == "bounded") ||
                   (name != "GUFO_LENIENT" && real != profile)))
        throw std::invalid_argument(
            "profile=exact: explicit value re-enables " + name);
    }
    if (suppressed && match && inputs.profile != "stock") {
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
    if (profile || (inputs.profile == "exact" && match &&
                    row.find("exact_off")->as_bool())) {
      setting.value = profile;
      setting.origin = suppressed ? "flag" : "profile:" + inputs.profile;
    }
    auto manifest = inputs.manifest_defaults.find(name);
    if (inputs.profile != "stock" && match && !suppressed &&
        manifest != inputs.manifest_defaults.end() &&
        !(inputs.profile == "exact" && row.find("exact_off")->as_bool())) {
      setting.value = manifest->second;
      setting.origin = "manifest";
    }
    if (real) {
      setting.value = real;
      setting.origin = "env";
      if (suppressed && !profile && inputs.profile != "stock")
        result.logs.push_back("env override: " + name + "=" + *real +
                              " (profile default says unset)");
      if (profile && real != profile)
        result.logs.push_back("env override: " + name + "=" + *real +
                              " (profile default says " + *profile + ")");
    }
    if (auto explicit_flag = inputs.explicit_flags.find(name);
        explicit_flag != inputs.explicit_flags.end()) {
      setting.value = explicit_flag->second;
      setting.origin = "flag";
    }
    if (name == "--speculative" && enabled("--no-mtp")) {
      setting.value = "off";
      setting.origin = "flag";
    }
    if (name == "--prompt-lookup" && enabled("--no-prompt-lookup")) {
      setting.value = "0";
      setting.origin = "flag";
    }
    if ((name == "--cache-disk" && enabled("--no-cache-disk")) ||
        (name == "--mmproj" && enabled("--no-vision")) ||
        (name == "--mtp-model" && enabled("--no-mtp"))) {
      setting.value.reset();
      setting.origin = "flag";
    }
    // Explicit CLI values only exist for flag rows. Env rows are handled
    // above; resolver tests may also simulate higher-precedence values.
    if (inputs.profile == "exact" && match &&
        row.find("exact_off")->as_bool() && setting.value &&
        ((name == "GUFO_LENIENT" && *setting.value == "bounded") ||
         (name != "GUFO_LENIENT" &&
          setting.value != Text(*row.find("values"), "exact"))))
      throw std::invalid_argument("profile=exact: explicit value re-enables " +
                                  name);
    if (setting.value) {
      try {
        ValidateValue(row, *setting.value);
      } catch (const std::invalid_argument& error) {
        if (match && inputs.profile != "stock" &&
            inputs.model_family == "qwen3.8-flash-next")
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
    if (inputs.profile == "exact" && match &&
        ((name == "--sessions" &&
          setting.value != std::optional<std::string>("1")) ||
         (name == "--cache-disk" && setting.value) ||
         (name == "GUFO_PREFILL_FORWARD_TOKENS" && setting.value &&
          std::strtol(setting.value->c_str(), nullptr, 10) != 4096) ||
         (name == "--no-cache" &&
          setting.value != std::optional<std::string>("1"))))
      throw std::invalid_argument("profile=exact: incompatible " + name);
    if (match && !real && !inputs.explicit_flags.contains(name) &&
        ((suppressed && !setting.value) ||
         (inputs.profile == "exact" && row.find("exact_off")->as_bool() &&
          !setting.value)))
      setting.skipped_reason =
          inputs.profile == "exact" && row.find("exact_off")->as_bool()
              ? "suppressed by profile=exact"
              : "suppressed by explicit flag";
    settings.emplace(name, std::move(setting));
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
          // The MTP artefact is required only by the MTP backend.
          if (setting.name == "--speculative" && setting.value &&
              *setting.value != "mtp")
            met = true;
        }
        if (met)
          continue;
        if (!Matches(row, inputs) || inputs.profile == "stock" ||
            inputs.model_family != "qwen3.8-flash-next") {
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
          throw std::invalid_argument(setting.name + ": needs " + req.str());
        setting.skipped_reason = "skipped: needs " + req.str();
        result.logs.push_back(setting.name + ": " + setting.skipped_reason);
        setting.value.reset();
        changed = true;
        break;
      }
    }
  } while (changed);
  for (const auto& row : rows) {
    auto& setting = settings.at(row.member_str("name"));
    if (Matches(row, inputs) && inputs.profile != "stock" &&
        setting.name == "GUFO_DENSE_TWIN_MAX_N" && setting.value &&
        settings.at("GUFO_DENSE_TWIN_BATCH").value ==
            std::optional<std::string>("1") &&
        settings.at("--sessions").value != std::optional<std::string>("1") &&
        models::qwen38_flash_next::TwinMaxNValue(setting.value->c_str()) < 8)
      throw std::invalid_argument(
          "GUFO_DENSE_TWIN_MAX_N must be at least 8 for batched twin");
    if (row.member_str("kind") == "flag" && setting.value)
      result.effective_flags[setting.name] = *setting.value;
    if (row.member_str("kind") == "env" && setting.value &&
        (setting.origin == "manifest" ||
         setting.origin.starts_with("profile:") || setting.origin == "flag") &&
        inputs.profile != "stock")
      result.injections.push_back(setting);
    result.settings.push_back(setting);
  }
  if (inputs.profile == "subagents" &&
      inputs.model_family == "qwen3.8-flash-next" &&
      !enabled("--allow-unmeasured-memory")) {
    const auto sessions = std::stoull(result.effective_flags.at("--sessions"));
    auto context = std::stoull(result.effective_flags.at("--context"));
    if (context == 0) {
      if (!inputs.native_context)
        throw std::invalid_argument(
            "profile=subagents: native context requires a model header or "
            "--allow-unmeasured-memory");
      context = *inputs.native_context;
    }
    if (context > 524288 ||
        sessions > 524288 / std::max<std::uint64_t>(context, 1))
      throw std::invalid_argument(
          "profile=subagents: sessions x context exceeds measured 524288 "
          "tokens; lower context or pass --allow-unmeasured-memory");
  }
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
        "profile=exact: turned off: none; pinned: none (model family does not "
        "match)");
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
        row->member_str("class") == "REMOVED" || !injection.value)
      throw std::invalid_argument("invalid injection: " + injection.name);
    ValidateValue(*row, *injection.value);
  }
  const auto real = SnapshotReleaseEnvironment();
  // Empty CRT values must not reach strict readers, even in stock or another
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
    if (found != real.end() && found->second && !found->second->empty())
      continue;
#ifdef _WIN32
    const int status =
        _putenv_s(injection.name.c_str(), injection.value->c_str());
#else
    const int status =
        ::setenv(injection.name.c_str(), injection.value->c_str(), 1);
#endif
    if (status != 0)
      throw std::runtime_error("cannot inject " + injection.name);
  }
}

ReleaseStartup PrepareReleaseSettings(std::span<const char* const> args) {
  ReleaseStartup startup;
  try {
    ReleaseInputs inputs;
    std::string modality = "llm";
    bool help = false;
    const std::set<std::string> server_options = {"--host",
                                                  "--port",
                                                  "--sessions",
                                                  "--max-connections",
                                                  "--max-request-bytes",
                                                  "--api-key",
                                                  "--log-level",
                                                  "--verbose",
                                                  "--debug-hooks",
                                                  "--max-score-tokens"};
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
            (server_options.contains(row.member_str("name")) ||
             row.member_str("name") == "--profile" ||
             row.member_str("name").starts_with("--no-") ||
             Has(row.member_str("name"),
                 {"--exact", "--print-effective-settings",
                  "--allow-unmeasured-memory"}))) {
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
    if (help || modality != "llm")
      return startup;
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
    if (help || modality != "llm")
      return startup;
    startup.active = true;
    if (auto found = inputs.explicit_flags.find("--profile");
        found != inputs.explicit_flags.end())
      inputs.profile = found->second;
    if (inputs.explicit_flags.contains("--exact")) {
      if (inputs.explicit_flags.contains("--profile") &&
          inputs.profile != "exact")
        throw std::invalid_argument("--exact conflicts with --profile");
      inputs.profile = "exact";
    }
    startup.print_effective =
        inputs.explicit_flags.contains("--print-effective-settings");
    if (auto found = inputs.explicit_flags.find("--model");
        found != inputs.explicit_flags.end())
      inputs.model_path = found->second;
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
      inputs.model_family = *architecture == "qwen4exp"
                                ? "qwen3.8-flash-next"
                                : std::string(*architecture);
      inputs.native_context =
          reader->GetMetadataUint64(inputs.architecture + ".context_length");
      if (!inputs.native_context)
        if (auto value = reader->GetMetadataUint32(inputs.architecture +
                                                   ".context_length"))
          inputs.native_context = *value;
    } else if (startup.print_effective)
      inputs.model_family = "qwen3.8-flash-next";
    inputs.environment = SnapshotReleaseEnvironment();
    startup.resolved = ResolveReleaseSettings(inputs);
    ApplyInjections(startup.resolved);
  } catch (const std::exception& error) {
    startup.error = error.what();
  }
  return startup;
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
  for (const auto& setting : resolved.settings)
    origins[setting.name] = setting.origin;
  return origins;
}
json::Value ReleaseSettingsJson(const ReleaseResolution& resolved) {
  auto output = json::Value::object();
  output["profile"] = resolved.profile;
  output["architecture"] = resolved.architecture.empty()
                               ? json::Value()
                               : json::Value(resolved.architecture);
  output["native_context"] = resolved.native_context
                                 ? json::Value(static_cast<unsigned long long>(
                                       *resolved.native_context))
                                 : json::Value();
  auto table = json::Value::array();
  for (const auto& setting : resolved.settings) {
    const auto& catalog = ReleaseSwitchTable();
    const auto found =
        std::find_if(catalog.begin(), catalog.end(), [&](const auto& item) {
          return item.member_str("name") == setting.name;
        });
    if (found->member_str("kind") == "env" && !found->find("echo")->as_bool())
      continue;
    auto row = json::Value::object();
    row["name"] = setting.name;
    row["value"] = setting.name == "--api-key" && setting.value
                       ? json::Value("<redacted>")
                   : setting.value ? json::Value(*setting.value)
                                   : json::Value();
    row["origin"] = setting.origin;
    row["class"] = setting.release_class;
    if (!setting.skipped_reason.empty())
      row["skipped_reason"] = setting.skipped_reason;
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
  for (const auto* name : {"twin_loaded", "draft_head_loaded", "mmproj_loaded"})
    engine[name] = "unknown until the manifest job";
  if (resolved.native_context && engine.member_str("context") == "0")
    engine["context"] = std::to_string(*resolved.native_context);
  output["engine"] = std::move(engine);
  return output;
}

std::string RenderReleaseInclude(std::span<const json::Value> rows) {
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
  return out + "#endif\n";
}
std::string RenderReleaseDocs(std::span<const json::Value> rows) {
  std::string out =
      "# Release switches\n\nGenerated from "
      "`src/cli/serve/release_switches.json`.\n\nProfile defaults apply to "
      "Qwen3.8 Flash Next.\nOther model families keep their existing "
      "defaults.\nExplicit flags and real environment values take "
      "precedence.\nEmpty environment values count as unset and produce a "
      "warning.\n\n";
  const auto table = [&](std::initializer_list<std::string_view> classes) {
    out +=
        "| Name | What it does | Default | When to change | Platform and "
        "hardware |\n| --- | --- | --- | --- | --- |\n";
    for (const auto& row : rows) {
      if (!Has(row.member_str("class"), classes))
        continue;
      const auto scope = row.member_str("platform") + "; " +
                         row.member_str("hardware_scope") + "; " +
                         row.member_str("model_family");
      out += "| `" + row.member_str("name") + "` | " +
             EscapeCell(row.member_str("purpose")) + " | " +
             EscapeCell(row.member_str("default_text")) + " | " +
             EscapeCell(row.member_str("note")) + " | " + scope + " |\n";
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
      "| stock | Keep upstream defaults. Inject no environment values or flag "
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
      "Output is identical to plain autoregressive decoding of that build.\n"
      "Output is not identical across builds or to other engines.\n"
      "Exact disables speculation, behaviour policies, lenient acceptance, the "
      "dense twin, the draft head, and RAM and disk prompt caches.\n"
      "Exact uses one session, forward size 4096, and seed 1 unless a seed is "
      "supplied.\n"
      "Exact refuses values that enable disabled settings or change its "
      "session count or prefill geometry.\n"
      "It runs at about half the decode speed, and every request prefills its "
      "full prompt.\n"
      "Manifest artefacts remain unavailable until the manifest job.\n"
      "Use `--print-effective-settings` to inspect settings without loading "
      "tensors or starting a server.\n"
      "\n";
  out += "## Diagnostics\n\n";
  table({"DEBUG", "INTERNAL"});
  out +=
      "## Removed\n\nOther model switches remain available for their model "
      "families.\n\n";
  table({"REMOVED", "OTHER_MODEL"});
  out.pop_back();  // Keep one final newline, matching the Python renderer.
  return out;
}
}  // namespace gufo::server
