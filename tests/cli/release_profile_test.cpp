#include "src/cli/serve/release_profile.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/model_families.hpp"
#include "src/cli/serve/debug_settings.hpp"
#include "src/cli/settings_support.hpp"

using namespace gufo::server;
namespace {
std::size_t passed = 0;
void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
  ++passed;
}
template<class F>
void Reject(F function, const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, message);
}
ReleaseInputs Inputs(const std::string& profile = "default") {
  ReleaseInputs inputs;
  inputs.model_family = "qwen3.8-flash-next";
  inputs.profile = profile;
  // Synthetic sidecar path: resolver tests never load tensors.
  if (profile != "exact" && profile != "vanilla")
    inputs.explicit_flags["--mtp-model"] = "mtp-test.gguf";
  return inputs;
}
const ReleaseSetting& Find(const ReleaseResolution& result,
                           const std::string& name) {
  for (const auto& setting : result.settings)
    if (setting.name == name)
      return setting;
  throw std::runtime_error("missing row " + name);
}
std::string Read(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::string text((std::istreambuf_iterator<char>(input)), {});
  text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
  return text;
}
void CheckSpec(const std::filesystem::path& path) {
  const auto spec = gufo::json::parse(Read(path));
  const auto result = ResolveReleaseSettings(Inputs());
  // Site/manifest keys intentionally have no release default.
  const std::set<std::string> site = {"GUFO_DENSE_TWIN_MODEL",
                                      "GUFO_DRAFT_OUT_MODEL",
                                      "--mtp-model",
                                      "--mmproj",
                                      "--model",
                                      "--port",
                                      "--served-model-name",
                                      "--host"};
  // Values equal to reader defaults are normalised to unset on both sides.
  const std::set<std::string> noops = {
      "GUFO_DENSE_TWIN_MAX_N", "GUFO_LENIENT_LAMBDA",
      "GUFO_LOOP_BREAKER_TRANSITION", "GUFO_DISK_INPASS",
      "GUFO_PREFILL_FORWARD_TOKENS"};
  for (const auto* section : {"env", "args", "env_defaults"}) {
    const auto* values = spec.find(section);
    if (!values)
      continue;
    for (const auto& [name, expected] : values->members()) {
      if (site.contains(name) || noops.contains(name))
        continue;
      const auto text = expected.is_bool() ? (expected.as_bool() ? "1" : "0")
                                           : expected.str();
      Expect(Find(result, name).value == std::optional<std::string>(text),
             "production parity: " + name);
    }
  }
  for (const auto& row : result.injections) {
    bool listed = false;
    for (const auto* section : {"env", "env_defaults"})
      if (const auto* values = spec.find(section))
        listed |= values->contains(row.name);
    Expect(listed || site.contains(row.name) || noops.contains(row.name),
           "production reverse parity: extra injected env " + row.name);
  }
  std::cout << "PASS: production comparison " << path.filename().string()
            << " (site/manifest keys excluded)\n";
}

void AuditCatalog() {
  const auto root = std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT);
  Expect(ReleaseInputs{}.profile == gufo::cli::kDefaultProfile,
         "resolver compile-time default follows the build brand");
  std::set<std::string> names, missing;
  for (const auto& row : ReleaseSwitchTable())
    names.insert(row.member_str("name"));
  const std::regex literal(R"audit("(GUFO_[A-Z0-9_]+)")audit");
  const std::regex getenv_literal(
      R"audit((?:std::getenv|\bgetenv)\s*\(\s*"([A-Za-z_][A-Za-z_0-9]*)")audit");
  const std::set<std::string> irrelevant = {"NO_COLOR"};
  const std::string gate = "GUFO_THINK_MARKER_ENTROPY_GATE";
  Expect(names.contains(gate) && DebugSettingsListsEnvironment(gate),
         "research entropy gate is registered and diagnosed");
  for (const char* profile : {"default", "subagents", "exact", "vanilla"}) {
    auto inputs = Inputs(profile);
    const auto unset = ResolveReleaseSettings(inputs);
    Expect(!Find(unset, gate).value &&
               std::ranges::none_of(unset.injections,
                                    [&](const auto& injection) {
                                      return injection.name == gate;
                                    }),
           "no profile injects the research entropy gate");
    const auto settings = ReleaseSettingsJson(unset);
    Expect(!ReleaseOrigins(unset).contains(gate) &&
               !settings.find("origin")->contains(gate) &&
               std::ranges::none_of(settings.find("settings")->items(),
                                    [&](const auto& row) {
                                      return row.member_str("name") == gate;
                                    }),
           "unset gate adds no effective release metadata");
    inputs.environment[gate] = "0.75";
    const auto configured = ResolveReleaseSettings(inputs);
    Expect(Find(configured, gate).value == std::optional<std::string>("0.75") &&
               ReleaseOrigins(configured).contains(gate) &&
               std::ranges::none_of(configured.injections,
                                    [&](const auto& injection) {
                                      return injection.name == gate;
                                    }),
           "explicit entropy gate resolves without profile injection");
    inputs.environment[gate] = "nan";
    if (std::string_view(profile) == "vanilla") {
      // Stock downgrades registry validation errors to warnings. The
      // behaviour reader still rejects nonfinite thresholds in every profile.
      const auto invalid = ResolveReleaseSettings(inputs);
      Expect(Find(invalid, gate).value == std::optional<std::string>("nan") &&
                 std::ranges::count(
                     invalid.logs,
                     "warning: " + gate + ": invalid value 'nan'") == 1,
             "vanilla retains the invalid entropy gate with a numeric warning");
    } else {
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "release profiles reject nonfinite entropy gates");
    }
  }
  for (const auto& file :
       std::filesystem::recursive_directory_iterator(root / "src")) {
    const auto ext = file.path().extension().string();
    if (!file.is_regular_file() ||
        (ext != ".cpp" && ext != ".hpp" && ext != ".hip" && ext != ".h" &&
         ext != ".cc" && ext != ".c"))
      continue;
    // Generated reads are metadata, never independent source evidence.
    if (file.path() == root / "src/cli/serve/release_switches.inc")
      continue;
    const auto text = Read(file.path());
    for (std::sregex_iterator match(text.begin(), text.end(), getenv_literal),
         end;
         match != end; ++match)
      if (!names.contains((*match)[1].str()) &&
          !irrelevant.contains((*match)[1].str()))
        missing.insert((*match)[1].str());
    for (std::sregex_iterator match(text.begin(), text.end(), literal), end;
         match != end; ++match)
      if (!names.contains((*match)[1].str()))
        missing.insert((*match)[1].str());
  }
  for (const auto& name : missing)
    std::cerr << "MISSING: " << name << '\n';
  Expect(missing.empty(), "source literal catalog completeness");
  // Extend the existing env/catalog scan with the public program-name policy.
  // Python keeps one implementation shared by hosted CI and this CPU test.
  std::string command = "\"" GUFO_BRAND_SCAN_PYTHON "\" -B \"" +
                        (root / "tools/release/check_brand.py").string() +
                        "\" --root \"" + root.string() + "\"";
#ifdef _WIN32
  command = "\"" + command + "\"";
#endif
  Expect(std::system(command.c_str()) == 0, "public program-name literal scan");
}

void ReviewRegression(int finding) {
  if (finding == 1) {
    auto inputs = Inputs();
    inputs.explicit_flags.clear();
    const auto result = ResolveReleaseSettings(inputs);
    Expect(!Find(result, "--speculative").value,
           "default without MTP does not inject speculation");
    for (const auto* name :
         {"GUFO_LENIENT", "GUFO_DRAFT_OUT_MODEL", "--mtp-policy",
          "--mtp-draft-vocab", "--prompt-lookup"})
      Expect(!Find(result, name).value &&
                 !Find(result, name).skipped_reason.empty(),
             "dependent default follows missing MTP: " + std::string(name));
    for (const auto* source : {"flag", "env", "manifest"}) {
      inputs = Inputs();
      inputs.explicit_flags.clear();
      if (std::string_view(source) == "flag")
        inputs.explicit_flags["--mtp-model"] = "mtp-test.gguf";
      else if (std::string_view(source) == "env")
        inputs.environment["--mtp-model"] = "mtp-test.gguf";
      else
        inputs.manifest_defaults["--mtp-model"] = "mtp-test.gguf";
      Expect(Find(ResolveReleaseSettings(inputs), "GUFO_LENIENT").value ==
                 std::optional<std::string>("bounded"),
             "resolved MTP artefact satisfies requirements from " +
                 std::string(source));
    }
  } else if (finding == 2) {
    auto inputs = Inputs();
    inputs.explicit_flags["--no-behaviour"] = "1";
    const auto result = ResolveReleaseSettings(inputs);
    Expect(Find(result, "GUFO_THINK_GUARD_ESCALATE").value ==
               std::optional<std::string>("0"),
           "no-behaviour turns guard escalation off");
    // The full resolver test checks the explicit suppression rows and every
    // unchanged row.
  } else if (finding == 3) {
    for (const auto* name :
         {"BUBO_API_KEY", "GUFO_EVAL_API_KEY", "GUFO_EVAL_ARTIFACT_ID",
          "GUFO_EVAL_MODEL_REVISION", "GUFO_EVAL_SERVER_COMMIT",
          "GUFO_EVAL_SERVER_CONFIG", "GUFO_EVAL_SERVER_LABEL", "PATH", "TEMP"})
      Expect(!DebugSettingsListsEnvironment(name),
             "echo excludes unsafe row: " + std::string(name));
    for (const auto* name :
         {"GUFO_EVAL_API_KEY", "GUFO_EVAL_ARTIFACT_ID",
          "GUFO_EVAL_MODEL_REVISION", "GUFO_EVAL_SERVER_COMMIT",
          "GUFO_EVAL_SERVER_CONFIG", "GUFO_EVAL_SERVER_LABEL"})
      (void)Find(ResolveReleaseSettings(Inputs("vanilla")), name);
    const auto settings =
        BuildEffectiveSettings(gufo::json::Value::object(), "v", "r", "b");
    Expect(!settings.find("environment")->contains("GUFO_EVAL_API_KEY"),
           "debug echo never contains a secret row");
    Expect(!settings.find("environment")->contains("BUBO_API_KEY"),
           "debug echo never contains the server API key row");
  } else if (finding == 4) {
    const auto text = Read(std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT) /
                           "cmake/Checks.cmake");
    Expect(text.substr(0, text.find("set(gufo_pr_tests"))
                   .find("release_profile_test") != std::string::npos,
           "check-pr builds and runs release_profile_test");
  } else if (finding == 5) {
    for (const auto* profile : {"vanilla", "default", "exact"}) {
      auto inputs = Inputs(profile);
      if (std::string_view(profile) != "vanilla")
        inputs.model_family = "other";
      inputs.environment["GUFO_DISK_IDLE_MS"] = "700000";
      inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "32768";
      inputs.environment["GUFO_LENIENT"] = "bounded";
      inputs.explicit_flags["--speculative"] = "off";
      const auto result = ResolveReleaseSettings(inputs);
      Expect(result.injections.empty(), "unowned settings inject nothing");
      Expect(std::any_of(
                 result.logs.begin(), result.logs.end(),
                 [](const auto& line) { return line.starts_with("warning:"); }),
             "unowned validation or requires produces warnings");
    }
  } else if (finding == 6) {
    AuditCatalog();
  } else if (finding == 7) {
    for (const auto* value : {"2147483648", "3000000000", "-1", "+1", "1x"}) {
      auto inputs = Inputs();
      inputs.environment["GUFO_THINK_GUARD_ESCALATE"] = value;
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "int31 rejects reader-invalid value");
    }
    for (const auto* value : {"0", "16", "2147483647"}) {
      auto inputs = Inputs();
      inputs.environment["GUFO_THINK_GUARD_ESCALATE"] = value;
      Expect(Find(ResolveReleaseSettings(inputs), "GUFO_THINK_GUARD_ESCALATE")
                     .value == std::optional<std::string>(value),
             "int31 accepts reader boundary");
    }
    const auto prior =
        SnapshotReleaseEnvironment().at("GUFO_THINK_GUARD_ESCALATE");
#ifdef _WIN32
    _putenv_s("GUFO_THINK_GUARD_ESCALATE", "");
#else
    setenv("GUFO_THINK_GUARD_ESCALATE", "", 1);
#endif
    ApplyInjections(ResolveReleaseSettings(Inputs("vanilla")));
    Expect(!SnapshotReleaseEnvironment().at("GUFO_THINK_GUARD_ESCALATE"),
           "empty real catalog value is removed in vanilla");
#ifdef _WIN32
    _putenv_s("GUFO_THINK_GUARD_ESCALATE", prior ? prior->c_str() : "");
#else
    if (prior)
      setenv("GUFO_THINK_GUARD_ESCALATE", prior->c_str(), 1);
    else
      unsetenv("GUFO_THINK_GUARD_ESCALATE");
#endif
  } else if (finding == 8) {
    for (const auto* name : {"GUFO_WEIGHT_UPLOAD_READERS",
                             "GUFO_PREFILL_STEP_LOG", "GUFO_PREFILL_TARGET_MS"})
      Expect(Find(ResolveReleaseSettings(Inputs()), name).release_class ==
                 "REMOVED",
             "unread row is REMOVED: " + std::string(name));
  } else if (finding == 10) {
    const char* args[] = {"--future-option", "video",
                          "--print-effective-settings"};
    const auto prior = SnapshotReleaseEnvironment();
    const auto startup = PrepareReleaseSettings(args);
    for (const auto& [name, value] : prior) {
#ifdef _WIN32
      _putenv_s(name.c_str(), value ? value->c_str() : "");
#else
      if (value)
        setenv(name.c_str(), value->c_str(), 1);
      else
        unsetenv(name.c_str());
#endif
    }
    Expect(startup.active && startup.error.empty(),
           "modality stops at first non-server token");
    CheckSpec(std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT) /
              "tests/fixtures/release_spec_fixture.json");
  } else if (finding == 16) {
    const std::map<std::string, std::optional<std::string>> off = {
        {"GUFO_THINK_EOG_GUARD", "0"},
        {"GUFO_ROUTED_LANEROW", "0"},
        {"GUFO_ROUTED_WMMA_IQ", "0"},
        {"GUFO_INCREMENTAL_TOKENIZE", "0"},
        {"GUFO_THINK_MARKER_BIAS", "0"},
        {"GUFO_LOGIT_BIAS", "0"},
        {"GUFO_MODEL_SUPPRESS_TOKENS", "0"},
        {"GUFO_REASONING_BUDGET", "0"},
        {"GUFO_PRODUCTION_SAMPLING", "0"},
        {"GUFO_LENIENT_THINK_ONLY", std::nullopt},
        {"GUFO_IQ4_WIDE_VARIANT", std::nullopt},
        {"GUFO_IQ4_WMMA_WAVES", std::nullopt},
        {"GUFO_EOG_BOUNDARY", "0"},
        {"GUFO_EOG_MIN_PROB", "-1"},
        {"GUFO_EOG_DEFER_P", "0"},
        {"GUFO_SPEC_EOG_STOP", "0"},
        {"GUFO_LOOP_BREAKER", "0"},
        {"GUFO_LOOP_BREAKER_EOG_CLOSE", "0"},
        {"GUFO_THINK_MARKER_BIAS_BY_EFFORT", std::nullopt},
        {"GUFO_THINK_GUARD_ESCALATE", "0"},
        {"GUFO_LENIENT", "off"},
        {"GUFO_DENSE_TWIN_MODEL", std::nullopt},
        {"GUFO_DENSE_TWIN_BATCH", "0"},
        {"GUFO_DENSE_TWIN_MAX_N", std::nullopt},
        {"GUFO_EXPERT_V2K", "0"},
        {"GUFO_DRAFT_OUT_MODEL", std::nullopt},
        {"--speculative", "off"}};
    const auto exact = ResolveReleaseSettings(Inputs("exact"));
    for (const auto& [name, value] : off) {
      Expect(Find(exact, name).value == value, "byte-exact off value: " + name);
      Expect(std::find(exact.turned_off.begin(), exact.turned_off.end(),
                       name) != exact.turned_off.end(),
             "exact lists the full byte-exact set: " + name);
      auto inputs = Inputs("exact");
      const auto on = name == "GUFO_LENIENT"                       ? "bounded"
                      : name == "--speculative"                    ? "mtp"
                      : name == "GUFO_EOG_MIN_PROB"                ? "0.2"
                      : name == "GUFO_EOG_DEFER_P"                 ? "0.9"
                      : name == "GUFO_THINK_MARKER_BIAS_BY_EFFORT" ? "medium:-1"
                      : name == "GUFO_DENSE_TWIN_MAX_N"            ? "8"
                      : name.find("MODEL") != std::string::npos ? "sidecar.gguf"
                                                                : "1";
      if (name.starts_with("--"))
        inputs.explicit_flags[name] = on;
      else
        inputs.environment[name] = on;
      if (name == "GUFO_DENSE_TWIN_MODEL" || name == "GUFO_DRAFT_OUT_MODEL") {
        Expect(!Find(ResolveReleaseSettings(inputs), name).value,
               "exact suppresses inherited artefact env before path checks: " +
                   name);
        inputs.explicit_flags[name == "GUFO_DENSE_TWIN_MODEL"
                                  ? "--twin-model"
                                  : "--draft-head-model"] = on;
      }
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact rejects positive re-enabling: " + name);
      inputs = Inputs("exact");
      inputs.manifest_defaults[name] = on;
      Expect(Find(ResolveReleaseSettings(inputs), name).value == value,
             "manifest cannot re-enable: " + name);
    }
    for (const auto* source : {"flag", "manifest"}) {
      for (const auto* value :
           {"-1", "-2", "+1", "1.5", "1x", "", " 1", "9223372036854775808"}) {
        auto inputs = Inputs("exact");
        if (std::string_view(source) == "flag")
          inputs.explicit_flags["--seed"] = value;
        else
          inputs.manifest_defaults["--seed"] = value;
        bool rejected = false;
        try {
          (void)ResolveReleaseSettings(inputs);
        } catch (const std::invalid_argument& error) {
          rejected = std::string_view(error.what()) ==
                     "profile=exact: --seed must be 0 or more";
        }
        Expect(rejected,
               "exact rejects invalid seed from " + std::string(source));
      }
    }
    for (const auto* value : {"0", "1", "01", "9223372036854775807"}) {
      auto inputs = Inputs("exact");
      inputs.explicit_flags["--seed"] = value;
      Expect(Find(ResolveReleaseSettings(inputs), "--seed").value ==
                 std::optional<std::string>(value),
             "exact accepts a fixed seed");
    }
    {
      auto inputs = Inputs("exact");
      inputs.explicit_flags["--no-mtp"] = "1";
      inputs.explicit_flags["--speculative"] = "mtp";
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact refuses explicit MTP even with no-mtp");
    }
    for (const auto* backend : {"mtp", "dspark", "dflash2"}) {
      auto inputs = Inputs("exact");
      inputs.explicit_flags["--speculative"] = backend;
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact refuses every speculative backend");
    }
    Expect(Find(exact, "--no-cache").value == std::optional<std::string>("1"),
           "exact injects no-cache");
    Expect(Find(exact, "--cache-disk").origin == "built-in",
           "an absent disk cache option retains its original origin");
    auto ignored_disk = Inputs("exact");
    ignored_disk.explicit_flags["--cache-disk"] = "cache";
    Expect(!ResolveReleaseSettings(ignored_disk)
                .effective_flags.contains("--cache-disk"),
           "exact accepts and ignores disk cache flags");
    gufo::cli::ArgParser parser("test");
    bool disabled = false;
    parser.AddFlag("", "--no-cache", "Disable caches", "Cache", &disabled);
    PrefillReleaseFlags(exact, parser);
    Expect(disabled, "no-cache effective setting reaches the parser sink");
    Expect(Find(ResolveReleaseSettings(Inputs()), "--no-cache").origin ==
                   "built-in" &&
               Find(ResolveReleaseSettings(Inputs("vanilla")), "--no-cache")
                       .origin == "built-in",
           "vanilla and default do not inject no-cache");
  } else {
    throw std::invalid_argument("unknown regression");
  }
  std::cout << "PASS: ReviewRegression" << finding << '\n';
}

void PlaceholderNotesCheck(std::span<const gufo::json::Value> rows) {
  for (const auto& row : rows) {
    if (row.member_str("class") != "OPTION" &&
        row.member_str("class") != "DEFAULT")
      continue;
    for (const auto* key : {"purpose", "note", "default_text"}) {
      const auto text = row.member_str(key);
      Expect(text.find("Inventory classification.") == std::string::npos &&
                 text.find("Control ") == std::string::npos,
             "public row has user wording: " + row.member_str("name"));
    }
    const auto note = row.member_str("note");
    std::istringstream words(note);
    std::size_t count = 0;
    for (std::string word; words >> word;)
      ++count;
    Expect(count > 0 && count <= 20 && note.ends_with('.') &&
               std::count(note.begin(), note.end(), '.') == 1,
           "public change note is one sentence of at most 20 words: " +
               row.member_str("name"));
  }
}

void Review2Regression(int finding) {
  auto inputs = Inputs("exact");
  if (finding == 1) {
    Expect(Find(ResolveReleaseSettings(inputs), "GUFO_ROUTED_LANEROW").value ==
               std::optional<std::string>("0"),
           "exact uses the vanilla routed kernel");
  } else if (finding == 2) {
    inputs.explicit_flags["--seed"] = "-1";
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "exact rejects random seed");
  } else if (finding == 3) {
    for (const auto* name :
         {"GUFO_THINK_MARKER_BIAS", "GUFO_LOGIT_BIAS",
          "GUFO_MODEL_SUPPRESS_TOKENS", "GUFO_REASONING_BUDGET",
          "GUFO_PRODUCTION_SAMPLING", "GUFO_LENIENT_THINK_ONLY",
          "GUFO_IQ4_WIDE_VARIANT", "GUFO_IQ4_WMMA_WAVES"}) {
      inputs = Inputs("exact");
      inputs.environment[name] = "1";
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact rejects output control: " + std::string(name));
      std::cout << "PASS: exact rejects " << name << '\n';
    }
    inputs = Inputs("exact");
    inputs.environment["GUFO_HIPBLASLT_PLAN_CACHE"] = "0";
    inputs.environment["GUFO_EVAL_API_KEY"] = "secret-marker";
    inputs.environment["BUBO_API_KEY"] = "bubo-secret-marker";
    const auto result = ResolveReleaseSettings(inputs);
    Expect(std::none_of(
               result.logs.begin(), result.logs.end(),
               [](const auto& line) {
                 return line.find("BUBO_API_KEY") != std::string::npos ||
                        line.find("bubo-secret-marker") != std::string::npos;
               }),
           "BUBO_API_KEY is not stripped, listed or logged by the profile");
    Expect(Find(result, "BUBO_API_KEY").value ==
               std::optional<std::string>("bubo-secret-marker"),
           "BUBO_API_KEY survives profile resolution");
    Expect(std::none_of(result.logs.begin(), result.logs.end(),
                        [](const auto& line) {
                          return line.find("secret-marker") !=
                                 std::string::npos;
                        }),
           "internal control diagnostics never expose secret values");
    Expect(
        std::count(result.logs.begin(), result.logs.end(),
                   "profile=exact: GUFO_HIPBLASLT_PLAN_CACHE is set (internal "
                   "control)") == 1,
        "exact logs the unread plan cache control once");
  } else if (finding == 4) {
    const auto docs = RenderReleaseDocs(ReleaseSwitchTable());
    Expect(docs.find("identical to plain autoregressive decoding of that build "
                     "with the same settings") != std::string::npos,
           "docs qualify reference equality");
    Expect(
        docs.find("--profile vanilla --speculative off --no-cache --sessions "
                  "1 --seed 1 --think on --reasoning-effort medium") !=
            std::string::npos,
        "docs state the complete reference command");
  } else if (finding == 5) {
    PlaceholderNotesCheck(ReleaseSwitchTable());
    for (const auto* classification : {"OPTION", "DEFAULT"}) {
      for (const auto* placeholder :
           {"Inventory classification.", "Control something."}) {
        auto rows = ReleaseSwitchTable();
        rows[0]["class"] = classification;
        rows[0]["model_family"] = "any";
        rows[0]["note"] = placeholder;
        bool rejected = false;
        try {
          PlaceholderNotesCheck(rows);
        } catch (const std::runtime_error&) {
          rejected = true;
        }
        Expect(rejected, "placeholder lint rejects any-family public rows");
      }
    }
    for (const auto& [name, value] : std::map<std::string, std::string>{
             {"--request-timeout-ms", "0 (off)"},
             {"--max-pending", "16"},
             {"--max-pending-per-client", "Same as --max-pending"},
             {"--max-output-bytes", "1048576 (1 MiB)"},
             {"--max-buffered-output-bytes", "65536 (64 KiB)"},
             {"--max-buffered-output-total", "262144 (256 KiB)"},
             {"--repeat-last-n", "64"},
             {"--repeat-penalty", "1.0 (off)"},
             {"--frequency-penalty", "0.0 (off)"}}) {
      const auto& rows = ReleaseSwitchTable();
      auto row = std::find_if(rows.begin(), rows.end(), [&](const auto& r) {
        return r.member_str("name") == name;
      });
      Expect(row != rows.end() && row->member_str("default_text") == value,
             "parser default documented: " + name);
    }
  } else if (finding == 6) {
    const auto docs = RenderReleaseDocs(ReleaseSwitchTable());
    Expect(docs.find("manifest job") == std::string::npos,
           "public docs omit internal manifest job wording");
    Expect(docs.find("Sidecar files (MTP, vision, dense twin, draft head) are "
                     "set with their own options.") != std::string::npos,
           "public docs explain sidecar options");
  } else if (finding == 7) {
    inputs.model_family = "other";
    const auto result = ResolveReleaseSettings(inputs);
    Expect(std::count(result.logs.begin(), result.logs.end(),
                      "profile=exact: byte-exact controls apply to Qwen3.8 "
                      "Flash Next only; seed=1 applied") == 1,
           "other family receives the exact scope message");
    Expect(result.effective_flags.at("--seed") == "1",
           "other family gets seed 1");
    Expect(result.injections.empty() && result.turned_off.empty(),
           "exact does not inject byte-exact controls into another family");
    for (const auto& row : ReleaseSwitchTable()) {
      if (row.member_str("kind") != "env" || !row.find("exact_off")->as_bool())
        continue;
      auto other = inputs;
      const auto name = row.member_str("name");
      other.environment[name] = "1";
      const auto external = ResolveReleaseSettings(other);
      Expect(Find(external, name).origin == "env" &&
                 external.injections.empty() && external.turned_off.empty(),
             "other-family exact leaves explicit environment control alone: " +
                 name);
    }
    auto concurrent = inputs;
    concurrent.explicit_flags["--sessions"] = "2";
    concurrent.explicit_flags["--cache-disk"] = "cache";
    concurrent.explicit_flags["--speculative"] = "off";
    Expect(
        ResolveReleaseSettings(concurrent).effective_flags.at("--sessions") ==
            "2",
        "other-family exact accepts its ordinary session and cache options");
    Expect(RenderReleaseDocs(ReleaseSwitchTable())
                   .find("Other model families keep their existing defaults "
                         "except for the exact seed and NPU profile policy.") !=
               std::string::npos,
           "docs state the other-family seed exception");
  } else if (finding == 8) {
    inputs = Inputs("default");
    inputs.explicit_flags["--no-cache"] = "1";
    inputs.explicit_flags["--cache-disk"] = "cache";
    const auto result = ResolveReleaseSettings(inputs);
    Expect(std::count(result.logs.begin(), result.logs.end(),
                      "cache: --no-cache overrides --cache-disk") == 1,
           "no-cache override logs once");
    inputs.explicit_flags["--cache-disk"] = "0";
    const auto zero_directory = ResolveReleaseSettings(inputs);
    Expect(std::count(zero_directory.logs.begin(), zero_directory.logs.end(),
                      "cache: --no-cache overrides --cache-disk") == 1,
           "a directory named zero also produces the override log");
    inputs.explicit_flags.erase("--no-cache");
    const auto unchanged = ResolveReleaseSettings(inputs);
    Expect(std::find(unchanged.logs.begin(), unchanged.logs.end(),
                     "cache: --no-cache overrides --cache-disk") ==
               unchanged.logs.end(),
           "disk cache alone has no override log");
  } else {
    throw std::invalid_argument("unknown review 2 regression");
  }
  std::cout << "PASS: Review2Regression" << finding << '\n';
}

void HeaderTest() {
  // A synthetic metadata-only GGUF needs no model or GPU. split.count proves
  // header inspection does not require a sibling shard.
  std::vector<unsigned char> bytes;
  const auto integer = [&](std::uint64_t value, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      bytes.push_back(static_cast<unsigned char>(value >> (8 * i)));
  };
  const auto text = [&](const std::string& value) {
    integer(value.size(), 8);
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  integer(0x46554747, 4);
  integer(3, 4);
  integer(0, 8);
  integer(3, 8);
  text("general.architecture");
  integer(8, 4);
  text("qwen4exp");
  text("qwen4exp.context_length");
  integer(4, 4);
  integer(262144, 4);
  text("split.count");
  integer(4, 4);
  integer(2, 4);
  bytes.resize((bytes.size() + 31) / 32 * 32);
  const auto path =
      std::filesystem::current_path() / "release-profile-header-test.gguf";
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  const auto prior = SnapshotReleaseEnvironment();
  const std::string filename = path.string();
  const char* argv[] = {"llm", "--model", filename.c_str(),
                        "--print-effective-settings"};
  const auto startup = PrepareReleaseSettings(argv);
  for (const auto& [name, value] : prior) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value->c_str() : "");
#else
    if (value)
      setenv(name.c_str(), value->c_str(), 1);
    else
      unsetenv(name.c_str());
#endif
  }
  std::filesystem::remove(path);
  Expect(startup.error.empty(),
         "header-only startup succeeds without sibling or tensor data");
  Expect(startup.resolved.architecture == "qwen4exp" &&
             startup.resolved.native_context ==
                 std::optional<std::uint64_t>(262144),
         "architecture and native context come from header");
}

void NpuPolicyTest() {
  using gufo::core::ExperimentalFeature;
#ifdef GUFO_NPU
  constexpr bool available = true;
#else
  constexpr bool available = false;
#endif
  for (const auto* profile : {"default", "subagents", "exact", "vanilla"}) {
    const bool restricted = std::string_view(profile) == "exact" ||
                            std::string_view(profile) == "vanilla";
    auto inputs = Inputs(profile);
    inputs.manifest_defaults["--npu-prefill"] = "off";
    inputs.environment["--npu-prefill"] = "off";
    auto resolved = ResolveReleaseSettings(inputs);
    // v0.1.1: the NPU is off by default in every profile. Only an explicit
    // command-line --npu-prefill on (below) turns it on; a manifest or an
    // environment value does not.
    Expect(!resolved.prefill_authorization.Allows(ExperimentalFeature::Prefill),
           "NPU is off by default; manifest and environment flag values do not "
           "enable it");
    Expect(Find(resolved, "--npu-prefill").value == "off",
           "NPU effective policy is off without an explicit flag");
    Expect(restricted || resolved.prefill_authorization.reason() ==
                             "profile default OFF",
           "NPU default reason names the default");
    if (restricted) {
      for (const auto& log : resolved.logs)
        Expect(log.find("npu") == std::string::npos &&
                   log.find("GUFO_NPU_") == std::string::npos,
               "restricted profiles emit no NPU log");
      inputs.environment["GUFO_NPU_KEEPALIVE"] = "";
      inputs.environment["GUFO_NPU_SLICES"] = "malformed";
      resolved = ResolveReleaseSettings(inputs);
      bool called = false;
      Expect(
          !gufo::core::CreateAuthorizedPrefill(resolved.prefill_authorization,
                                               [&](auto) { called = true; }) &&
              !called,
          "restricted profile never creates an NPU session");
    }
    inputs.explicit_flags["--npu-prefill"] = "off";
    inputs.manifest_defaults["--npu-prefill"] = "on";
    inputs.environment["--npu-prefill"] = "on";
    resolved = ResolveReleaseSettings(inputs);
    Expect(
        !resolved.prefill_authorization.Allows(ExperimentalFeature::Prefill) &&
            Find(resolved, "--npu-prefill").value == "off",
        "explicit NPU OFF cannot be overridden");
    inputs.explicit_flags["--npu-prefill"] = "on";
    if (restricted) {
      bool rejected = false;
      try {
        (void)ResolveReleaseSettings(inputs);
      } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what())
                       .find("effective=off: profile=" + std::string(profile) +
                             " forces OFF") != std::string::npos;
      }
      Expect(rejected, "restricted profile rejects explicit NPU ON");
    } else {
      resolved = ResolveReleaseSettings(inputs);
      Expect(resolved.prefill_authorization.Allows(
                 ExperimentalFeature::Prefill) == available,
             "explicit ON tolerates a build without NPU");
      Expect(Find(resolved, "--npu-prefill").value == (available ? "on" : "off"),
             "explicit --npu-prefill on opts in when the build has an NPU");
    }
  }
}
void TestResolver() {
  const auto baseline = ResolveReleaseSettings(Inputs());
  {
    auto inputs = Inputs("exact");
    inputs.environment["GUFO_LENIENT"] = "bounded";
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "exact refuses a flag that enables an approximation");
    inputs = Inputs("exact");
    inputs.manifest_defaults["GUFO_DRAFT_OUT_MODEL"] = "draft-head.gguf";
    Expect(!Find(ResolveReleaseSettings(inputs), "GUFO_DRAFT_OUT_MODEL").value,
           "exact suppresses the manifest draft head");
    inputs = Inputs("subagents");
    inputs.explicit_flags["--context"] = "0";
    inputs.native_context = 262144;
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "memory guard accounts for native context");
    inputs = Inputs();
    inputs.explicit_flags["--cache-disk"] = "0";
    Expect(Find(ResolveReleaseSettings(inputs), "GUFO_DISK_PERSIST").value ==
               std::optional<std::string>("evict_idle"),
           "directory named zero selects disk bundle");
    inputs.explicit_flags["--cache-disk"] = "";
    Expect(!Find(ResolveReleaseSettings(inputs), "GUFO_DISK_PERSIST").value,
           "empty disk path leaves bundle off");
    gufo::cli::ArgParser parser("test");
    std::size_t sessions = 99;
    parser.AddOption("-j", "--sessions", "N", "Sessions", "Test", &sessions);
    PrefillReleaseFlags(baseline, parser);
    Expect(sessions == 1 && !parser.WasSupplied("--sessions"),
           "profile prefill does not mark a flag supplied");
    const char* args[] = {"-j", "2"};
    Expect(
        parser.Parse(args) && sessions == 2 && parser.WasSupplied("--sessions"),
        "explicit flag overrides the prefilled sink");
    auto injection = Find(baseline, "GUFO_THINK_EOG_GUARD");
    injection.value = "bad";
    ReleaseResolution bad;
    bad.injections.push_back(injection);
    Reject([&] { ApplyInjections(bad); },
           "invalid injection refuses before an environment write");
  }
  for (const auto* name :
       {"GUFO_THINK_EOG_GUARD", "GUFO_EOG_BOUNDARY", "GUFO_LOOP_BREAKER",
        "GUFO_ROUTED_LANEROW", "GUFO_INCREMENTAL_TOKENIZE",
        "GUFO_EOG_MIN_PROB", "GUFO_EOG_DEFER_P"}) {
    auto inputs = Inputs();
    const std::string value = name == std::string("GUFO_EOG_MIN_PROB")  ? "0.3"
                              : name == std::string("GUFO_EOG_DEFER_P") ? "0.7"
                                                                        : "0";
    inputs.manifest_defaults[name] = value;
    Expect(Find(ResolveReleaseSettings(inputs), name).origin == "manifest",
           "manifest beats profile");
    inputs.environment[name] = value;
    Expect(Find(ResolveReleaseSettings(inputs), name).origin == "env",
           "env beats manifest");
    inputs.explicit_flags[name] = value;
    Expect(Find(ResolveReleaseSettings(inputs), name).origin == "flag",
           "flag beats env");
    Expect(Find(baseline, name).origin == "profile:default",
           "profile beats built-in");
  }
  auto inputs = Inputs("exact");
  inputs.explicit_flags["--sessions"] = "01";
  Expect(Find(ResolveReleaseSettings(inputs), "--sessions").value ==
             std::optional<std::string>("1"),
         "exact accepts a numerically single session with leading zeros");
  inputs = Inputs();
  inputs.explicit_flags["--sessions"] = "01";
  inputs.environment["GUFO_DENSE_TWIN_MODEL"] = "twin.gguf";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_DENSE_TWIN_MODEL").origin ==
             "env",
         "session requirements use the integer flag value");
  // More than one session with a twin selects the batched twin; one session,
  // no twin, vanilla and exact do not (decode gate 2026-10-14).
  inputs = Inputs();
  inputs.manifest_defaults["GUFO_DENSE_TWIN_MODEL"] = "twin.gguf";
  inputs.explicit_flags["--sessions"] = "2";
  {
    const auto multi = ResolveReleaseSettings(inputs);
    Expect(
        Find(multi, "GUFO_DENSE_TWIN_BATCH").value ==
                std::optional<std::string>("1") &&
            Find(multi, "GUFO_DENSE_TWIN_BATCH").origin == "profile:default" &&
            Find(multi, "GUFO_DENSE_TWIN_MODEL").value ==
                std::optional<std::string>("twin.gguf") &&
            Find(multi, "GUFO_DENSE_TWIN_MODEL").skipped_reason.empty() &&
            std::any_of(multi.injections.begin(), multi.injections.end(),
                        [](const auto& s) {
                          return s.name == "GUFO_DENSE_TWIN_BATCH" &&
                                 s.value == std::optional<std::string>("1");
                        }),
        "sessions > 1 with a twin sets and injects the batched twin");
  }
  {
    const auto document = ReleaseSettingsJson(ResolveReleaseSettings(inputs));
    const auto* twin_batched = document.find("engine")->find("twin_batched");
    Expect(twin_batched && twin_batched->as_bool(),
           "engine.twin_batched is true");
  }
  inputs.explicit_flags["--sessions"] = "1";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_DENSE_TWIN_BATCH").value !=
             std::optional<std::string>("1"),
         "one session leaves the batched twin off");
  {
    const auto document = ReleaseSettingsJson(ResolveReleaseSettings(inputs));
    Expect(!document.find("engine")->find("twin_batched")->as_bool(),
           "engine.twin_batched is false with one session");
  }
  inputs.explicit_flags["--sessions"] = "2";
  inputs.explicit_flags["--no-twin"] = "1";
  {
    const auto no_twin = ResolveReleaseSettings(inputs);
    Expect(!Find(no_twin, "GUFO_DENSE_TWIN_MODEL").value &&
               Find(no_twin, "GUFO_DENSE_TWIN_BATCH").value !=
                   std::optional<std::string>("1"),
           "no twin means no batched twin");
  }
  inputs.explicit_flags.erase("--no-twin");
  inputs.environment["GUFO_DENSE_TWIN_BATCH"] = "0";
  {
    const auto off = ResolveReleaseSettings(inputs);
    Expect(Find(off, "GUFO_DENSE_TWIN_BATCH").origin == "env" &&
               !Find(off, "GUFO_DENSE_TWIN_MODEL").value,
           "explicit batch 0 keeps the old skip");
  }
  inputs = Inputs("vanilla");
  inputs.explicit_flags["--sessions"] = "2";
  inputs.manifest_defaults["GUFO_DENSE_TWIN_MODEL"] = "twin.gguf";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_DENSE_TWIN_BATCH").value !=
             std::optional<std::string>("1"),
         "vanilla never sets the batched twin");
  inputs = Inputs();
  inputs.explicit_flags["--think"] = "off";
  Expect(!Find(ResolveReleaseSettings(inputs), "--reasoning-effort").value,
         "thinking off skips the medium effort default");
  inputs.explicit_flags["--reasoning-effort"] = "medium";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "explicit effort requires thinking");
  inputs.explicit_flags["--reasoning-effort"] = "auto";
  Expect(Find(ResolveReleaseSettings(inputs), "--reasoning-effort").value ==
             std::optional<std::string>("auto"),
         "explicit auto effort keeps thinking off compatible");
  inputs = Inputs();
  inputs.real_environment = [](const char* name) -> const char* {
    return std::string_view(name) == "GUFO_THINK_EOG_GUARD" ? "" : nullptr;
  };
  auto empty = ResolveReleaseSettings(inputs);
  Expect(Find(empty, "GUFO_THINK_EOG_GUARD").value ==
             std::optional<std::string>("1"),
         "empty env is unset");
  Expect(std::any_of(empty.logs.begin(), empty.logs.end(),
                     [](const auto& line) {
                       return line.find("empty environment") !=
                              std::string::npos;
                     }),
         "empty env warning");
  auto exact = ResolveReleaseSettings(Inputs("exact"));
  for (const auto& row : ReleaseSwitchTable())
    if (row.find("exact_off")->as_bool()) {
      const auto name = row.member_str("name");
      Expect(std::find(exact.turned_off.begin(), exact.turned_off.end(),
                       name) != exact.turned_off.end(),
             "exact lists every approximation");
      const auto* expected = row.find("values")->find("exact");
      Expect(expected->is_null()
                 ? !Find(exact, name).value
                 : Find(exact, name).value ==
                       std::optional<std::string>(expected->str()),
             "exact off value: " + name);
    }
  Expect(Find(exact, "GUFO_THINK_EOG_GUARD").value ==
             std::optional<std::string>("0"),
         "exact disables behaviour");
  Expect(exact.effective_flags.at("--seed") == "1", "exact seed pin");
  Expect(Find(exact, "GUFO_PREFILL_FORWARD_TOKENS").value ==
             std::optional<std::string>("4096"),
         "exact forward pin");
  for (const auto* name : {"GUFO_LENIENT", "GUFO_DENSE_TWIN_MODEL"}) {
    inputs = Inputs("exact");
    inputs.environment[name] =
        name == std::string("GUFO_LENIENT") ? "bounded" : "twin.gguf";
    if (name == std::string("GUFO_DENSE_TWIN_MODEL")) {
      Expect(ResolveReleaseSettings(inputs).artefacts.twin.empty(),
             "exact suppresses inherited twin before binding");
      inputs.explicit_flags["--twin-model"] = "twin.gguf";
    }
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "exact refuses explicit approximation");
  }
  inputs = Inputs("exact");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "2048";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "exact refuses a different prefill geometry");
  inputs = Inputs("exact");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "4097";
  {
    std::string message;
    try {
      (void)ResolveReleaseSettings(inputs);
    } catch (const std::invalid_argument& error) {
      message = error.what();
    }
    Expect(message.find("GUFO_PREFILL_FORWARD_TOKENS=4097") !=
                   std::string::npos &&
               message.find("requires 4096") != std::string::npos &&
               message.find("aligned to 4") != std::string::npos,
           "exact refuses an unaligned prefill forward size with a reason");
  }
  inputs = Inputs("exact");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "4096";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_PREFILL_FORWARD_TOKENS")
                 .value == std::optional<std::string>("4096"),
         "exact accepts forward size 4096");
  inputs = Inputs("default");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "4097";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_PREFILL_FORWARD_TOKENS")
                 .value == std::optional<std::string>("4097"),
         "default profile keeps forward size 4097");
  const std::map<std::string, std::optional<std::string>> off = {
      {"GUFO_THINK_GUARD_ESCALATE", "0"},
      {"GUFO_THINK_EOG_GUARD", "0"},
      {"GUFO_EOG_BOUNDARY", "0"},
      {"GUFO_EOG_MIN_PROB", "-1"},
      {"GUFO_EOG_DEFER_P", "0"},
      {"GUFO_SPEC_EOG_STOP", "0"},
      {"GUFO_LOOP_BREAKER", "0"},
      {"GUFO_LOOP_BREAKER_TOOL_ARGS", "0"},
      {"GUFO_LOOP_BREAKER_EOG_CLOSE", "0"},
      {"GUFO_THINK_MARKER_BIAS_BY_EFFORT", std::nullopt}};
  // The nine engine controls retain their values. The new CLI alias must
  // also show the loop breaker's disabled effective state.
  const std::map<std::string, std::optional<std::string>> aliases = {
      {"--loop-breaker", "off"}};
  inputs = Inputs();
  inputs.explicit_flags["--no-behaviour"] = "1";
  auto disabled = ResolveReleaseSettings(inputs);
  for (const auto& setting : baseline.settings) {
    if (off.contains(setting.name))
      Expect(Find(disabled, setting.name).value == off.at(setting.name),
             "behaviour off: " + setting.name);
    else if (aliases.contains(setting.name))
      Expect(Find(disabled, setting.name).value == aliases.at(setting.name),
             "behaviour flag alias off: " + setting.name);
    else if (setting.name != "--no-behaviour")
      Expect(Find(disabled, setting.name).value == setting.value,
             "no-behaviour changes only the explicit suppression rows");
  }
  inputs.environment["GUFO_THINK_EOG_GUARD"] = "1";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_THINK_EOG_GUARD").origin ==
             "env",
         "real env beats no-behaviour");
  auto sub = ResolveReleaseSettings(Inputs("subagents"));
  for (const auto& [name, value] :
       std::map<std::string, std::string>{{"--sessions", "4"},
                                          {"--context", "131072"},
                                          {"--cache-ram-bytes", "4294967296"},
                                          {"--prefill-chunk", "512"},
                                          {"--mtp-draft-vocab", "latin"},
                                          {"GUFO_DENSE_TWIN_BATCH", "1"}})
    Expect(Find(sub, name).value == std::optional(value),
           "subagent setting: " + name);
  Expect(Find(sub, "GUFO_LENIENT").value ==
                 std::optional<std::string>("bounded") &&
             Find(sub, "GUFO_LOOP_BREAKER").value ==
                 std::optional<std::string>("1"),
         "subagents keeps production behaviour and lenient policies");
  Expect(Find(sub, "GUFO_DENSE_TWIN_MAX_N").origin == "built-in",
         "subagents never injects twin width");
  inputs = Inputs("subagents");
  inputs.explicit_flags["--context"] = "262144";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "memory guard refuses unmeasured geometry");
  inputs.explicit_flags["--sessions"] = "1";
  inputs.explicit_flags["--context"] = "1048576";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "subagent memory cap also covers one session");
  inputs.explicit_flags["--allow-unmeasured-memory"] = "1";
  Expect(!ResolveReleaseSettings(inputs).settings.empty(),
         "memory override works");
  inputs = Inputs();
  inputs.explicit_flags["--speculative"] = "off";
  auto skipped = ResolveReleaseSettings(inputs);
  Expect(!Find(skipped, "GUFO_LENIENT").value &&
             Find(skipped, "GUFO_LENIENT").skipped_reason ==
                 "skipped: needs speculative==mtp",
         "unmet default requirement skips");
  inputs.environment["GUFO_LENIENT"] = "bounded";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "explicit requirement violation refuses");
  // Read the real catalog declaration rather than a test-owned family list.
  // New catalog families enter this matrix without updating the test.
  const auto catalog_source =
      Read(std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT) /
           "tools/release/gen_switches.py");
  std::smatch declaration;
  Expect(std::regex_search(catalog_source, declaration,
                           std::regex(R"(\bFAMILIES\s*=\s*\{([^}]*)\})")),
         "real catalog FAMILIES declaration exists");
  const auto family_source = declaration[1].str();
  const std::regex family_literal(R"family(['"]([^'"]+)['"])family");
  std::set<std::string> families;
  for (std::sregex_iterator
           match(family_source.begin(), family_source.end(), family_literal),
       end;
       match != end; ++match)
    families.insert((*match)[1].str());
  families.erase("any");
  families.insert("qwen3.6-a3b");
  const std::set<std::string> help_families(
      std::begin(gufo::cli::kModelFamilies),
      std::end(gufo::cli::kModelFamilies));
  Expect(help_families == families,
         "public family list matches the real catalog");
  for (const auto& family : families) {
    for (const auto* profile : {"default", "subagents", "vanilla", "exact"}) {
      for (const bool suppressed : {false, true}) {
        auto upstream_inputs = Inputs("vanilla");
        upstream_inputs.model_family = family;
        if (suppressed)
          upstream_inputs.explicit_flags["--no-behaviour"] = "1";
        const auto upstream = ResolveReleaseSettings(upstream_inputs);
        auto candidate_inputs = upstream_inputs;
        candidate_inputs.profile = profile;
        const auto candidate = ResolveReleaseSettings(candidate_inputs);
        if (family == "qwen3.8-flash-next" &&
            std::string_view(profile) != "vanilla") {
          Expect(!candidate.injections.empty() ||
                     candidate.effective_flags != upstream.effective_flags,
                 "Flash Next retains its tuned profile");
          continue;
        }
        auto flags = upstream.effective_flags;
        const bool exact = std::string_view(profile) == "exact";
        if (exact)
          flags["--seed"] = "1";
#ifdef GUFO_NPU
        flags["--npu-prefill"] = "off";  // v0.1.1: off in every profile unless
                                         // --npu-prefill on is given
#endif
        Expect(candidate.effective_flags == flags,
               "upstream flag diff: " + std::string(family) + "/" + profile);
        Expect(candidate.injections.empty(), "upstream env injections empty");
        for (const auto& setting : candidate.settings) {
          const auto& vanilla = Find(upstream, setting.name);
          Expect(setting.value ==
                     (setting.name == "--npu-prefill"
                          ? std::optional<std::string>(flags.at(setting.name))
                      : exact && setting.name == "--seed"
                          ? std::optional<std::string>("1")
                          : vanilla.value),
                 "effective env/flag diff: " + std::string(family) + "/" +
                     profile + "/" + setting.name);
        }
        Expect(candidate.turned_off.empty(), "other family policies retained");
        Expect(std::none_of(
                   candidate.settings.begin(), candidate.settings.end(),
                   [](const auto& row) {
                     return row.skipped_reason.starts_with("skipped: needs");
                   }),
               "no missing-artifact defaults for other families");
        if (exact) {
          Expect(std::find(
                     candidate.logs.begin(), candidate.logs.end(),
                     "profile=exact: byte-exact controls apply to Qwen3.8 "
                     "Flash Next only; seed=1 applied") != candidate.logs.end(),
                 "other-family exact documents seed-only scope");
          candidate_inputs.explicit_flags["--seed"] = "7";
          Expect(ResolveReleaseSettings(candidate_inputs)
                         .effective_flags.at("--seed") == "7",
                 "other-family exact retains an explicit seed");
        }
      }
      std::cout << "PASS: family=" << family << " profile=" << profile
                << (family == "qwen3.8-flash-next" &&
                            std::string_view(profile) != "vanilla"
                        ? " tuned defaults retained"
                    : std::string_view(profile) == "exact"
                        ? " diff=seed:1 only; message checked"
                        : " model defaults unchanged; NPU policy checked")
                << " (with/without --no-behaviour)\n";
    }
  }
  Expect(ResolveReleaseSettings(Inputs("vanilla")).injections.empty(),
         "vanilla injects nothing");
  inputs = Inputs();
  inputs.explicit_flags["--cache-disk"] = "cache";
  auto disk = ResolveReleaseSettings(inputs);
  for (const auto* name :
       {"GUFO_DISK_PERSIST", "GUFO_DISK_DEFER", "GUFO_DISK_EVICT_SKIP_TOKENS",
        "GUFO_DISK_EVICT_DOMINATED"}) {
    Expect(Find(disk, name).origin == "profile:default",
           "disk bundle selected");
    Expect(!Find(baseline, name).value, "disk bundle absent when off");
  }
  inputs = Inputs();
  inputs.profile = "unknown";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "unknown profile refuses");
  for (const auto& [name, value] : std::map<std::string, std::string>{
           {"GUFO_THINK_GUARD_ESCALATE", "3000000000"},
           {"GUFO_THINK_EOG_GUARD", "01"},
           {"GUFO_EOG_MIN_PROB", "2"},
           {"GUFO_EOG_DEFER_P", "nan"},
           {"GUFO_LENIENT_LAMBDA", "0"},
           {"GUFO_DISK_EVICT_SKIP_TOKENS", "1000001"},
           {"GUFO_DISK_IDLE_MS", "600001"},
           {"GUFO_DISK_DEFER_MS", "-1"},
           {"GUFO_PREFILL_FORWARD_TOKENS", "1e4"},
           {"GUFO_THINK_MARKER_BIAS_BY_EFFORT", "low:inf"}}) {
    inputs = Inputs();
    inputs.environment[name] = value;
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "invalid real value refuses");
  }
  inputs = Inputs("exact");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "4096tokens";
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_PREFILL_FORWARD_TOKENS")
                 .origin == "env",
         "forward validation retains the existing integer-prefix grammar");
  for (const auto* field : {"class", "purpose", "requires"}) {
    auto rows = ReleaseSwitchTable();
    if (std::string_view(field) == "requires") {
      rows[0][field] = gufo::json::Value::array();
      rows[0][field].push_back("unknown");
    } else
      rows[0][field] = std::string_view(field) == "class" ? "unknown" : "";
    Reject([&] { ValidateSwitchTable(rows); }, "C++ rejects invalid table");
  }
  const auto root = std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT);
  const auto& rows = ReleaseSwitchTable();
  const auto source =
      gufo::json::parse(Read(root / "src/cli/serve/release_switches.json"));
  Expect(source.find("switches")->size() == rows.size(),
         "JSON table row count agrees");
  Expect(RenderReleaseInclude(source.find("switches")->items()) ==
             Read(root / "src/cli/serve/release_switches.inc"),
         "JSON regenerates committed header in C++");
  Expect(RenderReleaseInclude(rows) ==
             Read(root / "src/cli/serve/release_switches.inc"),
         "compiled table regenerates header");
  Expect(RenderReleaseDocs(rows) == Read(root / "docs/FLAGS.md"),
         "compiled table regenerates docs");
  Expect(RenderReleaseDocs(rows) ==
             Read(root / "tests/fixtures/release_flags_golden.md"),
         "C++ renderer matches golden");
  CheckSpec(root / "tests/fixtures/release_spec_fixture.json");
  if (std::string_view(GUFO_PRODUCTION_ENV_SPEC).empty())
    std::cout << "SKIP: GUFO_PRODUCTION_ENV_SPEC is empty; committed fixture "
                 "always checked\n";
  else
    CheckSpec(GUFO_PRODUCTION_ENV_SPEC);
  // Exercise application with only one isolated CPU environment setting.
  const auto previous = SnapshotReleaseEnvironment().at("GUFO_THINK_EOG_GUARD");
#ifdef _WIN32
  _putenv_s("GUFO_THINK_EOG_GUARD", "0");
#else
  setenv("GUFO_THINK_EOG_GUARD", "0", 1);
#endif
  ReleaseResolution apply;
  apply.injections.push_back(Find(baseline, "GUFO_THINK_EOG_GUARD"));
  ApplyInjections(apply);
  Expect(SnapshotReleaseEnvironment().at("GUFO_THINK_EOG_GUARD") ==
             std::optional<std::string>("0"),
         "injection leaves existing value untouched");
#ifdef _WIN32
  _putenv_s("GUFO_THINK_EOG_GUARD", "");
#else
  unsetenv("GUFO_THINK_EOG_GUARD");
#endif
  ApplyInjections(apply);
  Expect(SnapshotReleaseEnvironment().at("GUFO_THINK_EOG_GUARD") ==
             std::optional<std::string>("1"),
         "injection fills unset value");
#ifdef _WIN32
  _putenv_s("GUFO_THINK_EOG_GUARD", previous ? previous->c_str() : "");
#else
  if (previous)
    setenv("GUFO_THINK_EOG_GUARD", previous->c_str(), 1);
  else
    unsetenv("GUFO_THINK_EOG_GUARD");
#endif
}
void RegistryNormalizationTest() {
  const auto strip = [](const gufo::json::Value& source, bool labels) {
    const std::set<std::string> optional = {"echo", "audience", "gui_control",
                                            "experimental", "restart_required"};
    auto row = gufo::json::Value::object();
    for (const auto& [key, value] : source.members())
      if (!optional.contains(key) && (labels || key != "label"))
        row[key] = value;
    return row;
  };
  auto start = passed;
  std::vector<gufo::json::Value> engine_rows;
  for (const auto& row : ReleaseSwitchTable())
    engine_rows.push_back(strip(row, true));
  const auto rows = NormalizeSwitchTable(engine_rows);
  ValidateSwitchTable(engine_rows);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    const auto& catalog = ReleaseSwitchTable()[i];
    Expect(row.find("allowed")->dump() == catalog.find("allowed")->dump(),
           "row retains its accepted values: " + row.member_str("name"));
    Expect(row.member_str("group") == catalog.member_str("group"),
           "row retains its chosen group");
    Expect(row.member_str("audience") == "debug" &&
               !row.find("gui_control")->as_bool() &&
               !row.find("experimental")->as_bool() &&
               !row.find("echo")->as_bool(),
           "engine-only row defaults to hidden debug metadata");
    Expect(row.find("restart_required")->as_bool() ==
               (row.member_str("class") != "REMOVED"),
           "removed switches do not require restart");
  }
  Expect(RenderReleaseInclude(engine_rows) == RenderReleaseInclude(rows),
         "include renderer applies the shared normalizer");
  Expect(RenderReleaseDocs(engine_rows) == RenderReleaseDocs(rows),
         "docs renderer applies the shared normalizer");
  std::cout << "PASS: RegistryMissingMetadata " << passed - start
            << " checks\n";

  start = passed;
  auto future = strip(ReleaseSwitchTable().front(), false);
  future["name"] = "--cache-future-engine-option";
  future["group"] = "Behaviour";
  future["requires"] = gufo::json::Value::array();
  future["suppressed_by"] = gufo::json::Value::array();
  future["inject_with"] = nullptr;
  std::vector<gufo::json::Value> new_rows = {future};
  ValidateSwitchTable(new_rows);
  const auto normalized = NormalizeSwitchTable(new_rows);
  Expect(normalized.front().member_str("group") == "Behaviour",
         "future engine switch accepts its row group independent of its name");
  Expect(RenderReleaseInclude(new_rows).find("--cache-future-engine-option") !=
             std::string::npos,
         "future engine switch generates with no GUI metadata");
  std::cout << "PASS: RegistryFutureEngineRow " << passed - start
            << " checks\n";

  start = passed;
  auto no_labels = ReleaseSwitchTable();
  for (auto& row : no_labels)
    row = strip(row, false);
  ValidateSwitchTable(no_labels);
  auto labeled = no_labels;
  labeled[0]["label"] = "New setting";
  ValidateSwitchTable(labeled);
  for (const auto& value : {gufo::json::Value(""), gufo::json::Value(" \t"),
                            gufo::json::Value(42), gufo::json::Value()}) {
    labeled[0]["label"] = value;
    Reject([&] { ValidateSwitchTable(labeled); },
           "invalid optional label refuses");
  }
  auto missing_core = gufo::json::Value::object();
  for (const auto& [key, value] : future.members())
    if (key != "purpose")
      missing_core[key] = value;
  new_rows = {missing_core};
  Reject([&] { ValidateSwitchTable(new_rows); },
         "missing engine metadata still refuses");
  std::cout << "PASS: RegistryOptionalLabelValidation " << passed - start
            << " checks\n";

  start = passed;
  std::size_t labels = 0;
  const auto settings = ReleaseSettingsJson(ResolveReleaseSettings(Inputs()));
  for (const auto& row : ReleaseSwitchTable()) {
    if (row.member_str("audience") != "user" &&
        row.member_str("audience") != "advanced")
      continue;
    ++labels;
    Expect(row.find("label") && !row.member_str("label").empty(),
           "user and advanced rows carry labels");
    const auto& effective = settings.find("effective_switches")->items();
    const auto found =
        std::find_if(effective.begin(), effective.end(), [&](const auto& item) {
          return item.member_str("name") == row.member_str("name");
        });
    Expect(found != effective.end() &&
               found->member_str("label") == row.member_str("label"),
           "effective switch emitter preserves registry label");
  }
  Expect(labels == 48, "all reviewed user and advanced labels are present");
  const auto runtime =
      std::find_if(ReleaseSwitchTable().begin(), ReleaseSwitchTable().end(),
                   [](const auto& row) {
                     return row.member_str("name") == "BUBO_RUNTIME_DIR";
                   });
  Expect(runtime != ReleaseSwitchTable().end() &&
             runtime->member_str("audience") == "debug" &&
             runtime->find("echo")->as_bool() &&
             !runtime->find("gui_control")->as_bool(),
         "runtime override is a complete debug row");
  std::cout << "PASS: RegistryLabelsAndRuntimeRow " << passed - start
            << " checks\n";
}
void DefaultSessionContextTest() {
  const auto start = passed;
  auto inputs = Inputs();
  inputs.native_context = 262144;
  bubo::cli::MemoryEstimateInputs memory;
  memory.package_id = std::string(bubo::prepare::kPackageId);
  memory.profile = "default";
  memory.context = *inputs.native_context;
  memory.sessions = 4;
  memory.mtp = memory.twin = memory.draft_head = memory.vision = true;
  memory.available_bytes = 102892924928ULL;
  memory.host_available_bytes = 11900000000ULL;
  memory.npu_prefill = true;
  std::string error;
  const auto estimate = bubo::cli::MemoryEstimate(memory, &error);
  Expect(estimate.is_object(), "context test uses the CLI estimator: " + error);
  std::uint64_t recommended = 0;
  for (const auto& row :
       estimate.find("recommended_context")->find("by_sessions")->items())
    if (row.member_size("sessions") == 4)
      recommended = row.member_size("context_per_session");
  Expect(recommended > 0 && recommended < *inputs.native_context,
         "four sessions have a smaller recommendation");
  auto single = ResolveReleaseSettings(inputs);
  Expect(!ResolveDefaultSessionContext(single, 1, estimate),
         "single session keeps the release context default");
  Expect(ReleaseSettingsJson(single).find("engine")->member_str("context") ==
             "200000",
         "single-session Flash Next context is 200000");
  std::cout << "PASS: DefaultContextSingleSession200000 2 checks\n";
  inputs.explicit_flags["--sessions"] = "4";
  auto four = ResolveReleaseSettings(inputs);
  Expect(ResolveDefaultSessionContext(four, 4, estimate) == recommended,
         "four-session default selects the estimator row");
  Expect(four.effective_flags.at("--context") == std::to_string(recommended) &&
             Find(four, "--context").origin == "recommended_context",
         "resolved context has its source");
  const auto output = ReleaseSettingsJson(four);
  bool reason_present = false;
  for (const auto& row : output.find("effective_switches")->items())
    if (row.member_str("name") == "--context")
      reason_present =
          row.member_str("source") == "default" &&
          row.member_str("reason").find("recommended_context.by_sessions") !=
              std::string::npos;
  Expect(reason_present && !four.logs.empty() &&
             four.logs.back().find(std::to_string(recommended)) !=
                 std::string::npos,
         "context reason reaches machine output and logs");
  std::cout << "PASS: DefaultContextFourSessionsRecommended 3 checks; context="
            << recommended << '\n';
  for (const auto* context : {"65536", "0"}) {
    inputs.explicit_flags["--context"] = context;
    auto explicit_context = ResolveReleaseSettings(inputs);
    Expect(!ResolveDefaultSessionContext(explicit_context, 4, estimate) &&
               explicit_context.effective_flags.at("--context") == context &&
               Find(explicit_context, "--context").origin == "flag",
           "an explicit context, including zero/native, is preserved");
  }
  std::cout << "PASS: DefaultContextExplicitOverrideUnchanged 2 checks\n";
  bool unsupported = false;
  try {
    ResolveDefaultSessionContext(four, 4, gufo::json::Value());
  } catch (const bubo::cli::ControlError& error) {
    unsupported = error.ExitCode() ==
                  static_cast<int>(bubo::cli::MachineExit::Unsupported);
  }
  Expect(unsupported, "missing recommendation returns typed unsupported");
  memory.available_bytes = 1;
  const auto short_estimate = bubo::cli::MemoryEstimate(memory, &error);
  Reject([&] { ResolveDefaultSessionContext(four, 4, short_estimate); },
         "a non-fitting recommendation fails before loading");
  auto no_margin = estimate;
  auto no_margin_rows = gufo::json::Value::array();
  for (auto row : no_margin["recommended_context"]["by_sessions"].items()) {
    if (row.member_size("sessions") == 4) {
      row["fits"] = true;
      row["recommended"] = false;
      row["reason"] = "Insufficient recommendation headroom.";
    }
    no_margin_rows.push_back(std::move(row));
  }
  no_margin["recommended_context"]["by_sessions"] = std::move(no_margin_rows);
  Reject(
      [&] { ResolveDefaultSessionContext(four, 4, no_margin); },
      "admission fit alone does not select an unrecommended default context");
  std::cout << "PASS: DefaultSessionContext " << passed - start << " checks\n";
}
void BuiltinDefaultDisplayTest() {
  const auto start = passed;
  for (const auto* profile : {"default", "subagents", "exact", "vanilla"}) {
    const auto resolved = ResolveReleaseSettings(Inputs(profile));
    const auto& floor = Find(resolved, "GUFO_CACHE_HOST_FLOOR_MB");
    Expect(!floor.value && floor.origin == "built-in",
           "host floor is not injected");
    Expect(std::none_of(resolved.injections.begin(), resolved.injections.end(),
                        [](const auto& row) {
                          return row.name == "GUFO_CACHE_HOST_FLOOR_MB";
                        }),
           "display defaults never become injections");
    const auto output = ReleaseSettingsJson(resolved);
    bool found = false;
    for (const auto& row : output.find("effective_switches")->items())
      if (row.member_str("name") == "GUFO_CACHE_HOST_FLOOR_MB") {
        found = row.member_str("effective") == "4096" &&
                row.member_str("source") == "default";
      }
    Expect(found, "effective host floor displays the engine built-in");
  }
  Expect(gufo::server::kDefaultHostFloorMiB == 4096,
         "independent engine host floor constant");
  std::cout << "PASS: BuiltinDefaultDisplay " << passed - start << " checks\n";
}
}  // namespace
static void StatusSummaryAndRejectionTest() {
  auto inputs = Inputs("default");
  const auto resolved = ResolveReleaseSettings(inputs);
  const auto settings = ReleaseSettingsJson(resolved);
  for (const auto& row : settings.find("effective_switches")->items()) {
    const auto summary = ReleaseSettingSummary(row);
    Expect(summary.size() == 5, "status setting retains five contract fields");
    for (const auto* key :
         {"requested", "effective", "source", "reason", "allowed"})
      Expect(summary.find(key)->dump() == row.find(key)->dump(),
             "status projection retains documented setting value");
  }
  for (const auto* name :
       {"--think-marker-bias", "--npu-prefill", "--sessions"}) {
    inputs.explicit_flags.clear();
    inputs.explicit_flags[name] = "not-a-value";
    bool rejected = false;
    try {
      ResolveReleaseSettings(inputs);
    } catch (const ReleaseSettingError& error) {
      rejected = error.switch_name == name;
    }
    Expect(rejected, "typed release rejection retains registry name");
  }
  const char* args[] = {"--think-marker-bias", "not-a-value"};
  const auto startup = PrepareReleaseSettings(args);
  Expect(
      startup.error_switch == std::optional<std::string>("--think-marker-bias"),
      "startup retains switch after catching typed rejection");
  const auto& rows = ReleaseSwitchTable();
  std::size_t secrets = 0;
  for (const auto& row : rows) {
    if (row.find("secret")->as_bool()) {
      ++secrets;
      Expect(!row.find("echo")->as_bool(), "secret cannot echo");
      Expect(!row.find("gui_control")->as_bool(),
             "secret cannot be GUI-writable");
    }
  }
  Expect(secrets == 4,
         "credential flag and environment secrets are registered");
}

void RegistryRecipe(const std::filesystem::path& path) {
  const auto recipe = gufo::json::parse(Read(path));
  const auto& rows = recipe.find("switches")->items();
  ValidateSwitchTable(rows);
  for (const auto& test : recipe.find("cases")->items()) {
    auto inputs = Inputs();
    const auto name = test.member_str("name");
    const auto value = test.member_str("value");
    if (const auto* direct = test.find("value_only");
        direct && direct->as_bool()) {
      const auto row = std::ranges::find_if(rows, [&](const auto& item) {
        return item.member_str("name") == name;
      });
      Expect(row != rows.end(), "value agreement row exists");
      if (test.find("accepted")->as_bool()) {
        ValidateSwitchValue(*row, value);
        Expect(true, "listed value agrees with C++ reader grammar");
      } else {
        Reject([&] { ValidateSwitchValue(*row, value); },
               "C++ rejects value outside the row contract");
      }
      continue;
    }
    if (name.starts_with("--"))
      inputs.explicit_flags[name] = value;
    else
      inputs.environment[name] = value;
    const auto resolve = [&] { return ResolveReleaseSettings(inputs, rows); };
    if (!test.find("accepted")->as_bool()) {
      Reject([&] { (void)resolve(); }, "recipe rejects " + name + "=" + value);
      continue;
    }
    const auto resolved = resolve();
    const auto settings = ReleaseSettingsJson(resolved, rows);
    Expect(settings.dump().find("effective_switches") != std::string::npos,
           "recipe emits effective settings through the production emitter");
    const auto& switches = settings.find("effective_switches")->items();
    const auto found = std::ranges::find_if(switches, [&](const auto& row) {
      return row.member_str("name") == name;
    });
    Expect(found != switches.end(), "recipe row reaches effective settings");
    Expect(found->find("allowed")->dump() ==
               std::ranges::find_if(rows,
                                    [&](const auto& row) {
                                      return row.member_str("name") == name;
                                    })
                   ->find("allowed")
                   ->dump(),
           "recipe row carries its own bounds to the emitter");
    if (const auto* secret = test.find("secret"); secret && secret->as_bool()) {
      Expect(settings.dump().find(value) == std::string::npos,
             "recipe credential is absent from every emitted field");
      for (const auto& log : resolved.logs)
        Expect(log.find(value) == std::string::npos,
               "recipe credential is absent from resolver logs");
    } else
      Expect(found->member_str("effective") == value,
             "recipe value reaches effective settings");
  }
  std::cout << "PASS: RegistryRecipe " << passed << " checks\n";
}

// `--speculative mtp` on a model whose file holds the MTP head needs no
// --mtp-model: no warning. A model without a head still gets the warning, and
// Flash-Next keeps requiring its own sidecar whatever the file says.
void InFileMtpWarningTest() {
  const auto warns = [](const ReleaseResolution& result) {
    return std::any_of(result.logs.begin(), result.logs.end(),
                       [](const auto& line) {
                         return line == "warning: --speculative: needs "
                                        "artefact:mtp";
                       });
  };
  for (const bool has_head : {false, true}) {
    ReleaseInputs inputs;
    inputs.profile = "default";
    inputs.model_family = "other";
    inputs.model_has_mtp = has_head;
    inputs.explicit_flags["--speculative"] = "mtp";
    const auto result = ResolveReleaseSettings(inputs);
    Expect(warns(result) != has_head,
           has_head ? "an in-file MTP head meets artefact:mtp: no warning"
                    : "no MTP head and no --mtp-model: the warning stays");
  }
  ReleaseInputs flash;
  flash.profile = "default";
  flash.model_family = "qwen3.8-flash-next";
  flash.model_has_mtp = true;
  flash.explicit_flags["--speculative"] = "mtp";
  Reject([&] { (void)ResolveReleaseSettings(flash); },
         "Flash-Next without --mtp-model is still refused, whatever the file "
         "says");
}
// ---- Gemma 4 MTP (item 7) -------------------------------------------------
// Flash Next's effective settings for every profile and for the controls the
// MTP rows touch. tests/cli/contract/flash_next_settings.golden.json holds the
// output of this function from before Gemma 4 shared those rows; the test
// compares it byte for byte. `--write-flash-baseline FILE` rewrites it.
std::string FlashNextBaseline() {
  using gufo::json::Value;
  const std::map<std::string, std::string> artefacts = {
      {"--mtp-model", "mtp.gguf"},
      {"--mmproj", "mmproj.gguf"},
      {"GUFO_DENSE_TWIN_MODEL", "twin.gguf"},
      {"GUFO_DRAFT_OUT_MODEL", "head.gguf"}};
  auto scenarios = Value::array();
  for (const char* profile : {"default", "subagents", "exact", "vanilla"})
    for (int scenario = 0; scenario < 6; ++scenario) {
      ReleaseInputs inputs;
      inputs.model_family = "qwen3.8-flash-next";
      inputs.profile = profile;
      inputs.manifest_present = true;
      inputs.native_context = 262144;
      if (scenario >= 1 && scenario <= 4)
        inputs.manifest_defaults = artefacts;
      if (scenario == 2)
        inputs.explicit_flags["--no-mtp"] = "1";
      if (scenario == 3)
        inputs.explicit_flags["--speculative"] = "off";
      if (scenario == 4)
        inputs.explicit_flags["--sessions"] = "2";
      if (scenario == 5)
        inputs.explicit_flags["--speculative"] = "mtp";
      auto entry = Value::object();
      entry["profile"] = profile;
      entry["scenario"] = scenario;
      try {
        const auto result = ResolveReleaseSettings(inputs);
        auto settings = Value::array();
        for (const auto& setting : result.settings) {
          if (!setting.value && setting.skipped_reason.empty())
            continue;  // rows with no value say nothing
          auto row = Value::object();
          row["name"] = setting.name;
          row["value"] = setting.value ? Value(*setting.value) : Value();
          row["origin"] = setting.origin;
          row["skipped"] = setting.skipped_reason;
          settings.push_back(std::move(row));
        }
        entry["settings"] = std::move(settings);
        auto flags = Value::object();
        for (const auto& [name, value] : result.effective_flags)
          flags[name] = value;
        entry["effective_flags"] = std::move(flags);
        auto logs = Value::array();
        for (const auto& line : result.logs)
          logs.push_back(line);
        entry["logs"] = std::move(logs);
        auto bound = Value::object();
        bound["present"] = result.artefacts.present;
        bound["mtp"] = result.artefacts.mtp;
        bound["twin"] = result.artefacts.twin;
        bound["draft_head"] = result.artefacts.draft_head;
        bound["mmproj"] = result.artefacts.mmproj;
        entry["artefacts"] = std::move(bound);
      } catch (const std::exception& error) {
        entry["error"] = std::string(error.what());
      }
      scenarios.push_back(std::move(entry));
    }
  return scenarios.dump() + "\n";
}
void FlashNextUnchangedTest() {
  const auto golden =
      Read(std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT) /
           "tests/cli/contract/flash_next_settings.golden.json");
  Expect(FlashNextBaseline() == golden,
         "Flash Next effective settings equal the recorded baseline");
}







void CacheRamDefaultTest() {
  const auto Row = [](const ReleaseInputs& inputs) {
    const auto result = ResolveReleaseSettings(inputs);
    const auto& row = Find(result, "--cache-ram-bytes");
    return std::pair{row.value, row.origin};
  };
  {
    const auto [value, origin] = Row(Inputs("default"));
    Expect(value && *value == "8589934592" && origin == "profile:default",
           "default profile: 8 GiB RAM cache from the profile");
  }
  {
    auto inputs = Inputs("default");
    inputs.explicit_flags["--sessions"] = "2";
    const auto [value, origin] = Row(inputs);
    Expect(value && *value == "8589934592" && origin == "profile:default",
           "default profile with --sessions 2: the budget stays 8 GiB (one "
           "shared cache)");
  }
  {
    const auto [value, origin] = Row(Inputs("subagents"));
    Expect(value && *value == "4294967296" && origin == "profile:subagents",
           "subagents profile keeps its 4 GiB RAM cache");
  }
  {
    auto inputs = Inputs("default");
    inputs.explicit_flags["--cache-ram-bytes"] = "6442450944";
    const auto [value, origin] = Row(inputs);
    Expect(value && *value == "6442450944" && origin == "flag",
           "an explicit --cache-ram-bytes wins over the profile");
  }
  {
    const auto [value, origin] = Row(Inputs("vanilla"));
    Expect(value && *value == "0" && origin == "built-in",
           "vanilla injects no RAM cache budget (built-in 0 = automatic)");
  }
}

// A Gemma 4 package whose files carry an mtp head binds it and decodes with
// MTP by default; without the head it stays off; --no-mtp wins.
void Gemma4MtpTest() {
  const auto resolve = [](const std::string& profile, bool head,
                          const std::string& flag = {}) {
    ReleaseInputs inputs;
    inputs.model_family = "gemma4";
    inputs.architecture = "gemma4";
    inputs.profile = profile;
    inputs.manifest_present = true;
    inputs.manifest_defaults["--mmproj"] = "mmproj.gguf";
    if (head)
      inputs.manifest_defaults["--mtp-model"] = "head.gguf";
    if (!flag.empty())
      inputs.explicit_flags[flag] = "1";
    return ResolveReleaseSettings(inputs);
  };
  for (const char* profile : {"default", "subagents"}) {
    const auto with_head = resolve(profile, true);
    Expect(Find(with_head, "--speculative").value ==
                   std::optional<std::string>("mtp") &&
               Find(with_head, "--mtp-model").value ==
                   std::optional<std::string>("head.gguf") &&
               Find(with_head, "--mtp-model").origin == "manifest" &&
               with_head.artefacts.mtp == "head.gguf",
           std::string("Gemma 4 with a head decodes with MTP: ") + profile);
    const auto without = resolve(profile, false);
    const auto speculative = Find(without, "--speculative").value;
    Expect((!speculative || *speculative == "off") &&
               !Find(without, "--mtp-model").value &&
               without.artefacts.mtp.empty(),
           std::string("Gemma 4 without a head stays off: ") + profile);
    const auto muted = resolve(profile, true, "--no-mtp");
    Expect(Find(muted, "--speculative").value ==
                   std::optional<std::string>("off") &&
               !Find(muted, "--mtp-model").value && muted.artefacts.mtp.empty(),
           std::string("--no-mtp wins over the head: ") + profile);
  }
  for (const char* profile : {"vanilla", "exact"}) {
    const auto result = resolve(profile, true);
    const auto speculative = Find(result, "--speculative").value;
    Expect(!speculative || *speculative != "mtp",
           std::string("profile ") + profile + " keeps Gemma 4 off MTP");
  }
  // The projector binding is the same as before.
  Expect(Find(resolve("default", true), "--mmproj").value ==
             std::optional<std::string>("mmproj.gguf"),
         "the Gemma 4 projector still binds");
  // Another family with a head file is not given MTP by this rule.
  ReleaseInputs other;
  other.model_family = "qwen3.8-27b";
  other.profile = "default";
  other.manifest_present = true;
  other.manifest_defaults["--mtp-model"] = "head.gguf";
  const auto other_result = ResolveReleaseSettings(other);
  Expect(Find(other_result, "--speculative").value !=
             std::optional<std::string>("mtp"),
         "only Flash Next and Gemma 4 take MTP by default");
}

// Item 9: the typed artefact rows say what they need, as --mmproj and
// --mtp-model already do.
void RegistryRequiresTest() {
  const auto row_of = [](const std::string& name) {
    for (const auto& row : ReleaseSwitchTable())
      if (row.member_str("name") == name)
        return row;
    throw std::runtime_error("missing registry row " + name);
  };
  const auto requires_of = [&](const std::string& name) {
    std::vector<std::string> out;
    const auto row = row_of(name);
    for (const auto& item : row.find("requires")->items())
      out.push_back(item.str());
    return out;
  };
  Expect(requires_of("--twin-model") ==
             std::vector<std::string>{"artefact:twin"},
         "--twin-model requires its twin file");
  Expect(requires_of("--draft-head-model") ==
             std::vector<std::string>{"artefact:draft_head"},
         "--draft-head-model requires its draft head file");
}
// Item 10: user and advanced rows use plain words, with no internal names and
// no measurement notes.
void RegistryWordingTest() {
  for (const auto& row : ReleaseSwitchTable()) {
    if (row.member_str("audience") == "debug")
      continue;
    for (const char* key : {"purpose", "note", "default_text"}) {
      auto text = row.member_str(key);
      std::transform(text.begin(), text.end(), text.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      for (const char* word : {"sidecar", "unmeasured", "not measured"})
        Expect(text.find(word) == std::string::npos,
               row.member_str("name") + " " + key + " says '" + word + "'");
    }
  }
}

// Decode's policy (2026-10-20): MTP is on by default at any --sessions, and an
// explicit --speculative off wins. The row's help says what the gain is.
void Gemma4MtpPolicyTest() {
  const auto resolve = [](const std::map<std::string, std::string>& flags) {
    ReleaseInputs inputs;
    inputs.model_family = "gemma4";
    inputs.architecture = "gemma4";
    inputs.profile = "default";
    inputs.manifest_present = true;
    inputs.manifest_defaults["--mtp-model"] = "head.gguf";
    inputs.explicit_flags = flags;
    return ResolveReleaseSettings(inputs);
  };
  for (const char* sessions : {"1", "2", "4"})
    Expect(Find(resolve({{"--sessions", sessions}}), "--speculative").value ==
               std::optional<std::string>("mtp"),
           std::string("Gemma 4 decodes with MTP at sessions=") + sessions);
  const auto off = resolve({{"--speculative", "off"}});
  Expect(Find(off, "--speculative").value == std::optional<std::string>("off"),
         "an explicit --speculative off wins");
  for (const auto& row : ReleaseSwitchTable())
    if (row.member_str("name") == "--speculative") {
      Expect(row.member_str("note").find(
                 "Speeds up replies for a single request; with two or more "
                 "sessions the gain is small.") != std::string::npos,
             "--speculative help names the single-request gain");
      Expect(row.member_str("note").find("percent") == std::string::npos &&
                 row.member_str("note").find("%") == std::string::npos,
             "--speculative help carries no speed number");
    }
}

// Item 11: the API model name defaults to the product name. An explicit
// --served-model-name wins. /v1/models lists the backend's model id, which
// serve sets from this name, so it reports the served name.
void ServedModelNameTest() {
  Expect(ServedModelAlias() == "bubo", "the compatibility alias stays bubo");
  Expect(AcceptsServedModel("real-model", "real-model"),
         "the selected served model is accepted");
  Expect(AcceptsServedModel(ServedModelAlias(), "real-model"),
         "the compatibility alias is accepted");
  Expect(!AcceptsServedModel("wrong-model", "real-model"),
         "other model names are refused");
  Expect(!AcceptsServedModel("Bubo", "real-model"),
         "model identity remains case sensitive");
  ModelPackage package;
  package.manifest_path = "fixture/bubo-package.json";
  package.package_id = "huihui-qwen38-flash-next-ud-q4-k-xl-v1";
  Expect(
      ServedModelName("", "fixture/model.gguf", package) == package.package_id,
      "a package serves its real id");
  Expect(ServedModelName("", "folder/plain-model.gguf") == "plain-model",
         "a plain GGUF serves its file name without the extension");
  Expect(ServedModelName("mine", "folder/model.gguf", package) == "mine",
         "an explicit served name wins over the package id");
  const auto plain = ResolveReleaseSettings(Inputs());
  Expect(!Find(plain, "--served-model-name").value,
         "the registry leaves the served name for model selection");
  auto named = Inputs();
  named.explicit_flags["--served-model-name"] = "mine";
  Expect(Find(ResolveReleaseSettings(named), "--served-model-name").value ==
             std::optional<std::string>("mine"),
         "an explicit --served-model-name wins in the settings");
  for (const auto& row : ReleaseSwitchTable())
    if (row.member_str("name") == "--served-model-name")
      Expect(row.member_str("default_text") == "Package ID or GGUF file name",
             "the help names the default: " + row.member_str("default_text"));
}

void V12SettingsTest() {
  auto canonical = Inputs("vanilla");
  auto alias = canonical;
  alias.profile = "stock";
  const auto canonical_settings = ResolveReleaseSettings(canonical);
  const auto alias_settings = ResolveReleaseSettings(alias);
  Expect(alias_settings.profile == "vanilla" &&
             ReleaseSettingsJson(canonical_settings).dump() ==
                 ReleaseSettingsJson(alias_settings).dump(),
         "stock aliases vanilla with identical effective settings");
  const std::array<const char*, 4> alias_args{"llm", "--profile", "stock",
                                              "--print-effective-settings"};
  const auto parsed_alias = PrepareReleaseSettings(alias_args, {}, true);
  Expect(
      parsed_alias.error.empty() && parsed_alias.resolved.profile == "vanilla",
      "the CLI parser accepts stock and resolves vanilla");
  const auto profiles = bubo::cli::SettingsProfiles();
  Expect(
      profiles.items()[2].member_str("name") == "vanilla" &&
          profiles.items()[2].member_str("label") == "Vanilla (no Bubo tuning)",
      "GUI profile contract names vanilla");
  auto flash = Inputs();
  flash.native_context = 262144;
  Expect(Find(ResolveReleaseSettings(flash), "--context").value ==
             std::optional<std::string>("200000"),
         "Flash Next default context is 200000");
  flash.explicit_flags["--context"] = "262144";
  Expect(Find(ResolveReleaseSettings(flash), "--context").value ==
             std::optional<std::string>("262144"),
         "Flash Next explicit native maximum remains available");
  flash.explicit_flags.clear();
  flash.model_family = "gemma4";
  Expect(Find(ResolveReleaseSettings(flash), "--context").value ==
             std::optional<std::string>("0"),
         "other model families retain their native context default");
  const std::set<std::string> debug = {
      "GUFO_DRAFT_OUT_MODEL",       "GUFO_EXPERT_V2K",
      "GUFO_ROUTED_WMMA_IQ",        "GUFO_IQ_ROUTE_MIN_ROWS",
      "GUFO_ROUTED_LANEROW",        "GUFO_LENIENT",
      "GUFO_LENIENT_LAMBDA",        "GUFO_INCREMENTAL_TOKENIZE",
      "GUFO_PREFILL_FORWARD_TOKENS"};
  const std::map<std::string, std::string> labels = {
      {"--top-k", "Top K (sampling candidate limit)"},
      {"--top-p", "Top P (cumulative probability filter)"},
      {"--min-p", "Min P (relative probability filter)"},
      {"--repeat-penalty", "Repeat Penalty (repetition penalty)"},
      {"--repeat-last-n", "Repeat Last N (repetition window)"},
      {"--presence-penalty", "Presence Penalty (repeated token penalty)"},
      {"--frequency-penalty", "Frequency Penalty (token frequency penalty)"},
      {"--max-tokens", "Max Tokens (new tokens per request)"},
      {"--seed", "Seed (random seed)"},
      {"--min-keep", "Min Keep (minimum sampling candidates)"}};
  const std::set<std::string> unbounded = {
      "--cache-disk-bytes", "--cache-disk-staging-bytes",
      "--cache-ram-bytes",  "--context",
      "--sessions",         "GUFO_MAX_IMAGE_TOKENS",
      "--max-connections",  "--max-request-bytes",
      "--prefill-chunk"};
  for (const auto& row : ReleaseSwitchTable()) {
    const auto name = row.member_str("name");
    if (debug.contains(name))
      Expect(row.member_str("audience") == "debug",
             name + " uses the debug tier");
    if (labels.contains(name))
      Expect(row.member_str("label") == labels.at(name),
             name + " has its approved label");
    if (unbounded.contains(name)) {
      Expect(!row.find("allowed")->find("maximum"),
             name + " omits its unlimited maximum");
      ValidateSwitchValue(row, "18446744073709551615");
      Expect(true, name + " retains uint64 maximum parsing");
      Reject([&] { ValidateSwitchValue(row, "18446744073709551616"); },
             name + " still rejects uint64 overflow");
    }
    if (const auto* maximum = row.find("allowed")->find("maximum"))
      Expect(maximum->as_double() <= 9007199254740991.0,
             name + " has a representable maximum");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--write-flash-baseline") {
      std::ofstream(argv[2], std::ios::binary) << FlashNextBaseline();
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--registry-recipe") {
      RegistryRecipe(argv[2]);
      return 0;
    }
    BuiltinDefaultDisplayTest();
    StatusSummaryAndRejectionTest();
    if (argc > 1 && std::string_view(argv[1]) == "--registry") {
      RegistryNormalizationTest();
      std::cout << "PASS: registry " << passed << " checks\n";
      return 0;
    }
    // CPU-only echo harness uses the same startup resolver as RunServe.
    if (argc > 1 && std::string_view(argv[1]) == "--print-effective-settings") {
      const auto startup = PrepareReleaseSettings(std::span<const char* const>(
          argv + 1, static_cast<std::size_t>(argc - 1)));
      if (!startup.error.empty())
        throw std::runtime_error(startup.error);
      std::cout << ReleaseSettingsJson(startup.resolved).dump() << '\n';
      return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--audit-catalog") {
      AuditCatalog();
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--review2-env") {
      auto inputs = Inputs("exact");
      inputs.environment[argv[2]] = "1";
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact rejects output control: " + std::string(argv[2]));
      std::cout << "PASS: exact rejects " << argv[2] << '\n';
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--review2") {
      Review2Regression(std::stoi(argv[2]));
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--regression") {
      ReviewRegression(std::stoi(argv[2]));
      return 0;
    }
    for (const int finding : {1, 2, 3, 4, 5, 6, 7, 8, 10, 16})
      ReviewRegression(finding);
    for (int finding = 1; finding <= 8; ++finding)
      Review2Regression(finding);
    NpuPolicyTest();
    RegistryNormalizationTest();
    TestResolver();
    DefaultSessionContextTest();
    HeaderTest();
    InFileMtpWarningTest();
    Gemma4MtpTest();
    Gemma4MtpPolicyTest();
    ServedModelNameTest();
    V12SettingsTest();
    RegistryRequiresTest();
    RegistryWordingTest();
    FlashNextUnchangedTest();
    CacheRamDefaultTest();
    AuditCatalog();
    if (argc > 2 && std::string_view(argv[1]) == "--production-spec")
      CheckSpec(argv[2]);
    std::cout << "PASS: release_profile_test " << passed << " checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
