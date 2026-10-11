#include "src/cli/serve/release_profile.hpp"

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

#include "src/cli/model_families.hpp"
#include "src/cli/serve/debug_settings.hpp"

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
    // The full resolver test checks all nine rows and every unchanged row.
  } else if (finding == 3) {
    for (const auto* name :
         {"GUFO_EVAL_API_KEY", "GUFO_EVAL_ARTIFACT_ID",
          "GUFO_EVAL_MODEL_REVISION", "GUFO_EVAL_SERVER_COMMIT",
          "GUFO_EVAL_SERVER_CONFIG", "GUFO_EVAL_SERVER_LABEL", "PATH", "TEMP"})
      Expect(!DebugSettingsListsEnvironment(name),
             "echo excludes unsafe row: " + std::string(name));
    for (const auto* name :
         {"GUFO_EVAL_API_KEY", "GUFO_EVAL_ARTIFACT_ID",
          "GUFO_EVAL_MODEL_REVISION", "GUFO_EVAL_SERVER_COMMIT",
          "GUFO_EVAL_SERVER_CONFIG", "GUFO_EVAL_SERVER_LABEL"})
      (void)Find(ResolveReleaseSettings(Inputs("stock")), name);
    const auto settings =
        BuildEffectiveSettings(gufo::json::Value::object(), "v", "r", "b");
    Expect(!settings.find("environment")->contains("GUFO_EVAL_API_KEY"),
           "debug echo never contains a secret row");
  } else if (finding == 4) {
    const auto text = Read(std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT) /
                           "cmake/Checks.cmake");
    Expect(text.substr(0, text.find("set(gufo_pr_tests"))
                   .find("release_profile_test") != std::string::npos,
           "check-pr builds and runs release_profile_test");
  } else if (finding == 5) {
    for (const auto* profile : {"stock", "default", "exact"}) {
      auto inputs = Inputs(profile);
      if (std::string_view(profile) != "stock")
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
    ApplyInjections(ResolveReleaseSettings(Inputs("stock")));
    Expect(!SnapshotReleaseEnvironment().at("GUFO_THINK_GUARD_ESCALATE"),
           "empty real catalog value is removed in stock");
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
      Reject([&] { (void)ResolveReleaseSettings(inputs); },
             "exact rejects re-enabling: " + name);
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
    gufo::cli::ArgParser parser("test");
    bool disabled = false;
    parser.AddFlag("", "--no-cache", "Disable caches", "Cache", &disabled);
    PrefillReleaseFlags(exact, parser);
    Expect(disabled, "no-cache effective setting reaches the parser sink");
    Expect(Find(ResolveReleaseSettings(Inputs()), "--no-cache").origin ==
                   "built-in" &&
               Find(ResolveReleaseSettings(Inputs("stock")), "--no-cache")
                       .origin == "built-in",
           "stock and default do not inject no-cache");
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
           "exact uses the stock routed kernel");
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
    const auto result = ResolveReleaseSettings(inputs);
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
    Expect(docs.find("--profile stock --speculative off --no-cache --sessions "
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
                         "except for the exact seed.") != std::string::npos,
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
        "GUFO_ROUTED_LANEROW", "GUFO_EOG_MIN_PROB", "GUFO_EOG_DEFER_P"}) {
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
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "exact refuses explicit approximation");
  }
  inputs = Inputs("exact");
  inputs.environment["GUFO_PREFILL_FORWARD_TOKENS"] = "2048";
  Reject([&] { (void)ResolveReleaseSettings(inputs); },
         "exact refuses a different prefill geometry");
  const std::map<std::string, std::optional<std::string>> off = {
      {"GUFO_THINK_GUARD_ESCALATE", "0"},
      {"GUFO_THINK_EOG_GUARD", "0"},
      {"GUFO_EOG_BOUNDARY", "0"},
      {"GUFO_EOG_MIN_PROB", "-1"},
      {"GUFO_EOG_DEFER_P", "0"},
      {"GUFO_SPEC_EOG_STOP", "0"},
      {"GUFO_LOOP_BREAKER", "0"},
      {"GUFO_LOOP_BREAKER_EOG_CLOSE", "0"},
      {"GUFO_THINK_MARKER_BIAS_BY_EFFORT", std::nullopt}};
  inputs = Inputs();
  inputs.explicit_flags["--no-behaviour"] = "1";
  auto disabled = ResolveReleaseSettings(inputs);
  for (const auto& setting : baseline.settings) {
    if (off.contains(setting.name))
      Expect(Find(disabled, setting.name).value == off.at(setting.name),
             "behaviour off: " + setting.name);
    else if (setting.name != "--no-behaviour")
      Expect(Find(disabled, setting.name).value == setting.value,
             "no-behaviour changes only nine rows");
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
    for (const auto* profile : {"default", "subagents", "stock", "exact"}) {
      for (const bool suppressed : {false, true}) {
        auto upstream_inputs = Inputs("stock");
        upstream_inputs.model_family = family;
        if (suppressed)
          upstream_inputs.explicit_flags["--no-behaviour"] = "1";
        const auto upstream = ResolveReleaseSettings(upstream_inputs);
        auto candidate_inputs = upstream_inputs;
        candidate_inputs.profile = profile;
        const auto candidate = ResolveReleaseSettings(candidate_inputs);
        if (family == "qwen3.8-flash-next" &&
            std::string_view(profile) != "stock") {
          Expect(!candidate.injections.empty() ||
                     candidate.effective_flags != upstream.effective_flags,
                 "Flash Next retains its tuned profile");
          continue;
        }
        auto flags = upstream.effective_flags;
        const bool exact = std::string_view(profile) == "exact";
        if (exact)
          flags["--seed"] = "1";
        Expect(candidate.effective_flags == flags,
               "upstream flag diff: " + std::string(family) + "/" + profile);
        Expect(candidate.injections.empty(), "upstream env injections empty");
        for (const auto& setting : candidate.settings) {
          const auto& stock = Find(upstream, setting.name);
          Expect(setting.value == (exact && setting.name == "--seed"
                                       ? std::optional<std::string>("1")
                                       : stock.value),
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
                            std::string_view(profile) != "stock"
                        ? " tuned defaults retained"
                    : std::string_view(profile) == "exact"
                        ? " diff=seed:1 only; message checked"
                        : " diff=EMPTY")
                << " (with/without --no-behaviour)\n";
    }
  }
  Expect(ResolveReleaseSettings(Inputs("stock")).injections.empty(),
         "stock injects nothing");
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
}  // namespace
int main(int argc, char** argv) {
  try {
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
    TestResolver();
    HeaderTest();
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
