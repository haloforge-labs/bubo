#include "src/cli/serve/release_profile.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

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
  std::cout << "PASS: production comparison " << path.filename().string()
            << " (site/manifest keys excluded)\n";
}

void AuditCatalog() {
  const auto root = std::filesystem::path(GUFO_RELEASE_SOURCE_ROOT);
  std::set<std::string> names, missing;
  for (const auto& row : ReleaseSwitchTable())
    names.insert(row.member_str("name"));
  const std::regex literal(R"audit("(GUFO_[A-Z0-9_]+)")audit");
  for (const auto& file :
       std::filesystem::recursive_directory_iterator(root / "src")) {
    const auto ext = file.path().extension().string();
    if (!file.is_regular_file() ||
        (ext != ".cpp" && ext != ".hpp" && ext != ".hip" && ext != ".h" &&
         ext != ".cc" && ext != ".c"))
      continue;
    const auto text = Read(file.path());
    for (std::sregex_iterator match(text.begin(), text.end(), literal), end;
         match != end; ++match)
      if (!names.contains((*match)[1].str()))
        missing.insert((*match)[1].str());
  }
  for (const auto& name : missing)
    std::cerr << "MISSING: " << name << '\n';
  Expect(missing.empty(), "source literal catalog completeness");
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
    inputs.explicit_flags["GUFO_LENIENT"] = "bounded";
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "exact refuses a flag that enables an approximation");
    inputs = Inputs("exact");
    inputs.manifest_defaults["GUFO_DRAFT_OUT_MODEL"] = "draft-head.gguf";
    Expect(
        Find(ResolveReleaseSettings(inputs), "GUFO_DRAFT_OUT_MODEL").origin ==
            "manifest",
        "exact keeps the draft-only head");
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
  auto inputs = Inputs();
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
      Expect(name == "GUFO_LENIENT"
                 ? Find(exact, name).value == std::optional<std::string>("off")
                 : !Find(exact, name).value,
             "exact off value");
    }
  Expect(Find(exact, "GUFO_THINK_EOG_GUARD").value ==
             std::optional<std::string>("1"),
         "exact keeps behaviour");
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
  Expect(Find(ResolveReleaseSettings(inputs), "GUFO_PREFILL_FORWARD_TOKENS")
                 .origin == "env",
         "real pin wins");
  const std::map<std::string, std::optional<std::string>> off = {
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
             "no-behaviour changes only eight rows");
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
  inputs = Inputs();
  inputs.model_family = "other";
  Expect(ResolveReleaseSettings(inputs).injections.empty(),
         "family mismatch injects nothing");
  inputs.explicit_flags["--no-behaviour"] = "1";
  auto other_disabled = ResolveReleaseSettings(inputs);
  Expect(other_disabled.injections.empty(),
         "suppression cannot inject another model's defaults");
  Expect(std::none_of(other_disabled.settings.begin(),
                      other_disabled.settings.end(),
                      [](const auto& row) {
                        return row.skipped_reason.starts_with("skipped: needs");
                      }),
         "family mismatch has no missing-artifact defaults");
  for (const auto* profile : {"subagents", "exact", "stock"}) {
    auto other = Inputs(profile);
    other.model_family = "other";
    Expect(ResolveReleaseSettings(other).injections.empty(),
           "every profile leaves another family unchanged");
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
           {"GUFO_THINK_EOG_GUARD", "01"},
           {"GUFO_EOG_MIN_PROB", "2"},
           {"GUFO_EOG_DEFER_P", "nan"},
           {"GUFO_LENIENT_LAMBDA", "0"},
           {"GUFO_DISK_EVICT_SKIP_TOKENS", "1000001"},
           {"GUFO_DISK_IDLE_MS", "600001"},
           {"GUFO_DISK_DEFER_MS", "-1"},
           {"GUFO_THINK_MARKER_BIAS_BY_EFFORT", "low:inf"}}) {
    inputs = Inputs();
    inputs.environment[name] = value;
    Reject([&] { (void)ResolveReleaseSettings(inputs); },
           "invalid real value refuses");
  }
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
