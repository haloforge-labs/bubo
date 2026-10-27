#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "src/cli/prepare/model_prepare.hpp"
#include "src/cli/prepare/runtime_prepare.hpp"
#include "src/cli/serve/model_manifest.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "tests/prepare/zip_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
using namespace bubo::prepare::test;
using namespace gufo::server;
void Check(bool ok, std::string text) {
  if (!ok)
    throw std::runtime_error(std::move(text));
}
std::string Read(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  Check(bool(file), "read fixture " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}
std::string Hash(std::string_view text) {
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}
void String(std::string& bytes, std::string_view value) {
  Put(bytes, value.size(), 8);
  bytes += value;
}
std::string Gguf(unsigned index) {
  std::string bytes;
  Put(bytes, 0x46554747, 4);
  Put(bytes, 3, 4);
  Put(bytes, 1, 8);
  Put(bytes, index < 4 ? 4 : 1, 8);
  String(bytes, "general.architecture");
  Put(bytes, 8, 4);
  String(bytes, "qwen4exp");
  if (index < 4) {
    String(bytes, "split.count");
    Put(bytes, 4, 4);
    Put(bytes, 4, 4);
    String(bytes, "split.no");
    Put(bytes, 4, 4);
    Put(bytes, index, 4);
    String(bytes, "split.tensors.count");
    Put(bytes, 10, 4);
    Put(bytes, 4, 8);
  }
  String(bytes, "fixture.tensor");
  Put(bytes, 1, 4);
  Put(bytes, 16, 8);
  Put(bytes, 0, 4);
  Put(bytes, 0, 8);
  while (bytes.size() % 32)
    bytes.push_back(0);
  bytes.append(64, static_cast<char>('A' + index));
  return bytes;
}
// A local repo mirror backed by the same Transport interface as curl. No
// sockets, DNS, AMD DLL loading, engine creation or model launch takes place.
class Mirror final : public Transport {
public:
  std::map<std::string, fs::path> files;
  std::function<void()> before_request;
  std::function<void()> on_request;
  unsigned requests = 0;
  TransferResult Perform(const Request& request, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    if (before_request)
      before_request();
    if (on_request)
      on_request();
    ++requests;
    const auto found = files.find(request.url);
    Check(found != files.end(), "unknown immutable fixture origin");
    const auto bytes = Read(found->second);
    if (headers({200,
                 {{"content-length", std::to_string(bytes.size())},
                  {"x-linked-size", std::to_string(bytes.size())},
                  {"x-linked-etag", Hash(bytes)},
                  {"etag", "\"fixture-v1\""}}}))
      body({reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
    return {};
  }
};
struct Result {
  int code;
  std::string out, err;
};
Result PrepareAll(const Options& options, const ModelTestEnvironment& model,
                  const RuntimeTestEnvironment& runtime, Mirror& mirror,
                  std::string answer = "") {
  std::istringstream input(answer);
  std::ostringstream out, err;
  mirror.before_request = [&] {
    Check(out.str().find("Approve the complete runtime and model plan?") !=
              std::string::npos,
          "no request before the complete all-step disclosure");
    Check(
        out.str().find("Other AMD terms can apply") != std::string::npos &&
            out.str().find("Bubo will not build a model") != std::string::npos,
        "both runtime/model disclosures precede requests");
  };
  const PreparePhase model_phase =
      [&](const Options& o, std::istream& in, std::ostream& output,
          std::ostream& errors, PreparationPlan* plan, bool approved) {
        return RunModelForTesting(o, model, in, output, errors, plan, approved);
      };
  const PreparePhase runtime_phase = [&](const Options& o, std::istream& in,
                                         std::ostream& output,
                                         std::ostream& errors,
                                         PreparationPlan* plan, bool approved) {
    return RunRuntimeForTesting(o, runtime, in, output, errors, plan, approved);
  };
  const auto code =
      RunAll(options, model_phase, runtime_phase, input, out, err);
  mirror.before_request = {};
  return {code, out.str(), err.str()};
}
const ReleaseSetting& Setting(const ReleaseResolution& settings,
                              std::string_view name) {
  for (const auto& setting : settings.settings)
    if (setting.name == name)
      return setting;
  throw std::runtime_error("setting absent: " + std::string(name));
}
template<class F>
void Reject(F operation, std::string_view named, std::string message) {
  try {
    operation();
  } catch (const std::exception& error) {
    Check(std::string_view(error.what()).find(named) != std::string_view::npos,
          message + ": wrong diagnostic: " + error.what());
    return;
  }
  throw std::runtime_error(message + ": accepted failing variant");
}
std::size_t Count(std::string_view text, std::string_view needle) {
  std::size_t result = 0, pos = 0;
  while ((pos = text.find(needle, pos)) != text.npos) {
    ++result;
    pos += needle.size();
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto root = fs::absolute(
        argc > 1 ? fs::u8path(argv[1]) : fs::path("j4-integration-fixture"));
    Check(!fs::exists(root), "fresh integration fixture root required");
    fs::create_directories(root);
    const auto original = ModelCatalogue();
    std::array<Entry, 8> entries;
    std::array<std::string, 8> hashes;
    std::array<FingerprintTable, 4> fingerprints;
    Mirror mirror;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      entries[i] = original[i];
      entries[i].source_repo = i < 4   ? "fixture/huihui"
                               : i < 6 ? "fixture/unsloth"
                                       : "fixture/bubo";
      entries[i].commit = "1111111111111111111111111111111111111111";
      const auto bytes = Gguf(static_cast<unsigned>(i));
      hashes[i] = Hash(bytes);
      entries[i].sha256 = hashes[i];
      entries[i].size = bytes.size();
      const auto file = root / "repo" / fs::u8path(entries[i].source_repo) /
                        fs::u8path(entries[i].repo_path);
      WriteFixture(file, bytes);
      mirror.files[OriginUrl(entries[i])] = file;
      if (i < 4) {
        const auto generated = GenerateGgufFingerprint(file, entries[i]);
        Check(bool(generated), "valid fixture GGUF fingerprint");
        fingerprints[i] = generated.table;
      }
    }
    ModelTestEnvironment model;
    model.entries = entries;
    model.transport = &mirror;
    for (const auto* name : {"LICENSE", "NOTICE", "PROVENANCE.md"}) {
      const std::string text =
          "Synthetic integration notice; no legal approval.\n";
      model.notices.push_back({name, text, Hash(text)});
    }
    model.volume = [](const fs::path&) {
      return VolumeInfo{
          "fixture-local", 16ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
    };
    model.fingerprint = [&](const Entry& entry, const fs::path& path) {
      const auto index = static_cast<std::size_t>(&entry - entries.data());
      Check(index < 4, "fingerprint called only for main shards");
      return CheckGgufFingerprint(path, entry, fingerprints[index]);
    };
    const std::array<fs::path, 3> wheel_paths{root / "repo/amd/core.whl",
                                              root / "repo/amd/libs.whl",
                                              root / "repo/amd/device.whl"};
    WriteFixture(
        wheel_paths[0],
        FixtureZip({{"_rocm_sdk_core/bin/amdhip64_7.dll", "abc", true}}));
    WriteFixture(wheel_paths[1],
                 FixtureZip({{"_rocm_sdk_libraries/bin/rocblas.dll"}}));
    WriteFixture(
        wheel_paths[2],
        FixtureZip({{"_rocm_sdk_libraries/bin/rocblas/library/gfx1151/a.co"}},
                   true));
    RuntimeTestEnvironment runtime;
    runtime.wheels = RuntimeWheels();
    runtime.members = GenerateRuntimeTable(wheel_paths, false);
    runtime.transport = &mirror;
    runtime.synthetic_disclosure = true;
    runtime.download_options.policy.source = HostPolicy::Source::Amd;
    runtime.volume = model.volume;
    runtime.holders = [] { return std::vector<std::string>{}; };
    const std::string notice = "Synthetic runtime notice; no legal approval.\n";
    const std::string notice_hash = Hash(notice);
    runtime.notices.push_back({"NOTICE-windows.md", notice_hash, notice});
    std::array<std::string, 3> wheel_hashes;
    for (unsigned i = 0; i < 3; ++i) {
      wheel_hashes[i] = HashFile(wheel_paths[i]);
      runtime.wheels[i].sha256 = wheel_hashes[i];
      runtime.wheels[i].size = fs::file_size(wheel_paths[i]);
      mirror.files[std::string(runtime.wheels[i].url)] = wheel_paths[i];
    }
    Options options;
    options.root = (root / "prepared").string();
    options.runtime_dir = (root / "runtime").string();
    options.cache_dir = (root / "cache").string();
    const auto declined = PrepareAll(options, model, runtime, mirror, "n\n");
    Check(declined.code == 3 && mirror.requests == 0 &&
              !fs::exists(options.root) && !fs::exists(options.runtime_dir) &&
              !fs::exists(options.cache_dir),
          "all-step decline has zero transport/payload/lock writes");
    auto missing_model = model;
    missing_model.notices.clear();
    options.yes = true;
    const auto gated = PrepareAll(options, missing_model, runtime, mirror);
    Check(gated.code == 4 && mirror.requests == 0 &&
              !fs::exists(options.runtime_dir),
          "OPEN model gate cannot partially install runtime");
    auto tight_runtime = runtime;
    tight_runtime.volume = [](const fs::path&) {
      return VolumeInfo{"fixture-local",
                        2ULL * 1024 * 1024 * 1024 + 5 * 1024 * 1024,
                        "NTFS",
                        false,
                        {}};
    };
    const auto tight = PrepareAll(options, model, tight_runtime, mirror);
    Check(tight.code == 7 && mirror.requests == 0 &&
              tight.err.find("Not enough free space on ") != tight.err.npos,
          "combined allocation rejects individually sufficient same volume");
    options.yes = false;
    const auto prepared = PrepareAll(options, model, runtime, mirror, "y\n");
    Check(prepared.code == 0, "prepare all: " + prepared.err);
    Check(mirror.requests == 11 &&
              Count(prepared.out,
                    "Approve the complete runtime and model plan?") == 1,
          "all eleven fixed inputs with one input approval");
    // One answer is read, so one question is printed.
    Check(Count(prepared.out, "[y/N]") == 1,
          "bare prepare prints one question: " +
              std::to_string(Count(prepared.out, "[y/N]")));
    const auto model_root = fs::u8path(ModelRoot(options));
    const auto manifest_path = model_root / "bubo-package.json";
    const auto manifest_text = Read(manifest_path);
    for (const auto& e : entries)
      Check(HashFile(model_root / fs::u8path(e.destination)) == e.sha256,
            "prepared payload matches independent fixture pin");
    Check(fs::exists(fs::u8path(options.runtime_dir) /
                     ".bubo/runtime-ready.json"),
          "runtime ready precedes model package readiness");
    const auto runtime_root = fs::u8path(options.runtime_dir);
    const auto marker = runtime_root / ".bubo/runtime-ready.json";
    const auto ready_bytes = Read(marker);
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime).empty(),
          "prepared runtime readiness consumed before engine load");
    WriteFixture(marker, "{}");
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime)
                  .find("runtime-ready.json") != std::string::npos,
          "broken runtime marker names file and fails");
    WriteFixture(marker, ready_bytes);
    const auto member =
        runtime_root / fs::u8path(runtime.members.front().destination);
    const auto member_bytes = Read(member);
    WriteFixture(member, member_bytes + "x");
    Check(
        RuntimeReadyErrorForTesting(runtime_root, runtime)
                .find(runtime.members.front().destination) != std::string::npos,
        "runtime member size mismatch caught before engine load");
    WriteFixture(member, std::string(member_bytes.size(), 'X'));
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime).empty(),
          "runtime startup performs only cheap metadata/size checks");
    WriteFixture(member, member_bytes);
    const auto runtime_notice = runtime_root / "licenses/NOTICE-windows.md";
    fs::remove(runtime_notice);
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime)
                  .find("NOTICE-windows.md") != std::string::npos,
          "missing runtime notice caught before engine load");
    WriteFixture(runtime_notice, notice);
    fs::remove(marker);
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime).empty(),
          "absent readiness allows explicit manual legacy installation");
    const auto journal = runtime_root / ".bubo/runtime/journal.json";
    WriteFixture(journal, "{}");
    Check(RuntimeReadyErrorForTesting(runtime_root, runtime)
                  .find("journal.json") != std::string::npos,
          "unfinished runtime journal fatal even with absent marker");
    fs::remove(journal);
    WriteFixture(marker, ready_bytes);
    auto package = LoadModelPackageForTesting(
        model_root / fs::u8path(entries[0].destination), "default", entries);
    Check(package.files.size() == 8 && package.defaults.size() == 4,
          "resolver consumes prepare's exact eight-file manifest");
    Check(package.hashed_files == 0 && !package.MainIdentity().empty(),
          "fresh streaming preparation supplies a zero-hash cache identity");
    ReleaseInputs inputs;
    inputs.model_path = model_root / fs::u8path(entries[0].destination);
    inputs.model_family = "qwen3.8-flash-next";
    inputs.architecture = "qwen4exp";
    inputs.manifest_defaults = package.defaults;
    inputs.manifest_present = true;
    const auto settings = ResolveReleaseSettings(inputs);
    for (const auto& [name, path] : package.defaults)
      Check(Setting(settings, name).value == path,
            "default profile binds prepared artefact " + name);
    const auto effective = ReleaseSettingsJson(settings).dump();
    std::cout << "prepare all output:\n"
              << prepared.out << "effective settings:\n"
              << effective << '\n';
    for (const auto* profile : {"stock", "exact"}) {
      inputs.profile = profile;
      const auto restricted = ResolveReleaseSettings(inputs);
      Check(restricted.artefacts.twin.empty() &&
                restricted.artefacts.draft_head.empty() &&
                restricted.artefacts.mtp.empty(),
            std::string(profile) +
                " typed engine-bound paths all speculative artefacts disabled");
      for (const auto* name :
           {"GUFO_DENSE_TWIN_MODEL", "GUFO_DRAFT_OUT_MODEL", "--mtp-model"})
        Check(!Setting(restricted, name).value ||
                  Setting(restricted, name).value->empty(),
              std::string(profile) + " binds no speculative artefacts");
      Check(Setting(restricted, "--mmproj").value ==
                package.defaults.at("--mmproj"),
            std::string(profile) + " retains vision unless disabled");
    }
    inputs.profile = "default";
    inputs.environment["GUFO_DENSE_TWIN_MODEL"] = "env-twin.gguf";
    inputs.explicit_flags["--twin-model"] = "flag-twin.gguf";
    const auto flag = ResolveReleaseSettings(inputs);
    Check(Setting(flag, "GUFO_DENSE_TWIN_MODEL").value == "flag-twin.gguf",
          "flag wins over env and manifest");
    inputs.explicit_flags.clear();
    const auto env = ResolveReleaseSettings(inputs);
    Check(Setting(env, "GUFO_DENSE_TWIN_MODEL").value == "env-twin.gguf",
          "env wins over manifest");
    inputs.environment.clear();
    Check(Setting(ResolveReleaseSettings(inputs), "GUFO_DENSE_TWIN_MODEL")
                  .value == package.defaults.at("GUFO_DENSE_TWIN_MODEL"),
          "manifest wins over profile");
    inputs.profile = "exact";
    inputs.explicit_flags["--speculative"] = "mtp";
    Reject([&] { ResolveReleaseSettings(inputs); }, "exact",
           "exact rejects positive MTP speculation");
    inputs.explicit_flags.clear();
    inputs.explicit_flags["--mtp-model"] = "positive-mtp.gguf";
    Reject([&] { ResolveReleaseSettings(inputs); }, "exact",
           "exact rejects positive MTP path");
    package.files.clear();  // Release startup leases before planted corruption.
    const auto mmproj = model_root / fs::u8path(entries[5].destination);
    auto corrupted = Read(mmproj);
    corrupted.back() ^= 1;
    WriteFixture(mmproj, corrupted);
    // This payload-only byte is outside the GGUF header/tensor table. Cheap
    // startup accepts it; explicit full verification must detect it.
    LoadModelPackageForTesting(inputs.model_path, "default", entries);
    std::ostringstream progress;
    Reject(
        [&] {
          LoadModelPackageForTesting(inputs.model_path, "default", entries,
                                     true, &progress);
        },
        fs::u8path(entries[5].destination).filename().string(),
        "verify-model catches planted payload byte");
    WriteFixture(mmproj, Gguf(5));
    progress.str("");
    auto verified = LoadModelPackageForTesting(inputs.model_path, "default",
                                               entries, true, &progress);
    std::uint64_t expected_bytes = 0;
    for (const auto& entry : entries)
      expected_bytes += entry.size;
    Check(verified.hashed_files == 8 &&
              verified.hashed_bytes == expected_bytes &&
              progress.str().find("checked 8 files") != progress.str().npos,
          "verify-model reports all eight named files");
    verified.files.clear();
    auto escaped = metadata::Parse(manifest_text);
    escaped["artefacts"]["twin"]["file"] = "../escape.gguf";
    WriteFixture(manifest_path, escaped.dump());
    Reject(
        [&] {
          LoadModelPackageForTesting(inputs.model_path, "default", entries);
        },
        "escape.gguf", "lexical path escape rejected");
    auto mismatch = metadata::Parse(manifest_text);
    mismatch["artefacts"]["twin"]["sha256"] = std::string(64, '0');
    WriteFixture(manifest_path, mismatch.dump());
    Reject(
        [&] {
          LoadModelPackageForTesting(inputs.model_path, "default", entries);
        },
        entries[6].file_name, "manifest entry mismatch rejected");
    WriteFixture(manifest_path, manifest_text);
    auto second = options;
    second.root = (root / "prepared-again").string();
    second.runtime_dir = (root / "runtime-again").string();
    second.yes = true;
    const auto repeated = PrepareAll(second, model, runtime, mirror);
    Check(repeated.code == 0 && Count(repeated.out, "approval: --yes") == 1,
          "--yes records exactly one complete-plan acceptance");
    const auto second_model = fs::u8path(ModelRoot(second));
    Check(Read(second_model / "bubo-package.json") == manifest_text,
          "independent roots produce identical canonical manifest bytes");
    for (const auto& entry : entries)
      Check(HashFile(second_model / fs::u8path(entry.destination)) ==
                entry.sha256,
            "independent-root acquisition preserves pinned bytes");
    auto race_options = options;
    race_options.root = (root / "ready-race-model").string();
    race_options.yes = true;
    bool removed = false;
    mirror.on_request = [&] {
      if (!removed) {
        fs::remove(member);
        removed = true;
      }
    };
    const auto raced = PrepareAll(race_options, model, runtime, mirror);
    mirror.on_request = {};
    Check(removed && raced.code == 7 &&
              raced.err.find("changed during") != raced.err.npos,
          "ready runtime changed during model download cannot advertise all "
          "success; removed=" +
              std::to_string(removed) + "; code=" + std::to_string(raced.code) +
              "; " + raced.err);
    WriteFixture(member, member_bytes);
    const auto count = mirror.requests;
    options.offline = true;
    const auto noop = PrepareAll(options, model, runtime, mirror);
    Check(noop.code == 0 && mirror.requests == count &&
              noop.out == "prepare: already prepared; nothing to do\n",
          "complete all-step offline no-op has no transfers or prompt");
    std::cout << "PASS: CPU prepare all -> eight-file manifest -> "
                 "resolver/effective settings; eleven local mirror inputs; one "
                 "approval; OPEN/decline/combined-space gates; full-byte "
                 "corruption; stock/exact; flag/env/manifest precedence; path "
                 "escape; entry mismatch; exact MTP rejection; reproducible "
                 "bytes/manifest; ready-phase race; offline no-op\n";
    fs::remove_all(root);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prepare_integration_test: FAIL " << error.what() << '\n';
    return 1;
  }
}
