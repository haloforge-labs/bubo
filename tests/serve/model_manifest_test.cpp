#include "src/cli/serve/model_manifest.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "src/cli/serve/artefact_binding_cases.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

namespace fs = std::filesystem;
namespace prep = bubo::prepare;
using namespace gufo::server;
using gufo::json::Value;
namespace {
std::size_t checks = 0;
void Expect(bool test, const std::string& message) {
  if (!test)
    throw std::runtime_error(message);
  ++checks;
}
template<class F>
void Reject(F call, const std::string& needle) {
  try {
    call();
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    Expect(message.find(needle) != message.npos,
           "rejection names " + needle + ": " + message);
    Expect(message.find_first_of("\r\n") == message.npos,
           "fatal error is one line");
    return;
  }
  throw std::runtime_error("accepted invalid case: " + needle);
}
void Write(const fs::path& path, std::string_view text) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!file)
    throw std::runtime_error("fixture write failed");
}
std::string Read(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {(std::istreambuf_iterator<char>(file)), {}};
}
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
std::string Gguf(unsigned index, bool main) {
  std::string bytes;
  const auto integer = [&](std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i)
      bytes += static_cast<char>(value >> (8 * i));
  };
  const auto text = [&](std::string_view value) {
    integer(value.size(), 8);
    bytes += value;
  };
  integer(0x46554747, 4);
  integer(3, 4);
  integer(1, 8);
  integer(main ? 4 : 1, 8);
  text("general.architecture");
  integer(8, 4);
  text(main ? "qwen4exp" : "fixture");
  if (main) {
    text("split.count");
    integer(4, 4);
    integer(4, 4);
    text("split.no");
    integer(4, 4);
    integer(index, 4);
    text("split.tensors.count");
    integer(10, 4);
    integer(4, 8);
  }
  text("fixture.weight");
  integer(1, 4);
  integer(8, 8);
  integer(0, 4);
  integer(0, 8);
  bytes.resize((bytes.size() + 31) / 32 * 32);
  bytes.append(32, static_cast<char>(index + 1));
  return bytes;
}
struct Fixture {
  fs::path root;
  std::array<prep::Entry, 8> entries;
  std::array<std::string, 8> sizes_unused, hashes;
  std::map<std::string, prep::FingerprintResult> pins;
  Value manifest;
  explicit Fixture(fs::path directory)
      : root(std::move(directory)), entries(prep::ModelCatalogue()) {
    fs::create_directories(root);
    auto files = Value::array(), artefacts = Value::object();
    for (std::size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries[i];
      const auto bytes = Gguf(static_cast<unsigned>(i), i < 4);
      hashes[i] = gufo::crypto::Sha256Hex(
          {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
      entry.size = bytes.size();
      entry.sha256 = hashes[i];
      Write(root / entry.destination, bytes);
      auto row = Value::object();
      row["file"] = std::string(entry.destination);
      row["size"] = static_cast<unsigned long long>(entry.size);
      row["sha256"] = hashes[i];
      if (i >= 4)
        row["recipe_id"] = "huihui-ud-q4-k-xl-iq4xs-nofullattn-nohc-q4k-v1";
      if (i >= 6)
        row["generator_version"] = "1.0.0";
      if (i < 4) {
        files.push_back(row);
        pins[std::string(entry.logical_name)] =
            prep::GenerateGgufFingerprint(root / entry.destination, entry);
        Expect(bool(pins.at(std::string(entry.logical_name))),
               "independent fixture pin generation");
      } else
        artefacts[std::string(entry.logical_name)] = row;
    }
    manifest = Value::object();
    manifest["schema_version"] = 1;
    manifest["package_id"] = std::string(prep::kPackageId);
    manifest["engine_min"] = "0.1.0";
    auto model = Value::object(), source = Value::object();
    source["repository"] = std::string(entries[0].source_repo);
    source["revision"] = std::string(entries[0].commit);
    model["source"] = source;
    model["family"] = "qwen3.8-flash-next";
    model["files"] = files;
    manifest["model"] = model;
    manifest["artefacts"] = artefacts;
    auto profiles = Value::array();
    for (auto profile : {"default", "stock", "exact", "subagents"})
      profiles.push_back(profile);
    manifest["profiles_supported"] = profiles;
    Save();
  }
  void Save() { Write(root / "bubo-package.json", manifest.dump()); }
  ModelPackage Load(bool verify = false, const fs::path& selected = {}) {
    return LoadModelPackageForTesting(
        root / entries[0].destination, "default", entries, verify, nullptr,
        [&](const prep::Entry& entry) {
          return pins.at(std::string(entry.logical_name));
        },
        selected);
  }
};
void Matrix() {
  const std::map<std::string, std::string> keys = {
      {"twin", "GUFO_DENSE_TWIN_MODEL"},
      {"draft_head", "GUFO_DRAFT_OUT_MODEL"},
      {"mtp", "--mtp-model"},
      {"mmproj", "--mmproj"}};
  const std::map<std::string, std::string> flags = {
      {"twin", "--twin-model"},
      {"draft_head", "--draft-head-model"},
      {"mtp", "--mtp-model"},
      {"mmproj", "--mmproj"}};
  for (const auto& row : kArtefactBindingCases) {
    ReleaseInputs inputs;
    inputs.profile = row.profile;
    inputs.model_family = "qwen3.8-flash-next";
    const auto artefact = std::string(row.artefact), key = keys.at(artefact),
               flag = flags.at(artefact);
    // Dependency support is explicit and separate from the artefact under test.
    if (artefact == "draft_head" && row.profile != "exact") {
      inputs.explicit_flags["--mtp-model"] = "mtp.gguf";
      inputs.explicit_flags["--speculative"] = "mtp";
    }
    if (row.source == "flag") {
      inputs.explicit_flags[flag] = "selected.gguf";
      if (artefact == "mtp")
        inputs.explicit_flags["--speculative"] = "mtp";
    } else if (row.source == "env")
      inputs.environment[key] = "selected.gguf";
    else
      inputs.manifest_defaults[key] = "selected.gguf";
    const auto label = std::string(row.profile) + "/" + artefact + "/" +
                       std::string(row.source);
    if (row.expected == BindingExpectation::rejected) {
      Reject([&] { ResolveReleaseSettings(inputs); }, "profile=exact");
      continue;
    }
    const auto resolved = ResolveReleaseSettings(inputs);
    const auto& typed = resolved.artefacts;
    const auto value = artefact == "twin"         ? typed.twin
                       : artefact == "draft_head" ? typed.draft_head
                       : artefact == "mtp"        ? typed.mtp
                                                  : typed.mmproj;
    Expect(typed.present,
           "typed selection prevents engine getenv fallback: " + label);
    Expect((value == "selected.gguf") ==
               (row.expected == BindingExpectation::bound),
           "T10 " + label);
  }
  ReleaseInputs inputs;
  inputs.model_family = "qwen3.8-flash-next";
  inputs.manifest_defaults["--mmproj"] = "manifest.gguf";
  Expect(ResolveReleaseSettings(inputs).artefacts.mmproj == "manifest.gguf",
         "manifest beats profile/builtin");
  inputs.environment["--mmproj"] = "env.gguf";
  Expect(ResolveReleaseSettings(inputs).artefacts.mmproj == "env.gguf",
         "env beats manifest");
  inputs.explicit_flags["--mmproj"] = "flag.gguf";
  Expect(ResolveReleaseSettings(inputs).artefacts.mmproj == "flag.gguf",
         "flag beats env");
  auto selector_inputs = ReleaseInputs{};
  selector_inputs.model_family = "qwen3.8-flash-next";
  selector_inputs.environment["BUBO_MTP_MODEL"] = "mtp-env.gguf";
  selector_inputs.environment["BUBO_MMPROJ"] = "vision-env.gguf";
  const auto selectors = ResolveReleaseSettings(selector_inputs);
  Expect(selectors.artefacts.mtp == "mtp-env.gguf" &&
             selectors.artefacts.mmproj == "vision-env.gguf",
         "documented environment selectors feed typed paths");
  inputs.explicit_flags["--no-vision"] = "1";
  Reject([&] { ResolveReleaseSettings(inputs); }, "conflicts");
  inputs.explicit_flags.erase("--mmproj");
  Expect(ResolveReleaseSettings(inputs).artefacts.vision_disabled &&
             ResolveReleaseSettings(inputs).artefacts.mmproj.empty(),
         "no-vision suppresses env and discovery");
  inputs = {};
  inputs.model_family = "qwen3.8-flash-next";
  inputs.manifest_defaults = {{"--mtp-model", "mtp.gguf"},
                              {"GUFO_DRAFT_OUT_MODEL", "head.gguf"}};
  inputs.explicit_flags["--no-mtp"] = "1";
  auto disabled = ResolveReleaseSettings(inputs);
  Expect(
      disabled.artefacts.mtp.empty() && disabled.artefacts.draft_head.empty(),
      "no-mtp also leaves unused manifest draft head inactive");
  inputs.explicit_flags.clear();
  inputs.explicit_flags["--speculative"] = "off";
  disabled = ResolveReleaseSettings(inputs);
  Expect(disabled.artefacts.draft_head.empty(),
         "speculative off leaves unused manifest head inactive");
  for (const auto* profile : {"default", "stock"}) {
    inputs = {};
    inputs.model_family = "qwen3.8-flash-next";
    inputs.profile = profile;
    inputs.explicit_flags["--draft-head-model"] = "head.gguf";
    Reject([&] { ResolveReleaseSettings(inputs); },
           "--draft-head-model: needs speculative==mtp");
  }
  const auto doc =
      Read(fs::path(__FILE__).parent_path().parent_path().parent_path() /
           "docs/prepare-binding.md");
  Expect(doc.find(RenderArtefactBindingTable()) != doc.npos,
         "ONE T10 docs/test table exactly matches renderer");
}
void ManifestCases(const fs::path& directory) {
  Fixture fixture(directory);
  auto package = fixture.Load();
  Expect(package.files.size() == 8 && package.defaults.size() == 4,
         "all eight names and artefacts loaded");
  Expect(package.hashed_files == 0 && package.hashed_bytes == 0,
         "fresh default zero payload hashing");
  Expect(package.MainIdentity().empty(),
         "target pins cannot authorize persistent cache identity");
  package = {};  // release native leases before fixture mutation
  package = fixture.Load(true);
  std::uint64_t total = 0;
  for (const auto& entry : fixture.entries)
    total += entry.size;
  Expect(package.hashed_files == 8 && package.hashed_bytes == total,
         "verify exactly eight pinned files");
  Expect(!package.MainIdentity().empty(),
         "current full verification authorizes identity");
  const auto override_path = fixture.root / "override.gguf";
  Write(override_path, Read(fixture.root / fixture.entries[6].destination));
  CheckPackageOverrides(package,
                        {{"GUFO_DENSE_TWIN_MODEL", Utf8(override_path)}}, true);
  Expect(package.hashed_files == 9 &&
             package.hashed_bytes == total + fixture.entries[6].size,
         "verify distinct selected override once");
  CheckPackageOverrides(package,
                        {{"GUFO_DENSE_TWIN_MODEL", Utf8(override_path)}}, true);
  Expect(package.hashed_files == 9, "same override deduplicated");
  Reject(
      [&] {
        CheckPackageOverrides(
            package,
            {{"GUFO_DENSE_TWIN_MODEL", Utf8(package.files.front().path)}},
            true);
      },
      "override manifest entry mismatch");
  package = {};
  auto corrupt = Read(fixture.root / fixture.entries[2].destination);
  corrupt.back() ^= 1;
  Write(fixture.root / fixture.entries[2].destination, corrupt);
  package = fixture.Load();
  Expect(package.hashed_files == 0,
         "valid-header same-size payload corruption not cold hashed");
  package = {};
  Reject([&] { fixture.Load(true); },
         std::string(fixture.entries[2].file_name));
  corrupt.back() ^= 1;
  Write(fixture.root / fixture.entries[2].destination, corrupt);
  Write(fixture.root / fixture.entries[2].destination,
        corrupt.substr(0, corrupt.size() - 1));
  Reject([&] { fixture.Load(); }, "size mismatch");
  Write(fixture.root / fixture.entries[2].destination, corrupt);
  auto wrong_header = corrupt;
  wrong_header[0] = '!';
  Write(fixture.root / fixture.entries[2].destination, wrong_header);
  Reject([&] { fixture.Load(); }, "header");
  Write(fixture.root / fixture.entries[2].destination, corrupt);
  const auto baseline = fixture.manifest;
  for (const auto* path :
       {"../escape.gguf", "a/../escape.gguf", "C:/escape.gguf",
        "//server/share.gguf", "a:stream", "a\\..\\escape.gguf"}) {
    fixture.manifest = baseline;
    fixture.manifest["artefacts"]["twin"]["file"] = path;
    fixture.Save();
    Reject([&] { fixture.Load(); }, "unsafe manifest relative path");
  }
  fixture.manifest = baseline;
  fixture.manifest["artefacts"]["twin"]["sha256"] = std::string(64, '0');
  fixture.Save();
  Reject([&] { fixture.Load(); }, std::string(fixture.entries[6].file_name));
  fixture.manifest = baseline;
  fixture.manifest["artefacts"]["mtp"]["recipe_id"] = "wrong";
  fixture.Save();
  Reject([&] { fixture.Load(); }, "recipe entry mismatch");
  fixture.manifest = baseline;
  fixture.manifest["schema_version"] = 2;
  fixture.Save();
  Reject([&] { fixture.Load(); }, "schema_version");
  fixture.manifest = baseline;
  fixture.manifest["engine_min"] = "0.2.0";
  fixture.Save();
  Reject([&] { fixture.Load(); }, "engine_min");
  fixture.manifest = baseline;
  fixture.Save();
#ifdef _WIN32
  // An ordinary user can create an NTFS junction without symbolic-link
  // privilege. Reads may follow this intentional model-storage junction.
  const auto junction = fixture.root / "twin";
  const auto destination = fixture.root / "actual-twin";
  fs::rename(junction, destination);
  fs::create_directory(junction);
  const std::wstring substitute =
      L"\\??\\" + fs::absolute(destination).wstring();
  const std::wstring printable = fs::absolute(destination).wstring();
  struct JunctionBuffer {
    DWORD tag;
    WORD length, reserved, substitute_offset, substitute_length, print_offset,
        print_length;
    WCHAR paths[4096];
  } buffer{};
  buffer.tag = IO_REPARSE_TAG_MOUNT_POINT;
  buffer.substitute_length =
      static_cast<WORD>(substitute.size() * sizeof(WCHAR));
  buffer.print_offset =
      static_cast<WORD>((substitute.size() + 1) * sizeof(WCHAR));
  buffer.print_length = static_cast<WORD>(printable.size() * sizeof(WCHAR));
  std::copy(substitute.begin(), substitute.end(), buffer.paths);
  std::copy(printable.begin(), printable.end(),
            buffer.paths + substitute.size() + 1);
  buffer.length = static_cast<WORD>(
      8 + (substitute.size() + printable.size() + 2) * sizeof(WCHAR));
  const auto handle = CreateFileW(
      junction.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  Expect(handle != INVALID_HANDLE_VALUE, "junction fixture handle");
  DWORD ignored = 0;
  const bool made = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, &buffer,
                                    static_cast<DWORD>(buffer.length + 8),
                                    nullptr, 0, &ignored, nullptr);
  CloseHandle(handle);
  Expect(made, "junction fixture creation without elevation");
  package = fixture.Load();
  Expect(package.files.size() == 8, "legitimate junction model read allowed");
  package = {};
  fs::remove(junction);
  fs::rename(destination, junction);
#endif
  fs::remove(fixture.root / fixture.entries[6].destination);
  Reject([&] { fixture.Load(); }, std::string(fixture.entries[6].file_name));
  Write(fixture.root / fixture.entries[6].destination, Gguf(6, false));
  fs::remove(fixture.root / "bubo-package.json");
  Expect(fixture.Load().files.empty(), "no manifest keeps legacy flags/env");
  Reject([&] { fixture.Load(true); }, "requires --manifest");
  Write(fixture.root / "legacy.bubo.json", "broken legacy ignored");
  Expect(fixture.Load().files.empty(), "only canonical filename discovered");
  fixture.Save();
  Expect(fixture.Load(false, fixture.root).files.size() == 8,
         "explicit prepare folder selects canonical manifest");
  const auto parent = fixture.root.parent_path() / "prepared-output";
  const auto package_root = parent / "models" / prep::kPackageId;
  fs::create_directories(package_root.parent_path());
  fs::rename(fixture.root, package_root);
  fixture.root = package_root;
  auto directory_package =
      LoadModelPackageForTesting(parent, "default", fixture.entries);
  Expect(directory_package.files.size() == 8, "prepare output ROOT discovery");
  directory_package = {};
}
}  // namespace
int main() {
  const auto scratch = fs::temp_directory_path() / "bubo-manifest-test";
  try {
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    Matrix();
    ManifestCases(scratch / fs::u8path("models with spaces-\xc3\xa9"));
    fs::remove_all(scratch);
    std::cout
        << "PASS: model_manifest_test " << checks
        << " checks; T10 48 rows; default zero payload hash; verify8+override; "
           "schema/path/header/pin/profile/discovery proofs\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
