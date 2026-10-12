#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/transport.hpp"
#include "tests/prepare/loopback_stub.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
constexpr std::string_view kAbc =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
// Deliberately synthetic: ASCII abc is the published SHA-256 known-answer
// vector, not a production GGUF. Separate fingerprint tests parse real tiny
// synthetic GGUF tables and reject a changed tensor prefix and truncation.
constexpr std::array<Entry, 8> kEntries = {
    {{"main_1", "one.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/one.gguf",
      Kind::MainShard, "qwen-community-1.0", "one.gguf",
      "fingerprints/one.json"},
     {"main_2", "two.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/two.gguf",
      Kind::MainShard, "qwen-community-1.0", "two.gguf",
      "fingerprints/two.json"},
     {"main_3", "three.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/three.gguf",
      Kind::MainShard, "qwen-community-1.0", "three.gguf",
      "fingerprints/three.json"},
     {"main_4", "four.gguf", 3, kAbc, "fixture/model",
      "1111111111111111111111111111111111111111", "UD-Q4_K_XL/four.gguf",
      Kind::MainShard, "qwen-community-1.0", "four.gguf",
      "fingerprints/four.json"},
     {"mtp", "mtp.gguf", 3, kAbc, "fixture/side",
      "2222222222222222222222222222222222222222", "MTP/mtp.gguf", Kind::Mtp,
      "qwen-community-1.0", "mtp/mtp.gguf"},
     {"mmproj", "mmproj.gguf", 3, kAbc, "fixture/side",
      "2222222222222222222222222222222222222222", "mmproj.gguf", Kind::Mmproj,
      "qwen-community-1.0", "mmproj/mmproj.gguf"},
     {"twin", "twin.gguf", 3, kAbc, "fixture/bubo",
      "3333333333333333333333333333333333333333", "twin/twin.gguf", Kind::Twin,
      "qwen-community-1.0", "twin/twin.gguf"},
     {"draft_head", "head.gguf", 3, kAbc, "fixture/bubo",
      "3333333333333333333333333333333333333333", "draft_head/head.gguf",
      Kind::DraftHead, "qwen-community-1.0", "draft_head/head.gguf"}}};
void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}
void Write(const fs::path& path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << bytes;
  Require(bool(file), "fixture write");
}
class Fake final : public Transport {
public:
  int requests = 0;
  bool wrong_byte = false, fail_last = false;
  std::function<void()> before_body;
  TransferResult Perform(const Request& request, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    ++requests;
    if (fail_last && request.url.ends_with("head.gguf")) {
      headers({401, {}});
      return {};
    }
    const Response response{200,
                            {{"content-length", "3"},
                             {"etag", "\"abc\""},
                             {"x-linked-etag", std::string(kAbc)},
                             {"x-linked-size", "3"}}};
    if (!headers(response))
      return {};
    if (before_body)
      before_body();
    const std::array<std::uint8_t, 3> data = {
        'a', static_cast<std::uint8_t>(wrong_byte ? 'B' : 'b'), 'c'};
    body(data);
    return {};
  }
};
class LoopbackTransport final : public Transport {
public:
  explicit LoopbackTransport(std::string base)
      : base_(std::move(base)), curl_(Policy()) {}
  TransferResult Perform(const Request& request, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>& cancel) override {
    auto local = request;
    constexpr std::string_view origin = "https://huggingface.co";
    Require(local.url.starts_with(origin),
            "only compiled HF fixture origins can be mapped");
    local.url = base_ + local.url.substr(origin.size());
    return curl_.Perform(local, headers, body, cancel);
  }

private:
  static HostPolicy Policy() {
    HostPolicy policy;
    policy.test_loopback = true;
    return policy;
  }
  std::string base_;
  CurlTransport curl_;
};
class NoProxy {
public:
  NoProxy() {
    if (const auto* value = std::getenv("NO_PROXY")) {
      previous_ = value;
      existed_ = true;
    }
    Set("127.0.0.1");
  }
  ~NoProxy() { Set(existed_ ? previous_.c_str() : ""); }

private:
  static void Set(const char* value) {
#ifdef _WIN32
    _putenv_s("NO_PROXY", value);
#else
    if (*value)
      setenv("NO_PROXY", value, 1);
    else
      unsetenv("NO_PROXY");
#endif
  }
  std::string previous_;
  bool existed_ = false;
};
ModelTestEnvironment Environment(Fake& transport, int& fingerprints) {
  ModelTestEnvironment env;
  env.entries = kEntries;
  env.transport = &transport;
  // TEST-ONLY text: never fills approved production legal evidence slots.
  env.notices = {{"LICENSE", "abc", std::string(kAbc)},
                 {"NOTICE", "abc", std::string(kAbc)},
                 {"PROVENANCE.md", "abc", std::string(kAbc)}};
  env.volume = [](const fs::path&) {
    return VolumeInfo{
        "fixture-volume", 16ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
  };
  env.fingerprint = [&fingerprints](const Entry& entry, const fs::path& path) {
    ++fingerprints;
    return HashFile(path) == entry.sha256 ? std::string{}
                                          : "synthetic fingerprint mismatch";
  };
  env.download_options.wait = [](std::chrono::seconds) {};
  return env;
}
int Run(const Options& options, const ModelTestEnvironment& env,
        std::string& output, std::string& error, std::string answer = {}) {
  std::istringstream input(answer);
  std::ostringstream out, err;
  const auto code = RunModelForTesting(options, env, input, out, err);
  output = out.str();
  error = err.str();
  return code;
}
using Snapshot =
    std::map<fs::path, std::pair<std::uintmax_t, fs::file_time_type>>;
Snapshot Files(const fs::path& root) {
  Snapshot result;
  if (!fs::exists(root))
    return result;
  result.emplace(root, std::make_pair(0, fs::last_write_time(root)));
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    result.emplace(
        entry.path(),
        std::make_pair(entry.is_regular_file() ? entry.file_size() : 0,
                       entry.last_write_time()));
  }
  return result;
}
std::string Normalize(std::string text) {
  std::replace(text.begin(), text.end(), '\\', '/');
  text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
  return text;
}
std::string Compact(std::string text) {
  text = Normalize(std::move(text));
  text.erase(
      std::remove_if(text.begin(), text.end(),
                     [](char c) { return c == ' ' || c == '\n' || c == '\t'; }),
      text.end());
  return text;
}
std::vector<std::uint8_t> TinyGguf() {
  std::vector<std::uint8_t> bytes;
  auto number = [&](std::uint64_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
      bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  number(0x46554747, 4);
  number(3, 4);
  number(1, 8);
  number(0, 8);
  number(4, 8);
  bytes.insert(bytes.end(), {'t', 'e', 's', 't'});
  number(1, 4);
  number(8, 8);
  number(0, 4);
  number(0, 8);
  while (bytes.size() % 32)
    bytes.push_back(0);
  bytes.resize(bytes.size() + 32, 0x41);
  return bytes;
}
#ifdef _WIN32
std::wstring QuoteArgument(const std::wstring& argument) {
  std::wstring quoted = L"\"";
  std::size_t backslashes = 0;
  for (const auto character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    quoted.append(character == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
    quoted += character;
    backslashes = 0;
  }
  quoted.append(backslashes * 2, L'\\');
  quoted += L'"';
  return quoted;
}
DWORD LockChild(const fs::path& path) {
  std::vector<wchar_t> executable(32768);
  const auto length = GetModuleFileNameW(nullptr, executable.data(),
                                         static_cast<DWORD>(executable.size()));
  Require(length > 0 && length < executable.size(),
          "native child executable path");
  const std::wstring image(executable.data(), length);
  auto command = QuoteArgument(image) + L" --lock-child " +
                 QuoteArgument(NativePath(path).wstring());
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  Require(CreateProcessW(image.c_str(), command.data(), nullptr, nullptr, FALSE,
                         CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                         nullptr, nullptr, &startup, &process) != FALSE,
          "native BelowNormal hidden lock child launch");
  CloseHandle(process.hThread);
  const auto wait = WaitForSingleObject(process.hProcess, 30000);
  DWORD code = 9;
  if (wait != WAIT_OBJECT_0) {
    TerminateProcess(process.hProcess,
                     9);  // Only this owned CPU fixture child.
    WaitForSingleObject(process.hProcess, 1000);
  } else {
    GetExitCodeProcess(process.hProcess, &code);
  }
  CloseHandle(process.hProcess);
  Require(wait == WAIT_OBJECT_0, "native lock child completed within test cap");
  return code;
}
#endif
}  // namespace
int main(int argc, char** argv) {
  try {
#ifdef _WIN32
    if (argc == 3 && std::string_view(argv[1]) == "--lock-child") {
      Require(
          GetPriorityClass(GetCurrentProcess()) == BELOW_NORMAL_PRIORITY_CLASS,
          "native lock child must run BelowNormal");
      PrepareLock lock;
      std::string error;
      return lock.Acquire(NativePath(fs::u8path(argv[2])), error)
                 ? 0
                 : (lock.busy() ? 8 : 7);
    }
#endif
    Require(argc == 3,
            "usage: prepare_model_test JOB_FIXTURE_ROOT CONSENT_GOLDEN");
    const auto base = fs::absolute(fs::u8path(argv[1])).lexically_normal();
    // Never remove an external or computed tree: caller supplies an unused
    // job-owned directory and the test refuses to overwrite existing fixtures.
    Require(!fs::exists(base), "fixture root must not exist");
    fs::create_directories(base);
    Options options;
    options.step = Step::Model;
    options.yes = true;
    options.root = (base / "happy").string();
    Fake transport;
    int fingerprints = 0;
    auto env = Environment(transport, fingerprints);
    const auto model = fs::u8path(ModelRoot(options));
    std::string out, err;
    bool last_checked = false;
    env.before_manifest = [&] {
      last_checked = true;
      Require(!fs::exists(model / "bubo-package.json"),
              "manifest published before final gate");
      Require(fs::exists(model / ".bubo" / "prepare-state.json"),
              "state must precede manifest");
      for (const auto& entry : kEntries)
        Require(HashFile(model / fs::u8path(entry.destination)) == entry.sha256,
                "every pin before manifest");
#ifdef _WIN32
      for (const auto& entry : kEntries) {
        const auto path = model / fs::u8path(entry.destination);
        std::ofstream concurrent(path, std::ios::binary | std::ios::trunc);
        Require(!concurrent,
                "verified payload lease must deny concurrent write before "
                "readiness");
        Require(!DeleteFileW(NativePath(path).c_str()),
                "verified payload lease must deny deletion before readiness");
      }
#endif
      for (const auto& notice : env.notices)
        Require(HashFile(model / "licenses" / notice.name) == notice.sha256,
                "notices before manifest");
      return false;
    };
    const auto first_code = Run(options, env, out, err);
    Require(first_code == 3 && last_checked && transport.requests == 8,
            "crash-before-manifest acquisition; code=" +
                std::to_string(first_code) + "; " + err);
    Require(out.find(" host huggingface.co") != std::string::npos,
            "per-file progress reports the validated safe host");
    Require(!fs::exists(model / "bubo-package.json"),
            "manifest-last crash invariant");
    env.before_manifest = {};
    Require(Run(options, env, out, err) == 0 && transport.requests == 8,
            "resume publication without payload requests");
    const auto snapshot = Files(model);
    fingerprints = 0;
    Require(Run(options, env, out, err) == 0, "no-op success");
    Require(out == "prepare: already prepared; nothing to do\n" && err.empty(),
            "exact no-op output");
    const auto after_no_op = Files(model);
    const bool same_snapshot = snapshot == after_no_op;
    std::ostringstream no_op_detail;
    no_op_detail << "no-op no-network/no-writes/stamp skip; requests="
                 << transport.requests << "; fingerprints=" << fingerprints
                 << "; same_snapshot=" << same_snapshot;
    if (!same_snapshot) {
      for (const auto& [path, before] : snapshot) {
        const auto after = after_no_op.find(path);
        if (after == after_no_op.end()) {
          no_op_detail << "\nremoved "
                       << path.lexically_relative(model).generic_string();
        } else if (after->second != before) {
          no_op_detail << "\nchanged "
                       << path.lexically_relative(model).generic_string()
                       << "; size=" << before.first << "->"
                       << after->second.first << "; mtime.count="
                       << before.second.time_since_epoch().count() << "->"
                       << after->second.second.time_since_epoch().count();
        }
      }
      for (const auto& [path, observed] : after_no_op) {
        if (!snapshot.contains(path))
          no_op_detail << "\nadded "
                       << path.lexically_relative(model).generic_string()
                       << "; size=" << observed.first << "; mtime.count="
                       << observed.second.time_since_epoch().count();
      }
    }
    Require(transport.requests == 8 && fingerprints == 0 && same_snapshot,
            no_op_detail.str());
    for (int repetition = 0; repetition < 5; ++repetition) {
      Require(Run(options, env, out, err) == 0 &&
                  out == "prepare: already prepared; nothing to do\n" &&
                  err.empty() && transport.requests == 8 && fingerprints == 0 &&
                  snapshot == Files(model),
              "repeated no-op preserves every file/directory without network "
              "or fingerprint reads");
    }
    options.verify = true;
    Require(Run(options, env, out, err) == 0 &&
                out.find("prepare verify: 8/8") != out.npos,
            "verify eight files progress");
    Require(transport.requests == 8 && snapshot == Files(model),
            "verify read-only no-op");
    options.verify = false;
    Write(model / ".bubo" / "prepare-state.json", "{ malformed state }");
    fingerprints = 0;
    Require(Run(options, env, out, err) == 0 && fingerprints == 8 &&
                out.find("approval: --yes") != out.npos &&
                out.find("already prepared") == out.npos,
            "malformed state rechecks shards then repairs state/readiness");
    fingerprints = 0;
    const auto corrupt = model / "one.gguf";
    const auto before = fs::last_write_time(corrupt);
    Write(corrupt, "aBc");
    fs::last_write_time(corrupt, before + std::chrono::seconds(1));
    Require(Run(options, env, out, err) == 7 && fingerprints == 1,
            "changed main stamp triggers fingerprint refusal");
    Require(err.find("one.gguf") != err.npos &&
                err.find("delete it and run bubo prepare again") != err.npos,
            "canonical mismatch diagnostic");
    Require(HashFile(corrupt) != kAbc, "planted byte remains untouched");
    Require(fs::exists(model / "bubo-package.json"),
            "existing canonical corruption is not silently repaired");
    Options offline = options;
    offline.root = (base / "offline").string();
    offline.offline = true;
    const auto requests = transport.requests;
    Require(Run(offline, env, out, err) == 4 &&
                transport.requests == requests && !fs::exists(offline.root),
            "offline missing before requests/writes");
    Options fat = options;
    fat.root = (base / "fat").string();
    auto fat_env = env;
    fat_env.volume = [](const fs::path&) {
      return VolumeInfo{"fat", 1ULL << 40, "FAT32", false, {}};
    };
    Require(Run(fat, fat_env, out, err) == 7 && err.find("FAT32") != err.npos &&
                transport.requests == requests,
            "injectable FAT32 refusal");
    auto net_env = env;
    net_env.volume = [](const fs::path&) {
      return VolumeInfo{"unc", 1ULL << 40, "NTFS", true, {}};
    };
    Require(Run(fat, net_env, out, err) == 7 && err.find("UNC") != err.npos,
            "explicit UNC/network policy");
    auto low_env = env;
    low_env.volume = [](const fs::path&) {
      return VolumeInfo{"low", 2ULL << 30, "NTFS", false, {}};
    };
    Require(Run(fat, low_env, out, err) == 7 &&
                err.find("insufficient") != err.npos,
            "new allocation plus margin");
    for (const auto spare_control_bytes : {0ULL, 2ULL * 1024 * 1024}) {
      Fake separate;
      int checks = 0;
      auto controls_env = Environment(separate, checks);
      bool control_low = false;
      controls_env.volume = [&](const fs::path& path) {
        const bool payload = path.extension() == ".gguf";
        return VolumeInfo{payload ? "payload-volume" : "control-volume",
                          !payload && control_low
                              ? 2ULL * 1024 * 1024 * 1024 + spare_control_bytes
                              : 16ULL * 1024 * 1024 * 1024,
                          "NTFS",
                          false,
                          {}};
      };
      separate.before_body = [&] { control_low = true; };
      Options controls = options;
      controls.root =
          (base / (spare_control_bytes ? "controls-notices" : "controls-space"))
              .string();
      Require(Run(controls, controls_env, out, err) == 7 &&
                  !fs::exists(fs::u8path(ModelRoot(controls)) /
                              "bubo-package.json"),
              "streaming recheck preserves root metadata/notice margin when "
              "payloads use another volume");
    }
    auto cloud_env = env;
    cloud_env.volume = [](const fs::path&) {
      return VolumeInfo{{},
                        0,
                        {},
                        false,
                        "Use a local folder. Select it with --output, "
                        "--runtime-dir or --cache-dir."};
    };
    Require(Run(fat, cloud_env, out, err) == 7 &&
                err.find("Use a local folder") != err.npos,
            "cloud recall blocked before payload");
    Options decline = options;
    decline.root = (base / "decline").string();
    decline.yes = false;
    Require(Run(decline, env, out, err) == 3 &&
                transport.requests == requests && !fs::exists(decline.root),
            "EOF declines before mutation/network");
    Require(
        out.find("approval: --yes") == out.npos &&
            out.find("Download and prepare all files in this plan? [y/N]") !=
                out.npos,
        "single consent question");
    Options failed = options;
    failed.root = (base / "failed").string();
    transport.fail_last = true;
    Require(
        Run(failed, env, out, err) == 5 &&
            !fs::exists(fs::u8path(ModelRoot(failed)) / "bubo-package.json"),
        "failed last file cannot publish manifest");
    transport.fail_last = false;
    Options wrong = options;
    wrong.root = (base / "wrong").string();
    transport.wrong_byte = true;
    Require(Run(wrong, env, out, err) == 6 &&
                !fs::exists(fs::u8path(ModelRoot(wrong)) / "bubo-package.json"),
            "download planted byte hash refusal");
    transport.wrong_byte = false;
    Options locked = options;
    locked.root = (base / "locked").string();
    const auto lock_dir = fs::u8path(ModelRoot(locked)) / ".bubo";
    fs::create_directories(lock_dir);
    std::string lock_error;
    {
      PrepareLock live;
      Require(live.Acquire(lock_dir / "prepare.lock", lock_error),
              "first exclusive lock");
      Require(Run(locked, env, out, err) == 8, "parallel live lock refusal");
#ifdef _WIN32
      Require(LockChild(lock_dir / "prepare.lock") == 8,
              "another native process cannot acquire a live prepare lock");
#endif
    }
    Require(fs::exists(lock_dir / "prepare.lock"),
            "released lock pathname retained");
#ifdef _WIN32
    Require(
        LockChild(lock_dir / "prepare.lock") == 0,
        "another native process can exclusively reopen a stale prepare lock");
#endif
    Write(lock_dir / "prepare.lock", "pid=dead creation=old\n");
    Require(Run(locked, env, out, err) == 0, "stale lock exclusive reopening");
    Require(fs::exists(lock_dir / "prepare.lock"),
            "stale lock pathname retained");
    {
      PrepareLock reopened;
      Require(reopened.Acquire(lock_dir / "prepare.lock", lock_error),
              "retained lock becomes acquirable after its owner releases it");
    }
    Options supplied = options;
    supplied.root = (base / "supplied").string();
    supplied.mtp_model = (base / "input" / "mtp.gguf").string();
    supplied.mmproj = (base / "input" / "mmproj.gguf").string();
    Write(fs::u8path(supplied.mtp_model), "abc");
    Write(fs::u8path(supplied.mmproj), "abc");
    const auto start_requests = transport.requests;
    auto supplied_env = env;
#ifdef _WIN32
    supplied_env.before_manifest = [&] {
      const auto root = NativePath(fs::u8path(ModelRoot(supplied)));
      for (const auto& entry : kEntries) {
        const auto path = root / fs::u8path(entry.destination);
        std::ofstream concurrent(path, std::ios::binary | std::ios::trunc);
        Require(!concurrent && !DeleteFileW(path.c_str()),
                "verified supplied/downloaded writer handles deny mutation "
                "through readiness");
      }
      return true;
    };
#endif
    Require(Run(supplied, supplied_env, out, err) == 0 &&
                transport.requests == start_requests + 6,
            "supplied sidecars copied through verified staging");
    Require(HashFile(fs::u8path(supplied.mtp_model)) == kAbc &&
                HashFile(fs::u8path(supplied.mmproj)) == kAbc,
            "supplied originals remain intact");
    auto cancelled_env = env;
    cancelled_env.cancelled = [] { return true; };
    Require(Run(fat, cancelled_env, out, err) == 3, "cancel before payload");
    auto missing_notice = env;
    missing_notice.notices.clear();
    Require(
        Run(fat, missing_notice, out, err) == 4 && err.find("OPEN") != err.npos,
        "notice approval gate");
    const std::array<ConsentVolume, 1> volume = {
        {{"fixture", 4294967296ULL, 2147483648ULL}}};
    auto prose = Normalize(ModelConsent({}, "M", {}, 0, volume));
    const auto golden = Normalize(metadata::Read(fs::u8path(argv[2])));
    Require(prose == golden,
            "independently authored model consent prose snapshot");
    Require(
        ModelConsent({}, "M", {}, 0, volume, false).find("main shards total") ==
            std::string::npos,
        "runtime selection omits shard line");
    std::vector<std::string> status(8, "download");
    const auto consent = ModelConsent(kEntries, "M", status, 24, volume);
    std::istringstream lines(consent);
    std::string line;
    while (std::getline(lines, line))
      Require(line.size() <= 80, "consent wraps at 80 columns");
    auto joined = consent;
    joined.erase(std::remove(joined.begin(), joined.end(), '\n'), joined.end());
    for (const auto& entry : kEntries) {
      Require(consent.find(entry.file_name) != consent.npos,
              "all eight consent rows");
      Require(joined.find(OriginUrl(entry)) != joined.npos &&
                  joined.find(entry.sha256) != joined.npos,
              "full origin and SHA in every row");
      const auto expected_row = std::string(entry.file_name) +
                                "|huggingface.co|" + OriginUrl(entry) +
                                "|3bytes|0.000GiB|ba7816bf8f01|" +
                                std::string(kAbc) + "|qwen-community-1.0|M/" +
                                std::string(entry.destination) + "|download";
      Require(Compact(consent).find(expected_row) != std::string::npos,
              "independent exact full-row consent snapshot fields/order");
    }
    Require(consent.find("The main shards total 12 bytes") != consent.npos &&
                consent.find("Total file size: 24 bytes") != consent.npos,
            "fixture consent exact totals");
#ifdef _WIN32
    const auto before_unc_requests = transport.requests;
    Require(CheckLocalPath(fs::path(L"\\\\unreachable.invalid\\share\\new"))
                    .find("UNC") != std::string::npos,
            "lexical UNC refusal occurs before existence queries or SMB/DNS");
    Require(
        CheckLocalPath(fs::path(L"\\\\?\\UNC\\unreachable.invalid\\share\\new"))
                .find("UNC") != std::string::npos,
        "extended UNC refusal occurs before existence queries or SMB/DNS");
    Require(transport.requests == before_unc_requests,
            "lexical UNC checks cannot contact the transport");
#endif
    const auto volume_real = QueryVolume(base / "new" / "nested");
    Require(volume_real.error.empty() && volume_real.free_bytes > 0,
            "nearest existing ancestor real volume query");
    // Real libcurl socket exercise against a local HF-layout mirror. No public
    // endpoint is contacted; only this test-only transport maps the origin.
    const auto mirror = base / "repo-mirror";
    const auto gguf = TinyGguf();
    const auto gguf_hash = gufo::crypto::Sha256Hex(gguf);
    auto real_entries = kEntries;
    std::map<std::string, FingerprintTable> tables;
    for (auto& entry : real_entries) {
      const auto source =
          mirror / fs::u8path(entry.source_repo) / fs::u8path(entry.repo_path);
      if (entry.kind == Kind::MainShard) {
        entry.size = gguf.size();
        entry.sha256 = gguf_hash;
        Write(source,
              std::string_view(reinterpret_cast<const char*>(gguf.data()),
                               gguf.size()));
        const auto generated = GenerateGgufFingerprint(source, entry);
        Require(bool(generated), "valid tiny GGUF fixture pin generation");
        tables.emplace(entry.logical_name, generated.table);
      } else
        Write(source, "abc");
    }
    test::LoopbackStub stub(mirror);
    NoProxy no_proxy;
    LoopbackTransport loopback(stub.url(""));
    auto socket_env = env;
    socket_env.entries = real_entries;
    socket_env.transport = &loopback;
    socket_env.fingerprint = [&](const Entry& entry, const fs::path& path) {
      return CheckGgufFingerprint(path, entry,
                                  tables.at(std::string(entry.logical_name)));
    };
    socket_env.download_options.policy.test_loopback = true;
    Options socket_options = options;
    socket_options.root = (base / "loopback").string();
    Require(Run(socket_options, socket_env, out, err) == 0,
            "real loopback eight-file folder-mirror acquisition: " + err);
    Require(stub.error().empty() && stub.records().size() >= 8,
            "loopback stub saw all fixed origins");
    const auto real_root = NativePath(fs::u8path(ModelRoot(socket_options)));
    const auto actual = real_root / "one.gguf";
    fs::last_write_time(actual,
                        fs::last_write_time(actual) + std::chrono::seconds(1));
    Require(Run(socket_options, socket_env, out, err) == 0 &&
                out == "prepare: already prepared; nothing to do\n",
            "actual GGUF size/table/prefix reuse");
    auto variant = gguf;
    variant.back() ^= 1;
    Write(actual,
          std::string_view(reinterpret_cast<const char*>(variant.data()),
                           variant.size()));
    Require(Run(socket_options, socket_env, out, err) == 7 &&
                err.find("tensor") != err.npos,
            "actual same-layout GGUF variant refusal");
    variant.pop_back();
    Write(actual,
          std::string_view(reinterpret_cast<const char*>(variant.data()),
                           variant.size()));
    Require(Run(socket_options, socket_env, out, err) == 7 &&
                err.find("size") != err.npos,
            "actual truncated GGUF refusal");
    // A Unicode selected root longer than MAX_PATH exercises native wide
    // extended-path writes, hashing, promotion, state and manifest publication.
    Options deep = options;
    deep.root = (base / "deep").string();
    for (int i = 0; i < 10; ++i)
      deep.root += "/directory-with-more-than-thirty-characters";
    const std::u8string unicode = u8"/données-模型";
    deep.root.append(reinterpret_cast<const char*>(unicode.data()),
                     unicode.size());
    Require(Run(deep, env, out, err) == 0,
            "deep Unicode extended-path model acquisition: " + err);
#ifdef _WIN32
    const auto sharing_dir = base / "sharing";
    const auto sharing_partial = sharing_dir / "payload.partial";
    const auto sharing_final = sharing_dir / "payload.gguf";
    Write(sharing_partial, "abc");
    ReadLease held(sharing_partial);
    std::thread release([lease = std::move(held)] {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    });
    const auto started = std::chrono::steady_clock::now();
    const bool promoted =
        AtomicPromote(sharing_partial, sharing_final, lock_error);
    release.join();
    Require(promoted && std::chrono::steady_clock::now() - started >=
                            std::chrono::milliseconds(750),
            "real sharing-violation rename retry");
#endif
    std::cout << "prepare_model_test: PASS consent snapshot/all rows/wrap; "
                 "eight-file acquisition; manifest-last crash/failure; "
                 "read-only no-op/stamps/verify; planted-byte refusal; "
                 "offline; FAT32/UNC/space; exclusive/stale lock; supplied "
                 "originals; cancellation; OPEN gate\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prepare_model_test: FAIL " << error.what() << '\n';
    return 1;
  }
}
