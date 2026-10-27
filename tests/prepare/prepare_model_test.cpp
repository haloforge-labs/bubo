#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/transport.hpp"
#include "src/core/prepare/verified_state.hpp"
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
  // Argument evaluation order hides `err` in "text" + err messages; this
  // prints it when BUBO_TEST_VERBOSE is set.
  if (code != 0 && std::getenv("BUBO_TEST_VERBOSE"))
    std::cerr << "  [run exit " << code << "] " << error;
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
// ---- W2R: `bubo prepare` repairs a damaged downloaded file -----------------
//
// Failure list, written before the code (testing doctrine). Each line is a way
// the repair could go wrong; the case that traps it is named after it.
//
//  P1 A damaged file (wrong size, same size with other bytes, main shard
//     found by the fingerprint, sidecar found by the full hash, --verify) is
//     still refused with exit 6 instead of being planned.            [plan]
//  P2 The plan hides the work: file_states says "reuse", the text plan has no
//     reason, "Total download size" or the space check leaves out the
//     replaced file (a replaced file needs its full size while the old copy
//     is still there).                                    [plan] [space]
//  P3 A plan, a declined question, an EOF, a cancel, an offline refusal, a
//     space refusal or a changed plan (authorize_plan) renames, deletes or
//     rewrites the damaged file.                                  [no-change]
//  P4 The damaged file is deleted before the approval, or before the new copy
//     is verified, or it is overwritten or truncated in place. Any other hard
//     link to its data then changes.                         [link] [no-change]
//  P5 The moved-aside copy is left behind after success, or the removal also
//     removes a name outside the package, or the link count is wrong.  [link]
//  P6 The file the user supplied (--mmproj, --mtp-model) is renamed, repaired
//     or accepted when it does not match; a damaged package copy next to a
//     bad supplied file is touched.                              [supplied]
//  P7 A file that only looks like part of the package (its folder is a
//     junction to another place) is renamed.                      [outside]
//  P8 The new download is not verified (a wrong byte from the server is
//     accepted), or the package marker is published after a failed download.
//                                                              [bad-bytes]
//  P9 The download fails and the damaged copy is lost, or the error does not
//     say where it is.                                              [failed]
// P10 A second damaged file gets the same moved-aside name and overwrites the
//     first; an existing file with that name is replaced.         [collision]
// P11 A stale .partial next to the damaged file breaks the repair.  [partial]
// P12 A prepared package is not "already prepared" afterwards (stamps, state
//     or marker not rewritten).                                      [plan]
// P13 Exit codes of other paths change: a foreign package marker, a wrong byte
//     from the server and a mismatching supplied file stay exit 6.
//                                         [existing cases] [supplied]
//                                         [bad-bytes]
// P14 The repair never starts because the read lease on the damaged file stays
//     open and the rename waits ten seconds (Windows).             [link]
// P15 A failed, cancelled or killed repair moves the damaged file aside first
//     and leaves a copy of tens of GB that the next run never removes. The
//     old file must keep its name until the new one has verified, and a retry
//     must finish with nothing left over.        [failed] [cancelled]
//     [bad-bytes]
// P16 The damaged file changed between the plan and the swap and is replaced
//     without a new approval.                                  [changed]
// P17 The machine plan cannot tell a repair from a missing file, or approval
//     of a repairing plan carries over to a plan that does not repair
//     (`replaces` and the digest).                                 [digest]
// P18 A file-level link to a file elsewhere is renamed like a package file.
//                                                               [symlink]
// P19 A rename refused because the model server holds the file gives no next
//     step, or leaves the package half changed.                  [sharing]
using Contents = std::map<fs::path, std::string>;
std::string Text(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  Require(bool(file), "fixture read " + path.string());
  return std::string(std::istreambuf_iterator<char>(file), {});
}
// Every regular file under root with its bytes, except the lock file that a
// refused run may rewrite with the same content.
Contents Bytes(const fs::path& root) {
  Contents result;
  for (const auto& entry : fs::recursive_directory_iterator(root))
    if (entry.is_regular_file() && entry.path().filename() != "prepare.lock")
      result[entry.path()] = Text(entry.path());
  return result;
}
std::vector<fs::path> Asides(const fs::path& root) {
  std::vector<fs::path> result;
  for (const auto& entry : fs::recursive_directory_iterator(root))
    if (entry.path().filename().string().find(".damaged-") != std::string::npos)
      result.push_back(entry.path());
  return result;
}
// The first "prepare repair:" line of a run, without its end of line.
std::string RepairLine(const std::string& out) {
  const auto at = out.find("prepare repair:");
  return at == std::string::npos ? std::string()
                                 : out.substr(at, out.find('\n', at) - at);
}
std::size_t Count(const std::string& text, std::string_view part) {
  std::size_t count = 0;
  for (auto at = text.find(part); at != std::string::npos;
       at = text.find(part, at + part.size()))
    ++count;
  return count;
}
void Bump(const fs::path& path) {
  fs::last_write_time(path,
                      fs::last_write_time(path) + std::chrono::seconds(1));
}
// A prepared eight-file package in its own folder.
struct Package {
  explicit Package(const fs::path& root) {
    options.step = Step::Model;
    options.yes = true;
    options.root = root.string();
    env = Environment(transport, fingerprints);
    model = fs::u8path(ModelRoot(options));
    std::string out, err;
    Require(Run(options, env, out, err) == 0 && transport.requests == 8,
            "repair fixture package: " + err);
  }
  Package(const Package&) = delete;
  Package& operator=(const Package&) = delete;
  Fake transport;
  int fingerprints = 0;
  Options options;
  ModelTestEnvironment env;
  fs::path model;
};
// Update failure cases, written before the refresh implementation:
// changed/missing notices or an old marker must not reject pinned payloads;
// an interrupted publication must be recoverable without downloading;
// a changed payload must still take the existing repair path.
void UpdateRefresh(const fs::path& base) {
  Package p(base / "update-refresh");
  const auto original_marker = Text(p.model / "bubo-package.json");
  std::map<fs::path, std::pair<std::string, fs::file_time_type>> payloads;
  for (const auto& entry : kEntries) {
    const auto path = p.model / fs::u8path(entry.destination);
    payloads[path] = {HashFile(path), fs::last_write_time(path)};
  }
  // The published zip is unprepared. Its embedded marker/state constructors
  // are retained in this fixture; payload bytes here remain the tiny CPU pins.
  const auto published = metadata::Parse(
      Text(fs::path(BUBO_PREPARE_FIXTURES) / "published-v0.1.0.json"));
  auto legacy_marker = metadata::Parse(original_marker);
  legacy_marker["profiles_supported"] = *published.find("model_profiles");
  auto legacy_state = *published.find("prepare_state");
  const auto state_path = p.model / ".bubo/prepare-state.json";
  const auto existing_state = metadata::Parse(Text(state_path));
  for (const auto& row : existing_state.find("files")->items()) {
    Require(row.size() == published.find("prepare_state_row_fields")->size(),
            "prepared state retains every 0.1.0 row field");
    for (const auto& field :
         published.find("prepare_state_row_fields")->items())
      Require(row.find(field.str()) != nullptr,
              "prepared state retains the published field names");
  }
  legacy_state["files"] = *existing_state.find("files");
  Write(state_path, legacy_state.dump() + "\n");
  Write(p.model / "bubo-package.json", legacy_marker.dump() + "\n");
  auto legacy_options = p.options;
  legacy_options.offline = true;
  std::string legacy_out, legacy_err;
  Require(Run(legacy_options, p.env, legacy_out, legacy_err) == 0 &&
              p.transport.requests == 8 &&
              Text(p.model / "bubo-package.json") == original_marker,
          "published stock marker and prepare-state refresh offline: " +
              legacy_err);
  auto next = p.env;
  next.recipe_id = "update-build-b";
  for (auto& notice : next.notices) {
    notice.text = "build B notice";
    notice.sha256 = gufo::crypto::Sha256Hex(
        std::span(reinterpret_cast<const std::uint8_t*>(notice.text.data()),
                  notice.text.size()));
  }
  fs::remove(p.model / "licenses/NOTICE");
  auto offline = p.options;
  offline.offline = true;
  std::string out, err;
  Require(Run(offline, next, out, err) == 0 && p.transport.requests == 8 &&
              out.find("notices refreshed") != std::string::npos,
          "first build B prepare refreshes without downloads: " + err);
  Require(Run(offline, p.env, out, err) == 0 && p.transport.requests == 8,
          "restore build A without touching payloads");
  next.before_manifest = [] { return false; };
  Require(Run(offline, next, out, err) == 3 && p.transport.requests == 8,
          "update refresh interruption reaches publication without network: " +
              err);
  Require(!fs::exists(p.model / "bubo-package.json"),
          "interrupted update withdraws readiness");
  next.before_manifest = {};
  Require(Run(offline, next, out, err) == 0 && p.transport.requests == 8,
          "build B recovers refresh offline: " + err);
  Require(Text(p.model / "bubo-package.json") != original_marker &&
              Text(p.model / "bubo-package.json").find("update-build-b") !=
                  std::string::npos,
          "build B package marker replaces build A");
  const auto current_marker = Text(p.model / "bubo-package.json");
  Write(p.model / "bubo-package.json", "\n" + current_marker);
  Require(
      Run(offline, next, out, err) == 0 && p.transport.requests == 8 &&
          Text(p.model / "bubo-package.json") == current_marker,
      "marker-only refresh restores exact current bytes without downloading");
  for (const auto& notice : next.notices)
    Require(Text(p.model / "licenses" / notice.name) == notice.text,
            "build B exact notice bytes");
  for (const auto& [path, previous] : payloads)
    Require(HashFile(path) == previous.first &&
                fs::last_write_time(path) == previous.second,
            "update preserves every shard and artefact hash and timestamp");
  Write(p.model / "mmproj/mmproj.gguf", "bad");
  Write(p.model / "licenses/NOTICE", "abc");
  Require(Run(offline, next, out, err) == 6 && p.transport.requests == 8 &&
              Text(p.model / "licenses/NOTICE") == "abc",
          "damaged payload retains notice-mismatch refusal without refreshing "
          "metadata");
  Write(p.model / "licenses/NOTICE", next.notices[1].text);
  Require(Run(offline, next, out, err) == 4 && p.transport.requests == 8,
          "update still refuses damaged payload offline");
  Require(Run(p.options, next, out, err) == 0 && p.transport.requests == 9,
          "update still repairs damaged payload online");
}
std::vector<std::string> PlanStates(Package& package, const Options& options,
                                    PreparationPlan& plan) {
  Options only = options;
  only.plan_only = true;
  only.machine = true;
  std::istringstream input;
  std::ostringstream out, err;
  Require(RunModelForTesting(only, package.env, input, out, err, &plan) == 0,
          "plan: " + err.str());
  std::vector<std::string> states;
  for (const auto& state : plan.machine["file_states"].items())
    states.push_back(state.str());
  return states;
}
bool MakeLinkedDirectory(const fs::path& link, const fs::path& target) {
  std::error_code error;
  fs::create_directory_symlink(target, link, error);
  if (!error)
    return true;
#ifdef _WIN32
  // A junction needs no privilege.
  auto from = link, to = target;
  from.make_preferred();
  to.make_preferred();
  const auto command =
      "mklink /J \"" + from.string() + "\" \"" + to.string() + "\" >nul 2>nul";
  return std::system(command.c_str()) == 0 && fs::exists(link);
#else
  return false;
#endif
}
void RepairCases(const fs::path& base) {
  constexpr std::uint64_t margin = 2ULL << 30;
  constexpr std::uint64_t control = 2 * metadata::kMaxJsonBytes;
  const std::string reason = "replaces a file that did not match";
  const std::vector<std::string> intact(8, "reuse");
  // [plan] [no-change] [space]: wrong size (mtp), same size with other bytes
  // in a sidecar (mmproj) and in a main shard (one), all at once.
  {
    Package p(base / "repair-plan");
    Write(p.model / "one.gguf", "aBc");
    Bump(p.model / "one.gguf");
    Write(p.model / "mtp" / "mtp.gguf", "ab");
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    const auto untouched = Bytes(p.model);
    PreparationPlan plan;
    const auto states = PlanStates(p, p.options, plan);
    auto expected = intact;
    expected[0] = expected[4] = expected[5] = "download";
    const auto text = Normalize(plan.text);
    Require(states == expected && !plan.ready,
            "machine plan: file_states says download for the damaged files");
    Require(Count(Compact(plan.text), Compact(reason)) == 3 &&
                text.find("Total download size: 9 bytes") != std::string::npos,
            "text plan: reason on each damaged row and their size in the "
            "download total: " +
                text);
    Require(plan.volumes.size() == 1 &&
                plan.volumes[0].required_bytes == margin + control + 9,
            "space: the replaced files count in full");
    Require(Bytes(p.model) == untouched && p.transport.requests == 8,
            "a plan changes nothing on disk and uses no network");
    {
      // [digest] `replaces` runs parallel to the sources and names the cause.
      const auto& why = plan.machine["replaces"].items();
      Require(why.size() == 8 &&
                  why[0].str() == "synthetic fingerprint mismatch" &&
                  why[4].str() == "size differs from the pin" &&
                  why[5].str() == "SHA-256 differs from the pin" &&
                  why[1].str().empty() && why[6].str().empty() &&
                  why[7].str().empty(),
              "machine plan: replaces names each damaged file and why");
    }
    std::string out, err;
    auto decline = p.options;
    decline.yes = false;
    Require(Run(decline, p.env, out, err, "n\n") == 3 &&
                Bytes(p.model) == untouched,
            "declined question changes nothing");
    Require(
        Run(decline, p.env, out, err, "") == 3 && Bytes(p.model) == untouched,
        "EOF changes nothing");
    auto cancelled = p.env;
    cancelled.cancelled = [] { return true; };
    Require(
        Run(p.options, cancelled, out, err) == 3 && Bytes(p.model) == untouched,
        "cancel changes nothing");
    auto refused = p.options;
    refused.authorize_plan = [](const gufo::json::Value&) { return false; };
    Require(Run(refused, p.env, out, err) == 3 &&
                err.find("plan changed") != std::string::npos &&
                Bytes(p.model) == untouched,
            "a plan that was not authorised changes nothing: " + err);
    auto offline = p.options;
    offline.offline = true;
    Require(Run(offline, p.env, out, err) == 4 && p.transport.requests == 8 &&
                Bytes(p.model) == untouched,
            "offline: exit 4 before any change: " + err);
    auto tight = p.env;
    tight.volume = [&](const fs::path&) {
      return VolumeInfo{
          "fixture-volume", margin + control + 8, "NTFS", false, {}};
    };
    Require(Run(p.options, tight, out, err) == 7 &&
                err.find("Not enough free space on") != std::string::npos &&
                Bytes(p.model) == untouched,
            "one byte short of the replaced files: exit 7, nothing changed");
    tight.volume = [&](const fs::path&) {
      return VolumeInfo{
          "fixture-volume", margin + control + 9, "NTFS", false, {}};
    };
    Require(Run(p.options, tight, out, err) == 0 && p.transport.requests == 11,
            "exactly enough space for the replaced files: " + err);
    // The inverted case: downloaded, verified, swapped in, old copy gone.
    Require(Count(out, "did not match (") == 3 &&
                Count(out, "); replaced") == 3 &&
                out.find("could not delete") == std::string::npos,
            "the run says what it replaced");
    for (const auto& entry : kEntries)
      Require(HashFile(p.model / fs::u8path(entry.destination)) == kAbc,
              "every file has the pinned hash after the repair");
    Require(Asides(p.model).empty(), "the damaged copies are gone");
    Require(fs::exists(p.model / "bubo-package.json"), "marker is back");
    Require(Run(p.options, p.env, out, err) == 0 &&
                out == "prepare: already prepared; nothing to do\n" &&
                p.transport.requests == 11,
            "[P12] a repaired package is prepared: no network, no writes");
  }
  // [plan] --verify finds a main shard whose size is right and bytes are not.
  {
    Package p(base / "repair-verify");
    Write(p.model / "two.gguf", "qrs");
    auto verify = p.options;
    verify.verify = true;
    std::string out, err;
    Require(Run(verify, p.env, out, err) == 0 && p.transport.requests == 9 &&
                HashFile(p.model / "two.gguf") == kAbc &&
                Asides(p.model).empty(),
            "--verify repairs a damaged main shard: " + err);
  }
  // [link] A second hard link to the damaged data must not change. While the
  // new copies download, the damaged files stay at their names and nothing
  // else exists beside them; the swap happens after each new copy verifies.
  {
    Package p(base / "repair-link");
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    Write(p.model / "two.gguf", "qrs");
    Bump(p.model / "two.gguf");
    const auto outside = base / "repair-link-outside";
    fs::create_directories(outside);
    fs::create_hard_link(p.model / "mmproj" / "mmproj.gguf",
                         outside / "mmproj-twin.gguf");
    fs::create_hard_link(p.model / "two.gguf", outside / "two-twin.gguf");
    Require(fs::hard_link_count(outside / "mmproj-twin.gguf") == 2 &&
                fs::hard_link_count(outside / "two-twin.gguf") == 2,
            "fixture: two names for each damaged file");
    std::vector<std::string> faults;
    int call = 0;
    p.transport.before_body = [&] {
      ++call;
      // two.gguf comes first, mmproj.gguf later.
      if (!Asides(p.model).empty())
        faults.push_back(std::to_string(call) + ": a copy exists mid-download");
      if (Text(p.model / "mmproj" / "mmproj.gguf") != "xyz" ||
          fs::hard_link_count(outside / "mmproj-twin.gguf") != 2)
        faults.push_back(std::to_string(call) + ": mmproj touched early");
      if (call == 1) {
        if (Text(p.model / "two.gguf") != "qrs" ||
            fs::hard_link_count(outside / "two-twin.gguf") != 2)
          faults.push_back("1: the damaged file was touched before its swap");
      } else if (call == 2) {
        if (HashFile(p.model / "two.gguf") != kAbc)
          faults.push_back("2: the first new copy is not in place");
        if (fs::hard_link_count(outside / "two-twin.gguf") != 1)
          faults.push_back("2: removing the copy left its link count");
      }
    };
    std::string out, err;
    const auto started = std::chrono::steady_clock::now();
    Require(Run(p.options, p.env, out, err) == 0 && call == 2,
            "hard-link repair: " + err);
    Require(
        std::chrono::steady_clock::now() - started < std::chrono::seconds(8),
        "[P14] no lease keeps the rename waiting");
    Require(faults.empty(), faults.empty() ? "" : faults.front());
    Require(Text(outside / "mmproj-twin.gguf") == "xyz" &&
                Text(outside / "two-twin.gguf") == "qrs",
            "the other names keep the old bytes: nothing was written through");
    Require(fs::hard_link_count(outside / "mmproj-twin.gguf") == 1 &&
                fs::hard_link_count(outside / "two-twin.gguf") == 1 &&
                fs::hard_link_count(p.model / "mmproj" / "mmproj.gguf") == 1 &&
                fs::hard_link_count(p.model / "two.gguf") == 1,
            "each link count dropped by one");
    Require(HashFile(p.model / "mmproj" / "mmproj.gguf") == kAbc &&
                HashFile(p.model / "two.gguf") == kAbc &&
                Asides(p.model).empty(),
            "repaired and no damaged copy left");
  }
  // [supplied] A supplied file that does not match: exit 6, nothing renamed,
  // and the damaged package copy next to it is not touched either.
  {
    Package p(base / "repair-supplied");
    Write(p.model / "mmproj" / "mmproj.gguf", "qrs");
    const auto input = base / "repair-supplied-input";
    Write(input / "bad-mmproj.gguf", "xyz");
    Write(input / "good-mmproj.gguf", "abc");
    const auto untouched = Bytes(p.model);
    auto bad = p.options;
    bad.mmproj = (input / "bad-mmproj.gguf").string();
    std::string out, err;
    Require(
        Run(bad, p.env, out, err) == 6 &&
            err.find("supplied file does not match") != std::string::npos &&
            Bytes(p.model) == untouched &&
            Text(input / "bad-mmproj.gguf") == "xyz" && Asides(input).empty() &&
            p.transport.requests == 8,
        "a supplied file that does not match: exit 6, nothing moved: " + err);
    // A good supplied file replaces only the package copy; it is read, not
    // changed, and no byte comes from the network.
    auto good = p.options;
    good.mmproj = (input / "good-mmproj.gguf").string();
    PreparationPlan plan;
    auto expected = intact;
    expected[5] = "import";
    Require(PlanStates(p, good, plan) == expected &&
                Count(Compact(plan.text), Compact(reason)) == 1,
            "a good supplied file: the damaged package copy is an import");
    Require(Run(good, p.env, out, err) == 0 && p.transport.requests == 8 &&
                Text(p.model / "mmproj" / "mmproj.gguf") == "abc" &&
                Text(input / "good-mmproj.gguf") == "abc" &&
                fs::hard_link_count(input / "good-mmproj.gguf") == 1 &&
                Asides(p.model).empty(),
            "import repairs the package copy: " + err);
  }
  // [outside] A folder of the package that is a link to another place holds
  // someone else's file: refused as before, not renamed.
  {
    Package p(base / "repair-outside");
    const auto elsewhere = base / "repair-outside-target";
    Write(elsewhere / "twin.gguf", "xyz");
    fs::remove_all(p.model / "twin");
    Require(MakeLinkedDirectory(p.model / "twin", elsewhere),
            "fixture: a folder junction or link");
    std::string out, err;
    const auto code = Run(p.options, p.env, out, err);
    const auto kept = Text(elsewhere / "twin.gguf");
    fs::remove(p.model / "twin");  // The link only.
    Require(code == 6 && err.find("twin.gguf") != std::string::npos &&
                err.find("delete it and run bubo prepare again") !=
                    std::string::npos,
            "a file reached through a link out of the package: exit 6: " + err);
    Require(kept == "xyz" && Asides(elsewhere).empty() &&
                fs::exists(elsewhere / "twin.gguf"),
            "the other place keeps its file and its name");
  }
  // [bad-bytes] [failed] [cancelled] The new download is verified. A repair
  // that fails or is cancelled leaves the damaged file at its name and nothing
  // else behind; the retry finishes the job.
  {
    Package p(base / "repair-bad-bytes");
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    p.transport.wrong_byte = true;
    std::string out, err;
    Require(Run(p.options, p.env, out, err) == 6 &&
                !fs::exists(p.model / "bubo-package.json"),
            "a wrong byte from the server is refused (exit 6), no marker");
    Require(Text(p.model / "mmproj" / "mmproj.gguf") == "xyz" &&
                Asides(p.model).empty(),
            "[P15] the damaged file keeps its name, no copy is left");
    p.transport.wrong_byte = false;
    Require(Run(p.options, p.env, out, err) == 0 &&
                HashFile(p.model / "mmproj" / "mmproj.gguf") == kAbc &&
                Asides(p.model).empty() &&
                fs::exists(p.model / "bubo-package.json"),
            "the retry repairs it with nothing left over");
  }
  {
    Package p(base / "repair-failed");
    Write(p.model / "draft_head" / "head.gguf", "xyz");
    p.transport.fail_last = true;
    std::string out, err;
    Require(Run(p.options, p.env, out, err) == 5 &&
                !fs::exists(p.model / "bubo-package.json"),
            "a failed transfer: exit 5, no marker");
    Require(Text(p.model / "draft_head" / "head.gguf") == "xyz" &&
                Asides(p.model).empty(),
            "[P15] the damaged file keeps its name, no copy is left");
    p.transport.fail_last = false;
    Require(Run(p.options, p.env, out, err) == 0 &&
                HashFile(p.model / "draft_head" / "head.gguf") == kAbc &&
                Asides(p.model).empty() &&
                out.find("); replaced") != std::string::npos,
            "the retry plans the repair again and finishes it");
  }
  {
    Package p(base / "repair-cancelled");
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    bool stop = false;
    p.env.cancelled = [&] { return stop; };
    p.transport.before_body = [&] { stop = true; };
    std::string out, err;
    Require(Run(p.options, p.env, out, err) == 3 &&
                !fs::exists(p.model / "bubo-package.json"),
            "a cancelled repair: exit 3");
    Require(Text(p.model / "mmproj" / "mmproj.gguf") == "xyz" &&
                Asides(p.model).empty(),
            "[P15] the damaged file keeps its name, no copy is left");
    stop = false;
    p.transport.before_body = {};
    Require(Run(p.options, p.env, out, err) == 0 &&
                HashFile(p.model / "mmproj" / "mmproj.gguf") == kAbc &&
                Asides(p.model).empty(),
            "the retry finishes the repair with nothing left over");
  }
  // [changed] The damaged file changes after the plan: no swap, exit 3.
  {
    Package p(base / "repair-changed");
    const auto file = p.model / "mmproj" / "mmproj.gguf";
    Write(file, "xyz");
    p.transport.before_body = [&] { Write(file, "qqq"); };
    std::string out, err;
    Require(
        Run(p.options, p.env, out, err) == 3 &&
            err.find("plan changed; approval required") != std::string::npos,
        "[P16] a changed file is not replaced without a new approval");
    Require(Text(file) == "qqq" && Asides(p.model).empty(),
            "the changed file keeps its name and bytes");
    p.transport.before_body = {};
    Require(Run(p.options, p.env, out, err) == 0 && HashFile(file) == kAbc &&
                Asides(p.model).empty(),
            "the next run plans it again and repairs it");
  }
#ifdef _WIN32
  // [sharing] Another program (the model server) holds the damaged file: the
  // swap rename is refused, the error says what to do, nothing is half done.
  {
    Package p(base / "repair-sharing");
    const auto file = p.model / "mmproj" / "mmproj.gguf";
    Write(file, "xyz");
    HANDLE held = INVALID_HANDLE_VALUE;
    p.transport.before_body = [&] {
      held =
          CreateFileW(NativePath(file).c_str(), GENERIC_READ, FILE_SHARE_READ,
                      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    };
    std::string out, err;
    const auto code = Run(p.options, p.env, out, err);
    if (held != INVALID_HANDLE_VALUE)
      CloseHandle(held);
    Require(held != INVALID_HANDLE_VALUE && code == 7 &&
                err.find("Stop Bubo's model server, then run prepare again.") !=
                    std::string::npos,
            "[P19] a held file: exit 7 with the next step: " + err);
    Require(Text(file) == "xyz" && Asides(p.model).empty(),
            "the held file is unchanged and no copy is left");
    p.transport.before_body = {};
    Require(Run(p.options, p.env, out, err) == 0 && HashFile(file) == kAbc,
            "once the file is free, prepare finishes the repair");
  }
#endif
  // [symlink] A file-level link to a file elsewhere is the user's own file.
  {
    Package p(base / "repair-symlink");
    const auto elsewhere = base / "repair-symlink-target";
    Write(elsewhere / "mmproj.gguf", "xyz");
    const auto link = p.model / "mmproj" / "mmproj.gguf";
    fs::remove(link);
    std::error_code made;
    fs::create_symlink(elsewhere / "mmproj.gguf", link, made);
    if (made) {
      // Creating a file link needs a privilege on Windows; say so, not pass.
      std::cerr << "prepare_model_test: NOTE symlink case not run: "
                << made.message() << '\n';
    } else {
      std::string out, err;
      const auto code = Run(p.options, p.env, out, err);
      const bool still_link = fs::is_symlink(link);
      const auto kept = Text(elsewhere / "mmproj.gguf");
      fs::remove(link);
      Require(
          code == 6 && err.find("mmproj.gguf") != std::string::npos &&
              still_link && kept == "xyz" && Asides(elsewhere).empty() &&
              Asides(p.model).empty(),
          "[P18] a file link out of the package: exit 6, not renamed: " + err);
    }
  }
  // [digest] A plan that replaces a file is not the plan that did not.
  {
    Package p(base / "repair-digest");
    PreparationPlan before, after;
    PlanStates(p, p.options, before);
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    PlanStates(p, p.options, after);
    Require(PlanDigest(before.machine) != PlanDigest(after.machine) &&
                before.ready && !after.ready,
            "[P17] the digest changes when a file becomes damaged");
    for (const auto& why : before.machine["replaces"].items())
      Require(why.str().empty(), "an intact package replaces nothing");
  }
  // [collision] [partial] Names that exist are never replaced; a stale
  // partial file does not stop the repair.
  {
    Package p(base / "repair-collision");
    Write(p.model / "mmproj" / "mmproj.gguf", "xyz");
    Write(p.model / "mmproj" / "mmproj.gguf.partial", "zz");
    const auto now = std::chrono::system_clock::now();
    std::vector<fs::path> taken;
    for (int offset = -2; offset <= 3; ++offset) {
      const auto time = std::chrono::system_clock::to_time_t(
          now + std::chrono::seconds(offset));
      std::tm parts{};
#ifdef _WIN32
      gmtime_s(&parts, &time);
#else
      gmtime_r(&time, &parts);
#endif
      char stamp[32]{};
      std::strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%SZ", &parts);
      taken.push_back(p.model / "mmproj" /
                      (std::string("mmproj.gguf.damaged-") + stamp));
      Write(taken.back(), "keep-me");
    }
    std::string out, err;
    Require(Run(p.options, p.env, out, err) == 0 &&
                HashFile(p.model / "mmproj" / "mmproj.gguf") == kAbc,
            "repair beside a stale partial file: " + err);
    for (const auto& path : taken)
      Require(fs::exists(path) && Text(path) == "keep-me",
              "an existing file with the moved-aside name is not replaced");
    Require(Asides(p.model).size() == taken.size() &&
                !fs::exists(p.model / "mmproj" / "mmproj.gguf.partial"),
            "only the moved-aside copy of the damaged file was removed");
  }
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
    UpdateRefresh(base);
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
    auto snapshot = Files(model);
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
    // An older Bubo marker is managed metadata and is refreshed in place.
    {
      const auto marker = model / "bubo-package.json";
      const auto text = metadata::Read(marker);
      const auto time = fs::last_write_time(marker);
      const std::string from = "\"engine_min\":\"0.1.0\"";
      const auto at = text.find(from);
      Require(at != std::string::npos, "marker names engine_min");
      auto other = text;
      other.replace(at, from.size(), "\"engine_min\":\"0.0.9\"");
      Write(marker, other);
      Require(Run(options, env, out, err) == 0 && err.empty() &&
                  transport.requests == 8 && metadata::Read(marker) == text,
              "an older package marker is refreshed without downloads: " + err);
      fs::last_write_time(marker, time);
      for (const auto& entry : kEntries) {
        const auto path = model / fs::u8path(entry.destination);
        Require(Files(model).at(path) == snapshot.at(path) &&
                    HashFile(path) == entry.sha256,
                "marker refresh preserves payload bytes and timestamps");
      }
      snapshot = Files(model);
    }
    options.verify = true;
    Require(Run(options, env, out, err) == 0 &&
                out.find("prepare verify: 8/8") != out.npos,
            "verify eight files progress");
    const auto after_verify = Files(model);
    for (const auto& [path, stamp] : snapshot)
      if (fs::is_regular_file(path) &&
          path.filename() != "verified-state.json" &&
          path.filename() != "verified-state.lock")
        Require(after_verify.at(path) == stamp,
                "verify preserves payloads and preparation metadata");
    Require(transport.requests == 8 && ReadVerifiedStamps(model).size() == 8,
            "verify refreshes eight observed stamp rows without transfers");
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
    // W2R: this case used to assert exit 6 and "delete it and run bubo
    // prepare again". A damaged file that Bubo placed is now replaced: the
    // changed stamp triggers the fingerprint (plan, then plan again under the
    // lock), and one download repairs the file.
    const auto before_repair = transport.requests;
    Require(Run(options, env, out, err) == 0 && fingerprints == 2 &&
                transport.requests == before_repair + 1,
            "changed main stamp triggers the fingerprint; one download "
            "repairs the file (exit 0): " +
                err);
    Require(
        RepairLine(out).find("one.gguf did not match") != std::string::npos &&
            HashFile(corrupt) == kAbc &&
            fs::exists(model / "bubo-package.json"),
        "the damaged main shard was replaced");
    RepairCases(base);
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
                err.find("Not enough free space on") != err.npos,
            "new allocation plus margin, plain message: " + err);
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
    // W2R: both used to be refused with exit 6. The fingerprint still finds
    // the damage (its reason is in the repair line); the file is downloaded
    // again through the real loopback socket and verified.
    Require(Run(socket_options, socket_env, out, err) == 0 &&
                RepairLine(out).find("tensor") != std::string::npos &&
                HashFile(actual) == gguf_hash,
            "actual same-layout GGUF variant is found and repaired: " + err);
    variant.pop_back();
    Write(actual,
          std::string_view(reinterpret_cast<const char*>(variant.data()),
                           variant.size()));
    Require(Run(socket_options, socket_env, out, err) == 0 &&
                RepairLine(out).find("size differs from the pin") !=
                    std::string::npos &&
                HashFile(actual) == gguf_hash,
            "actual truncated GGUF is found and repaired: " + err);
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
#ifdef _WIN32
    std::string unicode_root = "ROOT/";
    for (unsigned i = 0; i < 70; ++i)
      unicode_root += "\xe7\x95\x8c";
    const auto unicode_consent =
        ModelConsent(kEntries, fs::u8path(unicode_root), {}, 24, {}, true);
    Require(MultiByteToWideChar(
                CP_UTF8, MB_ERR_INVALID_CHARS, unicode_consent.data(),
                static_cast<int>(unicode_consent.size()), nullptr, 0) > 0,
            "long Unicode path consent must remain valid UTF-8 after wrapping");
    std::istringstream unicode_lines(unicode_consent);
    for (std::string line; std::getline(unicode_lines, line);)
      Require(line.size() <= 80, "Unicode consent byte width exceeds 80");
#endif
    std::cout << "prepare_model_test: PASS consent snapshot/all rows/wrap; "
                 "eight-file acquisition; manifest-last crash/failure; "
                 "read-only no-op/stamps/verify; planted-byte repair; "
                 "offline; FAT32/UNC/space; exclusive/stale lock; supplied "
                 "originals; cancellation; OPEN gate; damaged-file repair "
                 "(plan, no change, hard link, supplied, outside, failed)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prepare_model_test: FAIL " << error.what() << '\n';
    return 1;
  }
}
