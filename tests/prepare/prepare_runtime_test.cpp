#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepare/runtime_prepare.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "tests/prepare/loopback_stub.hpp"
#include "tests/prepare/zip_fixture.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
using namespace bubo::prepare::test;
std::size_t checks = 0;
void Check(bool ok, std::string text) {
  ++checks;
  if (!ok)
    throw std::runtime_error(std::move(text));
}
std::string Read(const fs::path& p) {
  std::ifstream file(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
std::string Normalize(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}
void Replace(std::string& text, std::string_view from, std::string_view to) {
  for (auto p = text.find(from); p != text.npos;
       p = text.find(from, p + to.size()))
    text.replace(p, from.size(), to);
}
using Snapshot = std::map<std::string, std::pair<std::string, FileStamp>>;
Snapshot Snap(const fs::path& root, const RuntimeTestEnvironment& env) {
  Snapshot result;
  std::vector<std::string> names;
  for (const auto& m : env.members)
    names.push_back(m.destination);
  for (const auto& n : env.notices)
    names.push_back("licenses/" + std::string(n.path));
  names.push_back(".bubo/runtime-ready.json");
  for (const auto& n : names) {
    const auto p = root / fs::u8path(n);
    if (fs::exists(p)) {
      ReadLease lease(p);
      result[n] = {HashFile(p), lease.Snapshot()};
    }
  }
  return result;
}
struct RunResult {
  int code;
  std::string out, err;
};
RunResult Run(const Options& options, const RuntimeTestEnvironment& env,
              std::string answer = "") {
  std::istringstream input(answer);
  std::ostringstream out, err;
  const auto code = RunRuntimeForTesting(options, env, input, out, err);
  return {code, out.str(), err.str()};
}
std::map<std::string, fs::file_time_type> Times(const fs::path& root) {
  std::map<std::string, fs::file_time_type> times;
  const auto remember = [&](const fs::path& path) {
    const auto utf8 = path.generic_u8string();
    times.emplace(std::string(utf8.begin(), utf8.end()),
                  fs::last_write_time(path));
  };
  // The fixture includes Unicode destinations beyond MAX_PATH. Native paths
  // keep traversal valid; UTF-8 keys avoid the Windows ANSI code page.
  const auto native_root = NativePath(root);
  for (const auto& f : fs::recursive_directory_iterator(native_root))
    remember(f.path());
  remember(native_root);
  return times;
}
void PublishedRuntimeMetadata(const fs::path& root) {
  const auto fixture = metadata::Parse(
      Read(fs::path(BUBO_PREPARE_FIXTURES) / "published-v0.1.0.json"));
  const auto& archive = *fixture.find("archive");
  constexpr std::string_view archive_sha =
      "d1c92dacf68e8e379b8e02c8d68ba18ba102b2fd22784e25cb06b0757fac1306";
  Check(archive.member_str("sha256") == archive_sha &&
            archive.member_str("revision") ==
                "3ff74dce9a67182e572dbddbd8279c058e60fe31" &&
            archive.member_str("version") == "0.1.0",
        "fixture identifies the verified published 0.1.0 archive");
  RuntimeTestEnvironment published{};
  const auto table = fixture.member_str("runtime_members");
  published.members = LoadRuntimeTable(table);
  Check(published.members.size() == 263 &&
            SerializeRuntimeTable(CompiledRuntimeTable()) == table,
        "all published runtime member paths, sizes and hashes still match");
  for (const auto& row : fixture.find("runtime_notices")->items()) {
    const auto& text = row.find("text")->str();
    Check(gufo::crypto::Sha256Hex(
              std::span(reinterpret_cast<const std::uint8_t*>(text.data()),
                        text.size())) == row.member_str("sha256"),
          "published notice text matches its embedded hash");
    published.notices.push_back(
        {row.find("path")->str(), row.find("sha256")->str(), text});
  }
  Check(published.notices.size() == 70,
        "all published runtime notices are retained independently");
  const auto dir = root / "published-metadata";
  WriteFixture(dir / ".bubo/runtime-ready.json",
               fixture.find("runtime_ready")->dump() + "\n");
  const auto error = RuntimeReadyErrorForTesting(dir, published);
  Check(error.find("missing runtime file or size mismatch") != error.npos &&
            error.find("readiness entry mismatch") == error.npos,
        "published readiness shape and digest pass before payload checks");
}
}  // namespace
int main(int argc, char** argv) {
  try {
#ifdef _WIN32
    if (argc == 3 && (std::string_view(argv[1]) == "--crash" ||
                      std::string_view(argv[1]) == "--refresh-crash")) {
      const auto run_root = fs::u8path(argv[2]);
      RuntimeTestEnvironment crash;
      crash.wheels = RuntimeWheels();
      crash.notices = RuntimeNotices();
      if (std::string_view(argv[1]) == "--refresh-crash")
        for (auto& notice : crash.notices) {
          notice.text = "abc";
          notice.sha256 = kAbcHash;
        }
      std::array<fs::path, 3> paths;
      std::array<std::string, 3> hashes;
      for (unsigned i = 0; i < 3; ++i) {
        paths[i] = run_root / "cache" / fs::u8path(crash.wheels[i].file);
        hashes[i] = HashFile(paths[i]);
        crash.wheels[i].sha256 = hashes[i];
        crash.wheels[i].size = fs::file_size(paths[i]);
      }
      crash.members = GenerateRuntimeTable(paths, false);
      crash.holders = [] { return std::vector<std::string>{}; };
      crash.after_install = [](std::size_t i) {
        if (i == 0)
          ExitProcess(42);
        return true;
      };
      Options o;
      o.step = Step::Runtime;
      o.yes = true;
      o.offline = true;
      o.root = run_root.string();
      o.cache_dir = (run_root / "cache").string();
      o.runtime_dir = (run_root / "install").string();
      return Run(o, crash).code;
    }
#endif
    const auto root = fs::absolute(argc > 1 ? fs::u8path(argv[1])
                                            : fs::path("j3-runtime-fixture"));
    Check(!fs::exists(root), "fresh runtime fixture root required");
    fs::create_directories(root);
    PublishedRuntimeMetadata(root);
    const auto consent_path = argc > 2 ? fs::u8path(argv[2])
                                       : fs::path(BUBO_PREPARE_FIXTURES) /
                                             "runtime-consent-source.md";
    auto source = Normalize(Read(consent_path));
    const auto start = source.find("```\n") + 4,
               end = source.find("```", start);
    Check(start != std::string::npos && end != std::string::npos,
          "binding consent fences");
    auto expected = source.substr(start, end - start);
    Replace(expected, "<per-user folder>", "C:\\Users\\fixture\\Bubo");
    Replace(expected, "<licenses folder>",
            "C:\\Users\\fixture\\Bubo\\licenses");
    Check(RuntimeConsent(fs::path("C:\\Users\\fixture\\Bubo"),
                         fs::path("C:\\Users\\fixture\\Bubo\\licenses")) ==
              expected,
          "binding consent exact equality");
    Check(expected.find("Other AMD terms can apply to these files. Bubo cannot "
                        "confirm which. If you continue, you accept the terms "
                        "that apply to the AMD files.") != expected.npos,
          "binding additional AMD terms sentence");
    for (const auto& n : RuntimeNotices()) {
      Check(Read(fs::path(BUBO_NOTICE_SOURCE) / fs::u8path(n.path)) == n.text,
            "exact release notice bytes: " + std::string(n.path));
      Check(HashFile(fs::path(BUBO_NOTICE_SOURCE) / fs::u8path(n.path)) ==
                n.sha256,
            "notice source digest");
    }
    std::cout << "PASS: printed consent equals " << consent_path.string()
              << " after code-filled placeholders; compiled pins; 70 exact "
                 "release notice bytes\n";
    const std::array<fs::path, 3> wheel_paths{root / "source/core.whl",
                                              root / "source/libraries.whl",
                                              root / "source/device.whl"};
    WriteFixture(wheel_paths[0],
                 FixtureZip({{"_rocm_sdk_core/bin/amdhip64_7.dll", "abc", true},
                             {"padding", std::string(2 * 1024 * 1024, 'x')}}));
    WriteFixture(wheel_paths[1],
                 FixtureZip({{"_rocm_sdk_libraries/bin/rocblas.dll"}}));
    WriteFixture(
        wheel_paths[2],
        FixtureZip(
            {{"_rocm_sdk_libraries/bin/rocblas/library/gfx1151/a.co"},
             {"_rocm_sdk_libraries/bin/hipblaslt/library/gfx1151/a.dat"}},
            true));
    RuntimeTestEnvironment env;
    env.wheels = RuntimeWheels();
    env.members = GenerateRuntimeTable(wheel_paths, false);
    env.notices = RuntimeNotices();
    std::array<std::string, 3> hashes;
    for (unsigned i = 0; i < 3; ++i) {
      hashes[i] = HashFile(wheel_paths[i]);
      env.wheels[i].size = fs::file_size(wheel_paths[i]);
      env.wheels[i].sha256 = hashes[i];
    }
    bool corrupt = false, drop = false;
    LoopbackStub stub([&](const StubRequest& request) {
      unsigned i = request.target == "/core"        ? 0
                   : request.target == "/libraries" ? 1
                                                    : 2;
      auto response = FileReply(request, wheel_paths[i]);
      if (corrupt && i == 0)
        response.corrupt_at = 50;
      if (drop && i == 0 && !request.headers.contains("range"))
        response.drop_after = 1024 * 1024;
      return response;
    });
    Check(stub.error().empty(), stub.error());
    HostPolicy policy;
    policy.source = HostPolicy::Source::Amd;
    policy.test_loopback = true;
    CurlTransport transport(policy);
    env.transport = &transport;
    env.download_options.policy = policy;
    env.download_options.wait = [](auto) {};
    env.origins = {stub.url("/core"), stub.url("/libraries"),
                   stub.url("/device")};
    env.volume = [](const fs::path&) {
      return VolumeInfo{
          "fixture-local", 20ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
    };
    env.holders = [] { return std::vector<std::string>{}; };
    auto options = [&](const char* name) {
      Options o;
      o.step = Step::Runtime;
      o.yes = true;
      o.root = (root / name).string();
      o.runtime_dir = (root / name / "install").string();
      o.cache_dir = (root / name / "cache").string();
      return o;
    };
    auto first = options("success");
    {
      auto preview = first;
      preview.plan_only = true;
      preview.machine = true;
      PreparationPlan plan;
      std::istringstream input;
      std::ostringstream output, errors;
      Check(RunRuntimeForTesting(preview, env, input, output, errors, &plan) ==
                    0 &&
                !plan.machine["volumes"].items().empty(),
            "runtime_remaining_space_plan_without_download");
      for (const auto& volume : plan.machine["volumes"].items())
        Check(volume.member_size("safety_margin_bytes") == 2147483648ULL &&
                  volume.member_size("bytes_to_write") > 0 &&
                  volume.member_size("free_after_download_bytes") ==
                      20ULL * 1024 * 1024 * 1024 -
                          volume.member_size("bytes_to_write"),
              "runtime_remaining_space_uses_planned_writes_only");
      Check(stub.records().empty(), "runtime_plan_has_zero_network_requests");
    }
    drop = true;
    const auto result = Run(first, env);
    drop = false;
    Check(result.code == 0, "runtime acquisition: " + result.err);
    // --runtime-dir puts the runtime where later commands do not look: the
    // output names the variable that makes them find it, with the folder.
    const auto note = "prepare runtime: set BUBO_RUNTIME_DIR=" +
                      first.runtime_dir +
                      " so other bubo commands find this runtime\n";
    Check(result.out.find(note) != result.out.npos,
          "runtime output names BUBO_RUNTIME_DIR and the folder: " +
              result.out);
    bool ranged = false;
    for (const auto& r : stub.records())
      ranged |= r.headers.contains("range");
    Check(ranged, "native J2 resume after dropped wheel");
    Check(result.out.find(RuntimeConsent(
              fs::path(first.runtime_dir),
              fs::path(first.runtime_dir) / "licenses")) != result.out.npos &&
              result.out.find("approval: --yes") != result.out.npos,
          "--yes logs exact consent");
    const auto installed = Snap(fs::path(first.runtime_dir), env);
    Check(installed.size() == env.members.size() + env.notices.size() + 1,
          "exact selected payload plus notices/ready");
    Check(!fs::exists(fs::path(first.runtime_dir) / "padding"),
          "unselected wheel content absent");
    const auto before = Times(root);
    const auto requests = stub.records().size();
    for (unsigned i = 0; i < 3; ++i) {
      const auto noop = Run(first, env);
      Check(
          noop.code == 0 &&
              noop.out ==
                  "prepare runtime: already prepared; nothing to do\n" + note,
          "read-only no-op still names BUBO_RUNTIME_DIR");
    }
    first.verify = true;
    first.offline = true;
    Check(Run(first, env).code == 0, "verified/offline no-op");
    first.verify = false;
    first.offline = false;
    Check(stub.records().size() == requests && Times(root) == before,
          "idempotence: zero requests and unchanged all file/directory mtimes");
    std::cout << "PASS: native stub wheel drop/Range resume; archive/member "
                 "pins; exact selection; notices and ready-last; --yes log; "
                 "no-op/offline/verify zero network/writes\n";
    // Update failure cases (written first): build B's notices/marker must
    // refresh without wheel access or member replacement; interruption must
    // roll back and allow an offline retry; payload mismatch still refuses.
    {
      auto updated = env;
      for (auto& notice : updated.notices) {
        notice.text = "abc";
        notice.sha256 = kAbcHash;
      }
      const auto install = fs::path(first.runtime_dir);
      auto previous = Snap(install, env);
      auto refresh = first;
      refresh.machine = true;
      refresh.offline = true;
      refresh.cache_dir = (root / "absent-wheel-cache").string();
      {
        const auto unchanged = Times(root);
        PreparationPlan plan;
        std::istringstream input;
        std::ostringstream out, err;
        Check(RunRuntimeForTesting(refresh, updated, input, out, err, &plan) ==
                      0 &&
                  !plan.ready && Times(root) == unchanged &&
                  stub.records().size() == requests,
              "notice refresh preview is read-only and needs no wheel cache");
        const auto* states = plan.machine.find("file_states");
        Check(states && states->is_array() &&
                  states->size() == updated.wheels.size(),
              "refresh plan has one file state per wheel source");
        for (const auto& state : states->items())
          Check(state.str() == "reuse", "refresh plan reuses every source");
        auto refused = refresh;
        refused.authorize_plan = [&](const auto& authorized) {
          Check(authorized.dump() == plan.machine.dump(),
                "refresh authorization matches the preview including reuse");
          return false;
        };
        Check(Run(refused, updated).code == 3 && Times(root) == unchanged,
              "unapproved runtime refresh changes nothing");
      }
#ifdef _WIN32
      bool payload_protected = false;
      updated.after_install = [&](std::size_t i) {
        if (i == 0) {
          const auto member = install / fs::u8path(env.members[0].destination);
          const auto handle = CreateFileW(
              member.c_str(), GENERIC_WRITE,
              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
          payload_protected = handle == INVALID_HANDLE_VALUE &&
                              GetLastError() == ERROR_SHARING_VIOLATION;
          if (handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
        }
        return true;
      };
#endif
      Check(
          Run(refresh, updated).code == 0 && stub.records().size() == requests,
          "first build B runtime prepare refreshes offline");
#ifdef _WIN32
      Check(payload_protected,
            "payload stays protected through metadata publication");
#endif
      updated.after_install = {};
      Check(Run(refresh, env).code == 0,
            "restore build A before interrupted refresh");
      previous = Snap(install, env);
      updated.after_install = [](std::size_t) { return false; };
      const auto interrupted = Run(refresh, updated);
      Check(interrupted.code == 7 && Snap(install, env) == previous,
            "interrupted notice refresh rolls back: " + interrupted.err);
      updated.after_install = {};
      fs::remove(install / "licenses" / fs::u8path(updated.notices[0].path));
      const auto upgraded = Run(refresh, updated);
      Check(upgraded.code == 0 && stub.records().size() == requests &&
                upgraded.out.find("already prepared") != upgraded.out.npos &&
                upgraded.out.find("notices refreshed") != upgraded.out.npos,
            "build B refresh offline without wheel cache: " + upgraded.err);
      const auto current = Snap(install, updated);
      for (const auto& member : env.members)
        Check(current.at(member.destination) == previous.at(member.destination),
              "refresh preserves member hash, identity and timestamp");
      for (const auto& notice : updated.notices)
        Check(
            Read(install / "licenses" / fs::u8path(notice.path)) == notice.text,
            "build B exact runtime notice bytes");
      Check(RuntimeReadyErrorForTesting(install, updated).empty(),
            "build B runtime marker is ready");
      fs::remove(install / ".bubo/runtime-ready.json");
      Check(Run(refresh, updated).code == 0 &&
                RuntimeReadyErrorForTesting(install, updated).empty() &&
                stub.records().size() == requests,
            "missing runtime marker refreshes offline without wheel cache");
      // Restore A for the remaining transaction tests.
      Check(Run(refresh, env).code == 0, "refresh back to build A");
#ifdef _WIN32
      std::wstring executable(32768, L'\0');
      const auto length = GetModuleFileNameW(
          nullptr, executable.data(), static_cast<DWORD>(executable.size()));
      executable.resize(length);
      auto command = L"\"" + executable + L"\" --refresh-crash \"" +
                     (root / "success").wstring() + L"\"";
      STARTUPINFOW startup{};
      startup.cb = sizeof(startup);
      PROCESS_INFORMATION process{};
      Check(
          CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                         FALSE, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                         nullptr, nullptr, &startup, &process) != 0,
          "launch native notice-refresh crash helper");
      const auto wait = WaitForSingleObject(process.hProcess, 30000);
      DWORD exit = 0;
      GetExitCodeProcess(process.hProcess, &exit);
      CloseHandle(process.hThread);
      CloseHandle(process.hProcess);
      Check(wait == WAIT_OBJECT_0 && exit == 42 &&
                fs::exists(install / ".bubo/runtime/journal.json"),
            "process death during metadata refresh retains recovery journal");
      Check(
          Run(refresh, updated).code == 0 &&
              RuntimeReadyErrorForTesting(install, updated).empty() &&
              stub.records().size() == requests,
          "metadata journal recovers after process death without wheel cache");
      const auto recovered = Snap(install, updated);
      for (const auto& member : env.members)
        Check(
            recovered.at(member.destination) == previous.at(member.destination),
            "crash recovery preserves member bytes, identity and timestamp");
      Check(Run(refresh, env).code == 0,
            "restore A after process death recovery");
#endif
    }
    auto offline = options("offline");
    offline.offline = true;
    Check(Run(offline, env).code == 4 && !fs::exists(root / "offline") &&
              stub.records().size() == requests,
          "offline missing refuses before writes/network");
    auto cancelled = options("decline");
    cancelled.yes = false;
    Check(Run(cancelled, env, "n\n").code == 3 &&
              !fs::exists(root / "decline") &&
              stub.records().size() == requests,
          "decline before writes/network");
    auto bad = options("archive-hash");
    corrupt = true;
    const auto hash_result = Run(bad, env);
    corrupt = false;
    Check(hash_result.code == 6 &&
              !fs::exists(fs::path(bad.runtime_dir) / "amdhip64_7.dll"),
          "archive hash mismatch before extraction");
    auto wrong_member = env;
    wrong_member.members[0].sha256 = std::string(64, '0');
    bad = options("member-hash");
    Check(
        Run(bad, wrong_member).code == 6 &&
            !fs::exists(fs::path(bad.runtime_dir) / ".bubo/runtime-ready.json"),
        "wrong member SHA never publishes");
    const auto original = Read(wheel_paths[0]);
    WriteFixture(
        wheel_paths[0],
        FixtureZip({{"_rocm_sdk_core/bin/amdhip64_7.dll"}, {"../escape"}}));
    auto traversal = env;
    const auto traversal_hash = HashFile(wheel_paths[0]);
    traversal.wheels[0].sha256 = traversal_hash;
    traversal.wheels[0].size = fs::file_size(wheel_paths[0]);
    bad = options("traversal");
    Check(
        Run(bad, traversal).code == 9 && !fs::exists(root / "escape") &&
            !fs::exists(fs::path(bad.runtime_dir) / ".bubo/runtime-ready.json"),
        "traversal archive refusal");
    WriteFixture(wheel_paths[0], original);
    // A missing member forces installation; marker-only changes now refresh.
    const auto install = fs::path(first.runtime_dir);
    WriteFixture(install / ".bubo/runtime-ready.json",
                 "previous-runtime-marker\n");
    fs::remove(install / fs::u8path(env.members.back().destination));
    const auto previous = Snap(install, env);
    for (const auto index : {std::size_t(0), env.members.size(),
                             env.members.size() + env.notices.size()}) {
      auto failing = env;
      failing.after_install = [index](std::size_t n) { return n != index; };
      Check(Run(first, failing).code == 7, "transaction injected failure");
      Check(Snap(install, env) == previous &&
                !fs::exists(install / ".bubo/runtime/journal.json"),
            "rollback preserves prior bytes, identities, mtimes and marker");
    }
    std::cout
        << "PASS: archive SHA/member SHA/traversal failures; rollback after "
           "DLL, notice and readiness moves preserves prior runtime\n";
#ifdef _WIN32
    std::thread releaser;
    std::chrono::steady_clock::time_point held_at{}, promoted_at{};
    auto sharing = env;
    sharing.after_install = [&](std::size_t index) {
      if (index == 0) {
        const auto dll = install / "rocblas.dll";
        const auto held = CreateFileW(
            dll.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        Check(held != INVALID_HANDLE_VALUE, "sharing retry fixture handle");
        held_at = std::chrono::steady_clock::now();
        releaser = std::thread([held] {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          CloseHandle(held);
        });
      }
      if (index == 2)
        promoted_at = std::chrono::steady_clock::now();
      return true;
    };
    const auto retry = Run(first, sharing);
    if (releaser.joinable())
      releaser.join();
    Check(retry.code == 0 &&
              promoted_at - held_at >= std::chrono::milliseconds(750),
          "native sharing violation retry");
    const auto holder_root = root / "native-holder";
    fs::create_directories(holder_root);
    fs::copy_file(fs::path(BUBO_INERT_DLL), holder_root / "amdhip64_7.dll");
    const auto module = LoadLibraryExW(
        (holder_root / "amdhip64_7.dll").c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    Check(module != nullptr, "CPU-only inert module fixture load");
    const auto holders = RuntimeHolders(holder_root, env.members);
    FreeLibrary(module);
    Check(!holders.empty(), "native Restart Manager detects loaded inert DLL");
    auto blocked = env;
    blocked.holders = [holders] { return holders; };
    const auto count = stub.records().size();
    const auto held_result = Run(options("held-runtime"), blocked);
    Check(
        held_result.code == 7 &&
            held_result.err.find("PID") != held_result.err.npos &&
            held_result.err.find("Stop the server.") != held_result.err.npos &&
            stub.records().size() == count,
        "running DLL holder named before download");
    std::cout << "PASS: native sharing retry; Restart Manager loaded CPU "
                 "fixture DLL PID detection; running-runtime refusal\n";
    WriteFixture(install / ".bubo/runtime-ready.json", "before-crash-marker\n");
    fs::remove(install / fs::u8path(env.members.back().destination));
    const auto before_crash = Snap(install, env);
    std::wstring executable(32768, L'\0');
    const auto exe_len = GetModuleFileNameW(
        nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    executable.resize(exe_len);
    auto command = L"\"" + executable + L"\" --crash \"" +
                   (root / "success").wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    Check(CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                         FALSE, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                         nullptr, nullptr, &startup, &process) != 0,
          "native crash helper launch");
    const auto wait = WaitForSingleObject(process.hProcess, 30000);
    DWORD exit = 0;
    GetExitCodeProcess(process.hProcess, &exit);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    Check(wait == WAIT_OBJECT_0 && exit == 42 &&
              fs::exists(install / ".bubo/runtime/journal.json"),
          "process death leaves recovery journal");
    WriteFixture(install / fs::u8path(env.members.front().destination),
                 "interrupted partial runtime member");
    {
      // Machine mode (the GUI) always previews first. A leftover journal is a
      // not-ready plan that the approved run recovers, not a preview failure.
      auto machine_preview = first;
      machine_preview.machine = true;
      PreparationPlan plan;
      std::istringstream input;
      std::ostringstream output, errors;
      const auto code = RunRuntimeForTesting(machine_preview, env, input,
                                             output, errors, &plan);
      Check(code == 0 && !plan.ready &&
                fs::exists(install / ".bubo/runtime/journal.json"),
            "unfinished_transaction_preview_is_not_ready: exit=" +
                std::to_string(code) + " " + errors.str());
    }
    const auto cache_core =
        fs::path(first.cache_dir) / fs::u8path(env.wheels[0].file);
    fs::remove(cache_core);
    first.offline = true;
    const auto crash_recovery = Run(first, env);
    first.offline = false;
    Check(crash_recovery.code == 4 && Snap(install, env) == before_crash &&
              !fs::exists(install / ".bubo/runtime/journal.json"),
          "crash rollback precedes offline cache prerequisite");
    fs::copy_file(wheel_paths[0], cache_core);
    {
      // A process killed before it wrote its journal leaves only the tool's
      // own staging folder. The next run removes it under the lock.
      WriteFixture(install / ".bubo/runtime/stage/amdhip64_7.dll", "partial");
      const auto leftover = Run(first, env);
      Check(leftover.code == 0 &&
                !fs::exists(install / ".bubo/runtime/stage") &&
                RuntimeReadyErrorForTesting(install, env).empty(),
            "scratch_without_journal_is_removed: exit=" +
                std::to_string(leftover.code) + " " + leftover.err);
    }
    {
      // A junction above the runtime and cache folders (a moved or redirected
      // profile) is the user's layout. Only the runtime and cache folders and
      // what is below them must be plain folders.
      const auto target = root / "junction-target";
      const auto profile = root / "junction-profile";
      fs::create_directories(target);
      const auto command = L"cmd /d /c mklink /J \"" + profile.wstring() +
                           L"\" \"" + target.wstring() + L"\" >nul";
      Check(_wsystem(command.c_str()) == 0 &&
                (GetFileAttributesW(profile.c_str()) &
                 FILE_ATTRIBUTE_REPARSE_POINT),
            "junction fixture");
      auto moved = options("junction-unused");
      moved.root = (profile / "data").string();
      moved.runtime_dir = (profile / "runtime").string();
      moved.cache_dir = (profile / "cache").string();
      const auto moved_result = Run(moved, env);
      Check(moved_result.code == 0 &&
                RuntimeReadyErrorForTesting(target / "runtime", env).empty(),
            "junctioned_profile_installs_runtime: exit=" +
                std::to_string(moved_result.code) + " " + moved_result.err);
      auto linked = moved;
      linked.runtime_dir = profile.string();
      linked.cache_dir = (root / "junction-cache").string();
      const auto linked_result = Run(linked, env);
      Check(linked_result.code == 7 &&
                linked_result.err.find("reparse point refused") !=
                    linked_result.err.npos,
            "a junction as the runtime folder itself is refused");
    }
    std::cout
        << "PASS: native process death after first promotion; journal restores "
           "previous runtime even with missing offline wheel cache; a "
           "leftover journal previews as not ready; scratch without a journal "
           "is removed; a junction above the runtime folder is accepted\n";
#endif
    auto cached_offline = options("cached-offline");
    cached_offline.offline = true;
    fs::create_directories(fs::path(cached_offline.cache_dir));
    for (unsigned i = 0; i < 3; ++i)
      fs::copy_file(wheel_paths[i], fs::path(cached_offline.cache_dir) /
                                        fs::u8path(env.wheels[i].file));
    const auto offline_count = stub.records().size();
    Check(Run(cached_offline, env).code == 0 &&
              stub.records().size() == offline_count,
          "complete offline cache installs without transport");
    auto unicode = first;
    unicode.offline = true;
    auto deep = root / fs::path(std::u8string(u8"runtime-データ"));
    for (unsigned i = 0; i < 6; ++i)
      deep /= std::string(45, 'd') + std::to_string(i);
    const auto deep_utf8 = deep.u8string();
    unicode.runtime_dir = std::string(deep_utf8.begin(), deep_utf8.end());
    const auto unicode_count = stub.records().size();
    Check(Run(unicode, env).code == 0 &&
              fs::exists(NativePath(deep / "amdhip64_7.dll")) &&
              stub.records().size() == unicode_count,
          "native Unicode extended-path offline runtime installation");
    std::cout << "PASS: native Unicode runtime destination beyond MAX_PATH; "
                 "offline cache; zero requests\n";
    {
      // --output selects model/archive storage. Runtime selection is the
      // same process search that the server uses, independent of that root.
      auto shared = options("shared-default");
      shared.runtime_dir.clear();
      auto shared_env = env;
      const auto local = root / "shared-process" / "Local";
      const auto local_text = local.string();
      const auto output = shared.root;
      const std::array<const char*, 4> args{"runtime", "--yes", "--output",
                                            output.c_str()};
      const auto parsed = Parse(args, local_text);
      Check(parsed && parsed.options.root == output &&
                parsed.options.runtime_dir.empty(),
            "prepare_output_root_keeps_runtime_selection_separate");
      shared.root = parsed.options.root;
      shared_env.runtime_search.local_app_data = local_text;
      shared_env.runtime_search.exe_dir = root / "shared-process" / "package";
      shared_env.runtime_search.check = [&](const fs::path& dir) {
        return fs::exists(dir / "amdhip64_7.dll")
                   ? RuntimeReadyErrorForTesting(dir, env)
                   : std::string("missing fixture runtime");
      };
      const auto shared_dir = SharedRuntimeDir(local / "Bubo");
      const auto shared_result = Run(shared, shared_env);
      Check(shared_result.code == 0,
            "default runtime install: " + shared_result.err);
      Check(fs::exists(shared_dir / "amdhip64_7.dll") &&
                fs::exists(shared_dir / ".bubo" / "runtime-ready.json") &&
                !fs::exists(shared_dir / "bubo.exe") &&
                !fs::exists(shared_dir / "bubo-engine.dll") &&
                !fs::exists(fs::path(output) / "runtime"),
            "prepare_output_root_does_not_redirect_runtime");
      Check(shared_result.out.find(RuntimeConsent(
                shared_dir, shared_dir / "licenses")) != shared_result.out.npos,
            "plan names the resolved runtime folder");
      Check(RuntimeReadyErrorForTesting(shared_dir, env).empty(),
            "installed shared runtime has a valid marker");
      const auto server_runtime = ResolveRuntime(shared_env.runtime_search);
      Check(server_runtime.dir == shared_dir && server_runtime.error.empty(),
            "prepare_output_runtime_matches_server_resolution");
      std::cout
          << "PASS: prepare_output_root_keeps_runtime_selection_separate; "
             "prepare_output_root_does_not_redirect_runtime; "
             "prepare_output_runtime_matches_server_resolution "
             "(3 cases)\n";

      // Read-only previews prove precedence without extra extraction/writes.
      const auto request_count = stub.records().size();
      const auto before_preview = Times(root);
      auto preview_options = shared;
      preview_options.machine = true;
      preview_options.cache_dir = cached_offline.cache_dir;
      // The crash-recovery proof deliberately leaves first.runtime_dir with
      // an older marker. Use the complete offline install for reuse previews.
      shared_env.runtime_search.environment = cached_offline.runtime_dir;
      const auto preview_runtime = [&](const Options& requested,
                                       const RuntimeTestEnvironment& test_env,
                                       const fs::path& expected,
                                       const std::string& name) {
        std::istringstream input;
        std::ostringstream out, err;
        PreparationPlan plan;
        const auto code =
            RunRuntimeForTesting(requested, test_env, input, out, err, &plan);
        Check(code == 0 && plan.ready &&
                  fs::u8path(plan.machine["paths"].member_str("runtime")) ==
                      expected,
              name + ": exit=" + std::to_string(code) +
                  " plan=" + plan.machine.dump() + " " + err.str());
      };
      preview_runtime(preview_options, shared_env,
                      fs::path(cached_offline.runtime_dir),
                      "prepare_runtime_environment_precedes_shared");
      const auto shared_utf8 = shared_dir.u8string();
      preview_options.runtime_dir =
          std::string(shared_utf8.begin(), shared_utf8.end());
      preview_runtime(preview_options, shared_env, shared_dir,
                      "prepare_runtime_option_precedes_environment");
      preview_options.runtime_dir.clear();
      shared_env.runtime_search.environment.clear();
      shared_env.runtime_search.local_app_data =
          (root / "legacy-process" / "Local").string();
      shared_env.runtime_search.exe_dir = fs::path(cached_offline.runtime_dir);
      const auto legacy_runtime = ResolveRuntime(shared_env.runtime_search);
      Check(legacy_runtime.dir == fs::path(cached_offline.runtime_dir) &&
                legacy_runtime.kind == RuntimeKind::Legacy &&
                legacy_runtime.error.empty(),
            "server_resolves_existing_legacy_runtime");
      preview_runtime(preview_options, shared_env,
                      fs::path(cached_offline.runtime_dir),
                      "prepare_runtime_legacy_matches_server");
      shared_env.runtime_search.environment = "relative-runtime-dir";
      const auto invalid = Run(preview_options, shared_env);
      Check(invalid.code == 2 &&
                invalid.err.find("absolute local path") != invalid.err.npos,
            "prepare_runtime_invalid_environment_refused");
      Check(stub.records().size() == request_count &&
                Times(root) == before_preview,
            "runtime_precedence_previews_have_zero_requests_or_writes");
      std::cout << "PASS: prepare_runtime_environment_precedes_shared; "
                   "prepare_runtime_option_precedes_environment; "
                   "server_resolves_existing_legacy_runtime; "
                   "prepare_runtime_legacy_matches_server; "
                   "prepare_runtime_invalid_environment_refused; "
                   "runtime_precedence_previews_have_zero_requests_or_writes "
                   "(6 cases)\n";
    }
    auto conflict = options("conflict");
    WriteFixture(fs::path(conflict.runtime_dir) / "amdhip64_7.dll", "bad");
    const auto conflict_count = stub.records().size();
    const auto refusal = Run(conflict, env);
    // A different existing file is an integrity failure (6), not storage (7):
    // the contract's code for "a file does not match its pin".
    Check(refusal.code == 6 &&
              refusal.err.find("amdhip64_7.dll") != refusal.err.npos &&
              refusal.err.find("delete it and run bubo prepare again") !=
                  refusal.err.npos &&
              stub.records().size() == conflict_count,
          "canonical conflict retained/refused");
    auto lowspace = env;
    lowspace.volume = [](auto) {
      return VolumeInfo{"fixture-local", 1, "NTFS", false, {}};
    };
    Check(Run(options("lowspace"), lowspace).code == 7, "space preflight");
    auto fat = env;
    fat.volume = [](auto) {
      return VolumeInfo{
          "fixture-local", 20ULL * 1024 * 1024 * 1024, "FAT32", false, {}};
    };
    Check(Run(options("fat"), fat).code == 7, "FAT32 refused");
    auto bad_notice = env;
    bad_notice.notices[0].sha256 = kAbcHash;
    Check(Run(options("badnotice"), bad_notice).code == 4,
          "invalid compiled notice pin before network");
    std::cout << "PASS: offline cached extraction; canonical conflict; free "
                 "space/FAT32; notice integrity preflight\n";
    fs::remove_all(NativePath(root));
    std::cout << "prepare_runtime_test: PASS (" << checks
              << " checks, CPU only)\n";
  } catch (const std::exception& e) {
    std::cerr << "prepare_runtime_test: FAIL " << e.what() << '\n';
    return 1;
  }
}
