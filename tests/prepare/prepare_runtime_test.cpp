#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cli/prepare/runtime_prepare.hpp"
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
void Check(bool ok, std::string text) {
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
  for (const auto& f : fs::recursive_directory_iterator(root))
    times[f.path().generic_string()] = fs::last_write_time(f.path());
  times[root.generic_string()] = fs::last_write_time(root);
  return times;
}
}  // namespace
int main(int argc, char** argv) {
  try {
#ifdef _WIN32
    if (argc == 3 && std::string_view(argv[1]) == "--crash") {
      const auto run_root = fs::u8path(argv[2]);
      RuntimeTestEnvironment crash;
      crash.wheels = RuntimeWheels();
      crash.notices = RuntimeNotices();
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
            "exact release-thread notice bytes: " + std::string(n.path));
      Check(HashFile(fs::path(BUBO_NOTICE_SOURCE) / fs::u8path(n.path)) ==
                n.sha256,
            "notice source digest");
    }
    std::cout << "PASS: printed consent equals " << consent_path.string()
              << " after code-filled placeholders; compiled pins; 70 exact "
                 "release-thread notice bytes\n";
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
    drop = true;
    const auto result = Run(first, env);
    drop = false;
    Check(result.code == 0, "runtime acquisition: " + result.err);
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
              noop.out == "prepare runtime: already prepared; nothing to do\n",
          "read-only no-op");
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
    // Restore an older marker to force a repair with a complete prior payload.
    const auto install = fs::path(first.runtime_dir);
    WriteFixture(install / ".bubo/runtime-ready.json",
                 "previous-runtime-marker\n");
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
    std::cout
        << "PASS: native process death after first promotion; journal restores "
           "previous runtime even with missing offline wheel cache\n";
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
    auto conflict = options("conflict");
    WriteFixture(fs::path(conflict.runtime_dir) / "amdhip64_7.dll", "bad");
    const auto conflict_count = stub.records().size();
    const auto refusal = Run(conflict, env);
    Check(refusal.code == 7 &&
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
    std::cout << "prepare_runtime_test: PASS (CPU only)\n";
  } catch (const std::exception& e) {
    std::cerr << "prepare_runtime_test: FAIL " << e.what() << '\n';
    return 1;
  }
}
