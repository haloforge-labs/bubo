// CPU test: where the AMD runtime lives and how bubo.exe finds it.
// Order: option > BUBO_RUNTIME_DIR > shared per-user folder > beside bubo.exe.
#include "src/cli/prepare/runtime_location.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/cli/prepare/runtime_prepare.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace bubo::prepare;
std::size_t checks = 0;
void Check(bool ok, const std::string& text) {
  ++checks;
  if (!ok)
    throw std::runtime_error(text);
}
void Touch(const fs::path& path, const std::string& body = "x") {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << body;
}
std::string Text(const fs::path& path) {
  const auto u = path.u8string();
  return std::string(u.begin(), u.end());
}
void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value.empty() ? nullptr : value.c_str());
#else
  if (value.empty())
    unsetenv(name);
  else
    setenv(name, value.c_str(), 1);
#endif
}
// A fake check: a folder is usable when it holds a file named "ok".
std::string FakeCheck(const fs::path& dir) {
  return fs::exists(dir / "ok") ? std::string()
                                : "fake: not usable " + Text(dir);
}
void Marker(const fs::path& dir) {
  Touch(dir / ".bubo" / "runtime-ready.json", "{}");
}
}  // namespace

int main() {
  try {
    const auto root = fs::absolute("k1-runtime-location-fixture");
    fs::remove_all(root);
    const auto appdata = root / "Local";
    const auto exe = root / "pkg" / "bin";
    fs::create_directories(exe);
    const auto shared = appdata / "Bubo" / "runtime" / "rocm-10.0.0-gfx1151";
    const auto explicit_dir = root / "elsewhere";

    // 1. The default folder name comes from the runtime id.
    Check(SharedRuntimeDirFor(Text(appdata)) == shared,
          "shared dir is LOCALAPPDATA\\Bubo\\runtime\\<runtime id>");
    Check(SharedRuntimeDirFor("").empty(), "no LOCALAPPDATA, no shared dir");
    Check(SharedRuntimeDir(appdata / "Bubo") == shared, "data root form");

    RuntimeSearch search;
    search.local_app_data = Text(appdata);
    search.exe_dir = exe;
    search.check = FakeCheck;

    // 2. Nothing installed: the shared folder, with an error that names it.
    auto none = ResolveRuntime(search);
    Check(none.dir == shared && none.kind == RuntimeKind::Shared &&
              none.source == "default" && !none.error.empty(),
          "empty machine resolves to the shared folder with an error");

    Check(none.absent, "nothing installed anywhere is reported as absent");
    // 3. A runtime beside bubo.exe (v0.1.0 layout) is used when no shared one.
    Touch(exe / "ok");
    Marker(exe);
    auto legacy = ResolveRuntime(search);
    Check(legacy.dir == exe && legacy.kind == RuntimeKind::Legacy &&
              legacy.error.empty() && legacy.source == "legacy",
          "legacy runtime beside bubo.exe keeps working");
    Check(RuntimeKindName(legacy.kind) == "legacy", "legacy name");

    // 4. A shared folder with a valid marker wins over the legacy one.
    Touch(shared / "ok");
    Marker(shared);
    auto picked = ResolveRuntime(search);
    Check(picked.dir == shared && picked.kind == RuntimeKind::Shared &&
              picked.error.empty() && picked.source == "shared",
          "valid shared runtime wins over legacy");

    // 5. A shared folder without a marker is not valid: legacy stays in use.
    fs::remove(shared / ".bubo" / "runtime-ready.json");
    Check(ResolveRuntime(search).dir == exe,
          "shared folder without marker is ignored");
    Marker(shared);

    // 6. A shared folder whose check fails (damaged) falls back to legacy.
    fs::remove(shared / "ok");
    auto damaged = ResolveRuntime(search);
    Check(damaged.dir == exe && damaged.kind == RuntimeKind::Legacy,
          "damaged shared runtime falls back to a usable legacy one");
    fs::remove(exe / "ok");
    auto both_bad = ResolveRuntime(search);
    Check(both_bad.dir == shared && !both_bad.error.empty(),
          "both damaged: report the shared folder");
    Touch(shared / "ok");

    // 7. Environment override beats the shared folder; option beats both.
    search.environment = Text(explicit_dir);
    Touch(explicit_dir / "ok");
    auto env = ResolveRuntime(search);
    Check(env.dir == explicit_dir && env.kind == RuntimeKind::Explicit &&
              env.source == "environment" && env.error.empty(),
          "BUBO_RUNTIME_DIR beats the shared folder");
    const auto option_dir = root / "from-option";
    Touch(option_dir / "ok");
    search.option = Text(option_dir);
    auto option = ResolveRuntime(search);
    Check(option.dir == option_dir && option.source == "option" &&
              option.error.empty(),
          "--runtime-dir beats BUBO_RUNTIME_DIR");
    // An explicit folder that is not usable is reported, never replaced.
    fs::remove(option_dir / "ok");
    auto explicit_bad = ResolveRuntime(search);
    Check(explicit_bad.dir == option_dir && !explicit_bad.error.empty(),
          "an unusable explicit folder is an error, not a silent fallback");
    search.option.clear();
    search.environment.clear();

    // 8. The option must be an absolute local path.
    Check(!CheckRuntimeDirOption("").empty(), "empty option refused");
    Check(!CheckRuntimeDirOption("relative\\dir").empty(),
          "relative option refused");
#ifdef _WIN32
    Check(!CheckRuntimeDirOption("\\\\server\\share\\rt").empty(),
          "UNC option refused");
    Check(CheckRuntimeDirOption("D:\\ROCm").empty(), "absolute path accepted");
    const char* namespace_paths[] = {
        "\\\\server\\share\\rt",
        "\\\\?\\UNC\\server\\share\\rt",
        "\\\\?\\unc\\server\\share\\rt",
        "//?/UNC/server/share/rt",
        "\\\\.\\C:\\rt",
        "\\\\.\\pipe\\runtime",
        "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\rt",
        "\\\\?\\Volume{00000000-0000-0000-0000-000000000000}\\rt",
        "\\??\\C:\\rt",
        "\\\\??\\C:\\rt",
    };
    for (const auto* value : namespace_paths) {
      auto guarded = search;
      bool probed = false;
      guarded.check = [&](const fs::path&) {
        probed = true;
        return std::string();
      };
      guarded.option = value;
      const auto option_result = ResolveRuntime(guarded);
      Check(!option_result.error.empty() && !probed &&
                option_result.source == "option",
            std::string("runtime_namespace_option_rejected_before_probe: ") +
                value);
      guarded.option.clear();
      guarded.environment = value;
      const auto environment_result = ResolveRuntime(guarded);
      Check(
          !environment_result.error.empty() && !probed &&
              environment_result.source == "environment",
          std::string("runtime_namespace_environment_rejected_before_probe: ") +
              value);
    }
    Check(CheckRuntimeDirOption("\\\\?\\D:\\ROCm").empty() &&
              CheckRuntimeDirOption("\\\\?\\d:\\runtime-データ").empty(),
          "runtime_extended_local_drive_paths_remain_supported");
    std::cout << "PASS: runtime_namespace_option_rejected_before_probe; "
                 "runtime_namespace_environment_rejected_before_probe "
                 "(10 paths, 20 cases); "
                 "runtime_extended_local_drive_paths_remain_supported "
                 "(1 case)\n";
#endif
    search.option = "relative\\dir";
    Check(!ResolveRuntime(search).error.empty() &&
              ResolveRuntime(search).kind == RuntimeKind::Explicit,
          "relative --runtime-dir fails resolution");
    search.option.clear();

    // 9. No LOCALAPPDATA: only the legacy folder can be used.
    search.local_app_data.clear();
    Touch(exe / "ok");
    Check(ResolveRuntime(search).dir == exe, "no LOCALAPPDATA uses legacy");
    search.local_app_data = Text(appdata);

    // 10. Migration decision: legacy only -> the server keeps using it and
    // nothing is moved or deleted; after the shared folder is installed the
    // server uses that one and the legacy files stay.
    fs::remove_all(shared);
    Check(ResolveRuntime(search).kind == RuntimeKind::Legacy &&
              fs::exists(exe / "ok"),
          "legacy-only machine keeps its legacy runtime");
    Touch(shared / "ok");
    Marker(shared);
    Check(ResolveRuntime(search).kind == RuntimeKind::Shared &&
              fs::exists(exe / "ok"),
          "after install the shared folder is used; legacy files untouched");

    // 11. Real marker validation (compiled tables, no fake): a junk marker is
    // not a valid runtime; an unfinished transaction is reported.
    RuntimeSearch real;
    real.local_app_data = Text(appdata);
    real.exe_dir = exe;
    fs::remove_all(shared);
    Check(!DefaultRuntimeCheck(shared).empty() &&
              DefaultRuntimeCheck(shared).find("run bubo prepare runtime") !=
                  std::string::npos,
          "missing folder is not a runtime");
    Touch(shared / ".bubo" / "runtime-ready.json", "{\"junk\":1}\n");
    const auto junk = DefaultRuntimeCheck(shared);
    Check(!junk.empty() &&
              junk.find("readiness entry mismatch") != std::string::npos,
          "junk marker refused: " + junk);
    Check(!ResolveRuntime(real).error.empty() && !ResolveRuntime(real).absent,
          "a junk shared marker is a damaged runtime, never absent");
    fs::remove_all(shared);
    Touch(shared / ".bubo" / "runtime" / "journal.json", "{}");
    Check(DefaultRuntimeCheck(shared).find("unfinished runtime transaction") !=
              std::string::npos,
          "journal reported");
    fs::remove_all(shared);
    Touch(shared / "amdhip64_7.dll");
    Check(DefaultRuntimeCheck(shared).empty(),
          "manual install without marker is accepted (as in v0.1.0)");

    // 12. ClassifyRuntime and the environment helpers.
    SetEnv("BUBO_RUNTIME_DIR", "");
    Check(RuntimeDirFromEnvironment().empty(), "env unset");
    Check(ClassifyRuntime(shared, Text(appdata)) == "shared",
          "classify shared");
    Check(ClassifyRuntime(exe, Text(appdata)) == "legacy", "classify legacy");
    SetEnv("BUBO_RUNTIME_DIR", Text(explicit_dir));
    Check(RuntimeDirFromEnvironment() == Text(explicit_dir), "env read");
    Check(ClassifyRuntime(explicit_dir, Text(appdata)) == "explicit",
          "classify explicit");
    Check(ResolveProcessRuntime(Text(appdata), exe).dir == explicit_dir,
          "process resolver reads BUBO_RUNTIME_DIR");
    const auto process = ProcessRuntimeSearch(Text(appdata), exe);
    Check(process.local_app_data == Text(appdata) && process.exe_dir == exe &&
              process.environment == Text(explicit_dir) &&
              process.option.empty(),
          "process_runtime_search_captures_common_inputs");
    const auto prepared = ResolveRuntime(process);
    const auto served = ResolveProcessRuntime(Text(appdata), exe);
    Check(prepared.dir == served.dir && prepared.kind == served.kind &&
              prepared.source == served.source &&
              prepared.error == served.error &&
              prepared.absent == served.absent,
          "prepare_and_server_share_process_runtime_resolution");
    std::cout << "PASS: process_runtime_search_captures_common_inputs; "
                 "prepare_and_server_share_process_runtime_resolution "
                 "(2 cases)\n";
    SetEnv("BUBO_RUNTIME_DIR", "");

    fs::remove_all(root);
    std::cout << "runtime_location_test: PASS (" << checks
              << " checks, CPU only)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "runtime_location_test: FAIL " << e.what() << '\n';
    return 1;
  }
}
