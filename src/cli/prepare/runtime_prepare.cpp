#include "src/cli/prepare/runtime_prepare.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/zip_reader.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
// The Windows SDK Restart Manager header requires windows.h first.
#ifdef _WIN32
#include <restartmanager.h>
#endif

namespace bubo::prepare {
namespace {
std::string RuntimeDisclosure(std::span<const WheelPin> wheels,
                              const std::filesystem::path& runtime,
                              const std::filesystem::path& licenses);
}
namespace {
namespace fs = std::filesystem;
fs::path Path(std::string_view text) {
  auto path = fs::path(std::u8string(
      reinterpret_cast<const char8_t*>(text.data()), text.size()));
  return path.make_preferred();
}
using metadata::Value;
struct Failure : std::runtime_error {
  int code;
  Failure(int c, std::string text)
      : std::runtime_error(std::move(text)), code(c) {}
};
struct Environment {
  RuntimeSearch runtime_search;
  std::array<WheelPin, 3> wheels;
  std::vector<RuntimeMember> members;
  std::vector<RuntimeNotice> notices;
  Transport* transport;
  DownloadOptions download;
  std::function<VolumeInfo(const fs::path&)> volume = QueryVolume;
  std::function<std::vector<std::string>()> holders;
  std::function<bool(std::size_t)> after_install;
  std::function<bool()> cancelled;
#ifdef BUBO_PREPARE_TESTING
  std::array<std::string, 3> origins;
  bool synthetic_disclosure = false;
#endif
};
std::string Utf8(const fs::path& p) {
  const auto u = p.u8string();
  std::string s(u.begin(), u.end());
  if (s.starts_with("\\\\?\\"))
    s.erase(0, 4);
  return s;
}
std::string Digest(std::string_view text) {
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}
std::string Commas(std::uint64_t n) {
  auto s = std::to_string(n);
  for (std::size_t i = s.size(); i > 3; i -= 3)
    s.insert(i - 3, ",");
  return s;
}
// Reparse points are refused from the runtime and cache folders downwards. A
// junction above them (a moved or redirected profile) is the user's layout.
// Empty outside a run: every ancestor is checked.
thread_local std::vector<fs::path> gReparseRoots;
class ReparseRoots {
public:
  explicit ReparseRoots(std::vector<fs::path> roots)
      : previous_(std::exchange(gReparseRoots, std::move(roots))) {}
  ~ReparseRoots() { gReparseRoots = std::move(previous_); }
  ReparseRoots(const ReparseRoots&) = delete;
  ReparseRoots& operator=(const ReparseRoots&) = delete;

private:
  std::vector<fs::path> previous_;
};
void Local(const fs::path& path) {
  const auto error = CheckLocalPath(path);
  if (!error.empty())
    throw Failure(7, error);
  for (auto p = path; !p.empty();) {
    const bool boundary = std::find(gReparseRoots.begin(), gReparseRoots.end(),
                                    p) != gReparseRoots.end();
#ifdef _WIN32
    const auto attr = GetFileAttributesW(p.c_str());
    if (attr != INVALID_FILE_ATTRIBUTES &&
        (attr & FILE_ATTRIBUTE_REPARSE_POINT))
      throw Failure(7, "runtime/cache reparse point refused: " + Utf8(p));
#else
    if (fs::is_symlink(fs::symlink_status(p)))
      throw Failure(7, "runtime/cache symlink refused");
#endif
    const auto parent = p.parent_path();
    if (boundary || parent == p || p == p.root_path())
      break;
    p = parent;
  }
}
void Move(const fs::path& from, const fs::path& to) {
  Local(from);
  Local(to);
  fs::create_directories(to.parent_path());
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
#ifdef _WIN32
    if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_WRITE_THROUGH))
      return;
    const auto code = GetLastError();
    if ((code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION) ||
        std::chrono::steady_clock::now() >= until)
      throw Failure(7, "runtime rename " + Utf8(from) + ": Windows error " +
                           std::to_string(code));
#else
    if (fs::exists(to))
      throw Failure(7, "runtime rename target exists");
    fs::rename(from, to);
    return;
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
}
void Save(const fs::path& path, std::string_view text) {
  Local(path);
  Local(fs::path(path.wstring() + L".new"));
  fs::create_directories(path.parent_path());
  std::string error;
  if (!AtomicWrite(path, text, error))
    throw Failure(7, error);
}
void Remove(const fs::path& path) {
  Local(path);
#ifdef _WIN32
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    if (DeleteFileW(path.c_str()))
      return;
    const auto code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND)
      return;
    if ((code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION) ||
        std::chrono::steady_clock::now() >= until)
      throw Failure(7, "runtime rollback/remove " + Utf8(path) +
                           ": Windows error " + std::to_string(code));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
#else
  fs::remove(path);
#endif
}
std::shared_ptr<ReadLease> Seal(const fs::path& path, std::uint64_t size,
                                std::string_view sha) {
  Local(path);
#ifdef _WIN32
  const auto handle = CreateFileW(
      path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, FILE_SHARE_READ,
      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    throw Failure(7, "cannot protect runtime staging file");
  auto lease = std::make_shared<ReadLease>(
      ReadLease::AdoptVerifiedHandle(reinterpret_cast<std::intptr_t>(handle)));
  gufo::crypto::Sha256Hasher hash;
  if (lease->Snapshot().size != size)
    throw Failure(6, "staged runtime length mismatch");
  HashInto(path, hash, size, {}, reinterpret_cast<std::intptr_t>(handle));
  if (hash.FinishHex() != sha)
    throw Failure(6, "staged runtime SHA-256 mismatch");
#else
  auto lease = std::make_shared<ReadLease>(path);
  if (lease->Snapshot().size != size || HashFile(path) != sha)
    throw Failure(6, "staged runtime pin mismatch");
#endif
  return lease;
}
std::string Ready(const Environment& e) {
  auto value = Value::object();
  value["schema_version"] = 1;
  value["runtime_id"] = "rocm-10.0.0-gfx1151";
  value["members_sha256"] = Digest(SerializeRuntimeTable(e.members));
  auto notices = Value::object();
  for (const auto& n : e.notices)
    notices[std::string(n.path)] = std::string(n.sha256);
  value["notices"] = notices;
  return value.dump() + "\n";
}
struct Installed {
  std::string path;
  bool existed = false;
};
std::vector<std::string> ManagedPaths(const Environment& e) {
  std::vector<std::string> names;
  for (const auto& m : e.members)
    names.push_back(m.destination);
  for (const auto& n : e.notices)
    names.push_back("licenses/" + std::string(n.path));
  names.push_back(".bubo/runtime-ready.json");
  return names;
}
// Journal is durable before moves. Recovery uses only compiled managed names;
// corrupted journal metadata never acquires path authority.
void Recover(const fs::path& root, const Environment& e) {
  const auto control = root / Path(".bubo/runtime");
  const auto journal = control / "journal.json";
  Local(journal);
  if (!fs::exists(journal))
    return;
  const auto data = metadata::Parse(metadata::Read(journal));
  metadata::Keys(data, {"phase", "files"});
  const auto& rows = metadata::Required(data, "files");
  const auto names = ManagedPaths(e);
  if (!rows.is_array() ||
      (rows.size() != names.size() && rows.size() != e.notices.size() + 1))
    throw Failure(7, "invalid runtime rollback journal; backups retained");
  const auto offset = rows.size() == names.size() ? 0 : e.members.size();
  std::vector<Installed> files;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows.items()[i];
    metadata::Keys(row, {"path", "existed"});
    const auto name = metadata::String(metadata::Required(row, "path"));
    const auto existed = metadata::Integer(metadata::Required(row, "existed"));
    if (name != names[offset + i] || existed > 1)
      throw Failure(7, "runtime journal differs from managed paths");
    files.push_back({name, existed != 0});
  }
  const auto phase = metadata::String(metadata::Required(data, "phase"));
  if (phase != "installing" && phase != "committed")
    throw Failure(7, "runtime journal phase");
  if (phase == "installing") {
    for (auto it = files.rbegin(); it != files.rend(); ++it) {
      const auto final = root / Path(it->path),
                 backup = control / "backup" / Path(it->path),
                 stage = control / "stage" / Path(it->path);
      Local(final);
      Local(backup);
      Local(stage);
      if (fs::exists(backup)) {
        Remove(final);
        Move(backup, final);
      } else if (!it->existed && !fs::exists(stage))
        Remove(final);
      else if (it->existed && !fs::exists(stage))
        throw Failure(7, "runtime rollback source missing; backups retained");
    }
  }
  // Narrow, verified owned scratch roots, never user payload roots.
  for (const auto* child : {"stage", "backup"}) {
    const auto dir = control / child;
    Local(dir);
    if (fs::exists(dir)) {
      for (const auto& entry : fs::recursive_directory_iterator(dir))
        Local(entry.path());
      fs::remove_all(dir);
    }
  }
  Remove(journal);
}
bool CheckInstalled(const fs::path& root, const Environment& e,
                    bool* payload_ready = nullptr) {
  bool complete = true;
  for (const auto& m : e.members) {
    const auto p = root / Path(m.destination);
    Local(p);
    if (!fs::exists(p)) {
      complete = false;
      continue;
    }
    ReadLease lease(p);
    if (lease.Snapshot().size != m.size || HashFile(p) != m.sha256)
      throw Failure(6, "different existing runtime file '" + Utf8(p) +
                           "'; delete it and run bubo prepare again");
  }
  if (payload_ready)
    *payload_ready = complete;
  for (const auto& n : e.notices) {
    const auto p = root / "licenses" / Path(n.path);
    Local(p);
    if (!fs::exists(p)) {
      complete = false;
      continue;
    }
    ReadLease lease(p);
    if (lease.Snapshot().size != n.text.size() || HashFile(p) != n.sha256)
      complete = false;
  }
  const auto marker = root / Path(".bubo/runtime-ready.json");
  Local(marker);
  if (!fs::exists(marker) || metadata::Read(marker) != Ready(e))
    complete = false;
  if (fs::exists(root / Path(".bubo/runtime/journal.json")))
    complete = false;
  return complete;
}
// Refresh only Bubo's metadata. The same durable journal and rollback used
// for installation protect each rename; no member is staged or replaced.
void RefreshNotices(const fs::path& root, const Environment& e) {
  const auto control = root / Path(".bubo/runtime");
  const auto all_names = ManagedPaths(e);
  const std::vector<std::string> names(all_names.begin() + e.members.size(),
                                       all_names.end());
  Local(control / "stage");
  Local(control / "backup");
  if (fs::exists(control / "backup"))
    throw Failure(7, "unowned runtime backup without journal");
  if (fs::exists(control / "stage")) {
    for (const auto& entry :
         fs::recursive_directory_iterator(control / "stage"))
      Local(entry.path());
    fs::remove_all(control / "stage");
  }
  std::vector<std::shared_ptr<ReadLease>> staged;
  try {
    for (const auto& n : e.notices) {
      const auto path = control / Path("stage/licenses") / Path(n.path);
      Save(path, n.text);
      staged.push_back(Seal(path, n.text.size(), n.sha256));
    }
    const auto ready = Ready(e);
    const auto marker = control / Path("stage/.bubo/runtime-ready.json");
    Save(marker, ready);
    staged.push_back(Seal(marker, ready.size(), Digest(ready)));
    // Keep matching members stable until the refreshed marker is published.
    std::vector<ReadLease> payload_leases;
    for (const auto& member : e.members)
      payload_leases.emplace_back(root / Path(member.destination));
    bool payload = false;
    CheckInstalled(root, e, &payload);
    if (!payload)
      throw Failure(
          7, "runtime changed during notice refresh; run bubo prepare again");
    auto journal = Value::object();
    journal["phase"] = "installing";
    journal["files"] = Value::array();
    for (const auto& name : names) {
      auto row = Value::object();
      row["path"] = name;
      Local(root / Path(name));
      row["existed"] = fs::exists(root / Path(name)) ? 1 : 0;
      journal["files"].push_back(row);
    }
    Save(control / "journal.json", journal.dump());
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (e.cancelled && e.cancelled())
        throw Failure(3, "cancelled");
      const auto final = root / Path(names[i]);
      fs::create_directories(final.parent_path());
      if (fs::exists(final))
        Move(final, control / "backup" / Path(names[i]));
      std::string error;
      if (!staged[i]->PromoteTo(control / "stage" / Path(names[i]), final,
                                error, false))
        throw Failure(7, error);
      if (e.after_install && !e.after_install(i))
        throw Failure(7, "injected runtime transaction failure");
    }
    journal["phase"] = "committed";
    Save(control / "journal.json", journal.dump());
    staged.clear();
    Recover(root, e);
  } catch (...) {
    staged.clear();
    Recover(root, e);
    throw;
  }
}
void Holders(const Environment& e) {
  const auto processes = e.holders();
  if (!processes.empty()) {
    std::string error = "runtime DLLs held by ";
    for (const auto& p : processes)
      error += p + "; ";
    throw Failure(7,
                  error + "Stop the server. Run bubo prepare runtime again.");
  }
}
Value MachinePlan(
    const Environment& e, const fs::path& root, const fs::path& cache,
    bool ready, bool payload_ready = false,
    const std::map<std::string, std::pair<std::uint64_t, std::uint64_t>>&
        budgets = {}) {
  auto value = Value::object();
  value["component"] = "runtime";
  value["ready"] = ready;
  value["sources"] = Value::array();
  if (ready || payload_ready)
    value["file_states"] = Value::array();
  value["licenses"] = Value::array();
  value["paths"] = Value::object();
  value["paths"]["runtime"] = Utf8(root);
  value["paths"]["cache"] = Utf8(cache);
  value["volumes"] = Value::array();
  for (const auto& wheel : e.wheels) {
    value["sources"].push_back(PlanSource(
        wheel.name, wheel.url, wheel.size, wheel.sha256, "AMD/TheRock",
        "10.0.0", Utf8(cache / Path(wheel.file)), "AMD component terms"));
    if (ready || payload_ready)
      value["file_states"].push_back("reuse");
  }
  for (const auto& notice : e.notices) {
    auto row = Value::object();
    row["path"] = std::string(notice.path);
    row["text"] = std::string(notice.text);
    row["sha256"] = std::string(notice.sha256);
    value["licenses"].push_back(row);
  }
  for (const auto& [name, budget] : budgets)
    value["volumes"].push_back(PlanVolume(
        {name, budget.first, budget.second + 2ULL * 1024 * 1024 * 1024}));
  return value;
}
int RunImpl(const Options& options, Environment e, std::istream& input,
            std::ostream& out, std::ostream& err, PreparationPlan* preview,
            bool approved) {
  try {
    if (e.members.empty() || e.notices.empty())
      throw Failure(4, "runtime member/notice pins are OPEN");
    for (const auto& n : e.notices)
      if (!SafeZipPath(n.path) || Digest(n.text) != n.sha256)
        throw Failure(4, "runtime notice pin mismatch; no downloads started");
    // Preparation and engine startup choose the same runtime. --output is
    // the model/archive root; it cannot redirect the runtime away from the
    // folder the server will load. Existing readiness errors are repaired by
    // preparation, while an invalid destination still fails before writes.
    auto search = e.runtime_search;
    search.option = options.runtime_dir;
    const auto runtime = ResolveRuntime(search);
    if (const auto bad = CheckRuntimeDirOption(Utf8(runtime.dir)); !bad.empty())
      throw Failure(2, "--runtime-dir: " + bad);
    const auto root = NativePath(runtime.dir);
    if (preview)
      preview->runtime_dir = Utf8(root);
    // --runtime-dir puts the runtime where later commands do not look. They
    // find it through BUBO_RUNTIME_DIR; say so, with the folder.
    const auto where = [&] {
      if (!options.runtime_dir.empty())
        out << "prepare runtime: set BUBO_RUNTIME_DIR=" << Utf8(root)
            << " so other bubo commands find this runtime\n";
    };
    const auto cache =
        NativePath(options.cache_dir.empty()
                       ? Path(options.root) / "cache/rocm-10.0.0-gfx1151"
                       : Path(options.cache_dir));
    const ReparseRoots reparse_roots({root, cache});
    Local(root);
    Local(cache);
    auto rp = Utf8(root), cp = Utf8(cache);
    std::transform(rp.begin(), rp.end(), rp.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::transform(cp.begin(), cp.end(), cp.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (rp == cp || cp.starts_with(rp + "\\") || rp.starts_with(cp + "\\") ||
        cp.starts_with(rp + "/") || rp.starts_with(cp + "/"))
      throw Failure(7, "runtime and cache roots must be separate");
    if (!e.holders)
      e.holders = [&] { return RuntimeHolders(root, e.members); };
    // A previous accepted transaction authorizes its recovery. Restore the
    // previous installation even if its wheel cache is now missing/offline.
    // A preview writes nothing: it reports the runtime as not ready, and the
    // approved run recovers under the lock before it installs.
    if (!preview && fs::exists(root / Path(".bubo/runtime/journal.json"))) {
      Holders(e);
      PrepareLock recovery_lock;
      std::string error;
      const auto lock_path = root / Path(".bubo/prepare.lock");
      Local(lock_path);
      if (!recovery_lock.Acquire(lock_path, error))
        throw Failure(recovery_lock.busy() ? 8 : 7, error);
      Recover(root, e);
    }
    bool payload_ready = false;
    const bool journal_exists =
        fs::exists(root / Path(".bubo/runtime/journal.json"));
    const bool installed =
        !journal_exists && CheckInstalled(root, e, &payload_ready);
    if (installed) {
      if (preview) {
        preview->ready = true;
        if (options.machine)
          preview->machine = MachinePlan(e, root, cache, true);
        std::ostringstream disclosure;
        disclosure << "Runtime status: reuse; already prepared.\n";
        for (const auto& notice : e.notices)
          disclosure << "License/notice: " << notice.path << '\n'
                     << notice.text << '\n';
#ifdef BUBO_PREPARE_TESTING
        if (e.synthetic_disclosure)
          disclosure << RuntimeDisclosure(e.wheels, root, root / "licenses");
        else
#endif
          disclosure << RuntimeConsent(root, root / "licenses");
        preview->text = disclosure.str();
        return 0;
      }
      out << "prepare runtime: already prepared; nothing to do\n";
      where();
      return 0;
    }
    if (payload_ready) {
      if (preview) {
        preview->ready = false;
        preview->machine = MachinePlan(e, root, cache, false, true);
        preview->text =
            "Runtime members verified; refresh notices and readiness.\n";
        return 0;
      }
      if (options.authorize_plan &&
          !options.authorize_plan(MachinePlan(e, root, cache, false, true)))
        throw Failure(3, "plan changed; approval required");
      PrepareLock lock;
      std::string error;
      Local(root / Path(".bubo/prepare.lock"));
      if (!lock.Acquire(root / Path(".bubo/prepare.lock"), error))
        throw Failure(lock.busy() ? 8 : 7, error);
      Recover(root, e);
      RefreshNotices(root, e);
      out << "prepare runtime: already prepared; nothing to download\n"
             "prepare runtime: notices refreshed\n";
      where();
      return 0;
    }
    Holders(e);
    std::array<Entry, 3> entries;
    std::array<bool, 3> cached{};
    std::vector<std::shared_ptr<ReadLease>> archives;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> budgets;
    auto budget = [&](const fs::path& p, std::uint64_t bytes) {
      const auto v = e.volume(p);
      if (!v.error.empty() || v.network || v.filesystem == "FAT32")
        throw Failure(7, "runtime volume refused: " + v.error);
      auto& b = budgets[v.identity];
      b.first = v.free_bytes;
      b.second += bytes;
    };
    std::uint64_t extracted = 0;
    for (const auto& m : e.members)
      extracted += m.size;
    for (const auto& n : e.notices)
      extracted += n.text.size();
    budget(root, extracted + 2 * 1024 * 1024);
    budget(cache, 2 * 1024 * 1024);
    for (unsigned i = 0; i < 3; ++i) {
      const auto& w = e.wheels[i];
      entries[i] = {w.name,
                    w.file,
                    w.size,
                    w.sha256,
                    "AMD/TheRock",
                    "10.0.0",
                    w.file,
                    Kind::MainShard,
                    "AMD component terms",
                    w.file,
                    {},
                    w.url};
      const auto p = cache / Path(w.file);
      Local(p);
      Local(fs::path(p.wstring() + L".partial"));
      Local(fs::path(p.wstring() + L".partial.json"));
      if (fs::exists(p)) {
        auto lease = std::make_shared<ReadLease>(p);
        if (lease->Snapshot().size != w.size || HashFile(p) != w.sha256)
          throw Failure(6, "cached wheel SHA-256/size mismatch: " + Utf8(p) +
                               "; delete it and run bubo prepare again");
        archives.push_back(lease);
        cached[i] = true;
      } else {
        if (options.offline)
          throw Failure(4, "offline missing pinned wheel: " + Utf8(p));
        auto partial = p;
        partial += ".partial";
        budget(cache, w.size - RetainedPrefix(entries[i], partial));
      }
    }
    std::ostringstream disclosure;
    for (const auto& [volume, b] : budgets) {
      const auto required = b.second + 2ULL * 1024 * 1024 * 1024;
      disclosure << "Runtime volume " << volume << ": free " << b.first
                 << " bytes; required " << required << " bytes\n";
      if (b.first < required)
        throw Failure(7, NotEnoughSpaceText({volume, b.first, required},
                                            "--runtime-dir"));
      if (preview)
        preview->volumes.push_back({volume, b.first, required});
    }
    // Exact notice texts are available before the affirmative prompt.
    for (const auto& n : e.notices)
      disclosure << "License/notice: " << n.path << '\n' << n.text << '\n';
#ifdef BUBO_PREPARE_TESTING
    if (e.synthetic_disclosure)
      disclosure << RuntimeDisclosure(e.wheels, root, root / "licenses");
    else
#endif
      disclosure << RuntimeConsent(root, root / "licenses");
    if (preview) {
      preview->ready = false;
      if (options.machine)
        preview->machine = MachinePlan(e, root, cache, false, false, budgets);
      preview->text = disclosure.str();
      return 0;
    }
    if (options.authorize_plan && !options.authorize_plan(MachinePlan(
                                      e, root, cache, false, false, budgets)))
      throw Failure(3, "plan changed; approval required");
    if (!approved) {
      out << disclosure.str();
      if (options.yes)
        out << "approval: --yes (displayed plan accepted)\n";
      else {
        std::string answer;
        if (!std::getline(input, answer) || (answer != "y" && answer != "Y"))
          throw Failure(3, "plan declined");
      }
    }
    if (e.cancelled && e.cancelled())
      throw Failure(3, "cancelled");
    std::vector<std::unique_ptr<PrepareLock>> locks;
    std::set<fs::path> lock_roots{root, cache};
    for (const auto& r : lock_roots) {
      Local(r / Path(".bubo/prepare.lock"));
      fs::create_directories(r / ".bubo");
      auto lock = std::make_unique<PrepareLock>();
      std::string error;
      if (!lock->Acquire(r / Path(".bubo/prepare.lock"), error))
        throw Failure(lock->busy() ? 8 : 7, error);
      locks.push_back(std::move(lock));
    }
    Recover(root, e);
    if (CheckInstalled(root, e)) {
      out << "prepare runtime: already prepared; nothing to do\n";
      where();
      return 0;
    }
    if (options.authorize_plan) {
      PreparationPlan fresh;
      if (const auto code = RunImpl(options, e, input, out, err, &fresh, false))
        return code;
      if (!options.authorize_plan(fresh.machine))
        throw Failure(3, "plan changed; approval required");
    }
    const auto control = root / Path(".bubo/runtime");
    Local(control / "stage");
    Local(control / "backup");
    // With no journal, nothing was moved yet: a stage folder is only this
    // tool's unfinished extraction, and the lock is held. Backups exist only
    // after a journal, so a backup without one is still refused.
    if (fs::exists(control / "stage")) {
      for (const auto& entry :
           fs::recursive_directory_iterator(control / "stage"))
        Local(entry.path());
      fs::remove_all(control / "stage");
    }
    if (fs::exists(control / "backup"))
      throw Failure(7, "unowned runtime backup without journal; remove " +
                           Utf8(control / "backup"));
    fs::create_directories(control / "stage");
    std::vector<std::shared_ptr<ReadLease>> staged;
    const auto names = ManagedPaths(e);
    try {
      for (unsigned i = 0; i < 3; ++i) {
        const auto& w = e.wheels[i];
        const auto final = cache / Path(w.file);
        if (!cached[i]) {
          auto partial = final;
          partial += ".partial";
          auto d = e.download;
          const auto resume_offset =
              options.resume ? RetainedPrefix(entries[i], partial) : 0;
          ReportProgress(options, "runtime", "download", w.name, resume_offset,
                         w.size, resume_offset);
          d.resume = options.resume;
          d.cancelled = e.cancelled;
          d.check_space = [&] {
            const auto v = e.volume(cache);
            return v.error.empty() && !v.network &&
                   v.free_bytes >= 2ULL * 1024 * 1024 * 1024;
          };
          d.progress = [&](auto done, auto total) {
            ReportProgress(options, "runtime", "download", w.name, done, total,
                           resume_offset);
            if (done == total)
              out << "Runtime wheel " << w.name << ' ' << done << '/' << total
                  << " bytes\n";
          };
#ifdef BUBO_PREPARE_TESTING
          if (!e.origins[i].empty())
            d.origin_override = e.origins[i];
#endif
          const auto result = Download(entries[i], partial, *e.transport, d);
          if (result.code)
            throw Failure(result.code, result.error);
          ReportProgress(options, "runtime", "verify", w.name, w.size, w.size);
          std::string error;
          if (!result.verified_file->PromoteTo(partial, final, error))
            throw Failure(7, error);
          archives.push_back(result.verified_file);
          fs::remove(fs::path(partial.wstring() + L".json"));
        }
        // Cached/new archives stay leased through extraction and publication.
        ZipReader zip(final);
        for (const auto& m : e.members)
          if (m.wheel == i) {
            std::uint64_t extracted_bytes = 0;
            ReportProgress(options, "runtime", "extract", m.destination, 0,
                           m.size);
            const auto space = e.volume(root);
            if (!space.error.empty() || space.network ||
                space.free_bytes < m.size + 2ULL * 1024 * 1024 * 1024)
              throw Failure(7, "insufficient runtime staging free space");
            const auto found =
                std::find_if(zip.members().begin(), zip.members().end(),
                             [&](const auto& z) { return z.name == m.member; });
            if (found == zip.members().end() || found->size != m.size)
              throw Failure(
                  6,
                  "missing/wrong-length selected runtime member: " + m.member);
            const auto target = control / "stage" / Path(m.destination);
            Local(target);
            fs::create_directories(target.parent_path());
            std::ofstream file(target, std::ios::binary);
            try {
              zip.Extract(
                  *found, m.size,
                  [&](auto bytes) {
                    file.write(reinterpret_cast<const char*>(bytes.data()),
                               static_cast<std::streamsize>(bytes.size()));
                    if (!file)
                      throw Failure(7, "runtime staging write failed");
                    extracted_bytes += bytes.size();
                    ReportProgress(options, "runtime", "extract", m.destination,
                                   extracted_bytes, m.size);
                  },
                  m.sha256, e.cancelled);
            } catch (const Failure&) {
              throw;
            } catch (const ZipIntegrityError& error) {
              throw Failure(6, error.what());
            } catch (const std::exception& error) {
              throw Failure(e.cancelled && e.cancelled() ? 3 : 9, error.what());
            }
            file.flush();
            if (!file)
              throw Failure(7, "runtime staging flush failed");
            file.close();
          }
      }
      for (const auto& n : e.notices)
        Save(control / Path("stage/licenses") / Path(n.path), n.text);
      Save(control / Path("stage/.bubo/runtime-ready.json"), Ready(e));
      // Protect and independently verify every staged on-disk file before
      // moves.
      for (const auto& m : e.members)
        staged.push_back(
            Seal(control / "stage" / Path(m.destination), m.size, m.sha256));
      for (const auto& n : e.notices)
        staged.push_back(Seal(control / Path("stage/licenses") / Path(n.path),
                              n.text.size(), n.sha256));
      const auto ready = Ready(e);
      staged.push_back(Seal(control / Path("stage/.bubo/runtime-ready.json"),
                            ready.size(), Digest(ready)));
      CheckInstalled(root, e);
      Holders(e);
      auto journal = Value::object();
      journal["phase"] = "installing";
      auto files = Value::array();
      for (const auto& name : names) {
        auto row = Value::object();
        row["path"] = name;
        row["existed"] = fs::exists(root / Path(name)) ? 1 : 0;
        files.push_back(row);
      }
      journal["files"] = files;
      Save(control / "journal.json", journal.dump());
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (e.cancelled && e.cancelled())
          throw Failure(3, "cancelled");
        const auto final = root / Path(names[i]),
                   stage = control / "stage" / Path(names[i]);
        Local(final);
        fs::create_directories(final.parent_path());
        if (fs::exists(final))
          Move(final, control / "backup" / Path(names[i]));
        std::string error;
        if (!staged[i]->PromoteTo(stage, final, error, false))
          throw Failure(7, error);
        ReportProgress(options, "runtime", "promote", names[i],
                       staged[i]->Snapshot().size, staged[i]->Snapshot().size);
        if (e.after_install && !e.after_install(i))
          throw Failure(7, "injected runtime transaction failure");
      }
      journal["phase"] = "committed";
      Save(control / "journal.json", journal.dump());
      staged.clear();
      Recover(root, e);
    } catch (...) {
      staged.clear();
      // mutation-anchor: runtime-rollback
      Recover(root, e);
      if (!fs::exists(control / "journal.json") &&
          fs::exists(control / "stage")) {
        for (const auto& f :
             fs::recursive_directory_iterator(control / "stage"))
          Local(f.path());
        fs::remove_all(control / "stage");
      }
      throw;
    }
    out << "prepare runtime: ready (ROCm 10.0.0 gfx1151)\n";
    where();
    return 0;
  } catch (const Failure& f) {
    err << "prepare: E" << f.code << ' ' << f.what() << '\n';
    return f.code;
  } catch (const std::exception& f) {
    err << "prepare: E9 " << f.what() << '\n';
    return 9;
  }
}
std::string ReadyError(const fs::path& root, const Environment& e) {
  try {
    const auto journal = root / Path(".bubo/runtime/journal.json");
    const auto marker = root / Path(".bubo/runtime-ready.json");
    if (fs::exists(journal))
      throw Failure(2, Utf8(journal) + ": unfinished runtime transaction");
    if (!fs::exists(marker))
      return {};  // Explicit legacy manual installation remains supported.
    if (metadata::Read(marker) != Ready(e))
      throw Failure(2, Utf8(marker) + ": runtime readiness entry mismatch");
    for (const auto& member : e.members) {
      const auto path = root / Path(member.destination);
      if (!fs::is_regular_file(path) || fs::file_size(path) != member.size)
        throw Failure(2,
                      Utf8(path) + ": missing runtime file or size mismatch");
    }
    for (const auto& notice : e.notices) {
      const auto path = root / "licenses" / Path(notice.path);
      if (!fs::is_regular_file(path) ||
          fs::file_size(path) != notice.text.size())
        throw Failure(2,
                      Utf8(path) + ": missing runtime notice or size mismatch");
    }
    return {};
  } catch (const std::exception& error) {
    return std::string(error.what()) + "; run bubo prepare runtime";
  }
}
}  // namespace
namespace {
std::string RuntimeDisclosure(std::span<const WheelPin> wheels,
                              const fs::path& runtime,
                              const fs::path& licenses) {
  std::ostringstream text;
  text << "Bubo needs the AMD ROCm runtime. Bubo does not include AMD files.\n"
          "Bubo downloads three packages directly from AMD (ROCm 10.0.0 for "
          "gfx1151):\n\n";
  std::uint64_t total = 0;
  for (const auto& w : wheels) {
    auto sha = std::string(w.sha256);
    std::transform(sha.begin(), sha.end(), sha.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    text << "  " << std::left << std::setw(25) << w.name << Commas(w.size)
         << " bytes  SHA-256 " << sha << "\n    " << w.url << '\n';
    total += w.size;
  }
  text
      << "  Total " << Commas(total) << " bytes (" << std::fixed
      << std::setprecision(1) << double(total) / (1024 * 1024)
      << " MiB). Bubo checks each hash before it extracts a file.\n\n"
      << "Files go to: " << Utf8(runtime)
      << ". Bubo needs no administrator access.\n\n"
         "Licenses: AMD licenses ROCm per component. Components can have "
         "third-party terms.\n"
         "  AMD license page: "
         "https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html\n"
      << "  License texts that come with the packages (COMGR, hipcc) are saved "
         "in: "
      << Utf8(licenses) << '\n'
      << "  Component licenses (MIT, Apache-2.0 with LLVM exception, BSD): see "
      << Utf8(licenses)
      << "\\NOTICE-windows.md\n"
         "Other AMD terms can apply to these files. Bubo cannot confirm which. "
         "If you continue, you accept the terms that apply to the AMD "
         "files.\n\n"
         "Continue? [y/N]\n";
  return text.str();
}
}  // namespace
std::string RuntimeConsent(const fs::path& runtime, const fs::path& licenses) {
  return RuntimeDisclosure(RuntimeWheels(), runtime, licenses);
}
std::vector<std::string> RuntimeHolders(
    const fs::path& root, const std::vector<RuntimeMember>& members) {
  std::vector<std::string> result;
#ifdef _WIN32
  DWORD session = 0;
  wchar_t key[CCH_RM_SESSION_KEY + 1]{};
  auto rc = RmStartSession(&session, 0, key);
  if (rc != ERROR_SUCCESS)
    throw Failure(7, "cannot check running runtime users: Windows error " +
                         std::to_string(rc));
  struct End {
    DWORD id;
    ~End() { RmEndSession(id); }
  } end{session};
  std::vector<std::wstring> files;
  std::vector<LPCWSTR> pointers;
  for (const auto& m : members)
    if (m.destination.ends_with(".dll")) {
      const auto p = NativePath(root / Path(m.destination));
      if (fs::exists(p))
        files.push_back(p.wstring());
    }
  if (files.empty())
    return result;
  for (const auto& f : files)
    pointers.push_back(f.c_str());
  rc = RmRegisterResources(session, static_cast<UINT>(pointers.size()),
                           pointers.data(), 0, nullptr, 0, nullptr);
  if (rc != ERROR_SUCCESS)
    throw Failure(7, "cannot register runtime DLLs for running-process check");
  UINT needed = 0, count = 0;
  DWORD reason = 0;
  rc = RmGetList(session, &needed, &count, nullptr, &reason);
  if (rc == ERROR_SUCCESS)
    return result;
  for (unsigned tries = 0; tries < 3 && rc == ERROR_MORE_DATA; ++tries) {
    if (needed > 4096)
      throw Failure(7, "too many runtime holders");
    std::vector<RM_PROCESS_INFO> processes(needed);
    count = needed;
    rc = RmGetList(session, &needed, &count, processes.data(), &reason);
    if (rc == ERROR_SUCCESS)
      for (UINT i = 0; i < count; ++i)
        result.push_back(Utf8(fs::path(processes[i].strAppName)) + " (PID " +
                         std::to_string(processes[i].Process.dwProcessId) +
                         ")");
  }
  if (rc != ERROR_SUCCESS)
    throw Failure(7, "cannot enumerate runtime holders");
#else
  (void)root;
  (void)members;
#endif
  return result;
}
int RunRuntime(const Options& options, std::ostream& out, std::ostream& err,
               const std::function<bool()>& cancelled, PreparationPlan* preview,
               bool approved) {
#ifndef _WIN32
  (void)options;
  (void)out;
  (void)cancelled;
  (void)preview;
  (void)approved;
  err << "prepare: E2 AMD Windows runtime preparation requires Windows\n";
  return 2;
#else
  try {
    HostPolicy policy;
    policy.source = HostPolicy::Source::Amd;
    CurlTransport transport(policy);
    Environment e;
    e.runtime_search = ProcessRuntimeSearch(options.local_app_data,
                                            bubo::cli::ExecutableDirectory());
    e.wheels = RuntimeWheels();
    e.members = CompiledRuntimeTable();
    e.notices = RuntimeNotices();
    e.transport = &transport;
    e.download.policy = policy;
    e.cancelled = cancelled;
    return RunImpl(options, std::move(e), std::cin, out, err, preview,
                   approved);
  } catch (const Failure& error) {
    err << "prepare: E" << error.code << ' ' << error.what() << '\n';
    return error.code;
  } catch (const std::exception& error) {
    err << "prepare: E4 " << error.what() << "; no downloads started\n";
    return 4;
  }
#endif
}
std::string RuntimeReadyError(const fs::path& root) {
  try {
    // Avoid catalogue work on the legacy absent-marker path.
    if (!fs::exists(root / Path(".bubo/runtime-ready.json")) &&
        !fs::exists(root / Path(".bubo/runtime/journal.json")))
      return {};
    Environment env;
    env.wheels = RuntimeWheels();
    env.members = CompiledRuntimeTable();
    env.notices = RuntimeNotices();
    return ReadyError(root, env);
  } catch (const std::exception& error) {
    return std::string(error.what()) + "; run bubo prepare runtime";
  }
}
#ifdef BUBO_PREPARE_TESTING
int RunRuntimeForTesting(const Options& options,
                         const RuntimeTestEnvironment& t, std::istream& input,
                         std::ostream& out, std::ostream& err,
                         PreparationPlan* preview, bool approved) {
  Environment e;
  e.runtime_search = t.runtime_search;
  e.wheels = t.wheels;
  e.members = t.members;
  e.notices = t.notices;
  e.transport = t.transport;
  e.download = t.download_options;
  e.origins = t.origins;
  e.holders = t.holders;
  e.after_install = t.after_install;
  e.cancelled = t.cancelled;
  e.synthetic_disclosure = t.synthetic_disclosure;
  if (t.volume)
    e.volume = t.volume;
  return RunImpl(options, std::move(e), input, out, err, preview, approved);
}
std::string RuntimeReadyErrorForTesting(const fs::path& root,
                                        const RuntimeTestEnvironment& test) {
  Environment env;
  env.wheels = test.wheels;
  env.members = test.members;
  env.notices = test.notices;
  return ReadyError(root, env);
}
#endif
}  // namespace bubo::prepare
