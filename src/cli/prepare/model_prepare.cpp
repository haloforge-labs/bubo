#include "src/cli/prepare/model_prepare.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/cli/model_catalogue.hpp"
#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/lowmemory_data.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/model_notices.hpp"
#include "src/core/prepare/transport.hpp"
#include "src/core/prepare/verified_state.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace bubo::prepare {
namespace {
namespace fs = std::filesystem;
using metadata::Value;
constexpr std::uint64_t kMargin = 2ULL * 1024 * 1024 * 1024;
struct Notice {
  std::string name, text, sha256;
};
struct Environment {
  std::span<const Entry> entries;
  // Identity of the model set. The defaults are the Huihui set.
  std::string_view package_id = kPackageId;
  std::string_view recipe_id = kRecipeId;
  // Catalogue download (not a fixed Flash Next set): one licence notice, any
  // number of rows, generic manifest and consent text.
  bool generic = false;
  std::string family = "qwen3.8-flash-next";
  std::vector<Notice> notices;
  Transport* transport = nullptr;
  DownloadOptions download_options;
  std::function<VolumeInfo(const fs::path&)> volume = QueryVolume;
  std::function<std::string(const Entry&, const fs::path&)> fingerprint;
  // Catalogue downloads check an existing main file by its full SHA-256;
  // set, it replaces `fingerprint` and reports verify progress.
  std::function<std::string(const Entry&, const fs::path&, const HashProgress&)>
      full_hash;
  std::function<bool()> cancelled;
  std::function<bool()> before_manifest;
};
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  std::string display(text.begin(), text.end());
  if (display.starts_with("\\\\?\\UNC\\"))
    return "\\\\" + display.substr(8);
  if (display.starts_with("\\\\?\\"))
    return display.substr(4);
  return display;
}
std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    throw std::runtime_error("space calculation overflow");
  return a + b;
}
std::string GiB(std::uint64_t size) {
  std::ostringstream text;
  text << std::fixed << std::setprecision(3)
       << double(size) / (1024 * 1024 * 1024);
  return text.str();
}
void Wrap(std::ostringstream& output, std::string_view line) {
  while (line.size() > 80) {
    auto end = line.rfind(' ', 80);
    if (end == 0 || end == line.npos) {
      end = 80;
      // Long native path tokens can contain multibyte UTF-8. Keep each
      // wrapped line valid rather than splitting a character at byte 80.
      while (end && (static_cast<unsigned char>(line[end]) & 0xc0) == 0x80)
        --end;
    }
    output << line.substr(0, end) << '\n';
    line.remove_prefix(end);
    if (!line.empty() && line.front() == ' ')
      line.remove_prefix(1);
  }
  output << line << '\n';
}
Value PinnedFile(const Entry& entry, std::string_view recipe_id) {
  auto result = Value::object();
  result["file"] = std::string(entry.destination);
  result["size"] = static_cast<unsigned long long>(entry.size);
  result["sha256"] = std::string(entry.sha256);
  if (entry.kind != Kind::MainShard && !recipe_id.empty())
    result["recipe_id"] = std::string(recipe_id);
  if (entry.kind == Kind::Twin || entry.kind == Kind::DraftHead) {
    result["generator_version"] = "1.0.0";
  }
  return result;
}
std::string Manifest(const Environment& env) {
  const auto entries = env.entries;
  auto result = Value::object();
  result["schema_version"] = 1;
  result["package_id"] = std::string(env.package_id);
  result["engine_min"] = "0.1.0";
  auto model = Value::object(), files = Value::array(),
       artefacts = Value::object();
  model["family"] = env.family;
  auto source = Value::object();
  for (const auto& entry : entries) {
    if (entry.kind == Kind::MainShard) {
      if (source.members().empty()) {
        source["repository"] = std::string(entry.source_repo);
        source["revision"] = std::string(entry.commit);
      }
      files.push_back(PinnedFile(entry, env.recipe_id));
    } else
      artefacts[std::string(entry.logical_name)] =
          PinnedFile(entry, env.recipe_id);
  }
  model["source"] = std::move(source);
  model["files"] = std::move(files);
  result["model"] = std::move(model);
  result["artefacts"] = std::move(artefacts);
  auto profiles = Value::array();
  for (auto profile : {"default", "subagents", "exact", "vanilla"})
    profiles.push_back(profile);
  result["profiles_supported"] = std::move(profiles);
  return result.dump() + "\n";
}
struct Plan {
  std::vector<std::string> status;
  // Per entry: why the file in the package folder fails its pin. Empty when
  // it passes or is absent. Such a file is planned like a missing one
  // (`missing` is true); its new copy is downloaded beside it and swapped in
  // only after it verifies. `stamps[i]` holds the damaged file's stamp from
  // the plan until then.
  std::vector<std::string> replaces;
  std::vector<bool> missing;
  std::vector<FileStamp> stamps;
  std::vector<ReadLease> leases;
  std::unique_ptr<ReadLease> manifest_lease;
  std::vector<ConsentVolume> volumes;
  std::uint64_t remaining = 0;
  bool ready = false;
  bool metadata_refresh = false;
  std::string state;
};
fs::path Supplied(const Options& options, const Entry& entry) {
  if (entry.kind == Kind::Mtp && !options.mtp_model.empty())
    return NativePath(fs::u8path(options.mtp_model));
  if (entry.kind == Kind::Mmproj && !options.mmproj.empty())
    return NativePath(fs::u8path(options.mmproj));
  return {};
}
int Error(std::ostream& err, int code, const std::string& text) {
  err << "prepare: E" << code << ' ' << text << '\n';
  return code;
}
// A file that differs from its pin is an integrity failure (6).
int Different(std::ostream& err, const fs::path& path, const std::string& why) {
  return Error(
      err, 6,
      Utf8(path) + ": " + why + "; delete it and run bubo prepare again");
}
// Only a file that really sits inside the package folder may be replaced. A
// link (or a folder junction) that leads elsewhere is the user's own file.
bool InsidePackage(const fs::path& root, const fs::path& file) {
  std::error_code error;
  if (fs::is_symlink(fs::symlink_status(file, error)) ||
      (error && error != std::errc::no_such_file_or_directory))
    return false;
  const auto real_root = fs::weakly_canonical(root, error);
  if (error)
    return false;
  const auto real_file = fs::weakly_canonical(file, error);
  if (error)
    return false;
  const auto relative = real_file.lexically_relative(real_root);
  return !relative.empty() && relative != "." && *relative.begin() != "..";
}
std::string UtcStamp() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm parts{};
#ifdef _WIN32
  gmtime_s(&parts, &now);
#else
  gmtime_r(&now, &parts);
#endif
  char text[32]{};
  std::strftime(text, sizeof text, "%Y%m%dT%H%M%SZ", &parts);
  return text;
}
// Renames a damaged file to a new name in its own folder. A rename moves the
// directory entry only: another hard link to the same data is not touched and
// no byte is written. The new name never replaces an existing file.
fs::path MoveAside(const fs::path& file, std::string& error) {
  const auto stamp = UtcStamp();
  for (int n = 0; n < 100; ++n) {
    auto aside = file;
    aside += ".damaged-" + stamp + (n ? "-" + std::to_string(n + 1) : "");
    std::error_code ignored;
    if (fs::exists(aside, ignored))
      continue;
    return AtomicPromote(file, aside, error) ? aside : fs::path();
  }
  error = "no free name for the damaged copy";
  return {};
}
// A rename that fails because another process holds the file (on Windows the
// model server does) gets a plain next step.
std::string SharingHint(const std::string& error) {
  return error.find("Windows error 32") != std::string::npos ||
                 error.find("Windows error 33") != std::string::npos
             ? ". Stop Bubo's model server, then run prepare again."
             : std::string();
}
std::map<std::string, FileStamp> StoredStamps(const fs::path& state,
                                              const Environment& env,
                                              bool& valid) {
  const auto entries = env.entries;
  valid = false;
  std::map<std::string, FileStamp> stamps;
  if (!fs::exists(state))
    return stamps;
  try {
    const auto json = metadata::Parse(metadata::Read(state));
    metadata::Keys(json, {"schema_version", "package_id", "files"});
    if (metadata::Integer(metadata::Required(json, "schema_version")) != 1 ||
        metadata::String(metadata::Required(json, "package_id")) !=
            env.package_id)
      return {};
    const auto& files = metadata::Required(json, "files");
    if (!files.is_array())
      return {};
    for (const auto& row : files.items()) {
      metadata::Keys(row, {"file", "sha256", "size", "modified", "identity"});
      const auto key = metadata::String(metadata::Required(row, "file")) + ":" +
                       metadata::String(metadata::Required(row, "sha256"));
      FileStamp stamp{metadata::Integer(metadata::Required(row, "size")),
                      metadata::String(metadata::Required(row, "modified")),
                      metadata::String(metadata::Required(row, "identity"))};
      if (!stamps.emplace(key, std::move(stamp)).second)
        return {};
    }
    if (stamps.size() != entries.size())
      return {};
    for (const auto& entry : entries) {
      const auto key =
          std::string(entry.destination) + ":" + std::string(entry.sha256);
      const auto found = stamps.find(key);
      if (found == stamps.end() || found->second.size != entry.size ||
          found->second.modified.empty() || found->second.identity.empty())
        return {};
    }
  } catch (...) {
    return {};
  }  // Invalid stamps authorize nothing.
  valid = true;
  return stamps;
}
std::string State(const Environment& env,
                  const std::vector<FileStamp>& stamps) {
  const auto entries = env.entries;
  auto result = Value::object(), files = Value::array();
  result["schema_version"] = 1;
  result["package_id"] = std::string(env.package_id);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    auto row = Value::object();
    row["file"] = std::string(entries[i].destination);
    row["sha256"] = std::string(entries[i].sha256);
    row["size"] = static_cast<unsigned long long>(stamps[i].size);
    row["modified"] = stamps[i].modified;
    row["identity"] = stamps[i].identity;
    files.push_back(std::move(row));
  }
  result["files"] = std::move(files);
  return result.dump() + "\n";
}
int MakePlan(const Options& options, const Environment& env,
             const fs::path& root, Plan& plan, std::ostream& out,
             std::ostream& err) {
  const auto local_state =
      CheckLocalPath(root / ".bubo" / "prepare-state.json");
  if (!local_state.empty())
    return Error(err, 7, local_state);
  bool valid_state = false;
  const auto stored =
      StoredStamps(root / ".bubo" / "prepare-state.json", env, valid_state);
  std::map<std::string, ConsentVolume> allocations;
  auto allocate = [&](const fs::path& path, std::uint64_t bytes) -> int {
    const auto volume = env.volume(path);
    if (!volume.error.empty())
      return Error(err, 7, volume.error);
    if (volume.filesystem == "FAT32")
      return Error(
          err, 7,
          "FAT32 cannot store this model; use a local NTFS or ReFS volume");
    if (volume.network)
      return Error(err, 7,
                   "UNC and network volumes are not supported; use a local "
                   "folder with --output");
    auto [row, inserted] = allocations.try_emplace(
        volume.identity,
        ConsentVolume{volume.identity, volume.free_bytes, kMargin});
    row->second.free_bytes =
        std::min(row->second.free_bytes, volume.free_bytes);
    row->second.required_bytes = Add(row->second.required_bytes, bytes);
    return 0;
  };
  if (const auto code = allocate(root, 2 * metadata::kMaxJsonBytes))
    return code;
  // A download that starts from nothing: a stale partial file may give a
  // prefix to resume. Used for a missing file and for a file that is replaced.
  auto plan_download = [&](const Entry& entry, const fs::path& path) -> int {
    auto partial = path;
    partial += ".partial";
    auto sidecar = partial;
    sidecar += ".json";
    for (const auto& input : {partial, sidecar}) {
      const auto local = CheckLocalPath(input);
      if (!local.empty())
        return Error(err, 7, local);
    }
    const auto prefix = options.resume ? RetainedPrefix(entry, partial) : 0;
    const auto outstanding = entry.size - std::min(prefix, entry.size);
    if (const auto code = allocate(path, outstanding))
      return code;
    plan.remaining = Add(plan.remaining, outstanding);
    return 0;
  };
  bool missing_notice = false;
  fs::path changed_notice;
  for (const auto& entry : env.entries) {
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled");
    const auto path = NativePath(root / fs::u8path(entry.destination));
    const bool missing = !fs::exists(path);
    plan.missing.push_back(missing);
    plan.replaces.emplace_back();
    plan.status.push_back(missing ? "download" : "reuse");
    plan.stamps.emplace_back();
    const auto supplied = Supplied(options, entry);
    if (!supplied.empty()) {
      const auto local = CheckLocalPath(supplied);
      if (!local.empty())
        return Error(err, 7, local);
      if (!fs::exists(supplied))
        return Error(err, 4, "missing supplied file " + Utf8(supplied));
      plan.leases.emplace_back(supplied);
      const HashProgress cancel_hash = [&](std::uint64_t done,
                                           std::uint64_t total) {
        ReportProgress(options, "model", "verify", entry.logical_name, done,
                       total);
        if (env.cancelled && env.cancelled())
          throw std::runtime_error("cancelled");
      };
      if (plan.leases.back().Snapshot().size != entry.size ||
          HashFile(supplied, cancel_hash) != entry.sha256)
        return Error(
            err, 6,
            Utf8(supplied) +
                ": supplied file does not match the compiled size/SHA-256");
    }
    if (missing) {
      if (!supplied.empty()) {
        plan.status.back() = "import";
        if (const auto code = allocate(path, entry.size))
          return code;
      } else if (const auto code = plan_download(entry, path))
        return code;
      continue;
    }
    if (const auto code = allocate(path, 0))
      return code;
    plan.leases.emplace_back(path);
    const auto stamp = plan.leases.back().Snapshot();
    const auto key =
        std::string(entry.destination) + ":" + std::string(entry.sha256);
    const auto previous = stored.find(key);
    // Empty while the file agrees with its pin.
    std::string damage;
    if (stamp.size != entry.size) {
      damage = "size differs from the pin";
    } else if (entry.kind == Kind::MainShard && !options.verify) {
      if (previous == stored.end() || previous->second != stamp) {
        const HashProgress progress = [&](std::uint64_t done,
                                          std::uint64_t total) {
          ReportProgress(options, "model", "verify", entry.logical_name, done,
                         total);
          if (env.cancelled && env.cancelled())
            throw std::runtime_error("cancelled");
        };
        damage = env.full_hash ? env.full_hash(entry, path, progress)
                               : env.fingerprint(entry, path);
      }
    } else {
      HashProgress progress = [&](std::uint64_t done, std::uint64_t total) {
        ReportProgress(options, "model", "verify", entry.logical_name, done,
                       total);
        if (env.cancelled && env.cancelled())
          throw std::runtime_error("cancelled");
      };
      if (options.verify) {
        out << "prepare verify: " << plan.missing.size() << '/'
            << env.entries.size() << ' ' << entry.destination << '\n';
        const auto start = std::chrono::steady_clock::now();
        progress = [&, start](std::uint64_t done, std::uint64_t total) {
          ReportProgress(options, "model", "verify", entry.logical_name, done,
                         total);
          if (env.cancelled && env.cancelled())
            throw std::runtime_error("cancelled");
          if (done == total) {
            const auto seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            out << "prepare verify: " << entry.destination << ' ' << done << '/'
                << total << " bytes, " << GiB(done) << " GiB in " << seconds
                << " seconds\n";
          }
        };
      }
      if (HashFile(path, progress) != entry.sha256)
        damage = "SHA-256 differs from the pin";
    }
    if (plan.leases.back().Snapshot() != stamp)
      return Different(err, path, "file changed during verification");
    if (!damage.empty()) {
      // A file that Bubo placed in the package folder and that fails its pin
      // is planned as a new download. Nothing changes on disk before the
      // approval: the plan only records it. A file that is not really inside
      // the package folder (a link to another place) is never replaced.
      if (!InsidePackage(root, path))
        return Different(err, path, damage);
      // The lease would deny the rename on Windows; the damaged file is not
      // verified content, so nothing needs to stay locked.
      plan.leases.pop_back();
      plan.missing.back() = true;
      plan.replaces.back() = damage;
      plan.stamps.back() = stamp;
      if (!supplied.empty()) {
        // The user's own file is the verified source; only the package copy
        // is replaced, and the user's file is read, not changed.
        plan.status.back() = "import";
        if (const auto code = allocate(path, entry.size))
          return code;
      } else {
        plan.status.back() = "download";
        if (const auto code = plan_download(entry, path))
          return code;
      }
      continue;
    }
    plan.stamps.back() = stamp;
  }
  for (const auto& notice : env.notices) {
    const auto path = root / "licenses" / notice.name;
    if (const auto local = CheckLocalPath(path); !local.empty())
      return Error(err, 7, local);
    if (!InsidePackage(root, path))
      return Different(err, path, "notice is a link outside the package");
    if (!fs::exists(path)) {
      missing_notice = true;
      if (const auto code = allocate(path, notice.text.size()))
        return code;
    } else {
      ReadLease lease(path);
      if (HashFile(path) != notice.sha256) {
        missing_notice = true;
        changed_notice = path;
      }
    }
  }
  const auto manifest = root / "bubo-package.json";
  if (const auto local = CheckLocalPath(manifest); !local.empty())
    return Error(err, 7, local);
  if (!InsidePackage(root, manifest))
    return Different(err, manifest,
                     "package marker is a link outside the package");
  bool valid_manifest = false;
  bool older_catalogue_marker = false;
  std::string malformed_manifest;
  if (fs::exists(manifest)) {
    plan.manifest_lease = std::make_unique<ReadLease>(manifest);
    try {
      const auto present_text = metadata::Read(manifest);
      const auto present = metadata::Parse(present_text);
      valid_manifest = present_text == Manifest(env);
      older_catalogue_marker =
          env.generic && present.member_str("package_id") == env.package_id;
    } catch (const std::exception& error) {
      // Readiness belongs to Bubo. A partial or older marker is repaired
      // only after the payload verification and existing repair rules above.
      valid_manifest = false;
      malformed_manifest = error.what();
    }
  }
  const bool any_missing =
      std::ranges::any_of(plan.missing, [](bool value) { return value; });
  // Refresh is justified by pinned payloads, never by the old metadata.
  // If payload acquisition or repair is needed, retain the previous refusal
  // rules for differing notices and foreign/malformed package markers.
  if (any_missing && !changed_notice.empty())
    return Different(err, changed_notice,
                     "notice SHA-256 differs from the pin");
  if (any_missing && !malformed_manifest.empty())
    return Error(err, 2, Utf8(manifest) + ": " + malformed_manifest);
  if (any_missing && plan.manifest_lease && !valid_manifest &&
      !older_catalogue_marker)
    return Error(err, 6,
                 Utf8(manifest) +
                     ": the package marker does not match this Bubo version; "
                     "delete it and run bubo prepare again");
  plan.ready = !any_missing && !missing_notice && valid_manifest && valid_state;
  plan.metadata_refresh = !any_missing && valid_state;
  if (plan.ready)
    return 0;
  if (options.offline && any_missing) {
    for (std::size_t i = 0; i < env.entries.size(); ++i) {
      if (!plan.missing[i] || !Supplied(options, env.entries[i]).empty())
        continue;
      return Error(err, 4,
                   std::string(plan.replaces[i].empty()
                                   ? "offline: missing "
                                   : "offline: cannot replace the damaged ") +
                       Utf8(root / fs::u8path(env.entries[i].destination)) +
                       " SHA-256 " + std::string(env.entries[i].sha256));
    }
  }
  for (const auto& [name, volume] : allocations) {
    plan.volumes.push_back(volume);
    // --plan reports free and required space; it does not stop on a shortage.
    if (volume.free_bytes < volume.required_bytes && !options.plan_only)
      return Error(err, 7, NotEnoughSpaceText(volume));
  }
  return 0;
}
// `awaiting[i]` marks a damaged file that is still in place: its new copy
// still needs the whole size (less what the partial file already holds).
bool StreamingSpace(const Environment& env, const fs::path& root,
                    const std::vector<bool>* awaiting = nullptr) {
  std::map<std::string, ConsentVolume> volumes;
  const auto allocate = [&](const fs::path& path, std::uint64_t bytes) {
    const auto volume = env.volume(path);
    if (!volume.error.empty() || volume.filesystem == "FAT32" || volume.network)
      return false;
    auto [row, inserted] = volumes.try_emplace(
        volume.identity,
        ConsentVolume{volume.identity, volume.free_bytes, kMargin});
    row->second.free_bytes =
        std::min(row->second.free_bytes, volume.free_bytes);
    row->second.required_bytes = Add(row->second.required_bytes, bytes);
    return true;
  };
  // Control metadata and legal texts may be on a different volume from
  // payloads reached through mounted directories. Keep their own margin.
  if (!allocate(root, 2 * metadata::kMaxJsonBytes))
    return false;
  for (const auto& notice : env.notices) {
    const auto path = root / "licenses" / notice.name;
    if (!allocate(path, fs::exists(path) ? 0 : notice.text.size()))
      return false;
  }
  for (std::size_t i = 0; i < env.entries.size(); ++i) {
    const auto& entry = env.entries[i];
    const auto final = NativePath(root / fs::u8path(entry.destination));
    std::uint64_t outstanding = 0;
    if (!fs::exists(final) || (awaiting && (*awaiting)[i])) {
      auto partial = final;
      partial += ".partial";
      const auto prefix = fs::exists(partial)
                              ? std::min(entry.size, fs::file_size(partial))
                              : 0;
      outstanding = entry.size - prefix;
    }
    if (!allocate(final, outstanding))
      return false;
  }
  return std::ranges::all_of(volumes, [](const auto& row) {
    return row.second.free_bytes >= row.second.required_bytes;
  });
}
int CopySupplied(const fs::path& source, const fs::path& partial,
                 const Entry& entry, const Environment& env,
                 const fs::path& root, std::ostream& err,
                 std::shared_ptr<ReadLease>& verified, const Options& options,
                 const std::vector<bool>* awaiting) {
  ReadLease lease(source);
  const auto stamp = lease.Snapshot();
  std::ifstream input(source, std::ios::binary);
  if (!input)
    return Error(err, 7, "cannot stage supplied file");
#ifdef _WIN32
  const auto handle = CreateFileW(
      NativePath(partial).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
      FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    return Error(err, 7,
                 "cannot open supplied-file partial: Windows error " +
                     std::to_string(GetLastError()));
  auto output = std::make_shared<ReadLease>(
      ReadLease::AdoptVerifiedHandle(reinterpret_cast<std::intptr_t>(handle)));
#else
  const auto handle = open(partial.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (handle == -1)
    return Error(err, 7, "cannot open supplied-file partial");
  auto output =
      std::make_shared<ReadLease>(ReadLease::AdoptVerifiedHandle(handle));
  if (flock(handle, LOCK_EX | LOCK_NB))
    return Error(err, 7, "cannot lease supplied-file partial");
#endif
  std::vector<std::uint8_t> buffer(4 * 1024 * 1024);
  gufo::crypto::Sha256Hasher hasher;
  std::uint64_t done = 0;
  while (input) {
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled; supplied original remains intact");
    if (!StreamingSpace(env, root, awaiting))
      return Error(err, 7, "insufficient space while copying supplied file");
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = static_cast<std::size_t>(input.gcount());
    if (count > entry.size - std::min(entry.size, done))
      return Error(err, 6, "supplied file grew during copy");
    hasher.Update(std::span(buffer).first(count));
#ifdef _WIN32
    DWORD written = 0;
    if (!WriteFile(handle, buffer.data(), static_cast<DWORD>(count), &written,
                   nullptr) ||
        written != count)
      return Error(err, 7, "cannot write supplied-file partial");
#else
    std::size_t written = 0;
    while (written < count) {
      const auto amount =
          write(handle, buffer.data() + written, count - written);
      if (amount <= 0)
        return Error(err, 7, "cannot write supplied-file partial");
      written += static_cast<std::size_t>(amount);
    }
#endif
    if (output->Snapshot().size != done + count)
      return Error(err, 7, "cannot write supplied-file partial");
    done += count;
    ReportProgress(options, "model", "copy", entry.logical_name, done,
                   entry.size);
  }
  if (input.bad())
    return Error(err, 7, "supplied file copy failed");
  if (done != entry.size || hasher.FinishHex() != entry.sha256 ||
      lease.Snapshot() != stamp)
    return Error(err, 6,
                 Utf8(source) + ": supplied-file SHA-256 or size mismatch");
#ifdef _WIN32
  FILETIME actual{}, frozen{0xffffffff, 0xffffffff};
  GetSystemTimeAsFileTime(&actual);
  if (!FlushFileBuffers(handle) ||
      !SetFileTime(handle, nullptr, nullptr, &actual) ||
      !SetFileTime(handle, nullptr, nullptr, &frozen))
    return Error(err, 7,
                 "cannot flush/freeze verified supplied file: Windows error " +
                     std::to_string(GetLastError()));
#else
  if (fsync(handle))
    return Error(err, 7, "cannot flush verified supplied file");
#endif
  verified =
      std::move(output);  // Original writer remains open through promotion.
  return 0;
}
void SavePreparedVerification(const fs::path& root, const Environment& env,
                              const Plan& plan, bool verify,
                              std::ostream& err) {
  const auto entries = env.entries;
  VerifiedStamps rows;
  for (std::size_t i = 0; i < entries.size(); ++i)
    if (verify || plan.missing[i])
      rows[{std::string(entries[i].logical_name),
            std::string(entries[i].sha256)}] = plan.stamps[i];
  std::string error;
  if (!SaveVerifiedStamps(root, rows, error, env.package_id)) {
    std::replace_if(
        error.begin(), error.end(),
        [](unsigned char c) { return c < 32 || c == 127; }, ' ');
    err << "verification stamp not saved: " << error << '\n';
  }
}
Value MachinePlan(const Options& options, const Environment& env,
                  const fs::path& root, const Plan& plan) {
  auto value = Value::object();
  value["component"] = "model";
  value["ready"] = plan.ready;
  value["sources"] = Value::array();
  value["licenses"] = Value::array();
  value["paths"] = Value::object();
  value["paths"]["model"] = Utf8(root);
  value["volumes"] = Value::array();
  value["file_states"] = Value::array();
  // Parallel to `sources`: why the file in the package folder is replaced, or
  // "" when nothing is replaced. Part of the plan digest.
  value["replaces"] = Value::array();
  for (std::size_t i = 0; i < env.entries.size(); ++i) {
    const auto& entry = env.entries[i];
    const auto url = OriginUrl(entry);
    value["sources"].push_back(PlanSource(
        entry.logical_name, url, entry.size, entry.sha256, entry.source_repo,
        entry.commit, Utf8(root / fs::u8path(entry.destination)),
        entry.licence_id));
    value["file_states"].push_back(plan.status[i]);
    value["replaces"].push_back(plan.replaces[i]);
    if (const auto supplied = Supplied(options, entry); !supplied.empty())
      value["paths"][std::string(entry.logical_name)] = Utf8(supplied);
  }
  for (const auto& notice : env.notices) {
    auto row = Value::object();
    row["path"] = notice.name;
    row["text"] = notice.text;
    row["sha256"] = notice.sha256;
    value["licenses"].push_back(row);
  }
  for (const auto& volume : plan.volumes)
    value["volumes"].push_back(PlanVolume(volume));
  return value;
}
// ---- catalogue downloads (any model, not a fixed Flash Next set) ----------
std::string ValidateGenericFields(std::span<const Entry> entries) {
  if (entries.empty())
    return "empty catalogue entry";
  std::set<std::string> names, destinations;
  for (const auto& entry : entries) {
    const auto fail = [&](std::string_view why) {
      return std::string(entry.logical_name) + ": " + std::string(why);
    };
    if (!entry.direct_url.empty())
      return fail("a model download cannot set an AMD origin");
    if (!IsRelativePath(entry.logical_name) ||
        entry.logical_name.find('/') != entry.logical_name.npos ||
        !names.insert(std::string(entry.logical_name)).second)
      return fail("empty or repeated file name");
    if (!IsRelativePath(entry.file_name) || !IsRelativePath(entry.repo_path) ||
        !IsRelativePath(entry.destination) ||
        !destinations.insert(std::string(entry.destination)).second)
      return fail("invalid or repeated path");
    if (!entry.size || entry.size > 9007199254740991ULL ||
        !IsLowerHex(entry.sha256, 64))
      return fail("invalid size or SHA-256");
    if (!IsLowerHex(entry.commit, 40) || entry.source_repo.empty())
      return fail("missing pinned repository or commit");
  }
  return {};
}
std::string GenericConsent(const Environment& env, const fs::path& root,
                           std::span<const std::string> status,
                           std::uint64_t remaining,
                           std::span<const ConsentVolume> volumes, bool ask) {
  std::ostringstream text;
  Wrap(text, "Bubo will prepare these files.");
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < env.entries.size(); ++i) {
    const auto& entry = env.entries[i];
    total = Add(total, entry.size);
    Wrap(text, std::string(entry.file_name) + " | huggingface.co | " +
                   OriginUrl(entry) + " | " + std::to_string(entry.size) +
                   " bytes | " + GiB(entry.size) + " GiB | " +
                   std::string(entry.sha256.substr(0, 12)) + " | " +
                   std::string(entry.sha256) + " | " +
                   std::string(entry.licence_id) + " | " +
                   Utf8(root / fs::u8path(entry.destination)) + " | " +
                   (i < status.size() ? status[i] : "download"));
  }
  Wrap(text, "Total file size: " + std::to_string(total) + " bytes (" +
                 GiB(total) + " GiB).");
  Wrap(text, "Total download size: " + std::to_string(remaining) + " bytes (" +
                 GiB(remaining) + " GiB).");
  for (const auto& volume : volumes)
    Wrap(text, "Free space: " + std::to_string(volume.free_bytes) +
                   " bytes. Required space: " +
                   std::to_string(volume.required_bytes) +
                   " bytes. Volume: " + volume.name + ".");
  // The licence block is the catalogue's own prompt line, link and note,
  // then one block per side file from another repository. The files under
  // licenses\ are written by the download, so the plan names the folder.
  for (const auto& notice : env.notices) {
    std::istringstream licence(notice.text);
    for (std::string line; std::getline(licence, line);)
      if (!line.empty())
        Wrap(text, line);
  }
  Wrap(text, "A copy is saved in " + Utf8(root / "licenses") +
                 " after the download.");
  Wrap(text, "Estimated download time: unknown; download speed is not known.");
  Wrap(text,
       "Bubo checks each download's full SHA-256 while it receives the file.");
  Wrap(text,
       "Downloads can redirect to allowed hosts under huggingface.co or "
       "hf.co.");
  Wrap(text, "Bubo will not build a model on this computer.");
  if (ask)
    Wrap(text, "Download and prepare all files in this plan? [y/N]");
  return text.str();
}
// The status column of the text plan. A file that replaces a damaged one says
// so in plain words; the machine plan keeps the bare state.
std::vector<std::string> ShownStatus(const Plan& plan) {
  auto shown = plan.status;
  for (std::size_t i = 0; i < shown.size() && i < plan.replaces.size(); ++i)
    if (!plan.replaces[i].empty())
      shown[i] += " | replaces a file that did not match";
  return shown;
}
int RunImpl(const Options& options, const Environment& env, std::istream& input,
            std::ostream& out, std::ostream& err, PreparationPlan* preview,
            bool approved) {
  try {
    // Production approval/text slots are intentionally OPEN. Do not invent
    // licence texts or fetch mutable legal text to get past the release gate.
    if (env.generic) {
      std::set<std::string> notice_names;
      const bool notices_ok =
          !env.notices.empty() && env.notices[0].name == "LICENSE-INFO.txt" &&
          std::ranges::all_of(env.notices, [&](const Notice& notice) {
            return !notice.text.empty() && IsRelativePath(notice.name) &&
                   notice.name.find('/') == std::string::npos &&
                   notice_names.insert(notice.name).second &&
                   gufo::crypto::Sha256Hex(
                       std::span(reinterpret_cast<const std::uint8_t*>(
                                     notice.text.data()),
                                 notice.text.size())) == notice.sha256;
          });
      if (!notices_ok)
        return Error(err, 4, "catalogue licence notice is missing or damaged");
      const auto invalid_generic = ValidateGenericFields(env.entries);
      if (!invalid_generic.empty())
        return Error(err, 4, invalid_generic + "; no downloads started");
    } else if (env.notices.size() != 3)
      return Error(err, 4,
                   "approved pinned model LICENSE, NOTICE and PROVENANCE.md "
                   "bundle is OPEN; no downloads started");
    for (std::size_t i = 0; !env.generic && i < env.notices.size(); ++i) {
      const auto& notice = env.notices[i];
      constexpr std::string_view names[] = {"LICENSE", "NOTICE",
                                            "PROVENANCE.md"};
      if (notice.name != names[i] || notice.text.empty() ||
          notice.text.size() > metadata::kMaxJsonBytes ||
          gufo::crypto::Sha256Hex(std::span(
              reinterpret_cast<const std::uint8_t*>(notice.text.data()),
              notice.text.size())) != notice.sha256)
        return Error(err, 4,
                     "model notice bundle does not match its compiled pins");
    }
    if (!env.generic && env.entries.size() != 8)
      return Error(err, 4, "model catalogue must have eight fixed entries");
    const auto invalid =
        env.generic ? std::string() : ValidateFields(env.entries, true);
    if (!invalid.empty())
      return Error(
          err, 4, invalid + "; production acquisition is gated before network");
    const auto root = NativePath(fs::u8path(ModelRoot(options)));
    const auto local = CheckLocalPath(root);
    if (!local.empty())
      return Error(err, 7, local);
    Plan initial;
    if (const auto code = MakePlan(options, env, root, initial, out, err))
      return code;
    if (preview) {
      if (options.machine)
        preview->machine = MachinePlan(options, env, root, initial);
      preview->ready = initial.ready;
      preview->volumes = initial.volumes;
      // --plan shows the plan only: it asks nothing.
      const auto shown = ShownStatus(initial);
      preview->text =
          env.generic
              ? GenericConsent(env, root, shown, initial.remaining,
                               initial.volumes, !options.plan_only)
              : ModelConsent(env.entries, root, shown, initial.remaining,
                             initial.volumes, options.step != Step::Runtime,
                             !options.plan_only);
      return 0;
    }
    // mutation-anchor: initial-readonly-check
    if (initial.ready) {
      if (options.verify)
        SavePreparedVerification(root, env, initial, true, err);
      out << "prepare: already prepared; nothing to do\n";
      return 0;
    }
    if (!approved) {
      const auto shown = ShownStatus(initial);
      out << (env.generic ? GenericConsent(env, root, shown, initial.remaining,
                                           initial.volumes, true)
                          : ModelConsent(env.entries, root, shown,
                                         initial.remaining, initial.volumes,
                                         options.step != Step::Runtime));
      if (options.yes)
        out << "approval: --yes (displayed plan accepted)\n";
      else {
        std::string answer;
        if (!std::getline(input, answer) || (answer != "y" && answer != "Y"))
          return Error(err, 3, "plan declined");
      }
    }
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled");
    initial.leases.clear();
    initial.manifest_lease.reset();
    fs::create_directories(root / ".bubo");
    PrepareLock lock;
    std::string error;
    if (!lock.Acquire(root / ".bubo" / "prepare.lock", error))
      return Error(err, lock.busy() ? 8 : 7, "prepare lock: " + error);
    Plan plan;
    if (const auto code = MakePlan(options, env, root, plan, out, err))
      return code;
    if (plan.ready) {
      if (options.verify)
        SavePreparedVerification(root, env, plan, true, err);
      out << "prepare: already prepared; nothing to do\n";
      return 0;
    }
    if (options.authorize_plan &&
        !options.authorize_plan(MachinePlan(options, env, root, plan)))
      return Error(err, 3, "plan changed; approval required");
    // An existing valid readiness file describes a formerly complete package.
    // Withdraw it before repair, so a crash never advertises incomplete files.
    plan.manifest_lease.reset();
    const bool refresh_only = plan.metadata_refresh;
    fs::remove(root / "bubo-package.json");
    // A damaged file stays in place until its new copy is swapped in, so the
    // new copy needs the whole size.
    std::vector<bool> awaiting(env.entries.size());
    for (std::size_t i = 0; i < env.entries.size(); ++i)
      awaiting[i] = !plan.replaces[i].empty();
    // mutation-anchor: before-file-loop
    for (std::size_t i = 0; i < env.entries.size(); ++i) {
      const auto& entry = env.entries[i];
      const auto final = NativePath(root / fs::u8path(entry.destination));
      if (!plan.missing[i]) {
        // Keep the original verified read lease from MakePlan through the
        // readiness write; no close/reopen gap permits a concurrent mutation.
        continue;
      }
      if (env.cancelled && env.cancelled())
        return Error(err, 3, "cancelled");
      fs::create_directories(final.parent_path());
      // A damaged file stays at its name while the new copy downloads beside
      // it, so a failed, cancelled or killed repair leaves nothing to clean up.
      auto partial = final;
      partial += ".partial";
      auto download = env.download_options;
      const auto resume_offset =
          options.resume ? RetainedPrefix(entry, partial) : 0;
      ReportProgress(options, "model", "download", entry.logical_name,
                     resume_offset, entry.size, resume_offset);
      download.resume = options.resume;
      download.cancelled = env.cancelled;
      download.check_space = [&] {
        return StreamingSpace(env, root, &awaiting);
      };
      download.on_host = [&](std::string_view host) {
        out << "prepare download: " << entry.destination << " host " << host
            << '\n';
      };
      std::uint64_t displayed = 0;
      download.progress = [&](std::uint64_t done, std::uint64_t total) {
        ReportProgress(options, "model", "download", entry.logical_name, done,
                       total, resume_offset);
        // One human line per GiB, and one at the end.
        if (done == total ||
            done - std::min(done, displayed) >= 1024ULL * 1024 * 1024) {
          out << "prepare download: " << entry.destination << ' ' << done << '/'
              << total << " bytes\n";
          displayed = done;
        }
      };
      const auto source = Supplied(options, entry);
      std::shared_ptr<ReadLease> verified;
      if (!source.empty()) {
        ReportProgress(options, "model", "copy", entry.logical_name, 0,
                       entry.size);
        if (const auto code = CopySupplied(source, partial, entry, env, root,
                                           err, verified, options, &awaiting))
          return code;
      } else {
        if (!env.transport)
          return Error(err, 9, "missing transport");
        const auto result = Download(entry, partial, *env.transport, download);
        if (result.code)
          return Error(err, result.code,
                       std::string(entry.destination) + ": " + result.error);
        verified = result.verified_file;
      }
      if (!verified)
        return Error(err, 9, "verified writer lease missing");
      ReportProgress(options, "model", "verify", entry.logical_name, entry.size,
                     entry.size);
      const auto verified_stamp = verified->Snapshot();
      if (verified_stamp.size != entry.size)
        return Error(err, 6, "verified partial size changed");
      // The new copy has passed its full SHA-256. A damaged file is swapped
      // out now, by renames only: bytes shared through another hard link stay
      // as they are, and nothing is written into the old file.
      fs::path aside;
      if (!plan.replaces[i].empty() && fs::exists(final)) {
        bool unchanged = false;
        try {
          ReadLease now(final);
          unchanged = now.Snapshot() == plan.stamps[i];
        } catch (const std::exception&) {
        }
        if (!unchanged)
          return Error(err, 3, "plan changed; approval required");
        aside = MoveAside(final, error);
        if (aside.empty())
          return Error(err, 7,
                       Utf8(final) + ": cannot move the damaged file aside: " +
                           error + SharingHint(error));
      }
      if (!verified->PromoteTo(partial, final, error)) {
        const auto hint = SharingHint(error);
        std::string back;
        if (!aside.empty() && !AtomicPromote(aside, final, back))
          return Error(err, 7,
                       Utf8(final) + ": " + error +
                           "; the damaged copy is at " + Utf8(aside) +
                           " (cannot rename it back: " + back + ")");
        return Error(err, 7, Utf8(final) + ": " + error + hint);
      }
      ReportProgress(options, "model", "promote", entry.logical_name,
                     entry.size, entry.size);
      auto sidecar = partial;
      sidecar += ".json";
      fs::remove(sidecar);
      plan.leases.push_back(std::move(*verified));
      plan.stamps[i] = plan.leases.back().Snapshot();
      if (plan.stamps[i] != verified_stamp)
        return Different(err, final,
                         "verified file identity changed during promotion");
      awaiting[i] = false;
      if (!aside.empty()) {
        // Removing the directory entry of the old copy leaves other hard
        // links to its data as they are.
        std::error_code removed;
        fs::remove(aside, removed);
        out << "prepare repair: " << entry.destination << " did not match ("
            << plan.replaces[i] << "); replaced\n";
        if (removed)
          out << "prepare repair: could not delete the damaged copy "
              << Utf8(aside) << " (" << removed.message()
              << "); delete it yourself\n";
      }
      // The streaming full hash proves the downloaded bytes match the pin;
      // repeating the fingerprint reads here would add unnecessary model I/O.
    }
    fs::create_directories(root / "licenses");
    for (const auto& notice : env.notices) {
      const auto path = root / "licenses" / notice.name;
      if (const auto local = CheckLocalPath(path); !local.empty())
        return Error(err, 7, local);
      if ((!fs::exists(path) || HashFile(path) != notice.sha256) &&
          !AtomicWrite(path, notice.text, error))
        return Error(err, 7, error);
      plan.leases.emplace_back(path);
      if (HashFile(path) != notice.sha256)
        return Different(err, path, "notice pin mismatch");
    }
    const auto state_path = root / ".bubo" / "prepare-state.json";
    SavePreparedVerification(root, env, plan, options.verify, err);
    const auto state_text = State(env, plan.stamps);
    if (!AtomicWrite(state_path, state_text, error))
      return Error(err, 7, error);
    plan.leases.emplace_back(state_path);
    if (metadata::Read(state_path) != state_text)
      return Error(err, 7,
                   "prepare state changed before readiness publication");
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled before readiness publication");
    if (env.before_manifest && !env.before_manifest())
      return Error(err, 3, "interrupted before readiness publication");
    // LAST: no later state, stamp, notice or payload write may follow
    // readiness.
    if (!AtomicWrite(root / "bubo-package.json", Manifest(env), error))
      return Error(err, 7, error);
    if (refresh_only)
      out << "prepare: already prepared; nothing to download\n"
             "prepare: notices refreshed\n";
    else
      out << "prepare: model prepared\n";
    return 0;
  } catch (const std::exception& error) {
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled");
    return Error(err, 7, error.what());
  }
}
}  // namespace

std::string ModelConsent(std::span<const Entry> entries, const fs::path& root,
                         std::span<const std::string> status,
                         std::uint64_t remaining,
                         std::span<const ConsentVolume> volumes,
                         bool model_selected, bool ask) {
  std::ostringstream text;
  Wrap(text, "Bubo will prepare these files.");
  std::uint64_t total = 0, shards = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    total = Add(total, entry.size);
    if (entry.kind == Kind::MainShard)
      shards = Add(shards, entry.size);
    Wrap(text, std::string(entry.file_name) + " | huggingface.co | " +
                   OriginUrl(entry) + " | " + std::to_string(entry.size) +
                   " bytes | " + GiB(entry.size) + " GiB | " +
                   std::string(entry.sha256.substr(0, 12)) + " | " +
                   std::string(entry.sha256) + " | " +
                   std::string(entry.licence_id) + " | " +
                   Utf8(root / fs::u8path(entry.destination)) + " | " +
                   (i < status.size() ? status[i] : "download"));
  }
  Wrap(text, "Total file size: " + std::to_string(total) + " bytes (" +
                 GiB(total) + " GiB).");
  Wrap(text, "Total download size: " + std::to_string(remaining) + " bytes (" +
                 GiB(remaining) + " GiB).");
  for (const auto& volume : volumes)
    Wrap(text, "Free space: " + std::to_string(volume.free_bytes) +
                   " bytes. Required space: " +
                   std::to_string(volume.required_bytes) +
                   " bytes. Volume: " + volume.name + ".");
  Wrap(text, "The derived files are hosted by the Bubo project.");
  Wrap(text,
       "Qwen Community License 1.0 applies to the model files. They are not "
       "MIT licensed.");
  // LICENSE and NOTICE are written by the download.
  Wrap(text, "A copy of LICENSE and NOTICE is saved in " +
                 Utf8(root / "licenses") + " after the download.");
  Wrap(text, "Read the MaaS and AI Work Assistant conditions in the licence.");
  if (model_selected)
    Wrap(text, "The main shards total " + std::to_string(shards) + " bytes (" +
                   GiB(shards) + " GiB).");
  Wrap(text, "Estimated download time: unknown; download speed is not known.");
  // Download policy requires streaming verification, superseding
  // the addendum's claim that new downloads receive a second complete read.
  Wrap(text,
       "Bubo checks each download's full SHA-256 while it receives the file.");
  Wrap(text,
       "Existing main shards use GGUF and tensor fingerprint checks; --verify "
       "checks full SHA-256.");
  Wrap(text,
       "HF downloads can redirect to allowed hosts under huggingface.co or "
       "hf.co.");
  Wrap(text, "Bubo will not build a model on this computer.");
  if (ask)
    Wrap(text, "Download and prepare all files in this plan? [y/N]");
  return text.str();
}
int RunModel(const Options& options, std::ostream& out, std::ostream& err,
             const std::function<bool()>& cancelled, PreparationPlan* preview,
             bool approved) {
  if (!options.catalogue_id.empty())
    if (const auto* entry = bubo::cli::FindCatalogueEntry(
            bubo::cli::ShippedCatalogue().entries, options.catalogue_id);
        entry && !entry->flash_next_set)
      return RunCatalogueModel(options, *entry, out, err, cancelled, preview,
                               approved);
  CurlTransport transport;
  Environment env;
  env.entries = ModelCatalogue(options.model_set);
  env.package_id = PackageId(options.model_set);
  env.recipe_id = RecipeId(options.model_set);
  for (const auto& notice : ModelNotices(options.model_set))
    env.notices.push_back({std::string(notice.name), std::string(notice.text),
                           std::string(notice.sha256)});
  env.transport = &transport;
  env.cancelled = cancelled;
  env.fingerprint = [](const Entry& entry, const fs::path& path) {
    const auto table = ProductionFingerprintTable(entry);
    return table ? CheckGgufFingerprint(path, entry, table.table) : table.error;
  };
  return RunImpl(options, env, std::cin, out, err, preview, approved);
}
namespace {
// Rows and strings of one catalogue download. Entry holds string_views, so
// the strings live here for as long as the run.
struct GenericPackage {
  std::deque<std::string> strings;
  std::vector<Entry> entries;
  // LICENSE-INFO.txt first, then one notice per side file that comes from
  // another repository (its own licence line).
  std::vector<Notice> notices;
  std::string error;
  const std::string& Keep(std::string text) {
    strings.push_back(std::move(text));
    return strings.back();
  }
};
bool BuildGenericPackage(const bubo::cli::CatalogueEntry& entry,
                         GenericPackage& package) {
  if (!entry.download_enabled) {
    package.error =
        "download is off for " + entry.id + ": " + entry.disabled_reason;
    return false;
  }
  if (const auto problem = bubo::cli::PinProblem(entry); !problem.empty()) {
    package.error = problem;
    return false;
  }
  const auto& licence = package.Keep(entry.licence.id);
  for (const auto& file : entry.files) {
    if (!file.downloadable)
      continue;
    Entry row{};
    row.logical_name = package.Keep(file.name);
    row.file_name = row.logical_name;
    row.size = file.bytes;
    row.sha256 = package.Keep(file.sha256);
    row.source_repo = package.Keep(file.repository.empty() ? entry.repository
                                                           : file.repository);
    row.commit = package.Keep(file.commit.empty() ? entry.commit : file.commit);
    row.repo_path = package.Keep(file.repo_path);
    if (file.role == "main")
      row.kind = Kind::MainShard;
    else if (file.role == "mtp")
      row.kind = Kind::Mtp;
    else if (file.role == "mmproj")
      row.kind = Kind::Mmproj;
    else if (file.role == "twin")
      row.kind = Kind::Twin;
    else if (file.role == "draft_head")
      row.kind = Kind::DraftHead;
    else
      row.kind = Kind::Support;
    row.licence_id = licence;
    row.destination = row.file_name;
    package.entries.push_back(row);
  }
  std::ostringstream notice;
  notice << entry.licence.prompt_line << '\n'
         << "Licence: " << entry.licence.name << " (" << entry.licence.id
         << ")\n"
         << "Licence link: " << entry.licence.url << '\n';
  if (!entry.licence.note.empty())
    notice << "Note: " << entry.licence.note << '\n';
  notice << "Model: " << entry.display_name << '\n';
  const auto add_notice = [&](std::string name, std::string text) {
    auto sha = gufo::crypto::Sha256Hex(std::span(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
    package.notices.push_back({std::move(name), std::move(text), sha});
  };
  add_notice("LICENSE-INFO.txt", notice.str());
  // PinProblem guarantees a licence line for every downloaded file from
  // another repository.
  for (const auto& file : entry.files) {
    if (!file.downloadable || file.licence_text.empty())
      continue;
    std::ostringstream side;
    side << file.licence_text << '\n'
         << "File: " << file.name << '\n'
         << "Source: https://huggingface.co/"
         << (file.repository.empty() ? entry.repository : file.repository)
         << "/tree/" << (file.commit.empty() ? entry.commit : file.commit)
         << '\n'
         << "Model: " << entry.display_name << '\n';
    add_notice("LICENSE-INFO-" + file.name + ".txt", side.str());
  }
  return true;
}
Environment GenericEnvironment(const bubo::cli::CatalogueEntry& entry,
                               const GenericPackage& package) {
  Environment env;
  env.generic = true;
  env.entries = package.entries;
  env.package_id = entry.id;
  env.recipe_id = {};
  env.family = entry.architecture;
  env.notices = package.notices;
  // The download pipeline hashes every byte while it receives it. A file that
  // is already on disk is checked in full as well.
  env.fingerprint = [](const Entry& row, const fs::path& path) {
    return HashFile(path) == row.sha256
               ? std::string()
               : std::string("SHA-256 differs from the pin");
  };
  env.full_hash = [](const Entry& row, const fs::path& path,
                     const HashProgress& progress) {
    return HashFile(path, progress) == row.sha256
               ? std::string()
               : std::string("SHA-256 differs from the pin");
  };
  return env;
}
}  // namespace

int RunCatalogueModel(const Options& options,
                      const bubo::cli::CatalogueEntry& entry, std::ostream& out,
                      std::ostream& err, const std::function<bool()>& cancelled,
                      PreparationPlan* preview, bool approved) {
  GenericPackage package;
  if (!BuildGenericPackage(entry, package))
    return Error(err, 4, package.error);
  CurlTransport transport;
  auto env = GenericEnvironment(entry, package);
  env.transport = &transport;
  env.cancelled = cancelled;
  return RunImpl(options, env, std::cin, out, err, preview, approved);
}
#ifdef BUBO_PREPARE_TESTING
int RunModelForTesting(const Options& options, const ModelTestEnvironment& test,
                       std::istream& input, std::ostream& out,
                       std::ostream& err, PreparationPlan* preview,
                       bool approved) {
  Environment env;
  env.entries = test.entries;
  env.package_id = test.package_id;
  env.recipe_id = test.recipe_id;
  for (const auto& notice : test.notices)
    env.notices.push_back({notice.name, notice.text, notice.sha256});
  env.transport = test.transport;
  env.download_options = test.download_options;
  if (test.volume)
    env.volume = test.volume;
  env.fingerprint = test.fingerprint;
  env.cancelled = test.cancelled;
  env.before_manifest = test.before_manifest;
  return RunImpl(options, env, input, out, err, preview, approved);
}
int RunCatalogueModelForTesting(const Options& options,
                                const bubo::cli::CatalogueEntry& entry,
                                const ModelTestEnvironment& test,
                                std::istream& input, std::ostream& out,
                                std::ostream& err, PreparationPlan* preview,
                                bool approved) {
  GenericPackage package;
  if (!BuildGenericPackage(entry, package))
    return Error(err, 4, package.error);
  auto env = GenericEnvironment(entry, package);
  env.transport = test.transport;
  env.download_options = test.download_options;
  if (test.volume)
    env.volume = test.volume;
  if (test.fingerprint) {
    env.fingerprint = test.fingerprint;
    env.full_hash = {};
  }
  env.cancelled = test.cancelled;
  env.before_manifest = test.before_manifest;
  return RunImpl(options, env, input, out, err, preview, approved);
}
#endif
}  // namespace bubo::prepare
