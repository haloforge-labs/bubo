#include "src/cli/prepare/model_prepare.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/hash_file.hpp"
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
  std::vector<Notice> notices;
  Transport* transport = nullptr;
  DownloadOptions download_options;
  std::function<VolumeInfo(const fs::path&)> volume = QueryVolume;
  std::function<std::string(const Entry&, const fs::path&)> fingerprint;
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
Value PinnedFile(const Entry& entry) {
  auto result = Value::object();
  result["file"] = std::string(entry.destination);
  result["size"] = static_cast<unsigned long long>(entry.size);
  result["sha256"] = std::string(entry.sha256);
  if (entry.kind != Kind::MainShard)
    result["recipe_id"] = "huihui-ud-q4-k-xl-iq4xs-nofullattn-nohc-q4k-v1";
  if (entry.kind == Kind::Twin || entry.kind == Kind::DraftHead) {
    result["generator_version"] = "1.0.0";
  }
  return result;
}
std::string Manifest(std::span<const Entry> entries) {
  auto result = Value::object();
  result["schema_version"] = 1;
  result["package_id"] = std::string(kPackageId);
  result["engine_min"] = "0.1.0";
  auto model = Value::object(), files = Value::array(),
       artefacts = Value::object();
  model["family"] = "qwen3.8-flash-next";
  auto source = Value::object();
  for (const auto& entry : entries) {
    if (entry.kind == Kind::MainShard) {
      if (source.members().empty()) {
        source["repository"] = std::string(entry.source_repo);
        source["revision"] = std::string(entry.commit);
      }
      files.push_back(PinnedFile(entry));
    } else
      artefacts[std::string(entry.logical_name)] = PinnedFile(entry);
  }
  model["source"] = std::move(source);
  model["files"] = std::move(files);
  result["model"] = std::move(model);
  result["artefacts"] = std::move(artefacts);
  auto profiles = Value::array();
  for (auto profile : {"default", "subagents", "exact", "stock"})
    profiles.push_back(profile);
  result["profiles_supported"] = std::move(profiles);
  return result.dump() + "\n";
}
struct Plan {
  std::vector<std::string> status;
  std::vector<bool> missing;
  std::vector<FileStamp> stamps;
  std::vector<ReadLease> leases;
  std::unique_ptr<ReadLease> manifest_lease;
  std::vector<ConsentVolume> volumes;
  std::uint64_t remaining = 0;
  bool ready = false;
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
int Different(std::ostream& err, const fs::path& path, const std::string& why) {
  return Error(
      err, 7,
      Utf8(path) + ": " + why + "; delete it and run bubo prepare again");
}
std::map<std::string, FileStamp> StoredStamps(const fs::path& state,
                                              std::span<const Entry> entries,
                                              bool& valid) {
  valid = false;
  std::map<std::string, FileStamp> stamps;
  if (!fs::exists(state))
    return stamps;
  try {
    const auto json = metadata::Parse(metadata::Read(state));
    metadata::Keys(json, {"schema_version", "package_id", "files"});
    if (metadata::Integer(metadata::Required(json, "schema_version")) != 1 ||
        metadata::String(metadata::Required(json, "package_id")) != kPackageId)
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
std::string State(std::span<const Entry> entries,
                  const std::vector<FileStamp>& stamps) {
  auto result = Value::object(), files = Value::array();
  result["schema_version"] = 1;
  result["package_id"] = std::string(kPackageId);
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
  const auto stored = StoredStamps(root / ".bubo" / "prepare-state.json",
                                   env.entries, valid_state);
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
  bool missing_notice = false;
  for (const auto& entry : env.entries) {
    if (env.cancelled && env.cancelled())
      return Error(err, 3, "cancelled");
    const auto path = NativePath(root / fs::u8path(entry.destination));
    const bool missing = !fs::exists(path);
    plan.missing.push_back(missing);
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
      const HashProgress cancel_hash = [&](std::uint64_t, std::uint64_t) {
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
      } else {
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
      }
      continue;
    }
    if (const auto code = allocate(path, 0))
      return code;
    plan.leases.emplace_back(path);
    const auto stamp = plan.leases.back().Snapshot();
    if (stamp.size != entry.size)
      return Different(err, path, "size differs from the pin");
    const auto key =
        std::string(entry.destination) + ":" + std::string(entry.sha256);
    const auto previous = stored.find(key);
    if (entry.kind == Kind::MainShard && !options.verify) {
      if (previous == stored.end() || previous->second != stamp) {
        const auto reason = env.fingerprint(entry, path);
        if (!reason.empty())
          return Different(err, path, reason);
      }
    } else {
      HashProgress progress = [&](std::uint64_t, std::uint64_t) {
        if (env.cancelled && env.cancelled())
          throw std::runtime_error("cancelled");
      };
      if (options.verify) {
        out << "prepare verify: " << plan.missing.size() << '/'
            << env.entries.size() << ' ' << entry.destination << '\n';
        const auto start = std::chrono::steady_clock::now();
        progress = [&, start](std::uint64_t done, std::uint64_t total) {
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
        return Different(err, path, "SHA-256 differs from the pin");
    }
    if (plan.leases.back().Snapshot() != stamp)
      return Different(err, path, "file changed during verification");
    plan.stamps.back() = stamp;
  }
  for (const auto& notice : env.notices) {
    const auto path = root / "licenses" / notice.name;
    if (!fs::exists(path)) {
      missing_notice = true;
      if (const auto code = allocate(path, notice.text.size()))
        return code;
    } else {
      plan.leases.emplace_back(path);
      if (HashFile(path) != notice.sha256)
        return Different(err, path, "notice SHA-256 differs from the pin");
    }
  }
  const auto manifest = root / "bubo-package.json";
  bool valid_manifest = false;
  if (fs::exists(manifest)) {
    plan.manifest_lease = std::make_unique<ReadLease>(manifest);
    try {
      if (metadata::Parse(metadata::Read(manifest)).dump() !=
          metadata::Parse(Manifest(env.entries)).dump())
        return Error(err, 2, Utf8(manifest) + ": invalid pinned manifest");
      valid_manifest = true;
    } catch (const std::exception& error) {
      return Error(err, 2, Utf8(manifest) + ": " + error.what());
    }
  }
  const bool any_missing =
      std::ranges::any_of(plan.missing, [](bool value) { return value; });
  plan.ready = !any_missing && !missing_notice && valid_manifest && valid_state;
  if (plan.ready)
    return 0;
  if (options.offline && any_missing) {
    for (std::size_t i = 0; i < env.entries.size(); ++i) {
      if (!plan.missing[i] || !Supplied(options, env.entries[i]).empty())
        continue;
      return Error(err, 4,
                   "offline: missing " +
                       Utf8(root / fs::u8path(env.entries[i].destination)) +
                       " SHA-256 " + std::string(env.entries[i].sha256));
    }
  }
  for (const auto& [name, volume] : allocations) {
    plan.volumes.push_back(volume);
    if (volume.free_bytes < volume.required_bytes)
      return Error(err, 7,
                   "insufficient free space on " + name + ": " +
                       std::to_string(volume.free_bytes) + " bytes free; " +
                       std::to_string(volume.required_bytes) +
                       " bytes required");
  }
  return 0;
}
bool StreamingSpace(const Environment& env, const fs::path& root) {
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
  for (const auto& entry : env.entries) {
    const auto final = NativePath(root / fs::u8path(entry.destination));
    std::uint64_t outstanding = 0;
    if (!fs::exists(final)) {
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
                 std::shared_ptr<ReadLease>& verified) {
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
    if (!StreamingSpace(env, root))
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
void SavePreparedVerification(const fs::path& root,
                              std::span<const Entry> entries,
                              const Plan& plan, bool verify,
                              std::ostream& err) {
  VerifiedStamps rows;
  for (std::size_t i = 0; i < entries.size(); ++i)
    if (verify || plan.missing[i])
      rows[{std::string(entries[i].logical_name),
            std::string(entries[i].sha256)}] = plan.stamps[i];
  std::string error;
  if (!SaveVerifiedStamps(root, rows, error)) {
    std::replace_if(error.begin(), error.end(),
                    [](unsigned char c) { return c < 32 || c == 127; }, ' ');
    err << "verification stamp not saved: " << error << '\n';
  }
}
int RunImpl(const Options& options, const Environment& env, std::istream& input,
            std::ostream& out, std::ostream& err, PreparationPlan* preview,
            bool approved) {
  try {
    // Production approval/text slots are intentionally OPEN. Do not invent
    // licence texts or fetch mutable legal text to get past the release gate.
    if (env.notices.size() != 3)
      return Error(err, 4,
                   "approved pinned model LICENSE, NOTICE and PROVENANCE.md "
                   "bundle is OPEN; no downloads started");
    for (std::size_t i = 0; i < env.notices.size(); ++i) {
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
    if (env.entries.size() != 8)
      return Error(err, 4, "model catalogue must have eight fixed entries");
    const auto invalid = ValidateFields(env.entries, true);
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
      preview->ready = initial.ready;
      preview->volumes = initial.volumes;
      preview->text =
          ModelConsent(env.entries, root, initial.status, initial.remaining,
                       initial.volumes, options.step != Step::Runtime);
      return 0;
    }
    // mutation-anchor: initial-readonly-check
    if (initial.ready) {
      if (options.verify)
        SavePreparedVerification(root, env.entries, initial, true, err);
      out << "prepare: already prepared; nothing to do\n";
      return 0;
    }
    if (!approved) {
      out << ModelConsent(env.entries, root, initial.status, initial.remaining,
                          initial.volumes, options.step != Step::Runtime);
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
        SavePreparedVerification(root, env.entries, plan, true, err);
      out << "prepare: already prepared; nothing to do\n";
      return 0;
    }
    // An existing valid readiness file describes a formerly complete package.
    // Withdraw it before repair, so a crash never advertises incomplete files.
    plan.manifest_lease.reset();
    fs::remove(root / "bubo-package.json");
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
      auto partial = final;
      partial += ".partial";
      auto download = env.download_options;
      download.resume = options.resume;
      download.cancelled = env.cancelled;
      download.check_space = [&] { return StreamingSpace(env, root); };
      download.on_host = [&](std::string_view host) {
        out << "prepare download: " << entry.destination << " host " << host
            << '\n';
      };
      std::uint64_t displayed = 0;
      download.progress = [&](std::uint64_t done, std::uint64_t total) {
        if (done == total || done - std::min(done, displayed) >= 1024 * 1024) {
          out << "prepare download: " << entry.destination << ' ' << done << '/'
              << total << " bytes\n";
          displayed = done;
        }
      };
      const auto source = Supplied(options, entry);
      std::shared_ptr<ReadLease> verified;
      if (!source.empty()) {
        if (const auto code =
                CopySupplied(source, partial, entry, env, root, err, verified))
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
      const auto verified_stamp = verified->Snapshot();
      if (verified_stamp.size != entry.size)
        return Error(err, 6, "verified partial size changed");
      if (!verified->PromoteTo(partial, final, error))
        return Error(err, 7, Utf8(final) + ": " + error);
      auto sidecar = partial;
      sidecar += ".json";
      fs::remove(sidecar);
      plan.leases.push_back(std::move(*verified));
      plan.stamps[i] = plan.leases.back().Snapshot();
      if (plan.stamps[i] != verified_stamp)
        return Different(err, final,
                         "verified file identity changed during promotion");
      // The streaming full hash proves the downloaded bytes match the pin;
      // repeating the fingerprint reads here would add unnecessary model I/O.
    }
    fs::create_directories(root / "licenses");
    for (const auto& notice : env.notices) {
      const auto path = root / "licenses" / notice.name;
      if (!fs::exists(path) && !AtomicWrite(path, notice.text, error))
        return Error(err, 7, error);
      plan.leases.emplace_back(path);
      if (HashFile(path) != notice.sha256)
        return Different(err, path, "notice pin mismatch");
    }
    const auto state_path = root / ".bubo" / "prepare-state.json";
    SavePreparedVerification(root, env.entries, plan, options.verify, err);
    const auto state_text = State(env.entries, plan.stamps);
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
    if (!AtomicWrite(root / "bubo-package.json", Manifest(env.entries), error))
      return Error(err, 7, error);
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
                         bool model_selected) {
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
  Wrap(text, "Read " + Utf8(root / "licenses" / "LICENSE") + " and " +
                 Utf8(root / "licenses" / "NOTICE") + ".");
  Wrap(text, "Read the MaaS and AI Work Assistant conditions in the licence.");
  if (model_selected)
    Wrap(text, "The main shards total " + std::to_string(shards) + " bytes (" +
                   GiB(shards) + " GiB).");
  Wrap(text, "Estimated download time: unknown; download speed is not known.");
  // Owner's later policy requires streaming download verification, superseding
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
  Wrap(text, "Download and prepare all files in this plan? [y/N]");
  return text.str();
}
int RunModel(const Options& options, std::ostream& out, std::ostream& err,
             const std::function<bool()>& cancelled, PreparationPlan* preview,
             bool approved) {
  CurlTransport transport;
  Environment env;
  env.entries = ModelCatalogue();
  for (const auto& notice : ModelNotices())
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
#ifdef BUBO_PREPARE_TESTING
int RunModelForTesting(const Options& options, const ModelTestEnvironment& test,
                       std::istream& input, std::ostream& out,
                       std::ostream& err, PreparationPlan* preview,
                       bool approved) {
  Environment env;
  env.entries = test.entries;
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
#endif
}  // namespace bubo::prepare
