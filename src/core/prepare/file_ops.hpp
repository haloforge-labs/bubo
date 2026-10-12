#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace bubo::prepare {
struct VolumeInfo {
  std::string identity;
  std::uint64_t free_bytes = 0;
  std::string filesystem;
  bool network = false;
  std::string error;
};
// Queries caller-available space on the nearest existing ancestor. Network
// volumes and recall-on-access cloud placeholders are rejected by the caller.
VolumeInfo QueryVolume(const std::filesystem::path& path);
std::string CheckLocalPath(const std::filesystem::path& path);
// Native wide extended paths on Windows, absolute paths elsewhere. Display
// strings should strip the extended prefix separately.
std::filesystem::path NativePath(const std::filesystem::path& path);
struct FileStamp {
  std::uint64_t size = 0;
  std::string modified;
  std::string identity;
  bool operator==(const FileStamp&) const = default;
};
// On Windows this read handle denies writes and deletion until released.
class ReadLease {
public:
  explicit ReadLease(const std::filesystem::path& path);
  ~ReadLease();
  ReadLease(ReadLease&& other) noexcept;
  ReadLease& operator=(ReadLease&& other) noexcept;
  ReadLease(const ReadLease&) = delete;
  ReadLease& operator=(const ReadLease&) = delete;
  FileStamp Snapshot() const;
  // Adopt a verified writer's still-open native handle; no close/reopen gap.
  // The caller transfers ownership and must not close the handle afterwards.
  static ReadLease AdoptVerifiedHandle(std::intptr_t handle);
  // Promote using the verified writer's ORIGINAL handle (Windows DELETE
  // access), while its sharing mode continues to deny writes and deletion.
  bool PromoteTo(const std::filesystem::path& partial,
                 const std::filesystem::path& final, std::string& error);

private:
  ReadLease() = default;
  std::intptr_t handle_ = -1;
};
// A leftover pathname is stale only when the OS permits exclusive reopening.
// The held handle, rather than a PID guess, defines whether a lock is live.
class PrepareLock {
public:
  PrepareLock() = default;
  ~PrepareLock();
  bool Acquire(const std::filesystem::path& path, std::string& error);
  bool busy() const { return busy_; }
  PrepareLock(const PrepareLock&) = delete;
  PrepareLock& operator=(const PrepareLock&) = delete;

private:
  std::intptr_t handle_ = -1;
  std::filesystem::path path_;
  bool busy_ = false;
};
// Same-directory promotion never overwrites a canonical payload. Windows
// sharing violations are retried every 250 ms for at most ten seconds.
bool AtomicPromote(const std::filesystem::path& partial,
                   const std::filesystem::path& final, std::string& error);
bool AtomicWrite(const std::filesystem::path& path, std::string_view text,
                 std::string& error);
}  // namespace bubo::prepare
