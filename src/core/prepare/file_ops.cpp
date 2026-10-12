#include "src/core/prepare/file_ops.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace bubo::prepare {
namespace {
namespace fs = std::filesystem;
#ifdef _WIN32
std::wstring Wide(const fs::path& path);
#endif
fs::path Ancestor(fs::path path) {
#ifdef _WIN32
  path = fs::path(Wide(path));
#else
  path = fs::absolute(path).lexically_normal();
#endif
  std::error_code error;
  while (!fs::exists(path, error)) {
    if (error)
      throw std::runtime_error(error.message());
#ifdef _WIN32
    const auto parent = fs::path(Wide(path.parent_path()));
#else
    const auto parent = path.parent_path();
#endif
    if (parent == path || parent.empty())
      throw std::runtime_error("no existing ancestor");
    path = parent;
  }
  return path;
}
#ifdef _WIN32
std::wstring Wide(const fs::path& path) {
  // Normalize DOS/UNC spelling before adding the native prefix. MSVC's
  // lexical normalization can strip the trailing slash from \\?\C:\.
  auto value = path.wstring();
  auto prefix = value.substr(0, 8);
  for (auto& character : prefix) {
    if (character >= L'a' && character <= L'z')
      character -= L'a' - L'A';
  }
  if (prefix == L"\\\\?\\UNC\\")
    value = L"\\\\" + value.substr(8);
  else if (value.starts_with(L"\\\\?\\"))
    value = value.substr(4);
  if (value.size() == 2 && value[1] == L':')
    value += L'\\';
  auto normalized = fs::absolute(fs::path(value)).lexically_normal();
  normalized.make_preferred();
  value = normalized.wstring();
  if (value.starts_with(L"\\\\"))
    return L"\\\\?\\UNC\\" + value.substr(2);
  return L"\\\\?\\" + value;
}
HANDLE Handle(std::intptr_t handle) {
  return reinterpret_cast<HANDLE>(handle);
}
std::string WinError(DWORD code) {
  return "Windows error " + std::to_string(code);
}
void Close(std::intptr_t& handle) {
  if (handle != -1)
    CloseHandle(Handle(std::exchange(handle, -1)));
}
bool Rename(const fs::path& from, const fs::path& to, bool replace,
            std::string& error) {
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  do {
    if (MoveFileExW(
            Wide(from).c_str(), Wide(to).c_str(),
            MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0)))
      return true;
    const auto code = GetLastError();
    if ((code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION) ||
        std::chrono::steady_clock::now() >= until) {
      error = WinError(code);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  } while (true);
}
#else
void Close(std::intptr_t& handle) {
  if (handle != -1)
    close(static_cast<int>(std::exchange(handle, -1)));
}
bool Rename(const fs::path& from, const fs::path& to, bool replace,
            std::string& error) {
  std::error_code code;
  if (!replace) {
    // link/unlink gives no-overwrite publication on the same volume.
    if (link(from.c_str(), to.c_str()) != 0) {
      error = "cannot publish canonical file";
      return false;
    }
    fs::remove(from, code);
  } else
    fs::rename(from, to, code);
  if (code)
    error = code.message();
  return !code;
}
#endif
}  // namespace

fs::path NativePath(const fs::path& path) {
#ifdef _WIN32
  return fs::path(Wide(path));
#else
  return fs::absolute(path).lexically_normal();
#endif
}
std::string CheckLocalPath(const fs::path& path) {
  try {
#ifdef _WIN32
    // Reject network paths before any existence/attribute query can cause an
    // SMB connection or DNS lookup, including for --offline preparation.
    const auto native = Wide(path);  // Pure absolute lexical normalization.
    auto prefix = native.substr(0, 8);
    for (auto& character : prefix) {
      if (character >= L'a' && character <= L'z')
        character -= L'a' - L'A';
    }
    if (prefix == L"\\\\?\\UNC\\")
      return "UNC and network volumes are not supported; use a local folder "
             "with --output";
    std::wstring root;
    if (native.size() >= 7 && native[5] == L':' && native[6] == L'\\')
      root = native.substr(4, 3);
    else
      root = fs::path(native).root_path().wstring();
    if (!root.empty() && GetDriveTypeW(root.c_str()) == DRIVE_REMOTE)
      return "UNC and network volumes are not supported; use a local folder "
             "with --output";
#endif
    auto current = Ancestor(path);
#ifdef _WIN32
    for (;;) {
      const auto normalized = Wide(current);
      const auto attributes = GetFileAttributesW(normalized.c_str());
      if (attributes == INVALID_FILE_ATTRIBUTES) {
        const auto code = GetLastError();
        const auto utf8 = fs::path(normalized).u8string();
        return "GetFileAttributesW " + std::string(utf8.begin(), utf8.end()) +
               ": " + WinError(code);
      }
      constexpr DWORD recall =
          0x00040000 | 0x00400000;  // recall-on-open/data-access
      if (attributes & (FILE_ATTRIBUTE_OFFLINE | recall))
        return "Use a local folder. Select it with --output, --runtime-dir or "
               "--cache-dir.";
      // Do not turn an extended drive root (\\?\C:\) into the drive-relative
      // spelling \\?\C: when walking parents; GetFileAttributes rejects it.
      const bool drive_root = normalized.size() == 7 && normalized[5] == L':' &&
                              normalized[6] == L'\\';
      if (drive_root || current == current.root_path())
        break;
      const auto parent = current.parent_path();
      if (parent.empty() || parent == current)
        break;
      current = parent;
    }
#endif
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}
VolumeInfo QueryVolume(const fs::path& path) {
  VolumeInfo result;
  try {
    result.error = CheckLocalPath(path);
    if (!result.error.empty())
      return result;
    const auto parent = Ancestor(path);
#ifdef _WIN32
    wchar_t volume[MAX_PATH + 1]{};
    if (!GetVolumePathNameW(Wide(parent).c_str(), volume, MAX_PATH))
      throw std::runtime_error(WinError(GetLastError()));
    wchar_t filesystem[MAX_PATH + 1]{};
    if (!GetVolumeInformationW(volume, nullptr, 0, nullptr, nullptr, nullptr,
                               filesystem, MAX_PATH))
      throw std::runtime_error(WinError(GetLastError()));
    ULARGE_INTEGER available{};
    if (!GetDiskFreeSpaceExW(volume, &available, nullptr, nullptr))
      throw std::runtime_error(WinError(GetLastError()));
    result.free_bytes = available.QuadPart;
    const std::wstring type(filesystem);
    result.filesystem.assign(type.begin(), type.end());
    const std::wstring root(volume);
    result.network =
        GetDriveTypeW(volume) == DRIVE_REMOTE ||
        root.starts_with(L"\\\\?\\UNC\\") ||
        (root.starts_with(L"\\\\") && !root.starts_with(L"\\\\?\\"));
    wchar_t guid[MAX_PATH + 1]{};
    if (GetVolumeNameForVolumeMountPointW(volume, guid, MAX_PATH)) {
      const std::wstring name(guid);
      result.identity.assign(name.begin(), name.end());
    } else if (result.network) {
      // UNC has no local volume GUID; it is still classified and rejected by
      // model preflight before any allocation or payload work.
      result.identity = "network:" + fs::path(root).generic_string();
    } else {
      throw std::runtime_error(WinError(GetLastError()));
    }
#else
    const auto info = fs::space(parent);
    result.free_bytes = info.available;
    struct stat stat{};
    if (::stat(parent.c_str(), &stat))
      throw std::runtime_error("cannot query volume");
    result.identity = std::to_string(stat.st_dev);
    result.filesystem = "local";
#endif
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

ReadLease::ReadLease(const fs::path& path) {
  const auto error = CheckLocalPath(path);
  if (!error.empty())
    throw std::runtime_error(error);
#ifdef _WIN32
  const auto handle =
      CreateFileW(Wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    throw std::runtime_error(WinError(GetLastError()));
  handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  handle_ = open(path.c_str(), O_RDONLY);
  if (handle_ == -1 || flock(static_cast<int>(handle_), LOCK_SH | LOCK_NB)) {
    Close(handle_);
    throw std::runtime_error("cannot lease file");
  }
#endif
}
ReadLease::~ReadLease() {
  Close(handle_);
}
ReadLease ReadLease::AdoptVerifiedHandle(std::intptr_t handle) {
  if (handle == -1)
    throw std::runtime_error("invalid verified file handle");
  ReadLease lease;
  lease.handle_ = handle;
  return lease;
}
bool ReadLease::PromoteTo(const fs::path& partial, const fs::path& final,
                          std::string& error) {
  if (NativePath(partial).parent_path() != NativePath(final).parent_path()) {
    error = "promotion must stay in the same directory";
    return false;
  }
#ifdef _WIN32
  const auto name = Wide(final);
  const auto bytes = name.size() * sizeof(wchar_t);
  // Win32 may inspect a terminator while converting the supplied filename;
  // its counted UTF-16 length excludes this zero-initialized final wchar_t.
  std::vector<std::uint8_t> buffer(offsetof(FILE_RENAME_INFO, FileName) +
                                   bytes + sizeof(wchar_t));
  auto* info = reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());
  info->ReplaceIfExists = FALSE;
  info->RootDirectory = nullptr;
  info->FileNameLength = static_cast<DWORD>(bytes);
  std::memcpy(info->FileName, name.data(), bytes);
  if (!FlushFileBuffers(Handle(handle_))) {
    error = WinError(GetLastError());
    return false;
  }
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  do {
    if (SetFileInformationByHandle(Handle(handle_), FileRenameInfo, info,
                                   static_cast<DWORD>(buffer.size()))) {
      if (!FlushFileBuffers(Handle(handle_))) {
        error = WinError(GetLastError());
        return false;
      }
      return true;
    }
    const auto code = GetLastError();
    if ((code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION) ||
        std::chrono::steady_clock::now() >= until) {
      error = "SetFileInformationByHandle(FileRenameInfo): " + WinError(code);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  } while (true);
#else
  return Rename(partial, final, false, error);
#endif
}
ReadLease::ReadLease(ReadLease&& other) noexcept
    : handle_(std::exchange(other.handle_, -1)) {}
ReadLease& ReadLease::operator=(ReadLease&& other) noexcept {
  Close(handle_);
  handle_ = std::exchange(other.handle_, -1);
  return *this;
}
FileStamp ReadLease::Snapshot() const {
  FileStamp result;
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(Handle(handle_), &info))
    throw std::runtime_error(WinError(GetLastError()));
  result.size = (std::uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
  result.modified = std::to_string(
      (std::uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
      info.ftLastWriteTime.dwLowDateTime);
  result.identity = std::to_string(info.dwVolumeSerialNumber) + ":" +
                    std::to_string((std::uint64_t(info.nFileIndexHigh) << 32) |
                                   info.nFileIndexLow);
#else
  struct stat info{};
  if (fstat(static_cast<int>(handle_), &info))
    throw std::runtime_error("cannot stamp file");
  result.size = info.st_size;
  result.modified = std::to_string(info.st_mtim.tv_sec) + ":" +
                    std::to_string(info.st_mtim.tv_nsec);
  result.identity =
      std::to_string(info.st_dev) + ":" + std::to_string(info.st_ino);
#endif
  return result;
}

PrepareLock::~PrepareLock() {
  // Keep the pathname on both platforms. Exclusive reopening defines stale
  // ownership; unlinking can race another opener (POSIX), and delete-on-close
  // can defer a directory mtime update past readiness publication (Windows).
  Close(handle_);
}
bool PrepareLock::Acquire(const fs::path& path, std::string& error) {
  if (handle_ != -1) {
    error = "lock already held";
    return false;
  }
#ifdef _WIN32
  auto handle =
      CreateFileW(Wide(path).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                  CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_EXISTS)
    handle =
        CreateFileW(Wide(path).c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const auto code = GetLastError();
    busy_ = code == ERROR_SHARING_VIOLATION || code == ERROR_LOCK_VIOLATION;
    error = WinError(code);
    return false;
  }
  handle_ = reinterpret_cast<std::intptr_t>(handle);
  FILETIME creation{}, exit{}, kernel{}, user{};
  GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
  const auto text =
      "pid=" + std::to_string(GetCurrentProcessId()) + " creation=" +
      std::to_string((std::uint64_t(creation.dwHighDateTime) << 32) |
                     creation.dwLowDateTime) +
      "\n";
  DWORD count = 0;
  // Disable automatic close-time timestamp writes for this diagnostic file.
  // Its explicit write time is set during acquisition, before the manifest.
  FILETIME frozen{0xffffffff, 0xffffffff}, actual{};
  GetSystemTimeAsFileTime(&actual);
  if (!SetFileTime(handle, nullptr, nullptr, &frozen) ||
      !SetFilePointerEx(handle, {}, nullptr, FILE_BEGIN) ||
      !SetEndOfFile(handle) ||
      !WriteFile(handle, text.data(), static_cast<DWORD>(text.size()), &count,
                 nullptr) ||
      count != text.size() || !FlushFileBuffers(handle) ||
      !SetFileTime(handle, nullptr, nullptr, &actual) ||
      !SetFileTime(handle, nullptr, nullptr, &frozen)) {
    error = WinError(GetLastError());
    Close(handle_);
    return false;
  }
#else
  handle_ = open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (handle_ == -1 || flock(static_cast<int>(handle_), LOCK_EX | LOCK_NB)) {
    busy_ = errno == EWOULDBLOCK || errno == EAGAIN;
    error = "live prepare lock or inaccessible lock";
    Close(handle_);
    return false;
  }
#endif
  path_ = path;
  return true;
}
bool AtomicPromote(const fs::path& partial, const fs::path& final,
                   std::string& error) {
  if (partial.parent_path() != final.parent_path()) {
    error = "promotion must stay in the same directory";
    return false;
  }
  return Rename(partial, final, false, error);
}
bool AtomicWrite(const fs::path& path, std::string_view text,
                 std::string& error) {
  if (text.size() > 1024 * 1024) {
    error = "metadata exceeds 1 MiB";
    return false;
  }
  auto temporary = path;
  temporary += ".new";
  // The caller holds the prepare lock; no foreign process shares this scratch.
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    if (!stream) {
      error = "cannot write metadata";
      return false;
    }
  }
#ifdef _WIN32
  auto handle = CreateFileW(Wide(temporary).c_str(), GENERIC_WRITE, 0, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = WinError(GetLastError());
    return false;
  }
  const bool flushed = FlushFileBuffers(handle) != 0;
  const auto code = GetLastError();
  CloseHandle(handle);
  if (!flushed) {
    error = WinError(code);
    return false;
  }
#endif
  return Rename(temporary, path, true, error);
}
}  // namespace bubo::prepare
