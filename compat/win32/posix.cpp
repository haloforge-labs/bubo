// Win32 implementation of the POSIX subset declared in include/gufo_posix.h.
//
// File descriptors are UCRT descriptors wrapping Win32 handles, so `int fd`
// code keeps working, but every read, write and seek goes straight to the
// handle rather than through the CRT's text-mode machinery.
//
// Semantics that differ from Linux and matter to Gufo:
// - O_DIRECT opens with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED. As on
//   Linux, offsets, lengths and buffers must be sector aligned; the overlapped
//   handle lets many threads pread() one descriptor concurrently.
// - O_DIRECT reads of a file that is also mapped are serialized by NTFS:
//   32 threads doing 4 KiB reads fall from ~70K to ~5K reads/s (measured on
//   a Kingston NVMe with a model shard mapped). O_CONCURRENT_RANDOM (Windows
//   only) opens a cached overlapped handle instead, which keeps full speed.
// - open("/proc/self/fd/N") reopens descriptor N with ReOpenFile, which is the
//   exact Win32 equivalent of Gufo's "independent file description for the
//   same inode" idiom.
// - flock() locks one byte far beyond any real file offset, because Windows
//   byte-range locks are mandatory and would otherwise block ordinary reads.
// - mmap() offsets are aligned down to the 64 KiB allocation granularity and
//   munmap() releases whole views.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#include <winternl.h>

#include <crtdbg.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <random>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "gufo_socket.h"

namespace {

// UCRT raises its invalid-parameter handler (which terminates by default) for
// a bad descriptor. POSIX code expects EBADF instead.
void IgnoreInvalidParameter(const wchar_t*, const wchar_t*, const wchar_t*,
                            unsigned int, uintptr_t) {}

// On Linux an uncaught exception prints its what() before aborting; the MSVC
// runtime exits silently with 0xC0000409. Keep the Linux behavior.
[[noreturn]] void ReportTermination() {
  if (const auto error = std::current_exception()) {
    try {
      std::rethrow_exception(error);
    } catch (const std::exception& e) {
      fprintf(stderr, "terminate called after throwing an instance of '%s'\n"
                      "  what():  %s\n", typeid(e).name(), e.what());
    } catch (...) {
      fprintf(stderr, "terminate called after throwing a non-standard "
                      "exception\n");
    }
  } else {
    fprintf(stderr, "terminate called without an active exception\n");
  }
  fflush(stderr);
  abort();
}

// A crash on Linux leaves a core and a signal name; on Windows the process
// just vanishes. Print the exception and the faulting module offset, which
// llvm-symbolizer resolves against the build's PDB.
// An uncaught C++ exception reaches the unhandled-exception filter before
// std::terminate. Find the std::exception base in the MSVC throw metadata
// (x64: RVAs from the thrower's image base) so its what() is still printed.
const std::exception* ThrownStdException(const EXCEPTION_RECORD* record) {
  constexpr DWORD kMsvcCxxException = 0xE06D7363;
  if (record->ExceptionCode != kMsvcCxxException ||
      record->NumberParameters < 4) {
    return nullptr;
  }
  struct ThrowInfo {
    unsigned attributes;
    int unwind, forward_compat, catchable_types;
  };
  struct CatchableType {
    unsigned properties;
    int type;
    int mdisp, pdisp, vdisp;
    int size, copy;
  };
  const auto object = record->ExceptionInformation[1];
  const auto* throw_info =
      reinterpret_cast<const ThrowInfo*>(record->ExceptionInformation[2]);
  const auto base = record->ExceptionInformation[3];
  if (object == 0 || throw_info == nullptr || base == 0) {
    return nullptr;
  }
  const auto* types =
      reinterpret_cast<const int*>(base + throw_info->catchable_types);
  for (int i = 0; i < types[0]; ++i) {
    const auto* type =
        reinterpret_cast<const CatchableType*>(base + types[1 + i]);
    const auto* name =
        reinterpret_cast<const char*>(base + type->type) + 2 * sizeof(void*);
    if (strcmp(name, ".?AVexception@std@@") == 0) {
      return reinterpret_cast<const std::exception*>(object + type->mdisp);
    }
  }
  return nullptr;
}

LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
  const auto* record = info->ExceptionRecord;
  if (const auto* error = ThrownStdException(record)) {
    fprintf(stderr,
            "terminate called after throwing an instance of '%s'\n"
            "  what():  %s\n",
            typeid(*error).name(), error->what());
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
  }
  HMODULE module = nullptr;
  char name[MAX_PATH] = "?";
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         static_cast<LPCSTR>(record->ExceptionAddress),
                         &module)) {
    GetModuleFileNameA(module, name, sizeof(name));
  }
  const auto offset = reinterpret_cast<uintptr_t>(record->ExceptionAddress) -
                      reinterpret_cast<uintptr_t>(module);
  fprintf(stderr, "fatal exception 0x%08lX at %s+0x%llX",
          static_cast<unsigned long>(record->ExceptionCode), name,
          static_cast<unsigned long long>(offset));
  if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
      record->NumberParameters >= 2) {
    fprintf(stderr, " (%s 0x%llX)",
            record->ExceptionInformation[0] == 0   ? "reading"
            : record->ExceptionInformation[0] == 1 ? "writing"
                                                   : "executing",
            static_cast<unsigned long long>(record->ExceptionInformation[1]));
  }
  fprintf(stderr, "\n");
  fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;
}

struct ProcessSetup {
  ProcessSetup() {
    std::set_terminate(ReportTermination);
    SetUnhandledExceptionFilter(ReportCrash);
    _set_invalid_parameter_handler(IgnoreInvalidParameter);
    _CrtSetReportMode(_CRT_ASSERT, 0);
    WSADATA data;
    (void)WSAStartup(MAKEWORD(2, 2), &data);
    // UTF-8 console output; Gufo prints UTF-8 model text.
    SetConsoleOutputCP(CP_UTF8);
    // Interpret the ANSI colors Gufo's logger writes, as Linux terminals do.
    for (const DWORD which : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
      const HANDLE console = GetStdHandle(which);
      DWORD mode = 0;
      if (console != INVALID_HANDLE_VALUE && GetConsoleMode(console, &mode)) {
        SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      }
    }
    for (int fd = 0; fd <= 2; ++fd) {
      (void)_setmode(fd, _O_BINARY);
    }
  }
};
const ProcessSetup process_setup;

int ErrnoFromWin32(DWORD error) {
  switch (error) {
    case ERROR_SUCCESS:
      return 0;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_DRIVE:
    case ERROR_BAD_NETPATH:
    case ERROR_BAD_PATHNAME:
    case ERROR_INVALID_NAME:
      return ENOENT;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
    case ERROR_WRITE_PROTECT:
    case ERROR_PRIVILEGE_NOT_HELD:
      return EACCES;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      return EEXIST;
    case ERROR_DIR_NOT_EMPTY:
      return ENOTEMPTY;
    case ERROR_DIRECTORY:
      return ENOTDIR;
    case ERROR_INVALID_HANDLE:
      return EBADF;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_COMMITMENT_LIMIT:
      return ENOMEM;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
      return ENOSPC;
    case ERROR_TOO_MANY_OPEN_FILES:
      return EMFILE;
    case ERROR_BROKEN_PIPE:
    case ERROR_NO_DATA:
      return EPIPE;
    case ERROR_NOT_SUPPORTED:
    case ERROR_CALL_NOT_IMPLEMENTED:
      return ENOSYS;
    case ERROR_OPERATION_ABORTED:
      return EINTR;
    case ERROR_CANT_RESOLVE_FILENAME:
      return ELOOP;
    case ERROR_FILENAME_EXCED_RANGE:
      return ENAMETOOLONG;
    default:
      return EINVAL;
  }
}

int Fail(DWORD error) {
  errno = ErrnoFromWin32(error);
  return -1;
}

std::wstring Widen(const char* utf8) {
  if (utf8 == nullptr || *utf8 == '\0') {
    return {};
  }
  const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
  std::wstring out(n > 0 ? n - 1 : 0, L'\0');
  if (n > 0) {
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out.data(), n);
  }
  for (auto& c : out) {
    if (c == L'/') {
      c = L'\\';
    }
  }
  return out;
}

std::wstring Backslashed(const wchar_t* wide) {
  std::wstring out = wide != nullptr ? wide : L"";
  for (auto& c : out) {
    if (c == L'/') {
      c = L'\\';
    }
  }
  return out;
}

std::string Narrow(const wchar_t* wide) {
  const int n =
      WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
  std::string out(n > 0 ? n - 1 : 0, '\0');
  if (n > 0) {
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), n, nullptr,
                        nullptr);
  }
  return out;
}

HANDLE HandleOf(int fd) {
  if (fd < 0) {
    return INVALID_HANDLE_VALUE;
  }
  return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}

// Descriptors opened with O_DIRECT own an overlapped handle.
std::mutex g_fd_mutex;
std::unordered_map<int, int> g_fd_flags;  // fd -> open flags (O_DIRECT, ...)

void RememberFlags(int fd, int flags) {
  std::lock_guard lock(g_fd_mutex);
  g_fd_flags[fd] = flags;
}

int FlagsOf(int fd) {
  std::lock_guard lock(g_fd_mutex);
  const auto it = g_fd_flags.find(fd);
  return it == g_fd_flags.end() ? 0 : it->second;
}

void ForgetFlags(int fd) {
  std::lock_guard lock(g_fd_mutex);
  g_fd_flags.erase(fd);
}

int WrapHandle(HANDLE handle, int flags) {
  int crt = _O_BINARY | _O_NOINHERIT;
  if ((flags & O_ACCMODE) == O_RDONLY) {
    crt |= _O_RDONLY;
  } else if ((flags & O_ACCMODE) == O_WRONLY) {
    crt |= _O_WRONLY;
  } else {
    crt |= _O_RDWR;
  }
  if (flags & O_APPEND) {
    crt |= _O_APPEND;
  }
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), crt);
  if (fd < 0) {
    CloseHandle(handle);
    errno = EMFILE;
    return -1;
  }
  RememberFlags(fd, flags);
  return fd;
}

DWORD AccessFor(int flags) {
  DWORD access = 0;
  switch (flags & O_ACCMODE) {
    case O_RDONLY:
      access = GENERIC_READ;
      break;
    case O_WRONLY:
      access = GENERIC_WRITE;
      break;
    default:
      access = GENERIC_READ | GENERIC_WRITE;
      break;
  }
  if (flags & O_APPEND) {
    // Append-only data access makes every write land at end of file.
    access &= ~static_cast<DWORD>(GENERIC_WRITE);
    access |= FILE_APPEND_DATA | FILE_WRITE_ATTRIBUTES | SYNCHRONIZE;
  }
  return access | FILE_READ_ATTRIBUTES;
}

DWORD AttributesFor(int flags) {
  DWORD attributes = FILE_ATTRIBUTE_NORMAL;
  if (flags & O_DIRECTORY) {
    attributes |= FILE_FLAG_BACKUP_SEMANTICS;
  }
  if (flags & O_NOFOLLOW) {
    attributes |= FILE_FLAG_OPEN_REPARSE_POINT;
  }
  if (flags & O_DIRECT) {
    attributes |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
  }
  if (flags & O_CONCURRENT_RANDOM) {
    attributes |= FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED;
  }
  if (flags & O_SYNC) {
    attributes |= FILE_FLAG_WRITE_THROUGH;
  }
  return attributes;
}

int ReopenDescriptor(int source, int flags) {
  const HANDLE handle = HandleOf(source);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  const HANDLE reopened =
      ReOpenFile(handle, AccessFor(flags) & ~FILE_READ_ATTRIBUTES,
                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                 AttributesFor(flags) & ~FILE_ATTRIBUTE_NORMAL);
  if (reopened == INVALID_HANDLE_VALUE) {
    return Fail(GetLastError());
  }
  return WrapHandle(reopened, flags);
}

bool IsDirectoryHandle(HANDLE handle) {
  BY_HANDLE_FILE_INFORMATION info{};
  return GetFileInformationByHandle(handle, &info) &&
         (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

int OpenPath(const std::wstring& path, int flags, mode_t mode) {
  DWORD disposition = OPEN_EXISTING;
  if (flags & O_CREAT) {
    if (flags & O_EXCL) {
      disposition = CREATE_NEW;
    } else if (flags & O_TRUNC) {
      disposition = CREATE_ALWAYS;
    } else {
      disposition = OPEN_ALWAYS;
    }
  } else if (flags & O_TRUNC) {
    disposition = TRUNCATE_EXISTING;
  }
  DWORD attributes = AttributesFor(flags);
  if ((flags & O_CREAT) && (mode & S_IWUSR) == 0 && mode != 0) {
    attributes |= FILE_ATTRIBUTE_READONLY;
  }
  SECURITY_ATTRIBUTES security{sizeof(security), nullptr,
                               (flags & O_CLOEXEC) == 0 ? FALSE : FALSE};
  const HANDLE handle = CreateFileW(
      path.c_str(), AccessFor(flags),
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &security,
      disposition, attributes, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    if (error == ERROR_ACCESS_DENIED && !(flags & O_DIRECTORY)) {
      // Opening a directory without O_DIRECTORY is EISDIR on Linux.
      const DWORD found = GetFileAttributesW(path.c_str());
      if (found != INVALID_FILE_ATTRIBUTES &&
          (found & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        errno = EISDIR;
        return -1;
      }
    }
    return Fail(error);
  }
  if (flags & O_DIRECTORY) {
    if (!IsDirectoryHandle(handle)) {
      CloseHandle(handle);
      errno = ENOTDIR;
      return -1;
    }
  }
  if (flags & O_NOFOLLOW) {
    FILE_ATTRIBUTE_TAG_INFO tag{};
    if (GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &tag,
                                     sizeof(tag)) &&
        (tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      CloseHandle(handle);
      errno = ELOOP;
      return -1;
    }
  }
  return WrapHandle(handle, flags);
}

std::wstring DirectoryPathOf(int dirfd) {
  if (dirfd == AT_FDCWD) {
    return {};
  }
  const HANDLE handle = HandleOf(dirfd);
  wchar_t buffer[32768];
  const DWORD n = GetFinalPathNameByHandleW(handle, buffer, 32768,
                                            FILE_NAME_NORMALIZED);
  if (n == 0 || n >= 32768) {
    return {};
  }
  return std::wstring(buffer, n);
}

std::wstring ResolveAt(int dirfd, const char* path) {
  std::wstring relative = Widen(path);
  const bool absolute =
      relative.size() >= 2 && (relative[1] == L':' || relative[0] == L'\\');
  if (dirfd == AT_FDCWD || absolute) {
    return relative;
  }
  std::wstring base = DirectoryPathOf(dirfd);
  if (base.empty()) {
    return {};
  }
  if (base.back() != L'\\') {
    base += L'\\';
  }
  return base + relative;
}

struct timespec TimespecFromFiletime(const FILETIME& ft) {
  ULARGE_INTEGER value;
  value.LowPart = ft.dwLowDateTime;
  value.HighPart = ft.dwHighDateTime;
  constexpr unsigned long long kEpochDelta = 116444736000000000ULL;
  const unsigned long long ticks =
      value.QuadPart > kEpochDelta ? value.QuadPart - kEpochDelta : 0;
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(ticks / 10000000ULL);
  ts.tv_nsec = static_cast<long>((ticks % 10000000ULL) * 100ULL);
  return ts;
}

int StatHandle(HANDLE handle, struct stat* st, bool no_follow) {
  memset(st, 0, sizeof(*st));
  const DWORD type = GetFileType(handle);
  if (type == FILE_TYPE_CHAR) {
    st->st_mode = S_IFCHR | 0600;
    return 0;
  }
  if (type == FILE_TYPE_PIPE) {
    st->st_mode = S_IFIFO | 0600;
    return 0;
  }
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle, &info)) {
    return Fail(GetLastError());
  }
  const bool directory = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  bool link = false;
  if (no_follow && (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
    FILE_ATTRIBUTE_TAG_INFO tag{};
    link = GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &tag,
                                        sizeof(tag)) &&
           (tag.ReparseTag == IO_REPARSE_TAG_SYMLINK ||
            tag.ReparseTag == IO_REPARSE_TAG_MOUNT_POINT);
  }
  mode_t permissions = directory ? 0700 : 0600;
  if (info.dwFileAttributes & FILE_ATTRIBUTE_READONLY) {
    permissions &= ~static_cast<mode_t>(0200);
  }
  st->st_mode =
      (link ? S_IFLNK : (directory ? S_IFDIR : S_IFREG)) | permissions;
  st->st_dev = info.dwVolumeSerialNumber;
  st->st_ino = (static_cast<unsigned long long>(info.nFileIndexHigh) << 32) |
               info.nFileIndexLow;
  st->st_nlink = info.nNumberOfLinks;
  st->st_size = static_cast<off_t>(
      (static_cast<unsigned long long>(info.nFileSizeHigh) << 32) |
      info.nFileSizeLow);
  st->st_blksize = 4096;
  st->st_blocks = (st->st_size + 511) / 512;
  st->st_atim = TimespecFromFiletime(info.ftLastAccessTime);
  st->st_mtim = TimespecFromFiletime(info.ftLastWriteTime);
  // st_ctime is the inode change time, which NTFS tracks as ChangeTime; it
  // moves on every write even when the modification time is restored.
  FILE_BASIC_INFO basic{};
  if (GetFileInformationByHandleEx(handle, FileBasicInfo, &basic,
                                   sizeof(basic))) {
    FILETIME change;
    change.dwLowDateTime = static_cast<DWORD>(basic.ChangeTime.QuadPart);
    change.dwHighDateTime = static_cast<DWORD>(basic.ChangeTime.QuadPart >> 32);
    st->st_ctim = TimespecFromFiletime(change);
  } else {
    st->st_ctim = st->st_mtim;
  }
  return 0;
}

int StatPath(const std::wstring& path, struct stat* st, bool no_follow) {
  DWORD attributes = FILE_FLAG_BACKUP_SEMANTICS;
  if (no_follow) {
    attributes |= FILE_FLAG_OPEN_REPARSE_POINT;
  }
  const HANDLE handle =
      CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, attributes, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Fail(GetLastError());
  }
  const int result = StatHandle(handle, st, no_follow);
  CloseHandle(handle);
  return result;
}

HANDLE ThreadEvent() {
  struct Holder {
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~Holder() {
      if (event != nullptr) {
        CloseHandle(event);
      }
    }
  };
  thread_local Holder holder;
  return holder.event;
}

// flock() bit kept in g_fd_flags: Windows releases byte-range locks
// asynchronously after CloseHandle, so close() unlocks explicitly first.
constexpr int kFdLocked = 0x40000000;
constexpr DWORD kLockOffsetLow = 0xFFFFFFFEu;
constexpr DWORD kLockOffsetHigh = 0x7FFFFFFFu;

// Positioned transfer that works on both synchronous and overlapped handles.
ssize_t Transfer(int fd, void* buffer, size_t count, off_t offset,
                 bool writing) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  const DWORD chunk =
      static_cast<DWORD>(count > (1U << 30) ? (1U << 30) : count);
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(static_cast<unsigned long long>(offset));
  overlapped.OffsetHigh =
      static_cast<DWORD>(static_cast<unsigned long long>(offset) >> 32);
  overlapped.hEvent = ThreadEvent();
  ResetEvent(overlapped.hEvent);
  DWORD done = 0;
  const BOOL ok =
      writing ? WriteFile(handle, buffer, chunk, &done, &overlapped)
              : ReadFile(handle, buffer, chunk, &done, &overlapped);
  if (!ok) {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      if (GetOverlappedResult(handle, &overlapped, &done, TRUE)) {
        return static_cast<ssize_t>(done);
      }
      error = GetLastError();
    }
    if (error == ERROR_HANDLE_EOF || error == ERROR_BROKEN_PIPE) {
      return 0;
    }
    return Fail(error);
  }
  return static_cast<ssize_t>(done);
}

// Current-position transfer for synchronous handles (files, pipes, console).
ssize_t Stream(int fd, void* buffer, size_t count, bool writing) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  const DWORD chunk =
      static_cast<DWORD>(count > (1U << 30) ? (1U << 30) : count);
  DWORD done = 0;
  const BOOL ok = writing ? WriteFile(handle, buffer, chunk, &done, nullptr)
                          : ReadFile(handle, buffer, chunk, &done, nullptr);
  if (!ok) {
    const DWORD error = GetLastError();
    if (!writing &&
        (error == ERROR_BROKEN_PIPE || error == ERROR_HANDLE_EOF)) {
      return 0;
    }
    return Fail(error);
  }
  return static_cast<ssize_t>(done);
}

SYSTEM_INFO SystemInfo() {
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info;
}

// Child processes started by the posix_spawn shim, so waitpid() can find a
// process that has already exited.
std::mutex g_child_mutex;
std::unordered_map<pid_t, HANDLE> g_children;

}  // namespace

extern "C" {

// Used by compat/win32/spawn.cpp.
void gufo_register_child(pid_t pid, void* process) {
  std::lock_guard lock(g_child_mutex);
  g_children[pid] = static_cast<HANDLE>(process);
}

int open(const char* path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args);
  }
  if (path == nullptr) {
    errno = EFAULT;
    return -1;
  }
  constexpr char kProcFd[] = "/proc/self/fd/";
  if (strncmp(path, kProcFd, sizeof(kProcFd) - 1) == 0) {
    return ReopenDescriptor(atoi(path + sizeof(kProcFd) - 1), flags);
  }
  return OpenPath(Widen(path), flags, mode);
}

int openat(int dirfd, const char* path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, int);
    va_end(args);
  }
  const std::wstring resolved = ResolveAt(dirfd, path);
  if (resolved.empty()) {
    errno = EBADF;
    return -1;
  }
  return OpenPath(resolved, flags, mode);
}

int close(int fd) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  if (FlagsOf(fd) & kFdLocked) {
    OVERLAPPED overlapped{};
    overlapped.Offset = kLockOffsetLow;
    overlapped.OffsetHigh = kLockOffsetHigh;
    (void)UnlockFileEx(handle, 0, 1, 0, &overlapped);
  }
  ForgetFlags(fd);
  return _close(fd);
}

ssize_t read(int fd, void* buf, size_t count) {
  return Stream(fd, buf, count, false);
}

ssize_t write(int fd, const void* buf, size_t count) {
  return Stream(fd, const_cast<void*>(buf), count, true);
}

ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  return Transfer(fd, buf, count, offset, false);
}

ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset) {
  return Transfer(fd, const_cast<void*>(buf), count, offset, true);
}

off_t lseek(int fd, off_t offset, int whence) {
  const HANDLE handle = HandleOf(fd);
  LARGE_INTEGER distance;
  distance.QuadPart = offset;
  LARGE_INTEGER position;
  const DWORD method = whence == SEEK_SET   ? FILE_BEGIN
                       : whence == SEEK_CUR ? FILE_CURRENT
                                            : FILE_END;
  if (!SetFilePointerEx(handle, distance, &position, method)) {
    return Fail(GetLastError());
  }
  return position.QuadPart;
}

int fsync(int fd) {
  if (!FlushFileBuffers(HandleOf(fd))) {
    const DWORD error = GetLastError();
    // Directories and read-only handles cannot be flushed; Linux accepts both.
    if (error == ERROR_ACCESS_DENIED || error == ERROR_INVALID_HANDLE) {
      return 0;
    }
    return Fail(error);
  }
  return 0;
}

int fdatasync(int fd) { return fsync(fd); }

int ftruncate(int fd, off_t length) {
  FILE_END_OF_FILE_INFO info{};
  info.EndOfFile.QuadPart = length;
  if (!SetFileInformationByHandle(HandleOf(fd), FileEndOfFileInfo, &info,
                                  sizeof(info))) {
    return Fail(GetLastError());
  }
  return 0;
}

int posix_fallocate(int fd, off_t offset, off_t length) {
  struct stat st;
  if (fstat(fd, &st) != 0) {
    return errno;
  }
  if (st.st_size < offset + length && ftruncate(fd, offset + length) != 0) {
    return errno;
  }
  return 0;
}

int posix_fadvise(int, off_t, off_t, int) { return 0; }

int fcntl(int fd, int cmd, ...) {
  switch (cmd) {
    case F_DUPFD:
    case F_DUPFD_CLOEXEC:
      return dup(fd);
    case F_GETFD:
      return FD_CLOEXEC;
    case F_SETFD:
      return 0;
    case F_GETFL:
      return FlagsOf(fd);
    case F_SETFL:
      return 0;
    default:
      errno = EINVAL;
      return -1;
  }
}

int dup(int fd) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  HANDLE copy = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &copy,
                       0, FALSE, DUPLICATE_SAME_ACCESS)) {
    return Fail(GetLastError());
  }
  return WrapHandle(copy, FlagsOf(fd) | O_RDWR);
}

int dup2(int fd, int fd2) {
  if (_dup2(fd, fd2) != 0) {
    return -1;
  }
  RememberFlags(fd2, FlagsOf(fd));
  return fd2;
}

int pipe(int fds[2]) {
  HANDLE read_end = nullptr;
  HANDLE write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, nullptr, 1 << 20)) {
    return Fail(GetLastError());
  }
  fds[0] = WrapHandle(read_end, O_RDONLY);
  fds[1] = WrapHandle(write_end, O_WRONLY);
  if (fds[0] < 0 || fds[1] < 0) {
    return -1;
  }
  return 0;
}

int unlink(const char* path) {
  const std::wstring wide = Widen(path);
  if (DeleteFileW(wide.c_str())) {
    return 0;
  }
  const DWORD error = GetLastError();
  if (error == ERROR_ACCESS_DENIED) {
    const DWORD attributes = GetFileAttributesW(wide.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
      errno = EISDIR;
      return -1;
    }
    if (attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_READONLY)) {
      SetFileAttributesW(wide.c_str(),
                         attributes & ~static_cast<DWORD>(FILE_ATTRIBUTE_READONLY));
      if (DeleteFileW(wide.c_str())) {
        return 0;
      }
    }
  }
  return Fail(error);
}

int unlinkat(int dirfd, const char* path, int flags) {
  const std::wstring resolved = ResolveAt(dirfd, path);
  const std::string narrow = Narrow(resolved.c_str());
  return (flags & 0x200 /* AT_REMOVEDIR */) ? rmdir(narrow.c_str())
                                            : unlink(narrow.c_str());
}

int rmdir(const char* path) {
  if (!RemoveDirectoryW(Widen(path).c_str())) {
    return Fail(GetLastError());
  }
  return 0;
}

int mkdir(const char* path, mode_t) {
  if (!CreateDirectoryW(Widen(path).c_str(), nullptr)) {
    return Fail(GetLastError());
  }
  return 0;
}

int access(const char* path, int mode) {
  const DWORD attributes = GetFileAttributesW(Widen(path).c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    return Fail(GetLastError());
  }
  if ((mode & W_OK) && (attributes & FILE_ATTRIBUTE_READONLY) &&
      !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
    errno = EACCES;
    return -1;
  }
  return 0;
}

int isatty(int fd) { return _isatty(fd); }

pid_t getpid(void) { return static_cast<pid_t>(GetCurrentProcessId()); }
uid_t getuid(void) { return 0; }
uid_t geteuid(void) { return 0; }
gid_t getgid(void) { return 0; }

long sysconf(int name) {
  switch (name) {
    case _SC_PAGESIZE:
      return static_cast<long>(SystemInfo().dwPageSize);
    case _SC_NPROCESSORS_CONF:
    case _SC_NPROCESSORS_ONLN:
      return static_cast<long>(
          GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    case _SC_PHYS_PAGES:
    case _SC_AVPHYS_PAGES: {
      MEMORYSTATUSEX status{};
      status.dwLength = sizeof(status);
      if (!GlobalMemoryStatusEx(&status)) {
        return -1;
      }
      const auto bytes = name == _SC_PHYS_PAGES ? status.ullTotalPhys
                                                : status.ullAvailPhys;
      return static_cast<long>(bytes / SystemInfo().dwPageSize);
    }
    default:
      errno = EINVAL;
      return -1;
  }
}

unsigned int sleep(unsigned int seconds) {
  Sleep(seconds * 1000U);
  return 0;
}

int usleep(unsigned long long microseconds) {
  Sleep(static_cast<DWORD>((microseconds + 999) / 1000));
  return 0;
}

int fstat(int fd, struct stat* st) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  return StatHandle(handle, st, (FlagsOf(fd) & O_NOFOLLOW) != 0);
}

int stat(const char* path, struct stat* st) {
  return StatPath(Widen(path), st, false);
}

int lstat(const char* path, struct stat* st) {
  return StatPath(Widen(path), st, true);
}

int fstatat(int dirfd, const char* path, struct stat* st, int flags) {
  const std::wstring resolved = ResolveAt(dirfd, path);
  return StatPath(resolved, st, (flags & 0x100 /* AT_SYMLINK_NOFOLLOW */) != 0);
}

int fchmod(int, mode_t) { return 0; }
int chmod(const char*, mode_t) { return 0; }

void* mmap(void* addr, size_t length, int prot, int flags, int fd,
           off_t offset) {
  if (length == 0 || (flags & MAP_FIXED) || addr != nullptr) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  if (flags & MAP_ANONYMOUS) {
    const DWORD protect =
        (prot & PROT_WRITE) ? PAGE_READWRITE
                            : ((prot & PROT_READ) ? PAGE_READONLY : PAGE_NOACCESS);
    void* memory =
        VirtualAlloc(nullptr, length, MEM_RESERVE | MEM_COMMIT, protect);
    if (memory == nullptr) {
      errno = ENOMEM;
      return MAP_FAILED;
    }
    return memory;
  }
  const HANDLE file = HandleOf(fd);
  if (file == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }
  const bool writable = (prot & PROT_WRITE) != 0;
  const bool copy_on_write = writable && (flags & MAP_PRIVATE);
  const DWORD protect =
      copy_on_write ? PAGE_WRITECOPY : (writable ? PAGE_READWRITE : PAGE_READONLY);
  const DWORD access = copy_on_write ? FILE_MAP_COPY
                       : writable    ? FILE_MAP_WRITE
                                     : FILE_MAP_READ;
  const HANDLE mapping =
      CreateFileMappingW(file, nullptr, protect, 0, 0, nullptr);
  if (mapping == nullptr) {
    Fail(GetLastError());
    return MAP_FAILED;
  }
  const unsigned long long granularity = SystemInfo().dwAllocationGranularity;
  const unsigned long long aligned =
      static_cast<unsigned long long>(offset) / granularity * granularity;
  const size_t skew = static_cast<size_t>(offset - aligned);
  void* view = MapViewOfFile(mapping, access, static_cast<DWORD>(aligned >> 32),
                             static_cast<DWORD>(aligned), length + skew);
  const DWORD error = GetLastError();
  CloseHandle(mapping);  // The view keeps the section alive.
  if (view == nullptr) {
    Fail(error);
    return MAP_FAILED;
  }
  void* result = static_cast<char*>(view) + skew;
  if (flags & MAP_POPULATE) {
    (void)madvise(result, length, MADV_POPULATE_READ);
  }
  return result;
}

int munmap(void* addr, size_t) {
  MEMORY_BASIC_INFORMATION info{};
  if (addr == nullptr || VirtualQuery(addr, &info, sizeof(info)) == 0) {
    errno = EINVAL;
    return -1;
  }
  if (info.Type == MEM_MAPPED) {
    if (!UnmapViewOfFile(info.AllocationBase)) {
      return Fail(GetLastError());
    }
    return 0;
  }
  if (!VirtualFree(info.AllocationBase, 0, MEM_RELEASE)) {
    return Fail(GetLastError());
  }
  return 0;
}

int madvise(void* addr, size_t length, int advice) {
  if (length == 0) {
    return 0;
  }
  switch (advice) {
    case MADV_WILLNEED: {
      WIN32_MEMORY_RANGE_ENTRY range{addr, length};
      (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
      return 0;
    }
    case MADV_POPULATE_READ:
    case MADV_POPULATE_WRITE: {
      // Queue one large read, then fault every page so the call, like Linux,
      // returns only once the range is resident.
      WIN32_MEMORY_RANGE_ENTRY range{addr, length};
      (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
      const size_t page = SystemInfo().dwPageSize;
      const auto* bytes = static_cast<const volatile char*>(addr);
      __try {
        for (size_t i = 0; i < length; i += page) {
          (void)bytes[i];
        }
        (void)bytes[length - 1];
      } __except (EXCEPTION_EXECUTE_HANDLER) {
        errno = EFAULT;
        return -1;
      }
      return 0;
    }
    case MADV_DONTNEED: {
      MEMORY_BASIC_INFORMATION info{};
      if (VirtualQuery(addr, &info, sizeof(info)) != 0 &&
          info.Type == MEM_PRIVATE) {
        // Linux zero-fills private anonymous pages after DONTNEED.
        memset(addr, 0, length);
      } else {
        // Drops clean file pages from the working set; failure is expected.
        (void)VirtualUnlock(addr, length);
      }
      return 0;
    }
    default:
      return 0;
  }
}

int posix_madvise(void* addr, size_t length, int advice) {
  return madvise(addr, length, advice) == 0 ? 0 : errno;
}

int mincore(void* addr, size_t length, unsigned char* vec) {
  const size_t page = SystemInfo().dwPageSize;
  const auto base = reinterpret_cast<uintptr_t>(addr) & ~(page - 1);
  const size_t pages =
      (reinterpret_cast<uintptr_t>(addr) - base + length + page - 1) / page;
  constexpr size_t kBatch = 4096;
  PSAPI_WORKING_SET_EX_INFORMATION info[kBatch];
  for (size_t done = 0; done < pages; done += kBatch) {
    const size_t count = pages - done < kBatch ? pages - done : kBatch;
    for (size_t i = 0; i < count; ++i) {
      info[i].VirtualAddress = reinterpret_cast<void*>(base + (done + i) * page);
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), info,
                           static_cast<DWORD>(count * sizeof(info[0])))) {
      return Fail(GetLastError());
    }
    for (size_t i = 0; i < count; ++i) {
      vec[done + i] = info[i].VirtualAttributes.Valid ? 1 : 0;
    }
  }
  return 0;
}

int mlock(const void*, size_t) { return 0; }
int munlock(const void*, size_t) { return 0; }

int msync(void* addr, size_t length, int) {
  if (!FlushViewOfFile(addr, length)) {
    return Fail(GetLastError());
  }
  return 0;
}

int flock(int fd, int operation) {
  const HANDLE handle = HandleOf(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  // One byte at the top of the offset range: advisory in practice, because
  // no real read or write ever touches it.
  OVERLAPPED overlapped{};
  overlapped.Offset = kLockOffsetLow;
  overlapped.OffsetHigh = kLockOffsetHigh;
  if (operation & LOCK_UN) {
    if (!UnlockFileEx(handle, 0, 1, 0, &overlapped)) {
      return Fail(GetLastError());
    }
    RememberFlags(fd, FlagsOf(fd) & ~kFdLocked);
    return 0;
  }
  if (FlagsOf(fd) & kFdLocked) {
    return 0;  // flock() on a held lock converts it; already ours.
  }
  DWORD lock_flags = 0;
  if (operation & LOCK_EX) {
    lock_flags |= LOCKFILE_EXCLUSIVE_LOCK;
  }
  if (operation & LOCK_NB) {
    lock_flags |= LOCKFILE_FAIL_IMMEDIATELY;
  }
  overlapped.hEvent = ThreadEvent();
  ResetEvent(overlapped.hEvent);
  if (!LockFileEx(handle, lock_flags, 0, 1, 0, &overlapped)) {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      DWORD ignored = 0;
      if (GetOverlappedResult(handle, &overlapped, &ignored, TRUE)) {
        RememberFlags(fd, FlagsOf(fd) | kFdLocked);
        return 0;
      }
      error = GetLastError();
    }
    if (error == ERROR_LOCK_VIOLATION) {
      errno = EWOULDBLOCK;
      return -1;
    }
    return Fail(error);
  }
  RememberFlags(fd, FlagsOf(fd) | kFdLocked);
  return 0;
}

static bool FillTemplate(char* path_template) {
  const size_t n = strlen(path_template);
  if (n < 6 || strcmp(path_template + n - 6, "XXXXXX") != 0) {
    errno = EINVAL;
    return false;
  }
  static constexpr char kAlphabet[] =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  thread_local std::mt19937_64 random{std::random_device{}() ^
                                      GetCurrentThreadId()};
  for (size_t i = n - 6; i < n; ++i) {
    path_template[i] = kAlphabet[random() % (sizeof(kAlphabet) - 1)];
  }
  return true;
}

int mkstemp(char* path_template) {
  const size_t n = strlen(path_template);
  std::string original(path_template, n);
  for (int attempt = 0; attempt < 100; ++attempt) {
    memcpy(path_template, original.data(), n);
    if (!FillTemplate(path_template)) {
      return -1;
    }
    const int fd = open(path_template, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0 || errno != EEXIST) {
      return fd;
    }
  }
  errno = EEXIST;
  return -1;
}

char* mkdtemp(char* path_template) {
  const size_t n = strlen(path_template);
  std::string original(path_template, n);
  for (int attempt = 0; attempt < 100; ++attempt) {
    memcpy(path_template, original.data(), n);
    if (!FillTemplate(path_template)) {
      return nullptr;
    }
    if (mkdir(path_template, 0700) == 0) {
      return path_template;
    }
    if (errno != EEXIST) {
      return nullptr;
    }
  }
  errno = EEXIST;
  return nullptr;
}

int setenv(const char* name, const char* value, int overwrite) {
  if (name == nullptr || *name == '\0' || strchr(name, '=') != nullptr) {
    errno = EINVAL;
    return -1;
  }
  if (!overwrite && getenv(name) != nullptr) {
    return 0;
  }
  // _putenv_s treats "" as removal. An empty value still reaches child
  // processes through the Win32 environment, but getenv() reports it unset.
  if (value == nullptr || *value == '\0') {
    (void)_putenv_s(name, "");
    return SetEnvironmentVariableA(name, "") ? 0 : -1;
  }
  return _putenv_s(name, value) == 0 ? 0 : -1;
}

int unsetenv(const char* name) {
  return _putenv_s(name, "") == 0 ? 0 : -1;
}

struct tm* gmtime_r(const time_t* timer, struct tm* result) {
  return gmtime_s(result, timer) == 0 ? result : nullptr;
}

struct tm* localtime_r(const time_t* timer, struct tm* result) {
  return localtime_s(result, timer) == 0 ? result : nullptr;
}

time_t timegm(struct tm* tm) { return _mkgmtime(tm); }

int clock_gettime(clockid_t clock, struct timespec* ts) {
  switch (clock) {
    case CLOCK_REALTIME: {
      FILETIME ft;
      GetSystemTimePreciseAsFileTime(&ft);
      *ts = TimespecFromFiletime(ft);
      return 0;
    }
    case CLOCK_PROCESS_CPUTIME_ID:
    case CLOCK_THREAD_CPUTIME_ID: {
      FILETIME creation, exit, kernel, user;
      const BOOL ok =
          clock == CLOCK_PROCESS_CPUTIME_ID
              ? GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel,
                                &user)
              : GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel,
                               &user);
      if (!ok) {
        return Fail(GetLastError());
      }
      ULARGE_INTEGER k, u;
      k.LowPart = kernel.dwLowDateTime;
      k.HighPart = kernel.dwHighDateTime;
      u.LowPart = user.dwLowDateTime;
      u.HighPart = user.dwHighDateTime;
      const unsigned long long ticks = k.QuadPart + u.QuadPart;
      ts->tv_sec = static_cast<time_t>(ticks / 10000000ULL);
      ts->tv_nsec = static_cast<long>((ticks % 10000000ULL) * 100ULL);
      return 0;
    }
    default: {
      static const long long frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f.QuadPart;
      }();
      LARGE_INTEGER counter;
      QueryPerformanceCounter(&counter);
      ts->tv_sec = static_cast<time_t>(counter.QuadPart / frequency);
      ts->tv_nsec = static_cast<long>((counter.QuadPart % frequency) *
                                      1000000000LL / frequency);
      return 0;
    }
  }
}

int getrusage(int who, struct rusage* usage) {
  memset(usage, 0, sizeof(*usage));
  FILETIME creation, exit, kernel, user;
  const BOOL ok = who == RUSAGE_THREAD
                      ? GetThreadTimes(GetCurrentThread(), &creation, &exit,
                                       &kernel, &user)
                      : GetProcessTimes(GetCurrentProcess(), &creation, &exit,
                                        &kernel, &user);
  if (ok) {
    auto convert = [](const FILETIME& ft, gufo_rusage_time* out) {
      ULARGE_INTEGER value;
      value.LowPart = ft.dwLowDateTime;
      value.HighPart = ft.dwHighDateTime;
      out->tv_sec = static_cast<long long>(value.QuadPart / 10000000ULL);
      out->tv_usec = static_cast<long long>((value.QuadPart % 10000000ULL) / 10);
    };
    convert(user, &usage->ru_utime);
    convert(kernel, &usage->ru_stime);
  }
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    usage->ru_maxrss = static_cast<long>(counters.PeakWorkingSetSize / 1024);
    // Windows does not separate minor from major faults.
    usage->ru_minflt = static_cast<long>(counters.PageFaultCount);
  }
  return 0;
}

int getrlimit(int, struct rlimit* limit) {
  limit->rlim_cur = RLIM_INFINITY;
  limit->rlim_max = RLIM_INFINITY;
  return 0;
}

int setrlimit(int, const struct rlimit*) { return 0; }

int uname(struct utsname* name) {
  memset(name, 0, sizeof(*name));
  strcpy_s(name->sysname, sizeof(name->sysname), "Windows");
  strcpy_s(name->machine, sizeof(name->machine), "x86_64");
  DWORD size = sizeof(name->nodename);
  (void)GetComputerNameA(name->nodename, &size);
  using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  const auto get_version = ntdll ? reinterpret_cast<RtlGetVersionFn>(
                                       GetProcAddress(ntdll, "RtlGetVersion"))
                                 : nullptr;
  RTL_OSVERSIONINFOW info{};
  info.dwOSVersionInfoSize = sizeof(info);
  if (get_version != nullptr && get_version(&info) == 0) {
    snprintf(name->release, sizeof(name->release), "%lu.%lu.%lu",
             info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber);
    snprintf(name->version, sizeof(name->version), "Windows %lu.%lu build %lu",
             info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber);
  }
  return 0;
}

unsigned long long gufo_resident_bytes(void) {
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))
             ? counters.WorkingSetSize
             : 0;
}

unsigned long long gufo_peak_resident_bytes(void) {
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))
             ? counters.PeakWorkingSetSize
             : 0;
}

unsigned long long gufo_available_physical_bytes(void) {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0;
}

unsigned long long gufo_total_physical_bytes(void) {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
}

int renameat(int olddirfd, const char* oldpath, int newdirfd,
             const char* newpath) {
  const std::wstring from = ResolveAt(olddirfd, oldpath);
  const std::wstring to = ResolveAt(newdirfd, newpath);
  if (from.empty() || to.empty()) {
    errno = EBADF;
    return -1;
  }
  // POSIX-semantics rename replaces the target even while another handle
  // has it open, like rename(2). Fall back to MoveFileEx on older systems.
  const HANDLE source =
      CreateFileW(from.c_str(), 0x00010000L /* DELETE */ | SYNCHRONIZE,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                  nullptr);
  if (source == INVALID_HANDLE_VALUE) {
    return Fail(GetLastError());
  }
  const size_t bytes = sizeof(FILE_RENAME_INFO) + to.size() * sizeof(wchar_t);
  std::string storage(bytes, '\0');
  auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
  info->Flags = 0x1 /* FILE_RENAME_FLAG_REPLACE_IF_EXISTS */ |
                0x2 /* FILE_RENAME_FLAG_POSIX_SEMANTICS */;
  info->RootDirectory = nullptr;
  info->FileNameLength = static_cast<DWORD>(to.size() * sizeof(wchar_t));
  memcpy(info->FileName, to.c_str(), to.size() * sizeof(wchar_t));
  const BOOL renamed = SetFileInformationByHandle(
      source, static_cast<FILE_INFO_BY_HANDLE_CLASS>(22 /* FileRenameInfoEx */),
      info, static_cast<DWORD>(bytes));
  CloseHandle(source);
  if (renamed) {
    return 0;
  }
  if (!MoveFileExW(from.c_str(), to.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return Fail(GetLastError());
  }
  return 0;
}

static FILETIME FiletimeFromTimespec(const struct timespec& ts) {
  const unsigned long long ticks =
      static_cast<unsigned long long>(ts.tv_sec) * 10000000ULL +
      static_cast<unsigned long long>(ts.tv_nsec) / 100ULL +
      116444736000000000ULL;
  FILETIME ft;
  ft.dwLowDateTime = static_cast<DWORD>(ticks);
  ft.dwHighDateTime = static_cast<DWORD>(ticks >> 32);
  return ft;
}

static int TouchHandle(HANDLE handle, const struct timespec times[2]) {
  FILETIME now;
  GetSystemTimeAsFileTime(&now);
  FILETIME access = now;
  FILETIME modify = now;
  FILETIME* access_ptr = &access;
  FILETIME* modify_ptr = &modify;
  if (times != nullptr) {
    if (times[0].tv_nsec == UTIME_OMIT) {
      access_ptr = nullptr;
    } else if (times[0].tv_nsec != UTIME_NOW) {
      access = FiletimeFromTimespec(times[0]);
    }
    if (times[1].tv_nsec == UTIME_OMIT) {
      modify_ptr = nullptr;
    } else if (times[1].tv_nsec != UTIME_NOW) {
      modify = FiletimeFromTimespec(times[1]);
    }
  }
  if (!SetFileTime(handle, nullptr, access_ptr, modify_ptr)) {
    return Fail(GetLastError());
  }
  return 0;
}

int utimensat(int dirfd, const char* path, const struct timespec times[2],
              int flags) {
  const std::wstring resolved = ResolveAt(dirfd, path);
  DWORD attributes = FILE_FLAG_BACKUP_SEMANTICS;
  if (flags & AT_SYMLINK_NOFOLLOW) {
    attributes |= FILE_FLAG_OPEN_REPARSE_POINT;
  }
  const HANDLE handle =
      CreateFileW(resolved.c_str(), FILE_WRITE_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, attributes, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Fail(GetLastError());
  }
  const int result = TouchHandle(handle, times);
  CloseHandle(handle);
  return result;
}

int futimens(int fd, const struct timespec times[2]) {
  return TouchHandle(HandleOf(fd), times);
}

FILE* gufo_tmpfile(void) {
  wchar_t directory[MAX_PATH + 1];
  const DWORD n = GetTempPathW(MAX_PATH + 1, directory);
  if (n == 0 || n > MAX_PATH) {
    errno = ENOENT;
    return nullptr;
  }
  wchar_t name[MAX_PATH + 1];
  if (GetTempFileNameW(directory, L"gfo", 0, name) == 0) {
    Fail(GetLastError());
    return nullptr;
  }
  const HANDLE handle = CreateFileW(
      name, GENERIC_READ | GENERIC_WRITE | 0x00010000L /* DELETE */, 0,
      nullptr, CREATE_ALWAYS,
      FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    Fail(GetLastError());
    DeleteFileW(name);
    return nullptr;
  }
  const int fd =
      _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDWR | _O_BINARY);
  if (fd < 0) {
    CloseHandle(handle);
    errno = EMFILE;
    return nullptr;
  }
  FILE* stream = _fdopen(fd, "w+b");
  if (stream == nullptr) {
    _close(fd);
  }
  return stream;
}

FILE* fmemopen(void* buffer, size_t size, const char* mode) {
  if (mode == nullptr || mode[0] != 'r' || strchr(mode, '+') != nullptr) {
    errno = ENOTSUP;
    return nullptr;
  }
  FILE* stream = gufo_tmpfile();
  if (stream == nullptr) {
    return nullptr;
  }
  if ((size != 0 && fwrite(buffer, 1, size, stream) != size) ||
      fflush(stream) != 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    errno = EIO;
    return nullptr;
  }
  return stream;
}

int fileno(FILE* stream) { return _fileno(stream); }

int dprintf(int fd, const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list copy;
  va_copy(copy, args);
  const int needed = vsnprintf(nullptr, 0, format, copy);
  va_end(copy);
  if (needed < 0) {
    va_end(args);
    return -1;
  }
  std::string text(static_cast<size_t>(needed) + 1, '\0');
  vsnprintf(text.data(), text.size(), format, args);
  va_end(args);
  size_t done = 0;
  while (done < static_cast<size_t>(needed)) {
    const ssize_t n = write(fd, text.data() + done, needed - done);
    if (n <= 0) {
      return -1;
    }
    done += static_cast<size_t>(n);
  }
  return needed;
}

int sched_yield(void) {
  SwitchToThread();
  return 0;
}

long syscall(long, ...) {
  errno = ENOSYS;
  return -1;
}

pid_t waitpid(pid_t pid, int* status, int options) {
  HANDLE process = nullptr;
  {
    std::lock_guard lock(g_child_mutex);
    const auto it = g_children.find(pid);
    if (it != g_children.end()) {
      process = it->second;
    }
  }
  if (process == nullptr) {
    errno = ECHILD;
    return -1;
  }
  const DWORD wait = WaitForSingleObject(
      process, (options & WNOHANG) ? 0 : INFINITE);
  if (wait == WAIT_TIMEOUT) {
    return 0;
  }
  DWORD code = 0;
  GetExitCodeProcess(process, &code);
  {
    std::lock_guard lock(g_child_mutex);
    g_children.erase(pid);
  }
  CloseHandle(process);
  if (status != nullptr) {
    // kill() exits with 128 + signal; report that as a signal like Linux.
    if (code > 128 && code < 128 + 32) {
      *status = static_cast<int>(code - 128);
    } else {
      *status = static_cast<int>((code & 0xff) << 8);
    }
  }
  return pid;
}

int kill(pid_t pid, int sig) {
  HANDLE process = nullptr;
  {
    std::lock_guard lock(g_child_mutex);
    const auto it = g_children.find(pid);
    if (it != g_children.end()) {
      process = it->second;
    }
  }
  const bool owned = process == nullptr;
  if (owned) {
    process = OpenProcess(PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid));
    if (process == nullptr) {
      return Fail(GetLastError());
    }
  }
  BOOL ok = TRUE;
  if (sig != 0) {
    ok = TerminateProcess(process, 128U + static_cast<unsigned>(sig));
  }
  const DWORD error = GetLastError();
  if (owned) {
    CloseHandle(process);
  }
  return ok ? 0 : Fail(error);
}

int sigaction(int sig, const struct sigaction* action,
              struct sigaction* previous) {
  if (previous != nullptr) {
    memset(previous, 0, sizeof(*previous));
  }
  if (sig == SIGPIPE || sig == SIGHUP || sig == SIGCHLD || sig == SIGUSR1) {
    return 0;  // Never delivered on Windows.
  }
  if (action == nullptr) {
    return 0;
  }
  void (*handler)(int) = action->sa_handler;
  const auto old = signal(sig, handler);
  if (old == SIG_ERR) {
    errno = EINVAL;
    return -1;
  }
  if (previous != nullptr) {
    previous->sa_handler = old;
  }
  return 0;
}

int sigemptyset(sigset_t* set) {
  *set = 0;
  return 0;
}

int sigfillset(sigset_t* set) {
  *set = ~0UL;
  return 0;
}

int sigaddset(sigset_t* set, int sig) {
  *set |= 1UL << (sig & 31);
  return 0;
}

int poll(struct pollfd* fds, unsigned long count, int timeout_ms) {
  constexpr SHORT kOutputOnly = POLLERR | POLLHUP | POLLNVAL | POLLPRI;
  SHORT requested[64];
  const unsigned long saved = count < 64 ? count : 64;
  for (unsigned long i = 0; i < saved; ++i) {
    requested[i] = fds[i].events;
    fds[i].events &= ~kOutputOnly;
  }
  const int ready = WSAPoll(fds, count, timeout_ms);
  const int error = ready < 0 ? WSAGetLastError() : 0;
  for (unsigned long i = 0; i < saved; ++i) {
    fds[i].events = requested[i];
  }
  if (ready < 0) {
    errno = error == WSAEINTR ? EINTR : (error == WSAENOTSOCK ? EBADF : EINVAL);
  }
  return ready;
}

}  // extern "C"

int open(const wchar_t* path, int flags, int mode) {
  if (path == nullptr) {
    errno = EFAULT;
    return -1;
  }
  return OpenPath(Backslashed(path), flags, static_cast<mode_t>(mode));
}

int openat(int dirfd, const wchar_t* path, int flags, int mode) {
  return openat(dirfd, Narrow(path).c_str(), flags, mode);
}

int stat(const wchar_t* path, struct stat* st) {
  return StatPath(Backslashed(path), st, false);
}

int lstat(const wchar_t* path, struct stat* st) {
  return StatPath(Backslashed(path), st, true);
}

int mkdir(const wchar_t* path, mode_t mode) {
  return mkdir(Narrow(path).c_str(), mode);
}

int rmdir(const wchar_t* path) { return rmdir(Narrow(path).c_str()); }

int unlink(const wchar_t* path) { return unlink(Narrow(path).c_str()); }

int chmod(const wchar_t*, mode_t) { return 0; }

int utimensat(int dirfd, const wchar_t* path, const struct timespec times[2],
              int flags) {
  return utimensat(dirfd, Narrow(path).c_str(), times, flags);
}

int access(const wchar_t* path, int mode) {
  return access(Narrow(path).c_str(), mode);
}
