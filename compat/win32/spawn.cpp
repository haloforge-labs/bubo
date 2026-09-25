// posix_spawnp on CreateProcessW. See include/spawn.h.
//
// POSIX children inherit descriptors by number. Win32 children inherit
// handles, and the CRT rebuilds its descriptor table from the lpReserved2
// block in STARTUPINFO (the protocol _spawn uses): an int count, one flag byte
// per descriptor, then one handle per descriptor. Filling that block lets the
// child's descriptor N be the handle we dup2()'d to N, which is how FFmpeg's
// "pipe:3" input finds its audio stream.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <errno.h>
#include <io.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <new>
#include <string>
#include <vector>

// POSIX code declares `extern char** environ;` itself. Spawned children
// always inherit this process's environment, so the value is never read.
char** environ = nullptr;

struct gufo_spawn_action {
  int kind;  // 0 = dup2, 1 = close
  int fd;
  int target;
};

extern "C" void gufo_register_child(pid_t pid, void* process);

namespace {

constexpr unsigned char kFopen = 0x01;
constexpr unsigned char kFpipe = 0x08;
constexpr unsigned char kFdev = 0x40;

int Append(posix_spawn_file_actions_t* actions, gufo_spawn_action action) {
  if (actions->count == actions->capacity) {
    const int capacity = actions->capacity == 0 ? 8 : actions->capacity * 2;
    auto* grown = static_cast<gufo_spawn_action*>(
        realloc(actions->actions, sizeof(gufo_spawn_action) * capacity));
    if (grown == nullptr) {
      return ENOMEM;
    }
    actions->actions = grown;
    actions->capacity = capacity;
  }
  actions->actions[actions->count++] = action;
  return 0;
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
  return out;
}

// Quotes one argument so CommandLineToArgvW / the CRT parse it back intact.
void AppendQuoted(std::wstring& command, const std::wstring& argument) {
  if (!command.empty()) {
    command += L' ';
  }
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    command += argument;
    return;
  }
  command += L'"';
  for (auto it = argument.begin();; ++it) {
    unsigned backslashes = 0;
    while (it != argument.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == argument.end()) {
      command.append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      command.append(backslashes * 2 + 1, L'\\');
      command += *it;
    } else {
      command.append(backslashes, L'\\');
      command += *it;
    }
  }
  command += L'"';
}

HANDLE ParentHandle(int fd) {
  return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}

int Spawn(pid_t* pid, const char* file,
          const posix_spawn_file_actions_t* actions, char* const argv[]) {
  // Child descriptor -> parent handle, starting from the inherited stdio.
  std::map<int, HANDLE> table;
  for (int fd = 0; fd <= 2; ++fd) {
    const HANDLE handle = ParentHandle(fd);
    if (handle != INVALID_HANDLE_VALUE && handle != nullptr) {
      table[fd] = handle;
    }
  }
  if (actions != nullptr) {
    for (int i = 0; i < actions->count; ++i) {
      const gufo_spawn_action& action = actions->actions[i];
      if (action.kind == 1) {
        table.erase(action.fd);
        continue;
      }
      const auto previous = table.find(action.fd);
      const HANDLE source =
          previous != table.end() ? previous->second : ParentHandle(action.fd);
      if (source == INVALID_HANDLE_VALUE || source == nullptr) {
        return EBADF;
      }
      table[action.target] = source;
    }
  }

  // Inheritable duplicates, restricted to exactly this list.
  std::map<int, HANDLE> inherited;
  std::vector<HANDLE> handle_list;
  auto cleanup = [&] {
    for (auto& [fd, handle] : inherited) {
      CloseHandle(handle);
    }
  };
  for (const auto& [fd, handle] : table) {
    HANDLE copy = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(),
                         &copy, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
      cleanup();
      return EBADF;
    }
    inherited[fd] = copy;
    handle_list.push_back(copy);
  }

  const int count = inherited.empty() ? 0 : inherited.rbegin()->first + 1;
  std::vector<unsigned char> reserved(
      sizeof(int) + static_cast<std::size_t>(count) * (1 + sizeof(intptr_t)),
      0);
  memcpy(reserved.data(), &count, sizeof(int));
  unsigned char* flags = reserved.data() + sizeof(int);
  unsigned char* handles = flags + count;
  for (int fd = 0; fd < count; ++fd) {
    intptr_t value = -1;
    const auto it = inherited.find(fd);
    if (it != inherited.end()) {
      value = reinterpret_cast<intptr_t>(it->second);
      const DWORD type = GetFileType(it->second);
      flags[fd] = kFopen | (type == FILE_TYPE_PIPE   ? kFpipe
                            : type == FILE_TYPE_CHAR ? kFdev
                                                     : 0);
    }
    memcpy(handles + static_cast<std::size_t>(fd) * sizeof(intptr_t), &value,
           sizeof(value));
  }

  SIZE_T attribute_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
  std::vector<unsigned char> attribute_storage(attribute_size);
  auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(
      attribute_storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_size)) {
    cleanup();
    return ENOMEM;
  }
  if (!handle_list.empty()) {
    UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                              handle_list.data(),
                              handle_list.size() * sizeof(HANDLE), nullptr,
                              nullptr);
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  auto std_handle = [&](int fd) {
    const auto it = inherited.find(fd);
    return it != inherited.end() ? it->second : nullptr;
  };
  startup.StartupInfo.hStdInput = std_handle(0);
  startup.StartupInfo.hStdOutput = std_handle(1);
  startup.StartupInfo.hStdError = std_handle(2);
  startup.StartupInfo.cbReserved2 = static_cast<WORD>(reserved.size());
  startup.StartupInfo.lpReserved2 = reserved.data();
  startup.lpAttributeList = attributes;

  // argv[0] is only a display name on Linux; run `file` itself.
  std::wstring command;
  const std::wstring program = Widen(file);
  AppendQuoted(command, program);
  if (argv != nullptr && argv[0] != nullptr) {
    for (int i = 1; argv[i] != nullptr; ++i) {
      AppendQuoted(command, Widen(argv[i]));
    }
  }
  PROCESS_INFORMATION process{};
  const BOOL ok = CreateProcessW(
      nullptr, command.data(), nullptr, nullptr, !handle_list.empty(),
      EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo,
      &process);
  const DWORD error = GetLastError();
  DeleteProcThreadAttributeList(attributes);
  cleanup();
  if (!ok) {
    return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND
               ? ENOENT
               : (error == ERROR_ACCESS_DENIED ? EACCES : EINVAL);
  }
  CloseHandle(process.hThread);
  const auto child = static_cast<pid_t>(process.dwProcessId);
  gufo_register_child(child, process.hProcess);
  if (pid != nullptr) {
    *pid = child;
  }
  return 0;
}

}  // namespace

extern "C" {

int posix_spawn_file_actions_init(posix_spawn_file_actions_t* actions) {
  actions->count = 0;
  actions->capacity = 0;
  actions->actions = nullptr;
  return 0;
}

int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t* actions) {
  free(actions->actions);
  actions->actions = nullptr;
  actions->count = 0;
  actions->capacity = 0;
  return 0;
}

int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t* actions,
                                     int fd, int target) {
  return Append(actions, {0, fd, target});
}

int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t* actions,
                                      int fd) {
  return Append(actions, {1, fd, -1});
}

int posix_spawnattr_init(posix_spawnattr_t* attributes) {
  attributes->flags = 0;
  return 0;
}

int posix_spawnattr_destroy(posix_spawnattr_t*) { return 0; }

int posix_spawnp(pid_t* pid, const char* file,
                 const posix_spawn_file_actions_t* actions,
                 const posix_spawnattr_t*, char* const argv[], char* const[]) {
  return Spawn(pid, file, actions, argv);
}

int posix_spawn(pid_t* pid, const char* path,
                const posix_spawn_file_actions_t* actions,
                const posix_spawnattr_t*, char* const argv[], char* const[]) {
  return Spawn(pid, path, actions, argv);
}

}  // extern "C"
