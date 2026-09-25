/* Win32 shadow of <spawn.h>: posix_spawnp over CreateProcessW; see
 * compat/win32/spawn.cpp. Supports the file actions Gufo uses (adddup2,
 * addclose); descriptors >= 3 reach the child through the CRT's
 * lpReserved2 descriptor-inheritance block, so FFmpeg's "pipe:N" works. */
#pragma once
#include "gufo_posix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int count;
  int capacity;
  struct gufo_spawn_action* actions;
} posix_spawn_file_actions_t;
typedef struct {
  int flags;
} posix_spawnattr_t;

int posix_spawn_file_actions_init(posix_spawn_file_actions_t* actions);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t* actions);
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t* actions,
                                     int fd, int target);
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t* actions,
                                      int fd);
int posix_spawnattr_init(posix_spawnattr_t* attributes);
int posix_spawnattr_destroy(posix_spawnattr_t* attributes);
/* envp is ignored: the child inherits this process's environment. */
int posix_spawnp(pid_t* pid, const char* file,
                 const posix_spawn_file_actions_t* actions,
                 const posix_spawnattr_t* attributes, char* const argv[],
                 char* const envp[]);
int posix_spawn(pid_t* pid, const char* path,
                const posix_spawn_file_actions_t* actions,
                const posix_spawnattr_t* attributes, char* const argv[],
                char* const envp[]);

#ifdef __cplusplus
}
#endif
