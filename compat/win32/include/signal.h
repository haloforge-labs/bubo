/* Win32 shadow of <signal.h>; see gufo_posix.h. */
#pragma once
#include_next <signal.h>
#include "gufo_posix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Signals Windows lacks. They are accepted and ignored: Winsock never raises
 * SIGPIPE, and SIGTERM/SIGHUP are only ever sent by kill(), which terminates. */
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#ifndef SIGHUP
#define SIGHUP 1
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif
#ifndef SIGUSR1
#define SIGUSR1 30
#endif
#ifndef SIGCHLD
#define SIGCHLD 17
#endif

typedef unsigned long sigset_t;
struct sigaction {
  void (*sa_handler)(int);
  void (*sa_sigaction)(int, void*, void*);
  sigset_t sa_mask;
  int sa_flags;
};
#define SA_RESTART 0x10000000
#define SA_SIGINFO 4
#define SA_RESETHAND 0x80000000

int sigaction(int sig, const struct sigaction* action, struct sigaction* previous);
int sigemptyset(sigset_t* set);
int sigfillset(sigset_t* set);
int sigaddset(sigset_t* set, int sig);

#ifdef __cplusplus
}
#endif
