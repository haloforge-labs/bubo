/* Win32 shadow of <sys/syscall.h>; see gufo_posix.h. */
#pragma once
#include "gufo_posix.h"
/* No memfd on Windows: syscall(SYS_memfd_create) fails with ENOSYS. */
#define SYS_memfd_create 319
#ifdef __cplusplus
extern "C"
#endif
long syscall(long number, ...);
