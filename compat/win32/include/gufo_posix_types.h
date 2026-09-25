/* Leaf header: POSIX scalar types only. Safe to include from anywhere,
 * including from inside UCRT headers mid-parse (see sys/types.h). */
#ifndef GUFO_COMPAT_WIN32_POSIX_TYPES_H_
#define GUFO_COMPAT_WIN32_POSIX_TYPES_H_

/* ---- types -------------------------------------------------------------- */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef long long ssize_t;
#endif
#ifndef GUFO_OFF_T_DEFINED
#define GUFO_OFF_T_DEFINED
typedef long long off_t;
#endif
#ifndef GUFO_PID_T_DEFINED
#define GUFO_PID_T_DEFINED
typedef int pid_t;
#endif
typedef unsigned int mode_t;
typedef unsigned int uid_t;
typedef unsigned int gid_t;
typedef unsigned long long rlim_t;

#endif /* GUFO_COMPAT_WIN32_POSIX_TYPES_H_ */
