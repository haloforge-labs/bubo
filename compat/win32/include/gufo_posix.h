/* POSIX surface used by Gufo, implemented on Win32 in compat/win32/posix.cpp.
 *
 * Every shadow header in this directory includes this file. It deliberately
 * does not include <windows.h>: translation units that only want unistd.h must
 * not inherit the Win32 macro namespace. The build defines
 * _CRT_DECLARE_NONSTDC_NAMES=0 so the UCRT's own deprecated POSIX aliases
 * (open, close, read, struct stat, ...) do not collide with these.
 */
#ifndef GUFO_COMPAT_WIN32_POSIX_H_
#define GUFO_COMPAT_WIN32_POSIX_H_

#include <stddef.h>
#include <stdint.h>
#include <time.h>
/* The UCRT's own <sys/stat.h> (reached through the shadow) must be parsed
 * before the st_atime/st_mtime/st_ctime macros below exist. The shadows of
 * <sys/types.h> and <time.h> include only leaf headers, so the UCRT can
 * include them mid-parse without dragging this file in early. */
#include <sys/types.h>
#include <sys/stat.h>
#include "gufo_posix_types.h"
#include "gufo_posix_time.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- open(2) flags ------------------------------------------------------ */
#ifndef O_RDONLY
#define O_RDONLY 0x0000
#define O_WRONLY 0x0001
#define O_RDWR 0x0002
#define O_APPEND 0x0008
#define O_CREAT 0x0100
#define O_TRUNC 0x0200
#define O_EXCL 0x0400
#endif
#define O_ACCMODE 0x0003
#define O_BINARY 0x8000
#define O_CLOEXEC 0x0080
#define O_NONBLOCK 0x00100000
#define O_DIRECT 0x01000000
#define O_NOFOLLOW 0x02000000
#define O_DIRECTORY 0x04000000
#define O_SYNC 0x08000000
#define O_DSYNC O_SYNC
#define O_NOATIME 0
/* Windows only: a cached handle that many threads can pread() concurrently
   (FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED). For small random reads
   of a file that is also mapped, where O_DIRECT reads are serialized. */
#define O_CONCURRENT_RANDOM 0x10000000

#define F_DUPFD 0
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define F_DUPFD_CLOEXEC 1030
#define FD_CLOEXEC 1

#define AT_FDCWD (-100)
#define AT_SYMLINK_NOFOLLOW 0x100
#define AT_REMOVEDIR 0x200
#define UTIME_NOW ((1L << 30) - 1L)
#define UTIME_OMIT ((1L << 30) - 2L)

#define F_OK 0
#define X_OK 1
#define W_OK 2
#define R_OK 4

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

#define POSIX_FADV_NORMAL 0
#define POSIX_FADV_RANDOM 1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_NOREUSE 5

/* ---- stat(2) ------------------------------------------------------------ */
#define S_IFMT 0170000
#define S_IFSOCK 0140000
#define S_IFLNK 0120000
#define S_IFREG 0100000
#define S_IFBLK 0060000
#define S_IFDIR 0040000
#define S_IFCHR 0020000
#define S_IFIFO 0010000
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#define S_ISLNK(m) (((m) & S_IFMT) == S_IFLNK)
#define S_ISCHR(m) (((m) & S_IFMT) == S_IFCHR)
#define S_ISFIFO(m) (((m) & S_IFMT) == S_IFIFO)
#define S_ISSOCK(m) (((m) & S_IFMT) == S_IFSOCK)
#define S_IRWXU 0700
#define S_IRUSR 0400
#define S_IWUSR 0200
#define S_IXUSR 0100
#define S_IRWXG 0070
#define S_IRGRP 0040
#define S_IWGRP 0020
#define S_IXGRP 0010
#define S_IRWXO 0007
#define S_IROTH 0004
#define S_IWOTH 0002
#define S_IXOTH 0001

/* Windows has no owner/group/other bits. Files report 0600 and directories
 * 0700, owned by uid 0 == geteuid(), which is what Gufo's "private cache
 * file" checks expect of files this user created. */
struct stat {
  unsigned long long st_dev;
  unsigned long long st_ino;
  mode_t st_mode;
  unsigned int st_nlink;
  uid_t st_uid;
  gid_t st_gid;
  unsigned long long st_rdev;
  off_t st_size;
  long long st_blksize;
  long long st_blocks;
  struct timespec st_atim;
  struct timespec st_mtim;
  struct timespec st_ctim;
};
#define st_atime st_atim.tv_sec
#define st_mtime st_mtim.tv_sec
#define st_ctime st_ctim.tv_sec

/* ---- mman --------------------------------------------------------------- */
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE 0x8000
#define MAP_FAILED ((void*)-1)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_FREE 8
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#define MADV_DONTDUMP 16
#define MADV_POPULATE_READ 22
#define MADV_POPULATE_WRITE 23
#define POSIX_MADV_NORMAL MADV_NORMAL
#define POSIX_MADV_RANDOM MADV_RANDOM
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_WILLNEED MADV_WILLNEED
#define POSIX_MADV_DONTNEED MADV_DONTNEED
#define MS_ASYNC 1
#define MS_SYNC 4
#define MS_INVALIDATE 2
#define MFD_CLOEXEC 1

/* ---- flock -------------------------------------------------------------- */
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8

/* ---- sysconf ------------------------------------------------------------ */
#define _SC_PAGESIZE 30
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_CONF 83
#define _SC_NPROCESSORS_ONLN 84
#define _SC_PHYS_PAGES 85
#define _SC_AVPHYS_PAGES 86

/* ---- resource ----------------------------------------------------------- */
struct gufo_rusage_time {
  long long tv_sec;
  long long tv_usec;
};
struct rusage {
  struct gufo_rusage_time ru_utime;
  struct gufo_rusage_time ru_stime;
  long ru_maxrss; /* KiB, like Linux */
  long ru_ixrss;
  long ru_idrss;
  long ru_isrss;
  long ru_minflt;
  long ru_majflt;
  long ru_nswap;
  long ru_inblock;
  long ru_oublock;
  long ru_msgsnd;
  long ru_msgrcv;
  long ru_nsignals;
  long ru_nvcsw;
  long ru_nivcsw;
};
#define RUSAGE_SELF 0
#define RUSAGE_CHILDREN (-1)
#define RUSAGE_THREAD 1
struct rlimit {
  rlim_t rlim_cur;
  rlim_t rlim_max;
};
#define RLIM_INFINITY (~0ULL)
#define RLIMIT_CPU 0
#define RLIMIT_FSIZE 1
#define RLIMIT_DATA 2
#define RLIMIT_STACK 3
#define RLIMIT_CORE 4
#define RLIMIT_NOFILE 7
#define RLIMIT_AS 9
#define RLIMIT_MEMLOCK 8

/* ---- utsname ------------------------------------------------------------ */
struct utsname {
  char sysname[65];
  char nodename[65];
  char release[65];
  char version[65];
  char machine[65];
};

/* ---- wait --------------------------------------------------------------- */
#define WNOHANG 1
#define WEXITSTATUS(s) (((s) >> 8) & 0xff)
#define WTERMSIG(s) ((s) & 0x7f)
#define WIFEXITED(s) (WTERMSIG(s) == 0)
#define WIFSIGNALED(s) (WTERMSIG(s) != 0)

/* ---- functions ---------------------------------------------------------- */
int open(const char* path, int flags, ...);
int openat(int dirfd, const char* path, int flags, ...);
int close(int fd);
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);
ssize_t pread(int fd, void* buf, size_t count, off_t offset);
ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset);
off_t lseek(int fd, off_t offset, int whence);
int fsync(int fd);
int fdatasync(int fd);
int ftruncate(int fd, off_t length);
int posix_fallocate(int fd, off_t offset, off_t length);
int posix_fadvise(int fd, off_t offset, off_t length, int advice);
int fcntl(int fd, int cmd, ...);
int dup(int fd);
int dup2(int fd, int fd2);
int pipe(int fds[2]);
int unlink(const char* path);
int unlinkat(int dirfd, const char* path, int flags);
int rmdir(const char* path);
int mkdir(const char* path, mode_t mode);
int access(const char* path, int mode);
int isatty(int fd);
pid_t getpid(void);
uid_t getuid(void);
uid_t geteuid(void);
gid_t getgid(void);
long sysconf(int name);
unsigned int sleep(unsigned int seconds);
int usleep(unsigned long long microseconds);
int fstat(int fd, struct stat* st);
int stat(const char* path, struct stat* st);
int lstat(const char* path, struct stat* st);
int fstatat(int dirfd, const char* path, struct stat* st, int flags);
int fchmod(int fd, mode_t mode);
int chmod(const char* path, mode_t mode);

void* mmap(void* addr, size_t length, int prot, int flags, int fd,
           off_t offset);
int munmap(void* addr, size_t length);
int madvise(void* addr, size_t length, int advice);
int posix_madvise(void* addr, size_t length, int advice);
int mincore(void* addr, size_t length, unsigned char* vec);
int mlock(const void* addr, size_t length);
int munlock(const void* addr, size_t length);
int msync(void* addr, size_t length, int flags);

int renameat(int olddirfd, const char* oldpath, int newdirfd,
             const char* newpath);
int utimensat(int dirfd, const char* path, const struct timespec times[2],
              int flags);
int futimens(int fd, const struct timespec times[2]);
/* Read-only memory streams ("r"/"rb") are backed by a delete-on-close temp
 * file; write modes are not supported (errno = ENOTSUP). */
FILE* fmemopen(void* buffer, size_t size, const char* mode);
/* A delete-on-close "w+b" stream in %TEMP%. */
FILE* gufo_tmpfile(void);

int flock(int fd, int operation);
int fileno(FILE* stream);
int dprintf(int fd, const char* format, ...);
int mkstemp(char* path_template);
char* mkdtemp(char* path_template);

int setenv(const char* name, const char* value, int overwrite);
int unsetenv(const char* name);


int getrusage(int who, struct rusage* usage);
int getrlimit(int resource, struct rlimit* limit);
int setrlimit(int resource, const struct rlimit* limit);
int uname(struct utsname* name);

pid_t waitpid(pid_t pid, int* status, int options);

/* Win32 stand-ins for /proc/self/status and /proc/meminfo reads. */
unsigned long long gufo_resident_bytes(void);
unsigned long long gufo_peak_resident_bytes(void);
unsigned long long gufo_available_physical_bytes(void);
unsigned long long gufo_total_physical_bytes(void);
int kill(pid_t pid, int sig);

#ifdef __cplusplus
}

/* std::filesystem::path::c_str() is wchar_t on Windows. These overloads take
 * it directly, so `open(path.c_str(), ...)` works on both platforms. */
int open(const wchar_t* path, int flags, int mode = 0);
int openat(int dirfd, const wchar_t* path, int flags, int mode = 0);
int stat(const wchar_t* path, struct stat* st);
int lstat(const wchar_t* path, struct stat* st);
int mkdir(const wchar_t* path, mode_t mode);
int rmdir(const wchar_t* path);
int unlink(const wchar_t* path);
int access(const wchar_t* path, int mode);
int chmod(const wchar_t* path, mode_t mode);
int utimensat(int dirfd, const wchar_t* path, const struct timespec times[2],
              int flags);
#endif

#endif /* GUFO_COMPAT_WIN32_POSIX_H_ */
