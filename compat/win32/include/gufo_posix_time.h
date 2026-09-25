/* Leaf header: POSIX time extensions, included by the <time.h> shadow after
 * the UCRT's <time.h>. */
#ifndef GUFO_COMPAT_WIN32_POSIX_TIME_H_
#define GUFO_COMPAT_WIN32_POSIX_TIME_H_

#ifdef __cplusplus
extern "C" {
#endif

/* ---- clocks ------------------------------------------------------------- */
typedef int clockid_t;
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3
#define CLOCK_MONOTONIC_RAW 4
#define CLOCK_BOOTTIME 7

struct tm* gmtime_r(const time_t* timer, struct tm* result);
struct tm* localtime_r(const time_t* timer, struct tm* result);
time_t timegm(struct tm* tm);
int clock_gettime(clockid_t clock, struct timespec* ts);

#ifdef __cplusplus
}
#endif

#endif /* GUFO_COMPAT_WIN32_POSIX_TIME_H_ */
