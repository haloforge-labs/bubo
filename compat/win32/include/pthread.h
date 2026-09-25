/* Win32 placeholder for <pthread.h>. Gufo threads with std::thread; a few
 * sources include this header without using it. Any real pthread call fails
 * to compile, which is intended. */
#pragma once
#include <sched.h>
