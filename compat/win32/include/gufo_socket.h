/* BSD socket names on Winsock. Sockets stay `int` in Gufo: Win32 kernel
 * handles are guaranteed to fit in 32 bits. Call sites still differ in three
 * ways Winsock cannot hide: sockets close with closesocket(), errors come from
 * WSAGetLastError() rather than errno, and SO_RCVTIMEO/SO_SNDTIMEO take a DWORD
 * of milliseconds. src/core/platform/socket.hpp wraps those. */
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include "gufo_posix.h"
#include "gufo_win32_unmacro.h"

#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif
#define SOCK_CLOEXEC 0
#define SOCK_NONBLOCK 0
typedef unsigned long nfds_t;

#ifdef __cplusplus
extern "C" {
#endif
/* WSAPoll with POSIX semantics: POLLERR/POLLHUP/POLLNVAL are accepted in
 * `events` (WSAPoll rejects them with WSAEINVAL) and errno is set. */
int poll(struct pollfd* fds, unsigned long count, int timeout_ms);
#ifdef __cplusplus
}
#endif
