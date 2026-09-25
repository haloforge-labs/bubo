#ifndef GUFO_CORE_PLATFORM_SOCKET_HPP_
#define GUFO_CORE_PLATFORM_SOCKET_HPP_

// The few socket operations whose POSIX spelling does not work on Winsock.
// Sockets stay `int` descriptors everywhere else; on Windows the value is the
// SOCKET handle, which always fits in 32 bits.

#include <sys/socket.h>
#include <sys/types.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>

#ifndef _WIN32
#include <poll.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace gufo::platform {

#ifdef _WIN32
namespace detail {
/// Maps the last Winsock error onto errno so POSIX error checks keep working.
inline void SetErrnoFromSocketError() noexcept {
  switch (::WSAGetLastError()) {
    case WSAEWOULDBLOCK:
    case WSAETIMEDOUT:  // SO_RCVTIMEO/SO_SNDTIMEO expiry is EAGAIN on Linux.
      errno = EAGAIN;
      break;
    case WSAEINTR:
      errno = EINTR;
      break;
    case WSA_OPERATION_ABORTED:  // StopSocketReads cancelled the recv.
      errno = ECANCELED;
      break;
    case WSAECONNRESET:
    case WSAECONNABORTED:
    case WSAENETRESET:
      errno = ECONNRESET;
      break;
    case WSAESHUTDOWN:
    case WSAENOTCONN:
      errno = EPIPE;
      break;
    case WSAENOTSOCK:
      errno = EBADF;
      break;
    default:
      errno = EIO;
      break;
  }
}

inline int ClampLength(std::size_t size) noexcept {
  return size > static_cast<std::size_t>(INT_MAX) ? INT_MAX
                                                  : static_cast<int>(size);
}
}  // namespace detail
#endif

inline int CloseSocket(int fd) noexcept {
#ifdef _WIN32
  return ::closesocket(static_cast<SOCKET>(fd));
#else
  return ::close(fd);
#endif
}

/// Closes an accepted connection. Closing with unread input sends a TCP
/// reset on both platforms, but Linux still lets the peer read what arrived
/// before it while Windows discards it, losing a final response or close
/// frame. On Windows, finish sending, drain briefly, then close.
inline int CloseConnection(int fd) noexcept {
#ifdef _WIN32
  const auto socket = static_cast<SOCKET>(fd);
  (void)::shutdown(socket, SD_SEND);
  u_long non_blocking = 1;
  (void)::ioctlsocket(socket, FIONBIO, &non_blocking);
  char sink[4096];
  for (int round = 0; round < 50; ++round) {
    const int n = ::recv(socket, sink, sizeof(sink), 0);
    if (n == 0)
      break;  // Peer finished too: an orderly close.
    if (n < 0) {
      if (::WSAGetLastError() != WSAEWOULDBLOCK)
        break;
      WSAPOLLFD readable{socket, POLLRDNORM, 0};
      if (::WSAPoll(&readable, 1, 10) <= 0 && round >= 10)
        break;  // Quiet for a while: the peer has what we sent.
    }
  }
  return ::closesocket(socket);
#else
  return ::close(fd);
#endif
}

/// Wakes a thread blocked reading `fd` and stops further reads, keeping the
/// send half usable. Linux does this with shutdown(SHUT_RD). Winsock resets
/// the connection when data arrives after SD_RECEIVE, which would discard
/// frames the peer has not read yet; Windows readers instead wait in short
/// WaitReadable slices and re-check their own stop flag, so this is a no-op.
inline void StopSocketReads([[maybe_unused]] int fd) noexcept {
#ifndef _WIN32
  (void)::shutdown(fd, SHUT_RD);
#endif
}

/// Waits up to `timeout` for `fd` to become readable (or closed/errored).
inline bool WaitReadable(int fd, std::chrono::milliseconds timeout) noexcept {
#ifdef _WIN32
  WSAPOLLFD readable{static_cast<SOCKET>(fd), POLLRDNORM, 0};
  return ::WSAPoll(&readable, 1, static_cast<INT>(timeout.count())) != 0;
#else
  pollfd readable{fd, POLLIN, 0};
  return ::poll(&readable, 1, static_cast<int>(timeout.count())) != 0;
#endif
}

/// read(2) on a connected socket.
inline ssize_t SocketRead(int fd, void* buffer, std::size_t size) noexcept {
#ifdef _WIN32
  const int n = ::recv(static_cast<SOCKET>(fd), static_cast<char*>(buffer),
                       detail::ClampLength(size), 0);
  if (n < 0)
    detail::SetErrnoFromSocketError();
  return n;
#else
  return ::read(fd, buffer, size);
#endif
}

inline ssize_t SocketRecv(int fd, void* buffer, std::size_t size,
                          int flags) noexcept {
#ifdef _WIN32
  const int n = ::recv(static_cast<SOCKET>(fd), static_cast<char*>(buffer),
                       detail::ClampLength(size), flags);
  if (n < 0)
    detail::SetErrnoFromSocketError();
  return n;
#else
  return ::recv(fd, buffer, size, flags);
#endif
}

inline ssize_t SocketSend(int fd, const void* data, std::size_t size,
                          int flags) noexcept {
#ifdef _WIN32
  const int n = ::send(static_cast<SOCKET>(fd), static_cast<const char*>(data),
                       detail::ClampLength(size), flags);
  if (n < 0)
    detail::SetErrnoFromSocketError();
  return n;
#else
  return ::send(fd, data, size, flags);
#endif
}

template<typename T>
int SetSocketOption(int fd, int level, int name, const T& value) noexcept {
#ifdef _WIN32
  return ::setsockopt(static_cast<SOCKET>(fd), level, name,
                      reinterpret_cast<const char*>(&value),
                      static_cast<int>(sizeof(value)));
#else
  return ::setsockopt(fd, level, name, &value, sizeof(value));
#endif
}

/// SO_RCVTIMEO / SO_SNDTIMEO. Winsock takes a DWORD of milliseconds where
/// Linux takes a timeval; passing a timeval on Windows silently sets
/// tv_sec milliseconds.
inline int SetSocketTimeout(int fd, int name,
                            std::chrono::milliseconds timeout) noexcept {
#ifdef _WIN32
  const DWORD ms = static_cast<DWORD>(timeout.count());
  return SetSocketOption(fd, SOL_SOCKET, name, ms);
#else
  const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(timeout);
  const timeval tv{static_cast<decltype(tv.tv_sec)>(seconds.count()),
                   static_cast<decltype(tv.tv_usec)>(
                       std::chrono::duration_cast<std::chrono::microseconds>(
                           timeout - seconds)
                           .count())};
  return SetSocketOption(fd, SOL_SOCKET, name, tv);
#endif
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_SOCKET_HPP_
