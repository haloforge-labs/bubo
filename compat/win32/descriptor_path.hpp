#ifndef GUFO_COMPAT_DESCRIPTOR_PATH_HPP_
#define GUFO_COMPAT_DESCRIPTOR_PATH_HPP_

#include <cerrno>
#include <cstddef>
#include <string>

#ifdef _WIN32
#include "gufo_posix.h"
extern "C" ssize_t gufo_descriptor_path(int fd, char* buffer, std::size_t size);
extern "C" ssize_t gufo_executable_path(char* buffer, std::size_t size);
#else
#include <unistd.h>
inline ssize_t gufo_descriptor_path(int fd, char* buffer, std::size_t size) {
  const std::string link = "/proc/self/fd/" + std::to_string(fd);
  return ::readlink(link.c_str(), buffer, size);
}
inline ssize_t gufo_executable_path(char* buffer, std::size_t size) {
  return ::readlink("/proc/self/exe", buffer, size);
}
#endif

#endif  // GUFO_COMPAT_DESCRIPTOR_PATH_HPP_
