#pragma once

#include <cstddef>

namespace gufo::server {

// Disk-cache limits accepted by `serve llm`. A directory with a zero byte
// budget is invalid. The staging size takes any value here: 0 selects the
// automatic size and the disk store rejects a value that is too small when it
// opens (continuation_disk_store.cpp).
[[nodiscard]] constexpr bool CacheDiskLimitsValid(
    bool directory_set, std::size_t cache_disk_bytes,
    std::size_t /*staging_bytes*/) noexcept {
  return !(directory_set && cache_disk_bytes == 0);
}

}  // namespace gufo::server
