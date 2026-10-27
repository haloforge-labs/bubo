#include <cstddef>
#include <cstdio>

#include "src/cli/serve/cache_disk_limits.hpp"

int main() {
  using gufo::server::CacheDiskLimitsValid;
  int failed = 0;
  const auto expect = [&](bool ok, const char* what) {
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", what);
      ++failed;
    }
  };
  const std::size_t gib = 1ull << 30;
  expect(!CacheDiskLimitsValid(true, 0, 0), "directory with 0 bytes rejected");
  expect(!CacheDiskLimitsValid(true, 0, 8589934592ull),
         "directory with 0 bytes rejected for any staging value");
  expect(CacheDiskLimitsValid(true, gib, 0), "staging 0 accepted");
  expect(CacheDiskLimitsValid(true, gib, 8589934592ull),
         "staging 8589934592 accepted");
  expect(CacheDiskLimitsValid(false, 0, 0), "no directory: 0 bytes not checked");
  expect(CacheDiskLimitsValid(true, 1, 0), "1 byte accepted");
  std::printf("cache_disk_limits_test: %s\n", failed ? "FAILED" : "6 checks passed");
  return failed ? 1 : 0;
}
