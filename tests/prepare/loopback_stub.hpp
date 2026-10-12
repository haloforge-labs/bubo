#pragma once

#ifndef BUBO_PREPARE_TESTING
#error "The loopback stub must never be compiled into a release target"
#endif

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace bubo::prepare::test {
struct StubRequest {
  std::string method, target;
  std::map<std::string, std::string> headers;
  unsigned number = 0;
};
struct StubReply {
  int status = 200;
  std::map<std::string, std::string> headers;
  std::string body;
  std::filesystem::path file;
  // Generated bodies use byte(offset % 251); no logical-size allocation/file.
  std::uint64_t generated_size = 0;
  std::uint64_t start = 0;
  std::uint64_t length = 0;
  // Send fewer bytes than advertised, then close (drop/short-body fixture).
  std::uint64_t drop_after = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t corrupt_at = std::numeric_limits<std::uint64_t>::max();
};
StubReply FileReply(const StubRequest&, const std::filesystem::path&,
                    std::string etag = "\"fixture-v1\"");
StubReply GeneratedReply(const StubRequest&, std::uint64_t size);

// A test-only native HTTP server. It binds exclusively to 127.0.0.1 and lets
// the OS assign a port, retrying if it assigns a reserved 8081-8083 port.
// The handler runs serially; records() is safe while the server is running.
class LoopbackStub {
public:
  using Handler = std::function<StubReply(const StubRequest&)>;
  explicit LoopbackStub(Handler);
  // Mirror is ROOT/<repo>/<path>; the URL is
  // /<repo>/resolve/<40 lowercase hex>/<path>. No production files are opened.
  explicit LoopbackStub(const std::filesystem::path& mirror);
  ~LoopbackStub();
  LoopbackStub(const LoopbackStub&) = delete;
  LoopbackStub& operator=(const LoopbackStub&) = delete;
  std::string url(std::string_view target = "/") const;
  std::uint16_t port() const;
  std::vector<StubRequest> records() const;
  std::string error() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace bubo::prepare::test
