#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
struct HostPolicy {
  enum class Source { HuggingFace, Amd } source = Source::HuggingFace;
#ifdef BUBO_PREPARE_TESTING
  bool test_loopback = false;
#endif
};
struct UrlResult {
  std::string url, host, error;
  explicit operator bool() const { return error.empty(); }
};
UrlResult CheckUrl(std::string_view url, const HostPolicy& policy = {});
UrlResult RedirectUrl(std::string_view current, std::string_view location,
                      unsigned hops, const HostPolicy& policy = {});
std::string OriginUrl(const Entry& entry);
using Headers = std::map<std::string, std::string>;
struct Request {
  std::string url;
  Headers headers;
};
struct Response {
  int status = 0;
  Headers headers;
};
using HeaderSink = std::function<bool(const Response&)>;
using BodySink = std::function<bool(std::span<const std::uint8_t>)>;
struct TransferResult {
  std::string error;
};
class Transport {
public:
  virtual ~Transport() = default;
  virtual TransferResult Perform(const Request&, const HeaderSink&,
                                 const BodySink&,
                                 const std::function<bool()>& cancelled) = 0;
};
class CurlTransport final : public Transport {
public:
  explicit CurlTransport(HostPolicy policy = {}) : policy_(policy) {}
  TransferResult Perform(const Request&, const HeaderSink&, const BodySink&,
                         const std::function<bool()>& cancelled) override;

private:
  HostPolicy policy_;
};
struct ContentRange {
  std::uint64_t start = 0, end = 0, total = 0;
  bool unsatisfied = false;
};
bool ParseContentRange(std::string_view text, ContentRange& range);
// Seconds or HTTP date. Negative/invalid values fail closed.
bool ParseRetryAfter(std::string_view text, std::int64_t now,
                     std::uint64_t& seconds);
struct DownloadOptions {
  bool resume = true;
  std::function<bool()> cancelled;
  std::function<bool()> check_space;
  std::function<void(std::uint64_t, std::uint64_t)> progress;
  std::function<void(const std::string&)> on_host;
  unsigned max_attempts = 256;
  // Save the resume checkpoint while streaming, every this many bytes, so a
  // stall or crash keeps what was written. Tests may lower it.
  std::uint64_t checkpoint_bytes = 256ULL << 20;
  std::chrono::seconds max_time{24 * 60 * 60};
  // Clock/wait injection has no URL or pin authority; J3 can reuse it.
  std::function<void(std::chrono::seconds)> wait;
  HostPolicy policy;
#ifdef BUBO_PREPARE_TESTING
  std::string origin_override;
#endif
};
struct DownloadResult {
  int code = 0;
  std::string error;
  std::uint64_t bytes = 0;
  std::uint64_t prefix_hash_bytes = 0;
  // Keeps the verified file object protected through handle-based promotion,
  // stamp capture and manifest publication. No extra payload read.
  std::shared_ptr<ReadLease> verified_file;
};
// Only writes <name>.partial and its sidecar. Caller promotes after success.
DownloadResult Download(const Entry&, const std::filesystem::path& partial,
                        Transport&, const DownloadOptions& options = {});
std::uint64_t RetainedPrefix(const Entry&,
                             const std::filesystem::path& partial);
}  // namespace bubo::prepare
