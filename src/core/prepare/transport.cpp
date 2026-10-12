#include "src/core/prepare/transport.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <charconv>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <thread>

namespace bubo::prepare {
namespace {
std::string Lower(std::string_view text) {
  std::string result(text);
  for (auto& c : result)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return result;
}
std::string Trim(std::string_view text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == text.npos)
    return {};
  return std::string(
      text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1));
}
std::string Part(CURLU* url, CURLUPart part) {
  char* raw = nullptr;
  if (curl_url_get(url, part, &raw, 0) != CURLUE_OK)
    return {};
  std::string result(raw);
  curl_free(raw);
  return result;
}
bool Suffix(std::string_view host, std::string_view suffix) {
  return host == suffix ||
         (host.size() > suffix.size() && host.ends_with(suffix) &&
          host[host.size() - suffix.size() - 1] == '.');
}
bool Number(std::string_view value, std::uint64_t& number) {
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), number);
  return !value.empty() && error == std::errc{} &&
         end == value.data() + value.size();
}
}  // namespace
UrlResult CheckUrl(std::string_view text, const HostPolicy& policy) {
  UrlResult result;
  // Reject parser-normalized controls, encoded authority tricks and ambiguous
  // backslashes before curl's standards-compliant URL parsing.
  if (text.empty() || std::ranges::any_of(text, [](unsigned char c) {
        return c <= 32 || c == 127 || c == '\\';
      })) {
    result.error = "unsafe URL";
    return result;
  }
  const auto scheme_end = text.find("://");
  if (scheme_end == text.npos) {
    result.error = "absolute HTTPS URL required";
    return result;
  }
  const auto authority =
      text.substr(scheme_end + 3,
                  text.find_first_of("/?#", scheme_end + 3) - (scheme_end + 3));
  if (authority.find_first_of("@%") != authority.npos) {
    result.error = "userinfo/escaped host forbidden";
    return result;
  }
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(),
                                                          curl_url_cleanup);
  const std::string owned(text);
  if (!url ||
      curl_url_set(url.get(), CURLUPART_URL, owned.c_str(), 0) != CURLUE_OK) {
    result.error = "invalid URL";
    return result;
  }
  result.host = Lower(Part(url.get(), CURLUPART_HOST));
  const auto scheme = Lower(Part(url.get(), CURLUPART_SCHEME));
  bool allowed = scheme == "https" && !result.host.empty() &&
                 (policy.source == HostPolicy::Source::HuggingFace
                      ? Suffix(result.host, "huggingface.co") ||
                            Suffix(result.host, "hf.co")
                      : result.host == "stable.repo.amd.com");
#ifdef BUBO_PREPARE_TESTING
  if (policy.test_loopback && scheme == "http" && result.host == "127.0.0.1")
    allowed = true;
#endif
  if (!allowed || !Part(url.get(), CURLUPART_USER).empty() ||
      !Part(url.get(), CURLUPART_PASSWORD).empty() ||
      !Part(url.get(), CURLUPART_FRAGMENT).empty()) {
    result.error =
        "scheme/host not allowed (HTTPS, reviewed host suffixes only)";
    return result;
  }
  result.url = Part(url.get(), CURLUPART_URL);
  const auto port = Part(url.get(), CURLUPART_PORT);
  if (!port.empty())
    result.host += ":" + port;  // Validator scope includes the serving port.
  return result;
}
UrlResult RedirectUrl(std::string_view current, std::string_view location,
                      unsigned hops, const HostPolicy& policy) {
  if (hops >= 5)
    return {{}, {}, "redirect limit (5 hops)"};
  if (location.empty() || std::ranges::any_of(location, [](unsigned char c) {
        return c <= 32 || c == 127 || c == '\\';
      }))
    return {{}, {}, "invalid redirect Location"};
  if (location.find("://") != location.npos)
    return CheckUrl(location, policy);
  if (location.starts_with("//")) {
    const auto scheme_end = current.find(':');
    if (scheme_end == current.npos)
      return {{}, {}, "invalid redirect base"};
    return CheckUrl(std::string(current.substr(0, scheme_end)) + ":" +
                        std::string(location),
                    policy);
  }
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(),
                                                          curl_url_cleanup);
  const std::string base(current), next(location);
  if (!url ||
      curl_url_set(url.get(), CURLUPART_URL, base.c_str(), 0) != CURLUE_OK ||
      curl_url_set(url.get(), CURLUPART_URL, next.c_str(), 0) != CURLUE_OK)
    return {{}, {}, "invalid redirect URL"};
  return CheckUrl(Part(url.get(), CURLUPART_URL), policy);
}
std::string OriginUrl(const Entry& entry) {
  if (entry.commit.empty() || !IsLowerHex(entry.commit, 40) ||
      entry.source_repo == "PIN_SLOT_BUBO_REPO_ID")
    throw std::invalid_argument("unresolved PIN SLOT; no network request made");
  auto encode = [](std::string_view path) {
    std::string result;
    constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char c : path) {
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
          c == '~' || c == '/')
        result += c;
      else {
        result += '%';
        result += hex[c >> 4];
        result += hex[c & 15];
      }
    }
    return result;
  };
  return "https://huggingface.co/" + encode(entry.source_repo) + "/resolve/" +
         std::string(entry.commit) + "/" + encode(entry.repo_path);
}
bool ParseContentRange(std::string_view text, ContentRange& result) {
  result = {};
  if (!text.starts_with("bytes "))
    return false;
  text.remove_prefix(6);
  const auto slash = text.find('/');
  if (slash == text.npos || !Number(text.substr(slash + 1), result.total))
    return false;
  if (text.substr(0, slash) == "*") {
    result.unsatisfied = true;
    return true;
  }
  const auto dash = text.find('-');
  return dash != text.npos && dash < slash &&
         Number(text.substr(0, dash), result.start) &&
         Number(text.substr(dash + 1, slash - dash - 1), result.end) &&
         result.start <= result.end && result.end < result.total;
}
bool ParseRetryAfter(std::string_view text, std::int64_t now,
                     std::uint64_t& seconds) {
  if (Number(text, seconds))
    return true;
  const std::string value(text);
  const auto date = curl_getdate(value.c_str(), nullptr);
  if (date < 0)
    return false;
  seconds = date <= now ? 0 : static_cast<std::uint64_t>(date - now);
  return true;
}
namespace {
struct CurlContext {
  Response response;
  const HeaderSink* headers;
  const BodySink* body;
  const std::function<bool()>* cancelled;
  bool accept_body = false;
  std::string callback_error;
  std::size_t header_bytes = 0;
};
std::size_t HeaderCallback(char* data, std::size_t size, std::size_t count,
                           void* user) noexcept {
  auto& ctx = *static_cast<CurlContext*>(user);
  try {
    const auto length = size * count;
    ctx.header_bytes += length;
    if (ctx.header_bytes > 64 * 1024)
      throw std::runtime_error("HTTP headers exceed 64 KiB");
    const std::string_view line(data, length);
    if (line.starts_with("HTTP/")) {
      ctx.response = {};
      ctx.accept_body = false;
      const auto space = line.find(' ');
      if (space == line.npos)
        throw std::runtime_error("malformed HTTP status");
      std::uint64_t status;
      if (!Number(line.substr(space + 1, 3), status))
        throw std::runtime_error("malformed HTTP status");
      ctx.response.status = static_cast<int>(status);
    } else if (line == "\r\n" || line == "\n") {
      if (ctx.response.status >= 200) {
        ctx.accept_body = (*ctx.headers)(ctx.response);
        if (!ctx.accept_body)
          return 0;
      }
    } else {
      const auto colon = line.find(':');
      if (colon == line.npos)
        throw std::runtime_error("malformed HTTP header");
      const auto name = Lower(line.substr(0, colon));
      if (!ctx.response.headers.emplace(name, Trim(line.substr(colon + 1)))
               .second &&
          (name == "location" || name == "content-length" ||
           name == "content-range" || name == "content-encoding" ||
           name == "etag" || name == "x-linked-etag" ||
           name == "x-linked-size" || name == "retry-after"))
        throw std::runtime_error("duplicate HTTP security header: " + name);
    }
    return length;
  } catch (const std::exception& error) {
    ctx.callback_error = error.what();
    return 0;
  } catch (...) {
    ctx.callback_error = "HTTP header callback failed";
    return 0;
  }
}
std::size_t BodyCallback(char* data, std::size_t size, std::size_t count,
                         void* user) noexcept {
  auto& ctx = *static_cast<CurlContext*>(user);
  try {
    const auto length = size * count;
    if (ctx.response.status >= 300)
      return length;  // Redirect/error bodies never become payload bytes.
    return ctx.accept_body &&
                   (*ctx.body)(std::span(
                       reinterpret_cast<const std::uint8_t*>(data), length))
               ? length
               : 0;
  } catch (const std::exception& error) {
    ctx.callback_error = error.what();
    return 0;
  } catch (...) {
    ctx.callback_error = "HTTP body callback failed";
    return 0;
  }
}
int ProgressCallback(void* user, curl_off_t, curl_off_t, curl_off_t,
                     curl_off_t) noexcept {
  auto& ctx = *static_cast<CurlContext*>(user);
  try {
    return *ctx.cancelled && (*ctx.cancelled)() ? 1 : 0;
  } catch (...) {
    return 1;
  }
}
}  // namespace
TransferResult CurlTransport::Perform(const Request& request,
                                      const HeaderSink& headers,
                                      const BodySink& body,
                                      const std::function<bool()>& cancelled) {
  const auto checked = CheckUrl(request.url, policy_);
  if (!checked)
    return {checked.error};
  static const auto init = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (init != CURLE_OK)
    return {"cannot initialize libcurl"};
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                           curl_easy_cleanup);
  if (!curl)
    return {"cannot create libcurl handle"};
  curl_slist* raw = nullptr;
  for (const auto& [name, value] : request.headers) {
    if (name.find_first_of("\r\n:") != name.npos ||
        value.find_first_of("\r\n") != value.npos) {
      curl_slist_free_all(raw);
      return {"unsafe request header"};
    }
    const auto key = Lower(name);
    if (key != "accept-encoding" && key != "range" && key != "if-range") {
      curl_slist_free_all(raw);
      return {"request header not allowlisted"};
    }
    auto* next = curl_slist_append(raw, (name + ": " + value).c_str());
    if (!next) {
      curl_slist_free_all(raw);
      return {"cannot allocate curl headers"};
    }
    raw = next;
  }
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> list(
      raw, curl_slist_free_all);
  CurlContext ctx{{}, &headers, &body, &cancelled};
  char errors[CURL_ERROR_SIZE]{};
  bool setup = true;
  const auto set = [&](auto key, auto value) {
    setup &= curl_easy_setopt(curl.get(), key, value) == CURLE_OK;
  };
  set(CURLOPT_URL, checked.url.c_str());
  set(CURLOPT_FOLLOWLOCATION, 0L);
  set(CURLOPT_FAILONERROR, 0L);
  set(CURLOPT_PROXYAUTH, static_cast<long>(CURLAUTH_ANY));
  set(CURLOPT_SSLVERSION, static_cast<long>(CURL_SSLVERSION_TLSv1_2));
  set(CURLOPT_SSL_VERIFYPEER, 1L);
  set(CURLOPT_SSL_VERIFYHOST, 2L);
  set(CURLOPT_CONNECTTIMEOUT_MS, 30000L);
  set(CURLOPT_LOW_SPEED_LIMIT, 1024L);
  set(CURLOPT_LOW_SPEED_TIME, 60L);
  set(CURLOPT_BUFFERSIZE, 512L * 1024L);
  set(CURLOPT_HTTP_CONTENT_DECODING, 0L);
  set(CURLOPT_NOSIGNAL, 1L);
  set(CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
  set(CURLOPT_NETRC, static_cast<long>(CURL_NETRC_IGNORED));
  set(CURLOPT_HTTPHEADER, list.get());
  set(CURLOPT_HEADERFUNCTION, &HeaderCallback);
  set(CURLOPT_HEADERDATA, &ctx);
  set(CURLOPT_WRITEFUNCTION, &BodyCallback);
  set(CURLOPT_WRITEDATA, &ctx);
  set(CURLOPT_XFERINFOFUNCTION, &ProgressCallback);
  set(CURLOPT_XFERINFODATA, &ctx);
  set(CURLOPT_NOPROGRESS, 0L);
  set(CURLOPT_ERRORBUFFER, errors);
#ifdef BUBO_PREPARE_TESTING
  set(CURLOPT_PROTOCOLS_STR, policy_.test_loopback ? "http,https" : "https");
#else
  set(CURLOPT_PROTOCOLS_STR, "https");
#endif
  if (!setup)
    return {"cannot configure libcurl securely"};
  const auto status = curl_easy_perform(curl.get());
  if (!ctx.callback_error.empty())
    return {ctx.callback_error};
  if (status == CURLE_OK)
    return {};
  if (status == CURLE_COULDNT_CONNECT ||
      status == CURLE_COULDNT_RESOLVE_PROXY ||
      status == CURLE_COULDNT_RESOLVE_HOST)
    return {
        "connection/proxy failed; set HTTPS_PROXY=http://host:port and run "
        "again (system PAC settings are not read)"};
  if (status == CURLE_PEER_FAILED_VERIFICATION ||
      status == CURLE_SSL_CONNECT_ERROR || status == CURLE_SSL_CERTPROBLEM)
    return {
        "TLS certificate/revocation validation failed; check Windows "
        "certificate trust, revocation access and any TLS-inspecting proxy "
        "(validation remains enabled)"};
  // Curl's detailed text can contain a presigned URL. Return its bounded enum
  // text only; diagnostic/progress URLs always use the immutable origin.
  return {std::string("libcurl: ") + curl_easy_strerror(status)};
}
}  // namespace bubo::prepare
