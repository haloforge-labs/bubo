#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/transport.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

namespace {
using namespace bubo::prepare;
void Require(bool value, std::string message) {
  if (!value)
    throw std::runtime_error(std::move(message));
}
struct Hop {
  Response response;
  std::string body, error;
  bool no_headers = false;
  std::function<void(const Request&)> inspect;
};
class Fake final : public Transport {
public:
  std::vector<Hop> hops;
  std::size_t calls = 0, offered_bytes = 0;
  TransferResult Perform(const Request& request, const HeaderSink& headers,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    Require(calls < hops.size(), "unexpected extra transport attempt");
    const auto& hop = hops[calls++];
    if (hop.inspect)
      hop.inspect(request);
    if (hop.no_headers)
      return {hop.error};
    if (!headers(hop.response))
      return {"header rejected"};
    if (hop.response.status < 300 && !hop.body.empty()) {
      offered_bytes += hop.body.size();
      if (!body(
              std::span(reinterpret_cast<const std::uint8_t*>(hop.body.data()),
                        hop.body.size())))
        return {"body rejected"};
    }
    return {hop.error};
  }
};
Hop Full(std::string body = "abc") {
  return {{200,
           {{"content-length", "3"},
            {"etag", "\"payload\""},
            {"x-linked-etag", std::string(fixture::kAbcHash)},
            {"x-linked-size", "3"}}},
          std::move(body)};
}
Hop Remaining() {
  return {{206,
           {{"content-length", "2"},
            {"content-range", "bytes 1-2/3"},
            {"etag", "\"payload\""},
            {"x-linked-etag", std::string(fixture::kAbcHash)},
            {"x-linked-size", "3"}}},
          "bc"};
}
Hop Redirect() {
  return {{302,
           {{"location", "https://cdn.hf.co/signed?private=NEVER_PERSIST"},
            {"x-linked-etag", std::string(fixture::kAbcHash)},
            {"x-linked-size", "3"}}}};
}
DownloadOptions Options() {
  DownloadOptions options;
  options.wait = [](auto) {};
  options.max_attempts = 8;
  options.max_time = std::chrono::seconds(10);
  return options;
}
struct OwnedFiles {
  std::filesystem::path root;
  std::vector<std::filesystem::path> files;
  OwnedFiles() {
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = 0;
#endif
    root = std::filesystem::current_path() /
           ("prepare-retry-policy-" + std::to_string(pid) + "-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(std::filesystem::create_directory(root),
            "test directory already exists");
  }
  std::filesystem::path Path(std::string name) {
    auto path = root / std::move(name);
    files.push_back(path);
    auto sidecar = path;
    sidecar += ".json";
    files.push_back(sidecar);
    sidecar += ".new";
    files.push_back(sidecar);
    return path;
  }
  ~OwnedFiles() {
    std::error_code ignored;
    for (const auto& file : files)
      std::filesystem::remove(file, ignored);
    // Remove only the empty directory we created; never recurse into it.
    std::filesystem::remove(root, ignored);
  }
};
std::string Read(const std::filesystem::path& path) {
#ifdef _WIN32
  const auto handle =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  Require(handle != INVALID_HANDLE_VALUE,
          "cannot inspect leased fixture prefix");
  LARGE_INTEGER size{};
  const auto measured = GetFileSizeEx(handle, &size);
  if (!measured || size.QuadPart < 0 || size.QuadPart > 64 * 1024) {
    CloseHandle(handle);
    throw std::runtime_error("fixture inspection exceeds64KiB bound");
  }
  std::string text(static_cast<std::size_t>(size.QuadPart), '\0');
  DWORD count = 0;
  const auto read = ReadFile(handle, text.data(),
                             static_cast<DWORD>(text.size()), &count, nullptr);
  CloseHandle(handle);
  Require(read && count == text.size(), "cannot read leased fixture prefix");
  return text;
#else
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "cannot inspect fixture prefix");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
#endif
}
void Seed(const Entry& entry, const std::filesystem::path& path) {
  bool cancel = false;
  auto options = Options();
  options.cancelled = [&] { return cancel; };
  options.progress = [&](auto, auto) { cancel = true; };
  Fake fake;
  fake.hops = {Full("a")};
  const auto result = Download(entry, path, fake, options);
  Require(result.code == 3 && result.bytes == 1 &&
              RetainedPrefix(entry, path) == 1 && Read(path) == "a",
          "cannot seed a flushed one-byte retained prefix");
}
void ResumeRange(const Request& request, const std::filesystem::path& path) {
  Require(request.headers.at("range") == "bytes=1-",
          "retained prefix was discarded");
  Require(Read(path) == "a", "saved prefix was rewritten before resume");
}
}  // namespace

int main() {
  try {
    const auto entry = fixture::kCatalogue[2];
    OwnedFiles owned;
    {
      const auto path = owned.Path("429.partial");
      Seed(entry, path);
      Fake fake;
      auto retry = Hop{{429, {{"retry-after", "2"}}}};
      retry.inspect = [&](const Request& request) {
        ResumeRange(request, path);
      };
      auto remaining = Remaining();
      remaining.inspect = retry.inspect;
      fake.hops = {retry, remaining};
      std::vector<long long> waits;
      auto options = Options();
      options.wait = [&](auto delay) {
        Require(Read(path) == "a", "429 truncated retained prefix before wait");
        waits.push_back(delay.count());
      };
      const auto result = Download(entry, path, fake, options);
      Require(result.code == 0 && fake.calls == 2 &&
                  result.prefix_hash_bytes == 1 &&
                  waits == std::vector<long long>{2} &&
                  HashFile(path) == fixture::kAbcHash,
              "resumed429 lost prefix, ignoredRetry-After, or failed digest");
      std::cout << "PASS resumed origin429 preserves prefix and honours "
                   "Retry-After2\n";
    }
    for (const int status : {401, 403}) {
      const auto path = owned.Path(std::to_string(status) + ".partial");
      Seed(entry, path);
      Fake fake;
      fake.hops = {{{status, {}}}};
      const auto result = Download(entry, path, fake, Options());
      Require(result.code == 5 && fake.calls == 1 && result.bytes == 1 &&
                  RetainedPrefix(entry, path) == 1 && Read(path) == "a" &&
                  result.error.find("pinned origin") != std::string::npos,
              "resumed originauth denial retried or destroyed retained prefix");
    }
    std::cout << "PASS resumed origin401/403 terminal on first response; "
                 "prefix preserved\n";
    {
      const auto path = owned.Path("redirect-connect.partial");
      Seed(entry, path);
      Fake fake;
      auto redirect = Redirect();
      redirect.inspect = [&](const Request& request) {
        ResumeRange(request, path);
      };
      auto remaining = Remaining();
      remaining.response.headers.erase("x-linked-etag");
      remaining.response.headers.erase("x-linked-size");
      remaining.inspect = redirect.inspect;
      Hop connect_failure;
      connect_failure.no_headers = true;
      connect_failure.error = "fixture connect failure before response headers";
      fake.hops = {redirect, connect_failure, redirect, remaining};
      const auto result = Download(entry, path, fake, Options());
      Require(
          result.code == 0 && fake.calls == 4 &&
              result.prefix_hash_bytes == 1 &&
              HashFile(path) == fixture::kAbcHash,
          "postredirect connection failure did not re-resolve pinned origin");
      auto sidecar = path;
      sidecar += ".json";
      Require(Read(sidecar).find("NEVER_PERSIST") == std::string::npos,
              "signed URL leaked into persisted metadata");
      std::cout << "PASS redirect payload-connect failure re-resolves; prefix "
                   "hashed once\n";
    }
    {
      unsigned index = 0;
      for (const auto& malformed :
           {std::string("not-a-SHA256"),
            std::string("W/\"") + std::string(fixture::kAbcHash) + "\"",
            std::string(63, 'a'), std::string(64, 'z')}) {
        const auto path =
            owned.Path("linked-" + std::to_string(index++) + ".partial");
        Fake fake;
        auto hop = Full();
        hop.response.headers["x-linked-etag"] = malformed;
        fake.hops = {hop};
        const auto result = Download(entry, path, fake, Options());
        Require(
            result.code == 6 && fake.offered_bytes == 0 && result.bytes == 0,
            "malformed/weak X-Linked-Etag accepted payload bytes");
      }
      std::cout
          << "PASS malformed/weak X-Linked-Etag rejected before payload\n";
    }
    {
      for (const bool at_origin : {true, false}) {
        const auto path = owned.Path(at_origin ? "origin-etag.partial"
                                               : "payload-etag.partial");
        Fake fake;
        auto large = Full();
        large.response.headers["etag"] = "\"" + std::string(1023, 'e') + "\"";
        if (at_origin) {
          fake.hops = {large};
        } else {
          large.response.headers.erase("x-linked-etag");
          large.response.headers.erase("x-linked-size");
          fake.hops = {Redirect(), large};
        }
        const auto result = Download(entry, path, fake, Options());
        Require(
            result.code != 0 && fake.offered_bytes == 0 && result.bytes == 0,
            "oversized origin/payload validator accepted payload bytes");
        auto sidecar = path;
        sidecar += ".json";
        Require(std::filesystem::file_size(sidecar) <= 64 * 1024,
                "partial sidecar exceeds64KiB cap");
      }
      std::cout
          << "PASS oversized origin/payload ETags rejected; sidecars bounded\n";
    }
    {
      const auto path = owned.Path("invalid-length.partial");
      Fake fake;
      auto malformed = Full();
      malformed.response.headers["content-length"] = "oops";
      fake.hops = {malformed};
      const auto result = Download(entry, path, fake, Options());
      Require(result.code == 6 && fake.offered_bytes == 0,
              "malformed Content-Length must produce integrity exit6");
      std::cout << "PASS malformed Content-Length exits6 before payload\n";
    }
    {
      const auto path = owned.Path("deadline.partial");
      Fake fake;
      auto options = Options();
      options.max_time = std::chrono::seconds(0);
      const auto result = Download(entry, path, fake, options);
      Require(result.code == 5 && fake.calls == 0,
              "expired per-file cap entered transport");
      std::cout
          << "PASS elapsed per-file deadline makes zero transport calls\n";
    }
#ifdef _WIN32
    for (const bool resume : {true, false}) {
      for (const bool sidecar_input : {false, true}) {
        const auto path = owned.Path(
            std::string("offline-") + (sidecar_input ? "marker-" : "partial-") +
            (resume ? "resume.partial" : "restart.partial"));
        Seed(entry, path);
        auto sidecar = path;
        sidecar += ".json";
        const auto prefix = Read(path), marker = Read(sidecar);
        const auto prefix_time = std::filesystem::last_write_time(path);
        const auto marker_time = std::filesystem::last_write_time(sidecar);
        const auto input = sidecar_input ? sidecar : path;
        {
          const auto attributes = GetFileAttributesW(input.c_str());
          Require(attributes != INVALID_FILE_ATTRIBUTES,
                  "cannot inspect owned offline fixture attributes");
          struct RestoreAttributes {
            std::filesystem::path path;
            DWORD attributes;
            ~RestoreAttributes() {
              SetFileAttributesW(path.c_str(), attributes);
            }
          } restore{input, attributes};
          Require(
              SetFileAttributesW(input.c_str(),
                                 attributes | FILE_ATTRIBUTE_OFFLINE) &&
                  (GetFileAttributesW(input.c_str()) & FILE_ATTRIBUTE_OFFLINE),
              "cannot mark owned native fixture offline");
          bool rejected = false;
          try {
            (void)RetainedPrefix(entry, path);
          } catch (const std::exception& error) {
            rejected =
                std::string_view(error.what()).find("Use a local folder") !=
                std::string_view::npos;
          }
          Require(rejected,
                  "RetainedPrefix read or swallowed an offline input");
          Fake fake;
          auto options = Options();
          options.resume = resume;
          const auto result = Download(entry, path, fake, options);
          Require(
              result.code == 7 && fake.calls == 0 &&
                  result.error.find("Use a local folder") != std::string::npos,
              "offline resume input was opened or entered transport");
        }
        // The fixture restores its owned attribute before inspecting data;
        // the production calls must leave both existing byte streams intact.
        Require(Read(path) == prefix && Read(sidecar) == marker &&
                    std::filesystem::last_write_time(path) == prefix_time &&
                    std::filesystem::last_write_time(sidecar) == marker_time,
                "offline refusal changed the partial or sidecar");
      }
    }
    std::cout << "PASS native offline partial/sidecar refused before reads or "
                 "writes; resume/restart; zero transport calls\n";
    {
      const auto path = owned.Path("read-lease.partial");
      Seed(entry, path);
      bool attempted = false, denied = false;
      auto options = Options();
      options.cancelled = [&] {
        if (!attempted) {
          attempted = true;
          const auto handle = CreateFileW(
              path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
          denied = handle == INVALID_HANDLE_VALUE &&
                   GetLastError() == ERROR_SHARING_VIOLATION;
          if (handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
        }
        return false;
      };
      Fake fake;
      fake.hops = {Remaining()};
      const auto result = Download(entry, path, fake, options);
      Require(attempted && denied && result.code == 0 &&
                  result.prefix_hash_bytes == 1 &&
                  HashFile(path) == fixture::kAbcHash,
              "partial could be mutated while saved prefix was hashed");
      std::cout << "PASS writer lease denies mutation during prefix hashing\n";
    }
#endif
    std::cout << "PASS prepare_retry_policy_test\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL prepare_retry_policy_test: " << error.what() << '\n';
    return 1;
  }
}
