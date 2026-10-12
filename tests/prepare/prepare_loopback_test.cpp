#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/prepare/loopback_stub.hpp"

#ifdef _WIN32
#define NOMINMAX
// clang-format off: PSAPI declarations require the Windows types first.
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/transport.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

namespace {
using namespace bubo::prepare;
using namespace bubo::prepare::test;
constexpr std::string_view kCommit = "1111111111111111111111111111111111111111";
constexpr std::string_view kRepo = "fixture/huihui";
constexpr std::string_view kRepoPath = "UD-Q4_K_XL/main.gguf";
std::string OriginPath() {
  return "/" + std::string(kRepo) + "/resolve/" + std::string(kCommit) + "/" +
         std::string(kRepoPath);
}
void Require(bool condition, std::string message) {
  if (!condition)
    throw std::runtime_error(std::move(message));
}
struct ProxyEnvironment {
  std::string previous;
  bool present = false;
  ProxyEnvironment() {
    if (const auto* value = std::getenv("NO_PROXY")) {
      present = true;
      previous = value;
    }
#ifdef _WIN32
    _putenv_s("NO_PROXY", "127.0.0.1");
#else
    setenv("NO_PROXY", "127.0.0.1", 1);
#endif
  }
  ~ProxyEnvironment() {
#ifdef _WIN32
    _putenv_s("NO_PROXY", present ? previous.c_str() : "");
#else
    if (present)
      setenv("NO_PROXY", previous.c_str(), 1);
    else
      unsetenv("NO_PROXY");
#endif
  }
};
Entry FixtureEntry(std::uint64_t size, std::string_view hash) {
  return {"fixture_main", "main.gguf", size,      hash,
          kRepo,          kCommit,     kRepoPath, Kind::MainShard,
          "fixture-only", "main.gguf"};
}
DownloadOptions Options(std::string url) {
  DownloadOptions options;
  options.policy.test_loopback = true;
  options.origin_override = std::move(url);
  options.max_attempts = 12;
  options.max_time = std::chrono::seconds(20);
  options.wait = [](auto) {};
  return options;
}
DownloadResult Fetch(const Entry& entry, const std::filesystem::path& partial,
                     const LoopbackStub& stub, std::string_view target,
                     DownloadOptions options = {}) {
  options.policy.test_loopback = true;
  options.origin_override = stub.url(target);
  if (!options.wait)
    options.wait = [](auto) {};
  options.max_time = std::chrono::seconds(20);
  CurlTransport transport(options.policy);
  const auto result = Download(entry, partial, transport, options);
  Require(stub.error().empty(), "native stub error: " + stub.error());
  return result;
}
void Write(const std::filesystem::path& file, std::string_view bytes) {
  std::filesystem::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(out), "fixture write failed");
}
void Pass(std::string_view name, std::chrono::steady_clock::time_point start) {
  std::cout << "PASS " << name << " ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count()
            << " ms)\n";
}

int ResumeChild(int argc, char** argv) {
  Require(argc == 7, "resume-child argument count");
  const std::filesystem::path path = std::filesystem::u8path(argv[2]);
  const std::string url = argv[3], hash = argv[4];
  const auto size = std::stoull(argv[5]);
  const auto prefix = std::stoull(argv[6]);
  auto options = Options(url);
  CurlTransport transport(options.policy);
  const auto result =
      Download(FixtureEntry(size, hash), path, transport, options);
  Require(result.code == 0, "separate-process resume: " + result.error);
  Require(result.prefix_hash_bytes == prefix,
          "separate-process resume did not hash exactly the saved prefix");
  Require(HashFile(path) == hash, "separate-process final file hash mismatch");
  std::cout << "PASS separate-process resume prefix_hash_bytes="
            << result.prefix_hash_bytes << "\n";
  return 0;
}
int SpawnResume(const std::filesystem::path& executable,
                const std::filesystem::path& partial, const std::string& url,
                const std::string& hash, std::uint64_t size,
                std::uint64_t prefix) {
#ifdef _WIN32
  // Every argument is fixture-owned. Quote paths without invoking a shell.
  auto quote = [](const std::wstring& value) {
    Require(value.find(L'"') == value.npos, "unsafe fixture argument");
    return L"\"" + value + L"\"";
  };
  std::wstring command = quote(executable.wstring()) + L" --resume-child " +
                         quote(partial.wstring()) + L" " +
                         quote(std::wstring(url.begin(), url.end())) + L" " +
                         std::wstring(hash.begin(), hash.end()) + L" " +
                         std::to_wstring(size) + L" " + std::to_wstring(prefix);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  Require(CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                         FALSE, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                         nullptr, nullptr, &startup, &process) != 0,
          "cannot spawn separate native resume process");
  CloseHandle(process.hThread);
  const auto waited = WaitForSingleObject(process.hProcess, 30000);
  DWORD code = 1;
  if (waited == WAIT_OBJECT_0)
    GetExitCodeProcess(process.hProcess, &code);
  else
    TerminateProcess(process.hProcess, 1);
  CloseHandle(process.hProcess);
  return static_cast<int>(code);
#else
  const auto pid = fork();
  Require(pid >= 0, "cannot fork separate native resume process");
  if (pid == 0) {
    const auto size_text = std::to_string(size);
    const auto prefix_text = std::to_string(prefix);
    execl(executable.c_str(), executable.c_str(), "--resume-child",
          partial.c_str(), url.c_str(), hash.c_str(), size_text.c_str(),
          prefix_text.c_str(), static_cast<char*>(nullptr));
    _exit(1);
  }
  int status = 0;
  Require(waitpid(pid, &status, 0) == pid, "resume child wait failed");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

void Run(const std::filesystem::path& root,
         const std::filesystem::path& executable) {
  const auto started = std::chrono::steady_clock::now();
  const auto abc = FixtureEntry(3, fixture::kAbcHash);
  const auto fixture_file = root / std::string(kRepo) / std::string(kRepoPath);
  Write(fixture_file, "abc");
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub mirror(root);
    Require(
        mirror.port() != 8081 && mirror.port() != 8082 && mirror.port() != 8083,
        "stub used reserved server port");
    const auto result =
        Fetch(abc, root / "smoke.partial", mirror, OriginPath());
    Require(result.code == 0, "native mirror smoke: " + result.error);
    Require(HashFile(root / "smoke.partial") == fixture::kAbcHash,
            "known-answer ASCII abc digest mismatch");
    Require(mirror.records().size() == 1, "mirror smoke request count");
    const auto request = mirror.records().front();
    Require(request.headers.at("accept-encoding") == "identity",
            "payload encoding was not identity");
    Require(!request.headers.contains("authorization") &&
                !request.headers.contains("cookie"),
            "fixture received credentials");
    Pass("native mirror pinned URL + abc SHA-256", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub payload([&](const StubRequest& request) {
      return FileReply(request, fixture_file);
    });
    LoopbackStub origin([&](const StubRequest&) {
      StubReply reply{302};
      reply.headers["location"] = payload.url("/signed?fixture=only");
      reply.headers["x-linked-etag"] = std::string(fixture::kAbcHash);
      reply.headers["x-linked-size"] = "3";
      return reply;
    });
    Require(origin.port() != payload.port(), "redirect used same endpoint");
    const auto result =
        Fetch(abc, root / "redirect.partial", origin, OriginPath());
    Require(result.code == 0, "two-port redirect: " + result.error);
    Require(origin.records().size() == 1 && payload.records().size() == 1,
            "manual redirect request count");
    Require(payload.records()[0].headers.at("accept-encoding") == "identity",
            "redirect lost identity header");
    Pass("native manual redirect to second allowed loopback endpoint", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub denied([](const StubRequest&) {
      StubReply reply{302};
      // Keep even a regressed policy entirely on loopback. Suffix near-miss
      // mutants are tested without sockets by prepare_transport_test.
      reply.headers["location"] = "http://127.0.0.2:1/never-contact";
      return reply;
    });
    const auto result =
        Fetch(abc, root / "denied.partial", denied, OriginPath());
    Require(result.code == 5 && denied.records().size() == 1,
            "disallowed redirect not stopped before next request");
    Pass("native disallowed redirect fails before external socket", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    for (const auto status : {401, 403}) {
      LoopbackStub denied(
          [status](const StubRequest&) { return StubReply{status}; });
      const auto result =
          Fetch(abc, root / ("status-" + std::to_string(status) + ".partial"),
                denied, OriginPath());
      Require(
          result.code == 5 &&
              result.error.find(std::to_string(status)) != result.error.npos &&
              result.error.find("pinned origin") != result.error.npos,
          "origin auth status lacks precise denial");
    }
    Pass("native 401 and 403 pinned-origin messages", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub retry([&](const StubRequest& request) {
      if (request.number == 1) {
        StubReply reply{429};
        reply.headers["retry-after"] = "2";
        return reply;
      }
      return FileReply(request, fixture_file);
    });
    std::vector<long long> waits;
    auto options = Options(retry.url(OriginPath()));
    options.wait = [&](auto delay) { waits.push_back(delay.count()); };
    const auto result =
        Fetch(abc, root / "retry429.partial", retry, OriginPath(), options);
    Require(result.code == 0 && retry.records().size() == 2 &&
                waits == std::vector<long long>{2},
            "native429 Retry-After not honoured");
    Pass("native 429 Retry-After with injected wait", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub changed([&](const StubRequest& request) {
      auto reply = FileReply(request, fixture_file);
      reply.corrupt_at = 1;
      return reply;
    });
    const auto result =
        Fetch(abc, root / "wrong.partial", changed, OriginPath());
    Require(result.code == 6 && result.bytes == 3 &&
                result.error.find("SHA-256 mismatch") != result.error.npos,
            "wrong content at right size was accepted");
    Require(!std::filesystem::exists(root / "wrong.gguf"),
            "hash mismatch published canonical file");
    Pass("native planted wrong byte at correct size rejected", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub bad_metadata([&](const StubRequest& request) {
      auto reply = FileReply(request, fixture_file);
      reply.headers["x-linked-size"] = "4";
      return reply;
    });
    const auto result =
        Fetch(abc, root / "metadata.partial", bad_metadata, OriginPath());
    Require(result.code == 6 && result.bytes == 0 &&
                std::filesystem::file_size(root / "metadata.partial") == 0,
            "X-Linked mismatch accepted a payload byte");
    Pass("native X-Linked pre-check stops before payload bytes", time);
  }

  // A bounded fixture exercises the one-MiB progress floor without large files.
  std::string large(3 * 1024 * 1024 + 137, '\0');
  for (std::size_t i = 0; i < large.size(); ++i)
    large[i] = static_cast<char>(i % 251);
  const auto hash = gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(large.data()), large.size()));
  const auto entry = FixtureEntry(large.size(), hash);
  const auto large_file = root / "bounded-generated-fixture.bin";
  Write(large_file, large);
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub drops([&](const StubRequest& request) {
      auto reply = FileReply(request, large_file);
      if (request.number == 1)
        reply.drop_after = 1024 * 1024 + 31;
      return reply;
    });
    const auto result =
        Fetch(entry, root / "drop.partial", drops, OriginPath());
    Require(result.code == 0 && result.prefix_hash_bytes == 0,
            "in-process drop retry rehashed saved prefix: " + result.error);
    const auto requests = drops.records();
    Require(requests.size() == 2 && requests[1].headers.contains("range") &&
                requests[1].headers.at("if-range") == "\"fixture-v1\"",
            "drop did not resume with payload validator");
    Require(HashFile(root / "drop.partial") == hash,
            "drop/resume final digest differs");
    Pass("native drop +206 Range/If-Range +live SHA context", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub short_body([&](const StubRequest& request) {
      auto reply = FileReply(request, large_file);
      reply.drop_after = 17;
      return reply;
    });
    const auto result =
        Fetch(entry, root / "short.partial", short_body, OriginPath());
    Require(result.code == 5 && short_body.records().size() == 3 &&
                result.error.find("no-progress budget") != result.error.npos,
            "short-body no-progress limit failed");
    Pass("native repeated short bodies stop at progress budget", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub validator([&](const StubRequest& request) {
      auto reply = FileReply(request, large_file,
                             request.number == 1 ? "\"old\"" : "\"new\"");
      if (request.number == 1)
        reply.drop_after = 1024 * 1024 + 7;
      return reply;
    });
    const auto result =
        Fetch(entry, root / "validator.partial", validator, OriginPath());
    const auto requests = validator.records();
    Require(result.code == 0 && requests.size() == 3 &&
                requests[1].headers.contains("range") &&
                !requests[2].headers.contains("range"),
            "changed validator did not restart from bytezero: " + result.error);
    Pass("native validator change resets partial", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    unsigned origin_count = 0;
    LoopbackStub payload([&](const StubRequest& request) {
      if (request.target.starts_with("/expired") && request.number > 1)
        return StubReply{403};
      auto reply = FileReply(request, large_file);
      if (request.number == 1)
        reply.drop_after = 1024 * 1024 + 11;
      return reply;
    });
    LoopbackStub origin([&](const StubRequest&) {
      StubReply reply{302};
      reply.headers["location"] =
          payload.url(++origin_count < 3 ? "/expired?token=fixture"
                                         : "/fresh?token=fixture");
      reply.headers["x-linked-etag"] = hash;
      reply.headers["x-linked-size"] = std::to_string(entry.size);
      return reply;
    });
    const auto result =
        Fetch(entry, root / "expiry.partial", origin, OriginPath());
    Require(result.code == 0 && origin.records().size() == 3 &&
                payload.records().size() == 3,
            "expired URL did not re-resolve pinned origin: " + result.error);
    Require(HashFile(root / "expiry.partial") == hash,
            "URL expiry/resume digest differs");
    Pass("native expiring payload403 re-resolves origin", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub resume([&](const StubRequest& request) {
      return FileReply(request, large_file);
    });
    bool cancel = false;
    auto options = Options(resume.url(OriginPath()));
    options.cancelled = [&] { return cancel; };
    options.progress = [&](auto bytes, auto) { cancel = bytes >= 1024 * 1024; };
    const auto partial = root / "process.partial";
    const auto first = Fetch(entry, partial, resume, OriginPath(), options);
    Require(first.code == 3 && first.bytes > 0 && first.bytes < entry.size,
            "cancellation did not preserve incomplete partial");
    const auto retained = RetainedPrefix(entry, partial);
    Require(retained == first.bytes,
            "sidecar checkpoint differs from flushed prefix");
    Require(SpawnResume(executable, partial, resume.url(OriginPath()), hash,
                        entry.size, retained) == 0,
            "native new-process partial reopen failed");
    Pass("native cancellation +new-process prefix rehash/resume", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    const auto partial = root / "complete416.partial";
    LoopbackStub full([&](const StubRequest& request) {
      return FileReply(request, fixture_file);
    });
    auto first = Fetch(abc, partial, full, OriginPath());
    Require(first.code == 0, "initial416 fixture transfer failed");
    // A later invocation starts after its predecessor releases the protected
    // verified writer. Keeping it open would correctly reject a second writer.
    first.verified_file.reset();
    const auto second = Fetch(abc, partial, full, OriginPath());
    Require(second.code == 0 && second.prefix_hash_bytes == 3 &&
                full.records().back().headers.at("range") == "bytes=3-",
            "complete partial416 hash proof failed: " + second.error);
    Pass("native416 complete retained prefix validates SHA-256", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    LoopbackStub bad_range([&](const StubRequest& request) {
      auto reply = FileReply(request, large_file);
      if (request.number == 1)
        reply.drop_after = 1024 * 1024 + 1;
      else
        reply.headers["content-range"] = "bytes 0-7/8";
      return reply;
    });
    const auto result =
        Fetch(entry, root / "bad-range.partial", bad_range, OriginPath());
    Require(result.code == 6 && bad_range.records().size() == 2,
            "bad206 Content-Range accepted");
    Pass("native malformed206 range rejected", time);
  }
  {
    const auto time = std::chrono::steady_clock::now();
    constexpr std::uint64_t size = (std::uint64_t{1} << 32) + 97;
    constexpr std::uint64_t suffix = 64;
    LoopbackStub generated([](const StubRequest& request) {
      return GeneratedReply(request, size);
    });
    HostPolicy policy;
    policy.test_loopback = true;
    CurlTransport transport(policy);
    Response headers;
    std::vector<std::uint8_t> bytes;
    const auto result = transport.Perform(
        {generated.url("/generated-over4GiB"),
         {{"accept-encoding", "identity"},
          {"range", "bytes=" + std::to_string(size - suffix) + "-"}}},
        [&](const Response& response) {
          headers = response;
          return true;
        },
        [&](auto part) {
          bytes.insert(bytes.end(), part.begin(), part.end());
          return bytes.size() <= suffix;
        },
        {});
    ContentRange range;
    Require(result.error.empty() && headers.status == 206 &&
                ParseContentRange(headers.headers.at("content-range"), range) &&
                range.total == size && range.start == size - suffix &&
                bytes.size() == suffix,
            "over4GiB generated suffix response failed");
    for (std::size_t i = 0; i < bytes.size(); ++i)
      Require(bytes[i] == (size - suffix + i) % 251,
              "over4GiB suffix offset truncated");
    std::cout << "logical_large_file_bytes=" << size
              << " transferred_bytes=" << bytes.size() << " disk_bytes=0\n";
    Pass("native>4GiB logical file only64-byte range", time);
  }
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  Require(K32GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                                  sizeof(counters)) != 0,
          "fixture working-set measurement unavailable");
  std::cout << "fixture_peak_working_set_bytes=" << counters.PeakWorkingSetSize
            << " stub_stream_buffer_bytes=65536 fixture_payload_bytes="
            << large.size() << '\n';
#endif
  Pass("prepare_loopback_test", started);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    ProxyEnvironment proxy;
    if (argc > 1 && std::string_view(argv[1]) == "--resume-child")
      return ResumeChild(argc, argv);
    const auto* fixture_root = std::getenv("BUBO_PREPARE_TEST_ROOT");
    Require(
        fixture_root && *fixture_root,
        "set BUBO_PREPARE_TEST_ROOT to an authorized job fixture directory");
    const auto root =
        std::filesystem::u8path(fixture_root) / "loopback-fixtures";
    Require(!std::filesystem::exists(root),
            "loopback fixture directory already exists; use a fresh job "
            "fixture root");
    std::filesystem::create_directories(root);
#ifdef _WIN32
    std::wstring module(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, module.data(),
                                           static_cast<DWORD>(module.size()));
    Require(length != 0 && length < module.size(),
            "test executable path unavailable");
    module.resize(length);
    const std::filesystem::path executable(module);
#else
    const auto executable = std::filesystem::absolute(argv[0]);
#endif
    Run(root, executable);
    // Keep the tiny fixtures for reviewer evidence; no production paths
    // touched.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL prepare_loopback_test: " << error.what() << '\n';
    return 1;
  }
}
