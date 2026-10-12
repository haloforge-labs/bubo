#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off: these declarations require Windows types first.
#include <windows.h>
#include <winioctl.h>
#include <psapi.h>
// clang-format on
#endif

#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/transport.hpp"
#include "tests/prepare/fixture_catalogue.hpp"
#include "tests/prepare/loopback_stub.hpp"

namespace {
void Require(bool condition, std::string message) {
  if (!condition)
    throw std::runtime_error(std::move(message));
}
#ifdef _WIN32
using namespace bubo::prepare;
using namespace bubo::prepare::test;
constexpr std::uint64_t kPrefix = (std::uint64_t{1} << 32) + 17;
constexpr std::uint64_t kSuffix = 64;
constexpr std::uint64_t kTotal = kPrefix + kSuffix;
constexpr std::string_view kEtag = "\"large-sparse-fixture-v1\"";

struct ProxyEnvironment {
  std::string previous;
  bool present = false;
  ProxyEnvironment() {
    if (const auto* value = std::getenv("NO_PROXY")) {
      present = true;
      previous = value;
    }
    Require(_putenv_s("NO_PROXY", "127.0.0.1") == 0,
            "cannot set process-local loopback proxy bypass");
  }
  ~ProxyEnvironment() {
    _putenv_s("NO_PROXY", present ? previous.c_str() : "");
  }
};
struct OwnedFiles {
  std::filesystem::path root, partial, sidecar, scratch;
  OwnedFiles() {
    const auto* supplied_root = std::getenv("BUBO_PREPARE_TEST_ROOT");
    const auto base = supplied_root && *supplied_root
                          ? std::filesystem::u8path(supplied_root)
                          : std::filesystem::current_path();
    root =
        base /
        ("prepare-large-resume-" + std::to_string(GetCurrentProcessId()) + "-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(std::filesystem::create_directory(root),
            "cannot create unique sparse fixture directory");
    partial = root / "logical-over4GiB.partial";
    sidecar = partial;
    sidecar += ".json";
    scratch = sidecar;
    scratch += ".new";
  }
  ~OwnedFiles() {
    std::error_code ignored;
    for (const auto& path : {partial, sidecar, scratch})
      std::filesystem::remove(path, ignored);
    // No recursive removal: only the directory and exact files we created.
    std::filesystem::remove(root, ignored);
  }
};
void CreateSparsePrefix(const std::filesystem::path& path) {
  const auto handle =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                  CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  Require(handle != INVALID_HANDLE_VALUE, "cannot create sparse test partial");
  DWORD returned = 0;
  const auto sparse = DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0,
                                      nullptr, 0, &returned, nullptr);
  LARGE_INTEGER end{};
  end.QuadPart = static_cast<LONGLONG>(kPrefix);
  const auto sized = sparse &&
                     SetFilePointerEx(handle, end, nullptr, FILE_BEGIN) &&
                     SetEndOfFile(handle) && FlushFileBuffers(handle);
  const auto error = GetLastError();
  CloseHandle(handle);
  Require(sized,
          "sparse fixture creation failed (requires sparse-capable volume), "
          "Windows error " +
              std::to_string(error));
}
std::uint64_t AllocatedBytes(const std::filesystem::path& path) {
  DWORD high = 0;
  SetLastError(ERROR_SUCCESS);
  const auto low = GetCompressedFileSizeW(path.c_str(), &high);
  Require(low != INVALID_FILE_SIZE || GetLastError() == ERROR_SUCCESS,
          "cannot measure sparse physical allocation");
  return (std::uint64_t{high} << 32) | low;
}
std::string FixtureHash(std::string_view suffix) {
  // Compute an independent synthetic pin through the already linked EVP
  // hasher. No model is opened and no zero-prefix bytes are written to disk.
  std::vector<std::uint8_t> zeroes(4 * 1024 * 1024, 0);
  gufo::crypto::Sha256Hasher hasher;
  for (std::uint64_t done = 0; done < kPrefix;) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(zeroes.size(), kPrefix - done));
    hasher.Update(std::span(zeroes.data(), count));
    done += count;
  }
  hasher.Update(std::span(reinterpret_cast<const std::uint8_t*>(suffix.data()),
                          suffix.size()));
  return hasher.FinishHex();
}
void WriteSidecar(const Entry& entry, const std::filesystem::path& sidecar,
                  std::string payload_host) {
  auto value = metadata::Value::object();
  value["schema_version"] = 1;
  value["repo"] = std::string(entry.source_repo);
  value["commit"] = std::string(entry.commit);
  value["path"] = std::string(entry.repo_path);
  value["expected_size"] = std::to_string(entry.size);
  value["expected_sha256"] = std::string(entry.sha256);
  value["prefix"] = std::to_string(kPrefix);
  value["origin_etag"] = std::string(kEtag);
  value["x_linked_etag"] = std::string(entry.sha256);
  value["payload_etag"] = std::string(kEtag);
  value["payload_host"] = std::move(payload_host);
  const auto text = value.dump() + "\n";
  Require(text.size() <= 64 * 1024, "synthetic sidecar too large");
  std::ofstream out(sidecar, std::ios::binary);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.flush();
  Require(static_cast<bool>(out), "cannot write synthetic identity sidecar");
}
std::string ReadSuffix(const std::filesystem::path& path) {
  const auto handle =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  Require(handle != INVALID_HANDLE_VALUE,
          "cannot inspect completed sparse suffix");
  LARGE_INTEGER position{};
  position.QuadPart = static_cast<LONGLONG>(kPrefix);
  std::string bytes(kSuffix, '\0');
  DWORD count = 0;
  const auto okay = SetFilePointerEx(handle, position, nullptr, FILE_BEGIN) &&
                    ReadFile(handle, bytes.data(),
                             static_cast<DWORD>(bytes.size()), &count, nullptr);
  CloseHandle(handle);
  Require(okay && count == kSuffix,
          "sparse suffix seek/read truncated above4GiB");
  return bytes;
}
void Run() {
  ProxyEnvironment proxy;
  OwnedFiles owned;
  std::string suffix(kSuffix, '\0');
  for (std::size_t i = 0; i < suffix.size(); ++i)
    suffix[i] = static_cast<char>(i);
  const auto begun = std::chrono::steady_clock::now();
  const auto hash = FixtureHash(suffix);
  auto entry = fixture::kCatalogue[2];
  entry.size = kTotal;
  entry.sha256 = hash;
  CreateSparsePrefix(owned.partial);
  const auto allocated_before = AllocatedBytes(owned.partial);
  Require(allocated_before <= 1024 * 1024 &&
              std::filesystem::file_size(owned.partial) == kPrefix,
          "sparse prefix consumed more than1MiB of physical storage");
  std::atomic<std::uint64_t> offered_bytes{0};
  LoopbackStub stub([&](const StubRequest& request) {
    Require(
        request.headers.at("range") == "bytes=" + std::to_string(kPrefix) + "-",
        "Download truncated a Range offset above4GiB");
    Require(request.headers.at("if-range") == kEtag,
            "sparse resume did not use the matching payload validator");
    StubReply reply{206};
    reply.headers["content-range"] = "bytes " + std::to_string(kPrefix) + "-" +
                                     std::to_string(kTotal - 1) + "/" +
                                     std::to_string(kTotal);
    reply.headers["content-length"] = std::to_string(kSuffix);
    reply.headers["etag"] = std::string(kEtag);
    reply.headers["x-linked-etag"] = hash;
    reply.headers["x-linked-size"] = std::to_string(kTotal);
    reply.length = kSuffix;
    reply.body = suffix;
    offered_bytes += kSuffix;
    return reply;
  });
  DownloadOptions options;
  options.policy.test_loopback = true;
  options.origin_override = stub.url(
      "/fixture/bubo/resolve/3333333333333333333333333333333333333333/twin/"
      "twin.gguf");
  options.wait = [](auto) {};
  options.max_attempts = 2;
  options.max_time = std::chrono::seconds(120);
  const auto checked = CheckUrl(options.origin_override, options.policy);
  Require(static_cast<bool>(checked), "sparse loopback origin rejected");
  WriteSidecar(entry, owned.sidecar, checked.host);
  Require(RetainedPrefix(entry, owned.partial) == kPrefix,
          "sidecar did not retain the full64-bit prefix");
  CurlTransport transport(options.policy);
  const auto result = Download(entry, owned.partial, transport, options);
  Require(stub.error().empty(), "sparse stub failure: " + stub.error());
  Require(result.code == 0 && result.bytes == kTotal &&
              result.prefix_hash_bytes == kPrefix &&
              offered_bytes.load() == kSuffix && stub.records().size() == 1,
          "real Download sparse resume failed: " + result.error);
  const auto allocated_after = AllocatedBytes(owned.partial);
  Require(allocated_after <= 1024 * 1024 &&
              std::filesystem::file_size(owned.partial) == kTotal &&
              ReadSuffix(owned.partial) == suffix,
          "sparse resume damaged size/suffix or materialized the prefix");
  PROCESS_MEMORY_COUNTERS memory{};
  memory.cb = sizeof(memory);
  Require(K32GetProcessMemoryInfo(GetCurrentProcess(), &memory,
                                  sizeof(memory)) != 0,
          "sparse fixture working-set measurement unavailable");
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - begun)
                           .count();
  std::cout << "logical_bytes=" << kTotal << " retained_prefix=" << kPrefix
            << " prefix_hash_bytes=" << result.prefix_hash_bytes
            << " http_body_bytes=" << offered_bytes.load()
            << " allocated_before=" << allocated_before
            << " allocated_after=" << allocated_after
            << " fixture_peak_working_set_bytes=" << memory.PeakWorkingSetSize
            << " elapsed_ms=" << elapsed << '\n';
  std::cout << "PASS prepare_large_resume_test: real Curl/Download206 at "
               "offset>4GiB; "
               "streaming SHA verified; physical allocation<=1MiB\n";
}
#endif
}  // namespace

int main() {
#ifdef _WIN32
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL prepare_large_resume_test: " << error.what() << '\n';
    return 1;
  }
#else
  std::cout
      << "SKIP prepare_large_resume_test: Windows sparse-volume fixture\n";
  return 0;
#endif
}
