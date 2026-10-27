#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "src/cli/instance_control.hpp"
#include "src/cli/serve/lifecycle.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <aclapi.h>
#include <sddl.h>
#include <windows.h>
#endif

namespace {
using bubo::cli::ControlError;
using bubo::cli::Value;
unsigned passed = 0;
void Check(bool condition, const char* name) {
  if (!condition)
    throw std::runtime_error(name);
  ++passed;
  std::cout << name << ": PASS\n";
}
template<class F>
void AccessFailure(F&& operation, const char* name) {
  try {
    operation();
  } catch (const ControlError& error) {
    Check(error.ExitCode() == 16, name);
    return;
  }
  throw std::runtime_error(name);
}
#ifdef _WIN32
std::string Utf8(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
std::string TestId() {
  char text[33]{};
  std::snprintf(text, sizeof(text), "%016llx%016llx",
                static_cast<unsigned long long>(GetCurrentProcessId()),
                static_cast<unsigned long long>(GetTickCount64()));
  return text;
}
void OwnerOnly(HANDLE handle, const char* name) {
  PSID owner = nullptr;
  PACL acl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  if (GetSecurityInfo(handle, SE_FILE_OBJECT,
                      OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                      &owner, nullptr, &acl, nullptr,
                      &descriptor) != ERROR_SUCCESS)
    throw std::runtime_error("GetSecurityInfo failed");
  bool valid = owner && acl && acl->AceCount == 1;
  void* raw = nullptr;
  if (valid && GetAce(acl, 0, &raw)) {
    const auto* ace = static_cast<const ACCESS_ALLOWED_ACE*>(raw);
    valid = ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE &&
            EqualSid(owner, const_cast<DWORD*>(&ace->SidStart));
  } else {
    valid = false;
  }
  wchar_t* dump = nullptr;
  if (ConvertSecurityDescriptorToStringSecurityDescriptorW(
          descriptor, SDDL_REVISION_1,
          OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &dump,
          nullptr)) {
    std::wcout << L"ACL " << dump << L'\n';
    LocalFree(dump);
  } else {
    valid = false;
  }
  LocalFree(descriptor);
  Check(valid, name);
}

void WindowsProof() {
  namespace cli = bubo::cli;
  namespace fs = std::filesystem;
  const auto id = TestId();
  const auto identity = cli::CurrentIdentity(id);
  const auto root = fs::current_path() / ("control-unit-" + id);
  if (!fs::create_directory(root))
    throw std::runtime_error("test root already exists");
  const auto root_text = Utf8(root);
  cli::DiscoveryRecord record{
      identity, "http://127.0.0.1:0/v1", "loading", std::nullopt, {}};
  cli::WriteDiscovery(record, root_text);
  const auto read = cli::ReadDiscovery(id, root_text);
  Check(read && read->identity.pid == identity.pid &&
            read->identity.process_creation_time ==
                identity.process_creation_time &&
            read->identity.exe_path == identity.exe_path,
        "discovery_atomic_round_trip");
  Check(cli::DiscoverRecords(root_text).size() == 1,
        "discovery_bounded_directory");
  const auto path = cli::DiscoveryDirectory(root_text) / (id + ".json");
  HANDLE file =
      CreateFileW(path.c_str(), GENERIC_READ | READ_CONTROL, FILE_SHARE_READ,
                  nullptr, OPEN_EXISTING, 0, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open test record");
  OwnerOnly(file, "discovery_owner_only_dacl");
  CloseHandle(file);

  // Match a short-lived discovery poller that denies atomic replacement.
  auto reader = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, 0, nullptr);
  if (reader == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot hold discovery reader");
  record.phase = "ready";
  std::exception_ptr publication_error;
  std::thread publisher([&] {
    try {
      cli::WriteDiscovery(record, root_text);
    } catch (...) {
      publication_error = std::current_exception();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CloseHandle(reader);
  publisher.join();
  if (publication_error)
    std::rethrow_exception(publication_error);
  Check(cli::ReadDiscovery(id, root_text)->phase == "ready",
        "discovery_publication_retries_short_lived_reader");
  reader = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                       OPEN_EXISTING, 0, nullptr);
  if (reader == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot hold discovery deadline reader");
  bool refused = false;
  auto blocked_candidate = record;
  blocked_candidate.phase = "stopping";
  try {
    cli::WriteDiscovery(blocked_candidate, root_text);
  } catch (const cli::ControlError& error) {
    refused = error.ExitCode() == 16;
  }
  CloseHandle(reader);
  Check(refused && cli::ReadDiscovery(id, root_text)->phase == "ready",
        "discovery_publication_deadline_preserves_typed_failure_and_record");

  std::atomic<bool> stopping{false};
  cli::ControlServer server(
      identity,
      [] {
        auto snapshot = Value::object();
        snapshot["phase"] = "loading";
        snapshot["unavailable"] = Value();
        return snapshot;
      },
      [&] { stopping.store(true, std::memory_order_release); });
  server.Start();
  auto instance = cli::VerifiedConnect(record);
  const auto sample = instance.Snapshot();
  Check(sample.member_str("phase") == "loading" && sample.find("unavailable") &&
            sample.find("unavailable")->is_null(),
        "verified_pipe_snapshot");
  const auto pipe_name = cli::ControlPipeName(id);
  HANDLE pipe = INVALID_HANDLE_VALUE;
  const auto connect_deadline = GetTickCount64() + 2000;
  do {
    pipe = CreateFileA(pipe_name.c_str(),
                       GENERIC_READ | GENERIC_WRITE | READ_CONTROL, 0, nullptr,
                       OPEN_EXISTING, 0, nullptr);
    if (pipe != INVALID_HANDLE_VALUE)
      break;
    if (GetLastError() != ERROR_PIPE_BUSY)
      break;
    WaitNamedPipeA(pipe_name.c_str(), 50);
  } while (GetTickCount64() < connect_deadline);
  if (pipe == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open test pipe ACL");
  OwnerOnly(pipe, "pipe_owner_only_dacl");
  auto mismatch = Value::object();
  mismatch["schema_version"] = 1;
  mismatch["control_version"] = 99;
  mismatch["action"] = "handshake";
  mismatch["identity"] = cli::IdentityJson(identity);
  const auto input = mismatch.dump() + "\n";
  DWORD mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
  if (!SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr))
    throw std::runtime_error("cannot configure bounded raw protocol client");
  DWORD request_written = 0;
  if (!WriteFile(pipe, input.data(), static_cast<DWORD>(input.size()),
                 &request_written, nullptr) ||
      request_written != input.size())
    throw std::runtime_error("cannot write mismatch handshake");
  std::string reply;
  const auto reply_deadline = GetTickCount64() + 2000;
  do {
    char buffer[4096]{};
    DWORD read_count = 0;
    if (ReadFile(pipe, buffer, sizeof(buffer), &read_count, nullptr))
      reply.append(buffer, read_count);
    else if (GetLastError() != ERROR_NO_DATA)
      break;
    if (reply.find('\n') != std::string::npos)
      break;
    Sleep(1);
  } while (GetTickCount64() < reply_deadline);
  CloseHandle(pipe);
  const auto error_reply = gufo::json::parse(reply);
  Check(error_reply.member_str("action") == "error" &&
            error_reply.member_size("exit_code") == 15,
        "pipe_protocol_mismatch_preserves_exit_15");

  auto stale = record;
  stale.identity.process_creation_time = "1";
  // Keep the file's actual owner-only ACL and mutate only the identity, as a
  // reused PID would appear in an untrusted discovery hint.
  file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0,
                     nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot write stale test record");
  const auto stale_text = cli::DiscoveryJson(stale).dump() + "\n";
  DWORD count = 0;
  const bool written =
      WriteFile(file, stale_text.data(), static_cast<DWORD>(stale_text.size()),
                &count, nullptr) &&
      count == stale_text.size() && SetEndOfFile(file);
  CloseHandle(file);
  if (!written)
    throw std::runtime_error("cannot publish stale test record");
  const auto stale_read = cli::ReadDiscovery(id, root_text);
  AccessFailure([&] { cli::VerifiedConnect(*stale_read); },
                "pid_reuse_wrong_creation_time_rejected");
  AccessFailure([&] { cli::WriteDiscovery(stale, root_text); },
                "record_publication_requires_own_identity");

  const auto ack = instance.RequestStop();
  Check(ack.find("acknowledged") && ack.find("acknowledged")->as_bool() &&
            ack.find("stopped") && !ack.find("stopped")->as_bool() &&
            !instance.WaitForExit(0),
        "stop_ack_is_separate_from_process_exit");
  // Receipt handling happens on the pipe thread; Shutdown joins that thread
  // after the accepted callback and makes the assertion deterministic.
  server.Shutdown();
  Check(stopping.load(std::memory_order_acquire), "stop_routes_to_callback");

  auto error = Value::object();
  error["code"] = "unexpected_failure";
  error["component"] = "serve";
  error["stage"] = "shutdown";
  error["retryable"] = false;
  error["remedy"] = "secret prompt and path";
  error["api_key"] = "secret";
  cli::WriteTombstone(record, error, root_text);
  const auto tombstone = cli::ReadDiscovery(id, root_text);
  Check(tombstone && tombstone->phase == "stopped" &&
            tombstone->log_path == record.log_path &&
            tombstone->last_error.dump().find("secret") == std::string::npos,
        "tombstone_error_redacted");
  Check(!cli::RecordedStopped(*tombstone),
        "tombstone_never_means_live_process_exited");

  // Delete only the exact files and directories created by this test. CTest
  // runs in its new build directory; no recursive removal is used.
  fs::remove(path);
  fs::remove(cli::DiscoveryDirectory(root_text));
  fs::remove(root / "Bubo");
  fs::remove(root);
}
#endif
}  // namespace

int main() {
  try {
    {
      bubo::cli::ServeLifecycle lifecycle(false);
      lifecycle.Configure("http://::1:0/v1", "fake", Value());
      Check(lifecycle.Snapshot()["endpoints"].is_null(),
            "endpoints_unknown_until_listener_is_bound");
      lifecycle.Configure("http://127.0.0.1:32145/v1", "fake", Value(), {},
                          true);
      auto snapshot = lifecycle.Snapshot();
      Check(snapshot["endpoints"].member_str("openai_base_url") ==
                    "http://127.0.0.1:32145/v1" &&
                snapshot["endpoints"].member_str("anthropic_base_url") ==
                    "http://127.0.0.1:32145",
            "endpoints_ipv4_from_configured_listener");
      lifecycle.Configure("http://[::1]:32146/v1", "fake", Value(), {}, true);
      snapshot = lifecycle.Snapshot();
      Check(snapshot["endpoints"].member_str("openai_base_url") ==
                    "http://[::1]:32146/v1" &&
                snapshot["endpoints"].member_str("anthropic_base_url") ==
                    "http://[::1]:32146",
            "endpoints_ipv6_keep_brackets_and_assigned_port");
      Check(snapshot.member_str("api_key_env") == "BUBO_API_KEY",
            "endpoints_api_key_name_only");
    }
    Check(bubo::cli::ValidInstanceId("0123456789abcdef0123456789abcdef") &&
              !bubo::cli::ValidInstanceId("../record") &&
              !bubo::cli::ValidInstanceId("0123456789ABCDEF0123456789ABCDEF"),
          "instance_id_canonical_and_path_safe");
#ifdef _WIN32
    WindowsProof();
#else
    Check(bubo::cli::ControlPipeName("0123456789abcdef0123456789abcdef") ==
              "\\\\.\\pipe\\Bubo.Control.0123456789abcdef0123456789abcdef",
          "pipe_name_versioned_identity");
#endif
    std::cout << "Control unit: " << passed << " passed, 0 failed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Control unit: " << passed
              << " passed, 1 failed: " << error.what() << '\n';
    return 1;
  }
}
