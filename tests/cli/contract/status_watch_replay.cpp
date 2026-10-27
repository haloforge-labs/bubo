// CPU-only bridge from a loopback /slots replay to the real private control
// protocol. Production bubo status discovers and verifies this exact process.
#include <curl/curl.h>
#include <windows.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/instance_control.hpp"

namespace {
BOOL WINAPI KeepLauncherAlive(DWORD event) {
  return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

int ConsoleRun(bool block_output) {
  // ConPTY attachment alone does not replace redirected parent standard
  // handles. Supply real read/write console handles, as a terminal does.
  SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
  const HANDLE input = CreateFileW(L"CONIN$", GENERIC_READ | GENERIC_WRITE,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &security, OPEN_EXISTING, 0, nullptr);
  const HANDLE output = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                                    &security, OPEN_EXISTING, 0, nullptr);
  if (input == INVALID_HANDLE_VALUE || output == INVALID_HANDLE_VALUE)
    return 1;
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = input;
  startup.hStdOutput = startup.hStdError = output;
  HANDLE unread = nullptr, blocked = nullptr;
  if (block_output) {
    if (!CreatePipe(&unread, &blocked, &security, 4096))
      return 1;
    SetHandleInformation(unread, HANDLE_FLAG_INHERIT, 0);
    // Arrange backpressure before starting the real CLI; no invented slots
    // or request counts are needed to fill its stdout pipe.
    const std::string padding(4096, ' ');
    DWORD written = 0;
    if (!WriteFile(blocked, padding.data(), static_cast<DWORD>(padding.size()),
                   &written, nullptr) ||
        written != padding.size())
      return 1;
    startup.hStdOutput = blocked;
  }
  PROCESS_INFORMATION child{};
  const std::wstring line = GetCommandLineW();
  const std::wstring marker =
      block_output ? L" --console-block-output " : L" --console-run ";
  const auto start = line.find(marker);
  if (start == std::wstring::npos)
    return 2;
  const auto command = line.substr(start + marker.size());
  std::vector<wchar_t> mutable_command(command.begin(), command.end());
  mutable_command.push_back(L'\0');
  SetConsoleCtrlHandler(KeepLauncherAlive, TRUE);
  const BOOL started =
      CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, TRUE, 0,
                     nullptr, nullptr, &startup, &child);
  CloseHandle(input);
  CloseHandle(output);
  if (!started)
    return 1;
  WaitForSingleObject(child.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(child.hProcess, &code);
  CloseHandle(child.hThread);
  CloseHandle(child.hProcess);
  if (unread)
    CloseHandle(unread);
  if (blocked)
    CloseHandle(blocked);
  return static_cast<int>(code);
}

std::size_t Receive(char* data, std::size_t size, std::size_t count,
                    void* out) {
  static_cast<std::string*>(out)->append(data, size * count);
  return size * count;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && (std::string_view(argv[1]) == "--console-run" ||
                    std::string_view(argv[1]) == "--console-block-output"))
    return ConsoleRun(std::string_view(argv[1]) == "--console-block-output");
  if (argc != 4)
    return 2;
  try {
    const std::string url =
        "http://127.0.0.1:" + std::string(argv[2]) + "/slots";
    const auto identity = bubo::cli::CurrentIdentity(argv[3]);
    bubo::cli::DiscoveryRecord record{identity, url, "ready", {}, {}};
    bubo::cli::WriteDiscovery(record, argv[1]);
    bubo::cli::ControlServer server(
        identity,
        [&] {
          std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
              curl_easy_init(), curl_easy_cleanup);
          if (!curl)
            throw std::runtime_error("curl initialization failed");
          std::string response;
          curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
          curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
          curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, 1000L);
          curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, Receive);
          curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response);
          if (curl_easy_perform(curl.get()) != CURLE_OK)
            throw std::runtime_error("slots replay failed");
          auto snapshot = bubo::cli::Value::object();
          auto replay = gufo::json::parse(response);
          if (const auto* stall = replay.find("stall_ms");
              stall && stall->is_number())
            Sleep(static_cast<DWORD>(stall->as_double()));
          snapshot["phase"] =
              replay.is_object() ? replay.member_str("phase") : "ready";
          snapshot["slots"] =
              replay.is_object() ? replay["slots"] : std::move(replay);
          snapshot["endpoint"] = url;
          snapshot["readiness"] = bubo::cli::Value::object();
          snapshot["readiness"]["server"] =
              snapshot.member_str("phase") == "ready";
          return snapshot;
        },
        [] {});
    server.Start();
    std::cout << bubo::cli::IdentityJson(identity).dump() << std::endl;
    std::string line;
    std::getline(std::cin, line);
    server.Shutdown();
    bubo::cli::WriteTombstone(record, {}, argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
