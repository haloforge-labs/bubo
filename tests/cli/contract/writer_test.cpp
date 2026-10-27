#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#include "src/cli/machine_output.hpp"

int main(int argc, char** argv) {
  using namespace bubo::cli;
  if (argc == 2) {
    if (std::string_view(argv[1]) == "--hold") {
      std::this_thread::sleep_for(std::chrono::seconds(4));
      return 0;
    }
    MachineWriter native(std::cout);
    if (std::string_view(argv[1]) == "--blocked" ||
        std::string_view(argv[1]) == "--closed") {
      auto payload = Value::object();
      payload["message"] = std::string(512 * 1024, 'x');
      const bool first = native.Write("warning", payload);
      // A failed stream stays disabled; retries must return promptly.
      return !first && !native.Write("warning", payload) ? 0 : 1;
    }
#ifdef _WIN32
    if (std::string_view(argv[1]) == "--child") {
      wchar_t executable[32768];
      if (!GetModuleFileNameW(nullptr, executable, 32768))
        return 1;
      auto command = L"\"" + std::wstring(executable) + L"\" --hold";
      STARTUPINFOW startup{};
      startup.cb = sizeof(startup);
      PROCESS_INFORMATION child{};
      if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                          CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                          nullptr, nullptr, &startup, &child))
        return 1;
      std::cerr << "holder_pid=" << child.dwProcessId << '\n' << std::flush;
      CloseHandle(child.hThread);
      CloseHandle(child.hProcess);
      return native.Write("capabilities", Capabilities()) ? 0 : 1;
    }
#endif
    std::cout << "stream noise\n" << std::flush;
    std::fprintf(stdout, "library noise\n");
    std::fflush(stdout);
#ifdef _WIN32
    DWORD count = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), "OS noise\n", 9, &count,
              nullptr);
#endif
    return native.Write("capabilities", Capabilities()) ? 0 : 1;
  }
  std::ostringstream output;
  MachineWriter writer(output);
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i)
    workers.emplace_back([&] {
      for (int j = 0; j < 32; ++j) {
        auto payload = Value::object();
        payload["message"] = "UTF-8: é\n\"\\";
        writer.Write("warning", payload);
      }
    });
  for (auto& worker : workers)
    worker.join();
  std::istringstream input(output.str());
  std::string line;
  unsigned count = 0;
  while (std::getline(input, line)) {
    const auto row = gufo::json::parse(line);
    if (row.member_size("seq") != ++count ||
        row.member_size("schema_version") != 1 ||
        row.member_str("operation_id") != writer.OperationId() ||
        row.member_str("time").back() != 'Z')
      return 1;
  }
  if (count != 128 || output.str().find('\r') != std::string::npos)
    return 1;
  std::cout << "Serializer: 128 passed, 0 failed\n";
}
