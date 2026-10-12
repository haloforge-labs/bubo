// SPDX-License-Identifier: MIT
#include <Windows.h>

#include <fstream>
#include <iostream>

#include "src/cli/serve/release_profile.hpp"

namespace {
[[noreturn]] void Counter(const std::string& path) {
  // Reserved expected-abort control; no XRT, HIP, SDK, model or server calls.
  HANDLE file =
      CreateFileA(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                  OPEN_ALWAYS, FILE_FLAG_WRITE_THROUGH, nullptr);
  const char record[] = "factory=1\n";
  DWORD bytes = 0;
  if (file == INVALID_HANDLE_VALUE ||
      !WriteFile(file, record, sizeof(record) - 1, &bytes, nullptr) ||
      bytes != sizeof(record) - 1 || !FlushFileBuffers(file))
    ExitProcess(84);
  CloseHandle(file);
  std::cout << "phase=factory counter=1 expected_exit=83\n" << std::flush;
  ExitProcess(83);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4)
    return 85;
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
  const std::string entry = argv[1], counter = argv[2];
  std::ifstream input(argv[3]);
  const auto text = std::string(std::istreambuf_iterator<char>(input), {});
  std::map<std::string, std::string> manifest;
  const auto document = gufo::json::parse(text);
  for (const auto& [name, value] : document.members())
    manifest.emplace(name, value.str());
  std::cout << "phase=cli manifest-read\n" << std::flush;
  // Planted variants prove the process oracle catches the original DIR enabler
  // and a removed creation guard. These paths exist ONLY in this CPU host.
  if (entry == "variant-unguarded" ||
      (entry == "variant-dir-enabler" && std::getenv("GUFO_NPU_DIR")))
    Counter(counter);
  auto startup = gufo::server::PrepareReleaseSettings(
      std::span<const char* const>(argv + 4, argc - 4), manifest);
  if (!startup.error.empty()) {
    std::cout << "phase=release-deny " << startup.error << '\n';
    return 2;
  }
  if (!startup.active || startup.resolved.architecture != "qwen4exp") {
    std::cout << "phase=wrong-model-header\n";
    return 86;
  }
  std::cout << "phase=resolved "
            << gufo::server::ReleaseSettingsJson(startup.resolved).dump()
            << '\n';
  if (entry == "noncli") {
    // A second caller resolves with no explicit flags, hostile manifest/env.
    gufo::server::ReleaseInputs inputs;
    inputs.profile = startup.resolved.profile;
    inputs.model_family = "qwen3.8-flash-next";
    inputs.manifest_defaults = manifest;
    inputs.environment = gufo::server::SnapshotReleaseEnvironment();
    startup.resolved = gufo::server::ResolveReleaseSettings(inputs);
  }
  std::cout << "phase=model-create effective="
            << startup.resolved.effective_flags.at("--npu-prefill") << '\n'
            << std::flush;
  gufo::core::CreateAuthorizedPrefill(
      startup.resolved.prefill_authorization,
      [&](const gufo::core::ExperimentalAuthorization&) { Counter(counter); });
  std::cout << "phase=model-ready factory=0\n";
  return 0;
}
