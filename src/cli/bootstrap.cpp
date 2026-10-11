#include <windows.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/brand.hpp"
#include "src/cli/cli_info.hpp"
#include "src/cli/prepare/prepare.hpp"

namespace {
std::string Utf8(std::wstring_view text) {
  if (text.empty())
    return {};
  const int count = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
      nullptr, 0, nullptr, nullptr);
  if (!count)
    throw std::runtime_error("invalid UTF-16 argument");
  std::string out(count, '\0');
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                           static_cast<int>(text.size()), out.data(), count,
                           nullptr, nullptr))
    throw std::runtime_error("UTF-8 conversion failed");
  return out;
}
std::string LocalAppData() {
  const auto count = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
  if (!count)
    return {};
  std::wstring value(count, L'\0');
  const auto length =
      GetEnvironmentVariableW(L"LOCALAPPDATA", value.data(), count);
  if (!length || length >= count)
    return {};
  value.resize(length);
  return Utf8(value);
}
std::filesystem::path EnginePath() {
  std::wstring name(32768, L'\0');
  const auto length =
      GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
  if (!length || length == name.size())
    throw std::runtime_error("cannot locate bubo.exe");
  name.resize(length);
  return std::filesystem::path(name).parent_path() / L"bubo-engine.dll";
}
int Run(std::span<const char* const> args) {
  const auto options = args.subspan(1);
  if (options.empty() || std::string_view(options[0]) == "--help" ||
      std::string_view(options[0]) == "-h") {
    gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
    return 0;
  }
  const std::string_view command = options[0];
  if (command == "--version" || command == "-v") {
    gufo::cli::PrintGlobalVersion();
    return 0;
  }
  if (command == "prepare")
    return bubo::prepare::Run(options.subspan(1), LocalAppData(), std::cout,
                              std::cerr);
  if (command == "help") {
    if (options.size() == 1) {
      gufo::cli::PrintGlobalHelp(gufo::cli::kProductName);
      return 0;
    }
    if (std::string_view(options[1]) == "prepare") {
      if (options.size() != 2) {
        std::cerr << "prepare: E2 extra help arguments\n";
        return 2;
      }
      bubo::prepare::PrintHelp(std::cout);
      return 0;
    }
  }
  // No engine import, HIP registration or inference initialisation precedes
  // here.
  const auto path = EnginePath();
  const auto module = LoadLibraryExW(
      path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!module) {
    std::cerr << "bubo: cannot load bubo-engine.dll (Windows error "
              << GetLastError() << "); run bubo prepare runtime\n";
    return 2;
  }
  const auto identity = reinterpret_cast<const char* (*)()>(
      GetProcAddress(module, "BuboEngineIdentityV1"));
  const auto run = reinterpret_cast<int (*)(int, const char* const*)>(
      GetProcAddress(module, "BuboRunCommandV1"));
  const std::string expected = gufo::cli::FormatVersion(
      gufo::cli::kProductName, gufo::cli::kProductVersion, gufo::cli::kRevision,
      gufo::cli::kEngineVersion);
  const char* actual = identity ? identity() : nullptr;
  if (!actual || !run || std::string_view(actual) != expected) {
    std::cerr << "bubo: engine ABI/version/revision mismatch; reinstall this "
                 "package\n";
    FreeLibrary(module);
    return 2;
  }
  // Keep the engine loaded until process teardown: its workers/static objects
  // have the same process lifetime as before the bootstrap split.
  return run(static_cast<int>(args.size()), args.data());
}
}  // namespace

int wmain(int argc, wchar_t* argv[]) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  try {
    std::vector<std::string> owned;
    owned.reserve(argc);
    for (int i = 0; i < argc; ++i)
      owned.push_back(Utf8(argv[i]));
    std::vector<const char*> pointers;
    for (const auto& value : owned)
      pointers.push_back(value.c_str());
    return Run(pointers);
  } catch (const std::exception& error) {
    std::cerr << "bubo: " << error.what() << '\n';
    return 2;
  }
}
