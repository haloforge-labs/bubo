#include "src/cli/prepare/vc_runtime.hpp"

#include <filesystem>
#include <ostream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace bubo::prepare {
VcRuntimeProbe NativeVcRuntimeProbe() {
#ifdef _WIN32
  return {[] {
            DWORD installed = 0, size = sizeof(installed);
            return RegGetValueW(HKEY_LOCAL_MACHINE,
                                L"SOFTWARE\\Microsoft\\VisualStudio\\14."
                                L"0\\VC\\Runtimes\\x64",
                                L"Installed",
                                RRF_RT_REG_DWORD | RRF_SUBKEY_WOW6464KEY,
                                nullptr, &installed, &size) == ERROR_SUCCESS &&
                   installed == 1;
          },
          [](std::string_view name) {
            wchar_t system[MAX_PATH];
            const auto length = GetSystemDirectoryW(system, MAX_PATH);
            if (!length || length >= MAX_PATH)
              return false;
            std::error_code error;
            return std::filesystem::is_regular_file(
                std::filesystem::path(system) / name, error);
          }};
#else
  // The prerequisite applies only to Windows builds.
  return {[] { return true; }, {}};
#endif
}

int CheckVcRuntime(const VcRuntimeProbe& probe, std::ostream& err) {
  if ((probe.installed && probe.installed()) ||
      (probe.system_file && probe.system_file("vcruntime140.dll") &&
       probe.system_file("msvcp140.dll")))
    return 0;
  err << "prepare: E4 Microsoft VC++ v14 x64 Redistributable is required "
         "(not shipped). Install it from Microsoft, then run bubo again:\n"
         "https://aka.ms/vs/17/release/vc_redist.x64.exe\n";
  return 4;
}
}  // namespace bubo::prepare
