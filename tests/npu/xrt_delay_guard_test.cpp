#include <windows.h>
#include <xrt/experimental/xrt_version.h>

#include <iostream>
#include <stdexcept>
#include <string_view>

// Exercise the production delay hook, without authorizing a DLL load and
// without linking any device API. The deliberate missing Ensure call must fail.
int main() {
  if (GetModuleHandleW(L"xrt_coreutil.dll")) {
    std::cerr << "FAIL: XRT was loaded at process startup\n";
    return 1;
  }
  try {
    (void)xrtVersionMajor();
  } catch (const std::runtime_error& error) {
    if (std::string_view(error.what())
                .find("before verified runtime loading") !=
            std::string_view::npos &&
        !GetModuleHandleW(L"xrt_coreutil.dll")) {
      std::cout << "PASS: unguarded delayed XRT call refused before DLL load; "
                   "real DLL loads=0 device calls=0\n";
      return 0;
    }
  }
  std::cerr << "FAIL: unguarded XRT call did not fail closed\n";
  return 1;
}
