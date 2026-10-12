#pragma once

#include <functional>
#include <iosfwd>
#include <string_view>

namespace bubo::prepare {
// Read-only probes: registry Installed value in the explicit 64-bit view,
// and individual DLLs in the native System32 directory. No acquisition hook.
struct VcRuntimeProbe {
  std::function<bool()> installed;
  std::function<bool(std::string_view)> system_file;
};
VcRuntimeProbe NativeVcRuntimeProbe();
int CheckVcRuntime(const VcRuntimeProbe&, std::ostream&);
}  // namespace bubo::prepare
