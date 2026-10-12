#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/vc_runtime.hpp"

int main(int argc, char** argv) {
  using namespace bubo::prepare;
  const std::string selected = argc > 1 ? argv[1] : "all";
  const auto check = [&](bool condition, const char* label) {
    if (!condition)
      std::cerr << "FAIL: " << label << '\n';
    return condition;
  };
  bool ok = true;
  if (selected == "all" || selected == "native") {
    try {
      const auto native = NativeVcRuntimeProbe();
      const bool registered = native.installed();
      std::cout << "native registry64 probe bool=" << registered << '\n';
#ifdef _WIN32
      for (const auto name : {"vcruntime140.dll", "msvcp140.dll"}) {
        const bool exists = native.system_file(name);
        std::cout << "native System32 " << name << " probe bool=" << exists
                  << '\n';
      }
#endif
      // Installation state is machine-dependent; only require callable probes.
    } catch (const std::exception& error) {
      std::cerr << "native probe exception: " << error.what() << '\n';
      ok &= check(false, "real read-only probes return bool without exception");
    }
  }
  for (const std::string kind : {"registry", "missing", "files", "partial"}) {
    if (selected != "all" && selected != kind)
      continue;
    int file_calls = 0;
    VcRuntimeProbe probe{
        [&] { return kind == "registry"; },
        [&](std::string_view name) {
          ++file_calls;
          return kind == "files" ||
                 (kind == "partial" && name == "vcruntime140.dll");
        }};
    std::ostringstream err;
    const bool present = kind == "registry" || kind == "files";
    ok &= check(CheckVcRuntime(probe, err) == (present ? 0 : 4), kind.c_str());
    ok &= check(present
                    ? err.str().empty()
                    : err.str().find("https://aka.ms/vs/17/release/"
                                     "vc_redist.x64.exe") != std::string::npos,
                "Microsoft link on refusal");
    if (kind == "registry")
      ok &= check(file_calls == 0, "registry success short circuits files");
    std::cout << kind << " detector: " << (present ? "present" : "missing")
              << '\n'
              << err.str();
  }
  if (selected == "all" || selected == "prepare") {
    VcRuntimeProbe missing{[] { return false; },
                           [](std::string_view) { return false; }};
    std::ostringstream out, err;
    const std::array<const char*, 3> args{"model", "--output", "unused"};
    ok &= check(Run(args, "", out, err, &missing) == 4 && out.str().empty() &&
                    err.str().find("vc_redist.x64.exe") != std::string::npos,
                "prepare refuses missing prerequisite before phase work");
    std::cout << "prepare missing prerequisite output:\n" << err.str();
    const std::array<const char*, 1> help{"--help"};
    err.str("");
    ok &= check(Run(help, "", out, err, &missing) == 0 && err.str().empty(),
                "help is available without prerequisite");
  }
  if (ok)
    std::cout << "PASS: injectable VC++ prerequisite and prepare guard\n";
  return ok ? 0 : 1;
}
