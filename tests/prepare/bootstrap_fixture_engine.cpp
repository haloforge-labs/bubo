#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/cli/brand.hpp"

// Inert DLL built only with BUILD_TESTING. No inference library or model.
extern "C" __declspec(dllexport) const char* BuboEngineIdentityV1() {
  if (std::getenv("BUBO_FIXTURE_BAD_IDENTITY"))
    return "fixture mismatched identity";
  static const std::string identity = gufo::cli::FormatVersion(
      gufo::cli::kProductName, gufo::cli::kProductVersion, gufo::cli::kRevision,
      gufo::cli::kEngineVersion);
  return identity.c_str();
}
extern "C" __declspec(dllexport) int BuboRunCommandV1(int argc,
                                                      const char* const* argv) {
  for (int i = 1; i < argc; ++i)
    std::cout << '[' << argv[i] << "]\n";
  return 37;
}
