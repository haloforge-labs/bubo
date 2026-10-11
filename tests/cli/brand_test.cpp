#include "src/cli/brand.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  using namespace gufo::cli;
  if (FormatVersion("bubo", "1.2.3", "abcd123", "4.5.6") !=
          "bubo 1.2.3 (abcd123), engine gufo 4.5.6" ||
      FormatVersion("bubo", "development", "", "development") !=
          "bubo development (unknown), engine gufo development" ||
      FormatVersion("gufo", "1", "unknown", "1") !=
          "gufo 1 (unknown), engine gufo 1" ||
      kDefaultProfile != (kProductName == "bubo" ? "default" : "stock") ||
      kExeName != kProductName || kCacheDirName != "gufo" ||
      kApiKeys[0] != "gufo" || kApiKeys[1] != "bubo")
    throw std::runtime_error("brand/version contract failed");
  std::cout
      << "PASS: brand_test version formatting and brand profile contract\n";
}
