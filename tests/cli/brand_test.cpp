#include "src/cli/brand.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  using namespace gufo::cli;
  if (FormatVersion("bubo", "1.2.3", "abcd123", "4.5.6") !=
          "bubo 1.2.3 (abcd123), engine gufo 4.5.6" ||
      FormatVersion("bubo", "2.3.4", "", "5.6.7") !=
          "bubo 2.3.4 (unknown), engine gufo 5.6.7" ||
      FormatVersion("gufo", "3.4.5", "unknown", "6.7.8") !=
          "gufo 3.4.5 (unknown), engine gufo 6.7.8" ||
      kDefaultProfile != (kProductName == "bubo" ? "default" : "stock") ||
      kExeName != kProductName || kCacheDirName != "gufo" ||
      kApiKeys[0] != "gufo" || kApiKeys[1] != "bubo")
    throw std::runtime_error("brand/version contract failed");
  std::cout
      << "PASS: brand_test version formatting and brand profile contract\n";
}
