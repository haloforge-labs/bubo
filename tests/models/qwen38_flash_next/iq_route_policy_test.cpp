#include "src/models/qwen38_flash_next/kernels/rocm/iq_route_policy.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace q = gufo::models::qwen38_flash_next::rocm;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  try {
    Require(q::kIqRouteMinRows == 256, "default threshold changed");
    Require(!q::UseIqWmmaRoute(false, 255, 256), "off below threshold");
    Require(!q::UseIqWmmaRoute(false, 256, 256), "off at threshold");
    Require(!q::UseIqWmmaRoute(true, 255, 256), "IQ below threshold");
    Require(q::UseIqWmmaRoute(true, 256, 256), "IQ at threshold");
    Require(q::UseIqWmmaRoute(true, 257, 256), "IQ above threshold");
    Require(!q::UseIqWmmaRoute(true, 511, 512), "override below threshold");
    Require(q::UseIqWmmaRoute(true, 512, 512), "override at threshold");
    Require(q::UseIqWmmaRoute(true, 1, 0), "zero selects IQ");
    Require(!q::UseIqWmmaRoute(true, 1024, 65536),
            "large threshold selects MMQ");

    std::uint32_t rows = 999;
    Require(q::ParseIqRouteMinRows("256", &rows) && rows == 256,
            "parse 256");
    Require(q::ParseIqRouteMinRows("0", &rows) && rows == 0,
            "parse zero");
    Require(q::ParseIqRouteMinRows("4294967295", &rows) &&
                rows == std::numeric_limits<std::uint32_t>::max(),
            "parse maximum");
    for (std::string_view bad : {"", "garbage", "256x", "-1", "+1", " 256",
                                 "4294967296"}) {
      rows = 999;
      Require(!q::ParseIqRouteMinRows(bad, &rows) && rows == 999,
              "garbage accepted or output changed");
    }
    Require(!q::ParseIqRouteMinRows("1", nullptr), "null output accepted");
    std::cout << "IQ ROUTE POLICY PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "IQ ROUTE POLICY FAIL: " << e.what() << '\n';
    return 1;
  }
}
