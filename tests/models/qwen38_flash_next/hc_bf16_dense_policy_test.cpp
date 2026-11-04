#include "src/models/qwen38_flash_next/hc_bf16_dense_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace p = gufo::models::qwen38_flash_next::hc_bf16_dense;

void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() try {
  Require(p::Parse(nullptr) == 1 && p::Parse("0") == 0 && p::Parse("1") == 1,
          "switch default or valid value changed");
  for (const char* value : {"", "2", "true", "off", "10"})
    Require(p::Parse(value) == -1, "unknown switch value accepted");
  for (unsigned n = 0; n <= 32; ++n) {
    const bool narrow = n >= 1 && n <= 8;
    Require(
        p::Shape(n, 320, 10240) == narrow && p::Shape(n, 10240, 320) == narrow,
        "prefill route or HC shape boundary changed");
    for (const auto shape : {std::pair{4U, 10240U},
                             {512U, 2560U},
                             {48U, 2560U},
                             {640U, 2560U},
                             {2560U, 2560U}})
      Require(!p::Shape(n, shape.first, shape.second),
              "inject/router/alpha/shared/PLE selected");
  }
  Require(p::Aligned(256, 256) && p::Aligned(264, 272),
          "valid vector alignment refused");
  Require(!p::Aligned(260, 256) && !p::Aligned(256, 260),
          "unaligned vector load selected");
  std::cout << "BF16 HC POLICY PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
