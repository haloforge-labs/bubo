#include "src/models/qwen38_flash_next/iq4_wide_policy.hpp"

#include <iostream>
#include <stdexcept>

namespace q = gufo::models::qwen38_flash_next;

void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("IQ4 wide dispatch contract");
}

int main() {
  const char* invalid_values[] = {nullptr, "", "-1", "4", "01", "true", "2 "};
  for (const char* invalid : invalid_values)
    Require(q::Iq4WideVariant(invalid) == 2);
  for (int variant = 0; variant <= 3; ++variant) {
    const char value[] = {char('0' + variant), '\0'};
    Require(q::Iq4WideVariant(value) == variant);
    for (int n = 1; n <= 32; ++n) {
      Require(q::Iq4WideTokenTile(n) == (n <= 8    ? 8
                                         : n <= 16 ? 16
                                         : n <= 24 ? 24
                                                   : 32));
      for (int rows :
           {1, 7, 8, 9, 65, 640, 1279, 1280, 1281, 2560, 6144, 10240}) {
        const int tile = q::Iq4WideRowsPerBlock(variant, rows);
        Require(tile == (variant >= 2 && rows >= 1280 ? 16 : 8));
        const int grid = (rows + tile - 1) / tile;
        Require(grid * tile >= rows && (grid - 1) * tile < rows);
        if (tile == 16)
          Require(grid >= 80);
        // Token groups partition the batch exactly, including 9/17/25/31.
        int visits[32]{};
        for (int g = 0; g < (n + 7) / 8; ++g)
          for (int t = 0; t < 8 && 8 * g + t < n; ++t)
            ++visits[8 * g + t];
        for (int t = 0; t < n; ++t)
          Require(visits[t] == 1);
      }
    }
  }
  std::cout << "PASS IQ4 wide variants=0..3 N=1..32 geometry/tails/grid\n";
}
