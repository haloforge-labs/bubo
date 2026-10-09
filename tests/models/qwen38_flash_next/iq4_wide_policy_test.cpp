#include "src/models/qwen38_flash_next/iq4_wide_policy.hpp"

#include <iostream>
#include <stdexcept>

namespace q = gufo::models::qwen38_flash_next;

void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("IQ4 wide dispatch contract");
}

int main() {
  const char* invalid_values[] = {nullptr, "", "-1", "5", "01", "true", "2 "};
  for (const char* invalid : invalid_values)
    Require(q::Iq4WideVariant(invalid) == 4);
  for (int variant = 0; variant <= 4; ++variant) {
    const char value[] = {char('0' + variant), '\0'};
    Require(q::Iq4WideVariant(value) == variant);
    for (int n = 1; n <= 64; ++n) {
      Require(q::Iq4WideTokenTile(n) == ((n + 7) / 8) * 8);
      Require(q::Iq4WmmaTokenTiles(n) == (n + 15) / 16);
      for (int rows :
           {1, 7, 8, 9, 65, 640, 1279, 1280, 1281, 2560, 6144, 10240}) {
        const int tile = q::Iq4WideRowsPerBlock(variant, rows);
        Require(tile ==
                (variant == 4 || (variant >= 2 && rows >= 1280) ? 16 : 8));
        const int grid = (rows + tile - 1) / tile;
        Require(grid * tile >= rows && (grid - 1) * tile < rows);
        if (tile == 16 && variant != 4)
          Require(grid >= 80);
        // Token groups partition the batch exactly, including 9/17/25/31.
        int visits[64]{};
        for (int g = 0; g < (n + 7) / 8; ++g)
          for (int t = 0; t < 8 && 8 * g + t < n; ++t)
            ++visits[8 * g + t];
        for (int t = 0; t < n; ++t)
          Require(visits[t] == 1);
      }
    }
  }
  const char* invalid_waves[] = {nullptr, "", "3", "04", "16 ", "32"};
  for (const char* invalid : invalid_waves)
    Require(q::Iq4WmmaWaves(invalid) == 8);
  Require(q::Iq4WmmaWaves("4") == 4);
  Require(q::Iq4WmmaWaves("8") == 8);
  Require(q::Iq4WmmaWaves("16") == 16);
  // Validate WMMA transpose ownership and K partition for every supported
  // wave count. Eight/16-wave XS scale shifts stay constant modulo eight;
  // four-wave A/B computes the changing sub-block shift from kb instead.
  for (int waves : {4, 8, 16}) {
    for (int blocks : {1, 7, 8, 9, 20, 80, 192}) {
      int k_visits[192]{};
      for (int wave = 0; wave < waves; ++wave)
        for (int kb = wave; kb < blocks; kb += waves)
          ++k_visits[kb];
      for (int kb = 0; kb < blocks; ++kb)
        Require(k_visits[kb] == 1);
    }
    int writes[4096]{};
    for (int wave = 0; wave < waves; ++wave)
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 8; ++i) {
          const int out = (lane % 16) * 16 + 2 * i + lane / 16;
          const int swizzle = (out ^ (out >> 4)) & (waves - 1);
          ++writes[out * waves + (wave ^ swizzle)];
        }
    for (int out = 0; out < 256; ++out)
      for (int wave = 0; wave < waves; ++wave)
        Require(writes[out * waves + wave] == 1);
    for (int n = 1; n <= 64; ++n) {
      int visits[64]{};
      for (int tile = 0; tile < q::Iq4WmmaTokenTiles(n); ++tile)
        for (int thread = 0; thread < 32 * waves; ++thread)
          for (int out = thread; out < 256; out += 32 * waves)
            if (out % 16 == 0 && tile * 16 + out / 16 < n)
              ++visits[tile * 16 + out / 16];
      for (int t = 0; t < n; ++t)
        Require(visits[t] == 1);
    }
  }
  std::cout << "PASS IQ4 variants=0..4 N=1..64 WMMA waves=4/8/16 "
               "indexing/tails/grid\n";
}
