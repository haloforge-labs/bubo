// SPDX-License-Identifier: MIT
#include "src/npu/prefill_plan.hpp"

#include <cassert>
#include <iostream>
#include <vector>
using namespace gufo::npu;
int main() {
  const std::vector<std::size_t> lengths{1,    2047, 2048, 4095, 4096,
                                         4097, 7703, 8192, 16384};
  const std::vector<std::vector<std::size_t>> expected{
      {1},          {2047},       {2048},
      {4095},       {4096},       {4096, 1},
      {4096, 3607}, {4096, 4096}, {4096, 4096, 4096, 4096}};
  for (std::size_t cap : {2048U, 4096U, 8192U})
    for (std::size_t budget : {1U, 2048U, 4096U, 16384U}) {
      for (auto n : lengths) {
        std::vector<std::size_t> old, off;
        auto remaining = n;
        while (remaining) {
          auto c = std::min(
              {budget, remaining, remaining <= cap + 128 ? cap + 128 : cap});
          old.push_back(c);
          remaining -= c;
        }
        remaining = n;
        while (remaining) {
          auto c = PrefillChunk(false, budget, remaining, cap, cap + 128);
          off.push_back(c);
          remaining -= c;
        }
        assert(old == off);
      }
    }
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    std::vector<std::size_t> chunks;
    auto remaining = lengths[i];
    while (remaining) {
      auto budget = PrefillBudget(true, 2048, remaining);
      auto c = PrefillChunk(true, budget, remaining, 2048, 2176);
      chunks.push_back(c);
      remaining -= c;
    }
    assert(chunks == expected[i]);
  }
  assert(PrefillChunk(true, 3000, 7703, 2048, 2176) == 2048);  // hard boundary
  assert(PrefillChunk(true, 1000, 7703, 2048, 2176) == 1000);
  assert(PrefillChunk(true, 4096, 7703, 2048, 2176) == 4096);
  assert(PrefillBoundaryBudget(4096, 2048, 7703, 3000) == 2048);
  assert(PrefillChunk(true, PrefillBoundaryBudget(4096, 2048, 7703, 3000), 3000,
                      2048, 2176) == 2048);
  assert(PrefillBoundaryBudget(0, 2048, 7703, 3000) == 3000);
  std::cout << "PASS planner: OFF original sequences; ON exact 4096 and hard "
               "boundaries\n";
}
