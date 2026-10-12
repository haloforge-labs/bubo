// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstddef>
namespace gufo::npu {
inline std::size_t PrefillChunk(bool on, std::size_t budget,
                                std::size_t remaining, std::size_t capacity,
                                std::size_t through_capacity) {
  if (on && budget >= std::min<std::size_t>(4096, remaining))
    return std::min<std::size_t>(4096, remaining);
  // A hard prefix/checkpoint boundary falls back to the original split.
  return std::min(
      {budget, remaining,
       remaining <= through_capacity ? through_capacity : capacity});
}
inline std::size_t PrefillBudget(bool on, std::size_t budget,
                                 std::size_t remaining) {
  return on ? std::max(budget, std::min<std::size_t>(4096, remaining)) : budget;
}
inline std::size_t PrefillBoundaryBudget(std::size_t preferred,
                                         std::size_t legacy,
                                         std::size_t remaining,
                                         std::size_t clipped) {
  return preferred && remaining >= preferred && clipped < preferred
             ? std::min(clipped, legacy)
             : clipped;
}
}  // namespace gufo::npu
