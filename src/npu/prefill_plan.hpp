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
// Spacing of the intermediate checkpoints of a cold prompt. With the NPU
// prefill on it is the NPU chunk. On the GPU alone it is the forward chunk
// (`legacy_capacity`), so a checkpoint ends a whole pass: 2048..4096 as is,
// the smallest multiple of the chunk that reaches 2048 below that, and 8192
// above 8192 (a checkpoint can then cut a larger pass). At the default chunk
// of 4096 or less the grid is the same as before; it differs only when
// GUFO_PREFILL_FORWARD_TOKENS is raised above 4096.
inline std::size_t CheckpointGrid(std::size_t preferred,
                                  std::size_t legacy_capacity) {
  if (preferred)
    return preferred;
  if (legacy_capacity == 0)
    return 2048;
  if (legacy_capacity < 2048)
    return (2048 + legacy_capacity - 1) / legacy_capacity * legacy_capacity;
  return std::min<std::size_t>(legacy_capacity, 8192);
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
