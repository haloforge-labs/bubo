// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_LINEAR_Z_GEOMETRY_HPP_
#define GUFO_NPU_LINEAR_Z_GEOMETRY_HPP_
namespace gufo::npu {
// Host launch validation shared with the CPU oracle. A zero request retains
// the original full projection; invalid nonzero requests launch nothing.
constexpr unsigned LinearZLaunchRows(unsigned requested, unsigned rows,
                                     unsigned channels) noexcept {
  if (requested == 0)
    return rows;
  return requested >= 4096 && requested <= rows && requested % 1024 == 0 &&
                 rows == 16384 && channels == 10240
             ? requested
             : 0;
}
}  // namespace gufo::npu
#endif
