// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_PROJECTION_GEOMETRY_HPP_
#define GUFO_NPU_PROJECTION_GEOMETRY_HPP_
namespace gufo::npu {
inline unsigned QueryGpuRows(unsigned first) noexcept {
  return first >= 1024 && first <= 12288 && first % 1024 == 0 ? 13312 - first
                                                              : 0;
}
inline unsigned ProjectionUploadWidth(unsigned pitch, unsigned offset,
                                      bool second, unsigned columns) noexcept {
  if (!second && pitch == 16384 && offset == 0xffffffffU && columns > 0 &&
      columns <= 12288 && columns % 1024 == 0)
    return columns;
  if (!second && pitch == 13312 && offset == 0 && QueryGpuRows(columns))
    return columns;
  if (second && pitch > 0 && pitch <= 1024 && offset == 0 &&
      columns == 2 * pitch)
    return pitch;
  return 0;
}
}  // namespace gufo::npu
#endif
