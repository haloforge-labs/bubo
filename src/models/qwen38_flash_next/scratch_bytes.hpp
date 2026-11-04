#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SCRATCH_BYTES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SCRATCH_BYTES_HPP_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next {
// Match the allocator's nonzero payload for empty optional arrays. Account
// requested payload bytes, not driver/page overhead or a simulated slope.
inline std::size_t AllocationRequestBytes(std::size_t count,
                                          std::size_t width) {
  count = std::max<std::size_t>(1, count);
  if (width == 0 || count > std::numeric_limits<std::size_t>::max() / width)
    throw std::overflow_error("invalid allocation byte size");
  return count * width;
}
inline void AddAllocationBytes(std::size_t& total, std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - total)
    throw std::overflow_error("allocation byte accounting overflow");
  total += bytes;
}
inline std::size_t ResidentWithScratch(std::size_t resident,
                                       std::size_t device_scratch,
                                       std::size_t host_scratch) {
  AddAllocationBytes(resident, device_scratch);
  AddAllocationBytes(resident, host_scratch);
  return resident;
}
}  // namespace gufo::models::qwen38_flash_next
#endif
