#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// Bench-only cache traffic. Every word is read and changed, including the tail.
__global__ void Evict(volatile std::uint32_t* words, std::size_t count) {
  const std::size_t first = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = first; i < count; i += stride)
    words[i] = words[i] + 1U;
}

}  // namespace

hipError_t ExpertBenchEvict(void* buffer, std::size_t bytes) {
  Evict<<<1024, 256, 0, nullptr>>>(static_cast<std::uint32_t*>(buffer),
                                   bytes / sizeof(std::uint32_t));
  return hipGetLastError();
}
