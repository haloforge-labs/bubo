#include "src/models/qwen/hip/kernels/copy_27b.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace gufo::hip {
namespace {

constexpr unsigned kThreads = 256;

__global__ void CopyVec16Kernel(uint4* __restrict__ dst,
                                const uint4* __restrict__ src,
                                std::size_t count) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                       threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

__global__ void CopyBytesKernel(std::uint8_t* __restrict__ dst,
                                const std::uint8_t* __restrict__ src,
                                std::size_t count) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                       threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

__global__ void Copy2DVec16Kernel(std::uint8_t* __restrict__ dst,
                                  std::size_t dst_pitch,
                                  const std::uint8_t* __restrict__ src,
                                  std::size_t src_pitch,
                                  std::size_t row_vectors) {
  const auto* in = reinterpret_cast<const uint4*>(src + blockIdx.y * src_pitch);
  auto* out = reinterpret_cast<uint4*>(dst + blockIdx.y * dst_pitch);
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                       threadIdx.x;
       i < row_vectors; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    out[i] = in[i];
}

unsigned Blocks(std::size_t items) {
  // Enough waves to saturate memory; the loops are grid-stride.
  return static_cast<unsigned>(
      std::clamp<std::size_t>((items + kThreads - 1) / kThreads, 1, 2560));
}

bool Aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
}

constexpr std::size_t kMaxPublishWords = 4;

struct PublishSources {
  const std::uint32_t* words[kMaxPublishWords];
};

__global__ void PublishWordsKernel(PublishSources sources, std::uint32_t count,
                                   std::uint32_t* host_mapped,
                                   std::uint32_t sequence) {
  for (std::uint32_t i = 0; i < count; ++i)
    host_mapped[i] = *sources.words[i];
  __threadfence_system();
  host_mapped[count] = sequence;
  __threadfence_system();
}

}  // namespace

void LaunchPublishWords27(const std::uint32_t* const* sources,
                          std::size_t count, std::uint32_t* host_mapped,
                          std::uint32_t sequence, hipStream_t stream) {
  PublishSources packed{};
  for (std::size_t i = 0; i < count && i < kMaxPublishWords; ++i)
    packed.words[i] = sources[i];
  PublishWordsKernel<<<1, 1, 0, stream>>>(
      packed, static_cast<std::uint32_t>(std::min(count, kMaxPublishWords)),
      host_mapped, sequence);
}

void WaitPublished27(volatile std::uint32_t* host_words, std::size_t slot,
                     std::uint32_t sequence, hipStream_t stream) {
  // HIP on Windows holds launches until a flush; hipStreamQuery submits them.
  (void)hipStreamQuery(stream);
  const auto start = std::chrono::steady_clock::now();
  std::uint32_t spins = 0;
  while (host_words[slot] != sequence) {
#if !defined(__HIP_DEVICE_COMPILE__)
    __builtin_ia32_pause();
#endif
    if (++spins % 4096 == 0 &&
        std::chrono::steady_clock::now() - start > std::chrono::seconds(1)) {
      if (hipStreamSynchronize(stream) != hipSuccess ||
          host_words[slot] != sequence)
        throw std::runtime_error("Qwen published result did not arrive");
      return;
    }
  }
}

void LaunchDeviceCopy27(void* dst, const void* src, std::size_t bytes,
                        hipStream_t stream) {
  if (bytes == 0)
    return;
  if (Aligned16(dst) && Aligned16(src)) {
    const std::size_t vectors = bytes / 16;
    if (vectors != 0)
      CopyVec16Kernel<<<Blocks(vectors), kThreads, 0, stream>>>(
          static_cast<uint4*>(dst), static_cast<const uint4*>(src), vectors);
    const std::size_t done = vectors * 16;
    if (done != bytes)
      CopyBytesKernel<<<1, kThreads, 0, stream>>>(
          static_cast<std::uint8_t*>(dst) + done,
          static_cast<const std::uint8_t*>(src) + done, bytes - done);
    return;
  }
  CopyBytesKernel<<<Blocks(bytes), kThreads, 0, stream>>>(
      static_cast<std::uint8_t*>(dst), static_cast<const std::uint8_t*>(src),
      bytes);
}

void LaunchDeviceCopy2D27(void* dst, std::size_t dst_pitch, const void* src,
                          std::size_t src_pitch, std::size_t row_bytes,
                          std::size_t rows, hipStream_t stream) {
  if (rows == 0 || row_bytes == 0)
    return;
  if (Aligned16(dst) && Aligned16(src) && dst_pitch % 16 == 0 &&
      src_pitch % 16 == 0 && row_bytes % 16 == 0) {
    const std::size_t vectors = row_bytes / 16;
    const dim3 grid(Blocks(vectors), static_cast<unsigned>(rows));
    Copy2DVec16Kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<std::uint8_t*>(dst), dst_pitch,
        static_cast<const std::uint8_t*>(src), src_pitch, vectors);
    return;
  }
  for (std::size_t row = 0; row < rows; ++row)
    LaunchDeviceCopy27(static_cast<std::uint8_t*>(dst) + row * dst_pitch,
                       static_cast<const std::uint8_t*>(src) + row * src_pitch,
                       row_bytes, stream);
}

}  // namespace gufo::hip
