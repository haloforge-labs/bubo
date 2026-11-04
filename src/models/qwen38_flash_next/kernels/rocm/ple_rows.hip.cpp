#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/ple_decode.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
__global__ void DecodePleRowsKernel(const std::uint8_t* raw, float* out,
                                    std::uint32_t dim, std::size_t row_bytes,
                                    bool iq4) {
  const auto row = blockIdx.x;
  for (std::uint32_t col = threadIdx.x; col < dim; col += blockDim.x)
    out[std::size_t{row} * dim + col] =
        PleRowElement(raw + std::size_t{row} * row_bytes, col, iq4);
}
}  // namespace

void DecodePleRows(const std::uint8_t* raw, float* out, std::uint32_t rows,
                   std::uint32_t dim, bool iq4, hipStream_t stream) {
  const std::size_t row_bytes = iq4 ? std::size_t{dim / 32} * 18 : dim * 2;
  hipLaunchKernelGGL(DecodePleRowsKernel, dim3(rows), dim3(128), 0, stream, raw,
                     out, dim, row_bytes, iq4);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
