// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_BFP_HIP_HPP_
#define GUFO_NPU_BFP_HIP_HPP_
#include <hip/hip_runtime.h>

#include "bfp.hpp"

namespace gufo::npu::bfp {
// HIP encoding support. All spans are DEVICE storage, kept alive until
// the stream completes. Caller zeros invalid once per chunk and refuses the
// chunk after its existing completion wait if any encoder/decoder set it.
// The word is a bit set (invalid_bits.hpp): each cause is one atomicOr, so
// non-zero still means invalid and the bits say why.
// Stream success alone does not establish finite/valid results.
hipError_t EncodeAHip(std::span<const std::uint16_t> input, std::size_t pitch,
                      std::span<std::uint8_t> output, unsigned* invalid,
                      hipStream_t stream);
hipError_t EncodeBHip(std::span<const std::byte> input,
                      std::span<std::uint8_t> output, unsigned* invalid,
                      hipStream_t stream);
hipError_t DecodeCHip(std::span<const std::uint8_t> input,
                      std::span<float> output, std::size_t pitch,
                      std::size_t col0, std::size_t valid_cols, bool accumulate,
                      unsigned* invalid, hipStream_t stream);
}  // namespace gufo::npu::bfp
#endif
