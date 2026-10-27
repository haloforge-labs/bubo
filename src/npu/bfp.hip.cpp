// SPDX-License-Identifier: MIT
// Selective adaptation of mikealanni/gufo 8e0fb1de BFP staging kernels.
// No unrelated GEMM, requantization or executor changes. Scalar loads avoid
// the fork's vector alignment assumptions and partial-group overwrites.
#include <hip/hip_fp16.h>

#include <stdexcept>

#include "bfp_hip.hpp"
#include "invalid_bits.hpp"

namespace gufo::npu::bfp {
namespace {
__device__ float F16(std::uint16_t bits) {
  union {
    std::uint16_t bits;
    __half value;
  } h;
  h.bits = bits;
  return __half2float(h.value);
}
__device__ void Encode(const float* v, std::uint8_t* out, unsigned* invalid) {
  unsigned e = 0;
  for (unsigned i = 0; i < 8; ++i)
    e = max(e, (__float_as_uint(v[i]) >> 23) & 255U);
  if (e == 255) {
    atomicOr(invalid, kInvalidEncoderNonFinite);
    for (unsigned i = 0; i < 9; ++i)
      out[i] = 0;
    return;
  }
  const float inv = __uint_as_float(e >= 7 ? (260U - e) << 23 : 0U);
  out[0] = static_cast<std::uint8_t>(e);
  for (unsigned i = 0; i < 8; ++i) {
    const float q = fminf(127.0F, fmaxf(-128.0F, rintf(v[i] * inv)));
    // Bits 0 and 1 are shared by the A (activation) and B (weight) encoders;
    // EncodeBHip has no caller in the server, so a bit seen in a chunk is A.
    if (e == 254 && q == -128.0F)
      atomicOr(invalid, kInvalidEncoderSaturated);
    out[i + 1] = static_cast<std::uint8_t>(static_cast<int>(q) & 255);
  }
}
__global__ void A(const std::uint16_t* x, std::size_t pitch, std::uint8_t* out,
                  unsigned* invalid) {
  const auto g = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= ABytes / 9)
    return;
  const auto p = Position(Layout::A, g);
  float v[8];
  for (unsigned i = 0; i < 8; ++i)
    v[i] = F16(x[p.row * pitch + p.col + i]);
  Encode(v, out + g * 9, invalid);
}
__global__ void B(const std::uint8_t* x, std::uint8_t* out, unsigned* invalid) {
  const auto g = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= BBytes / 9)
    return;
  const auto p = Position(Layout::B, g);
  const auto block = (p.col * K + p.row) / 32 * 34;
  const float scale = F16(x[block] | (unsigned(x[block + 1]) << 8));
  float v[8];
  for (unsigned i = 0; i < 8; ++i) {
    const int q = x[block + 2 + p.row % 32 + i];
    v[i] = scale * float(q < 128 ? q : q - 256);
  }
  Encode(v, out + g * 9, invalid);
}
__global__ void C(const std::uint8_t* x, float* out, std::size_t pitch,
                  std::size_t col0, std::size_t valid, bool accumulate,
                  unsigned* invalid) {
  const auto g = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= CBytes / 9)
    return;
  const auto p = Position(Layout::C, g);
  const auto* bytes = x + g * 9;
  if (bytes[0] == 255) {
    atomicOr(invalid, kInvalidDecoderExp255);
    return;
  }
  if (bytes[0] == 254) {
    for (unsigned i = 1; i < 9; ++i) {
      if (bytes[i] == 128) {
        atomicOr(invalid, kInvalidDecoderExp254Mant128);
        return;
      }
    }
  }
  const float mult = ldexpf(1.0F, int(bytes[0]) - 127) / 64.0F;
  for (unsigned i = 0; i < 8 && p.col + i < valid; ++i) {
    const int q = bytes[i + 1];
    const float value = float(q < 128 ? q : q - 256) * mult;
    auto& dst = out[p.row * pitch + col0 + p.col + i];
    dst = accumulate ? dst + value : value;
  }
}
void Require(bool condition) {
  if (!condition)
    throw std::invalid_argument("NPU BFP HIP storage invalid");
}
}  // namespace
hipError_t EncodeAHip(std::span<const std::uint16_t> input, std::size_t pitch,
                      std::span<std::uint8_t> output, unsigned* invalid,
                      hipStream_t stream) {
  Require(input.size() >= MatrixExtent(M, pitch, 0, K) && input.data() &&
          output.size() == ABytes && output.data() && invalid);
  hipLaunchKernelGGL(A, dim3(ABytes / 9 / 256), dim3(256), 0, stream,
                     input.data(), pitch, output.data(), invalid);
  return hipGetLastError();
}
hipError_t EncodeBHip(std::span<const std::byte> input,
                      std::span<std::uint8_t> output, unsigned* invalid,
                      hipStream_t stream) {
  Require(input.size() == Q8Bytes && input.data() && output.size() == BBytes &&
          output.data() && invalid);
  hipLaunchKernelGGL(B, dim3(BBytes / 9 / 256), dim3(256), 0, stream,
                     reinterpret_cast<const std::uint8_t*>(input.data()),
                     output.data(), invalid);
  return hipGetLastError();
}
hipError_t DecodeCHip(std::span<const std::uint8_t> input,
                      std::span<float> output, std::size_t pitch,
                      std::size_t col0, std::size_t valid_cols, bool accumulate,
                      unsigned* invalid, hipStream_t stream) {
  Require(valid_cols > 0 && valid_cols <= N && input.size() == CBytes &&
          input.data() &&
          output.size() >= MatrixExtent(M, pitch, col0, valid_cols) &&
          output.data() && invalid);
  hipLaunchKernelGGL(C, dim3(CBytes / 9 / 256), dim3(256), 0, stream,
                     input.data(), output.data(), pitch, col0, valid_cols,
                     accumulate, invalid);
  return hipGetLastError();
}
}  // namespace gufo::npu::bfp
