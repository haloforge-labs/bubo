// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_INVALID_BITS_HPP_
#define GUFO_NPU_INVALID_BITS_HPP_
// Cause bits of the sticky "invalid" word that the BFP kernels set in mapped
// host memory. Non-zero still means "invalid"; the bits only say why. Kernels
// use one atomicOr per cause. Header has no includes so HIP and host code can
// share it. The encoder bits are set by both the A (activation) and B (weight)
// encoders; B is encoded at load time, so a bit seen during a chunk is A.
namespace gufo::npu {
inline constexpr unsigned kInvalidEncoderNonFinite = 1U << 0;  // exponent 255
inline constexpr unsigned kInvalidEncoderSaturated = 1U
                                                     << 1;  // exp 254, q=-128
inline constexpr unsigned kInvalidDecoderExp255 = 1U
                                                  << 2;  // C exponent byte 255
inline constexpr unsigned kInvalidDecoderExp254Mant128 = 1U
                                                         << 3;  // C 254 + 128
}  // namespace gufo::npu
#endif
