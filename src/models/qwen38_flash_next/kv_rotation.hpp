#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_ROTATION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_ROTATION_HPP_

#include <span>

namespace gufo::models::qwen38_flash_next {

enum class KvRotationKind { kKeyQuery, kValueOutput };

// Exactly one 256-element F32 head. K/Q: H256, V/output: four contiguous
// H64 blocks. Natural Sylvester order, normalization FIRST (1/16 or 1/8),
// then ascending RNE-F32 butterflies, high=u-v. Same operation for inverse;
// an arbitrary floating-point round trip is not promised to be exact.
// Requires round-to-nearest and gradual underflow (FTZ/DAZ clear). Strict CPU
// signed-zero semantics. Throws invalid_argument on size/kind/nonfinite input
// or intermediate overflow, leaving the input unchanged. No codec or gate.
// Store callers must separately EncodeKvRow: finite transformed values may
// still fail the codec's finite-F16 reconstruction constraint.
void RotateKvHead(std::span<float> head, KvRotationKind kind);

}  // namespace gufo::models::qwen38_flash_next
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_ROTATION_HPP_
