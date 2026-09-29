#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// A3B-private kernels: the few operations Flash-Next's launchers do not
/// cover for this model.
namespace gufo::models::qwen36_a3b {

/// res[i] += x[i].
void AddInPlace(float* res, const float* x, std::size_t count,
                hipStream_t stream);

/// x[i] *= z[i]. Qwen3.5/3.6 gate the GDN output with silu(z); Flash-Next's
/// GDN epilogue applies sigmoid(z), and silu(z) = z * sigmoid(z).
void MulInPlace(float* x, const float* z, std::size_t count,
                hipStream_t stream);

/// F32 projection out[t][m] = dot(w[m][:], x[t][:]) for 1-8 token rows, one
/// wave per output row (the router: 257 x 2048). A token's sum order does
/// not depend on the row count. k must be a multiple of 128.
void F32Gemv(const float* w, const float* x, float* out, std::uint32_t tokens,
             std::uint32_t m, std::uint32_t k, hipStream_t stream);

/// Routed BF16 expert projection:
/// out[r*used + s][m] = dot(W[ids[r*used + s]][m][:], x[r][:])
/// for r < rows, s < used. W is [experts][m][k] BF16, x is [rows][k] F32.
void Bf16Experts(const void* w, const float* x, const std::int32_t* ids,
                 float* out, std::uint32_t rows, std::uint32_t used,
                 std::uint32_t m, std::uint32_t k, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_HPP_
