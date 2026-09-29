#ifndef GUFO_MODELS_QWEN36_A3B_PREFILL_KERNELS_HPP_
#define GUFO_MODELS_QWEN36_A3B_PREFILL_KERNELS_HPP_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Glue kernels for A3B's wide prefill (prefill.cpp): layout changes around
/// Flash-Next's prefill launchers, which are used read-only.
namespace gufo::models::qwen36_a3b {

/// dst[t][g * padded + j] = src[t][g * group + j] for j < group, zero rows
/// for group <= j < padded (heads of `d` floats). Pads A3B's 8 query heads
/// per KV head to the 12 of Flash-Next's WMMA attention geometry.
void PadHeads(const float* src, float* dst, std::uint32_t n_tokens,
              std::uint32_t kv_heads, std::uint32_t group,
              std::uint32_t padded, std::uint32_t d, hipStream_t stream);
/// The inverse selection: dst[t][g * group + j] = src[t][g * padded + j].
void UnpadHeads(const float* src, float* dst, std::uint32_t n_tokens,
                std::uint32_t kv_heads, std::uint32_t group,
                std::uint32_t padded, std::uint32_t d, hipStream_t stream);

/// BF16 hi/lo split: hi = BF16(x), lo = BF16(x - hi), so hi + lo carries
/// ~16 mantissa bits of x (two BF16 GEMMs then match F32 activations).
void SplitBf16(const float* x, __hip_bfloat16* hi, __hip_bfloat16* lo,
               std::size_t count, hipStream_t stream);
/// dst[r][:] = BF16 split of src[rows[r]][:] (zeros for rows[r] < 0),
/// `width` wide; `lo` may be null. src is F32 or (half_src) F16.
void GatherRowsBf16(const void* src, bool half_src, const std::int32_t* rows,
                    std::uint32_t n_rows, std::uint32_t width,
                    __hip_bfloat16* dst, __hip_bfloat16* lo,
                    hipStream_t stream);
/// dst[rows[r]][:] = F16(src[r][:]) for rows[r] >= 0.
void ScatterRowsHalf(const float* src, const std::int32_t* rows,
                     std::uint32_t n_rows, std::uint32_t width, __half* dst,
                     hipStream_t stream);
/// out/lo = BF16 split of silu(gate[i]) * up[i] (lo may be null).
void SwigluBf16(const float* gate, const float* up, __hip_bfloat16* out,
                __hip_bfloat16* lo, std::size_t count, hipStream_t stream);
/// F32 out[t][j] = sum_k w[j][k] x[t][k] (w [m][k], x [n][k], out [n][m]),
/// each sum in K order. For the small F32 projections (router: 257 rows,
/// GDN alpha/beta: 64), where hipBLAS's SGEMM runs at ~1 TFLOPS.
void F32GemmNT(const float* w, const float* x, float* out, std::uint32_t m,
               std::uint32_t n, std::uint32_t k, hipStream_t stream);
/// Token rows per launched tile of GroupedBf16Gemm.
inline constexpr std::uint32_t kGroupedTileRows = 64;
/// Routed BF16 expert GEMM on the WMMA cores: out[c][0..m) = W_e x[c] for
/// every compacted row c of expert e's bucket [bounds[e], bounds[e + 1]).
/// W is [experts][m][k] BF16; x (and x_lo, the BF16 split's low part,
/// accumulated when non-null) is [rows][k] BF16; out is [rows][m] F32.
/// `tiles[y]` packs expert | (token tile << 16), tiles of kGroupedTileRows
/// rows. m % 16 == 0 and k % 16 == 0.
void GroupedBf16Gemm(const void* w, const void* x, const void* x_lo,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* bounds, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream);
/// out[t] = [a[t] ; b[t]] (rows of `width` floats each).
void Concat2(const float* a, const float* b, float* out, std::uint32_t n,
             std::uint32_t width, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_PREFILL_KERNELS_HPP_
