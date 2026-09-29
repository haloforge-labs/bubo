#ifndef GUFO_MODELS_QWEN36_A3B_ATTENTION_HPP_
#define GUFO_MODELS_QWEN36_A3B_ATTENTION_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen36_a3b {

/// Decode/verify attention for 1-8 query rows at A3B's geometry (8 query
/// heads per KV head, head dim 256, F16 cache [pos][kv_head][256]).
///
/// One workgroup takes one KV head and one slice of the keys, and scores all
/// 8 heads x `rows` queries against it, so every K/V byte is read once per
/// step (Flash-Next's per-head kernel reads it 8 x rows times); the default
/// kernel does both matrix products on the WMMA cores. Keys are cut
/// into fixed kAttnChunk slices from position 0, so a query's arithmetic
/// depends only on its position (batch-width invariant, like every other
/// decode kernel); the grid covers max_context and slices past *start_pos
/// exit at once, so a captured graph fits every depth. A second launch
/// merges the slices (log-sum-exp).
///
/// q and out are [row][head][256] F32 (q already normed and rotated); row t
/// sits at position *start_pos + t and sees keys 0..*start_pos + t.
/// `partials` holds AttentionPartialFloats(kv_heads, max_context) floats.
inline constexpr std::uint32_t kAttnChunk = 256;
std::uint32_t AttentionSlices(std::uint32_t max_context);
std::size_t AttentionPartialFloats(std::uint32_t kv_heads,
                                   std::uint32_t max_context);
/// vt_cache (optional): V transposed in 16-position tiles (see StoreVt), kept
/// in step with v_cache; the direct / cooperative WMMA kernels read it.
bool DecodeAttention(const float* q, const __half* k_cache,
                     const __half* v_cache, const __half* vt_cache, float* out,
                     float* partials, const std::uint32_t* start_pos,
                     std::uint32_t rows, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim,
                     std::uint32_t max_context, hipStream_t stream);
/// Copies V rows [*pos, *pos + n) of v_cache ([pos][kv_head][256]) into the
/// tiled transposed cache vt ([kv_head][max_context / 16][256][16]).
void StoreVt(const __half* v_cache, __half* vt_cache, const std::uint32_t* pos,
             std::uint32_t n, std::uint32_t kv_heads, std::uint32_t max_context,
             hipStream_t stream);
/// Kernel choice for DecodeAttention: 0 = the ALU kernel, 1 = WMMA (default;
/// needs vt_cache, else the ALU kernel), 2 = WMMA with P split hi + lo.
/// -1 = the A3B_DECODE_ATTN environment variable.
void SetDecodeAttentionMode(int mode);
/// `--bench attn`: decode attention alone on random KV at several depths and
/// widths, every mode timed, compared and checked for width invariance.
int RunDecodeAttentionBench();

/// Prefill (wide batch) causal attention on the F16 WMMA cores at the same
/// geometry: q [t][heads][256] F32 (normed, rotated) at positions
/// start_pos + t, keys 0..start_pos + t from the F16 caches; out[t][h][d] =
/// attention * sigmoid(gate[t][h * 256 + d]). A block takes 16 queries x 4
/// heads of one KV group; K/V tiles are staged once in LDS for all four.
/// Q and the probabilities are F16 in the matrix products (Flash-Next's
/// prefill attention does the same). Returns false for another geometry.
bool PrefillAttention(const float* q, const float* gate, const __half* k_cache,
                      const __half* v_cache, float* out,
                      std::uint32_t start_pos, std::uint32_t n_tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_ATTENTION_HPP_
