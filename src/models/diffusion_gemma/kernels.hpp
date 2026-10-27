#ifndef GUFO_MODELS_DIFFUSION_GEMMA_KERNELS_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_KERNELS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace gufo::models::diffusion_gemma::kernels {

using bf16_t = std::uint16_t;

/// out[t] = float(bf16(embed[ids[t]] * scale)), the checkpoint-dtype product.
void Embed(const std::int32_t* ids, const bf16_t* embed, float scale,
           float* out, int tokens, int hidden, hipStream_t stream);

/// out_bf16 = bf16(rms(x) * w), w may be null (no scale).
void RmsNormToBf16(const float* x, const bf16_t* w, bf16_t* out, int rows,
                   int hidden, float eps, hipStream_t stream);

/// After the output projection: h += rms(o) * post_attention_norm; then the
/// three feed-forward inputs from the updated residual h:
///   ff   = bf16(rms(h) * pre_ff_norm)
///   ff2  = bf16(rms(h) * pre_ff_norm_2)
///   rout = bf16(rms(h) * router_scale * hidden^-0.5)
void PostAttention(float* h, const float* o, const bf16_t* post_attention,
                   const bf16_t* pre_ff, const bf16_t* pre_ff_2,
                   const bf16_t* router_scale, bf16_t* ff, bf16_t* ff2,
                   bf16_t* rout, int rows, int hidden, float eps,
                   hipStream_t stream);

/// The mixture-of-experts output of each row: the k expert outputs of its
/// assignments (rows of `expert_out` at `slot[t*k + j]`) weighted by
/// `weight[t*k + j]`, summed in j order in FP32 (see Dispatch, ExpertDown).
struct ExpertOutputs {
  const bf16_t* expert_out{nullptr};
  const std::int32_t* slot{nullptr};
  const float* weight{nullptr};
  int k{0};
};

/// h = (h + rms(rms(mlp)*n1 + rms(moe)*n2) * n) * scalar, with moe the
/// combined expert outputs.
void FinishLayer(float* h, const float* mlp, const ExpertOutputs& moe,
                 const bf16_t* post_ff_1, const bf16_t* post_ff_2,
                 const bf16_t* post_ff, float layer_scalar, int rows,
                 int hidden, float eps, hipStream_t stream);

/// Q/K RMS norms, RoPE and V norm from the fused projection (fp32 rows of
/// q_heads*hd + kv*hd [+ kv*hd]). Q goes to q_out [t][head][hd]; K and V to
/// the cache [kv][capacity][hd] at slot first_slot + t. Global layers
/// (`shared_kv`) take V from K's projection before its norm and RoPE.
struct RopeTable {
  const float* inv_freq;  ///< hd/2 entries; zero entries do not rotate.
};
void QkvNormRope(const float* qkv, const bf16_t* q_norm, const bf16_t* k_norm,
                 RopeTable rope, bf16_t* q_out, bf16_t* k_cache,
                 bf16_t* v_cache, int tokens, int q_heads, int kv_heads,
                 int head_dim, bool shared_kv, int first_position,
                 int first_slot, int capacity, float eps, hipStream_t stream);

/// Attention with scale 1. Query t has absolute position q_pos0 + t, cache
/// slot s has position k_pos0 + s. A key is visible when s < kv_len and,
/// if causal, its position does not exceed the query's, and, with a window,
/// it lies less than `window` positions behind the query.
/// Scratch for bidirectional passes too short in queries to fill the GPU:
/// the key range is split across blocks and merged. A null `out` disables
/// splitting; `capacity` is in floats of `out`.
struct SplitWorkspace {
  float* out{nullptr};
  float* stats{nullptr};
  std::size_t capacity{0};
  int max_splits{1};
};
void Attention(const bf16_t* q, const bf16_t* k_cache, const bf16_t* v_cache,
               bf16_t* out, int tokens, int q_heads, int kv_heads, int head_dim,
               int capacity, int kv_len, int q_pos0, int k_pos0, bool causal,
               int window, hipStream_t stream,
               const SplitWorkspace& workspace = {});

/// Softmax over expert logits, top-k, renormalise, scale per expert.
void Route(const float* logits, const bf16_t* per_expert_scale,
           std::int32_t* index, float* weight, int rows, int experts, int k,
           hipStream_t stream);

/// Groups the rows*k assignments by expert. counts/offsets: experts entries;
/// source[pos] = token of the assignment at sorted position pos;
/// slot[t*k + j] = sorted position of token t's j-th assignment. `groups`
/// lists each expert's `group_rows`-row groups (64 or 128) as (expert,
/// first row within the expert) pairs, *group_count of them (at most
/// ExpertGroupBound).
void Dispatch(const std::int32_t* index, std::int32_t* counts,
              std::int32_t* offsets, std::int32_t* source, std::int32_t* slot,
              std::int32_t* groups, std::int32_t* group_count, int rows,
              int experts, int k, int group_rows, hipStream_t stream);

/// Upper bound of Dispatch's group count for `rows` rows.
int ExpertGroupBound(int rows, int experts, int k, int group_rows);

/// Dispatch's grouping, read by the expert products: one workgroup column
/// per group of `group_rows` rows.
struct ExpertRouting {
  const std::int32_t* counts{nullptr};
  const std::int32_t* offsets{nullptr};
  const std::int32_t* source{nullptr};
  const std::int32_t* groups{nullptr};
  const std::int32_t* group_count{nullptr};
  int max_groups{0};   ///< ExpertGroupBound for the pass's rows.
  int group_rows{64};  ///< Rows per group: 64 or 128.
};

/// Grouped expert gate/up projection with fused activation:
/// act[pos] = bf16(gelu(x[source[pos]] . Wg^T) * (x[source[pos]] . Wu^T)).
/// Q8_0 expert weights (int8 values + FP16 scale per 32 along the
/// reduction), see LayerWeights. Null `values` selects the BF16 weights.
struct ExpertQ8 {
  const std::int8_t* values{nullptr};
  const std::uint16_t* scales{nullptr};
};
/// FP8 E4M3 expert weights in the BF16 layout with one BF16 scale per weight
/// row (output channel), see LayerWeights. Takes precedence over Q8 when set.
struct ExpertFp8 {
  const std::uint8_t* values{nullptr};
  const bf16_t* scales{nullptr};
};
/// NVFP4 expert weights: E2M1 values packed two per byte (first value in the
/// low nibble) in the BF16 layout, one E4M3 scale per 16 values along the
/// reduction, and one FP32 scale per weight matrix stored for each of its
/// rows. Takes precedence over FP8 and Q8 when set.
struct ExpertFp4 {
  const std::uint8_t* values{nullptr};
  const std::uint8_t* block_scales{nullptr};
  const float* tensor_scales{nullptr};
};

void ExpertGateUp(const bf16_t* x, const bf16_t* gate_up,
                  const ExpertRouting& routing, bf16_t* act, int hidden,
                  int inter, hipStream_t stream, ExpertQ8 q8 = {},
                  ExpertFp8 fp8 = {}, ExpertFp4 fp4 = {});

/// Grouped expert down projection: out[pos] = bf16(act[pos] . Wd[e]^T), the
/// reference's checkpoint-dtype expert output.
void ExpertDown(const bf16_t* act, const bf16_t* down,
                const ExpertRouting& routing, bf16_t* out, int hidden,
                int inter, hipStream_t stream, ExpertQ8 q8 = {},
                ExpertFp8 fp8 = {}, ExpertFp4 fp4 = {});

/// Per canvas row of FP32 logits: z = cap*tanh(logit/cap)/temperature,
/// entropy of softmax(z), argmax(z), a Gumbel-max sample of softmax(z) and the
/// self-conditioning probabilities bf16(softmax(bf16(z))), blocked as
/// [vocab / 32][rows][32] for SoftEmbedProduct.
/// Token ids the sampler never picks (fixed-length generation suppresses the
/// end tokens): z = -inf, so they have zero probability, are never the
/// argmax or the sample, and take no self-conditioning mass.
struct SuppressedTokens {
  static constexpr int kMax = 4;
  std::int32_t ids[kMax]{};
  int count{0};
};
void SampleRows(const float* logits, bf16_t* probs, float* entropy,
                std::int32_t* argmax, std::int32_t* sample, int rows, int vocab,
                float softcap, float temperature, std::uint64_t seed,
                SuppressedTokens suppressed, hipStream_t stream);

/// Compact temperature-1 scores from raw LM-head logits. The vocabulary is
/// split into 4096-ID tiles; each tile supplies a log-sum-exp partial and
/// its top 20, then a row pass merges the partials. No full logits leave the
/// device. Scratch sizes are rows * ceil(vocab/4096) for max/sum and 20
/// times that for top IDs/values. Outputs are [rows][20], [rows][labels],
/// and [rows] log-sum-exp and argmax. Ties choose the smaller token ID.
struct ScoreWorkspace {
  float* tile_max{nullptr};
  float* tile_sum{nullptr};
  std::int32_t* tile_ids{nullptr};
  float* tile_values{nullptr};
  std::int32_t* top_ids{nullptr};
  float* top_logprobs{nullptr};
  float* label_logprobs{nullptr};
  float* logsumexp{nullptr};
  std::int32_t* argmax{nullptr};
};
void ScoreRows(const float* logits, int rows, int vocab, float softcap,
               const std::int32_t* labels, int label_count,
               ScoreWorkspace workspace, hipStream_t stream);

/// The pinned rows contribute no soft embedding to the next denoise step.
void ZeroPinnedSoftRows(float* soft, const std::uint8_t* pinned, int rows,
                        int hidden, hipStream_t stream);

/// Self-conditioning input: bf16(rms(bf16(bf16(soft) * scale)) * pre_norm).
void SoftEmbedNorm(const float* soft, const bf16_t* pre_norm, float scale,
                   bf16_t* out, int rows, int hidden, float eps,
                   hipStream_t stream);

/// h = rms(bf16(embeds + bf16(signal))), no scale; signal may be null.
void SelfConditionCombine(const float* embeds, const float* signal, float* h,
                          int rows, int hidden, float eps, hipStream_t stream);

/// out[i] = sum over s < splits of partials[s * count + i].
void SumPartials(const float* partials, float* out, std::size_t count,
                 int splits, hipStream_t stream);

/// out[256 x hidden] (fp32) = probs * embed[vocab x hidden], with probs in
/// SampleRows' blocked layout. The vocabulary is split over `splits`
/// workgroup columns whose fp32 partials (splits * 256 * hidden floats) are
/// summed in order.
void SoftEmbedProduct(const bf16_t* probs, const bf16_t* embed, float* partials,
                      float* out, int rows, int vocab, int hidden, int splits,
                      hipStream_t stream);

/// Dense product out[m x n] (fp32) = a[m x k] (row-major) * w[n x k]^T,
/// with w blocked by BlockLinearWeights; n % 128 == 0, k % 64 == 0. Shapes
/// with few output tiles split the reduction through `partials`
/// (LinearPartialFloats(m, n, k) floats, summed in order; may be null when
/// that is zero).
std::size_t LinearPartialFloats(int m, int n, int k);
void Linear(const bf16_t* a, const bf16_t* w, float* partials, float* out,
            int m, int n, int k, hipStream_t stream);
/// w[n][k] to [n / 128][k / 64][128][64].
void BlockLinearWeights(const bf16_t* w, bf16_t* out, int n, int k,
                        hipStream_t stream);
/// The MLP's gate/up product with its activation:
/// act[m x inter] = bf16(gelu(bf16(gate)) * bf16(up)) with gate, up the two
/// halves of a[m x k] * w^T, w (2 inter x k, gate rows then up rows) blocked
/// by BlockGateUpWeights (gelu: the tanh approximation).
void LinearGeluMul(const bf16_t* a, const bf16_t* w, bf16_t* act, int m,
                   int inter, int k, hipStream_t stream);
/// w[2 inter][k] to BlockLinearWeights' layout with each 128-row tile
/// holding 64 gate rows and the matching 64 up rows.
void BlockGateUpWeights(const bf16_t* w, bf16_t* out, int inter, int k,
                        hipStream_t stream);

/// Q8_0 blocks (FP16 scale + 32 int8) to BF16: out = bf16(scale * q).
void DequantQ8(const std::uint8_t* blocks, bf16_t* out, std::size_t values,
               hipStream_t stream);
/// FP8 E4M3 rows with one BF16 scale per row to BF16:
/// out[r][c] = bf16(fp8[r][c] * scale[r]).
void DequantFp8Rows(const std::uint8_t* values, const bf16_t* scales,
                    bf16_t* out, std::size_t rows, std::size_t cols,
                    hipStream_t stream);
/// Q8_0 blocks to separate int8 values and FP16 scales.
void RepackQ8(const std::uint8_t* blocks, std::int8_t* values,
              std::uint16_t* scales, std::size_t block_count,
              hipStream_t stream);
/// out = bf16(in), round to nearest even.
void F32ToBf16(const float* in, bf16_t* out, std::size_t count,
               hipStream_t stream);

/// Moves `count` slots starting at `from` to slot 0 of a [heads][capacity][hd]
/// cache, through `scratch` (at least heads*count*hd elements).
void ShiftCache(bf16_t* cache, bf16_t* scratch, int heads, int capacity,
                int head_dim, int from, int count, hipStream_t stream);

}  // namespace gufo::models::diffusion_gemma::kernels

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_KERNELS_HPP_
