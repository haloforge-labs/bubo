// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {
int qfn_mmq_init(int device);

void qfn_mmq_set_routed_max_expert_rows(int rows);

void qfn_mmq_set_routed_tile_cols(int cols);

int qfn_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

int qfn_mmq_k_dense(int weight_type, const void* W, const float* X, float* out, int M, int N, int K, hipStream_t stream);
int qfn_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    hipStream_t  stream);

int qfn_mmq_q4_K_moe_pair_unique(const void* W_a, const void* W_b,
                                 const float* X_f32, const int32_t* ids,
                                 float* out_a, float* out_b, int M, int K,
                                 int n_tokens, int n_experts, int n_expert_used,
                                 hipStream_t stream);

// Routed vector projection for GGML Q4_K/Q5_K/Q5_1/Q8_0. Optional paired
// weights use the same shape, routing and quantized input; both outputs are
// [n_tokens][n_expert_used][M]. W_b and out_b must be supplied together.
int qfn_mmq_moe_vec(int weight_type, const void* W, const float* X_f32,
                    const int32_t* ids, float* out, int M, int K, int n_tokens,
                    int n_experts, int n_expert_used, hipStream_t stream,
                    const void* W_b = nullptr, float* out_b = nullptr);

// Q4_K/Q5_K/Q8_0 gate/up for 1–8 tokens, with the SwiGLU result in out.
// Q4_K also accepts up to 64 independent rows, grouped by expert.
int qfn_mmq_moe_gated_vec(int weight_type, const void* gate, const void* up,
                          const float* x, const int32_t* ids, float* out, int m,
                          int k, int tokens, int experts, int experts_used,
                          hipStream_t stream);

// Fixed 2560/640, top-10, 512-expert decode chain. Q4_K gate/up and
// Q5_1/Q8_0 down, N=2..8. Caller scratch persists across graph replay.
size_t qfn_mmq_expert_v2k_scratch_bytes(int max_tokens);
int qfn_mmq_expert_v2k(int gu_type, int down_type, const void* gate_w,
                       const void* up_w, const void* down_w, const float* x,
                       const int32_t* ids, float* gate, float* up, float* out,
                       int n_tokens, void* scratch, size_t scratch_bytes,
                       hipStream_t stream);
// Stages used by the whole chain and the opt-in routed microbench.
// GU outputs and hq are in compact entry order; down restores slot order.
int qfn_mmq_expert_v2k_gu(const void* gate_w, const void* up_w, const float* x,
                          const int32_t* ids, float* gate, float* up,
                          int n_tokens, void* scratch, size_t scratch_bytes,
                          hipStream_t stream);
int qfn_mmq_expert_v2k_bridge(const float* gate, const float* up, void* hq,
                              int entries, hipStream_t stream);
int qfn_mmq_expert_v2k_down(int down_type, const void* weights, float* out,
                            int n_tokens, void* scratch, size_t scratch_bytes,
                            hipStream_t stream);
// qfn_mmq_moe_gated_vec over x already in qfn_mmq_quantize_q8_1's layout.
int qfn_mmq_moe_gated_vec_preq(int weight_type, const void* gate,
                               const void* up, const void* x_q8,
                               const int32_t* ids, float* out, int m, int k,
                               int tokens, int experts, int experts_used,
                               hipStream_t stream);

size_t qfn_mmq_q8_1_bytes(int N, int K);

int qfn_mmq_quantize_q8_1(const float* X_f32, void* X_q8, int N, int K,
                          hipStream_t stream);

// Exact decode arithmetic for 1–8 rows, or up to 32 ungated rows.
// Wide projections (M >= 8192, K = 2560) accept up to 48 ungated rows.
// The 320×10240 HC projection also accepts larger multiples of eight.
// Only N output rows are written.
/// The HC mixer's 320 x 10240 down projection for N <= 8 tokens with its
/// SiluScale (silu(x * scale)) fused into the write.
int qfn_mmq_q8_0_hc_down_silu(const void* W_q8_0, const void* X_q8,
                              float* out_f32, int N, float scale,
                              hipStream_t stream);
/// Deeper load prefetch in the HC down projection for 2-8 tokens (outputs
/// are bit-identical; one token always keeps depth 2). Off by default.
void qfn_mmq_set_hc_down_deep_prefetch(int enabled);
int qfn_mmq_q8_0_dense_vec_preq(const void* W_q8_0, const void* W_gate,
                                const void* X_q8, float* out_f32, int M, int N,
                                int K, hipStream_t stream);
/// Host launch submissions (including graph capture), process lifetime.
unsigned long long qfn_mmq_iq4_dense_launch_count();
/// Decode twin: dense projections of 1..8 pre-quantised Q8_1 activation rows.
/// IQ4_XS / IQ4_NL decode weights once for all columns. Other types retain
/// the one-expert vector dispatch. zero_ids holds >= N zeros; output is [N][M].
/// Invalid types, widths, pointers or block dimensions return an error.
int qfn_mmq_dense_vec_preq(int weight_type, const void* W, const void* X_q8,
                           const int32_t* zero_ids, float* out_f32, int M, int N,
                           int K, hipStream_t stream);

/// Batched-only IQ4, N=1..64 valid rows; loads/stores never touch padded rows.
/// Single-session callers retain qfn_mmq_dense_vec_preq unchanged.
int qfn_mmq_iq4_dense_batch_preq(int weight_type, const void* W,
    const void* X_q8, float* out, int M, int N, int K, hipStream_t stream);

int qfn_mmq_build_ids_maps(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, hipStream_t stream);

int qfn_mmq_q8_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q4_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_1_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q3_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q6_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

// Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 by GGML type id; any other type returns -1.
int qfn_mmq_k_moe_raw(
    int weight_type, const void * W, const float * X_f32, const int32_t * ids,
    float * out, int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

// IQ experts (e.g. unsloth UD-IQ4_XS: IQ3_S/IQ4_XS gate/up, IQ4_NL down).
int qfn_mmq_iq3_s_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
// ISTA GSQ-RCO IQ3_S rung: IQ3_XXS and IQ2_S gate/up, Q2_0 down (memory lane).
int qfn_mmq_iq3_xxs_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_iq2_xxs_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_iq2_xs_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_iq2_s_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_q2_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_iq4_xs_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);
int qfn_mmq_iq4_nl_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

}
