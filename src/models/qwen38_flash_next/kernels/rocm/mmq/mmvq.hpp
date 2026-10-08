#pragma once
#include "common.hpp"

constexpr int MMVQ_MAX_BATCH_SIZE = 8;
constexpr int MMVQ_MAX_ROUTED_BATCH = 64;

// One expert's independent inputs, in groups of eight. The anchor also
// identifies an inactive slot that must be zeroed.
struct MoeBatchGroup {
  int expert;
  int anchor;
  int token[MMVQ_MAX_BATCH_SIZE];
  uint32_t slots[MMVQ_MAX_BATCH_SIZE];
};

// Tuned routed-vector batch limits for gfx1151.
constexpr __host__ __device__ int mmvq_moe_max_batch(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 4 : 8;
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream);
void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream);
void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream);

void mul_mat_vec_moe_batch_wave64(const void* gate, const void* up,
                                 const block_q8_1* input, const int32_t* groups,
                                 float* output, int k, int rows,
                                 int experts_used, int input_stride,
                                 hipStream_t stream);

// Dense IQ4 batched path; routed projections retain their existing dispatch.
void mul_mat_vec_iq4_dense_dispatch(const void* weights, ggml_type type,
                                    const block_q8_1* input, float* output,
                                    int k, int rows, int tokens, int input_stride,
                                    hipStream_t stream);

// Separate wide adapter; the original dense dispatch remains N<=8 only.
void mul_mat_vec_iq4_dense_batch_dispatch(const void* weights, ggml_type type,
    const block_q8_1* input, float* output, int k, int rows, int n_valid,
    int input_stride, hipStream_t stream);
