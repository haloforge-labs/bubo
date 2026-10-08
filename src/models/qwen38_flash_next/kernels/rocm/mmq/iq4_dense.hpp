// Dense IQ4 only: retain the routed kernel's lane/block mapping and sum tree.
// Each lane decodes its weight fragment once for all activation columns.
template<ggml_type type>
struct IQ4DenseFragment {
  int2 values[type == GGML_TYPE_IQ4_XS ? 4 : 2];
  float scale;
  int subscale;

  __device__ __forceinline__ IQ4DenseFragment(const void* weights, int block,
                                             int iqs) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
      const auto* w = static_cast<const block_iq4_xs*>(weights) + block;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        values[j] = get_int_from_table_16(get_int_b4(w->qs, iqs + j),
                                         kvalues_iq4nl);
      subscale = (((w->scales_l[iqs / 8] >> (iqs & 4)) & 15) |
                  (((w->scales_h >> (iqs / 2)) & 3) << 4)) - 32;
      scale = __half2float(w->d);
    } else {
      const auto* w = static_cast<const block_iq4_nl*>(weights) + block;
#pragma unroll
      for (int j = 0; j < 2; ++j)
        values[j] = get_int_from_table_16(get_int_b2(w->qs, iqs + j),
                                         kvalues_iq4nl);
      scale = __half2float(w->d);
    }
  }

  __device__ __forceinline__ float Accumulate(const block_q8_1* x, int iqs, float acc) const {
    int sumi = 0;
    if constexpr (type == GGML_TYPE_IQ4_XS) {
      x += iqs / 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sumi = ggml_hip_dp4a(values[j].x, get_int_b4(x->qs, j), sumi);
        sumi = ggml_hip_dp4a(values[j].y, get_int_b4(x->qs, j + 4), sumi);
      }
      sumi *= subscale;
    } else {
      const auto* q = reinterpret_cast<const int*>(x->qs) + iqs;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        sumi = ggml_hip_dp4a(values[j].x, q[j], sumi);
        sumi = ggml_hip_dp4a(values[j].y, q[j + 4], sumi);
      }
    }
    // Match the original generated FP32 products and FMA, which differ by
    // format under fast-math. XS rounds the two scales first; NL rounds the
    // weight-scale/integer-dot product first. Neither uses half products.
    if constexpr (type == GGML_TYPE_IQ4_XS) {
      const float d = __fmul_rn(scale, __low2float(x->ds));
      return __fmaf_rn(d, float(sumi), acc);
    } else {
      const float d = __fmul_rn(scale, float(sumi));
      return __fmaf_rn(d, __low2float(x->ds), acc);
    }
  }
};

template<ggml_type type, int tokens, int row_tile, int waves>
__launch_bounds__(32 * waves) static __global__ void mul_mat_vec_iq4_dense(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    float* __restrict__ output, int k, int rows, int input_stride) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
  constexpr int step = vdr * 32 / qi;
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * waves + threadIdx.x / 32) * row_tile;
  if (row0 >= rows)
    return;
  const int blocks = k / qk;
  const int iqs = vdr * (lane % (qi / vdr));
  float sum[tokens][row_tile]{};
  for (int kb = lane / (qi / vdr); kb < blocks; kb += step) {
#pragma unroll
    for (int r = 0; r < row_tile; ++r) {
      const IQ4DenseFragment<type> w(weights,
          min(row0 + r, rows - 1) * blocks + kb, iqs);
#pragma unroll
      for (int t = 0; t < tokens; ++t)
        sum[t][r] = w.Accumulate(input + t * input_stride + kb * (qk / QK8_1),
                                  iqs, sum[t][r]);
    }
  }
#pragma unroll
  for (int t = 0; t < tokens; ++t) {
#pragma unroll
    for (int r = 0; r < row_tile; ++r)
      sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
    if (lane < row_tile && row0 + lane < rows) {
      const float value = sum[t][lane];
      output[t * rows + row0 + lane] = isfinite(value) ? value : 0.0f;
    }
  }
}

template<ggml_type type, int tokens, int row_tile>
static void launch_iq4_dense_tile(const void* weights, const block_q8_1* input,
                                   float* output, int k, int rows,
                                   int input_stride, hipStream_t stream) {
  constexpr int waves = 4;
  mul_mat_vec_iq4_dense<type, tokens, row_tile, waves>
      <<<(rows + row_tile * waves - 1) / (row_tile * waves),
         32 * waves, 0, stream>>>(weights, input, output, k, rows, input_stride);
}

template<ggml_type type, int tokens = 1>
static void launch_iq4_dense(const void* weights, const block_q8_1* input,
                              float* output, int k, int rows, int n,
                              int input_stride, hipStream_t stream) {
  if (n == tokens) {
    // Match the original dense-as-one-expert row tiles, including the lane
    // that stores each row's XOR reduction. Other NL widths use two rows.
    if constexpr (type == GGML_TYPE_IQ4_NL && tokens > 1) {
      if (k == 640)
        launch_iq4_dense_tile<type, tokens, 4>(weights, input, output, k, rows,
                                               input_stride, stream);
      else
        launch_iq4_dense_tile<type, tokens, 2>(weights, input, output, k, rows,
                                               input_stride, stream);
    } else {
      launch_iq4_dense_tile<type, tokens, 2>(weights, input, output, k, rows,
                                             input_stride, stream);
    }
  } else if constexpr (tokens < 8) {
    launch_iq4_dense<type, tokens + 1>(weights, input, output, k, rows, n,
                                        input_stride, stream);
  }
}

void mul_mat_vec_iq4_dense_dispatch(const void* weights, ggml_type type,
                                    const block_q8_1* input, float* output,
                                    int k, int rows, int tokens, int input_stride,
                                    hipStream_t stream) {
  if (type == GGML_TYPE_IQ4_XS)
    launch_iq4_dense<GGML_TYPE_IQ4_XS>(weights, input, output, k, rows, tokens,
                                       input_stride, stream);
  else
    launch_iq4_dense<GGML_TYPE_IQ4_NL>(weights, input, output, k, rows, tokens,
                                       input_stride, stream);
}
