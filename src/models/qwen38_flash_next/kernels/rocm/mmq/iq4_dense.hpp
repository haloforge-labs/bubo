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

// Both IQ4 formats pack K0..15 in low nibbles, K16..31 in high nibbles.
// Decode directly from global memory; chain two full K16 WMMA operations
// in int32 before one FP32 scale/FMA. No LDS weight or activation tile.
template<ggml_type type, int token_tiles, bool ragged, int waves>
__launch_bounds__(32 * waves) static __global__ void mul_mat_iq4_decode_batch(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    float* __restrict__ output, int k, int rows, int input_stride, int tokens) {
  using Int4 = __attribute__((ext_vector_type(4))) int;
  using Int8 = __attribute__((ext_vector_type(8))) int;
  static_assert(waves == 4 || waves == 8 || waves == 16);
  __shared__ float partial[16 * 16 * waves];
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % 16;
  const int half = lane / 16;
  const int first_row = blockIdx.x * 16;
  const int row = min(first_row + sub, rows - 1);
  const int blocks = k / QK8_1;
  float acc[token_tiles][8]{};
  for (int kb = wave; kb < blocks; kb += waves) {
    Int4 low, high;
    float scale;
    if constexpr (type == GGML_TYPE_IQ4_XS) {
      const auto* w = static_cast<const block_iq4_xs*>(weights) +
          row * (blocks / 8) + kb / 8;
      const int block = kb % 8;
      const int ls = ((w->scales_l[block / 2] >> (4 * (block % 2))) & 15) |
          (((w->scales_h >> (2 * block)) & 3) << 4);
      // FP16 significand times a six-bit integer is exact in FP32.
      scale = __fmul_rn(__half2float(w->d), float(ls - 32));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int2 codes = get_int_from_table_16(
            get_int_b4(w->qs, 4 * block + j), kvalues_iq4nl);
        low[j] = codes.x;
        high[j] = codes.y;
      }
    } else {
      const auto* w = static_cast<const block_iq4_nl*>(weights) +
          row * blocks + kb;
      scale = __half2float(w->d);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // NL qs is only two-byte aligned, including the final packed block.
        const int2 codes = get_int_from_table_16(
            get_int_b2(w->qs, j), kvalues_iq4nl);
        low[j] = codes.x;
        high[j] = codes.y;
      }
    }
    float scales[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      scales[i] = __shfl(scale, 2 * i + half, 32);
#pragma unroll
    for (int tile = 0; tile < token_tiles; ++tile) {
      const int token = ragged && tile == token_tiles - 1
                            ? min(tile * 16 + sub, tokens - 1)
                            : tile * 16 + sub;
      const auto* x = input + token * input_stride + kb;
      const float x_scale = __low2float(x->ds);
      const Int4 values0 = {get_int_b4(x->qs, 0), get_int_b4(x->qs, 1),
                            get_int_b4(x->qs, 2), get_int_b4(x->qs, 3)};
      const Int4 values1 = {get_int_b4(x->qs, 4), get_int_b4(x->qs, 5),
                            get_int_b4(x->qs, 6), get_int_b4(x->qs, 7)};
      Int8 dots{};
      dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
          true, low, true, values0, dots, false);
      dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
          true, high, true, values1, dots, false);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[tile][i] =
            __fmaf_rn(x_scale, __fmul_rn(scales[i], float(dots[i])), acc[tile][i]);
    }
  }
  // Reuse the 4/8/16 KiB output transpose for each tile. Every valid
  // (row, token) has the same wave K traversal and descending-delta sum tree.
#pragma unroll
  for (int tile = 0; tile < token_tiles; ++tile) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int out = sub * 16 + 2 * i + half;
      const int swizzle = (out ^ (out >> 4)) & (waves - 1);
      partial[out * waves + (wave ^ swizzle)] = acc[tile][i];
    }
    __syncthreads();
    // Four waves assign two outputs/thread; sixteen waves leave half idle
    // during reduction only. All threads still participate in barriers.
    for (int out = threadIdx.x; out < 256; out += 32 * waves) {
      const int swizzle = (out ^ (out >> 4)) & (waves - 1);
      float sums[waves];
#pragma unroll
      for (int i = 0; i < waves; ++i)
        sums[i] = partial[out * waves + (i ^ swizzle)];
#pragma unroll
      for (int delta = waves / 2; delta; delta /= 2) {
#pragma unroll
        for (int i = 0; i < delta; ++i)
          sums[i] = __fadd_rn(sums[i], sums[i + delta]);
      }
      const int out_row = first_row + out % 16;
      const int token = tile * 16 + out / 16;
      if (out_row < rows && (!ragged || token < tokens)) {
        const float value = sums[0];
        output[token * rows + out_row] = isfinite(value) ? value : 0.0f;
      }
    }
    __syncthreads();
  }
}

template<ggml_type type, int token_tiles, int waves>
static void launch_iq4_matrix_tile(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int tokens, int input_stride,
    hipStream_t stream) {
  if (tokens == 16 * token_tiles)
    mul_mat_iq4_decode_batch<type, token_tiles, false, waves>
        <<<(rows + 15) / 16, 32 * waves, 0, stream>>>(
            weights, input, output, k, rows, input_stride, tokens);
  else
    mul_mat_iq4_decode_batch<type, token_tiles, true, waves>
        <<<(rows + 15) / 16, 32 * waves, 0, stream>>>(
            weights, input, output, k, rows, input_stride, tokens);
}

template<ggml_type type, int waves>
static void launch_iq4_matrix(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int n, int input_stride, hipStream_t stream) {
  if (n <= 16)
    launch_iq4_matrix_tile<type, 1, waves>(
        weights, input, output, k, rows, n, input_stride, stream);
  else if (n <= 32)
    launch_iq4_matrix_tile<type, 2, waves>(
        weights, input, output, k, rows, n, input_stride, stream);
  else if (n <= 48)
    launch_iq4_matrix_tile<type, 3, waves>(
        weights, input, output, k, rows, n, input_stride, stream);
  else
    launch_iq4_matrix_tile<type, 4, waves>(
        weights, input, output, k, rows, n, input_stride, stream);
}


// Wide tiles are separate: the 1..8 kernels and launch recursion above stay
// unchanged. Stage each K tile once through LDS for all four output waves.
// The lane's K traversal, FP32 fragment arithmetic and XOR sum tree match N=1.
template<ggml_type type, int tokens>
__launch_bounds__(128) static __global__ void mul_mat_vec_iq4_dense_batch(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    float* __restrict__ output, int k, int rows, int input_stride, int n_valid) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
  constexpr int step = vdr * 32 / qi;
  constexpr int xblocks = step * (qk / QK8_1);
  // Two output rows also for NL K=640: match the N=1 storing lane and keep
  // sum registers bounded at 32 rows. G1 checks this shape explicitly.
  constexpr int row_tile = 2;
  __shared__ block_q8_1 staged[tokens][xblocks];
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * 4 + threadIdx.x / 32) * row_tile;
  const int blocks = k / qk;
  const int iqs = vdr * (lane % (qi / vdr));
  float sum[tokens][row_tile]{};
  for (int tile = 0; tile < blocks; tile += step) {
    for (int i = threadIdx.x; i < tokens * xblocks; i += 128) {
      const int t = i / xblocks, b = i % xblocks;
      const int src = tile * (qk / QK8_1) + b;
      if (t < n_valid && src < k / QK8_1)
        staged[t][b] = input[t * input_stride + src];
    }
    __syncthreads();
    const int kb = tile + lane / (qi / vdr);
    if (row0 < rows && kb < blocks) {
#pragma unroll
      for (int r = 0; r < row_tile; ++r) {
        const IQ4DenseFragment<type> w(weights,
            min(row0 + r, rows - 1) * blocks + kb, iqs);
#pragma unroll
        for (int t = 0; t < tokens; ++t) {
          if (t < n_valid)
            sum[t][r] = w.Accumulate(staged[t] + (kb - tile) * (qk / QK8_1),
                                    iqs, sum[t][r]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int t = 0; t < tokens; ++t) {
    if (t < n_valid) {
#pragma unroll
      for (int r = 0; r < row_tile; ++r)
        sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
      if (lane < row_tile && row0 + lane < rows) {
        // Fixed indexing keeps the accumulator array in VGPRs at 24/32.
        const float value = lane == 0 ? sum[t][0] : sum[t][1];
        output[t * rows + row0 + lane] = isfinite(value) ? value : 0.0f;
      }
    }
  }
}

template<ggml_type type, int tokens>
static void launch_iq4_dense_batch_tile(const void* weights,
    const block_q8_1* input, float* output, int k, int rows, int input_stride,
    int n_valid, hipStream_t stream) {
  mul_mat_vec_iq4_dense_batch<type, tokens>
      <<<(rows + 7) / 8, 128, 0, stream>>>(weights, input, output, k, rows,
                                           input_stride, n_valid);
}

template<ggml_type type>
static void launch_iq4_dense_batch(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int n_valid, int input_stride,
    hipStream_t stream) {
  if (n_valid <= 8)
    launch_iq4_dense_batch_tile<type, 8>(weights, input, output, k, rows,
                                         input_stride, n_valid, stream);
  else if (n_valid <= 16)
    launch_iq4_dense_batch_tile<type, 16>(weights, input, output, k, rows,
                                          input_stride, n_valid, stream);
  else if (n_valid <= 24)
    launch_iq4_dense_batch_tile<type, 24>(weights, input, output, k, rows,
                                          input_stride, n_valid, stream);
  else
    launch_iq4_dense_batch_tile<type, 32>(weights, input, output, k, rows,
                                          input_stride, n_valid, stream);
}

// Copy packed weights exactly once from global memory for all token groups.
// The padded LDS row pitch is word-aligned even for the 18-byte NL blocks.
// memcpy permits NL's two-byte global alignment and the final half-word tail.
template<ggml_type type, int row_waves, int groups>
__device__ __forceinline__ void stage_iq4_wide_weights(
    unsigned int* dst, const void* weights, int tile, int blocks, int rows) {
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
  constexpr int step = vdr * 32 / qi;
  constexpr int bytes = type == GGML_TYPE_IQ4_XS ? sizeof(block_iq4_xs) : sizeof(block_iq4_nl);
  constexpr int words = step * bytes / 4;
  constexpr int block_rows = 2 * row_waves;
  for (int i = threadIdx.x; i < block_rows * words;
       i += 32 * row_waves * groups) {
    const int r = i / words;
    const int pos = (i % words) * 4;
    const int row = blockIdx.x * block_rows + r;
    const int available = min(step, blocks - tile) * bytes;
    unsigned int value = 0;
    if (row < rows && pos < available) {
      const auto* src = static_cast<const unsigned char*>(weights) +
          (static_cast<size_t>(row) * blocks + tile) * bytes + pos;
      if (pos + 4 <= available) {
        __builtin_memcpy(&value, src, 4);
      } else {
        unsigned short tail;
        __builtin_memcpy(&tail, src, 2);
        value = tail;
      }
    }
    dst[i] = value;
  }
}

// Eight tokens per wave, two output rows per wave: only sixteen accumulators.
// Every token group decodes the shared packed fragment; global weight reads
// are owned by the cooperative loader, never repeated by token groups.
// Activations remain in the cache hierarchy, eliminating N*K LDS re-staging.
template<ggml_type type, int groups, int row_waves, bool double_buffer>
__launch_bounds__(32 * groups * row_waves)
static __global__ void mul_mat_vec_iq4_wide(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    float* __restrict__ output, int k, int rows, int input_stride, int n_valid) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
  constexpr int step = vdr * 32 / qi;
  constexpr int bytes = type == GGML_TYPE_IQ4_XS ? sizeof(block_iq4_xs) : sizeof(block_iq4_nl);
  constexpr int row_words = step * bytes / 4;
  constexpr int tile_words = 2 * row_waves * row_words;
  __shared__ unsigned int staged[double_buffer ? 2 : 1][tile_words];
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
  const int row_wave = wave % row_waves;
  const int token0 = (wave / row_waves) * 8;
  const int row0 = (blockIdx.x * row_waves + row_wave) * 2;
  const int blocks = k / qk;
  const int local_kb = lane / (qi / vdr);
  const int iqs = vdr * (lane % (qi / vdr));
  float sum[8][2]{};
  int buffer = 0;
  if constexpr (double_buffer) {
    stage_iq4_wide_weights<type, row_waves, groups>(
        staged[0], weights, 0, blocks, rows);
    __syncthreads();
  }
  for (int tile = 0; tile < blocks; tile += step) {
    if constexpr (double_buffer) {
      // The alternate buffer is free after the preceding tile's barrier.
      if (tile + step < blocks)
        stage_iq4_wide_weights<type, row_waves, groups>(
            staged[buffer ^ 1], weights, tile + step, blocks, rows);
    } else {
      stage_iq4_wide_weights<type, row_waves, groups>(
          staged[0], weights, tile, blocks, rows);
      __syncthreads();
    }
    const int kb = tile + local_kb;
    if (kb < blocks) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (row0 + r < rows) {
          const IQ4DenseFragment<type> w(
              staged[buffer] + (2 * row_wave + r) * row_words,
              local_kb, iqs);
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            if (token0 + t < n_valid)
              sum[t][r] = w.Accumulate(
                  input + (token0 + t) * input_stride + kb * (qk / QK8_1),
                  iqs, sum[t][r]);
          }
        }
      }
    }
    // Completes both current-buffer readers and next-buffer writers. No wave
    // returns early, including an empty output wave or partial token group.
    __syncthreads();
    if constexpr (double_buffer)
      buffer ^= 1;
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    if (token0 + t < n_valid) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
      if (lane < 2 && row0 + lane < rows) {
        const float value = lane == 0 ? sum[t][0] : sum[t][1];
        output[(token0 + t) * rows + row0 + lane] = isfinite(value) ? value : 0.0f;
      }
    }
  }
}

template<ggml_type type, int groups, int row_waves, bool double_buffer>
static void launch_iq4_wide_tile(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int n, int input_stride, hipStream_t stream) {
  mul_mat_vec_iq4_wide<type, groups, row_waves, double_buffer>
      <<<(rows + 2 * row_waves - 1) / (2 * row_waves),
         32 * groups * row_waves, 0, stream>>>(
          weights, input, output, k, rows, input_stride, n);
}

template<ggml_type type, int row_waves, bool double_buffer>
static void launch_iq4_wide(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int n, int input_stride, hipStream_t stream) {
  if (n <= 16)
    launch_iq4_wide_tile<type, 2, row_waves, double_buffer>(
        weights, input, output, k, rows, n, input_stride, stream);
  else if (n <= 24)
    launch_iq4_wide_tile<type, 3, row_waves, double_buffer>(
        weights, input, output, k, rows, n, input_stride, stream);
  else
    launch_iq4_wide_tile<type, 4, row_waves, double_buffer>(
        weights, input, output, k, rows, n, input_stride, stream);
}

template<ggml_type type>
static void launch_iq4_wide_variant(const void* weights, const block_q8_1* input,
    float* output, int k, int rows, int n, int input_stride, hipStream_t stream) {
  using namespace gufo::models::qwen38_flash_next;
  const int variant = Iq4WideVariant();
  if (variant == 4) {
    const int waves = Iq4WmmaWaves();
    if (waves == 4)
      launch_iq4_matrix<type, 4>(weights, input, output, k, rows, n, input_stride, stream);
    else if (waves == 16)
      launch_iq4_matrix<type, 16>(weights, input, output, k, rows, n, input_stride, stream);
    else
      launch_iq4_matrix<type, 8>(weights, input, output, k, rows, n, input_stride, stream);
    return;
  }
  // Legacy A/B variants retain their <=32 geometry and arithmetic.
  if (n > 32) {
    launch_iq4_wide_variant<type>(weights, input, output, k, rows, 32, input_stride, stream);
    launch_iq4_wide_variant<type>(weights, input + 32 * input_stride, output + 32 * rows,
        k, rows, n - 32, input_stride, stream);
    return;
  }
  if (n <= 8) {
    launch_iq4_dense<type>(weights, input, output, k, rows, n, input_stride, stream);
    return;
  }
  if (variant == 0) {
    launch_iq4_dense_batch<type>(weights, input, output, k, rows, n, input_stride, stream);
  } else if (variant == 3) {
    if (Iq4WideRowsPerBlock(variant, rows) == 16)
      launch_iq4_wide<type, 8, true>(weights, input, output, k, rows, n, input_stride, stream);
    else
      launch_iq4_wide<type, 4, true>(weights, input, output, k, rows, n, input_stride, stream);
  } else {
    if (Iq4WideRowsPerBlock(variant, rows) == 16)
      launch_iq4_wide<type, 8, false>(weights, input, output, k, rows, n, input_stride, stream);
    else
      launch_iq4_wide<type, 4, false>(weights, input, output, k, rows, n, input_stride, stream);
  }
}

void mul_mat_vec_iq4_dense_batch_dispatch(const void* weights, ggml_type type,
    const block_q8_1* input, float* output, int k, int rows, int n_valid,
    int input_stride, hipStream_t stream) {
  if (type == GGML_TYPE_IQ4_XS)
    launch_iq4_wide_variant<GGML_TYPE_IQ4_XS>(weights, input, output, k, rows,
                                             n_valid, input_stride, stream);
  else
    launch_iq4_wide_variant<GGML_TYPE_IQ4_NL>(weights, input, output, k, rows,
                                             n_valid, input_stride, stream);
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
