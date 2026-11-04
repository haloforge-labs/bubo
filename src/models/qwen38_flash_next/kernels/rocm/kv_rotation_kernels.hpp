// Included only by kernels.hip.cpp, inside its private namespace, when the
// experimental rotated-q8 build is enabled. No legacy kernel body is changed.
// Numerics revision 23: normalized natural-order Sylvester FWHT, F32 RNE at
// normalization and every ascending butterfly output. No FMA or reciprocal.

/// One channel per thread, a complete D256 head per 256-thread block. The
/// first five stages use wave32 register transport. Higher channel bits use
/// this block's existing shared head, with every read finished before a write.
template<unsigned kRotation>
__device__ __forceinline__ float RotateKvChannel(float value, float* shared) {
  static_assert(kRotation == 64 || kRotation == 256);
  const unsigned tid = threadIdx.x;
  value = OpaqueQ8KvF32(value * (kRotation == 256 ? 0.0625F : 0.125F));
#pragma unroll
  for (unsigned distance = 1; distance < 32; distance <<= 1) {
    const float other = XorLane(value, distance);
    value =
        OpaqueQ8KvF32((tid & distance) == 0 ? value + other : other - value);
  }
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (unsigned distance = 32; distance < kRotation; distance <<= 1) {
    const float other = shared[tid ^ distance];
    __syncthreads();
    value =
        OpaqueQ8KvF32((tid & distance) == 0 ? value + other : other - value);
    shared[tid] = value;
    __syncthreads();
  }
  return value;
}

/// Separate correctness-first producer: keep all legacy RMS/RoPE operations,
/// then transform before the canonical Store. This also handles both predictor
/// cache-only catch-up callers (heads == 0, null Q/gate/query gamma).
template<bool kYarn>
__global__ void PrepareRotatedAttentionKernel(
    const float* __restrict__ packed, std::uint32_t stride,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float* __restrict__ q, float* __restrict__ gate,
    void* __restrict__ key_storage, void* __restrict__ value_storage,
    std::uint32_t heads, const std::uint32_t* start_pos, float theta, float eps,
    const qwen::vision::DeviceRope* rope, YarnRope yarn) {
  __shared__ float partial[8];
  __shared__ float norm[256];
  const unsigned t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const unsigned lane = tid % 32, wave = tid / 32;
  const bool query = h < heads;
  const unsigned head = query ? h : h - heads;
  const std::size_t width = std::size_t{heads} * 256;
  constexpr std::size_t kv_width = 512;
  const float* row = packed + std::size_t{t} * stride;
  const float value = row[(query ? head * 512 : 2 * width + head * 256) + tid];
  float ss = value * value;
  asm volatile("" : "+v"(ss));
  ss = WaveSum(ss);
  if (lane == 0)
    partial[wave] = ss;
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w)
    total += partial[w];
  const float scale = rsqrtf(total / 256.0F + eps);
  const float* gamma = query ? q_gamma : k_gamma;
  norm[tid] = value * scale * (gamma != nullptr ? gamma[tid] : 1.0F);
  if (query)
    gate[std::size_t{t} * width + head * 256 + tid] =
        row[head * 512 + 256 + tid];
  __syncthreads();
  if (tid < 32) {
    float frequency = powf(theta, -2.0F * static_cast<float>(tid) / 64.0F);
    if constexpr (kYarn)
      frequency = YarnFrequency(frequency, tid, yarn);
    float sine, cosine;
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, tid) * frequency,
            &sine, &cosine);
    if constexpr (kYarn)
      YarnScale(sine, cosine, yarn);
    const float a = norm[tid], b = norm[tid + 32];
    norm[tid] = __fmaf_rn(a, cosine, -__fmul_rn(b, sine));
    norm[tid + 32] = __fmaf_rn(a, sine, __fmul_rn(b, cosine));
  }
  __syncthreads();
  const float rotated = RotateKvChannel<256>(norm[tid], norm);
  if (query) {
    q[std::size_t{t} * width + head * 256 + tid] = rotated;
  } else {
    const std::size_t index =
        std::size_t{*start_pos + t} * kv_width + head * 256 + tid;
    // StoreQ8KvLane preserves the qualified divide/round/scale contract and
    // explicitly NaN-poisons nonfinite/half-reconstruction-overflow payloads.
    StoreQ8KvLane(key_storage, index, rotated);
    const float v = row[2 * width + kv_width + head * 256 + tid];
    const float rotated_v = RotateKvChannel<64>(v, norm);
    StoreQ8KvLane(value_storage, index, rotated_v);
  }
}

/// Independent, in-place row-local transform for the separate producer path
/// and inverse context boundary. One wave owns one H256 or one H64 block.
/// All values are loaded before stores; no scratch, allocation or host state.
template<unsigned kRotation>
__global__ void RotateKvRowsKernel(float* rows, std::uint64_t segments) {
  static_assert(kRotation == 64 || kRotation == 256);
  const std::uint64_t segment = std::uint64_t{blockIdx.x} * 4 + threadIdx.y;
  if (segment >= segments)
    return;  // uniform across this entire wave
  const unsigned lane = threadIdx.x;
  constexpr unsigned kValues = kRotation / 32;
  float value[kValues];
#pragma unroll
  for (unsigned j = 0; j < kValues; ++j)
    value[j] = OpaqueQ8KvF32(rows[segment * kRotation + j * 32 + lane] *
                             (kRotation == 256 ? 0.0625F : 0.125F));
#pragma unroll
  for (unsigned distance = 1; distance < 32; distance <<= 1) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const float other = XorLane(value[j], distance);
      value[j] = OpaqueQ8KvF32((lane & distance) == 0 ? value[j] + other
                                                      : other - value[j]);
    }
  }
#pragma unroll
  for (unsigned distance = 1; distance < kValues; distance <<= 1) {
#pragma unroll
    for (unsigned base = 0; base < kValues; base += 2 * distance) {
#pragma unroll
      for (unsigned j = 0; j < distance; ++j) {
        const float lo = value[base + j], hi = value[base + j + distance];
        value[base + j] = OpaqueQ8KvF32(lo + hi);
        value[base + j + distance] = OpaqueQ8KvF32(lo - hi);
      }
    }
  }
#pragma unroll
  for (unsigned j = 0; j < kValues; ++j)
    rows[segment * kRotation + j * 32 + lane] = value[j];
}
