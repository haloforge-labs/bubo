// Self-conditioning soft embedding, soft[256 x H] = probs[256 x V] * E[V x H],
// with the reduction over the whole vocabulary. hipBLASLt's best algorithm
// for this row-major "NN" shape on gfx1151 reaches about 13-16 TFLOPS. Here
// one workgroup covers all 256 canvas rows, so the 1.5 GB embedding table is
// read from memory once, and the vocabulary is split across workgroups that
// write FP32 partials, summed in a fixed order afterwards.
//
// The probabilities arrive blocked, [V / 32][256][32] (SampleRows writes
// them so): each stage then reads one contiguous 16 KB block instead of 256
// rows 512 KB apart, whose page misses left the loads unhidden.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/device.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"

namespace gufo::models::diffusion_gemma::kernels {
namespace {

constexpr int kRows = 256;          // Canvas rows (all of them).
constexpr int kCols = 128;          // Hidden columns per workgroup.
constexpr int kDepth = 32;          // Vocabulary entries per stage.
constexpr int kPitch = kDepth + 8;  // Padded LDS row: 80 bytes.
constexpr int kStage = (kRows + kCols) * kPitch;  // bf16 elements.

using U32x4 = __attribute__((ext_vector_type(4))) unsigned int;

// Grid: (hidden / 128 column tiles, splits). Column tiles of one split are
// adjacent in launch order, so they stream the same probability slice
// through the cache at about the same time.
__global__ void __launch_bounds__(256)
    SoftEmbedKernel(const bf16_t* probs, const bf16_t* embed, float* partials,
                    int hidden, int span) {
  __shared__ __attribute__((aligned(16))) bf16_t stages[2 * kStage];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  // Wave layout: 4 row groups (64 rows) x 2 column groups (64 columns).
  const int wave_m = wave / 2, wave_n = wave % 2;
  const int col0 = blockIdx.x * kCols;
  const std::size_t k_begin = static_cast<std::size_t>(blockIdx.y) * span;

  // Probabilities: one contiguous block of 256 rows x 32 entries, 1024
  // pieces of 8, 4 per thread.
  const bf16_t* p_src[4];
  int p_dst[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int index = j * 256 + tid;
    const int r = index / 4, chunk = (index % 4) * 8;
    p_src[j] = probs + k_begin * kRows + static_cast<std::size_t>(index) * 8;
    p_dst[j] = r * kPitch + chunk;
  }
  // Embedding: 32 entries x 128 columns; each thread loads 8 columns of two
  // consecutive entries and stores them transposed (column-major in LDS, so
  // fragments read 16 consecutive entries per column). Adjacent lanes take
  // adjacent entry pairs: their 4-byte stores then fall in distinct banks.
  const int e_pair = tid % 16, e_cols = (tid / 16) * 8;
  const bf16_t* e_src = embed + (k_begin + 2 * e_pair) * hidden + col0 + e_cols;

  Accumulator acc[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n)
      acc[m][n] = Accumulator{};

  // Native vector types: HIP's uint4 kept these registers in scratch.
  U32x4 p_next[4], e_next[2];
  const auto fetch = [&](int k0) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      p_next[j] = *reinterpret_cast<const U32x4*>(
          p_src[j] + static_cast<std::size_t>(k0) * kRows);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      e_next[j] = *reinterpret_cast<const U32x4*>(
          e_src + static_cast<std::size_t>(k0 + j) * hidden);
  };
  fetch(0);
  int buffer = 0;
  for (int k0 = 0; k0 < span; k0 += kDepth) {
    bf16_t* stage = stages + buffer * kStage;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      *reinterpret_cast<U32x4*>(stage + p_dst[j]) = p_next[j];
    {
      // Column c of the two entries: halves of word c / 2.
      bf16_t* et = stage + kRows * kPitch;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int shift = 16 * (c % 2);
        const unsigned lo = (e_next[0][c / 2] >> shift) & 0xffffU;
        const unsigned hi = (e_next[1][c / 2] >> shift) & 0xffffU;
        *reinterpret_cast<unsigned*>(et + (e_cols + c) * kPitch + 2 * e_pair) =
            lo | (hi << 16);
      }
    }
    LdsBarrier();
    // The last stage loads itself again: an unconditional prefetch keeps
    // the registers out of scratch.
    fetch(min(k0 + kDepth, span - kDepth));
    const bf16_t* pt = stage;
    const bf16_t* et = stage + kRows * kPitch;
#pragma unroll
    for (int s = 0; s < kDepth / 16; ++s) {
      Fragment b[4];
#pragma unroll
      for (int n = 0; n < 4; ++n)
        b[n] = LoadFragment(et + (wave_n * 64 + n * 16 + lane % 16) * kPitch +
                            s * 16);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const Fragment a = LoadFragment(
            pt + (wave_m * 64 + m * 16 + lane % 16) * kPitch + s * 16);
#pragma unroll
        for (int n = 0; n < 4; ++n)
          acc[m][n] = Mma(a, b[n], acc[m][n]);
      }
      // One sub-step at a time: hoisted fragment loads spill.
      __builtin_amdgcn_sched_barrier(0);
    }
    // The other buffer is written next; its readers finished before the
    // barrier above.
    buffer ^= 1;
  }

  float* out = partials + static_cast<std::size_t>(blockIdx.y) * kRows * hidden;
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = wave_m * 64 + m * 16 + 2 * i + lane / 16;
        out[static_cast<std::size_t>(r) * hidden + col0 + wave_n * 64 + n * 16 +
            lane % 16] = acc[m][n][i];
      }
}

}  // namespace

void SoftEmbedProduct(const bf16_t* probs, const bf16_t* embed, float* partials,
                      float* out, int rows, int vocab, int hidden, int splits,
                      hipStream_t stream) {
  if (rows != kRows || hidden % kCols != 0 || splits < 1 ||
      vocab % (splits * kDepth) != 0)
    throw std::invalid_argument("SoftEmbedProduct: unsupported shape");
  const dim3 grid(hidden / kCols, splits);
  SoftEmbedKernel<<<grid, 256, 0, stream>>>(
      probs, embed, splits == 1 ? out : partials, hidden, vocab / splits);
  CheckLaunch("SoftEmbed");
  if (splits > 1)
    SumPartials(partials, out, static_cast<std::size_t>(rows) * hidden, splits,
                stream);
}

}  // namespace gufo::models::diffusion_gemma::kernels
