// Dense products C[m x n] = A[m x k] * W[n x k]^T with BF16 operands and
// FP32 accumulation and output. A is row-major; W is blocked once at load as
// [n / 128][k / 64][128][64], so each stage of a workgroup reads one
// contiguous 16 KB block of weights.
//
// A workgroup covers 256 (or 128) rows and 128 output columns with eight
// waves in a 4 x 2 layout, consuming the reduction 64 deep per stage. The
// next stage is loaded into registers while the current one is multiplied
// from LDS. Wide outputs use 256-row tiles, which read each weight once per
// 256 rows; the 2816-wide projections use 128-row tiles, which give more
// workgroups to fill the GPU. Outputs with fewer tiles than the GPU has
// workgroup slots split the reduction; FP32 partials are summed in order.
//
// The prefetch registers are native vector types: HIP's uint4 kept them in
// scratch memory, which cost about 15%.
//
// LinearGeluMul applies the MLP activation in the epilogue: its weights are
// blocked with each 128-row tile holding 64 gate rows and the matching 64 up
// rows (BlockGateUpWeights), so a tile's outputs give 64 activations, and
// the FP32 gate/up product is never written.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/device.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"

namespace gufo::models::diffusion_gemma::kernels {
namespace {

constexpr int kCols = 128;          // Output columns per workgroup.
constexpr int kDepth = 64;          // Reduction depth per stage.
constexpr int kPitch = kDepth + 8;  // Padded LDS row: 144 bytes.
constexpr int kThreads = 256;       // Eight waves, 4 (rows) x 2 (columns).
// Workgroups that run at once (two per WGP on gfx1151); fewer tiles than
// this split the reduction.
constexpr int kSlots = 40;

using U32x4 = __attribute__((ext_vector_type(4))) unsigned int;

__device__ float GeluTanh(float x) {
  const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

// Grid: (n / 128 column tiles, row tiles, splits). Split s reduces
// [s * span, (s + 1) * span) into out + s * m * n. kGeluMul (one split):
// act[m][n / 2] = bf16(gelu(bf16(gate)) * bf16(up)) instead of out.
template<int kRows, bool kGeluMul>
__global__ void __launch_bounds__(kThreads)
    LinearKernel(const bf16_t* a, const bf16_t* w, float* out, bf16_t* act,
                 int m, int n, int k, int span) {
  constexpr int kTilesM = kRows / 4 / 16;  // Row fragments per wave.
  constexpr int kTilesN = kCols / 2 / 16;  // Column fragments per wave.
  constexpr int kChunks = kDepth / 8;      // 16-byte pieces per row.
  constexpr int kAPieces = kRows * kChunks / kThreads;
  constexpr int kWPieces = kCols * kChunks / kThreads;
  __shared__
      __attribute__((aligned(16))) bf16_t stage[(kRows + kCols) * kPitch];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int wave_m = wave / 2, wave_n = wave % 2;
  const int row0 = blockIdx.y * kRows;
  const std::size_t k_begin = static_cast<std::size_t>(blockIdx.z) * span;
  const bf16_t* w_src =
      w + static_cast<std::size_t>(blockIdx.x) * kCols * k + k_begin * kCols;
  // Each thread's A pieces. Rows past m repeat row m - 1, whose products
  // are never stored: the loads need no branch.
  const bf16_t* a_src[kAPieces];
#pragma unroll
  for (int j = 0; j < kAPieces; ++j) {
    const int index = j * kThreads + tid;
    const int r = min(index / kChunks, m - 1 - row0);
    a_src[j] = a + static_cast<std::size_t>(row0 + r) * k + k_begin +
               (index % kChunks) * 8;
  }

  Accumulator acc[kTilesM][kTilesN];
#pragma unroll
  for (int i = 0; i < kTilesM; ++i)
#pragma unroll
    for (int j = 0; j < kTilesN; ++j)
      acc[i][j] = Accumulator{};

  U32x4 a_next[kAPieces], w_next[kWPieces];
  const auto fetch = [&](int k0) {
#pragma unroll
    for (int j = 0; j < kAPieces; ++j)
      a_next[j] = *reinterpret_cast<const U32x4*>(a_src[j] + k0);
    const auto* w4 = reinterpret_cast<const U32x4*>(
        w_src + static_cast<std::size_t>(k0) * kCols);
#pragma unroll
    for (int j = 0; j < kWPieces; ++j)
      w_next[j] = w4[j * kThreads + tid];
  };
  fetch(0);
  for (int k0 = 0; k0 < span; k0 += kDepth) {
    LdsBarrier();  // The previous stage's fragments are consumed.
#pragma unroll
    for (int j = 0; j < kAPieces; ++j) {
      const int index = j * kThreads + tid;
      *reinterpret_cast<U32x4*>(stage + (index / kChunks) * kPitch +
                                (index % kChunks) * 8) = a_next[j];
    }
#pragma unroll
    for (int j = 0; j < kWPieces; ++j) {
      const int index = j * kThreads + tid;
      *reinterpret_cast<U32x4*>(stage + (kRows + index / kChunks) * kPitch +
                                (index % kChunks) * 8) = w_next[j];
    }
    LdsBarrier();
    // The last stage loads itself again: an unconditional prefetch keeps
    // the registers out of scratch.
    fetch(min(k0 + kDepth, span - kDepth));
    const bf16_t* at = stage;
    const bf16_t* wt = stage + kRows * kPitch;
#pragma unroll
    for (int s = 0; s < kDepth / 16; ++s) {
      Fragment b[kTilesN];
#pragma unroll
      for (int j = 0; j < kTilesN; ++j)
        b[j] = LoadFragment(
            wt + (wave_n * kTilesN * 16 + j * 16 + lane % 16) * kPitch +
            s * 16);
#pragma unroll
      for (int i = 0; i < kTilesM; ++i) {
        const Fragment f = LoadFragment(
            at + (wave_m * kTilesM * 16 + i * 16 + lane % 16) * kPitch +
            s * 16);
#pragma unroll
        for (int j = 0; j < kTilesN; ++j)
          acc[i][j] = Mma(f, b[j], acc[i][j]);
      }
      // Without this the compiler hoists every sub-step's fragment loads
      // and spills; one sub-step at a time is about 15% faster.
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  if constexpr (kGeluMul) {
    // 64 rows at a time (one wave row group) through LDS: thread t then
    // takes 16 activations of row t / 4 (gate columns 16 (t % 4) + j of the
    // tile, up columns 64 + 16 (t % 4) + j), written as 32 contiguous
    // bytes. The gate/up values are rounded to BF16 first: the reference's
    // gate/up projection is a checkpoint-dtype tensor.
    auto* tile = reinterpret_cast<float (*)[kCols + 1]>(stage);
    const int r_local = tid / 4, quarter = tid % 4;
#pragma unroll
    for (int g = 0; g < kRows / 64; ++g) {
      LdsBarrier();  // The stage, or the previous group, is consumed.
      // Waves holding this group's rows (kTilesM * 16 rows per wave row).
      if (wave_m * kTilesM / 4 == g) {
#pragma unroll
        for (int i = 0; i < kTilesM; ++i)
#pragma unroll
          for (int j = 0; j < kTilesN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              tile[(wave_m * kTilesM + i) % 4 * 16 + 2 * e + lane / 16]
                  [wave_n * kTilesN * 16 + j * 16 + lane % 16] = acc[i][j][e];
      }
      LdsBarrier();
      const int row = row0 + g * 64 + r_local;
      if (row < m) {
        U32x4 packed[2];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          const float gate = RoundBf16(tile[r_local][quarter * 16 + c]);
          const float up = RoundBf16(tile[r_local][64 + quarter * 16 + c]);
          const unsigned value = ToBf16(RoundBf16(GeluTanh(gate)) * up);
          const unsigned half = value << (16 * (c % 2));
          packed[c / 8][(c % 8) / 2] =
              c % 2 ? packed[c / 8][(c % 8) / 2] | half : half;
        }
        auto* dst = reinterpret_cast<U32x4*>(
            act + static_cast<std::size_t>(row) * (n / 2) + blockIdx.x * 64 +
            quarter * 16);
        dst[0] = packed[0];
        dst[1] = packed[1];
      }
    }
    return;
  }
  float* dst = out + static_cast<std::size_t>(blockIdx.z) * m * n;
  const int col0 = blockIdx.x * kCols + wave_n * kTilesN * 16 + lane % 16;
  const bool full = row0 + kRows <= m;
#pragma unroll
  for (int i = 0; i < kTilesM; ++i)
#pragma unroll
    for (int j = 0; j < kTilesN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = row0 + wave_m * kTilesM * 16 + i * 16 + 2 * e + lane / 16;
        if (full || r < m)
          dst[static_cast<std::size_t>(r) * n + col0 + j * 16] = acc[i][j][e];
      }
}

// [rows][k] row-major to [rows / 128][k / 64][128][64], 8 values per thread.
// gate_up: rows are gate then up halves, interleaved so that each 128-row
// tile holds 64 gate rows and the matching 64 up rows.
__global__ void BlockWeightsKernel(const uint4* in, uint4* out, int rows, int k,
                                   bool gate_up) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t pieces = static_cast<std::size_t>(rows) * k / 8;
  if (i >= pieces)
    return;
  std::size_t row = i / (k / 8);
  if (gate_up) {
    // Gate row j lands at tile row 128 (j / 64) + j % 64, up row j 64 later.
    const std::size_t half = rows / 2, j = row % half;
    row = (j / 64) * 128 + (row >= half ? 64 : 0) + j % 64;
  }
  const int col = static_cast<int>(i % (k / 8)) * 8;
  const std::size_t dst = ((row / kCols) * (k / kDepth) + col / kDepth) *
                              (static_cast<std::size_t>(kCols) * kDepth) +
                          (row % kCols) * kDepth + col % kDepth;
  out[dst / 8] = in[i];
}

struct Plan {
  int rows;    // Tile rows: 256 or 128.
  int splits;  // Reduction splits.
};

Plan LinearPlan(int m, int n, int k) {
  const int tiles_n = n / kCols;
  // 256-row tiles once the outputs are wide enough to fill the GPU.
  const int rows = tiles_n >= 32 ? 256 : 128;
  const int tiles = tiles_n * ((m + rows - 1) / rows);
  // Split the reduction (in whole stages) until the tiles fill the GPU.
  const int stages = k / kDepth;
  int splits = 1;
  while (tiles * splits < kSlots / 2 && splits * 2 <= stages &&
         stages % (splits * 2) == 0)
    splits *= 2;
  return {rows, splits};
}

}  // namespace

std::size_t LinearPartialFloats(int m, int n, int k) {
  const Plan plan = LinearPlan(m, n, k);
  return plan.splits > 1 ? static_cast<std::size_t>(plan.splits) * m * n : 0;
}

void Linear(const bf16_t* a, const bf16_t* w, float* partials, float* out,
            int m, int n, int k, hipStream_t stream) {
  if (m < 1 || n % kCols != 0 || k % kDepth != 0)
    throw std::invalid_argument("Linear: unsupported shape");
  const Plan plan = LinearPlan(m, n, k);
  const dim3 grid(n / kCols, (m + plan.rows - 1) / plan.rows, plan.splits);
  float* target = plan.splits > 1 ? partials : out;
  if (plan.rows == 256)
    LinearKernel<256, false><<<grid, kThreads, 0, stream>>>(
        a, w, target, nullptr, m, n, k, k / plan.splits);
  else
    LinearKernel<128, false><<<grid, kThreads, 0, stream>>>(
        a, w, target, nullptr, m, n, k, k / plan.splits);
  CheckLaunch("Linear");
  if (plan.splits > 1)
    SumPartials(partials, out, static_cast<std::size_t>(m) * n, plan.splits,
                stream);
}

void BlockLinearWeights(const bf16_t* w, bf16_t* out, int n, int k,
                        hipStream_t stream) {
  if (n % kCols != 0 || k % kDepth != 0)
    throw std::invalid_argument("BlockLinearWeights: unsupported shape");
  const std::size_t pieces = static_cast<std::size_t>(n) * k / 8;
  BlockWeightsKernel<<<(pieces + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(w), reinterpret_cast<uint4*>(out), n, k,
      false);
  CheckLaunch("BlockLinearWeights");
}

void BlockGateUpWeights(const bf16_t* w, bf16_t* out, int inter, int k,
                        hipStream_t stream) {
  if (inter % 64 != 0 || k % kDepth != 0)
    throw std::invalid_argument("BlockGateUpWeights: unsupported shape");
  const std::size_t pieces = static_cast<std::size_t>(2 * inter) * k / 8;
  BlockWeightsKernel<<<(pieces + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(w), reinterpret_cast<uint4*>(out),
      2 * inter, k, true);
  CheckLaunch("BlockGateUpWeights");
}

void LinearGeluMul(const bf16_t* a, const bf16_t* w, bf16_t* act, int m,
                   int inter, int k, hipStream_t stream) {
  const int n = 2 * inter;
  if (m < 1 || n % kCols != 0 || k % kDepth != 0)
    throw std::invalid_argument("LinearGeluMul: unsupported shape");
  // The activation needs whole columns, so the reduction is never split
  // (the model's 4224-wide product fills the GPU without one).
  const Plan plan{LinearPlan(m, n, k).rows, 1};
  const dim3 grid(n / kCols, (m + plan.rows - 1) / plan.rows, 1);
  if (plan.rows == 256)
    LinearKernel<256, true>
        <<<grid, kThreads, 0, stream>>>(a, w, nullptr, act, m, n, k, k);
  else
    LinearKernel<128, true>
        <<<grid, kThreads, 0, stream>>>(a, w, nullptr, act, m, n, k, k);
  CheckLaunch("LinearGeluMul");
}

}  // namespace gufo::models::diffusion_gemma::kernels
