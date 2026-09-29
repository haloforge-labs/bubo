#ifndef GUFO_MODELS_QWEN36_A3B_DECODE_KERNELS_HPP_
#define GUFO_MODELS_QWEN36_A3B_DECODE_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// A3B fused decode kernels for 1-8 token rows. They replace chains of small
/// launches (norm, add, quantize, gate, GEMV, epilogue) with one launch each:
/// at hidden 2048 a decode step is ~1000 dependent kernels, and each small
/// one costs ~3.5 us whatever it does.
///
/// Activations for Q8_0 weights use GGML's Q8_1 block layout (d = amax / 127,
/// q = round(x / d), ds = {d, sum}) at a stride of K / 32 blocks per row;
/// weights stay in their GGUF layout. Q8_0 dot products are the MMVQ
/// arithmetic: an exact integer sum per 8 codes, scaled by both block
/// scales. BF16 and F32 weights take the F32 activations.
namespace gufo::models::qwen36_a3b {

struct Q8_1Block {
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_1Block) == 36);

enum class WeightKind : std::uint32_t { kQ8 = 0, kBf16 = 1, kF32 = 2 };

/// res += add (when add is non-null), then xn = RMSNorm(res) * gamma and
/// xq = Q8_1(xn). One block per token row; h % 1024 == 0, h <= 4096. Output
/// rows are `xn_stride` floats / `xq_stride` blocks apart (0: packed).
void AddNormQuant(float* res, const float* add, const float* gamma, float* xn,
                  Q8_1Block* xq, std::uint32_t rows, std::uint32_t h, float eps,
                  hipStream_t stream, std::uint32_t xn_stride = 0,
                  std::uint32_t xq_stride = 0);

enum class GateMode : std::uint32_t {
  /// x * g. Flash-Next's GDN epilogue gates with sigmoid(z) and Qwen3.6 wants
  /// silu(z) = z * sigmoid(z): the caller passes the sigmoid-gated output as
  /// x and z as g.
  kMul = 0,
  /// x * sigmoid(g): the attention output gate.
  kSigmoid = 1,
};

/// y = x * gate(g) over rows of k (k % 32 == 0), g rows `g_stride` apart.
/// Writes xq = Q8_1(y) and, when non-null, y itself to xf.
void GateQuant(const float* x, const float* g, std::uint32_t g_stride,
               GateMode mode, Q8_1Block* xq, float* xf, std::uint32_t rows,
               std::uint32_t k, hipStream_t stream);

enum class GemvMode : std::uint32_t {
  kStore = 0,       // out = W x
  kAccumulate = 1,  // out += W x
  kGated = 2,       // out = silu(W x) * (W2 x)
};

struct GemvSeg {
  const void* w{nullptr};
  const void* w2{nullptr};  // kGated: the up projection
  float* out{nullptr};
  std::uint32_t rows{0};
  WeightKind type{WeightKind::kQ8};
  GemvMode mode{GemvMode::kStore};
  std::uint32_t out_stride{0};  // floats between token rows of `out`
};

/// Up to four projections of one input in one launch, one wave per output
/// row. `xq` is the Q8_1 input (Q8 segments), `xf` the F32 input (BF16/F32
/// segments); both have k columns.
constexpr int kMaxSegs = 4;
void MultiGemv(const GemvSeg* segs, int count, const Q8_1Block* xq,
               const float* xf, std::uint32_t rows, std::uint32_t k,
               hipStream_t stream);

/// Routed gate/up: act[(t*used + s)][m] = silu(G_e x_t) * (U_e x_t) with
/// e = ids[t*used + s], over [experts][m][k] Q8_0 (xq input) or BF16 (xf).
void ExpertsGated(const void* gate, const void* up, WeightKind type,
                  const Q8_1Block* xq, const float* xf,
                  const std::int32_t* ids, float* act, std::uint32_t rows,
                  std::uint32_t used, std::uint32_t m, std::uint32_t k,
                  hipStream_t stream);

/// Routed and shared down projections with the MoE epilogue and the residual
/// add fused:
///   res[t][h] += sum_s weights[t][s] * (D_e act[t][s])[h]
///              + sigmoid(router[t*router_stride + shared_gate]) *
///                (S sh_act[t])[h]
/// D is [experts][h][f], S is [h][f], each Q8_0 or BF16. A first launch
/// quantizes the activations for the Q8_0 paths into `act_q`
/// (rows * (used + 1) * f / 32 blocks).
void ExpertsDown(const void* down, WeightKind down_type, const void* sh_down,
                 WeightKind sh_type, const float* act, const float* sh_act,
                 Q8_1Block* act_q, const std::int32_t* ids,
                 const float* weights,
                 const float* router, std::uint32_t router_stride,
                 std::uint32_t shared_gate, float* res, std::uint32_t rows,
                 std::uint32_t used, std::uint32_t h, std::uint32_t f,
                 hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_DECODE_KERNELS_HPP_
