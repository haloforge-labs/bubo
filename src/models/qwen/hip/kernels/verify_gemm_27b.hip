#include "src/models/qwen/hip/kernels/verify_gemm_27b.hpp"

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"

namespace gufo::hip {
namespace {

using core::GgmlType;

// One lane walks its sub-blocks in the same order and the lanes reduce in the
// same butterfly for every block shape, so these launches only change how
// output rows are spread over waves and how much activation staging a block
// shares. Measured on real UD-Q4_K_XL and DFlash2 Q4_K_M weights against the
// production dispatcher, alternating rounds, median of nine
// (kernel-bench/kb27_sweep, kb27_sweep2); only gains of 5% or more are kept.
// Lanes = 16 runs two sixteen-lane groups per wave; each keeps both original
// lane partials and combines them in the same butterfly, so it is exact too.
// MinWaves is only the occupancy hint of the launch bounds. The 16-lane and
// MinWaves shapes at 3-4 rows come from kernel-bench/kb27_sweep4 and
// kb27_confirm4 (eleven alternating rounds against this dispatch).
template<GgmlType T, std::size_t Batch, std::uint32_t Waves, std::size_t Rows,
         std::size_t Tiles, std::size_t TokensPerStep = Batch,
         std::uint32_t MinWaves = 4, std::uint32_t Lanes = 32>
bool Launch(const void* w, const float* x, float* y, std::size_t m,
            std::size_t k, hipStream_t stream) {
  constexpr std::size_t kRowsPerBlock = Waves * (32 / Lanes) * Rows;
  SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, T, Tiles, MinWaves,
                                      1, true, TokensPerStep, 32, Lanes>
      <<<static_cast<unsigned>((m + kRowsPerBlock - 1) / kRowsPerBlock),
         Waves * 32, 0, stream>>>(w, x, y, m, k);
  return true;
}

bool Is(std::size_t m, std::size_t k, std::size_t want_m, std::size_t want_k) {
  return m == want_m && k == want_k;
}

// Target UD-Q4_K_XL projections.
bool LaunchQ5K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ5_K;
  const bool ffn_up = Is(m, k, 17408, 5120);
  const bool ffn_down = Is(m, k, 5120, 17408);
  const bool ssm_qkv = Is(m, k, 10240, 5120);
  const bool ssm_gate = Is(m, k, 6144, 5120);
  const bool attn_q = Is(m, k, 12288, 5120);
  const bool out_6144 = Is(m, k, 5120, 6144);  // ssm_out, attn_output
  const bool any =
      ffn_up || ffn_down || ssm_qkv || ssm_gate || attn_q || out_6144;
  switch (batch) {
    case 3:
      if (ffn_down) return Launch<T, 3, 16, 4, 1, 1>(w, x, y, m, k, stream);
      if (ssm_qkv) return Launch<T, 3, 16, 4, 1>(w, x, y, m, k, stream);
      if (ssm_gate) return Launch<T, 3, 8, 4, 1, 1>(w, x, y, m, k, stream);
      if (attn_q) return Launch<T, 3, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (out_6144)
        return Launch<T, 3, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (ffn_up) return Launch<T, 4, 16, 4, 1, 1>(w, x, y, m, k, stream);
      if (ffn_down || ssm_qkv || ssm_gate || attn_q)
        return Launch<T, 4, 8, 4, 1, 4, 4, 16>(w, x, y, m, k, stream);
      if (out_6144)
        return Launch<T, 4, 4, 4, 1, 1, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 5:
      if (ssm_gate) return Launch<T, 5, 4, 4, 1>(w, x, y, m, k, stream);
      if (ffn_down || out_6144)
        return Launch<T, 5, 16, 4, 2>(w, x, y, m, k, stream);
      return false;
    // 6-8 rows (kb27_sweep5, 2026-09-27): 16-lane groups, 22-38% faster
    // than the 32-lane shapes on every projection.
    case 6:
      if (any) return Launch<T, 6, 4, 4, 1, 6, 8, 16>(w, x, y, m, k, stream);
      return false;
    case 7:
      if (any) return Launch<T, 7, 4, 4, 1, 7, 8, 16>(w, x, y, m, k, stream);
      return false;
    case 8:
      if (any) return Launch<T, 8, 4, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      return false;
    default:
      return false;
  }
}

// Target FFN gate.
bool LaunchIQ4XS(const void* w, const float* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kIQ4_XS;
  if (!Is(m, k, 17408, 5120))
    return false;
  if (batch == 2) return Launch<T, 2, 4, 3, 2>(w, x, y, m, k, stream);
  if (batch == 3) return Launch<T, 3, 4, 4, 2, 3, 1>(w, x, y, m, k, stream);
  if (batch == 4)
    return Launch<T, 4, 8, 4, 1, 4, 8, 16>(w, x, y, m, k, stream);
  if (batch == 5)
    return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
  if (batch == 6)
    return Launch<T, 6, 4, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
  if (batch == 7)
    return Launch<T, 7, 4, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
  if (batch == 8)
    return Launch<T, 8, 4, 4, 1, 8, 4, 16>(w, x, y, m, k, stream);
  return false;
}

// Output head (shared by verification and the draft) and attention V.
bool LaunchQ6K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ6_K;
  const bool head = Is(m, k, 248320, 5120);
  const bool attn_v = Is(m, k, 1024, 5120);
  switch (batch) {
    case 2:
      if (attn_v) return Launch<T, 2, 4, 3, 2>(w, x, y, m, k, stream);
      return false;
    case 3:
      if (attn_v) return Launch<T, 3, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (head) return Launch<T, 3, 16, 4, 1>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (attn_v) return Launch<T, 4, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (head) return Launch<T, 4, 16, 4, 2>(w, x, y, m, k, stream);
      return false;
    // The head at 5-8 rows (kb27_sweep6): 15-31% faster, 16-lane groups.
    case 5:
      if (attn_v) return Launch<T, 5, 8, 4, 2>(w, x, y, m, k, stream);
      if (head) return Launch<T, 5, 4, 4, 1, 5, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 6:
      if (head) return Launch<T, 6, 16, 4, 1, 6, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 7:
      if (head) return Launch<T, 7, 16, 4, 1, 7, 4, 16>(w, x, y, m, k, stream);
      return false;
    case 8:
      if (head) return Launch<T, 8, 16, 4, 1, 8, 8, 16>(w, x, y, m, k, stream);
      return false;
    default:
      return false;
  }
}

// DFlash2 Q4_K_M draft.
bool LaunchQ4K(const void* w, const float* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr auto T = GgmlType::kQ4_K;
  const bool ffn_down = Is(m, k, 5120, 17408);
  const bool attn_q = Is(m, k, 4096, 5120);
  const bool attn_output = Is(m, k, 5120, 4096);
  const bool conv = Is(m, k, 1280, 5120);
  const bool fc = Is(m, k, 5120, 25600);
  switch (batch) {
    case 2:
      if (attn_q) return Launch<T, 2, 4, 4, 2>(w, x, y, m, k, stream);
      if (conv) return Launch<T, 2, 8, 3, 1>(w, x, y, m, k, stream);
      return false;
    case 3:
      if (attn_q) return Launch<T, 3, 8, 4, 2>(w, x, y, m, k, stream);
      if (attn_output) return Launch<T, 3, 8, 4, 2>(w, x, y, m, k, stream);
      if (conv) return Launch<T, 3, 8, 4, 1>(w, x, y, m, k, stream);
      if (fc) return Launch<T, 3, 16, 4, 2>(w, x, y, m, k, stream);
      return false;
    case 4:
      if (ffn_down) return Launch<T, 4, 8, 4, 1>(w, x, y, m, k, stream);
      if (attn_q) return Launch<T, 4, 4, 4, 1>(w, x, y, m, k, stream);
      if (attn_output) return Launch<T, 4, 8, 4, 1>(w, x, y, m, k, stream);
      if (conv) return Launch<T, 4, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (fc) return Launch<T, 4, 16, 4, 1>(w, x, y, m, k, stream);
      return false;
    case 5:
      if (Is(m, k, 17408, 5120))  // FFN gate and up
        return Launch<T, 5, 8, 4, 1>(w, x, y, m, k, stream);
      if (attn_q) return Launch<T, 5, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (attn_output) return Launch<T, 5, 16, 4, 2>(w, x, y, m, k, stream);
      if (conv) return Launch<T, 5, 4, 4, 1, 1>(w, x, y, m, k, stream);
      if (fc) return Launch<T, 5, 16, 4, 2>(w, x, y, m, k, stream);
      return false;
    default:
      return false;
  }
}

}  // namespace

bool TryLaunchVerifyGemm27(GgmlType type, const void* w, const float* x,
                           float* y, std::size_t batch, std::size_t m,
                           std::size_t k, hipStream_t stream) {
  if (batch < 2 || batch > 8)
    return false;
  switch (type) {
    case GgmlType::kQ5_K:
      return LaunchQ5K(w, x, y, batch, m, k, stream);
    case GgmlType::kIQ4_XS:
      return LaunchIQ4XS(w, x, y, batch, m, k, stream);
    case GgmlType::kQ6_K:
      return LaunchQ6K(w, x, y, batch, m, k, stream);
    case GgmlType::kQ4_K:
      return LaunchQ4K(w, x, y, batch, m, k, stream);
    default:
      return false;
  }
}

}  // namespace gufo::hip
