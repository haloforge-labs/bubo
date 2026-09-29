// A3B wide prefill: the prompt in chunks of Options::prefill_chunk tokens
// through Flash-Next's prefill launchers (read-only): F16 / int8 WMMA GEMMs
// for the Q8_0 projections, routed F16 WMMA GEMMs for the Q8_0 experts, the
// row-split Gated DeltaNet and the WMMA causal attention (A3B's 8 query
// heads per KV head padded to its 12). F32 and BF16 weights go through
// hipBLAS; BF16 experts (layers 34, 38 and 39 of the UD-Q8_K_XL file) run
// one GEMM per expert over the compacted rows.
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen36_a3b/attention.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen36_a3b/kernels.hpp"
#include "src/models/qwen36_a3b/prefill_kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

namespace fn = gufo::models::qwen38_flash_next::rocm;
using core::GgmlType;

// Flash-Next's WMMA attention geometry: 12 query heads per KV head.
constexpr std::uint32_t kWmmaGroup = 12;
// Dense Q8_0 shapes that take the F16 GEMM: Flash-Next measured K <= 2560 on
// its shapes; A3B's 2048 x 4096 outputs are faster on it too (-25%).
constexpr std::uint32_t kF16MinRows = 512;
constexpr std::uint32_t kF16MaxCols = 4096;
// Routed expert buckets pad to 16 rows (RoutedCompact).
constexpr std::uint32_t kBucket = 16;

void Fail(std::string* error, std::string message) {
  if (error != nullptr && error->empty())
    *error = std::move(message);
}

bool Ok(hipError_t err, const char* what, std::string* error) {
  if (err == hipSuccess)
    return true;
  Fail(error, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

bool Blas(hipblasStatus_t status, const char* what, std::string* error) {
  if (status == HIPBLAS_STATUS_SUCCESS)
    return true;
  Fail(error, std::string(what) + ": hipBLAS status " +
                  std::to_string(static_cast<int>(status)));
  return false;
}

fn::WeightType Small(GgmlType t) {
  return static_cast<fn::WeightType>(t);
}

WeightKind Kind(const Tensor& t) {
  switch (t.type) {
    case GgmlType::kQ8_0:
      return WeightKind::kQ8;
    case GgmlType::kBF16:
      return WeightKind::kBf16;
    default:
      return WeightKind::kF32;
  }
}

hipblasHandle_t Handle(void* p) {
  return static_cast<hipblasHandle_t>(p);
}

}  // namespace

void Engine::WideMark(int phase) {
  static const bool enabled = std::getenv("A3B_WIDE_PROFILE") != nullptr;
  if (!enabled)
    return;
  // Synced host timestamps: event times are not trustworthy on Windows HIP
  // (the stream runs in submission batches). Adds a sync per phase.
  Wide& s = wide_;
  (void)hipStreamSynchronize(stream_);
  s.stamps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count());
  s.marks.push_back(phase);
}

void Engine::WideReport(std::size_t tokens) {
  Wide& s = wide_;
  if (s.marks.empty())
    return;
  (void)hipStreamSynchronize(stream_);
  double ms[kWPhases]{};
  double total = 0;
  for (std::size_t i = 0; i + 1 < s.marks.size(); ++i) {
    const double t = s.stamps[i + 1] - s.stamps[i];
    ms[s.marks[i]] += t;
    total += t;
  }
  s.marks.clear();
  s.stamps.clear();
  static constexpr const char* kNames[kWPhases] = {
      "gdn proj",   "gdn core",    "gdn out",      "attn proj",
      "attn core",  "attn out",    "router",       "shared exp",
      "route host", "exp gate/up", "exp down",     "moe epilogue",
      "norm/head",  "mtp block",   "bf16 experts", "topk+counts"};
  std::fprintf(stderr,
               "a3b wide prefill profile, %zu tokens, %.1f ms "
               "(%.0f t/s):\n",
               tokens, total, tokens * 1000.0 / total);
  for (int p = 0; p < kWPhases; ++p)
    std::fprintf(stderr, "  %-13s %8.1f ms %5.1f%%\n", kNames[p], ms[p],
                 100.0 * ms[p] / total);
}

bool Engine::AllocateWide(std::string* error) {
  if (wide_.rows != 0)
    return true;
  const std::size_t N = options_.prefill_chunk;
  const std::size_t H = c_.hidden;
  const std::size_t E = c_.experts;
  const std::size_t U = c_.experts_used;
  const std::size_t ff = c_.expert_ff;
  const std::size_t sf = c_.shared_ff;
  const std::size_t ch = c_.ConvChannels();
  const std::size_t vd = c_.SsmValueDim();
  const std::size_t qd = c_.QDim();
  const std::size_t kv = c_.KvDim();
  const std::size_t padded =
      static_cast<std::size_t>(c_.kv_heads) * kWmmaGroup * c_.head_dim;
  const std::size_t max_k = std::max({H, vd, qd, ff, sf, 2 * H});
  const std::size_t slots = N * U;
  const std::size_t compact = fn::RoutedCompactRows(slots, E);
  // The 48-row map plus the 64-row down map (RouteHints-style).
  const std::size_t tiles = (slots + E * (kBucket - 1)) / kBucket + 2 * E + 2;
  bool ok = true;
  const auto alloc = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (!ok || !Ok(hipMalloc(&p, std::max<std::size_t>(bytes, 256)),
                   "prefill alloc", error)) {
      ok = false;
      return nullptr;
    }
    owned_.push_back(p);
    return p;
  };
  const auto f32 = [&](std::size_t n) {
    return static_cast<float*>(alloc(n * sizeof(float)));
  };
  const auto i32 = [&](std::size_t n) {
    return static_cast<std::int32_t*>(alloc(n * sizeof(std::int32_t)));
  };
  Wide& w = wide_;
  w.res = f32(N * H);
  w.xn = f32(N * H);
  w.blk = f32(N * H);
  w.h = f32(N * H);
  w.qkv = f32(N * std::max(ch, 2 * qd));
  w.z = f32(N * vd);
  w.ab = f32(N * 2 * c_.ssm_v_heads);
  w.conv = f32((N + c_.ssm_conv_kernel) * ch);
  w.qn = f32(N * c_.ssm_k_heads * c_.ssm_head_dim);
  w.kn = f32(N * std::max<std::size_t>(c_.ssm_k_heads * c_.ssm_head_dim,
                                       2 * c_.ssm_v_heads));
  w.raw = f32(N * vd);
  w.gdn = f32(N * std::max(vd, qd));
  w.q = f32(N * qd);
  w.gate = f32(N * qd);
  w.k = f32(N * kv);
  w.v = f32(N * kv);
  w.q24 = f32(N * padded);
  w.g24 = f32(N * padded);
  w.o24 = f32(N * padded);
  w.router = f32(N * (E + 1));
  w.weights = f32(N * U);
  w.gate_e = f32(slots * ff);
  w.shg = f32(N * sf);
  w.shu = f32(N * sf);
  w.sh_out = f32(N * H);
  w.gc = f32(compact * ff);
  w.uc = f32(compact * ff);
  w.dc = f32(compact * H);
  w.emb = f32(N * H);
  w.cat = f32(N * 2 * H);
  w.mh = f32(N * H);
  w.half = static_cast<__half*>(alloc(N * max_k * sizeof(__half)));
  w.up_half = static_cast<__half*>(alloc(slots * ff * sizeof(__half)));
  w.down_half = static_cast<__half*>(alloc(slots * H * sizeof(__half)));
  w.bf = alloc(N * max_k * 2);
  w.bf_lo = alloc(N * max_k * 2);
  w.q8t = alloc(fn::Q8TiledBytes(N, max_k));
  w.xc = alloc(compact * std::max(H, ff) * 2);
  w.xc_lo = alloc(compact * std::max(H, ff) * 2);
  w.ac = alloc(compact * ff * 2);
  w.ac_lo = alloc(compact * ff * 2);
  w.ids = i32(N * U);
  w.bounds = i32(E + 1);
  w.cursors = i32(E);
  w.rows_token = i32(compact);
  w.rows_slot = i32(compact);
  w.tiles = i32(tiles);
  const std::size_t bf_tiles = slots / kGroupedTileRows + E + 1;
  w.tiles_bf = i32(bf_tiles);
  w.tokens = i32(N);
  w.counts = static_cast<std::uint32_t*>(alloc(E * sizeof(std::uint32_t)));
  if (!ok)
    return false;
  if (!Ok(hipHostMalloc(reinterpret_cast<void**>(&w.counts_host),
                        E * sizeof(std::uint32_t)),
          "pinned counts", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&w.tiles_host),
                        tiles * sizeof(std::int32_t)),
          "pinned tiles", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&w.tiles_bf_host),
                        bf_tiles * sizeof(std::int32_t)),
          "pinned BF16 tiles", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&w.tokens_host),
                        N * sizeof(std::int32_t)),
          "pinned prefill tokens", error) ||
      !Ok(hipEventCreateWithFlags(&w.counts_ready, hipEventDisableTiming),
          "counts event", error))
    return false;
  hipblasHandle_t h = nullptr;
  if (!Blas(hipblasCreate(&h), "hipblasCreate", error))
    return false;
  w.blas = h;
  if (!Blas(hipblasSetStream(h, stream_), "hipblasSetStream", error))
    return false;
  w.rows = static_cast<std::uint32_t>(N);
  return true;
}

void Engine::FreeWide() {
  Wide& w = wide_;
  if (w.blas != nullptr)
    (void)hipblasDestroy(Handle(w.blas));
  if (w.counts_ready != nullptr)
    (void)hipEventDestroy(w.counts_ready);
  for (hipEvent_t e : w.events)
    (void)hipEventDestroy(e);
  if (w.counts_host != nullptr)
    (void)hipHostFree(w.counts_host);
  if (w.tiles_host != nullptr)
    (void)hipHostFree(w.tiles_host);
  if (w.tiles_bf_host != nullptr)
    (void)hipHostFree(w.tiles_bf_host);
  if (w.tokens_host != nullptr)
    (void)hipHostFree(w.tokens_host);
  w = Wide{};
}

bool Engine::BlasBf16(const void* w, const void* x, const void* x_lo,
                      float* out, std::uint32_t m, std::uint32_t n,
                      std::uint32_t k, std::string* error) {
  // The decode kernels feed BF16 weights F32 activations. Here each
  // activation is split into BF16 hi + lo parts (~16 mantissa bits) and the
  // second GEMM accumulates, so the prefill keeps that precision.
  const float one = 1.0F;
  const float zero = 0.0F;
  for (int part = 0; part < (x_lo != nullptr ? 2 : 1); ++part) {
    if (!Blas(hipblasGemmEx(
                  Handle(wide_.blas), HIPBLAS_OP_T, HIPBLAS_OP_N,
                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                  &one, w, HIP_R_16BF, static_cast<int>(k),
                  part == 0 ? x : x_lo, HIP_R_16BF, static_cast<int>(k),
                  part == 0 ? &zero : &one, out, HIP_R_32F, static_cast<int>(m),
                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
              "BF16 GEMM", error))
      return false;
  }
  return true;
}

bool Engine::WideDense(const Tensor& w, const float* x, float* out,
                       std::uint32_t n, std::string* error) {
  const std::uint32_t m = w.rows;
  const std::uint32_t k = w.cols;
  Wide& s = wide_;
  if (s.staged != x) {
    s.staged = x;
    s.staged_kinds = 0;
  }
  const std::size_t count = static_cast<std::size_t>(n) * k;
  if (w.type == GgmlType::kQ8_0) {
    // Diagnostic: A3B_WIDE_NO_F16 keeps every Q8_0 projection on the int8
    // GEMM (the decode kernels' activation quantization).
    static const bool no_f16 = std::getenv("A3B_WIDE_NO_F16") != nullptr;
    // Diagnostic: A3B_WIDE_F16_MAXCOLS overrides the F16 route's K bound.
    static const std::uint32_t max_cols = [] {
      const char* v = std::getenv("A3B_WIDE_F16_MAXCOLS");
      return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v))
                          : kF16MaxCols;
    }();
    static const std::uint32_t min_rows = [] {
      const char* v = std::getenv("A3B_WIDE_F16_MINROWS");
      return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v))
                          : kF16MinRows;
    }();
    if (!no_f16 && m >= min_rows && k <= max_cols) {
      if (!(s.staged_kinds & 1u)) {
        fn::NarrowActivations(x, s.half, false, count, stream_);
        s.staged_kinds |= 1u;
      }
      if (fn::DenseF16Gemm(w.data, s.half, out, n, m, k, stream_))
        return true;
    }
    if (!(s.staged_kinds & 2u)) {
      fn::QuantizeQ8Tiled(x, s.q8t, n, k, stream_);
      s.staged_kinds |= 2u;
    }
    if (!fn::W8A8Gemm(w.data, s.q8t, out, n, m, k, stream_)) {
      Fail(error, "W8A8 GEMM rejected a prefill shape");
      return false;
    }
    return true;
  }
  if (w.type == GgmlType::kF32) {
    // The F32 projections here are narrow (router, alpha/beta): a plain
    // tiled kernel beats hipBLAS's SGEMM on these shapes by ~10x.
    // A3B_WIDE_BLAS_F32 keeps hipBLAS (reference).
    static const bool blas_f32 = std::getenv("A3B_WIDE_BLAS_F32") != nullptr;
    if (!blas_f32) {
      F32GemmNT(w.f32(), x, out, m, n, k, stream_);
      return true;
    }
    const float one = 1.0F;
    const float zero = 0.0F;
    return Blas(
        hipblasSgemm(Handle(s.blas), HIPBLAS_OP_T, HIPBLAS_OP_N,
                     static_cast<int>(m), static_cast<int>(n),
                     static_cast<int>(k), &one, w.f32(), static_cast<int>(k), x,
                     static_cast<int>(k), &zero, out, static_cast<int>(m)),
        "F32 GEMM", error);
  }
  if (!(s.staged_kinds & 4u)) {
    SplitBf16(x, static_cast<__hip_bfloat16*>(s.bf),
              static_cast<__hip_bfloat16*>(s.bf_lo), count, stream_);
    s.staged_kinds |= 4u;
  }
  return BlasBf16(w.data, s.bf, s.bf_lo, out, m, n, k, error);
}

bool Engine::WideGdn(const Layer& l, std::uint32_t idx, std::uint32_t n,
                     std::string* error) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t ch = c_.ConvChannels();
  const std::uint32_t vd = c_.SsmValueDim();
  Wide& s = wide_;
  Tensor ab;
  ab.data = l.alpha_beta;
  ab.type = GgmlType::kF32;
  ab.rows = 2 * c_.ssm_v_heads;
  ab.cols = H;
  WideMark(kWGdnProj);
  if (!WideDense(l.ssm_qkv, s.xn, s.qkv, n, error) ||
      !WideDense(l.ssm_z, s.xn, s.z, n, error) ||
      !WideDense(ab, s.xn, s.ab, n, error))
    return false;
  WideMark(kWGdnCore);
  fn::GatedDeltaNet(s.qkv, ch, s.z, vd, s.ab, l.ssm_conv1d.f32(), l.ssm_a.f32(),
                    l.ssm_dt.f32(), l.ssm_norm.f32(), conv_state_[idx], s.conv,
                    s.qn, s.kn, s.raw, ssm_state_[idx], s.gdn, nullptr,
                    fn::RollbackRows{}, fn::RollbackRows{}, n, c_.ssm_k_heads,
                    c_.ssm_v_heads, c_.ssm_head_dim, c_.ssm_conv_kernel,
                    /*row_split=*/true, /*convolved=*/false, c_.eps, stream_);
  // Flash-Next gates with sigmoid(z); Qwen3.6 wants silu(z) = z * sigmoid(z).
  if (!options_.fn_gate)
    MulInPlace(s.gdn, s.z, static_cast<std::size_t>(n) * vd, stream_);
  WideMark(kWGdnOut);
  if (!WideDense(l.ssm_out, s.gdn, s.blk, n, error))
    return false;
  AddInPlace(s.res, s.blk, static_cast<std::size_t>(n) * H, stream_);
  return true;
}

bool Engine::WideAttention(const Layer& l, __half* k_cache, __half* v_cache,
                           __half* vt_cache, float* res, std::uint32_t n,
                           std::uint32_t start_pos, std::string* error) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t qd = c_.QDim();
  const std::uint32_t kv = c_.KvDim();
  const std::uint32_t d = c_.head_dim;
  const std::uint32_t group = c_.heads / c_.kv_heads;
  Wide& s = wide_;
  float* qg = s.qkv;  // [q | gate] interleaved per head
  WideMark(kWAttnProj);
  if (!WideDense(l.attn_q, s.xn, qg, n, error) ||
      !WideDense(l.attn_k, s.xn, s.k, n, error) ||
      !WideDense(l.attn_v, s.xn, s.v, n, error))
    return false;
  WideMark(kWAttnCore);
  fn::UnpackQGate(qg, 2 * qd, s.q, s.gate, nullptr, nullptr, n, c_.heads, d, 0,
                  stream_);
  fn::RmsNormRows(s.q, l.attn_q_norm.f32(), s.q, n * c_.heads, d, 1, c_.eps,
                  stream_);
  fn::RmsNormRows(s.k, l.attn_k_norm.f32(), s.k, n * c_.kv_heads, d, 1, c_.eps,
                  stream_);
  fn::Rope(s.q, n, c_.heads, d, c_.rotary_dim, pos_dev_, c_.rope_theta,
           stream_);
  fn::Rope(s.k, n, c_.kv_heads, d, c_.rotary_dim, pos_dev_, c_.rope_theta,
           stream_);
  fn::StoreKv(s.k, k_cache, n, kv, pos_dev_, stream_);
  fn::StoreKv(s.v, v_cache, n, kv, pos_dev_, stream_);
  StoreVt(v_cache, vt_cache, pos_dev_, n, c_.kv_heads, options_.max_context,
          stream_);
  // A3B's prefill attention (attention.hip). A3B_WIDE_FN_ATTN runs
  // Flash-Next's WMMA kernel on heads padded to its 12 per KV head instead;
  // A3B_WIDE_ATTN_CHECK runs both and prints the largest difference.
  static const bool fn_attn = std::getenv("A3B_WIDE_FN_ATTN") != nullptr;
  static const bool check = std::getenv("A3B_WIDE_ATTN_CHECK") != nullptr;
  if (!fn_attn) {
    if (!PrefillAttention(s.q, s.gate, k_cache, v_cache, s.gdn, start_pos, n,
                          c_.heads, c_.kv_heads, d, stream_)) {
      Fail(error, "prefill attention rejected the geometry");
      return false;
    }
    if (!check) {
      WideMark(kWAttnOut);
      if (!WideDense(l.attn_out, s.gdn, s.blk, n, error))
        return false;
      AddInPlace(res, s.blk, static_cast<std::size_t>(n) * H, stream_);
      return true;
    }
  }
  // Flash-Next's WMMA attention is built for 12 query heads per KV head:
  // A3B's 8 sit in the first 8 slots of each group, the rest are zero rows
  // whose outputs are dropped.
  PadHeads(s.q, s.q24, n, c_.kv_heads, group, kWmmaGroup, d, stream_);
  PadHeads(s.gate, s.g24, n, c_.kv_heads, group, kWmmaGroup, d, stream_);
  if (!fn::WmmaCausalAttention(s.q24, s.g24, k_cache, v_cache, nullptr, 0,
                               s.o24, n, start_pos, c_.kv_heads * kWmmaGroup,
                               c_.kv_heads, d, 4, stream_)) {
    Fail(error, "WMMA causal attention rejected the padded geometry");
    return false;
  }
  if (check && !fn_attn) {
    // Both results to the host: A3B's in s.gdn, Flash-Next's unpadded into
    // s.q24 (free again after the padded launch).
    UnpadHeads(s.o24, s.q24, n, c_.kv_heads, group, kWmmaGroup, d, stream_);
    const std::size_t count = static_cast<std::size_t>(n) * qd;
    std::vector<float> mine(count), ref(count);
    (void)hipStreamSynchronize(stream_);
    (void)hipMemcpy(mine.data(), s.gdn, count * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ref.data(), s.q24, count * 4, hipMemcpyDeviceToHost);
    double worst = 0, scale = 0;
    for (std::size_t i = 0; i < count; ++i) {
      scale = std::max(scale, static_cast<double>(std::fabs(ref[i])));
      worst = std::max(worst, static_cast<double>(std::fabs(mine[i] - ref[i])));
    }
    std::fprintf(stderr,
                 "prefill attn check pos %u n %u: max |diff| %.3g, "
                 "max |ref| %.3g\n",
                 start_pos, n, worst, scale);
  } else {
    UnpadHeads(s.o24, s.gdn, n, c_.kv_heads, group, kWmmaGroup, d, stream_);
  }
  WideMark(kWAttnOut);
  if (!WideDense(l.attn_out, s.gdn, s.blk, n, error))
    return false;
  AddInPlace(res, s.blk, static_cast<std::size_t>(n) * H, stream_);
  return true;
}

bool Engine::WideMoe(const Layer& l, float* res, std::uint32_t n,
                     std::string* error) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t E = c_.experts;
  const std::uint32_t U = c_.experts_used;
  const std::uint32_t ff = l.gate_exps.rows;
  const std::uint32_t sf = l.sh_up.rows;
  const std::uint32_t slots = n * U;
  Wide& s = wide_;
  Tensor router;
  router.data = l.router;
  router.type =
      l.router_type == WeightKind::kBf16 ? GgmlType::kBF16 : GgmlType::kF32;
  router.rows = E + 1;
  router.cols = H;
  WideMark(kWRouter);
  if (!WideDense(router, s.xn, s.router, n, error))
    return false;
  WideMark(kWTopK);
  fn::RouterTopK(s.router, E + 1, s.ids, s.weights, n, E, U, stream_);
  fn::ExpertCounts(s.ids, s.counts, n, E, U, stream_);
  if (!Ok(hipMemcpyAsync(s.counts_host, s.counts, E * sizeof(std::uint32_t),
                         hipMemcpyDeviceToHost, stream_),
          "expert counts download", error) ||
      !Ok(hipEventRecord(s.counts_ready, stream_), "counts event", error))
    return false;
  WideMark(kWShared);
  // The shared expert does not depend on routing: queue it while the host
  // waits for the counts.
  if (!WideDense(l.sh_gate, s.xn, s.shg, n, error) ||
      !WideDense(l.sh_up, s.xn, s.shu, n, error))
    return false;
  fn::Swiglu(s.shg, s.shu, static_cast<std::size_t>(n) * sf, stream_);
  if (!WideDense(l.sh_down, s.shg, s.sh_out, n, error))
    return false;

  WideMark(kWRouteHost);
  if (!Ok(hipEventSynchronize(s.counts_ready), "expert counts", error))
    return false;
  // One launched tile per (expert, 16- or 48-row slice of its padded
  // bucket); host bounds mirror RoutedCompact's padding.
  const std::uint32_t tile_rows = slots >= 16u * E ? 48u : 16u;
  std::uint32_t n_tiles = 0;
  std::vector<std::uint32_t> bound(E + 1, 0);
  for (std::uint32_t e = 0; e < E; ++e) {
    const std::uint32_t padded =
        (s.counts_host[e] + kBucket - 1) / kBucket * kBucket;
    bound[e + 1] = bound[e] + padded;
    for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j)
      s.tiles_host[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
  }
  const std::uint32_t compact = bound[E];
  // Flash-Next's wide down map: 64-row tiles amortize the weight decode when
  // they launch at most 3/4 of the 48-row map's tiles (appended after it).
  std::uint32_t down_tiles = n_tiles;
  std::uint32_t down_offset = 0;
  std::uint32_t down_rows = tile_rows;
  static const bool no_wide_down = std::getenv("A3B_WIDE_NO_DOWN64") != nullptr;
  if (tile_rows == 48 && !no_wide_down) {
    std::uint32_t n64 = 0;
    for (std::uint32_t e = 0; e < E; ++e)
      for (std::uint32_t j = 0; j * 64 < bound[e + 1] - bound[e]; ++j)
        s.tiles_host[n_tiles + n64++] =
            static_cast<std::int32_t>(e | (j << 16));
    if (n64 * 4 <= n_tiles * 3) {
      down_tiles = n64;
      down_offset = n_tiles;
      down_rows = 64;
    }
  }
  const std::uint32_t uploaded = down_offset + down_tiles;
  if (uploaded > 0 &&
      !Ok(hipMemcpyAsync(s.tiles, s.tiles_host, uploaded * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "routed tile upload", error))
    return false;
  fn::RoutedCompact(s.ids, s.counts, s.bounds, s.cursors, s.rows_token,
                    s.rows_slot, n, U, E, stream_);

  WideMark(kWGateUp);
  const bool q8_gate_up =
      l.gate_exps.type == GgmlType::kQ8_0 && l.up_exps.type == GgmlType::kQ8_0;
  const bool q8_down = l.down_exps.type == GgmlType::kQ8_0;
  auto* xc = static_cast<__hip_bfloat16*>(s.xc);
  auto* ac = static_cast<__hip_bfloat16*>(s.ac);
  auto* xc_lo = static_cast<__hip_bfloat16*>(s.xc_lo);
  auto* ac_lo = static_cast<__hip_bfloat16*>(s.ac_lo);
  // BF16 experts over the compacted rows [bound[e], bound[e] + count): the
  // grouped WMMA GEMM (tiles of kGroupedTileRows rows, built once per layer),
  // or with A3B_WIDE_BF16_BLAS one hipBLAS GEMM per expert (reference).
  static const bool blas_experts = std::getenv("A3B_WIDE_BF16_BLAS") != nullptr;
  std::uint32_t bf_tiles = 0;
  if ((!q8_gate_up || !q8_down) && !blas_experts) {
    for (std::uint32_t e = 0; e < E; ++e)
      for (std::uint32_t j = 0; j * kGroupedTileRows < s.counts_host[e]; ++j)
        s.tiles_bf_host[bf_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    if (bf_tiles > 0 && !Ok(hipMemcpyAsync(s.tiles_bf, s.tiles_bf_host,
                                           bf_tiles * sizeof(std::int32_t),
                                           hipMemcpyHostToDevice, stream_),
                            "BF16 tile upload", error))
      return false;
  }
  const auto bf16_experts = [&](const Tensor& w, const void* x,
                                const void* x_lo, std::uint32_t k, float* out,
                                std::uint32_t m) {
    if (!blas_experts) {
      GroupedBf16Gemm(w.data, x, x_lo, s.tiles_bf, bf_tiles, s.bounds, out, m,
                      k, stream_);
      return true;
    }
    const std::size_t per = static_cast<std::size_t>(m) * k * 2;
    for (std::uint32_t e = 0; e < E; ++e) {
      const std::uint32_t count = s.counts_host[e];
      if (count == 0)
        continue;
      if (!BlasBf16(static_cast<const std::uint8_t*>(w.data) + e * per,
                    static_cast<const std::uint8_t*>(x) +
                        static_cast<std::size_t>(bound[e]) * k * 2,
                    static_cast<const std::uint8_t*>(x_lo) +
                        static_cast<std::size_t>(bound[e]) * k * 2,
                    out + static_cast<std::size_t>(bound[e]) * m, m, count, k,
                    error))
        return false;
    }
    return true;
  };

  if (q8_gate_up) {
    if (!(s.staged == s.xn && (s.staged_kinds & 1u))) {
      if (s.staged != s.xn)
        s.staged_kinds = 0;
      s.staged = s.xn;
      fn::NarrowActivations(s.xn, s.half, false,
                            static_cast<std::size_t>(n) * H, stream_);
      s.staged_kinds |= 1u;
    }
    if (n_tiles > 0 &&
        (!fn::RoutedF16Gemm(l.gate_exps.data, fn::WeightType::kQ8_0, s.half,
                            s.tiles, n_tiles, tile_rows, s.bounds, s.rows_token,
                            s.rows_slot, nullptr, s.gate_e, nullptr, ff, H,
                            stream_) ||
         !fn::RoutedF16Gemm(l.up_exps.data, fn::WeightType::kQ8_0, s.half,
                            s.tiles, n_tiles, tile_rows, s.bounds, s.rows_token,
                            s.rows_slot, s.gate_e, nullptr, s.up_half, ff, H,
                            stream_))) {
      Fail(error, "routed F16 gate/up GEMM failed");
      return false;
    }
  } else {
    // BF16 gate/up: gather the token rows in bucket order, one GEMM per
    // expert, SwiGLU straight into the down projection's BF16 rows.
    WideMark(kWBf16Exp);
    GatherRowsBf16(s.xn, false, s.rows_token, compact, H, xc, xc_lo, stream_);
    if (!bf16_experts(l.gate_exps, xc, xc_lo, H, s.gc, ff) ||
        !bf16_experts(l.up_exps, xc, xc_lo, H, s.uc, ff))
      return false;
    SwigluBf16(s.gc, s.uc, ac, ac_lo, static_cast<std::size_t>(compact) * ff,
               stream_);
  }
  WideMark(kWDown);
  if (q8_down && q8_gate_up) {
    if (down_tiles > 0 &&
        !fn::RoutedF16Gemm(l.down_exps.data, fn::WeightType::kQ8_0, s.up_half,
                           s.tiles + down_offset, down_tiles, down_rows,
                           s.bounds, s.rows_slot, s.rows_slot, nullptr, nullptr,
                           s.down_half, H, ff, stream_)) {
      Fail(error, "routed F16 down GEMM failed");
      return false;
    }
  } else if (!q8_down) {
    // BF16 down: the SwiGLU rows in bucket order (gathered from the routed
    // F16 output when gate/up were Q8_0), then scattered back per slot.
    WideMark(kWBf16Exp);
    if (q8_gate_up)
      GatherRowsBf16(s.up_half, true, s.rows_slot, compact, ff, ac, ac_lo,
                     stream_);
    if (!bf16_experts(l.down_exps, ac, ac_lo, ff, s.dc, H))
      return false;
    ScatterRowsHalf(s.dc, s.rows_slot, compact, H, s.down_half, stream_);
  } else {
    Fail(error, "BF16 gate/up with Q8_0 down experts is not handled");
    return false;
  }
  WideMark(kWEpilogue);
  fn::MoeEpilogueVec4F16(s.down_half, s.weights, s.sh_out, s.router + E, E + 1,
                         s.blk, n, U, H, stream_);
  AddInPlace(res, s.blk, static_cast<std::size_t>(n) * H, stream_);
  return true;
}

bool Engine::WideTrunk(std::uint32_t n, std::string* error) {
  const std::uint32_t H = c_.hidden;
  Wide& s = wide_;
  // Diagnostic (A3B_WIDE_DEBUG bits): run a part with the decode kernels in
  // 8-row groups instead (1 MoE, 2 attention, 4 GDN) to bisect numerics.
  static const unsigned debug = [] {
    const char* v = std::getenv("A3B_WIDE_DEBUG");
    return v != nullptr ? static_cast<unsigned>(std::atoi(v)) : 0u;
  }();
  if (debug != 0) {
    // Group start positions for the decode kernels' device position reads.
    const std::uint32_t groups = (n + kMaxRows - 1) / kMaxRows;
    if (s.pos_list == nullptr) {
      void* p = nullptr;
      if (!Ok(hipMalloc(&p, (s.rows / kMaxRows + 1) * 4), "pos list", error))
        return false;
      owned_.push_back(p);
      s.pos_list = static_cast<std::uint32_t*>(p);
    }
    std::vector<std::uint32_t> list(groups);
    for (std::uint32_t g = 0; g < groups; ++g)
      list[g] = pos_ + g * kMaxRows;
    if (!Ok(hipMemcpy(s.pos_list, list.data(), groups * 4,
                      hipMemcpyHostToDevice),
            "pos list upload", error))
      return false;
  }
  const auto grouped = [&](const Tensor& gamma, auto&& part) {
    for (std::uint32_t t0 = 0; t0 < n; t0 += kMaxRows) {
      const std::uint32_t r = std::min<std::uint32_t>(kMaxRows, n - t0);
      float* rows = s.res + static_cast<std::size_t>(t0) * H;
      AddNormQuant(rows, nullptr, gamma.f32(), xn_, aq_, r, H, c_.eps, stream_);
      part(rows, r, s.pos_list + t0 / kMaxRows);
    }
  };
  fn::EmbedTokens(token_embd_.data, Small(token_embd_.type), s.tokens, s.res, n,
                  H, 1, stream_);
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    const Layer& l = layers_[i];
    if (l.linear && (debug & 4u)) {
      grouped(l.attn_norm, [&](float* rows, std::uint32_t r,
                               const std::uint32_t*) {
        // FusedGdn accumulates into res_.
        fn::CopyDevice(rows, res_, static_cast<std::size_t>(r) * H, stream_);
        FusedGdn(l, i, r, false);
        fn::CopyDevice(res_, rows, static_cast<std::size_t>(r) * H, stream_);
      });
    } else if (!l.linear && (debug & 2u)) {
      grouped(l.attn_norm, [&](float* rows, std::uint32_t r,
                               const std::uint32_t* pos) {
        FusedAttention(l, k_cache_[i], v_cache_[i], vt_cache_[i], pos, rows, r);
      });
    } else {
      WideMark(kWOther);
      fn::RmsNormRows(s.res, l.attn_norm.f32(), s.xn, n, H, 1, c_.eps, stream_);
      Unstage();
      const bool ok = l.linear
                          ? WideGdn(l, i, n, error)
                          : WideAttention(l, k_cache_[i], v_cache_[i],
                                          vt_cache_[i], s.res, n, pos_, error);
      if (!ok)
        return false;
    }
    if (debug & 1u) {
      grouped(l.post_norm, [&](float* rows, std::uint32_t r,
                               const std::uint32_t*) { FusedMoe(l, rows, r); });
      continue;
    }
    WideMark(kWOther);
    fn::RmsNormRows(s.res, l.post_norm.f32(), s.xn, n, H, 1, c_.eps, stream_);
    Unstage();
    if (!WideMoe(l, s.res, n, error))
      return false;
    if (df_)
      DFlashTap(i, s.res, n, true);
  }
  return true;
}

bool Engine::WideLogits(std::span<const std::int32_t> tokens, float* logits,
                        std::string* error) {
  const std::uint32_t H = c_.hidden;
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (!AllocateWide(error))
    return false;
  Wide& s = wide_;
  if (n == 0 || n > s.rows || pos_ + n > options_.max_context) {
    Fail(error, "wide logits chunk does not fit");
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(n) * c_.vocab * 4;
  if (s.logits_rows < n) {
    void* p = nullptr;
    if (!Ok(hipMalloc(&p, bytes), "wide logits alloc", error))
      return false;
    owned_.push_back(p);
    s.logits = static_cast<float*>(p);
    s.logits_rows = n;
  }
  if (!Sync(error))
    return false;
  std::copy_n(tokens.data(), n, s.tokens_host);
  *pos_host_ = pos_;
  if (!Ok(hipMemcpyAsync(s.tokens, s.tokens_host, n * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "token upload", error) ||
      !Ok(hipMemcpyAsync(pos_dev_, pos_host_, sizeof(std::uint32_t),
                         hipMemcpyHostToDevice, stream_),
          "position upload", error) ||
      !WideTrunk(n, error))
    return false;
  fn::RmsNormRows(s.res, output_norm_.f32(), s.h, n, H, 1, c_.eps, stream_);
  Unstage();
  if (!WideDense(output_, s.h, s.logits, n, error))
    return false;
  pos_ += n;
  return Ok(hipMemcpyAsync(logits, s.logits, bytes, hipMemcpyDeviceToHost,
                           stream_),
            "logits download", error) &&
         Sync(error);
}

bool Engine::SpecPrefillWide(std::span<const std::int32_t> tokens,
                             Candidates* last, std::string* error) {
  const std::uint32_t H = c_.hidden;
  const bool mtp = HasMtp() && mtp_k_ != nullptr;
  if (tokens.empty() ||
      pos_ + tokens.size() + kMaxRows > options_.max_context) {
    Fail(error, "prompt does not fit the context");
    return false;
  }
  if (!AllocateWide(error))
    return false;
  Wide& s = wide_;
  const auto end = static_cast<std::uint32_t>(pos_ + tokens.size());
  for (std::size_t off = 0; off < tokens.size(); off += s.rows) {
    const auto n = static_cast<std::uint32_t>(
        std::min<std::size_t>(s.rows, tokens.size() - off));
    const bool final = off + n == tokens.size();
    // The pinned staging is read when the queued copies run.
    if (!Sync(error))
      return false;
    std::copy_n(tokens.data() + off, n, s.tokens_host);
    *pos_host_ = pos_;
    if (!Ok(hipMemcpyAsync(s.tokens, s.tokens_host, n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_),
            "prefill token upload", error) ||
        !Ok(hipMemcpyAsync(pos_dev_, pos_host_, sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream_),
            "position upload", error))
      return false;
    if (!WideTrunk(n, error))
      return false;
    if (df_ && !DFlashInject(n, pos_, end, true, nullptr, error))
      return false;
    WideMark(kWOther);
    if (final) {
      // The last row's logits through the decode head (same arithmetic as
      // the 8-row path's final chunk).
      AddNormQuant(s.res + static_cast<std::size_t>(n - 1) * H, nullptr,
                   output_norm_.f32(), xn_, aq_, 1, H, c_.eps, stream_);
      const GemvSeg head{output_.data,  nullptr,          logits_, output_.rows,
                         Kind(output_), GemvMode::kStore, c_.vocab};
      MultiGemv(&head, 1, aq_, xn_, 1, H, stream_);
      TopCandidates(1);
    }
    WideMark(kWMtp);
    if (mtp) {
      // MTP rows pair token i with the trunk's normed hidden i - 1: the
      // pending row from the previous chunk (zeros at position 0), then
      // this chunk's rows but the last, which becomes the new pending row.
      fn::RmsNormRows(s.res, output_norm_.f32(), s.h, n, H, 1, c_.eps, stream_);
      fn::CopyDevice(mtp_pending_, s.mh, H, stream_);
      if (n > 1)
        fn::CopyDevice(s.h, s.mh + H, static_cast<std::size_t>(n - 1) * H,
                       stream_);
      fn::CopyDevice(s.h + static_cast<std::size_t>(n - 1) * H, mtp_pending_, H,
                     stream_);
      fn::EmbedTokens(token_embd_.data, Small(token_embd_.type), s.tokens,
                      s.emb, n, H, 1, stream_);
      fn::RmsNormRows(s.emb, mtp_.enorm.f32(), s.emb, n, H, 1, c_.eps, stream_);
      fn::RmsNormRows(s.mh, mtp_.hnorm.f32(), s.mh, n, H, 1, c_.eps, stream_);
      Concat2(s.emb, s.mh, s.cat, n, H, stream_);
      Unstage();
      const Layer& b = mtp_.block;
      if (!WideDense(mtp_.eh_proj, s.cat, s.res, n, error))
        return false;
      fn::RmsNormRows(s.res, b.attn_norm.f32(), s.xn, n, H, 1, c_.eps, stream_);
      Unstage();
      if (!WideAttention(b, mtp_k_, mtp_v_, mtp_vt_, s.res, n, pos_, error))
        return false;
      fn::RmsNormRows(s.res, b.post_norm.f32(), s.xn, n, H, 1, c_.eps, stream_);
      Unstage();
      if (!WideMoe(b, s.res, n, error))
        return false;
    }
    WideMark(kWOther);
    pos_ += n;
  }
  WideReport(tokens.size());
  if (mtp) {
    fn::CopyDevice(mtp_pending_, mtp_h_, H, stream_);
    mtp_rows_ = 1;
  }
  verify_rows_ = 0;
  if (!Ok(hipGetLastError(), "prefill launch", error) ||
      !Ok(hipMemcpyAsync(last, cand_out_, sizeof(Candidates),
                         hipMemcpyDeviceToHost, stream_),
          "candidates download", error))
    return false;
  return Sync(error);
}

}  // namespace gufo::models::qwen36_a3b
