// GPU correctness and DRAM pass bench of the decode-twin dense kernels.
// For every selected tensor of a real trunk and its twin, at each N = 1..32:
//   kernel8 = qfn_mmq_q8_0_dense_vec_preq (the existing Q8 dense path), kernel4
//   = qfn_mmq_dense_vec_preq, ref4 / ref8 = CPU dot of the DEQUANTISED weights
//   with the DEQUANTISED Q8_1 activations (the same activation bits the kernels
//   read).
// Reports per tensor and N: kernel4 vs ref4 (kernel / layout correctness: must
// agree to float tolerance), kernel8 vs ref8 (sanity of the reference), kernel4
// vs kernel8
// (= the 4-bit quantisation error: max abs diff, relative RMS, top-1 agreement
// over the output rows). Exit 1 if kernel4 vs ref4 or kernel8 vs ref8 relative
// RMS > 2e-3. usage: dense_twin_gpu_test TRUNK_SHARD1.gguf TWIN.gguf
// [tensor-name ...]   (default tensors: blk.0.attn_qkv.weight
// blk.0.attn_gate.weight blk.0.ssm_out.weight blk.3.attn_q.weight
//        blk.3.attn_v.weight blk.0.hc_attn_down.weight blk.0.hc_attn_up.weight
//        blk.0.ffn_gate_shexp.weight blk.0.ffn_down_shexp.weight)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

extern "C" {
int qfn_mmq_iq4_dense_batch_preq(int type, const void* W, const void* X,
                                 float* out, int M, int N, int K,
                                 hipStream_t stream);
size_t qfn_mmq_q8_1_bytes(int N, int K);
int qfn_mmq_quantize_q8_1(const float* X_f32, void* X_q8, int N, int K,
                          hipStream_t stream);
int qfn_mmq_q8_0_dense_vec_preq(const void* W_q8_0, const void* W_gate,
                                const void* X_q8, float* out_f32, int M, int N,
                                int K, hipStream_t stream);
int qfn_mmq_k_dense(int weight_type, const void* W, const float* X, float* out,
                    int M, int N, int K, hipStream_t stream);
unsigned long long qfn_mmq_iq4_dense_launch_count();
int qfn_mmq_dense_vec_preq_reference(int weight_type, const void* W,
                                     const void* X_q8, const int32_t* zero_ids,
                                     float* out_f32, int M, int N, int K,
                                     hipStream_t stream);
int qfn_mmq_dense_vec_preq(int weight_type, const void* W, const void* X_q8,
                           const int32_t* zero_ids, float* out_f32, int M,
                           int N, int K, hipStream_t stream);
}

namespace {
namespace core = gufo::core;
constexpr std::size_t kTail = 4096;

#define HIPCHECK(x)                                                          \
  do {                                                                       \
    const hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                   __FILE__, __LINE__);                                      \
      return 2;                                                              \
    }                                                                        \
  } while (0)

float Half(const std::uint8_t* p) {
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  return gufo::quant::Fp16ToFloat(h);
}

// one row of cols elements -> float
void DequantRow(core::GgmlType type, const std::uint8_t* row, float* out,
                std::size_t cols) {
  switch (type) {
    case core::GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(row, out, cols);
      break;
    case core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(row, out, cols);
      break;
    case core::GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(row, out, cols);
      break;
    case core::GgmlType::kIQ4_NL: {  // IQ4_NL: d (half) + 16 bytes of nibbles
                                     // per 32 values, value = d *
                                     // kvalues[nibble]
      static const int kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1,    13,   25,  38,  53,  69,  89,  113};
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const std::uint8_t* blk = row + b * 18;
        const float d = Half(blk);
        for (int j = 0; j < 16; ++j) {
          out[b * 32 + j] = d * static_cast<float>(kv[blk[2 + j] & 0x0F]);
          out[b * 32 + j + 16] = d * static_cast<float>(kv[blk[2 + j] >> 4]);
        }
      }
      break;
    }
    case core::GgmlType::kIQ4_XS: {  // IQ4_XS: d (half), scales_h (u16),
                                     // scales_l[4], qs[128] per 256 values
      static const int kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1,    13,   25,  38,  53,  69,  89,  113};
      for (std::size_t b = 0; b < cols / 256; ++b) {
        const std::uint8_t* blk = row + b * 136;
        const float d = Half(blk);
        std::uint16_t sh;
        std::memcpy(&sh, blk + 2, 2);
        const std::uint8_t* sl = blk + 4;
        const std::uint8_t* qs = blk + 8;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                         (((sh >> (2 * ib)) & 3) << 4);
          const float dl = d * static_cast<float>(ls - 32);
          for (int j = 0; j < 16; ++j) {
            out[b * 256 + ib * 32 + j] =
                dl * static_cast<float>(kv[qs[ib * 16 + j] & 0x0F]);
            out[b * 256 + ib * 32 + j + 16] =
                dl * static_cast<float>(kv[qs[ib * 16 + j] >> 4]);
          }
        }
      }
      break;
    }
    case core::GgmlType::kQ5_1:
      for (std::size_t b = 0; b < cols / 32; ++b) {
        const std::uint8_t* blk = row + b * 24;
        const float d = Half(blk), m = Half(blk + 2);
        std::uint32_t qh;
        std::memcpy(&qh, blk + 4, 4);
        const std::uint8_t* qs = blk + 8;
        for (int j = 0; j < 16; ++j) {
          const std::uint8_t xh0 =
              static_cast<std::uint8_t>(((qh >> j) << 4) & 0x10);
          const std::uint8_t xh1 =
              static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
          out[b * 32 + j] = static_cast<float>((qs[j] & 0x0F) | xh0) * d + m;
          out[b * 32 + j + 16] = static_cast<float>((qs[j] >> 4) | xh1) * d + m;
        }
      }
      break;
    default:
      std::fprintf(stderr, "unsupported type\n");
      std::exit(2);
  }
}

std::size_t RowBytes(core::GgmlType type, std::size_t cols) {
  switch (type) {
    case core::GgmlType::kQ8_0:
      return cols / 32 * 34;
    case core::GgmlType::kQ4_K:
      return cols / 256 * 144;
    case core::GgmlType::kQ5_K:
      return cols / 256 * 176;
    case core::GgmlType::kQ5_1:
      return cols / 32 * 24;
    case core::GgmlType::kIQ4_NL:
      return cols / 32 * 18;
    case core::GgmlType::kIQ4_XS:
      return cols / 256 * 136;
    default:
      return 0;
  }
}

struct Dev {
  void* p{nullptr};
  ~Dev() {
    if (p)
      (void)hipFree(p);
  }
};

bool Upload(Dev& d, const void* src, std::size_t bytes) {
  if (hipMalloc(&d.p, bytes + kTail) != hipSuccess)
    return false;
  (void)hipMemset(d.p, 0, bytes + kTail);
  return hipMemcpy(d.p, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

struct Stats {
  double max_abs{0}, rel_rms{0};
};
Stats Diff(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0, mx = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
    mx = std::max(mx, std::abs(d));
  }
  return {mx, den > 0 ? std::sqrt(num / den) : 0.0};
}
int Project(int type, const void* w, const void* q, const std::int32_t* zeros,
            float* out, int M, int N, int K, hipStream_t stream) {
  const bool iq4 = type == int(core::GgmlType::kIQ4_XS) ||
                   type == int(core::GgmlType::kIQ4_NL);
  const bool q8 = type == int(core::GgmlType::kQ8_0);
  if (N <= 8)
    return q8 ? qfn_mmq_q8_0_dense_vec_preq(w, nullptr, q, out, M, N, K, stream)
              : qfn_mmq_dense_vec_preq(type, w, q, zeros, out, M, N, K, stream);
  if (q8 && M == 320 && K == 10240) {
    const int full = N / 8 * 8;
    const int rc =
        qfn_mmq_q8_0_dense_vec_preq(w, nullptr, q, out, M, full, K, stream);
    if (rc != 0 || full == N)
      return rc;
    return qfn_mmq_q8_0_dense_vec_preq(
        w, nullptr,
        static_cast<const std::uint8_t*>(q) + qfn_mmq_q8_1_bytes(full, K),
        out + static_cast<std::size_t>(full) * M, M, N - full, K, stream);
  }
  const int chunk =
      iq4 || (q8 && ((K == 2560 && M >= 1024) || (K == 320 && M == 10240))) ? 32
                                                                            : 8;
  for (int r = 0; r < N; r += chunk) {
    const int n = std::min(chunk, N - r);
    const auto* x =
        static_cast<const std::uint8_t*>(q) + qfn_mmq_q8_1_bytes(r, K);
    const int rc =
        iq4  ? qfn_mmq_iq4_dense_batch_preq(
                   type, w, x, out + static_cast<std::size_t>(r) * M, M, n, K,
                   stream)
        : q8 ? qfn_mmq_q8_0_dense_vec_preq(
                   w, nullptr, x, out + static_cast<std::size_t>(r) * M, M, n,
                   K, stream)
             : qfn_mmq_dense_vec_preq(type, w, x, zeros,
                                      out + static_cast<std::size_t>(r) * M, M,
                                      n, K, stream);
    if (rc != 0)
      return rc;
  }
  return 0;
}

// --pass-bench: DRAM-regime timing. Every twin tensor and its Q8_0 trunk tensor
// stay resident (about 6 GB); one timed "pass" launches all of them back to
// back, so the 3.8 GB (Q8_0) / ~2.2 GB (4-bit) of weights stream from DRAM as
// in the model, unlike the per-shape --bench (200 launches of one <= 33 MB
// tensor, which run from the Infinity Cache / L2 and overstate bandwidth ~3x).
int PassBench(const core::GgufReader& trunk, const core::GgufReader& twin,
              bool batch) {
  hipStream_t stream = nullptr;
  HIPCHECK(hipStreamCreate(&stream));
  struct T {
    std::string name;
    core::GgmlType ty4;
    std::size_t M, K;
    Dev w8, w4;
  };
  std::vector<std::unique_ptr<T>> ts;
  double bytes8 = 0, bytes4 = 0;
  for (const auto& ti : twin.GetTensors()) {
    const auto* t8 = trunk.FindTensor(std::string(ti.name));
    if (t8 == nullptr || t8->type != core::GgmlType::kQ8_0)
      continue;
    auto x = std::make_unique<T>();
    x->name = std::string(ti.name);
    x->ty4 = ti.type;
    x->K = t8->dimensions[0];
    x->M = t8->dimensions[1];
    const std::size_t b8 = RowBytes(core::GgmlType::kQ8_0, x->K) * x->M,
                      b4 = RowBytes(x->ty4, x->K) * x->M;
    if (!Upload(x->w8, t8->data, b8) || !Upload(x->w4, ti.data, b4)) {
      std::fprintf(stderr, "upload failed (%s)\n", x->name.c_str());
      return 2;
    }
    bytes8 += static_cast<double>(b8);
    bytes4 += static_cast<double>(b4);
    ts.push_back(std::move(x));
  }
  std::map<std::size_t, std::unique_ptr<Dev>> act;
  std::mt19937 rng(7);
  std::normal_distribution<float> gauss(0.0F, 1.0F);
  for (const auto& x : ts) {
    if (act.count(x->K) != 0)
      continue;
    std::vector<float> h(32 * x->K);
    for (auto& v : h)
      v = gauss(rng);
    Dev dx;
    auto dq = std::make_unique<Dev>();
    if (!Upload(dx, h.data(), h.size() * 4) ||
        hipMalloc(&dq->p, qfn_mmq_q8_1_bytes(32, static_cast<int>(x->K)) +
                              kTail) != hipSuccess)
      return 2;
    if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p), dq->p, 32,
                              static_cast<int>(x->K), stream) != 0)
      return 2;
    HIPCHECK(hipStreamSynchronize(stream));
    act[x->K] = std::move(dq);
  }
  Dev zero_ids, out;
  const std::int32_t zeros[8] = {};
  if (!Upload(zero_ids, zeros, sizeof zeros) ||
      hipMalloc(&out.p, std::size_t{262144} * 32 * 4) != hipSuccess)
    return 2;
  std::printf(
      "pass bench: %zu tensors, Q8_0 %.3f GB, 4-bit %.3f GB (bytes ratio "
      "%.3f)\n",
      ts.size(), bytes8 / 1e9, bytes4 / 1e9, bytes4 / bytes8);
  hipEvent_t e0 = nullptr, e1 = nullptr;
  HIPCHECK(hipEventCreate(&e0));
  HIPCHECK(hipEventCreate(&e1));
  // ABAB: per N, 10 rounds of (one Q8_0 pass, one 4-bit pass), each pass timed
  // alone after a sync; launch errors checked after every pass; min / median /
  // max per path.
  for (int N = batch ? 9 : 1; N <= (batch ? 32 : 8); ++N) {
    std::vector<double> v[2];
    for (int round = -1; round < 10; ++round) {
      for (int which = 0; which < 2; ++which) {
        HIPCHECK(hipStreamSynchronize(stream));
        HIPCHECK(hipEventRecord(e0, stream));
        int rc = 0;
        for (const auto& x : ts) {
          const void* q = act[x->K]->p;
          rc |= Project(which == 0 ? int(core::GgmlType::kQ8_0) : int(x->ty4),
                        which == 0 ? x->w8.p : x->w4.p, q,
                        static_cast<const std::int32_t*>(zero_ids.p),
                        static_cast<float*>(out.p), static_cast<int>(x->M), N,
                        static_cast<int>(x->K), stream);
        }
        HIPCHECK(hipEventRecord(e1, stream));
        HIPCHECK(hipEventSynchronize(e1));
        if (rc != 0 || hipGetLastError() != hipSuccess) {
          std::fprintf(stderr, "launch error in pass (N=%d, %s)\n", N,
                       which ? "4-bit" : "Q8_0");
          return 2;
        }
        float f = 0;
        HIPCHECK(hipEventElapsedTime(&f, e0, e1));
        if (round >= 0)
          v[which].push_back(f);
      }
    }
    for (auto& x : v)
      std::sort(x.begin(), x.end());
    const double m8 = v[0][v[0].size() / 2], m4 = v[1][v[1].size() / 2];
    std::printf(
        "PASS N=%d | Q8_0 median %7.2f ms (min %6.2f max %6.2f, %5.1f GB/s) | "
        "4-bit median %7.2f ms (min %6.2f max %6.2f, %5.1f GB/s) | 4-bit/Q8 "
        "time %.2f\n",
        N, m8, v[0].front(), v[0].back(), bytes8 / (m8 * 1e6), m4, v[1].front(),
        v[1].back(), bytes4 / (m4 * 1e6), m4 / m8);
  }
  std::printf("iq4_dense_launches=%llu\n", qfn_mmq_iq4_dense_launch_count());
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: dense_twin_gpu_test TRUNK_SHARD1.gguf TWIN.gguf "
                 "[tensor-name ...]\n");
    return 2;
  }
  std::string err;
  auto trunk = core::GgufReader::OpenFile(argv[1], &err);
  if (!trunk) {
    std::fprintf(stderr, "trunk: %s\n", err.c_str());
    return 2;
  }
  auto twin = core::GgufReader::OpenFile(argv[2], &err);
  if (!twin) {
    std::fprintf(stderr, "twin: %s\n", err.c_str());
    return 2;
  }
  if (argc == 4 && (std::string(argv[3]) == "--pass-bench" ||
                    std::string(argv[3]) == "--batch-pass-bench"))
    return PassBench(*trunk, *twin,
                     std::string(argv[3]) == "--batch-pass-bench");
  std::vector<std::string> names;
  for (int i = 3; i < argc; ++i)
    names.emplace_back(argv[i]);
  bool bench = false;  // --bench: also time the Q8_0 vector GEMV and the twin
                       // kernel per shape at N = 1, 2, 4, 8 (hipEvents, 20 warm
                       // + 200 timed launches)
  for (auto it = names.begin(); it != names.end();) {
    if (*it == "--bench") {
      bench = true;
      it = names.erase(it);
    } else {
      ++it;
    }
  }
  if (names.size() == 1 &&
      names[0] == "--all") {  // every tensor of the twin file (the whole twin
                              // set), N = 1..32
    names.clear();
    for (const auto& ti : twin->GetTensors())
      names.emplace_back(ti.name);
  }
  if (names.empty())
    names = {"blk.0.attn_qkv.weight",      "blk.0.attn_gate.weight",
             "blk.0.ssm_out.weight",       "blk.3.attn_q.weight",
             "blk.3.attn_v.weight",        "blk.0.hc_attn_down.weight",
             "blk.0.hc_attn_up.weight",    "blk.0.ffn_gate_shexp.weight",
             "blk.0.ffn_down_shexp.weight"};
  hipStream_t stream = nullptr;
  HIPCHECK(hipStreamCreate(&stream));
  Dev zero_ids;
  const std::int32_t zeros[8] = {};
  if (!Upload(zero_ids, zeros, sizeof zeros))
    return 2;
  int failures = 0;
  std::printf(
      "%-30s %-5s %2s | k4-vs-ref4 maxabs relRMS | k8-vs-ref8 maxabs relRMS | "
      "k4-vs-k8 maxabs relRMS top1\n",
      "tensor", "type", "N");
  for (const auto& name : names) {
    const auto* t8 = trunk->FindTensor(name);
    const auto* t4 = twin->FindTensor(name);
    if (!t8 || !t4) {
      std::fprintf(stderr, "tensor %s missing (trunk %d twin %d)\n",
                   name.c_str(), t8 != nullptr, t4 != nullptr);
      ++failures;
      continue;
    }
    const std::size_t K = t8->dimensions[0], M = t8->dimensions[1];
    if (t8->type != core::GgmlType::kQ8_0 || t4->dimensions[0] != K ||
        t4->dimensions[1] != M) {
      std::fprintf(stderr, "%s: type / shape mismatch\n", name.c_str());
      ++failures;
      continue;
    }
    const auto ty4 = t4->type;
    const std::size_t rb8 = RowBytes(core::GgmlType::kQ8_0, K),
                      rb4 = RowBytes(ty4, K);
    Dev w8, w4;
    if (!Upload(w8, t8->data, rb8 * M) || !Upload(w4, t4->data, rb4 * M)) {
      std::fprintf(stderr, "upload failed\n");
      return 2;
    }
    // CPU dequantised weights
    std::vector<float> dq8(M * K), dq4(M * K);
    for (std::size_t r = 0; r < M; ++r) {
      DequantRow(core::GgmlType::kQ8_0,
                 static_cast<const std::uint8_t*>(t8->data) + r * rb8,
                 dq8.data() + r * K, K);
      DequantRow(ty4, static_cast<const std::uint8_t*>(t4->data) + r * rb4,
                 dq4.data() + r * K, K);
    }
    for (int N = 1; N <= 32; ++N) {
      std::mt19937 rng(1234 + N);
      std::normal_distribution<float> gauss(0.0F, 1.0F);
      std::vector<float> x(static_cast<std::size_t>(N) * K);
      for (auto& v : x)
        v = gauss(rng) *
            (1.0F + 3.0F * (rng() % 97 ==
                            0));  // heavy-tailed: a few large activations
      Dev dx, dq, dout8, dout4;
      const std::size_t qbytes = qfn_mmq_q8_1_bytes(N, static_cast<int>(K));
      if (!Upload(dx, x.data(), x.size() * 4) ||
          hipMalloc(&dq.p, qbytes + kTail) != hipSuccess ||
          hipMalloc(&dout8.p, N * M * 4 + kTail) != hipSuccess ||
          hipMalloc(&dout4.p, N * M * 4 + kTail) != hipSuccess)
        return 2;
      (void)hipMemset(dq.p, 0, qbytes + kTail);
      HIPCHECK(
          hipMemsetAsync(dout4.p, 0xA5, N * M * sizeof(float) + kTail, stream));
      if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p), dq.p, N,
                                static_cast<int>(K), stream) != 0) {
        std::fprintf(stderr, "quantize failed\n");
        return 2;
      }
      if (Project(int(core::GgmlType::kQ8_0), w8.p, dq.p,
                  static_cast<const std::int32_t*>(zero_ids.p),
                  static_cast<float*>(dout8.p), static_cast<int>(M), N,
                  static_cast<int>(K), stream) != 0) {
        std::fprintf(stderr, "q8 dense failed\n");
        return 2;
      }
      if (Project(int(ty4), w4.p, dq.p,
                  static_cast<const std::int32_t*>(zero_ids.p),
                  static_cast<float*>(dout4.p), static_cast<int>(M), N,
                  static_cast<int>(K), stream) != 0) {
        std::fprintf(stderr, "twin dense failed\n");
        return 2;
      }
      HIPCHECK(hipStreamSynchronize(stream));
      std::vector<float> k8(N * M), k4(N * M);
      HIPCHECK(hipMemcpy(k8.data(), dout8.p, N * M * 4, hipMemcpyDeviceToHost));
      HIPCHECK(hipMemcpy(k4.data(), dout4.p, N * M * 4, hipMemcpyDeviceToHost));
      // Padded template columns may not write beyond the valid output rows.
      std::vector<std::uint8_t> output_guard(kTail);
      HIPCHECK(hipMemcpy(
          output_guard.data(),
          static_cast<const std::uint8_t*>(dout4.p) + N * M * sizeof(float),
          output_guard.size(), hipMemcpyDeviceToHost));
      if (std::any_of(output_guard.begin(), output_guard.end(),
                      [](std::uint8_t b) { return b != 0xA5; })) {
        std::printf("FAIL OUTPUT_GUARD %s N=%d\n", name.c_str(), N);
        ++failures;
      }
      if (N <= 8 &&
          (ty4 == core::GgmlType::kIQ4_XS || ty4 == core::GgmlType::kIQ4_NL)) {
        if (qfn_mmq_dense_vec_preq_reference(
                static_cast<int>(ty4), w4.p, dq.p,
                static_cast<const std::int32_t*>(zero_ids.p),
                static_cast<float*>(dout4.p), static_cast<int>(M), N,
                static_cast<int>(K), stream) != 0)
          return 2;
        HIPCHECK(hipStreamSynchronize(stream));
        std::vector<float> old(N * M);
        HIPCHECK(
            hipMemcpy(old.data(), dout4.p, N * M * 4, hipMemcpyDeviceToHost));
        std::size_t different = 0;
        for (std::size_t i = 0; i < old.size(); ++i)
          different += std::memcmp(&old[i], &k4[i], sizeof(float)) != 0;
        failures += different != 0;
        const auto d = Diff(k4, old);
        std::printf(
            "IDENTITY %s N=%d different=%zu/%zu maxabs=%.9e relRMS=%.9e\n",
            name.c_str(), N, different, old.size(), d.max_abs, d.rel_rms);
      }
      const bool iq4 =
          ty4 == core::GgmlType::kIQ4_XS || ty4 == core::GgmlType::kIQ4_NL;
      std::vector<float> candidate = k4;
      if (iq4) {
        // Exercise the wide adapter even at N<=8: its eight-row instance
        // serves the final chunk of 33..40-row decode batches. Tight input
        // allocation and an output sentinel expose invalid padded accesses.
        Dev tightq;
        HIPCHECK(hipMalloc(&tightq.p, qbytes));
        HIPCHECK(hipMemcpyAsync(tightq.p, dq.p, qbytes, hipMemcpyDeviceToDevice,
                                stream));
        HIPCHECK(hipMemsetAsync(dout4.p, 0xA5, N * M * sizeof(float) + kTail,
                                stream));
        if (qfn_mmq_iq4_dense_batch_preq(int(ty4), w4.p, tightq.p,
                                         static_cast<float*>(dout4.p), M, N, K,
                                         stream) != 0)
          return 2;
        HIPCHECK(hipStreamSynchronize(stream));
        HIPCHECK(hipMemcpy(candidate.data(), dout4.p, N * M * sizeof(float),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(
            output_guard.data(),
            static_cast<const std::uint8_t*>(dout4.p) + N * M * sizeof(float),
            output_guard.size(), hipMemcpyDeviceToHost));
        if (std::any_of(output_guard.begin(), output_guard.end(),
                        [](std::uint8_t b) { return b != 0xA5; })) {
          std::printf("FAIL WIDE_OUTPUT_GUARD %s N=%d\n", name.c_str(), N);
          ++failures;
        }
      }
      // Compute each exact activation row alone, including its own quantize.
      // No reuse of batch output or padded input can hide a row-count error.
      Dev oneq, oneout;
      if (hipMalloc(&oneq.p, qfn_mmq_q8_1_bytes(1, K)) != hipSuccess ||
          hipMalloc(&oneout.p, M * sizeof(float)) != hipSuccess)
        return 2;
      std::vector<float> alone(N * M);
      for (int t = 0; t < N; ++t) {
        if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p) +
                                      static_cast<std::size_t>(t) * K,
                                  oneq.p, 1, K, stream) != 0 ||
            qfn_mmq_dense_vec_preq(int(ty4), w4.p, oneq.p,
                                   static_cast<const std::int32_t*>(zero_ids.p),
                                   static_cast<float*>(oneout.p), M, 1, K,
                                   stream) != 0)
          return 2;
        HIPCHECK(hipMemcpyAsync(alone.data() + static_cast<std::size_t>(t) * M,
                                oneout.p, M * sizeof(float),
                                hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipStreamSynchronize(stream));
      }
      std::size_t row_differences = 0;
      for (std::size_t i = 0; i < alone.size(); ++i)
        row_differences +=
            std::memcmp(&alone[i], &candidate[i], sizeof(float)) != 0;
      std::printf("ROW_IDENTITY %s N=%d different=%zu/%zu\n", name.c_str(), N,
                  row_differences, alone.size());
      // Every valid row of each wide instance must match the unchanged N=1
      // serving path. Legacy <=8 kernels are independently checked above.
      if (iq4 || N > 8)
        failures += row_differences != 0;
      // activations as the kernels read them: block_q8_1 = { half d; half s;
      // int8 qs[32] }, row stride GGML_PAD(K, 512) / 32 blocks
      std::vector<std::uint8_t> qh(qbytes);
      HIPCHECK(hipMemcpy(qh.data(), dq.p, qbytes, hipMemcpyDeviceToHost));
      const std::size_t stride_blocks = (K + 511) / 512 * 512 / 32;
      std::vector<float> xq(static_cast<std::size_t>(N) * K);
      for (int n = 0; n < N; ++n)
        for (std::size_t b = 0; b < K / 32; ++b) {
          const std::uint8_t* blk =
              qh.data() +
              (static_cast<std::size_t>(n) * stride_blocks + b) * 36;
          const float d = Half(blk);
          for (int j = 0; j < 32; ++j)
            xq[static_cast<std::size_t>(n) * K + b * 32 + j] =
                d * static_cast<float>(static_cast<std::int8_t>(blk[4 + j]));
        }
      std::vector<float> ref8(N * M), ref4(N * M);
      for (int n = 0; n < N; ++n)
        for (std::size_t m = 0; m < M; ++m) {
          double s8 = 0, s4 = 0;
          for (std::size_t k = 0; k < K; ++k) {
            const double a = xq[static_cast<std::size_t>(n) * K + k];
            s8 += a * dq8[m * K + k];
            s4 += a * dq4[m * K + k];
          }
          ref8[n * M + m] = static_cast<float>(s8);
          ref4[n * M + m] = static_cast<float>(s4);
        }
      if (bench) {
        hipEvent_t e0 = nullptr, e1 = nullptr;
        HIPCHECK(hipEventCreate(&e0));
        HIPCHECK(hipEventCreate(&e1));
        const auto run8 = [&] {
          return Project(int(core::GgmlType::kQ8_0), w8.p, dq.p,
                         static_cast<const std::int32_t*>(zero_ids.p),
                         static_cast<float*>(dout8.p), static_cast<int>(M), N,
                         static_cast<int>(K), stream);
        };
        const auto run4 = [&] {
          return Project(int(ty4), w4.p, dq.p,
                         static_cast<const std::int32_t*>(zero_ids.p),
                         static_cast<float*>(dout4.p), static_cast<int>(M), N,
                         static_cast<int>(K), stream);
        };
        const auto runmmq = [&] {
          return qfn_mmq_k_dense(
              static_cast<int>(ty4), w4.p, static_cast<const float*>(dx.p),
              static_cast<float*>(dout4.p), static_cast<int>(M), N,
              static_cast<int>(K), stream);
        };
        double us[3] = {0, 0, 0};
        for (int which = 0; which < 3; ++which) {
          for (int i = 0; i < 20; ++i)
            (void)(which == 0 ? run8() : which == 1 ? run4() : runmmq());
          HIPCHECK(hipStreamSynchronize(stream));
          HIPCHECK(hipEventRecord(e0, stream));
          for (int i = 0; i < 200; ++i)
            (void)(which == 0 ? run8() : which == 1 ? run4() : runmmq());
          HIPCHECK(hipEventRecord(e1, stream));
          HIPCHECK(hipEventSynchronize(e1));
          float ms = 0;
          HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
          us[which] = ms * 1000.0 / 200.0;
        }
        std::printf(
            "BENCH %-30s %-5s N=%d rows=%zu cols=%zu | Q8_0 %8.1f us (%6.1f "
            "GB/s) | %-5s %8.1f us (%6.1f GB/s) | twin/Q8 time ratio %.2f\n",
            name.c_str(), std::string(core::ToString(ty4)).c_str(), N, M, K,
            us[0], static_cast<double>(rb8 * M) / (us[0] * 1e3),
            std::string(core::ToString(ty4)).c_str(), us[1],
            static_cast<double>(rb4 * M) / (us[1] * 1e3), us[1] / us[0]);
        std::printf(
            "BENCHMMQ %-30s %-5s N=%d rows=%zu cols=%zu | Q8_0 vec %8.1f us | "
            "%s vec %8.1f us | %s MMQ tile %8.1f us (%6.1f GB/s) | MMQ/Q8 "
            "%.2f\n",
            name.c_str(), std::string(core::ToString(ty4)).c_str(), N, M, K,
            us[0], std::string(core::ToString(ty4)).c_str(), us[1],
            std::string(core::ToString(ty4)).c_str(), us[2],
            static_cast<double>(rb4 * M) / (us[2] * 1e3), us[2] / us[0]);
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
      }
      const auto a = Diff(k4, ref4), b = Diff(k8, ref8), c = Diff(k4, k8);
      const auto wide = Diff(candidate, ref4);
      if (iq4) {
        std::printf("WIDE_CPU_REF %s N=%d maxabs=%.9e relRMS=%.9e\n",
                    name.c_str(), N, wide.max_abs, wide.rel_rms);
        if (wide.rel_rms > 2e-3)
          ++failures;
      }
      int agree = 0;
      for (int n = 0; n < N; ++n) {
        const auto top = [&](const std::vector<float>& v) {
          return std::max_element(v.begin() + n * M, v.begin() + (n + 1) * M) -
                 (v.begin() + n * M);
        };
        agree += top(k4) == top(k8);
      }
      std::printf(
          "%-30s %-5s %2d | %10.3e %9.2e | %10.3e %9.2e | %10.3e %9.2e %d/%d\n",
          name.c_str(), std::string(core::ToString(ty4)).c_str(), N, a.max_abs,
          a.rel_rms, b.max_abs, b.rel_rms, c.max_abs, c.rel_rms, agree, N);
      if (a.rel_rms > 2e-3 || b.rel_rms > 2e-3) {
        std::printf("  FAIL: kernel disagrees with its own CPU reference\n");
        ++failures;
      }
    }
  }
  std::printf("iq4_dense_launches=%llu\n", qfn_mmq_iq4_dense_launch_count());
  std::printf("dense_twin_gpu_test: %s (%d failures)\n",
              failures ? "FAIL" : "all kernels agree with their CPU references",
              failures);
  return failures ? 1 : 0;
}
