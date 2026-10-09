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
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/iq4_wide_policy.hpp"

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
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return {std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()};
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
    mx = std::max(mx, std::abs(d));
  }
  return {mx, den > 0
                  ? std::sqrt(num / den)
                  : (num == 0 ? 0.0 : std::numeric_limits<double>::infinity())};
}
int Project(int type, const void* w, const void* q, const std::int32_t* zeros,
            float* out, int M, int N, int K, hipStream_t stream,
            bool multi_session = false) {
  const bool iq4 = type == int(core::GgmlType::kIQ4_XS) ||
                   type == int(core::GgmlType::kIQ4_NL);
  const bool q8 = type == int(core::GgmlType::kQ8_0);
  if (iq4 && multi_session)
    return qfn_mmq_iq4_dense_batch_preq(type, w, q, out, M, N, K, stream);
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
      iq4 || (q8 && ((K == 2560 && M >= 1024) || (K == 320 && M == 10240)))
          ? (q8 && K == 2560 && M >= 8192 && N > 32 && N <= 48 ? 48 : 32)
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

// DenseBatch's eight-row fallback calls Dense on each original FP32 slice.
// Quantize each slice independently, including the unchanged N=1 kernel for
// N=9/17/25's last row. Tight allocations end at the last valid activation.
int ProjectSlices(int type, const void* w, const float* x,
                  const std::int32_t* zeros, float* out, int M, int N, int K,
                  hipStream_t stream, bool multi_session = false) {
  for (int r = 0; r < N; r += 8) {
    const int n = std::min(8, N - r);
    Dev q;
    HIPCHECK(hipMalloc(&q.p, qfn_mmq_q8_1_bytes(n, K)));
    if (qfn_mmq_quantize_q8_1(x + static_cast<std::size_t>(r) * K, q.p, n, K,
                              stream) != 0 ||
        Project(type, w, q.p, zeros, out + static_cast<std::size_t>(r) * M, M,
                n, K, stream, multi_session) != 0)
      return 2;
    HIPCHECK(hipStreamSynchronize(stream));
  }
  return 0;
}

const core::GgmlType kSliceTypes[] = {
    core::GgmlType::kIQ4_XS, core::GgmlType::kIQ4_NL, core::GgmlType::kQ4_K,
    core::GgmlType::kQ5_K, core::GgmlType::kQ5_1};

// Operator fixtures, explicitly synthetic: random packed integer payloads and
// finite FP16 block scales. They cover types absent from the real twin file;
// they do not claim model-level numerical or text qualification.
int SliceTypes() {
  hipStream_t stream = nullptr;
  HIPCHECK(hipStreamCreate(&stream));
  Dev zero_ids;
  const std::int32_t zeros[8] = {};
  if (!Upload(zero_ids, zeros, sizeof zeros))
    return 2;
  int failures = 0;
  const bool wmma = gufo::models::qwen38_flash_next::Iq4WideVariant() == 4;
  for (const auto type : kSliceTypes) {
    const bool iq4 =
        type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ4_NL;
    const bool small_block =
        type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kQ5_1;
    // K=640 reproduces the IQ4_NL half-quantum tail in real HC-up weights.
    const int tail_k = small_block ? 96 : 256;
    const int shapes[] = {512, 2560, 10240, small_block ? 640 : 768, tail_k};
    for (const int K : shapes) {
      // The large ragged fixture selects sixteen rows/block for V2/V3;
      // NL K=96 also tests an odd packed block count / final half-word load.
      const int M = K == tail_k ? 1281 : 65;
      const std::size_t block_cols = small_block ? 32 : 256;
      const std::size_t rb = RowBytes(type, K);
      const std::size_t block_bytes = rb / (K / block_cols);
      std::mt19937 rng(71 + static_cast<int>(type) * 13 + K);
      std::vector<std::uint8_t> packed(rb * M);
      for (auto& b : packed)
        b = static_cast<std::uint8_t>(rng());
      for (std::size_t b = 0; b < packed.size(); b += block_bytes) {
        const std::uint16_t d = 0x1800;  // 2^-9: avoid saturated dot products
        std::memcpy(packed.data() + b, &d, sizeof d);
        if (type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
            type == core::GgmlType::kQ5_1) {
          const std::uint16_t offset = 0x1400;  // finite dmin / m
          std::memcpy(packed.data() + b + 2, &offset, sizeof offset);
        }
      }
      Dev w;
      if (!Upload(w, packed.data(), packed.size()))
        return 2;
      std::vector<float> weights(M * K);
      for (int m = 0; m < M; ++m)
        DequantRow(type, packed.data() + m * rb, weights.data() + m * K, K);
      for (int N = 1; N <= (wmma && iq4 ? 64 : 32); ++N) {
        std::normal_distribution<float> gauss(0.0F, 0.125F);
        std::vector<float> x(N * K);
        for (auto& v : x)
          v = gauss(rng);
        Dev dx, out, oneq, oneout;
        HIPCHECK(hipMalloc(&dx.p, x.size() * sizeof(float)));
        HIPCHECK(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        HIPCHECK(hipMalloc(&out.p, N * M * sizeof(float) + kTail));
        HIPCHECK(
            hipMemsetAsync(out.p, 0xA5, N * M * sizeof(float) + kTail, stream));
        if (ProjectSlices(int(type), w.p, static_cast<const float*>(dx.p),
                          static_cast<const std::int32_t*>(zero_ids.p),
                          static_cast<float*>(out.p), M, N, K, stream,
                          wmma && iq4) != 0)
          return 2;
        std::vector<float> sliced(N * M), alone(N * M), ref(N * M);
        HIPCHECK(hipMemcpy(sliced.data(), out.p, N * M * sizeof(float),
                           hipMemcpyDeviceToHost));
        std::vector<std::uint8_t> guard(kTail);
        HIPCHECK(hipMemcpy(
            guard.data(),
            static_cast<const std::uint8_t*>(out.p) + N * M * sizeof(float),
            guard.size(), hipMemcpyDeviceToHost));
        const bool guard_ok =
            std::none_of(guard.begin(), guard.end(),
                         [](std::uint8_t b) { return b != 0xA5; });
        failures += !guard_ok;
        const std::size_t qbytes = qfn_mmq_q8_1_bytes(1, K);
        HIPCHECK(hipMalloc(&oneq.p, qbytes));
        HIPCHECK(hipMalloc(&oneout.p, M * sizeof(float)));
        std::vector<std::uint8_t> qhost(qbytes);
        for (int n = 0; n < N; ++n) {
          if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p) + n * K,
                                    oneq.p, 1, K, stream) != 0 ||
              Project(int(type), w.p, oneq.p,
                      static_cast<const std::int32_t*>(zero_ids.p),
                      static_cast<float*>(oneout.p), M, 1, K, stream,
                      wmma && iq4) != 0)
            return 2;
          HIPCHECK(hipStreamSynchronize(stream));
          HIPCHECK(hipMemcpy(alone.data() + n * M, oneout.p, M * sizeof(float),
                             hipMemcpyDeviceToHost));
          HIPCHECK(
              hipMemcpy(qhost.data(), oneq.p, qbytes, hipMemcpyDeviceToHost));
          // Independent FP64 CPU dot of dequantized weights and Q8_1 inputs.
          for (int m = 0; m < M; ++m) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
              const auto* b = qhost.data() + (k / 32) * 36;
              const double a =
                  Half(b) *
                  static_cast<float>(static_cast<std::int8_t>(b[4 + k % 32]));
              sum += a * weights[m * K + k];
            }
            ref[n * M + m] = static_cast<float>(sum);
          }
        }
        std::size_t different = 0;
        for (std::size_t i = 0; i < alone.size(); ++i)
          different += std::memcmp(&alone[i], &sliced[i], sizeof(float)) != 0;
        const auto d = Diff(sliced, ref);
        const auto one = Diff(alone, ref);
        failures += different != 0 || d.rel_rms > 2e-3 || one.rel_rms > 2e-3;
        if (type == core::GgmlType::kIQ4_XS ||
            type == core::GgmlType::kIQ4_NL) {
          Dev batchq;
          HIPCHECK(hipMalloc(&batchq.p, qfn_mmq_q8_1_bytes(N, K)));
          if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p), batchq.p,
                                    N, K, stream) != 0)
            return 2;
          HIPCHECK(hipMemsetAsync(out.p, 0xA5, N * M * sizeof(float) + kTail,
                                  stream));
          if (qfn_mmq_iq4_dense_batch_preq(int(type), w.p, batchq.p,
                                           static_cast<float*>(out.p), M, N, K,
                                           stream) != 0)
            return 2;
          std::vector<float> wide(N * M);
          HIPCHECK(hipMemcpy(wide.data(), out.p, N * M * sizeof(float),
                             hipMemcpyDeviceToHost));
          HIPCHECK(hipMemcpy(
              guard.data(),
              static_cast<const std::uint8_t*>(out.p) + N * M * sizeof(float),
              guard.size(), hipMemcpyDeviceToHost));
          const bool wide_guard =
              std::none_of(guard.begin(), guard.end(),
                           [](std::uint8_t b) { return b != 0xA5; });
          std::size_t row_diff = 0, slice_diff = 0;
          for (std::size_t i = 0; i < wide.size(); ++i) {
            row_diff += std::memcmp(&wide[i], &alone[i], sizeof(float)) != 0;
            slice_diff += std::memcmp(&wide[i], &sliced[i], sizeof(float)) != 0;
          }
          failures += row_diff != 0 || slice_diff != 0 || !wide_guard;
          std::printf(
              "WIDE_SYNTHETIC_ROW_IDENTITY type=%s K=%d M=%d N=%d "
              "different=%zu/%zu sliced_different=%zu/%zu guard=%s\n",
              std::string(core::ToString(type)).c_str(), K, M, N, row_diff,
              wide.size(), slice_diff, wide.size(),
              wide_guard ? "PASS" : "FAIL");
        }
        std::printf(
            "SYNTHETIC_ROW_IDENTITY type=%s K=%d M=%d N=%d slices=%d "
            "last=%d different=%zu/%zu guard=%s cpu_relRMS=%.9e "
            "one_relRMS=%.9e\n",
            std::string(core::ToString(type)).c_str(), K, M, N, (N + 7) / 8,
            (N - 1) % 8 + 1, different, alone.size(),
            guard_ok ? "PASS" : "FAIL", d.rel_rms, one.rel_rms);
      }
    }
    std::printf("SYNTHETIC_TYPE_COVERAGE %s N=1..32 slices=8\n",
                std::string(core::ToString(type)).c_str());
  }
  HIPCHECK(hipStreamDestroy(stream));
  std::printf("slice-types: %s (%d failures)\n", failures ? "FAIL" : "PASS",
              failures);
  return failures ? 1 : 0;
}

struct WideVariantEnvironment {
  const char* original = std::getenv("GUFO_IQ4_WIDE_VARIANT");
  bool present = original != nullptr;
  std::string saved = present ? original : "";

  void Set(const char* value) {
#ifdef _WIN32
    _putenv_s("GUFO_IQ4_WIDE_VARIANT", value ? value : "");
#else
    if (value)
      setenv("GUFO_IQ4_WIDE_VARIANT", value, 1);
    else
      unsetenv("GUFO_IQ4_WIDE_VARIANT");
#endif
  }
  ~WideVariantEnvironment() { Set(present ? saved.c_str() : nullptr); }
};

// The resident pass deliberately covers only the 252 IQ4 tensors. Q4_K,
// Q5_K and Q5_1 serving slices are outside this measurement.
int TensorClass(std::size_t M, std::size_t K, core::GgmlType type) {
  if (type == core::GgmlType::kIQ4_XS && K == 2560) {
    if (M == 10240)
      return 0;
    if (M == 6144)
      return 1;
    if (M == 640)
      return 3;
  }
  if (type == core::GgmlType::kIQ4_XS && M == 2560 && K == 6144)
    return 2;
  if (type == core::GgmlType::kIQ4_NL && M == 2560 && K == 640)
    return 4;
  return -1;
}
constexpr const char* kClassNames[] = {"qkv", "gate", "ssm_out",
                                       "shexp_gate_up", "shexp_down"};

int PassBench(const core::GgufReader& trunk, const core::GgufReader& twin,
              bool batch) {
  (void)batch;  // Both modes now exercise the same resident serving dispatch.
  hipStream_t stream = nullptr;
  HIPCHECK(hipStreamCreate(&stream));
  struct T {
    std::string name;
    core::GgmlType ty4;
    std::size_t M, K;
    int cls;
    Dev w8, w4;
  };
  std::vector<std::unique_ptr<T>> ts;
  double bytes8 = 0, bytes4 = 0;
  std::size_t counts[5] = {};
  for (const auto& ti : twin.GetTensors()) {
    const auto* t8 = trunk.FindTensor(std::string(ti.name));
    if (t8 == nullptr || t8->type != core::GgmlType::kQ8_0 ||
        (ti.type != core::GgmlType::kIQ4_XS &&
         ti.type != core::GgmlType::kIQ4_NL))
      continue;
    auto x = std::make_unique<T>();
    x->name = std::string(ti.name);
    x->ty4 = ti.type;
    x->K = t8->dimensions[0];
    x->M = t8->dimensions[1];
    x->cls = TensorClass(x->M, x->K, x->ty4);
    if (x->cls < 0 || ti.dimensions[0] != x->K || ti.dimensions[1] != x->M) {
      std::fprintf(stderr, "INVALID pass tensor shape (%s)\n", x->name.c_str());
      return 2;
    }
    ++counts[x->cls];
    const auto b8 = RowBytes(core::GgmlType::kQ8_0, x->K) * x->M;
    const auto b4 = RowBytes(x->ty4, x->K) * x->M;
    if (!Upload(x->w8, t8->data, b8) || !Upload(x->w4, ti.data, b4))
      return 2;
    bytes8 += static_cast<double>(b8);
    bytes4 += static_cast<double>(b4);
    ts.push_back(std::move(x));
  }
  constexpr std::size_t expected[] = {36, 36, 36, 96, 48};
  if (ts.size() != 252 ||
      !std::equal(std::begin(counts), std::end(counts), std::begin(expected))) {
    std::fprintf(stderr,
                 "INVALID: resident pass requires all 252 IQ4 tensors\n");
    return 2;
  }
  std::map<std::size_t, std::unique_ptr<Dev>> act;
  std::mt19937 rng(7);
  std::normal_distribution<float> gauss(0.0F, 1.0F);
  for (const auto& x : ts) {
    if (act.count(x->K))
      continue;
    std::vector<float> h(64 * x->K);
    for (auto& v : h)
      v = gauss(rng);
    Dev dx;
    auto dq = std::make_unique<Dev>();
    if (!Upload(dx, h.data(), h.size() * sizeof(float)))
      return 2;
    HIPCHECK(hipMalloc(&dq->p, qfn_mmq_q8_1_bytes(64, x->K)));
    if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p), dq->p, 64, x->K,
                              stream) != 0)
      return 2;
    HIPCHECK(hipStreamSynchronize(stream));
    act[x->K] = std::move(dq);
  }
  Dev zero_ids, out;
  const std::int32_t zeros[8] = {};
  if (!Upload(zero_ids, zeros, sizeof zeros))
    return 2;
  HIPCHECK(hipMalloc(&out.p, std::size_t{10240} * 64 * sizeof(float)));
  std::printf(
      "pass bench: %zu tensors, Q8_0 %.3f GB, 4-bit %.3f GB "
      "(bytes ratio %.3f); IQ4 only; excludes Q4_K/Q5_K/Q5_1\n",
      ts.size(), bytes8 / 1e9, bytes4 / 1e9, bytes4 / bytes8);
  std::printf("WMMA_WAVES=%d\n",
              gufo::models::qwen38_flash_next::Iq4WmmaWaves());
  hipEvent_t e0 = nullptr, e1 = nullptr;
  HIPCHECK(hipEventCreate(&e0));
  HIPCHECK(hipEventCreate(&e1));
  WideVariantEnvironment variant_env;
  for (int variant : {0, 4}) {
    const char value[] = {char('0' + variant), '\0'};
    variant_env.Set(value);
    std::printf("WIDE_VARIANT_BEGIN %d\n", variant);
    for (int N = 1; N <= 64; ++N) {
      for (int cls = -1; cls < 5; ++cls) {
        std::vector<double> times[3];
        const auto measure = [&](int which, double* elapsed) -> int {
          HIPCHECK(hipStreamSynchronize(stream));
          HIPCHECK(hipEventRecord(e0, stream));
          int rc = 0;
          for (const auto& x : ts) {
            if (cls != -1 && x->cls != cls)
              continue;
            rc |= Project(which == 0 ? int(core::GgmlType::kQ8_0) : int(x->ty4),
                          which == 0 ? x->w8.p : x->w4.p, act[x->K]->p,
                          static_cast<const std::int32_t*>(zero_ids.p),
                          static_cast<float*>(out.p), x->M, N, x->K, stream,
                          which == 1);
          }
          HIPCHECK(hipEventRecord(e1, stream));
          HIPCHECK(hipEventSynchronize(e1));
          if (rc != 0 || hipGetLastError() != hipSuccess)
            return 2;
          float ms = 0;
          HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
          *elapsed = ms;
          return 0;
        };
        // Q8 and the selected batch variant always alternate strictly ABAB.
        for (int round = -1; round < 10; ++round) {
          for (int which = 0; which < 2; ++which) {
            double elapsed = 0;
            if (measure(which, &elapsed) != 0)
              return 2;
            if (round >= 0)
              times[which].push_back(elapsed);
          }
        }
        // Narrow is an observation after the registered two-path experiment.
        if (cls == -1 && N <= 8) {
          for (int round = -1; round < 10; ++round) {
            double elapsed = 0;
            if (measure(2, &elapsed) != 0)
              return 2;
            if (round >= 0)
              times[2].push_back(elapsed);
          }
        }
        for (auto& v : times)
          std::sort(v.begin(), v.end());
        const double m8 = (times[0][4] + times[0][5]) / 2;
        const double m4 = (times[1][4] + times[1][5]) / 2;
        if (cls == -1) {
          std::printf(
              "PASS N=%d | Q8_0 median %.6f ms (min %.6f max %.6f, "
              "%5.1f GB/s) | 4-bit median %.6f ms (min %.6f max %.6f, "
              "%5.1f GB/s) | 4-bit/Q8 time %.9f\n",
              N, m8, times[0].front(), times[0].back(), bytes8 / (m8 * 1e6), m4,
              times[1].front(), times[1].back(), bytes4 / (m4 * 1e6), m4 / m8);
          std::printf("BANDWIDTH N=%d IQ4_GBps=%.6f fraction_narrow_195=%.6f\n",
                      N, bytes4 / (m4 * 1e6), bytes4 / (m4 * 1e6) / 195.0);
          if (!times[2].empty()) {
            const double narrow = (times[2][4] + times[2][5]) / 2;
            std::printf(
                "NARROW_RATIO N=%d variant=%d narrow_ms=%.6f "
                "batch_over_narrow=%.9f gate=REPORT_ONLY\n",
                N, variant, narrow, m4 / narrow);
          }
        } else {
          std::printf(
              "CLASS_PASS variant=%d N=%d class=%s tensors=%zu "
              "Q8_ms=%.6f IQ4_ms=%.6f IQ4_over_Q8=%.9f\n",
              variant, N, kClassNames[cls], counts[cls], m8, m4, m4 / m8);
        }
      }
    }
    std::printf("WIDE_VARIANT_END %d\n", variant);
  }
  HIPCHECK(hipEventDestroy(e0));
  HIPCHECK(hipEventDestroy(e1));
  HIPCHECK(hipStreamDestroy(stream));
  std::printf("iq4_dense_launches=%llu\n", qfn_mmq_iq4_dense_launch_count());
  return 0;
}

// This mode compares WMMA only with WMMA for identity. Narrow is a separate
// closeness observation because its floating-point accumulation order differs.
int WmmaCase(const std::string& name, core::GgmlType type, const void* weights,
             int M, int K, bool closeness, hipStream_t stream) {
  Dev w, dx, out, slices, alone, narrow;
  if (!Upload(w, weights, RowBytes(type, K) * M))
    return 2;
  std::mt19937 rng(1234);
  std::normal_distribution<float> gauss(0.0F, 0.125F);
  std::vector<float> x(64 * K);
  for (auto& v : x)
    v = gauss(rng);
  HIPCHECK(hipMalloc(&dx.p, x.size() * sizeof(float)));
  HIPCHECK(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                          hipMemcpyHostToDevice, stream));
  const std::size_t maxbytes = std::size_t{64} * M * sizeof(float);
  HIPCHECK(hipMalloc(&out.p, maxbytes + kTail));
  HIPCHECK(hipMalloc(&slices.p, maxbytes + kTail));
  HIPCHECK(hipMalloc(&alone.p, M * sizeof(float)));
  HIPCHECK(hipMalloc(&narrow.p, maxbytes));
  Dev zero_ids;
  const std::int32_t zeros[8] = {};
  if (!Upload(zero_ids, zeros, sizeof zeros))
    return 2;
  int failures = 0;
  for (int N = 1; N <= (closeness ? 8 : 64); ++N) {
    Dev q;
    HIPCHECK(hipMalloc(&q.p, qfn_mmq_q8_1_bytes(N, K)));
    HIPCHECK(hipMemsetAsync(out.p, 0xA5, maxbytes + kTail, stream));
    if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p), q.p, N, K,
                              stream) != 0 ||
        Project(int(type), w.p, q.p, nullptr, static_cast<float*>(out.p), M, N,
                K, stream, true) != 0)
      return 2;
    HIPCHECK(hipStreamSynchronize(stream));
    std::vector<float> batch(N * M);
    HIPCHECK(hipMemcpy(batch.data(), out.p, batch.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
    std::vector<std::uint8_t> guard(kTail);
    HIPCHECK(hipMemcpy(
        guard.data(),
        static_cast<const std::uint8_t*>(out.p) + batch.size() * sizeof(float),
        guard.size(), hipMemcpyDeviceToHost));
    const bool guard_ok = std::all_of(guard.begin(), guard.end(),
                                      [](std::uint8_t b) { return b == 0xA5; });
    failures += !guard_ok;
    if (closeness) {
      if (Project(int(type), w.p, q.p,
                  static_cast<const std::int32_t*>(zero_ids.p),
                  static_cast<float*>(narrow.p), M, N, K, stream) != 0)
        return 2;
      HIPCHECK(hipStreamSynchronize(stream));
      std::vector<float> baseline(N * M);
      HIPCHECK(hipMemcpy(baseline.data(), narrow.p,
                         baseline.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
      for (int row = 0; row < N; ++row) {
        const std::vector<float> a(batch.begin() + row * M,
                                   batch.begin() + (row + 1) * M);
        const std::vector<float> b(baseline.begin() + row * M,
                                   baseline.begin() + (row + 1) * M);
        const auto d = Diff(a, b);
        failures += !std::isfinite(d.rel_rms);
        std::printf(
            "WMMA_NARROW_CLOSENESS %s N=%d row=%d maxabs=%.9e "
            "relRMS=%.9e guard=%s\n",
            name.c_str(), N, row, d.max_abs, d.rel_rms,
            guard_ok ? "PASS" : "FAIL");
      }
      continue;
    }
    HIPCHECK(hipMemsetAsync(slices.p, 0xA5, maxbytes + kTail, stream));
    if (ProjectSlices(int(type), w.p, static_cast<const float*>(dx.p), nullptr,
                      static_cast<float*>(slices.p), M, N, K, stream,
                      true) != 0)
      return 2;
    std::vector<float> sliced(N * M), individual(N * M);
    HIPCHECK(hipMemcpy(sliced.data(), slices.p, sliced.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(guard.data(),
                       static_cast<const std::uint8_t*>(slices.p) +
                           sliced.size() * sizeof(float),
                       guard.size(), hipMemcpyDeviceToHost));
    const bool slice_guard = std::all_of(
        guard.begin(), guard.end(), [](std::uint8_t b) { return b == 0xA5; });
    Dev oneq;
    HIPCHECK(hipMalloc(&oneq.p, qfn_mmq_q8_1_bytes(1, K)));
    for (int row = 0; row < N; ++row) {
      if (qfn_mmq_quantize_q8_1(static_cast<const float*>(dx.p) + row * K,
                                oneq.p, 1, K, stream) != 0 ||
          Project(int(type), w.p, oneq.p, nullptr, static_cast<float*>(alone.p),
                  M, 1, K, stream, true) != 0)
        return 2;
      HIPCHECK(hipMemcpyAsync(individual.data() + row * M, alone.p,
                              M * sizeof(float), hipMemcpyDeviceToHost,
                              stream));
      HIPCHECK(hipStreamSynchronize(stream));
    }
    std::size_t row_diff = 0, slice_diff = 0;
    for (std::size_t i = 0; i < batch.size(); ++i) {
      row_diff += std::memcmp(&batch[i], &individual[i], sizeof(float)) != 0;
      slice_diff += std::memcmp(&batch[i], &sliced[i], sizeof(float)) != 0;
      failures += !std::isfinite(batch[i]);
    }
    failures += row_diff != 0 || slice_diff != 0 || !slice_guard;
    std::printf("ROW_IDENTITY %s N=%d different=%zu/%zu guard=%s\n",
                name.c_str(), N, row_diff, batch.size(),
                guard_ok ? "PASS" : "FAIL");
    std::printf(
        "SLICED_ROW_IDENTITY %s type=%s N=%d slices=%d last=%d "
        "different=%zu/%zu guard=%s\n",
        name.c_str(), std::string(core::ToString(type)).c_str(), N, (N + 7) / 8,
        (N - 1) % 8 + 1, slice_diff, batch.size(),
        slice_guard ? "PASS" : "FAIL");
  }
  return failures ? 1 : 0;
}

int WmmaG1(const core::GgufReader& twin, bool closeness, bool all) {
  WideVariantEnvironment variant_env;
  variant_env.Set("4");
  hipStream_t stream = nullptr;
  HIPCHECK(hipStreamCreate(&stream));
  std::size_t coverage[5] = {};
  int failures = 0;
  for (const auto& t : twin.GetTensors()) {
    if (t.dimensions.size() != 2)
      continue;
    const int cls = TensorClass(t.dimensions[1], t.dimensions[0], t.type);
    if (cls < 0 || (!all && coverage[cls] != 0))
      continue;
    ++coverage[cls];
    const int rc =
        WmmaCase(std::string(t.name), t.type, t.data, t.dimensions[1],
                 t.dimensions[0], closeness, stream);
    if (rc == 2)
      return 2;
    failures += rc;
  }
  constexpr std::size_t expected[] = {36, 36, 36, 96, 48};
  for (int cls = 0; cls < 5; ++cls) {
    std::printf("WMMA_REAL_CLASS_COVERAGE class=%s tensors=%zu N=1..%d\n",
                kClassNames[cls], coverage[cls], closeness ? 8 : 64);
    failures += coverage[cls] == 0;
    if (all || closeness)
      failures += coverage[cls] != expected[cls];
  }
  // Synthetic alignment, K-block and output-row tails are operator fixtures.
  if (!closeness) {
    struct Shape {
      core::GgmlType type;
      int M, K;
    };
    for (const auto shape : {Shape{core::GgmlType::kIQ4_XS, 65, 2560},
                             Shape{core::GgmlType::kIQ4_XS, 1281, 768},
                             Shape{core::GgmlType::kIQ4_NL, 65, 640},
                             Shape{core::GgmlType::kIQ4_NL, 1281, 96}}) {
      std::mt19937 rng(71 + shape.K);
      const auto bytes = RowBytes(shape.type, shape.K) * shape.M;
      std::vector<std::uint8_t> packed(bytes);
      for (auto& b : packed)
        b = static_cast<std::uint8_t>(rng());
      const int block_bytes = shape.type == core::GgmlType::kIQ4_XS ? 136 : 18;
      for (std::size_t b = 0; b < bytes; b += block_bytes) {
        const std::uint16_t d = 0x1800;
        std::memcpy(packed.data() + b, &d, sizeof d);
      }
      const auto name = "synthetic_" + std::string(core::ToString(shape.type)) +
                        "_M" + std::to_string(shape.M) + "_K" +
                        std::to_string(shape.K);
      const int rc = WmmaCase(name, shape.type, packed.data(), shape.M, shape.K,
                              false, stream);
      if (rc == 2)
        return 2;
      failures += rc;
    }
  }
  HIPCHECK(hipStreamDestroy(stream));
  std::printf("WMMA_G1 mode=%s %s failures=%d\n",
              closeness ? "closeness" : "identity", failures ? "FAIL" : "PASS",
              failures);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("IQ4_WIDE_VARIANT=%d\n",
              gufo::models::qwen38_flash_next::Iq4WideVariant());
  if (argc == 2 && std::string(argv[1]) == "--slice-types")
    return SliceTypes();
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: dense_twin_gpu_test TRUNK_SHARD1.gguf TWIN.gguf "
                 "[tensor-name ...] | --slice-types\n");
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
  if (argc == 4 && (std::string(argv[3]) == "--wmma-identity" ||
                    std::string(argv[3]) == "--wmma-identity-all" ||
                    std::string(argv[3]) == "--wmma-closeness"))
    return WmmaG1(*twin, std::string(argv[3]) == "--wmma-closeness",
                  std::string(argv[3]) != "--wmma-identity");
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
  std::map<core::GgmlType, std::size_t> real_type_coverage;
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
    if (rb4 == 0) {
      std::fprintf(stderr, "%s: unsupported twin type\n", name.c_str());
      ++failures;
      continue;
    }
    ++real_type_coverage[ty4];
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
      Dev sliceout;
      HIPCHECK(hipMalloc(&sliceout.p, N * M * sizeof(float) + kTail));
      HIPCHECK(hipMemsetAsync(sliceout.p, 0xA5, N * M * sizeof(float) + kTail,
                              stream));
      if (ProjectSlices(
              int(ty4), w4.p, static_cast<const float*>(dx.p),
              static_cast<const std::int32_t*>(zero_ids.p),
              static_cast<float*>(sliceout.p), M, N, K, stream,
              iq4 && gufo::models::qwen38_flash_next::Iq4WideVariant() == 4) !=
          0)
        return 2;
      std::vector<float> sliced(N * M);
      HIPCHECK(hipMemcpy(sliced.data(), sliceout.p, N * M * sizeof(float),
                         hipMemcpyDeviceToHost));
      HIPCHECK(hipMemcpy(
          output_guard.data(),
          static_cast<const std::uint8_t*>(sliceout.p) + N * M * sizeof(float),
          output_guard.size(), hipMemcpyDeviceToHost));
      if (std::any_of(output_guard.begin(), output_guard.end(),
                      [](std::uint8_t b) { return b != 0xA5; })) {
        std::printf("FAIL SLICED_OUTPUT_GUARD %s N=%d\n", name.c_str(), N);
        ++failures;
      }
      std::vector<float> candidate = k4;
      if (iq4) {
        // Exercise the wide adapter even at N<=8: it dispatches the unchanged
        // narrow kernel for the final chunk of 33..40-row batches. Tight input
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
            Project(int(ty4), w4.p, oneq.p,
                    static_cast<const std::int32_t*>(zero_ids.p),
                    static_cast<float*>(oneout.p), M, 1, K, stream,
                    iq4 && gufo::models::qwen38_flash_next::Iq4WideVariant() ==
                               4) != 0)
          return 2;
        HIPCHECK(hipMemcpyAsync(alone.data() + static_cast<std::size_t>(t) * M,
                                oneout.p, M * sizeof(float),
                                hipMemcpyDeviceToHost, stream));
        HIPCHECK(hipStreamSynchronize(stream));
      }
      std::size_t row_differences = 0;
      std::size_t slice_differences = 0;
      for (std::size_t i = 0; i < alone.size(); ++i)
        row_differences +=
            std::memcmp(&alone[i], &candidate[i], sizeof(float)) != 0;
      for (std::size_t i = 0; i < alone.size(); ++i)
        slice_differences +=
            std::memcmp(&candidate[i], &sliced[i], sizeof(float)) != 0;
      std::printf("ROW_IDENTITY %s N=%d different=%zu/%zu\n", name.c_str(), N,
                  row_differences, alone.size());
      std::printf(
          "SLICED_ROW_IDENTITY %s type=%s N=%d slices=%d last=%d "
          "different=%zu/%zu\n",
          name.c_str(), std::string(core::ToString(ty4)).c_str(), N,
          (N + 7) / 8, (N - 1) % 8 + 1, slice_differences, alone.size());
      // All five types, every N and every row: no tolerance on byte identity.
      failures += row_differences != 0 || slice_differences != 0;
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
      const auto slice_ref = Diff(sliced, ref4);
      std::printf("SLICED_CPU_REF %s N=%d maxabs=%.9e relRMS=%.9e\n",
                  name.c_str(), N, slice_ref.max_abs, slice_ref.rel_rms);
      failures += slice_ref.rel_rms > 2e-3;
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
  for (const auto type : kSliceTypes) {
    const auto count = real_type_coverage[type];
    std::printf(
        "REAL_TYPE_COVERAGE %s tensors=%zu N=1..32 %s\n",
        std::string(core::ToString(type)).c_str(), count,
        count ? "covered" : "GAP: absent; require synthetic --slice-types");
  }
  std::printf("iq4_dense_launches=%llu\n", qfn_mmq_iq4_dense_launch_count());
  std::printf("dense_twin_gpu_test: %s (%d failures)\n",
              failures ? "FAIL" : "all kernels agree with their CPU references",
              failures);
  return failures ? 1 : 0;
}
