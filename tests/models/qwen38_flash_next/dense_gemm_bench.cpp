// Times the dense F16 prefill projections on synthetic weights
// (diagnostic): the Q8_0 2560 x 6144 output projection, the 2560 x 640
// shared-expert down projection, the fused SSM projection (16384 x 2560 with
// its convolution), the stacked attention projection (13312 x 2560 with norms
// and RoPE), and the F16 router (513 x 2560) and recurrent gates (96 x 2560).
//
// dense_gemm_bench [tokens] [reps]
//
// Each line prints the median launch time, TFLOPS and an FNV-1a hash of the
// output, so launch-order changes can be checked for identical results.
//
// dense_gemm_bench --prefill-native [N list, default 4096,100] [reps, default 20]
//
// Prefill trunk GEMM on native-quant weights (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S,
// IQ4_NL) against the Q8_0 W8A8 route, plus the BF16/F16 hyper-connection
// GEMMs; see PrefillNative for the columns.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dense_half_kquant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/native_dispatch.hpp"

// The Qwen 27B lane's tiled-Q8_1 quantiser (with the activation-sum sidecar the
// affine K-quant types read) and its decode-once int8-WMMA GEMMs. They live in
// gufo_core (src/models/qwen/hip/kernels/prefill_quant_gemm.hip) and are
// redeclared here (same signatures as src/models/qwen/hip/ops/gemm.hpp).
namespace gufo::hip {
void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream);
std::size_t QuantizedActivationBytes(std::size_t batch, std::size_t k);
void LaunchKQuantWmmaGEMM(core::GgmlType type, const void* w,
                          const void* q8_1_x, float* y, std::size_t batch,
                          std::size_t m, std::size_t k, hipStream_t stream);
void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream);
}  // namespace gufo::hip

// F16 dequantiser of the candidate e (dense_gemm_bench_dequant.hip.cpp).
namespace gufo::bench {
bool DequantizeToHalf(core::GgmlType type, const void* w, void* out,
                      std::size_t rows, std::size_t cols, hipStream_t stream);
}

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace g4 = gufo::models::gemma4::rocm;
namespace core = gufo::core;
namespace quant = gufo::quant;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

void* Zeros(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

// Q8_0 rows: per 32 codes a F16 scale near 1/256 and random codes.
std::vector<std::uint8_t> Q8Weights(std::size_t rows, std::size_t k,
                                    std::uint32_t seed) {
  const std::size_t blocks = rows * (k / 32);
  std::vector<std::uint8_t> w(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d = __float2half(
        (1.0F + static_cast<float>(Next(&seed) % 64) / 64.0F) / 256.0F);
    std::memcpy(&w[b * 34], &d, 2);
    for (int i = 0; i < 32; ++i)
      w[b * 34 + 2 + i] = static_cast<std::uint8_t>(Next(&seed));
  }
  return w;
}

std::vector<__half> Activations(std::size_t count, std::uint32_t seed) {
  std::vector<__half> x(count);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  return x;
}

std::uint64_t Hash(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  CheckHip(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

double MedianMs(const std::function<void()>& launch, int reps,
                int warmup = 2) {
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (int i = 0; i < warmup; ++i)
    launch();
  CheckHip(hipDeviceSynchronize(), "warm-up launch");
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
    launch();
    CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
    CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
    CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
  }
  CheckHip(hipEventDestroy(start), "hipEventDestroy");
  CheckHip(hipEventDestroy(stop), "hipEventDestroy");
  std::sort(ms.begin(), ms.end());
  return ms[reps / 2];
}

// ---- --prefill-native ------------------------------------------------------

// MedianMs with the bench's three warm-ups and a launch-error check: the
// launchers it times return void, so a failed launch shows only in
// hipGetLastError.
double TimedMs(const std::function<void()>& launch, int reps) {
  (void)hipGetLastError();
  const double ms = MedianMs(launch, reps, 3);
  CheckHip(hipGetLastError(), "launch");
  return ms;
}

// Device allocation freed on scope exit, so one shape's buffers never outlive
// it even when a section throws.
class DevBuf {
public:
  static constexpr std::size_t kPad = 64 * 1024;
  explicit DevBuf(std::size_t bytes) : bytes_(bytes) {
    CheckHip(hipMalloc(&p_, bytes ? bytes : 1), "hipMalloc");
    CheckHip(hipMemset(p_, 0, bytes ? bytes : 1), "hipMemset");
  }
  // Uploaded data is followed by kPad zero bytes, so a tile loader that steps
  // a few blocks past the last row (mmq IQ4_NL, 8-block step) stays inside the
  // allocation.
  template<typename T>
  DevBuf(const std::vector<T>& host)
      : DevBuf(host.size() * sizeof(T) + kPad) {
    CheckHip(hipMemcpy(p_, host.data(), host.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "hipMemcpy");
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { (void)hipFree(p_); }
  // Sets every byte; 0xFF makes unwritten F32 outputs read back as NaN.
  void Fill(int byte) const {
    CheckHip(hipMemset(p_, byte, bytes_ ? bytes_ : 1), "hipMemset");
  }
  template<typename T = void>
  T* get() const {
    return static_cast<T*>(p_);
  }

private:
  void* p_{nullptr};
  std::size_t bytes_{0};
};

std::uint16_t Fp16Bits(float v) {
  const __half h = __float2half(v);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, 2);
  return bits;
}

std::uint16_t Bf16Bits(float f) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, 4);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                    16);
}

// Uniform in [-1, 1].
std::vector<float> RandomFloats(std::size_t count, std::uint32_t seed,
                                float scale) {
  std::vector<float> v(count);
  for (auto& e : v)
    e = scale *
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F;
  return v;
}

struct NativeType {
  const char* name;
  core::GgmlType type;
  void (*dequant)(const void*, float*, std::size_t);
  std::size_t block_bytes;
  std::size_t d_offset;     // fp16 super-block scale
  std::size_t dmin_offset;  // fp16 minimum scale, or SIZE_MAX
  float d_scale;            // d in [d_scale, 2 d_scale)
};

constexpr std::size_t kNoOffset = static_cast<std::size_t>(-1);

// d_scale keeps decoded weights near 0.05-0.3 in magnitude for each format's
// code and sub-scale range, so the F32 outputs stay finite and well scaled.
const NativeType kNativeTypes[] = {
    {"Q4_K", core::GgmlType::kQ4_K, quant::DequantizeQ4_K, 144, 0, 2,
     1.0F / 16384.0F},
    {"Q5_K", core::GgmlType::kQ5_K, quant::DequantizeQ5_K, 176, 0, 2,
     1.0F / 32768.0F},
    {"Q6_K", core::GgmlType::kQ6_K, quant::DequantizeQ6_K, 210, 208, kNoOffset,
     1.0F / 65536.0F},
    {"IQ4_XS", core::GgmlType::kIQ4_XS, quant::DequantizeIQ4_XS, 136, 0,
     kNoOffset, 1.0F / 65536.0F},
    {"IQ3_S", core::GgmlType::kIQ3_S, quant::DequantizeIQ3_S, 110, 0,
     kNoOffset, 1.0F / 4096.0F},
    {"IQ4_NL", core::GgmlType::kIQ4_NL, quant::DequantizeIQ4_NL, 18, 0,
     kNoOffset, 1.0F / 1024.0F},
};

// Random quant bits with valid (finite, small) fp16 scales. Random bytes would
// give NaN/inf scales.
std::vector<std::uint8_t> NativeWeights(const NativeType& t, std::size_t rows,
                                        std::size_t cols,
                                        std::uint32_t seed) {
  const std::size_t row_bytes = quant::QuantizedRowBytes(t.type, cols);
  if (row_bytes == 0 || row_bytes % t.block_bytes != 0)
    throw std::runtime_error(std::string("row bytes for ") + t.name);
  std::vector<std::uint8_t> w(rows * row_bytes);
  for (auto& b : w)
    b = static_cast<std::uint8_t>(Next(&seed) >> 7);
  for (std::size_t off = 0; off < w.size(); off += t.block_bytes) {
    const auto scale = [&] {
      return t.d_scale *
             (1.0F + static_cast<float>(Next(&seed) % 64) / 64.0F);
    };
    const std::uint16_t d = Fp16Bits(scale());
    std::memcpy(&w[off + t.d_offset], &d, 2);
    if (t.dmin_offset != kNoOffset) {
      const std::uint16_t dmin = Fp16Bits(scale());
      std::memcpy(&w[off + t.dmin_offset], &dmin, 2);
    }
  }
  return w;
}

struct Diff {
  double max_abs{0};
  double mean_abs{0};
  double rms_ref{0};
  std::size_t nonfinite_ref{0};
  std::size_t nonfinite_test{0};
  [[nodiscard]] double Rel() const {
    if (std::isinf(max_abs))
      return INFINITY;
    return rms_ref > 0 ? max_abs / rms_ref : 0;
  }
  [[nodiscard]] double MeanRel() const {
    return rms_ref > 0 ? mean_abs / rms_ref : 0;
  }
};

// Compares two device F32 buffers in chunks. Non-finite values are counted and
// left out of the error statistics.
Diff CompareDevice(const float* ref, const float* test, std::size_t count) {
  Diff d;
  constexpr std::size_t kChunk = std::size_t{1} << 22;
  std::vector<float> a(std::min(kChunk, count));
  std::vector<float> b(a.size());
  double sum_abs = 0;
  double sum_sq = 0;
  std::size_t finite = 0;
  for (std::size_t off = 0; off < count; off += kChunk) {
    const std::size_t n = std::min(kChunk, count - off);
    CheckHip(hipMemcpy(a.data(), ref + off, n * 4, hipMemcpyDeviceToHost),
             "hipMemcpy");
    CheckHip(hipMemcpy(b.data(), test + off, n * 4, hipMemcpyDeviceToHost),
             "hipMemcpy");
    for (std::size_t i = 0; i < n; ++i) {
      const bool fa = std::isfinite(a[i]);
      const bool fb = std::isfinite(b[i]);
      d.nonfinite_ref += fa ? 0 : 1;
      d.nonfinite_test += fb ? 0 : 1;
      if (!fa || !fb) {
        // A pair with a non-finite side is an infinite error, not a skipped
        // one: an all-NaN output must not read as 0.
        d.max_abs = INFINITY;
        continue;
      }
      const double e = std::abs(double{a[i]} - double{b[i]});
      d.max_abs = std::max(d.max_abs, e);
      sum_abs += e;
      sum_sq += double{a[i]} * double{a[i]};
      ++finite;
    }
  }
  if (finite) {
    d.mean_abs = sum_abs / static_cast<double>(finite);
    d.rms_ref = std::sqrt(sum_sq / static_cast<double>(finite));
  }
  return d;
}

// Non-finite values among the first `count` floats of a device buffer that was
// filled with 0xFF before the launch: unwritten elements count here.
std::size_t CountNonFinite(const float* device, std::size_t count) {
  constexpr std::size_t kChunk = std::size_t{1} << 22;
  std::vector<float> a(std::min(kChunk, count));
  std::size_t bad = 0;
  for (std::size_t off = 0; off < count; off += kChunk) {
    const std::size_t n = std::min(kChunk, count - off);
    CheckHip(hipMemcpy(a.data(), device + off, n * 4, hipMemcpyDeviceToHost),
             "hipMemcpy");
    for (std::size_t i = 0; i < n; ++i)
      bad += std::isfinite(a[i]) ? 0 : 1;
  }
  return bad;
}

// The activation as the Q8_1 kernels see it: per 32-wide block absmax / 127,
// codes round(v / d), the same recipe as the device quantisers.
std::vector<double> Q8Dequantized(const float* x, std::size_t k) {
  std::vector<double> out(k);
  for (std::size_t b = 0; b < k / 32; ++b) {
    float max_abs = 0;
    for (std::size_t i = 0; i < 32; ++i)
      max_abs = std::max(max_abs, std::abs(x[b * 32 + i]));
    const float d = max_abs / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    for (std::size_t i = 0; i < 32; ++i) {
      const auto qv = static_cast<std::int8_t>(std::round(x[b * 32 + i] * id));
      out[b * 32 + i] = double{d} * static_cast<double>(qv);
    }
  }
  return out;
}

struct Ref64 {
  double max_f32{0};  // max |kernel - fp64(F32 activations)|
  double max_q8{0};   // max |kernel - fp64(Q8-quantised activations)|
  double rms{0};      // RMS of the fp64 (F32 activation) references
};

// FP64 reference for 64 output elements per kernel output: dequantised weight
// row (CPU dequantiser) dot the activation row, in double.
struct SampleRefs {
  std::vector<std::size_t> token;
  std::vector<std::size_t> row;
  std::vector<double> f32;
  std::vector<double> q8;
  double rms{0};
};

SampleRefs MakeSampleRefs(const NativeType& t, const std::uint8_t* w_host,
                          const float* x_host, std::size_t n,
                          std::size_t rows, std::size_t cols,
                          std::uint32_t seed) {
  SampleRefs s;
  const std::size_t row_bytes = quant::QuantizedRowBytes(t.type, cols);
  std::vector<float> wrow(cols);
  double sum_sq = 0;
  for (int i = 0; i < 64; ++i) {
    const std::size_t tok = Next(&seed) % n;
    const std::size_t row = Next(&seed) % rows;
    t.dequant(w_host + row * row_bytes, wrow.data(), cols);
    const float* x = x_host + tok * cols;
    const std::vector<double> xq = Q8Dequantized(x, cols);
    double f = 0;
    double g = 0;
    for (std::size_t c = 0; c < cols; ++c) {
      f += double{wrow[c]} * double{x[c]};
      g += double{wrow[c]} * xq[c];
    }
    s.token.push_back(tok);
    s.row.push_back(row);
    s.f32.push_back(f);
    s.q8.push_back(g);
    sum_sq += f * f;
  }
  s.rms = std::sqrt(sum_sq / 64.0);
  return s;
}

Ref64 CompareSamples(const SampleRefs& s, const float* out_device,
                     std::size_t rows) {
  Ref64 r;
  r.rms = s.rms;
  for (std::size_t i = 0; i < s.token.size(); ++i) {
    float v = 0;
    CheckHip(hipMemcpy(&v, out_device + s.token[i] * rows + s.row[i], 4,
                       hipMemcpyDeviceToHost),
             "hipMemcpy");
    if (!std::isfinite(v)) {
      r.max_f32 = r.max_q8 = INFINITY;
      continue;
    }
    r.max_f32 = std::max(r.max_f32, std::abs(double{v} - s.f32[i]));
    r.max_q8 = std::max(r.max_q8, std::abs(double{v} - s.q8[i]));
  }
  return r;
}

// Bit-for-bit comparison of the first `count` floats of two device buffers.
// Returns the index of the first difference, or count when they are equal.
std::size_t FirstDifference(const float* a, const float* b,
                            std::size_t count) {
  constexpr std::size_t kChunk = std::size_t{1} << 22;
  std::vector<float> x(std::min(kChunk, count));
  std::vector<float> y(x.size());
  for (std::size_t off = 0; off < count; off += kChunk) {
    const std::size_t n = std::min(kChunk, count - off);
    CheckHip(hipMemcpy(x.data(), a + off, n * 4, hipMemcpyDeviceToHost),
             "hipMemcpy");
    CheckHip(hipMemcpy(y.data(), b + off, n * 4, hipMemcpyDeviceToHost),
             "hipMemcpy");
    for (std::size_t i = 0; i < n; ++i)
      if (std::memcmp(&x[i], &y[i], 4) != 0)
        return off + i;
  }
  return count;
}

// The condition under which TryLaunchQuantPrefillWave64
// (prefill_quant_wave64.hip) takes a launch, copied here only to label the
// b64 line; the launch itself goes through the library entry point.
bool TakesWave64(core::GgmlType type, std::size_t n, std::size_t m,
                 std::size_t k) {
  if (n < 96 || m < 1024 || k < 1024 || k % 256 != 0)
    return false;
  switch (type) {
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
      return true;
    case core::GgmlType::kIQ4_XS:
      return m <= k;
    default:
      return false;
  }
}

// The Gemma 4 dense binary16 GEMM's formats for the types of this bench
// (Gemma 4's DenseHalfFormat also maps Q4_0 and Q5_0, which this bench does
// not run).
std::optional<g4::ExpertFormat> HalfFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return g4::ExpertFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return g4::ExpertFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return g4::ExpertFormat::kQ6_K;
    default:
      return std::nullopt;
  }
}

// The Flash-Next launcher's formats (the engine maps the same three types).
std::optional<q::DenseHalfFormat> FlashHalfFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return q::DenseHalfFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return q::DenseHalfFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return q::DenseHalfFormat::kQ6_K;
    default:
      return std::nullopt;
  }
}

// ms columns: a_ms, b_ms, b64_ms, c_ms and prod_ms are end-to-end medians (one
// event window around quantise + GEMM); bq_ms, bg_ms and b64g_ms are the
// separately timed parts.
struct SummaryRow {
  std::string shape;
  std::string type;
  int n{0};
  double a_ms{-1};
  double b_ms{-1};
  double bq_ms{-1};
  double bg_ms{-1};
  double b64_ms{-1};
  double b64g_ms{-1};
  double c_ms{-1};
  double prod_ms{-1};
  double rel_ba{-1};      // max |b - a| / rms(a)
  double b_f64_f32{-1};   // b vs FP64 on 64 samples, F32 activations, / rms
  double b_f64_q8{-1};    // same, Q8-quantised activations
  const char* rowind{"-"};
  // d: Gemma 4's dense binary16-activation K-quant GEMM (NarrowActivations +
  // LaunchDenseHalfGemm). d_ms is one window around both; dn_ms and dg_ms are
  // the parts. -1 = not run (n/a).
  double d_ms{-1};
  double dn_ms{-1};
  double dg_ms{-1};
  double rel_da{-1};      // max |d - a| / rms(a)
  double rel_db{-1};      // max |d - b| / rms(b)
  double d_f64_f32{-1};   // d vs FP64 on 64 samples, F32 activations, / rms
  double d_f64_q8{-1};    // same, Q8-quantised activations
  const char* rowind_d{"-"};
  // f: Flash-Next's DenseHalfKQuantGemm (the engine's value 2 route), one
  // window around NarrowActivations + the launcher. bits_f compares its output
  // with d's at N and the widths of the f block; -1 / "-" = not run.
  double f_ms{-1};
  double fg_ms{-1};
  double f_f64_f32{-1};
  const char* bits_f{"-"};
  const char* rowind_f{"-"};
  // e: dequantise the whole weight to an F16 scratch + NarrowActivations +
  // BlasLt::Gemm (F16). e_* is the unpadded GEMM (m = rows); ep_* the best of
  // the padded variants (rows rounded up to 128 / 256 with zero weight rows),
  // -1 where padding changes nothing. One event window each.
  double e_ms{-1};
  double e_deq_ms{-1};
  double e_nar_ms{-1};
  double e_gemm_ms{-1};
  double rel_ea{-1};      // max |e - a| / rms(a)
  double rel_eb{-1};      // max |e - b| / rms(b)
  double e_f64_f32{-1};
  double e_f64_q8{-1};
  const char* rowind_e{"-"};
  double ep_ms{-1};
  std::size_t ep_rows{0};
};

// Median launch times of the Q8_0 W8A8 route (and the F16 route production
// prefers for some shapes) on one shape and one token count. total and
// f16_total are one event window around the whole pipeline; quant/gemm and
// f16_narrow/f16_gemm are the parts timed on their own.
struct Q8Times {
  double quant{-1};
  double gemm{-1};
  double total{-1};
  double Total() const { return total; }
  double f16_narrow{-1};
  double f16_gemm{-1};
  double f16_total{-1};
  double F16Total() const { return f16_total; }
  std::size_t nonfinite{0};      // of the W8A8 output, buffer pre-filled 0xFF
  std::size_t f16_nonfinite{0};  // same for the F16 route
};

constexpr double kMiB = 1024.0 * 1024.0;

void PrintHeader(int reps, const std::vector<int>& ns) {
  std::printf(
      "dense_gemm_bench --prefill-native  (median of %d launches, 3 warm-ups, "
      "one stream, nothing evicted)\n",
      reps);
  std::printf("token counts:");
  for (int n : ns)
    std::printf(" %d", n);
  std::printf(
      "\n"
      "Columns, per kernel line:\n"
      "  ms      median time. 'total' = everything the engine would launch for "
      "the projection (quantise, then GEMM), timed in ONE event window; the "
      "'quantise' and 'gemm' parts are timed separately as a breakdown, so "
      "their sum can differ slightly from the total.\n"
      "  TOPS    2*rows*cols*N / time, on total time (b also on GEMM-only "
      "time).\n"
      "  wGB/s   weight bytes / time, on the same time as TOPS.\n"
      "  x c     throughput ratio to the Q8_0 W8A8 route on the same shape and "
      "N: c_total_ms / ms. Above 1 = faster than Q8_0.\n"
      "  x prod  same, against the route production prefill takes for the "
      "Q8_0 trunk (c2, the F16 route, when rows >= 2048 and cols <= 2560).\n"
      "Kernels:\n"
      "  a   qfn_mmq_k_dense (current engine path for native types): "
      "quantise + MMQ GEMM + output sanitise in one call.\n"
      "  b   Q8_1 tiled quantiser with activation-sum sidecar "
      "(LaunchQuantizeActivationQ8_1FromFp32) + LaunchKQuantWmmaGEMM "
      "(decode-once int8 WMMA, wave32 128x128 tile).\n"
      "  b64 LaunchBatchedQuantGEMMPreQuantized on the same quantised input: "
      "the 27B lane's shipped dispatch (wave64 kernels where it takes them, "
      "else the same kernel as b).\n"
      "  d   NarrowActivations (F32 -> F16) + g4::LaunchDenseHalfGemm, the "
      "Gemma 4 lane's dense K-quant GEMM with binary16 activations (weights "
      "decoded to binary16 in LDS, F16 WMMA, 128 rows x 96 tokens per block), "
      "Q4_K / Q5_K / Q6_K only; one event window = total, parts as a "
      "breakdown. Shown with x c and x prod like the others; errors against a "
      "and b are max abs / rms of the reference.\n"
      "  e   dequantise the whole weight to an F16 scratch on every call "
      "(bench-local kernel over the Qwen lane's DecodeQuantSub16; value "
      "rounded once to binary16) + NarrowActivations + BlasLt::Gemm with "
      "HIP_R_16F, Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_S; one event window = "
      "total, parts dequant + narrow + gemm. 'e unpadded' runs m = rows; "
      "'e padded' rounds the row count up to a multiple of 128 / 256 with "
      "zero weight rows (shown only when it differs). 'x b64' on the parts "
      "line is b64_total / e_total. Scratch = rows x cols x 2 bytes.\n"
      "  c   QuantizeQ8Tiled + W8A8Gemm on Q8_0 weights of the same shape "
      "(what production's trunk runs).\n"
      "  c2  NarrowActivations (F32 -> F16) + DenseF16Gemm on the same Q8_0 "
      "weights (production's F16 route).\n"
      "Errors (b vs a, over the whole output): max abs, mean abs, and both "
      "relative to the RMS of a's output.\n"
      "f64 columns: 64 random output elements recomputed on the host in double "
      "from the CPU-dequantised weights. 'F32 act' uses the F32 activations "
      "(error includes activation quantisation); 'Q8 act' uses the "
      "Q8_1-quantised activations the kernels see. Max abs error over the 64, "
      "and that error relative to the RMS of the 64 references.\n"
      "Row independence: b at N (rows 0..99) must equal b at N = 100 on the "
      "first 100 tokens, bit for bit.\n"
      "Caveats: (a) includes qfn_mmq's output sanitise pass (a launch over the "
      "whole output) that b and c do not run. N = 100 numbers are warm-cache: "
      "the weights (19-29 MB for the small types) stay in the 32 MB Infinity "
      "Cache between launches, so small-N weight GB/s can exceed DRAM "
      "bandwidth. Output buffers are filled with 0xFF before each run, so a "
      "kernel that writes nothing shows as NaN (non-finite counts, inf "
      "errors).\n\n");
}

void TimeQ8Reference(const std::vector<int>& ns, int max_n, std::size_t rows,
                     std::size_t cols, int reps,
                     std::vector<Q8Times>* times) {
  times->assign(ns.size(), {});
  if (cols % 32 != 0)
    return;
  const std::vector<std::uint8_t> w_host = Q8Weights(rows, cols, 7U + rows);
  const std::vector<float> x_host = RandomFloats(
      std::size_t{static_cast<unsigned>(max_n)} * cols, 1234U, 1.0F);
  DevBuf w(w_host);
  DevBuf x(x_host);
  DevBuf xq(q::Q8TiledBytes(max_n, cols));
  DevBuf xh(std::size_t{static_cast<unsigned>(max_n)} * cols * 2);
  DevBuf out(std::size_t{static_cast<unsigned>(max_n)} * rows * 4);
  const bool f16_route = rows >= 2048 && cols <= 2560;
  for (std::size_t i = 0; i < ns.size(); ++i) {
    const std::size_t n = ns[i];
    Q8Times& t = (*times)[i];
    const auto quantise = [&] {
      q::QuantizeQ8Tiled(x.get<float>(), xq.get(), n, cols, nullptr);
    };
    bool ok = true;
    const auto gemm = [&] {
      ok = q::W8A8Gemm(w.get(), xq.get(), out.get<float>(), n, rows, cols,
                       nullptr) &&
           ok;
    };
    // Unwritten outputs must read back as NaN.
    out.Fill(0xFF);
    t.total = TimedMs(
        [&] {
          quantise();
          gemm();
        },
        reps);
    t.nonfinite = CountNonFinite(out.get<float>(), n * rows);
    t.quant = TimedMs(quantise, reps);
    t.gemm = TimedMs(gemm, reps);
    if (!ok)
      t.quant = t.gemm = t.total = -1;
    if (f16_route) {
      const auto narrow = [&] {
        q::NarrowActivations(x.get<float>(), xh.get(), false, n * cols,
                             nullptr);
      };
      bool f16_ok = true;
      const auto f16_gemm = [&] {
        f16_ok = q::DenseF16Gemm(w.get(), xh.get<__half>(), out.get<float>(),
                                 n, rows, cols, nullptr) &&
                 f16_ok;
      };
      out.Fill(0xFF);
      t.f16_total = TimedMs(
          [&] {
            narrow();
            f16_gemm();
          },
          reps);
      t.f16_nonfinite = CountNonFinite(out.get<float>(), n * rows);
      t.f16_narrow = TimedMs(narrow, reps);
      t.f16_gemm = TimedMs(f16_gemm, reps);
      if (!f16_ok)
        t.f16_narrow = t.f16_gemm = t.f16_total = -1;
    }
  }
}

void PrintKernelLine(const char* tag, const char* what, double ms, double gemm_ms,
                     std::size_t rows, std::size_t cols, std::size_t n,
                     double weight_bytes, double c_ms, double prod_ms) {
  const double flops = 2.0 * rows * cols * n;
  std::printf("  %-4s %-34s %8.3f ms  %7.2f TOPS  %7.1f wGB/s", tag, what, ms,
              flops / ms / 1e9, weight_bytes / ms / 1e6);
  if (gemm_ms > 0 && gemm_ms != ms)
    std::printf("  (gemm-only %7.2f TOPS %7.1f wGB/s)", flops / gemm_ms / 1e9,
                weight_bytes / gemm_ms / 1e6);
  if (c_ms > 0)
    std::printf("  x c %5.2f", c_ms / ms);
  if (prod_ms > 0 && prod_ms != c_ms)
    std::printf("  x prod %5.2f", prod_ms / ms);
  std::printf("\n");
}

// All measured rows in one table, printed before the hyper-connection section
// and again at the end. Times are end-to-end medians (one event window per
// pipeline) except the parts; ratios are throughput versus c (Q8_0 W8A8, above
// 1 = faster) or versus prod (the route production takes for Q8_0).
void PrintSummary(const std::vector<SummaryRow>& summary) {
  std::printf("=== summary (ms; a, b, b64, d, c, prod are end-to-end; b.quant, "
              "b.gemm, b64.gemm, d.narrow, d.gemm are parts; ratios are "
              "throughput vs c = Q8_0 W8A8, above 1 = faster; prod = "
              "production route; f64 = kernel vs FP64 on 64 samples, max abs "
              "/ rms; d = Gemma 4 binary16 K-quant GEMM, n/a = -1 / 0) ===\n");
  std::printf("%-12s %-7s %5s %8s %8s %8s %8s %8s %8s %8s %8s %6s %6s %6s %6s "
              "%8s %8s %8s %s | %8s %8s %8s %6s %6s %8s %8s %8s %8s %s\n",
              "shape", "type", "N", "a", "b", "b.quant", "b.gemm", "b64",
              "b64.gemm", "c", "prod", "a/c", "b/c", "b64/c", "b/prod",
              "b-a/rms", "f64.F32", "f64.Q8", "rowind", "d", "d.narrow",
              "d.gemm", "d/c", "d/prod", "d-a/rms", "d-b/rms", "d.f64F32",
              "d.f64Q8", "rowind.d");
  const auto ratio = [](double base, double ms) {
    return base > 0 && ms > 0 ? base / ms : 0.0;
  };
  for (const SummaryRow& r : summary) {
    std::printf(
        "%-12s %-7s %5d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %6.2f "
        "%6.2f %6.2f %6.2f %8.3g %8.3g %8.3g %s | %8.3f %8.3f %8.3f %6.2f "
        "%6.2f %8.3g %8.3g %8.3g %8.3g %s\n",
        r.shape.c_str(), r.type.c_str(), r.n, r.a_ms, r.b_ms, r.bq_ms, r.bg_ms,
        r.b64_ms, r.b64g_ms, r.c_ms, r.prod_ms, ratio(r.c_ms, r.a_ms),
        ratio(r.c_ms, r.b_ms), ratio(r.c_ms, r.b64_ms),
        ratio(r.prod_ms, r.b_ms), r.rel_ba, r.b_f64_f32, r.b_f64_q8, r.rowind,
        r.d_ms, r.dn_ms, r.dg_ms, ratio(r.c_ms, r.d_ms),
        ratio(r.prod_ms, r.d_ms), r.rel_da, r.rel_db, r.d_f64_f32, r.d_f64_q8,
        r.rowind_d);
  }
  // Second table, same rows: candidate e (dequantise to an F16 scratch +
  // narrow + hipBLASLt F16 GEMM). e is unpadded (m = rows); e.pad is the best
  // padded variant (rows rounded up to 128 / 256) and ep.rows its row count,
  // 0 where padding changes nothing.
  std::printf("=== summary e (ms; e is one window; e.deq, e.narrow, e.gemm are "
              "parts; e/c, e/prod, e/b64 throughput ratios, above 1 = faster; "
              "e-a/rms, e-b/rms, f64 = max abs / rms) ===\n");
  std::printf("%-12s %-7s %5s %8s %8s %8s %8s %6s %6s %6s %8s %8s %8s %8s %8s "
              "| %8s %7s %7s\n",
              "shape", "type", "N", "e", "e.deq", "e.narrow", "e.gemm", "e/c",
              "e/prod", "e/b64", "e-a/rms", "e-b/rms", "e.f64F32", "e.f64Q8",
              "rowind.e", "e.pad", "ep.rows", "ep/prod");
  for (const SummaryRow& r : summary) {
    std::printf("%-12s %-7s %5d %8.3f %8.3f %8.3f %8.3f %6.2f %6.2f %6.2f %8.3g "
                "%8.3g %8.3g %8.3g %8s | %8.3f %7zu %7.2f\n",
                r.shape.c_str(), r.type.c_str(), r.n, r.e_ms, r.e_deq_ms,
                r.e_nar_ms, r.e_gemm_ms, ratio(r.c_ms, r.e_ms),
                ratio(r.prod_ms, r.e_ms), ratio(r.b64_ms, r.e_ms), r.rel_ea,
                r.rel_eb, r.e_f64_f32, r.e_f64_q8, r.rowind_e, r.ep_ms,
                r.ep_rows, ratio(r.prod_ms, r.ep_ms));
  }
  // Third table: candidate f, Flash-Next's DenseHalfKQuantGemm (engine route of
  // GUFO_PREFILL_NATIVE_GEMM=2) against d, the Gemma 4 launcher on the same
  // F16 rows. bits = f equals d bit for bit at N and the widths 100, 95, 64,
  // 9, 1; f/d is the time ratio (1.00 = the same).
  std::printf(
      "=== summary f (ms; f is one window around narrow + launcher; "
      "f/d = d_ms / f_ms; bits = f output equals d bit for bit; "
      "f64 = max abs / rms) ===\n");
  std::printf("%-12s %-7s %5s %8s %8s %6s %6s %-10s %-8s %8s\n", "shape",
              "type", "N", "f", "f.gemm", "f/d", "f/prod", "bits", "rowind.f",
              "f.f64F32");
  for (const SummaryRow& r : summary) {
    std::printf("%-12s %-7s %5d %8.3f %8.3f %6.2f %6.2f %-10s %-8s %8.3g\n",
                r.shape.c_str(), r.type.c_str(), r.n, r.f_ms, r.fg_ms,
                ratio(r.d_ms, r.f_ms), ratio(r.prod_ms, r.f_ms), r.bits_f,
                r.rowind_f, r.f_f64_f32);
  }
  int failures = 0;
  int failures_d = 0;
  int failures_e = 0;
  int failures_f = 0;
  for (const SummaryRow& r : summary) {
    failures_f += std::strcmp(r.bits_f, "FAIL") == 0 ||
                          std::strcmp(r.rowind_f, "FAIL") == 0
                      ? 1
                      : 0;
    failures += std::strcmp(r.rowind, "FAIL") == 0 ? 1 : 0;
    failures_d += std::strcmp(r.rowind_d, "FAIL") == 0 ? 1 : 0;
    failures_e += std::strcmp(r.rowind_e, "FAIL") == 0 ? 1 : 0;
  }
  std::printf(
      "row independence failures: b %d, d %d, e %d; f (bits or rows) "
      "%d\n\n",
      failures, failures_d, failures_e, failures_f);
}

int PrefillNative(const std::vector<int>& ns_in, int reps) {
  // Unbuffered, so a crash or exception later loses nothing already measured.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<int> ns = ns_in;
  const int max_n = *std::max_element(ns.begin(), ns.end());
  PrintHeader(reps, ns);
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init failed");
  std::vector<SummaryRow> summary;
  // hipBLASLt handle of candidate e (a null handle prints e as n/a).
  std::string blas_error;
  auto blas_e = q::BlasLt::Create(nullptr, &blas_error);

  struct Shape {
    const char* name;
    std::size_t rows;
    std::size_t cols;
  };
  const Shape shapes[] = {
      {"ssm_out/attn_output", 2560, 6144},
      {"fused SSM in", 16384, 2560},
      {"attn qkv", 13312, 2560},
      {"shared-expert gate/up", 640, 2560},
      {"shared-expert down", 2560, 640},
  };

  for (const Shape& shape : shapes) {
    const std::size_t rows = shape.rows;
    const std::size_t cols = shape.cols;
    char shape_label[64];
    std::snprintf(shape_label, sizeof(shape_label), "%zux%zu", rows, cols);
    std::printf("=== %s: %s (rows x cols) ===\n", shape.name, shape_label);

    std::vector<Q8Times> c;
    TimeQ8Reference(ns, max_n, rows, cols, reps, &c);
    const double q8_bytes = static_cast<double>(rows) * (cols / 32) * 34;
    for (std::size_t i = 0; i < ns.size(); ++i) {
      const std::size_t n = ns[i];
      if (c[i].gemm < 0) {
        std::printf("  N=%zu c  : n/a (Q8_0 needs cols %% 32 == 0)\n", n);
        continue;
      }
      std::printf("  N=%zu\n", n);
      PrintKernelLine("c", "Q8_0 quant + W8A8 GEMM", c[i].Total(), c[i].gemm,
                      rows, cols, n, q8_bytes, c[i].Total(), -1);
      std::printf("       c quant %.3f ms + gemm %.3f ms (parts timed "
                  "separately; non-finite outputs %zu)\n",
                  c[i].quant, c[i].gemm, c[i].nonfinite);
      if (c[i].f16_gemm > 0) {
        PrintKernelLine("c2", "Q8_0 narrow + F16 GEMM", c[i].F16Total(),
                        c[i].f16_gemm, rows, cols, n, q8_bytes, c[i].Total(),
                        -1);
        std::printf("       c2 narrow %.3f ms + gemm %.3f ms (non-finite "
                    "outputs %zu)\n",
                    c[i].f16_narrow, c[i].f16_gemm, c[i].f16_nonfinite);
      }
    }

    // Activations and output buffers are shared by every type of this shape;
    // the tokens of a smaller N are the first rows of the same activations.
    // 13 spare rows: the row-independence check runs a max_n + 13 pass (a
    // 13-token final chunk after a full one) on the same activations.
    constexpr std::size_t kTailRows = 13;
    const std::vector<float> x_host = RandomFloats(
        (std::size_t{static_cast<unsigned>(max_n)} + kTailRows) * cols, 1234U,
        1.0F);
    const std::size_t out_bytes = std::size_t{static_cast<unsigned>(max_n)} * rows * 4;
    DevBuf x(x_host);
    DevBuf xq(gufo::hip::QuantizedActivationBytes(max_n, cols));
    DevBuf out_a(out_bytes);
    DevBuf out_b(out_bytes);
    DevBuf out_c(out_bytes);
    // F16 activations of d and e: NarrowActivations' [tokens][cols] rows.
    DevBuf xh16((std::size_t{static_cast<unsigned>(max_n)} + kTailRows) * cols * 2);
    // e's F16 weight scratch, one per shape: the whole weight dequantised on
    // every call, with room for the row count padded up to a multiple of 256
    // (the pad rows stay zero: the dequantiser never writes them).
    const auto pad_to = [](std::size_t v, std::size_t m) {
      return (v + m - 1) / m * m;
    };
    DevBuf wh_scratch(pad_to(rows, 256) * cols * 2);
    const float* xf = x.get<float>();

    for (const NativeType& t : kNativeTypes) {
      const std::size_t block_elems = quant::QuantizedBlockElements(t.type);
      if (cols % block_elems != 0) {
        std::printf("--- %s %s: n/a (cols %zu is not a multiple of the "
                    "%zu-element block)\n",
                    shape_label, t.name, cols, block_elems);
        continue;
      }
      const std::vector<std::uint8_t> w_host =
          NativeWeights(t, rows, cols, 99U + static_cast<std::uint32_t>(rows));
      const double weight_bytes = static_cast<double>(w_host.size());
      std::printf("--- %s %s  (weights %.1f MiB, %.3f bits/weight) ---\n",
                  shape_label, t.name, weight_bytes / kMiB,
                  weight_bytes * 8.0 / (static_cast<double>(rows) * cols));

      DevBuf w(w_host);

      for (std::size_t i = 0; i < ns.size(); ++i) {
        const std::size_t n = ns[i];
        SummaryRow row;
        row.shape = shape_label;
        row.type = t.name;
        row.n = static_cast<int>(n);
        row.c_ms = c[i].gemm > 0 ? c[i].Total() : -1;
        row.prod_ms = c[i].f16_gemm > 0 ? c[i].F16Total() : row.c_ms;
        std::printf("  N=%zu\n", n);
        // Outputs are reused across N and kernels: a kernel that writes
        // nothing at this N must show up as NaN, not as the last N's values.
        out_a.Fill(0xFF);
        out_b.Fill(0xFF);
        out_c.Fill(0xFF);

        // (a) current engine path.
        bool a_ok = true;
        try {
          row.a_ms = TimedMs(
              [&] {
                const int rc = qfn_mmq_k_dense(
                    static_cast<int>(t.type), w.get(), xf, out_a.get<float>(),
                    static_cast<int>(rows), static_cast<int>(n),
                    static_cast<int>(cols), nullptr);
                if (rc != 0)
                  throw std::runtime_error("qfn_mmq_k_dense returned " +
                                           std::to_string(rc));
              },
              reps);
          PrintKernelLine("a", "qfn_mmq_k_dense (total)", row.a_ms, -1, rows,
                          cols, n, weight_bytes, row.c_ms, row.prod_ms);
        } catch (const std::exception& e) {
          a_ok = false;
          std::printf("  a    n/a: %s\n", e.what());
        }

        // (b) quantise once into the tiled layout with the sum sidecar, then
        // the decode-once int8 WMMA GEMM.
        const auto quantise = [&] {
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(xf, xq.get(), n, cols,
                                                          nullptr);
        };
        const auto gemm_b = [&] {
          gufo::hip::LaunchKQuantWmmaGEMM(t.type, w.get(), xq.get(),
                                          out_b.get<float>(), n, rows, cols,
                                          nullptr);
        };
        const auto gemm_b64 = [&] {
          gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
              t.type, w.get(), xq.get(), out_c.get<float>(), n, rows, cols,
              nullptr);
        };
        // Headline: quantise then GEMM in one event window. The parts are timed
        // separately below as a breakdown.
        row.b_ms = TimedMs(
            [&] {
              quantise();
              gemm_b();
            },
            reps);
        row.bq_ms = TimedMs(quantise, reps);
        row.bg_ms = TimedMs(gemm_b, reps);
        PrintKernelLine("b", "quantise + LaunchKQuantWmmaGEMM", row.b_ms,
                        row.bg_ms, rows, cols, n, weight_bytes, row.c_ms,
                        row.prod_ms);
        std::printf("       b parts: quantise %.3f ms + gemm %.3f ms = %.3f ms "
                    "(total above is one window)\n",
                    row.bq_ms, row.bg_ms, row.bq_ms + row.bg_ms);

        // (b64) the 27B lane's shipped dispatch.
        row.b64_ms = TimedMs(
            [&] {
              quantise();
              gemm_b64();
            },
            reps);
        row.b64g_ms = TimedMs(gemm_b64, reps);
        const bool w64 = TakesWave64(t.type, n, rows, cols);
        char what[64];
        std::snprintf(what, sizeof(what), "quantise + shipped GEMM (%s)",
                      w64 ? "wave64" : "same as b");
        PrintKernelLine("b64", what, row.b64_ms, row.b64g_ms, rows, cols, n,
                        weight_bytes, row.c_ms, row.prod_ms);

        // Correctness. xq holds the quantised activations of this N.
        const SampleRefs refs = MakeSampleRefs(
            t, w_host.data(), x_host.data(), n, rows, cols,
            777U + static_cast<std::uint32_t>(n));
        const std::size_t count = n * rows;
        const Ref64 rb_sum = CompareSamples(refs, out_b.get<float>(), rows);
        row.b_f64_f32 = rb_sum.max_f32 / rb_sum.rms;
        row.b_f64_q8 = rb_sum.max_q8 / rb_sum.rms;
        if (a_ok) {
          // out_a holds a's last timed launch at this N.
          const Diff d = CompareDevice(out_a.get<float>(), out_b.get<float>(),
                                       count);
          row.rel_ba = d.Rel();
          std::printf(
              "  err  b vs a: max %.4g  mean %.4g  max/rms %.3g  mean/rms %.3g"
              "  (rms a %.4g; non-finite a %zu b %zu)\n",
              d.max_abs, d.mean_abs, d.Rel(), d.MeanRel(), d.rms_ref,
              d.nonfinite_ref, d.nonfinite_test);
          const Ref64 ra = CompareSamples(refs, out_a.get<float>(), rows);
          const Ref64 rb = CompareSamples(refs, out_b.get<float>(), rows);
          std::printf(
              "  f64  F32 act: a max %.4g (%.3g of rms)  b max %.4g (%.3g)"
              "   Q8 act: a max %.4g (%.3g)  b max %.4g (%.3g)   rms ref %.4g\n",
              ra.max_f32, ra.max_f32 / ra.rms, rb.max_f32, rb.max_f32 / rb.rms,
              ra.max_q8, ra.max_q8 / ra.rms, rb.max_q8, rb.max_q8 / rb.rms,
              ra.rms);
        } else {
          const Ref64 rb = CompareSamples(refs, out_b.get<float>(), rows);
          std::printf("  f64  F32 act: b max %.4g (%.3g of rms)   Q8 act: b max "
                      "%.4g (%.3g)   rms ref %.4g\n",
                      rb.max_f32, rb.max_f32 / rb.rms, rb.max_q8,
                      rb.max_q8 / rb.rms, rb.rms);
        }
        // b64 against b: the two launchers must agree where they differ.
        {
          const Diff d = CompareDevice(out_b.get<float>(), out_c.get<float>(),
                                       count);
          std::printf("  err  b64 vs b: max %.4g  mean %.4g  max/rms %.3g%s\n",
                      d.max_abs, d.mean_abs, d.Rel(),
                      d.max_abs == 0 ? "  (identical)" : "");
        }

        // Row independence of b: tokens 0..99 at this N against a separate
        // N = 100 run on the first 100 tokens of the same activations.
        if (n > 100) {
          DevBuf xq100(gufo::hip::QuantizedActivationBytes(100, cols));
          DevBuf out100(std::size_t{100} * rows * 4);
          out100.Fill(0xFF);
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(xf, xq100.get(), 100,
                                                          cols, nullptr);
          gufo::hip::LaunchKQuantWmmaGEMM(t.type, w.get(), xq100.get(),
                                          out100.get<float>(), 100, rows, cols,
                                          nullptr);
          CheckHip(hipDeviceSynchronize(), "N=100 run");
          const std::size_t diff = FirstDifference(
              out_b.get<float>(), out100.get<float>(), std::size_t{100} * rows);
          const std::size_t bad100 =
              CountNonFinite(out100.get<float>(), std::size_t{100} * rows);
          if (diff == std::size_t{100} * rows && bad100 != 0) {
            // Equal bits that are NaN (unwritten) are not a pass.
            row.rowind = "FAIL";
            std::printf("  rowind b: FAIL N=100 output has %zu non-finite "
                        "values\n",
                        bad100);
          } else if (diff == std::size_t{100} * rows) {
            row.rowind = "PASS";
            std::printf("  rowind b: PASS (N=%zu rows 0..99 == N=100 run)\n", n);
          } else {
            row.rowind = "FAIL";
            float va = 0;
            float vb = 0;
            CheckHip(hipMemcpy(&va, out_b.get<float>() + diff, 4,
                               hipMemcpyDeviceToHost),
                     "hipMemcpy");
            CheckHip(hipMemcpy(&vb, out100.get<float>() + diff, 4,
                               hipMemcpyDeviceToHost),
                     "hipMemcpy");
            std::printf("  rowind b: FAIL first difference at token %zu row "
                        "%zu: N=%zu gives %.9g, N=100 gives %.9g\n",
                        diff / rows, diff % rows, n, va, vb);
          }
        }
        // Row independence of the shipped launcher (b64) across pass widths,
        // for the shapes the engine would route (NativeWmmaGemm). The reference
        // is this N's b64 output (out_c); token t of every other width must
        // match it bit for bit. "padded" runs quantise and launch 96 rows
        // whose rows past `real` are zero (the discarded padding): the same
        // calls, in the same order, as Executor::NativeGemm for a pass under
        // 96 rows (a private member, so the bench models it rather than
        // calling it; the pad copies are not part of the arithmetic); "plain"
        // runs launch `real` rows and are informational: below 96 rows the
        // launcher takes other kernels, which is what padding would avoid.
        if (static_cast<int>(n) == max_n && n >= 96 &&
            gufo::models::qwen38_flash_next::NativeWmmaGemm(t.type, rows, cols, true)) {
          struct Width {
            std::size_t real;
            std::size_t launch;
            bool gating;
          };
          const Width widths[] = {
              {n + kTailRows, n + kTailRows, true}, {130, 130, true},
              {100, 100, true}, {96, 96, true},
              {95, 96, true},   {64, 96, true},
              {9, 96, true},    {8, 96, true},
              {1, 96, true},    {95, 95, false},
              {64, 64, false},  {9, 9, false},
              {8, 8, false},    {1, 1, false}};
          const std::size_t max_launch = n + kTailRows;
          DevBuf xw(gufo::hip::QuantizedActivationBytes(max_launch, cols));
          DevBuf xin(max_launch * cols * 4);
          DevBuf ow(max_launch * rows * 4);
          bool gating_fail = false;
          int info_diff = 0;
          for (const Width& wd : widths) {
            xin.Fill(0);
            CheckHip(hipMemcpy(xin.get(), xf, wd.real * cols * 4,
                               hipMemcpyDeviceToDevice),
                     "hipMemcpy");
            ow.Fill(0xFF);
            gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(
                xin.get<float>(), xw.get(), wd.launch, cols, nullptr);
            gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
                t.type, w.get(), xw.get(), ow.get<float>(), wd.launch, rows,
                cols, nullptr);
            CheckHip(hipDeviceSynchronize(), "width run");
            const std::size_t tokens = std::min(wd.real, n);
            const std::size_t count = tokens * rows;
            const std::size_t diff =
                FirstDifference(out_c.get<float>(), ow.get<float>(), count);
            const std::size_t bad = CountNonFinite(ow.get<float>(), count);
            const char* kind = wd.real == wd.launch ? "plain " : "padded";
            if (diff == count && bad == 0) {
              std::printf("  rowind b64 %s real=%zu launch=%zu: PASS%s\n", kind,
                          wd.real, wd.launch, wd.gating ? "" : " (info)");
              continue;
            }
            float va = 0;
            float vb = 0;
            if (diff < count) {
              CheckHip(hipMemcpy(&va, out_c.get<float>() + diff, 4,
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy");
              CheckHip(hipMemcpy(&vb, ow.get<float>() + diff, 4,
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy");
            }
            std::printf("  rowind b64 %s real=%zu launch=%zu: %s first "
                        "difference at token %zu row %zu: N=%zu gives %.9g, "
                        "this width gives %.9g (non-finite %zu)\n",
                        kind, wd.real, wd.launch, wd.gating ? "FAIL" : "DIFFERS",
                        diff / rows, diff % rows, n, va, vb, bad);
            if (wd.gating)
              gating_fail = true;
            else
              ++info_diff;
          }
          std::printf("  rowind b64 summary: %s (%d plain sub-96 widths differ)\n",
                      gating_fail ? "FAIL" : "PASS", info_diff);
          if (gating_fail)
            row.rowind = "FAIL";
        }
        // (d) Gemma 4's dense binary16-activation K-quant GEMM. An engine
        // integration would narrow the F32 trunk activations to F16 rows
        // ([tokens][cols], NarrowActivations) and call g4::LaunchDenseHalfGemm
        // on them, as Gemma 4's Executor::Project does; the headline total is
        // that pair in one event window, the parts a breakdown. Output is F32
        // [tokens][rows], like b.
        {
          const auto d_format = HalfFormat(t.type);
          std::string d_reason;
          if (!d_format)
            d_reason = "the launcher takes Q4_K, Q5_K, Q6_K, Q4_0, Q5_0 only";
          else if (cols % 256 != 0)
            d_reason = "k must be a multiple of 256";
          DevBuf out_d(n * rows * 4);
          out_d.Fill(0xFF);
          const auto narrow = [&](std::size_t tokens, void* dst) {
            q::NarrowActivations(xf, dst, false, tokens * cols, nullptr);
          };
          const auto launch_d = [&](std::size_t tokens, const void* act,
                                    float* dst) {
            if (!g4::LaunchDenseHalfGemm(
                    *d_format, w.get(), act, dst, nullptr,
                    static_cast<std::uint32_t>(tokens),
                    static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(cols), nullptr))
              throw std::runtime_error("LaunchDenseHalfGemm returned false");
          };
          bool d_ok = false;
          if (d_reason.empty()) {
            try {
              // One untimed call: a launcher that declines the shape prints
              // n/a instead of failing the run.
              narrow(n, xh16.get());
              launch_d(n, xh16.get(), out_d.get<float>());
              CheckHip(hipDeviceSynchronize(), "d launch");
              CheckHip(hipGetLastError(), "d launch");
              d_ok = true;
            } catch (const std::exception& e) {
              d_reason = e.what();
            }
          }
          if (!d_ok) {
            std::printf("  d    n/a: %s\n", d_reason.c_str());
          } else {
            row.d_ms = TimedMs(
                [&] {
                  narrow(n, xh16.get());
                  launch_d(n, xh16.get(), out_d.get<float>());
                },
                reps);
            row.dn_ms = TimedMs([&] { narrow(n, xh16.get()); }, reps);
            row.dg_ms = TimedMs(
                [&] { launch_d(n, xh16.get(), out_d.get<float>()); }, reps);
            PrintKernelLine("d", "NarrowActivations + DenseHalfGemm", row.d_ms,
                            row.dg_ms, rows, cols, n, weight_bytes, row.c_ms,
                            row.prod_ms);
            std::printf("       d parts: narrow %.3f ms + gemm %.3f ms = %.3f "
                        "ms (total above is one window)\n",
                        row.dn_ms, row.dg_ms, row.dn_ms + row.dg_ms);
            // out_d holds the last timed launch at this N.
            if (a_ok) {
              const Diff da =
                  CompareDevice(out_a.get<float>(), out_d.get<float>(), count);
              row.rel_da = da.Rel();
              std::printf("  err  d vs a: max %.4g  mean %.4g  max/rms %.3g  "
                          "mean/rms %.3g  (rms a %.4g; non-finite a %zu d "
                          "%zu)\n",
                          da.max_abs, da.mean_abs, da.Rel(), da.MeanRel(),
                          da.rms_ref, da.nonfinite_ref, da.nonfinite_test);
            }
            const Diff db =
                CompareDevice(out_b.get<float>(), out_d.get<float>(), count);
            row.rel_db = db.Rel();
            std::printf("  err  d vs b: max %.4g  mean %.4g  max/rms %.3g  "
                        "mean/rms %.3g  (rms b %.4g; non-finite b %zu d %zu)\n",
                        db.max_abs, db.mean_abs, db.Rel(), db.MeanRel(),
                        db.rms_ref, db.nonfinite_ref, db.nonfinite_test);
            const Ref64 rd = CompareSamples(refs, out_d.get<float>(), rows);
            row.d_f64_f32 = rd.max_f32 / rd.rms;
            row.d_f64_q8 = rd.max_q8 / rd.rms;
            std::printf("  f64  d: F32 act max %.4g (%.3g of rms)   Q8 act max "
                        "%.4g (%.3g)   (d runs F16 activations, not Q8)\n",
                        rd.max_f32, rd.max_f32 / rd.rms, rd.max_q8,
                        rd.max_q8 / rd.rms);

            // Row independence of d at this N against narrower passes on the
            // first tokens of the same activations. The launcher's token tile
            // is 96 rows, so widths 130 (a partial second tile) and 100 (a
            // 4-row tail) gate; the rest are informational. Activation rows
            // past the pass width are zero, like an engine's unused tail.
            if (static_cast<int>(n) == max_n && n >= 130) {
              struct DWidth {
                std::size_t real;
                bool gating;
              };
              const DWidth widths[] = {{130, true}, {100, true},
                                       {n + kTailRows, false}, {97, false},
                                       {96, false},  {95, false},
                                       {64, false},  {9, false},
                                       {1, false}};
              DevBuf xin16((n + kTailRows) * cols * 2);
              DevBuf ow((n + kTailRows) * rows * 4);
              bool gating_fail = false;
              for (const DWidth& wd : widths) {
                xin16.Fill(0);
                narrow(wd.real, xin16.get());
                ow.Fill(0xFF);
                try {
                  launch_d(wd.real, xin16.get(), ow.get<float>());
                } catch (const std::exception& e) {
                  std::printf("  rowind d real=%zu: n/a: %s\n", wd.real,
                              e.what());
                  continue;
                }
                CheckHip(hipDeviceSynchronize(), "d width run");
                const std::size_t tokens = std::min(wd.real, n);
                const std::size_t cnt = tokens * rows;
                const std::size_t diff =
                    FirstDifference(out_d.get<float>(), ow.get<float>(), cnt);
                const std::size_t bad = CountNonFinite(ow.get<float>(), cnt);
                if (diff == cnt && bad == 0) {
                  std::printf("  rowind d real=%zu: PASS%s\n", wd.real,
                              wd.gating ? "" : " (info)");
                  continue;
                }
                float va = 0;
                float vb = 0;
                if (diff < cnt) {
                  CheckHip(hipMemcpy(&va, out_d.get<float>() + diff, 4,
                                     hipMemcpyDeviceToHost),
                           "hipMemcpy");
                  CheckHip(hipMemcpy(&vb, ow.get<float>() + diff, 4,
                                     hipMemcpyDeviceToHost),
                           "hipMemcpy");
                }
                std::printf("  rowind d real=%zu: %s first difference at token "
                            "%zu row %zu: N=%zu gives %.9g, this width gives "
                            "%.9g (non-finite %zu)\n",
                            wd.real, wd.gating ? "FAIL" : "DIFFERS (info)",
                            diff / rows, diff % rows, n, va, vb, bad);
                if (wd.gating)
                  gating_fail = true;
              }
              row.rowind_d = gating_fail ? "FAIL" : "PASS";
              std::printf("  rowind d summary (widths 130 and 100 gate): %s\n",
                          row.rowind_d);
            }

            // (f) Flash-Next's DenseHalfKQuantGemm, the engine's route for
            // GUFO_PREFILL_NATIVE_GEMM=2, on the same F16 rows. It instantiates
            // the shared-header kernel d runs, so its output must equal d's bit
            // for bit at every width (a wiring check); the timing is the engine
            // pair (narrow + launcher) in one window.
            {
              const auto f_format = FlashHalfFormat(t.type);
              const auto launch_f = [&](std::size_t tokens, const void* act,
                                        float* dst) {
                if (!q::DenseHalfKQuantGemm(*f_format, w.get(), act, dst,
                                            static_cast<std::uint32_t>(tokens),
                                            static_cast<std::uint32_t>(rows),
                                            static_cast<std::uint32_t>(cols),
                                            nullptr))
                  throw std::runtime_error(
                      "DenseHalfKQuantGemm returned false");
              };
              DevBuf out_f(n * rows * 4);
              out_f.Fill(0xFF);
              narrow(n, xh16.get());
              launch_f(n, xh16.get(), out_f.get<float>());
              CheckHip(hipDeviceSynchronize(), "f launch");
              CheckHip(hipGetLastError(), "f launch");
              row.f_ms = TimedMs(
                  [&] {
                    narrow(n, xh16.get());
                    launch_f(n, xh16.get(), out_f.get<float>());
                  },
                  reps);
              row.fg_ms = TimedMs(
                  [&] { launch_f(n, xh16.get(), out_f.get<float>()); }, reps);
              PrintKernelLine("f", "NarrowActivations + DenseHalfKQuantGemm",
                              row.f_ms, row.fg_ms, rows, cols, n, weight_bytes,
                              row.c_ms, row.prod_ms);
              // Bit comparison against d, f at the full width against d, and
              // f at narrower widths against f at the full width (row
              // independence). Activation rows past the pass width are zero.
              bool bits_ok = true;
              bool rowind_ok = true;
              DevBuf xin16((n + kTailRows) * cols * 2);
              DevBuf od(n * rows * 4);
              DevBuf of(n * rows * 4);
              const std::size_t full_diff = FirstDifference(
                  out_d.get<float>(), out_f.get<float>(), n * rows);
              std::printf("  bits f vs d N=%zu: %s\n", n,
                          full_diff == n * rows ? "IDENTICAL" : "DIFFERS");
              bits_ok = bits_ok && full_diff == n * rows;
              for (const std::size_t width :
                   {std::size_t{4096}, std::size_t{100}, std::size_t{95},
                    std::size_t{64}, std::size_t{9}, std::size_t{1}}) {
                if (width > n)
                  continue;
                xin16.Fill(0);
                narrow(width, xin16.get());
                od.Fill(0xFF);
                of.Fill(0xFF);
                launch_d(width, xin16.get(), od.get<float>());
                launch_f(width, xin16.get(), of.get<float>());
                CheckHip(hipDeviceSynchronize(), "f width run");
                const std::size_t cnt = width * rows;
                const std::size_t bit_diff =
                    FirstDifference(od.get<float>(), of.get<float>(), cnt);
                const std::size_t ind_diff =
                    FirstDifference(out_f.get<float>(), of.get<float>(), cnt);
                const std::size_t bad = CountNonFinite(of.get<float>(), cnt);
                std::printf(
                    "  f width %5zu: bits vs d %s; rows vs f at N=%zu "
                    "%s (non-finite %zu)\n",
                    width, bit_diff == cnt ? "IDENTICAL" : "DIFFERS", n,
                    ind_diff == cnt ? "PASS" : "FAIL", bad);
                bits_ok = bits_ok && bit_diff == cnt && bad == 0;
                rowind_ok = rowind_ok && ind_diff == cnt && bad == 0;
              }
              row.bits_f = bits_ok ? "IDENTICAL" : "FAIL";
              row.rowind_f = rowind_ok ? "PASS" : "FAIL";
              const Ref64 rf = CompareSamples(refs, out_f.get<float>(), rows);
              row.f_f64_f32 = rf.max_f32 / rf.rms;
              std::printf(
                  "  f summary: bits vs d %s, row independence %s, f64 "
                  "F32-act %.3g of rms\n",
                  row.bits_f, row.rowind_f, row.f_f64_f32);
            }
          }
        }

        // (e) Dequantise the whole quantised weight to an F16 scratch on every
        // call, narrow the activations, and run the engine's 16-bit GEMM
        // (BlasLt::Gemm, F16), the recipe GSQHalo reports for Q6_K above 1024
        // tokens. One event window = total; dequant, narrow and GEMM are the
        // parts. The weight scratch is wh_scratch (rows x cols x 2 bytes,
        // allocated per shape). Run unpadded (m = rows) and with the row count
        // padded up to a multiple of 128 and of 256 with zero weight rows
        // (variants equal to the unpadded one are skipped). Output is F32
        // [tokens][m]; padded variants are compacted to [tokens][rows] (not
        // timed) for the error columns.
        {
          std::string e_reason;
          if (!blas_e)
            e_reason = "BlasLt::Create failed: " + blas_error;
          else if (!(t.type == core::GgmlType::kQ4_K ||
                     t.type == core::GgmlType::kQ5_K ||
                     t.type == core::GgmlType::kQ6_K ||
                     t.type == core::GgmlType::kIQ4_XS ||
                     t.type == core::GgmlType::kIQ3_S))
            e_reason = "no F16 dequantiser for this type";
          else if (cols % 256 != 0)
            e_reason = "k must be a multiple of 256";
          std::size_t variants[3] = {};
          int n_variants = 0;
          for (const std::size_t p :
               {rows, pad_to(rows, 128), pad_to(rows, 256)}) {
            if (n_variants == 0 || p != variants[n_variants - 1])
              variants[n_variants++] = p;
          }
          const auto dequant = [&] {
            if (!gufo::bench::DequantizeToHalf(t.type, w.get(),
                                               wh_scratch.get(), rows, cols,
                                               nullptr))
              throw std::runtime_error("DequantizeToHalf returned false");
          };
          const auto narrow_e = [&](std::size_t tokens, void* dst) {
            q::NarrowActivations(xf, dst, false, tokens * cols, nullptr);
          };
          const auto gemm_e = [&](std::size_t mp, std::size_t tokens,
                                  const void* act, float* dst) {
            if (!blas_e->Gemm(wh_scratch.get(), act, dst, HIP_R_16F,
                              static_cast<int>(mp), static_cast<int>(tokens),
                              static_cast<int>(cols), &blas_error))
              throw std::runtime_error(blas_error);
          };
          DevBuf out_e(n * variants[n_variants - 1] * 4);
          bool e_ok = false;
          if (e_reason.empty()) {
            try {
              out_e.Fill(0xFF);
              dequant();
              narrow_e(n, xh16.get());
              gemm_e(rows, n, xh16.get(), out_e.get<float>());
              CheckHip(hipDeviceSynchronize(), "e launch");
              CheckHip(hipGetLastError(), "e launch");
              e_ok = true;
            } catch (const std::exception& ex) {
              e_reason = ex.what();
            }
          }
          if (!e_ok) {
            std::printf("  e    n/a: %s\n", e_reason.c_str());
          } else {
            std::printf("  e    F16 weight scratch %.1f MiB (%zu rows x %zu "
                        "cols x 2 bytes)\n",
                        static_cast<double>(rows * cols * 2) / kMiB, rows,
                        cols);
            for (int vi = 0; vi < n_variants; ++vi) {
              const std::size_t mp = variants[vi];
              char label[48];
              if (mp == rows)
                std::snprintf(label, sizeof(label), "e unpadded m=%zu", mp);
              else
                std::snprintf(label, sizeof(label), "e padded m=%zu", mp);
              std::string pad_err;
              const bool own =
                  blas_e->UsesOwnKernel(HIP_R_16F, static_cast<int>(mp),
                                        static_cast<int>(n),
                                        static_cast<int>(cols), &pad_err);
              out_e.Fill(0xFF);
              const double total = TimedMs(
                  [&] {
                    dequant();
                    narrow_e(n, xh16.get());
                    gemm_e(mp, n, xh16.get(), out_e.get<float>());
                  },
                  reps);
              const double t_deq = TimedMs(dequant, reps);
              const double t_nar = TimedMs([&] { narrow_e(n, xh16.get()); }, reps);
              const double t_gem = TimedMs(
                  [&] { gemm_e(mp, n, xh16.get(), out_e.get<float>()); }, reps);
              char what[64];
              std::snprintf(what, sizeof(what), "%s%s", label,
                            own ? " (own kernel)" : "");
              PrintKernelLine("e", what, total, t_gem, rows, cols, n,
                              weight_bytes, row.c_ms, row.prod_ms);
              std::printf("       e parts: dequant %.3f + narrow %.3f + gemm "
                          "%.3f = %.3f ms (total above is one window); vs b64 "
                          "x %.2f\n",
                          t_deq, t_nar, t_gem, t_deq + t_nar + t_gem,
                          row.b64_ms / total);
              // [tokens][rows] view of the output for the error columns.
              const float* view = out_e.get<float>();
              if (mp != rows) {
                CheckHip(hipMemcpy2D(out_c.get(), rows * 4, out_e.get(),
                                     mp * 4, rows * 4, n,
                                     hipMemcpyDeviceToDevice),
                         "compact padded output");
                view = out_c.get<float>();
              }
              if (a_ok) {
                const Diff ea = CompareDevice(out_a.get<float>(), view, count);
                std::printf("  err  e vs a: max %.4g  mean %.4g  max/rms %.3g  "
                            "mean/rms %.3g  (rms a %.4g; non-finite a %zu e "
                            "%zu)\n",
                            ea.max_abs, ea.mean_abs, ea.Rel(), ea.MeanRel(),
                            ea.rms_ref, ea.nonfinite_ref, ea.nonfinite_test);
                if (vi == 0)
                  row.rel_ea = ea.Rel();
              }
              const Diff eb = CompareDevice(out_b.get<float>(), view, count);
              std::printf("  err  e vs b: max %.4g  mean %.4g  max/rms %.3g  "
                          "mean/rms %.3g  (rms b %.4g; non-finite b %zu e "
                          "%zu)\n",
                          eb.max_abs, eb.mean_abs, eb.Rel(), eb.MeanRel(),
                          eb.rms_ref, eb.nonfinite_ref, eb.nonfinite_test);
              const Ref64 re = CompareSamples(refs, view, rows);
              std::printf("  f64  e: F32 act max %.4g (%.3g of rms)   Q8 act "
                          "max %.4g (%.3g)   (e runs F16 weights and "
                          "activations)\n",
                          re.max_f32, re.max_f32 / re.rms, re.max_q8,
                          re.max_q8 / re.rms);
              if (vi == 0) {
                row.e_ms = total;
                row.e_deq_ms = t_deq;
                row.e_nar_ms = t_nar;
                row.e_gemm_ms = t_gem;
                row.rel_eb = eb.Rel();
                row.e_f64_f32 = re.max_f32 / re.rms;
                row.e_f64_q8 = re.max_q8 / re.rms;
              } else if (row.ep_ms < 0 || total < row.ep_ms) {
                row.ep_ms = total;
                row.ep_rows = mp;
              }

              // Row independence of e at this N against narrower passes on the
              // first tokens (padded layout compared as is: the pad columns
              // are zero in both). hipBLASLt edge tiles can change the K order
              // across batch sizes; widths 130 and 100 gate, the rest inform.
              if (static_cast<int>(n) == max_n && n >= 130) {
                struct EWidth {
                  std::size_t real;
                  bool gating;
                };
                const EWidth widths[] = {{130, true}, {100, true},
                                         {n + kTailRows, false}, {97, false},
                                         {96, false},  {95, false},
                                         {64, false},  {9, false},
                                         {1, false}};
                DevBuf xin16((n + kTailRows) * cols * 2);
                DevBuf ow((n + kTailRows) * mp * 4);
                bool gating_fail = false;
                for (const EWidth& wd : widths) {
                  xin16.Fill(0);
                  narrow_e(wd.real, xin16.get());
                  ow.Fill(0xFF);
                  try {
                    gemm_e(mp, wd.real, xin16.get(), ow.get<float>());
                  } catch (const std::exception& ex) {
                    std::printf("  rowind %s real=%zu: n/a: %s\n", label,
                                wd.real, ex.what());
                    continue;
                  }
                  CheckHip(hipDeviceSynchronize(), "e width run");
                  const std::size_t tokens = std::min(wd.real, n);
                  const std::size_t cnt = tokens * mp;
                  const std::size_t diff = FirstDifference(
                      out_e.get<float>(), ow.get<float>(), cnt);
                  const std::size_t bad = CountNonFinite(ow.get<float>(), cnt);
                  if (diff == cnt && bad == 0) {
                    std::printf("  rowind %s real=%zu: PASS%s\n", label,
                                wd.real, wd.gating ? "" : " (info)");
                    continue;
                  }
                  float va = 0;
                  float vb = 0;
                  if (diff < cnt) {
                    CheckHip(hipMemcpy(&va, out_e.get<float>() + diff, 4,
                                       hipMemcpyDeviceToHost),
                             "hipMemcpy");
                    CheckHip(hipMemcpy(&vb, ow.get<float>() + diff, 4,
                                       hipMemcpyDeviceToHost),
                             "hipMemcpy");
                  }
                  std::printf("  rowind %s real=%zu: %s first difference at "
                              "token %zu row %zu: N=%zu gives %.9g, this "
                              "width gives %.9g (non-finite %zu)\n",
                              label, wd.real,
                              wd.gating ? "FAIL" : "DIFFERS (info)", diff / mp,
                              diff % mp, n, va, vb, bad);
                  if (wd.gating)
                    gating_fail = true;
                }
                if (vi == 0)
                  row.rowind_e = gating_fail ? "FAIL" : "PASS";
                std::printf("  rowind %s summary (widths 130 and 100 gate): "
                            "%s\n",
                            label, gating_fail ? "FAIL" : "PASS");
              }
            }
          }
        }
        summary.push_back(row);
      }
    }
    std::printf("\n");
  }

  PrintSummary(summary);

  // Hyper-connection GEMMs: BF16 / F16 through hipBLASLt (Gemm is the engine's
  // call, LibraryGemm the plain library kernel) against the Q8_0 W8A8 route.
  {
    std::string error;
    auto blas = q::BlasLt::Create(nullptr, &error);
    if (!blas)
      throw std::runtime_error(error);
    struct Hc {
      int m;
      int k;
    };
    const Hc hcs[] = {{320, 10240}, {10240, 320}, {4, 10240}};
    std::printf("=== hyper-connection GEMMs: weights [m][k], out [N][m] ===\n");
    std::printf("  ratio = Q8_0 W8A8 total ms / this ms (above 1 = faster than "
                "Q8_0); TFLOPS = 2*m*k*N / time.\n");
    for (const Hc& hc : hcs) {
      const std::size_t m = hc.m;
      const std::size_t k = hc.k;
      const std::vector<float> w_f = RandomFloats(m * k, 31U + hc.m, 0.05F);
      const std::vector<float> x_f = RandomFloats(
          std::size_t{static_cast<unsigned>(max_n)} * k, 32U, 1.0F);
      std::vector<std::uint16_t> w_bf(w_f.size());
      std::vector<std::uint16_t> w_h(w_f.size());
      for (std::size_t i = 0; i < w_f.size(); ++i) {
        w_bf[i] = Bf16Bits(w_f[i]);
        w_h[i] = Fp16Bits(w_f[i]);
      }
      std::vector<std::uint16_t> x_bf(x_f.size());
      std::vector<std::uint16_t> x_h(x_f.size());
      for (std::size_t i = 0; i < x_f.size(); ++i) {
        x_bf[i] = Bf16Bits(x_f[i]);
        x_h[i] = Fp16Bits(x_f[i]);
      }
      DevBuf wbf(w_bf);
      DevBuf wh(w_h);
      DevBuf xbf(x_bf);
      DevBuf xh(x_h);
      DevBuf xf(x_f);
      DevBuf w8(Q8Weights(m, k, 5U + hc.m));
      DevBuf xq(q::Q8TiledBytes(max_n, k));
      // One output buffer per (variant, call); the Q8_0 reference borrows the
      // last one before the F16 library run refills it. All are filled with
      // 0xFF before each use so unwritten values read back as NaN.
      const std::size_t out_bytes =
          std::size_t{static_cast<unsigned>(max_n)} * m * 4;
      DevBuf out_bf_gemm(out_bytes);
      DevBuf out_bf_lib(out_bytes);
      DevBuf out_h_gemm(out_bytes);
      DevBuf out_h_lib(out_bytes);
      const DevBuf* outs[2][2] = {{&out_bf_gemm, &out_bf_lib},
                                  {&out_h_gemm, &out_h_lib}};
      std::printf("--- m=%zu k=%zu ---\n", m, k);
      for (const int n : ns) {
        const double flops = 2.0 * m * k * n;
        std::printf("  N=%d\n", n);
        double c_ms = -1;
        if (k % 32 == 0) {
          bool ok = true;
          const auto quantise = [&] {
            q::QuantizeQ8Tiled(xf.get<float>(), xq.get(), n, k, nullptr);
          };
          const auto gemm = [&] {
            ok = q::W8A8Gemm(w8.get(), xq.get(), out_h_lib.get<float>(), n, m,
                             k, nullptr) &&
                 ok;
          };
          out_h_lib.Fill(0xFF);
          const double total_ms = TimedMs(
              [&] {
                quantise();
                gemm();
              },
              reps);
          const std::size_t bad = CountNonFinite(out_h_lib.get<float>(), n * m);
          const double qms = TimedMs(quantise, reps);
          const double gms = TimedMs(gemm, reps);
          if (ok) {
            c_ms = total_ms;
            std::printf("    Q8_0 W8A8 %8.3f ms (one window)  %7.2f TFLOPS "
                        "(parts: quant %.3f + gemm %.3f; gemm-only %7.2f "
                        "TFLOPS; non-finite outputs %zu)\n",
                        total_ms, flops / total_ms / 1e9, qms, gms,
                        flops / gms / 1e9, bad);
          } else {
            std::printf("    Q8_0 W8A8: n/a (W8A8Gemm declined this shape)\n");
          }
        }
        struct Variant {
          const char* label;
          hipDataType type;
          const void* weights;
          const void* input;
        };
        const Variant variants[] = {
            {"BF16", HIP_R_16BF, wbf.get(), xbf.get()},
            {"F16 ", HIP_R_16F, wh.get(), xh.get()},
        };
        bool done[2][2] = {{false, false}, {false, false}};
        const std::size_t count = std::size_t{static_cast<unsigned>(n)} * m;
        for (int vi = 0; vi < 2; ++vi) {
          const Variant& v = variants[vi];
          const char* label = v.label;
          const hipDataType type = v.type;
          const void* wp = v.weights;
          const void* xp = v.input;
          const bool own = blas->UsesOwnKernel(type, hc.m, n, hc.k, &error);
          // Gemm runs the library kernel itself unless it has its own kernel
          // for the shape; then a second call would repeat the first.
          for (int li = 0; li < (own ? 2 : 1); ++li) {
            const bool library = li == 1;
            const char* name =
                library ? "library" : (own ? "Gemm(own)" : "Gemm(=lib)");
            float* dst = outs[vi][li]->get<float>();
            outs[vi][li]->Fill(0xFF);
            try {
              const double ms = TimedMs(
                  [&] {
                    const bool ok =
                        library
                            ? blas->LibraryGemm(wp, xp, dst, type, hc.m, n,
                                                hc.k, &error)
                            : blas->Gemm(wp, xp, dst, type, hc.m, n, hc.k,
                                         &error);
                    if (!ok)
                      throw std::runtime_error(error);
                  },
                  reps);
              done[vi][li] = true;
              std::printf("    %s %-10s %8.3f ms  %7.2f TFLOPS", label, name,
                          ms, flops / ms / 1e9);
              if (c_ms > 0)
                std::printf("  x Q8_0 %5.2f", c_ms / ms);
              std::printf("  (non-finite outputs %zu)\n",
                          CountNonFinite(dst, count));
            } catch (const std::exception& e) {
              std::printf("    %s %-10s n/a: %s\n", label, name, e.what());
            }
          }
        }
        // Sanity: the paths should agree to rounding (BF16 and F16 round their
        // inputs differently, so those two differ by more).
        const auto sanity = [&](const char* what, const DevBuf& ref,
                                const DevBuf& test) {
          const Diff d =
              CompareDevice(ref.get<float>(), test.get<float>(), count);
          std::printf("    check %s: max %.4g  mean %.4g  max/rms %.3g  "
                      "(non-finite %zu/%zu)\n",
                      what, d.max_abs, d.mean_abs, d.Rel(), d.nonfinite_ref,
                      d.nonfinite_test);
        };
        if (done[0][0] && done[1][0])
          sanity("BF16 Gemm vs F16 Gemm", out_bf_gemm, out_h_gemm);
        if (done[0][0] && done[0][1])
          sanity("BF16 Gemm vs library", out_bf_gemm, out_bf_lib);
        if (done[1][0] && done[1][1])
          sanity("F16 Gemm vs library", out_h_gemm, out_h_lib);
      }
    }
    std::printf("\n");
  }

  PrintSummary(summary);
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc > 1 && std::strcmp(argv[1], "--prefill-native") == 0) {
    std::vector<int> ns;
    for (const char* p = argc > 2 ? argv[2] : "4096,100"; *p;) {
      char* end = nullptr;
      const long n = std::strtol(p, &end, 10);
      if (end == p || n <= 0 || n > 65536)
        throw std::runtime_error("bad token list; expected e.g. 4096,100");
      ns.push_back(static_cast<int>(n));
      p = *end == ',' ? end + 1 : end;
    }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    if (ns.empty() || reps < 1)
      throw std::runtime_error("bad token list or reps");
    return PrefillNative(ns, reps);
  }
  const std::uint32_t tokens = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  const auto report = [&](const char* name, double ms, double flops,
                          std::uint64_t hash) {
    std::printf("%-5s tokens %u  %8.3f ms  %6.2f TFLOPS  hash %016llx\n", name,
                tokens, ms, flops / ms / 1e9,
                static_cast<unsigned long long>(hash));
  };

  // Output projection: [tokens][6144] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 6144;
    auto* w = Upload(Q8Weights(m, k, 11U));
    auto* x = Upload(Activations(tokens * k, 12U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("out", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    if (tokens >= 1024) {
      constexpr auto stride = k + 32;
      auto* padded =
          static_cast<__half*>(Zeros(tokens * stride * sizeof(__half)));
      CheckHip(
          hipMemcpy2D(padded, stride * sizeof(__half), x, k * sizeof(__half),
                      k * sizeof(__half), tokens, hipMemcpyDeviceToDevice),
          "padded activation");
      const auto expected = Hash(y, tokens * m * sizeof(float));
      const double padded_ms = MedianMs(
          [&] {
            if (!q::DenseF16Gemm(w, padded, y, tokens, m, k, nullptr, stride))
              throw std::runtime_error("strided DenseF16Gemm");
          },
          reps);
      const auto actual = Hash(y, tokens * m * sizeof(float));
      if (actual != expected)
        throw std::runtime_error("strided projection changed output");
      report("out-padded", padded_ms, 2.0 * tokens * m * k, actual);
      CheckHip(hipFree(padded), "hipFree");
    }
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // Shared-expert down projection: [tokens][640] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 640;
    auto* w = Upload(Q8Weights(m, k, 41U));
    auto* x = Upload(Activations(tokens * k, 42U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("down", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // F16 router and recurrent gates (unquantized weights).
  for (const std::size_t m : {std::size_t{513}, std::size_t{96}}) {
    constexpr std::size_t k = 2560;
    auto* w = Upload(Activations(m * k, 51U));
    auto* x = Upload(Activations(tokens * k, 52U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::UnquantizedF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("UnquantizedF16Gemm");
        },
        reps);
    report(m == 513 ? "route" : "gates", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // HC up projection fused with the stream mixer: [tokens][320] gates over
  // four 2560-wide F16 streams into F16 and tiled Q8 mixer rows.
  {
    constexpr std::uint32_t hidden = 2560;
    constexpr std::uint32_t rank = 320;
    auto* w = Upload(Q8Weights(4 * hidden, rank, 51U));
    auto* low = Upload(Activations(std::size_t{tokens} * rank, 52U));
    auto* xn = Upload(Activations(std::size_t{tokens} * 4 * hidden, 53U));
    auto* mixed = static_cast<__half*>(
        Zeros(std::size_t{tokens} * hidden * sizeof(__half)));
    const std::size_t q8_bytes =
        ((tokens + 15) / 16) * std::size_t{hidden / 32} * 576;
    auto* mixed_q8 = Zeros(q8_bytes);
    const double ms = MedianMs(
        [&] {
          if (!q::HcMixF16Gemm(w, low, xn, nullptr, nullptr, mixed, mixed_q8,
                               nullptr, tokens, hidden, rank, nullptr))
            throw std::runtime_error("HcMixF16Gemm");
        },
        reps);
    report("hcmix", ms, 2.0 * tokens * 4 * hidden * rank,
           Hash(mixed, std::size_t{tokens} * hidden * sizeof(__half)) ^
               Hash(mixed_q8, q8_bytes));
    for (void* p :
         {static_cast<void*>(w), static_cast<void*>(low),
          static_cast<void*>(xn), static_cast<void*>(mixed), mixed_q8})
      CheckHip(hipFree(p), "hipFree");
  }

  // SSM projection fused with the four-tap convolution.
  {
    constexpr std::uint32_t m = 16384;
    constexpr std::uint32_t k = 2560;
    constexpr std::uint32_t channels = 10240;
    auto* w = Upload(Q8Weights(m, k, 21U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 22U));
    std::vector<float> conv(channels * 4);
    std::uint32_t seed = 23U;
    for (auto& v : conv)
      v = static_cast<float>(Next(&seed) % 1000) / 2000.0F;
    auto* conv_w = Upload(conv);
    auto* history =
        static_cast<float*>(Zeros(std::size_t{3} * channels * sizeof(float)));
    auto* qkvz =
        static_cast<float*>(Zeros(std::size_t{tokens} * m * sizeof(float)));
    auto* convolved = static_cast<float*>(
        Zeros(std::size_t{tokens} * channels * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16SsmGemm(w, x, conv_w, history, qkvz, convolved,
                                  tokens, m, k, channels, 4, nullptr))
            throw std::runtime_error("DenseF16SsmGemm");
        },
        reps);
    report("ssm", ms, 2.0 * tokens * m * k,
           Hash(convolved, std::size_t{tokens} * channels * sizeof(float)));
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(conv_w), static_cast<void*>(history),
                    static_cast<void*>(qkvz), static_cast<void*>(convolved)})
      CheckHip(hipFree(p), "hipFree");
  }

  // Stacked attention projection: 24 query, 24 gate, 2 key and 2 value heads.
  {
    constexpr std::uint32_t heads = 24;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t m = 2 * (heads + kv_heads) * 256;
    constexpr std::uint32_t k = 2560;
    auto* w = Upload(Q8Weights(m, k, 31U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 32U));
    std::vector<float> gamma(256, 1.0F);
    auto* q_gamma = Upload(gamma);
    auto* k_gamma = Upload(gamma);
    auto* position = Upload(std::vector<std::uint32_t>{0U});
    const std::size_t q_bytes =
        std::size_t{tokens} * heads * 256 * sizeof(float);
    const std::size_t kv_bytes =
        std::size_t{tokens} * kv_heads * 256 * sizeof(__half);
    auto* query = static_cast<float*>(Zeros(q_bytes));
    auto* gate = static_cast<float*>(Zeros(q_bytes));
    auto* keys = static_cast<__half*>(Zeros(kv_bytes));
    auto* values = static_cast<__half*>(Zeros(kv_bytes));
    const double ms = MedianMs(
        [&] {
          if (!q::AttentionF16Gemm(w, x, q_gamma, k_gamma, query, gate, keys,
                                   values, tokens, position, 1.0e7F, 1.0e-6F,
                                   nullptr))
            throw std::runtime_error("AttentionF16Gemm");
        },
        reps);
    const std::uint64_t hash = Hash(query, q_bytes) ^ Hash(gate, q_bytes) ^
                               Hash(keys, kv_bytes) ^ Hash(values, kv_bytes);
    report("attn", ms, 2.0 * tokens * m * k, hash);
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(q_gamma), static_cast<void*>(k_gamma),
                    static_cast<void*>(position), static_cast<void*>(query),
                    static_cast<void*>(gate), static_cast<void*>(keys),
                    static_cast<void*>(values)})
      CheckHip(hipFree(p), "hipFree");
  }

  // BF16 indexer projections (query 512 x 2560, key 128 x 2560, and a TP2
  // rank's halves): the library kernel against BlasLt::Gemm's own kernel,
  // which sums every token in K order. Timed at the requested tokens; the two
  // agree to rounding, and the own kernel's rows must not depend on where a
  // token sits in the batch.
  {
    std::string error;
    auto blas = q::BlasLt::Create(nullptr, &error);
    if (!blas)
      throw std::runtime_error(error);
    constexpr int k = 2560;
    const auto bf16 = [](std::size_t count, std::uint32_t seed, float scale) {
      std::vector<std::uint16_t> v(count);
      for (auto& e : v) {
        const float f =
            scale *
            static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
            1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        e = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                       16);
      }
      return v;
    };
    const auto host = [](const float* d, std::size_t count) {
      std::vector<float> h(count);
      CheckHip(hipMemcpy(h.data(), d, count * 4, hipMemcpyDeviceToHost),
               "hipMemcpy");
      return h;
    };
    for (const int m : {512, 128, 256, 64}) {
      auto* w = Upload(bf16(std::size_t{512} * k, 51U + m, 0.05F));
      auto* x = Upload(bf16(std::size_t{tokens} * k, 52U, 3.0F));
      const std::size_t bytes = std::size_t{tokens} * m * sizeof(float);
      auto* y_lib = static_cast<float*>(Zeros(bytes));
      auto* y_own = static_cast<float*>(Zeros(bytes));
      const auto lib = [&](int n) {
        if (!blas->LibraryGemm(w, x, y_lib, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const auto own = [&](int n) {
        if (!blas->Gemm(w, x, y_own, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const int t = static_cast<int>(tokens);
      char name[16];
      std::snprintf(name, sizeof(name), "bf%d", m);
      std::printf("%s-lib", name);
      report("", MedianMs([&] { lib(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_lib, bytes));
      std::printf("%s-own", name);
      report("", MedianMs([&] { own(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_own, bytes));
      if (!blas->UsesOwnKernel(HIP_R_16BF, m, t, k, &error))
        throw std::runtime_error(std::string(name) + " runs the library");
      lib(t);
      own(t);
      CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
      const std::size_t count = std::size_t{static_cast<unsigned>(t)} * m;
      const auto library = host(y_lib, count);
      const auto whole = host(y_own, count);
      double worst = 0.0;
      for (std::size_t i = 0; i < count; ++i)
        worst = std::max(worst, std::abs(double{library[i]} - whole[i]));
      // The same rows at other offsets and batch widths.
      int moved = 0;
      for (const int offset : {1, 31, 37, 128, 777}) {
        if (offset >= t)
          continue;
        const int n = t - offset;
        if (!blas->Gemm(
                w, static_cast<std::uint16_t*>(x) + std::size_t(offset) * k,
                y_own, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
        CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
        const auto part =
            host(y_own, std::size_t{static_cast<unsigned>(n)} * m);
        if (std::memcmp(part.data(), whole.data() + std::size_t(offset) * m,
                        part.size() * 4) != 0) {
          ++moved;
          std::printf("%s offset %d DIFF\n", name, offset);
        }
      }
      std::printf(
          "%s n %d: max |own - library| %.3g; rows changed with their "
          "batch offset at %d of 5 offsets\n",
          name, t, worst, moved);
      for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                      static_cast<void*>(y_lib), static_cast<void*>(y_own)})
        CheckHip(hipFree(p), "hipFree");
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "dense_gemm_bench: %s\n", e.what());
  return 1;
}
