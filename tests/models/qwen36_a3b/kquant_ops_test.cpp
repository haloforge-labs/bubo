// Independent CPU dequantization + FP64 dots check vendor MMVQ/MMQ. Exact
// Q8_1-representable activations keep the tolerance about kernel arithmetic.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen36_a3b {

struct EngineTestAccess {
  static void Dense(const Tensor& weight, const float* input, float* output,
                    std::uint32_t rows, void* scratch, std::int32_t* zero_ids) {
    Engine engine;
    engine.xq_ = scratch;
    engine.zero_ids_ = zero_ids;
    engine.Dense(weight, input, output, rows);
  }
};

}  // namespace gufo::models::qwen36_a3b

namespace {

using gufo::core::GgmlType;
namespace a3b = gufo::models::qwen36_a3b;
namespace fn = gufo::models::qwen38_flash_next::rocm;

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

struct Device {
  void* data{};
  explicit Device(std::size_t bytes) { Hip(hipMalloc(&data, bytes)); }
  template<class T>
  explicit Device(const std::vector<T>& host)
      : Device(host.size() * sizeof(T)) {
    Hip(hipMemcpy(data, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice));
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template<class T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

std::uint32_t Random(std::uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

struct Weights {
  std::vector<std::uint8_t> encoded;
  std::vector<float> values;
};

Weights MakeWeights(GgmlType type, int rows, int cols, std::uint32_t seed) {
  Weights w;
  const auto stride = gufo::quant::QuantizedRowBytes(type, cols);
  const auto block_bytes = stride / (cols / 256);
  w.encoded.resize(rows * stride);
  for (auto& v : w.encoded)
    v = static_cast<std::uint8_t>(Random(seed));
  for (std::size_t at = 0; at < w.encoded.size(); at += block_bytes) {
    const auto put = [&](std::size_t offset, float scale) {
      const __half d = __float2half(scale);
      std::memcpy(w.encoded.data() + at + offset, &d, sizeof(d));
    };
    const float scale = 0.00003F + (Random(seed) % 100) * 0.000001F;
    if (type == GgmlType::kQ3_K) {
      put(108, scale);
    } else if (type == GgmlType::kQ6_K) {
      put(208, scale);
    } else {
      put(0, scale);
      put(2, scale * 0.75F);
    }
  }
  w.values.resize(std::size_t(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    const void* src = w.encoded.data() + r * stride;
    float* dst = w.values.data() + std::size_t(r) * cols;
    switch (type) {
      case GgmlType::kQ3_K:
        gufo::quant::DequantizeQ3_K(src, dst, cols);
        break;
      case GgmlType::kQ4_K:
        gufo::quant::DequantizeQ4_K(src, dst, cols);
        break;
      case GgmlType::kQ5_K:
        gufo::quant::DequantizeQ5_K(src, dst, cols);
        break;
      case GgmlType::kQ6_K:
        gufo::quant::DequantizeQ6_K(src, dst, cols);
        break;
      default:
        throw std::runtime_error("bad test type");
    }
  }
  return w;
}

std::vector<float> Inputs(int rows, int cols) {
  std::vector<float> x(std::size_t(rows) * cols);
  std::uint32_t seed = 7183;
  for (std::size_t i = 0; i < x.size(); ++i) {
    // Each 32-value Q8_1 block has d = 1/128, exactly stored in FP16.
    x[i] = (i % 32 == 0 ? 127 : int(Random(seed) % 255) - 127) / 128.F;
  }
  return x;
}

std::vector<double> Reference(const Weights& w, const std::vector<float>& x,
                              const std::vector<std::int32_t>& ids, int rows,
                              int cols, int tokens, int used) {
  std::vector<double> out(std::size_t(tokens) * used * rows);
  for (int t = 0; t < tokens; ++t)
    for (int s = 0; s < used; ++s)
      for (int r = 0; r < rows; ++r) {
        const int slot = t * used + s;
        if (ids[slot] < 0)
          continue;
        const float* weight =
            w.values.data() + (std::size_t(ids[slot]) * rows + r) * cols;
        double sum = 0;
        for (int k = 0; k < cols; ++k)
          sum += double(weight[k]) * x[std::size_t(t) * cols + k];
        out[std::size_t(slot) * rows + r] = sum;
      }
  return out;
}

float Half(float v) { return __half2float(__float2half(v)); }

void ScaleMinK4(int j, const std::uint8_t* q, int& sc, int& m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// The WMMA tile load for Q4_K/Q5_K (mmq.hpp load_tiles_q4_K/q5_K) forms
// d*scale and dmin*min in FP16, and the DS4 Q8_1 tile keeps each 32-value sum
// in FP16, as upstream ggml does. Stage both; everything else stays FP64.
// With stage=false it must reproduce Reference(), which checks the decoder.
// One output element: weight row `row` of w against the activation row xs.
double StagedQ45KDot(GgmlType type, const Weights& w, int row, const float* x,
                     int cols, bool stage) {
  const auto round = [stage](float v) { return stage ? Half(v) : v; };
  const bool q5 = type == GgmlType::kQ5_K;
  const std::size_t block_bytes = q5 ? 176 : 144;
  const auto stride = gufo::quant::QuantizedRowBytes(type, cols);
  double sum = 0;
  for (int b = 0; b < cols / 256; ++b) {
    const std::uint8_t* block = w.encoded.data() + row * stride + b * block_bytes;
    __half dh, mh;
    std::memcpy(&dh, block, sizeof(dh));
    std::memcpy(&mh, block + 2, sizeof(mh));
    const float d = __half2float(dh), dmin = __half2float(mh);
    const std::uint8_t* qh = block + 16;
    const std::uint8_t* qs = block + (q5 ? 48 : 16);
    for (int j = 0; j < 8; ++j) {
      int sc, m;
      ScaleMinK4(j, block + 4, sc, m);
      const float* xs = x + b * 256 + j * 32;
      double dot = 0, xsum = 0;
      for (int l = 0; l < 32; ++l) {
        const std::uint8_t byte = qs[(j / 2) * 32 + l];
        int q = j % 2 == 0 ? byte & 0xF : byte >> 4;
        if (q5 && (qh[l] >> j) & 1)
          q += 16;
        dot += double(q) * xs[l];
        xsum += xs[l];
      }
      sum += double(round(d * float(sc))) * dot -
             double(round(dmin * float(m))) * round(float(xsum));
    }
  }
  return sum;
}

std::vector<double> StagedQ45KReference(GgmlType type, const Weights& w,
                                        const std::vector<float>& x, int rows,
                                        int cols, int tokens, bool stage) {
  std::vector<double> out(std::size_t(tokens) * rows);
  for (int t = 0; t < tokens; ++t)
    for (int r = 0; r < rows; ++r)
      out[std::size_t(t) * rows + r] =
          StagedQ45KDot(type, w, r, x.data() + std::size_t(t) * cols, cols,
                        stage);
  return out;
}

// Routed reference with the FP16-staged Q4_K/Q5_K arithmetic: slot s of
// token t reads expert ids[slot]; inactive (-1) slots stay zero.
std::vector<double> StagedRoutedReference(GgmlType type, const Weights& w,
                                          const std::vector<float>& x,
                                          const std::vector<std::int32_t>& ids,
                                          int rows, int cols, int tokens,
                                          int used) {
  std::vector<double> out(std::size_t(tokens) * used * rows);
  for (int t = 0; t < tokens; ++t)
    for (int s = 0; s < used; ++s) {
      const int slot = t * used + s;
      if (ids[slot] < 0)
        continue;
      for (int r = 0; r < rows; ++r)
        out[std::size_t(slot) * rows + r] = StagedQ45KDot(
            type, w, ids[slot] * rows + r, x.data() + std::size_t(t) * cols,
            cols, true);
    }
  return out;
}

constexpr std::size_t kGuard = 17;
constexpr float kCanary = 123456.F;

void Compare(const Device& output, const std::vector<double>& expected,
             const char* what, bool required = true) {
  std::vector<float> got(expected.size() + kGuard);
  Hip(hipMemcpy(got.data(), output.data, got.size() * sizeof(float),
                hipMemcpyDeviceToHost));
  double scale = 0, error = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Require(std::isfinite(got[i]), "nonfinite K-quant result");
    error = std::max(error, std::abs(double(got[i]) - expected[i]));
    scale = std::max(scale, std::abs(expected[i]));
  }
  for (std::size_t i = expected.size(); i < got.size(); ++i)
    Require(got[i] == kCanary, "K-quant output overran a ragged row");
  std::cout << what << " error=" << error << " scale=" << scale << '\n';
  if (required)
    Require(error <= 1e-6 + scale * 1e-4, "K-quant CPU reference mismatch");
}

Device Output(std::size_t size) {
  return Device(std::vector<float>(size + kGuard, kCanary));
}

void Moe(GgmlType type, int tokens, bool down) {
  const int rows = down ? 67 : 37;
  const int cols = down ? 512 : 2048;
  constexpr int experts = 11, used = 8;
  const int n = down ? tokens * used : tokens;
  const int u = down ? 1 : used;
  const auto gate = MakeWeights(type, experts * rows, cols, 1113);
  const auto up = MakeWeights(type, experts * rows, cols, 883);
  const auto x = Inputs(n, cols);
  std::vector<std::int32_t> ids(n * u);
  for (int t = 0; t < n; ++t)
    for (int s = 0; s < u; ++s)
      ids[t * u + s] = (t % 2 == 0 ? s : t + s * 3) % experts;
  ids.back() = -1;  // inactive expert slots must write zero
  Device dg(gate.encoded), du(up.encoded), dx(x), di(ids);
  auto ya = Output(std::size_t(n) * u * rows);
  auto yb = Output(std::size_t(n) * u * rows);
  auto expected = Reference(gate, x, ids, rows, cols, n, u);
  auto expected_up = Reference(up, x, ids, rows, cols, n, u);
  Require(qfn_mmq_moe_vec(static_cast<int>(type), dg.data, dx.As<float>(),
                          di.As<std::int32_t>(), ya.As<float>(), rows, cols, n,
                          experts, u, nullptr, du.data, yb.As<float>()) == 0,
          "K-quant MoE vec rejected");
  Compare(ya, expected, "MoE vec gate/down");
  Compare(yb, expected_up, "MoE vec paired up");
  if (!down) {
    Require(qfn_mmq_moe_gated_vec(static_cast<int>(type), dg.data, du.data,
                                  dx.As<float>(), di.As<std::int32_t>(),
                                  ya.As<float>(), rows, cols, n, experts, u,
                                  nullptr) == 0,
            "K-quant gated MoE rejected");
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] =
          expected[i] / (1.0 + std::exp(-expected[i])) * expected_up[i];
    Compare(ya, expected, "MoE gated SwiGLU");
  }
}

// Wide prefill expert GEMM: the entry point and hints Engine::WideMoe uses
// for K-quant layers (qfn_mmq_k_moe_raw after the counts-based grid hints).
// 11 experts, top 8; experts 9 and 10 receive nothing and the last slot is
// inactive (-1), which MMQ never writes (the output stays at its zero fill).
void WideMoe(GgmlType type, int tokens, bool down) {
  const int rows = down ? 67 : 37;
  const int cols = down ? 512 : 2048;
  constexpr int experts = 11, used = 8;
  const int n = down ? tokens * used : tokens;
  const int u = down ? 1 : used;
  const auto w = MakeWeights(type, experts * rows, cols, down ? 4421 : 3307);
  const auto x = Inputs(n, cols);
  std::vector<std::int32_t> ids(n * u);
  for (int t = 0; t < n; ++t)
    for (int s = 0; s < u; ++s)
      ids[t * u + s] = (down ? (t / used + t % used) : (t + s)) % 9;
  ids.back() = -1;
  std::vector<std::uint32_t> counts(experts, 0);
  for (auto id : ids)
    if (id >= 0)
      ++counts[id];
  std::uint32_t max_rows = 1;
  for (auto c : counts)
    max_rows = std::max(max_rows, c);
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(max_rows));
  qfn_mmq_set_routed_tile_cols(
      qfn_mmq_routed_tile_cols_for_counts(counts.data(), experts));
  Device dw(w.encoded), dx(x), di(ids);
  std::vector<float> fill(std::size_t(n) * u * rows + kGuard, 0.F);
  std::fill(fill.end() - kGuard, fill.end(), kCanary);
  Device output(fill);
  const int rc = qfn_mmq_k_moe_raw(static_cast<int>(type), dw.data,
                                   dx.As<float>(), di.As<std::int32_t>(),
                                   output.As<float>(), rows, cols, n, experts,
                                   u, nullptr);
  qfn_mmq_set_routed_max_expert_rows(0);
  qfn_mmq_set_routed_tile_cols(0);
  Require(rc == 0, "wide K-quant MoE MMQ rejected");
  const auto expected =
      type == GgmlType::kQ4_K || type == GgmlType::kQ5_K
          ? StagedRoutedReference(type, w, x, ids, rows, cols, n, u)
          : Reference(w, x, ids, rows, cols, n, u);
  Compare(output, expected, down ? "wide MoE MMQ down" : "wide MoE MMQ gate/up");
}

void Head(GgmlType type, int tokens, bool gathered) {
  constexpr int rows = 257, cols = 2048;
  const auto w = MakeWeights(type, rows, cols, 8721);
  const auto x = Inputs(tokens, cols);
  std::vector<std::int32_t> zero_ids(tokens, 0);
  Device dw(w.encoded), dx(x), di(zero_ids);
  Device scratch(qfn_mmq_q8_1_bytes(tokens, cols));
  auto output = Output(std::size_t(tokens) * rows);
  a3b::Tensor weight{dw.data, type, cols, rows, 1};
  a3b::EngineTestAccess::Dense(weight, dx.As<float>(), output.As<float>(),
                               tokens, scratch.data, di.As<std::int32_t>());
  const auto expected = Reference(w, x, zero_ids, rows, cols, tokens, 1);
  Compare(output, expected, "A3B head GEMV");
  if (gathered) {
    const std::vector<std::int32_t> map{256, 3, 129, 0, 64};
    Device dm(map);
    const auto stride = gufo::quant::EncodedSizeBytes(type, cols);
    Device gathered_rows(map.size() * stride);
    fn::GatherRows(dw.data, gathered_rows.data, dm.As<std::int32_t>(),
                   map.size(), stride, nullptr);
    weight.data = gathered_rows.data;
    weight.rows = static_cast<std::uint32_t>(map.size());
    a3b::EngineTestAccess::Dense(weight, dx.As<float>(), output.As<float>(),
                                 tokens, scratch.data, di.As<std::int32_t>());
    std::vector<float> actual(std::size_t(tokens) * map.size());
    Hip(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
    for (int t = 0; t < tokens; ++t)
      for (std::size_t r = 0; r < map.size(); ++r) {
        const double ref = expected[std::size_t(t) * rows + map[r]];
        Require(std::isfinite(actual[t * map.size() + r]) &&
                    std::abs(actual[t * map.size() + r] - ref) <=
                        1e-6 + std::abs(ref) * 1e-4,
                "pruned K-quant head row gather mismatch");
      }
  }
}

void Mmq(GgmlType type) {
  constexpr int rows = 37, cols = 512, tokens = 9;
  const auto w = MakeWeights(type, rows, cols, 719);
  const auto x = Inputs(tokens, cols);
  const std::vector<std::int32_t> ids(tokens, 0);
  Device dw(w.encoded), dx(x);
  auto output = Output(rows * tokens);
  Require(qfn_mmq_k_dense(static_cast<int>(type), dw.data, dx.As<float>(),
                          output.As<float>(), rows, tokens, cols, nullptr) == 0,
          "K-quant MMQ rejected");
  if (type != GgmlType::kQ4_K && type != GgmlType::kQ5_K) {
    Compare(output, Reference(w, x, ids, rows, cols, tokens, 1), "K-quant MMQ");
    return;
  }
  const auto expected = Reference(w, x, ids, rows, cols, tokens, 1);
  const auto decoded = StagedQ45KReference(type, w, x, rows, cols, tokens, false);
  for (std::size_t i = 0; i < expected.size(); ++i)
    Require(std::abs(decoded[i] - expected[i]) <=
                1e-12 + std::abs(expected[i]) * 1e-9,
            "staged K-quant decoder disagrees with the CPU dequantizer");
  Compare(output, expected, "K-quant MMQ unstaged (info)", false);
  Compare(output, StagedQ45KReference(type, w, x, rows, cols, tokens, true),
          "K-quant MMQ FP16-staged");
}

}  // namespace

int main() {
  try {
    Hip(hipSetDevice(0));
    Require(qfn_mmq_init(0) == 0, "MMQ init failed");
    for (auto type :
         {GgmlType::kQ3_K, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K}) {
      std::cout << gufo::core::ToString(type) << '\n';
      for (int tokens : {1, 2, 3, 7, 8}) {
        Moe(type, tokens, false);
        Moe(type, tokens, true);
        Head(type, tokens, tokens == 1);
      }
      Mmq(type);
      for (int tokens : {37, 300}) {
        WideMoe(type, tokens, false);
        WideMoe(type, tokens, true);
      }
    }
    Hip(hipDeviceSynchronize());
    std::cout << "PASS: K-quant MoE, wide MoE MMQ, head, pruned gather and MMQ\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
