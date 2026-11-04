// Operator-only speed and real-tensor byte comparison. --help uses no device.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/dense_gemv.hpp"
#include "src/models/qwen38_flash_next/hc_bf16_dense_policy.hpp"
#include "src/models/qwen38_flash_next/iq4_wide_policy.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/kquant_stage.hpp"

hipError_t ExpertBenchEvict(void*, std::size_t);

// Bench-only tubo control (tubo_k4.hip.cpp, copied MIT kernels).
extern "C" {
int bench_tubo_config_count();
int bench_tubo_config(int cfg, const char** name, int* type, int* rows32,
                      int* act);
unsigned long long bench_tubo_launch_count(int cfg, int width);
int bench_tubo_repack(const void* src, void* dst, int rows, int k,
                      int block_bytes);
int bench_tubo_prepare(int cfg, const float* x, void* xq, void* xds, void* xsum,
                       int n, int k);
int bench_tubo_launch(int cfg, const void* w, const void* xq, const void* xds,
                      const void* xsum, float* y, int n, int m, int k);
}

namespace {
namespace d = gufo::models::qwen38_flash_next::dense_gemv;
void Check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
void Require(bool ok, const char* reason) {
  if (!ok)
    throw std::runtime_error(reason);
}
struct Buffer {
  void* p = nullptr;
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&p, bytes)); }
  ~Buffer() {
    if (p)
      (void)hipFree(p);
  }
  Buffer(const Buffer&) = delete;
};
struct Shape {
  const char* name;
  int k, m;
};
constexpr Shape shapes[]{{"attn_qkv", 2560, 10240},
                         {"attn_gate", 2560, 6144},
                         {"ssm_out+attn_output", 6144, 2560},
                         {"attn_q", 2560, 12288},
                         {"attn_k+attn_v", 2560, 512},
                         {"shared_gate+up", 2560, 640},
                         {"shared_down", 640, 2560},
                         {"hc_down", 10240, 320},
                         {"hc_up", 320, 10240}};
const char* Format(int type) {
  switch (type) {
    case 0:
      return "F32";
    case 1:
      return "F16";
    case 30:
      return "BF16";
    case 23:
      return "IQ4_XS";
    case 20:
      return "IQ4_NL";
    case 12:
      return "Q4_K";
    case 13:
      return "Q5_K";
    case 7:
      return "Q5_1";
    case 14:
      return "Q6_K";
    case 21:
      return "IQ3_S";
    case 18:
      return "IQ3_XXS";
    case 17:
      return "IQ2_XS";
    default:
      return "Q8_0";
  }
}
const char* Name(int v) {
  return v == 0     ? "existing"
         : v == 100 ? "iq4_wmma"
                    : d::VariantName(static_cast<d::Variant>(v));
}
bool Eligible(int type, int variant) {
  if (variant == 100)
    return type == 20 || type == 23;
  if (variant == int(d::Variant::kKQuantWmma))
    return d::IsKQuantTarget(type);
  return variant < 3 || d::IsIqTarget(type);
}
int Block(int type) {
  return type == 8 ? 32 : d::BlockSize(type);
}
int Bytes(int type) {
  return type == 8 ? 34 : d::BlockBytes(type);
}
std::size_t WeightBytes(int type, Shape s) {
  return std::size_t(s.m) * (s.k / Block(type)) * Bytes(type);
}
std::uint32_t Next(std::uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}
std::vector<int> Variants() {
  const char* env = std::getenv("GUFO_DENSE_BENCH_VARIANTS");
  std::string text = env ? env : "baseline,reuse1";
  std::vector<int> out{0};
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto end = text.find(',', pos);
    const auto part = text.substr(pos, end - pos);
    const int v = part == "iq4_wmma" ? 100 : d::ParseVariant(part.c_str());
    Require(v >= 0, "unknown GUFO_DENSE_BENCH_VARIANTS value");
    if (std::find(out.begin(), out.end(), v) == out.end())
      out.push_back(v);
    if (end == std::string::npos)
      break;
    pos = end + 1;
  }
  return out;
}
int Call(int type, int variant, const void* w, const void* x, const void* ids,
         void* y, Shape s, int n) {
  if (variant == 100) {
    Require(gufo::models::qwen38_flash_next::Iq4WideVariant() == 4,
            "iq4_wmma control requires GUFO_IQ4_WIDE_VARIANT=4 or unset");
    return qfn_mmq_iq4_dense_batch_preq(type, w, x, static_cast<float*>(y), s.m,
                                        n, s.k, nullptr);
  }
  if (type == 8)
    return qfn_mmq_q8_0_dense_vec_preq(w, nullptr, x, static_cast<float*>(y),
                                       s.m, n, s.k, nullptr);
  return qfn_mmq_dense_vec_preq_variant(
      type, w, x, static_cast<const int32_t*>(ids), static_cast<float*>(y), s.m,
      n, s.k, variant, nullptr);
}
std::vector<float> InputHost(int n, int k, float amplitude) {
  std::vector<float> host(std::size_t(n) * k);
  std::uint32_t seed = 38127;
  for (auto& v : host)
    v = float(int(Next(seed) % 2001) - 1000) / 997.0F * amplitude;
  return host;
}
void Inputs(Buffer& x, Buffer& ids, Shape s, int n, float amplitude = 1.0F) {
  const auto host = InputHost(n, s.k, amplitude);
  Buffer raw(host.size() * sizeof(float));
  Check(hipMemcpy(raw.p, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  Require(qfn_mmq_quantize_q8_1(static_cast<float*>(raw.p), x.p, n, s.k,
                                nullptr) == 0,
          "quantize");
  Check(hipMemset(ids.p, 0, n * sizeof(int32_t)));
  Check(hipDeviceSynchronize());
}
void WeightsAt(void* base, std::size_t bytes, int type) {
  const auto block = std::size_t(Bytes(type));
  std::vector<std::uint8_t> chunk((8U << 20) / block * block);
  std::uint32_t seed = 92817U + type;
  for (auto& v : chunk)
    v = std::uint8_t(Next(seed));
  for (std::size_t i = 0; i < chunk.size(); i += block) {
    const auto scale = i + (type == 14 ? 208 : 0);
    chunk[scale] = 0;
    chunk[scale + 1] = 0x20;
    if (type == 12 || type == 13 || type == 7) {
      chunk[i + 2] = 0;
      chunk[i + 3] = 0x1C;
    }
  }
  for (std::size_t i = 0; i < bytes; i += chunk.size())
    Check(hipMemcpy(static_cast<char*>(base) + i, chunk.data(),
                    std::min(chunk.size(), bytes - i), hipMemcpyHostToDevice));
}
void Weights(Buffer& w, std::size_t bytes, int type) {
  WeightsAt(w.p, bytes, type);
}
double TimeCall(const std::function<int(int)>& call, Buffer& eviction,
                std::size_t evict_bytes, int sets) {
  // One event interval over distinct matrices; eviction is outside the timer.
  Require(call(0) == 0, "warm launch");
  Check(ExpertBenchEvict(eviction.p, evict_bytes));
  Check(hipDeviceSynchronize());
  hipEvent_t start, stop;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&stop));
  Check(hipEventRecord(start, nullptr));
  for (int i = 0; i < sets; ++i)
    Require(call(i) == 0, "dense launch");
  Check(hipEventRecord(stop, nullptr));
  Check(hipEventSynchronize(stop));
  float ms = 0;
  Check(hipEventElapsedTime(&ms, start, stop));
  Check(hipEventDestroy(start));
  Check(hipEventDestroy(stop));
  return double(ms) * 1000.0 / sets;
}
double Time(int type, int variant, Buffer& w, Buffer& x, Buffer& ids, Buffer& y,
            Buffer& eviction, Shape s, int n, int sets) {
  return TimeCall(
      [&](int i) {
        return Call(
            type, variant,
            static_cast<char*>(w.p) + std::size_t(i) * WeightBytes(type, s),
            x.p, ids.p, y.p, s, n);
      },
      eviction, 64U << 20, sets);
}
void BenchWidths(int type, const std::vector<int>& variants, Buffer& weights,
                 Buffer& input, Buffer& ids, Buffer& output, Buffer& eviction,
                 Shape s, int sets, int shape,
                 std::map<std::pair<int, int>, double>& q8) {
  const auto bytes = WeightBytes(type, s);
  std::map<int, double> width_one;
  for (int n = 1; n <= 8; ++n) {
    double existing = 0;
    for (int variant : type == 8 ? std::vector<int>{0} : variants) {
      if (!Eligible(type, variant))
        continue;
      const double us = Time(type, variant, weights, input, ids, output,
                             eviction, s, n, sets);
      const double gbps = double(bytes) / us / 1000.0;
      if (!(us > 0) || gbps > 256) {
        std::printf("fmt=%s shape=%s N=%d variant=%s discarded_above_peak\n",
                    Format(type), s.name, n, Name(variant));
        continue;
      }
      if (variant == 0)
        existing = us;
      if (n == 1)
        width_one[variant] = us;
      if (type == 8)
        q8[{shape, n}] = us;
      const double control = q8.contains({shape, n}) ? q8.at({shape, n}) : 0;
      std::printf(
          "fmt=%s shape=%s K=%d M=%d N=%d variant=%s sets=%d us=%.3f "
          "GB/s=%.2f ratio_existing=%.4f ratio_Q8=%.4f "
          "ratio_reuse1_N1=%.4f ratio_variant_N1=%.4f\n",
          Format(type), s.name, s.k, s.m, n, Name(variant), sets, us, gbps,
          existing > 0 ? us / existing
                       : std::numeric_limits<double>::quiet_NaN(),
          control > 0 ? us / control : std::numeric_limits<double>::quiet_NaN(),
          width_one.contains(type == 8 ? 0 : 1)
              ? us / width_one.at(type == 8 ? 0 : 1)
              : std::numeric_limits<double>::quiet_NaN(),
          width_one.contains(variant)
              ? us / width_one.at(variant)
              : std::numeric_limits<double>::quiet_NaN());
      std::fflush(stdout);
    }
  }
}
int Sets(std::size_t bytes) {
  // Whole banks of 48 layers; small shapes get more banks to exceed cache.
  return 48 * std::max(1, int(((128U << 20) + 48 * bytes - 1) / (48 * bytes)));
}
void Bench(const std::vector<int>& variants) {
  std::map<std::pair<int, int>, double> q8;
  Buffer eviction(64U << 20);
  Check(hipMemset(eviction.p, 0, 64U << 20));
  for (int type : {8, 23, 20, 12, 13, 7, 14, 21, 18, 17}) {
    for (int shape = 0; shape < int(std::size(shapes)); ++shape) {
      const auto s = shapes[shape];
      if (s.k % Block(type)) {
        std::printf("fmt=%s shape=%s K=%d M=%d unsupported_K\n", Format(type),
                    s.name, s.k, s.m);
        continue;
      }
      const auto bytes = WeightBytes(type, s);
      const int sets = Sets(bytes);
      Buffer weights(sets * bytes), input(qfn_mmq_q8_1_bytes(8, s.k));
      Buffer ids(8 * sizeof(int32_t)),
          output(std::size_t(8) * s.m * sizeof(float));
      Weights(weights, sets * bytes, type);
      Inputs(input, ids, s, 8);
      BenchWidths(type, variants, weights, input, ids, output, eviction, s,
                  sets, shape, q8);
    }
  }
}
void BenchReal(const std::string& path, const std::vector<int>& variants) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(bool(reader), error.c_str());
  using Key = std::tuple<int, int, int>;
  std::map<Key, int> counts;
  const auto eligible = [](const auto& tensor) {
    const int type = int(tensor.type);
    return tensor.dimensions.size() == 2 && tensor.name.starts_with("blk.") &&
           (d::IsIqTarget(type) || type == 20 || type == 23);
  };
  for (const auto& tensor : reader->GetTensors())
    if (eligible(tensor))
      ++counts[{int(tensor.type), int(tensor.dimensions[0]),
                int(tensor.dimensions[1])}];
  Buffer eviction(64U << 20);
  Check(hipMemset(eviction.p, 0, 64U << 20));
  std::set<Key> seen;
  std::map<std::pair<int, int>, double> q8;
  for (const auto& tensor : reader->GetTensors()) {
    if (!eligible(tensor))
      continue;
    const int type = int(tensor.type);
    const Key key{type, int(tensor.dimensions[0]), int(tensor.dimensions[1])};
    if (!seen.insert(key).second)
      continue;
    const std::string name(tensor.name);
    const Shape s{name.c_str(), int(tensor.dimensions[0]),
                  int(tensor.dimensions[1])};
    const auto bytes = WeightBytes(type, s);
    Require(s.k % Block(type) == 0 && bytes <= tensor.size_bytes,
            "unsupported real benchmark shape");
    const int sets = Sets(bytes);
    Buffer weights(sets * bytes), input(qfn_mmq_q8_1_bytes(8, s.k));
    Buffer ids(8 * sizeof(int32_t)),
        output(std::size_t(8) * s.m * sizeof(float));
    // Same real payload, physically disjoint banks. No address is reused in
    // one timed interval. Other tensors with this geometry are byte-checked.
    for (int i = 0; i < sets; ++i)
      Check(hipMemcpy(static_cast<char*>(weights.p) + std::size_t(i) * bytes,
                      tensor.data, bytes, hipMemcpyHostToDevice));
    Inputs(input, ids, s, 8);
    std::printf(
        "real_file=%s representative=%s tensors_with_shape=%d "
        "cold_disjoint_copies=%d\n",
        path.c_str(), name.c_str(), counts.at(key), sets);
    BenchWidths(type, variants, weights, input, ids, output, eviction, s, sets,
                0, q8);
  }
  std::printf(
      "real_file=%s unique_IQ_dense_shapes=%zu (IQ2_XS/IQ3_XXS expert slices "
      "are proxies only)\n",
      path.c_str(), seen.size());
}

// Phase 0: every actual projection, including the full vocabulary head and
// native BF16 HC matrices. A timed interval visits physically distinct copies
// of the actual matrix, once each. --budget-sets 1 retains the old protocol.
void BudgetReal(const std::string& path, const std::string& twin_path = {},
                int sets = 16, bool hc_only = false, bool kquant_only = false) {
  namespace r = gufo::models::qwen38_flash_next::rocm;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(bool(reader), error.c_str());
  std::unique_ptr<gufo::core::GgufReader> twin;
  if (!twin_path.empty()) {
    twin = gufo::core::GgufReader::OpenFile(twin_path, &error);
    Require(bool(twin), error.c_str());
  }
  Buffer eviction(256U << 20);
  Check(hipMemset(eviction.p, 0, 256U << 20));
  const bool fused_hc = gufo::platform::PlatformTuning().fused_hc_down;
  qfn_mmq_set_hc_down_deep_prefetch(fused_hc ? 1 : 0);
  std::size_t covered = 0;
  for (const auto& original : reader->GetTensors()) {
    const auto* mirror = twin ? twin->FindTensor(original.name) : nullptr;
    if (mirror)
      Require(
          int(original.type) == 8 && original.dimensions == mirror->dimensions,
          "invalid Q8 twin mirror");
    const auto& t = mirror ? *mirror : original;
    if (t.dimensions.size() != 2 ||
        t.name.find("token_embd") != std::string::npos ||
        t.name.find("conv1d") != std::string::npos)
      continue;
    const std::string name(t.name);
    const int type = int(t.type);
    if (kquant_only &&
        (!d::IsKQuantTarget(type) ||
         !d::KQuantMatrixShape(int(t.dimensions[0]), int(t.dimensions[1]))))
      continue;
    if (hc_only &&
        (type != 30 || !gufo::models::qwen38_flash_next::hc_bf16_dense::Shape(
                           2, int(t.dimensions[1]), int(t.dimensions[0]))))
      continue;
    if (type == 0 && name.find("hc_") != std::string::npos &&
        name.find("inject") != std::string::npos) {
      std::printf("BUDGET_SKIP name=%s reason=fused_hc_inject_not_projection\n",
                  name.c_str());
      continue;
    }
    if (type != 0 && type != 1 && type != 30 && type != 8 &&
        d::FormatIndex(type) < 0) {
      std::printf(
          "BUDGET_SKIP name=%s type=%d reason=unsupported_projection_type\n",
          name.c_str(), type);
      continue;
    }
    const Shape s{name.c_str(), int(t.dimensions[0]), int(t.dimensions[1])};
    const bool small = type == 0 || type == 1 || type == 30;
    const bool hc_down = fused_hc && type == 8 && s.k == 10240 && s.m == 320 &&
                         name.find("hc_down") != std::string::npos;
    Require(t.size_bytes <= std::numeric_limits<std::size_t>::max() / sets,
            "budget weight allocation overflow");
    Buffer w(t.size_bytes * sets), raw(std::size_t(8) * s.k * sizeof(float)),
        x(qfn_mmq_q8_1_bytes(8, s.k)), ids(8 * sizeof(int32_t)),
        y(std::size_t(8) * s.m * sizeof(float));
    for (int i = 0; i < sets; ++i)
      Check(hipMemcpy(static_cast<char*>(w.p) + i * t.size_bytes, t.data,
                      t.size_bytes, hipMemcpyHostToDevice));
    // Raw rows and the prequantized view have identical deterministic input
    // at every width. Quantization is deliberately outside projection timing.
    std::vector<float> host(std::size_t(8) * s.k);
    std::uint32_t seed = 38127;
    for (auto& v : host)
      v = float(int(Next(seed) % 2001) - 1000) / 997.0F;
    Check(hipMemcpy(raw.p, host.data(), host.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    if (!small)
      Inputs(x, ids, s, 8);
    for (int n = 1; n <= 8; ++n) {
      const bool hc_candidate =
          type == 30 &&
          gufo::models::qwen38_flash_next::hc_bf16_dense::Enabled() &&
          gufo::models::qwen38_flash_next::hc_bf16_dense::Shape(n, s.m, s.k);
      for (int sample = 0; sample < 9; ++sample) {
        const double us = TimeCall(
            [&](int matrix) {
              const void* weights =
                  static_cast<const char*>(w.p) + matrix * t.size_bytes;
              if (small) {
                r::SmallGemm(weights, static_cast<r::WeightType>(type),
                             static_cast<const float*>(raw.p),
                             static_cast<float*>(y.p), n, s.m, s.k, nullptr);
                return int(hipGetLastError());
              }
              if (hc_down)
                return qfn_mmq_q8_0_hc_down_silu(
                    weights, x.p, static_cast<float*>(y.p), n, 0.25F, nullptr);
              if (type == 8)
                return qfn_mmq_q8_0_dense_vec_preq(weights, nullptr, x.p,
                                                   static_cast<float*>(y.p),
                                                   s.m, n, s.k, nullptr);
              return qfn_mmq_dense_vec_preq(
                  type, weights, x.p, static_cast<const int32_t*>(ids.p),
                  static_cast<float*>(y.p), s.m, n, s.k, nullptr);
            },
            eviction, 256U << 20, sets);
        const double gbps = double(t.size_bytes) / us / 1000;
        std::printf(
            "BUDGET name=%s fmt=%s K=%d M=%d N=%d sample=%d us=%.6f GBps=%.3f "
            "valid=%d scope=isolated_projection allocation=hipMalloc "
            "input=synthetic weights=real quantization=excluded kernel=%s "
            "evict_bytes=268435456 sets=%d protocol=%s routing=kquant-from2 "
            "prod_variant=%s bf16_hc_selected=%d\n",
            name.c_str(), Format(type), s.k, s.m, n, sample, us, gbps,
            int(us > 0 && gbps <= 256),
            hc_down        ? "hc_down_silu"
            : hc_candidate ? "bf16_hc_vector_split"
                           : "projection",
            sets, sets == 1 ? "single-call" : "distinct-matrices",
            type == 8 || small
                ? "shape-dispatch"
                : d::VariantName(d::SelectVariant(
                      type, n, s.k, s.m,
                      d::ParseVariant(std::getenv("GUFO_DENSE_IQ_VARIANT")))),
            int(hc_candidate));
        std::fflush(stdout);
      }
    }
    ++covered;
  }
  std::printf(
      "BUDGET_SUMMARY projections=%zu coverage=all_supported_rank2 "
      "excludes=gathers,convolutions,norms,fusion,producer_quantization\n",
      covered);
}
void Compare(const std::string& path, const std::vector<int>& variants) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(bool(reader), error.c_str());
  std::size_t checked = 0, expert_slices = 0, skipped = 0;
  for (const auto& tensor : reader->GetTensors()) {
    // Native shards also contain norms, gathers and routed expert stacks.
    // IQ2_XS/IQ3_XXS have no dense tensors in the published rungs; use one
    // real expert matrix per such tensor, explicitly labelled as a slice.
    const int type = int(tensor.type);
    const bool slice =
        tensor.dimensions.size() == 3 && (type == 17 || type == 18);
    if (d::FormatIndex(type) < 0 || (tensor.dimensions.size() != 2 && !slice))
      continue;
    const Shape s{"real", int(tensor.dimensions[0]), int(tensor.dimensions[1])};
    // Hashed-row lookup reads this table; production never multiplies it.
    if (tensor.name == "per_layer_token_embd.weight") {
      ++skipped;
      std::printf(
          "skipped %s type=%d fmt=%s M=%d K=%d reason=PLE embedding lookup, "
          "not a production dense projection\n",
          std::string(tensor.name).c_str(), type, Format(type), s.m, s.k);
      std::fflush(stdout);
      continue;
    }
    const auto weight_bytes = WeightBytes(type, s);
    Require(s.k % Block(type) == 0 && weight_bytes <= tensor.size_bytes,
            "unsupported real tensor shape");
    Buffer w(weight_bytes), x(qfn_mmq_q8_1_bytes(8, s.k)),
        ids(8 * sizeof(int32_t));
    Buffer y(std::size_t(8) * s.m * sizeof(float));
    Check(hipMemcpy(w.p, tensor.data, weight_bytes, hipMemcpyHostToDevice));
    Inputs(x, ids, s, 8);
    const auto launch = [&](int variant, int n) {
      const int result = Call(type, variant, w.p, x.p, ids.p, y.p, s, n);
      if (result == 0)
        return;
      const std::string reason =
          std::string(variant ? "new kernel" : "old kernel") +
          " tensor=" + std::string(tensor.name) +
          " type=" + std::to_string(type) + " fmt=" + Format(type) +
          " M=" + std::to_string(s.m) + " K=" + std::to_string(s.k) +
          " N=" + std::to_string(n) + " variant=" + Name(variant) +
          " return=" + std::to_string(result);
      Require(result == 0, reason.c_str());
    };
    for (int n = 1; n <= 8; ++n) {
      const auto bytes = std::size_t(n) * s.m * sizeof(float);
      std::vector<std::uint8_t> old(bytes), now(bytes);
      const int production = int(d::ProductionVariant(type, n, s.k));
      launch(production, n);
      Check(hipMemcpy(old.data(), y.p, bytes, hipMemcpyDeviceToHost));
      for (int v : variants) {
        if (v == production || !Eligible(type, v))
          continue;
        // An unwritten output must not retain the baseline's bytes.
        Check(hipMemset(y.p, 0xFF, bytes));
        launch(v, n);
        Check(hipMemcpy(now.data(), y.p, bytes, hipMemcpyDeviceToHost));
        if (old != now) {
          std::fprintf(
              stderr, "Byte mismatch: %s type=%d M=%d K=%d N=%d variant=%s\n",
              std::string(tensor.name).c_str(), type, s.m, s.k, n, Name(v));
          throw std::runtime_error("real tensor outputs differ");
        }
      }
    }
    ++checked;
    expert_slices += slice;
    std::printf("%s fmt=%s%s widths=1..8 identical\n",
                std::string(tensor.name).c_str(), Format(type),
                slice ? " expert=0" : "");
    std::fflush(stdout);
  }
  Require(checked + skipped > 0, "no supported real tensors");
  std::printf(
      "PASS: %s checked=%zu dense=%zu expert_slices=%zu skipped=%zu "
      "widths=1..8\n",
      path.c_str(), checked, checked - expert_slices, expert_slices, skipped);
  std::fflush(stdout);
}
#include "ablate.hpp"
#include "hc_bf16_compare.hpp"
#include "kquant_compare.hpp"
}  // namespace
int main(int argc, char** argv) try {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts(
        "--bench | --bench-real WEIGHTS.gguf [...] | --budget-real "
        "WEIGHTS.gguf [...] | --budget-twin TRUNK.gguf TWIN.gguf | --compare "
        "WEIGHTS.gguf [...]\n"
        "--bench-hc WEIGHTS.gguf [...] | --compare-hc WEIGHTS.gguf [...]\n"
        "--bench-kquant WEIGHTS.gguf [...] | --compare-kquant WEIGHTS.gguf "
        "[...] | --compare-dot4 WEIGHTS.gguf [...]\n"
        "GUFO_DENSE_BENCH_VARIANTS=baseline,reuse1,reuse2,reuse1_cap6,reuse1_"
        "lds,"
        "reuse1_ldsgrid,reuse1_splitk_lds,kquant_wmma,dot4,iq4_wmma\n"
        "IQ2_XS/IQ3_XXS expert slices are dense-kernel proxies; iq4_wmma is "
        "the existing batch entry point.\n"
        "Budget defaults to 16 distinct matrices/interval. Append "
        "--budget-sets 1 for the historical single-call column, or 16..64.\n"
        "--ablate --fmt FMT --kernel K[,K...] (--shape KxM ... | --gguf FILE) "
        "[--widths 1-8|4,8] [--sets 16] [--samples 9] [--occupancy R | "
        "--extra-lds BYTES] [--max-gib 6]\n"
        "  kernels: production existing reuse1 reuse2 reuse1_cap6 reuse1_lds "
        "reuse1_ldsgrid reuse1_splitk_lds kquant_wmma dot4 "
        "stage:A|A0|B|B0|C1|C2|C3|D "
        "reuse1_loads q8_dp4a q8_wmma iq4_wmma\n"
        "--ablate-selfcheck (no timing; one tiny cell per kernel and stage)\n"
        "--ablate selects kernels by flags only and refuses selection "
        "environment variables.");
    return 0;
  }
  Require(argc >= 2, "use --help");
  if (std::string(argv[1]) == "--ablate")
    return AblRun(argc, argv);
  if (std::string(argv[1]) == "--ablate-selfcheck")
    return AblSelfcheck();
  int budget_sets = 16;
  if (argc >= 4 && std::string(argv[argc - 2]) == "--budget-sets") {
    const std::string value = argv[argc - 1];
    std::size_t consumed = 0;
    budget_sets = std::stoi(value, &consumed);
    Require(consumed == value.size() &&
                (budget_sets == 1 || (budget_sets >= 16 && budget_sets <= 64)),
            "budget sets must be 1 or 16..64");
    argc -= 2;
    Require(std::string(argv[1]) == "--budget-real" ||
                std::string(argv[1]) == "--budget-twin" ||
                std::string(argv[1]) == "--bench-hc" ||
                std::string(argv[1]) == "--bench-kquant",
            "--budget-sets is only for budget modes");
  }
  const auto variants = Variants();
  Check(hipSetDevice(0));
  if (argc == 2 && std::string(argv[1]) == "--bench")
    Bench(variants);
  else if (argc >= 3 && std::string(argv[1]) == "--bench-real") {
    for (int i = 2; i < argc; ++i)
      BenchReal(argv[i], variants);
  } else if (argc == 4 && std::string(argv[1]) == "--budget-twin") {
    BudgetReal(argv[2], argv[3], budget_sets);
  } else if (argc >= 3 && (std::string(argv[1]) == "--budget-real" ||
                           std::string(argv[1]) == "--bench-hc" ||
                           std::string(argv[1]) == "--bench-kquant")) {
    for (int i = 2; i < argc; ++i)
      BudgetReal(argv[i], {}, budget_sets, std::string(argv[1]) == "--bench-hc",
                 std::string(argv[1]) == "--bench-kquant");
  } else if (argc >= 3 && (std::string(argv[1]) == "--compare" ||
                           std::string(argv[1]) == "--compare-hc" ||
                           std::string(argv[1]) == "--compare-kquant" ||
                           std::string(argv[1]) == "--compare-dot4")) {
    bool failed = false;
    for (int i = 2; i < argc; ++i) {
      try {
        if (std::string(argv[1]) == "--compare-hc")
          CompareHcBf16(argv[i]);
        else if (std::string(argv[1]) == "--compare-kquant")
          CompareKQuant(argv[i]);
        else if (std::string(argv[1]) == "--compare-dot4")
          CompareKQuant(argv[i], int(d::Variant::kDot4));
        else
          Compare(argv[i], variants);
      } catch (const std::exception& e) {
        std::fflush(stdout);
        std::fprintf(stderr, "FAIL %s: %s\n", argv[i], e.what());
        failed = true;
      }
    }
    return failed ? 1 : 0;
  } else
    throw std::runtime_error("use --help");
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
