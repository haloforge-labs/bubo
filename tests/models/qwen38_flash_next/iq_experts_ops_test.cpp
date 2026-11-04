// Routed expert projections for IQ weight formats (unsloth UD-IQ4_XS:
// IQ3_S/IQ4_XS gate and up, IQ4_NL down) against a double-precision
// reference built from the core CPU dequantizers, which are independent of
// the vendored device code. Covers the decode vector path (single and paired
// projections, top-k and one-slot-per-row layouts) and the tiled MMQ path.

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

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

using gufo::core::GgmlType;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

struct Format {
  const char* name;
  GgmlType type;
  std::size_t block;
  std::size_t bytes;
};

constexpr Format kIQ3_S{"IQ3_S", GgmlType::kIQ3_S, 256, 110};
constexpr Format kIQ4_XS{"IQ4_XS", GgmlType::kIQ4_XS, 256, 136};
constexpr Format kIQ4_NL{"IQ4_NL", GgmlType::kIQ4_NL, 32, 18};
// ISTA GSQ-RCO IQ3_S rung (memory lane): IQ3_XXS and IQ2_S gate/up, Q2_0 down.
constexpr Format kIQ3_XXS{"IQ3_XXS", GgmlType::kIQ3_XXS, 256, 98};
constexpr Format kIQ2_S{"IQ2_S", GgmlType::kIQ2_S, 256, 82};
constexpr Format kIQ2_XXS{"IQ2_XXS", GgmlType::kIQ2_XXS, 256, 66};
constexpr Format kIQ2_XS{"IQ2_XS", GgmlType::kIQ2_XS, 256, 74};
constexpr Format kIQ1_M{"IQ1_M", GgmlType::kIQ1_M, 256, 56};
constexpr Format kQ2_0{"Q2_0", GgmlType::kQ2_0, 64, 18};
// K-quants as routed experts (community quants put Q3_K/Q6_K on them).
constexpr Format kQ3_K{"Q3_K", GgmlType::kQ3_K, 256, 110};
constexpr Format kQ6_K{"Q6_K", GgmlType::kQ6_K, 256, 210};

/// Byte offset of the fp16 super-block scale `d`: the first two bytes of every
/// block, except Q3_K and Q6_K, which store it in the last two.
std::size_t ScaleOffset(const Format& f) {
  if (f.type == GgmlType::kQ3_K)
    return 108;
  if (f.type == GgmlType::kQ6_K)
    return 208;
  return 0;
}

/// Random but valid blocks: every byte pattern decodes, so only the fp16
/// super-block scale (see ScaleOffset) is controlled.
struct Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<float> values;  // [experts * rows][cols]
};

Weights MakeWeights(const Format& f, std::size_t matrices, std::size_t cols,
                    std::uint32_t seed) {
  Weights w;
  const std::size_t row_bytes = cols / f.block * f.bytes;
  w.blocks.resize(matrices * row_bytes);
  for (auto& byte : w.blocks)
    byte = static_cast<std::uint8_t>(NextRandom(&seed));
  for (std::size_t b = 0; b < w.blocks.size() / f.bytes; ++b) {
    const float scale =
        0.002F + 0.004F * static_cast<float>(NextRandom(&seed) % 1000) / 1000.F;
    const __half d = __float2half(scale);
    if (f.type == GgmlType::kIQ1_M) {
      std::uint16_t half;
      std::memcpy(&half, &d, sizeof(half));
      std::uint16_t scales[4];
      std::memcpy(scales, w.blocks.data() + b * f.bytes + 48, sizeof(scales));
      for (int i = 0; i < 4; ++i)
        scales[i] = (scales[i] & 0x0FFF) | (((half >> (4 * i)) & 15) << 12);
      std::memcpy(w.blocks.data() + b * f.bytes + 48, scales, sizeof(scales));
    } else {
      std::memcpy(w.blocks.data() + b * f.bytes + ScaleOffset(f), &d,
                  sizeof(d));
    }
  }
  w.values.resize(matrices * cols);
  for (std::size_t r = 0; r < matrices; ++r) {
    const auto* src = w.blocks.data() + r * row_bytes;
    float* dst = w.values.data() + r * cols;
    switch (f.type) {
      case GgmlType::kIQ3_S:
        gufo::quant::DequantizeIQ3_S(src, dst, cols);
        break;
      case GgmlType::kIQ4_XS:
        gufo::quant::DequantizeIQ4_XS(src, dst, cols);
        break;
      case GgmlType::kIQ3_XXS:
        gufo::quant::DequantizeIQ3_XXS(src, dst, cols);
        break;
      case GgmlType::kIQ2_S:
        gufo::quant::DequantizeIQ2_S(src, dst, cols);
        break;
      case GgmlType::kIQ2_XXS:
        gufo::quant::DequantizeIQ2_XXS(src, dst, cols);
        break;
      case GgmlType::kIQ2_XS:
        gufo::quant::DequantizeIQ2_XS(src, dst, cols);
        break;
      case GgmlType::kIQ1_M:
        gufo::quant::DequantizeIQ1_M(src, dst, cols);
        break;
      case GgmlType::kQ2_0:
        gufo::quant::DequantizeQ2_0(src, dst, cols);
        break;
      case GgmlType::kQ3_K:
        gufo::quant::DequantizeQ3_K(src, dst, cols);
        break;
      case GgmlType::kQ6_K:
        gufo::quant::DequantizeQ6_K(src, dst, cols);
        break;
      case GgmlType::kIQ4_NL:
        gufo::quant::DequantizeIQ4_NL(src, dst, cols);
        break;
      default:
        throw std::logic_error(std::string("no reference dequantizer for ") +
                               f.name);
    }
  }
  return w;
}

template<typename T>
T* Upload(const std::vector<T>& host, const char* what) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), what);
  CheckHip(hipMemset(device, 0, host.size() * sizeof(T) + 4096), what);
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           what);
  return device;
}

/// out[(t * used + s) * rows + r] = W[ids[t * used + s]][r] . x[t or slot].
/// With `per_slot`, each (t, s) row has its own input, as for the down
/// projection, which the executor runs with n_used == 1.
double Compare(const std::vector<float>& actual, const Weights& w,
               const std::vector<float>& x, const std::vector<int>& ids,
               int rows, int cols, int tokens, int used, bool per_slot) {
  double max_error = 0, max_reference = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int s = 0; s < used; ++s) {
      const int slot = t * used + s;
      const int expert = ids[slot];
      const float* input = x.data() + (per_slot ? slot : t) * cols;
      for (int r = 0; r < rows; ++r) {
        const float* weight =
            w.values.data() + (std::size_t(expert) * rows + r) * cols;
        double expected = 0;
        for (int k = 0; k < cols; ++k)
          expected += double(weight[k]) * input[k];
        const double got = actual[std::size_t(slot) * rows + r];
        if (!std::isfinite(got))
          return INFINITY;
        max_error = std::max(max_error, std::abs(got - expected));
        max_reference = std::max(max_reference, std::abs(expected));
      }
    }
  }
  return max_reference > 0 ? max_error / max_reference : max_error;
}

int FormatId(const Format& f) {
  return static_cast<int>(f.type);
}

int Raw(const Format& f, const void* w, const float* x, const std::int32_t* ids,
        float* out, int m, int k, int t, int e, int u) {
  switch (f.type) {
    case GgmlType::kIQ3_S:
      return qfn_mmq_iq3_s_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ4_XS:
      return qfn_mmq_iq4_xs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ3_XXS:
      return qfn_mmq_iq3_xxs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ2_S:
      return qfn_mmq_iq2_s_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ2_XXS:
      return qfn_mmq_iq2_xxs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ2_XS:
      return qfn_mmq_iq2_xs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ1_M:
      return qfn_mmq_iq1_m_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kQ2_0:
      return qfn_mmq_q2_0_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kQ3_K:
      return qfn_mmq_q3_K_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kQ6_K:
      return qfn_mmq_q6_K_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ4_NL:
      return qfn_mmq_iq4_nl_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    default:
      throw std::logic_error(std::string("no tiled kernel selected for ") +
                             f.name);
  }
}

/// Activations quantize to Q8_1 on the device; 2% of the largest output
/// covers that rounding with margin while catching layout or table errors.
constexpr double kTolerance = 2e-2;

bool Run(const Format& f, int rows, int cols, int tokens, int experts, int used,
         bool per_slot, bool tiled, bool paired, std::uint32_t seed) {
  const auto a = MakeWeights(f, std::size_t(experts) * rows, cols, seed);
  const auto b = MakeWeights(f, std::size_t(experts) * rows, cols, seed ^ 77U);
  const int inputs = per_slot ? tokens * used : tokens;
  std::vector<float> x(std::size_t(inputs) * cols);
  for (auto& v : x)
    v = static_cast<float>(int(NextRandom(&seed) % 2001) - 1000) / 1000.F;
  std::vector<int> ids(std::size_t(tokens) * used);
  for (int t = 0; t < tokens; ++t)
    for (int s = 0; s < used; ++s)
      ids[t * used + s] = (t * 7 + s * 3) % experts;

  void* dwa = Upload(a.blocks, "weights");
  void* dwb = Upload(b.blocks, "weights");
  float* dx = Upload(x, "input");
  auto* di = reinterpret_cast<std::int32_t*>(Upload(ids, "ids"));
  const std::size_t out_size = std::size_t(tokens) * used * rows;
  std::vector<float> zeros(out_size, 0.F);
  float* dya = Upload(zeros, "output");
  float* dyb = Upload(zeros, "output");

  // Per-slot layouts run as one input row per slot with one expert each.
  const int T = per_slot ? tokens * used : tokens;
  const int U = per_slot ? 1 : used;
  int rc = 0;
  if (tiled) {
    qfn_mmq_set_routed_max_expert_rows(T * U);
    qfn_mmq_set_routed_tile_cols(32);
    rc = Raw(f, dwa, dx, di, dya, rows, cols, T, experts, U);
    if (rc == 0 && paired)
      rc = Raw(f, dwb, dx, di, dyb, rows, cols, T, experts, U);
  } else {
    rc = qfn_mmq_moe_vec(FormatId(f), dwa, dx, di, dya, rows, cols, T, experts,
                         U, nullptr, paired ? dwb : nullptr,
                         paired ? dyb : nullptr);
  }
  CheckHip(hipDeviceSynchronize(), "projection");
  std::vector<float> ya(out_size), yb(out_size);
  CheckHip(hipMemcpy(ya.data(), dya, out_size * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(yb.data(), dyb, out_size * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  for (void* p : {dwa, dwb, static_cast<void*>(dx), static_cast<void*>(di),
                  static_cast<void*>(dya), static_cast<void*>(dyb)})
    (void)hipFree(p);

  double error =
      rc == 0 ? Compare(ya, a, x, ids, rows, cols, tokens, used, per_slot)
              : INFINITY;
  if (paired && rc == 0)
    error = std::max(
        error, Compare(yb, b, x, ids, rows, cols, tokens, used, per_slot));
  const bool ok = error < kTolerance;
  std::cout << (ok ? "PASS " : "FAIL ") << f.name << (tiled ? " tiled" : " vec")
            << (paired ? " pair" : "") << (per_slot ? " per-slot" : "")
            << " rows=" << rows << " cols=" << cols << " tokens=" << tokens
            << " used=" << used << " rc=" << rc << " rel_error=" << error
            << '\n';
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    for (const Format* f : {&kIQ3_S, &kIQ4_XS, &kIQ3_XXS, &kIQ2_S,
                           &kIQ2_XXS, &kIQ2_XS, &kIQ1_M}) {
      // Gate/up shapes: hidden-width K, decode (1 and a draft of 4 tokens
      // with top-8), the paired decode route, and wide tiled prefill.
      ok = Run(*f, 64, 2560, 1, 16, 8, false, false, true, 0xA11CEU) && ok;
      ok = Run(*f, 64, 2560, 4, 16, 8, false, false, true, 0xB0B5U) && ok;
      ok = Run(*f, 64, 2560, 65, 16, 8, false, true, true, 0xC0FFEEU) && ok;
      ok = Run(*f, 96, 2560, 300, 16, 8, false, true, false, 0xD00DU) && ok;
    }
    // Down shapes: K = expert width, one input row per (token, slot).
    ok = Run(kIQ4_NL, 128, 640, 1, 16, 8, true, false, false, 0xE1U) && ok;
    ok = Run(kIQ4_NL, 128, 640, 3, 16, 8, true, false, false, 0xE2U) && ok;
    ok = Run(kIQ4_NL, 128, 640, 65, 16, 8, true, true, false, 0xE3U) && ok;
    ok = Run(kIQ4_NL, 2560, 640, 40, 16, 8, true, true, false, 0xE4U) && ok;
    // Q2_0 down: 640 = ten 64-value blocks per row.
    ok = Run(kQ2_0, 128, 640, 1, 16, 8, true, false, false, 0xF1U) && ok;
    ok = Run(kQ2_0, 128, 640, 3, 16, 8, true, false, false, 0xF2U) && ok;
    ok = Run(kQ2_0, 128, 640, 65, 16, 8, true, true, false, 0xF3U) && ok;
    ok = Run(kQ2_0, 2560, 640, 40, 16, 8, true, true, false, 0xF4U) && ok;
    // K-quant routed experts. Gate/up as above. The production expert width
    // (640) is not a multiple of 256, so a K-quant down tensor needs K = 512.
    for (const Format* f : {&kQ3_K, &kQ6_K}) {
      ok = Run(*f, 64, 2560, 1, 16, 8, false, false, true, 0x3A11U) && ok;
      ok = Run(*f, 64, 2560, 4, 16, 8, false, false, true, 0x3B0BU) && ok;
      ok = Run(*f, 64, 2560, 65, 16, 8, false, true, true, 0x3C0FU) && ok;
      ok = Run(*f, 96, 2560, 300, 16, 8, false, true, false, 0x3D0DU) && ok;
      // Large prefill: 1024 tokens x 8 used experts = 8192 slots.
      ok = Run(*f, 96, 2560, 1024, 16, 8, false, true, true, 0x3E0EU) && ok;
      ok = Run(*f, 128, 512, 1, 16, 8, true, false, false, 0x3F01U) && ok;
      ok = Run(*f, 128, 512, 3, 16, 8, true, false, false, 0x3F02U) && ok;
      ok = Run(*f, 128, 512, 65, 16, 8, true, true, false, 0x3F03U) && ok;
      ok = Run(*f, 2560, 512, 40, 16, 8, true, true, false, 0x3F04U) && ok;
      // Large down: 1024 tokens x 8 used experts = 8192 slots, one expert each.
      ok = Run(*f, 128, 512, 1024, 16, 8, true, true, false, 0x3F05U) && ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
