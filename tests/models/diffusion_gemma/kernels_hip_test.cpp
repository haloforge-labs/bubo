// DiffusionGemma GPU kernels against float64 CPU references on small random
// shapes: attention (causal with a window, bidirectional with split keys,
// head dimensions 256 and 512), expert products (BF16, Q8_0, FP8 and NVFP4
// weights), the dense product, the self-conditioning product, the
// sampler, and exact FP8 / Q8_0 dequantization. Exits 77 without a GPU.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/read.hpp"

namespace k = gufo::models::diffusion_gemma::kernels;
using bf16 = std::uint16_t;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void Hip(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

bf16 ToBf16(float x) {
  std::uint32_t bits;
  std::memcpy(&bits, &x, 4);
  return static_cast<bf16>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float ToFloat(bf16 x) {
  const std::uint32_t bits = static_cast<std::uint32_t>(x) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float RoundBf16(float x) {
  return ToFloat(ToBf16(x));
}

/// Device copy of a host vector, freed on scope exit.
template<typename T>
class Device {
public:
  explicit Device(const std::vector<T>& host) : count_(host.size()) {
    Hip(hipMalloc(&data_, std::max<std::size_t>(1, count_) * sizeof(T)));
    if (count_ > 0)
      Hip(hipMemcpy(data_, host.data(), count_ * sizeof(T),
                    hipMemcpyHostToDevice));
  }
  explicit Device(std::size_t count) : Device(std::vector<T>(count)) {}
  ~Device() { (void)hipFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* get() const { return data_; }
  std::vector<T> Read() const {
    std::vector<T> host(count_);
    Hip(hipDeviceSynchronize());
    Hip(hipMemcpy(host.data(), data_, count_ * sizeof(T),
                  hipMemcpyDeviceToHost));
    return host;
  }

private:
  T* data_{nullptr};
  std::size_t count_;
};

std::vector<bf16> RandomBf16(std::size_t count, float sigma, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, sigma);
  std::vector<bf16> values(count);
  for (auto& v : values)
    v = ToBf16(normal(rng));
  return values;
}

/// Largest |got - want| relative to max|want| (plus a floor).
double RelativeError(const std::vector<double>& want,
                     const std::vector<float>& got) {
  double worst = 0, scale = 1e-30;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::abs(want[i] - got[i]));
    scale = std::max(scale, std::abs(want[i]));
  }
  return worst / scale;
}

// ---------------------------------------------------------------- attention

struct AttentionCase {
  const char* name;
  int head_dim, q_heads, kv_heads, tokens, kv_len, q_pos0, k_pos0;
  bool causal;
  int window;
  bool split;
};

void Attention(const AttentionCase& c) {
  const int hd = c.head_dim, group = c.q_heads / c.kv_heads;
  const int capacity = c.kv_len + 16;
  const auto q = RandomBf16(std::size_t(c.tokens) * c.q_heads * hd, 1.0F, 1);
  // Keys scaled so that logits are of order one (as after the Q/K norms).
  const auto kc = RandomBf16(std::size_t(c.kv_heads) * capacity * hd,
                             1.0F / std::sqrt(static_cast<float>(hd)), 2);
  const auto vc = RandomBf16(std::size_t(c.kv_heads) * capacity * hd, 1.0F, 3);
  Device<bf16> dq(q), dk(kc), dv(vc), dout(q.size());
  constexpr int kMaxSplits = 16;
  Device<float> split_out(std::size_t(kMaxSplits) * c.tokens * c.q_heads * hd);
  Device<float> split_stats(std::size_t(kMaxSplits) * c.tokens * c.q_heads * 2);
  const k::SplitWorkspace workspace =
      c.split ? k::SplitWorkspace{.out = split_out.get(),
                                  .stats = split_stats.get(),
                                  .capacity = std::size_t(kMaxSplits) *
                                              c.tokens * c.q_heads * hd,
                                  .max_splits = kMaxSplits}
              : k::SplitWorkspace{};
  k::Attention(dq.get(), dk.get(), dv.get(), dout.get(), c.tokens, c.q_heads,
               c.kv_heads, hd, capacity, c.kv_len, c.q_pos0, c.k_pos0, c.causal,
               c.window, nullptr, workspace);
  const auto out = dout.Read();

  // P and the output are BF16 in the kernel: absolute errors of a few
  // BF16 steps of |V| (about 3), relative ones of a BF16 step or two.
  double worst_absolute = 0, worst_relative = 0;
  std::vector<double> score(c.kv_len), acc(hd);
  for (int t = 0; t < c.tokens; ++t)
    for (int h = 0; h < c.q_heads; ++h) {
      const int kvh = h / group, q_position = c.q_pos0 + t;
      const bf16* qr = &q[(std::size_t(t) * c.q_heads + h) * hd];
      double top = -INFINITY;
      for (int s = 0; s < c.kv_len; ++s) {
        const int k_position = c.k_pos0 + s;
        bool visible = true;
        if (c.causal)
          visible = visible && k_position <= q_position;
        if (c.window > 0)
          visible = visible && k_position > q_position - c.window;
        score[s] = -INFINITY;
        if (!visible)
          continue;
        const bf16* kr = &kc[(std::size_t(kvh) * capacity + s) * hd];
        double dot = 0;
        for (int d = 0; d < hd; ++d)
          dot += double(ToFloat(qr[d])) * ToFloat(kr[d]);
        score[s] = dot;
        top = std::max(top, dot);
      }
      std::fill(acc.begin(), acc.end(), 0.0);
      double sum = 0;
      for (int s = 0; s < c.kv_len; ++s) {
        if (score[s] == -INFINITY)
          continue;
        const double p = std::exp(score[s] - top);
        sum += p;
        const bf16* vr = &vc[(std::size_t(kvh) * capacity + s) * hd];
        for (int d = 0; d < hd; ++d)
          acc[d] += p * ToFloat(vr[d]);
      }
      const bf16* got = &out[(std::size_t(t) * c.q_heads + h) * hd];
      for (int d = 0; d < hd; ++d) {
        const double want = sum > 0 ? acc[d] / sum : 0.0;
        const double error = std::abs(ToFloat(got[d]) - want);
        worst_absolute = std::max(worst_absolute, error);
        if (std::abs(want) > 0.5)
          worst_relative = std::max(worst_relative, error / std::abs(want));
      }
    }
  std::printf("attention %-28s abs error %.2e  rel error %.2e\n", c.name,
              worst_absolute, worst_relative);
  Expect(worst_absolute < 1e-2 && worst_relative < 1e-2, c.name);
}

// ------------------------------------------------------------------ experts

enum class Weights { kBf16, kQ8, kFp8, kFp4 };

/// E4M3 (no NaN) to float, the reference decode.
float Fp8(std::uint8_t byte) {
  const int sign = byte >> 7, exponent = (byte >> 3) & 15, mantissa = byte & 7;
  const float magnitude =
      exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                    : std::ldexp(1.0F + mantissa / 8.0F, exponent - 7);
  return sign ? -magnitude : magnitude;
}

/// E2M1 (NVFP4 value, low 4 bits) to float.
float Fp4(unsigned code) {
  static const float kMagnitude[8] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
  return code & 8 ? -kMagnitude[code & 7] : kMagnitude[code & 7];
}

/// One expert matrix set in all four formats: rows x depth per expert.
struct ExpertMatrix {
  int experts, rows, depth;
  std::vector<bf16> bf16_values;
  std::vector<std::int8_t> q8_values;
  std::vector<std::uint16_t> q8_scales;  // FP16, one per 32 along depth.
  std::vector<std::uint8_t> fp8_values;
  std::vector<bf16> fp8_scales;          // One per row.
  std::vector<std::uint8_t> fp4_values;  // Two per byte, first in low bits.
  std::vector<std::uint8_t> fp4_blocks;  // E4M3, one per 16 along depth.
  std::vector<float> fp4_tensor;         // One per row (per matrix).

  ExpertMatrix(int e, int r, int d, unsigned seed)
      : experts(e), rows(r), depth(d) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.05F);
    std::uniform_int_distribution<int> q(-127, 127), byte(0, 255);
    const std::size_t n = std::size_t(e) * r * d;
    bf16_values.resize(n);
    for (auto& v : bf16_values)
      v = ToBf16(normal(rng));
    q8_values.resize(n);
    for (auto& v : q8_values)
      v = static_cast<std::int8_t>(q(rng));
    q8_scales.resize(n / 32);
    for (auto& s : q8_scales) {
      // FP16 scales in [2^-11, 2^-9): exponent 4..5, random mantissa.
      s = static_cast<std::uint16_t>(((4 + rng() % 2) << 10) | (rng() & 1023));
    }
    fp8_values.resize(n);
    for (auto& v : fp8_values) {
      v = static_cast<std::uint8_t>(byte(rng));
      if ((v & 0x7f) == 0x7f)
        v ^= 1;  // E4M3FN NaN codes are not weights.
    }
    fp8_scales.resize(std::size_t(e) * r);
    for (auto& s : fp8_scales)
      s = ToBf16(0.01F * (1.0F + static_cast<float>(rng() % 100) / 100.0F));
    fp4_values.resize(n / 2);
    for (auto& v : fp4_values)
      v = static_cast<std::uint8_t>(byte(rng));
    fp4_blocks.resize(n / 16);
    for (auto& b : fp4_blocks)
      b = static_cast<std::uint8_t>(0x20 + rng() % 0x30);  // Normal scales.
    fp4_tensor.resize(std::size_t(e) * r);
    for (int m = 0; m < e; ++m) {
      // One matrix scale per expert, as in the checkpoint.
      const float scale = 0.002F * (1.0F + static_cast<float>(m) / 8.0F);
      std::fill_n(fp4_tensor.begin() + std::size_t(m) * r, r, scale);
    }
  }

  static float Half(std::uint16_t h) {
    const int exponent = (h >> 10) & 31, mantissa = h & 1023;
    return std::ldexp(1.0F + mantissa / 1024.0F, exponent - 15);
  }

  /// Exact row . x for expert e, as the kernel's arithmetic contract
  /// defines it: Q8_0 sums 32-blocks and scales each; FP8 scales the row.
  double Dot(Weights format, int e, int row, const bf16* x) const {
    const std::size_t base = (std::size_t(e) * rows + row) * depth;
    double total = 0;
    for (int b = 0; b < depth; b += 32) {
      double block = 0;
      for (int i = b; i < b + 32; ++i) {
        double w = 0;
        switch (format) {
          case Weights::kBf16:
            w = ToFloat(bf16_values[base + i]);
            break;
          case Weights::kQ8:
            w = q8_values[base + i];
            break;
          case Weights::kFp8:
            w = Fp8(fp8_values[base + i]);
            break;
          case Weights::kFp4:
            w = double(Fp4(fp4_values[(base + i) / 2] >> (4 * (i % 2)))) *
                Fp8(fp4_blocks[(base + i) / 16]);
            break;
        }
        block += w * ToFloat(x[i]);
      }
      if (format == Weights::kQ8)
        block *= Half(q8_scales[(base + b) / 32]);
      total += block;
    }
    if (format == Weights::kFp8)
      total *= ToFloat(fp8_scales[std::size_t(e) * rows + row]);
    if (format == Weights::kFp4)
      total *= fp4_tensor[std::size_t(e) * rows + row];
    return total;
  }
};

double Gelu(double x) {
  return 0.5 * x *
         (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

void Experts(Weights format, const char* name, int group_rows) {
  constexpr int kExperts = 4, kHidden = 256, kInter = 128, kTokens = 80;
  // Expert 0 gets 150 rows (three 64-row or two 128-row groups), expert 2
  // none.
  std::vector<std::int32_t> counts = {150, 25, 0, 41};
  std::vector<std::int32_t> offsets(kExperts), source;
  std::mt19937 rng(11);
  for (int e = 0, at = 0; e < kExperts; ++e) {
    offsets[e] = at;
    for (int r = 0; r < counts[e]; ++r)
      source.push_back(static_cast<std::int32_t>(rng() % kTokens));
    at += counts[e];
  }
  const int assignments = static_cast<int>(source.size());
  const auto x = RandomBf16(std::size_t(kTokens) * kHidden, 1.0F, 12);
  const ExpertMatrix gate_up(kExperts, 2 * kInter, kHidden, 13);
  const ExpertMatrix down(kExperts, kHidden, kInter, 14);

  // Dispatch's group list: expert 0's groups, experts 1 and 3.
  std::vector<std::int32_t> groups;
  for (int first = 0; first < counts[0]; first += group_rows)
    groups.insert(groups.end(), {0, first});
  groups.insert(groups.end(), {1, 0, 3, 0});
  const std::vector<std::int32_t> group_count = {
      static_cast<std::int32_t>(groups.size() / 2)};
  Device<bf16> dx(x), act(std::size_t(assignments) * kInter),
      out(std::size_t(assignments) * kHidden);
  Device<std::int32_t> dcounts(counts), doffsets(offsets), dsource(source),
      dgroups(groups), dgroup_count(group_count);
  const k::ExpertRouting routing{.counts = dcounts.get(),
                                 .offsets = doffsets.get(),
                                 .source = dsource.get(),
                                 .groups = dgroups.get(),
                                 .group_count = dgroup_count.get(),
                                 .max_groups = group_count[0] + 1,
                                 .group_rows = group_rows};
  Device<bf16> gu_bf16(gate_up.bf16_values), dn_bf16(down.bf16_values);
  Device<std::int8_t> gu_q8(gate_up.q8_values), dn_q8(down.q8_values);
  Device<std::uint16_t> gu_q8s(gate_up.q8_scales), dn_q8s(down.q8_scales);
  Device<std::uint8_t> gu_fp8(gate_up.fp8_values), dn_fp8(down.fp8_values);
  Device<bf16> gu_fp8s(gate_up.fp8_scales), dn_fp8s(down.fp8_scales);
  Device<std::uint8_t> gu_fp4(gate_up.fp4_values), dn_fp4(down.fp4_values),
      gu_fp4b(gate_up.fp4_blocks), dn_fp4b(down.fp4_blocks);
  Device<float> gu_fp4t(gate_up.fp4_tensor), dn_fp4t(down.fp4_tensor);
  k::ExpertFp4 fp4_gu{}, fp4_dn{};
  k::ExpertQ8 q8_gu{}, q8_dn{};
  k::ExpertFp8 fp8_gu{}, fp8_dn{};
  if (format == Weights::kQ8) {
    q8_gu = {gu_q8.get(), gu_q8s.get()};
    q8_dn = {dn_q8.get(), dn_q8s.get()};
  } else if (format == Weights::kFp8) {
    fp8_gu = {gu_fp8.get(), gu_fp8s.get()};
    fp8_dn = {dn_fp8.get(), dn_fp8s.get()};
  } else if (format == Weights::kFp4) {
    fp4_gu = {gu_fp4.get(), gu_fp4b.get(), gu_fp4t.get()};
    fp4_dn = {dn_fp4.get(), dn_fp4b.get(), dn_fp4t.get()};
  }
  k::ExpertGateUp(dx.get(), gu_bf16.get(), routing, act.get(), kHidden, kInter,
                  nullptr, q8_gu, fp8_gu, fp4_gu);
  k::ExpertDown(act.get(), dn_bf16.get(), routing, out.get(), kHidden, kInter,
                nullptr, q8_dn, fp8_dn, fp4_dn);
  const auto got_act = act.Read();
  const auto got_out = out.Read();

  // Both outputs are BF16: compared within one BF16 step.
  double act_error = 0, down_error = 0;
  for (int e = 0; e < kExperts; ++e)
    for (int r = 0; r < counts[e]; ++r) {
      const int pos = offsets[e] + r;
      const bf16* xr = &x[std::size_t(source[pos]) * kHidden];
      for (int c = 0; c < kInter; ++c) {
        const double g = RoundBf16(float(gate_up.Dot(format, e, c, xr)));
        const double u =
            RoundBf16(float(gate_up.Dot(format, e, kInter + c, xr)));
        const double want = RoundBf16(float(Gelu(g))) * u;
        const double got = ToFloat(got_act[std::size_t(pos) * kInter + c]);
        act_error =
            std::max(act_error, std::abs(got - want) / (1e-3 + std::abs(want)));
      }
      // Down from the kernel's own activations, so only this product is
      // compared.
      for (int c = 0; c < kHidden; ++c) {
        const double want =
            down.Dot(format, e, c, &got_act[std::size_t(pos) * kInter]);
        const double got = ToFloat(got_out[std::size_t(pos) * kHidden + c]);
        down_error = std::max(down_error,
                              std::abs(got - want) / (1e-3 + std::abs(want)));
      }
    }
  std::printf(
      "experts %-5s %3d-row groups gate/up error %.2e  down error "
      "%.2e\n",
      name, group_rows, act_error, down_error);
  Expect(act_error < 2e-2, "expert gate/up");
  Expect(down_error < 1e-2, "expert down");
}

// --------------------------------------------------------- dense products

// One shape per kernel configuration: 256-row tiles (wide outputs), 128-row
// tiles with a ragged last tile, and a split reduction (one column tile).
void Linear(int rows, int n, int depth) {
  const auto a = RandomBf16(std::size_t(rows) * depth, 1.0F, 21);
  const auto w = RandomBf16(std::size_t(n) * depth, 0.05F, 22);
  std::vector<double> want(std::size_t(rows) * n);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; ++c) {
      double sum = 0;
      for (int i = 0; i < depth; ++i)
        sum += double(ToFloat(a[std::size_t(r) * depth + i])) *
               ToFloat(w[std::size_t(c) * depth + i]);
      want[std::size_t(r) * n + c] = sum;
    }
  Device<bf16> da(a), dw(w), wb(w.size());
  Device<float> out(want.size()),
      partials(
          std::max<std::size_t>(1, k::LinearPartialFloats(rows, n, depth)));
  k::BlockLinearWeights(dw.get(), wb.get(), n, depth, nullptr);
  k::Linear(da.get(), wb.get(), partials.get(), out.get(), rows, n, depth,
            nullptr);
  const double error = RelativeError(want, out.Read());
  std::printf("linear %dx%dx%d error %.2e\n", rows, n, depth, error);
  Expect(error < 1e-5, "linear");
}

// The MLP gate/up product with its activation, against float64 products
// rounded as the reference rounds them; a ragged 128-row tile.
void LinearGeluMul() {
  constexpr int kRows = 300, kInter = 192, kDepth = 256;
  const auto a = RandomBf16(std::size_t(kRows) * kDepth, 1.0F, 23);
  const auto w = RandomBf16(std::size_t(2 * kInter) * kDepth, 0.08F, 24);
  Device<bf16> da(a), dw(w), wb(w.size()), act(std::size_t(kRows) * kInter);
  k::BlockGateUpWeights(dw.get(), wb.get(), kInter, kDepth, nullptr);
  k::LinearGeluMul(da.get(), wb.get(), act.get(), kRows, kInter, kDepth,
                   nullptr);
  const auto got = act.Read();
  const auto dot = [&](int r, int c) {
    double sum = 0;
    for (int i = 0; i < kDepth; ++i)
      sum += double(ToFloat(a[std::size_t(r) * kDepth + i])) *
             ToFloat(w[std::size_t(c) * kDepth + i]);
    return sum;
  };
  double error = 0;
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kInter; ++c) {
      const double g = RoundBf16(float(dot(r, c)));
      const double u = RoundBf16(float(dot(r, kInter + c)));
      const double want = RoundBf16(float(Gelu(g))) * u;
      const double value = ToFloat(got[std::size_t(r) * kInter + c]);
      error = std::max(error, std::abs(value - want) / (1e-3 + std::abs(want)));
    }
  std::printf("linear gelu-mul %dx%dx%d error %.2e\n", kRows, kInter, kDepth,
              error);
  Expect(error < 2e-2, "linear gelu-mul");
}

void SoftEmbed() {
  constexpr int kRows = 256, kVocab = 2048, kHidden = 256;
  // Row-normalised probabilities, written in SampleRows' blocked layout.
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  std::vector<bf16> probs(std::size_t(kRows) * kVocab), blocked(probs.size());
  for (int r = 0; r < kRows; ++r) {
    double sum = 0;
    std::vector<float> row(kVocab);
    for (auto& v : row) {
      v = std::pow(uniform(rng), 8.0F);
      sum += v;
    }
    for (int i = 0; i < kVocab; ++i) {
      probs[std::size_t(r) * kVocab + i] = ToBf16(float(row[i] / sum));
      blocked[(std::size_t(i / 32) * kRows + r) * 32 + i % 32] =
          probs[std::size_t(r) * kVocab + i];
    }
  }
  const auto embed = RandomBf16(std::size_t(kVocab) * kHidden, 0.05F, 32);
  std::vector<double> want(std::size_t(kRows) * kHidden);
  for (int r = 0; r < kRows; ++r)
    for (int i = 0; i < kVocab; ++i) {
      const double p = ToFloat(probs[std::size_t(r) * kVocab + i]);
      for (int c = 0; c < kHidden; ++c)
        want[std::size_t(r) * kHidden + c] +=
            p * ToFloat(embed[std::size_t(i) * kHidden + c]);
    }
  Device<bf16> dp(blocked), de(embed);
  Device<float> out(want.size()), partials(8 * want.size());
  for (const int splits : {1, 4, 8}) {
    k::SoftEmbedProduct(dp.get(), de.get(), partials.get(), out.get(), kRows,
                        kVocab, kHidden, splits, nullptr);
    const double error = RelativeError(want, out.Read());
    std::printf("soft embedding splits=%d error %.2e\n", splits, error);
    Expect(error < 1e-5, "soft embedding");
  }
}

// ------------------------------------------------------------------ sampler

void Sampler() {
  constexpr int kRows = 4, kVocab = 4096;
  constexpr float kSoftcap = 30.0F, kTemperature = 0.7F;
  std::mt19937 rng(41);
  std::normal_distribution<float> normal(0.0F, 4.0F);
  std::vector<float> logits(std::size_t(kRows) * kVocab);
  for (auto& v : logits)
    v = normal(rng);
  logits[17] = 40.0F;  // A confident first row.
  Device<float> dl(logits), entropy(kRows);
  Device<bf16> probs(logits.size());
  Device<std::int32_t> argmax(kRows), sample(kRows);
  k::SampleRows(dl.get(), probs.get(), entropy.get(), argmax.get(),
                sample.get(), kRows, kVocab, kSoftcap, kTemperature, 5, {},
                nullptr);
  const auto got_entropy = entropy.Read();
  const auto got_argmax = argmax.Read();
  const auto got_sample = sample.Read();
  const auto got_probs = probs.Read();
  for (int r = 0; r < kRows; ++r) {
    std::vector<double> z(kVocab);
    int best = 0;
    for (int i = 0; i < kVocab; ++i) {
      z[i] = kSoftcap *
             std::tanh(double(logits[std::size_t(r) * kVocab + i]) / kSoftcap) /
             kTemperature;
      if (z[i] > z[best])
        best = i;
    }
    double sum = 0, weighted = 0;
    for (const double v : z) {
      sum += std::exp(v - z[best]);
      weighted += std::exp(v - z[best]) * (v - z[best]);
    }
    const double want_entropy = std::log(sum) - weighted / sum;
    double prob_sum = 0;
    for (int i = 0; i < kVocab; ++i)
      prob_sum +=
          ToFloat(got_probs[(std::size_t(i / 32) * kRows + r) * 32 + i % 32]);
    Expect(got_argmax[r] == best, "sampler argmax");
    Expect(std::abs(got_entropy[r] - want_entropy) <
               1e-4 * std::max(1.0, want_entropy),
           "sampler entropy");
    Expect(got_sample[r] >= 0 && got_sample[r] < kVocab, "sample in range");
    Expect(std::abs(prob_sum - 1.0) < 1e-2, "blocked probabilities sum to 1");
  }
  Expect(got_sample[0] == 17, "confident row samples its argmax");

  // Suppressing the confident token: it is never chosen and gets no
  // probability, and the entropy stays finite.
  k::SuppressedTokens suppressed;
  suppressed.ids[0] = 17;
  suppressed.count = 1;
  k::SampleRows(dl.get(), probs.get(), entropy.get(), argmax.get(),
                sample.get(), kRows, kVocab, kSoftcap, kTemperature, 5,
                suppressed, nullptr);
  const auto masked_entropy = entropy.Read();
  const auto masked_argmax = argmax.Read();
  const auto masked_sample = sample.Read();
  const auto masked_probs = probs.Read();
  Expect(masked_argmax[0] != 17 && masked_sample[0] != 17,
         "suppressed token is not chosen");
  Expect(ToFloat(masked_probs[(std::size_t(17 / 32) * kRows) * 32 + 17 % 32]) ==
             0.0F,
         "suppressed token has no probability");
  Expect(std::isfinite(masked_entropy[0]), "entropy finite with suppression");
}

void ReadScoresAndPins() {
  constexpr int kRows = 2, kVocab = 262144, kTiles = 64, kLabels = 3;
  std::vector<float> logits(std::size_t(kRows) * kVocab, -3.0F);
  for (int row = 0; row < kRows; ++row) {
    logits[std::size_t(row) * kVocab + 4] = 12.0F;
    logits[std::size_t(row) * kVocab + 7] = 12.0F;
    logits[std::size_t(row) * kVocab + 4100] = 11.0F;
  }
  const std::vector<std::int32_t> labels = {4, 31, 262143};
  Device<float> raw(logits), tile_max(kRows * kTiles), tile_sum(kRows * kTiles),
      tile_values(kRows * kTiles * 20), top_values(kRows * 20),
      label_values(kRows * kLabels), lse(kRows);
  Device<std::int32_t> tile_ids(kRows * kTiles * 20), top_ids(kRows * 20),
      argmax(kRows), device_labels(labels);
  k::ScoreRows(raw.get(), kRows, kVocab, 30.0F, device_labels.get(), kLabels,
               {.tile_max = tile_max.get(),
                .tile_sum = tile_sum.get(),
                .tile_ids = tile_ids.get(),
                .tile_values = tile_values.get(),
                .top_ids = top_ids.get(),
                .top_logprobs = top_values.get(),
                .label_logprobs = label_values.get(),
                .logsumexp = lse.get(),
                .argmax = argmax.get()},
               nullptr);
  const auto ids = top_ids.Read();
  const auto best = argmax.Read();
  const auto top = top_values.Read();
  const auto exact = label_values.Read();
  for (int row = 0; row < kRows; ++row) {
    const auto host = gufo::models::diffusion_gemma::ScoreLogitsRow(
        std::span<const float>(logits.data() + std::size_t(row) * kVocab,
                               kVocab),
        30.0F, 20, labels);
    Expect(best[row] == host.token, "read score argmax");
    for (int rank = 0; rank < 20; ++rank) {
      Expect(ids[row * 20 + rank] == host.top_logprobs[rank].token,
             "read score top ID and tie order");
      Expect(std::abs(top[row * 20 + rank] - host.top_logprobs[rank].logprob) <
                 2e-4,
             "read score top logprob");
    }
    for (int label = 0; label < kLabels; ++label) {
      const auto wanted = [&](const auto& candidate) {
        return candidate.token == labels[label];
      };
      auto found = std::find_if(host.top_logprobs.begin(),
                                host.top_logprobs.end(), wanted);
      const double want = found != host.top_logprobs.end()
                              ? found->logprob
                              : std::find_if(host.label_logprobs.begin(),
                                             host.label_logprobs.end(), wanted)
                                    ->logprob;
      Expect(std::abs(exact[row * kLabels + label] - want) < 2e-4,
             "read score exact label logprob");
    }
  }
  std::vector<float> soft(4 * 32, 1.0F);
  const std::vector<std::uint8_t> pinned = {1, 0, 1, 0};
  Device<float> device_soft(soft);
  Device<std::uint8_t> device_pins(pinned);
  k::ZeroPinnedSoftRows(device_soft.get(), device_pins.get(), 4, 32, nullptr);
  const auto after = device_soft.Read();
  for (int row = 0; row < 4; ++row)
    for (int i = 0; i < 32; ++i)
      Expect(after[row * 32 + i] == (pinned[row] ? 0.0F : 1.0F),
             "pinned soft embedding is zero");
}

// ----------------------------------------------------------- dequantization

void Dequant() {
  // Every FP8 byte except the NaN codes, scale 1, exact.
  std::vector<std::uint8_t> bytes;
  for (int b = 0; b < 256; ++b)
    if ((b & 0x7f) != 0x7f)
      bytes.push_back(static_cast<std::uint8_t>(b));
  const std::vector<bf16> one = {ToBf16(1.0F)};
  Device<std::uint8_t> dv(bytes);
  Device<bf16> ds(one), out(bytes.size());
  k::DequantFp8Rows(dv.get(), ds.get(), out.get(), 1, bytes.size(), nullptr);
  const auto fp8 = out.Read();
  bool exact = true;
  for (std::size_t i = 0; i < bytes.size(); ++i)
    exact = exact && ToFloat(fp8[i]) == Fp8(bytes[i]);
  Expect(exact, "FP8 E4M3 dequantization is exact");

  // Q8_0 blocks: bf16(fp16 scale * q).
  std::vector<std::uint8_t> blocks(4 * 34);
  std::mt19937 rng(51);
  std::vector<float> want;
  for (int b = 0; b < 4; ++b) {
    const std::uint16_t scale =
        static_cast<std::uint16_t>((12 << 10) | (rng() & 1023));
    std::memcpy(&blocks[b * 34], &scale, 2);
    for (int i = 0; i < 32; ++i) {
      const auto q = static_cast<std::int8_t>(rng() % 255 - 127);
      blocks[b * 34 + 2 + i] = static_cast<std::uint8_t>(q);
      want.push_back(RoundBf16(ExpertMatrix::Half(scale) * q));
    }
  }
  Device<std::uint8_t> db(blocks);
  Device<bf16> q8(want.size());
  k::DequantQ8(db.get(), q8.get(), want.size(), nullptr);
  const auto got = q8.Read();
  exact = true;
  for (std::size_t i = 0; i < want.size(); ++i)
    exact = exact && ToFloat(got[i]) == want[i];
  Expect(exact, "Q8_0 dequantization is exact");
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("diffusion_gemma_kernels_hip_test: no GPU, skipped\n");
    return 77;
  }
  Attention({"sliding causal hd256 window", 256, 16, 8, 96, 160, 64, 0, true,
             64, false});
  Attention(
      {"global causal hd512", 512, 16, 2, 70, 130, 60, 0, true, 0, false});
  Attention({"global canvas hd512 split", 512, 16, 2, 64, 1100, 1036, 0, false,
             0, true});
  Attention({"sliding canvas hd256 offset", 256, 16, 8, 64, 400, 1300, 964,
             false, 0, true});
  for (const int group_rows : {64, 128}) {
    Experts(Weights::kBf16, "bf16", group_rows);
    Experts(Weights::kQ8, "q8", group_rows);
    Experts(Weights::kFp8, "fp8", group_rows);
    Experts(Weights::kFp4, "nvfp4", group_rows);
  }
  Linear(256, 4096, 256);
  Linear(300, 384, 512);
  Linear(256, 128, 1024);
  LinearGeluMul();
  SoftEmbed();
  Sampler();
  ReadScoresAndPins();
  Dequant();
  if (failures == 0)
    std::printf("diffusion_gemma_kernels_hip_test: OK\n");
  return failures == 0 ? 0 : 1;
}
