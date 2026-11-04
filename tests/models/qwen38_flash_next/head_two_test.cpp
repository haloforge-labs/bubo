#include "src/models/qwen38_flash_next/head_two.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/sampling.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::vector<int> Top(std::span<const float> scores, int count) {
  std::vector<int> ids(scores.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    if (std::isfinite(scores[a]) != std::isfinite(scores[b]))
      return std::isfinite(scores[a]);
    if (!std::isfinite(scores[a]))
      return a < b;
    return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
  });
  ids.resize(count);
  return ids;
}
void Policy() {
  using R = q::HeadTwoReason;
  Require(
      q::HeadTwoConsumer(128, 128, false, false, false, false) == R::kShortlist,
      "bounded consumer declined");
  Require(q::HeadTwoConsumer(0, 1, false, false, false, false) == R::kTopk0,
          "unbounded top-k");
  Require(
      q::HeadTwoConsumer(129, 1, false, false, false, false) == R::kTopkWide,
      "wide top-k");
  Require(q::HeadTwoConsumer(20, std::numeric_limits<std::size_t>::max(), false,
                             false, false, false) == R::kOther,
          "wide min-keep");
  for (int field = 0; field < 4; ++field)
    Require(q::HeadTwoConsumer(20, 1, field == 0, field == 1, field == 2,
                               field == 3) != R::kShortlist,
            "unsafe consumer");
  Require(!q::HeadTwoWidth(9), "width 9 must use the original full head");
}

std::uint16_t Half(float x) {
  const unsigned bits = std::bit_cast<unsigned>(x);
  const unsigned sign = (bits >> 16) & 0x8000;
  const int exponent = int((bits >> 23) & 255) - 127 + 15;
  unsigned fraction = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10)
      return sign;
    fraction |= 0x800000;
    const unsigned shift = 14 - exponent;
    const unsigned value = fraction >> shift;
    const unsigned rem = fraction & ((1u << shift) - 1);
    return sign | (value + (rem > (1u << (shift - 1)) ||
                            (rem == (1u << (shift - 1)) && (value & 1))));
  }
  if (exponent >= 31)
    return sign | 0x7c00;
  const unsigned value = (unsigned(exponent) << 10) | (fraction >> 13);
  const unsigned rem = fraction & 8191;
  return sign | (value + (rem > 4096 || (rem == 4096 && (value & 1))));
}
struct Q81 {
  std::uint16_t d, sum;
  std::int8_t qs[32];
};
static_assert(sizeof(Q81) == 36);
std::vector<Q81> Quantize(std::span<const float> x) {
  std::vector<Q81> out(x.size() / 32);
  for (std::size_t b = 0; b < out.size(); ++b) {
    float maximum = 0;
    for (int c = 0; c < 32; ++c)
      maximum = std::max(maximum, std::abs(x[b * 32 + c]));
    const float d = maximum / 127;
    out[b].d = Half(d);
    int sum = 0;
    for (int c = 0; c < 32; ++c) {
      out[b].qs[c] = maximum == 0 ? 0 : std::round(x[b * 32 + c] / d);
      sum += out[b].qs[c];
    }
    out[b].sum = Half(d * sum);
  }
  return out;
}
float Scale(const std::uint8_t* row) {
  std::uint16_t h;
  std::memcpy(&h, row, 2);
  return gufo::quant::Fp16ToFloat(h);
}

// Scalar transcription of the original Q8 row's 32 lane accumulators,
// eight signed channels per lane/block and descending XOR addition tree.
// The multiply/FMA order follows the existing row-kernel arithmetic comment.
float OldQ8(const std::uint8_t* row, const Q81* x, int cols) {
  std::array<float, 32> lane{};
  for (int l = 0; l < 32; ++l) {
    for (int b = l / 4; b < cols / 32; b += 8) {
      const auto* w = row + b * 34;
      int dot = 0;
      for (int c = (l % 4) * 8; c < (l % 4 + 1) * 8; ++c)
        dot += std::int8_t(w[c + 2]) * int(x[b].qs[c]);
      const float p = Scale(w) * float(dot);
      lane[l] = std::fma(p, gufo::quant::Fp16ToFloat(x[b].d), lane[l]);
    }
  }
  for (int offset = 16; offset; offset /= 2) {
    const auto previous = lane;
    for (int l = 0; l < 32; ++l)
      lane[l] = previous[l] + previous[l ^ offset];
  }
  return lane[0];
}

float KernelPrescoreA(const std::uint8_t* raw, const Q81* x, int cols) {
  std::array<float, 32> lane{};
  for (int l = 0; l < 32; ++l) {
    const int g = l / 4, channel = (l & 3) * 8;
    for (int sb = 0; sb < cols / 256; ++sb) {
      const auto* w = raw + sb * 144;
      const auto* s = w + 4;
      const int scale =
          g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
      const int minimum =
          g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
      int dot = 0, sum = 0;
      const auto& a = x[sb * 8 + g];
      for (int c = channel; c < channel + 8; ++c) {
        const int quant = (w[16 + (g / 2) * 32 + c] >> ((g & 1) * 4)) & 15;
        dot += quant * int(a.qs[c]);
        sum += a.qs[c];
      }
      const float d = Scale(w) * scale, m = Scale(w + 2) * minimum;
      const float local = std::fma(d, float(dot), -(m * float(sum)));
      lane[l] = std::fma(local, gufo::quant::Fp16ToFloat(a.d), lane[l]);
    }
  }
  for (int offset = 16; offset; offset /= 2) {
    const auto previous = lane;
    for (int l = 0; l < 32; ++l)
      lane[l] = previous[l] + previous[l ^ offset];
  }
  return lane[0];
}

void RealData(const std::filesystem::path& model,
              const std::filesystem::path& cheap_path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  auto cheap_reader = gufo::core::GgufReader::OpenFile(cheap_path, &error);
  Require(reader && cheap_reader, error.c_str());
  const auto* tensor = reader->FindTensor("output.weight");
  const auto* cheap = cheap_reader->FindTensor("output.weight");
  Require(tensor && cheap && tensor->type == gufo::core::GgmlType::kQ8_0 &&
              cheap->type == gufo::core::GgmlType::kQ4_K &&
              tensor->dimensions == cheap->dimensions,
          "real output head shape/type differs");
  const int cols = tensor->dimensions[0], rows = tensor->dimensions[1],
            sample = 256;
  const auto* weights = static_cast<const std::uint8_t*>(tensor->data);
  const auto* cheap_weights = static_cast<const std::uint8_t*>(cheap->data);
  std::vector<int> physical(sample);
  for (int i = 0; i < sample; ++i)
    physical[i] = (7 + i * 967) % rows;
  std::vector<float> a(std::size_t(sample) * cols);
  for (int i = 0; i < sample; ++i)
    gufo::quant::DequantizeQ4_K(
        cheap_weights + std::size_t(physical[i]) * cols / 256 * 144,
        a.data() + std::size_t(i) * cols, cols);
  std::size_t bytes_compared = 0;
  double maximum_prescore_error = 0;
  for (int width : {1, 2, 8, 9}) {
    std::vector<float> old(width * sample), gathered(width * 128);
    for (int r = 0; r < width; ++r) {
      std::vector<float> activation(cols), x(cols), scores(sample);
      for (int c = 0; c < cols; ++c)
        activation[c] = .17f * std::sin(float(c * 7 + r * 13));
      const auto quantized = Quantize(activation);
      for (int c = 0; c < cols; ++c) {
        x[c] = gufo::quant::Fp16ToFloat(quantized[c / 32].d) *
               quantized[c / 32].qs[c % 32];
      }
      for (int i = 0; i < sample; ++i) {
        const auto* raw = weights + std::size_t(physical[i]) * cols / 32 * 34;
        old[r * sample + i] = OldQ8(raw, quantized.data(), cols);
        double reference = 0;
        auto row = std::span(a).subspan(std::size_t(i) * cols, cols);
        scores[i] = KernelPrescoreA(
            cheap_weights + std::size_t(physical[i]) * cols / 256 * 144,
            quantized.data(), cols);
        for (int c = 0; c < cols; ++c)
          reference += double(row[c]) * x[c];
        maximum_prescore_error = std::max(
            maximum_prescore_error, std::abs(double(scores[i]) - reference));
        Require(std::abs(double(scores[i]) - reference) <= 1e-5,
                "prescore differs from scalar reconstruction");
      }
      const auto ids = Top(scores, 128);
      if (width == 1 && r == 0) {
        // A known biased ID outside the pre-score top-128 must be gathered
        // even when its raw logit is low. Compare the real sampler to a full
        // 256-row oracle, with default-profile EOG/marker controls enabled.
        int extra = 0;
        while (std::ranges::find(ids, extra) != ids.end())
          ++extra;
        gufo::sampling::SamplingConfig c;
        c.top_k = 20;
        auto b = std::make_shared<gufo::sampling::BehaviourConfig>();
        b->eog_boundary = true;
        b->eog_min_prob = .2;
        b->eog_defer_p = .9;
        b->loop_breaker = true;
        b->spec_eog_stop = true;
        b->think_eog_guard = true;
        b->loop_eog_close = true;
        b->eog_ids = {1, 2};
        b->marker_bias = 1;
        b->marker_ids = {3};
        b->logit_bias_enabled = true;
        b->logit_bias = {{static_cast<std::uint32_t>(extra), 100.0f}};
        c.behaviour = b;
        gufo::sampling::SamplerState sampler(c);
        const auto support = sampler.HeadSupport();
        Require(!support.full_normalizer,
                "default bounded behaviour needs full Z");
        Require(std::ranges::find(support.ids, extra) != support.ids.end(),
                "biased extra row missing");
        std::vector<float> sparse(sample, -INFINITY);
        for (int id : ids)
          sparse[id] = old[id];
        for (auto id : support.ids) {
          const auto* raw =
              weights + std::size_t(physical[id]) * cols / 32 * 34;
          sparse[id] = OldQ8(raw, quantized.data(), cols);
          Require(std::memcmp(&sparse[id], &old[id], sizeof(float)) == 0,
                  "extra row byte comparison failed");
          bytes_compared += sizeof(float);
        }
        auto oracle = sampler;
        Require(
            sampler.Sample(sparse) == unsigned(extra) &&
                oracle.Sample(std::span(old).first(sample)) == unsigned(extra),
            "biased extra row differs from full-head oracle");
        b->eog_min_prob_exact = true;
        Require(gufo::sampling::SamplerState(c).HeadSupport().full_normalizer,
                "exact EOG normalizer must keep full head");
        std::cout << "Known extra row outside pre-score top-128: bias/full "
                     "oracle PASS\n";
      }
      if (q::HeadTwoWidth(width)) {
        // Gather packed bytes first, then run the retained row arithmetic.
        std::vector<std::uint8_t> packed(std::size_t(128) * cols / 32 * 34);
        for (int j = 0; j < 128; ++j) {
          const auto rb = std::size_t(cols / 32 * 34);
          std::memcpy(packed.data() + j * rb,
                      weights + std::size_t(physical[ids[j]]) * rb, rb);
          gathered[r * 128 + j] =
              OldQ8(packed.data() + j * rb, quantized.data(), cols);
          Require(std::memcmp(&gathered[r * 128 + j], &old[r * sample + ids[j]],
                              sizeof(float)) == 0,
                  "gathered row bytes differ from old full-row arithmetic");
          bytes_compared += sizeof(float);
        }
      } else {
        // The wide full-head implementation is left unchanged. CPU work can
        // prove this route guard; its F16/WMMA instruction bits need a GPU.
        Require(!q::HeadTwoWidth(width), "width 9 entered shortlist");
      }
    }
  }
  std::cout << "CPU reference: A stages 1-3, real rows=" << sample
            << " N=1,2,8; N=9 route guard; max_prescore_error="
            << maximum_prescore_error << " PASS\n"
            << "CPU byte comparison: old row arithmetic versus gathered packed "
               "real weights, bytes="
            << bytes_compared
            << " PASS (HIP instruction equivalence requires GUFO_HEAD_CHECK)\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Policy();
    // Real-model directory: BUBO_TEST_MODEL_ROOT, else the directory of the
    // shard given as argv[1]. It holds the UD-Q4_K_XL shards and
    // draft_head/draft-out-q4k.gguf.
    std::filesystem::path root;
    if (const char* env = std::getenv("BUBO_TEST_MODEL_ROOT"); env && *env)
      root = env;
    else if (argc > 1)
      root = std::filesystem::path(argv[1]).parent_path();
    if (root.empty()) {
      std::cout << "real-data check skipped: set BUBO_TEST_MODEL_ROOT or pass "
                   "the main shard path\n";
    } else {
      RealData(argc > 1
                   ? std::filesystem::path(argv[1])
                   : root / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
               root / "draft_head/draft-out-q4k.gguf");
    }
    std::cout << "shortlist consumer policy and N=9 full-route guard PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
