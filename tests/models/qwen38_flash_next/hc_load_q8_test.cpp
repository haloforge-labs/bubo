// GUFO_HC_LOAD_Q8: switch parsing, the all-or-nothing predicate, and the
// shared BF16 -> Q8_0 quantiser on random data and on crops of real HC
// tensors. `--real <gguf>...` reads crops from native files, `--load-time`
// quantises a synthetic tensor set of the real total size and prints the host
// milliseconds.
#include "src/models/qwen38_flash_next/hc_load_q8_policy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/bf16_q8_0.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace p = gufo::models::qwen38_flash_next::hc_load_q8;
using gufo::core::GgmlType;
using gufo::quant::block_q8_0;

namespace {

void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}

float Decode(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t{bits} << 16);
}

std::uint16_t Encode(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

// Nearest finite binary16 with ties to even, found in the sorted table of every
// non-negative finite half: independent of the production cast.
std::uint16_t ReferenceHalf(float value) {
  static const std::vector<double> table = [] {
    std::vector<double> t;
    for (std::uint32_t h = 0; h <= 0x7bff; ++h)
      t.push_back(gufo::quant::Fp16ToFloat(static_cast<std::uint16_t>(h)));
    return t;
  }();
  const auto upper = std::lower_bound(table.begin(), table.end(), double(value));
  if (upper == table.begin())
    return 0;
  if (upper == table.end())
    return 0x7bff;
  const auto lower = upper - 1;
  const double below = double(value) - *lower;
  const double above = *upper - double(value);
  auto index = static_cast<std::uint16_t>(
      (below < above || (below == above && ((lower - table.begin()) & 1) == 0))
          ? lower - table.begin()
          : upper - table.begin());
  return index;
}

// The documented GGML convention written out independently of the library.
std::vector<block_q8_0> Reference(const std::vector<std::uint16_t>& bf16) {
  std::vector<block_q8_0> out(bf16.size() / 32);
  for (std::size_t b = 0; b < out.size(); ++b) {
    float amax = 0;
    for (std::size_t i = 0; i < 32; ++i)
      amax = std::max(amax, std::abs(Decode(bf16[b * 32 + i])));
    const float d = amax / 127.0F;
    const float inverse = d == 0 ? 0 : 1.0F / d;
    out[b].d = ReferenceHalf(d);
    for (std::size_t i = 0; i < 32; ++i)
      out[b].qs[i] = static_cast<std::int8_t>(
          std::clamp(std::lround(Decode(bf16[b * 32 + i]) * inverse), -127L,
                     127L));
  }
  return out;
}

struct Stats {
  double max_abs{0}, rel_rms{0}, max_weight{0};
};

Stats RoundTrip(const std::vector<std::uint16_t>& bf16,
                const std::vector<block_q8_0>& q8) {
  std::vector<float> back(bf16.size());
  for (std::size_t b = 0; b < q8.size(); ++b)
    gufo::quant::DequantizeQ8_0(&q8[b], back.data() + b * 32, 32);
  Stats s;
  double err2 = 0, ref2 = 0;
  for (std::size_t i = 0; i < bf16.size(); ++i) {
    const double x = Decode(bf16[i]);
    const double e = std::abs(x - back[i]);
    s.max_abs = std::max(s.max_abs, e);
    s.max_weight = std::max(s.max_weight, std::abs(x));
    err2 += e * e;
    ref2 += x * x;
  }
  s.rel_rms = ref2 > 0 ? std::sqrt(err2 / ref2) : 0;
  return s;
}

// `rms_limit` bounds the relative RMS: tight for real weights; the stress data
// puts a 40-sigma outlier in one block in eleven, which costs a few percent.
void Check(const std::string& label, const std::vector<std::uint16_t>& bf16,
           std::size_t rows, std::size_t cols, double rms_limit = 0.01) {
  std::vector<block_q8_0> actual(bf16.size() / 32);
  std::string error;
  Require(gufo::quant::QuantizeBf16Q8_0(bf16.data(), rows, cols, actual, &error),
          label + ": " + error);
  const auto expected = Reference(bf16);
  Require(std::memcmp(actual.data(), expected.data(),
                      actual.size() * sizeof(block_q8_0)) == 0,
          label + ": quantiser differs from the independent reference");
  // The loader's threaded form must give the same bytes for any split.
  for (const unsigned threads : {2U, 3U, p::kQuantizeThreads, 7U}) {
    std::vector<block_q8_0> split(actual.size());
    Require(gufo::quant::QuantizeBf16Q8_0Threads(bf16.data(), rows, cols, split,
                                                 &error, threads) &&
                std::memcmp(split.data(), actual.data(),
                            actual.size() * sizeof(block_q8_0)) == 0,
            label + ": threaded quantiser differs from the serial one");
  }
  // Per block: half a quantisation step, plus the stored scale's rounding
  // error accumulated over at most 127 levels, plus F32 reciprocal error.
  for (std::size_t b = 0; b < actual.size(); ++b) {
    float amax = 0;
    for (std::size_t i = 0; i < 32; ++i)
      amax = std::max(amax, std::abs(Decode(bf16[b * 32 + i])));
    const double d = amax / 127.0F;
    const double stored = gufo::quant::Fp16ToFloat(actual[b].d);
    const double bound = 0.5 * d + 127 * std::abs(d - stored) +
                         4 * std::numeric_limits<float>::epsilon() * amax;
    for (std::size_t i = 0; i < 32; ++i)
      Require(std::abs(Decode(bf16[b * 32 + i]) - stored * actual[b].qs[i]) <=
                  bound,
              label + ": error above the Q8_0 bound in block " +
                  std::to_string(b));
  }
  const Stats s = RoundTrip(bf16, actual);
  Require(s.rel_rms < rms_limit,
          label + ": relative RMS above " + std::to_string(rms_limit));
  std::cout << "  " << label << ": values=" << bf16.size()
            << " max_weight=" << s.max_weight << " max_abs_err=" << s.max_abs
            << " rel_rms=" << s.rel_rms << " (matches reference bit for bit)\n";
}

void CheckParse() {
  Require(p::Parse(nullptr) == 0 && p::Parse("0") == 0 && p::Parse("1") == 1,
          "unset must mean off for the debug switches; 0 and 1 are the only values");
  Require(p::ParseDefaultOn(nullptr) == 1 && p::ParseDefaultOn("0") == 0 &&
              p::ParseDefaultOn("1") == 1,
          "GUFO_HC_LOAD_Q8: unset must mean on; 0 and 1 are the only values");
  for (const char* value : {"", "2", "true", " 1", "1 "})
    Require(p::ParseDefaultOn(value) == -1,
            std::string("GUFO_HC_LOAD_Q8: unknown value accepted: '") + value +
                "'");
  for (const char* value : {"", "2", "true", "false", "on", "off", "10", "01",
                            " 1", "-1", "1 "})
    Require(p::Parse(value) == -1, std::string("unknown value accepted: '") +
                                       value + "'");
}

constexpr std::uint64_t kDownBf16 = 6553600;  // 320 x 10240 x 2
constexpr std::uint64_t kDownQ8 = 3481600;    // 102400 blocks x 34

struct Model {
  std::vector<p::Tensor> list;
  std::uint32_t next{0};
  void Mixer(p::Group group, GgmlType projection, GgmlType inject,
             bool with_inject = true) {
    ++next;
    list.push_back({next, group, p::Role::kDown, projection, 320, 10240});
    list.push_back({next, group, p::Role::kUp, projection, 10240, 320});
    if (with_inject)
      list.push_back({next, group, p::Role::kInject, inject, 4, 10240});
  }
};

Model Native(bool mtp_q8 = true, bool mtp = true) {
  Model m;
  m.Mixer(p::Group::kOutputHead, GgmlType::kBF16, GgmlType::kF32, false);
  for (int layer = 0; layer < 48; ++layer) {
    m.Mixer(p::Group::kTrunkAttn, GgmlType::kBF16, GgmlType::kBF16);
    m.Mixer(p::Group::kTrunkFfn, GgmlType::kBF16, GgmlType::kBF16);
  }
  if (mtp) {
    // The shared MTP sidecar stores Q8_0 projections and a Q8_0 inject.
    const GgmlType t = mtp_q8 ? GgmlType::kQ8_0 : GgmlType::kBF16;
    m.Mixer(p::Group::kMtpAttn, t, mtp_q8 ? GgmlType::kQ8_0 : GgmlType::kBF16);
    m.Mixer(p::Group::kMtpFfn, t, mtp_q8 ? GgmlType::kQ8_0 : GgmlType::kBF16);
    m.Mixer(p::Group::kMtpHead, t, t, false);
  }
  return m;
}

void CheckPlan() {
  // The native files: 97 mixers of BF16 projections.
  {
    const p::Plan plan = p::Decide(Native().list);
    Require(plan.convert && plan.reason.empty(), "native file must convert");
    Require(plan.down == 97 && plan.up == 97 && plan.inject == 96,
            "class counts: 97 down, 97 up, 96 inject");
    Require(plan.bf16_bytes == 1271398400 && plan.q8_bytes == 675430400 &&
                plan.resident_saved() == 595968000,
            "the 194 tensors are 1,271,398,400 B BF16 and 675,430,400 B Q8_0");
    Require(plan.bf16_bytes == 194 * kDownBf16 &&
                plan.q8_bytes == 194 * kDownQ8,
            "per-tensor sizes");
    Require(plan.inject_added_bytes == 96 * (163840 - 81920),
            "inject widening adds 7,864,320 B");
    Require(plan.mixers_converted[std::size_t(p::Group::kTrunkAttn)] == 48 &&
                plan.mixers_converted[std::size_t(p::Group::kTrunkFfn)] == 48 &&
                plan.mixers_converted[std::size_t(p::Group::kOutputHead)] == 1 &&
                plan.mixers_converted[std::size_t(p::Group::kMtpAttn)] == 0,
            "mixers covered");
  }
  // MTP sidecar in BF16 as well: converted in the same step.
  {
    const p::Plan plan = p::Decide(Native(false).list);
    Require(plan.convert && plan.down == 100 && plan.up == 100 &&
                plan.mixers_converted[std::size_t(p::Group::kMtpHead)] == 1,
            "a BF16 MTP block converts with the trunk");
  }
  // Without an MTP block.
  Require(p::Decide(Native(true, false).list).convert, "no MTP block");
  // Production package: nothing is BF16, so the switch does nothing.
  {
    Model m;
    m.Mixer(p::Group::kOutputHead, GgmlType::kQ8_0, GgmlType::kF32, false);
    for (int layer = 0; layer < 48; ++layer) {
      m.Mixer(p::Group::kTrunkAttn, GgmlType::kQ8_0, GgmlType::kF32);
      m.Mixer(p::Group::kTrunkFfn, GgmlType::kQ8_0, GgmlType::kF32);
    }
    m.Mixer(p::Group::kMtpAttn, GgmlType::kQ8_0, GgmlType::kQ8_0);
    m.Mixer(p::Group::kMtpFfn, GgmlType::kQ8_0, GgmlType::kQ8_0);
    const p::Plan plan = p::Decide(m.list);
    Require(!plan.convert && plan.nothing_to_do && plan.bf16_bytes == 0,
            "a production-style package must be a no-op");
  }
  // Refusals: each leaves the file entirely on the BF16 route.
  const auto refused = [](Model m, const char* what) {
    const p::Plan plan = p::Decide(m.list);
    Require(!plan.convert && !plan.nothing_to_do && !plan.reason.empty() &&
                plan.down == 0 && plan.up == 0 && plan.inject == 0,
            std::string("must refuse: ") + what);
  };
  {
    Model m = Native();
    m.list[32].type = GgmlType::kF16;
    refused(m, "an F16 projection");
  }
  {
    Model m = Native();
    m.list[33].type = GgmlType::kF32;
    refused(m, "an F32 projection");
  }
  {
    Model m = Native();
    m.list[32].type = GgmlType::kQ8_0;  // down Q8_0, up still BF16
    refused(m, "down and up of one mixer in different types");
  }
  {
    Model m = Native();
    m.list[32].rows = 256;
    refused(m, "a down projection with another shape");
  }
  {
    Model m = Native();
    m.list[33].cols = 321;
    refused(m, "an up projection with another column count");
  }
  {
    Model m = Native();
    m.list[34].type = GgmlType::kF16;
    refused(m, "an F16 inject");
  }
  {
    Model m = Native();
    m.list[34].cols = 8192;
    refused(m, "an inject with another shape");
  }
  {
    Model m = Native();
    m.list.erase(m.list.begin(), m.list.begin() + 2);  // output head gone
    refused(m, "a missing output head");
  }
  {
    Model m = Native();
    m.list.back().type = GgmlType::kIQ3_S;  // MTP head, last tensor
    refused(m, "a draft-head projection in another type");
  }
  refused(Model{}, "an empty listing");
  // One BF16 draft mixer in an otherwise Q8_0 package converts that mixer.
  {
    Model m;
    m.Mixer(p::Group::kOutputHead, GgmlType::kQ8_0, GgmlType::kF32, false);
    m.Mixer(p::Group::kTrunkAttn, GgmlType::kQ8_0, GgmlType::kF32);
    m.Mixer(p::Group::kTrunkFfn, GgmlType::kQ8_0, GgmlType::kF32);
    m.Mixer(p::Group::kMtpAttn, GgmlType::kBF16, GgmlType::kBF16);
    const p::Plan plan = p::Decide(m.list);
    Require(plan.convert && plan.down == 1 && plan.up == 1 && plan.inject == 1,
            "a lone BF16 mixer is converted so every mixer ends Q8_0");
  }
}

void CheckWiden() {
  std::vector<std::uint16_t> bits;
  for (std::uint32_t v = 0; v < 65536; ++v)
    bits.push_back(static_cast<std::uint16_t>(v));
  std::vector<float> wide(bits.size());
  p::WidenBf16(bits.data(), bits.size(), wide.data());
  for (std::size_t i = 0; i < bits.size(); ++i) {
    const std::uint32_t round_trip = std::bit_cast<std::uint32_t>(wide[i]);
    Require((round_trip >> 16) == bits[i] && (round_trip & 0xffff) == 0,
            "BF16 -> F32 must be exact");
  }
}

// The loader uploads blocks and the zero tail in one image: every byte must be
// written even into a buffer that holds poison, for every thread count.
void CheckImage() {
  std::mt19937_64 rng(77);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> bf16(40 * 64);
  for (auto& v : bf16)
    v = Encode(normal(rng));
  const auto expected = Reference(bf16);
  for (const unsigned threads : {1U, 2U, 3U, 7U, 100U}) {
    std::vector<std::uint8_t> image(1000000, 0xA5);
    std::string error;
    Require(gufo::quant::BuildBf16Q8_0Image(bf16.data(), 40, 64, 4096, threads, image, &error),
            "image build: " + error);
    Require(image.size() == expected.size() * sizeof(block_q8_0) + 4096, "image size");
    Require(std::memcmp(image.data(), expected.data(), expected.size() * sizeof(block_q8_0)) == 0,
            "image blocks differ from the reference");
    for (std::size_t i = image.size() - 4096; i < image.size(); ++i)
      Require(image[i] == 0, "image tail must be zero");
  }
  std::vector<std::uint8_t> f32;
  std::string error;
  Require(gufo::quant::BuildBf16F32Image(bf16.data(), bf16.size(), 4096, f32, &error) &&
              f32.size() == bf16.size() * 4 + 4096,
          "F32 image");
  for (std::size_t i = 0; i < bf16.size(); ++i) {
    float value;
    std::memcpy(&value, f32.data() + i * 4, 4);
    Require(std::bit_cast<std::uint32_t>(value) == std::uint32_t{bf16[i]} << 16, "F32 image value");
  }
  for (std::size_t i = f32.size() - 4096; i < f32.size(); ++i)
    Require(f32[i] == 0, "F32 image tail must be zero");
}

// GUFO_HC_LOAD_Q8_POISON: the pattern decodes to NaN in every type the engine
// reads, and the ledger accounts exactly the bytes written.
void CheckPoison() {
  // Switch parsing is the shared strict 0/1 rule (unset = off).
  Require(p::Parse(nullptr) == 0 && p::Parse("1") == 1 && p::Parse("2") == -1 && p::Parse("") == -1,
          "poison switch parse");
  Require(p::kPoisonByte == 0xFF, "poison byte");
  std::vector<std::uint8_t> a(1000, 0x11), b(37, 0x22);
  p::PoisonLedger ledger;
  ledger.Host(a.data(), a.size());
  ledger.Host(b.data(), b.size());
  ledger.Host(nullptr, 5);        // no buffer: nothing counted
  ledger.Host(a.data(), 0);       // empty: nothing counted
  ledger.KeptDevice(4096 + 100);
  ledger.KeptDevice(8192);
  Require(ledger.host_events == 2 && ledger.host_bytes == 1037, "host accounting");
  Require(ledger.kept_device == 2 && ledger.kept_device_bytes == 4096 + 100 + 8192, "device accounting");
  for (auto v : a) Require(v == 0xFF, "buffer a not fully poisoned");
  for (auto v : b) Require(v == 0xFF, "buffer b not fully poisoned");
  // Decoded in every storage type: BF16 0xFFFF, F16 0xFFFF, F32 0xFFFFFFFF are NaN; a Q8_0 block's binary16 scale is NaN.
  std::uint16_t bf16 = 0xFFFF, f16 = 0xFFFF;
  std::uint32_t f32bits = 0xFFFFFFFFu;
  const float from_bf16 = Decode(bf16);
  float from_f32;
  std::memcpy(&from_f32, &f32bits, 4);
  Require(std::isnan(from_bf16) && std::isnan(from_f32), "poison is not NaN in BF16/F32");
  Require(((f16 >> 10) & 0x1F) == 0x1F && (f16 & 0x3FF) != 0, "poison is not an F16 NaN");
  block_q8_0 block;
  std::memset(&block, p::kPoisonByte, sizeof block);
  Require(((block.d >> 10) & 0x1F) == 0x1F && (block.d & 0x3FF) != 0 && block.qs[0] == -1,
          "poisoned Q8_0 block scale is not NaN");
}

void CheckRandom() {
  std::mt19937_64 rng(0xC0FFEE);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const auto& shape : {std::pair<std::size_t, std::size_t>{320, 10240},
                            {10240, 320}, {4, 10240}}) {
    std::vector<std::uint16_t> bf16(shape.first * shape.second);
    for (std::size_t b = 0; b < bf16.size() / 32; ++b) {
      // Block scales across six decades, some blocks all zero, some with one
      // outlier, so scale rounding and the clamp are both exercised.
      const float scale = std::pow(10.0F, float(rng() % 600) / 100.0F - 4.0F);
      const bool zero = rng() % 97 == 0;
      const bool outlier = rng() % 11 == 0;
      for (std::size_t i = 0; i < 32; ++i)
        bf16[b * 32 + i] =
            zero ? 0 : Encode(normal(rng) * scale * (outlier && i == 7 ? 40.0F : 1.0F));
    }
    Check("random " + std::to_string(shape.second) + "x" +
              std::to_string(shape.first),
          bf16, shape.first, shape.second, 0.2);
  }
  // Rejections: nonfinite weights and a scale that overflows binary16.
  std::vector<std::uint16_t> bf16(64, Encode(1.0F));
  std::vector<block_q8_0> out(2);
  std::string error;
  for (const std::uint16_t bad : {0x7f80, 0xff80, 0x7fc1})
    for (std::size_t at : {0u, 33u}) {
      bf16[at] = bad;
      Require(!gufo::quant::QuantizeBf16Q8_0(bf16.data(), 1, 64, out, &error) &&
                  !error.empty(),
              "nonfinite weight accepted");
      bf16[at] = Encode(1.0F);
    }
  bf16[3] = 0x7f7f;  // about 3.4e38: d/127 does not fit binary16
  Require(!gufo::quant::QuantizeBf16Q8_0(bf16.data(), 1, 64, out, &error),
          "binary16 scale overflow accepted");
  Require(!gufo::quant::QuantizeBf16Q8_0(bf16.data(), 1, 63, out, &error),
          "cols not a multiple of 32 accepted");
  {
    // A bad weight in the middle row range must fail the threaded form too.
    std::vector<std::uint16_t> rows8(8 * 32, Encode(0.5F));
    std::vector<block_q8_0> out8(8);
    rows8[5 * 32 + 9] = 0x7fc1;
    error.clear();
    Require(!gufo::quant::QuantizeBf16Q8_0Threads(rows8.data(), 8, 32, out8,
                                                  &error, 4) &&
                !error.empty(),
            "threaded quantiser accepted a nonfinite weight");
  }
}

void CheckReal(const std::vector<std::string>& files) {
  int crops = 0;
  for (const auto& file : files) {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenHeaderFile(file, &error);
    Require(reader != nullptr, file + ": " + error);
    for (const auto& info : reader->GetTensors()) {
      const std::string name(info.name);
      const bool hc = name.find("hc_") != std::string::npos ||
                      name.find("output_hc") != std::string::npos;
      if (!hc || info.type != GgmlType::kBF16)
        continue;
      if (name.find("_down.weight") == std::string::npos &&
          name.find("_up.weight") == std::string::npos)
        continue;
      // Crops: the first 16 rows of blk.0, the middle rows of blk.24 and the
      // last rows of anything else (so every mixer family is seen).
      const bool picked = name.rfind("blk.0.", 0) == 0 ||
                          name.rfind("blk.24.", 0) == 0 ||
                          name.rfind("blk.47.", 0) == 0 ||
                          name.rfind("output_hc", 0) == 0;
      if (!picked)
        continue;
      const std::size_t cols = info.dimensions[0];
      const std::size_t rows_total = info.dimensions[1];
      const std::size_t rows = std::min<std::size_t>(rows_total, cols > 1000 ? 16 : 512);
      const std::size_t first = name.rfind("blk.24.", 0) == 0 ? rows_total / 2 : 0;
      std::vector<std::uint16_t> bf16(rows * cols);
      std::memcpy(bf16.data(),
                  static_cast<const std::uint8_t*>(info.data) + first * cols * 2,
                  bf16.size() * 2);
      Check(name + " rows " + std::to_string(first) + ".." +
                std::to_string(first + rows - 1),
            bf16, rows, cols);
      ++crops;
    }
  }
  Require(crops > 0, "no BF16 HC tensor found in the given files");
  std::cout << "  real crops checked: " << crops << '\n';
}

void LoadTime() {
  // 194 tensors of 320 x 10240 BF16 values: the real total, one slice each.
  constexpr std::size_t kTensor = 320ull * 10240;
  constexpr std::size_t kTensors = 194;
  std::vector<std::uint16_t> source(kTensor * kTensors);
  {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> normal(0.0F, 0.05F);
    for (std::size_t i = 0; i < kTensor; ++i)
      source[i] = Encode(normal(rng));
    for (std::size_t t = 1; t < kTensors; ++t)
      std::memcpy(source.data() + t * kTensor, source.data(), kTensor * 2);
  }
  std::vector<block_q8_0> out(kTensor / 32);
  for (const unsigned threads : {1U, p::kQuantizeThreads}) {
    std::vector<std::uint8_t> sink;
    sink.reserve(kTensors * out.size() * sizeof(block_q8_0));
    const auto begin = std::chrono::steady_clock::now();
    std::string error;
    for (std::size_t t = 0; t < kTensors; ++t) {
      Require(gufo::quant::QuantizeBf16Q8_0Threads(
                  source.data() + t * kTensor, 320, 10240, out, &error, threads),
              error);
      // The loader hands each tensor to a synchronous device copy; a copy into
      // a sink stands in for it so the output is written, not optimised away.
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(out.data());
      sink.insert(sink.end(), bytes, bytes + out.size() * sizeof(block_q8_0));
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    std::cout << "  load-time prediction (" << threads << " thread"
              << (threads == 1 ? "" : "s") << "): " << kTensors << " tensors, "
              << kTensor * kTensors << " BF16 values (" << kTensor * kTensors * 2
              << " B) -> " << sink.size() << " B Q8_0 in " << ms
              << " ms host, warm memory ("
              << ms / (double(kTensor * kTensors * 2) / 1e9)
              << " ms per GB of BF16)\n";
  }
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<std::string> real;
  bool load_time = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--load-time")
      load_time = true;
    else if (arg == "--real")
      for (++i; i < argc; ++i)
        real.push_back(argv[i]);
    else
      throw std::runtime_error("unknown argument " + arg);
  }
  CheckParse();
  CheckPlan();
  CheckWiden();
  CheckImage();
  CheckPoison();
  CheckRandom();
  if (!real.empty())
    CheckReal(real);
  if (load_time)
    LoadTime();
  std::cout << "HC LOAD Q8 PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
