// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
#include "src/models/qwen38_flash_next/dense_gemv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/iq_grids.hpp"
#include "src/models/qwen38_flash_next/kquant_stage.hpp"

namespace {
namespace d = gufo::models::qwen38_flash_next::dense_gemv;
namespace q = gufo::quant;
std::uint64_t checks = 0;

void Require(bool ok, const char* reason) {
  ++checks;
  if (!ok)
    throw std::runtime_error(reason);
}
std::uint32_t Next(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
int Signed(std::uint32_t v) {
  return int(v & 127) - int(v & 128);
}

struct Math {
  template<int alignment>
  static std::uint32_t Word(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
  }
  static std::uint16_t HalfBits(const std::uint8_t* p) {
    std::uint16_t value;
    std::memcpy(&value, p, 2);
    return value;
  }
  static float Half(const std::uint8_t* p) {
    return q::Fp16ToFloat(HalfBits(p));
  }
  static d::Pair Lookup(std::uint32_t word) {
    d::Pair pair{};
    for (int j = 0; j < 4; ++j) {
      pair.lo |=
          std::uint32_t(std::uint8_t(q::kValuesIq4Nl[(word >> (8 * j)) & 15]))
          << (8 * j);
      pair.hi |= std::uint32_t(
                     std::uint8_t(q::kValuesIq4Nl[(word >> (8 * j + 4)) & 15]))
                 << (8 * j);
    }
    return pair;
  }
  template<int Type>
  static d::Pair Grid(int index, const std::uint32_t* staged = nullptr) {
    if (staged) {
      if constexpr (Type == 17)
        return {staged[2 * index], staged[2 * index + 1]};
      else
        return {staged[index], 0};
    }
    if constexpr (Type == 17) {
      const auto value = q::Iq2xsGrid()[index];
      return {std::uint32_t(value), std::uint32_t(value >> 32)};
    } else if constexpr (Type == 18)
      return {q::Iq3xxsGrid()[index], 0};
    else
      return {q::Iq3sGrid()[index], 0};
  }
  static int Dot(std::uint32_t w, std::uint32_t x, int acc) {
    for (int j = 0; j < 4; ++j)
      acc += Signed(w >> (8 * j)) * Signed(x >> (8 * j));
    return acc;
  }
  static std::uint32_t Subtract32(std::uint32_t word) {
    std::uint32_t result = 0;
    for (int j = 0; j < 4; ++j)
      result |= std::uint32_t(std::uint8_t(((word >> (8 * j)) & 255) - 32))
                << (8 * j);
    return result;
  }
  static float Mul(float a, float b) {
    volatile float v = a * b;
    return v;
  }
  static float Add(float a, float b) {
    volatile float v = a + b;
    return v;
  }
  static float Fma(float a, float b, float c) { return std::fma(a, b, c); }
};

void PutHalf(std::uint8_t* p, std::uint16_t value) {
  std::memcpy(p, &value, 2);
}

// Scalar element addressing from the vendored dequantizers. This reference
// never calls Fragment, its packed-word helpers, or its scale extraction.
int Scale(const std::uint8_t* w, int group, bool minimum) {
  const auto* s = w + 4;
  if (group < 4)
    return s[group + (minimum ? 4 : 0)] & 63;
  return minimum ? (s[group + 4] >> 4) | ((s[group] >> 6) << 4)
                 : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
}

float ReferenceStep(int type, const std::uint8_t* w, const std::uint8_t* x,
                    const float* codes, int iqs, float acc) {
  if (type == 17 || type == 18 || type == 21) {
    const int group = iqs / 2;
    const auto* a = x + group * 36;
    int dots[2]{};
    for (int j = 0; j < 32; ++j)
      dots[j / 16] += int(codes[group * 32 + j]) * Signed(a[4 + j]);
    int integer = dots[0] + dots[1];
    if (type == 17) {
      const int sc = w[66 + group];
      integer = (dots[0] * (sc & 15) + dots[1] * (sc >> 4) + integer / 2) / 4;
    } else if (type == 18) {
      const int sc = w[69 + group * 4] >> 4;
      integer = (sc * integer + integer / 2) / 2;
    } else {
      const int sc = (w[106 + group / 2] >> (4 * (group % 2))) & 15;
      integer *= 1 + 2 * sc;
    }
    return std::fma(Math::Mul(Math::Half(w), Math::Half(a)), float(integer),
                    acc);
  }
  if (type == 14) {
    const int group = (iqs / 16) * 4 + (iqs % 16) / 8;
    const int first = (iqs % 8) * 4;
    float sd[2];
    for (int part = 0; part < 2; ++part) {
      const int block = group + 2 * part;
      const auto* a = x + block * 36;
      int dot = 0;
      for (int j = 0; j < 4; ++j)
        dot += int(codes[block * 32 + first + j]) * Signed(a[4 + first + j]);
      sd[part] = float(dot * Signed(w[192 + block * 2 + first / 16]));
    }
    const float dot = std::fma(Math::Half(x + (group + 2) * 36), sd[1],
                               Math::Mul(Math::Half(x + group * 36), sd[0]));
    return std::fma(Math::Half(w + 208), dot, acc);
  }
  const float scale = Math::Half(w);
  if (type == 12 || type == 13) {
    const int group = (iqs / 8) * 2;
    const int first = ((iqs / 2) % 4) * 4;
    float sd[2], sm[2];
    for (int part = 0; part < 2; ++part) {
      int dot = 0, sum = 0;
      const auto* a = x + (group + part) * 36;
      for (int half = 0; half < 2; ++half)
        for (int j = 0; j < 4; ++j) {
          const int index = first + 16 * half + j;
          const int v = Signed(a[4 + index]);
          dot += int(codes[(group + part) * 32 + index]) * v;
          sum += v;
        }
      sd[part] = float(dot * Scale(w, group + part, false));
      sm[part] = float(sum * Scale(w, group + part, true));
    }
    const float dot = std::fma(Math::Half(x + (group + 1) * 36), sd[1],
                               Math::Mul(Math::Half(x + group * 36), sd[0]));
    const float min = std::fma(Math::Half(x + (group + 1) * 36), sm[1],
                               Math::Mul(Math::Half(x + group * 36), sm[0]));
    return std::fma(-Math::Half(w + 2), min, std::fma(scale, dot, acc));
  }
  int dot = 0, sum = 0;
  if (type == 23) {
    const int group = iqs / 4;
    const auto* a = x + group * 36;
    for (int j = 0; j < 32; ++j)
      dot += int(codes[group * 32 + j]) * Signed(a[4 + j]);
    const int sub = ((w[4 + group / 2] >> (4 * (group % 2))) & 15) |
                    (((w[2 + group / 4] >> (2 * (group % 4))) & 3) << 4);
    return std::fma(Math::Mul(scale, Math::Half(a)), float(dot * (sub - 32)),
                    acc);
  }
  for (int half = 0; half < 2; ++half)
    for (int j = 0; j < 8; ++j) {
      const int index = iqs * 4 + half * 16 + j;
      const int v = Signed(x[4 + index]);
      dot += int(codes[index]) * v;
      sum += v;
    }
  if (type == 20)
    return std::fma(Math::Mul(scale, float(dot)), Math::Half(x), acc);
  const float value =
      std::fma(Math::Half(w + 2), float(sum), Math::Mul(scale, float(dot)));
  return std::fma(Math::Half(x), value, acc);
}

float Reduce(std::array<float, 32> values, int lane) {
  for (int mask : {16, 8, 4, 2, 1}) {
    const auto before = values;
    for (int j = 0; j < 32; ++j)
      values[j] = Math::Add(before[j], before[j ^ mask]);
  }
  return std::isfinite(values[lane]) ? values[lane] : 0.0F;
}

float ReduceSplit(const std::array<float, 32>& original, int lane) {
  std::array<float, 8> values{};
  for (int sub = 0; sub < 8; ++sub)
    values[sub] = Math::Add(Math::Add(original[sub], original[sub + 16]),
                            Math::Add(original[sub + 8], original[sub + 24]));
  for (int mask : {4, 2, 1}) {
    const auto before = values;
    for (int sub = 0; sub < 8; ++sub)
      values[sub] = Math::Add(before[sub], before[sub ^ mask]);
  }
  return std::isfinite(values[lane]) ? values[lane] : 0.0F;
}

// Model only the new scheduling/address transformations. Integer decode and
// FP32 expressions use Fragment; expected values below remain the independent
// vendored-dequant reference. The last input row ends at its allocation bound.
template<int Type>
void CheckIqSchedules(const std::vector<std::uint8_t>& weights,
                      const std::vector<std::uint8_t>& input, int stride,
                      int blocks, int n, int row,
                      const std::array<float, 32> (&expected)[8]) {
  constexpr int grid_entries = Type == 18 ? 256 : 512;
  std::vector<std::uint32_t> grid;
  for (int i = 0; i < grid_entries; ++i) {
    const auto pair = Math::Grid<Type>(i);
    grid.push_back(pair.lo);
    if constexpr (Type == 17)
      grid.push_back(pair.hi);
  }
  constexpr int chunk_bytes = 4 * 8 * 36;
  std::vector<std::uint8_t> staged(n * chunk_bytes, 0xCD);
  for (auto variant :
       {d::Variant::kReuse1Cap6, d::Variant::kReuse1Lds,
        d::Variant::kReuse1LdsGrid, d::Variant::kReuse1SplitKLds}) {
    std::array<float, 32> sums[8]{};
    const int first =
        variant == d::Variant::kReuse1Cap6 ? d::Cap6FirstWidth(n) : n;
    const bool lds = variant == d::Variant::kReuse1Lds ||
                     variant == d::Variant::kReuse1SplitKLds;
    for (int base = 0; base < n; base += first) {
      const int count = std::min(first, n - base);
      for (int tile = 0; tile < blocks; tile += 4) {
        if (lds)
          for (int t = 0; t < n; ++t)
            std::memcpy(staged.data() + t * chunk_bytes,
                        input.data() + t * stride + tile * 8 * 36,
                        std::min(4, blocks - tile) * 8 * 36);
        for (int lane = 0; lane < 32; ++lane) {
          const int kb = tile + lane / 8;
          if (kb >= blocks)
            continue;
          const int iqs = 2 * (lane % 8);
          const auto* w =
              weights.data() + (row * blocks + kb) * d::BlockBytes(Type);
          const d::Fragment<Type, Math> fragment(
              w, iqs,
              variant == d::Variant::kReuse1LdsGrid ? grid.data() : nullptr);
          for (int t = base; t < base + count; ++t) {
            const auto* x =
                lds ? staged.data() + t * chunk_bytes + (kb - tile) * 8 * 36
                    : input.data() + t * stride + kb * 8 * 36;
            sums[t][lane] = fragment.Accumulate(x, iqs, sums[t][lane]);
          }
        }
      }
    }
    for (int t = 0; t < n; ++t) {
      const float value = variant == d::Variant::kReuse1SplitKLds
                              ? ReduceSplit(sums[t], row % 2)
                              : Reduce(sums[t], row % 2);
      Require(std::bit_cast<std::uint32_t>(value) ==
                  std::bit_cast<std::uint32_t>(Reduce(expected[t], row % 2)),
              d::VariantName(variant));
    }
  }
}

void CheckVariantControls() {
  Require(d::ParseVariant(nullptr) == -1 && d::ParseVariant("") == -1,
          "unset diagnostic selection");
  for (int v = 0; v < d::kVariantCount; ++v)
    Require(d::ParseVariant(d::VariantName(static_cast<d::Variant>(v))) == v,
            "diagnostic variant round trip");
  for (const char* invalid :
       {"reuse", "reuse1_lds_extra", "REUSE1", "reuse1,baseline", "unknown"})
    Require(d::ParseVariant(invalid) == -2,
            "unknown diagnostic name must fail");
  for (int n = 1; n <= 8; ++n) {
    const int first = d::Cap6FirstWidth(n);
    Require(first >= 1 && first <= 6 && first <= n && n - first <= 6,
            "capped passes cover each token exactly once");
  }
  for (int type : {17, 18, 21}) {
    Require(d::IsIqTarget(type) && !d::Covers(type, 8, 640) &&
                d::ProductionVariant(type, 8, 640) == d::Variant::kBaseline,
            "IQ block-unaligned K retains fallback");
    Require(d::ProductionVariant(type, 1, 2560) == d::Variant::kBaseline &&
                d::ProductionVariant(type, 8, 2560) == d::Variant::kReuse1,
            "production policy remains unchanged");
  }
  Require(!d::IsIqTarget(20) && !d::IsIqTarget(23) && !d::IsIqTarget(42),
          "diagnostic override excludes IQ4 and Q2_0");
  for (int pattern = 0; pattern < 3; ++pattern) {
    std::array<float, 32> values{};
    for (int lane = 0; lane < 32; ++lane)
      values[lane] = pattern == 0   ? (lane % 3 == 0   ? 1.0e20F
                                       : lane % 3 == 1 ? -1.0e20F
                                                       : 3.0F)
                     : pattern == 1 ? (lane & 1 ? -0.0F : 0.0F)
                                    : std::ldexp(float(lane - 17), lane % 13);
    for (int lane : {0, 1})
      Require(
          std::bit_cast<std::uint32_t>(ReduceSplit(values, lane)) ==
              std::bit_cast<std::uint32_t>(Reduce(values, lane)),
          "split-K butterfly must retain cancellation and signed-zero order");
  }
}

// Decode integer codes with the vendored dequantizer, using unit scale/minimum.
// The scalar dot below then applies the ORIGINAL block metadata and FP32 DAG.
void Codes(int type, const std::uint8_t* block, float* values) {
  std::vector<std::uint8_t> unit(block, block + d::BlockBytes(type));
  PutHalf(unit.data() + (type == 14 ? 208 : 0), 0x3C00);
  if (type == 17) {
    PutHalf(unit.data(), 0x4800);  // d=8, zero subscales: integer grids.
    std::fill_n(unit.data() + 66, 8, 0);
    q::DequantizeIQ2_XS(unit.data(), values, 256);
  } else if (type == 18) {
    PutHalf(unit.data(), 0x4400);  // d=4, zero subscales: integer grids.
    for (int group = 0; group < 8; ++group)
      unit[69 + group * 4] &= 15;
    q::DequantizeIQ3_XXS(unit.data(), values, 256);
  } else if (type == 21) {
    std::fill_n(unit.data() + 106, 4, 0);
    q::DequantizeIQ3_S(unit.data(), values, 256);
  } else if (type == 23) {
    PutHalf(unit.data() + 2, 0xAAAA);
    std::fill_n(unit.data() + 4, 4, 0x11);
    q::DequantizeIQ4_XS(unit.data(), values, 256);
  } else if (type == 14) {
    std::fill_n(unit.data() + 192, 16, 1);
    q::DequantizeQ6_K(unit.data(), values, 256);
  } else if (type == 20)
    q::DequantizeIQ4_NL(unit.data(), values, 32);
  else if (type == 7) {
    PutHalf(unit.data() + 2, 0);
    q::DequantizeQ5_1(unit.data(), values, 32);
  } else {
    PutHalf(unit.data() + 2, 0);
    const std::uint8_t scales[]{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
    std::copy_n(scales, 12, unit.data() + 4);
    if (type == 12)
      q::DequantizeQ4_K(unit.data(), values, 256);
    else
      q::DequantizeQ5_K(unit.data(), values, 256);
  }
}
template<int Type>
void CheckKQuantStaging() {
  struct Sink {
    std::array<std::uint32_t, 64> words{}, writes{};
    std::array<float, 16> scales{}, minima{};
    std::array<int, 16> scale_writes{}, minimum_writes{};
    void Word(int element, std::uint32_t value) {
      words.at(element / 4) = value;
      ++writes.at(element / 4);
    }
    void Scale(int group, float value) {
      scales.at(group) = value;
      ++scale_writes.at(group);
    }
    void Minimum(int group, float value) {
      minima.at(group) = value;
      ++minimum_writes.at(group);
    }
  };
  std::uint32_t seed = 71239 + Type;
  for (int sample = 0; sample < 100; ++sample) {
    std::vector<std::uint8_t> w(d::BlockBytes(Type));
    for (auto& byte : w)
      byte = std::uint8_t(Next(seed));
    PutHalf(w.data() + (Type == 14 ? 208 : 0), 0x2800 | (Next(seed) & 1023));
    if constexpr (Type != 14)
      PutHalf(w.data() + 2, 0x2400 | (Next(seed) & 1023));
    Sink sink;
    for (int iqs = 0; iqs < 32; iqs += Type == 14 ? 1 : 2)
      d::Fragment<Type, Math>(w.data(), iqs).StoreKTile(iqs, sink);
    float codes[256], decoded[256];
    Codes(Type, w.data(), codes);
    if constexpr (Type == 14)
      q::DequantizeQ6_K(w.data(), decoded, 256);
    else if constexpr (Type == 13)
      q::DequantizeQ5_K(w.data(), decoded, 256);
    else
      q::DequantizeQ4_K(w.data(), decoded, 256);
    for (int word = 0; word < 64; ++word)
      Require(sink.writes[word] == 1, "every staged code word has one owner");
    constexpr int groups = Type == 14 ? 16 : 8;
    for (int group = 0; group < groups; ++group) {
      Require(sink.scale_writes[group] == 1, "one scale writer per subgroup");
      if constexpr (Type != 14)
        Require(sink.minimum_writes[group] == 1, "one minimum writer");
    }
    for (int i = 0; i < 256; ++i) {
      const int integer = Signed(sink.words[i / 4] >> (8 * (i % 4)));
      Require(integer == int(codes[i]),
              "staged signed bytes match dequantizer");
      const int group = i / (Type == 14 ? 16 : 32);
      const float value = sink.scales[group] * integer - sink.minima[group];
      Require(std::abs(value - decoded[i]) < 1e-5F,
              "staged metadata matches independent dequantizer");
    }
  }
}

void CheckKQuantDispatchAndLayout() {
  std::array<int, 1024> owners{};
  for (int row = 0; row < 16; ++row)
    for (int k = 0; k < 256; k += 4) {
      const int word = d::KQuantStageWordIndex(row, k);
      Require(word >= 0 && word < 1024, "LDS swizzle remains inside tile");
      ++owners[word];
      if (k % 16 == 0)
        Require(word % 4 == 0, "matrix fragment has 16-byte alignment");
    }
  for (int count : owners)
    Require(count == 1, "LDS swizzle is bijective");
  for (int type : {8, 12, 13, 14, 17, 18, 20, 21, 23})
    for (int n : {1, 2, 3, 4, 8, 9, 16})
      for (const auto [k, m] : {std::pair{2560, 6144},
                                {2560, 10240},
                                {2560, 12288},
                                {6144, 2560},
                                {2560, 248320},
                                {2560, 512},
                                {2560, 640},
                                {640, 2560},
                                {10240, 320}}) {
        const auto current = d::ProductionVariant(type, n, k);
        const auto candidate =
            d::SelectVariant(type, n, k, m, int(d::Variant::kKQuantWmma));
        const bool expected = (type == 12 || type == 13 || type == 14) &&
                              n >= 4 && n <= 8 &&
                              ((k == 2560 && m >= 6144 && m <= 12288) ||
                               (k == 6144 && m == 2560));
        Require(candidate == (expected ? d::Variant::kKQuantWmma : current),
                "candidate excludes head, skinny rows, Q8, IQ and N1..3");
        Require(d::SelectVariant(type, n, k, m, -1) == current,
                "unset override retains production selection");
      }
}

// The dot4 override: chosen by (format, K) only, never by width or M (no
// shape list), width one included even with GUFO_DENSE_W1=0, and every other
// call keeps its route with a named reason.
void CheckDot4Selection() {
  Require(d::ParseVariant("dot4") == int(d::Variant::kDot4) &&
              d::ParseVariant("dot") == -2 && d::ParseVariant("dot4 ") == -2 &&
              d::ParseVariant("DOT4") == -2,
          "dot4 parses strictly");
  for (int type : {7, 8, 12, 13, 14, 17, 18, 20, 21, 23}) {
    Require(d::Dot4Selected(type) == (type == 12 || type == 13 || type == 14),
            "dot4 selects Q4_K, Q5_K and Q6_K");
    Require(!d::Dot4Selected(type) || d::Dot4Kernel(type),
            "every selected format is instantiated");
    // Non-Qwen shapes, an M tail, small M, odd K256-block counts.
    for (const auto [k, m] :
         {std::pair{2560, 10240}, {6144, 2560}, {4096, 14336}, {14336, 4096},
          {5120, 13824}, {8192, 28672}, {768, 4097}, {3840, 512},
          {256, 1}, {2560, 248320}, {640, 2560}, {320, 10240}, {32, 64}})
      for (int n = 1; n <= 8; ++n)
        for (bool width_one : {true, false}) {
          const auto route = d::Dot4RouteFor(type, n, k);
          const bool dot4 =
              (type == 12 || type == 13 || type == 14) && k % 256 == 0;
          Require((route == d::Dot4Route::kDot4) == dot4,
                  "dot4 route is (format, K) only");
          Require(dot4 || route == (d::Dot4Selected(type)
                                        ? d::Dot4Route::kKTail
                                        : d::Dot4Route::kFormat),
                  "kept routes name their reason");
          const auto chosen = d::SelectVariant(
              type, n, k, m, int(d::Variant::kDot4), width_one);
          const auto kept = d::SelectVariant(type, n, k, m, -1, width_one);
          Require(chosen == (dot4 ? d::Variant::kDot4 : kept),
                  "dot4 or the unchanged production route");
        }
  }
  Require(d::Dot4RouteFor(13, 9, 2560) == d::Dot4Route::kWidth &&
              d::Dot4RouteFor(14, 0, 2560) == d::Dot4Route::kWidth,
          "the per-launch width contract is 1..8");
  for (int r = 0; r < d::kDot4RouteCount; ++r)
    Require(d::Dot4RouteName(d::Dot4Route(r)) != nullptr, "route names");
}

// The ablation stages are bench-only: they parse as stages, never as
// production variants, and the variant table keeps its size.
void CheckAblationStages() {
  Require(d::kVariantCount == 9 &&
              d::Variant(d::kVariantCount - 1) == d::Variant::kDot4,
          "ablation stages do not add dense variants (dot4 is the last)");
  Require(d::kStageFull == d::kStageCount && d::kStageCount == 8,
          "stage D is the full kernel");
  for (int stage = 1; stage <= d::kStageCount; ++stage) {
    const char* name = d::KQuantStageName(stage);
    Require(name != nullptr, "every stage has a name");
    Require(d::ParseKQuantStage(name) == stage, "stage names round-trip");
    Require(d::ParseVariant(name) == -2, "a stage name is not a variant");
    for (int other = stage + 1; other <= d::kStageCount; ++other)
      Require(std::string_view(name) != d::KQuantStageName(other),
              "stage names are unique");
  }
  Require(d::KQuantStageName(0) == nullptr &&
              d::KQuantStageName(d::kStageCount + 1) == nullptr,
          "stage range");
  Require(d::ParseKQuantStage("") == -1 && d::ParseKQuantStage("E") == -1 &&
              d::ParseKQuantStage(nullptr) == -1 &&
              d::ParseKQuantStage("kquant_wmma") == -1,
          "unknown stage names are refused");
  for (int type : {12, 13, 14})
    for (int n = 1; n <= 8; ++n)
      Require(d::SelectVariant(type, n, 2560, 10240, -1) ==
                  d::ProductionVariant(type, n, 2560),
              "production selection untouched by the ablation entries");
}

template<int Type>
void Compare() {
  constexpr int qk = d::BlockSize(Type), qi = d::Qi(Type), vdr = d::Vdr(Type);
  std::uint32_t seed = 381727U + Type;
  for (int shape_case = 0; shape_case < 6; ++shape_case) {
    const int pattern = shape_case / 2;
    const int k = shape_case % 2 ? 2560 : 3 * qk;
    const int blocks = k / qk, stride = ((k + 511) / 512) * 16 * 36;
    std::vector<std::uint8_t> weights(3 * blocks * d::BlockBytes(Type));
    std::vector<std::uint8_t> x(7 * stride + (k / 32) * 36);
    std::vector<float> decoded(3 * k);
    for (auto& value : weights)
      value = std::uint8_t(Next(seed));
    for (auto& value : x)
      value = std::uint8_t(Next(seed));
    for (int b = 0; b < 3 * blocks; ++b) {
      auto* w = weights.data() + b * d::BlockBytes(Type);
      PutHalf(w + (Type == 14 ? 208 : 0),
              std::uint16_t(0x2800 | (Next(seed) & 1023)));
      if constexpr (Type == 12 || Type == 13 || Type == 7)
        PutHalf(w + 2, std::uint16_t(0x2800 | (Next(seed) & 1023)));
      Codes(Type, w, decoded.data() + b * qk);
    }
    for (int t = 0; t < 8; ++t)
      for (int b = 0; b < k / 32; ++b) {
        auto* a = x.data() + t * stride + b * 36;
        if (pattern != 0)
          for (int j = 0; j < 32; ++j)
            a[4 + j] = pattern == 1 ? 0 : (j % 2 ? 0x80 : 0x7F);
        PutHalf(x.data() + t * stride + b * 36,
                std::uint16_t(0x3000 | (Next(seed) & 1023)));
      }
    for (int n = 1; n <= 8; ++n)
      for (int row = 0; row < 3; ++row) {
        std::array<float, 32> actual[8]{}, expected[8]{};
        for (int lane = 0; lane < 32; ++lane)
          for (int kb = lane / (qi / vdr); kb < blocks; kb += 32 * vdr / qi) {
            const int iqs = vdr * (lane % (qi / vdr));
            const auto* w =
                weights.data() + (row * blocks + kb) * d::BlockBytes(Type);
            const d::Fragment<Type, Math> fragment(w, iqs);
            for (int t = 0; t < n; ++t) {
              const auto* a = x.data() + t * stride + kb * (qk / 32) * 36;
              actual[t][lane] = fragment.Accumulate(a, iqs, actual[t][lane]);
              expected[t][lane] =
                  ReferenceStep(Type, w, a, decoded.data() + row * k + kb * qk,
                                iqs, expected[t][lane]);
            }
          }
        for (int t = 0; t < n; ++t)
          Require(
              std::bit_cast<std::uint32_t>(Reduce(actual[t], row % 2)) ==
                  std::bit_cast<std::uint32_t>(Reduce(expected[t], row % 2)),
              "multi-row arithmetic differs from scalar reference");
        if constexpr (d::IsIqTarget(Type))
          CheckIqSchedules<Type>(weights, x, stride, blocks, n, row, expected);
      }
  }
  for (int n : {1, 2, 3, 4, 8, 9}) {
    Require(d::Covers(Type, n, 2560) == (n >= 1 && n <= 8),
            "row dispatch boundary");
    const bool selected =
        n >= 1 && n <= 8 &&
        (n == 1 ? Type != 17 && Type != 18 && Type != 21
                : Type == 13 || Type == 17 || Type == 18 || Type == 21 ||
                      ((Type == 12 || Type == 14) && n >= 2) ||
                      (Type == 7 && n >= 4));
    Require(d::ProductionVariant(Type, n, 2560) ==
                (selected ? d::Variant::kReuse1 : d::Variant::kBaseline),
            "production format/width gate");
    Require(d::ProductionVariant(Type, n, 2561) == d::Variant::kBaseline,
            "unsupported K falls back");
  }
}
}  // namespace
int main() try {
  CheckVariantControls();
  CheckKQuantDispatchAndLayout();
  CheckAblationStages();
  CheckDot4Selection();
  CheckKQuantStaging<12>();
  CheckKQuantStaging<13>();
  CheckKQuantStaging<14>();
  Compare<23>();
  Compare<20>();
  Compare<12>();
  Compare<13>();
  Compare<7>();
  Compare<14>();
  Compare<21>();
  Compare<18>();
  Compare<17>();
  std::printf(
      "PASS: scalar reference, nine formats, R=1..8 and R=9 fallback; %llu "
      "bit comparisons and dispatch checks\n",
      static_cast<unsigned long long>(checks));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
