#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/iq_grids.hpp"
#include "src/models/qwen38_flash_next/iq_decode_bits.hpp"

// Include the immutable device tables without the HIP block declarations.
#define GGML_COMMON_DECL
#define __device__
namespace device_tables {
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/ggml-common.h"
}
#undef __device__
#undef GGML_COMMON_DECL

namespace {

namespace bits = gufo::models::qwen38_flash_next::iq_decode;
using gufo::quant::block_iq2_s;
using gufo::quant::block_q2_0;
using gufo::quant::Fp16ToFloat;

std::uint64_t checks = 0;

void Require(bool condition, const char* what) {
  ++checks;
  if (!condition)
    throw std::runtime_error(what);
}

std::uint32_t Next(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

template<typename T>
void Randomize(T& value, std::uint32_t& state) {
  auto* bytes = reinterpret_cast<std::uint8_t*>(&value);
  for (std::size_t i = 0; i < sizeof(value); ++i)
    bytes[i] = static_cast<std::uint8_t>(Next(state));
}

int SignedByte(std::uint32_t word, int byte) {
  const int value = (word >> (8 * byte)) & 255;
  return value < 128 ? value : value - 256;
}

struct Q8 {
  std::uint16_t scale;
  std::array<std::int8_t, 32> qs;
};

int DotWord(std::uint32_t word, const std::int8_t* x) {
  int sum = 0;
  for (int j = 0; j < 4; ++j)
    sum += SignedByte(word, j) * int(x[j]);
  return sum;
}

int IntegerDot(const block_q2_0& w, const Q8* x, int chunk, bool packed,
               const std::uint64_t*) {
  int sum = 0;
  if (packed) {
    for (int j = 0; j < 8; ++j)
      sum += DotWord(bits::ExpandQ2Byte(w.qs[8 * chunk + j]),
                     x[chunk].qs.data() + 4 * j);
  } else {
    for (int j = 0; j < 32; ++j) {
      const int code = (w.qs[8 * chunk + j / 4] >> (2 * (j % 4))) & 3;
      sum += (code - 1) * int(x[chunk].qs[j]);
    }
  }
  return sum;
}

int IntegerDot(const block_iq2_s& w, const Q8* x, int chunk, bool packed,
               const std::uint64_t* grid) {
  int sum[2] = {};
  for (int l = 0; l < 4; ++l) {
    const int index =
        w.qs[4 * chunk + l] | ((int(w.qh[chunk]) << (8 - 2 * l)) & 0x300);
    const std::uint64_t values = grid[index];
    const std::uint32_t signs = w.qs[32 + 4 * chunk + l];
    if (packed) {
      const auto lo = bits::ApplyIq2Signs(std::uint32_t(values), signs);
      const auto hi =
          bits::ApplyIq2Signs(std::uint32_t(values >> 32), signs >> 4);
      sum[l / 2] += DotWord(lo, x[chunk].qs.data() + 8 * l);
      sum[l / 2] += DotWord(hi, x[chunk].qs.data() + 8 * l + 4);
    } else {
      for (int j = 0; j < 8; ++j) {
        const int magnitude = (values >> (8 * j)) & 255;
        const int weight = (signs & (1U << j)) ? -magnitude : magnitude;
        sum[l / 2] += weight * int(x[chunk].qs[8 * l + j]);
      }
    }
  }
  if (packed)
    return bits::ScaleIq2S(sum[0], sum[1], w.scales[chunk]);
  const int ls0 = w.scales[chunk] & 15;
  const int ls1 = w.scales[chunk] >> 4;
  return (sum[0] * ls0 + sum[1] * ls1 + (sum[0] + sum[1]) / 2) / 4;
}

int IntegerDot(const gufo::quant::block_iq2_xxs& w, const Q8* x, int chunk,
               bool packed, const std::uint64_t*) {
  std::uint32_t words[2];
  std::memcpy(words, reinterpret_cast<const std::uint8_t*>(w.qs) + 8 * chunk,
              sizeof(words));
  int sum = 0;
  for (int l = 0; l < 4; ++l) {
    const auto values = gufo::quant::Iq2xxsGrid()[(words[0] >> (8 * l)) & 255];
    const int code = (words[1] >> (7 * l)) & 127;
    if (packed) {
      const auto signs = bits::ParitySigns(words[1] >> (7 * l));
      sum += DotWord(bits::ApplyIq2Signs(std::uint32_t(values), signs),
                     x[chunk].qs.data() + 8 * l);
      sum +=
          DotWord(bits::ApplyIq2Signs(std::uint32_t(values >> 32), signs >> 4),
                  x[chunk].qs.data() + 8 * l + 4);
    } else {
      const int signs = gufo::quant::SignsIq2xs()[code];
      for (int j = 0; j < 8; ++j) {
        const int magnitude = int((values >> (8 * j)) & 255);
        sum += (signs & (1 << j) ? -magnitude : magnitude) *
               int(x[chunk].qs[8 * l + j]);
      }
    }
  }
  return packed ? bits::ScaleIq2Xxs(sum, words[1])
                : int(std::trunc(double(sum) * (2 * int(words[1] >> 28) + 1) /
                                 8.0));
}

int IntegerDot(const gufo::quant::block_iq2_xs& w, const Q8* x, int chunk,
               bool packed, const std::uint64_t*) {
  int sum[2] = {};
  for (int l = 0; l < 4; ++l) {
    const auto code = w.qs[4 * chunk + l];
    const auto values = gufo::quant::Iq2xsGrid()[code & 511];
    const int signs = packed ? int(bits::ParitySigns(code >> 9))
                             : gufo::quant::SignsIq2xs()[code >> 9];
    if (packed) {
      sum[l / 2] += DotWord(bits::ApplyIq2Signs(std::uint32_t(values), signs),
                            x[chunk].qs.data() + 8 * l);
      sum[l / 2] +=
          DotWord(bits::ApplyIq2Signs(std::uint32_t(values >> 32), signs >> 4),
                  x[chunk].qs.data() + 8 * l + 4);
    } else {
      for (int j = 0; j < 8; ++j) {
        const int magnitude = int((values >> (8 * j)) & 255);
        sum[l / 2] += (signs & (1 << j) ? -magnitude : magnitude) *
                      int(x[chunk].qs[8 * l + j]);
      }
    }
  }
  return packed ? bits::ScaleIq2S(sum[0], sum[1], w.scales[chunk])
                : (sum[0] * (w.scales[chunk] & 15) +
                   sum[1] * (w.scales[chunk] >> 4) + (sum[0] + sum[1]) / 2) /
                      4;
}

template<typename Block>
int Iq3IntegerDot(const Block& w, const Q8* x, int chunk, bool packed) {
  constexpr bool small = std::is_same_v<Block, gufo::quant::block_iq3_s>;
  std::uint32_t sign_word;
  if constexpr (small)
    std::memcpy(&sign_word, w.signs + 4 * chunk, sizeof(sign_word));
  else
    std::memcpy(&sign_word, w.qs + 64 + 4 * chunk, sizeof(sign_word));
  int sum = 0;
  if (packed) {
    for (int l = 0; l < 8; ++l) {
      int index = w.qs[8 * chunk + l];
      if constexpr (small)
        index |= ((w.qh[chunk] >> l) & 1) << 8;
      const auto grid = small ? gufo::quant::Iq3sGrid()[index]
                              : gufo::quant::Iq3xxsGrid()[index];
      const auto signs = small ? sign_word >> (8 * (l / 2))
                               : bits::ParitySigns(sign_word >> (7 * (l / 2)));
      const auto word = bits::ApplyIq2Signs(grid, signs >> (4 * (l % 2)));
      sum += DotWord(word, x[chunk].qs.data() + 4 * l);
    }
  } else {
    for (int j = 0; j < 32; ++j) {
      int index = w.qs[8 * chunk + j / 4];
      if constexpr (small)
        index |= ((w.qh[chunk] >> (j / 4)) & 1) << 8;
      const auto grid = small ? gufo::quant::Iq3sGrid()[index]
                              : gufo::quant::Iq3xxsGrid()[index];
      const int magnitude = (grid >> (8 * (j % 4))) & 255;
      int signs;
      if constexpr (small)
        signs = w.signs[4 * chunk + j / 8];
      else
        signs = gufo::quant::SignsIq2xs()[(sign_word >> (7 * (j / 8))) & 127];
      sum += (signs & (1 << (j % 8)) ? -magnitude : magnitude) *
             int(x[chunk].qs[j]);
    }
  }
  if constexpr (small) {
    const int scale = w.scales[chunk / 2] >> (4 * (chunk % 2));
    return packed ? bits::ScaleIq3S(sum, scale) : sum * (1 + 2 * (scale & 15));
  } else {
    const int scale = sign_word >> 28;
    return packed ? bits::ScaleIq3Xxs(sum, scale)
                  : int(std::trunc(
                        (double(scale) * sum + std::trunc(double(sum) / 2.0)) /
                        2.0));
  }
}

int IntegerDot(const gufo::quant::block_iq3_s& w, const Q8* x, int chunk,
               bool packed, const std::uint64_t*) {
  return Iq3IntegerDot(w, x, chunk, packed);
}
int IntegerDot(const gufo::quant::block_iq3_xxs& w, const Q8* x, int chunk,
               bool packed, const std::uint64_t*) {
  return Iq3IntegerDot(w, x, chunk, packed);
}

float Round(float value) {
  volatile float rounded = value;
  return rounded;
}

// Model the existing expression under each relevant contraction choice.
float Accumulate(float acc, float dw, float dx, int integer, int mode) {
  if (mode == 0)
    return Round(acc + Round(Round(dw * dx) * float(integer)));
  if (mode == 1)
    return std::fma(Round(dw * dx), float(integer), acc);
  return std::fma(Round(dw * float(integer)), dx, acc);
}

float Finite(float value) {
  return std::isfinite(value) ? value : 0.0F;
}

std::uint32_t Word(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

void ExhaustCodesAndSigns() {
  const auto* grid = gufo::quant::Iq2sGrid();
  for (int index = 0; index < 1024; ++index) {
    Require(grid[index] == device_tables::iq2s_grid[index],
            "independent core and device IQ2_S grids differ");
    for (int j = 0; j < 8; ++j) {
      const int magnitude = (grid[index] >> (8 * j)) & 255;
      Require(magnitude > 0 && magnitude < 128,
              "IQ2_S grid violates carry-free sign precondition");
    }
    block_iq2_s w{};
    w.d = 0x4800;
    std::fill_n(w.qs, 32, std::uint8_t(index));
    std::fill_n(w.qh, 8, std::uint8_t((index >> 8) * 0x55));
    for (int signs = 0; signs < 256; ++signs) {
      std::fill_n(w.qs + 32, 32, std::uint8_t(signs));
      std::array<float, 256> reference{};
      gufo::quant::DequantizeIQ2_S(&w, reference.data(), reference.size());
      const std::uint32_t words[2] = {
          bits::ApplyIq2Signs(std::uint32_t(grid[index]), signs),
          bits::ApplyIq2Signs(std::uint32_t(grid[index] >> 32), signs >> 4)};
      for (int j = 0; j < 256; ++j) {
        const int magnitude = (grid[index] >> (8 * (j % 8))) & 255;
        const int expected = signs & (1 << (j % 8)) ? -magnitude : magnitude;
        Require(reference[j] == float(expected),
                "IQ2_S index/high-bit/sign layout differs from CPU reference");
        Require(SignedByte(words[(j % 8) / 4], j % 4) == expected,
                "IQ2_S packed signs differ from independent byte reference");
      }
    }
  }
  for (int code = 0; code < 256; ++code) {
    block_q2_0 w{};
    w.d = 0x3C00;
    std::fill_n(w.qs, 16, std::uint8_t(code));
    std::array<float, 64> reference{};
    gufo::quant::DequantizeQ2_0(&w, reference.data(), reference.size());
    const auto packed = bits::ExpandQ2Byte(code);
    for (int j = 0; j < 64; ++j) {
      const int expected = ((code >> (2 * (j % 4))) & 3) - 1;
      Require(reference[j] == float(expected), "Q2_0 CPU byte order differs");
      Require(SignedByte(packed, j % 4) == expected,
              "Q2_0 packed byte expansion or borrow differs");
    }
  }
  Require(bits::ScaleIq2S(-7, 0, 0) == 0,
          "IQ2_S divisions must truncate toward zero");
  for (int scale = 0; scale < 256; ++scale) {
    for (int low : {-260096, -257, -7, -1, 0, 1, 7, 257, 260096}) {
      for (int high : {-260096, -257, -7, -1, 0, 1, 7, 257, 260096}) {
        const int reference =
            (low * (scale & 15) + high * (scale >> 4) + (low + high) / 2) / 4;
        Require(bits::ScaleIq2S(low, high, scale) == reference,
                "IQ2_S scale nibble or signed division boundary differs");
      }
    }
  }
}

constexpr std::array<std::uint16_t, 16> kScales{
    0,      0x8000, 1,      0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400,
    0x1C00, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E01};

template<typename Block, int Chunks>
void CheckFragments() {
  std::uint32_t state = 0xAE103987;
  const auto* grid = gufo::quant::Iq2sGrid();
  for (int sample = 0; sample < 4096; ++sample) {
    Block w;
    Randomize(w, state);
    std::array<Q8, Chunks> x;
    for (auto& chunk : x)
      Randomize(chunk, state);
    if (sample < 8) {
      constexpr std::int8_t extremes[]{-128, -127, -1, 0, 1, 2, 126, 127};
      for (auto& chunk : x)
        chunk.qs.fill(extremes[sample]);
    }
    for (int chunk = 0; chunk < Chunks; ++chunk)
      Require(IntegerDot(w, x.data(), chunk, false, grid) ==
                  IntegerDot(w, x.data(), chunk, true, grid),
              "random integer fragment differs");
  }
  Block w;
  Randomize(w, state);
  std::array<Q8, Chunks> x;
  for (auto& chunk : x)
    Randomize(chunk, state);
  for (unsigned half = 0; half < 65536; ++half) {
    for (int swap = 0; swap < 2; ++swap) {
      w.d = swap ? kScales[half % kScales.size()] : std::uint16_t(half);
      x[0].scale = swap ? std::uint16_t(half) : kScales[half % kScales.size()];
      const int old_dot = IntegerDot(w, x.data(), 0, false, grid);
      const int new_dot = IntegerDot(w, x.data(), 0, true, grid);
      for (int mode = 0; mode < 3; ++mode) {
        const float prefix = half & 1 ? 0x1.000002p-20F : -0x1.0p10F;
        const float old_sum = Accumulate(
            prefix, Fp16ToFloat(w.d), Fp16ToFloat(x[0].scale), old_dot, mode);
        const float new_sum = Accumulate(
            prefix, Fp16ToFloat(w.d), Fp16ToFloat(x[0].scale), new_dot, mode);
        Require(Word(Finite(old_sum)) == Word(Finite(new_sum)),
                "half-scale boundary or FP32 contraction model differs");
      }
    }
  }
}

template<typename T>
void Put(std::vector<std::uint8_t>& bytes, T value) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
  bytes.insert(bytes.end(), p, p + sizeof(value));
}

void CheckNewTypeHeaders() {
  using gufo::core::GgmlType;
  for (auto type : {GgmlType::kIQ2_XXS, GgmlType::kIQ2_XS, GgmlType::kIQ1_M}) {
    const std::size_t size = type == GgmlType::kIQ2_XXS  ? 66
                             : type == GgmlType::kIQ2_XS ? 74
                                                         : 56;
    Require(gufo::quant::QuantizedBlockElements(type) == 256,
            "new type block elements differ");
    Require(gufo::quant::EncodedSizeBytes(type, 256) == size &&
                gufo::quant::EncodedSizeBytes(type, 255) == 0,
            "new type encoded size differs");
    std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
    Put<std::uint32_t>(bytes, 3);
    Put<std::uint64_t>(bytes, 1);
    Put<std::uint64_t>(bytes, 0);
    constexpr char name[] = "iq.weight";
    Put<std::uint64_t>(bytes, sizeof(name) - 1);
    bytes.insert(bytes.end(), name, name + sizeof(name) - 1);
    Put<std::uint32_t>(bytes, 3);
    const auto shape_offset = bytes.size();
    Put<std::uint64_t>(bytes, 256);
    Put<std::uint64_t>(bytes, 3);
    Put<std::uint64_t>(bytes, 2);
    Put<std::uint32_t>(bytes, static_cast<std::uint32_t>(type));
    const auto tensor_offset = bytes.size();
    Put<std::uint64_t>(bytes, 0);
    bytes.resize((bytes.size() + 31) / 32 * 32 + 6 * size);
    std::string error;
    const auto reader =
        gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
    Require(reader != nullptr, "new type GGUF header rejected");
    const auto* tensor = reader->FindTensor(name);
    Require(tensor && tensor->type == type && tensor->size_bytes == 6 * size,
            "new type GGUF extent differs");
    const auto truncated = gufo::core::GgufReader::OpenMemory(
        bytes.data(), bytes.size() - 1, &error);
    Require(!truncated, "truncated new type GGUF accepted");
    auto invalid = bytes;
    constexpr std::uint64_t huge = std::uint64_t(1) << 63;
    std::memcpy(invalid.data() + shape_offset, &huge, sizeof(huge));
    Require(!gufo::core::GgufReader::OpenMemory(invalid.data(), invalid.size(),
                                                &error),
            "overflowing new type dimensions accepted");
    invalid = bytes;
    constexpr std::uint64_t bad_offset = ~std::uint64_t(0);
    std::memcpy(invalid.data() + tensor_offset, &bad_offset,
                sizeof(bad_offset));
    Require(!gufo::core::GgufReader::OpenMemory(invalid.data(), invalid.size(),
                                                &error),
            "overflowing new type tensor offset accepted");
  }
  Require(static_cast<int>(GgmlType::kIQ1_M) == 29 &&
              gufo::core::ToString(GgmlType::kIQ1_M) == "IQ1_M",
          "IQ1_M type id/name differs");
}

void SetIq1Half(gufo::quant::block_iq1_m& w, std::uint16_t half, int low,
                int high) {
  std::uint16_t sc[4];
  for (int i = 0; i < 4; ++i)
    sc[i] = std::uint16_t(((half >> (4 * i)) & 15) << 12) |
            std::uint16_t(low | (high << 3) | (high << 6) | (low << 9));
  std::memcpy(w.scales, sc, sizeof(sc));
}

void CheckIq1Layout() {
  using gufo::quant::block_iq1_m;
  const auto* grid = gufo::quant::Iq1sGrid();
  for (int index = 0; index < 2048; ++index) {
    const auto packed = device_tables::iq1s_grid_gpu[index];
    for (int j = 0; j < 8; ++j) {
      const int value =
          SignedByte(std::uint32_t(grid[index] >> (32 * (j / 4))), j % 4);
      const int nibble = (packed >> (8 * (j % 4) + 4 * (j / 4))) & 15;
      Require(nibble >= 0 && nibble <= 2 && value == nibble - 1,
              "IQ1 CPU/device grid or nibble ordering differs");
      for (bool negative : {false, true}) {
        const auto word = bits::ExpandIq1(packed >> (4 * (j / 4)), negative);
        Require(SignedByte(word, j % 4) == 8 * value + (negative ? -1 : 1),
                "IQ1 packed expansion differs from signed scalar byte");
      }
    }
    for (int signs = 0; signs < 16; ++signs) {
      for (int scale = 0; scale < 8; ++scale) {
        block_iq1_m w{};
        SetIq1Half(w, 0x3C00, scale, (scale + 3) % 8);
        for (int group = 0; group < 32; ++group) {
          w.qs[group] = std::uint8_t(index);
          const auto high = (index >> 8) | (((signs >> (group % 4)) & 1) << 3);
          w.qh[group / 2] |= std::uint8_t(high << (4 * (group % 2)));
        }
        std::array<float, 258> decoded;
        decoded.fill(std::bit_cast<float>(0x7FCABCDEU));
        gufo::quant::DequantizeIQ1_M(&w, decoded.data() + 1, 256);
        Require(Word(decoded.front()) == 0x7FCABCDEU &&
                    Word(decoded.back()) == 0x7FCABCDEU,
                "IQ1 CPU dequantizer overwrites guard");
        for (int j = 0; j < 256; ++j) {
          const int ib = j / 32;
          const int sub = (j % 32) / 16;
          const int code = ib % 2 == 0 ? (sub == 0 ? scale : (scale + 3) % 8)
                                       : (sub == 0 ? (scale + 3) % 8 : scale);
          Require(bits::Iq1Scale(w.scales, ib, sub) == 2 * code + 1,
                  "IQ1 shared subscale unpack differs");
          const int value = SignedByte(
              std::uint32_t(grid[index] >> (32 * ((j % 8) / 4))), j % 4);
          const float delta = signs & (1 << ((j / 8) % 4)) ? -0.125F : 0.125F;
          Require(Word(decoded[j + 1]) ==
                      Word(float(2 * code + 1) * (value + delta)),
                  "IQ1 index/high bits/subscale/delta differs from scalar "
                  "reference");
        }
      }
    }
  }
  block_iq1_m w{};
  std::array<float, 256> decoded;
  for (int word = 0; word < 4; ++word) {
    for (unsigned low = 0; low < 4096; ++low) {
      SetIq1Half(w, 0x3C00, 0, 0);
      std::uint16_t scales[4];
      std::memcpy(scales, w.scales, sizeof(scales));
      scales[word] |= std::uint16_t(low);
      std::memcpy(w.scales, scales, sizeof(scales));
      Require(bits::Iq1Half(w.scales) == 0x3C00,
              "IQ1 lower scale bits alter the shared half");
      gufo::quant::DequantizeIQ1_M(&w, decoded.data(), 256);
      for (int chunk = 0; chunk < 8; ++chunk) {
        for (int sub = 0; sub < 2; ++sub) {
          const unsigned packed = chunk / 2 == word ? low : 0;
          const int odd = 1 + 2 * ((packed >> (6 * (chunk % 2) + 3 * sub)) & 7);
          Require(bits::Iq1Scale(w.scales, chunk, sub) == odd,
                  "IQ1 independent scale-word or per-16 packing differs");
          Require(Word(decoded[32 * chunk + 16 * sub]) == Word(-0.875F * odd),
                  "IQ1 CPU per-16 scale differs from independent packed bits");
        }
      }
    }
  }
  for (unsigned half = 0; half < 65536; ++half) {
    SetIq1Half(w, std::uint16_t(half), 2, 5);
    Require(bits::Iq1Half(w.scales) == half, "IQ1 shared half unpack differs");
    gufo::quant::DequantizeIQ1_M(&w, decoded.data(), 256);
    const float expected = Fp16ToFloat(std::uint16_t(half)) * 5.0F * -0.875F;
    Require(std::isnan(expected) ? std::isnan(decoded[0])
                                 : Word(expected) == Word(decoded[0]),
            "IQ1 packed half-scale pattern differs");
  }
}

void CheckIq2Tables() {
  for (int code = 0; code < 128; ++code) {
    for (auto upper : {0U, 0xFFFFFF80U, 0x73ABDE00U})
      Require(
          bits::ParitySigns(code | upper) == gufo::quant::SignsIq2xs()[code],
          "packed parity signs or seven-bit mask differs");
  }
  for (int scale = 0; scale < 16; ++scale) {
    for (int sum : {-176128, -257, -9, -8, -7, -1, 0, 1, 7, 8, 9, 257, 176128})
      Require(
          bits::ScaleIq2Xxs(sum, (std::uint32_t(scale) << 28) | 0x0FFFFFFFU) ==
              int(std::trunc(double(sum) * (2 * scale + 1) / 8.0)),
          "IQ2_XXS signed division boundary differs");
  }
  for (bool xs : {false, true}) {
    for (int index = 0; index < (xs ? 512 : 256); ++index) {
      const auto grid = xs ? gufo::quant::Iq2xsGrid()[index]
                           : gufo::quant::Iq2xxsGrid()[index];
      Require(grid == (xs ? device_tables::iq2xs_grid[index]
                          : device_tables::iq2xxs_grid[index]),
              "IQ2 CPU/device grid differs");
      for (int code = 0; code < 128; ++code) {
        gufo::quant::block_iq2_xxs a{};
        gufo::quant::block_iq2_xs b{};
        a.d = b.d = 0x4800;
        for (int chunk = 0; chunk < 8; ++chunk) {
          const std::uint32_t words[]{
              std::uint32_t(index) * 0x01010101U,
              std::uint32_t(code) | (std::uint32_t(code) << 7) |
                  (std::uint32_t(code) << 14) | (std::uint32_t(code) << 21)};
          std::memcpy(reinterpret_cast<std::uint8_t*>(a.qs) + 8 * chunk, words,
                      sizeof(words));
        }
        std::fill_n(b.qs, 32, std::uint16_t(index | (code << 9)));
        std::array<float, 256> decoded;
        if (xs)
          gufo::quant::DequantizeIQ2_XS(&b, decoded.data(), 256);
        else
          gufo::quant::DequantizeIQ2_XXS(&a, decoded.data(), 256);
        for (int j = 0; j < 256; ++j) {
          const auto signs = bits::ParitySigns(code);
          const auto word =
              bits::ApplyIq2Signs(std::uint32_t(grid >> (32 * ((j % 8) / 4))),
                                  signs >> (4 * ((j % 8) / 4)));
          Require(decoded[j] == float(SignedByte(word, j % 4)),
                  "IQ2 high index/sign layout differs from CPU dequantizer");
        }
      }
    }
  }
}

void CheckIq3Tables() {
  for (bool small : {false, true}) {
    for (int index = 0; index < (small ? 512 : 256); ++index) {
      const auto grid = small ? gufo::quant::Iq3sGrid()[index]
                              : gufo::quant::Iq3xxsGrid()[index];
      Require(grid == (small ? device_tables::iq3s_grid[index]
                             : device_tables::iq3xxs_grid[index]),
              "IQ3 CPU/device grid differs");
      for (int byte = 0; byte < 4; ++byte)
        Require(((grid >> (8 * byte)) & 255) > 0 &&
                    ((grid >> (8 * byte)) & 255) < 128,
                "IQ3 grid violates carry-free sign precondition");
      for (int code = 0; code < (small ? 256 : 128); ++code) {
        gufo::quant::block_iq3_s a{};
        gufo::quant::block_iq3_xxs b{};
        a.d = 0x3C00;
        b.d = 0x4400;
        std::fill_n(a.qs, 64, std::uint8_t(index));
        std::fill_n(a.qh, 8, std::uint8_t(index >> 8 ? 255 : 0));
        std::fill_n(a.signs, 32, std::uint8_t(code));
        std::fill_n(b.qs, 64, std::uint8_t(index));
        for (int chunk = 0; chunk < 8; ++chunk) {
          const std::uint32_t word =
              std::uint32_t(code) | (std::uint32_t(code) << 7) |
              (std::uint32_t(code) << 14) | (std::uint32_t(code) << 21);
          std::memcpy(b.qs + 64 + 4 * chunk, &word, sizeof(word));
        }
        std::array<float, 256> decoded;
        if (small)
          gufo::quant::DequantizeIQ3_S(&a, decoded.data(), 256);
        else
          gufo::quant::DequantizeIQ3_XXS(&b, decoded.data(), 256);
        const int signs = small ? code : int(bits::ParitySigns(code));
        for (int j = 0; j < 256; ++j) {
          const auto word =
              bits::ApplyIq2Signs(grid, signs >> (4 * ((j / 4) % 2)));
          Require(
              decoded[j] == float(SignedByte(word, j % 4)),
              "IQ3 grid high bits/sign layout differs from scalar dequantizer");
        }
      }
    }
  }
  for (int scale = 0; scale < 16; ++scale) {
    for (int sum : {-253952, -257, -9, -7, -1, 0, 1, 7, 9, 257, 253952})
      Require(
          bits::ScaleIq3Xxs(sum, scale) ==
              int(std::trunc(
                  (double(scale) * sum + std::trunc(double(sum) / 2.0)) / 2.0)),
          "IQ3_XXS signed division boundary differs");
  }
}

void CheckIq1Arithmetic() {
  std::uint32_t state = 0x731903;
  for (int sample = 0; sample < 4096; ++sample) {
    gufo::quant::block_iq1_m w;
    Randomize(w, state);
    SetIq1Half(w, 0x3C00, int(Next(state) & 7), int(Next(state) & 7));
    std::array<Q8, 8> x;
    for (auto& chunk : x)
      Randomize(chunk, state);
    std::array<float, 256> decoded;
    gufo::quant::DequantizeIQ1_M(&w, decoded.data(), 256);
    for (int chunk = 0; chunk < 8; ++chunk) {
      int sum[2] = {};
      for (int l = 0; l < 4; ++l) {
        const int high = w.qh[2 * chunk + l / 2] >> (4 * (l % 2));
        const auto grid = device_tables::iq1s_grid_gpu[w.qs[4 * chunk + l] |
                                                       ((high & 7) << 8)];
        sum[l / 2] += DotWord(bits::ExpandIq1(grid, high & 8),
                              x[chunk].qs.data() + 8 * l);
        sum[l / 2] += DotWord(bits::ExpandIq1(grid >> 4, high & 8),
                              x[chunk].qs.data() + 8 * l + 4);
      }
      const float actual =
          float(sum[0]) * 0.125F * bits::Iq1Scale(w.scales, chunk, 0) +
          float(sum[1]) * 0.125F * bits::Iq1Scale(w.scales, chunk, 1);
      float expected = 0;
      for (int j = 0; j < 32; ++j)
        expected += decoded[32 * chunk + j] * int(x[chunk].qs[j]);
      Require(Word(actual) == Word(expected),
              "IQ1 packed dot differs from independent dequantized dot");
    }
  }
}

}  // namespace

int main() try {
  CheckNewTypeHeaders();
  CheckIq1Layout();
  CheckIq2Tables();
  CheckIq3Tables();
  CheckIq1Arithmetic();
  ExhaustCodesAndSigns();
  CheckFragments<block_iq2_s, 8>();
  CheckFragments<block_q2_0, 2>();
  CheckFragments<gufo::quant::block_iq2_xxs, 8>();
  CheckFragments<gufo::quant::block_iq2_xs, 8>();
  CheckFragments<gufo::quant::block_iq3_xxs, 8>();
  CheckFragments<gufo::quant::block_iq3_s, 8>();
  std::printf(
      "PASS IQ decode helpers: checks=%llu; 1024 grids x 256 signs, "
      "256 Q2 bytes, all subscales and half scales; three FP32 models\n",
      static_cast<unsigned long long>(checks));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "FAIL IQ decode: %s\n", error.what());
  return 1;
}
