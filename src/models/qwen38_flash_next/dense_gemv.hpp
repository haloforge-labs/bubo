// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/iq_decode_bits.hpp"

#if defined(__HIPCC__)
#define GUFO_DENSE_HD __host__ __device__ __forceinline__
#define GUFO_DENSE_FRAGMENT __device__ __forceinline__
#else
#define GUFO_DENSE_HD inline
#define GUFO_DENSE_FRAGMENT inline
#endif

namespace gufo::models::qwen38_flash_next::dense_gemv {

enum class Variant {
  kBaseline,
  kReuse1,
  kReuse2,
  kReuse1Cap6,
  kReuse1Lds,
  kReuse1LdsGrid,
  kReuse1SplitKLds,
  kKQuantWmma,
  kDot4
};
inline constexpr int kVariantCount = 9;

GUFO_DENSE_HD constexpr const char* VariantName(Variant variant) {
  switch (variant) {
    case Variant::kBaseline:
      return "baseline";
    case Variant::kReuse1:
      return "reuse1";
    case Variant::kReuse2:
      return "reuse2";
    case Variant::kReuse1Cap6:
      return "reuse1_cap6";
    case Variant::kReuse1Lds:
      return "reuse1_lds";
    case Variant::kReuse1LdsGrid:
      return "reuse1_ldsgrid";
    case Variant::kReuse1SplitKLds:
      return "reuse1_splitk_lds";
    case Variant::kKQuantWmma:
      return "kquant_wmma";
    case Variant::kDot4:
      return "dot4";
  }
  return nullptr;
}

// -1 means no override, -2 is invalid. The adapter owns getenv and failure.
GUFO_DENSE_HD constexpr int ParseVariant(const char* text) {
  if (!text || !*text)
    return -1;
  for (int v = 0; v < kVariantCount; ++v) {
    const char* name = VariantName(static_cast<Variant>(v));
    int i = 0;
    while (text[i] && text[i] == name[i])
      ++i;
    if (text[i] == name[i])
      return v;
  }
  return -2;
}

GUFO_DENSE_HD constexpr bool IsIqTarget(int type) {
  return type == 21 || type == 18 || type == 17;
}

GUFO_DENSE_HD constexpr bool IsKQuantTarget(int type) {
  return type == 12 || type == 13 || type == 14;
}

// Only the four large trunk shapes. Vocabulary heads, shared experts and
// skinny projections retain their current route, including with the override.
GUFO_DENSE_HD constexpr bool KQuantMatrixShape(int k, int rows) {
  return (k == 2560 && (rows == 6144 || rows == 10240 || rows == 12288)) ||
         (k == 6144 && rows == 2560);
}

// Four-byte words in a 16-row, K256 signed-byte LDS tile. Every aligned K16
// fragment stays contiguous; row swizzling avoids a single-bank vector gather.
GUFO_DENSE_HD constexpr int KQuantStageWordIndex(int row, int element) {
  return (row * 16 + ((element / 16) ^ ((row >> 1) & 15))) * 4 +
         (element % 16) / 4;
}

// Both passes retain the original complete K traversal and storing lane.
GUFO_DENSE_HD constexpr int Cap6FirstWidth(int width) {
  return width > 6 ? 4 : width;
}

GUFO_DENSE_HD constexpr int FormatIndex(int type) {
  switch (type) {
    case 23:
      return 0;
    case 20:
      return 1;
    case 12:
      return 2;
    case 13:
      return 3;
    case 7:
      return 4;
    case 14:
      return 5;
    case 21:
      return 6;
    case 18:
      return 7;
    case 17:
      return 8;
    default:
      return -1;
  }
}

GUFO_DENSE_HD constexpr int BlockSize(int type) {
  return type == 20 || type == 7 ? 32 : 256;
}

GUFO_DENSE_HD constexpr int BlockBytes(int type) {
  switch (type) {
    case 23:
      return 136;
    case 20:
      return 18;
    case 12:
      return 144;
    case 13:
      return 176;
    case 7:
      return 24;
    case 14:
      return 210;
    case 21:
      return 110;
    case 18:
      return 98;
    case 17:
      return 74;
    default:
      return 0;
  }
}

GUFO_DENSE_HD constexpr bool Covers(int type, int width, int k) {
  return FormatIndex(type) >= 0 && width >= 1 && width <= 8 && k > 0 &&
         k % BlockSize(type) == 0;
}

// The host adapter applies the optional width-one override. Wider IQ4 keeps
// its existing reuse kernel; unsupported widths retain their original route.
// Q6_K and Q4_K take the reuse kernel from width two: on the FU3 shapes it was
// never slower than the existing one there (time ratio 0.94-0.99 and 0.86-0.98
// at N=2, 0.71-1.02 and 0.81-1.00 at N=3) and the existing kernel falls to 160
// and 176 GB/s at N=3 where reuse holds 211 and 206. Q5_1 keeps width four.
GUFO_DENSE_HD constexpr Variant ProductionVariant(int type, int width, int k) {
  if (!Covers(type, width, k))
    return Variant::kBaseline;
  if (width == 1 && (type == 21 || type == 18 || type == 17))
    return Variant::kBaseline;
  if (width == 1 || type == 13 || type == 21 || type == 18 || type == 17 ||
      ((type == 12 || type == 14) && width >= 2) || (type == 7 && width >= 4))
    return Variant::kReuse1;
  return Variant::kBaseline;
}

// Formats the few-row dot4 kernel instantiates (dense_dot4.hpp). An explicit
// variant request (bench) may run any of them.
GUFO_DENSE_HD constexpr bool Dot4Kernel(int type) {
  return type == 12 || type == 13 || type == 14 || type == 21 || type == 23;
}

// Formats the engine routes to dot4 under the override: the measured cells
// where it beats the production selection (the phase 2 report lists them).
// IQ3_S and IQ4_XS stay on their kernels (slower at N4 / N8).
GUFO_DENSE_HD constexpr bool Dot4Selected(int type) {
  return type == 12 || type == 13 || type == 14;
}

// Why the dot4 override keeps a call on its existing route. No shape list:
// any M (tails clamp) and any K that is a whole number of 256-blocks.
enum class Dot4Route { kDot4, kFormat, kKTail, kWidth };
inline constexpr int kDot4RouteCount = 4;

GUFO_DENSE_HD constexpr Dot4Route Dot4RouteFor(int type, int width, int k) {
  if (!Dot4Selected(type))
    return Dot4Route::kFormat;
  if (k <= 0 || k % 256 != 0)
    return Dot4Route::kKTail;
  if (width < 1 || width > 8)
    return Dot4Route::kWidth;
  return Dot4Route::kDot4;
}

GUFO_DENSE_HD constexpr const char* Dot4RouteName(Dot4Route route) {
  switch (route) {
    case Dot4Route::kDot4:
      return "dot4";
    case Dot4Route::kFormat:
      return "format";
    case Dot4Route::kKTail:
      return "k_tail";
    case Dot4Route::kWidth:
      return "width";
  }
  return nullptr;
}

GUFO_DENSE_HD constexpr Variant SelectVariant(int type, int width, int k,
                                              int rows, int override_value,
                                              bool width_one = true) {
  const auto selected = width == 1 && !width_one
                            ? Variant::kBaseline
                            : ProductionVariant(type, width, k);
  // The choice depends on (format, K) only, never on the width, so a row
  // computes to the same bits at N = 1..8 (width one included, whatever
  // GUFO_DENSE_W1 says, for the formats dot4 selects).
  if (override_value == int(Variant::kDot4))
    return Dot4RouteFor(type, width, k) == Dot4Route::kDot4 ? Variant::kDot4
                                                            : selected;
  if (override_value == int(Variant::kKQuantWmma))
    return IsKQuantTarget(type) && width >= 4 && width <= 8 &&
                   KQuantMatrixShape(k, rows)
               ? Variant::kKQuantWmma
               : selected;
  return override_value >= 0 && IsIqTarget(type) && Covers(type, width, k)
             ? static_cast<Variant>(override_value)
             : selected;
}

GUFO_DENSE_HD constexpr int Vdr(int type) {
  return type == 23 ? 4 : type == 14 ? 1 : 2;
}
GUFO_DENSE_HD constexpr int Qi(int type) {
  return type == 17 || type == 18 || type == 21 ? 16 : BlockSize(type) / 8;
}

struct Pair {
  std::uint32_t lo, hi;
};

GUFO_DENSE_HD constexpr std::uint32_t SpreadNibble(std::uint32_t value) {
  return ((value & 15U) * 0x00204081U) & 0x01010101U;
}

// A fragment contains decoded integers and scales, never an activation.
// Math supplies the same rounded FP32 operations on the host and device.
// Weight-only work precedes the exact-width activation loop in the caller.
template<int Type, class Math>
struct Fragment {
  static_assert(Type == 23 || Type == 20 || Type == 12 || Type == 13 ||
                Type == 7 || Type == 14 || Type == 21 || Type == 18 ||
                Type == 17);
  Pair values[Type == 23 || Type == 21 || Type == 18 || Type == 17 ? 4 : 2]{};
  float scale{}, minimum{};
  int subscale{}, scales[2]{}, minima[2]{};

  GUFO_DENSE_FRAGMENT Fragment(const std::uint8_t* w, int iqs,
                               const std::uint32_t* grid = nullptr) {
    scale = Math::Half(w + (Type == 14 ? 208 : 0));
    if constexpr (Type == 21 || Type == 18 || Type == 17) {
      const auto aux =
          Type == 18 ? Math::template Word<2>(w + 66 + 2 * iqs) : 0U;
      if constexpr (Type == 17) {
        scales[0] = w[66 + iqs / 2] & 15;
        scales[1] = w[66 + iqs / 2] >> 4;
      } else if constexpr (Type == 18)
        subscale = int(aux >> 28);
      else
        subscale = 1 + 2 * ((w[106 + iqs / 4] >> ((iqs << 1) & 4)) & 15);
      for (int j = 0; j < 4; ++j) {
        std::uint32_t signs;
        if constexpr (Type == 17) {
          const auto code = Math::HalfBits(w + 2 + 4 * iqs + 2 * j);
          values[j] = Math::template Grid<17>(code & 511, grid);
          signs = iq_decode::ParitySigns(code >> 9);
        } else {
          const auto* codes = w + 2 + 4 * iqs + 2 * j;
          const int high = Type == 21 ? w[66 + iqs / 2] >> (2 * j) : 0;
          values[j] = {
              Math::template Grid<Type>(codes[0] | ((high & 1) << 8), grid).lo,
              Math::template Grid<Type>(codes[1] | ((high & 2) << 7), grid).lo};
          signs = Type == 18 ? iq_decode::ParitySigns(aux >> (7 * j))
                             : w[74 + 2 * iqs + j];
        }
        values[j].lo = iq_decode::ApplyIq2Signs(values[j].lo, signs);
        values[j].hi = iq_decode::ApplyIq2Signs(values[j].hi, signs >> 4);
      }
    } else if constexpr (Type == 23 || Type == 20) {
      constexpr int count = Type == 23 ? 4 : 2;
      constexpr int offset = Type == 23 ? 8 : 2;
      for (int j = 0; j < count; ++j)
        values[j] = Math::Lookup(Math::template Word<Type == 20 ? 2 : 4>(
            w + offset + 4 * (iqs + j)));
      if constexpr (Type == 23)
        subscale = (((w[4 + iqs / 8] >> (iqs & 4)) & 15) |
                    (((Math::HalfBits(w + 2) >> (iqs / 2)) & 3) << 4)) -
                   32;
    } else if constexpr (Type == 14) {
      const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
      const auto low = Math::template Word<2>(w + 4 * iqs);
      const auto high =
          Math::template Word<2>(w + 128 + 4 * (8 * (iqs / 16) + iqs % 8)) >>
          (2 * ((iqs % 16) / 8));
      for (int i = 0; i < 2; ++i) {
        const auto code = ((low >> (4 * i)) & 0x0F0F0F0FU) |
                          (((high >> (4 * i)) << 4) & 0x30303030U);
        values[i].lo = Math::Subtract32(code);
        scales[i] = int(std::int8_t(w[192 + scale_offset + 4 * i]));
      }
    } else if constexpr (Type == 7) {
      minimum = Math::Half(w + 2);
      const auto high = Math::template Word<4>(w + 4);
      for (int j = 0; j < 2; ++j) {
        const auto low = Math::template Word<4>(w + 8 + 4 * (iqs + j));
        const auto bits = high >> (4 * (iqs + j));
        values[j] = {
            (low & 0x0F0F0F0FU) | (SpreadNibble(bits) << 4),
            ((low >> 4) & 0x0F0F0F0FU) | (SpreadNibble(bits >> 16) << 4)};
      }
    } else {
      minimum = Math::Half(w + 2);
      const int offset = 2 * ((iqs / 2) / 4);
      const int index = (iqs / 2) % 4;
      constexpr int code_offset = Type == 12 ? 16 : 48;
      const auto low0 =
          Math::template Word<4>(w + code_offset + 16 * offset + 4 * index);
      const auto low1 = Math::template Word<4>(w + code_offset + 16 * offset +
                                               4 * index + 16);
      std::uint32_t high0 = 0, high1 = 0;
      if constexpr (Type == 13) {
        high0 = Math::template Word<4>(w + 16 + 4 * index) >> offset;
        high1 = Math::template Word<4>(w + 16 + 4 * index + 16) >> offset;
      }
      const int j = offset / 2;
      const auto lo = Math::HalfBits(w + 4 + 2 * (j & 1));
      const auto mid = Math::HalfBits(w + 8 + 2 * (j & 1));
      const auto hi = Math::HalfBits(w + 12 + 2 * (j & 1));
      const auto sc =
          j < 2 ? lo & 0x3F3FU : (hi & 0x0F0FU) | ((lo & 0xC0C0U) >> 2);
      const auto mn = j < 2 ? mid & 0x3F3FU
                            : ((hi >> 4) & 0x0F0FU) | ((mid & 0xC0C0U) >> 2);
      for (int i = 0; i < 2; ++i) {
        values[i] = {(low0 >> (4 * i)) & 0x0F0F0F0FU,
                     (low1 >> (4 * i)) & 0x0F0F0F0FU};
        if constexpr (Type == 13) {
          values[i].lo |= ((high0 >> i) << 4) & 0x10101010U;
          values[i].hi |= ((high1 >> i) << 4) & 0x10101010U;
        }
        scales[i] = (sc >> (8 * i)) & 255;
        minima[i] = (mn >> (8 * i)) & 255;
      }
    }
  }

  // Reuse this decoder when staging for matrix units. The sink owns LDS
  // addressing; no packed-code decoding remains in the WMMA inner loop.
  template<class Sink>
  GUFO_DENSE_FRAGMENT void StoreKTile(int iqs, Sink& sink) const {
    static_assert(IsKQuantTarget(Type));
    if constexpr (Type == 14) {
      const int base = (iqs / 16) * 128 + (iqs % 16) * 4;
      for (int i = 0; i < 2; ++i) {
        sink.Word(base + 64 * i, values[i].lo);
        if (iqs % 4 == 0)
          sink.Scale((base + 64 * i) / 16, Math::Mul(scale, float(scales[i])));
      }
    } else {
      const int group = 2 * (iqs / 8), index = (iqs / 2) % 4;
      for (int i = 0; i < 2; ++i) {
        const int base = 32 * (group + i) + 4 * index;
        sink.Word(base, values[i].lo);
        sink.Word(base + 16, values[i].hi);
        if (index == 0) {
          sink.Scale(group + i, Math::Mul(scale, float(scales[i])));
          sink.Minimum(group + i, Math::Mul(minimum, float(minima[i])));
        }
      }
    }
  }

  // x points to the first 36-byte Q8_1 block of this weight block.
  GUFO_DENSE_FRAGMENT float Accumulate(const std::uint8_t* x, int iqs,
                                       float acc) const {
    if constexpr (Type == 21 || Type == 18 || Type == 17) {
      x += (iqs / 2) * 36;
      int low = 0, high = 0;
      for (int j = 0; j < 4; ++j) {
        const auto u0 = Math::template Word<4>(x + 4 + 8 * j);
        const auto u1 = Math::template Word<4>(x + 8 + 8 * j);
        if (Type == 17 && j >= 2) {
          high = Math::Dot(values[j].lo, u0, high);
          high = Math::Dot(values[j].hi, u1, high);
        } else {
          low = Math::Dot(values[j].lo, u0, low);
          low = Math::Dot(values[j].hi, u1, low);
        }
      }
      int dot;
      if constexpr (Type == 17)
        dot = (low * scales[0] + high * scales[1] + (low + high) / 2) / 4;
      else if constexpr (Type == 18)
        dot = (subscale * low + low / 2) / 2;
      else
        dot = low * subscale;
      return Math::Fma(Math::Mul(scale, Math::Half(x)), float(dot), acc);
    } else if constexpr (Type == 14) {
      x += (4 * (iqs / 16) + (iqs % 16) / 8) * 36;
      float sd = 0.0F;
      for (int i = 0; i < 2; ++i) {
        const auto* block = x + 2 * i * 36;
        const auto u = Math::template Word<4>(block + 4 + 4 * (iqs % 8));
        const float dot = float(Math::Dot(values[i].lo, u, 0) * scales[i]);
        sd = i == 0 ? Math::Mul(Math::Half(block), dot)
                    : Math::Fma(Math::Half(block), dot, sd);
      }
      return Math::Fma(scale, sd, acc);
    } else if constexpr (Type == 12 || Type == 13) {
      const int offset = 2 * ((iqs / 2) / 4);
      const int index = (iqs / 2) % 4;
      float sd = 0.0F, sm = 0.0F;
      for (int i = 0; i < 2; ++i) {
        const auto* block = x + (offset + i) * 36;
        const auto u0 = Math::template Word<4>(block + 4 + 4 * index);
        const auto u1 = Math::template Word<4>(block + 4 + 4 * index + 16);
        const int dot =
            Type == 13
                ? Math::Dot(values[i].lo, u0, Math::Dot(values[i].hi, u1, 0))
                : Math::Dot(values[i].hi, u1, Math::Dot(values[i].lo, u0, 0));
        const int sum =
            Type == 13
                ? Math::Dot(0x01010101U, u0, Math::Dot(0x01010101U, u1, 0))
                : Math::Dot(0x01010101U, u1, Math::Dot(0x01010101U, u0, 0));
        const float d8 = Math::Half(block);
        sd = i == 0 ? Math::Mul(d8, float(dot * scales[i]))
                    : Math::Fma(d8, float(dot * scales[i]), sd);
        sm = i == 0 ? Math::Mul(d8, float(sum * minima[i]))
                    : Math::Fma(d8, float(sum * minima[i]), sm);
      }
      // Match the scalar kernel's two contracted outer FMAs.
      return Math::Fma(-minimum, sm, Math::Fma(scale, sd, acc));
    } else {
      int dot = 0, sum = 0;
      if constexpr (Type == 23)
        x += (iqs / 4) * 36;
      constexpr int count = Type == 23 ? 4 : 2;
      for (int j = 0; j < count; ++j) {
        const int index = Type == 23 ? j : iqs + j;
        const auto u0 = Math::template Word<4>(x + 4 + 4 * index);
        const auto u1 = Math::template Word<4>(x + 4 + 4 * index + 16);
        dot = Math::Dot(values[j].lo, u0, dot);
        dot = Math::Dot(values[j].hi, u1, dot);
        if constexpr (Type == 7) {
          sum = Math::Dot(0x01010101U, u0, sum);
          sum = Math::Dot(0x01010101U, u1, sum);
        }
      }
      if constexpr (Type == 23)
        return Math::Fma(Math::Mul(scale, Math::Half(x)), float(dot * subscale),
                         acc);
      else if constexpr (Type == 20)
        return Math::Fma(Math::Mul(scale, float(dot)), Math::Half(x), acc);
      else
        return Math::Fma(
            Math::Half(x),
            Math::Fma(minimum, float(sum), Math::Mul(scale, float(dot))), acc);
    }
  }
};

}  // namespace gufo::models::qwen38_flash_next::dense_gemv

#undef GUFO_DENSE_HD
#undef GUFO_DENSE_FRAGMENT
