// CPU checks of the register-fed IQ pair kernel's decode (pf-iq-pair,
// iq_pair_decode.hpp): the lane field layout against the CPU dequantizer,
// the exactness of the mixed FMA for every scale, code and sign, the half2
// sign masks, and that every lane load stays inside its block.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/iq_grids.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/iq_pair_decode.hpp"

namespace {

namespace ip = gufo::models::qwen38_flash_next::rocm::iq_pair;
using gufo::quant::Fp16ToFloat;

std::uint64_t checks = 0;

void Require(bool condition, const char* what) {
  ++checks;
  if (!condition)
    throw std::runtime_error(what);
}

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error(what);
}

std::uint32_t Next(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

std::uint32_t Bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

/// F32 to F16, rounding to nearest even (the GPU's v_cvt_f16_f32 and the
/// F16 output of v_fma_mix with the default rounding mode).
std::uint16_t HalfRn(float f) {
  const std::uint32_t x = Bits(f);
  const std::uint32_t sign = (x >> 16) & 0x8000U;
  const std::uint32_t ax = x & 0x7FFFFFFFU;
  if (ax >= 0x7F800000U)
    return static_cast<std::uint16_t>(sign | (ax > 0x7F800000U ? 0x7E00U : 0x7C00U));
  if (ax >= 0x477FF000U)  // 65520 and above round to infinity
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (ax < 0x38800000U) {  // below 2^-14: subnormal or zero
    const int e = static_cast<int>(ax >> 23);
    if (e < 102)
      return static_cast<std::uint16_t>(sign);
    const std::uint32_t m = (ax & 0x7FFFFFU) | 0x800000U;
    const int shift = 126 - e;
    std::uint32_t q = m >> shift;
    const std::uint32_t rem = m & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (rem > halfway || (rem == halfway && (q & 1U)))
      ++q;
    return static_cast<std::uint16_t>(sign | q);
  }
  std::uint32_t h = (((ax >> 23) - 112U) << 10) | ((ax & 0x7FFFFFU) >> 13);
  const std::uint32_t rem = ax & 0x1FFFU;
  if (rem > 0x1000U || (rem == 0x1000U && (h & 1U)))
    ++h;
  return static_cast<std::uint16_t>(sign | h);
}

void CheckHalfRn() {
  // Every F16 value converts back to itself, and midpoints round to even.
  for (std::uint32_t h = 0; h < 0x10000U; ++h) {
    if ((h & 0x7C00U) == 0x7C00U && (h & 0x3FFU) != 0)
      continue;  // NaN
    const float f = Fp16ToFloat(static_cast<std::uint16_t>(h));
    Require(HalfRn(f) == h, "HalfRn round trip");
  }
  Require(HalfRn(1.0F + std::ldexp(1.0F, -11)) == 0x3C00U, "tie to even down");
  Require(HalfRn(1.0F + 3.0F * std::ldexp(1.0F, -11)) == 0x3C02U, "tie to even up");
  Require(HalfRn(std::ldexp(1.0F, -25)) == 0, "half the smallest subnormal");
  Require(HalfRn(65520.0F) == 0x7C00U, "overflow");
}

/// The pair kernel's arithmetic is exact for every F16 d, every 4-bit scale
/// code, every grid byte (the grids hold 1..62) and both signs. An FMA rounds
/// its exact result once, and the exact result of (1024 + q) * s - 1024 * s
/// is q * s; so the F32 FMA is exact iff q * s is an F32 value, which this
/// checks in double (where every term here is exact). That F32 value is the
/// product RoutedIqWmmaKernel rounds, so the F16 weights are bit-identical.
/// (std::fmaf itself runs in the random-block check below.)
void CheckArithmetic() {
  const ip::Type types[3] = {ip::Type::kIQ2_S, ip::Type::kIQ3_XXS,
                             ip::Type::kIQ3_S};
  std::uint64_t compared = 0;
  for (const ip::Type t : types) {
    for (std::uint32_t dh = 0; dh < 0x10000U; ++dh) {
      if ((dh & 0x7C00U) == 0x7C00U)
        continue;  // d infinite or NaN
      const float d = Fp16ToFloat(static_cast<std::uint16_t>(dh));
      for (std::uint32_t ls = 0; ls < 16; ++ls) {
        const float scale = ip::Scale(t, d, ls);
        const float bias = -1024.0F * scale;
        const double scale_d = static_cast<double>(scale);
        Require(static_cast<double>(bias) == -1024.0 * scale_d, "1024 * scale exact");
        for (std::uint32_t q = 0; q < 128; ++q) {
          const double exact = (1024.0 + q) * scale_d + static_cast<double>(bias);
          const float current = scale * static_cast<float>(q);
          if (exact != scale_d * q || static_cast<double>(current) != exact) {
            Fail("mixed FMA is not exact (type " + std::to_string(static_cast<int>(t)) +
                 " d " + std::to_string(dh) + " ls " + std::to_string(ls) + " q " +
                 std::to_string(q) + ")");
          }
          // The current kernel rounds the signed product, the pair kernel
          // rounds the magnitude and flips the F16 sign bit.
          if (HalfRn(-current) != (HalfRn(current) ^ 0x8000U))
            Fail("rounding is not sign-symmetric at d " + std::to_string(dh));
          compared += 2;
        }
      }
    }
  }
  checks += compared;
  // IQ2_XXS and IQ2_XS share IQ2_S's scale formula (ip::Scale), so the three
  // formulas cover all five types.
  Require(ip::Scale(ip::Type::kIQ2_XXS, 1.0F, 7) ==
                  ip::Scale(ip::Type::kIQ2_S, 1.0F, 7) &&
              ip::Scale(ip::Type::kIQ2_XS, 1.0F, 7) ==
                  ip::Scale(ip::Type::kIQ2_S, 1.0F, 7),
          "IQ2 scale formula");
  std::printf("arithmetic: %llu weights (all finite F16 d x 16 scale codes x 128 grid bytes x 2 signs x 3 scale formulas) bit-identical\n",
              static_cast<unsigned long long>(compared));
}

void CheckSignMasks() {
  for (std::uint32_t bits = 0; bits < 256; ++bits) {
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t mask = ip::SignMask(bits, j / 2);
      const std::uint32_t bit = (mask >> ((j & 1) != 0 ? 31 : 15)) & 1U;
      Require(bit == ((bits >> j) & 1U), "sign mask bit");
    }
    for (int p = 0; p < 4; ++p)
      Require((ip::SignMask(bits, p) & ~0x80008000U) == 0, "sign mask touches only sign bits");
  }
}

/// The magnitudes of codebook entry `entry` (eight bytes for IQ2, four for IQ3).
const std::uint8_t* GridBytes(ip::Type t, std::uint32_t entry) {
  switch (t) {
    case ip::Type::kIQ2_XXS:
      return reinterpret_cast<const std::uint8_t*>(gufo::quant::Iq2xxsGrid() + entry);
    case ip::Type::kIQ2_XS:
      return reinterpret_cast<const std::uint8_t*>(gufo::quant::Iq2xsGrid() + entry);
    case ip::Type::kIQ2_S:
      return reinterpret_cast<const std::uint8_t*>(gufo::quant::Iq2sGrid() + entry);
    case ip::Type::kIQ3_XXS:
      return reinterpret_cast<const std::uint8_t*>(gufo::quant::Iq3xxsGrid() + entry);
    default:
      return reinterpret_cast<const std::uint8_t*>(gufo::quant::Iq3sGrid() + entry);
  }
}

void Dequantize(ip::Type t, const std::uint8_t* block, float* out) {
  switch (t) {
    case ip::Type::kIQ2_XXS: gufo::quant::DequantizeIQ2_XXS(block, out, 256); break;
    case ip::Type::kIQ2_XS: gufo::quant::DequantizeIQ2_XS(block, out, 256); break;
    case ip::Type::kIQ2_S: gufo::quant::DequantizeIQ2_S(block, out, 256); break;
    case ip::Type::kIQ3_XXS: gufo::quant::DequantizeIQ3_XXS(block, out, 256); break;
    default: gufo::quant::DequantizeIQ3_S(block, out, 256); break;
  }
}

/// The grid magnitudes are 7-bit (the exhaustive arithmetic check covers
/// q < 128) and the codebook sizes match GridEntries.
void CheckGrids() {
  const ip::Type types[5] = {ip::Type::kIQ2_XXS, ip::Type::kIQ2_XS, ip::Type::kIQ2_S,
                             ip::Type::kIQ3_XXS, ip::Type::kIQ3_S};
  for (const ip::Type t : types) {
    const int bytes = ip::Grid64(t) ? 8 : 4;
    for (int e = 0; e < ip::GridEntries(t); ++e) {
      const std::uint8_t* g = GridBytes(t, static_cast<std::uint32_t>(e));
      for (int b = 0; b < bytes; ++b)
        Require(g[b] < 128 && g[b] > 0, "grid magnitude in 1..127");
    }
  }
  // The register variant of the pair kernel rebuilds ksigns_iq2xs[code] as
  // the code plus an even-parity eighth bit.
  for (std::uint32_t c = 0; c < 128; ++c) {
    std::uint32_t parity = 0;
    for (int b = 0; b < 7; ++b)
      parity ^= (c >> b) & 1U;
    Require(gufo::quant::SignsIq2xs()[c] == (c | (parity << 7)),
            "ksigns_iq2xs is the code with an even-parity eighth bit");
  }
}

/// Decode of one super-block the way RoutedIqPairKernel does it (LoadBlock,
/// NormalizeBlock, SubBlockFields, the codebook, the sign table, the mixed
/// FMA, the F16 sign flip), for both halves; out[k] is the F16 of value k.
template<ip::Type T>
void PairDecode(const std::uint8_t* block, std::uint16_t* out) {
  for (int half = 0; half < 2; ++half) {
    ip::Block blk = ip::LoadBlock<T>(block, half);
    ip::NormalizeBlock<T>(blk, half);
    const float d = Fp16ToFloat(static_cast<std::uint16_t>(blk.d));
    for (int i = 0; i < 8; ++i) {
      const ip::Groups f = ip::SubBlockFields<T>(blk.q[i], blk.s[i], blk.e, i, half);
      const float scale = ip::Scale(T, d, f.ls);
      const float bias = -1024.0F * scale;
      for (int g = 0; g < 2; ++g) {
        const std::uint32_t bits =
            ip::RawSigns(T) ? f.sign[g] : gufo::quant::SignsIq2xs()[f.sign[g]];
        Require(f.sign[g] < static_cast<std::uint32_t>(ip::SignEntries(T)), "sign index range");
        Require(f.grid[g][0] < static_cast<std::uint32_t>(ip::GridEntries(T)) &&
                    f.grid[g][1] < static_cast<std::uint32_t>(ip::GridEntries(T)),
                "grid index range");
        std::uint8_t q[8];
        if constexpr (ip::Grid64(T)) {
          std::memcpy(q, GridBytes(T, f.grid[g][0]), 8);
        } else {
          std::memcpy(q, GridBytes(T, f.grid[g][0]), 4);
          std::memcpy(q + 4, GridBytes(T, f.grid[g][1]), 4);
        }
        for (int v = 0; v < 8; ++v) {
          const float fused =
              std::fmaf(1024.0F + static_cast<float>(q[v]), scale, bias);
          const std::uint32_t mask = ip::SignMask(bits, v / 2);
          const std::uint16_t sign =
              static_cast<std::uint16_t>((v & 1) != 0 ? mask >> 16 : mask & 0xFFFFU);
          out[32 * i + 8 * (2 * half + g) + v] =
              static_cast<std::uint16_t>(HalfRn(fused) ^ sign);
        }
      }
    }
  }
}

template<ip::Type T>
void CheckType(const char* name) {
  constexpr int kBytes = ip::BlockBytes(T);
  std::uint32_t state = 0x2545F491U ^ static_cast<std::uint32_t>(T);
  // Bounds: a block of byte values 0..kBytes-1 followed by 0xEE guard bytes;
  // no loaded byte may come from the guard.
  {
    std::vector<std::uint8_t> buffer(kBytes + 32, 0xEE);
    for (int b = 0; b < kBytes; ++b)
      buffer[b] = static_cast<std::uint8_t>(b);
    for (int half = 0; half < 2; ++half) {
      const ip::Block blk = ip::LoadBlock<T>(buffer.data(), half);
      const std::uint32_t* words[3] = {blk.q, blk.s, blk.e};
      const int counts[3] = {8, 8, 4};
      for (int a = 0; a < 3; ++a)
        for (int wi = 0; wi < counts[a]; ++wi)
          for (int byte = 0; byte < 4; ++byte)
            if (((words[a][wi] >> (8 * byte)) & 0xFFU) == 0xEE)
              Fail(std::string(name) + ": a lane load leaves the block");
    }
  }
  std::vector<std::uint8_t> block(kBytes);
  float reference[256];
  std::uint16_t pair[256];
  std::uint64_t weights = 0;
  for (int n = 0; n < 20000; ++n) {
    for (auto& b : block)
      b = static_cast<std::uint8_t>(Next(state) >> 11);
    // d: every 16th block a random finite F16 (subnormals included), else a
    // weight-like scale.
    std::uint16_t d = 0;
    if (n % 16 == 0) {
      do {
        d = static_cast<std::uint16_t>(Next(state));
      } while ((d & 0x7C00U) == 0x7C00U);
    } else {
      d = static_cast<std::uint16_t>(0x1000U + Next(state) % 0x1800U);
    }
    std::memcpy(block.data(), &d, 2);
    Dequantize(T, block.data(), reference);
    PairDecode<T>(block.data(), pair);
    for (int v = 0; v < 256; ++v) {
      if (pair[v] != HalfRn(reference[v])) {
        Fail(std::string(name) + ": pair decode differs from the dequantizer at block " +
             std::to_string(n) + " value " + std::to_string(v));
      }
      ++weights;
      ++checks;
    }
  }
  std::printf("%s: %llu weights of 20000 random blocks bit-identical to RN16(DequantizeIQ*), both halves; lane loads inside the %d-byte block\n",
              name, static_cast<unsigned long long>(weights), kBytes);
}

}  // namespace

int main() {
  try {
    CheckHalfRn();
    CheckSignMasks();
    CheckGrids();
    CheckType<ip::Type::kIQ2_XXS>("IQ2_XXS");
    CheckType<ip::Type::kIQ2_XS>("IQ2_XS");
    CheckType<ip::Type::kIQ2_S>("IQ2_S");
    CheckType<ip::Type::kIQ3_XXS>("IQ3_XXS");
    CheckType<ip::Type::kIQ3_S>("IQ3_S");
    CheckArithmetic();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  std::printf("iq_pair_decode_test: OK (%llu checks)\n",
              static_cast<unsigned long long>(checks));
  return 0;
}
