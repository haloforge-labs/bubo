#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_PAIR_DECODE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_PAIR_DECODE_HPP_

#include <cstdint>

#if defined(__HIPCC__)
#define GUFO_IQ_PAIR_HD __host__ __device__
#else
#define GUFO_IQ_PAIR_HD
#endif

// Field layout of the IQ grid types as the register-fed gate/up pair kernel
// (RoutedIqPairKernel, GUFO_ROUTED_WMMA_IQ=2) reads it; written from the
// block definitions of ggml-common.h. The CPU test iq_pair_decode_test checks
// every function here against the CPU dequantizer.
//
// A lane of that kernel owns one weight row and one wave half `half`; per
// 32-value sub-block i it decodes the half's K16 fragment, the groups of
// eight il = 2 half and 2 half + 1. Per 256-value super-block the lane loads
// the bytes of its groups into registers (LoadBlock). IQ blocks are 66 to
// 110 bytes, so fields are only 2-byte aligned: the loads are unaligned 4-
// and 16-byte loads (the Q8_0 route already reads 2-byte aligned 16-byte
// chunks) at offsets relative to the half, which puts each field at a fixed
// bit position; nibble fields are shifted to the half once per super-block
// (NormalizeBlock). Every load stays inside the block.
//
// Arithmetic: RoutedIqWmmaKernel rounds the F32 product scale * q once to
// F16, and that product is exact (scale has at most 16 significant bits, q at
// most 6). The pair kernel feeds 1024 + q as F16 (the CodesToHalves byte
// permute) to one mixed F16/F32 FMA, (1024 + q) * scale - 1024 * scale, whose
// F32 result is the same exact product, then rounds to F16: the same value bit
// for bit. Signs XOR the rounded F16 bits (rounding to nearest is
// sign-symmetric), as kVar bit 3 of RoutedIqWmmaKernel already does.

namespace gufo::models::qwen38_flash_next::rocm::iq_pair {

/// The covered types; the values are the GGML type ids (WeightType).
enum class Type : std::uint32_t {
  kIQ2_XXS = 16,
  kIQ2_XS = 17,
  kIQ3_XXS = 18,
  kIQ3_S = 21,
  kIQ2_S = 22,
};

GUFO_IQ_PAIR_HD constexpr int BlockBytes(Type t) {
  return t == Type::kIQ2_XXS  ? 66
         : t == Type::kIQ2_XS ? 74
         : t == Type::kIQ2_S  ? 82
         : t == Type::kIQ3_XXS ? 98
                               : 110;
}

/// IQ2 grids hold eight magnitudes per 64-bit entry, IQ3 grids four per
/// 32-bit entry (two entries per group of eight).
GUFO_IQ_PAIR_HD constexpr bool Grid64(Type t) {
  return t == Type::kIQ2_XXS || t == Type::kIQ2_XS || t == Type::kIQ2_S;
}

GUFO_IQ_PAIR_HD constexpr int GridEntries(Type t) {
  return t == Type::kIQ2_XXS  ? 256
         : t == Type::kIQ2_XS ? 512
         : t == Type::kIQ2_S  ? 1024
         : t == Type::kIQ3_XXS ? 256
                               : 512;
}

/// IQ2_S and IQ3_S store a sign byte per group; the others a 7-bit index
/// into ksigns_iq2xs. The sign table has one entry per byte or per index.
GUFO_IQ_PAIR_HD constexpr bool RawSigns(Type t) {
  return t == Type::kIQ2_S || t == Type::kIQ3_S;
}

GUFO_IQ_PAIR_HD constexpr int SignEntries(Type t) {
  return RawSigns(t) ? 256 : 128;
}

/// XOR mask of the half2 holding values 2 pair and 2 pair + 1 of a group of
/// eight with sign bits `bits` < 256 (bit j set: value j is negative): bit
/// 2 pair lands on bit 15, bit 2 pair + 1 on bit 31 (two shifts, one OR and
/// one AND; the higher bits of both shifts fall off the mask).
GUFO_IQ_PAIR_HD constexpr std::uint32_t SignMask(std::uint32_t bits,
                                                 int pair) {
  return ((bits << (15 - 2 * pair)) | (bits << (30 - 2 * pair))) &
         0x80008000U;
}

/// The F32 scale of sub-block i exactly as RoutedIqWmmaKernel computes it
/// (every step is exact), from the super-block's d and the 4-bit code ls.
GUFO_IQ_PAIR_HD constexpr float Scale(Type t, float d, std::uint32_t ls) {
  if (t == Type::kIQ3_S)
    return d * (1.0F + 2.0F * static_cast<float>(ls));
  if (t == Type::kIQ3_XXS)
    return d * (0.5F + static_cast<float>(ls)) * 0.5F;
  return d * (0.5F + static_cast<float>(ls)) * 0.25F;
}

/// One lane's registers for one super-block: q, s and e as LoadBlock fills
/// them per type, d the raw F16 bits of the super-block scale.
struct Block {
  std::uint32_t q[8];  ///< grid index bytes or halves of sub-block i
  std::uint32_t s[8];  ///< sign bytes (raw-sign types) or aux words of sub-block i
  std::uint32_t e[4];  ///< high index bits and 4-bit scales
  std::uint32_t d;
};

// __builtin_memcpy: one spelling for the device compile and the host test.
GUFO_IQ_PAIR_HD inline std::uint32_t LoadU32(const std::uint8_t* p) {
  std::uint32_t v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}

//   IQ2_XXS  q[i] = the four grid bytes at 2 + 8i (the half's in bits
//            16 half..16 half + 15), s[i] = the aux word at 6 + 8i, both
//            from one 64-byte read;
//   IQ2_XS   q[i] = 4 bytes at 2 + 8i + 4 half (the two 16-bit codes of the
//            half's groups), e[0..1] = scales[0..7];
//   IQ2_S    q[i] = 4 bytes at 2 + 4i + 2 half (low index bytes in bits
//            0-15), s[i] = 4 bytes at 34 + 4i + 2 half (sign bytes in bits
//            0-15), e[0..1] = qh[0..7], e[2..3] = scales[0..7];
//   IQ3_XXS  q[i] = 4 bytes at 2 + 8i + 4 half (two index bytes per group),
//            s[i] = the aux word at 66 + 4i;
//   IQ3_S    q[i] as IQ3_XXS, s[i] = 4 bytes at 74 + 4i + 2 half (sign
//            bytes in bits 0-15), e[0..1] = qh[0..7], e[2] = scales[0..3].
template<Type T>
GUFO_IQ_PAIR_HD inline Block LoadBlock(const std::uint8_t* b, int half) {
  Block r{};
  std::uint16_t d;
  __builtin_memcpy(&d, b, sizeof(d));
  r.d = d;
  const int h2 = 2 * half, h4 = 4 * half;
  if constexpr (T == Type::kIQ2_S) {
    __builtin_memcpy(r.q, b + 2 + h2, 32);
    __builtin_memcpy(r.s, b + 34 + h2, 32);
    __builtin_memcpy(r.e, b + 66, 16);
  } else if constexpr (T == Type::kIQ3_S) {
    for (int i = 0; i < 8; ++i)
      r.q[i] = LoadU32(b + 2 + 8 * i + h4);
    __builtin_memcpy(r.s, b + 74 + h2, 32);
    __builtin_memcpy(r.e, b + 66, 8);
    r.e[2] = LoadU32(b + 106);
  } else if constexpr (T == Type::kIQ3_XXS) {
    for (int i = 0; i < 8; ++i)
      r.q[i] = LoadU32(b + 2 + 8 * i + h4);
    __builtin_memcpy(r.s, b + 66, 32);
  } else if constexpr (T == Type::kIQ2_XS) {
    for (int i = 0; i < 8; ++i)
      r.q[i] = LoadU32(b + 2 + 8 * i + h4);
    __builtin_memcpy(r.e, b + 66, 8);
  } else {
    std::uint32_t words[16];
    __builtin_memcpy(words, b + 2, 64);
    for (int i = 0; i < 8; ++i) {
      r.q[i] = words[2 * i];
      r.s[i] = words[2 * i + 1];
    }
  }
  return r;
}

/// Shift the per-half nibbles (qh pairs, scales of il / 2) to bits 0-3 of
/// each byte. Called once per super-block, after the loads have landed.
template<Type T>
GUFO_IQ_PAIR_HD inline void NormalizeBlock(Block& r, int half) {
  constexpr int kWords = T == Type::kIQ2_S                          ? 4
                         : T == Type::kIQ2_XS || T == Type::kIQ3_S ? 2
                                                                    : 0;
  const unsigned shift = 4U * static_cast<unsigned>(half);
  for (int w = 0; w < kWords; ++w)
    r.e[w] = (r.e[w] >> shift) & 0x0F0F0F0FU;
}

/// The lane's two groups of eight of sub-block i (il = 2 half + g).
struct Groups {
  std::uint32_t grid[2][2];  ///< grid entries [g][entry]; IQ2 types use entry 0
  std::uint32_t sign[2];     ///< sign-table index of group g
  std::uint32_t ls;          ///< 4-bit scale code of the sub-block half
};

/// `q` and `s` are Block::q[i] and Block::s[i], `e` the normalized
/// Block::e; `i` is the sub-block (0-7) of the super-block.
template<Type T>
GUFO_IQ_PAIR_HD inline Groups SubBlockFields(std::uint32_t q, std::uint32_t s,
                                             const std::uint32_t* e, int i,
                                             int half) {
  Groups f{};
  const unsigned byte = 8U * static_cast<unsigned>(i % 4);
  for (int g = 0; g < 2; ++g) {
    if constexpr (T == Type::kIQ2_S) {
      const std::uint32_t nib = (e[i / 4] >> byte) & 0xFU;
      f.grid[g][0] = ((q >> (8 * g)) & 0xFFU) | (((nib >> (2 * g)) & 3U) << 8);
      f.sign[g] = (s >> (8 * g)) & 0xFFU;
    } else if constexpr (T == Type::kIQ2_XS) {
      const std::uint32_t code = (q >> (16 * g)) & 0xFFFFU;
      f.grid[g][0] = code & 511U;
      f.sign[g] = code >> 9;
    } else if constexpr (T == Type::kIQ2_XXS) {
      f.grid[g][0] = (q >> (16 * half + 8 * g)) & 0xFFU;
      f.sign[g] = (s >> (14 * half + 7 * g)) & 127U;
    } else if constexpr (T == Type::kIQ3_XXS) {
      f.grid[g][0] = (q >> (16 * g)) & 0xFFU;
      f.grid[g][1] = (q >> (16 * g + 8)) & 0xFFU;
      f.sign[g] = (s >> (14 * half + 7 * g)) & 127U;
    } else {
      const std::uint32_t nib = (e[i / 4] >> byte) & 0xFU;
      f.grid[g][0] = ((q >> (16 * g)) & 0xFFU) | (((nib >> (2 * g)) & 1U) << 8);
      f.grid[g][1] =
          ((q >> (16 * g + 8)) & 0xFFU) | (((nib >> (2 * g + 1)) & 1U) << 8);
      f.sign[g] = (s >> (8 * g)) & 0xFFU;
    }
  }
  if constexpr (T == Type::kIQ2_S)
    f.ls = (e[2 + i / 4] >> byte) & 0xFU;
  else if constexpr (T == Type::kIQ2_XS)
    f.ls = (e[i / 4] >> byte) & 0xFU;
  else if constexpr (T == Type::kIQ3_S)
    f.ls = (e[2] >> (4 * i)) & 0xFU;
  else
    f.ls = s >> 28;
  return f;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::iq_pair

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_PAIR_DECODE_HPP_
