// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_GATHER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NATIVE_GATHER_HPP_

#include <cstdint>

#if defined(__HIPCC__)
#define QFN_GATHER_HD __host__ __device__ __forceinline__
#else
#define QFN_GATHER_HD inline
#endif

namespace gufo::models::qwen38_flash_next::native {

// Adapted from dequantize.cuh. Each lane writes its original packed block
// elements directly to the destination; no weight expansion is retained.
template<class Block>
QFN_GATHER_HD void Iq3S(const Block& x, float d_all, float* yy, int tid,
                       const std::uint32_t* grid) {
  const int il = tid / 8;
  const int ib = tid % 8;
  auto* y = yy + 32 * ib + 8 * il;
  const auto* qs = x.qs + 8 * ib;
  const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
      grid + (qs[2 * il] | ((x.qh[ib] << (8 - 2 * il)) & 256)));
  const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
      grid + (qs[2 * il + 1] | ((x.qh[ib] << (7 - 2 * il)) & 256)));
  const float d = d_all * (1 + 2 * ((x.scales[ib / 2] >> (4 * (ib % 2))) & 15));
  const std::uint8_t signs = x.signs[4 * ib + il];
  for (int j = 0; j < 4; ++j) {
    y[j] = d * grid1[j] * (signs & (1 << j) ? -1.0F : 1.0F);
    y[j + 4] = d * grid2[j] * (signs & (1 << (j + 4)) ? -1.0F : 1.0F);
  }
}

template<class Block>
QFN_GATHER_HD void Iq4Xs(const Block& x, float d_all, float* yy, int tid,
                        const std::int8_t* values) {
  const int il = tid / 8;
  const int ib = tid % 8;
  auto* y = yy + 32 * ib + 4 * il;
  const auto* q4 = x.qs + 16 * ib + 4 * il;
  const float d = d_all *
                  ((((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 15) |
                    (((x.scales_h >> (2 * ib)) & 3) << 4)) - 32);
  for (int j = 0; j < 4; ++j) {
    y[j] = d * values[q4[j] & 15];
    y[j + 16] = d * values[q4[j] >> 4];
  }
}

template<class Block>
QFN_GATHER_HD void Q6K(const Block& x, float d, float* yy, int tid) {
  const int ip = tid / 32;
  const int il = tid - 32 * ip;
  const int is = 8 * ip + il / 16;
  auto* y = yy + 128 * ip + il;
  const auto* ql = x.ql + 64 * ip + il;
  const std::uint8_t qh = x.qh[32 * ip + il];
  const auto* sc = x.scales + is;
  y[0] = d * sc[0] * (static_cast<int>((ql[0] & 15) | (((qh >> 0) & 3) << 4)) - 32);
  y[32] = d * sc[2] * (static_cast<int>((ql[32] & 15) | (((qh >> 2) & 3) << 4)) - 32);
  y[64] = d * sc[4] * (static_cast<int>((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
  y[96] = d * sc[6] * (static_cast<int>((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

}  // namespace gufo::models::qwen38_flash_next::native

#undef QFN_GATHER_HD
#endif
