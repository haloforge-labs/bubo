// F16 dequantiser for dense_gemm_bench's candidate e (dequantise the whole
// quantised weight into a binary16 scratch, then run a 16-bit GEMM). The decode
// is the Qwen lane's own DecodeQuantSub16 (src/models/qwen/hip/quant_ops.hpp),
// the one the int8 WMMA GEMM stages from, so the weights equal what b sees; the
// value is scale * q - offset (the QuantBlockElement convention), rounded once
// to binary16 (no BF16 rounding of the product). Bench only.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::bench {
namespace {

// One thread per 16-element sub-block: decode once, store 16 binary16 values.
template<core::GgmlType Type>
__global__ void DequantizeToHalfKernel(const std::uint8_t* __restrict__ w,
                                       __half* __restrict__ out,
                                       std::size_t rows, std::size_t cols) {
  const std::size_t subs_per_row = cols / 16;
  const std::size_t idx =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (idx >= rows * subs_per_row) {
    return;
  }
  const std::size_t row = idx / subs_per_row;
  const std::size_t sub = idx % subs_per_row;
  const std::size_t row_bytes = hip::QuantRowBytes(Type, cols);
  hip::QuantSub16 s;
  hip::DecodeQuantSub16(Type, w + (row * row_bytes), sub, s);
  alignas(16) __half h[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    h[j] = __float2half((s.scale * static_cast<float>(s.q[j])) - s.offset);
  }
  uint4 v[2];
  __builtin_memcpy(v, h, 32);
  auto* dst = reinterpret_cast<uint4*>(out + (idx * 16));
  dst[0] = v[0];
  dst[1] = v[1];
}

template<core::GgmlType Type>
void Launch(const void* w, void* out, std::size_t rows, std::size_t cols,
            hipStream_t stream) {
  const std::size_t threads = rows * (cols / 16);
  constexpr unsigned kBlock = 256;
  DequantizeToHalfKernel<Type>
      <<<static_cast<unsigned>((threads + kBlock - 1) / kBlock), kBlock, 0,
         stream>>>(static_cast<const std::uint8_t*>(w),
                   static_cast<__half*>(out), rows, cols);
}

}  // namespace

/// Dequantises `rows` x `cols` weights of `type` into binary16 `out`
/// ([rows][cols]). Returns false, launching nothing, for other types or a
/// cols that is not a multiple of the type's block size.
bool DequantizeToHalf(core::GgmlType type, const void* w, void* out,
                      std::size_t rows, std::size_t cols, hipStream_t stream) {
  if (rows == 0 || cols % 256 != 0) {
    return false;
  }
  switch (type) {
    case core::GgmlType::kQ4_K:
      Launch<core::GgmlType::kQ4_K>(w, out, rows, cols, stream);
      return true;
    case core::GgmlType::kQ5_K:
      Launch<core::GgmlType::kQ5_K>(w, out, rows, cols, stream);
      return true;
    case core::GgmlType::kQ6_K:
      Launch<core::GgmlType::kQ6_K>(w, out, rows, cols, stream);
      return true;
    case core::GgmlType::kIQ4_XS:
      Launch<core::GgmlType::kIQ4_XS>(w, out, rows, cols, stream);
      return true;
    case core::GgmlType::kIQ3_S:
      Launch<core::GgmlType::kIQ3_S>(w, out, rows, cols, stream);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::bench
