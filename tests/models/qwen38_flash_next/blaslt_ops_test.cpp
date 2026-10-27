#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Check(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}

struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

std::uint16_t Bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

// DenseBf16Gemm takes an optional device pointer to the absolute start token
// (C4). Before that parameter exists the call drops it, so this test builds
// on both sides and reports the old behaviour as a failure.
template<class Offset>
bool CallBf16(const void* w, const void* x, float* out, std::size_t batch,
              std::size_t m, std::size_t k, const Offset* offset) {
  namespace rocm = gufo::models::qwen38_flash_next::rocm;
  if constexpr (requires {
                  rocm::DenseBf16Gemm(w, x, out, batch, m, k, nullptr, offset);
                }) {
    return rocm::DenseBf16Gemm(w, x, out, batch, m, k, nullptr, offset);
  } else {
    return rocm::DenseBf16Gemm(w, x, out, batch, m, k, nullptr);
  }
}

// C4 BF16 absolute K rotation: the summation order of row r must depend on
// its absolute token position only, never on where the launch started.
void CheckBf16AbsoluteOffset() {
  constexpr std::size_t m = 64, k = 2560, total = 700, pad = 600;
  std::vector<std::uint16_t> weights(m * k), input(total * k);
  unsigned rng = 0xC4BF1600U;
  for (auto* values : {&weights, &input}) {
    for (auto& v : *values) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      // A wide dynamic range makes the F32 sum order visible.
      const float magnitude = static_cast<float>(1U << (rng % 12));
      v = Bf16(static_cast<float>(int(rng % 2049) - 1024) / 1024.0F *
               magnitude);
    }
  }
  Buffer w(weights.size() * 2), x(input.size() * 2);
  Buffer out((pad + total + 64) * m * sizeof(float));
  Buffer offset_device(sizeof(std::uint32_t));
  Check(hipMemcpy(w.data, weights.data(), weights.size() * 2,
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                  hipMemcpyHostToDevice));
  const auto fill_canary = [&] {
    std::vector<float> canary((pad + total + 64) * m, -777.0F);
    Check(hipMemcpy(out.data, canary.data(), canary.size() * sizeof(float),
                    hipMemcpyHostToDevice));
  };
  const auto download = [&] {
    std::vector<float> all((pad + total + 64) * m);
    Check(hipMemcpy(all.data(), out.data, all.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    return all;
  };
  float* base = static_cast<float*>(out.data) + pad * m;
  const std::uint32_t zero = 0;
  Check(hipMemcpy(offset_device.data, &zero, sizeof(zero),
                  hipMemcpyHostToDevice));
  fill_canary();
  Require(CallBf16(w.data, x.data, base, total, m, k,
                   static_cast<const std::uint32_t*>(offset_device.data)),
          "BF16 full launch rejected");
  Check(hipDeviceSynchronize());
  const auto full = download();
  for (const std::uint32_t start : {1U, 31U, 32U, 127U, 128U, 511U, 512U}) {
    const std::size_t rows = total - start;
    Check(hipMemcpy(offset_device.data, &start, sizeof(start),
                    hipMemcpyHostToDevice));
    fill_canary();
    Require(CallBf16(w.data, static_cast<const std::uint16_t*>(x.data) +
                                 std::size_t(start) * k,
                     base + std::size_t(start) * m, rows, m, k,
                     static_cast<const std::uint32_t*>(offset_device.data)),
            "BF16 suffix launch rejected");
    Check(hipDeviceSynchronize());
    const auto suffix = download();
    for (std::size_t i = 0; i < pad * m; ++i)
      Require(suffix[i] == -777.0F, "BF16 launch wrote before its output");
    for (std::size_t i = (pad + total) * m; i < suffix.size(); ++i)
      Require(suffix[i] == -777.0F, "BF16 launch wrote past its output");
    for (std::size_t i = 0; i < start * m; ++i)
      Require(suffix[pad * m + i] == -777.0F,
              "BF16 suffix launch wrote rows before its start");
    Require(std::memcmp(suffix.data() + (pad + start) * m,
                        full.data() + (pad + start) * m,
                        rows * m * sizeof(float)) == 0,
            "BF16 projection of absolute start " + std::to_string(start) +
                " differs from the whole launch");
  }
  std::cout << "PASS: BF16 absolute K rotation across starts\n";
}

// Shape-stable dispatch: the first rows of a wider batch equal the same rows
// of a narrower one (the old dispatch changed the library kernel at 4096).
void CheckBf16ShapeStable() {
  using gufo::models::qwen38_flash_next::rocm::BlasLt;
  constexpr int m = 64, k = 2560, narrow = 600, wide = 4200;
  std::string error;
  auto blas = BlasLt::Create(nullptr, &error);
  Require(blas != nullptr, error);
  std::vector<std::uint16_t> weights(std::size_t(m) * k),
      input(std::size_t(wide) * k);
  unsigned rng = 0xC4BF1601U;
  for (auto* values : {&weights, &input})
    for (auto& v : *values) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      v = Bf16(static_cast<float>(int(rng % 2049) - 1024) / 1024.0F *
               static_cast<float>(1U << (rng % 12)));
    }
  Buffer w(weights.size() * 2), x(input.size() * 2);
  Buffer c(std::size_t(wide) * m * sizeof(float));
  Check(hipMemcpy(w.data, weights.data(), weights.size() * 2,
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                  hipMemcpyHostToDevice));
  std::vector<float> a(std::size_t(narrow) * m), b(std::size_t(wide) * m);
  Require(blas->Gemm(w.data, x.data, static_cast<float*>(c.data), HIP_R_16BF,
                     m, narrow, k, &error),
          error);
  Check(hipDeviceSynchronize());
  Check(hipMemcpy(a.data(), c.data, a.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  Require(blas->Gemm(w.data, x.data, static_cast<float*>(c.data), HIP_R_16BF,
                     m, wide, k, &error),
          error);
  Check(hipDeviceSynchronize());
  Check(hipMemcpy(b.data(), c.data, b.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  Require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "BF16 projection changed with the batch width");
  std::cout << "PASS: BF16 projection is shape-stable across 4096\n";
}

}  // namespace

int main() {
  try {
    using gufo::models::qwen38_flash_next::rocm::BlasLt;
    std::string error;
    auto warmed = BlasLt::Create(nullptr, &error);
    Require(warmed != nullptr, error);
    constexpr int max_n = 4096;
    for (const auto [m, k] :
         {std::pair{96, 2560}, std::pair{513, 2560}, std::pair{33, 96}}) {
      std::vector<__half> weights(static_cast<std::size_t>(m) * k);
      std::vector<__half> input(static_cast<std::size_t>(max_n) * k);
      unsigned rng = 123;
      for (auto* values : {&weights, &input}) {
        for (auto& v : *values) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          v = __float2half_rn(static_cast<float>(int(rng % 8193) - 4096) /
                              4096.0F);
        }
      }
      Buffer a(weights.size() * sizeof(__half));
      Buffer b(input.size() * sizeof(__half));
      Buffer c(static_cast<std::size_t>(m) * max_n * sizeof(float));
      Check(hipMemcpy(a.data, weights.data(), weights.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(b.data, input.data(), input.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      if (k == 2560) {
        // Exercise both an unaligned cache boundary and a wide prefill split.
        for (const auto [total, boundary] :
             {std::pair{136, 94}, std::pair{2048, 1025}}) {
          std::vector<float> bulk(std::size_t(m) * total), split(bulk.size());
          Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                               HIP_R_16F, m, total, k, &error),
                  error);
          Check(hipMemcpy(bulk.data(), c.data, bulk.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          for (const auto [off, n] : {std::pair{0, boundary},
                                      std::pair{boundary, total - boundary}}) {
            Require(
                warmed->Gemm(
                    a.data, static_cast<__half*>(b.data) + std::size_t(off) * k,
                    static_cast<float*>(c.data) + std::size_t(off) * m,
                    HIP_R_16F, m, n, k, &error),
                error);
          }
          Check(hipMemcpy(split.data(), c.data, split.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          Require(bulk == split, "projection changed across prefill chunks");
        }
      }
      // Vary shape order and use ragged sizes between the calibrated widths.
      for (int n : {4096, 2048, 9, 1024, 512, 129, 128, 513, 256}) {
        auto fresh = BlasLt::Create(nullptr, &error);
        Require(fresh != nullptr, error);
        std::vector<float> reference(static_cast<std::size_t>(m) * n);
        std::vector<float> actual(reference.size());
        for (int pass = 0; pass < 3; ++pass) {
          auto& blas = pass == 1 ? fresh : warmed;
          Require(blas->Gemm(a.data, b.data, static_cast<float*>(c.data),
                             HIP_R_16F, m, n, k, &error),
                  error);
          Check(hipDeviceSynchronize());
          Check(hipMemcpy(actual.data(), c.data, actual.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          if (pass == 0) {
            reference = actual;
          } else {
            Require(std::memcmp(reference.data(), actual.data(),
                                actual.size() * sizeof(float)) == 0,
                    "projection changed across instances or plan reuse");
          }
        }
        for (int sample = 0; sample < 32; ++sample) {
          const int row = (sample * 79) % n;
          const int col = (sample * 127) % m;
          double expected = 0.0;
          for (int j = 0; j < k; ++j) {
            expected += static_cast<double>(
                            __half2float(weights[std::size_t(col) * k + j])) *
                        __half2float(input[std::size_t(row) * k + j]);
          }
          const float result = actual[std::size_t(row) * m + col];
          Require(std::isfinite(result) && std::abs(result - expected) < 0.002,
                  "projection differs from the F64 reference");
        }
      }
    }
    std::cout << "PASS: F16 projection replay, chunk boundaries, ragged shapes "
                 "and F64\n";
    CheckBf16AbsoluteOffset();
    CheckBf16ShapeStable();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
