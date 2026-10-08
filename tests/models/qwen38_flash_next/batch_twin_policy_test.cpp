#include "src/models/qwen38_flash_next/batch_twin_policy.hpp"

#include <iostream>
#include <stdexcept>

namespace q = gufo::models::qwen38_flash_next;
using T = gufo::core::GgmlType;
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("batch twin contract");
}
int main() {
  for (std::uint32_t n = 1; n <= q::kTwinBatchMax; ++n) {
    for (bool armed : {false, true})
      for (bool enabled : {false, true})
        for (bool prefill : {false, true}) {
          Require(q::BatchUsesTwin(armed, enabled, prefill) ==
                  (armed && enabled && !prefill));
          for (T t : {T::kQ8_0, T::kF16, T::kF32, T::kIQ4_XS, T::kIQ4_NL,
                      T::kQ4_K, T::kQ5_K, T::kQ5_1}) {
            const auto p = q::BatchDenseDispatch(t, n, armed && enabled);
            if (n <= 8)
              Require(p == q::BatchDensePath::kDense);
            else if (armed && enabled && q::TwinDenseType(t))
              Require(p == q::BatchDensePath::kTwinSlices);
            else
              Require(p == (t == T::kQ8_0 ? q::BatchDensePath::kQ8
                                          : q::BatchDensePath::kSmallGemm));
            Require(!(armed && enabled && q::TwinDenseType(t) &&
                      p == q::BatchDensePath::kSmallGemm));
          }
        }
  }
  for (std::uint32_t c = 1; c <= 8; ++c)
    for (std::uint32_t k = 1; k <= 8; ++k)
      Require(q::TwinBatchLoadError(true, true, c, k, 8) == nullptr);
  Require(q::TwinBatchLoadError(true, true, 8, 9, 8) != nullptr);
  Require(q::TwinBatchLoadError(true, true, 9, 8, 8) != nullptr);
  Require(q::TwinBatchLoadError(true, true, 1, 0, 8) != nullptr);
  for (long n = 0; n < 8; ++n)
    Require(q::TwinBatchLoadError(true, true, 4, 8, n) != nullptr);
  Require(q::TwinBatchLoadError(true, false, 2, 8, 8) != nullptr);
  Require(q::TwinBatchLoadError(true, false, 1, 8, 6) != nullptr);
  Require(q::TwinBatchLoadError(false, true, 9, 9, 1) == nullptr);
  std::cout << "PASS batch layer selector 1..64; dispatch type x rows; load "
               "refusal; OFF contract\n";
}
