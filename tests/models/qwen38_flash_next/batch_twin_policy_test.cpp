#include "src/models/qwen38_flash_next/batch_twin_policy.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace q = gufo::models::qwen38_flash_next;
using T = gufo::core::GgmlType;
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("batch twin contract");
}
void BatchEnv(const char* value) {
#ifdef _WIN32
  _putenv_s("GUFO_DENSE_TWIN_BATCH", value == nullptr ? "" : value);
#else
  if (value != nullptr)
    setenv("GUFO_DENSE_TWIN_BATCH", value, 1);
  else
    unsetenv("GUFO_DENSE_TWIN_BATCH");
#endif
}
int main() {
  const char* old_env = std::getenv("GUFO_DENSE_TWIN_BATCH");
  const bool had_env = old_env != nullptr;
  const std::string saved_env = had_env ? old_env : "";
  for (const char* value :
       std::array<const char*, 6>{nullptr, "", "0", "1", "11", "true"}) {
    BatchEnv(value);
    Require(q::TwinBatchEnabled() ==
            (value != nullptr && std::string(value) == "1"));
  }
  BatchEnv(had_env ? saved_env.c_str() : nullptr);
  for (std::uint32_t n = 1; n <= q::kTwinBatchMax; ++n) {
    for (bool armed : {false, true})
      for (bool enabled : {false, true})
        for (bool prefill : {false, true}) {
          Require(q::BatchUsesTwin(armed, enabled, prefill) ==
                  (armed && enabled && !prefill));
          for (bool multi_session : {false, true})
            for (T t : {T::kQ8_0, T::kF16, T::kF32, T::kBF16, T::kIQ4_XS,
                        T::kIQ4_NL, T::kQ4_K, T::kQ5_K, T::kQ5_1}) {
              const auto p =
                  q::BatchDenseDispatch(t, n, armed && enabled, multi_session);
              if (armed && enabled && multi_session &&
                  (t == T::kIQ4_XS || t == T::kIQ4_NL))
                Require(p == q::BatchDensePath::kTwinMultiRow);
              else if (n <= 8)
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
  Require(q::TwinBatchLoadError(true, true, 2, 0, 8) != nullptr);
  for (std::uint32_t n = 0; n < 8; ++n) {
    Require(q::TwinBatchLoadError(true, true, 4, 8, n) != nullptr);
    Require(q::TwinBatchLoadError(true, false, 1, 8, n) == nullptr);
  }
  Require(q::TwinBatchLoadError(true, false, 2, 8, 8) != nullptr);
  Require(q::TwinBatchLoadError(true, false, 1, 8, 6) == nullptr);
  Require(q::TwinBatchLoadError(false, true, 9, 9, 1) == nullptr);
  std::cout << "PASS batch layer selector 1..64; dispatch type x rows x "
               "session flag; load "
               "refusal; OFF contract\n";
}
