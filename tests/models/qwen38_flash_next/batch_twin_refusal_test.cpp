#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/models/qwen38_flash_next/batch_twin_policy.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {
void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("twin load refusal contract");
}

void SetBatchSwitch(const char* value) {
#ifdef _WIN32
  Require(_putenv_s("GUFO_DENSE_TWIN_BATCH", value == nullptr ? "" : value) ==
          0);
#else
  Require((value == nullptr ? unsetenv("GUFO_DENSE_TWIN_BATCH")
                            : setenv("GUFO_DENSE_TWIN_BATCH", value, 1)) == 0);
#endif
}
}  // namespace

int main() {
  const char* previous = std::getenv("GUFO_DENSE_TWIN_BATCH");
  const bool had_previous = previous != nullptr;
  const std::string saved = had_previous ? previous : "";
  // Independent golden copied from B5c Model::Load, including punctuation.
  constexpr std::string_view refusal =
      "GUFO_DENSE_TWIN_MODEL supports one decode session (the batched "
      "decode loop is not twinned)";
  for (const char* value :
       std::array<const char*, 5>{nullptr, "0", "true", "01", "1"}) {
    SetBatchSwitch(value);
    const bool enabled = q::TwinBatchEnabled();
    Require(enabled == (value != nullptr && std::string_view(value) == "1"));
    for (const char* max_n : std::array<const char*, 8>{
             nullptr, "", "6", "8", "99", "abc", "0", "-1"}) {
      const auto cutoff = q::TwinMaxNValue(max_n);
      // Call the exact predicate and parser used by Model::Load.
      const char* failure = q::TwinBatchLoadError(true, enabled, 2, 8, cutoff);
      Require(q::TwinBatchLoadError(true, enabled, 1, 8, cutoff) == nullptr);
      if (!enabled) {
        Require(failure != nullptr && std::string_view(failure) == refusal);
      } else {
        Require((failure == nullptr) == (cutoff == 8));
      }
      Require(q::TwinBatchLoadError(false, enabled, 2, 8, cutoff) == nullptr);
    }
    for (std::uint32_t rows = 9; rows <= 64; ++rows)
      Require(q::BatchDenseDispatch(gufo::core::GgmlType::kQ8_0, rows,
                                    enabled) == q::BatchDensePath::kQ8);
  }
  Require(q::TwinMaxNValue(nullptr) == 8);
  Require(q::TwinMaxNValue("6") == 6);
  Require(q::TwinMaxNValue("abc") == 1);
  Require(q::TwinMaxNValue("99") == 8);
  Require(q::TwinBatchLoadError(true, true, 2, 9, 8) != nullptr);
  Require(q::TwinBatchLoadError(true, true, 2, std::uint64_t{1} << 32, 8) !=
          nullptr);
  SetBatchSwitch(had_previous ? saved.c_str() : nullptr);
  std::cout << "PASS real Load predicate: exact B5c refusal unset/0/true/01; "
               "c1 cutoff accepted with every switch; literal 1 only; shared "
               "cutoff parser; "
               "armed Q8 logits rows 9..64; chain bounds\n";
}
