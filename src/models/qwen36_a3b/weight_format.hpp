#ifndef GUFO_MODELS_QWEN36_A3B_WEIGHT_FORMAT_HPP_
#define GUFO_MODELS_QWEN36_A3B_WEIGHT_FORMAT_HPP_

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen36_a3b {

constexpr bool IsKQuant(core::GgmlType type) {
  return type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K;
}

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_WEIGHT_FORMAT_HPP_
