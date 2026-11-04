#ifndef GUFO_MODELS_QWEN36_A3B_PREFILL_FAILURE_HPP_
#define GUFO_MODELS_QWEN36_A3B_PREFILL_FAILURE_HPP_

#include <stdexcept>
#include <string_view>

namespace gufo::models::qwen36_a3b {
template<class Diagnostic>
[[noreturn]] inline void FailPrefillExperts(std::string_view detail,
                                            Diagnostic diagnostic) {
  diagnostic(detail);
  throw std::runtime_error("inference execution failed");
}
}  // namespace gufo::models::qwen36_a3b
#endif
