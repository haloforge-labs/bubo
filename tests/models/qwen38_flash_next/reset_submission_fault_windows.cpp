// Windows uses model-local seams because HIP DLL imports cannot be interposed
// with RTLD_NEXT. Faults are consumed after real transfer completion.
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

void FailNextResetSubmission() {
  gufo::models::qwen38_flash_next::rocm::FailNextContinuationResetForTest();
}

void FailNextTransferWait() {
  gufo::models::qwen38_flash_next::rocm::FailNextContinuationTransferForTest();
}
