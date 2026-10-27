// Process-local device headroom limit for snapshot storage tests. Linked only
// into the model-local snapshot test, not the model or serving library.
// Upstream interposes hipMemGetInfo with dlsym(RTLD_NEXT), which Windows does
// not offer, so this forwards to the engine's test seam instead.
#include <cstddef>

#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

/// Caps the free device memory HIP reports; the maximum removes the cap.
void LimitReportedDeviceFree(std::size_t bytes) {
  gufo::models::qwen38_flash_next::rocm::LimitSnapshotDeviceFreeForTest(bytes);
}
