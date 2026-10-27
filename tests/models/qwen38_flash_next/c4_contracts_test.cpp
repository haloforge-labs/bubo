// CPU-checkable contracts of the C4 correctness fixes. The kernel-level
// contracts (sparse split invariance, BF16 absolute K rotation, paired
// SwiGLU rounding, continuation sync splits) are GPU tests in
// attention_ops_test, blaslt_ops_test, routed_wmma_ops_test and session_test.
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// The C4 fixes change inference arithmetic (sparse key tiling, BF16 K order,
// SwiGLU product order). engine.hpp requires a snapshot version bump for that,
// so states made by the old arithmetic are rejected instead of being resumed.
void CheckSnapshotVersionCoversC4() {
  Require(qfn::Session::kSnapshotPayloadVersion >= 18,
          "snapshot payload version " +
              std::to_string(qfn::Session::kSnapshotPayloadVersion) +
              " does not invalidate pre-C4 arithmetic (need >= 18)");
}

}  // namespace

int main() {
  try {
    CheckSnapshotVersionCoversC4();
    std::cout << "PASS: C4 contracts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
