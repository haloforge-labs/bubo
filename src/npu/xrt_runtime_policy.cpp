#include <stdexcept>

#include "src/npu/xrt_runtime.hpp"

namespace gufo::npu::detail {

// Adapted from npu-tooling-sol/host.cpp runtimeCheck; build/code are unsigned
// integers, not a Git hash. No device API belongs in this loading boundary.
void ValidateAndLoad(RuntimeOperations& operations) {
  const auto identity = operations.Inspect();
  if (!identity.path.is_absolute() ||
      identity.path.filename() != "xrt_coreutil.dll") {
    throw std::runtime_error("NPU XRT requires an absolute runtime DLL path");
  }
  if (identity.sha256 !=
      "04a26d37c6e0c713491ad0bfae74ce74ea94c74136d2aa056333616dac6c3a44") {
    throw std::runtime_error("NPU XRT DLL SHA256 mismatch");
  }
  if (identity.file_version != std::array<unsigned, 4>{32, 0, 20102, 3931}) {
    throw std::runtime_error("NPU XRT DLL file version mismatch");
  }
  operations.Load(identity.path);
  if (operations.Versions() !=
      std::array<unsigned, 5>{2, 21, 0, 8679, 131093}) {
    throw std::runtime_error("NPU XRT five-function version mismatch");
  }
}

}  // namespace gufo::npu::detail
