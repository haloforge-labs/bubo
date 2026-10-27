#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_DRAFT_UPLOAD_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_DRAFT_UPLOAD_HPP_

#include <string>

#include "src/models/gemma4/kernels/rocm/device_model.hpp"

namespace gufo::models::gemma4::rocm {

/// Preserve the RC's native Q8_0 repack decision.
[[nodiscard]] inline bool ShouldRepackDraftVocabAsQ4K(
    const DeviceTensor& head) {
  return head.type == core::GgmlType::kQ8_0 && head.cols % 256 == 0;
}

/// BF16 vocabulary always enters Q4_K through host Q8_0 quantization and
/// the existing GPU repack. Pre/post projections retain the layer uploader's
/// in-place binary16 conversion and named overflow accounting. Native formats
/// keep their old copy/repack path. Host-only for CPU policy tests.
template<class Uploader>
DeviceTensor UploadDraftProjection(const TensorRef& host, Uploader& up) {
  if (!host.empty() && host.name == "token_embd.weight" &&
      host.type == core::GgmlType::kBF16) {
    if (host.cols == 0 || host.cols % 256 != 0) {
      up.Fail("tensor " + std::string(host.name) +
              ": BF16 draft vocabulary Q4_K requires cols aligned to 256");
      return {};
    }
    return up.QuantizeBf16(host);
  }
  DeviceTensor device = up.Copy(host);
  if (!device.empty() && device.type == core::GgmlType::kBF16) {
    up.Half(device, host.name);
  }
  return device;
}

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_DRAFT_UPLOAD_HPP_
