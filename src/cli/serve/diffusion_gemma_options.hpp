#ifndef GUFO_SERVER_DIFFUSION_GEMMA_OPTIONS_HPP_
#define GUFO_SERVER_DIFFUSION_GEMMA_OPTIONS_HPP_

#include <cstddef>
#include <filesystem>
#include <stdexcept>

namespace gufo::server {

/// Same sidecar tower and largest-image scratch allowance as Gemma 4.
inline constexpr std::size_t kDiffusionGemmaVisionReserveBytes =
    std::size_t{2} * 1024 * 1024 * 1024;

inline void ValidateDiffusionGemmaVisionReserve(std::size_t free_bytes) {
  if (free_bytes < kDiffusionGemmaVisionReserveBytes)
    throw std::runtime_error(
        "DiffusionGemma vision needs 2 GiB free device memory for the "
        "projector and image scratch");
}

inline bool ShouldBindDiffusionGemmaVision(const std::filesystem::path& mmproj,
                                           bool vision_disabled) {
  return !vision_disabled && !mmproj.empty();
}

struct DiffusionGemmaUnsupported {
  bool mmproj{false};
  bool speculative{false};
  bool cache_disk{false};
  bool npu_prefill{false};
  bool draft_model{false};
};

/// Sidecar vision is supported; retain the other DiffusionGemma refusals.
inline const char* UnsupportedDiffusionGemmaOption(
    const DiffusionGemmaUnsupported& options) {
  (void)options.mmproj;
  if (options.speculative)
    return "speculative decoding";
  if (options.cache_disk)
    return "--cache-disk";
  if (options.npu_prefill)
    return "--npu-prefill on";
  if (options.draft_model)
    return "--twin-model and --draft-head-model";
  return nullptr;
}

inline void ValidateDiffusionGemmaImageRequest(bool has_images,
                                               bool projector_bound) {
  if (has_images && !projector_bound)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_GEMMA_OPTIONS_HPP_
