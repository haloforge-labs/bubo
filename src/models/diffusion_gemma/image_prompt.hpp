#ifndef GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_PROMPT_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_PROMPT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::models::diffusion_gemma {

/// Absolute prompt position of one run of vision soft tokens. The identity
/// includes this image and all earlier images, their positions and projector.
struct ImageSpan {
  std::uint32_t offset{0};
  std::uint32_t rows{0};
  std::array<std::uint8_t, 32> prefix_identity{};
  bool operator==(const ImageSpan&) const = default;
};

/// The encoder owns `embedding` while Generate runs; rows are device FP32.
struct ImageInput : ImageSpan {
  const float* embedding{nullptr};
};

struct PromptChunk {
  int rows{0};
  /// Index in the prompt's image array, or -1 for causal text.
  int image{-1};
};

/// Plan one prefill pass. Text stops before an image; image rows are a whole
/// pass with bidirectional attention. A zero-row result means the caller must
/// wait for enough batch capacity. Throws for invalid placements or a span
/// larger than the configured chunk or context.
PromptChunk PlanPromptChunk(std::size_t prompt_size, std::size_t offset,
                            int available_rows, int chunk_rows, int max_context,
                            std::span<const ImageSpan> images);

void ValidateImageSpans(std::size_t prompt_size, int chunk_rows,
                        int max_context, std::span<const ImageSpan> images);

/// Validate an expanded prompt before admitting a request or starting work.
void ValidateImagePrompt(std::span<const std::int32_t> prompt, int chunk_rows,
                         int max_context, int canvas,
                         std::span<const ImageSpan> images);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_PROMPT_HPP_
