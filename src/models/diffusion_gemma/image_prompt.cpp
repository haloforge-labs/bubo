#include "src/models/diffusion_gemma/image_prompt.hpp"

#include <algorithm>
#include <stdexcept>

namespace gufo::models::diffusion_gemma {

void ValidateImageSpans(std::size_t prompt_size, int chunk_rows,
                        int max_context, std::span<const ImageSpan> images) {
  std::size_t previous_end = 0;
  for (const auto& image : images) {
    const std::size_t end = std::size_t{image.offset} + image.rows;
    if (image.rows == 0 || image.offset == 0 || image.offset < previous_end ||
        end >= prompt_size)
      throw std::invalid_argument("invalid image span in prompt");
    if (image.rows > static_cast<std::uint32_t>(chunk_rows))
      throw std::length_error("image exceeds the configured prefill chunk");
    if (end > static_cast<std::size_t>(max_context))
      throw std::length_error("image exceeds the configured context");
    previous_end = end;
  }
  if (prompt_size > static_cast<std::size_t>(max_context))
    throw std::length_error("image prompt exceeds the configured context");
}

void ValidateImagePrompt(std::span<const std::int32_t> prompt, int chunk_rows,
                         int max_context, int canvas,
                         std::span<const ImageSpan> images) {
  if (!images.empty()) {
    ValidateImageSpans(prompt.size(), chunk_rows, max_context, images);
    for (const auto& image : images)
      if (prompt[image.offset - 1] != 255999 ||
          prompt[image.offset + image.rows] != 258882 ||
          !std::all_of(prompt.begin() + image.offset,
                       prompt.begin() + image.offset + image.rows,
                       [](std::int32_t id) { return id == 258880; }))
        throw std::invalid_argument("invalid DiffusionGemma image tokens");
  }
  if (prompt.size() + static_cast<std::size_t>(canvas) >
      static_cast<std::size_t>(max_context))
    throw std::length_error(
        "prompt leaves no room for a canvas within the configured context");
}

PromptChunk PlanPromptChunk(std::size_t prompt_size, std::size_t offset,
                            int available_rows, int chunk_rows, int max_context,
                            std::span<const ImageSpan> images) {
  ValidateImageSpans(prompt_size, chunk_rows, max_context, images);
  if (offset > prompt_size || available_rows < 0)
    throw std::invalid_argument("invalid prompt chunk offset or capacity");
  const auto take = std::min<std::size_t>(
      {prompt_size - offset, static_cast<std::size_t>(available_rows),
       static_cast<std::size_t>(chunk_rows)});
  for (std::size_t i = 0; i < images.size(); ++i) {
    const auto& image = images[i];
    if (offset > image.offset && offset < image.offset + image.rows)
      throw std::logic_error("prefill resumed inside an image");
    if (image.offset == offset) {
      if (take < image.rows)
        return {};
      return {static_cast<int>(image.rows), static_cast<int>(i)};
    }
    if (image.offset > offset && image.offset < offset + take)
      return {static_cast<int>(image.offset - offset), -1};
  }
  return {static_cast<int>(take), -1};
}

}  // namespace gufo::models::diffusion_gemma
