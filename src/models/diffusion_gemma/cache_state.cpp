#include "src/models/diffusion_gemma/cache_state.hpp"

#include <algorithm>

namespace gufo::models::diffusion_gemma {

CacheState::CacheState(int sliding_window, int reuse_history)
    : sliding_window_(sliding_window), reuse_history_(reuse_history) {}

CacheState::Pass CacheState::GlobalPass() const {
  return {.first_slot = length_, .skip = 0, .first_position = 0};
}

CacheState::Pass CacheState::SlidingPass() const {
  // Only the newest window - 1 cached positions are attended to.
  const int skip =
      sliding_valid_ - std::min(sliding_valid_, sliding_window_ - 1);
  return {.first_slot = sliding_valid_,
          .skip = skip,
          .first_position = length_ - sliding_valid_ + skip};
}

int CacheState::ReusableLength(std::span<const std::int32_t> prompt,
                               std::span<const ImageSpan> images) const {
  const int limit = std::min<int>(length_, static_cast<int>(prompt.size()));
  int reuse = 0;
  while (reuse < limit && tokens_[reuse] == prompt[reuse])
    ++reuse;
  // Matching soft-token IDs say nothing about their pixels. A different
  // image invalidates the cache starting at its first soft-token row.
  for (std::size_t i = 0; i < std::max(images.size(), images_.size()); ++i) {
    if (i < images.size() && i < images_.size() && images[i] == images_[i])
      continue;
    if (i < images.size())
      reuse = std::min(reuse, static_cast<int>(images[i].offset));
    if (i < images_.size())
      reuse = std::min(reuse, static_cast<int>(images_[i].offset));
    break;
  }
  for (const auto& image : images)
    if (reuse > static_cast<int>(image.offset) &&
        reuse < static_cast<int>(image.offset + image.rows))
      reuse = image.offset;
  for (const auto& image : images_)
    if (reuse > static_cast<int>(image.offset) &&
        reuse < static_cast<int>(image.offset + image.rows))
      reuse = image.offset;
  // The sliding layers continue at `reuse` only if they still hold the
  // window before it.
  const int window_start = reuse - std::min(reuse, sliding_window_ - 1);
  if (reuse == 0 || window_start < length_ - sliding_valid_)
    return 0;
  return reuse;
}

int CacheState::Reuse(std::span<const std::int32_t> prompt,
                      std::span<const ImageSpan> images) {
  const int reuse = ReusableLength(prompt, images);
  if (reuse == 0) {
    Clear();
    return 0;
  }
  sliding_valid_ -= length_ - reuse;
  length_ = reuse;
  tokens_.resize(reuse);
  std::erase_if(images_, [reuse](const ImageSpan& image) {
    return image.offset >= static_cast<std::uint32_t>(reuse);
  });
  return reuse;
}

CacheState::Shift CacheState::Append(std::span<const std::int32_t> tokens,
                                     std::span<const ImageSpan> images) {
  const int t = static_cast<int>(tokens.size());
  const int filled = sliding_valid_ + t;
  const int keep = std::min(filled, sliding_keep());
  length_ += t;
  sliding_valid_ = keep;
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  images_.insert(images_.end(), images.begin(), images.end());
  return {.from = filled - keep, .count = keep};
}

void CacheState::Clear() {
  length_ = 0;
  sliding_valid_ = 0;
  tokens_.clear();
  images_.clear();
}

}  // namespace gufo::models::diffusion_gemma
