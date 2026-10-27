#ifndef GUFO_MODELS_DIFFUSION_GEMMA_CACHE_STATE_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_CACHE_STATE_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/models/diffusion_gemma/image_prompt.hpp"

namespace gufo::models::diffusion_gemma {

/// Host bookkeeping of the KV caches; the engine owns the device buffers.
///
/// Global layers hold every encoded position, at slot = position. Sliding
/// layers hold the newest `sliding_valid()` positions at slots
/// [0, sliding_valid()): the attention window (window - 1 positions) plus up
/// to `reuse_history` older ones. The older ones are never attended to; they
/// let a later prompt that diverges from the cached tokens up to that far
/// before the end reuse the cache (see Reuse).
class CacheState {
public:
  CacheState(int sliding_window, int reuse_history);

  /// Positions encoded so far, and their tokens.
  [[nodiscard]] int length() const { return length_; }
  [[nodiscard]] std::span<const std::int32_t> tokens() const { return tokens_; }
  [[nodiscard]] int sliding_valid() const { return sliding_valid_; }
  /// Positions the sliding caches keep after a pass.
  [[nodiscard]] int sliding_keep() const {
    return sliding_window_ - 1 + reuse_history_;
  }

  /// Where a pass writes its keys and values, and which cached slots it
  /// attends to: slots [skip, first_slot + rows), the first of them at
  /// position `first_position`.
  struct Pass {
    int first_slot{0};
    int skip{0};
    int first_position{0};
  };
  [[nodiscard]] Pass GlobalPass() const;
  [[nodiscard]] Pass SlidingPass() const;

  /// Keeps the longest cached prefix of `prompt` that the sliding layers can
  /// continue from (the window before it is still held) and drops the rest;
  /// returns its length. Clears the cache and returns 0 when no prefix can
  /// be reused.
  int Reuse(std::span<const std::int32_t> prompt,
            std::span<const ImageSpan> images = {});
  /// What Reuse(prompt) would keep, without changing the cache.
  [[nodiscard]] int ReusableLength(
      std::span<const std::int32_t> prompt,
      std::span<const ImageSpan> images = {}) const;

  /// Records an encoder pass over `tokens`. The sliding caches must then
  /// move slots [from, from + count) to slot 0.
  struct Shift {
    int from{0};
    int count{0};
  };
  Shift Append(std::span<const std::int32_t> tokens,
               std::span<const ImageSpan> images = {});

  void Clear();

private:
  int sliding_window_;
  int reuse_history_;
  int length_{0};
  int sliding_valid_{0};
  std::vector<std::int32_t> tokens_;
  std::vector<ImageSpan> images_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_CACHE_STATE_HPP_
