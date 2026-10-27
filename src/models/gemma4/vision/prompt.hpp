#ifndef GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_
#define GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/tokenizer.hpp"

namespace gufo::models::gemma4::vision {

inline constexpr std::uint32_t kPatchSize = 16;
/// Average-pooling kernel over the patch grid; one soft token per 3x3.
inline constexpr std::uint32_t kPoolSize = 3;
inline constexpr std::uint32_t kSideMultiple = kPatchSize * kPoolSize;
/// Gemma4ImageProcessor's default `max_soft_tokens`.
inline constexpr std::uint32_t kDefaultSoftTokens = 280;
/// The soft-token budgets the checkpoint supports (model card, "variable
/// image resolution"): more tokens keep more detail at a higher cost.
inline constexpr std::array<std::uint32_t, 5> kSoftTokenBudgets = {70, 140, 280,
                                                                   560, 1120};
[[nodiscard]] constexpr bool IsSoftTokenBudget(std::uint32_t tokens) {
  for (const auto budget : kSoftTokenBudgets) {
    if (budget == tokens) {
      return true;
    }
  }
  return false;
}

struct ImageSize {
  std::uint32_t width{0};
  std::uint32_t height{0};
  bool operator==(const ImageSize&) const = default;
};

/// `soft_tokens == kNativeSize` keeps an image near its real size (see
/// NativeResizedSize); any other value is a fixed budget (ResizedSize).
inline constexpr std::uint32_t kNativeSize = 0;
/// The range of soft tokens one image may take at native size, as the model's
/// variable-resolution support (70 to 1120).
inline constexpr std::uint32_t kMinSoftTokens = 70;
inline constexpr std::uint32_t kMaxSoftTokens = 1120;

/// Gemma4ImageProcessor's aspect-preserving target: the largest multiple-of-48
/// size with at most `soft_tokens` pooled cells, filling the budget whether
/// that enlarges or shrinks the image.
[[nodiscard]] ImageSize ResizedSize(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t soft_tokens = kDefaultSoftTokens);
/// Native-size target, the way Flash Next and llama.cpp treat an image: each
/// side rounds to the nearest multiple of 48 (at least 48); an image above
/// `max_tokens` cells shrinks to fit, one below `min_tokens` enlarges to
/// reach it. Small text and UI details keep their pixels up to the ceiling.
[[nodiscard]] ImageSize NativeResizedSize(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t min_tokens = kMinSoftTokens,
    std::uint32_t max_tokens = kMaxSoftTokens);
/// `kNativeSize` (the default) or a fixed budget.
[[nodiscard]] core::Image ResizeImage(const core::Image& image,
                                      std::uint32_t soft_tokens = kNativeSize);

/// `<|image>`, the soft token `<|image|>` and `<image|>`, looked up by text.
struct ImageTokens {
  TokenId start{-1};
  TokenId soft{-1};
  TokenId end{-1};
};
[[nodiscard]] ImageTokens FindImageTokens(const Tokenizer& tokenizer);

/// Soft-token rows of one image. They attend to each other bidirectionally
/// in sliding layers, so a forward never splits them.
struct ImageSpan {
  std::uint32_t offset{0};  ///< first soft token
  std::uint32_t rows{0};
  bool operator==(const ImageSpan&) const = default;
};

struct PreparedImage {
  core::Image pixels;  ///< resized RGB8, both sides multiples of 48
  ImageSpan span;
  /// Covers every image up to and including this one, with its placement,
  /// the preprocessing version and the encoder identity.
  std::array<std::uint8_t, 32> prefix_identity{};
};

struct Prompt {
  std::vector<TokenId> tokens;
  /// Tokens before the generation prompt; zero when it does not tokenize
  /// apart.
  std::size_t stable_prefix_tokens{0};
  std::vector<PreparedImage> images;
  /// Identity of all images; empty for a text-only prompt.
  std::vector<std::uint8_t> cache_identity;
  /// Identity of the images starting before `token_count`.
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
};

/// Renders the conversation and expands each image into `<|image>`, its soft
/// tokens and `<image|>`. Images are decoded and resized here, once.
[[nodiscard]] Prompt Prepare(
    const Tokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
    std::string_view encoder_identity, std::uint32_t max_context,
    std::uint32_t soft_tokens = kNativeSize);

/// Shared image expansion after a model has rendered its own chat template.
/// `image_offsets` locate kImagePlaceholder in `rendered`; `encode` must
/// tokenize each intervening text run without adding a BOS token.
using TextEncoder = std::function<std::vector<TokenId>(std::string_view)>;
[[nodiscard]] Prompt PrepareRendered(
    std::string_view rendered, std::span<const std::size_t> image_offsets,
    std::span<const tokenization::ChatMessage> messages,
    const TextEncoder& encode, ImageTokens image_tokens,
    std::string_view encoder_identity, std::uint32_t max_context,
    std::uint32_t soft_tokens = kNativeSize,
    std::size_t stable_prefix_offset = 0);

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_
