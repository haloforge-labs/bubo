#ifndef GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::gemma4 {

struct ChatOptions {
  bool add_generation_prompt{true};
  /// Gemma 4 thinking is opt-in: the template default is off.
  bool enable_thinking{false};
  bool preserve_thinking{false};
};

/// Resolves provider-neutral controls against the Gemma 4 template defaults.
/// Effort levels have no Gemma equivalent; any level other than minimal
/// enables thinking when it was not set explicitly.
[[nodiscard]] ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning);

struct RenderedPrompt {
  std::string text;
  /// Byte offset where the generation prompt starts; equals text.size()
  /// without one. Everything before it is the stable conversation prefix.
  std::size_t generation_prompt_offset{0};
  /// Byte offset of the prefix the next request is expected to share: the
  /// generation prompt, or, when the prompt ends with a text-only user turn
  /// after a tool result or another user turn, the end of the turn before
  /// it. Agent clients send per-turn context there and replace it on the
  /// next request. A tool result stays inside its model turn, which the next
  /// request continues, so the boundary precedes that turn's closing marker.
  std::size_t stable_prefix_offset{0};
};

/// Compiled renderer of Unsloth's Gemma 4 chat template. The GGUF template
/// is only accepted when its SHA-256 matches a qualified revision; the Jinja
/// source is kept in reference/ as data.
///
/// Known, documented approximations of the Jinja source: message content
/// parts arrive flattened by the HTTP layer, JSON numbers render as Python
/// ints when integral (Python would print 5.0 for a float literal), text
/// parts between two images are trimmed as one segment, a string `const`
/// in a tool schema is shown as the equivalent one-value `enum` (the Jinja
/// source drops `const`), and a replayed thought ends directly at
/// `<channel|>`. The Jinja source adds a newline there that Gemma does not
/// generate, so a replayed tool turn could not reuse its cached tokens. A tool
/// result's images render inside its response value, at their part's place,
/// where llama.cpp puts the media marker. The Jinja source emits them after
/// the block, which can leave a prompt that ends on an image; the 26B-A4B
/// then answers with repeated "-1" instead of reading it.
class ChatTemplate {
public:
  [[nodiscard]] static constexpr std::string_view
  UnslothTemplateSha256() noexcept {
    return "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
  }
  /// The template text that llama.cpp's converter embeds in the community
  /// abliterated/uncensored GGUFs (TrevorJS 26B-A4B and Huihui 31B v2 both
  /// carry it; 12,045 bytes). It is the earlier Google text, without the
  /// Unsloth fixes this renderer implements (null arguments, the
  /// `<|tool_response>` block, thinking-channel placement); the renderer, not
  /// the embedded text, defines the prompt, so the GGUF is accepted with that
  /// difference on record.
  [[nodiscard]] static constexpr std::string_view
  EarlyGoogleTemplateSha256() noexcept {
    return "2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c";
  }
  [[nodiscard]] static constexpr std::string_view TemplateId() noexcept {
    return "gemma4-unsloth-845f1ee4";
  }

  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<RenderedPrompt> Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
      std::string* error_msg = nullptr,
      std::vector<std::size_t>* image_offsets = nullptr);
};

/// Reasoning channel markup produced by the model.
inline constexpr std::string_view kThoughtStart = "<|channel>thought\n";
inline constexpr std::string_view kThoughtEnd = "<channel|>";
inline constexpr std::string_view kToolCallStart = "<|tool_call>";
inline constexpr std::string_view kToolCallEnd = "<tool_call|>";
/// Rendered in place of each user image; the prompt builder replaces it with
/// the image's soft tokens. Render reports the byte offset of each one it
/// emits through `image_offsets`; without that output images are rejected.
inline constexpr std::string_view kImagePlaceholder = "<|image|>";
/// String delimiter of the tool-call argument syntax.
inline constexpr std::string_view kStringQuote = "<|\"|>";

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
