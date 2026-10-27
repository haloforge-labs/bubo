#ifndef GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_CHAT_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_CHAT_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "src/models/diffusion_gemma/chat.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::diffusion_gemma {

struct ImageChatRendered {
  std::string text;
  std::vector<std::size_t> image_offsets;
};

/// Add image placeholders at the HTTP content-part offsets without changing
/// DiffusionGemma's chat framing, including tool responses and later turns.
ImageChatRendered RenderImageChat(
    std::span<const tokenization::ChatMessage> source,
    std::vector<Message> messages, const std::vector<json::Value>& tools,
    bool thinking);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_IMAGE_CHAT_HPP_
