#include "src/models/diffusion_gemma/image_chat.hpp"

#include <stdexcept>

#include "src/models/gemma4/chat_template.hpp"

namespace gufo::models::diffusion_gemma {

ImageChatRendered RenderImageChat(
    std::span<const tokenization::ChatMessage> source,
    std::vector<Message> messages, const std::vector<json::Value>& tools,
    bool thinking) {
  if (source.size() != messages.size())
    throw std::invalid_argument("image chat messages do not match");
  std::vector<std::string> markers;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (source[i].images.empty())
      continue;
    std::string content;
    std::size_t cursor = 0;
    for (const auto& image : source[i].images) {
      if (!image.bytes || image.offset < cursor ||
          image.offset > source[i].content.size())
        throw std::invalid_argument("invalid image part in chat message");
      content.append(source[i].content, cursor, image.offset - cursor);
      std::string marker =
          "\x1f"
          "gufo-image-" +
          std::to_string(markers.size()) + "\x1e";
      for (const auto& original : source)
        if (original.content.find(marker) != std::string::npos)
          throw std::invalid_argument("image marker collision in chat text");
      content += marker;
      markers.push_back(std::move(marker));
      cursor = image.offset;
    }
    content.append(source[i].content, cursor);
    messages[i].content = std::move(content);
  }
  ImageChatRendered out;
  out.text = RenderChat(messages, tools, thinking);
  std::size_t cursor = 0;
  for (const auto& marker : markers) {
    const auto at = out.text.find(marker, cursor);
    if (at == std::string::npos)
      throw std::invalid_argument(
          "DiffusionGemma chat template did not render an image part");
    out.text.replace(at, marker.size(), gemma4::kImagePlaceholder);
    out.image_offsets.push_back(at);
    cursor = at + gemma4::kImagePlaceholder.size();
  }
  return out;
}

}  // namespace gufo::models::diffusion_gemma
