#include "src/core/image.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/diffusion_gemma_options.hpp"
#include "src/core/json.hpp"
#include "src/models/diffusion_gemma/cache_state.hpp"
#include "src/models/diffusion_gemma/image_chat.hpp"
#include "src/models/diffusion_gemma/image_prompt.hpp"
#include "src/models/diffusion_gemma/tokenizer.hpp"
#include "src/models/gemma4/vision/prompt.hpp"

namespace dg = gufo::models::diffusion_gemma;
namespace vision = gufo::models::gemma4::vision;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;

namespace {
int failures = 0;
void Expect(bool condition, const char* name) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", name);
    ++failures;
  }
}

template<typename Exception, typename Function>
void ExpectThrows(Function function, const char* name) {
  try {
    function();
    Expect(false, name);
  } catch (const Exception&) {
  }
}

std::filesystem::path MakeTokenizerFixture() {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("bubo-diffusion-gemma-image-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  auto document = gufo::json::Value::object();
  auto model = gufo::json::Value::object();
  model["type"] = "BPE";
  model["byte_fallback"] = true;
  auto vocab = gufo::json::Value::object();
  for (int byte = 0; byte < 256; ++byte) {
    char name[8];
    std::snprintf(name, sizeof(name), "<0x%02X>", byte);
    vocab[name] = byte;
  }
  model["vocab"] = std::move(vocab);
  model["merges"] = gufo::json::Value::array();
  document["model"] = std::move(model);
  auto added = gufo::json::Value::array();
  for (const auto& [text, id] :
       std::vector<std::pair<const char*, int>>{{"<bos>", 2},
                                                {"<|turn>", 105},
                                                {"<turn|>", 106},
                                                {"<|image>", 255999},
                                                {"<|image|>", 258880},
                                                {"<image|>", 258882}}) {
    auto token = gufo::json::Value::object();
    token["content"] = text;
    token["id"] = id;
    token["special"] = true;
    added.push_back(std::move(token));
  }
  document["added_tokens"] = std::move(added);
  std::ofstream(root / "tokenizer.json") << document.dump();
  return root;
}

std::shared_ptr<const std::vector<std::uint8_t>> Png() {
  return std::make_shared<const std::vector<std::uint8_t>>(
      gufo::core::ReadImageUrl(
          "data:image/png;base64,"
          "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYklEQVR4nO3PMQ0AIADA"
          "MEAD/jUiAREcDcmqYJtn7/GzpQNeNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWg"
          "NaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaBdCLsBmEpLi1UAAAAASUVORK5C"
          "YII="));
}

vision::Prompt Prepare(const dg::Tokenizer& tokenizer,
                       const std::vector<ChatMessage>& source,
                       std::string_view identity = "projector",
                       std::string* rendered_text = nullptr) {
  std::vector<dg::Message> messages;
  for (const auto& m : source) {
    dg::Message out;
    out.role = m.role == ChatRole::kAssistant ? "assistant" : "user";
    out.content = m.content;
    messages.push_back(std::move(out));
  }
  const auto rendered =
      dg::RenderImageChat(source, std::move(messages), {}, false);
  if (rendered_text)
    *rendered_text = rendered.text;
  return vision::PrepareRendered(
      rendered.text, rendered.image_offsets, source,
      [&](std::string_view text) { return tokenizer.Encode(text); },
      {255999, 258880, 258882}, identity, 4096);
}

void CheckPrompt(const dg::Tokenizer& tokenizer) {
  Expect(tokenizer.IsSpecial(255999) && tokenizer.IsSpecial(258880) && tokenizer.IsSpecial(258882) && tokenizer.IsSpecial(105) &&
             !tokenizer.IsSpecial(65),
         "added tokens beyond the vocabulary are special (the flag vector grows with the pieces)");
  const auto png = Png();
  std::vector<ChatMessage> one{{ChatRole::kUser, "look"}};
  one[0].images.push_back({4, png});
  std::string rendered;
  const auto prompt = Prepare(tokenizer, one, "projector", &rendered);
  Expect(prompt.images.size() == 1 && prompt.images[0].span.rows == 81,
         "one image expands to 81 native soft tokens");
  const auto span = prompt.images[0].span;
  Expect(prompt.tokens[span.offset - 1] == 255999 &&
             prompt.tokens[span.offset + span.rows] == 258882 &&
             std::all_of(prompt.tokens.begin() + span.offset,
                         prompt.tokens.begin() + span.offset + span.rows,
                         [](int id) { return id == 258880; }),
         "DiffusionGemma image token IDs and framing");
  Expect(
      prompt.tokens.size() == tokenizer.Encode(rendered).size() + span.rows + 1,
      "one-image token count includes both framing tokens");

  std::vector<ChatMessage> two{{ChatRole::kUser, "a b"}};
  two[0].images = {{1, png}, {3, png}};
  const auto twice = Prepare(tokenizer, two, "projector", &rendered);
  Expect(twice.images.size() == 2 &&
             twice.images[1].span.offset >
                 twice.images[0].span.offset + twice.images[0].span.rows &&
             twice.images[0].prefix_identity != twice.images[1].prefix_identity,
         "two images stay ordered with separate identities");
  Expect(twice.IdentityForPrefix(twice.images[0].span.offset).empty() &&
             twice.IdentityForPrefix(twice.tokens.size()).size() == 32,
         "prefix image identities follow token positions");
  Expect(twice.tokens.size() == tokenizer.Encode(rendered).size() +
                                    twice.images[0].span.rows +
                                    twice.images[1].span.rows + 2,
         "two-image token count includes both expanded spans");

  std::vector<ChatMessage> later{{ChatRole::kUser, "first"},
                                 {ChatRole::kAssistant, "answer"},
                                 {ChatRole::kUser, "second"}};
  later[2].images.push_back({0, png});
  const auto late = Prepare(tokenizer, later, "projector", &rendered);
  Expect(late.images.size() == 1 &&
             late.images[0].span.offset >
                 tokenizer.Encode("<bos><|turn>user\nfirst").size(),
         "later-turn image follows earlier turns");
  Expect(late.tokens.size() ==
             tokenizer.Encode(rendered).size() + late.images[0].span.rows + 1,
         "later-turn image token count");
  ExpectThrows<std::invalid_argument>(
      [&] { (void)Prepare(tokenizer, one, ""); },
      "image without projector identity is rejected");
}

dg::ImageSpan Span(std::uint32_t offset, std::uint32_t rows,
                   std::uint8_t identity = 1) {
  dg::ImageSpan out;
  out.offset = offset;
  out.rows = rows;
  out.prefix_identity[0] = identity;
  return out;
}

void CheckChunks() {
  const std::vector<dg::ImageSpan> spans{Span(7, 4), Span(16, 3, 2)};
  dg::ValidateImageSpans(24, 8, 24, spans);
  auto chunk = dg::PlanPromptChunk(24, 0, 8, 8, 24, spans);
  Expect(chunk.rows == 7 && chunk.image == -1,
         "chunk boundary moves before image");
  chunk = dg::PlanPromptChunk(24, 7, 8, 8, 24, spans);
  Expect(chunk.rows == 4 && chunk.image == 0, "image is a whole prefill pass");
  chunk = dg::PlanPromptChunk(24, 11, 8, 8, 24, spans);
  Expect(chunk.rows == 5 && chunk.image == -1,
         "text between images stops before the next image");
  chunk = dg::PlanPromptChunk(24, 16, 8, 8, 24, spans);
  Expect(chunk.rows == 3 && chunk.image == 1, "second image uses its own pass");
  chunk = dg::PlanPromptChunk(24, 16, 2, 8, 24, spans);
  Expect(chunk.rows == 0, "image waits when batch capacity is too small");
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImageSpans(24, 3, 24, spans); },
      "image larger than prefill chunk is rejected");
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImageSpans(24, 8, 20, spans); },
      "prompt larger than context is rejected");
  const std::vector<dg::ImageSpan> too_large{Span(16, 5)};
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImageSpans(24, 8, 20, too_large); },
      "image larger than context is rejected");
  ExpectThrows<std::logic_error>(
      [&] { (void)dg::PlanPromptChunk(24, 9, 8, 8, 24, spans); },
      "prefill cannot resume inside an image");
}

void CheckPromptAdmission() {
  std::vector<std::int32_t> tokens(24, 7);
  tokens[6] = 255999;
  std::fill(tokens.begin() + 7, tokens.begin() + 11, 258880);
  tokens[11] = 258882;
  const std::array<dg::ImageSpan, 1> images{Span(7, 4)};
  dg::ValidateImagePrompt(tokens, 8, 280, 256, images);
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImagePrompt(tokens, 3, 280, 256, images); },
      "image above prefill chunk fails before streaming");
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImagePrompt(tokens, 8, 10, 256, images); },
      "image above context fails before streaming");
  ExpectThrows<std::length_error>(
      [&] { dg::ValidateImagePrompt(tokens, 8, 279, 256, images); },
      "image prompt without canvas room fails before streaming");
  tokens[8] = 7;
  ExpectThrows<std::invalid_argument>(
      [&] { dg::ValidateImagePrompt(tokens, 8, 280, 256, images); },
      "invalid soft image token fails before streaming");
}

void CheckReuse() {
  dg::CacheState cache(8, 4);
  const std::vector<std::int32_t> tokens{1, 2, 3, 4, 258880, 258880, 258880, 5};
  (void)cache.Append(std::span(tokens).first(4));
  const auto image = Span(4, 3);
  (void)cache.Append(std::span(tokens).subspan(4, 3), std::span(&image, 1));
  (void)cache.Append(std::span(tokens).last(1));
  Expect(cache.ReusableLength(tokens, std::span(&image, 1)) == 8,
         "same image and text reuse whole prefix");
  const auto different = Span(4, 3, 9);
  Expect(cache.ReusableLength(tokens, std::span(&different, 1)) == 4,
         "different image stops reuse before image rows");
  Expect(cache.Reuse(tokens, std::span(&different, 1)) == 4,
         "cache truncates before changed image");

  dg::CacheState two_images(16, 4);
  const std::vector<std::int32_t> longer{
      1, 2, 3, 4, 258880, 258880, 5, 6, 7, 258880, 258880, 8, 9};
  const std::array<dg::ImageSpan, 2> matching{Span(4, 2), Span(9, 2, 2)};
  (void)two_images.Append(std::span(longer).first(4));
  (void)two_images.Append(std::span(longer).subspan(4, 2),
                          std::span(&matching[0], 1));
  (void)two_images.Append(std::span(longer).subspan(6, 3));
  (void)two_images.Append(std::span(longer).subspan(9, 2),
                          std::span(&matching[1], 1));
  (void)two_images.Append(std::span(longer).last(2));
  auto changed = matching;
  changed[1].prefix_identity[0] = 3;
  Expect(two_images.ReusableLength(longer, changed) == 9,
         "changed second image retains first image prefix");
}

void CheckServeOptions(const std::filesystem::path& root) {
  std::ofstream(root / "model.gguf").put('\0');
  std::ofstream(root / "mmproj-BF16.gguf").put('\0');
  Expect(std::filesystem::is_regular_file(root / "mmproj-BF16.gguf") &&
             !gufo::server::ShouldBindDiffusionGemmaVision({}, false),
         "stray adjacent sidecar is not bound without --mmproj");
  Expect(gufo::server::ShouldBindDiffusionGemmaVision(root / "mmproj-BF16.gguf",
                                                      false) &&
             !gufo::server::ShouldBindDiffusionGemmaVision(
                 root / "mmproj-BF16.gguf", true),
         "explicit sidecar binds unless vision is disabled");
  Expect(gufo::server::UnsupportedDiffusionGemmaOption({.mmproj = true}) ==
             nullptr,
         "DiffusionGemma accepts a vision sidecar");
  Expect(std::string_view(gufo::server::UnsupportedDiffusionGemmaOption(
             {.cache_disk = true})) == "--cache-disk",
         "other unsupported options remain rejected");
  ExpectThrows<std::invalid_argument>(
      [] { gufo::server::ValidateDiffusionGemmaImageRequest(true, false); },
      "backend rejects images without a projector");
  gufo::server::ValidateDiffusionGemmaImageRequest(true, true);
  ExpectThrows<std::runtime_error>(
      [] { gufo::server::ValidateDiffusionGemmaVisionReserve(1); },
      "bound projector requires Gemma 4 vision reserve");
  gufo::server::ValidateDiffusionGemmaVisionReserve(
      gufo::server::kDiffusionGemmaVisionReserveBytes);
}
}  // namespace

int main() {
  const auto root = MakeTokenizerFixture();
  try {
    const dg::Tokenizer tokenizer(root);
    CheckPrompt(tokenizer);
    CheckChunks();
    CheckPromptAdmission();
    CheckReuse();
    CheckServeOptions(root);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    ++failures;
  }
  std::filesystem::remove_all(root);
  if (failures == 0)
    std::puts("diffusion_gemma_image_test: OK");
  return failures == 0 ? 0 : 1;
}
