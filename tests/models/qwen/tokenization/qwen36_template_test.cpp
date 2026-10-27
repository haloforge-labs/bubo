// Qwen3.6-27B's own chat template, compiled (QwenTemplateDialect::kQwen36).
//
// The goldens in tests/fixtures/qwen36_template/ are jinja2 renderings of the
// template stored in the Qwen3.6-27B GGUF (qwen36-template.jinja), written by
// the independent upstream template oracle. Every
// case must render to the same bytes. The Qwen3.8 classification and renderer
// stay as they were.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"

namespace {

using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTemplateOptions;
using gufo::tokenization::ChatTool;
using gufo::tokenization::ContentSpan;
using gufo::tokenization::QwenChatTemplate;
using gufo::tokenization::QwenReasoningEffort;
using gufo::tokenization::QwenTemplateDialect;
namespace json = gufo::json;

void Expect(bool condition, const std::string& msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Expect(input.good(), "cannot open " + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

std::string FixturePath(const std::string& name) {
  return std::string(GUFO_QWEN36_FIXTURES) + "/" + name;
}

// A GGUF header that holds only string metadata.
std::vector<std::uint8_t> BuildGguf(
    const std::vector<std::pair<std::string, std::string>>& strings) {
  std::vector<std::uint8_t> buffer;
  const auto bytes = [&](const void* data, std::size_t size) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), p, p + size);
  };
  const auto u32 = [&](std::uint32_t value) { bytes(&value, sizeof(value)); };
  const auto u64 = [&](std::uint64_t value) { bytes(&value, sizeof(value)); };
  const auto text = [&](std::string_view value) {
    u64(value.size());
    bytes(value.data(), value.size());
  };
  bytes("GGUF", 4);
  u32(3);
  u64(0);
  u64(strings.size());
  for (const auto& [key, value] : strings) {
    text(key);
    u32(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    text(value);
  }
  while (buffer.size() % 32 != 0) {
    buffer.push_back(0);
  }
  return buffer;
}

std::unique_ptr<QwenChatTemplate> TemplateFromGguf(
    std::string_view name, std::string_view chat_template,
    bool* valid = nullptr) {
  const auto binary = BuildGguf({{"general.name", std::string(name)},
                                 {"tokenizer.chat_template",
                                  std::string(chat_template)}});
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, "in-memory GGUF opens: " + error);
  if (valid != nullptr) {
    *valid = QwenChatTemplate::ValidateGgufTemplate(*reader, &error);
  }
  return QwenChatTemplate::CreateFromGguf(*reader, &error);
}

// ---- cases.json -> engine inputs -------------------------------------------

bool OptionalBool(const json::Value& object, const char* key, bool fallback) {
  const json::Value* value = object.find(key);
  return value == nullptr || value->is_null() ? fallback : value->as_bool();
}

ChatMessage ToMessage(const json::Value& source,
                      const std::shared_ptr<const std::vector<std::uint8_t>>&
                          image) {
  ChatMessage message;
  const std::string role = source.member_str("role");
  message.role = role == "system"      ? ChatRole::kSystem
                 : role == "developer" ? ChatRole::kDeveloper
                 : role == "user"      ? ChatRole::kUser
                 : role == "assistant" ? ChatRole::kAssistant
                                       : ChatRole::kTool;
  const json::Value* content = source.find("content");
  if (content != nullptr && content->is_string()) {
    message.content = content->get_str();
  } else if (content != nullptr && content->is_array()) {
    for (const auto& part : content->items()) {
      if (part.member_str("type") == "image") {
        message.images.push_back({message.content.size(), image});
      } else {
        message.content += part.member_str("text");
      }
    }
  }
  message.thought = source.member_str("reasoning_content");
  if (const json::Value* calls = source.find("tool_calls")) {
    for (const auto& call : calls->items()) {
      const json::Value* function = call.find("function");
      Expect(function != nullptr, "tool call has a function");
      ChatMessage::ToolCall converted;
      converted.id = call.member_str("id");
      converted.name = function->member_str("name");
      // As the HTTP parser stores them: strings plain, the rest as tojson.
      for (const auto& [name, value] : function->find("arguments")->members()) {
        converted.arguments.push_back(
            {name, value.is_string() ? value.get_str() : value.tojson(),
             value.is_string()});
      }
      message.tool_calls.push_back(std::move(converted));
    }
  }
  return message;
}

ChatTool ToTool(const json::Value& source) {
  const json::Value* function = source.find("function");
  Expect(function != nullptr, "tool has a function");
  ChatTool tool;
  tool.name = function->member_str("name");
  tool.description = function->member_str("description");
  tool.parameters_json = function->find("parameters")->dump();
  tool.definition_json = source.dump();
  return tool;
}

// What the serving path builds from the request: preserve_thinking is off
// unless the request sets it, and the effort kwarg has no effect.
ChatTemplateOptions ToOptions(const json::Value& source) {
  gufo::ReasoningOptions reasoning;
  if (const json::Value* value = source.find("enable_thinking");
      value != nullptr && !value->is_null()) {
    reasoning.enabled = value->as_bool();
  }
  if (const json::Value* value = source.find("preserve_thinking");
      value != nullptr && !value->is_null()) {
    reasoning.preserve_thinking = value->as_bool();
  }
  auto options = gufo::tokenization::ResolveQwenChatOptions(
      reasoning, OptionalBool(source, "add_vision_id", false),
      QwenTemplateDialect::kQwen36);
  options.add_generation_prompt =
      OptionalBool(source, "add_generation_prompt", true);
  return options;
}

void ExpectSameBytes(const std::string& name, const std::string& got,
                     const std::string& want) {
  if (got == want) {
    return;
  }
  std::size_t at = 0;
  while (at < got.size() && at < want.size() && got[at] == want[at]) {
    ++at;
  }
  std::cerr << "case " << name << " differs at byte " << at << " (got "
            << got.size() << " bytes, want " << want.size() << ")\n  got:  "
            << got.substr(at > 40 ? at - 40 : 0, 100) << "\n  want: "
            << want.substr(at > 40 ? at - 40 : 0, 100) << "\n";
  Expect(false, "case " + name + " renders the golden bytes");
}

void TestGoldens() {
  const auto document = json::parse(ReadFile(FixturePath("cases.json")));
  const auto image = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  const auto& cases = document.find("cases")->items();
  Expect(cases.size() >= 40, "at least 40 cases");
  for (const auto& source : cases) {
    const std::string name = source.member_str("name");
    std::vector<ChatMessage> messages;
    for (const auto& message : source.find("messages")->items()) {
      messages.push_back(ToMessage(message, image));
    }
    std::vector<ChatTool> tools;
    if (const json::Value* list = source.find("tools"); list != nullptr) {
      for (const auto& tool : list->items()) {
        tools.push_back(ToTool(tool));
      }
    }
    const auto options = ToOptions(source);
    std::string error;
    std::vector<std::size_t> image_offsets;
    std::size_t stable = 0;
    std::vector<ContentSpan> spans;
    const auto rendered = QwenChatTemplate::Render(
        messages, tools, options, &error, &image_offsets, &stable, &spans);
    Expect(rendered.has_value(), "case " + name + " renders: " + error);

    const std::string golden = ReadFile(FixturePath(name + ".txt"));
    ExpectSameBytes(name, *rendered, golden);
    Expect(Sha256(golden) == source.member_str("output_sha256"),
           "golden file hash matches cases.json for " + name);

    // Side outputs the serving path relies on.
    std::size_t images_in_messages = 0;
    for (const auto& message : messages) {
      images_in_messages += message.images.size();
    }
    Expect(image_offsets.size() == images_in_messages,
           "one pad offset per image in " + name);
    for (const auto offset : image_offsets) {
      Expect(rendered->compare(offset, gufo::tokenization::kImagePad.size(),
                               gufo::tokenization::kImagePad) == 0,
             "image offset points at the pad in " + name);
    }
    Expect(stable <= rendered->size() &&
               (stable == rendered->size() ||
                rendered->compare(stable, gufo::tokenization::kImStart.size(),
                                  gufo::tokenization::kImStart) == 0),
           "stable prefix ends on a turn boundary in " + name);
    std::size_t previous_end = 0;
    for (const auto& span : spans) {
      Expect(span.offset >= previous_end &&
                 span.offset + span.size <= rendered->size(),
             "content spans are ordered and inside the prompt in " + name);
      previous_end = span.offset + span.size;
    }
  }
  std::cout << "  " << cases.size() << " golden cases match byte for byte\n";
}

// ---- selection ---------------------------------------------------------------

void TestSelectionByHash() {
  const std::string source = ReadFile(FixturePath("qwen36-template.jinja"));
  Expect(Sha256(source) == QwenChatTemplate::Qwen36TemplateSha256(),
         "the recorded hash is the hash of the GGUF template");
  const auto document = json::parse(ReadFile(FixturePath("cases.json")));
  Expect(document.member_str("template_sha256") ==
             QwenChatTemplate::Qwen36TemplateSha256(),
         "cases.json records the same hash");

  // A Qwen3.6-27B file has the dimensions of Qwen3.8-27B; its name keeps it
  // out of the Qwen3.8 pin.
  bool valid = false;
  auto selected = TemplateFromGguf("Qwen3.6-27B", source, &valid);
  Expect(selected != nullptr, "the Qwen3.6 template loads");
  Expect(valid, "a Qwen3.6-27B GGUF validates");
  Expect(selected->GetProfile() == QwenChatTemplate::Profile::kQwen36,
         "the exact template selects the Qwen3.6 dialect");
  Expect(selected->GetDialect() == QwenTemplateDialect::kQwen36,
         "the profile maps to the dialect");
  Expect(selected->GetTemplateId() == "qwen36-compiled-v1",
         "the dialect has a stable id");
  Expect(QwenChatTemplate::CreateDefault(source)->GetProfile() ==
             QwenChatTemplate::Profile::kQwen36,
         "CreateDefault classifies by the same hash");

  // One changed byte, anywhere, falls back to the existing formatter.
  auto edited = TemplateFromGguf("Qwen3.6-27B", source + "\n");
  Expect(edited != nullptr &&
             edited->GetProfile() == QwenChatTemplate::Profile::kLegacyChatMl,
         "a changed template does not select the Qwen3.6 dialect");

  // Qwen3.8 stays pinned: the Qwen3.6 template is not accepted for a file
  // that names itself Qwen3.8, and an unknown template is still refused.
  valid = true;
  auto strict = TemplateFromGguf("Qwen3.8-27B", source, &valid);
  Expect(strict == nullptr && !valid,
         "a Qwen3.8 file refuses the Qwen3.6 template");
  Expect(TemplateFromGguf("Qwen3.8-27B", source + "\n") == nullptr,
         "a Qwen3.8 file refuses an unknown template");

  // The pinned Qwen3.8 templates keep their profile and renderer.
  std::string reference = ReadFile(GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE);
  if (reference.ends_with('\n')) {
    reference.pop_back();
  }
  auto official = TemplateFromGguf("Qwen3.8-27B", reference);
  Expect(official != nullptr && official->GetProfile() ==
                                    QwenChatTemplate::Profile::kQwen38Reasoning,
         "the official Qwen3.8 template keeps its profile");
  Expect(official->GetDialect() == QwenTemplateDialect::kQwen,
         "the Qwen3.8 profile renders with the existing dialect");
  const std::string unsloth = ReadFile(FixturePath("qwen38-unsloth-template.jinja"));
  Expect(Sha256(unsloth) == QwenChatTemplate::UnslothArtifactTemplateSha256(),
         "the Unsloth Qwen3.8 fixture is the pinned artifact template");
  auto artifact = TemplateFromGguf("Qwen3.8-27B", unsloth);
  Expect(artifact != nullptr && artifact->GetProfile() ==
                                    QwenChatTemplate::Profile::kQwen38Reasoning,
         "the Unsloth Qwen3.8 template keeps its profile");
  Expect(artifact->GetTemplateId() == "qwen38-reasoning-compiled-v3",
         "the Qwen3.8 template id is unchanged");

  Expect(ChatTemplateOptions{}.dialect == QwenTemplateDialect::kQwen,
         "default options use the existing renderer");
}

void TestOptions() {
  gufo::ReasoningOptions none;
  const auto existing = gufo::tokenization::ResolveQwenChatOptions(none);
  Expect(existing.preserve_thinking && existing.dialect == QwenTemplateDialect::kQwen &&
             existing.reasoning_effort == QwenReasoningEffort::kXHigh,
         "Qwen3.8 defaults are unchanged");
  const auto q36 = gufo::tokenization::ResolveQwenChatOptions(
      none, false, QwenTemplateDialect::kQwen36);
  Expect(!q36.preserve_thinking && q36.enable_thinking &&
             q36.dialect == QwenTemplateDialect::kQwen36,
         "Qwen3.6 keeps earlier reasoning only on request");
  gufo::ReasoningOptions preserve;
  preserve.preserve_thinking = true;
  preserve.enabled = false;
  const auto asked = gufo::tokenization::ResolveQwenChatOptions(
      preserve, false, QwenTemplateDialect::kQwen36);
  Expect(asked.preserve_thinking && !asked.enable_thinking,
         "Qwen3.6 honours preserve_thinking and enable_thinking");
}

// ---- the existing renderer is unchanged ---------------------------------------

void TestExistingRendererUnchanged() {
  // Reference renderings of the pinned Qwen3.8 template (jinja2), written for
  // the Cyber-Tiel tests and shared here.
  {
    std::vector<ChatMessage> messages{{ChatRole::kUser, "Hi"}};
    ChatTool tool;
    tool.name = "get_weather";
    tool.definition_json =
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get the current weather for a city.\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{"
        "\"type\":\"string\",\"description\":\"City name, e.g. Montreal\"},"
        "\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\","
        "\"fahrenheit\"]}},\"required\":[\"city\"]}}}";
    ChatTemplateOptions options;
    options.reasoning_effort = QwenReasoningEffort::kMedium;
    std::string error;
    const auto rendered = QwenChatTemplate::Render(
        messages, std::span<const ChatTool>(&tool, 1), options, &error);
    Expect(rendered.has_value(), "existing renderer renders tools: " + error);
    ExpectSameBytes(
        "qwen_reference_tools_medium", *rendered,
        ReadFile(std::string(GUFO_CYBERTIEL_FIXTURES) +
                 "/qwen_reference_tools_medium.txt"));
  }
  {
    std::vector<ChatMessage> messages{{ChatRole::kSystem, "S"},
                                      {ChatRole::kUser, "Hi"}};
    ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto rendered = QwenChatTemplate::Render(messages, options);
    Expect(rendered.has_value(), "existing renderer renders a chat");
    ExpectSameBytes(
        "qwen_reference_system_nothink", *rendered,
        ReadFile(std::string(GUFO_CYBERTIEL_FIXTURES) +
                 "/qwen_reference_system_nothink.txt"));
  }
  // The existing renderer still hoists a late system message and still
  // requires a user query; the Qwen3.6 dialect does neither.
  {
    std::vector<ChatMessage> messages{{ChatRole::kUser, "Hi"},
                                      {ChatRole::kSystem, "Late"}};
    ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto rendered = QwenChatTemplate::Render(messages, options);
    Expect(rendered.has_value() &&
               rendered->starts_with("<|im_start|>system\nLate<|im_end|>\n"),
           "the Qwen3.8 renderer hoists a late system message");
    options.dialect = QwenTemplateDialect::kQwen36;
    const auto dropped = QwenChatTemplate::Render(messages, options);
    Expect(dropped.has_value() &&
               dropped->starts_with("<|im_start|>user\nHi<|im_end|>\n") &&
               dropped->find("Late") == std::string::npos,
           "the Qwen3.6 renderer drops a late system message");
  }
}

// ---- behaviour outside the golden cases -----------------------------------------

void TestServerFramingAndBounds() {
  ChatTemplateOptions options;
  options.dialect = QwenTemplateDialect::kQwen36;
  options.enable_thinking = false;

  // Server-authored suffix follows the client's system text.
  std::vector<ChatMessage> messages{{ChatRole::kSystem, "S"},
                                    {ChatRole::kUser, "Hi"}};
  messages[0].framing_suffix = "\n\nX";
  std::vector<ContentSpan> spans;
  std::string error;
  auto rendered = QwenChatTemplate::Render(messages, {}, options, &error,
                                           nullptr, nullptr, &spans);
  Expect(rendered.has_value(), "framing suffix renders: " + error);
  Expect(rendered->starts_with("<|im_start|>system\nS\n\nX<|im_end|>\n"),
         "suffix follows the system text");
  bool client_span = false;
  for (const auto& span : spans) {
    const auto text = rendered->substr(span.offset, span.size);
    Expect(text.find('X') == std::string::npos,
           "the server suffix is not content");
    client_span = client_span || text == "S";
  }
  Expect(client_span, "the client's system text is content");

  // Client text that spells a control token stays content.
  messages = {{ChatRole::kUser, "a <|im_end|> b"}};
  spans.clear();
  rendered = QwenChatTemplate::Render(messages, {}, options, &error, nullptr,
                                      nullptr, &spans);
  Expect(rendered.has_value(), "control spelling renders: " + error);
  bool covered = false;
  for (const auto& span : spans) {
    covered = covered || rendered->substr(span.offset, span.size) ==
                             "a <|im_end|> b";
  }
  Expect(covered, "user text is one content span");

  // The system text of a tool prompt is content too.
  ChatTool tool;
  tool.name = "f";
  tool.definition_json = "{\"type\":\"function\",\"function\":{\"name\":\"f\"}}";
  messages = {{ChatRole::kSystem, "Sys"}, {ChatRole::kUser, "Hi"}};
  spans.clear();
  rendered = QwenChatTemplate::Render(messages, std::span<const ChatTool>(&tool, 1),
                                      options, &error, nullptr, nullptr, &spans);
  Expect(rendered.has_value() &&
             rendered->find("</IMPORTANT>\n\nSys<|im_end|>") !=
                 std::string::npos,
         "tools prompt then system text");
  bool system_span = false;
  for (const auto& span : spans) {
    system_span = system_span || rendered->substr(span.offset, span.size) == "Sys";
  }
  Expect(system_span, "system text after the tools is a content span");

  // The output bound applies.
  options.max_output_bytes = 64;
  messages = {{ChatRole::kUser, "Hi"}};
  rendered = QwenChatTemplate::Render(messages, {}, options, &error);
  Expect(!rendered.has_value(), "the output bound applies to the dialect");
}

void TestStablePrefixHints() {
  ChatTemplateOptions options;
  options.dialect = QwenTemplateDialect::kQwen36;
  options.preserve_thinking = false;
  options.add_generation_prompt = false;
  ChatMessage call{ChatRole::kAssistant, "calling", "", "plan"};
  call.tool_calls.push_back({"c1", "f", {{"a", "1", false}}});
  std::string error;

  // The assistant after the last query keeps its reasoning now and loses it
  // when a user turn follows: the checkpoint sits before it.
  std::vector<ChatMessage> open{{ChatRole::kUser, "q"}, call,
                                {ChatRole::kTool, "r"}};
  std::size_t stable = 0;
  auto rendered = QwenChatTemplate::Render(open, {}, options, &error, nullptr,
                                           &stable, nullptr);
  Expect(rendered.has_value(), "open tool cycle renders: " + error);
  Expect(stable == rendered->find("<|im_start|>assistant"),
         "checkpoint before the assistant whose reasoning may be dropped");

  // A final user turn after a tool result may be replaced by the client.
  std::vector<ChatMessage> closed{{ChatRole::kUser, "q"}, call,
                                  {ChatRole::kTool, "r"},
                                  {ChatRole::kUser, "next"}};
  rendered = QwenChatTemplate::Render(closed, {}, options, &error, nullptr,
                                      &stable, nullptr);
  Expect(rendered.has_value(), "closed tool cycle renders: " + error);
  Expect(stable == rendered->find("<|im_start|>user\nnext"),
         "checkpoint before the final user turn");

  // Nothing mutable: the prompt is its own stable prefix.
  std::vector<ChatMessage> plain{{ChatRole::kUser, "q"}};
  rendered = QwenChatTemplate::Render(plain, {}, options, &error, nullptr,
                                      &stable, nullptr);
  Expect(rendered.has_value() && stable == rendered->size(),
         "a lone user turn is stable");
}

void TestEdgeInputs() {
  ChatTemplateOptions options;
  options.dialect = QwenTemplateDialect::kQwen36;
  options.preserve_thinking = false;
  std::string error;

  // The template raises on no messages.
  Expect(!QwenChatTemplate::Render({}, options, &error).has_value(),
         "no messages are refused");

  // No user query is not an error in this template.
  std::vector<ChatMessage> assistant_only{{ChatRole::kAssistant, "A", "", "r"}};
  auto rendered = QwenChatTemplate::Render(assistant_only, options, &error);
  Expect(rendered.has_value() &&
             rendered->starts_with("<|im_start|>assistant\nA<|im_end|>\n"),
         "a conversation without a user turn renders: " + error);

  // A system message that carries an image is refused, as the template does.
  auto image = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  ChatMessage with_image{ChatRole::kSystem, "look"};
  with_image.images.push_back({0, image});
  std::vector<ChatMessage> messages{with_image, {ChatRole::kUser, "Hi"}};
  rendered = QwenChatTemplate::Render(messages, options, &error);
  Expect(!rendered.has_value(), "images in system text are refused");
}

// Optional: the real Qwen3.6-27B GGUF header (set GUFO_QWEN36_GGUF). It has the
// dimensions of Qwen3.8-27B and the pinned template, so it must be accepted as
// Qwen3.6, not refused as Qwen3.8.
void TestRealGgufHeader() {
  const char* path = std::getenv("GUFO_QWEN36_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "  (GUFO_QWEN36_GGUF unset: real-file check skipped)\n";
    return;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenHeaderFile(path, &error);
  Expect(reader != nullptr, std::string("real GGUF opens: ") + error);
  const auto config = reader->ExtractModelConfig();
  Expect(config.has_value() && config->num_layers == 64 &&
             config->hidden_size == 5120 && config->vocab_size == 248320,
         "the real file has the Qwen3.8-27B dimensions");
  Expect(QwenChatTemplate::ValidateGgufTemplate(*reader, &error),
         "the real Qwen3.6-27B file validates: " + error);
  const auto chat_template = QwenChatTemplate::CreateFromGguf(*reader, &error);
  Expect(chat_template != nullptr &&
             chat_template->GetProfile() == QwenChatTemplate::Profile::kQwen36 &&
             chat_template->GetTemplateSha256() ==
                 QwenChatTemplate::Qwen36TemplateSha256(),
         "the real file selects the Qwen3.6 dialect by hash");
  std::cout << "  real GGUF header selects " << chat_template->GetTemplateId()
            << "\n";
}

}  // namespace

int main() {
  std::cout << "Running Qwen3.6 chat template tests...\n";
  TestGoldens();
  TestSelectionByHash();
  TestOptions();
  TestExistingRendererUnchanged();
  TestServerFramingAndBounds();
  TestStablePrefixHints();
  TestEdgeInputs();
  TestRealGgufHeader();
  std::cout << "All Qwen3.6 chat template tests passed\n";
  return 0;
}
