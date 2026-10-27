// Cyber-Tiel's own chat template, compiled (QwenTemplateDialect::kCyberTiel).
//
// The goldens in tests/fixtures/cybertiel_template/ are jinja2 renderings of
// the template stored in the Cyber-Tiel-Coder GGUF (cybertiel-template.jinja),
// written by the independent upstream template oracle;
// the tokenizer applies the template independently. Every case must render to the same bytes.

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
using gufo::tokenization::QwenToolFormat;
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
  return std::string(GUFO_CYBERTIEL_FIXTURES) + "/" + name;
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
    std::string_view name, std::string_view chat_template) {
  const auto binary = BuildGguf({{"general.name", std::string(name)},
                                 {"tokenizer.chat_template",
                                  std::string(chat_template)}});
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, "in-memory GGUF opens: " + error);
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

ChatTemplateOptions ToOptions(const json::Value& source) {
  ChatTemplateOptions options;
  options.dialect = QwenTemplateDialect::kCyberTiel;
  options.add_generation_prompt =
      OptionalBool(source, "add_generation_prompt", true);
  options.enable_thinking = OptionalBool(source, "enable_thinking", true);
  options.preserve_thinking = OptionalBool(source, "preserve_thinking", true);
  options.add_vision_id = OptionalBool(source, "add_vision_id", false);
  options.reasoning_effort = QwenReasoningEffort::kMedium;
  // Undefined in the template means XML; "json" is opt-in.
  options.tool_format = source.member_str("tool_call_format") == "json"
                            ? QwenToolFormat::kJson
                            : QwenToolFormat::kXml;
  // The effort strings the template reads.
  const std::string effort = source.member_str("reasoning_effort");
  if (effort == "none" || effort == "off") {
    options.enable_thinking = false;
  } else if (effort == "minimal" || effort == "low") {
    options.reasoning_effort = QwenReasoningEffort::kLow;
  } else if (effort == "high" || effort == "xhigh" || effort == "max") {
    options.reasoning_effort = QwenReasoningEffort::kXHigh;
  }
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
  const auto document =
      json::parse(ReadFile(FixturePath("cases.json")));
  const auto image = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  const auto& cases = document.find("cases")->items();
  Expect(cases.size() >= 12, "at least 12 cases");
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
}

// ---- selection ---------------------------------------------------------------

void TestSelectionByHash() {
  const std::string source = ReadFile(FixturePath("cybertiel-template.jinja"));
  Expect(Sha256(source) == QwenChatTemplate::CyberTielTemplateSha256(),
         "the recorded hash is the hash of the GGUF template");
  const auto document = json::parse(ReadFile(FixturePath("cases.json")));
  Expect(document.member_str("template_sha256") ==
             QwenChatTemplate::CyberTielTemplateSha256(),
         "cases.json records the same hash");

  // Not a Qwen3.8 artifact, as the real Cyber-Tiel GGUF (qwen35moe).
  auto selected = TemplateFromGguf("Huihui Ornith 1.5 35B A3B", source);
  Expect(selected != nullptr, "the Cyber-Tiel template loads");
  Expect(selected->GetProfile() == QwenChatTemplate::Profile::kCyberTiel,
         "the exact template selects the Cyber-Tiel dialect");
  Expect(selected->GetTemplateId() == "cybertiel-compiled-v1",
         "the dialect has a stable id");
  Expect(QwenChatTemplate::CreateDefault(source)->GetProfile() ==
             QwenChatTemplate::Profile::kCyberTiel,
         "CreateDefault classifies by the same hash");

  // One changed byte, anywhere, falls back to the existing formatter.
  auto edited = TemplateFromGguf("Huihui Ornith 1.5 35B A3B", source + "\n");
  Expect(edited != nullptr &&
             edited->GetProfile() == QwenChatTemplate::Profile::kLegacyChatMl,
         "a changed template does not select the Cyber-Tiel dialect");
  std::string version_changed = source;
  const auto at = version_changed.find("v22.5.0");
  Expect(at != std::string::npos, "template names its version");
  version_changed[at + 6] = '1';
  auto other = TemplateFromGguf("Huihui Ornith 1.5 35B A3B", version_changed);
  Expect(other != nullptr &&
             other->GetProfile() == QwenChatTemplate::Profile::kLegacyChatMl,
         "another template version does not select the Cyber-Tiel dialect");

  // The pinned Qwen3.8 template keeps its own profile, and the default
  // options never select the Cyber-Tiel dialect.
  std::string reference = ReadFile(GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE);
  if (reference.ends_with('\n')) {
    reference.pop_back();
  }
  auto official = TemplateFromGguf("Qwen3.8-27B", reference);
  Expect(official != nullptr && official->GetProfile() ==
                                    QwenChatTemplate::Profile::kQwen38Reasoning,
         "the Qwen3.8 template keeps its profile");
  Expect(ChatTemplateOptions{}.dialect == QwenTemplateDialect::kQwen,
         "default options use the existing renderer");
  Expect(ChatTemplateOptions{}.tool_format == QwenToolFormat::kXml,
         "the Cyber-Tiel dialect writes XML tool calls unless asked");
}

// ---- the existing renderer is unchanged ---------------------------------------

void TestExistingRendererUnchanged() {
  // Reference renderings of the pinned Qwen3.8 template (jinja2).
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
    ExpectSameBytes("qwen_reference_tools_medium", *rendered,
                    ReadFile(FixturePath("qwen_reference_tools_medium.txt")));
  }
  {
    std::vector<ChatMessage> messages{{ChatRole::kSystem, "S"},
                                      {ChatRole::kUser, "Hi"}};
    ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto rendered = QwenChatTemplate::Render(messages, options);
    Expect(rendered.has_value(), "existing renderer renders a chat");
    ExpectSameBytes("qwen_reference_system_nothink", *rendered,
                    ReadFile(FixturePath("qwen_reference_system_nothink.txt")));
  }
}

// ---- behaviour outside the golden cases -----------------------------------------

void TestServerFramingAndErrors() {
  ChatTemplateOptions options;
  options.dialect = QwenTemplateDialect::kCyberTiel;
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
  Expect(rendered->starts_with(
             "<|im_start|>system\nS\n\nX\n\nYou are Cyber-Tiel-Coder"),
         "suffix follows the system text, then the built-in text");
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

  // A message with an image cannot lose a think-control marker: refused.
  auto image = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  ChatMessage with_image{ChatRole::kUser, "look <|think_off|>"};
  with_image.images.push_back({0, image});
  messages = {with_image};
  rendered = QwenChatTemplate::Render(messages, {}, options, &error);
  Expect(!rendered.has_value() && !error.empty(),
         "markers beside an image are refused, not approximated");

  // The output bound applies.
  options.max_output_bytes = 64;
  messages = {{ChatRole::kUser, "Hi"}};
  rendered = QwenChatTemplate::Render(messages, {}, options, &error);
  Expect(!rendered.has_value(), "the output bound applies to the dialect");
}

}  // namespace

int main() {
  std::cout << "Running Cyber-Tiel chat template tests...\n";
  TestGoldens();
  TestSelectionByHash();
  TestExistingRendererUnchanged();
  TestServerFramingAndErrors();
  std::cout << "All Cyber-Tiel chat template tests passed\n";
  return 0;
}
