#include "src/models/qwen/chat_template.hpp"

#include <unicode/uchar.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

std::size_t RenderedPromptBoundBytes(std::uint32_t context_tokens) noexcept {
  constexpr std::size_t kFloor = 1024ULL * 1024ULL;
  return std::max(kFloor,
                  std::size_t{context_tokens} * kMaxRenderedBytesPerToken);
}

ChatTemplateOptions ResolveQwenChatOptions(const ReasoningOptions& reasoning,
                                           bool add_vision_id,
                                           QwenTemplateDialect dialect) {
  ChatTemplateOptions options;
  options.add_vision_id = add_vision_id;
  options.enable_thinking = reasoning.enabled.value_or(options.enable_thinking);
  if (dialect == QwenTemplateDialect::kQwen36) {
    // The Qwen3.6 template reads preserve_thinking only when it is defined and
    // true, and has no reasoning-effort line.
    options.dialect = dialect;
    options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
    options.reasoning_effort = QwenReasoningEffort::kMedium;
    return options;
  }
  options.preserve_thinking =
      reasoning.preserve_thinking.value_or(options.preserve_thinking);
  switch (reasoning.effort.value_or(ReasoningEffort::kXHigh)) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      options.reasoning_effort = QwenReasoningEffort::kLow;
      break;
    case ReasoningEffort::kMedium:
      options.reasoning_effort = QwenReasoningEffort::kMedium;
      break;
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      options.reasoning_effort = QwenReasoningEffort::kXHigh;
      break;
  }
  return options;
}

namespace {

constexpr std::string_view kDefaultChatmlTemplate =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

constexpr std::string_view kLowReasoningInstruction =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstruction =
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer.";

std::string TemplateSha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

QwenChatTemplate::Profile ClassifyTemplate(std::string_view value) {
  const std::string hash = TemplateSha256(value);
  if (hash == QwenChatTemplate::OfficialTemplateSha256() ||
      hash == QwenChatTemplate::UnslothArtifactTemplateSha256()) {
    return QwenChatTemplate::Profile::kQwen38Reasoning;
  }
  if (hash == QwenChatTemplate::CyberTielTemplateSha256()) {
    return QwenChatTemplate::Profile::kCyberTiel;
  }
  if (hash == QwenChatTemplate::Qwen36TemplateSha256()) {
    return QwenChatTemplate::Profile::kQwen36;
  }
  return QwenChatTemplate::Profile::kLegacyChatMl;
}

// Qwen3.6-27B has the same dimensions as Qwen3.8-27B (64 layers, hidden 5120,
// vocabulary 248320), so the dimensions alone do not tell the two apart. A
// file whose general.name says Qwen3.6 is not a Qwen3.8 artifact.
bool IsQwen38Artifact(const core::GgufReader& reader) {
  const auto name = reader.GetMetadataString("general.name");
  if (name.has_value() && name->find("Qwen3.8") != std::string_view::npos) {
    return true;
  }
  if (name.has_value() && name->find("Qwen3.6") != std::string_view::npos) {
    return false;
  }
  const auto config = reader.ExtractModelConfig();
  return config.has_value() && config->num_layers == 64 &&
         config->hidden_size == 5120 && config->vocab_size == 248320;
}

std::string_view Trim(std::string_view value) {
  const auto whitespace = [](UChar32 cp) {
    // Python str.strip also includes the four ASCII information separators.
    return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
  };
  std::int32_t first = 0;
  auto last = static_cast<std::int32_t>(value.size());
  while (first < last) {
    auto next = first;
    UChar32 cp;
    U8_NEXT(value.data(), next, last, cp);
    if (!whitespace(cp))
      break;
    first = next;
  }
  while (first < last) {
    auto previous = last;
    UChar32 cp;
    U8_PREV(value.data(), first, previous, cp);
    if (!whitespace(cp))
      break;
    last = previous;
  }
  return value.substr(first, last - first);
}

std::string PythonJsonSpacing(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    output.push_back(character);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

void AppendReasoningInstruction(std::string& output,
                                const ChatTemplateOptions& options) {
  if (!options.enable_thinking) {
    return;
  }
  switch (options.reasoning_effort) {
    case QwenReasoningEffort::kLow:
      output.append(kLowReasoningInstruction);
      return;
    case QwenReasoningEffort::kMedium:
      return;
    case QwenReasoningEffort::kXHigh:
      output.append(kXHighReasoningInstruction);
      return;
  }
}

// Message text is text. Content spans are tokenized without special matching,
// so a vocabulary token a client sends — or the model itself wrote — reaches
// the model as the characters it contains, never as a control token that ends
// a turn in the middle of a conversation (#383).
void AppendContent(std::string& output, std::vector<ContentSpan>* spans,
                   std::string_view text) {
  const auto begin = output.size();
  output.append(text);
  if (spans != nullptr && !text.empty()) {
    spans->push_back({begin, output.size() - begin});
  }
}

// Keep images inside their owning message, including grouped tool results.
// Only the inserted vision markers are framing; surrounding text stays literal.
bool AppendMessageContent(std::string& output, const ChatMessage& message,
                          const ChatTemplateOptions& options,
                          std::size_t& image_count,
                          std::vector<std::size_t>* image_offsets,
                          std::vector<ContentSpan>* content_spans) {
  if (message.images.empty()) {
    const auto content = Trim(message.content);
    AppendContent(output, content_spans, content);
    return !content.empty();
  }
  std::string rendered;
  std::vector<ContentSpan> spans;
  std::vector<std::size_t> offsets;
  std::size_t cursor = 0;
  for (const auto& image : message.images) {
    AppendContent(rendered, &spans,
                  std::string_view(message.content)
                      .substr(cursor, image.offset - cursor));
    ++image_count;
    if (options.add_vision_id)
      rendered.append("Picture " + std::to_string(image_count) + ": ");
    rendered.append(kVisionStart);
    offsets.push_back(rendered.size());
    rendered.append(kImagePad).append(kVisionEnd);
    cursor = image.offset;
  }
  AppendContent(rendered, &spans,
                std::string_view(message.content).substr(cursor));
  const auto content = Trim(rendered);
  const auto removed =
      static_cast<std::size_t>(content.data() - rendered.data());
  const auto start = output.size();
  if (image_offsets != nullptr)
    for (const auto offset : offsets)
      image_offsets->push_back(start + offset - removed);
  if (content_spans != nullptr) {
    for (const auto& span : spans) {
      const auto begin = std::max(span.offset, removed);
      const auto end =
          std::min(span.offset + span.size, removed + content.size());
      if (end > begin)
        content_spans->push_back({start + begin - removed, end - begin});
    }
  }
  output.append(content);
  return !content.empty();
}

void AppendJsonString(std::string& output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        output.push_back(static_cast<char>(character));
        break;
    }
  }
  output.push_back('"');
}

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools,
                       bool require_tool_call) {
  if (tools.empty()) {
    return;
  }

  if (!output.empty()) {
    output.append("\n\n");
  }
  output.append(
      "# Tools\n\nYou have access to the following functions:\n\n<tools>");
  for (const auto& tool : tools) {
    if (!tool.definition_json.empty()) {
      output.push_back('\n');
      output.append(PythonJsonSpacing(tool.definition_json));
      continue;
    }
    output.append("\n{\"type\": \"function\", \"function\": {\"name\": ");
    AppendJsonString(output, tool.name);
    output.append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output.append(", \"parameters\": ");
    output.append(tool.parameters_json.empty()
                      ? "{}"
                      : PythonJsonSpacing(tool.parameters_json));
    output.append("}}");
  }
  output.append(
      "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
      "following format with NO suffix:\n\n<tool_call>\n"
      "<function=example_function_name>\n<parameter=example_parameter_1>\n"
      "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
      "This is the value for the second parameter\nthat can span\nmultiple "
      "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
      "Reminder:\n- Function calls MUST follow the specified format: an inner "
      "<function=...></function> block must be nested within "
      "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
      "specified\n- You may provide optional reasoning for your function call "
      "in natural language BEFORE the function call, but NOT after\n- If "
      "there is no function call available, answer the question like normal "
      "with your current knowledge and do not tell the user about function "
      "calls\n</IMPORTANT>");
  if (require_tool_call) {
    output.append(
        "\n\nYou must call at least one available function. Do not answer the "
        "user directly.");
  }
}

void AppendToolCalls(std::string& output,
                     std::span<const ChatMessage::ToolCall> calls,
                     std::vector<ContentSpan>* content_spans) {
  bool first_call = true;
  for (const auto& call : calls) {
    if (!first_call) {
      output.push_back('\n');
    }
    first_call = false;
    output.append("<tool_call>\n<function=");
    output.append(call.name);
    output.append(">\n");
    for (const auto& argument : call.arguments) {
      output.append("<parameter=");
      output.append(argument.name);
      output.append(">\n");
      AppendContent(output, content_spans, argument.value);
      output.append("\n</parameter>\n");
    }
    output.append("</function>\n</tool_call>");
  }
}

// ---- Cyber-Tiel dialect -----------------------------------------------------
// Compiled form of the chat template inside the Cyber-Tiel-Coder GGUFs
// (tokenizer.chat_template, template_version qwen3.8-froggeric-v22.5.0). The
// default is the template with its kwargs undefined, as stock llama.cpp
// serves it: XML tool calls, the built-in system text on, and no length caps.
// ChatTemplateOptions::tool_format = kJson is the template's tool_call_format
// "json". tests/fixtures/cybertiel_template holds jinja2 renderings of the
// original template; qwen_cybertiel_template_test compares them byte for byte.

constexpr std::array<std::string_view, 10> kThinkMarkers{
    "<|think_off|>",       "<|think_on|>",        "<|think_xhigh|>",
    "<|think_high|>",      "<|think_ultracode|>", "<|think_extreme|>",
    "<|think_max|>",       "<|think_medium|>",    "<|think_low|>",
    "<|think_minimal|>",
};

constexpr std::string_view kCyberTielLeadThinking =
    "You are Cyber-Tiel-Coder, a variant of Ornith-1.5-35B-A3B. Answer "
    "directly, after thinking. Lead with the answer, then only what it needs "
    "to be correct and usable.";

constexpr std::string_view kCyberTielLeadPlain =
    "You are Cyber-Tiel-Coder, a variant of Ornith-1.5-35B-A3B. Answer "
    "directly and concisely. Give the answer with only what it needs to be "
    "correct and usable.";

constexpr std::string_view kCyberTielCore =
    "Never: open with preamble or pleasantries; restate the question; add "
    "filler transitions; hedge with niceties; or repeat a point you've "
    "already made.\n"
    "Always: keep essential steps, caveats, uncertainties, and specifics — "
    "never drop correctness or a needed warning for brevity. Keep the final "
    "answer lean. Use the least structure that conveys it (plain prose when "
    "short; lists or code only when they earn their place). If genuinely "
    "uncertain, say so and explain why — never omit uncertainty for the sake "
    "of brevity.\n"
    "If a user request is genuinely ambiguous, ask a sharp question, don't "
    "guess.";

constexpr std::string_view kCyberTielToolsIntro =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

constexpr std::string_view kCyberTielToolFormat =
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n";

constexpr std::string_view kCyberTielExampleThought =
    "<think>\nBrief explanation of tool call\n</think>\n";

constexpr std::string_view kCyberTielExampleCallXml =
    "<tool_call>\n<function=example_function_name>\n<parameter=example_"
    "parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n";

constexpr std::string_view kCyberTielExampleCallJson =
    "<tool_call>\n{\"name\": \"example_function_name\", \"arguments\": "
    "{\"example_parameter_1\": \"value_1\", \"example_parameter_2\": \"This "
    "is the value for the second parameter\"}}\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n";

constexpr std::string_view kCyberTielReminderThinking =
    "- You can use the <think></think> block to plan your next tool call OR "
    "to synthesize data and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the "
    "<think></think> block.\n";

constexpr std::string_view kCyberTielReminderFormatXml =
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags.\n";

constexpr std::string_view kCyberTielReminderFormatJson =
    "- Function calls MUST follow the specified format: a single JSON object "
    "with \"name\" and \"arguments\" keys inside <tool_call></tool_call> XML "
    "tags.\n";

constexpr std::string_view kCyberTielImmediateThinking =
    "- If you choose to call a tool, you MUST output the <tool_call> block "
    "IMMEDIATELY after thinking, with NO conversational text before it.\n";

constexpr std::string_view kCyberTielImmediatePlain =
    "- If you choose to call a tool, you MUST output the <tool_call> block "
    "IMMEDIATELY, with NO conversational text before it.\n";

constexpr std::string_view kCyberTielReminderEndXml =
    "- The <tool_call> and <function> tags MUST be at the very beginning of "
    "a new line, with NO spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest "
    "<tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to "
    "the user without any tool call.\n</IMPORTANT>";

constexpr std::string_view kCyberTielReminderEndJson =
    "- The <tool_call> tag MUST be at the very beginning of a new line, with "
    "NO spaces or indentation before it.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest "
    "<tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to "
    "the user without any tool call.\n</IMPORTANT>";

constexpr std::string_view kCyberTielOneFailure =
    "\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. "
    "Diagnose the failure and retry with completely corrected arguments.";

constexpr std::string_view kCyberTielManyFailuresHead =
    "\n\n⚠️ SYSTEM WARNING: ";

constexpr std::string_view kCyberTielManyFailuresTail =
    " consecutive tool errors detected. Your previous approach is incorrect. "
    "You MUST use a fundamentally different approach or corrected arguments.";

bool Contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

bool ContainsAny(std::string_view text,
                 std::initializer_list<std::string_view> needles) {
  return std::any_of(needles.begin(), needles.end(),
                     [&](std::string_view needle) {
                       return Contains(text, needle);
                     });
}

// The template's split(marker) | join('') | trim for every think-control
// marker, in the template's order.
std::string StripThinkMarkers(std::string_view text) {
  std::string result(text);
  for (const auto marker : kThinkMarkers) {
    if (!Contains(result, marker)) {
      continue;
    }
    std::string removed;
    removed.reserve(result.size());
    std::size_t cursor = 0;
    for (auto at = result.find(marker); at != std::string::npos;
         at = result.find(marker, cursor)) {
      removed.append(result, cursor, at - cursor);
      cursor = at + marker.size();
    }
    removed.append(result, cursor, std::string::npos);
    result = std::string(Trim(removed));
  }
  return result;
}

// The template's if/elif chain over the markers of one message.
void ApplyThinkMarkers(std::string_view text, bool& thinking,
                       QwenReasoningEffort& effort) {
  if (Contains(text, "<|think_off|>")) {
    thinking = false;
  } else if (Contains(text, "<|think_on|>")) {
    thinking = true;
  } else if (ContainsAny(text, {"<|think_xhigh|>", "<|think_high|>",
                                "<|think_ultracode|>", "<|think_extreme|>",
                                "<|think_max|>"})) {
    thinking = true;
    effort = QwenReasoningEffort::kXHigh;
  } else if (ContainsAny(text, {"<|think_low|>", "<|think_minimal|>"})) {
    thinking = true;
    effort = QwenReasoningEffort::kLow;
  } else if (Contains(text, "<|think_medium|>")) {
    thinking = true;
    effort = QwenReasoningEffort::kMedium;
  }
}

std::string_view LstripNewlines(std::string_view text) {
  const auto first = text.find_first_not_of('\n');
  return first == std::string_view::npos ? std::string_view{}
                                         : text.substr(first);
}

std::string_view RstripNewlines(std::string_view text) {
  const auto last = text.find_last_not_of('\n');
  return last == std::string_view::npos ? std::string_view{}
                                        : text.substr(0, last + 1);
}

// Python's text.split(separator)[-1] and text.split(separator)[0]; the caller
// has checked that the separator occurs.
std::string_view AfterLast(std::string_view text, std::string_view separator) {
  return text.substr(text.rfind(separator) + separator.size());
}

std::string_view BeforeFirst(std::string_view text,
                             std::string_view separator) {
  return text.substr(0, text.find(separator));
}

// json.dumps(ensure_ascii=False) of one string.
std::string JsonString(std::string_view value) {
  std::string output;
  output.reserve(value.size() + 2);
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        if (character < 0x20) {
          constexpr std::string_view kHex = "0123456789abcdef";
          output.append("\\u00");
          output.push_back(kHex[character >> 4]);
          output.push_back(kHex[character & 0x0f]);
        } else {
          output.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  output.push_back('"');
  return output;
}

// Valid JSON in the spelling of the template's tojson: no whitespace outside
// strings, then ": " after a colon and ", " after a comma.
std::string NormalizeJson(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    if (in_string) {
      output.push_back(character);
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == ' ' || character == '\t' || character == '\r' ||
        character == '\n') {
      continue;
    }
    output.push_back(character);
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

// The first `limit` code points of `value`.
std::string_view FirstCodePoints(std::string_view value, std::size_t limit) {
  std::int32_t index = 0;
  const auto length = static_cast<std::int32_t>(value.size());
  for (std::size_t seen = 0; seen < limit && index < length; ++seen) {
    U8_FWD_1(value.data(), index, length);
  }
  return value.substr(0, static_cast<std::size_t>(index));
}

std::size_t CodePointCount(std::string_view value) {
  std::int32_t index = 0;
  const auto length = static_cast<std::int32_t>(value.size());
  std::size_t count = 0;
  while (index < length) {
    U8_FWD_1(value.data(), index, length);
    ++count;
  }
  return count;
}

std::string AsciiLower(std::string_view value) {
  std::string lowered(value);
  for (auto& character : lowered) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  return lowered;
}

// Python's str.lower(): full Unicode case mapping, root locale. Text that is
// not valid UTF-8 (Python would have refused it) gets ASCII lowering.
std::string LowerCase(std::string_view value) {
  UErrorCode status = U_ZERO_ERROR;
  std::vector<UChar> wide(value.size() + 1);
  std::int32_t wide_length = 0;
  u_strFromUTF8(wide.data(), static_cast<std::int32_t>(wide.size()),
                &wide_length, value.data(),
                static_cast<std::int32_t>(value.size()), &status);
  if (U_FAILURE(status)) {
    return AsciiLower(value);
  }
  std::vector<UChar> lowered(static_cast<std::size_t>(wide_length) * 3 + 8);
  const auto lowered_length = u_strToLower(
      lowered.data(), static_cast<std::int32_t>(lowered.size()), wide.data(),
      wide_length, "", &status);
  if (U_FAILURE(status)) {
    return AsciiLower(value);
  }
  std::string narrow(static_cast<std::size_t>(lowered_length) * 3 + 8, '\0');
  std::int32_t narrow_length = 0;
  u_strToUTF8(narrow.data(), static_cast<std::int32_t>(narrow.size()),
              &narrow_length, lowered.data(), lowered_length, &status);
  if (U_FAILURE(status)) {
    return AsciiLower(value);
  }
  narrow.resize(static_cast<std::size_t>(narrow_length));
  return narrow;
}

// The template's running count of consecutive failing tool results after one
// more result `content`.
std::size_t ToolFailureCount(std::string_view content, std::size_t failures) {
  const std::string lowered = LowerCase(content);
  const std::string_view lower = lowered;
  const std::string_view head = FirstCodePoints(lower, 120);
  const bool code_or_grep =
      ContainsAny(lower, {"throw new ", "throw error", "console.error",
                          "logger.error", "logging.error"}) ||
      ContainsAny(head, {"import ", "def ", "function "});
  const bool exit_code_zero =
      ContainsAny(head, {"exit code: 0", "process exited with code 0"});
  const bool error_field_ok =
      ContainsAny(head, {"\"error\": null", "\"error\":null",
                         "\"error\": false", "\"error\":false",
                         "\"error\": \"\"", "\"error\":\"\""});
  const bool strong_error =
      (Contains(head, "\"error\":") && !error_field_ok) ||
      ContainsAny(head, {"\"status\": \"error\"", "\"status\":\"error\"",
                         "traceback (most recent call last):",
                         "command not found", "invalid syntax", "fatal:"}) ||
      (ContainsAny(head, {"exit code: ", "process exited with code"}) &&
       !exit_code_zero) ||
      head.starts_with("exception:") || head.starts_with("failed to ");
  const bool weak_error = ContainsAny(head, {"error:", "err!"});
  const bool weak_suppressed =
      ContainsAny(head, {"$ ", "took "}) || CodePointCount(content) >= 600;
  if (!code_or_grep && (strong_error || (weak_error && !weak_suppressed))) {
    return failures + 1;
  }
  return 0;
}

// A message's rendered content with the image placeholders it holds, as the
// template's render_content | trim writes it.
struct Piece {
  std::string text;
  std::vector<std::size_t> images;  ///< Offsets of the image pads in `text`.
  std::vector<ContentSpan> spans;
};

Piece MakePiece(const ChatMessage& message, const ChatTemplateOptions& options,
                std::size_t& image_count) {
  Piece piece;
  AppendMessageContent(piece.text, message, options, image_count,
                       &piece.images, &piece.spans);
  return piece;
}

// Replaces the text of a piece. The whole new text is client text; image
// offsets cannot follow a rewrite, so a piece with images is refused.
bool Rewrite(Piece& piece, std::string text, std::string* error_msg) {
  if (text == piece.text) {
    return true;
  }
  if (!piece.images.empty()) {
    if (error_msg != nullptr) {
      *error_msg =
          "think-control markers cannot be removed from a message with images";
    }
    return false;
  }
  piece.text = std::move(text);
  piece.spans.clear();
  if (!piece.text.empty()) {
    piece.spans.push_back({0, piece.text.size()});
  }
  return true;
}

void AppendPiece(std::string& output, const Piece& piece,
                 std::vector<std::size_t>* image_offsets,
                 std::vector<ContentSpan>* content_spans) {
  const auto start = output.size();
  output.append(piece.text);
  if (image_offsets != nullptr) {
    for (const auto offset : piece.images) {
      image_offsets->push_back(start + offset);
    }
  }
  if (content_spans != nullptr) {
    for (const auto& span : piece.spans) {
      content_spans->push_back({start + span.offset, span.size});
    }
  }
}

// tojson of a tool definition. Complete HTTP definitions keep their field
// order; the rest are built in the OpenAI shape.
std::string ToolDefinitionJson(const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    return NormalizeJson(tool.definition_json);
  }
  std::string json = "{\"type\": \"function\", \"function\": {\"name\": ";
  json.append(JsonString(tool.name));
  json.append(", \"description\": ");
  json.append(JsonString(tool.description));
  json.append(", \"parameters\": ");
  json.append(tool.parameters_json.empty()
                  ? std::string("{}")
                  : NormalizeJson(tool.parameters_json));
  json.append("}}");
  return json;
}

// tojson of a call's arguments mapping.
std::string ToolArgumentsJson(const ChatMessage::ToolCall& call) {
  std::string json = "{";
  bool first = true;
  for (const auto& argument : call.arguments) {
    if (!first) {
      json.append(", ");
    }
    first = false;
    json.append(JsonString(argument.name));
    json.append(": ");
    json.append(argument.is_string ? JsonString(argument.value)
                                   : NormalizeJson(argument.value));
  }
  json.push_back('}');
  return json;
}

std::optional<std::string> RenderCyberTiel(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets, std::size_t* stable_prefix_bytes,
    std::vector<ContentSpan>* content_spans) {
  const auto fail = [&](std::string message) -> std::optional<std::string> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return std::nullopt;
  };
  const auto is_system = [](const ChatMessage& message) {
    return message.role == ChatRole::kSystem ||
           message.role == ChatRole::kDeveloper;
  };

  const bool json_calls = options.tool_format == QwenToolFormat::kJson;

  // Think-control markers in any system, developer or user message set the
  // thinking state of the whole prompt, later messages last.
  bool thinking = options.enable_thinking;
  QwenReasoningEffort effort = options.reasoning_effort;
  for (const auto& message : messages) {
    if (is_system(message) || message.role == ChatRole::kUser) {
      ApplyThinkMarkers(message.content, thinking, effort);
    }
  }
  std::string_view reasoning_instruction;
  if (thinking && effort == QwenReasoningEffort::kXHigh) {
    reasoning_instruction = kXHighReasoningInstruction;
  } else if (thinking && effort == QwenReasoningEffort::kLow) {
    reasoning_instruction = kLowReasoningInstruction;
  }

  // Leading system and developer messages become one system text.
  std::size_t head = 0;
  while (head < messages.size() && is_system(messages[head])) {
    ++head;
  }
  std::string system_content;
  std::vector<ContentSpan> system_spans;
  for (std::size_t index = 0; index < head; ++index) {
    const auto& message = messages[index];
    const std::string client = StripThinkMarkers(Trim(message.content));
    const std::string untrimmed = client + message.framing_suffix;
    const auto part = Trim(untrimmed);
    if (part.empty()) {
      continue;
    }
    if (!system_content.empty()) {
      system_content.append("\n\n");
    }
    const auto begin =
        static_cast<std::size_t>(part.data() - untrimmed.data());
    const auto client_end = std::min(client.size(), begin + part.size());
    const auto client_size = client_end > begin ? client_end - begin : 0;
    AppendContent(system_content, &system_spans, part.substr(0, client_size));
    system_content.append(part.substr(client_size));
  }
  // The built-in text always follows the client's system text.
  std::string system_text = system_content;
  if (!system_text.empty()) {
    system_text.append("\n\n");
  }
  system_text.append(thinking ? kCyberTielLeadThinking : kCyberTielLeadPlain);
  system_text.push_back('\n');
  system_text.append(kCyberTielCore);

  std::string output;
  output.append(kImStart).append("system\n");
  if (!reasoning_instruction.empty()) {
    output.append(reasoning_instruction).append("\n\n");
  }
  // Runtimes that bring their own tool protocol say so in the system text.
  if (!tools.empty() && !Contains(system_text, "[TOOL_REQUEST]")) {
    output.append(kCyberTielToolsIntro);
    for (const auto& tool : tools) {
      output.push_back('\n');
      AppendContent(output, content_spans, ToolDefinitionJson(tool));
    }
    output.append("\n</tools>");
    output.append(kCyberTielToolFormat);
    if (thinking) {
      output.append(kCyberTielExampleThought);
    }
    output.append(json_calls ? kCyberTielExampleCallJson
                             : kCyberTielExampleCallXml);
    if (thinking) {
      output.append(kCyberTielReminderThinking);
    }
    output.append(json_calls ? kCyberTielReminderFormatJson
                             : kCyberTielReminderFormatXml);
    output.append(thinking ? kCyberTielImmediateThinking
                           : kCyberTielImmediatePlain);
    output.append(json_calls ? kCyberTielReminderEndJson
                             : kCyberTielReminderEndXml);
    output.append("\n\n");
  }
  const auto system_text_start = output.size();
  output.append(system_text);
  if (content_spans != nullptr) {
    for (const auto& span : system_spans) {
      content_spans->push_back({system_text_start + span.offset, span.size});
    }
  }
  output.append(kImEnd).append("\n");

  // The last user message that is not a tool-result wrapper is the query.
  // Reasoning of assistant turns before it is dropped unless preserved.
  const auto turns = messages.subspan(head);
  bool query_found = false;
  std::size_t last_query = 0;
  for (std::size_t index = turns.size(); index > 0; --index) {
    if (turns[index - 1].role != ChatRole::kUser) {
      continue;
    }
    std::size_t scratch_images = 0;
    const Piece scan = MakePiece(turns[index - 1], options, scratch_images);
    if (!(scan.text.starts_with("<tool_response>") &&
          scan.text.ends_with("</tool_response>"))) {
      query_found = true;
      last_query = index - 1;
      break;
    }
  }
  if (!query_found) {
    last_query = turns.size() > 51 ? turns.size() - 1 : 0;
  }

  // Checkpoint hints, as in the Qwen renderer: the prompt up to these offsets
  // does not change when the conversation grows by a new user turn.
  const bool final_user_turn =
      query_found && last_query + 1 == turns.size() &&
      turns.back().images.empty() && turns.size() >= 2 &&
      (turns[turns.size() - 2].role == ChatRole::kTool ||
       turns[turns.size() - 2].role == ChatRole::kUser);
  std::optional<std::size_t> mutable_reasoning;
  std::optional<std::size_t> final_user_start;
  bool seen_assistant = false;

  std::size_t image_count = 0;
  std::size_t failures = 0;
  ChatRole previous_role = ChatRole::kSystem;
  bool has_previous = false;
  for (std::size_t index = 0; index < turns.size(); ++index) {
    const auto& message = turns[index];
    Piece piece = MakePiece(message, options, image_count);
    const bool system_turn = is_system(message);
    if (system_turn || message.role == ChatRole::kUser) {
      std::string stripped = StripThinkMarkers(piece.text);
      if (!Rewrite(piece, std::move(stripped), error_msg)) {
        return std::nullopt;
      }
    }
    if (message.role == ChatRole::kAssistant) {
      if (!options.preserve_thinking && index > last_query &&
          !mutable_reasoning) {
        mutable_reasoning = output.size();
      }
      seen_assistant = true;
    } else if (message.role == ChatRole::kUser && final_user_turn &&
               seen_assistant && index == last_query) {
      final_user_start = output.size();
    }

    if (system_turn) {
      output.append(kImStart).append("system\n");
      AppendPiece(output, piece, image_offsets, content_spans);
      output.append(kImEnd).append("\n");
    } else if (message.role == ChatRole::kUser) {
      failures = 0;
      output.append(kImStart).append("user\n");
      AppendPiece(output, piece, image_offsets, content_spans);
      output.append(kImEnd).append("\n");
    } else if (message.role == ChatRole::kAssistant) {
      std::string_view content = piece.text;
      std::string_view reasoning;
      if (!message.thought.empty()) {
        // Explicit reasoning wins; a leading think block in the content is
        // dropped.
        std::string_view lead_end;
        if (content.starts_with("<think>") && Contains(content, "</think>")) {
          lead_end = "</think>";
        } else if (content.starts_with("<thinking>") &&
                   Contains(content, "</thinking>")) {
          lead_end = "</thinking>";
        } else if (content.starts_with("</think>")) {
          lead_end = "</think>";
        } else if (content.starts_with("</thinking>")) {
          lead_end = "</thinking>";
        }
        if (!lead_end.empty()) {
          content = LstripNewlines(AfterLast(content, lead_end));
        }
        reasoning = message.thought;
      } else {
        // Reasoning inside the content, up to the closing tag.
        std::string_view think_end;
        if (content.starts_with("</think>")) {
          think_end = "</think>";
        } else if (content.starts_with("</thinking>")) {
          think_end = "</thinking>";
        } else if (Contains(content, "\n</think>")) {
          think_end = "\n</think>";
        } else if (Contains(content, "\n</thinking>")) {
          think_end = "\n</thinking>";
        } else if (Contains(content, "\n</ think>")) {
          think_end = "\n</ think>";
        } else if (Contains(content, "\n</think >")) {
          think_end = "\n</think >";
        } else if (content.starts_with("<think>") &&
                   Contains(content, "</think>")) {
          think_end = "</think>";
        } else if (content.starts_with("<thinking>") &&
                   Contains(content, "</thinking>")) {
          think_end = "</thinking>";
        }
        if (!think_end.empty()) {
          const std::string_view think_start =
              Contains(think_end, "thinking") ? "<thinking>" : "<think>";
          reasoning = RstripNewlines(BeforeFirst(content, think_end));
          if (Contains(reasoning, think_start)) {
            reasoning = LstripNewlines(AfterLast(reasoning, think_start));
          }
          content = LstripNewlines(AfterLast(content, think_end));
        }
      }
      reasoning = Trim(reasoning);

      output.append(kImStart).append("assistant\n");
      if (options.preserve_thinking || index > last_query) {
        output.append("<think>\n");
        AppendContent(output, content_spans, reasoning);
        output.append("\n</think>\n\n");
      }
      AppendContent(output, content_spans, content);
      bool first_call = true;
      for (const auto& call : message.tool_calls) {
        if (!first_call) {
          output.push_back('\n');
        } else if (!Trim(content).empty()) {
          output.append("\n\n");
        }
        first_call = false;
        if (json_calls) {
          output.append("<tool_call>\n{\"name\": ");
          AppendContent(output, content_spans, JsonString(call.name));
          output.append(", \"arguments\": ");
          AppendContent(output, content_spans, ToolArgumentsJson(call));
          output.append("}\n</tool_call>");
          continue;
        }
        output.append("<tool_call>\n<function=");
        AppendContent(output, content_spans, call.name);
        output.append(">\n");
        for (const auto& argument : call.arguments) {
          output.append("<parameter=");
          AppendContent(output, content_spans, argument.name);
          output.append(">\n");
          AppendContent(output, content_spans,
                        argument.is_string ? argument.value
                                           : NormalizeJson(argument.value));
          output.append("\n</parameter>\n");
        }
        output.append("</function>\n</tool_call>");
      }
      output.append(kImEnd).append("\n");
    } else {
      // Tool results share one user turn until the next non-tool message.
      failures = ToolFailureCount(piece.text, failures);
      if (!has_previous || previous_role != ChatRole::kTool) {
        output.append(kImStart).append("user");
      }
      output.append("\n<tool_response>\n");
      AppendPiece(output, piece, image_offsets, content_spans);
      if (failures >= 2) {
        output.append(kCyberTielManyFailuresHead);
        output.append(std::to_string(failures));
        output.append(kCyberTielManyFailuresTail);
      } else if (failures == 1) {
        output.append(kCyberTielOneFailure);
      }
      output.append("\n</tool_response>");
      if (index + 1 == turns.size() ||
          turns[index + 1].role != ChatRole::kTool) {
        output.append(kImEnd).append("\n");
      }
    }
    previous_role = message.role;
    has_previous = true;

    if (output.size() > options.max_output_bytes) {
      return fail("Prompt exceeded max output bytes during message rendering");
    }
  }

  if (stable_prefix_bytes != nullptr) {
    *stable_prefix_bytes = std::min(mutable_reasoning.value_or(output.size()),
                                    final_user_start.value_or(output.size()));
  }
  if (options.add_generation_prompt) {
    output.append(kImStart).append("assistant\n");
    output.append(thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
  }
  if (output.size() > options.max_output_bytes) {
    return fail("Prompt exceeded max output bytes after generation prompt");
  }
  return output;
}

// ---- Qwen3.6 dialect --------------------------------------------------------
// Compiled form of the chat template inside the Qwen3.6-27B GGUFs
// (unsloth/Qwen3.6-27B-GGUF, SHA-256 QwenChatTemplate::Qwen36TemplateSha256).
// It differs from the Qwen3.8 template in these ways:
//   - no reasoning-effort line;
//   - only the first one or two leading system/developer messages count, joined
//     by one newline; later system/developer messages are dropped, not an
//     error, and are still the "previous message" of a tool result;
//   - an assistant turn without reasoning_content whose content holds </think>
//     is split there;
//   - preserve_thinking is read only when it is true (default off);
//   - tool-call arguments that are not containers are written with Python's
//     str(), so true is "True" and null is "None".
// tests/fixtures/qwen36_template holds jinja2 renderings of the original
// template; qwen_qwen36_template_test compares them byte for byte.

constexpr std::string_view kQwen36ToolsIntro =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

constexpr std::string_view kQwen36ToolsFooter =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
    "following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call "
    "in natural language BEFORE the function call, but NOT after\n- If "
    "there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function "
    "calls\n</IMPORTANT>";

// Python's str(float(token)) for a JSON number token that is a float.
std::string PythonFloatString(std::string_view token) {
  const std::string text(token);
  const double value = std::strtod(text.c_str(), nullptr);
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return value < 0 ? "-inf" : "inf";
  }
  std::array<char, 64> buffer{};
  const auto result =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                    std::fabs(value), std::chars_format::scientific);
  const std::string_view scientific(buffer.data(), result.ptr - buffer.data());
  // Shortest round-trip digits d[.ddd]e[+-]XX; the value is 0.DIGITS * 10^decpt.
  const auto marker = scientific.find('e');
  std::string digits;
  for (const char character : scientific.substr(0, marker)) {
    if (character != '.') {
      digits.push_back(character);
    }
  }
  while (digits.size() > 1 && digits.back() == '0') {
    digits.pop_back();
  }
  const int exponent = std::atoi(std::string(scientific.substr(marker + 1)).c_str());
  const int decpt = exponent + 1;
  const int count = static_cast<int>(digits.size());
  std::string output = std::signbit(value) ? "-" : "";
  if (exponent < -4 || exponent >= 16) {
    output.push_back(digits[0]);
    if (count > 1) {
      output.push_back('.');
      output.append(digits, 1, std::string::npos);
    }
    output.push_back('e');
    output.push_back(exponent < 0 ? '-' : '+');
    const int magnitude = exponent < 0 ? -exponent : exponent;
    if (magnitude < 10) {
      output.push_back('0');
    }
    output.append(std::to_string(magnitude));
  } else if (decpt <= 0) {
    output.append("0.");
    output.append(static_cast<std::size_t>(-decpt), '0');
    output.append(digits);
  } else if (decpt >= count) {
    output.append(digits);
    output.append(static_cast<std::size_t>(decpt - count), '0');
    output.append(".0");
  } else {
    output.append(digits, 0, static_cast<std::size_t>(decpt));
    output.push_back('.');
    output.append(digits, static_cast<std::size_t>(decpt), std::string::npos);
  }
  return output;
}

// The template writes `value | tojson | safe` for a mapping or a list and
// `value | string` for every other value that is not already a string.
std::string Qwen36ArgumentText(const ChatMessage::ToolArgument& argument) {
  if (argument.is_string) {
    return argument.value;
  }
  const std::string_view value = argument.value;
  if (value.starts_with('{') || value.starts_with('[')) {
    return NormalizeJson(value);
  }
  if (value == "true") {
    return "True";
  }
  if (value == "false") {
    return "False";
  }
  if (value == "null") {
    return "None";
  }
  const auto digits = value.starts_with('-') ? value.substr(1) : value;
  if (!digits.empty() &&
      std::all_of(digits.begin(), digits.end(),
                  [](char character) { return character >= '0' && character <= '9'; })) {
    return value == "-0" ? "0" : std::string(value);
  }
  return PythonFloatString(value);
}

std::optional<std::string> RenderQwen36(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets, std::size_t* stable_prefix_bytes,
    std::vector<ContentSpan>* content_spans) {
  const auto fail = [&](std::string message) -> std::optional<std::string> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return std::nullopt;
  };
  const auto is_system = [](const ChatMessage& message) {
    return message.role == ChatRole::kSystem ||
           message.role == ChatRole::kDeveloper;
  };

  // Only the first one or two leading system/developer messages count. Each is
  // trimmed; two are joined by a single newline even when one is empty.
  std::string merged;
  std::vector<ContentSpan> merged_spans;
  bool image_in_system = false;
  const auto append_system = [&](const ChatMessage& message) {
    image_in_system = image_in_system || !message.images.empty();
    const std::string untrimmed = message.content + message.framing_suffix;
    const auto part = Trim(untrimmed);
    const auto begin = static_cast<std::size_t>(part.data() - untrimmed.data());
    const auto client_end =
        std::min(message.content.size(), begin + part.size());
    const auto client_size = client_end > begin ? client_end - begin : 0;
    AppendContent(merged, &merged_spans, part.substr(0, client_size));
    merged.append(part.substr(client_size));
  };
  if (!messages.empty() && is_system(messages[0])) {
    append_system(messages[0]);
    if (messages.size() > 1 && is_system(messages[1])) {
      merged.push_back('\n');
      append_system(messages[1]);
    }
  }
  if (image_in_system) {
    return fail("System message cannot contain images.");
  }

  std::string output;
  const auto append_merged = [&] {
    const auto offset = output.size();
    output.append(merged);
    if (content_spans != nullptr) {
      for (const auto& span : merged_spans) {
        content_spans->push_back({offset + span.offset, span.size});
      }
    }
  };
  if (!tools.empty()) {
    output.append(kImStart).append("system\n");
    output.append(kQwen36ToolsIntro);
    for (const auto& tool : tools) {
      output.push_back('\n');
      AppendContent(output, content_spans, ToolDefinitionJson(tool));
    }
    output.append(kQwen36ToolsFooter);
    if (!merged.empty()) {
      output.append("\n\n");
      append_merged();
    }
    output.append(kImEnd).append("\n");
  } else if (!merged.empty()) {
    output.append(kImStart).append("system\n");
    append_merged();
    output.append(kImEnd).append("\n");
  }

  // The last user message that is not a tool-result wrapper is the query; the
  // scan covers the whole list. With no such message every index is at or
  // before it, so reasoning is kept only when preserve_thinking is true.
  bool query_found = false;
  std::size_t last_query = messages.size() - 1;
  for (std::size_t index = messages.size(); index > 0; --index) {
    if (messages[index - 1].role != ChatRole::kUser) {
      continue;
    }
    std::size_t scratch_images = 0;
    const Piece scan = MakePiece(messages[index - 1], options, scratch_images);
    if (!(scan.text.starts_with("<tool_response>") &&
          scan.text.ends_with("</tool_response>"))) {
      query_found = true;
      last_query = index - 1;
      break;
    }
  }

  // Checkpoint hints, as in the other renderers.
  const bool final_user_turn =
      query_found && last_query + 1 == messages.size() &&
      messages.back().images.empty() && messages.size() >= 2 &&
      (messages[messages.size() - 2].role == ChatRole::kTool ||
       messages[messages.size() - 2].role == ChatRole::kUser);
  std::optional<std::size_t> mutable_reasoning;
  std::optional<std::size_t> final_user_start;
  bool seen_assistant = false;

  std::size_t image_count = 0;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    if (is_system(message)) {
      continue;  // Merged above, or dropped.
    }
    const Piece piece = MakePiece(message, options, image_count);
    if (message.role == ChatRole::kAssistant) {
      if (!options.preserve_thinking && index > last_query &&
          !mutable_reasoning) {
        mutable_reasoning = output.size();
      }
      seen_assistant = true;
    } else if (message.role == ChatRole::kUser && final_user_turn &&
               seen_assistant && index == last_query) {
      final_user_start = output.size();
    }

    if (message.role == ChatRole::kUser) {
      output.append(kImStart).append("user\n");
      AppendPiece(output, piece, image_offsets, content_spans);
      output.append(kImEnd).append("\n");
    } else if (message.role == ChatRole::kAssistant) {
      std::string_view content = piece.text;
      std::string_view reasoning;
      if (!message.thought.empty()) {
        reasoning = message.thought;
      } else if (Contains(content, "</think>")) {
        auto before = RstripNewlines(BeforeFirst(content, "</think>"));
        if (Contains(before, "<think>")) {
          before = AfterLast(before, "<think>");
        }
        reasoning = LstripNewlines(before);
        content = LstripNewlines(AfterLast(content, "</think>"));
      }
      reasoning = Trim(reasoning);

      output.append(kImStart).append("assistant\n");
      if (options.preserve_thinking || index > last_query) {
        output.append("<think>\n");
        AppendContent(output, content_spans, reasoning);
        output.append("\n</think>\n\n");
      }
      AppendContent(output, content_spans, content);
      bool first_call = true;
      for (const auto& call : message.tool_calls) {
        if (!first_call) {
          output.push_back('\n');
        } else if (!Trim(content).empty()) {
          output.append("\n\n");
        }
        first_call = false;
        output.append("<tool_call>\n<function=");
        AppendContent(output, content_spans, call.name);
        output.append(">\n");
        for (const auto& argument : call.arguments) {
          output.append("<parameter=");
          AppendContent(output, content_spans, argument.name);
          output.append(">\n");
          AppendContent(output, content_spans, Qwen36ArgumentText(argument));
          output.append("\n</parameter>\n");
        }
        output.append("</function>\n</tool_call>");
      }
      output.append(kImEnd).append("\n");
    } else {
      // A tool result opens a user turn unless the previous message in the
      // list, system or not, is a tool result too (or there is none).
      if (index > 0 && messages[index - 1].role != ChatRole::kTool) {
        output.append(kImStart).append("user");
      }
      output.append("\n<tool_response>\n");
      AppendPiece(output, piece, image_offsets, content_spans);
      output.append("\n</tool_response>");
      if (index + 1 == messages.size() ||
          messages[index + 1].role != ChatRole::kTool) {
        output.append(kImEnd).append("\n");
      }
    }

    if (output.size() > options.max_output_bytes) {
      return fail("Prompt exceeded max output bytes during message rendering");
    }
  }

  if (stable_prefix_bytes != nullptr) {
    *stable_prefix_bytes = std::min(mutable_reasoning.value_or(output.size()),
                                    final_user_start.value_or(output.size()));
  }
  if (options.add_generation_prompt) {
    output.append(kImStart).append("assistant\n");
    output.append(options.enable_thinking ? "<think>\n"
                                          : "<think>\n\n</think>\n\n");
  }
  if (output.size() > options.max_output_bytes) {
    return fail("Prompt exceeded max output bytes after generation prompt");
  }
  return output;
}

}  // namespace

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto template_str = reader.GetMetadataString("tokenizer.chat_template");
  if (!template_str.has_value() || template_str->empty()) {
    if (IsQwen38Artifact(reader)) {
      if (error_msg != nullptr) {
        *error_msg = "Qwen3.8 GGUF is missing tokenizer.chat_template metadata";
      }
      return nullptr;
    }
    if (error_msg != nullptr) {
      *error_msg =
          "GGUF metadata missing 'tokenizer.chat_template', using default "
          "ChatML template";
    }
    return CreateDefault();
  }
  const Profile profile = ClassifyTemplate(*template_str);
  // A file named Qwen3.8 keeps the strict pin. The Qwen3.6 template, by exact
  // hash, is also accepted for a file that is Qwen3.8-sized but does not name
  // itself Qwen3.8.
  const auto name = reader.GetMetadataString("general.name");
  const bool names_qwen38 =
      name.has_value() && name->find("Qwen3.8") != std::string_view::npos;
  if (IsQwen38Artifact(reader) && profile != Profile::kQwen38Reasoning &&
      !(profile == Profile::kQwen36 && !names_qwen38)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned "
          "version: " +
          TemplateSha256(*template_str);
    }
    return nullptr;
  }
  return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
      std::string(*template_str), TemplateSha256(*template_str), profile));
}

bool QwenChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                            std::string* error_msg) {
  if (!IsQwen38Artifact(reader)) {
    return true;
  }
  return CreateFromGguf(reader, error_msg) != nullptr;
}

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateDefault(
    std::string_view raw_template) {
  if (raw_template.empty()) {
    return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
        std::string(kDefaultChatmlTemplate),
        TemplateSha256(kDefaultChatmlTemplate), Profile::kLegacyChatMl));
  }
  return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
      std::string(raw_template), TemplateSha256(raw_template),
      ClassifyTemplate(raw_template)));
}

std::string_view QwenChatTemplate::GetTemplateId() const noexcept {
  switch (profile_) {
    case Profile::kLegacyChatMl:
      return "qwen-chatml-compiled-v1";
    case Profile::kQwen38Reasoning:
      return "qwen38-reasoning-compiled-v3";
    case Profile::kCyberTiel:
      return "cybertiel-compiled-v1";
    case Profile::kQwen36:
      return "qwen36-compiled-v1";
  }
  return "qwen-chatml-compiled-v1";
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, const ChatTemplateOptions& options,
    std::string* error_msg) {
  return Render(messages, {}, options, error_msg);
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets, std::size_t* stable_prefix_bytes,
    std::vector<ContentSpan>* content_spans) {
  if (image_offsets != nullptr)
    image_offsets->clear();
  if (content_spans != nullptr)
    content_spans->clear();
  if (messages.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "No messages provided";
    }
    return std::nullopt;
  }

  const auto template_messages = messages;

  // The reference template accepts one leading system turn, but OpenAI clients
  // such as Codex send system/developer messages mid-conversation. Hoist them,
  // in order, into that turn; templates that allow them keep them in place.
  const auto is_system = [](const ChatMessage& message) {
    return message.role == ChatRole::kSystem ||
           message.role == ChatRole::kDeveloper;
  };
  std::vector<ChatMessage> hoisted;
  if (std::any_of(std::find_if_not(messages.begin(), messages.end(), is_system),
                  messages.end(), is_system)) {
    hoisted.assign(messages.begin(), messages.end());
    std::stable_partition(hoisted.begin(), hoisted.end(), is_system);
    messages = hoisted;
  }

  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len += msg.content.size() + msg.framing_suffix.size() +
                     msg.thought.size() + 32 + msg.images.size() * 64;
    std::size_t previous = 0;
    for (const auto& image : msg.images) {
      if ((msg.role != ChatRole::kUser && msg.role != ChatRole::kTool) ||
          image.bytes == nullptr || image.offset < previous ||
          image.offset > msg.content.size()) {
        if (error_msg != nullptr)
          *error_msg = "invalid user or tool image content";
        return std::nullopt;
      }
      previous = image.offset;
    }
    for (const auto& call : msg.tool_calls) {
      estimated_len += call.name.size() + 64;
      for (const auto& argument : call.arguments) {
        estimated_len += argument.name.size() + argument.value.size() + 40;
      }
    }
  }
  for (const auto& tool : tools) {
    estimated_len += tool.name.size() + tool.description.size() +
                     tool.parameters_json.size() +
                     2 * tool.definition_json.size() + 96;
  }
  if (options.add_generation_prompt) {
    estimated_len += 64;
  }
  estimated_len += kXHighReasoningInstruction.size() + 64;

  if (estimated_len > options.max_output_bytes ||
      estimated_len >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    if (error_msg != nullptr) {
      *error_msg = "Rendered prompt estimated length (" +
                   std::to_string(estimated_len) +
                   " bytes) exceeds maximum bound (" +
                   std::to_string(options.max_output_bytes) + " bytes)";
    }
    return std::nullopt;
  }

  if (options.dialect == QwenTemplateDialect::kCyberTiel) {
    // That template keeps system messages where the client put them, so it
    // reads the messages as sent, not as hoisted above.
    return RenderCyberTiel(template_messages, tools, options, error_msg,
                               image_offsets, stable_prefix_bytes,
                               content_spans);
  }
  if (options.dialect == QwenTemplateDialect::kQwen36) {
    // That template drops later system messages instead of hoisting them, so
    // it also reads the messages as sent.
    return RenderQwen36(template_messages, tools, options, error_msg,
                        image_offsets, stable_prefix_bytes, content_spans);
  }

  output.reserve(estimated_len);

  std::size_t message_index = 0;
  std::string system_content;
  std::vector<ContentSpan> system_content_spans;
  while (message_index < messages.size() &&
         (messages[message_index].role == ChatRole::kSystem ||
          messages[message_index].role == ChatRole::kDeveloper)) {
    const auto& message = messages[message_index];
    const std::string untrimmed = message.content + message.framing_suffix;
    const auto content = Trim(untrimmed);
    if (!content.empty()) {
      if (!system_content.empty())
        system_content.push_back('\n');
      const auto begin =
          static_cast<std::size_t>(content.data() - untrimmed.data());
      const auto client_end =
          std::min(message.content.size(), begin + content.size());
      const auto client_size = client_end > begin ? client_end - begin : 0;
      AppendContent(system_content, &system_content_spans,
                    content.substr(0, client_size));
      system_content.append(content.substr(client_size));
    }
    ++message_index;
  }

  std::string system_prefix;
  std::vector<ContentSpan> system_spans;
  AppendReasoningInstruction(system_prefix, options);
  AppendToolsPrompt(system_prefix, tools, options.require_tool_call);
  if (!system_prefix.empty() && !system_content.empty()) {
    system_prefix.append("\n\n");
  }
  const auto system_content_offset = system_prefix.size();
  system_prefix.append(system_content);
  for (const auto& span : system_content_spans)
    system_spans.push_back({system_content_offset + span.offset, span.size});
  if (!system_prefix.empty()) {
    output.append(kImStart).append("system\n");
    const auto prefix_offset = output.size();
    output.append(system_prefix);
    output.append(kImEnd).append("\n");
    if (content_spans != nullptr) {
      for (const auto& span : system_spans)
        content_spans->push_back({prefix_offset + span.offset, span.size});
    }
  }

  std::size_t last_user_index = messages.size();
  for (std::size_t index = messages.size(); index > 0; --index) {
    if (messages[index - 1].role == ChatRole::kUser) {
      const auto content = Trim(messages[index - 1].content);
      if (messages[index - 1].images.empty() &&
          content.starts_with("<tool_response>") &&
          content.ends_with("</tool_response>"))
        continue;
      last_user_index = index - 1;
      break;
    }
  }
  if (last_user_index == messages.size()) {
    if (error_msg != nullptr)
      *error_msg = "No user query found in messages.";
    return std::nullopt;
  }

  std::size_t image_count = 0;
  std::optional<std::size_t> mutable_reasoning;
  // Agent clients may end every request with a user message that the next
  // request replaces (per-turn runtime context) rather than keeps. A
  // checkpoint at the end of the prompt then never prefixes the next request.
  // Such context follows a tool result or another user message, not an
  // assistant reply, so only then keep the checkpoint before the final
  // text-only user turn. A client that keeps the message prefills it again.
  const bool final_user_turn =
      last_user_index + 1 == messages.size() &&
      messages.back().images.empty() && messages.size() >= 2 &&
      (messages[messages.size() - 2].role == ChatRole::kTool ||
       messages[messages.size() - 2].role == ChatRole::kUser);
  bool seen_assistant = false;
  std::optional<std::size_t> final_user_start;
  for (; message_index < messages.size(); ++message_index) {
    const auto& msg = messages[message_index];
    const bool tool_result = msg.role == ChatRole::kTool;
    if (tool_result) {
      output.append(kImStart).append("user\n");
      while (message_index < messages.size() &&
             messages[message_index].role == ChatRole::kTool) {
        const auto& tool_message = messages[message_index];
        output.append("<tool_response>\n");
        AppendMessageContent(output, tool_message, options, image_count,
                             image_offsets, content_spans);
        output.append("\n</tool_response>");
        ++message_index;
        if (message_index < messages.size() &&
            messages[message_index].role == ChatRole::kTool) {
          output.push_back('\n');
        }
      }
      --message_index;
      output.append(kImEnd).append("\n");
      continue;
    }
    // Current tool-cycle assistants retain their reasoning even when older
    // reasoning is disabled. A new user turn removes it, including empty
    // <think> framing. Keep a checkpoint before the first affected assistant.
    if (!options.preserve_thinking && msg.role == ChatRole::kAssistant &&
        message_index > last_user_index && !mutable_reasoning)
      mutable_reasoning = output.size();
    if (msg.role == ChatRole::kAssistant)
      seen_assistant = true;
    else if (final_user_turn && seen_assistant &&
             message_index == last_user_index)
      final_user_start = output.size();
    const auto role_name = ToString(msg.role);
    output.append(kImStart);
    output.append(role_name);
    output.push_back('\n');

    const std::string_view thought = Trim(msg.thought);
    if (msg.role == ChatRole::kAssistant &&
        (options.preserve_thinking || message_index > last_user_index)) {
      output.append("<think>\n");
      AppendContent(output, content_spans, thought);
      output.append("\n</think>\n\n");
    }

    const bool has_content = AppendMessageContent(
        output, msg, options, image_count, image_offsets, content_spans);
    if (msg.role == ChatRole::kAssistant && !msg.tool_calls.empty()) {
      if (has_content) {
        output.append("\n\n");
      }
      AppendToolCalls(output, msg.tool_calls, content_spans);
    }
    output.append(kImEnd).append("\n");

    if (output.size() > options.max_output_bytes) {
      if (error_msg != nullptr) {
        *error_msg =
            "Prompt exceeded max output bytes during message rendering";
      }
      return std::nullopt;
    }
  }

  if (stable_prefix_bytes != nullptr)
    *stable_prefix_bytes = std::min(mutable_reasoning.value_or(output.size()),
                                    final_user_start.value_or(output.size()));
  if (options.add_generation_prompt)
    output.append(GenerationPrompt(options.enable_thinking));

  if (output.size() > options.max_output_bytes) {
    if (error_msg != nullptr) {
      *error_msg = "Prompt exceeded max output bytes after generation prompt";
    }
    return std::nullopt;
  }

  return output;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  static constexpr std::string_view kThinkingSuffix = "assistant\n<think>\n";
  static constexpr std::string_view kAnsweredSuffix =
      "assistant\n<think>\n\n</think>\n\n";
  static constexpr auto kThinking =
      ConcatControlText<kImStart, kThinkingSuffix>();
  static constexpr auto kAnswered =
      ConcatControlText<kImStart, kAnsweredSuffix>();
  return enable_thinking ? std::string_view(kThinking.data(), kThinking.size())
                         : std::string_view(kAnswered.data(), kAnswered.size());
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    const ChatTemplateOptions& options, std::string* error_msg) {
  return RenderAndTokenize(tokenizer, messages, {}, options, error_msg);
}

std::optional<std::string> QwenChatTemplate::RenderTail(
    std::span<const ChatMessage> messages, const ChatTemplateOptions& options,
    std::string* error_msg, std::vector<ContentSpan>* content_spans) {
  if (content_spans)
    content_spans->clear();
  if (messages.empty() || options.dialect != QwenTemplateDialect::kQwen ||
      !options.preserve_thinking ||
      std::any_of(messages.begin(), messages.end(), [](const auto& message) {
        return message.role == ChatRole::kSystem ||
               message.role == ChatRole::kDeveloper || !message.images.empty();
      })) {
    if (error_msg)
      *error_msg = "incremental tail requires preserved Qwen text turns";
    return std::nullopt;
  }
  std::string output;
  std::size_t image_count = 0;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    if (message.role == ChatRole::kTool) {
      output.append(kImStart).append("user\n");
      do {
        output.append("<tool_response>\n");
        AppendMessageContent(output, messages[index], options, image_count,
                             nullptr, content_spans);
        output.append("\n</tool_response>");
        ++index;
        if (index < messages.size() && messages[index].role == ChatRole::kTool)
          output.push_back('\n');
      } while (index < messages.size() &&
               messages[index].role == ChatRole::kTool);
      --index;
    } else {
      output.append(kImStart).append(ToString(message.role)).append("\n");
      if (message.role == ChatRole::kAssistant) {
        output.append("<think>\n");
        AppendContent(output, content_spans, Trim(message.thought));
        output.append("\n</think>\n\n");
      }
      const bool content = AppendMessageContent(
          output, message, options, image_count, nullptr, content_spans);
      if (message.role == ChatRole::kAssistant && !message.tool_calls.empty()) {
        if (content)
          output.append("\n\n");
        AppendToolCalls(output, message.tool_calls, content_spans);
      }
    }
    output.append(kImEnd).append("\n");
    if (output.size() > options.max_output_bytes) {
      if (error_msg)
        *error_msg = "incremental tail exceeds rendered bound";
      return std::nullopt;
    }
  }
  if (options.add_generation_prompt)
    output.append(GenerationPrompt(options.enable_thinking));
  return output;
}

std::vector<TokenId> QwenChatTemplate::EncodeRendered(
    const QwenTokenizer& tokenizer, std::string_view rendered,
    std::size_t begin, std::size_t end,
    std::span<const ContentSpan> content_spans,
    const TokenizerOptions& options) {
  const auto slice = rendered.substr(begin, end - begin);
  // Only split spans containing literal vocabulary-token spellings. Splitting
  // every content span when a later message contains one changes earlier BPE
  // merges at content/framing boundaries and invalidates unchanged history.
  TokenizerOptions as_framing = options;
  as_framing.parse_special_tokens = true;
  TokenizerOptions as_content = options;
  as_content.parse_special_tokens = false;
  const auto encode = [&](std::size_t from, std::size_t to,
                          const TokenizerOptions& opts) {
    return tokenizer.Encode(rendered.substr(from, to - from), opts);
  };

  std::vector<TokenId> tokens;
  std::size_t cursor = begin;
  for (const auto& span : content_spans) {
    const auto span_end = span.offset + span.size;
    if (span_end <= begin || span.offset >= end) {
      continue;
    }
    const auto from = std::max(span.offset, begin);
    const auto to = std::min(span_end, end);
    const auto content_text = rendered.substr(from, to - from);
    if (std::none_of(tokenizer.SpecialTokens().begin(),
                     tokenizer.SpecialTokens().end(), [&](const auto& token) {
                       return content_text.find(token.first) !=
                              std::string_view::npos;
                     }))
      continue;
    if (from > cursor) {
      const auto framing = encode(cursor, from, as_framing);
      tokens.insert(tokens.end(), framing.begin(), framing.end());
    }
    const auto content = encode(from, to, as_content);
    tokens.insert(tokens.end(), content.begin(), content.end());
    cursor = to;
  }
  if (cursor == begin)
    return tokenizer.Encode(slice, options);
  if (cursor < end) {
    const auto tail = encode(cursor, end, as_framing);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
  }
  return tokens;
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  if (std::any_of(messages.begin(), messages.end(), [](const auto& message) {
        return !message.images.empty();
      })) {
    if (error_msg != nullptr) {
      *error_msg =
          "image messages require the model's vision prompt preparation";
    }
    return std::nullopt;
  }
  std::vector<ContentSpan> content_spans;
  const auto rendered = Render(messages, tools, options, error_msg, nullptr,
                               nullptr, &content_spans);
  if (!rendered.has_value()) {
    return std::nullopt;
  }

  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;
  return EncodeRendered(tokenizer, *rendered, 0, rendered->size(),
                        content_spans, tok_opts);
}

}  // namespace gufo::tokenization
