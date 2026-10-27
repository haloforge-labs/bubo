#ifndef GUFO_MODELS_DIFFUSION_GEMMA_CHAT_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_CHAT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::models::diffusion_gemma {

struct ToolCall {
  std::string id;
  std::string name;
  json::Value arguments{json::Value::object()};  ///< A JSON object.
};

struct Message {
  /// "system", "developer", "user", "assistant" or "tool".
  std::string role{"user"};
  std::string content;
  std::vector<ToolCall> tool_calls;  ///< Assistant messages.
  std::string tool_call_id;          ///< Tool messages.
  std::string name;                  ///< Tool messages: the function.
};

/// Renders the checkpoint's chat_template.jinja (Gemma 4) with
/// add_generation_prompt: BOS, the system turn (thinking marker, leading
/// system/developer text, tool declarations), each message (assistant
/// thinking channels stripped, tool calls, the tool results that follow
/// them, turn continuation rules), then the generation prompt. `tools` are
/// OpenAI tool objects ({"type":"function","function":{...}}).
std::string RenderChat(const std::vector<Message>& messages,
                       const std::vector<json::Value>& tools, bool thinking);

/// The template's format_argument macro.
std::string FormatArgument(const json::Value& value, bool escape_keys);
/// The template's format_function_declaration macro (without <|tool>).
std::string FormatToolDeclaration(const json::Value& tool);

/// Rewrites every complete native call (`<|tool_call>call:NAME{...}
/// <tool_call|>`) as a Qwen XML call block, which the HTTP adapter parses
/// into tool calls. Text from an unterminated call onward is cut and its
/// offset in `text` stored in `*held` (npos when nothing is held).
std::string NativeToolCallsToXml(std::string_view text, std::size_t* held);

/// Removes a leading thinking channel (`<|channel>...<channel|>`) from
/// generated text; returns the text unchanged when there is none.
std::string StripThinking(const std::string& text);

/// Token-level form: drops every `open ... close` span (inclusive); an
/// unterminated span runs to the end.
std::vector<std::int32_t> StripThinkingTokens(
    std::span<const std::int32_t> tokens, std::int32_t open,
    std::int32_t close);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_CHAT_HPP_
