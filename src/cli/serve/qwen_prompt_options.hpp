#ifndef GUFO_SERVER_QWEN_PROMPT_OPTIONS_HPP_
#define GUFO_SERVER_QWEN_PROMPT_OPTIONS_HPP_

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

inline tokenization::ChatTemplateOptions QwenChatOptions(
    const ChatRequest& request, std::uint32_t max_context,
    tokenization::QwenTemplateDialect dialect =
        tokenization::QwenTemplateDialect::kQwen) {
  auto options = tokenization::ResolveQwenChatOptions(
      request.reasoning, request.add_vision_id, dialect);
  options.require_tool_call =
      request.tool_choice == ChatRequest::ToolChoice::kRequired;
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

// Shared with the CPU HTTP preparation check: constructing the Flash-Next
// runner itself requires a loaded GPU model.
inline std::string_view QwenIncrementalRequestGuard(
    const ChatRequest& request) {
  return request.response_format ? "response_format"
         : request.tool_choice != ChatRequest::ToolChoice::kAuto ||
                 !request.forced_tool_name.empty() || request.json_tool_calls
             ? "request_options"
             : "";
}

}  // namespace gufo::server
#endif  // GUFO_SERVER_QWEN_PROMPT_OPTIONS_HPP_
