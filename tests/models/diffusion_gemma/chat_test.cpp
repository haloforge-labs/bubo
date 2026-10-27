// DiffusionGemma prompt rendering, tool-call translation and
// thinking-channel removal (CPU only).
#include "src/models/diffusion_gemma/chat.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

dg::Message Msg(std::string role, std::string content) {
  dg::Message m;
  m.role = std::move(role);
  m.content = std::move(content);
  return m;
}

}  // namespace

int main() {
  // chat_template.jinja output for these conversations, thinking disabled.
  Expect(dg::RenderChat({Msg("user", "Why is the sky blue?")}, {}, false) ==
             "<bos><|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n",
         "single user turn");
  Expect(dg::RenderChat(
             {Msg("system", "  Be brief. "), Msg("user", " Hi "),
              Msg("assistant", "<|channel>thought\nplan<channel|>Hello!"),
              Msg("user", "Again")},
             {}, false) ==
             "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n"
             "<|turn>model\nHello!<turn|>\n<|turn>user\nAgain<turn|>\n"
             "<|turn>model\n",
         "system, history and stripped assistant thinking");
  Expect(dg::RenderChat({Msg("user", "Hi")}, {}, true) ==
             "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n"
             "<|turn>model\n",
         "thinking marker");
  // A tool round trip: the result follows the call in the same model turn,
  // which stays open for the reply.
  dg::Message call = Msg("assistant", "");
  call.tool_calls.push_back(dg::ToolCall{
      "c1", "read", gufo::json::parse(R"({"path":"a.txt","limit":5})")});
  dg::Message result = Msg("tool", "hello");
  result.tool_call_id = "c1";
  const auto tool = gufo::json::parse(
      R"({"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path"},"limit":{"type":"integer"}},"required":["path"]}}})");
  Expect(dg::RenderChat({Msg("user", "Read a.txt"), call, result}, {tool},
                        false) ==
             "<bos><|turn>system\n<|tool>declaration:read{description:<|\"|>"
             "Read a file<|\"|>,parameters:{properties:{limit:{type:<|\"|>"
             "INTEGER<|\"|>},path:{description:<|\"|>Path<|\"|>,type:<|\"|>"
             "STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT"
             "<|\"|>}}<tool|><turn|>\n<|turn>user\nRead a.txt<turn|>\n"
             "<|turn>model\n<|tool_call>call:read{limit:5,path:<|\"|>a.txt"
             "<|\"|>}<tool_call|><|tool_response>response:read{value:<|\"|>"
             "hello<|\"|>}<tool_response|>",
         "tool declaration, call and response");
  const std::vector<std::int32_t> tokens = {7, 100, 8, 9, 101, 10, 100, 11};
  Expect(dg::StripThinkingTokens(tokens, 100, 101) ==
             std::vector<std::int32_t>{7, 10},
         "token-level channel removal");
  Expect(dg::StripThinking("<|channel>thought\n<channel|>Answer") == "Answer",
         "text-level channel removal");
  std::size_t held = 0;
  Expect(dg::NativeToolCallsToXml(
             "Reading.<|tool_call>call:read{path:<|\"|>/a b<|\"|>,limit:5}"
             "<tool_call|>",
             &held) ==
             "Reading.<tool_call>\n<function=read>\n<parameter=path>\n/a b\n"
             "</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n"
             "</tool_call>",
         "native call to XML");
  Expect(dg::NativeToolCallsToXml("x<|tool_call>call:read{pa", &held) == "x" &&
             held == 1,
         "unterminated call is held");
  if (failures == 0)
    std::puts("PASS diffusion_gemma_chat_test");
  return failures == 0 ? 0 : 1;
}
