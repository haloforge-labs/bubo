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
  // Issue 11: the model drops the opener token. The declared tools are
  // bash, read and write.
  const std::vector<std::string> tools = {"bash", "read", "write"};
  const std::string bash_xml =
      "<tool_call>\n<function=bash>\n<parameter=command>\ncd tienda && npm "
      "install lucide-react\n</parameter>\n</function>\n</tool_call>";
  const std::string quoted =
      "{command:<|\"|>cd tienda && npm install lucide-react<|\"|>}<tool_call|>";
  // Fence with a language, bare name, closing fence.
  Expect(dg::NativeToolCallsToXml("Instalo.\n\n```bash\nbash" + quoted + "\n```",
                                  &held, tools) == "Instalo.\n\n" + bash_xml &&
             held == std::string::npos,
         "fenced call without opener");
  // Fence directly followed by call:.
  Expect(dg::NativeToolCallsToXml("```call:bash" + quoted, &held, tools) ==
             bash_xml,
         "fence and call: without opener");
  // Language and name on one line, with a colon (seen in issue 11 run 06).
  Expect(dg::NativeToolCallsToXml("Instalo.\n```bash:bash" + quoted + "\n```",
                                  &held, tools) == "Instalo.\n" + bash_xml &&
             held == std::string::npos,
         "fence with lang:NAME on one line");
  Expect(dg::NativeToolCallsToXml("```bash:bash" + quoted, &held, tools) ==
             bash_xml,
         "fence lang:NAME without closing fence");
  Expect(dg::NativeToolCallsToXml("```bash:call:bash" + quoted, &held, tools) ==
             bash_xml,
         "fence lang:call:NAME on one line");
  // The fence line may separate the language and the name in other ways, and
  // the language may be the name (issue 11 A/B, runs 21, 47 and 56).
  Expect(dg::NativeToolCallsToXml("Primero:\n\n```bash-bash" + quoted, &held,
                                  tools) == "Primero:\n\n" + bash_xml,
         "fence lang-NAME");
  Expect(dg::NativeToolCallsToXml("```bash" + quoted + "\n```", &held, tools) ==
             bash_xml,
         "fence language is the name");
  Expect(dg::NativeToolCallsToXml("Voy.\n\n```bash\n" + quoted, &held, tools) ==
             "Voy.\n\n" + bash_xml,
         "name only as the fence language, arguments on the next line");
  Expect(dg::NativeToolCallsToXml("```bash bash" + quoted, &held, tools) ==
             bash_xml,
         "fence lang NAME with a space");
  // call: alone and the bare name alone.
  Expect(dg::NativeToolCallsToXml("Ahora call:bash" + quoted, &held, tools) ==
             "Ahora " + bash_xml,
         "call: without opener");
  Expect(dg::NativeToolCallsToXml("bash" + quoted + " fin", &held, tools) ==
             bash_xml + " fin",
         "bare name without opener, text after");
  // Two calls, one with the opener and one without.
  Expect(dg::NativeToolCallsToXml(
             "<|tool_call>call:read{path:<|\"|>a<|\"|>}<tool_call|> y bash" +
                 quoted,
             &held, tools) ==
             "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>"
             "\n</function>\n</tool_call> y " +
                 bash_xml,
         "mixed calls");
  // Waiting for the closer: the call is held, text before it is not.
  Expect(dg::NativeToolCallsToXml("Voy a instalar.\n```bash\nbash{command:<|\"|>cd",
                                  &held, tools) == "Voy a instalar.\n" &&
             held == 16,
         "bare call waits for its closer");
  // The closing fence may still be arriving after the closer.
  Expect(dg::NativeToolCallsToXml("```bash\nbash" + quoted + "\n``", &held,
                                  tools) == bash_xml &&
             held != std::string::npos,
         "closing fence in progress");
  // Negative cases: all stay text.
  Expect(dg::NativeToolCallsToXml("bash" + quoted, &held, {}) ==
             "bash" + quoted,
         "no declared tools: stays text");
  Expect(dg::NativeToolCallsToXml("grep{command:<|\"|>x<|\"|>}<tool_call|>",
                                  &held, tools) ==
             "grep{command:<|\"|>x<|\"|>}<tool_call|>",
         "undeclared name stays text");
  Expect(dg::NativeToolCallsToXml("mybash{command:<|\"|>x<|\"|>}<tool_call|>",
                                  &held, tools) ==
             "mybash{command:<|\"|>x<|\"|>}<tool_call|>",
         "name inside a longer word stays text");
  Expect(dg::NativeToolCallsToXml("bash{command:<|\"|>x<|\"|>", &held, tools) ==
             "" &&
             held == 0,
         "unfinished arguments are held");
  Expect(dg::NativeToolCallsToXml("nota:bash" + quoted, &held, tools) ==
             "nota:bash" + quoted,
         "colon without a fence stays text");
  Expect(dg::NativeToolCallsToXml("```bash:grep" + quoted, &held, tools) ==
             "```bash:grep" + quoted,
         "fence lang:undeclared name stays text");
  Expect(dg::NativeToolCallsToXml("```js" + quoted, &held, tools) ==
             "```js" + quoted,
         "undeclared fence language stays text");
  Expect(dg::NativeToolCallsToXml("```js\n" + quoted, &held, tools) ==
             "```js\n" + quoted,
         "undeclared fence language, arguments on the next line, stays text");
  Expect(dg::NativeToolCallsToXml("```algo de prosa:bash" + quoted, &held,
                                  tools) == "```algo de prosa:bash" + quoted,
         "prose after a fence stays text");
  Expect(dg::NativeToolCallsToXml("```bash\nbash<tool_call|>", &held, tools) ==
             "```bash\nbash<tool_call|>",
         "closer with no argument object stays text");
  Expect(dg::NativeToolCallsToXml("```bash<tool_call|>", &held, tools) ==
             "```bash<tool_call|>",
         "fence language and closer only stays text");
  // The opener is present but the name is missing: stays text.
  Expect(dg::NativeToolCallsToXml("<|tool_call>call:" + quoted, &held, tools) ==
             "<|tool_call>call:" + quoted,
         "opener without a name stays text");
  Expect(dg::NativeToolCallsToXml("bash{oops}<tool_call|>", &held, tools) ==
             "bash{oops}<tool_call|>",
         "bad arguments stay text");
  // Streaming: each prefix of a reply gives text that the final text keeps
  // (nothing streamed is retracted), and a call that never closes is flushed
  // whole when the turn ends (the backend appends text from `*held`).
  const auto final_text = [&](const std::string& reply) {
    std::size_t at = 0;
    std::string out = dg::NativeToolCallsToXml(reply, &at, tools);
    if (at != std::string::npos)
      out += reply.substr(at);
    return out;
  };
  const std::vector<std::string> replies = {
      "Instalo.\n\n```bash\nbash" + quoted + "\n```",
      "```call:bash" + quoted + "\nListo",
      "Instalo.\n```bash:bash" + quoted + "\n```\nListo",
      "```bash:call:bash" + quoted,
      "Primero:\n\n```bash-bash" + quoted + "\n```\nListo",
      "Voy.\n```bash" + quoted + "\nListo",
      "Voy.\n```bash\n" + quoted,
      "Antes ```js" + quoted,
      "Ejemplo ```bash:bash{command:<|\"|>ls<|\"|>} fin",
      "Ahora call:read{path:<|\"|>a.txt<|\"|>}<tool_call|> y bash" + quoted,
      "Con `bash{command:<|\"|>ls<|\"|>}` se lista.",
      "Ejemplo:\n```bash\nbash{command:<|\"|>ls<|\"|>}\n```\nFin.",
      "Todo bien, sin llamadas. ```python\nprint(1)\n``` y read más.",
  };
  for (const auto& reply : replies) {
    const std::string whole = final_text(reply);
    bool stable = true;
    for (std::size_t k = 0; k <= reply.size(); ++k) {
      std::size_t at = 0;
      const std::string part =
          dg::NativeToolCallsToXml(reply.substr(0, k), &at, tools);
      if (!whole.starts_with(part)) {
        std::fprintf(stderr, "retracted at %zu of %zu: [%s] vs [%s]\n", k,
                     reply.size(), part.c_str(), whole.c_str());
        stable = false;
        break;
      }
    }
    Expect(stable, "streamed text is never retracted");
  }
  // No closer: the held text comes back as it was written, in order.
  for (const auto* prose :
       {"Con `bash{command:<|\"|>ls<|\"|>}` se lista.",
        "Ejemplo:\n```bash\nbash{command:<|\"|>ls<|\"|>}\n```\nFin.",
        "Usa ```bash\nbash{command:<|\"|>ls<|\"|>}\n``` listo",
        "Escribe call:write{path:<|\"|>x<|\"|>,content:<|\"|>y<|\"|>} aqui",
        "Termina en una valla ```bash\nba",
        "Ejemplo ```bash:bash{command:<|\"|>ls<|\"|>} fin",
        "Termina ```bash:ba",
        "Ejemplo ```bash\n{command:<|\"|>ls<|\"|>} fin",
        "Ejemplo ```bash-bash{command:<|\"|>ls<|\"|>}",
        "Datos ```js{a:1} y ```json\n{\"a\":1} fin",
        "Usa ```bash\nls -la\nbash -c 'x'\n``` y listo"})
    Expect(final_text(prose) == prose, "held text is flushed unchanged");
  if (failures == 0)
    std::puts("PASS diffusion_gemma_chat_test");
  return failures == 0 ? 0 : 1;
}
