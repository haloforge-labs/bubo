#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/qwen_prompt_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/incremental_tokenize.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace {
namespace s = gufo::server;
namespace q = gufo::tokenization;
namespace json = gufo::json;
constexpr std::uint32_t kContext = 1048576;
std::size_t failures = 0, turns = 0, comparisons = 0, rejections = 0;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL " << message << '\n';
  }
}

template<typename Change>
void Edit(json::Value& array, std::size_t index, Change change) {
  Require(array.is_array() && index < array.size(), "invalid array edit");
  auto replacement = json::Value::array();
  for (std::size_t at = 0; at < array.size(); ++at) {
    auto item = array.items()[at];
    if (at == index)
      change(item);
    replacement.push_back(std::move(item));
  }
  array = std::move(replacement);
}

const json::Value& Field(const json::Value& value, const std::string& name) {
  const auto* field = value.find(name);
  Require(field != nullptr, "missing JSON field " + name);
  return *field;
}

// Stop the real HTTP handler immediately after parsing, at start_chat. No
// scheduler, listener, model execution or invented generation result is used.
struct ParsedRequest {
  s::ChatRequest request;
};

class ParsingBoundary final : public s::TextGenerationBackend {
public:
  std::string model_id() const override { return "bubo"; }
  bool ready() const override { return true; }
  bool supports_timeline() const override { return true; }
  std::shared_ptr<GenerationRequest> start_chat(
      const s::ChatRequest& request, std::size_t,
      const gufo::sampling::SamplingConfig&, const CancellationCheck&,
      bool) override {
    throw ParsedRequest{request};
  }
  Result complete(std::string_view, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck&, const TokenCallback&,
                  std::string_view, const std::vector<std::string>&,
                  bool) override {
    throw std::logic_error("unexpected generation");
  }
  Result chat(const s::ChatRequest&, std::size_t,
              const gufo::sampling::SamplingConfig&, const CancellationCheck&,
              const TokenCallback&) override {
    throw std::logic_error("unexpected generation");
  }
  std::size_t count_tokens(std::string_view) const override {
    throw std::logic_error("unexpected token count");
  }
  std::optional<std::size_t> count_chat_tokens(
      const s::ChatRequest&) const override {
    throw std::logic_error("unexpected token count");
  }
};

s::ChatRequest Parse(const json::Value& body, std::string client = "A") {
  ParsingBoundary backend;
  s::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/chat/completions";
  http.body = body.dump();
  http.client_id = std::move(client);
  http.debug_hooks = true;
  try {
    const auto response = s::HandleOpenAiChat(http, backend);
    throw std::runtime_error("HTTP parser did not reach start_chat: " +
                             std::to_string(response.status) + " " +
                             response.body);
  } catch (const ParsedRequest& parsed) {
    return parsed.request;
  }
}

auto Tools(const s::ChatRequest& request) {
  return request.tool_choice == s::ChatRequest::ToolChoice::kNone
             ? std::span<const q::ChatTool>{}
             : std::span<const q::ChatTool>{request.tools};
}

auto Full(const q::QwenTokenizer& tokenizer, const s::ChatRequest& request) {
  return gufo::models::qwen::vision::Prepare(
      tokenizer, request.messages, Tools(request),
      s::QwenChatOptions(request, kContext), {}, kContext);
}

q::IncrementalPrompt Check(const q::QwenTokenizer& tokenizer,
                           q::QwenIncrementalTokenizer& cache,
                           const s::ChatRequest& request,
                           const std::string& label,
                           std::string_view expected_route,
                           std::string_view expected_error = {}) {
  std::vector<q::TokenId> expected;
  std::size_t prefix = 0;
  std::string full_error, candidate_error;
  try {
    auto full = Full(tokenizer, request);
    expected = std::move(full.tokens);
    prefix = full.stable_prefix_tokens;
  } catch (const std::exception& error) {
    full_error = error.what();
  }
  const auto before = cache.Statistics();
  q::IncrementalPrompt actual;
  try {
    actual =
        cache.Prepare(tokenizer, request.client_id, request.messages,
                      Tools(request), s::QwenChatOptions(request, kContext),
                      kContext, s::QwenIncrementalRequestGuard(request));
  } catch (const std::exception& error) {
    candidate_error = error.what();
  }
  ++turns;
  Require(full_error == candidate_error,
          label + ": error differs: full=" + full_error +
              " candidate=" + candidate_error);
  if (!expected_error.empty())
    Require(full_error == expected_error, label + ": wrong rejection text");
  if (full_error.empty()) {
    Require(expected.size() == actual.tokens.size(),
            label + ": token vector length differs");
    for (std::size_t index = 0; index < expected.size(); ++index) {
      Require(expected[index] == actual.tokens[index],
              label + ": token ID mismatch at " + std::to_string(index));
      ++comparisons;
    }
    Require(prefix == actual.cache_prefix_tokens,
            label + ": serving prefix differs");
    Expect(actual.reason == expected_route,
           label + ": expected " + std::string(expected_route) + ", got " +
               std::string(actual.reason));
  } else {
    ++rejections;
    const auto after = cache.Statistics();
    const auto count = [&](const json::Value& stats) {
      const auto* reason =
          stats.find("full_path_reasons")->find(std::string(expected_route));
      return reason ? reason->as_size() : 0;
    };
    Expect(count(after) == count(before) + 1,
           label + ": rejected request route " + std::string(expected_route));
  }
  if (full_error.empty()) {
    const auto after = cache.Statistics();
    if (expected_route == "hit") {
      Expect(after.member_size("incremental_hits") ==
                 before.member_size("incremental_hits") + 1,
             label + ": hit counter must advance");
    } else {
      const auto* prior =
          before.find("full_path_reasons")->find(std::string(expected_route));
      const auto* current =
          after.find("full_path_reasons")->find(std::string(expected_route));
      Expect(
          current && current->as_size() >= (prior ? prior->as_size() : 0) + 1,
          label + ": named full-path counter must advance");
    }
  }
  std::cout << "TURN " << label << " route=" << actual.reason
            << " tokens=" << expected.size() << " error=" << full_error
            << " stats=" << cache.Statistics().dump() << '\n';
  return actual;
}

json::Value Seed(bool tools = true) {
  auto body = json::parse(R"({
    "model":"bubo","messages":[
      {"role":"system","content":"Be precise."},
      {"role":"user","content":"Inspect the public example."}],
    "tool_choice":"auto","parallel_tool_calls":true,
    "max_tokens":32,"temperature":0,"top_k":1,"top_p":1,"min_p":0,
    "seed":1,"stream":false,"cache_prompt":true,"reasoning_effort":"xhigh",
    "chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}
  })");
  if (tools)
    body["tools"] = json::parse(R"([
      {"type":"function","function":{"name":"read_file",
       "description":"Read an example file.","parameters":{"type":"object",
       "properties":{"path":{"type":"string"}},"required":["path"],
       "additionalProperties":false}}},
      {"type":"function","function":{"name":"list_files",
       "description":"List example files.","parameters":{"type":"object",
       "properties":{"path":{"type":"string"}},"required":["path"],
       "additionalProperties":false}}}
    ])");
  return body;
}

void Cycle(json::Value& body, int turn, int count = 1) {
  auto assistant = json::Value::object();
  assistant["role"] = "assistant";
  assistant["content"] = nullptr;
  assistant["tool_calls"] = json::Value::array();
  for (int index = 0; index < count; ++index) {
    auto call = json::Value::object();
    call["id"] = "call_" + std::to_string(turn) + "_" + std::to_string(index);
    call["type"] = "function";
    call["function"]["name"] = index % 2 ? "list_files" : "read_file";
    call["function"]["arguments"] = R"({"path":"example.txt"})";
    assistant["tool_calls"].push_back(std::move(call));
  }
  body["messages"].push_back(std::move(assistant));
  for (int index = 0; index < count; ++index) {
    auto result = json::Value::object();
    result["role"] = "tool";
    result["tool_call_id"] =
        "call_" + std::to_string(turn) + "_" + std::to_string(index);
    result["content"] = "Public example result " + std::to_string(turn);
    body["messages"].push_back(std::move(result));
  }
}

void ToolLoops(const q::QwenTokenizer& tokenizer) {
  for (const bool parallel : {true, false}) {
    auto body = Seed();
    body["parallel_tool_calls"] = parallel;
    q::QwenIncrementalTokenizer cache;
    std::size_t hits = 0;
    for (int turn = 0; turn < 10; ++turn) {
      if (turn)
        Cycle(body, turn);
      if (turn < 2)
        std::cout << "HTTP_BODY parallel=" << parallel << " turn=" << turn
                  << " " << body.dump() << '\n';
      const auto request = Parse(body);
      Require(request.constrained_tools &&
                  request.parallel_tool_calls == parallel &&
                  request.tool_choice == s::ChatRequest::ToolChoice::kAuto,
              "real parser lost tool controls");
      hits += Check(tokenizer, cache, request,
                    "P1_loop_" + std::to_string(parallel) + "_" +
                        std::to_string(turn),
                    turn ? "hit" : "no_entry")
                  .incremental;
    }
    Expect(hits == 9, "P1 ten-turn HTTP tool loop must hit 9/10");
    std::cout << "TOOL_LOOP parallel=" << parallel << " turns=10 hits=" << hits
              << " stats=" << cache.Statistics().dump() << '\n';
  }
  auto body = Seed();
  q::QwenIncrementalTokenizer cache;
  Check(tokenizer, cache, Parse(body), "P1_multi_seed", "no_entry");
  Cycle(body, 1, 2);
  Check(tokenizer, cache, Parse(body), "P1_multi_results", "hit");
  body["messages"].push_back(json::parse(
      R"({"role":"tool","tool_call_id":"call_extra","content":"extra"})"));
  Check(tokenizer, cache, Parse(body), "P1_extended_group",
        "tool_group_changed");
  Cycle(body, 2);
  body["parallel_tool_calls"] = false;
  Check(tokenizer, cache, Parse(body), "P1_new_group_toggle_parallel", "hit");
  Edit(body["messages"], 3,
       [](auto& result) { result["content"] = "edited result"; });
  Check(tokenizer, cache, Parse(body), "P1_edited_result", "history_changed");
  Edit(body["messages"], 2, [](auto& message) {
    Edit(message["tool_calls"], 0, [](auto& call) {
      call["function"]["arguments"] = R"({"path":"edited.txt"})";
    });
  });
  Check(tokenizer, cache, Parse(body), "P1_edited_arguments",
        "history_changed");

  body = Seed();
  Edit(body["tools"], 0, [](auto& tool) { tool["function"]["strict"] = true; });
  q::QwenIncrementalTokenizer strict;
  Check(tokenizer, strict, Parse(body), "P1_strict_seed", "no_entry");
  Cycle(body, 1);
  Check(tokenizer, strict, Parse(body), "P1_strict_append", "hit");
  Edit(body["tools"], 0,
       [](auto& tool) { tool["function"]["strict"] = false; });
  Check(tokenizer, strict, Parse(body), "P1_strict_changed", "options_changed");
  Cycle(body, 2);
  Check(tokenizer, strict, Parse(body), "P1_strict_reseed", "hit");

  for (const auto choice : {json::Value("required"), json::Value("none"),
                            json::parse(R"({"type":"function",
                              "function":{"name":"read_file"}})")}) {
    body = Seed();
    body["tool_choice"] = choice;
    q::QwenIncrementalTokenizer guarded;
    Check(tokenizer, guarded, Parse(body), "P1_choice_seed", "request_options");
    Cycle(body, 1);
    Check(tokenizer, guarded, Parse(body), "P1_choice_append",
          "request_options");
    Expect(guarded.Statistics().member_size("entries") == 0,
           "P1 guarded tool choices must not publish");
  }
  for (const auto format : {json::parse(R"({"type":"json_object"})"),
                            json::parse(R"({"type":"json_schema","json_schema":{
             "name":"answer","description":"Answer format","strict":true,
             "schema":{"type":"object","properties":{"answer":{"type":"string"}},
                       "required":["answer"],"additionalProperties":false}}})")}) {
    body = Seed();
    body["response_format"] = format;
    q::QwenIncrementalTokenizer guarded;
    Check(tokenizer, guarded, Parse(body), "P1_format_seed", "response_format");
    Cycle(body, 1);
    Check(tokenizer, guarded, Parse(body), "P1_format_append",
          "response_format");
    Expect(guarded.Statistics().member_size("entries") == 0,
           "P1 response formats must not publish");
  }
  auto internal = Parse(Seed());
  internal.json_tool_calls = true;
  q::QwenIncrementalTokenizer guarded;
  Check(tokenizer, guarded, internal, "P1_json_override", "request_options");

  auto a = Parse(Seed()), b = a;
  b.parallel_tool_calls = false;
  b.constrained_tools = false;
  Require(Full(tokenizer, a).tokens == Full(tokenizer, b).tokens,
          "grammar-only fields unexpectedly alter native Qwen tokens");
  b = a;
  b.tool_choice = s::ChatRequest::ToolChoice::kRequired;
  Require(Full(tokenizer, a).tokens != Full(tokenizer, b).tokens,
          "required choice must change the native Qwen instruction");
}

void Publication(const q::QwenTokenizer& tokenizer) {
  auto long_body = Seed(false);
  std::string long_text = "long";
  for (int index = 0; index < 100000; ++index)
    long_text += " a";
  Edit(long_body["messages"], 1,
       [&](auto& user) { user["content"] = long_text; });
  const auto a = Parse(long_body, "slow-A");
  auto next_a = a;
  next_a.messages.emplace_back(q::ChatRole::kAssistant, "done");
  next_a.messages.emplace_back(q::ChatRole::kUser, "continue");
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto b = Parse(Seed(false), "fast-B");
    bool injected = false;
    q::QwenIncrementalTokenizer cache([&](std::string_view) {
      if (!injected) {
        injected = true;
        Check(tokenizer, cache, b, "P2_fast_publishes_first", "no_entry");
      }
    });
    Check(tokenizer, cache, a, "P2_slow_finishes_second", "no_entry");
    Expect(
        cache.Statistics().member_size("entries") == 2 &&
            cache.Statistics().member_size("publication_skips") == 0,
        "P2 another client must not starve slow publication with free slots");
    Check(tokenizer, cache, next_a, "P2_slow_progress", "hit");
    b.messages.emplace_back(q::ChatRole::kAssistant, "done");
    b.messages.emplace_back(q::ChatRole::kUser, "continue");
    Check(tokenizer, cache, b, "P2_fast_progress", "hit");
    std::cout << "PUBLICATION free " << cache.Statistics().dump() << '\n';
  }
  bool injected = false;
  q::QwenIncrementalTokenizer same([&](std::string_view) {
    if (!injected) {
      injected = true;
      Check(tokenizer, same, next_a, "P2_same_newer_first", "no_entry");
    }
  });
  Check(tokenizer, same, a, "P2_same_older_second", "no_entry");
  Expect(same.Statistics().member_size("publication_skips") == 1,
         "P2 older same-client publication must be skipped");
  auto newer = next_a;
  newer.messages.emplace_back(q::ChatRole::kAssistant, "done again");
  newer.messages.emplace_back(q::ChatRole::kUser, "continue again");
  Check(tokenizer, same, newer, "P2_same_retained_newer", "hit");
  std::cout << "PUBLICATION same " << same.Statistics().dump() << '\n';
  injected = false;
  q::QwenIncrementalTokenizer bounded([&](std::string_view) {
    if (!injected) {
      injected = true;
      for (int index = 0; index < 4; ++index)
        Check(tokenizer, bounded,
              Parse(Seed(false), "fast-" + std::to_string(index)),
              "P2_fill_slots", "no_entry");
    }
  });
  Check(tokenizer, bounded, a, "P2_slow_full_slots", "no_entry");
  Expect(bounded.Statistics().member_size("entries") == 4 &&
             bounded.Statistics().member_size("evictions") == 1,
         "P2 bounded LRU must permit older independent publication");
  Check(tokenizer, bounded, next_a, "P2_slow_LRU_progress", "hit");
  std::cout << "PUBLICATION bounded " << bounded.Statistics().dump() << '\n';
}

void UserTags(const q::QwenTokenizer& tokenizer) {
  for (const std::string text :
       {"Explain the <tool_response> tag.", "<tool_response>incomplete",
        " \t<tool_response>wrapped</tool_response> suffix",
        "\xE3\x80\x80mention <tool_response> here"}) {
    auto body = Seed(false);
    Edit(body["messages"], 1, [&](auto& user) { user["content"] = text; });
    q::QwenIncrementalTokenizer cache;
    Check(tokenizer, cache, Parse(body), "P3_mention_seed", "no_entry");
    Expect(cache.Statistics().member_size("entries") == 1,
           "P3 valid user mention must seed the cache");
    body["messages"].push_back(
        json::parse(R"({"role":"assistant","content":"A tool result tag."})"));
    body["messages"].push_back(json::parse(
        R"({"role":"user","content":"Show another <tool_response> example."})"));
    Check(tokenizer, cache, Parse(body), "P3_mention_append", "hit");
  }
  for (const std::string text :
       {"<tool_response>only</tool_response>",
        " \t<tool_response>only</tool_response>\r\n",
        "\xE3\x80\x80<tool_response>only</tool_response>\xE3\x80\x80"}) {
    auto body = Seed(false);
    Edit(body["messages"], 1, [&](auto& user) { user["content"] = text; });
    const auto request = Parse(body);
    q::QwenIncrementalTokenizer empty;
    Check(tokenizer, empty, request, "P3_exact_wrapper_only", "no_entry",
          "No user query found in messages.");
    Expect(empty.Statistics().member_size("entries") == 0,
           "P3 invalid wrappers must not publish");
    q::QwenIncrementalTokenizer retained;
    Check(tokenizer, retained, Parse(Seed(false)), "P3_accepted_seed",
          "no_entry");
    Check(tokenizer, retained, request, "P3_wrapper_after_seed",
          "history_changed", "No user query found in messages.");
    auto no_messages = request;
    no_messages.messages.clear();
    Check(tokenizer, retained, no_messages, "P3_empty_after_seed",
          "invalid_request", "No messages provided");
  }
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  Require(stream.good(), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(stream), {}};
}

// Fixture creation is outside any timed region. Every size and identity check
// goes through the HTTP parser and the production native Qwen options/guard.
void Fixtures(const q::QwenTokenizer& tokenizer,
              const std::filesystem::path& previous,
              const std::filesystem::path& destination) {
  Require(!std::filesystem::exists(destination), "fixture output exists");
  const auto old = json::parse(Read(previous));
  auto output = json::Value::object();
  output["source"] = old.member_str("source");
  output["cases"] = json::Value::array();
  for (const auto& plain : old.find("cases")->items()) {
    const auto target = plain.member_size("context_tokens");
    if (target != 32768 && target != 100000)
      continue;
    auto control = plain;
    control["shape"] = "plain";
    output["cases"].push_back(control);
    auto seed = Seed();
    seed["debug"] = *plain.find("seed")->find("debug");
    const auto& plain_messages =
        Field(Field(plain, "seed"), "messages").items();
    Edit(seed["messages"], 0,
         [&](auto& system) { system = plain_messages[0]; });
    const auto& public_text = Field(plain_messages[1], "content").str();
    const auto fit = [&](json::Value& body, std::size_t size,
                         std::string_view padding) {
      const auto set_content = [&](const std::string& text) {
        Edit(body["messages"], body["messages"].size() - 1,
             [&](auto& message) { message["content"] = text; });
      };
      set_content("");
      const auto overhead = Full(tokenizer, Parse(body)).tokens.size();
      Require(overhead + 64 < size, "fixture has insufficient content room");
      q::TokenizerOptions options;
      options.parse_special_tokens = false;
      auto tokens = tokenizer.Encode(padding, options);
      tokens.resize(std::min(tokens.size(), size - overhead - 64));
      auto content = tokenizer.Decode(tokens);
      set_content(content);
      auto count = Full(tokenizer, Parse(body)).tokens.size();
      Require(count < size, "fixture truncated content exceeds target");
      content += " a";
      set_content(content);
      Require(Full(tokenizer, Parse(body)).tokens.size() == count + 1,
              "fixture padding unit is not one token");
      for (++count; count < size; ++count)
        content += " a";
      set_content(content);
      Require(Full(tokenizer, Parse(body)).tokens.size() == size,
              "fixture size differs through full preparation");
    };
    fit(seed, target, public_text);
    auto next = seed;
    Cycle(next, 1);
    fit(next, target + 300, public_text);
    auto tool = json::Value::object();
    tool["shape"] = "tool_loop";
    tool["context_tokens"] = target;
    tool["appended_tokens"] = 300;
    tool["tool_result_tokens"] =
        tokenizer
            .Encode(Field(next["messages"].items().back(), "content").str())
            .size();
    tool["seed"] = seed;
    tool["next"] = next;
    output["cases"].push_back(std::move(tool));
  }
  Require(output["cases"].size() == 4, "fixture contexts absent");
  for (const auto& pair : output["cases"].items()) {
    q::QwenIncrementalTokenizer cache;
    const auto seed = Parse(Field(pair, "seed"));
    const auto next = Parse(Field(pair, "next"));
    const auto a = Check(tokenizer, cache, seed, "fixture_seed", "no_entry");
    const auto b = Check(tokenizer, cache, next, "fixture_next", "hit");
    Require(a.tokens.size() == pair.member_size("context_tokens") &&
                b.tokens.size() == a.tokens.size() + 300,
            "fixture net appended size differs");
    std::cout << "FIXTURE shape=" << pair.member_str("shape")
              << " seed_tokens=" << a.tokens.size()
              << " next_tokens=" << b.tokens.size() << " tool_result_tokens="
              << pair.member_size("tool_result_tokens")
              << " stats=" << cache.Statistics().dump() << '\n';
  }
  Require(failures == 0, "fixture route failure");
  std::ofstream stream(destination, std::ios::binary);
  stream << output.dump() << '\n';
  Require(stream.good(), "cannot write fixtures");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const char* model = std::getenv("GUFO_QWEN_GGUF");
    if (!model || !*model) {
      std::cout << "SKIP: set GUFO_QWEN_GGUF\n";
      return 77;
    }
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenHeaderFile(model, &error);
    Require(reader != nullptr, error);
    const auto tokenizer = q::QwenTokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, error);
    Require(q::QwenChatTemplate::ValidateGgufTemplate(*reader, &error), error);
    if (argc == 4 && std::string_view(argv[1]) == "--fixtures")
      Fixtures(*tokenizer, argv[2], argv[3]);
    else {
      Require(argc == 1, "usage: [--fixtures previous.json destination.json]");
      ToolLoops(*tokenizer);
      Publication(*tokenizer);
      UserTags(*tokenizer);
    }
    std::cout << "HTTP_FIX turns=" << turns
              << " token_comparisons=" << comparisons
              << " mismatches=0 rejected_identically=" << rejections
              << " route_failures=" << failures << '\n';
    return failures ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
