#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/vision/prompt.hpp"

// This checker is committed and run before the incremental implementation.
#if __has_include("src/models/qwen/incremental_tokenize.hpp")
#include "src/models/qwen/incremental_tokenize.hpp"
#define GUFO_HAS_INCREMENTAL_TOKENIZE 1
#endif

namespace {
namespace q = gufo::tokenization;
namespace json = gufo::json;
using Clock = std::chrono::steady_clock;
using Role = q::ChatRole;
using Message = q::ChatMessage;
constexpr std::uint32_t kContext = 1048576;
bool baseline = false;
bool broken_join = false;

void Require(bool condition, const std::string& detail) {
  if (!condition)
    throw std::runtime_error(detail);
}

struct Input {
  std::string client{"A"};
  std::vector<Message> messages{{Role::kSystem, "Be precise."},
                                {Role::kUser, "hello"}};
  std::vector<q::ChatTool> tools;
  q::ChatTemplateOptions options;
  std::uint32_t context{kContext};
  std::string guard;
  Clock::time_point now{Clock::now()};
};

struct Result {
  std::vector<q::TokenId> tokens;
  std::size_t cache_prefix_tokens{0};
  bool incremental{false};
  std::string reason;
  std::size_t reused_tokens{0};
  std::size_t tokenized_tokens{0};
};

Result Full(const q::QwenTokenizer& tokenizer, const Input& input) {
  auto prompt = gufo::models::qwen::vision::Prepare(
      tokenizer, input.messages, input.tools, input.options, {}, input.context);
  Result result;
  result.tokens = std::move(prompt.tokens);
  result.cache_prefix_tokens = prompt.stable_prefix_tokens;
  result.reason = "baseline";
  result.tokenized_tokens = result.tokens.size();
  return result;
}

class Candidate {
public:
  Result Prepare(const q::QwenTokenizer& tokenizer, const Input& input) {
    Result result;
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
    if (!baseline) {
      auto prepared =
          cache_.Prepare(tokenizer, input.client, input.messages, input.tools,
                         input.options, input.context, input.guard, input.now);
      result = {std::move(prepared.tokens), prepared.cache_prefix_tokens,
                prepared.incremental,       std::string(prepared.reason),
                prepared.reused_tokens,     prepared.tokenized_tokens};
    } else
#endif
    {
      result = Full(tokenizer, input);
    }
    if (broken_join) {
      // Deliberate defect: keep the previous generation header at the join.
      if (!previous_.empty())
        result.tokens.insert(result.tokens.begin(), previous_.begin(),
                             previous_.end());
      previous_ = result.tokens;
    }
    return result;
  }

  json::Value Statistics() const {
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
    if (!baseline)
      return cache_.Statistics();
#endif
    auto value = json::Value::object();
    value["implementation"] = "original_full_path";
    return value;
  }

private:
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
  q::QwenIncrementalTokenizer cache_;
#endif
  std::vector<q::TokenId> previous_;
};

struct Totals {
  std::size_t turns{0};
  std::size_t comparisons{0};
  std::size_t rejected{0};
  std::size_t hits{0};
  std::size_t reused{0};
  std::size_t tokenized{0};
  std::map<std::string, std::size_t> full;
  std::mutex mutex;

  json::Value Json() const {
    auto value = json::Value::object();
    value["turns"] = turns;
    value["token_comparisons"] = comparisons;
    value["mismatches"] = 0;
    value["rejected_identically"] = rejected;
    value["incremental_hits"] = hits;
    value["tokens_reused"] = reused;
    value["tokens_tokenized"] = tokenized;
    auto reasons = json::Value::object();
    for (const auto& [reason, count] : full)
      reasons[reason] = count;
    value["full_path_reasons"] = std::move(reasons);
    return value;
  }
};

Result Check(const q::QwenTokenizer& tokenizer, Candidate& candidate,
             const Input& input, const std::string& label, Totals& totals,
             std::string_view expected_reason = {}) {
  Result expected;
  std::string expected_error;
  try {
    expected = Full(tokenizer, input);
  } catch (const std::exception& error) {
    expected_error = error.what();
  }
  Result actual;
  std::string actual_error;
  try {
    actual = candidate.Prepare(tokenizer, input);
  } catch (const std::exception& error) {
    actual_error = error.what();
  }
  Require(actual_error == expected_error,
          label + ": error differs: full=" + expected_error +
              " candidate=" + actual_error);
  const std::lock_guard lock(totals.mutex);
  ++totals.turns;
  if (!expected_error.empty()) {
    ++totals.rejected;
    return actual;
  }
  const auto common = std::min(expected.tokens.size(), actual.tokens.size());
  for (std::size_t index = 0; index < common; ++index) {
    Require(expected.tokens[index] == actual.tokens[index],
            label + ": token mismatch at " + std::to_string(index) +
                " full=" + std::to_string(expected.tokens[index]) +
                " candidate=" + std::to_string(actual.tokens[index]));
    ++totals.comparisons;
  }
  Require(expected.tokens.size() == actual.tokens.size(),
          label + ": token length mismatch: full=" +
              std::to_string(expected.tokens.size()) +
              " candidate=" + std::to_string(actual.tokens.size()));
  Require(expected.cache_prefix_tokens == actual.cache_prefix_tokens,
          label + ": serving cache-prefix count differs");
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
  if (!baseline && !expected_reason.empty())
    Require(actual.reason == expected_reason, label + ": expected route " +
                                                  std::string(expected_reason) +
                                                  ", got " + actual.reason);
#else
  static_cast<void>(expected_reason);
#endif
  if (actual.incremental)
    ++totals.hits;
  else
    ++totals.full[actual.reason];
  totals.reused += actual.reused_tokens;
  totals.tokenized += actual.tokenized_tokens;
  return actual;
}

void Append(Input& input, std::string answer = "answer",
            std::string user = "next") {
  input.messages.emplace_back(Role::kAssistant, std::move(answer));
  input.messages.emplace_back(Role::kUser, std::move(user));
}

void Sequence(const q::QwenTokenizer& tokenizer, const std::string& label,
              const std::vector<Input>& inputs, Totals& totals) {
  Candidate candidate;
  for (std::size_t index = 0; index < inputs.size(); ++index)
    Check(tokenizer, candidate, inputs[index],
          label + "/" + std::to_string(index), totals);
  std::cout << "PASS " << label << " requests=" << inputs.size() << '\n';
}

void BoundaryChecks(const q::QwenTokenizer& tokenizer) {
  Require(
      tokenizer.Encode("hello") !=
          [&] {
            auto tokens = tokenizer.Encode("hel");
            auto tail = tokenizer.Encode("lo");
            tokens.insert(tokens.end(), tail.begin(), tail.end());
            return tokens;
          }(),
      "F01: real vocabulary must expose an unsafe ordinary-text join");
  const auto start_id = tokenizer.FindSpecialToken("<|im_start|>");
  const auto end_id = tokenizer.FindSpecialToken("<|im_end|>");
  Require(start_id && end_id, "F21: ChatML specials exist");
  Require(
      tokenizer.Encode("<|im_start|>") == std::vector<q::TokenId>{*start_id} &&
          tokenizer.Encode("<|im_end|>") == std::vector<q::TokenId>{*end_id},
      "F21: framing specials are indivisible");
  for (const auto& text :
       {std::string("hello \r\n \n"), std::string("e\xCC\x81 \xE4\xB8\xAD"),
        std::string(4096, 'a'), std::string(4096, ' ')}) {
    const std::vector<Message> messages{
        {Role::kUser, text}, {Role::kAssistant, text}, {Role::kUser, "next"}};
    std::vector<q::ContentSpan> spans;
    auto rendered = q::QwenChatTemplate::Render(messages, {}, {}, nullptr,
                                                nullptr, nullptr, &spans);
    Require(rendered.has_value(), "F21: boundary fixture renders");
    const auto whole = q::QwenChatTemplate::EncodeRendered(
        tokenizer, *rendered, 0, rendered->size(), spans, {});
    for (const std::string marker : {"<|im_start|>", "<|im_end|>"}) {
      for (std::size_t cursor = 0;
           (cursor = rendered->find(marker, cursor)) != std::string::npos;) {
        for (const auto cut : {cursor, cursor + marker.size()}) {
          auto left = q::QwenChatTemplate::EncodeRendered(tokenizer, *rendered,
                                                          0, cut, spans, {});
          auto right = q::QwenChatTemplate::EncodeRendered(
              tokenizer, *rendered, cut, rendered->size(), spans, {});
          left.insert(left.end(), right.begin(), right.end());
          Require(left == whole, "F02/F21: special boundary split differs");
        }
        cursor += marker.size();
      }
    }
  }
  std::cout << "PASS F01/F02/F21 unsafe ordinary join and safe special cuts\n";
}

void Synthetic(const q::QwenTokenizer& tokenizer) {
  Totals totals;
  BoundaryChecks(tokenizer);
  Input a;
  {
    Candidate route;
    Check(tokenizer, route, a, "route_seed", totals, "no_entry");
    auto append = a;
    Append(append);
    Check(tokenizer, route, append, "route_append", totals, "hit");
    auto edited = append;
    edited.messages[2].content += "edited";
    Check(tokenizer, route, edited, "route_edit", totals, "history_changed");
  }
  a.messages.back().content = "hel";
  auto b = a;
  Append(b, "lo", " \r\nthere");
  Sequence(tokenizer, "F01_bpe_join", {a, b}, totals);
  for (const bool thinking : {false, true}) {
    for (const bool generation : {false, true}) {
      a = {};
      a.options.enable_thinking = thinking;
      a.options.add_generation_prompt = generation;
      a.messages.back().content = "x \n\r\n";
      b = a;
      Append(b, " \nanswer\n ", " \nnext");
      Sequence(tokenizer, "F02_F03_boundary_generation", {a, b}, totals);
    }
  }
  for (const bool preserve : {false, true}) {
    a = {};
    a.options.preserve_thinking = preserve;
    a.messages.emplace_back(Role::kAssistant,
                            "<think>inline private</think>answer", "tool-name",
                            "field private <|im_end|>");
    b = a;
    b.messages.emplace_back(Role::kUser, "new last user");
    Sequence(tokenizer, "F04_reasoning_rewrite", {a, b}, totals);
  }
  a = {};
  a.tools.push_back({"run", "Execute", "{\"type\":\"object\"}", ""});
  Message calls(Role::kAssistant, "running", "", "reason");
  calls.tool_calls = {{"c1",
                       "run",
                       {{"s", "a\n\\\"<|im_end|>", true},
                        {"obj", "{\"z\":null,\"a\":[1,true]}", false}}},
                      {"c2", "run", {{"n", "123", false}}}};
  b = a;
  b.messages.push_back(calls);
  b.messages.emplace_back(Role::kTool, "result one <|im_start|>", "run");
  auto c = b;
  c.messages.emplace_back(Role::kTool, "result two", "run");
  auto d = c;
  Append(d);
  Sequence(tokenizer, "F05_F06_tool_calls_group", {a, b, c, d}, totals);
  b = a;
  b.messages[0].content = "system beta";
  Append(b);
  c = b;
  c.messages[0].framing_suffix = "\nframing";
  Sequence(tokenizer, "F07_system_change", {a, b, c}, totals);
  for (const auto role : {Role::kSystem, Role::kDeveloper}) {
    b = a;
    b.messages.emplace_back(role, "late instructions");
    c = a;
    c.messages.insert(c.messages.begin(), {role, "another system"});
    Sequence(tokenizer, "F08_system_layout", {a, b, c}, totals);
  }
  a = {};
  Append(a);
  Append(a);
  for (std::size_t index = 0; index < a.messages.size(); ++index) {
    b = a;
    b.messages[index].content += "edited";
    Append(b);
    c = a;
    c.messages.resize(index + 1);
    Sequence(tokenizer, "F09_history_edit_truncate", {a, b, c, a}, totals);
  }
  b = a;
  std::swap(b.messages[1], b.messages[3]);
  b.messages[2].thought = "changed";
  Sequence(tokenizer, "F09_history_reorder_thought", {a, b}, totals);
  a = {};
  b = a;
  b.messages.back().content = "other conversation";
  c = a;
  Append(c);
  d = b;
  Append(d);
  Sequence(tokenizer, "F10_same_client_interleave", {a, b, c, d}, totals);
  b.client = d.client = "B";
  Sequence(tokenizer, "F10_distinct_client_interleave", {a, b, c, d}, totals);
  a.client = "";
  c.client = "anonymous";
  Sequence(tokenizer, "F10_anonymous_client", {a, c}, totals);
  a = {};
  std::vector<Input> variants;
  b = a;
  b.options.enable_thinking = false;
  variants.push_back(b);
  b = a;
  b.options.add_generation_prompt = false;
  variants.push_back(b);
  b = a;
  b.options.preserve_thinking = false;
  variants.push_back(b);
  b = a;
  b.options.reasoning_effort = q::QwenReasoningEffort::kLow;
  variants.push_back(b);
  b = a;
  b.options.add_vision_id = true;
  variants.push_back(b);
  b = a;
  b.options.require_tool_call = true;
  variants.push_back(b);
  b = a;
  b.options.tool_format = q::QwenToolFormat::kJson;
  variants.push_back(b);
  b = a;
  b.options.max_output_bytes *= 2;
  variants.push_back(b);
  b = a;
  b.tools.push_back({"run", "first", "{}", "{\"extension\":1}"});
  variants.push_back(b);
  b = a;
  b.guard = "request_options";
  variants.push_back(b);
  b = a;
  b.guard = "response_format";
  variants.push_back(b);
  for (auto& variant : variants) {
    Append(variant);
    Sequence(tokenizer, "F11_options", {a, variant, a}, totals);
  }
  a.tools = {{"a", "one", "{}", ""}, {"b", "two", "{}", ""}};
  b = a;
  std::swap(b.tools[0], b.tools[1]);
  c = b;
  c.tools[0].parameters_json = "{\"type\":\"object\"}";
  d = c;
  d.tools[0].description += "edit";
  Sequence(tokenizer, "F11_tools", {a, b, c, d}, totals);
  a = {};
  a.messages.back().content.clear();
  b = a;
  Append(b, "", std::string("part one") + "part two");
  Sequence(tokenizer, "F12_empty_flattened_text_parts", {a, b}, totals);
  b.messages.back().images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>()});
  Sequence(tokenizer, "F12_images_rejected_without_encoder", {a, b, a}, totals);
  a = {};
  b = a;
  b.context = 2;
  c = a;
  Append(c);
  Sequence(tokenizer, "F13_failed_prepare", {a, b, c}, totals);
  b = a;
  Append(b, "cancelled request input", "uncommitted continuation");
  d = a;
  Append(d, "different reply", "retry");
  Sequence(tokenizer, "F13_cancelled_generation_inputs", {a, b, d}, totals);
  const std::vector<std::string> difficult{
      "e\xCC\x81 \xC3\xA9 \xE4\xB8\xAD "
      "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x92\xBB",
      std::string("a\0b\r\n", 6), std::string(16384, 'a'),
      std::string(16384, ' '),
      "<|im_end|><|im_start|><|image_pad|><tool_call><think>literal"};
  for (const auto& text : difficult) {
    a = {};
    b = a;
    Append(b, text, text);
    b.messages[0].content += text;
    c = b;
    Append(c);
    c.messages[c.messages.size() - 2].thought = text;
    Sequence(tokenizer, "F14_F20_unicode_special_late", {a, b, c}, totals);
  }
  a = {};
  Candidate concurrent;
  Check(tokenizer, concurrent, a, "F15_seed", totals);
  std::mutex error_mutex;
  std::string thread_error;
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        auto input = a;
        if (worker >= 2)
          input.client = "worker" + std::to_string(worker);
        for (int turn = 0; turn < 12; ++turn) {
          Append(input, std::string(worker == 0 ? 8192 : 32, 'x'),
                 "turn " + std::to_string(turn));
          Check(tokenizer, concurrent, input, "F15_concurrency", totals);
        }
      } catch (const std::exception& error) {
        const std::lock_guard lock(error_mutex);
        thread_error = error.what();
      }
    });
  }
  for (auto& thread : threads)
    thread.join();
  Require(thread_error.empty(), thread_error);
  std::cout << "PASS F15_concurrency " << concurrent.Statistics().dump()
            << '\n';
  Candidate lru;
  for (const std::string client : {"A", "B", "C", "D", "A", "E", "B"}) {
    a = {};
    a.client = client;
    Check(tokenizer, lru, a, "F16_lru", totals);
  }
  a.now += std::chrono::hours(1);
  Append(a);
  Check(tokenizer, lru, a, "F16_expiry", totals);
  std::cout << "PASS F16_lru_expiry " << lru.Statistics().dump() << '\n';
  a = {};
  a.messages.back().content = std::string(9 * 1024 * 1024, ' ');
  a.options.max_output_bytes = 32 * 1024 * 1024;
  b = a;
  Append(b);
  Sequence(tokenizer, "F16_retained_byte_bound", {a, b}, totals);
  a = {};
  for (const auto dialect :
       {q::QwenTemplateDialect::kCyberTiel, q::QwenTemplateDialect::kQwen36}) {
    b = a;
    b.options.dialect = dialect;
    Append(b);
    Sequence(tokenizer, "F17_dialect", {a, b, a}, totals);
  }
  b = a;
  b.messages.clear();
  c = a;
  c.messages.resize(1);
  d = a;
  d.messages.back().content = "<tool_response>only</tool_response>";
  Sequence(tokenizer, "F18_invalid_no_user", {a, b, c, d, a}, totals);
  a = {};
  Append(a);
  a.messages.emplace_back(Role::kTool, "result");
  a.messages.emplace_back(Role::kUser, "runtime one");
  b = a;
  b.messages.back().content = "runtime two";
  c = a;
  c.messages.back().role = Role::kUser;
  c.messages.emplace_back(Role::kUser, "another user");
  Sequence(tokenizer, "F19_replaceable_user", {a, b, c}, totals);
  a = {};
  a.options.max_output_bytes = 512;
  b = a;
  b.messages.back().content = std::string(1000, ' ');
  c = a;
  c.tools.push_back({"x", std::string(1000, ' '), "{}", ""});
  Sequence(tokenizer, "F22_estimated_bound", {a, b, c}, totals);
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
  if (!baseline)
    Require(totals.hits > 0, "synthetic suite exercised no incremental hits");
#endif
  std::cout << "SYNTHETIC " << totals.Json().dump() << '\n';
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

Message RecordedMessage(const json::Value& value) {
  const auto role = value.member_str("role");
  Require(role == "user" || role == "assistant" || role == "toolResult" ||
              role == "tool" || role == "system" || role == "developer",
          "unsupported recorded role: " + role);
  Message message;
  message.role = role == "assistant"                      ? Role::kAssistant
                 : role == "system"                       ? Role::kSystem
                 : role == "developer"                    ? Role::kDeveloper
                 : role == "tool" || role == "toolResult" ? Role::kTool
                                                          : Role::kUser;
  message.name = value.member_str("name", value.member_str("toolName"));
  message.tool_call_id =
      value.member_str("tool_call_id", value.member_str("toolCallId"));
  message.thought = value.member_str("reasoning_content");
  const auto argument = [](const json::Value& args, Message::ToolCall& call) {
    Require(args.is_object(), "recorded tool arguments must be an object");
    for (const auto& [name, item] : args.members())
      call.arguments.push_back({name,
                                item.is_string() ? item.str() : item.dump(),
                                item.is_string()});
  };
  if (const auto* calls = value.find("tool_calls");
      calls && !calls->is_null()) {
    for (const auto& item : calls->items()) {
      const auto* function = item.find("function");
      Require(function != nullptr, "recorded call lacks function");
      Message::ToolCall call;
      call.id = item.member_str("id");
      call.name = function->member_str("name");
      const auto* args = function->find("arguments");
      if (args)
        argument(args->is_string() ? json::parse(args->str()) : *args, call);
      message.tool_calls.push_back(std::move(call));
    }
  }
  const auto* content = value.find("content");
  if (!content || content->is_null())
    return message;
  if (content->is_string()) {
    message.content = content->str();
    return message;
  }
  Require(content->is_array(), "recorded content must be text or parts");
  for (const auto& part : content->items()) {
    const auto type = part.member_str("type");
    if (type == "text" || type == "input_text")
      message.content += part.member_str("text");
    else if (type == "thinking")
      message.thought += part.member_str("thinking");
    else if (type == "toolCall") {
      Message::ToolCall call;
      call.id = part.member_str("id");
      call.name = part.member_str("name");
      if (const auto* args = part.find("arguments"))
        argument(*args, call);
      message.tool_calls.push_back(std::move(call));
    } else
      throw std::runtime_error("unsupported recorded part: " + type);
  }
  return message;
}

std::vector<std::filesystem::path> Sources(
    const std::filesystem::path& manifest,
    const std::filesystem::path& allowed) {
  const auto data = json::parse(Read(manifest));
  const auto root = std::filesystem::canonical(allowed);
  Require(std::filesystem::canonical(data.member_str("source_root")) == root,
          "manifest source_root is not the authorized public root");
  std::vector<std::filesystem::path> sources;
  for (const auto& item : data.find("sessions")->items()) {
    const auto source = std::filesystem::canonical(item.member_str("source"));
    Require(source.parent_path() == root && source.extension() == ".jsonl",
            "source outside authorized public root");
    Require(
        gufo::crypto::Sha256FileHex(source) == item.member_str("source_sha256"),
        "public source hash differs: " + source.string());
    sources.push_back(source);
  }
  return sources;
}

void Public(const q::QwenTokenizer& tokenizer,
            const std::vector<std::filesystem::path>& sources) {
  Totals totals;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    Input input;
    input.client = "public-" + std::to_string(index);
    input.messages.clear();
    Candidate candidate;
    std::ifstream file(sources[index], std::ios::binary);
    std::string line;
    std::size_t turn = 0;
    while (std::getline(file, line)) {
      if (line.empty())
        continue;
      const auto record = json::parse(line);
      const auto type = record.member_str("type");
      if (type == "session")
        continue;
      Require(type == "message",
              "unhandled record (compaction/branch?): " + type);
      const auto* value = record.find("message");
      Require(value != nullptr, "message record lacks message");
      auto message = RecordedMessage(*value);
      if (message.role == Role::kAssistant) {
        Check(tokenizer, candidate, input,
              sources[index].filename().string() + "/" + std::to_string(turn++),
              totals);
      }
      input.messages.push_back(std::move(message));
    }
    std::cout << "SESSION " << index + 1 << '/' << sources.size() << ' '
              << sources[index].filename().string() << " turns=" << turn
              << " stats=" << candidate.Statistics().dump() << std::endl;
  }
  auto summary = totals.Json();
  summary["manifest_entries"] = sources.size();
  auto unique = sources;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  summary["unique_sources"] = unique.size();
  std::cout << "PUBLIC " << summary.dump() << '\n';
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

void Timing(const q::QwenTokenizer& tokenizer,
            const std::vector<std::filesystem::path>& sources,
            const std::filesystem::path& fixture_path) {
  Require(!sources.empty(), "timing requires authorized public sources");
  std::ifstream file(sources.front(), std::ios::binary);
  std::string corpus, line;
  while (std::getline(file, line) && corpus.size() < 32768) {
    const auto record = json::parse(line);
    if (const auto* message = record.find("message"))
      corpus += RecordedMessage(*message).content + "\n";
  }
  Require(!corpus.empty(), "empty public timing corpus");
  const auto context_text = [&](std::size_t count) {
    q::TokenizerOptions content_options;
    content_options.parse_special_tokens = false;
    std::string text;
    while (tokenizer.Encode(text, content_options).size() < count + 256)
      text += corpus;
    auto tokens = tokenizer.Encode(text, content_options);
    tokens.resize(count);
    return tokenizer.Decode(tokens);
  };
  const auto size_content = [&](Input& input, std::size_t target) {
    input.messages.back().content.clear();
    const auto framing = Full(tokenizer, input).tokens.size();
    Require(target > framing + 32, "timing content has no sizing headroom");
    auto body_count = target - framing - 32;
    std::size_t size = 0;
    do {
      input.messages.back().content = context_text(body_count);
      size = Full(tokenizer, input).tokens.size();
      if (size <= target)
        break;
      Require(body_count > 32, "timing public prefix exceeds target");
      body_count -= 32;
    } while (true);
    if (size < target) {
      // Verify the padding unit through full preparation. Re-encoding a
      // truncated BPE prefix need not retain its original token count.
      input.messages.back().content += " a";
      Require(Full(tokenizer, input).tokens.size() == size + 1,
              "timing padding is not one token");
      for (std::size_t index = size + 1; index < target; ++index)
        input.messages.back().content += " a";
    }
    Require(Full(tokenizer, input).tokens.size() == target,
            "timing content does not reach exact target");
  };
  const auto body = [](const Input& input) {
    auto value = json::Value::object();
    value["model"] = "bubo";
    value["messages"] = json::Value::array();
    for (const auto& message : input.messages) {
      auto item = json::Value::object();
      item["role"] = std::string(q::ToString(message.role));
      item["content"] = message.content;
      value["messages"].push_back(std::move(item));
    }
    value["max_tokens"] = 32;
    value["temperature"] = 0;
    value["top_k"] = 1;
    value["top_p"] = 1;
    value["min_p"] = 0;
    value["seed"] = 1;
    value["stream"] = false;
    value["cache_prompt"] = true;
    value["reasoning_effort"] = "xhigh";
    value["chat_template_kwargs"]["enable_thinking"] = true;
    value["chat_template_kwargs"]["preserve_thinking"] = true;
    value["debug"]["timeline"] = true;
    value["debug"]["raw_output"] = true;
    return value;
  };
  auto fixtures = json::Value::object();
  fixtures["source"] = sources.front().string();
  fixtures["cases"] = json::Value::array();
  for (const std::size_t target : {600, 8192, 32768, 100000}) {
    Input seed;
    size_content(seed, target);
    Input next = seed;
    Append(next, "Acknowledged.", "");
    const auto seed_full = Full(tokenizer, seed);
    size_content(next, target + 300);
    const auto next_full = Full(tokenizer, next);
    Require(seed_full.tokens.size() == target &&
                next_full.tokens.size() - seed_full.tokens.size() == 300,
            "timing fixture cannot reach exact context/append sizes");
    if (!fixture_path.empty()) {
      auto fixture = json::Value::object();
      fixture["context_tokens"] = target;
      fixture["appended_tokens"] = 300;
      fixture["seed"] = body(seed);
      fixture["next"] = body(next);
      fixtures["cases"].push_back(std::move(fixture));
    }
    std::vector<double> full_times, incremental_times;
    std::size_t hits = 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (int pair = 0; pair < 2; ++pair) {
        Candidate candidate;
        candidate.Prepare(tokenizer, seed);
        const auto begin_a = Clock::now();
        const auto full = Full(tokenizer, next);
        const auto end_a = Clock::now();
        const auto incremental = candidate.Prepare(tokenizer, next);
        const auto end_b = Clock::now();
        Require(full.tokens == incremental.tokens &&
                    full.cache_prefix_tokens == incremental.cache_prefix_tokens,
                "timing token identity failure");
        const double a_ms =
            std::chrono::duration<double, std::milli>(end_a - begin_a).count();
        const double b_ms =
            std::chrono::duration<double, std::milli>(end_b - end_a).count();
        full_times.push_back(a_ms);
        incremental_times.push_back(b_ms);
        hits += incremental.incremental;
        auto sample = json::Value::object();
        sample["target"] = target;
        sample["repeat"] = repeat + 1;
        sample["pair"] = pair + 1;
        sample["full_ms"] = a_ms;
        sample["incremental_ms"] = b_ms;
        std::cout << "SAMPLE " << sample.dump() << '\n';
      }
    }
    auto summary = json::Value::object();
    summary["target"] = target;
    summary["context_tokens"] = seed_full.tokens.size();
    summary["appended_tokens"] =
        next_full.tokens.size() - seed_full.tokens.size();
    summary["full_median_ms"] = Median(full_times);
    summary["incremental_median_ms"] = Median(incremental_times);
    summary["saving_percent"] =
        100 * (1 - Median(incremental_times) / Median(full_times));
    summary["hits"] = hits;
    summary["samples_per_path"] = full_times.size();
    std::cout << "TIMING " << summary.dump() << std::endl;
  }
  if (!fixture_path.empty()) {
    Require(!std::filesystem::exists(fixture_path), "fixture output exists");
    std::ofstream output(fixture_path, std::ios::binary);
    output << fixtures.dump() << '\n';
    Require(output.good(), "cannot write operator fixtures");
    std::cout << "FIXTURES " << fixture_path.string() << '\n';
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string manifest, source_root, fixture_path;
    bool public_gate = false, timing = false, synthetic = true;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--baseline")
        baseline = true;
      else if (arg == "--broken-join")
        broken_join = true;
      else if (arg == "--public") {
        public_gate = true;
        synthetic = false;
      } else if (arg == "--timing") {
        timing = true;
        synthetic = false;
      } else if (arg == "--manifest" && index + 1 < argc)
        manifest = argv[++index];
      else if (arg == "--source-root" && index + 1 < argc)
        source_root = argv[++index];
      else if (arg == "--fixtures" && index + 1 < argc)
        fixture_path = argv[++index];
      else
        throw std::runtime_error("unknown/incomplete argument: " + arg);
    }
    const char* model = std::getenv("GUFO_QWEN_GGUF");
    if (!model || !*model) {
      std::cout << "SKIP: set GUFO_QWEN_GGUF\n";
      return 77;
    }
    std::string error;
    // Open exactly shard 1. No model loader or tensor/shard discovery is used.
    const auto reader = gufo::core::GgufReader::OpenHeaderFile(model, &error);
    Require(reader != nullptr, "GGUF open failed: " + error);
    const auto tokenizer = q::QwenTokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer load failed: " + error);
    Require(q::QwenChatTemplate::ValidateGgufTemplate(*reader, &error), error);
    std::cout << "CHECKER implementation="
#ifdef GUFO_HAS_INCREMENTAL_TOKENIZE
              << (baseline ? "forced_full" : "incremental")
#else
              << "absent_full"
#endif
              << " tokenizer=" << model << std::endl;
    if (synthetic)
      Synthetic(*tokenizer);
    if (public_gate || timing) {
      Require(!manifest.empty() && !source_root.empty(),
              "public manifest/root required");
      const auto sources = Sources(manifest, source_root);
      if (public_gate)
        Public(*tokenizer, sources);
      if (timing)
        Timing(*tokenizer, sources, fixture_path);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
