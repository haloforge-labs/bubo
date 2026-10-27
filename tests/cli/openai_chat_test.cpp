#include "src/cli/serve/openai_chat.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "src/cli/serve/behaviour_options.hpp"
#include "src/cli/serve/debug_request.hpp"
#include "src/cli/serve/debug_settings.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"

using gufo::tokenization::kImEnd;

namespace {

using namespace std::chrono_literals;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

bool Accepts(const gufo::sampling::JsonConstraint& grammar,
             std::string_view text) {
  auto state = grammar.Start();
  for (const unsigned char byte : text) {
    state = grammar.Advance(state, byte);
    if (state.empty())
      return false;
  }
  return grammar.Complete(state);
}

/// Streaming splits deltas differently from the buffered path, so content that
/// differs only in surrounding whitespace is the same response.
std::string Trimmed(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

/// Reports fixed prompt progress before delegating token generation.
class ProgressRequest final
    : public gufo::server::TextGenerationBackend::GenerationRequest {
public:
  using Backend = gufo::server::TextGenerationBackend;

  ProgressRequest(std::shared_ptr<GenerationRequest> inner,
                  std::vector<Backend::PromptProgress> progress,
                  std::optional<gufo::sampling::JsonConstraint::ToolFormat>
                      tool_format = {},
                  std::function<Backend::ReasoningEnd()> reasoning_end = {})
      : inner_(std::move(inner)),
        progress_(std::move(progress)),
        tool_format_(tool_format),
        reasoning_end_(std::move(reasoning_end)) {}

  Backend::Result Wait(const Backend::TokenCallback& on_token,
                       const Backend::ProgressCallback& on_progress,
                       const Backend::StartCallback& on_start) override {
    for (const auto& value : progress_) {
      if (on_progress && !on_progress(value)) {
        inner_->Cancel();
        break;
      }
    }
    return inner_->Wait(on_token, on_progress, on_start);
  }
  void Cancel() noexcept override { inner_->Cancel(); }
  Backend::ReasoningEnd CurrentReasoningEnd() const override {
    return reasoning_end_ ? reasoning_end_() : inner_->CurrentReasoningEnd();
  }
  std::optional<gufo::sampling::JsonConstraint::ToolFormat> ToolFormat()
      const override {
    return tool_format_;
  }

private:
  std::shared_ptr<GenerationRequest> inner_;
  std::vector<Backend::PromptProgress> progress_;
  const std::optional<gufo::sampling::JsonConstraint::ToolFormat> tool_format_;
  const std::function<Backend::ReasoningEnd()> reasoning_end_;
};

class FakeBackend final : public gufo::server::TextGenerationBackend {
public:
  std::optional<gufo::models::qwen38_flash_next::PrefillDiagTotals>
      prefill_diag;
  [[nodiscard]] std::string model_id() const override { return "test-model"; }
  [[nodiscard]] bool ready() const override { return true; }
  [[nodiscard]] SamplingDefaults sampling_defaults() const override {
    return defaults;
  }
  [[nodiscard]] gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning_defaults_value;
  }
  [[nodiscard]] InitialOutputState initial_output_state(
      const gufo::server::ChatRequest& request) const override {
    if (initial_output_state_override)
      return *initial_output_state_override;
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kReasoning
               : InitialOutputState::kContent;
  }
  [[nodiscard]] OutputMarkup output_markup() const override { return markup; }

  Result complete(std::string_view, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck&, const TokenCallback&,
                  std::string_view, const std::vector<std::string>&,
                  bool) override {
    return {};
  }

  Result chat(const gufo::server::ChatRequest& request, std::size_t max_tokens,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled,
              const TokenCallback& on_token) override {
    ++chat_calls;
    last_request = request;
    last_max_tokens = max_tokens;
    last_temperature = sampling.temperature;
    last_sampling = sampling;

    Result result;
    result.prefill_diag = prefill_diag;
    result.prompt_tokens = 7;
    result.cached_prompt_tokens = 5;
    result.prefill_tokens = 2;
    result.reasoning_tokens = reasoning_tokens;
    result.prefill_chunks = 1;
    result.queue_depth_at_submit = 3;
    result.client_queue_depth_at_submit = 1;
    result.resident_requests_at_admission = 2;
    result.requested_logical_concurrency = 4;
    result.physical_execution_width = 2;
    result.queue_ms = 1.25;
    result.prefill_ms = 2.5;
    result.decode_ms = 4.0;
    result.ttft_ms = 3.75;
    result.mean_inter_token_ms = 2.0;
    result.max_inter_token_ms = 2.5;
    result.execution_plan = "serial-fallback";
    result.cache_hit = cache_miss_reason.empty();
    result.cache_miss_reason = cache_miss_reason;
    result.cache_snapshots_skipped = snapshots_skipped;
    result.cache_snapshot_skip_reason = snapshot_skip_reason;
    if (!result.cache_hit) {
      result.cached_prompt_tokens = 0;
      result.prefill_tokens = result.prompt_tokens;
      result.cache_common_prefix_tokens = 2;
      result.cache_checkpoint_tokens = 5;
    }
    // Like the scheduler: the generated tag token's offset is known before
    // its piece reaches the consumer.
    result.reasoning_end.tracked = track_reasoning_end;
    live_reasoning_end_byte.store(kNoReasoningEnd);
    for (const std::string& piece : pieces) {
      if (track_reasoning_end && reasoning_end_piece == result.tokens.size()) {
        result.reasoning_end.byte = result.text.size();
        live_reasoning_end_byte.store(result.text.size());
      }
      if ((is_cancelled && is_cancelled()) || (on_token && !on_token(piece))) {
        result.cancelled = true;
        result.finish_reason = FinishReason::kCancelled;
        return result;
      }
      result.text += piece;
      result.tokens.push_back(
          static_cast<gufo::tokenization::TokenId>(result.tokens.size()));
      if (block_after_first_piece &&
          result.tokens.size() == static_cast<std::size_t>(1)) {
        {
          const std::lock_guard<std::mutex> lock(mutex);
          first_piece_emitted = true;
        }
        condition.notify_all();
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return released; });
      }
    }
    result.completion_tokens = result.tokens.size();
    result.finish_reason = finish_reason;
    result.termination_cause = termination_cause;
    result.stop_sequence = stop_sequence;
    completed.store(true);
    return result;
  }

  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, bool stream_output) override {
    if (reject_on_start.has_value()) {
      throw gufo::server::TextGenerationError(*reject_on_start,
                                              "injected admission rejection");
    }
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                          is_cancelled, stream_output),
        progress, tool_format, [this] { return CurrentReasoningEnd(); });
  }

  [[nodiscard]] ReasoningEnd CurrentReasoningEnd() const {
    const auto byte = live_reasoning_end_byte.load();
    return {.tracked = track_reasoning_end,
            .byte = byte == kNoReasoningEnd ? std::nullopt
                                            : std::optional<std::size_t>(byte)};
  }

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  [[nodiscard]] std::optional<std::size_t> count_chat_tokens(
      const gufo::server::ChatRequest&) const override {
    return 7;  // The fixed prompt-token count reported by chat().
  }
  [[nodiscard]] bool supports_images() const override { return image_support; }
  bool image_support{true};

  bool WaitForFirstPiece() {
    std::unique_lock<std::mutex> lock(mutex);
    return condition.wait_for(lock, 2s, [&] { return first_piece_emitted; });
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    condition.notify_all();
  }

  std::vector<std::string> pieces;
  /// Token-level reasoning end (L-REL-8): when tracked, the piece at this
  /// index is the generated tag token; other tag text is ordinary text.
  bool track_reasoning_end{false};
  std::optional<std::size_t> reasoning_end_piece;
  static constexpr std::size_t kNoReasoningEnd = static_cast<std::size_t>(-1);
  std::atomic<std::size_t> live_reasoning_end_byte{kNoReasoningEnd};
  std::vector<PromptProgress> progress;
  std::optional<gufo::sampling::JsonConstraint::ToolFormat> tool_format;
  std::string cache_miss_reason;
  std::size_t snapshots_skipped = 0;
  std::string snapshot_skip_reason;
  FinishReason finish_reason{FinishReason::kStop};
  TerminationCause termination_cause{TerminationCause::kNone};
  std::string stop_sequence;
  bool block_after_first_piece{false};
  std::atomic<bool> completed{false};
  std::atomic<int> chat_calls{0};
  gufo::server::ChatRequest last_request;
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults_value;
  std::optional<InitialOutputState> initial_output_state_override;
  OutputMarkup markup;
  std::size_t last_max_tokens{0};
  std::size_t reasoning_tokens{0};
  float last_temperature{0.0F};
  gufo::sampling::SamplingConfig last_sampling;
  std::optional<gufo::server::TextGenerationErrorCode> reject_on_start;

private:
  std::mutex mutex;
  std::condition_variable condition;
  bool first_piece_emitted{false};
  bool released{false};
};

gufo::server::HttpRequest Request(
    std::string body,
    std::vector<std::pair<std::string, std::string>> headers = {}) {
  return {
      .method = "POST",
      .path = "/v1/chat/completions",
      .query = {},
      .body = std::move(body),
      .headers = std::move(headers),
      .is_cancelled = {},
  };
}

void DebugHooksRawOutputTest() {
  using gufo::json::Value;
  for (const auto* value : {"null", "1", "\"true\"", "[]", "{}"}) {
    bool timeline = false, raw = false;
    const auto body = gufo::json::parse(
        std::string("{\"debug\":{\"raw_output\":") + value + "}}");
    Expect(gufo::server::ParseDebugRequest(body, true, &timeline, true, &raw)
               .has_value(),
           "raw_output accepts only booleans");
  }
  for (const bool enabled : {false, true}) {
    bool timeline = false, raw = false;
    const auto body = gufo::json::parse(R"({"debug":{"raw_output":true}})");
    Expect(!gufo::server::ParseDebugRequest(body, enabled, &timeline, true,
                                            &raw) &&
               raw == enabled,
           "raw_output requires debug hooks");
    Expect(gufo::server::ParseDebugRequest(body, true, &timeline, false, &raw)
               .has_value(),
           "raw_output requires loopback");
    Expect(gufo::server::ParseDebugRequest(body, true, &timeline).has_value(),
           "other endpoints do not accept raw_output");
  }

  const std::string unfinished =
      "<tool_call>\n<function=read>\n<parameter=path>\ncaf\xC3\xA9.txt";
  const std::string complete =
      "<tool_call>\n<function=read>\n<parameter=path>\nx.txt\n"
      "</parameter>\n</function>\n</tool_call>";
  struct Case {
    std::string text;
    std::size_t calls;
    std::size_t dropped;
    std::size_t content_chars;
  };
  for (const auto& item :
       {Case{"Visible\n" + unfinished, 0, unfinished.size(), 8},
        Case{complete, 1, 0, 0},
        Case{complete + "\n" + unfinished, 1, unfinished.size(), 1},
        Case{"Visible", 0, 0, 7}}) {
    for (const bool stream : {false, true}) {
      for (const bool include_usage : {false, true}) {
        auto body = gufo::json::parse(R"({
          "model":"test-model","messages":[{"role":"user","content":"Read"}],
          "tools":[{"type":"function","function":{"name":"read",
            "parameters":{"type":"object","properties":{
              "path":{"type":"string"}},"required":["path"]}}}],
          "max_tokens":4096,"reasoning_effort":"high"})");
        body["stream"] = stream;
        body["stream_options"]["include_usage"] = include_usage;
        const std::string raw = "Réfléchir</think>\n\n" + item.text;
        const auto run = [&](bool hooks, int diagnostic) {
          FakeBackend backend;
          backend.tool_format =
              gufo::sampling::JsonConstraint::ToolFormat::kQwen;
          backend.finish_reason = FakeBackend::FinishReason::kLength;
          // Include split UTF-8, reasoning delimiters and call tags.
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          auto input = body;
          if (diagnostic >= 0)
            input["debug"]["raw_output"] = diagnostic != 0;
          auto request = Request(input.dump());
          request.debug_hooks = hooks;
          auto response = gufo::server::HandleOpenAiChat(request, backend);
          Expect(response.status == 200, "raw output request succeeds");
          std::vector<Value> events;
          std::string bytes;
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              bytes += chunk;
              if (chunk.starts_with("data: ") &&
                  !chunk.substr(6).starts_with("[DONE]"))
                events.push_back(gufo::json::parse(chunk.substr(6)));
              return true;
            });
          } else {
            bytes = response.body;
            events.push_back(gufo::json::parse(bytes));
          }
          // IDs and creation seconds are intentionally different per request.
          bytes = std::regex_replace(bytes, std::regex("chatcmpl-[^\"]+"),
                                     "chatcmpl-ID");
          bytes =
              std::regex_replace(bytes, std::regex("call_[^\"]+"), "call_ID");
          bytes = std::regex_replace(bytes, std::regex("\"created\":[0-9]+"),
                                     "\"created\":0");
          return std::pair{bytes, events};
        };
        const auto baseline = run(false, -1);
        Expect(run(true, -1).first == baseline.first &&
                   run(true, 0).first == baseline.first &&
                   run(false, 1).first == baseline.first,
               "absent, false or disabled raw_output leaves response bytes "
               "unchanged");
        const auto output = run(true, 1);
        const auto& last = output.second.back();
        const auto* debug = last.find("gufo_debug");
        Expect(debug && (!stream || last.contains("usage")),
               "raw_output is on the final usage chunk or buffered top level");
        const auto* diagnostic = debug->find("raw_output");
        Expect(
            diagnostic && diagnostic->member_str("text") == raw &&
                diagnostic->member_size("tokens") == raw.size() &&
                diagnostic->member_str("finish_reason") == "length",
            "raw_output retains exactly the decoded parser input and finish");
        const auto* parse = diagnostic->find("parse");
        Expect(
            parse && parse->member_size("reasoning_chars") == 9 &&
                parse->member_size("content_chars") == item.content_chars &&
                parse->member_size("parsed_tool_calls") == item.calls &&
                parse->member_size("unfinished_call_bytes") == item.dropped,
            "parse counts Unicode characters, calls and dropped suffix bytes");
        for (std::size_t i = 0; i + 1 < output.second.size(); ++i)
          Expect(!output.second[i].contains("gufo_debug"),
                 "intermediate chunks do not contain raw_output");
      }
    }
  }
}

void TestPrefillDiagnosticUsage() {
  for (const bool enabled : {false, true}) {
    FakeBackend backend;
    if (enabled) {
      backend.prefill_diag.emplace();
      backend.prefill_diag->passes = 2;
      backend.prefill_diag->table_reads = 9;
      backend.prefill_diag->table_bytes = 123;
      backend.prefill_diag->AddLayer(3, 64);
    }
    auto response = gufo::server::HandleOpenAiChat(Request(R"({
      "model":"test-model","messages":[{"role":"user","content":"hello"}],
      "max_tokens":2
    })"),
                                                   backend);
    Expect(response.status == 200, "diagnostic usage fake request succeeds");
    const auto body = gufo::json::parse(response.body);
    const auto* usage = body.find("usage");
    const auto* gufo = usage ? usage->find("gufo") : nullptr;
    const auto* diagnostic = gufo ? gufo->find("prefill_diag") : nullptr;
    Expect(bool(diagnostic) == enabled, "API diagnostic usage is opt-in");
    if (diagnostic) {
      Expect(diagnostic->member_size("passes") == 2 &&
                 diagnostic->member_size("table_reads") == 9 &&
                 diagnostic->member_size("table_bytes") == 123 &&
                 diagnostic->member_size("expert_bytes") == 192,
             "API retains additive measured totals");
    }
  }
}

void TestReasoningTokensInUsage() {
  for (const bool stream : {false, true}) {
    for (const std::size_t reasoning_tokens : {std::size_t{0}, std::size_t{3}}) {
      FakeBackend backend;
      backend.pieces = {"a", "b", "c", "d"};
      backend.reasoning_tokens = reasoning_tokens;
      auto response = gufo::server::HandleOpenAiChat(
          Request(std::string(R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":)") +
                  (stream ? "true" : "false") +
                  R"(,"stream_options":{"include_usage":true}})"),
          backend);
      Expect(response.status == 200, "reasoning usage request succeeds");
      gufo::json::Value usage = gufo::json::Value::object();
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          if (chunk.starts_with("data: ") &&
              !chunk.substr(6).starts_with("[DONE]")) {
            const auto event = gufo::json::parse(chunk.substr(6));
            if (const auto* found = event.find("usage"))
              usage = *found;
          }
          return true;
        });
      } else {
        usage = *gufo::json::parse(response.body).find("usage");
      }
      const auto* details = usage.find("completion_tokens_details");
      Expect(details != nullptr &&
                 details->member_size("reasoning_tokens") == reasoning_tokens,
             "Chat usage reports completion_tokens_details.reasoning_tokens "
             "in buffered and streamed responses");
    }
  }
}

void AppendToolDeltas(std::vector<gufo::json::Value>& calls,
                      const gufo::json::Value& deltas, bool stream) {
  for (const auto& delta : deltas.items()) {
    if (!stream) {
      calls.push_back(delta);
      continue;
    }
    const auto index = delta.member_size("index");
    Expect(index <= calls.size(), "tool indexes have no gaps");
    if (index == calls.size())
      calls.push_back(gufo::json::Value::object());
    auto& call = calls[index];
    if (!delta.member_str("id").empty()) {
      Expect(call.member_str("id").empty(), "tool ID is sent once");
      call["id"] = delta.member_str("id");
      call["type"] = delta.member_str("type");
    }
    if (const auto* function = delta.find("function"))
      for (const auto* key : {"name", "arguments"})
        call["function"][key] =
            call["function"].member_str(key) + function->member_str(key);
  }
}

void TestStreamingIsLive() {
  FakeBackend backend;
  backend.pieces = {"Hel<tool_call>", "lo"};
  backend.finish_reason =
      gufo::server::TextGenerationBackend::FinishReason::kLength;
  backend.block_after_first_piece = true;

  auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":2,
        "stream":true,
        "stream_options":{"include_usage":true}
      })"),
                                                 backend);
  Expect(response.status == 200, "Streaming request is accepted");
  Expect(static_cast<bool>(response.streaming_body),
         "Streaming request returns a streaming body");

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::string output;
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view chunk) {
      {
        const std::lock_guard<std::mutex> lock(output_mutex);
        output.append(chunk);
      }
      output_condition.notify_all();
      return true;
    });
  });

  Expect(backend.WaitForFirstPiece(), "Backend emits the first token");
  {
    std::unique_lock<std::mutex> lock(output_mutex);
    Expect(output_condition.wait_for(
               lock, 2s,
               [&] {
                 return output.find(R"("content":"Hel<tool_call>")") !=
                        std::string::npos;
               }),
           "First content delta is written promptly");
    Expect(!backend.completed.load(),
           "First content delta arrives before generation completes");
  }

  backend.Release();
  writer.join();

  Expect(output.find(R"("finish_reason":"length")") != std::string::npos,
         "Token limit is reported as finish_reason length");
  Expect(output.find(R"("prompt_tokens":7)") != std::string::npos,
         "Usage is emitted when requested");
  Expect(output.find(R"("cached_tokens":5)") != std::string::npos,
         "Usage reports transparently reused prompt tokens");
  Expect(output.find(R"("prefill_tokens":2)") != std::string::npos,
         "Usage reports actual prefill work");
  Expect(output.find(R"("prompt_n":2)") != std::string::npos,
         "Timings exclude cached tokens");
  Expect(output.find(R"("prompt_tokens_per_second":800)") != std::string::npos,
         "Usage throughput counts only tokens actually prefilled");
  Expect(output.find(R"("prefill_ms":2.5)") != std::string::npos,
         "Usage reports server prefill time");
  Expect(output.find(R"("decode_ms":4)") != std::string::npos,
         "Usage reports server decode time");
  Expect(
      output.find(R"("requested_logical_concurrency":4)") != std::string::npos,
      "Usage reports configured logical concurrency");
  Expect(
      output.find(R"("execution_plan":"serial-fallback")") != std::string::npos,
      "Usage reports the executed serving plan");
  Expect(output.ends_with("data: [DONE]\n\n"),
         "Stream terminates with the OpenAI DONE sentinel");
  Expect(response.stream_log &&
             response.stream_log->details.find("cached_tokens=5") !=
                 std::string::npos &&
             response.stream_log->details.find("finish=length") !=
                 std::string::npos,
         "Streaming completion retains request diagnostics");
}

void TestLessThanProseStreamsBeforeCompletion() {
  FakeBackend backend;
  backend.pieces = {"3 < 5 and x < y", " are comparisons."};
  backend.block_after_first_piece = true;
  auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model","messages":[{"role":"user","content":"explain"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
    "stream":true,"reasoning_effort":"none"
  })"),
                                                 backend);
  std::mutex mutex;
  std::condition_variable ready;
  std::string content;
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view part) {
      if (part == "data: [DONE]\n\n")
        return true;
      const auto event = gufo::json::parse(part.substr(6));
      {
        const std::lock_guard lock(mutex);
        for (const auto& choice : event.find("choices")->items()) {
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        }
      }
      ready.notify_all();
      return true;
    });
  });
  Expect(backend.WaitForFirstPiece(), "backend reaches the comparison prose");
  bool arrived;
  {
    std::unique_lock lock(mutex);
    arrived =
        ready.wait_for(lock, 2s, [&] { return content == backend.pieces[0]; });
  }
  const auto completed = backend.completed.load();
  backend.Release();
  writer.join();
  Expect(arrived && !completed,
         "all comparison prose streams while generation is still blocked");
  Expect(content == "3 < 5 and x < y are comparisons.",
         "comparison content stays exact");
}

void TestCachePromptOption() {
  for (bool stream : {false, true}) {
    for (const auto value : {"true", "false", "null", "0", "\"false\""}) {
      FakeBackend backend;
      auto response = gufo::server::HandleOpenAiChat(
          Request(
              std::string(
                  R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":)") +
              (stream ? "true" : "false") + R"(,"cache_prompt":)" + value +
              "}"),
          backend);
      // As in llama-server, null keeps the default.
      const bool valid = std::string_view(value) == "true" ||
                         std::string_view(value) == "false" ||
                         std::string_view(value) == "null";
      Expect(response.status == (valid ? 200 : 400),
             "cache_prompt accepts only JSON booleans or null");
      if (valid) {
        if (response.streaming_body)
          response.streaming_body([](std::string_view) { return true; });
        Expect(backend.last_request.cache_prompt ==
                   (std::string_view(value) != "false"),
               "cache_prompt reaches both buffered and streaming backends");
      }
    }
  }
}

void TestStreamingWithoutUsage() {
  for (const auto* options :
       {"", R"(,"stream_options":{"include_usage":false})"}) {
    FakeBackend backend;
    backend.pieces = {"ok"};
    auto response = gufo::server::HandleOpenAiChat(
        Request(
            std::string(
                R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true)") +
            options + "}"),
        backend);
    Expect(response.status == 200 && response.streaming_body,
           "Stream without usage is accepted");
    std::string output;
    response.streaming_body([&](std::string_view chunk) {
      output += chunk;
      return true;
    });
    Expect(output.find(R"("usage":)") == std::string::npos &&
               output.ends_with("data: [DONE]\n\n"),
           "Usage chunk is opt-in");
    Expect(
        output.find(R"("prompt_per_second":800)") != std::string::npos &&
            output.find(R"("cache_n":5)") != std::string::npos,
        "Terminal timings survive omitted or disabled usage on cached turns");
    Expect(response.stream_log &&
               response.stream_log->details.find("generated_tokens=1") !=
                   std::string::npos,
           "Request diagnostics do not depend on client usage preference");
  }
}

void TestUtf8Output() {
  const auto check = [](std::vector<std::string> pieces,
                        const std::string& expected, bool reasoning) {
    FakeBackend backend;
    backend.pieces = std::move(pieces);
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"hello"}]
    })");
    body["chat_template_kwargs"]["enable_thinking"] = reasoning;
    const auto field = reasoning ? "reasoning_content" : "content";
    const auto complete =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(complete.status == 200, "Unicode completion succeeds");
    const auto parsed = gufo::json::parse(complete.body);
    Expect(
        parsed.find("choices")->items().front().find("message")->member_str(
            field) == expected,
        "Non-streaming UTF-8 preserves scalars and replaces malformed bytes");
    body["stream"] = true;
    const auto stream =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(stream.status == 200 && stream.streaming_body,
           "Unicode stream succeeds");
    std::string text;
    stream.streaming_body([&](std::string_view chunk) {
      if (chunk == "data: [DONE]\n\n")
        return true;
      Expect(chunk.starts_with("data: "), "SSE has a data prefix");
      const auto event = gufo::json::parse(chunk.substr(6));
      for (const auto& choice : event.find("choices")->items()) {
        const auto* delta = choice.find("delta");
        if (delta)
          text += delta->member_str(field);
      }
      Expect(expected.starts_with(text) &&
                 (text.size() == expected.size() ||
                  (static_cast<unsigned char>(expected[text.size()]) & 0xC0) !=
                      0x80),
             "Every SSE event ends at a complete Unicode scalar");
      return true;
    });
    Expect(text == expected, "Streaming and non-streaming UTF-8 agree");
  };
  const std::string valid = "Aé中┌😀Z";
  const std::string replacement = "\xEF\xBF\xBD";
  for (const bool reasoning : {false, true}) {
    for (std::size_t split = 1; split < valid.size(); ++split)
      check({valid.substr(0, split), valid.substr(split)}, valid, reasoning);
    std::vector<std::string> bytes;
    for (char byte : valid)
      bytes.emplace_back(1, byte);
    check(bytes, valid, reasoning);
    check({"\xE2\x82", "X"}, replacement + "X", reasoning);
    check({"\xF0", "\x9F"}, replacement, reasoning);
    check({"\x80", "ok"}, replacement + "ok", reasoning);
    check({"\xC0\xAF"}, replacement + replacement, reasoning);
    check({"\xED", "\xA0\x80"}, replacement + replacement + replacement,
          reasoning);
    check({"\xF4\x90\x80\x80"},
          replacement + replacement + replacement + replacement, reasoning);
  }
}

void TestCachedPrefillMetrics() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})"),
      backend);
  Expect(response.status == 200, "Cached non-streaming response succeeds");
  const auto body = gufo::json::parse(response.body);
  const auto* usage = body.find("usage");
  const auto* timings = body.find("timings");
  Expect(body.find("metrics") == nullptr,
         "One timing schema prevents proxies from subtracting cached tokens "
         "twice");
  Expect(usage && usage->member_size("prompt_tokens") == 7,
         "Token usage includes cached tokens");
  Expect(timings && timings->member_size("prompt_n") == 2 &&
             timings->member_double("prompt_per_second") == 800 &&
             timings->member_double("prompt_per_token_ms") == 1.25,
         "Non-streaming timings report executed prefill work");

  gufo::server::TextGenerationBackend::Result cached;
  cached.prompt_tokens = cached.cached_prompt_tokens = 1024;
  cached.prefill_ms = 0.01;
  Expect(gufo::server::PrefillTokensPerSecond(cached) == 0,
         "Full cache hits cannot report artificial prefill throughput");
  cached.prefill_tokens = 10;
  cached.prefill_ms = 0;
  Expect(gufo::server::PrefillTokensPerSecond(cached) == 0,
         "Untimed work does not divide by zero");

  backend.cache_miss_reason = "prefix_changed";
  const auto miss = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"changed"}]})"),
      backend);
  const auto miss_body = gufo::json::parse(miss.body);
  const auto* miss_usage = miss_body.find("usage");
  const auto* metrics = miss_usage ? miss_usage->find("gufo") : nullptr;
  Expect(metrics &&
             metrics->member_str("cache_miss_reason") == "prefix_changed" &&
             metrics->member_size("cache_common_prefix_tokens") == 2 &&
             metrics->member_size("cache_checkpoint_tokens") == 5,
         "Usage explains cache misses without exposing prompt text");
  Expect(metrics->find("cache_snapshots_skipped") == nullptr, "no skipped copies: the fields are absent");
  // GitHub #9: a dropped checkpoint and the copies this request was refused are named in usage.gufo.
  backend.cache_miss_reason = "evicted_byte_capacity";
  backend.snapshots_skipped = 2;
  backend.snapshot_skip_reason = "skipped_capacity";
  const auto dropped = gufo::server::HandleOpenAiChat(
      Request(R"({"model":"test-model","messages":[{"role":"user","content":"again"}]})"), backend);
  const auto dropped_body = gufo::json::parse(dropped.body);
  const auto* dropped_metrics = dropped_body.find("usage") ? dropped_body.find("usage")->find("gufo") : nullptr;
  Expect(dropped_metrics && dropped_metrics->member_str("cache_miss_reason") == "evicted_byte_capacity" &&
             dropped_metrics->member_size("cache_snapshots_skipped") == 2 &&
             dropped_metrics->member_str("cache_snapshot_skip_reason") == "skipped_capacity",
         "Usage reports the true miss reason and the request's skipped snapshot copies with their reason");
}

void TestToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>",
  };

  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"weather in Rome"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"get_weather",
            "description":"Get weather",
            "parameters":{
              "type":"object",
              "properties":{"city":{"type":"string"}},
              "required":["city"]
            }
          }
        }],
        "tool_choice":"required",
        "stream":false
      })"),
                                                       backend);

  Expect(response.status == 200, "Tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Tool generation reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"get_weather")") != std::string::npos,
         "Tool name is translated to OpenAI format");
  Expect(response.body.find(R"(\"city\":\"Rome\")") != std::string::npos,
         "Tool arguments are translated to JSON");
  Expect(backend.last_request.tools.size() == 1,
         "Tool schema reaches the model backend");
  Expect(backend.last_request.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kRequired,
         "Required tool choice reaches the model backend");
}

void TestToolParameterCompatibility() {
  using gufo::json::Value;
  const std::pair<const char*, const char*> cases[] = {
      {R"({"name":"f"})", "{}"},
      {R"({"name":"f","parameters":null})", "{}"},
      {R"({"name":"f","parameters":{}})", "{}"},
      {R"({"name":"f","parametersJsonSchema":null})", "{}"},
      {R"({"name":"f","parametersJsonSchema":{"type":"object"}})",
       R"({"type":"object"})"},
      {R"({"name":"f","parameters":null,"parametersJsonSchema":{"type":"object"}})",
       R"({"type":"object"})"},
      {R"({"name":"f","parameters":{},"parametersJsonSchema":"ignored"})",
       "{}"},
  };
  for (bool flat : {false, true}) {
    for (bool stream : {false, true}) {
      for (const auto& [function_json, expected_parameters] : cases) {
        FakeBackend backend;
        backend.pieces = {
            "<tool_call>\n<function=f>\n</function>\n</tool_call>"};
        auto body = gufo::json::parse(R"({
          "model":"test-model","messages":[{"role":"user","content":"call f"}],
          "tool_choice":"required","tools":[]
        })");
        auto definition = Value::object();
        definition["type"] = "function";
        auto function = gufo::json::parse(function_json);
        if (flat) {
          for (const auto& [key, value] : function.members())
            definition.append_member(key, value);
        } else {
          definition["function"] = function;
        }
        body["tools"].push_back(std::move(definition));
        body["stream"] = stream;
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 200, "Compatible tool schema is accepted");
        std::string output = response.body;
        if (stream) {
          Expect(static_cast<bool>(response.streaming_body),
                 "Compatible tool request supports streaming");
          response.streaming_body([&](std::string_view chunk) {
            output += chunk;
            return true;
          });
        }
        Expect(output.find(R"("name":"f")") != std::string::npos &&
                   output.find(R"("finish_reason":"tool_calls")") !=
                       std::string::npos,
               "No-argument function returns a structured tool call");
        Expect(backend.last_request.tools.size() == 1,
               "Normalized tool reaches the backend");
        const auto& tool = backend.last_request.tools.front();
        Expect(tool.parameters_json == expected_parameters,
               "Missing/null schemas normalize and parameters take precedence");
        auto expected_function = Value::object();
        expected_function["name"] = "f";
        expected_function["parameters"] =
            gufo::json::parse(expected_parameters);
        auto expected_definition = Value::object();
        expected_definition["type"] = "function";
        expected_definition["function"] = expected_function;
        Expect(tool.definition_json == expected_definition.dump(),
               "Templates receive the normalized nested definition");
      }
    }
  }
}

void TestToolArgumentLoopStopOutput() {
  using Backend = gufo::server::TextGenerationBackend;
  const std::string open =
      "<tool_call>\n<function=execute_bash>\n<parameter=command>\n";
  const std::string period = "</function>\n</tool_call>\n" + open;
  const std::string complete = open + "printf ok\n</parameter>\n</function>\n</tool_call>\n";
  for (const bool stream : {false, true}) {
    for (const int choice : {0, 1, 2}) {
      for (const bool preceding_call : {false, true}) {
        for (const bool loop_cause : {false, true}) {
          FakeBackend backend;
          backend.finish_reason = Backend::FinishReason::kStop;
          backend.termination_cause = loop_cause ? Backend::TerminationCause::kToolArgumentLoop
                                                : Backend::TerminationCause::kNone;
          backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
          backend.pieces = {"Before.\n"};
          if (preceding_call) backend.pieces.push_back(complete);
          backend.pieces.push_back(open);
          for (int repeat = 0; repeat < 16; ++repeat) backend.pieces.push_back(period);
          auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"run a command"}],
            "tools":[{"type":"function","function":{"name":"execute_bash",
              "parameters":{"type":"object","properties":{"command":{"type":"string"}},
                "required":["command"],"additionalProperties":false}}}],
            "parallel_tool_calls":true,"chat_template_kwargs":{"enable_thinking":false}
          })");
          body["stream"] = stream;
          body["tool_choice"] = choice == 2
              ? gufo::json::parse(R"({"type":"function","function":{"name":"execute_bash"}})")
              : gufo::json::Value(choice == 1 ? "required" : "auto");
          const auto response = gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          if (!loop_cause && choice != 0 && !preceding_call) {
            // Ordinary EOS must still enforce required / forced-name tools.
            std::string wire = response.body;
            if (stream) response.streaming_body([&](std::string_view chunk) { wire += chunk; return true; });
            Expect(response.status == (stream ? 200 : 502) && wire.find("tool_choice_unsatisfied") != std::string::npos,
                   "ordinary EOS still rejects an unsatisfied required tool choice");
            continue;
          }
          Expect(response.status == 200, "loop stop succeeds for auto, required, forced name and parallel calls");
          std::string content, finish, error_code, error_message;
          std::size_t calls = 0;
          // Streamed calls arrive as deltas; the call cut by the stop is sent
          // before the stop is known, so it stays as an unfinished prefix.
          std::vector<gufo::json::Value> streamed_calls;
          const auto inspect = [&](const gufo::json::Value& output) {
            if (const auto* error = output.find("error")) {
              error_code = error->member_str("code");
              error_message = error->member_str("message");
              return;
            }
            if (output.find("choices")->items().empty()) return;
            const auto& selected = output.find("choices")->items().front();
            if (const auto* reason = selected.find("finish_reason"); reason && reason->is_string()) finish = reason->str();
            const auto* message = selected.find(stream ? "delta" : "message");
            if (const auto* emitted = message->find("tool_calls")) {
              if (stream) {
                AppendToolDeltas(streamed_calls, *emitted, true);
              } else {
                calls += emitted->size();
                for (const auto& call : emitted->items()) {
                  const auto* function = call.find("function");
                  Expect(function && function->member_str("name") == "execute_bash" &&
                             gufo::json::parse(function->member_str("arguments")).member_str("command") == "printf ok",
                         "only the preceding complete call survives parsing");
                }
              }
            }
            if (const auto* text = message->find("content"); text && text->is_string()) content += text->str();
          };
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              const auto start = chunk.find("data: ");
              if (start != std::string_view::npos && chunk.substr(start + 6).starts_with("{"))
                inspect(gufo::json::parse(chunk.substr(start + 6)));
              return true;
            });
          } else inspect(gufo::json::parse(response.body));
          if (stream) {
            // A call cut after part of its arguments was streamed cannot be
            // reported as a successful turn. Ordinary EOS and the loop stop
            // both end with the same error and no finish reason. The loop stop
            // uses a fixed message. TestToolArgumentLoopStopClosesStreamedCall
            // pins the streamed loop stop without a partial call on the wire.
            Expect(error_code == "generation_failed" && finish.empty(),
                   "a cut streamed call fails instead of finishing the turn");
            if (loop_cause)
              Expect(error_message ==
                         "generation stopped: the tool call repeated its arguments (loop breaker)",
                     "the loop stop error carries the fixed message");
            // Complete preceding calls stay delivered exactly as streamed.
            Expect(streamed_calls.size() >= static_cast<std::size_t>(preceding_call),
                   "complete preceding calls were streamed before the error");
            for (std::size_t i = 0; i < static_cast<std::size_t>(preceding_call); ++i)
              Expect(gufo::json::parse(streamed_calls[i].find("function")->member_str("arguments"))
                             .member_str("command") == "printf ok",
                     "the preceding complete call is intact");
            continue;
          }
          Expect(error_code.empty(), "tool-argument loop has no response error");
          Expect(finish == (loop_cause || !preceding_call ? "stop" : "tool_calls"),
                 "loop cause returns stop before has-tool-calls; ordinary EOS retains its finish contract");
          Expect(calls == static_cast<std::size_t>(preceding_call), "unfinished call is dropped and previous calls are kept");
          // The unchanged parser retains the newline after a preceding complete
          // call, in addition to the newline before it. Buffered ordinary EOS
          // reaches this check with the same bytes, so the loop cause cannot
          // alter content parsing.
          Expect(content == (preceding_call ? "Before.\n\n" : "Before.\n"), "content before the unfinished call is returned: stream=" +
                 std::to_string(stream) + " choice=" + std::to_string(choice) +
                 " preceding=" + std::to_string(preceding_call) + " loop=" + std::to_string(loop_cause) +
                 " actual=" + gufo::json::Value(content).dump());
        }
      }
    }
  }
}

void TestInvalidToolsFailBeforeGeneration() {
  const char* invalid[] = {
      "null",
      "42",
      R"({"function":{"name":"f"}})",
      R"({"type":42,"function":{"name":"f"}})",
      R"({"type":"custom","custom":{"name":"shell"}})",
      R"({"type":"function","function":null,"name":"f"})",
      R"({"type":"function","function":[],"name":"f"})",
      R"({"type":"function","function":{}})",
      R"({"type":"function","name":""})",
      R"({"type":"function","name":42})",
      R"({"type":"function","name":"f","parameters":"bad"})",
      R"({"type":"function","name":"f","parameters":[]})",
      R"({"type":"function","name":"f","parameters":false})",
      R"({"type":"function","name":"f","parametersJsonSchema":[]})",
  };
  for (bool stream : {false, true}) {
    for (bool valid_first : {false, true}) {
      for (const char* entry : invalid) {
        FakeBackend backend;
        auto body = gufo::json::parse(R"({
          "model":"test-model","messages":[{"role":"user","content":"use tools"}],
          "tools":[]
        })");
        if (valid_first)
          body["tools"].push_back(gufo::json::parse(
              R"({"type":"function","function":{"name":"valid","parameters":{}}})"));
        body["tools"].push_back(gufo::json::parse(entry));
        body["stream"] = stream;
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 400 &&
                   response.body.find("invalid_tools") != std::string::npos &&
                   !response.streaming_body && backend.chat_calls == 0,
               "Invalid tools fail before generation, including mixed lists");
      }
    }
  }
}

void TestResponsesClientCompatTolerances() {
  // Hosted tool types (Responses-only) are skipped, not rejected. Namespaces
  // group client-executed functions and flatten to the function list, while a
  // malformed function still fails the controls.
  {
    auto body = gufo::json::parse(R"({
      "tools":[
        {"type":"function","name":"exec","parameters":{"type":"object",
          "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
        {"type":"web_search","external_web_access":false},
        {"type":"namespace","name":"agents","tools":[{"type":"function",
          "name":"spawn"}]},
        {"type":"code_interpreter"}]})");
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses skips hosted tool types without failing");
    Expect(chat.tools.size() == 2 && chat.tools[0].name == "exec" &&
               chat.tools[1].name == "spawn",
           "Function tools survive, including those nested in namespaces");
  }
  {
    // A namespace is a client-side grouping, not a hosted tool: its functions
    // must reach the model with their schema and strictness intact.
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[{"type":"namespace","name":"crm",
                 "description":"Local customer tools","tools":[{"type":"function",
                 "name":"lookup","parameters":{"type":"object","properties":{},
                 "required":[],"additionalProperties":false},"strict":true}]}]})"),
               &chat) &&
               chat.tools.size() == 1 && chat.tools[0].name == "lookup" &&
               chat.tools[0].definition_json.find("\"strict\":true") !=
                   std::string::npos,
           "Namespace function tools are flattened with their definitions");
    // #643: a Responses function tool without `strict` (or with null) keeps its
    // schema as sent. Optional properties stay optional; nothing is forced
    // into `required` and no `strict` key is added.
    for (const char* strict : {"", R"(,"strict":null)"}) {
      gufo::server::ChatRequest optional;
      const auto schema =
          std::string(R"({"type":"object","properties":{"path":{"type":"string"},)"
                      R"("limit":{"type":"integer"}},"required":["path"]})");
      Expect(!gufo::server::ParseOpenAiResponseControls(
                 gufo::json::parse(R"({"tools":[{"type":"function","name":"read",)"
                                   R"("parameters":)" +
                                   schema + strict + "}]}"),
                 &optional) &&
                 optional.tools.size() == 1,
             "Responses tool without strict parses");
      const auto parameters =
          gufo::json::parse(optional.tools[0].parameters_json);
      const auto* required = parameters.find("required");
      std::string required_names;
      if (required && required->is_array())
        for (const auto& name : required->items())
          required_names += name.get_str() + ";";
      Expect(required_names == "path;" &&
                 !parameters.contains("additionalProperties"),
             "Responses tool without strict keeps optional properties optional");
      Expect(optional.tools[0].definition_json.find("\"strict\":true") ==
                 std::string::npos,
             "Responses tool without strict is not made strict");
    }
    // Rejected parses leave the request untouched, so every case parses into
    // fresh state instead of inheriting tools from a previous parse.
    gufo::server::ChatRequest ambiguous;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[
                 {"type":"function","name":"lookup"},
                 {"type":"namespace","name":"crm","tools":[{"type":"function",
                  "name":"lookup"}]}]})"),
               &ambiguous)
               .has_value(),
           "Namespace routing is rejected when function names are ambiguous");
    gufo::server::ChatRequest malformed;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[{"type":"namespace",
                 "name":"crm"}]})"),
               &malformed)
               .has_value(),
           "A namespace without a tools array is rejected");
  }
  {
    // The shared 128-function cap counts flattened functions at append time,
    // so both orderings around a full namespace reject the 129th function.
    std::string nested = R"({"type":"namespace","name":"crm","tools":[)";
    for (int i = 0; i < 128; ++i) {
      if (i != 0)
        nested += ",";
      nested += R"({"type":"function","name":"f)" + std::to_string(i) + "\"}";
    }
    nested += "]}";
    const std::string outside = R"({"type":"function","name":"outside"})";
    gufo::server::ChatRequest accepted;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[)" + nested + "]}"), &accepted) &&
               accepted.tools.size() == 128,
           "Exactly 128 functions flattened from a namespace are accepted");
    for (const bool outside_first : {false, true}) {
      const auto body = outside_first
                            ? R"({"tools":[)" + outside + "," + nested + "]}"
                            : R"({"tools":[)" + nested + "," + outside + "]}";
      gufo::server::ChatRequest overflow;
      Expect(gufo::server::ParseOpenAiResponseControls(gufo::json::parse(body),
                                                       &overflow)
                 .has_value(),
             "Both boundary orderings reject the 129th flattened function");
    }
  }
  {
    gufo::server::ChatRequest chat;
    Expect(
        gufo::server::ParseOpenAiResponseControls(
            gufo::json::parse(
                R"({"tools":[{"type":"function","function":null,"name":"f"}]})"),
            &chat)
            .has_value(),
        "Responses still rejects a malformed function tool");
  }

  // reasoning.summary is accepted and ignored; effort still applies; a
  // genuinely unknown reasoning member still fails.
  {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(
                   R"({"reasoning":{"effort":"low","summary":"auto"}})"),
               &chat) &&
               chat.reasoning.enabled == true &&
               chat.reasoning.effort == gufo::ReasoningEffort::kLow,
           "reasoning.summary is accepted while effort still applies");
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"reasoning":{"effort":"low","bogus":1}})"),
               &chat)
               .has_value(),
           "Unknown reasoning members are still rejected");
  }

  // text.verbosity is accepted and leaves the response format unset;
  // text.format still applies beside it; an unknown text member still fails.
  {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"verbosity":"low"}})"), &chat) &&
               !chat.response_format,
           "text.verbosity is accepted without forcing a response format");
    gufo::server::ChatRequest formatted;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"format":{"type":"json_object"},
                 "verbosity":"low"}})"),
               &formatted) &&
               formatted.response_format,
           "text.format still applies alongside verbosity");
    gufo::server::ChatRequest rejected;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"bogus":1}})"), &rejected)
               .has_value(),
           "Unknown text members are still rejected");
  }

  // Codex routes calls by namespace and name: a flattened function call must
  // carry its namespace, while a top-level function call carries none.
  for (const bool stream : {false, true}) {
    auto body = gufo::json::parse(R"({"input":"go","tools":[
      {"type":"function","name":"exec","strict":false,
       "parameters":{"type":"object","properties":{}}},
      {"type":"namespace","name":"multi_agent_v1","tools":[
        {"type":"function","name":"close_agent","strict":false,
         "parameters":{"type":"object","properties":{}}}]}]})");
    body["stream"] = stream;
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts a namespace beside a function");
    FakeBackend backend;
    backend.pieces = {
        "<tool_call>{\"name\":\"close_agent\",\"arguments\":{}}</tool_call>"
        "<tool_call>{\"name\":\"exec\",\"arguments\":{}}</tool_call>"};
    auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 256, {}, stream);
    Expect(response.status == 200, "namespaced tool request succeeds");
    std::vector<gufo::json::Value> results;
    if (stream) {
      response.streaming_body([&](std::string_view chunk) {
        auto pos = chunk.find("data: ");
        if (pos != std::string_view::npos &&
            !chunk.substr(pos + 6).starts_with("[DONE]")) {
          auto event = gufo::json::parse(chunk.substr(pos + 6));
          if (event.member_str("type") == "response.completed")
            results.push_back(*event.find("response"));
        }
        return true;
      });
    } else {
      results.push_back(gufo::json::parse(response.body));
    }
    std::map<std::string, std::string> namespaces;
    for (const auto& result : results)
      for (const auto& item : result.find("output")->items())
        if (item.member_str("type") == "function_call")
          namespaces[item.member_str("name")] =
              item.contains("namespace") ? item.member_str("namespace") : "-";
    Expect(namespaces.size() == 2 &&
               namespaces["close_agent"] == "multi_agent_v1" &&
               namespaces["exec"] == "-",
           "function calls echo only their own namespace");
  }
}

void TestMidConversationSystemMessagesKeepOrder() {
  // Codex sends developer messages mid-conversation. The adapter keeps them in
  // place; each model template decides how to render them.
  FakeBackend backend;
  const auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[
      {"role":"user","content":"first"},
      {"role":"assistant","content":"noted"},
      {"role":"developer","content":"compacted state"},
      {"role":"user","content":"second"}]})");
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(response.status == 200 && backend.last_request.messages.size() == 4 &&
             backend.last_request.messages[0].content == "first" &&
             backend.last_request.messages[1].content == "noted" &&
             backend.last_request.messages[2].role ==
                 gufo::tokenization::ChatRole::kDeveloper &&
             backend.last_request.messages[2].content == "compacted state" &&
             backend.last_request.messages[3].content == "second",
         "Mid-conversation developer messages reach the backend in place");
}

void TestToolNameCharacters() {
  const auto declare = [](const std::string& name) {
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use it"}],
      "tools":[]
    })");
    auto tool = gufo::json::parse(
        R"({"type":"function","function":{"name":"PLACEHOLDER",
            "parameters":{"type":"object","properties":{}}}})");
    tool["function"]["name"] = name;
    body["tools"].push_back(std::move(tool));
    return body.dump();
  };
  const auto replay = [](const std::string& name) {
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use it"}]
    })");
    auto assistant =
        gufo::json::parse(R"({"role":"assistant","tool_calls":[]})");
    auto call = gufo::json::parse(
        R"({"id":"call_1","type":"function","function":{"name":"PLACEHOLDER",
            "arguments":"{\"path\":\"a.txt\"}"}})");
    call["function"]["name"] = name;
    assistant["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(assistant));
    body["messages"].push_back(gufo::json::parse(
        R"({"role":"tool","tool_call_id":"call_1","content":"done"})"));
    return body.dump();
  };

  // Agent harnesses name bridged tools after their server. Nothing between
  // the request and either renderer treats these characters as structure.
  const std::vector<std::string> renderable{
      "read_file",   "read-file", "readFile",
      "server:tool", "fs/read",   "github.create_issue",
      "tool@v1",     "a.b.c~d+e", std::string(64, 'n')};
  for (const std::string& name : renderable) {
    FakeBackend declared;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(declare(name)), declared);
    Expect(response.status == 200 && declared.last_request.tools.size() == 1 &&
               declared.last_request.tools.front().name == name,
           "A renderable tool name reaches the backend unchanged");

    FakeBackend replayed;
    Expect(gufo::server::HandleOpenAiChat(Request(replay(name)), replayed)
                       .status == 200 &&
               replayed.chat_calls == 1,
           "The same name is accepted when a message replays a call");
  }

  // New declarations retain their name restrictions. Historical calls are
  // records from an earlier turn and must not strand the conversation (#357).
  const std::vector<std::string> unrenderable{"bad>name",
                                              "bad<name",
                                              "bad\"name",
                                              "bad\\name",
                                              "bad name",
                                              "bad\nname",
                                              "bad\x7F"
                                              "name",
                                              "outil_traçage",
                                              "…",
                                              "reаd",
                                              "​read",
                                              std::string(65, 'n')};
  for (const std::string& name : unrenderable) {
    FakeBackend declared;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(declare(name)), declared);
    Expect(response.status == 400 &&
               response.body.find("invalid_tools") != std::string::npos &&
               declared.chat_calls == 0,
           "An unrenderable declared name fails with invalid_tools");

    FakeBackend replayed;
    const auto replay_response =
        gufo::server::HandleOpenAiChat(Request(replay(name)), replayed);
    Expect(replay_response.status == 200 && replayed.chat_calls == 1 &&
               replayed.last_request.messages[1].tool_calls[0].name == name &&
               replayed.last_request.messages[1].tool_calls[0].id == "call_1" &&
               replayed.last_request.messages[2].tool_call_id == "call_1",
           "Historical names and result pairing survive unchanged");

    auto item = gufo::json::parse(
        R"({"type":"function_call","call_id":"call_1","name":"placeholder",
            "arguments":"{\"path\":\"a.txt\"}"})");
    item["name"] = name;
    gufo::tokenization::ChatMessage message;
    gufo::core::ImageReadBudget budget;
    std::string error;
    Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                    &error) &&
               message.tool_calls.size() == 1 &&
               message.tool_calls[0].name == name &&
               message.tool_calls[0].id == "call_1" &&
               message.tool_calls[0].arguments[0].value == "a.txt",
           "Responses preserves the same historical names and arguments");
  }
}

// Chat templates render typed arguments with Jinja tojson, so the model
// generates that spelling. Replayed history must render it the same way, or
// the next turn re-prefills the call it generated.
void TestHistoricalTypedArgumentsUseTojson() {
  const auto item = gufo::json::parse(
      R"({"type":"function_call","call_id":"call_1","name":"edit",
          "arguments":"{\"path\":\"a.txt\",\"edits\":[{\"oldText\":\"a\",\"newText\":\"b\"}],\"n\":2}"})");
  gufo::tokenization::ChatMessage message;
  gufo::core::ImageReadBudget budget;
  std::string error;
  Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                  &error) &&
             message.tool_calls.size() == 1 &&
             message.tool_calls[0].arguments.size() == 3 &&
             message.tool_calls[0].arguments[0].value == "a.txt" &&
             message.tool_calls[0].arguments[0].is_string &&
             message.tool_calls[0].arguments[1].value ==
                 R"([{"oldText": "a", "newText": "b"}])" &&
             !message.tool_calls[0].arguments[1].is_string &&
             message.tool_calls[0].arguments[2].value == "2",
         "Typed historical arguments render as the template's tojson");
}

// Numbers in replayed tool arguments keep the model's exact text (census row 30):
// an integer above 2^53 must not round through double, and spellings such as
// 1.0 / 1e5 / -0.50 must not be normalised, or the replayed call no longer
// matches the tokens the model generated.
void TestHistoricalNumberArgumentsKeepSourceText() {
  const auto item = gufo::json::parse(
      R"({"type":"function_call","call_id":"call_1","name":"tool",
          "arguments":"{\"id\":9007199254740993,\"f\":1.0,\"e\":1e5,\"ids\":[12345678901234567890123,-0.50]}"})");
  gufo::tokenization::ChatMessage message;
  gufo::core::ImageReadBudget budget;
  std::string error;
  Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                  &error) &&
             message.tool_calls.size() == 1 &&
             message.tool_calls[0].arguments.size() == 4 &&
             message.tool_calls[0].arguments[0].value == "9007199254740993" &&
             !message.tool_calls[0].arguments[0].is_string &&
             message.tool_calls[0].arguments[1].value == "1.0" &&
             message.tool_calls[0].arguments[2].value == "1e5" &&
             message.tool_calls[0].arguments[3].value ==
                 "[12345678901234567890123, -0.50]",
         "Historical number arguments keep their source text");
}

void TestMalformedHistoricalFunctions() {
  for (const auto source :
       {R"({"arguments":"{}"})", R"({"name":"","arguments":"{}"})",
        R"({"name":null,"arguments":"{}"})", R"({"name":42,"arguments":"{}"})",
        R"({"name":"read"})", R"({"name":"read","arguments":{}})",
        R"({"name":"read","arguments":null})",
        R"({"name":"read","arguments":"[]"})",
        R"({"name":"read","arguments":"["})",
        R"({"name":"read","arguments":"{\"x\":\"raw\nnewline\"}"})",
        R"({"name":"read\u0000file","arguments":"{}"})"}) {
    const auto function = gufo::json::parse(source);
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"continue"}]
    })");
    auto assistant =
        gufo::json::parse(R"({"role":"assistant","tool_calls":[]})");
    auto call = gufo::json::parse(R"({"id":"call_1","type":"function"})");
    call["function"] = function;
    assistant["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(assistant));
    FakeBackend backend;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 400 && backend.chat_calls == 0 &&
               response.body.find("invalid_messages") != std::string::npos,
           "Malformed history is rejected before model work");
    auto item = function;
    item["type"] = "function_call";
    item["call_id"] = "call_1";
    gufo::tokenization::ChatMessage message;
    gufo::core::ImageReadBudget budget;
    std::string error;
    Expect(!gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                     &error) &&
               !error.empty(),
           "Responses rejects malformed history with an explanation");
  }
}

void TestQwenToolBoundariesAndSchema() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model", "messages":[{"role":"user","content":"use f"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"},
      "count":{"type":"integer"},"flag":{"type":"boolean"},
      "limit":{"type":["integer","null"]},"tags":{"type":"array"}}}}}]
  })");
  const std::string good =
      "<tool_call><function=f><parameter=text>42</parameter>"
      "<parameter=count>42</parameter></function></tool_call>";
  struct Case {
    std::string text;
    std::size_t calls;
    std::string argument;
  };
  for (const auto& item :
       {Case{good, 1, R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>literal </tool_call> "
             "and </function></parameter></function></tool_call>",
             1, R"({"text":"literal </tool_call> and </function>"})"},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>oops</parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>42</function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>42</function></tool_call>" +
                 good,
             1, R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>unclosed" + good, 1,
             R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>literal </think>"
             "</parameter></function></tool_call>",
             1, R"({"text":"literal </think>"})"},
        // A pipe-wrapped spelling that is not a vocabulary token is argument
        // data, not framing: nothing in the output made it a control token.
        Case{"<tool_call><function=f><parameter=text>\n<|not_a_vocab_entry|>\n"
             "</parameter></function></tool_call>",
             1, R"({"text":"<|not_a_vocab_entry|>"})"},
        // One newline on each side of a value is template framing; a file's
        // final newline, indentation and an empty value survive.
        Case{"<tool_call>\n<function=f>\n<parameter=text>\n  line 1\n"
             "line 2\n\n</parameter>\n<parameter=count>\n42\n</parameter>\n"
             "</function>\n</tool_call>",
             1, R"({"text":"  line 1\nline 2\n","count":42})"},
        Case{"<tool_call>\n<function=f>\n<parameter=text>\n\n</parameter>\n"
             "</function>\n</tool_call>",
             1, R"({"text":""})"},
        Case{"<tool_call>\r\n<function=f>\r\n<parameter=text>\r\n\r\nx\r\n"
             "</parameter>\r\n</function>\r\n</tool_call>",
             1, R"({"text":"\r\nx"})"},
        // Python literals where the schema wants JSON; a string keeps them.
        Case{"<tool_call><function=f><parameter=flag>\nTrue\n</parameter>"
             "<parameter=limit>None</parameter><parameter=tags>"
             "[False, \"None\", \"a \\\" True\"]</parameter>"
             "<parameter=text>True</parameter></function></tool_call>",
             1,
             R"({"flag":true,"limit":null,"tags":[false,"None","a \" True"],)"
             R"("text":"True"})"},
        Case{"<tool_call><function=f><parameter=flag>Yes</parameter>"
             "</function></tool_call>",
             0, ""},
        // A repeated parameter with the same value is dropped; a conflicting
        // repeat still rejects the call.
        Case{"<tool_call>\n<function=f>\n<parameter=text>\na\n</parameter>\n"
             "<parameter=count>42</parameter><parameter=flag>True</parameter>"
             "<parameter=text>a</parameter><parameter=count>\n42\n"
             "</parameter><parameter=flag>true</parameter>\n</function>\n"
             "</tool_call>",
             1, R"({"text":"a","count":42,"flag":true})"},
        Case{"<tool_call><function=f><parameter=text>a</parameter>"
             "<parameter=text>b</parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>a</parameter>"
             "<parameter=text>a </parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":"
             "\"literal </tool_call>\"}}</tool_call>",
             1, R"({"text":"literal </tool_call>"})"}}) {
    for (bool reasoning : {false, true}) {
      for (bool stream : {false, true}) {
        auto body = schema;
        body["stream"] = stream;
        body["chat_template_kwargs"] = Value::object();
        body["chat_template_kwargs"]["enable_thinking"] = reasoning;
        FakeBackend backend;
        const auto text =
            (reasoning ? "Considering. </think>" : "") + item.text;
        // Split every marker and argument across token callbacks.
        for (char c : text)
          backend.pieces.emplace_back(1, c);
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 200, "tool boundary request succeeds");
        std::vector<Value> calls;
        if (!stream) {
          const auto output = gufo::json::parse(response.body);
          const auto& message =
              *output.find("choices")->items()[0].find("message");
          if (const auto* found = message.find("tool_calls"))
            calls.assign(found->items().begin(), found->items().end());
        } else {
          std::string output;
          response.streaming_body([&](std::string_view part) {
            output += part;
            return true;
          });
          std::size_t cursor = 0;
          while ((cursor = output.find("data: ", cursor)) !=
                 std::string::npos) {
            const auto begin = cursor + 6;
            cursor = output.find('\n', begin);
            const auto payload = output.substr(begin, cursor - begin);
            if (payload == "[DONE]")
              break;
            const auto event = gufo::json::parse(payload);
            const auto* choices = event.find("choices");
            if (!choices || choices->empty())
              continue;
            const auto* delta = choices->items()[0].find("delta");
            if (const auto* found = delta ? delta->find("tool_calls") : nullptr)
              AppendToolDeltas(calls, *found, stream);
            if (item.calls && delta)
              Expect(delta->member_str("reasoning_content").find('<') ==
                         std::string::npos,
                     "tool markers do not leak into reasoning deltas");
          }
        }
        if (calls.size() != item.calls)
          std::cerr << "Tool input: " << item.text
                    << "\nExpected calls: " << item.calls
                    << ", actual: " << calls.size() << '\n';
        Expect(calls.size() == item.calls, "only complete tool calls emitted");
        if (!calls.empty()) {
          if (calls.back().find("function")->member_str("arguments") !=
              item.argument)
            std::cerr << "Tool input: " << item.text
                      << "\nExpected: " << item.argument << "\nActual: "
                      << calls.back().find("function")->member_str("arguments")
                      << '\n';
          Expect(calls.back().find("function")->member_str("arguments") ==
                     item.argument,
                 "tool argument values and schema types preserved");
        }
      }
    }
  }
}

void TestToolChoiceEnforcement() {
  for (bool stream : {false, true}) {
    for (const auto* text :
         {"<tool_call><function=f></function></tool_call>",
          "<｜DSML｜tool_calls｜><｜DSML｜invoke "
          "name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls｜>"}) {
      for (const auto* choice : {"auto", "none", "required"}) {
        for (const auto* declared : {"", "f", "other"}) {
          auto body = gufo::json::parse(
              R"({"model":"test-model","messages":[{"role":"user","content":"use a tool"}]})");
          body["stream"] = stream;
          body["tool_choice"] = choice;
          if (*declared) {
            auto tool = gufo::json::parse(
                R"({"type":"function","function":{"name":"f","parameters":{"type":"object"}}})");
            tool["function"]["name"] = declared;
            body["tools"] = gufo::json::Value::array();
            body["tools"].push_back(std::move(tool));
          }
          FakeBackend backend;
          backend.pieces = {text};
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          const bool required = std::string_view(choice) == "required";
          if (required && !*declared) {
            Expect(response.status == 400, "required needs declared tools");
            continue;
          }
          std::string output = response.body;
          if (response.streaming_body)
            response.streaming_body([&](std::string_view part) {
              output += part;
              return true;
            });
          const bool allowed = std::string_view(declared) == "f" &&
                               std::string_view(choice) != "none";
          Expect(
              (output.find("\"tool_calls\":") != std::string::npos) == allowed,
              "only declared, enabled tool calls can enter API output");
          if (required && !allowed)
            Expect(
                output.find("tool_choice_unsatisfied") != std::string::npos &&
                    (stream || response.status == 502),
                "required cannot silently return text");
        }
      }
    }
  }
  FakeBackend backend;
  backend.pieces = {"ordinary text"};
  auto body = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"call f"}],"tool_choice":"required","tools":[{"type":"function","function":{"name":"f","parameters":{}}}]})");
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             502,
         "ordinary text cannot fulfill required tool choice");
}

void TestDeepSeekToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<｜DSML｜tool_calls｜>\n"
      "<｜DS｜invoke name=\"read\">\n"
      "<｜DS｜parameter name=\"path\" string=\"true\">"
      "/etc/hostname</｜DS｜parameter>\n"
      "</｜DS｜invoke>\n"
      "</｜DSML｜tool_calls｜>",
  };

  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"read the hostname"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"read",
            "description":"Read a file",
            "parameters":{
              "type":"object",
              "properties":{"path":{"type":"string"}},
              "required":["path"]
            }
          }
        }],
        "stream":false
      })"),
                                                       backend);

  Expect(response.status == 200, "DeepSeek tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Hybrid DeepSeek syntax reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"read")") != std::string::npos,
         "Hybrid DeepSeek tool name is translated");
  Expect(
      response.body.find(R"(\"path\":\"/etc/hostname\")") != std::string::npos,
      "Hybrid DeepSeek tool arguments are translated");
}

void TestDeepSeekRepeatedToolParameters() {
  const auto call = [](std::string_view repeat) {
    return "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"read\">"
           "<｜DSML｜parameter name=\"path\" string=\"true\">/a"
           "</｜DSML｜parameter><｜DSML｜parameter name=\"path\" "
           "string=\"true\">" +
           std::string(repeat) +
           "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls｜>";
  };
  for (const auto& [repeat, accepted] :
       {std::pair{"/a", true}, std::pair{"\n/a\n", false},
        std::pair{"/b", false}}) {
    FakeBackend backend;
    backend.pieces = {call(repeat)};
    const auto response = gufo::server::HandleOpenAiChat(
        Request(
            R"({"model":"test-model","messages":[{"role":"user","content":"read"}],)"
            R"("tools":[{"type":"function","function":{"name":"read",)"
            R"("parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]})"),
        backend);
    Expect(response.status == 200, "DeepSeek repeated parameter request");
    Expect((response.body.find(R"("arguments":"{\"path\":\"/a\"}")") !=
            std::string::npos) == accepted,
           "identical DeepSeek repeats are dropped, conflicts rejected");
    Expect((response.body.find(R"("finish_reason":"tool_calls")") !=
            std::string::npos) == accepted,
           "a conflicting DeepSeek repeat is not a tool call");
  }
}

void TestBackendSamplingDefaults() {
  FakeBackend backend;
  backend.defaults = {
      .max_tokens = 37,
      .sampling = {.temperature = 0.25F},
  };

  const auto default_response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                               backend);
  Expect(default_response.status == 200, "Defaulted request is accepted");
  Expect(backend.last_max_tokens == 37,
         "Backend max-token default reaches generation");
  Expect(backend.last_temperature > 0.24F && backend.last_temperature < 0.26F,
         "Backend temperature default reaches generation");

  const auto override_response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":11,
        "temperature":0
      })"),
                                                                backend);
  Expect(override_response.status == 200, "Sampling override is accepted");
  Expect(backend.last_max_tokens == 11,
         "Explicit max tokens override the backend default");
  Expect(backend.last_temperature == 0.0F,
         "Explicit temperature overrides the backend default");
}

void TestModelSamplingDefaults() {
  using gufo::sampling::TextModelPreset;
  FakeBackend backend;
  backend.defaults.model = TextModelPreset::kQwen38;
  backend.defaults.supplied = {};
  auto constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  constraint->grammar = gufo::sampling::JsonConstraint::Object();
  constraint->vocabulary =
      std::make_shared<gufo::sampling::ConstraintVocabulary>(
          2, [](std::uint32_t id) {
            return gufo::sampling::ConstraintVocabulary::Piece{
                .text = id == 0 ? "{}" : "", .stop = id == 1};
          });
  backend.defaults.sampling.constraint = constraint;
  auto send = [&](std::string fields) {
    const auto response = gufo::server::HandleOpenAiChat(
        Request("{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hello\"}]" +
                fields + "}"),
        backend);
    Expect(response.status == 200, "Model default request accepted");
    Expect(backend.last_sampling.constraint ==
               backend.defaults.sampling.constraint,
           "Model presets preserve output constraints");
    return backend.last_sampling;
  };
  auto config = send("");
  Expect(config.temperature == 1.0F && config.top_p == 0.95F &&
             config.top_k == 20 && config.presence_penalty == 0.0F,
         "Qwen defaults follow thinking-on template");
  config = send(R"(,"chat_template_kwargs":{"enable_thinking":false})");
  Expect(config.temperature == 0.7F && config.top_p == 0.8F &&
             config.presence_penalty == 1.5F,
         "Request thinking-off selects its model preset");
  backend.reasoning_defaults_value.enabled = false;
  config = send(R"(,"temperature":null,"top_p":null,"presence_penalty":null)");
  Expect(config.temperature == 0.7F && config.presence_penalty == 1.5F,
         "Nulls inherit effective server thinking preset");
  config = send(R"(,"reasoning_effort":"high")");
  Expect(config.temperature == 1.0F && config.presence_penalty == 0.0F,
         "Effort enables thinking before sampling resolves");
  backend.defaults.sampling.temperature = 0.2F;
  backend.defaults.sampling.top_k = 0;
  backend.defaults.supplied.temperature = true;
  backend.defaults.supplied.top_k = true;
  config = send("");
  Expect(config.temperature == 0.2F && config.top_k == 0 &&
             config.top_p == 0.8F && config.presence_penalty == 1.5F,
         "Partial server override preserves other model defaults");
  config = send(R"(,"temperature":0,"presence_penalty":0,"top_k":40)");
  Expect(config.temperature == 0.0F && config.presence_penalty == 0.0F &&
             config.top_k == 40,
         "Explicit request zero overrides server and model presets");
  backend.defaults.model = TextModelPreset::kDeepSeekV4Flash;
  backend.defaults.supplied = {};
  for (const bool thinking : {false, true}) {
    backend.reasoning_defaults_value.enabled = thinking;
    config = send("");
    Expect(config.temperature == 1.0F && config.top_p == 0.95F &&
               config.top_k == 0 && config.min_p == 0.0F &&
               config.presence_penalty == 0.0F &&
               config.frequency_penalty == 0.0F &&
               config.repeat_penalty == 1.0F,
           "DeepSeek agentic defaults do not depend on reasoning");
  }
  backend.defaults.supplied = gufo::sampling::SamplingOverrides::All();
  config = send("");
  Expect(config.constraint == backend.defaults.sampling.constraint,
         "Explicit public-API configuration preserves its output constraint");
}

void TestCompleteToolDefinitionsReachTemplate() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "messages":[{"role":"user","content":"Emit a value."}],
    "tools":[{"type":"function","function":{"parameters":{
      "type":"object","additionalProperties":false},"strict":true,
      "description":"","name":"emit"},
      "vendor":{"version":2}}]
  })"),
                                                       backend);
  Expect(response.status == 200 && backend.last_request.tools.size() == 1,
         "complete function definition reaches the backend");
  const auto& tool = backend.last_request.tools.front();
  Expect(
      tool.definition_json ==
          R"({"type":"function","function":{"parameters":{"type":"object","additionalProperties":false},"strict":true,"description":"","name":"emit"},"vendor":{"version":2}})",
      "Valid nested definitions retain every field and its original order");
  gufo::tokenization::ChatTemplateOptions options;
  options.enable_thinking = false;
  const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
      backend.last_request.messages, backend.last_request.tools, options);
  Expect(
      rendered.has_value() &&
          rendered->find(
              R"({"type": "function", "function": {"parameters": {"type": "object", "additionalProperties": false}, "strict": true, "description": "", "name": "emit"}, "vendor": {"version": 2}})") !=
              std::string::npos,
      "Existing valid tool JSON is unchanged in the Qwen prompt");
}

void TestFlatToolFieldsReachTemplate() {
  for (const char* strict : {"true", "false", "null"}) {
    FakeBackend backend;
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use f"}],
      "tools":[]
    })");
    auto function = gufo::json::parse(R"({
      "name":"f","description":"","parameters":{"type":"object","additionalProperties":false},
      "strict":null,"vendor":{"version":2}
    })");
    function["strict"] = gufo::json::parse(strict);
    auto flat = function;
    flat["type"] = "function";
    body["tools"].push_back(flat);
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 && backend.last_request.tools.size() == 1,
           "Flat function with additional fields is accepted");
    const auto flat_tools = backend.last_request.tools;
    const auto definition = gufo::json::parse(flat_tools[0].definition_json);
    const auto* nested = definition.find("function");
    Expect(nested && nested->dump() == function.dump() &&
               !definition.contains("strict") && !definition.contains("vendor"),
           "All flat function fields survive DS4's nested function extraction");
    gufo::tokenization::ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto flat_prompt = gufo::tokenization::QwenChatTemplate::Render(
        backend.last_request.messages, flat_tools, options);
    auto canonical = gufo::json::Value::object();
    canonical["type"] = "function";
    canonical["function"] = function;
    body["tools"] = gufo::json::Value::array();
    body["tools"].push_back(canonical);
    const auto nested_response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(nested_response.status == 200, "Nested equivalent is accepted");
    const auto nested_prompt = gufo::tokenization::QwenChatTemplate::Render(
        backend.last_request.messages, backend.last_request.tools, options);
    Expect(flat_prompt.has_value() && nested_prompt == flat_prompt,
           "Flat and nested tools produce identical Qwen prompts");
  }
}

// Each dialect's renderer and parser describe one wire format. Render a
// historical call through the model's own chat template, hand exactly those
// bytes back as generated text, and require every string argument to return
// unchanged (#290).
void TestToolCallRenderParseRoundTrip() {
  const std::vector<std::string> values = {
      "plain",
      "ends with a newline\n",
      "  indented first line\n    and a second\n",
      "",
      "\nstarts with a newline",
      "blank\n\nlines\n\n",
      "tab\tand trailing space ",
      "quoted <b>markup</b> and \"quotes\"",
  };
  for (const bool deepseek : {false, true}) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      const auto& value = values[index];
      std::string text;
      if (deepseek) {
        gufo::models::deepseek_v4_flash::ChatMessage call;
        call.role = "assistant";
        call.tool_calls.push_back(
            {.name = "record",
             .arguments = {{.name = "text", .value = value},
                           {.name = "other", .value = "x"}},
             .id = "call_0"});
        const auto rendered = gufo::models::deepseek_v4_flash::RenderChat(
            std::vector<gufo::models::deepseek_v4_flash::ChatMessage>{
                {.role = "user", .content = "go"}, call});
        const auto begin = rendered.find("<｜DSML｜tool_calls>");
        const auto end_tag = std::string_view("</｜DSML｜tool_calls>");
        const auto end = rendered.find(end_tag);
        Expect(begin != std::string::npos && end != std::string::npos,
               "DeepSeek renders a historical call");
        text = rendered.substr(begin, end + end_tag.size() - begin);
      } else {
        gufo::tokenization::ChatMessage call;
        call.role = gufo::tokenization::ChatRole::kAssistant;
        call.tool_calls.push_back(
            {.id = "call_0",
             .name = "record",
             .arguments = {{.name = "text", .value = value},
                           {.name = "other", .value = "x"}}});
        std::vector<gufo::tokenization::ChatMessage> history = {
            {gufo::tokenization::ChatRole::kUser, "go"}, call};
        gufo::tokenization::ChatTemplateOptions options;
        options.enable_thinking = false;
        const auto rendered =
            gufo::tokenization::QwenChatTemplate::Render(history, {}, options);
        Expect(rendered.has_value(), "Qwen renders a historical call");
        const auto begin = rendered->find("<tool_call>");
        const auto end_tag = std::string_view("</tool_call>");
        const auto end = rendered->find(end_tag);
        Expect(begin != std::string::npos && end != std::string::npos,
               "Qwen renders a historical call envelope");
        text = rendered->substr(begin, end + end_tag.size() - begin);
      }

      FakeBackend backend;
      backend.pieces = {text};
      const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"record"}],
        "tools":[{"type":"function","function":{"name":"record",
          "parameters":{"type":"object","properties":{
            "text":{"type":"string"},"other":{"type":"string"}}}}}],
        "stream":false
      })"),
                                                           backend);
      const std::string label = std::string(deepseek ? "DeepSeek" : "Qwen") +
                                " case " + std::to_string(index);
      Expect(response.status == 200, label + ": request is accepted");
      const auto body = gufo::json::parse(response.body);
      const auto* choices = body.find("choices");
      const auto* message = choices != nullptr && !choices->items().empty()
                                ? choices->items()[0].find("message")
                                : nullptr;
      const auto* calls =
          message != nullptr ? message->find("tool_calls") : nullptr;
      Expect(calls != nullptr && calls->items().size() == 1,
             label + ": the rendered call parses back as one call");
      const auto* function = calls->items()[0].find("function");
      Expect(function != nullptr, label + ": the call has a function");
      const auto arguments =
          gufo::json::parse(function->member_str("arguments"));
      const auto* text_argument = arguments.find("text");
      Expect(text_argument != nullptr && text_argument->is_string() &&
                 text_argument->get_str() == value &&
                 arguments.member_str("other") == "x",
             label +
                 ": string arguments survive the round trip byte for "
                 "byte");
    }
  }
}

void TestAllSamplingControlsReachBackend() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "temperature":0.8,
        "top_k":40,
        "top_p":0.9,
        "min_p":0.05,
        "min_keep":3,
        "seed":123,
        "repeat_penalty":1.1,
        "repeat_last_n":32,
        "frequency_penalty":0.25,
        "presence_penalty":0.5
      })"),
                                                       backend);

  Expect(response.status == 200, "Complete sampling request is accepted");
  const auto& sampling = backend.last_sampling;
  Expect(sampling.temperature > 0.79F && sampling.temperature < 0.81F,
         "temperature reaches backend");
  Expect(sampling.top_k == 40, "top-k reaches backend");
  Expect(sampling.top_p > 0.89F && sampling.top_p < 0.91F,
         "top-p reaches backend");
  Expect(sampling.min_p > 0.04F && sampling.min_p < 0.06F,
         "min-p reaches backend");
  Expect(sampling.min_keep == 3, "min-keep reaches backend");
  Expect(sampling.seed == 123, "seed reaches backend");
  Expect(sampling.repeat_penalty > 1.09F && sampling.repeat_penalty < 1.11F,
         "repeat penalty reaches backend");
  Expect(sampling.repeat_last_n == 32, "repeat window reaches backend");
  Expect(
      sampling.frequency_penalty > 0.24F && sampling.frequency_penalty < 0.26F,
      "frequency penalty reaches backend");
  Expect(sampling.presence_penalty > 0.49F && sampling.presence_penalty < 0.51F,
         "presence penalty reaches backend");
}

void TestUnsupportedSamplingControlsAreRejected() {
  FakeBackend backend;
  for (const auto& [field, value] :
       {std::pair{"temperature", 2.01}, std::pair{"presence_penalty", 2.01},
        std::pair{"frequency_penalty", -2.01}}) {
    for (const bool stream : {false, true}) {
      auto request = Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
      auto body = gufo::json::parse(request.body);
      body[field] = value;
      body["stream"] = stream;
      request.body = body.dump();
      const auto response = gufo::server::HandleOpenAiChat(request, backend);
      Expect(response.status == 400 &&
                 response.body.find(std::string("invalid_") + field) !=
                     std::string::npos,
             "out-of-range OpenAI sampling fails before streaming");
    }
  }
  for (const double choices : {0.0, 1.4, 2.0, 1e100}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
    auto body = gufo::json::parse(request.body);
    body["n"] = choices;
    request.body = body.dump();
    Expect(gufo::server::HandleOpenAiChat(request, backend).status == 400,
           "n must be exactly one, without truncation or overflow");
  }
  for (const auto* field : {"draft_temperature",  "temperature_draft",
                            "draft_top_k",        "draft_top_p",
                            "draft_min_p",        "draft_seed",
                            "draft_policy",       "samplers",
                            "typical_p",          "tfs_z",
                            "mirostat",           "mirostat_eta",
                            "mirostat_tau",       "dynatemp_range",
                            "dynatemp_exponent",  "xtc_probability",
                            "xtc_threshold",      "dry_multiplier",
                            "dry_base",           "dry_allowed_length",
                            "dry_penalty_last_n", "dry_sequence_breakers",
                            "top_n_sigma",        "logit_bias"}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
    auto body = gufo::json::parse(request.body);
    body[field] = 0.8;
    // Dependent controls only matter while their sampler is enabled.
    const std::string name = field;
    if (name.starts_with("mirostat_"))
      body["mirostat"] = 2;
    if (name == "dynatemp_exponent")
      body["dynatemp_range"] = 0.5;
    if (name == "xtc_threshold")
      body["xtc_probability"] = 0.5;
    if (name.starts_with("dry_") && name != "dry_multiplier") {
      body["dry_multiplier"] = 0.8;
    }
    request.body = body.dump();
    const auto response = gufo::server::HandleOpenAiChat(request, backend);
    Expect(response.status == 400 &&
               response.body.find("unsupported_sampling") != std::string::npos,
           "unsupported sampling must not be silently ignored");
  }
  // Disabled values, as OpenAI SDKs and llama.cpp front ends send them, leave
  // the output unchanged and are accepted.
  {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
            "stop":null,"logprobs":false,"top_logprobs":0,"n":1,
            "response_format":{"type":"text"},"modalities":["text"],
            "typical_p":1.0,"tfs_z":1.0,"mirostat":0,"mirostat_eta":0.1,
            "mirostat_tau":5.0,"dynatemp_range":0,"dynatemp_exponent":1.0,
            "xtc_probability":0,"xtc_threshold":0.1,"dry_multiplier":0,
            "dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1,
            "dry_sequence_breakers":["\n"],"top_n_sigma":-1,"logit_bias":{}})");
    Expect(gufo::server::HandleOpenAiChat(request, backend).status == 200,
           "disabled compatibility fields are accepted");
    auto body = gufo::json::parse(request.body);
    body["stop"] = gufo::json::Value::array();
    request.body = body.dump();
    Expect(gufo::server::HandleOpenAiChat(request, backend).status == 200,
           "an empty stop list is accepted");
  }
  // SDK message objects serialize absent tool fields as null.
  {
    auto request = Request(
        R"({"model":"test-model","tools":null,"tool_choice":null,
            "messages":[{"role":"user","content":"hello"},
                        {"role":"assistant","content":"hi","tool_calls":null,
                         "reasoning_content":null},
                        {"role":"user","content":"again"}]})");
    Expect(gufo::server::HandleOpenAiChat(request, backend).status == 200,
           "null tools, tool_choice and tool_calls mean absent");
  }
}

void TestSamplingRanges() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  for (const auto* field : {"temperature", "top_p", "min_p",
                            "frequency_penalty", "presence_penalty"}) {
    const bool penalty = std::string_view(field).ends_with("penalty");
    const double minimum = penalty ? -2 : 0;
    const double maximum =
        penalty || std::string_view(field) == "temperature" ? 2 : 1;
    for (const auto value :
         {minimum - 1e-8, minimum, maximum, maximum + 1e-8}) {
      auto request = Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
      auto body = gufo::json::parse(request.body);
      body[field] = value;
      request.body = body.dump();
      const auto response = gufo::server::HandleOpenAiChat(request, backend);
      Expect(
          response.status == (value < minimum || value > maximum ? 400 : 200),
          "sampling range is inclusive and checked before float rounding");
    }
  }
}

void TestAssistantReasoningContentReachesBackend() {
  FakeBackend backend;
  backend.pieces = {"Blue"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[
          {"role":"user","content":"Name one color."},
          {
            "role":"assistant",
            "reasoning_content":"I should answer concisely.",
            "content":"Red"
          },
          {"role":"user","content":"Name another."}
        ],
        "max_tokens":1
      })"),
                                                       backend);

  Expect(response.status == 200, "Assistant reasoning history is accepted");
  Expect(backend.last_request.messages.size() == 3,
         "Complete reasoning history reaches the backend");
  Expect(
      backend.last_request.messages[1].thought == "I should answer concisely.",
      "Assistant reasoning_content reaches the model template");
  Expect(backend.last_request.messages[1].content == "Red",
         "Assistant visible content remains separate from reasoning");
}

void TestBehaviourRequestOptions() {
  struct EnvironmentScope {
    std::vector<std::pair<std::string, std::optional<std::string>>> previous;
    static void Set(const char* name, const char* value) {
#ifdef _WIN32
      _putenv_s(name, value ? value : "");
#else
      if (value)
        setenv(name, value, 1);
      else
        unsetenv(name);
#endif
    }
    EnvironmentScope() {
      for (const auto* name :
           {"GUFO_THINK_EOG_GUARD", "GUFO_THINK_GUARD_ESCALATE",
            "GUFO_EOG_BOUNDARY", "GUFO_EOG_MIN_PROB_EXACT", "GUFO_EOG_MIN_PROB",
            "GUFO_EOG_DEFER_P", "GUFO_SPEC_EOG_STOP", "GUFO_LOOP_BREAKER",
            "GUFO_LOOP_BREAKER_EOG_CLOSE", "GUFO_LOOP_BREAKER_TOOL_ARGS",
            "GUFO_LOOP_BREAKER_TRANSITION", "GUFO_REASONING_BUDGET",
            "GUFO_LOGIT_BIAS", "GUFO_THINK_MARKER_BIAS",
            "GUFO_THINK_MARKER_BIAS_BY_EFFORT",
            "GUFO_THINK_MARKER_ENTROPY_GATE", "GUFO_PRODUCTION_SAMPLING",
            "GUFO_MODEL_SUPPRESS_TOKENS"}) {
        const char* value = std::getenv(name);
        previous.emplace_back(name, value ? std::optional<std::string>(value)
                                          : std::nullopt);
        Set(name, nullptr);
      }
    }
    ~EnvironmentScope() {
      for (const auto& [name, value] : previous)
        Set(name.c_str(), value ? value->c_str() : nullptr);
    }
  } environment;
  using gufo::server::ParseBehaviourOptions;
  using gufo::server::ResolveMarkerBias;
  const auto parse = [](std::string_view body) {
    return ParseBehaviourOptions(gufo::json::parse(body));
  };
  const auto invalid = [&](std::string_view body) {
    bool rejected = false;
    try {
      (void)parse(body);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "invalid behaviour request is rejected deterministically");
  };
  Expect(parse("{}") == nullptr, "all unset switches preserve native sampling");
  Expect(gufo::server::ThinkGuardEscalate() == 16,
         "unset guard escalation defaults to sixteen");
  EnvironmentScope::Set("GUFO_LOOP_BREAKER", "1");
  Expect(parse("{}")->loop_tool_args, "unset tool-argument kill switch defaults on");
  EnvironmentScope::Set("GUFO_LOOP_BREAKER_TOOL_ARGS", "0");
  Expect(!parse("{}")->loop_tool_args, "tool-argument kill switch disables only TOOL");
  EnvironmentScope::Set("GUFO_LOOP_BREAKER_TOOL_ARGS", "1");
  Expect(parse("{}")->loop_tool_args, "tool-argument kill switch accepts one");
  EnvironmentScope::Set("GUFO_LOOP_BREAKER_TOOL_ARGS", "bad");
  invalid("{}");
  EnvironmentScope::Set("GUFO_LOOP_BREAKER_TOOL_ARGS", nullptr);
  EnvironmentScope::Set("GUFO_LOOP_BREAKER", nullptr);
  EnvironmentScope::Set("GUFO_THINK_EOG_GUARD", "1");
  for (const auto* value : {"0", "1", "16", "2147483647"}) {
    EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", value);
    Expect(parse("{}")->think_guard_escalate == std::stoi(value),
           "guard escalation parses a strict nonnegative integer");
  }
  for (const auto* value : {"-1", "+1", " 16", "16 ", "1.5", "1e1", "16junk",
                            "2147483648", "nan"}) {
    EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", value);
    invalid("{}");
  }
#ifndef _WIN32
  EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", "");
  invalid("{}");
#endif
  // Validate at startup even when every behaviour switch is off.
  EnvironmentScope::Set("GUFO_THINK_EOG_GUARD", nullptr);
  EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", "bad");
  invalid("{}");
  EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", nullptr);

  {
    FakeBackend backend;
    const auto response = gufo::server::HandleOpenAiChat(Request(R"({
      "model":"test-model","messages":[{"role":"user","content":"hello"}],
      "logit_bias":{"12":1}})"), backend);
    Expect(response.status == 400 &&
               response.body.find("unsupported_sampling") != std::string::npos &&
               backend.chat_calls == 0,
           "unset bias switch retains gufo's unsupported_sampling contract");
  }
  EnvironmentScope::Set("GUFO_LOGIT_BIAS", "1");
  auto cfg = parse(R"({"logit_bias":{"12":1.25,"hello":false}})");
  Expect(cfg && cfg->logit_bias_enabled && cfg->logit_bias.size() == 1 &&
             cfg->logit_bias[0].first == 12 &&
             cfg->logit_bias[0].second == 1.25F &&
             cfg->text_logit_bias.size() == 1 &&
             cfg->text_logit_bias[0].first == "hello" &&
             std::isinf(cfg->text_logit_bias[0].second) &&
             cfg->text_logit_bias[0].second < 0,
         "enabled logit bias accepts numeric ids and false bans of tokenized text");
  cfg = parse(R"({"logit_bias":[[12,-2],["hello",0.5],[13,false]]})");
  Expect(cfg->logit_bias.size() == 2 && cfg->text_logit_bias.size() == 1,
         "production array form preserves numeric ids and text bias");
  for (const auto* body : {R"({"logit_bias":null})",
                           R"({"logit_bias":{}})",
                           R"({"logit_bias":[]})"}) {
    cfg = parse(body);
    Expect(cfg && cfg->logit_bias_enabled && cfg->logit_bias.empty() &&
               cfg->text_logit_bias.empty(),
           "null and empty bias containers preserve the unmodified distribution");
  }
  for (const auto* body : {
           R"({"logit_bias":{"-1":1}})", R"({"logit_bias":{"+1":1}})",
           R"({"logit_bias":[[1.5,1]]})", R"({"logit_bias":[[-1,1]]})",
           R"({"logit_bias":[[4294967296,1]]})",
           R"({"logit_bias":{"4294967296":1}})",
           R"({"logit_bias":{"12":true}})",
           R"({"logit_bias":{"12":"1"}})",
           R"({"logit_bias":12})", R"({"logit_bias":true})",
           R"({"logit_bias":false})", R"({"logit_bias":"oops"})",
           R"({"logit_bias":""})"})
    invalid(body);
  EnvironmentScope::Set("GUFO_REASONING_BUDGET", "1");
  for (const auto* body : {R"({"reasoning_budget_tokens":0})",
                           R"({"reasoning_budget_tokens":-1})",
                           R"({"reasoning_budget_tokens":2147483647})"}) {
    cfg = parse(body);
    Expect(cfg && cfg->reasoning_budget_enabled &&
               cfg->reasoning_budget ==
                   static_cast<std::int64_t>(gufo::json::parse(body)
                                                .find("reasoning_budget_tokens")
                                                ->as_double()),
           "budget accepts zero, unlimited and the production integer maximum");
  }
  for (const auto* body : {R"({"reasoning_budget_tokens":0.5})",
                           R"({"reasoning_budget_tokens":-2})",
                           R"({"reasoning_budget_tokens":2147483648})",
                           R"({"reasoning_budget_tokens":"0"})"})
    invalid(body);
  cfg = parse(R"({"reasoning_budget_start_tag":"<reason>",
                   "reasoning_budget_end_tag":"</reason>"})");
  Expect(cfg->reasoning_start_tag == "<reason>" &&
             cfg->reasoning_end_tags == std::vector<std::string>{"</reason>"},
         "budget start and single end tags preserve custom phase delimiters");
  cfg = parse(R"({"reasoning_budget_end_tag":42,
                   "reasoning_budget_end_tags":["</reason>","<done>"]})");
  Expect(cfg->reasoning_end_tags ==
             std::vector<std::string>{"</reason>", "<done>"},
         "budget end-tag list takes precedence over the single-tag field");
  for (const auto* body : {
           R"({"reasoning_budget_start_tag":""})",
           R"({"reasoning_budget_start_tag":null})",
           R"({"reasoning_budget_start_tag":42})",
           R"({"reasoning_budget_end_tag":""})",
           R"({"reasoning_budget_end_tag":null})",
           R"({"reasoning_budget_end_tag":[]})",
           R"({"reasoning_budget_end_tags":[]})",
           R"({"reasoning_budget_end_tags":null})",
           R"({"reasoning_budget_end_tags":"</reason>"})",
           R"({"reasoning_budget_end_tags":[""]})",
           R"({"reasoning_budget_end_tags":["</reason>",42]})"})
    invalid(body);
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS", "-0.25");
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS_BY_EFFORT",
                        "low:-2,medium:-1,xhigh:0");
  cfg = parse("{}");
  Expect(ResolveMarkerBias(*cfg, "low") == -2 &&
             ResolveMarkerBias(*cfg, "minimal") == -2 &&
             ResolveMarkerBias(*cfg, "medium") == -1 &&
             ResolveMarkerBias(*cfg, "xhigh") == 0 &&
             ResolveMarkerBias(*cfg, "high") == 0 &&
             ResolveMarkerBias(*cfg, "max") == 0 &&
             ResolveMarkerBias(*cfg, "unknown") == -0.25,
         "effort map overrides global bias and aliases preserve production semantics");
  cfg = parse(R"({"think_marker_bias":0.75})");
  Expect(ResolveMarkerBias(*cfg, "low") == 0.75 &&
             ResolveMarkerBias(*cfg, "xhigh") == 0.75,
         "explicit request bias wins over effort mapping");
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS_BY_EFFORT",
                        " low : -2 ,\tmedium : -1 , xhigh : 0 \r\n");
  cfg = parse("{}");
  Expect(ResolveMarkerBias(*cfg, "low") == -2 &&
             ResolveMarkerBias(*cfg, "medium") == -1 &&
             ResolveMarkerBias(*cfg, "xhigh") == 0,
         "production effort map trims whitespace around labels and values");
  // server-common.cpp:40-74 accepts arbitrary nonempty labels. Keep that
  // compatibility for future efforts while enforcing the original syntax.
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS_BY_EFFORT",
                        "experimental:-3,low:-2");
  cfg = parse("{}");
  Expect(ResolveMarkerBias(*cfg, "experimental") == -3 &&
             ResolveMarkerBias(*cfg, "medium") == -0.25,
         "unknown nonempty map labels remain legal and unmatched efforts inherit global bias");
  for (const auto* map : {"low:-2,low:-1", "low:-2,", "low:-2, ",
                          "low", ":-2", "low:", "low:-2:3",
                          "low:nan", "low:inf", "low:1e999"}) {
    bool rejected = false;
    try {
      EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS_BY_EFFORT", map);
      cfg = parse("{}");
      (void)ResolveMarkerBias(*cfg, "low");
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected,
           "production effort map rejects duplicate, empty, malformed and nonfinite entries");
  }
}

void DebugHooksSettingsTest() {
  struct EnvironmentScope {
    std::vector<std::pair<std::string, std::optional<std::string>>> previous;
    static void Set(const char* name, const char* value) {
#ifdef _WIN32
      _putenv_s(name, value ? value : "");
#else
      if (value)
        setenv(name, value, 1);
      else
        unsetenv(name);
#endif
    }
    EnvironmentScope() {
      for (const auto* name : gufo::server::kDebugSettingsEnvironment) {
        const auto* value = std::getenv(name);
        previous.emplace_back(name, value ? std::optional<std::string>(value)
                                          : std::nullopt);
        Set(name, nullptr);
      }
    }
    ~EnvironmentScope() {
      for (const auto& [name, value] : previous)
        Set(name.c_str(), value ? value->c_str() : nullptr);
    }
  } environment;
  using gufo::server::BuildEffectiveSettings;
  auto resolved = gufo::json::Value::object();
  resolved["ram_bytes"] = 1234;
  const auto unset = BuildEffectiveSettings(resolved, "test-version", "revision", "build-id");
  Expect(unset.member_str("version") == "test-version" &&
             unset.member_str("revision") == "revision" &&
             unset.member_str("exe_build_id") == "build-id",
         "debug settings identify the exact executable");
  const auto* raw = unset.find("environment");
  Expect(raw && raw->is_object(), "settings contain an explicit environment catalog");
  for (const auto* name : {"GUFO_EVAL_API_KEY", "PATH", "TEMP"})
    Expect(!raw->contains(name),
           std::string("settings exclude unsafe environment: ") + name);
  for (const auto* name : gufo::server::kDebugSettingsEnvironment) {
    if (std::string_view(name) == "GUFO_THINK_MARKER_ENTROPY_GATE")
      Expect(!raw->contains(name),
             "unset entropy gate has no raw environment key");
    else
      Expect(raw->find(name) && raw->find(name)->is_null(),
             std::string("unset environment is visible: ") + name);
  }
  const auto* values = unset.find("resolved");
  Expect(values && values->member_size("ram_bytes") == 1234,
         "settings retain the owning component's resolved auto budget");
  const auto* behaviour = values->find("behaviour");
  Expect(behaviour && !behaviour->find("enabled")->as_bool() &&
             !behaviour->contains("marker_entropy_gate") &&
             behaviour->member_double("eog_min_prob") == -1.0 &&
             behaviour->member_size("think_guard_escalate") == 16,
         "disabled behaviour uses its parser's actual minimum default");
  const auto unset_request =
      BuildEffectiveSettings(unset, gufo::sampling::SamplingConfig{}, false);
  Expect(!unset_request.find("resolved")
                 ->find("behaviour")
                 ->contains("marker_entropy_gate") &&
             !unset_request.find("environment")
                  ->contains("GUFO_THINK_MARKER_ENTROPY_GATE"),
         "unset request settings retain the prior key shape");
  const auto* tuning = values->find("platform_tuning");
  for (const auto& field : gufo::platform::detail::kTuningFields) {
    const auto* value = tuning->find(std::string(field.name));
    Expect(value && value->is_bool() &&
               value->as_bool() == gufo::platform::PlatformTuning().*field.member,
           "settings echo each resolved platform tuning switch");
  }
  const auto* lenient = values->find("lenient");
  const auto& config = gufo::models::qwen38_flash_next::MtpLenient();
  Expect(lenient && lenient->find("enabled")->as_bool() == config.enabled &&
             lenient->member_double("lambda") == config.lambda &&
             lenient->find("think_only")->as_bool() == config.think_only,
         "settings use the shared lenient parser");
  EnvironmentScope::Set("GUFO_THINK_EOG_GUARD", "1");
  EnvironmentScope::Set("GUFO_THINK_GUARD_ESCALATE", "7");
  EnvironmentScope::Set("GUFO_EOG_MIN_PROB", "0.25");
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS_BY_EFFORT", "low:0.5,xhigh:2");
  EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", "0.75");
  EnvironmentScope::Set("GUFO_MTP_TRACE", "01");
  EnvironmentScope::Set("GUFO_DISK_BOUNDARY_ALIGN", "128");
  const auto configured = BuildEffectiveSettings(resolved, "v", "r", "b");
  const auto* effective = configured.find("resolved");
  behaviour = effective->find("behaviour");
  Expect(
      behaviour->find("think_eog_guard")->as_bool() &&
          behaviour->member_double("eog_min_prob") == 0.25 &&
          behaviour->member_size("think_guard_escalate") == 7 &&
          behaviour->member_double("marker_entropy_gate") == 0.75 &&
          behaviour->find("marker_bias_by_effort")->member_double("high") ==
              2.0 &&
          behaviour->find("marker_bias_by_effort")->member_double("minimal") ==
              0.5,
      "settings return parsed numeric values and resolved effort aliases");
  gufo::sampling::SamplingConfig request_policy;
  auto request_behaviour = std::make_shared<gufo::sampling::BehaviourConfig>();
  request_behaviour->think_eog_guard = true;
  request_behaviour->think_guard_escalate = 3;
  request_behaviour->marker_entropy_gate = 0.25;
  request_policy.behaviour = request_behaviour;
  const auto request_settings =
      BuildEffectiveSettings(configured, request_policy, true);
  Expect(request_settings.find("resolved")
                 ->find("behaviour")
                 ->member_size("think_guard_escalate") == 3,
         "request settings echo the actual guard escalation policy");
  Expect(request_settings.find("resolved")
                     ->find("behaviour")
                     ->member_double("marker_entropy_gate") == 0.25 &&
             configured.find("environment")
                     ->member_str("GUFO_THINK_MARKER_ENTROPY_GATE") == "0.75",
         "settings show the raw gate env and actual request threshold");
  request_behaviour->marker_entropy_gate.reset();
  const auto ungated_request =
      BuildEffectiveSettings(configured, request_policy, true);
  Expect(!ungated_request.find("resolved")
              ->find("behaviour")
              ->contains("marker_entropy_gate"),
         "request without a gate removes a cached process threshold");
  EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", nullptr);
  const auto unconfigured = BuildEffectiveSettings(resolved, "v", "r", "b");
  Expect(!unconfigured.find("environment")
                 ->contains("GUFO_THINK_MARKER_ENTROPY_GATE") &&
             !unconfigured.find("resolved")
                  ->find("behaviour")
                  ->contains("marker_entropy_gate"),
         "unsetting the gate removes both process keys");
  Expect(!effective->find("mtp_trace")->as_bool(),
         "MTP trace accepts only the literal one");
  Expect(effective->member_size("disk_boundary_align") == 128,
         "settings echo parsed disk boundary alignment");
  EnvironmentScope::Set("GUFO_MTP_TRACE", "1");
  Expect(BuildEffectiveSettings(resolved, "v", "r", "b")
             .find("resolved")->find("mtp_trace")->as_bool(),
         "MTP trace literal one is enabled");
  Expect(gufo::server::DebugSettingsListsEnvironment("NO_COLOR"),
         "irrelevant environment has an explicit documented allowance");
}

// Raw stderr gate. An allowance is keyed by file, exact call text and the
// 1-based occurrence among identical call texts in that file, in source order.
// Line numbers are not part of the key, so edits that only move code do not
// break the gate. A new call, a changed call and a deleted call all still fail.
const std::regex& StderrCallRegex() {
  static const std::regex calls(
      R"scan((?:std::)?\bfprintf\s*\(\s*stderr\s*,[\s\S]*?;|(?:std::)?\bfputs\s*\([^;]*?\bstderr\s*\)[\s\S]*?;|std::(?:cerr|clog)\s*(?:<<|\.write\s*\()[\s\S]*?;|(?:::)?\bwrite\s*\(\s*STDERR_FILENO\s*,[\s\S]*?;)scan");
  return calls;
}

std::string StderrSiteKey(const std::string& file, std::size_t occurrence,
                          const std::string& call) {
  return file + "\n" + std::to_string(occurrence) + "\n" + call;
}

/// Every raw stderr call in `source`, as (key, call) in source order.
std::vector<std::pair<std::string, std::string>> FindStderrSites(
    const std::string& file, const std::string& source) {
  std::vector<std::pair<std::string, std::string>> sites;
  std::map<std::string, std::size_t> seen;
  for (std::sregex_iterator m(source.begin(), source.end(), StderrCallRegex()),
       end;
       m != end; ++m) {
    const auto call = m->str();
    sites.emplace_back(StderrSiteKey(file, ++seen[call], call), call);
  }
  return sites;
}

/// Allowance table from the allow-list JSON. Problems are appended to `out`.
std::map<std::string, std::string> ParseStderrAllowlist(
    const gufo::json::Value& allowance, std::vector<std::string>& out) {
  std::map<std::string, std::string> table;
  const auto* sites = allowance.find("sites");
  if (!sites || !sites->is_array()) {
    out.push_back("stderr allow-list has no sites array");
    return table;
  }
  for (const auto& site : sites->items()) {
    const auto file = site.member_str("file");
    const auto call = site.member_str("call");
    const auto occurrence = site.member_str("occurrence");
    const bool digits =
        !occurrence.empty() &&
        occurrence.find_first_not_of("0123456789") == std::string::npos &&
        std::stoull(occurrence) >= 1;
    if (file.empty() || call.empty() || !digits ||
        site.member_str("reason").empty()) {
      out.push_back("stderr allowance needs file, call, occurrence (>= 1) and "
                    "reason: " + file + " / " + call);
      continue;
    }
    if (!table.emplace(StderrSiteKey(file, std::stoull(occurrence), call),
                       site.member_str("reason"))
             .second)
      out.push_back(file + ": duplicate stderr allowance, occurrence " +
                    occurrence + ": " + call);
  }
  return table;
}

/// Checks sources (file -> text) against the allowance table. Returns one
/// message per unreviewed call (with file and call text) and per stale entry.
std::vector<std::string> StderrGateProblems(
    const std::map<std::string, std::string>& sources,
    std::map<std::string, std::string> allowed) {
  std::vector<std::string> problems;
  for (const auto& [file, source] : sources)
    for (const auto& [key, call] : FindStderrSites(file, source))
      if (allowed.erase(key) != 1)
        problems.push_back(file + ": unreviewed raw stderr call: " + call);
  for (const auto& entry : allowed) {
    const auto first = entry.first.find('\n');
    const auto second = entry.first.find('\n', first + 1);
    problems.push_back(entry.first.substr(0, first) +
                       ": stale stderr allowance, occurrence " +
                       entry.first.substr(first + 1, second - first - 1) +
                       ": " + entry.first.substr(second + 1));
  }
  return problems;
}

void TestStderrGateIsLineIndependent() {
  const std::string call_a = "std::fprintf(stderr, \"a\\n\");";
  const std::string call_b = "std::cerr << \"b\";";
  const std::string source =
      "void f() {\n  " + call_a + "\n  " + call_b + "\n  " + call_a + "\n}\n";
  const auto sites = FindStderrSites("x.cpp", source);
  Expect(sites.size() == 3, "stderr gate finds all three calls");
  Expect(sites[0].first != sites[2].first &&
             sites[0].first == StderrSiteKey("x.cpp", 1, call_a) &&
             sites[2].first == StderrSiteKey("x.cpp", 2, call_a) &&
             sites[1].first == StderrSiteKey("x.cpp", 1, call_b),
         "identical calls get distinct 1-based occurrence indices");
  const std::map<std::string, std::string> allowed = {
      {StderrSiteKey("x.cpp", 1, call_a), "r"},
      {StderrSiteKey("x.cpp", 2, call_a), "r"},
      {StderrSiteKey("x.cpp", 1, call_b), "r"}};
  Expect(StderrGateProblems({{"x.cpp", source}}, allowed).empty(),
         "the complete allowance passes");
  const std::string shifted = "// one\n// two\n// three\n\n\n" + source;
  Expect(StderrGateProblems({{"x.cpp", shifted}}, allowed).empty(),
         "shifting lines does not fail the gate");
  const std::string added = source + "void g() { std::fputs(\"n\", stderr); }\n";
  auto problems = StderrGateProblems({{"x.cpp", added}}, allowed);
  Expect(problems.size() == 1 &&
             problems[0].find("x.cpp: unreviewed raw stderr call: "
                              "std::fputs(\"n\", stderr);") == 0,
         "a new raw stderr call fails with file and call text");
  const std::string third = source + "void g() { " + call_a + " }\n";
  problems = StderrGateProblems({{"x.cpp", third}}, allowed);
  Expect(problems.size() == 1 &&
             problems[0].find("unreviewed raw stderr call: " + call_a) !=
                 std::string::npos,
         "a third copy of an allowed call is a new call");
  const std::string deleted =
      "void f() {\n  " + call_a + "\n  " + call_b + "\n}\n";
  problems = StderrGateProblems({{"x.cpp", deleted}}, allowed);
  Expect(problems.size() == 1 &&
             problems[0].find("x.cpp: stale stderr allowance, occurrence 2: " +
                              call_a) == 0,
         "a deleted call leaves a reported stale entry");
  std::vector<std::string> parse_problems;
  const auto table = ParseStderrAllowlist(
      gufo::json::parse(R"({"schema":2,"sites":[
        {"file":"x.cpp","call":"c","reason":"r","occurrence":"1"},
        {"file":"x.cpp","call":"c","reason":"r","occurrence":"2"},
        {"file":"x.cpp","call":"c","reason":"r","occurrence":"2"},
        {"file":"x.cpp","call":"c","reason":"r"},
        {"file":"x.cpp","call":"c","reason":"r","occurrence":"0"}]})"),
      parse_problems);
  Expect(table.size() == 2 && parse_problems.size() == 3,
         "allow-list parser rejects duplicate, missing and zero occurrence");
}

void DebugHooksSettingsEnvironmentTest() {
  namespace fs = std::filesystem;
  const fs::path root = fs::absolute(fs::path(__FILE__)).parent_path()
                            .parent_path().parent_path();
  const std::regex calls(
      R"debug((std::getenv|\bgetenv|BehaviourSwitch|BehaviourNumber)\s*\(\s*([^)]*)\))debug");
  const std::regex literal(R"debug(^"([A-Za-z_][A-Za-z_0-9]*)")debug");
  std::map<std::string, std::size_t> dynamic_reads;
  std::size_t literal_reads = 0;
  for (const char* directory : {
           "src/core", "src/cli/serve", "src/models/qwen38_flash_next"}) {
    const auto source = root / directory;
    Expect(fs::is_directory(source), "environment completeness source root exists");
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
      if (entry.path() == root / "src/cli/serve/release_switches.inc")
        continue;
      if (!entry.is_regular_file())
        continue;
      const auto extension = entry.path().extension().string();
      if (extension != ".cpp" && extension != ".hpp" && extension != ".hip" &&
          extension != ".h" && extension != ".cc" && extension != ".c" &&
          extension != ".inc")
        continue;
      std::ifstream input(entry.path(), std::ios::binary);
      Expect(input.good(), "environment completeness can read each source file");
      const std::string text((std::istreambuf_iterator<char>(input)), {});
      const std::string path = fs::relative(entry.path(), root).generic_string();
      for (std::sregex_iterator match(text.begin(), text.end(), calls), end;
           match != end; ++match) {
        const std::string function = (*match)[1].str();
        const std::string argument = (*match)[2].str();
        std::smatch name;
        if (std::regex_search(argument, name, literal)) {
          ++literal_reads;
          const auto& catalog = gufo::server::ReleaseSwitchTable();
          const bool catalogued =
              std::any_of(catalog.begin(), catalog.end(), [&](const auto& row) {
                return row.member_str("name") == name[1].str() &&
                       row.member_str("class") != "REMOVED";
              });
          Expect(catalogued ||
                     gufo::server::DebugSettingsListsEnvironment(name[1].str()),
                 path + ": environment missing from echo/allow-list: " +
                     name[1].str());
        } else if (function == "std::getenv" || function == "getenv") {
          Expect(argument == "name" &&
                     (path == "src/cli/serve/behaviour_options.hpp" ||
                      path == "src/cli/serve/debug_settings.hpp"),
                 path + ": unreviewed dynamic environment read: " + argument);
          ++dynamic_reads[path];
        } else {
          Expect(path == "src/cli/serve/behaviour_options.hpp" &&
                     (argument == "const char* name" ||
                      argument == "const char* name, double disabled"),
                 path + ": unreviewed dynamic behaviour wrapper call: " + argument);
        }
      }
    }
  }

  // The original reviewed-wrapper check above retains its scope and counts.
  // All source literals are now catalogued, including other model families.
  const auto& rows = gufo::server::ReleaseSwitchTable();
  std::map<std::string, std::string> classes, kinds;
  for (const auto& row : rows) {
    classes[row.member_str("name")] = row.member_str("class");
    kinds[row.member_str("name")] = row.member_str("kind");
  }
  const std::regex all_names(R"debug("(GUFO_[A-Z0-9_]+)")debug");
  const std::regex options(
      R"debug(\.Add(?:Option|Flag|InverseFlag|CustomOption)\s*\(\s*"[^"]*"\s*,\s*"(--[a-z0-9-]+)")debug");
  for (const auto& entry : fs::recursive_directory_iterator(root / "src")) {
    if (!entry.is_regular_file())
      continue;
    const auto extension = entry.path().extension().string();
    if (extension != ".cpp" && extension != ".hpp" && extension != ".hip" &&
        extension != ".h" && extension != ".cc" && extension != ".c" &&
        extension != ".inc")
      continue;
    std::ifstream input(entry.path(), std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(input)), {});
    if (entry.path() == root / "src/cli/serve/release_switches.inc")
      continue;
    const std::regex getenv_literal(
        R"debug((?:std::getenv|\bgetenv)\s*\(\s*"([A-Za-z_][A-Za-z_0-9]*)")debug");
    for (std::sregex_iterator match(text.begin(), text.end(), getenv_literal),
         end;
         match != end; ++match) {
      const auto name = (*match)[1].str();
      Expect(classes.contains(name) || name == "NO_COLOR",
             entry.path().string() +
                 ": literal getenv missing from catalog: " + name);
      Expect(!classes.contains(name) || classes[name] != "REMOVED",
             "removed getenv has a reader: " + name);
    }
    for (std::sregex_iterator match(text.begin(), text.end(), all_names), end;
         match != end; ++match) {
      const auto name = (*match)[1].str();
      Expect(classes.contains(name),
             entry.path().string() +
                 ": source literal missing from release table: " + name);
      Expect(classes[name] != "REMOVED",
             entry.path().string() +
                 ": removed switch remains in source: " + name);
    }
    if (entry.path() == root / "src/cli/serve/serve.cpp") {
      for (std::sregex_iterator match(text.begin(), text.end(), options), end;
           match != end; ++match)
        Expect(kinds[(*match)[1].str()] == "flag",
               "serve flag missing from release table: " + (*match)[1].str());
    }
  }
  // Explicit serving files; each exception is tied to its exact call text.
  // New sites fail until individually reviewed in the allow-list.
  const auto read_source = [&](const fs::path& file) {
    std::ifstream input(file);
    Expect(input.good(), "stderr scan can read its explicit input");
    std::string text((std::istreambuf_iterator<char>(input)), {});
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
  };
  std::vector<std::string> allowlist_problems;
  const auto exceptions = ParseStderrAllowlist(
      gufo::json::parse(read_source(root / "tools/ci/log_stderr_allowlist.json")),
      allowlist_problems);
  Expect(allowlist_problems.empty(),
         allowlist_problems.empty() ? "" : allowlist_problems.front());
  std::map<std::string, std::string> stderr_sources;
  for (const char* name :
       {"src/models/qwen38_flash_next/kernels/rocm/executor.cpp",
        "src/models/qwen38_flash_next/engine.cpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp",
        "src/cli/serve/text_generation_scheduler.cpp",
        "src/cli/serve/http_server.cpp", "src/cli/serve/text_model_runner.cpp",
        "src/models/qwen38_flash_next/kernels/rocm/trace.hpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/mmq.hpp",
        "src/models/qwen38_flash_next/kernels/rocm/device_model.cpp",
        "src/models/qwen38_flash_next/mtp_sampling.hpp",
        "src/cli/serve/continuation_disk_store.cpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_ggml_stubs.hip.cpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_ggml_stubs.h",
        "src/cli/serve/serve.cpp", "src/cli/serve/serve_command.cpp",
        "src/cli/serve/logging.cpp"})
    stderr_sources[name] = read_source(root / name);
  // Unreviewed calls and stale entries both fail, each with file and call text.
  const auto stderr_problems = StderrGateProblems(stderr_sources, exceptions);
  for (const auto& problem : stderr_problems)
    std::cerr << problem << "\n";
  Expect(stderr_problems.empty(),
         "raw stderr calls match the allow-list (see the lines above)");
  // Five lines inserted at the top of serve.cpp (in memory, the real file stays
  // untouched) move every call and must not change the result.
  const std::string serve_file = "src/cli/serve/serve.cpp";
  std::map<std::string, std::string> serve_allowed;
  for (const auto& [key, reason] : exceptions)
    if (key.compare(0, serve_file.size() + 1, serve_file + "\n") == 0)
      serve_allowed.emplace(key, reason);
  Expect(serve_allowed.size() > 10, "serve.cpp has reviewed stderr allowances");
  Expect(StderrGateProblems(
             {{serve_file,
               "// shift 1\n// shift 2\n// shift 3\n// shift 4\n// shift 5\n" +
                   stderr_sources[serve_file]}},
             serve_allowed)
             .empty(),
         "five inserted lines in serve.cpp do not break the stderr gate");
  for (const char* name :
       {"src/models/qwen38_flash_next/kernels/rocm/mmq/mmq.hpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_ggml_stubs.hip.cpp",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_ggml_stubs.h"}) {
    const auto source = read_source(root / name);
    const std::regex fatal_print(
        R"scan(fprintf\(stderr, "(?:mmq_x_best=|HIP error:|qfn_stack_pool:|GGML_ASSERT|GGML_ABORT))scan");
    for (std::sregex_iterator m(source.begin(), source.end(), fatal_print), end;
         m != end; ++m) {
      const auto prefix = source.substr(0, m->position());
      const auto flush =
          prefix.rfind("Logger::FlushSync(std::chrono::seconds(2))");
      Expect(
          flush != std::string::npos && prefix.size() - flush < 100,
          std::string(name) + ": fatal print must have a bounded prefix drain");
    }
  }
  const auto& stderr_calls = StderrCallRegex();
  const std::string new_site = "std::fprintf ( stderr, \"new site\\n\" );";
  Expect(std::regex_search(new_site, stderr_calls),
         "stderr scan detects a new call with whitespace");
  Expect(
      std::regex_search(std::string("std::fputs(line, stderr);"), stderr_calls),
      "stderr scan detects fputs");
  Expect(std::regex_search(std::string("std::cerr << line;"), stderr_calls),
         "stderr scan detects stream output");

  const auto read_generated = [&](const fs::path& file) {
    std::ifstream input(file);
    Expect(input.good(), "generated release file exists");
    std::string text((std::istreambuf_iterator<char>(input)), {});
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
  };
  gufo::server::ValidateSwitchTable(rows);
  Expect(gufo::server::RenderReleaseInclude(rows) ==
             read_generated(root / "src/cli/serve/release_switches.inc"),
         "release header matches compiled table");
  Expect(gufo::server::RenderReleaseDocs(rows) ==
             read_generated(root / "docs/FLAGS.md"),
         "release docs match compiled table");
  Expect(gufo::server::RenderReleaseDocs(rows) ==
             read_generated(root / "tests/fixtures/release_flags_golden.md"),
         "C++ docs renderer matches Python golden");

  Expect(literal_reads > 20, "environment completeness actually scanned the sources");
  Expect(dynamic_reads["src/cli/serve/behaviour_options.hpp"] == 2 &&
             dynamic_reads["src/cli/serve/debug_settings.hpp"] == 1,
         "only the reviewed environment wrappers use dynamic names");
  Expect(!gufo::server::DebugSettingsListsEnvironment("GUFO_NEW_UNREVIEWED_SWITCH"),
         "a newly added environment read fails until explicitly listed");
}

void TestNullReasoningHistory() {
  // Production common/chat.cpp:454 assigned null to a string and rejected
  // valid OpenAI history. gufo already accepts null; retain that protection.
  for (const bool stream : {false, true}) {
    FakeBackend backend;
    backend.pieces = {"next"};
    auto body = gufo::json::parse(R"({"model":"test-model","messages":[
      {"role":"assistant","content":"previous","reasoning_content":null},
      {"role":"user","content":"continue"}]})");
    body["stream"] = stream;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "null reasoning history is accepted");
    if (response.streaming_body)
      response.streaming_body([](std::string_view) { return true; });
    Expect(backend.last_request.messages[0].thought.empty() &&
               backend.last_request.messages[0].content == "previous",
           "null reasoning means absent reasoning, preserving visible history");
  }
}

void TestProductionSamplingSelection() {
  using namespace gufo::sampling;
  SamplingConfig configured;
  auto policy = std::make_shared<BehaviourConfig>();
  policy->think_eog_guard = true;
  configured.behaviour = policy;
  for (const bool thinking : {false, true}) {
    const auto native = ResolveTextSampling(TextModelPreset::kQwen38, thinking,
                                           configured, {}, false);
    const auto production = ResolveTextSampling(
        TextModelPreset::kQwen38, thinking, configured, {}, true);
    Expect(native.temperature == (thinking ? 1.0F : 0.7F) &&
               native.top_p == (thinking ? 0.95F : 0.8F) &&
               native.presence_penalty == (thinking ? 0.0F : 1.5F),
           "native gufo presets remain selectable");
    Expect(production.temperature == (thinking ? 1.0F : 0.7F) &&
               production.top_p == 0.95F && production.top_k == 20 &&
               production.min_p == 0.0F && production.presence_penalty == 0.0F,
           "production thinking on/off uses the launcher distribution");
    Expect(production.behaviour->think_eog_guard &&
               production.behaviour->production_sampling &&
               native.behaviour == configured.behaviour,
           "preset selection preserves the request policy and native comparator");
    Expect(production.behaviour->native_temperature == native.temperature &&
               production.behaviour->native_top_p == native.top_p &&
               production.behaviour->native_presence_penalty ==
                   native.presence_penalty,
           "preset comparator records the effective native distribution");
  }
  configured.temperature = 0.0F;
  configured.top_p = 0.0F;
  configured.top_k = 0;
  configured.min_p = 0.25F;
  configured.presence_penalty = 0.5F;
  configured.seed = 123;
  const auto explicit_config = ResolveTextSampling(
      TextModelPreset::kQwen38, false, configured, SamplingOverrides::All(),
      true);
  Expect(explicit_config.temperature == 0.0F && explicit_config.top_p == 0.0F &&
             explicit_config.top_k == 0 && explicit_config.min_p == 0.25F &&
             explicit_config.presence_penalty == 0.5F &&
             explicit_config.seed == 123,
         "explicit overrides, including zero, win over production presets");
  Expect(explicit_config.behaviour->native_temperature == 0.0F &&
             explicit_config.behaviour->native_top_k == 0 &&
             explicit_config.behaviour->native_top_p == 0.0F &&
             explicit_config.behaviour->native_presence_penalty == 0.5F,
         "explicit overrides also apply to the native intervention comparator");
}

void TestProductionParserRecovery() {
  // Port the output cases from wt-pfbt/tests/test-chat.cpp:2157-2175.
  const std::string replacement = "\xEF\xBF\xBD";
  struct Case {
    std::string raw;
    std::string thought;
    std::string content;
  };
  for (const auto& item : {
           Case{"We need to think further.", "We need to think further.", ""},
           Case{"</think>\n\n", "", ""},
           Case{"</think>\n\nThe bound is $x^2+", "", "The bound is $x^2+"},
           Case{std::string("</think>\n\nCost: ") + "\xE2\x82", "",
                "Cost: " + replacement},
           Case{std::string("Reasoning ") + "\xB0" + "geq 0</think>Answer",
                "Reasoning " + replacement + "geq 0", "Answer"}}) {
    for (const bool stream : {false, true}) {
      FakeBackend backend;
      for (const char byte : item.raw)
        backend.pieces.emplace_back(1, byte);
      auto body = gufo::json::parse(R"({"model":"test-model","messages":[
        {"role":"user","content":"solve"}],
        "chat_template_kwargs":{"enable_thinking":true}})");
      body["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "partial production output stays valid");
      std::string thought, content;
      if (!stream) {
        const auto output = gufo::json::parse(response.body);
        const auto& message =
            *output.find("choices")->items()[0].find("message");
        thought = message.member_str("reasoning_content");
        content = message.member_str("content");
      } else {
        response.streaming_body([&](std::string_view chunk) {
          if (chunk == "data: [DONE]\n\n")
            return true;
          const auto event = gufo::json::parse(chunk.substr(6));
          for (const auto& choice : event.find("choices")->items()) {
            if (const auto* delta = choice.find("delta")) {
              thought += delta->member_str("reasoning_content");
              content += delta->member_str("content");
            }
          }
          return true;
        });
      }
      Expect(thought == item.thought && content == item.content,
             "split UTF-8, unfinished reasoning and truncated math are retained");
    }
  }
  for (const bool malformed : {false, true}) {
    FakeBackend backend;
    backend.pieces = {
        std::string(16384, 'r') + "\n</think>\n\n<tool_call>\n"
        "<function=run_in_terminal>\n<parameter=command>\npwd\n" +
        (malformed ? "" : "</parameter>\n</function>\n</tool_call>")};
    const auto response = gufo::server::HandleOpenAiChat(Request(R"({
      "model":"test-model","messages":[{"role":"user","content":"pwd"}],
      "chat_template_kwargs":{"enable_thinking":true},"tools":[{
        "type":"function","function":{"name":"run_in_terminal",
          "parameters":{"type":"object","properties":{
            "command":{"type":"string"}},"required":["command"]}}}]})"),
        backend);
    Expect(response.status == 200, "long reasoning tool output is valid");
    const auto output = gufo::json::parse(response.body);
    const auto& message = *output.find("choices")->items()[0].find("message");
    const auto* calls = message.find("tool_calls");
    Expect(malformed ? calls == nullptr : calls && calls->items().size() == 1,
           "only a complete call invokes a tool; malformed output is never invented");
    if (calls)
      Expect(calls->items()[0].find("function")->member_str("name") ==
                 "run_in_terminal",
             "long reasoning preserves the production tool name");
  }
}

void TestPiReasoningControlsAndOutputFraming() {
  FakeBackend backend;
  backend.pieces = {"I should verify this.", "</think>\n\n", "Forty-two."};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"What is six times seven?"}],
        "reasoning_effort":"high",
        "chat_template_kwargs":{
          "enable_thinking":true,
          "reasoning_effort":"high",
          "preserve_thinking":false
        }
      })"),
                                                       backend);

  Expect(response.status == 200, "Pi reasoning request is accepted");
  Expect(backend.last_request.reasoning.enabled == true,
         "Pi enable_thinking reaches the backend");
  Expect(backend.last_request.reasoning.effort == gufo::ReasoningEffort::kHigh,
         "Pi reasoning effort reaches the backend");
  Expect(backend.last_request.reasoning.preserve_thinking == false,
         "Pi preservation control reaches the backend");
  Expect(response.body.find(R"("reasoning_content":"I should verify this.")") !=
             std::string::npos,
         "Prompt-opened reasoning is returned separately");
  Expect(response.body.find(R"("content":"Forty-two.")") != std::string::npos,
         "Visible answer excludes reasoning");

  for (const auto effort : {"low", "medium", "xhigh"}) {
    const auto native = gufo::server::HandleOpenAiChat(
        Request("{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hello\"}],\"chat_template_kwargs\":{"
                "\"reasoning_effort\":\"" +
                std::string(effort) + "\"}}"),
        backend);
    Expect(
        native.status == 200 && backend.last_request.reasoning.enabled == true,
        "Native effort alone enables thinking");
  }
  backend.pieces = {"Done."};
  const auto disabled = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low"}})"),
      backend);
  Expect(
      disabled.status == 200 && backend.last_request.reasoning.enabled == false,
      "Template enable_thinking=false suppresses the configured effort");
  Expect(disabled.body.find(R"("content":"Done.")") != std::string::npos,
         "Explicit thinking-off produces visible content");
  const auto vision_ids = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"add_vision_id":true}})"),
      backend);
  Expect(vision_ids.status == 200 && backend.last_request.add_vision_id,
         "Official add_vision_id option reaches the model formatter");
  const auto invalid_ids = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"add_vision_id":"true"}})"),
      backend);
  Expect(invalid_ids.status == 400, "add_vision_id requires a boolean");
}

void TestPiNativeDeepSeekThinkingObject() {
  FakeBackend backend;
  backend.pieces = {"Check.", "</think>", "Done."};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Check this."}],
        "thinking":{"type":"enabled"},
        "reasoning_effort":"high"
      })"),
                                                       backend);
  Expect(response.status == 200,
         "Pi native DeepSeek thinking object is accepted");
  Expect(backend.last_request.reasoning.enabled == true,
         "Pi DeepSeek thinking.type enables reasoning");
  Expect(backend.last_request.reasoning.effort == gufo::ReasoningEffort::kHigh,
         "Pi DeepSeek reasoning effort reaches the backend");

  const auto disabled = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Answer directly."}],
        "thinking":{"type":"disabled"}
      })"),
                                                       backend);
  Expect(disabled.status == 200,
         "Pi native DeepSeek disabled thinking object is accepted");
  Expect(backend.last_request.reasoning.enabled == false,
         "Pi DeepSeek thinking.type disables reasoning");
}

void TestStreamingPromptOpenedReasoning() {
  FakeBackend backend;
  backend.pieces = {"Check", " carefully", "</thi", "nk>\n\n", "Done"};
  auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Check it."}],
        "reasoning_effort":"xhigh",
        "stream":true
      })"),
                                                 backend);
  Expect(response.status == 200, "Streaming reasoning request is accepted");
  std::string output;
  response.streaming_body([&](std::string_view chunk) {
    output.append(chunk);
    return true;
  });
  Expect(output.find(R"("reasoning_content":"Check")") != std::string::npos,
         "Streaming reasoning uses reasoning_content deltas");
  Expect(output.find(R"("content":"Done")") != std::string::npos,
         "Streaming answer switches to content after think end");
  Expect(output.find(R"("content":"Check")") == std::string::npos,
         "Reasoning is never exposed as visible content");
}

void TestInitialOutputPhases() {
  using gufo::json::Value;
  struct Case {
    std::string text;
    bool automatic{false};
  };
  for (const auto& item :
       {Case{"<think>literal example</think>"},
        Case{" \n<think>literal example</think>\nanswer"},
        Case{"<think>unfinished example"},
        Case{"<think>Check carefully.</think>\nAnswer.", true}}) {
    for (const bool stream : {false, true}) {
      FakeBackend backend;
      // Disabled thinking starts in the content phase. Tags requested as
      // literal output remain data. Automatic detection still recognizes an
      // initial reasoning block, including across token boundaries.
      if (item.automatic)
        backend.initial_output_state_override =
            FakeBackend::InitialOutputState::kAuto;
      for (const char byte : item.text)
        backend.pieces.emplace_back(1, byte);
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[]
      })");
      if (!item.automatic)
        body["reasoning_effort"] = "none";
      auto message = Value::object();
      message["role"] = "user";
      message["content"] =
          item.automatic
              ? "Answer carefully."
              : "Copy this XML exactly, without code fences: " + item.text;
      body["messages"].push_back(std::move(message));
      body["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "output-phase request succeeds");
      std::string content, reasoning;
      const auto collect = [&](const Value& message) {
        content += message.member_str("content");
        reasoning += message.member_str("reasoning_content");
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]")) {
            const auto event = gufo::json::parse(payload);
            for (const auto& choice : event.find("choices")->items())
              collect(*choice.find("delta"));
          }
          return true;
        });
      } else {
        const auto output = gufo::json::parse(response.body);
        collect(*output.find("choices")->items()[0].find("message"));
      }
      Expect(content == (item.automatic ? "Answer." : item.text) &&
                 reasoning == (item.automatic ? "Check carefully." : ""),
             "initial output phase controls reasoning tag interpretation");
    }
  }
}

void TestReasoningAnswerSeparator() {
  using gufo::json::Value;
  for (const bool thinking : {false, true})
    for (const bool stream : {false, true})
      for (const bool split : {false, true})
        for (const int mode : {0, 1, 2})
          for (const std::string separator :
               {"", "\n\n", "\r\n\r\n", " \t\n\n\f\v"})
            for (const bool truncated : {false, true}) {
              const std::string answer =
                  truncated   ? ""
                  : mode == 2 ? R"({"answer":9,"text":"\n\nliteral"})"
                              : "9\n\nNext paragraph.\n    Indented line.";
              const std::string raw =
                  (thinking ? "Check</think>" : "") + separator + answer;
              FakeBackend backend;
              if (truncated)
                backend.finish_reason = FakeBackend::FinishReason::kLength;
              backend.tool_format =
                  gufo::sampling::JsonConstraint::ToolFormat::kQwen;
              if (split) {
                for (const char byte : raw)
                  backend.pieces.emplace_back(1, byte);
              } else {
                backend.pieces = {raw};
              }
              auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"4+5?"}],
            "temperature":0})");
              body["stream"] = stream;
              body["reasoning_effort"] = thinking ? "low" : "none";
              if (mode == 1) {
                body["tools"] =
                    gufo::json::parse(R"([{"type":"function","function":{
              "name":"unused","parameters":{"type":"object","properties":{},
              "additionalProperties":false}}}])");
                body["tool_choice"] = "auto";
              } else if (mode == 2) {
                body["response_format"] = gufo::json::parse(R"({
              "type":"json_schema","json_schema":{"name":"answer","strict":true,
              "schema":{"type":"object","properties":{"answer":{"type":"integer"},
              "text":{"type":"string"}},
              "required":["answer","text"],"additionalProperties":false}}})");
              }
              const auto response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
              Expect(response.status == 200,
                     "reasoning separator request succeeds");
              std::string content;
              if (stream) {
                response.streaming_body([&](std::string_view piece) {
                  const auto payload = piece.substr(piece.find("data: ") + 6);
                  if (!payload.starts_with("[DONE]")) {
                    const auto event = gufo::json::parse(payload);
                    for (const auto& choice : event.find("choices")->items())
                      if (const auto* delta = choice.find("delta"))
                        content += delta->member_str("content");
                  }
                  return true;
                });
              } else {
                const auto parsed = gufo::json::parse(response.body);
                content = parsed.find("choices")
                              ->items()[0]
                              .find("message")
                              ->member_str("content");
              }
              const auto expected = thinking ? answer : separator + answer;
              if (content != expected)
                std::cerr << "separator thinking=" << thinking
                          << " stream=" << stream << " split=" << split
                          << " mode=" << mode << " truncated=" << truncated
                          << " content=" << Value(content).dump() << "\n";
              Expect(content == expected,
                     "only the reasoning separator is removed; answer "
                     "whitespace survives");
            }
}

void TestConflictingReasoningControlsAreRejected() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "reasoning_effort":"high",
        "chat_template_kwargs":{"enable_thinking":false}
      })"),
                                                       backend);
  Expect(response.status == 400, "Conflicting reasoning controls are rejected");
  Expect(response.body.find("invalid_reasoning") != std::string::npos,
         "Reasoning conflict has a stable error code");
  Expect(backend.chat_calls.load() == 0,
         "Invalid reasoning request never reaches generation");
}

void TestWrongModelIsRejected() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"wrong-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                       backend);

  Expect(response.status == 404, "Unknown model is rejected");
  Expect(response.body.find("model_not_found") != std::string::npos,
         "Unknown model returns a stable error code");
  Expect(response.body.find("this process serves 'test-model'") !=
             std::string::npos,
         "Unknown model names the model being served");
  Expect(response.body.find("is not served: this process serves") !=
             std::string::npos,
         "Unknown model remedy survives the log punctuation filter");
  Expect(backend.chat_calls.load() == 0,
         "Rejected request never reaches the backend");
  const auto alias = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"bubo", "messages":[{"role":"user","content":"hello"}]
      })"),
                                                    backend);
  Expect(alias.status == 200, "Bubo alias reaches the served model");
}

void TestClientIdentityReachesBackend() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  for (const auto* header : {"pi-agent-2", "contains spaces", "rotated"}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})",
        {{"X-Client-ID", header}});
    request.client_id = "192.0.2.7";
    const auto response = gufo::server::HandleOpenAiChat(request, backend);
    Expect(response.status == 200 &&
               backend.last_request.client_id == request.client_id,
           "untrusted headers cannot change the transport identity");
  }
}

void TestStreamingOverloadIsRejectedBeforeHeaders() {
  FakeBackend backend;
  backend.reject_on_start = gufo::server::TextGenerationErrorCode::kQueueFull;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "stream":true
      })"),
                                                       backend);

  Expect(response.status == 429,
         "streaming overload is rejected synchronously");
  Expect(!response.streaming_body,
         "overloaded stream does not commit successful SSE headers");
  Expect(response.body.find("queue_full") != std::string::npos,
         "overload response carries a stable retry code");
  bool retry_after = false;
  for (const auto& [name, value] : response.headers) {
    retry_after = retry_after || (name == "Retry-After" && value == "1");
  }
  Expect(retry_after, "retryable overload advertises Retry-After");
}

void TestImagePartsRetainOrderAndIdentity() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "stop":["END"],
    "messages":[{"role":"user","content":[
      {"type":"text","text":"left"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"auto"}},
      {"type":"text","text":"right"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BAUG"}}
    ]}]
  })"),
                                                       backend);
  Expect(response.status == 200, "image content parts reach the backend");
  Expect(backend.last_request.stop_sequences == std::vector<std::string>{"END"},
         "image requests preserve explicit stop sequences");
  const auto& message = backend.last_request.messages.front();
  Expect(message.content == "leftright" && message.images.size() == 2,
         "images do not become text placeholders before model preparation");
  Expect(message.images[0].offset == 4 && message.images[1].offset == 9,
         "image placement relative to text is preserved");
  Expect(*message.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}) &&
             *message.images[1].bytes == std::vector<std::uint8_t>({4, 5, 6}),
         "each image retains its own decoded transport bytes");
  for (
      const auto* body :
      {R"({"messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]})",
       R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"low"}}]}]})",
       R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///tmp/image.png"}}]}]})"}) {
    Expect(gufo::server::HandleOpenAiChat(Request(body), backend).status == 400,
           "unsupported image role, policy and URL are rejected");
  }
}

void TestTextOnlyBackendRefusesImagesBeforeReading() {
  FakeBackend backend;
  backend.image_support = false;
  // An unresolvable HTTPS host would fail differently if it were fetched, and
  // a data URL would fail differently if it were decoded.
  for (const auto* url : {"https://image-fetch-must-not-happen.invalid/x.png",
                          "data:image/png;base64,!!!not-base64!!!"}) {
    const auto body =
        std::string(R"({"model":"test-model","messages":[{"role":"user",)"
                    R"("content":[{"type":"image_url","image_url":{"url":")") +
        url + R"("}}]}]})";
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body), backend);
    Expect(response.status == 400 &&
               response.body.find("does not support image input") !=
                   std::string::npos,
           "a text-only model refuses images before reading them");
  }
}

void TestImageCountAndByteBudget() {
  const auto message = gufo::json::parse(R"({"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]})");
  for (const unsigned count : {17u, 257u}) {
    for (const bool split : {false, true}) {
      FakeBackend backend;
      auto body = gufo::json::Value::object();
      body["model"] = "test-model";
      body["messages"] = gufo::json::Value::array();
      if (split) {
        for (unsigned i = 0; i < count; ++i)
          body["messages"].push_back(message);
      } else {
        auto combined = message;
        for (unsigned i = 1; i < count; ++i)
          combined["content"].push_back(message.find("content")->items()[0]);
        body["messages"].push_back(std::move(combined));
      }
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      std::size_t observed = 0;
      for (const auto& parsed : backend.last_request.messages)
        observed += parsed.images.size();
      Expect(response.status == 200 && observed == count,
             "all image parts reach the model within the byte budget");
    }
  }
  const auto response_message = gufo::json::parse(R"({
    "role":"user","content":[{"type":"input_image",
    "image_url":"data:image/png;base64,AQID"}]})");
  gufo::core::ImageReadBudget budget;
  budget.remaining_bytes = 17 * 3;
  std::string error;
  for (unsigned i = 0; i < 17; ++i) {
    gufo::tokenization::ChatMessage parsed;
    Expect(gufo::server::ParseOpenAiResponseMessage(response_message, &parsed,
                                                    budget, &error) &&
               parsed.images.size() == 1,
           "Responses accepts more than sixteen images across messages");
  }
  gufo::tokenization::ChatMessage parsed;
  Expect(!gufo::server::ParseOpenAiResponseMessage(response_message, &parsed,
                                                   budget, &error) &&
             error.find("byte") != std::string::npos,
         "aggregate byte protection still covers every message");
}

void TestToolImagesRetainOrderAndIdentity() {
  using gufo::json::parse;
  using gufo::tokenization::ChatRole;
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "messages":[{"role":"tool","tool_call_id":"call_picture","content":[
      {"type":"text","text":"left "},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},
      {"type":"text","text":" right"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,BAUG"}}
    ]}]
  })"),
                                                       backend);
  Expect(response.status == 200, "Chat tool images reach the backend");
  const auto& chat = backend.last_request.messages.front();
  Expect(chat.role == ChatRole::kTool && chat.tool_call_id == "call_picture" &&
             chat.content == "left  right" && chat.images.size() == 2 &&
             chat.images[0].offset == 5 && chat.images[1].offset == 11,
         "Chat images stay attached to the tool result at their text offsets");

  for (const auto* kind : {"function_call_output", "custom_tool_call_output"}) {
    for (const bool text : {false, true}) {
      auto item = parse(R"({"call_id":"call_picture","output":[
        {"type":"input_image","image_url":"data:image/png;base64,AQID"}
      ]})");
      item["type"] = kind;
      if (text)
        item["output"] = parse(R"([
          {"type":"input_text","text":"left "},
          {"type":"input_image","image_url":"data:image/png;base64,AQID"},
          {"type":"input_text","text":" right"}
        ])");
      gufo::core::ImageReadBudget budget;
      budget.remaining_bytes = 3;
      gufo::tokenization::ChatMessage message;
      std::string error;
      Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                      &error),
             "Responses accepts image-bearing tool outputs: " + error);
      Expect(
          message.role == ChatRole::kTool &&
              message.tool_call_id == "call_picture" &&
              message.content == (text ? "left  right" : "") &&
              message.images.size() == 1 &&
              message.images[0].offset == (text ? 5 : 0) &&
              *message.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}),
          "Responses retains call association, image bytes and order");
      gufo::tokenization::ChatMessage overflow;
      Expect(!gufo::server::ParseOpenAiResponseMessage(item, &overflow, budget,
                                                       &error),
             "tool output images consume the shared request byte budget");
    }
    auto literal = parse(R"({"call_id":"literal",
      "output":"data:image/png;base64,AQID"})");
    literal["type"] = kind;
    gufo::core::ImageReadBudget budget;
    budget.remaining_bytes = 0;
    gufo::tokenization::ChatMessage message;
    std::string error;
    Expect(gufo::server::ParseOpenAiResponseMessage(literal, &message, budget,
                                                    &error) &&
               message.images.empty() &&
               message.content == "data:image/png;base64,AQID",
           "string tool output stays literal text even when it spells an image "
           "URL");
  }
  for (const auto* role : {"assistant", "system", "developer"}) {
    auto message = parse(R"({"content":[
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}
    ]})");
    message["role"] = role;
    auto body = gufo::json::Value::object();
    body["model"] = "test-model";
    body["messages"] = gufo::json::Value::array();
    body["messages"].push_back(std::move(message));
    Expect(
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
            400,
        "images remain rejected on non-observation roles");
  }
  for (const auto* image :
       {R"({"url":17})",
        R"({"url":"data:image/png;base64,AQID","detail":"high"})",
        R"({"url":"file:///tmp/image.png"})",
        R"({"url":"data:image/png;base64,A==="})"}) {
    auto part = parse(R"({"type":"image_url"})");
    part["image_url"] = parse(image);
    auto message = parse(R"({"role":"tool","content":[]})");
    message["content"].push_back(std::move(part));
    auto body = parse(R"({"messages":[]})");
    body["model"] = "test-model";
    body["messages"].push_back(std::move(message));
    Expect(
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
            400,
        "tool images preserve URL, encoding and resolution validation");
  }
}

void TestStopSequencesAndDefaultFields() {
  using gufo::json::Value;
  const auto base = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
  for (const auto& [field, value, message] :
       {std::tuple{"model", "5", "'model' must be a string"},
        std::tuple{"model", "{}", "'model' must be a string"},
        std::tuple{"model", "[]", "'model' must be a string"},
        std::tuple{"model", "true", "'model' must be a string"},
        std::tuple{"reasoning_effort", "\"bogus\"",
                   "reasoning_effort must be none, minimal"}}) {
    auto invalid = base;
    invalid[field] = gufo::json::parse(value);
    FakeBackend backend;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(invalid.dump()), backend);
    Expect(response.status == 400 &&
               response.body.find(message) != std::string::npos,
           "validation errors describe the accepted values");
  }
  for (const auto* stop : {"null", "[]", "\"END\"", "[\"END\",\"終\"]"}) {
    auto body = base;
    body["stop"] = gufo::json::parse(stop);
    FakeBackend backend;
    backend.pieces = {"hello"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "supported stop forms are accepted");
    const auto& expected = body["stop"];
    const std::size_t count = expected.is_null()     ? 0
                              : expected.is_string() ? 1
                                                     : expected.items().size();
    Expect(backend.last_request.stop_sequences.size() == count,
           "stop sequences reach the backend");
    if (count)
      Expect(backend.last_request.stop_sequences.front() == "END",
             "stop string bytes are preserved");
  }
  for (const bool stream : {false, true}) {
    for (const auto* stop :
         {"true", "123", "{}", "\"\"", "[\"\"]", "[\"END\",null]",
          "[\"1\",\"2\",\"3\",\"4\",\"5\"]"}) {
      auto body = base;
      body["stream"] = stream;
      body["stop"] = gufo::json::parse(stop);
      FakeBackend backend;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 400 && backend.chat_calls == 0 &&
                 !response.streaming_body,
             "invalid stops fail before generation or streaming headers");
    }
  }
  auto body = base;
  body["stop"] = std::string(4097, 's');
  FakeBackend backend;
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             400,
         "oversized stop sequences are bounded");
  backend.defaults.max_tokens = 37;
  backend.defaults.sampling.temperature = 0.7F;
  backend.defaults.sampling.top_p = 0.9F;
  backend.defaults.sampling.seed = 123;
  backend.defaults.sampling.frequency_penalty = 0.5F;
  backend.defaults.sampling.presence_penalty = 0.3F;
  body = base;
  for (const auto* field :
       {"tools", "tool_choice", "stream", "stream_options", "n", "max_tokens",
        "max_completion_tokens", "logprobs", "top_logprobs", "response_format",
        "modalities", "audio", "temperature", "top_p", "seed", "logit_bias",
        "frequency_penalty", "presence_penalty", "reasoning_effort"})
    body[field] = Value();
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "nullable API defaults do not enable unsupported features");
  Expect(backend.last_max_tokens == backend.defaults.max_tokens,
         "null token limits retain the configured default");
  Expect(backend.last_sampling.temperature ==
                 backend.defaults.sampling.temperature &&
             backend.last_sampling.top_p == backend.defaults.sampling.top_p &&
             backend.last_sampling.seed == backend.defaults.sampling.seed &&
             backend.last_sampling.frequency_penalty ==
                 backend.defaults.sampling.frequency_penalty &&
             backend.last_sampling.presence_penalty ==
                 backend.defaults.sampling.presence_penalty,
         "null sampling fields retain server defaults");
  for (const std::string field : {"max_tokens", "max_completion_tokens"}) {
    for (const char* value : {"0", "-2", "1.5", "4294967296", "\"2\""}) {
      auto invalid = base;
      invalid[field] = gufo::json::parse(value);
      const auto response =
          gufo::server::HandleOpenAiChat(Request(invalid.dump()), backend);
      Expect(
          response.status == 400 &&
              response.body.find("'" + field + "' must be a positive integer") !=
                  std::string::npos,
          "an invalid token limit names the field that carried it");
    }
    // -1 (llama.cpp: no limit) leaves the configured default in place, like null (v0.1.1 HTTP fix).
    auto unset = base;
    unset[field] = gufo::json::parse("-1");
    const auto accepted =
        gufo::server::HandleOpenAiChat(Request(unset.dump()), backend);
    Expect(accepted.status == 200 &&
               backend.last_max_tokens == backend.defaults.max_tokens,
           "-1 leaves the configured token limit in place");
  }
  {
    auto invalid = base;
    invalid["max_completion_tokens"] = Value();
    invalid["max_tokens"] = 0;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(invalid.dump()), backend);
    Expect(response.status == 400 &&
               response.body.find("'max_tokens' must be a positive integer") !=
                   std::string::npos,
           "a null completion limit names the fallback max_tokens field");
  }
  body["logprobs"] = false;
  body["top_logprobs"] = 0;
  body["logit_bias"] = Value::object();
  body["response_format"] = gufo::json::parse(R"({"type":"text"})");
  body["modalities"] = gufo::json::parse(R"(["text"])");
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "explicit text-only and disabled logprobs defaults work");
  for (const auto* request :
       {R"({"logprobs":true})", R"({"top_logprobs":3})",
        R"({"modalities":["text","audio"]})", R"({"audio":{}})",
        R"({"response_format":{"type":"text","unexpected":true}})"}) {
    auto invalid = base;
    const auto fields = gufo::json::parse(request);
    for (const auto& [key, value] : fields.members())
      invalid[key] = value;
    const auto before = backend.chat_calls.load();
    Expect(gufo::server::HandleOpenAiChat(Request(invalid.dump()), backend)
                       .status == 400 &&
               backend.chat_calls == before,
           "actual unsupported feature requests remain explicit errors");
  }
}

void TestStructuredResponseFormat() {
  FakeBackend backend;
  backend.reasoning_defaults_value.enabled = true;
  auto body = gufo::json::parse(
      R"({"model":"test-model", "messages":[{"role":"user","content":"Return a value"}]})");
  for (
      const char* format :
      {R"({"type":"json_object"})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}})"}) {
    body["response_format"] = gufo::json::parse(format);
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 && backend.last_request.response_format,
           "compiled format reaches backend");
    Expect(backend.last_request.reasoning.enabled == true,
           "structured output retains the model reasoning default");
    for (
        const char* incompatible :
        {R"({"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})"}) {
      auto request = body;
      const auto fields = gufo::json::parse(incompatible);
      for (const auto& [key, value] : fields.members())
        request[key] = value;
      const auto accepted =
          gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
      Expect(accepted.status == 200 && backend.last_request.tools.size() == 1,
             "structured requests keep their active tools");
    }
  }
  body["response_format"]["json_schema"]["description"] =
      "Use the requested labels.";
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
                 200 &&
             backend.last_request.response_format_description ==
                 "Use the requested labels.",
         "format description reaches the model request");
  const std::string literal =
      R"({"text":"<think>literal</think> <tool_call>literal</tool_call> ┌"})";
  backend.pieces = {literal.substr(0, literal.size() - 4),
                    literal.substr(literal.size() - 4, 1),
                    literal.substr(literal.size() - 3)};
  body["response_format"] = gufo::json::parse(R"({"type":"json_object"})");
  body["reasoning_effort"] = "none";
  const auto buffered =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(gufo::json::parse(buffered.body)
                 .find("choices")
                 ->items()[0]
                 .find("message")
                 ->member_str("content") == literal,
         "JSON strings do not activate output protocol markers");
  body["stream"] = true;
  const auto streamed =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  std::string content;
  streamed.streaming_body([&](std::string_view chunk) {
    if (chunk == "data: [DONE]\n\n")
      return true;
    const auto event = gufo::json::parse(chunk.substr(6));
    for (const auto& choice : event.find("choices")->items()) {
      const auto* delta = choice.find("delta");
      if (delta) {
        Expect(!delta->contains("reasoning_content") &&
                   !delta->contains("tool_calls"),
               "JSON literals remain ordinary content in SSE");
        content += delta->member_str("content");
      }
    }
    return true;
  });
  Expect(content == literal,
         "streaming retains JSON bytes and UTF-8 boundaries");
  for (const bool stream : {false, true}) {
    body["stream"] = stream;
    body["reasoning_effort"] = "high";
    backend.pieces = {"Consider <tool_call> as text.", "</thi", "nk>", literal};
    const auto reply =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(reply.status == 200, "thinking and structured output coexist");
    std::string answer, reasoning;
    if (stream) {
      reply.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items()) {
          if (const auto* delta = choice.find("delta")) {
            answer += delta->member_str("content");
            reasoning += delta->member_str("reasoning_content");
          }
        }
        return true;
      });
    } else {
      const auto response = gufo::json::parse(reply.body);
      const auto* message =
          response.find("choices")->items()[0].find("message");
      answer = message->member_str("content");
      reasoning = message->member_str("reasoning_content");
    }
    Expect(answer == literal && reasoning == "Consider <tool_call> as text.",
           "only the reasoning delimiter changes structured-output phase");
  }
  // #580: as in llama.cpp, response schemas that omit additionalProperties
  // are admitted and keep declared objects closed; additionalProperties true
  // or a schema admits extra keys. Strict tool parameters keep requiring
  // explicitly closed objects.
  {
    auto open = gufo::json::parse(
        R"({"model":"test-model","messages":[{"role":"user","content":"Who?"}],
            "response_format":{"type":"json_schema","json_schema":{
              "name":"person","strict":true,"schema":{"type":"object",
              "properties":{"name":{"type":"string"},
                "born":{"type":"integer","description":"birth year"}},
              "required":["name","born"]}}}})");
    for (const bool stream : {false, true}) {
      open["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(open.dump()), backend);
      Expect(response.status == 200 && backend.last_request.response_format,
             "response schemas without additionalProperties are admitted");
      Expect(
          Accepts(*backend.last_request.response_format,
                  R"({"name":"Nostradamus","born":1503})") &&
              !Accepts(*backend.last_request.response_format,
                       R"({"name":"Nostradamus","born":1503,"note":"seer"})") &&
              !Accepts(*backend.last_request.response_format,
                       R"({"name":"Nostradamus"})"),
          "omitted additionalProperties keeps declared objects closed");
    }
    open["stream"] = false;
    open["tools"] = gufo::json::parse(
        R"([{"type":"function","function":{"name":"f","strict":true,
            "parameters":{"type":"object","properties":{"x":{"type":"string"}},
            "required":["x"]}}}])");
    const auto calls = backend.chat_calls.load();
    const auto rejected =
        gufo::server::HandleOpenAiChat(Request(open.dump()), backend);
    Expect(rejected.status == 400 && backend.chat_calls == calls &&
               rejected.body.find("invalid_tools") != std::string::npos,
           "strict tools still require closed parameter objects");
    gufo::server::ChatRequest responses;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"format":{"type":"json_schema",
                 "name":"person","strict":true,"schema":{"type":"object",
                 "properties":{"name":{"type":"string"}},"required":["name"],
                 "additionalProperties":{"type":"integer"}}}}})"),
               &responses) &&
               responses.response_format &&
               Accepts(*responses.response_format, R"({"name":"a","b":1})") &&
               !Accepts(*responses.response_format, R"({"name":"a","b":"1"})"),
           "Responses text.format admits typed additional properties");
  }
  body["stop"] = "END";
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "structured requests accept explicit stops");
  body = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"value"}]})");
  for (
      const char* invalid :
      {R"({"type":"unknown"})",
       R"({"type":"json_schema","json_schema":{"name":"bad name","schema":{}}})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","strict":1,"schema":{}}})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","schema":{"type":"object","properties":{},"additionalProperties":false,"not":{}}}})"}) {
    body["response_format"] = gufo::json::parse(invalid);
    body["stream"] = true;
    const auto calls = backend.chat_calls.load();
    const auto rejected =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(
        rejected.status == 400 && !rejected.streaming_body &&
            backend.chat_calls == calls &&
            rejected.body.find("invalid_response_format") != std::string::npos,
        "invalid schema fails before stream headers or admission");
  }
}

void TestExplicitStopOutputFraming() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool stream : {false, true}) {
    for (const bool thinking : {false, true}) {
      FakeBackend backend;
      // The scheduler already removed the stop marker. Test protocol framing
      // independently, including a required tool interrupted before its call.
      backend.pieces = {"safe"};
      backend.finish_reason = Finish::kStopSequence;
      backend.stop_sequence = "END";
      auto body = gufo::json::parse(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
              "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
              "tool_choice":"required","stop":"END"})");
      body["stream"] = stream;
      body["chat_template_kwargs"] = gufo::json::Value::object();
      body["chat_template_kwargs"]["enable_thinking"] = thinking;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "explicit stop completes successfully");
      std::string output = response.body;
      if (response.streaming_body)
        response.streaming_body([&](std::string_view piece) {
          output += piece;
          return true;
        });
      Expect(output.find("\"finish_reason\":\"stop\"") != std::string::npos &&
                 output.find("tool_choice_unsatisfied") == std::string::npos &&
                 output.find("END") == std::string::npos,
             "explicit stop wins over required tools and hides its marker");
      Expect(
          output.find(thinking ? "\"reasoning_content\":\"safe\""
                               : "\"content\":\"safe\"") != std::string::npos,
          "stopped text retains reasoning/content framing");
    }
  }
}

void TestStructuredToolTruncation() {
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{}}</tool_call>)";
  for (const bool stream : {false, true}) {
    for (std::size_t length = 0; length <= call.size(); ++length) {
      FakeBackend backend;
      backend.pieces = {call.substr(0, length)};
      backend.finish_reason =
          gufo::server::TextGenerationBackend::FinishReason::kLength;
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "response_format":{"type":"json_object"},"reasoning_effort":"none",
        "tool_choice":"required",
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]
      })");
      body["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200,
             "truncated constrained tool is a length result");
      if (stream) {
        std::string output;
        response.streaming_body([&](std::string_view chunk) {
          output += chunk;
          return true;
        });
        Expect(
            output.find("\"finish_reason\":\"length\"") != std::string::npos &&
                output.find("\"error\"") == std::string::npos &&
                output.find("<tool") == std::string::npos,
            "partial tool delimiters are not exposed as structured content");
      } else {
        const auto output = gufo::json::parse(response.body);
        const auto& choice = output.find("choices")->items()[0];
        Expect(choice.member_str("finish_reason") == "length" &&
                   choice.find("message")->member_str("content").empty() &&
                   choice.find("message")->contains("tool_calls") ==
                       (length == call.size()),
               "buffered truncated tools retain the length contract");
      }
    }
  }
}

void TestStrictToolSchema() {
  auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"call f"}],
    "reasoning_effort":"none","parallel_tool_calls":false,
    "tools":[{"type":"function","function":{"name":"f","strict":true,
      "parameters":{"type":"object","properties":{"text":{"type":"string"}},
        "required":["text"],"additionalProperties":false}}}]
  })");
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{"text":"<think>x</think></tool_call>"}}</tool_call>)";
  for (const bool stream : {false, true}) {
    FakeBackend backend;
    backend.pieces = {"Calling f. "};
    for (char byte : call)
      backend.pieces.emplace_back(1, byte);
    backend.pieces.emplace_back(" After f.");
    body["stream"] = stream;
    body["tool_choice"] =
        gufo::json::parse(R"({"type":"function","function":{"name":"f"}})");
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string output = response.body;
    std::string content;
    const auto collect = [&](const gufo::json::Value& event) {
      for (const auto& choice : event.find("choices")->items()) {
        const auto* message = choice.find(stream ? "delta" : "message");
        if (message)
          content += message->member_str("content");
      }
    };
    if (response.streaming_body)
      response.streaming_body([&](std::string_view chunk) {
        output += chunk;
        if (chunk != "data: [DONE]\n\n")
          collect(gufo::json::parse(chunk.substr(6)));
        return true;
      });
    else
      collect(gufo::json::parse(response.body));
    Expect(response.status == 200 && backend.last_request.constrained_tools &&
               !backend.last_request.response_format &&
               !backend.last_request.parallel_tool_calls &&
               backend.last_request.tool_choice ==
                   gufo::server::ChatRequest::ToolChoice::kRequired,
           "strict tool schemas and forced choice reach the runner "
           "independently of response_format");
    Expect(
        output.find("\"finish_reason\":\"tool_calls\"") != std::string::npos &&
            content == "Calling f.  After f." &&
            output.find("<think>x</think></tool_call>") != std::string::npos &&
            output.find("reasoning_content") == std::string::npos,
        "strict tool payload markers remain argument data in buffered and "
        "streaming responses");
  }
  body["tool_choice"] = "auto";
  body["parallel_tool_calls"] = true;
  for (const bool stream : {false, true}) {
    FakeBackend backend;
    const auto raw = " Before " + call + " between " + call + " after <";
    for (char byte : raw)
      backend.pieces.emplace_back(1, byte);
    body["stream"] = stream;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string content;
    if (stream) {
      response.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        return true;
      });
    } else {
      content = gufo::json::parse(response.body)
                    .find("choices")
                    ->items()[0]
                    .find("message")
                    ->member_str("content");
    }
    Expect(content == " Before  between  after <",
           "parallel strict calls preserve surrounding prose and whitespace");
  }
  body["parallel_tool_calls"] = false;
  for (
      const char* invalid :
      {R"({"tool_choice":{"type":"function","function":{"name":"absent"}}})",
       R"({"parallel_tool_calls":"false"})",
       R"({"tools":[{"type":"function","function":{"name":"bad<name","parameters":{}}}]})",
       R"({"tools":[{"type":"function","function":{"name":"f"}},{"type":"function","function":{"name":"f"}}]})",
       R"({"tools":[{"type":"function","function":{"name":"bad","strict":true,"parameters":{"type":"object"}}}]})"}) {
    FakeBackend backend;
    auto request = body;
    const auto fields = gufo::json::parse(invalid);
    for (const auto& [key, value] : fields.members())
      request[key] = value;
    Expect(gufo::server::HandleOpenAiChat(Request(request.dump()), backend)
                       .status == 400 &&
               backend.chat_calls == 0,
           "invalid tool constraints fail before generation");
  }
  body["tool_choice"] = "auto";
  for (const bool stream : {false, true}) {
    FakeBackend prose;
    prose.pieces = {"A literal ", "<"};
    body["stream"] = stream;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), prose);
    std::string content;
    if (response.streaming_body)
      response.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        return true;
      });
    else
      content = gufo::json::parse(response.body)
                    .find("choices")
                    ->items()[0]
                    .find("message")
                    ->member_str("content");
    Expect(
        response.status == 200 && content == "A literal <",
        "automatic strict tools preserve ordinary text ending with a partial "
        "marker");
  }
  for (std::size_t length = 1; length < call.size(); ++length) {
    FakeBackend backend;
    backend.pieces = {"Calling f. ", call.substr(0, length)};
    backend.finish_reason =
        gufo::server::TextGenerationBackend::FinishReason::kLength;
    body["stream"] = false;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 &&
               response.body.find("\"finish_reason\":\"length\"") !=
                   std::string::npos &&
               response.body.find("\"tool_calls\"") == std::string::npos,
           "incomplete strict calls never become API tool calls");
  }
  FakeBackend no_arguments;
  no_arguments.pieces = {
      R"(<tool_call>{"name":"f","arguments":{}}</tool_call>)"};
  body["tools"] = gufo::json::parse(
      R"([{"type":"function","function":{"name":"f","strict":true}}])");
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), no_arguments);
  Expect(response.status == 200 && no_arguments.last_request.constrained_tools,
         "omitted parameters define a strict function without arguments");
  body["tools"] = gufo::json::parse(
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object"},"strict":false}}])");
  body["parallel_tool_calls"] = false;
  FakeBackend non_strict;
  non_strict.pieces = no_arguments.pieces;
  Expect(
      gufo::server::HandleOpenAiChat(Request(body.dump()), non_strict).status ==
              200 &&
          non_strict.last_request.constrained_tools,
      "parallel_tool_calls false also constrains non-strict tools");
}

void TestStopInsideToolArguments() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool stream : {false, true}) {
    for (const bool thinking : {false, true}) {
      for (const auto* partial :
           {"<tool_call><function=f><parameter=text>partial-",
            "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"f\">"
            "<｜DSML｜parameter name=\"text\" string=\"true\">partial-"}) {
        for (const bool earlier_call : {false, true}) {
          FakeBackend backend;
          backend.finish_reason = Finish::kStopSequence;
          backend.stop_sequence = "STOP";
          const std::string raw =
              std::string(earlier_call
                              ? "<tool_call><function=f></function></tool_call>"
                              : "") +
              partial;
          backend.pieces = {"safe"};
          // Constrained reasoning must close before a tool can start.
          if (thinking)
            backend.pieces.emplace_back("</think>");
          // Exercise splits inside XML delimiters and multibyte DSML tags.
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          auto body = gufo::json::parse(
              R"({"model":"test-model","messages":[{"role":"user","content":"call f"}],
                  "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
                  "tool_choice":"required","stop":"STOP"})");
          body["stream"] = stream;
          body["chat_template_kwargs"] = gufo::json::Value::object();
          body["chat_template_kwargs"]["enable_thinking"] = thinking;
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200,
                 "a stop inside a tool argument succeeds");
          std::string output = response.body;
          if (response.streaming_body)
            response.streaming_body([&](std::string_view piece) {
              output += piece;
              return true;
            });
          Expect(output.find("partial-") == std::string::npos &&
                     output.find("<tool_call>") == std::string::npos &&
                     output.find("DSML") == std::string::npos,
                 "unfinished tool markup is never emitted as ordinary content");
          Expect((output.find("\"tool_calls\":") != std::string::npos) ==
                     earlier_call,
                 "only calls completed before the stop can be emitted");
          Expect(
              output.find("\"finish_reason\":\"stop\"") != std::string::npos &&
                  output.find(thinking ? "\"reasoning_content\":\"safe\""
                                       : "\"content\":\"safe\"") !=
                      std::string::npos,
              "preceding text and reasoning survive interrupted calls");
        }
      }
    }
  }
}

void TestNativeToolImplicitReasoningEnd() {
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const bool qwen = format == Format::kQwen;
    const std::string call =
        qwen ? "<tool_call>\n<function=read>\n<parameter=path>\nfile.txt\n"
               "</parameter>\n</function>\n</tool_call>"
             : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n"
               "<｜DSML｜parameter name=\"path\" string=\"true\">file.txt"
               "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    for (const bool responses : {false, true})
      for (const bool stream : {false, true})
        for (const bool quoted : {false, true})
          for (const bool interrupted : {false, true}) {
            const std::string thought = quoted ? "Example:\n```xml\n" + call +
                                                     "\n```\nRead the file.\n\n"
                                               : "Read the file.\n\n";
            const std::string text =
                thought + (quoted ? "</think>" : "") +
                (interrupted ? call.substr(0, call.find("file.txt") + 4)
                             : call);
            // llama.cpp treats the two newlines before a DeepSeek call as
            // TC_SEPARATOR, including when the model omitted </think>.
            const std::string expected_thought =
                qwen || quoted ? thought
                               : thought.substr(0, thought.size() - 2);
            auto body = gufo::json::parse(R"({
              "model":"test-model","messages":[{"role":"user","content":"read the file"}],
              "reasoning_effort":"low","tool_choice":"required","parallel_tool_calls":false,
              "tools":[{"type":"function","function":{"name":"read","parameters":{
                "type":"object","properties":{"path":{"type":"string"}},
                "required":["path"],"additionalProperties":false}}}]})");
            body["stream"] = stream;
            FakeBackend backend;
            backend.tool_format = format;
            backend.finish_reason =
                interrupted ? Finish::kLength : Finish::kStop;
            for (const char byte : text)
              backend.pieces.emplace_back(1, byte);
            gufo::server::HttpResponse response;
            if (responses) {
              auto flat = *body["tools"].items()[0].find("function");
              flat["type"] = "function";
              body["tools"] = gufo::json::Value::array();
              body["tools"].push_back(std::move(flat));
              gufo::server::ChatRequest chat;
              chat.reasoning.enabled = true;
              Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                     "implicit tool fixture has valid Responses controls");
              response = gufo::server::CreateOpenAiResponse(
                  Request(body.dump()), backend, chat, 512, {}, stream);
            } else {
              response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
            }
            Expect(response.status == 200,
                   "implicit native tool request succeeds");
            gufo::json::Value result;
            std::string streamed_reasoning, streamed_content;
            std::string streamed_arguments, finish;
            if (!stream)
              result = gufo::json::parse(response.body);
            else
              response.streaming_body([&](std::string_view piece) {
                if (piece == "data: [DONE]\n\n")
                  return true;
                const auto event =
                    gufo::json::parse(piece.substr(piece.find("data: ") + 6));
                if (responses) {
                  if (event.member_str("type") ==
                      "response.reasoning_summary_text.delta")
                    streamed_reasoning += event.member_str("delta");
                  if (event.member_str("type") == "response.output_text.delta")
                    streamed_content += event.member_str("delta");
                  if (event.member_str("type") == "response.completed" ||
                      event.member_str("type") == "response.incomplete")
                    result = *event.find("response");
                } else if (const auto* choices = event.find("choices")) {
                  for (const auto& choice : choices->items()) {
                    if (!choice.member_str("finish_reason").empty())
                      finish = choice.member_str("finish_reason");
                    if (const auto* delta = choice.find("delta")) {
                      streamed_reasoning +=
                          delta->member_str("reasoning_content");
                      streamed_content += delta->member_str("content");
                      if (const auto* tools = delta->find("tool_calls"))
                        for (const auto& tool : tools->items())
                          streamed_arguments +=
                              tool.find("function")->member_str("arguments");
                    }
                  }
                }
                return true;
              });
            std::string reasoning, content;
            std::size_t calls = 0;
            if (responses) {
              for (const auto& item : result.find("output")->items()) {
                if (item.member_str("type") == "function_call") {
                  calls += item.member_str("status") == "completed";
                  streamed_arguments = item.member_str("arguments");
                }
                for (const auto* field : {"summary", "content"})
                  if (const auto* parts = item.find(field))
                    for (const auto& part : parts->items())
                      (std::string_view(field) == "summary" ? reasoning
                                                            : content) +=
                          part.member_str("text");
              }
            } else if (!stream) {
              const auto& message =
                  *result.find("choices")->items()[0].find("message");
              reasoning = message.member_str("reasoning_content");
              content = message.member_str("content");
              if (const auto* tools = message.find("tool_calls"))
                calls = tools->size();
            } else {
              reasoning = streamed_reasoning;
              content = streamed_content;
              calls = finish == "tool_calls";
            }
            Expect(calls == (interrupted ? 0u : 1u),
                   "only a complete unquoted native call is invoked");
            if (stream && interrupted)
              Expect(streamed_arguments == R"({"path":"file)",
                     "an interrupted call retains only its unfinished prefix");
            Expect(content.empty() && reasoning == expected_thought,
                   "implicit call markup is not leaked into either output "
                   "channel");
            if (stream)
              Expect(streamed_content.empty() &&
                         streamed_reasoning == expected_thought,
                     "implicit reasoning boundary is correct during streaming");
          }
  }
}

void TestToolMarkersInsideConstrainedReasoning() {
  using gufo::json::Value;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const std::string reasoning =
      "Confirmed garbage on the docstring line. The broken line is:\n"
      "`    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{`\n"
      "A quoted <tool_call> is file data, not the end of reasoning.\n"
      "Also <｜DSML｜tool_calls> is quoted data. Preserve oldText exactly.";
  auto arguments = Value::object();
  arguments["path"] = "tests/unit/shared/test_vpc_block_public_access.py";
  arguments["edits"] = Value::array();
  auto replacement = Value::object();
  replacement["oldText"] =
      "    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{";
  replacement["newText"] =
      "    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"";
  arguments["edits"].push_back(std::move(replacement));
  const std::string call =
      "<tool_call>\n<function=edit>\n<parameter=path>\n" +
      arguments.member_str("path") + "\n</parameter>\n<parameter=edits>\n" +
      arguments["edits"].dump() + "\n</parameter>\n</function>\n</tool_call>";
  const std::string prose = "\nVerify the file, then run mypy and tests.";
  // The reasoning separator is framing; newlines inside the edit and the
  // prose following its call remain data in both APIs and transports.
  const std::string complete = "</think>\n\n" + call + prose;
  for (const bool responses : {false, true})
    for (const bool stream : {false, true})
      for (const bool bytewise : {false, true})
        for (const bool strict : {false, true})
          for (const std::string& tail :
               {complete, std::string{}, std::string{"</thi"}}) {
            const bool truncated = tail != complete;
            const std::string expected_reasoning =
                reasoning + (truncated ? tail : "");
            const std::string expected_content = truncated ? "" : prose;
            auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"fix the docstring"}],
            "chat_template_kwargs":{"enable_thinking":true},
            "parallel_tool_calls":false,"tool_choice":"auto",
            "tools":[{"type":"function","function":{"name":"edit",
              "parameters":{"type":"object","properties":{
                "path":{"type":"string"},"edits":{"type":"array","items":{
                  "type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
                  "required":["oldText","newText"],"additionalProperties":false}}},
                "required":["path","edits"],"additionalProperties":false}}}]
          })");
            body["stream"] = stream;
            auto tool = body["tools"].items()[0];
            tool["function"]["strict"] = strict;
            body["tools"] = Value::array();
            body["tools"].push_back(std::move(tool));
            FakeBackend backend;
            backend.finish_reason = truncated ? Finish::kLength : Finish::kStop;
            const std::string raw = reasoning + tail;
            if (bytewise)
              for (char byte : raw)
                backend.pieces.emplace_back(1, byte);
            else
              backend.pieces = {raw};
            gufo::server::HttpResponse response;
            if (responses) {
              auto flat = *body["tools"].items()[0].find("function");
              flat["type"] = "function";
              body["tools"] = Value::array();
              body["tools"].push_back(std::move(flat));
              gufo::server::ChatRequest chat;
              chat.reasoning.enabled = true;
              Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                     "reasoning marker fixture has valid Responses controls");
              response = gufo::server::CreateOpenAiResponse(
                  Request(body.dump()), backend, chat, 4096, {}, stream);
            } else {
              response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
            }
            Expect(response.status == 200, "reasoning marker fixture succeeds");
            std::vector<Value> events;
            if (stream)
              response.streaming_body([&](std::string_view chunk) {
                const auto pos = chunk.find("data: ");
                if (pos != std::string_view::npos &&
                    !chunk.substr(pos + 6).starts_with("[DONE]"))
                  events.push_back(gufo::json::parse(chunk.substr(pos + 6)));
                return true;
              });
            else
              events.push_back(gufo::json::parse(response.body));
            std::string content, thought, emitted_arguments;
            std::string streamed_content, streamed_thought;
            std::size_t calls = 0;
            bool terminal = false;
            for (const auto& event : events) {
              if (responses) {
                if (event.member_str("type") == "response.output_text.delta")
                  streamed_content += event.member_str("delta");
                if (event.member_str("type") ==
                    "response.reasoning_summary_text.delta")
                  streamed_thought += event.member_str("delta");
                // Terminal Responses items are authoritative in both modes.
                const auto* result = stream ? event.find("response") : &event;
                if (result && result->member_str("status") ==
                                  (truncated ? "incomplete" : "completed")) {
                  terminal = true;
                  for (const auto& item : result->find("output")->items()) {
                    if (item.member_str("type") == "function_call") {
                      ++calls;
                      emitted_arguments = item.member_str("arguments");
                    } else if (item.member_str("type") == "reasoning") {
                      for (const auto& part : item.find("summary")->items())
                        thought += part.member_str("text");
                    } else if (item.member_str("type") == "message") {
                      for (const auto& part : item.find("content")->items())
                        content += part.member_str("text");
                    }
                  }
                }
              } else if (const auto* choices = event.find("choices")) {
                for (const auto& choice : choices->items()) {
                  if (!choice.member_str("finish_reason").empty()) {
                    terminal = true;
                    Expect(choice.member_str("finish_reason") ==
                               (truncated ? "length" : "tool_calls"),
                           "reasoning limits retain the correct finish reason");
                  }
                  const auto* message =
                      choice.find(stream ? "delta" : "message");
                  if (!message)
                    continue;
                  thought += message->member_str("reasoning_content");
                  content += message->member_str("content");
                  if (const auto* tools = message->find("tool_calls"))
                    for (const auto& tool : tools->items()) {
                      calls += !stream || !tool.member_str("id").empty();
                      emitted_arguments +=
                          tool.find("function")->member_str("arguments");
                    }
                }
              }
            }
            Expect(terminal, "reasoning marker fixture has a terminal result");
            Expect(
                thought == expected_reasoning,
                "literal tool openers cannot terminate constrained reasoning");
            Expect(
                content == expected_content,
                "reasoning text and closing think tags never leak as content");
            Expect(truncated
                       ? calls == 0 && emitted_arguments.empty()
                       : calls == 1 && emitted_arguments == arguments.dump(),
                   "the edit preserves exact file data without protocol "
                   "pollution");
            if (responses && stream)
              Expect(
                  streamed_content == expected_content &&
                      streamed_thought == expected_reasoning,
                  "Responses deltas agree with the buffered phase boundaries");
          }
}

void TestToolMarkersWhenToolsDisabled() {
  using gufo::json::Value;
  const std::string thought =
      "Quoted <tool_call> and <｜DSML｜tool_calls> are reasoning data.";
  const std::string answer =
      "Literal <tool_call><function=f></function></tool_call> end.";
  for (const bool responses : {false, true})
    for (const bool stream : {false, true})
      for (const bool declared : {false, true})
        for (const int phase : {0, 1, 2}) {
          auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"Explain syntax"}],
            "tool_choice":"none"})");
          body["stream"] = stream;
          body["chat_template_kwargs"]["enable_thinking"] = phase != 0;
          if (declared)
            body["tools"] =
                gufo::json::parse(R"([{"type":"function","function":{
              "name":"f","parameters":{"type":"object","properties":{},"additionalProperties":false}}}])");
          const bool incomplete = phase == 2;
          const auto expected_thought = phase ? thought : std::string{};
          const auto expected_answer = incomplete ? std::string{} : answer;
          const auto raw = expected_thought + (phase == 1 ? "</think>" : "") +
                           expected_answer;
          FakeBackend backend;
          backend.finish_reason =
              incomplete
                  ? gufo::server::TextGenerationBackend::FinishReason::kLength
                  : gufo::server::TextGenerationBackend::FinishReason::kStop;
          for (char byte : raw)
            backend.pieces.emplace_back(1, byte);
          gufo::server::HttpResponse response;
          if (responses) {
            if (declared) {
              auto tool = *body["tools"].items()[0].find("function");
              tool["type"] = "function";
              body["tools"] = Value::array();
              body["tools"].push_back(std::move(tool));
            }
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = phase != 0;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "disabled tools have valid Responses controls");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 4096, {}, stream);
          } else
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          std::string text, reasoning;
          const auto inspect = [&](const Value& event) {
            if (responses) {
              if (stream) {
                if (event.member_str("type") == "response.output_text.delta")
                  text += event.member_str("delta");
                if (event.member_str("type") ==
                    "response.reasoning_summary_text.delta")
                  reasoning += event.member_str("delta");
              } else
                for (const auto& item : event.find("output")->items()) {
                  Expect(item.member_str("type") != "function_call",
                         "disabled tools never become API calls");
                  for (const auto* field : {"content", "summary"})
                    if (const auto* parts = item.find(field))
                      for (const auto& part : parts->items())
                        (std::string_view(field) == "content" ? text
                                                              : reasoning) +=
                            part.member_str("text");
                }
            } else
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                if (!message)
                  continue;
                Expect(!message->contains("tool_calls"),
                       "disabled tools never become API calls");
                text += message->member_str("content");
                reasoning += message->member_str("reasoning_content");
              }
          };
          Expect(response.status == 200,
                 "disabled tool marker request succeeds");
          if (stream)
            response.streaming_body([&](std::string_view chunk) {
              const auto pos = chunk.find("data: ");
              if (pos != std::string_view::npos &&
                  !chunk.substr(pos + 6).starts_with("[DONE]"))
                inspect(gufo::json::parse(chunk.substr(pos + 6)));
              return true;
            });
          else
            inspect(gufo::json::parse(response.body));
          Expect(text == expected_answer && reasoning == expected_thought,
                 "disabled tool markers preserve content and reasoning phases");
        }
}

void TestNativeToolTransports() {
  using gufo::json::Value;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const std::string literal =
      " <think>literal</think><tool_call></tool_call>"
      "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":\"nested\"}}</"
      "tool_call>"
      "<｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"></｜DSML｜invoke>"
      "</｜DSML｜tool_calls>"
      "</function></｜DSML｜invoke> \"42\" \\\nπ🦉\n \r";
  auto args = Value::object();
  args["text"] = literal;
  const std::vector<std::string> calls{
      "<tool_call>\n<function=f>\n<parameter=text>\n" + literal +
          "\n</parameter>\n</function>\n</tool_call>",
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
      "<｜DSML｜parameter name=\"text\" string=\"true\">" +
          literal +
          "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
          "</｜DSML｜tool_calls>",
      "<tool_call>{\"name\":\"f\",\"arguments\":" + args.dump() +
          "}</tool_call>"};
  for (const auto& call : calls)
    for (bool thinking : {false, true})
      for (bool stream : {false, true})
        for (bool parallel : {false, true})
          for (bool responses : {false, true})
            for (bool constrained : {false, true}) {
              auto body = gufo::json::parse(R"({
              "model":"test-model","messages":[{"role":"user","content":"call f"}],
              "tools":[{"type":"function","function":{"name":"f","strict":false,
                "parameters":{"type":"object","properties":{"text":{"type":"string"}},
                "required":["text"],"additionalProperties":false}}}],
              "tool_choice":"required"})");
              body["stream"] = stream;
              body["parallel_tool_calls"] = parallel;
              body["tool_choice"] = constrained ? "required" : "auto";
              body["chat_template_kwargs"]["enable_thinking"] = thinking;
              FakeBackend backend;
              if (constrained)
                backend.tool_format =
                    call.starts_with("<｜DSML｜tool_calls>")
                        ? gufo::sampling::JsonConstraint::ToolFormat::kDeepSeek
                        : gufo::sampling::JsonConstraint::ToolFormat::kQwen;
              const auto raw =
                  (thinking ? "Reasoning.</think>" : "") + call + " \n";
              for (char byte : raw)
                backend.pieces.emplace_back(1, byte);
              gufo::server::HttpResponse response;
              if (responses) {
                // Exercise the actual flat Responses function shape.
                auto flat = *body["tools"].items()[0].find("function");
                flat["type"] = "function";
                body["tools"] = Value::array();
                body["tools"].push_back(std::move(flat));
                gufo::server::ChatRequest chat;
                chat.reasoning.enabled = thinking;
                Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                       "Responses shares tool controls");
                response = gufo::server::CreateOpenAiResponse(
                    Request(body.dump()), backend, chat, 256, {}, stream);
              } else {
                response = gufo::server::HandleOpenAiChat(Request(body.dump()),
                                                          backend);
              }
              Expect(response.status == 200, "native tool request succeeds");
              std::vector<Value> output;
              if (stream) {
                response.streaming_body([&](std::string_view chunk) {
                  auto pos = chunk.find("data: ");
                  if (pos != std::string_view::npos &&
                      !chunk.substr(pos + 6).starts_with("[DONE]"))
                    output.push_back(gufo::json::parse(chunk.substr(pos + 6)));
                  return true;
                });
              } else {
                output.push_back(gufo::json::parse(response.body));
              }
              std::string arguments, reasoning;
              std::size_t call_count = 0;
              for (const auto& event : output) {
                if (responses) {
                  if (event.member_str("type") ==
                      "response.function_call_arguments.done")
                    Expect(event.member_str("name") == "f" &&
                               event.member_str("arguments") == args.dump(),
                           "Responses argument completion identifies the "
                           "function");
                  const auto* result = stream ? event.find("response") : &event;
                  if (result && result->member_str("status") == "completed")
                    for (const auto& item : result->find("output")->items())
                      if (item.member_str("type") == "function_call") {
                        ++call_count;
                        arguments = item.member_str("arguments");
                      }
                } else if (const auto* choices = event.find("choices")) {
                  for (const auto& choice : choices->items()) {
                    const auto* message =
                        choice.find(stream ? "delta" : "message");
                    if (!message)
                      continue;
                    reasoning += message->member_str("reasoning_content");
                    if (const auto* tools = message->find("tool_calls"))
                      for (const auto& tool : tools->items()) {
                        call_count += !stream || !tool.member_str("id").empty();
                        arguments +=
                            tool.find("function")->member_str("arguments");
                      }
                  }
                }
              }
              Expect(call_count == 1 && arguments == args.dump(),
                     "native and JSON arguments survive both API transports");
              Expect(
                  backend.last_request.constrained_tools,
                  "automatic non-strict tools also bind decoding constraints");
              Expect(reasoning.find("literal") == std::string::npos,
                     "argument tags never enter reasoning");
            }
  for (const auto& call : calls)
    for (const auto finish : {Finish::kLength, Finish::kStopSequence})
      for (std::size_t length = 0; length < call.size(); ++length) {
        FakeBackend backend;
        backend.pieces = {call.substr(0, length)};
        backend.finish_reason = finish;
        auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "tool_choice":"required","tools":[{"type":"function","function":{
          "name":"f","parameters":{}}}]})");
        const auto result =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(result.status == 200, "tool truncation terminates successfully");
        const auto output = gufo::json::parse(result.body);
        const auto* emitted =
            output.find("choices")->items()[0].find("message")->find(
                "tool_calls");
        // A DSML invocation may finish before its outer list closes. Never
        // recover a nested literal call from a truncated argument.
        if (emitted)
          Expect(emitted->size() == 1 &&
                     emitted->items()[0]
                             .find("function")
                             ->member_str("arguments") == args.dump(),
                 "every native truncation preserves complete arguments only");
      }
}

void TestJsonToolStringOwnership() {
  using gufo::json::Value;
  using Constraint = gufo::sampling::JsonConstraint;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const auto schema = gufo::json::parse(R"({
    "type":"object","properties":{"content":{"type":"string","pattern":".*"}},
    "required":["content"],"additionalProperties":false
  })");
  // Native runners keep raw text for a pattern, as llama.cpp; the JSON
  // envelope below is the syntax of runners without a native format.
  Expect(Constraint::ToolParameters(schema, false,
                                    Constraint::ToolFormat::kQwen) != nullptr,
         "string patterns keep native tool framing");
  const auto grammar = Constraint::WithTools(
      nullptr, {{"write", Constraint::Compile(schema, false)}}, false, true);
  auto request = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"write"}],
    "tools":[]
  })");
  auto tool = Value::object();
  tool["type"] = "function";
  tool["function"]["name"] = "write";
  tool["function"]["parameters"] = schema;
  request["tools"].push_back(std::move(tool));
  // Complete @aarononeal's write example from #383 as JSON string data.
  std::string literal =
      "<tool_call>\n<function=write>\n<parameter=content>\nAAAA\n"
      "</parameter>\n</function>\n</tool_call>";
  std::erase(literal, '\n');
  for (const auto* prefix : {"", "quoted \"text\" and \\ "}) {
    Value arguments = Value::object();
    arguments["content"] = std::string(prefix) + literal;
    const std::string complete =
        "<tool_call>{\"name\":\"write\",\"arguments\":" + arguments.dump() +
        "}</tool_call>";
    const auto truncated =
        complete.substr(0, complete.find(literal) + literal.size());
    for (const bool partial : {false, true}) {
      const auto& raw = partial ? truncated : complete;
      auto state = grammar->Start();
      for (const unsigned char byte : raw)
        state = grammar->Advance(state, byte);
      Expect(!state.empty() && grammar->Complete(state) == !partial,
             "literal XML is an admitted JSON call or an unfinished prefix");
      for (const auto finish : {Finish::kLength, Finish::kStopSequence}) {
        for (const bool stream : {false, true}) {
          auto body = request;
          body["stream"] = stream;
          FakeBackend backend;
          backend.finish_reason = finish;
          if (finish == Finish::kStopSequence)
            backend.stop_sequence = "STOP";
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200,
                 "JSON tool ownership request succeeds");
          std::vector<Value> calls;
          std::string content;
          const auto inspect = [&](const Value& event) {
            if (const auto* choices = event.find("choices"))
              for (const auto& choice : choices->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                if (!message)
                  continue;
                content += message->member_str("content");
                if (const auto* found = message->find("tool_calls"))
                  AppendToolDeltas(calls, *found, stream);
              }
          };
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              const auto pos = chunk.find("data: ");
              if (pos != std::string_view::npos &&
                  !chunk.substr(pos + 6).starts_with("[DONE]"))
                inspect(gufo::json::parse(chunk.substr(pos + 6)));
              return true;
            });
          } else {
            inspect(gufo::json::parse(response.body));
          }
          if (partial && !calls.empty())
            std::cerr << "Truncated JSON produced: " << calls.front().dump()
                      << '\n';
          Expect(calls.size() == (partial ? 0 : 1),
                 "an unfinished JSON string cannot invoke its literal XML");
          Expect(content.empty(), "JSON tool framing stays out of content");
          if (!partial)
            Expect(calls.front().find("function")->member_str("arguments") ==
                       arguments.dump(),
                   "a complete JSON call preserves its literal XML argument");
        }
      }
    }
    for (const int rejected_shape : {0, 1, 2}) {
      auto rejected = Value::object();
      rejected["name"] = rejected_shape == 0 ? "undeclared" : "write";
      rejected["arguments"] = arguments;
      if (rejected_shape == 1)
        rejected["name"] = "";
      if (rejected_shape == 2) {
        rejected["arguments"] = Value::array();
        rejected["arguments"].push_back(std::string(prefix) + literal);
      }
      FakeBackend backend;
      backend.pieces = {"<tool_call>" + rejected.dump() + "</tool_call>" +
                        complete};
      const auto response =
          gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
      Expect(response.status == 200,
             "rejected JSON calls permit later recovery");
      const auto output = gufo::json::parse(response.body);
      const auto* calls =
          output.find("choices")->items()[0].find("message")->find(
              "tool_calls");
      Expect(
          calls && calls->size() == 1 &&
              calls->items()[0].find("function")->member_str("arguments") ==
                  arguments.dump(),
          "rejected JSON envelopes cannot invoke quoted XML during recovery");
    }
  }
}

void TestNativeToolDialectSelection() {
  using gufo::json::Value;
  using gufo::sampling::JsonConstraint;
  using Format = JsonConstraint::ToolFormat;
  const auto native_call = [](Format format, std::string_view path) {
    if (format == Format::kDeepSeek)
      return "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n"
             "<｜DSML｜parameter name=\"path\" string=\"true\">" +
             std::string(path) +
             "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
             "</｜DSML｜tool_calls>";
    return "<tool_call>\n<function=read>\n<parameter=path>\n" +
           std::string(path) + "\n</parameter>\n</function>\n</tool_call>";
  };
  // The JSON fallback leaves text before <tool_call> unconstrained. A native
  // DeepSeek call written there remains a call, as before this change.
  {
    auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string","pattern":"^[a-z.]+$"}},
    "required":["path"],"additionalProperties":false})");
    // A native runner keeps DSML for this schema; the JSON-envelope runner
    // below is the remaining fallback syntax.
    Expect(JsonConstraint::ToolParameters(schema, false, Format::kDeepSeek) !=
               nullptr,
           "patterned string keeps native tool framing");
    auto body = gufo::json::parse(R"({
    "model":"test-model","reasoning_effort":"none",
    "messages":[{"role":"user","content":"Read fixture.xml."}],
    "tool_choice":"auto","tools":[{"type":"function","function":{
      "name":"read","strict":false}}]})");
    auto tool = body["tools"].items()[0];
    tool["function"]["parameters"] = schema;
    body["tools"] = Value::array();
    body["tools"].push_back(std::move(tool));
    for (const bool stream : {false, true}) {
      body["stream"] = stream;
      FakeBackend backend;
      backend.tool_format = Format::kJson;
      backend.pieces = {native_call(Format::kDeepSeek, "fixture.xml")};
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "JSON fallback request succeeds");
      std::string content;
      std::vector<std::string> arguments;
      const auto collect = [&](const Value& event) {
        for (const auto& choice : event.find("choices")->items()) {
          const auto* message = choice.find(stream ? "delta" : "message");
          content += message->member_str("content");
          if (const auto* calls = message->find("tool_calls"))
            for (const auto& item : calls->items())
              arguments.push_back(
                  item.find("function")->member_str("arguments"));
        }
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]"))
            collect(gufo::json::parse(payload));
          return true;
        });
      } else {
        collect(gufo::json::parse(response.body));
      }
      Expect(arguments == std::vector<std::string>{R"({"path":"fixture.xml"})"},
             "a native call written under the JSON fallback is returned");
      Expect(content.find("DSML") == std::string::npos,
             "the recognized native call is not visible content");
    }
  }
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string"}},
    "required":["path"],"additionalProperties":false})");
    const auto parameters =
        JsonConstraint::ToolParameters(schema, false, format);
    Expect(parameters != nullptr, "file tool has a supported wire format");
    const auto grammar = JsonConstraint::WithTools(
        nullptr, {{"read", parameters}}, false, true, format);
    const std::string call = native_call(format, "fixture.xml");
    const auto foreign =
        format == Format::kDeepSeek ? Format::kQwen : Format::kDeepSeek;
    // As in llama.cpp, calls end the output in both formats.
    const std::string suffix;
    for (const std::string& prefix :
         {std::string{"The XML root is `<tool_calls></tool_calls>`.\n"},
          std::string{"The foreign wrapper is `"} +
              (foreign == Format::kQwen ? "<tool_call>"
                                        : "<｜DSML｜tool_calls>") +
              "`.\n",
          "Example syntax:\n" + native_call(foreign, "example.xml") +
              "\nNow inspect the fixture.\n"}) {
      const auto raw = prefix + call + suffix;
      auto state = grammar->Start();
      for (const unsigned char byte : raw)
        state = grammar->Advance(state, byte);
      Expect(grammar->Complete(state),
             "foreign visible literals and the actual call satisfy the "
             "production tool grammar");
      for (const bool responses : {false, true}) {
        for (const bool stream : {false, true}) {
          auto body = gufo::json::parse(R"({
          "model":"test-model","reasoning_effort":"none",
          "messages":[{"role":"user","content":"Inspect fixture.xml and its tool_calls root element."}],
          "tool_choice":"auto","tools":[{"type":"function","function":{
            "name":"read","strict":false}}]})");
          auto tool = body["tools"].items()[0];
          tool["function"]["parameters"] = schema;
          body["tools"] = Value::array();
          body["tools"].push_back(std::move(tool));
          body["stream"] = stream;
          FakeBackend backend;
          backend.tool_format = format;
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          gufo::server::HttpResponse response;
          if (responses) {
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = false;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "Responses accepts the file tool");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 256, {}, stream);
          } else {
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          }
          Expect(response.status == 200, "native file tool request succeeds");
          // A later backend configuration must not change an admitted stream's
          // format. The fake snapshots this metadata on its GenerationRequest.
          backend.tool_format.reset();
          std::string content;
          std::vector<std::string> arguments;
          const auto collect = [&](const Value& event) {
            if (responses) {
              const auto* result = stream ? event.find("response") : &event;
              if (!result || result->member_str("status") != "completed")
                return;
              for (const auto& item : result->find("output")->items()) {
                if (item.member_str("type") == "function_call")
                  arguments.push_back(item.member_str("arguments"));
                else if (item.member_str("type") == "message")
                  for (const auto& part : item.find("content")->items())
                    content += part.member_str("text");
              }
            } else {
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                content += message->member_str("content");
                if (const auto* calls = message->find("tool_calls"))
                  for (const auto& item : calls->items()) {
                    const auto index =
                        stream ? item.member_size("index") : arguments.size();
                    if (index == arguments.size())
                      arguments.emplace_back();
                    Expect(index < arguments.size(), "tool index is stable");
                    arguments[index] +=
                        item.find("function")->member_str("arguments");
                  }
              }
            }
          };
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              const auto payload = chunk.substr(chunk.find("data: ") + 6);
              if (!payload.starts_with("[DONE]"))
                collect(gufo::json::parse(payload));
              return true;
            });
          } else {
            collect(gufo::json::parse(response.body));
          }
          Expect(arguments ==
                     std::vector<std::string>{R"({"path":"fixture.xml"})"},
                 "only the actual selected-dialect call reaches the client");
          Expect(content == prefix + suffix,
                 "foreign tool syntax remains literal visible content");
        }
      }
    }
  }
  // DeepSeek's template writes "\n\n" before its call block. That separator
  // is framing, as in llama.cpp's parser; returned as content, the replayed
  // history would carry it twice and miss the generated continuation.
  for (const std::string prose : {"", "Reading the fixture."}) {
    for (const bool stream : {false, true}) {
      auto body = gufo::json::parse(R"({
      "model":"test-model","reasoning_effort":"none",
      "messages":[{"role":"user","content":"Read fixture.xml."}],
      "tool_choice":"auto","tools":[{"type":"function","function":{
        "name":"read","parameters":{"type":"object",
        "properties":{"path":{"type":"string"}},"required":["path"]}}}]})");
      body["stream"] = stream;
      FakeBackend backend;
      backend.tool_format = Format::kDeepSeek;
      for (const char byte :
           prose + "\n\n" + native_call(Format::kDeepSeek, "fixture.xml"))
        backend.pieces.emplace_back(1, byte);
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      std::string content;
      std::size_t calls = 0;
      const auto collect = [&](const Value& event) {
        for (const auto& choice : event.find("choices")->items()) {
          const auto* message = choice.find(stream ? "delta" : "message");
          content += message->member_str("content");
          if (const auto* found = message->find("tool_calls"))
            for (const auto& call : found->items())
              calls += !stream || !call.member_str("id").empty();
        }
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]"))
            collect(gufo::json::parse(payload));
          return true;
        });
      } else {
        collect(gufo::json::parse(response.body));
      }
      Expect(response.status == 200 && calls == 1 && content == prose,
             "the separator before a DeepSeek call block is not content");
    }
  }
}

void TestNativeToolTextOutsideEnvelopes() {
  using gufo::json::Value;
  using gufo::sampling::JsonConstraint;
  using Format = JsonConstraint::ToolFormat;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string"}},
    "required":["path"],"additionalProperties":false})");
  const auto invoke = [](std::string_view path) {
    return "\n<｜DSML｜invoke name=\"read\">\n"
           "<｜DSML｜parameter name=\"path\" string=\"true\">" +
           std::string(path) + "</｜DSML｜parameter>\n</｜DSML｜invoke>";
  };
  const auto envelope = [](const std::string& calls) {
    return "<｜DSML｜tool_calls>" + calls + "\n</｜DSML｜tool_calls>";
  };
  const std::string suffix = "\nNo edits are requested.";
  const auto first = envelope(invoke("fixture.txt"));
  const auto legacy =
      "<tool_calls>" + invoke("example.txt") + "\n</tool_calls>";
  const std::string literal = "Keep </｜DSML｜tool_calls> as argument data.";
  struct Fixture {
    Format format;
    std::string raw;
    std::string text;
    std::vector<std::string> paths;
    bool complete{true};
    bool canonical{true};
    Finish finish{Finish::kStop};
    bool required{false};
    std::optional<Format> admitted_format{};
  };
  const std::vector<Fixture> fixtures{
      {Format::kQwen,
       "<tool_call>\n<function=read>\n<parameter=path>\nfixture.txt\n"
       "</parameter>\n</function>\n</tool_call>" +
           suffix,
       suffix,
       {"fixture.txt"}},
      {Format::kJson,
       R"(<tool_call>{"name":"read","arguments":{"path":"fixture.txt"}}</tool_call>)" +
           suffix,
       suffix,
       {"fixture.txt"}},
      {Format::kDeepSeek, first + suffix, suffix, {"fixture.txt"}},
      // Native framing may be part of the admitted opener, not its tag name.
      {.format = Format::kDeepSeek,
       .raw = "\n\n" + first + suffix,
       .text = suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      {Format::kDeepSeek,
       "Before.\n" + envelope(invoke("fixture.txt") + invoke("second.txt")) +
           suffix,
       "Before.\n" + suffix,
       {"fixture.txt", "second.txt"}},
      {Format::kDeepSeek,
       first + "\nBetween.\n" + envelope(invoke("second.txt")) + suffix,
       "\nBetween.\n" + suffix,
       {"fixture.txt", "second.txt"}},
      {Format::kDeepSeek,
       envelope(invoke(literal)) + suffix,
       suffix,
       {literal}},
      // Interrupted output may keep a complete invocation even before its
      // outer list closes, but must not expose any unfinished markup.
      {Format::kDeepSeek,
       first + "\nBetween.\n<｜DSML｜tool_calls>" + invoke("second.txt"),
       "\nBetween.\n",
       {"fixture.txt", "second.txt"},
       false},
      {Format::kDeepSeek,
       first + "\nBetween.\n<｜DSML｜tool_calls>\n"
               "<｜DSML｜invoke name=\"read\">\n"
               "<｜DSML｜parameter name=\"path\" string=\"true\">partial",
       "\nBetween.\n",
       {"fixture.txt"},
       false},
      // Existing compatibility spelling is accepted by extraction, rather
      // than generated by the canonical native grammar.
      {Format::kDeepSeek,
       "<｜DSML｜tool_calls｜>" + invoke("fixture.txt") +
           "\n</｜DSML｜tool_calls｜>" + suffix,
       suffix,
       {"fixture.txt"},
       true,
       false},
      // A partial marker is legal ordinary text in auto mode. Required tools
      // instead hold it back when a length limit or explicit stop interrupts.
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n<｜DSML｜tool_calls",
       .paths = {"fixture.txt"},
       .finish = Finish::kLength},
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n",
       .paths = {"fixture.txt"},
       .finish = Finish::kLength,
       .required = true},
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n",
       .paths = {"fixture.txt"},
       .finish = Finish::kStopSequence,
       .required = true},
      // Only an outer envelope starts DSML calls. Bare invocation examples
      // outside it are ordinary text, even before another real envelope.
      {Format::kDeepSeek,
       first + invoke("example.txt") + suffix,
       invoke("example.txt") + suffix,
       {"fixture.txt"}},
      {Format::kDeepSeek,
       first + invoke("example.txt") + envelope(invoke("second.txt")) + suffix,
       invoke("example.txt") + suffix,
       {"fixture.txt", "second.txt"}},
      {.format = Format::kDeepSeek,
       .raw = first + suffix,
       .text = suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      // A known native grammar owns only its canonical envelope. Foreign
      // wrapper examples remain prose, including native-looking invocations.
      {.format = Format::kDeepSeek,
       .raw = first + "\n<tool_calls></tool_calls>" + suffix,
       .text = "\n<tool_calls></tool_calls>" + suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      {.format = Format::kDeepSeek,
       .raw = first + legacy + suffix,
       .text = legacy + suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      // Unknown metadata and JSON fallback deliberately retain legacy native
      // recognition. Both wrapped calls still belong to the tool output.
      {.format = Format::kDeepSeek,
       .raw = first + legacy + suffix,
       .text = suffix,
       .paths = {"fixture.txt", "example.txt"}},
      {.format = Format::kJson,
       .raw = first + legacy + suffix,
       .text = suffix,
       .paths = {"fixture.txt", "example.txt"},
       .admitted_format = Format::kJson},
  };
  for (const auto& fixture : fixtures) {
    auto parameters_schema = schema;
    if (fixture.format == Format::kJson) {
      parameters_schema["enum"] = gufo::json::parse(
          R"([{"path":"fixture.txt"},{"path":"example.txt"}])");
      // A native runner keeps DSML for finite objects too; this fixture
      // covers runners whose only call syntax is the JSON envelope.
      Expect(JsonConstraint::ToolParameters(parameters_schema, false,
                                            Format::kDeepSeek) != nullptr,
             "finite object parameters keep native tool framing");
    }
    if (fixture.canonical) {
      const auto parameters = JsonConstraint::ToolParameters(
          parameters_schema, false, fixture.format);
      Expect(parameters != nullptr, "native tool schema is supported");
      const auto grammar =
          JsonConstraint::WithTools(nullptr, {{"read", parameters}},
                                    fixture.required, true, fixture.format);
      // As in llama.cpp, native calls end the output (Qwen may add the
      // whitespace llama.cpp's `space` admits). Text after them is reachable
      // only without that grammar, e.g. the JSON envelope runners use.
      std::string_view admitted = fixture.raw;
      if (fixture.format == Format::kDeepSeek) {
        constexpr std::string_view kClose = "\n</｜DSML｜tool_calls>";
        admitted = admitted.substr(0, admitted.find(kClose) + kClose.size());
      } else if (fixture.format == Format::kQwen) {
        constexpr std::string_view kClose = "</tool_call>";
        auto end = admitted.find(kClose) + kClose.size();
        while (end < admitted.size() && admitted[end] == '\n')
          ++end;
        admitted = admitted.substr(0, end);
      }
      auto state = grammar->Start();
      for (const unsigned char byte : admitted) {
        state = grammar->Advance(state, byte);
        Expect(!state.empty(), "fixture is admitted by the production grammar");
      }
      Expect(grammar->Complete(state) ==
                 (fixture.complete || admitted.size() < fixture.raw.size()),
             "only complete fixture may terminate normally");
      if (admitted.size() < fixture.raw.size())
        Expect(grammar
                   ->Advance(state, static_cast<unsigned char>(
                                        fixture.raw[admitted.size()]))
                   .empty(),
               "native output ends after its calls");
    }
    for (const bool responses : {false, true})
      for (const bool stream : {false, true})
        for (const bool fragmented : {false, true}) {
          auto body = gufo::json::parse(R"({"model":"test-model",
            "reasoning_effort":"none","tool_choice":"auto",
            "messages":[{"role":"user","content":"Read the fixtures."}],
            "tools":[{"type":"function","function":{"name":"read",
              "strict":false}}]})");
          auto tool = body["tools"].items()[0];
          tool["function"]["parameters"] = parameters_schema;
          body["tools"] = Value::array();
          body["tools"].push_back(responses ? *tool.find("function") : tool);
          if (responses) {
            auto flat = body["tools"].items()[0];
            flat["type"] = "function";
            body["tools"] = Value::array();
            body["tools"].push_back(std::move(flat));
          }
          body["stream"] = stream;
          body["tool_choice"] = fixture.required ? "required" : "auto";
          FakeBackend backend;
          backend.tool_format = fixture.admitted_format;
          backend.finish_reason =
              fixture.complete ? fixture.finish : Finish::kLength;
          if (fragmented)
            for (const char byte : fixture.raw)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces = {fixture.raw};
          gufo::server::HttpResponse response;
          if (responses) {
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = false;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "Responses accepts tool controls");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 512, {}, stream);
          } else {
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          }
          Expect(response.status == 200, "native tool request succeeds");
          backend.tool_format.reset();
          std::string text, deltas;
          std::vector<std::string> paths;
          std::vector<Value> chat_calls;
          const auto call = [&](const Value& function) {
            Expect(function.member_str("name") == "read",
                   "only the declared intended call is returned");
            const auto arguments =
                gufo::json::parse(function.member_str("arguments"));
            Expect(arguments.size() == 1, "call has exactly its path argument");
            paths.push_back(arguments.member_str("path"));
          };
          const auto collect = [&](const Value& event) {
            if (responses) {
              if (event.member_str("type") == "response.output_text.delta")
                deltas += event.member_str("delta");
              const auto* result = stream ? event.find("response") : &event;
              if (!result ||
                  (stream && event.member_str("type") != "response.completed" &&
                   event.member_str("type") != "response.incomplete"))
                return;
              for (const auto& item : result->find("output")->items())
                if (item.member_str("type") == "function_call")
                  call(item);
                else if (item.member_str("type") == "message")
                  for (const auto& part : item.find("content")->items())
                    text += part.member_str("text");
            } else {
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                text += message->member_str("content");
                if (const auto* calls = message->find("tool_calls"))
                  AppendToolDeltas(chat_calls, *calls, stream);
              }
            }
          };
          if (stream)
            response.streaming_body([&](std::string_view chunk) {
              const auto data = chunk.find("data: ");
              Expect(data != std::string_view::npos, "stream uses SSE framing");
              const auto payload = chunk.substr(data + 6);
              if (!payload.starts_with("[DONE]"))
                collect(gufo::json::parse(payload));
              return true;
            });
          else
            collect(gufo::json::parse(response.body));
          for (const auto& tool : chat_calls)
            call(*tool.find("function"));
          Expect(backend.last_request.constrained_tools && backend.completed,
                 "tool request uses constrained generation");
          Expect(paths == fixture.paths,
                 "complete calls and literal arguments retain ownership");
          Expect(text == fixture.text,
                 "ordinary text outside complete tool envelopes survives");
          if (responses && stream)
            Expect(deltas == text,
                   "Responses text deltas agree with the final output");
        }
  }
}

void TestNativeReferencedArgumentTypes() {
  using gufo::json::Value;
  const auto arguments = gufo::json::parse(
      R"({"n":42,"b":true,"a":[1],"o":{"x":2},"s":"42","z":null})");
  const auto schema = gufo::json::parse(R"({
    "type":"object","properties":{
      "n":{"$ref":"#/$defs/number~1type~0"},
      "b":{"$ref":"#/$defs/boolean"},"a":{"$ref":"#/$defs/array"},
      "o":{"$ref":"#/$defs/object"},"s":{"$ref":"#/$defs/string"},
      "z":{"$ref":"#/$defs/null"}},
    "required":["n","b","a","o","s","z"],"additionalProperties":false,
    "$defs":{"number/type~":{"type":"integer"},"boolean":{"type":"boolean"},
      "array":{"type":"array","items":{"type":"integer"}},
      "object":{"type":"object","properties":{"x":{"type":"integer"}},
                "required":["x"],"additionalProperties":false},
      "string":{"type":"string"},"null":{"type":"null"}}})");
  for (const bool root_reference : {false, true})
    for (const bool responses : {false, true}) {
      auto parameters = schema;
      if (root_reference) {
        parameters = Value::object();
        parameters["$ref"] = "#/$defs/Arguments";
        parameters["$defs"] = *schema.find("$defs");
        auto object = Value::object();
        for (const auto& [key, value] : schema.members())
          if (key != "$defs")
            object[key] = value;
        parameters["$defs"]["Arguments"] = std::move(object);
      }
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "tools":[{"type":"function","function":{"name":"f","strict":true}}],
        "tool_choice":"required"})");
      auto tool = body["tools"].items()[0];
      tool["function"]["parameters"] = parameters;
      body["tools"] = Value::array();
      body["tools"].push_back(std::move(tool));
      FakeBackend backend;
      std::string call = "<tool_call>\n<function=f>\n";
      for (const auto& [name, value] : arguments.members())
        call += "<parameter=" + name + ">\n" +
                (value.is_string() ? value.str() : value.dump()) +
                "\n</parameter>\n";
      backend.pieces = {call + "</function>\n</tool_call>"};
      gufo::server::HttpResponse response;
      if (responses) {
        gufo::server::ChatRequest chat;
        Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
               "referenced Responses tool schema is accepted");
        response = gufo::server::CreateOpenAiResponse(
            Request(body.dump()), backend, chat, 256, {}, false);
      } else {
        response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      }
      Expect(response.status == 200, "referenced native tool succeeds");
      const auto output = gufo::json::parse(response.body);
      const auto encoded =
          responses
              ? output.find("output")->items().back().member_str("arguments")
              : output.find("choices")
                    ->items()[0]
                    .find("message")
                    ->find("tool_calls")
                    ->items()[0]
                    .find("function")
                    ->member_str("arguments");
      Expect(encoded == arguments.dump(),
             "native argument types follow property and root references");
    }
}

// Schemas that used to force a JSON envelope now stay native (#438). Their raw
// values are typed as llama.cpp's qwen3-coder parser types them: a schema that
// admits other JSON types tries them first, an exact string stays text.
void TestNativeArgumentTypingMatrix() {
  struct Case {
    const char* schema;
    const char* raw;
    const char* expected;
  };
  const std::vector<Case> cases{
      {R"({"type":"string","pattern":"^[0-9]+$"})", "42", R"("42")"},
      {R"({"type":"string","format":"uri"})", "https://a.b",
       R"("https://a.b")"},
      {R"({"type":"string","default":"x","examples":["y"]})", "true",
       R"("true")"},
      {R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "42", "42"},
      {R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "forty",
       R"("forty")"},
      {R"({"allOf":[{"type":"string"},{"minLength":1}]})", "42", R"("42")"},
      {R"({"allOf":[{"type":"integer"},{"minimum":1}]})", "42", "42"},
      {R"({"const":"42"})", "42", R"("42")"},
      {R"({"enum":["true","false"]})", "true", R"("true")"},
      {R"({"enum":["null","42"]})", "null", R"("null")"},
      {R"({"anyOf":[{"const":"42"},{"type":"null"}]})", "42", R"("42")"},
      {R"({"enum":["42",17]})", "17", "17"},
      {R"({"type":"boolean"})", "True", "true"},
      {R"({"minLength":1})", "42", R"("42")"},
      {R"({"pattern":".*"})", "true", R"("true")"},
      {R"({"not":{"type":"null"}})", "plain", R"("plain")"},
      {R"({"type":["string","null"]})", "null", "null"},
      {R"({"anyOf":[{"type":"integer"},{"type":"null"}]})", "null", "null"},
      {R"({"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991})",
       "240000", "240000"},
      {R"({"type":"object","properties":{"x":{"type":"integer"}}})",
       R"({"x": 1})", R"({"x":1})"},
      {R"({"type":"array","items":{"type":"object"}})", R"([{"k": "v"}])",
       R"([{"k":"v"}])"},
      // A recursive schema the grammar cannot enforce keeps its JSON value.
      {R"({"$ref":"#"})", "5", "5"},
      {R"({"type":"string"})", "line 1\n<b>\"q\"</b>",
       R"("line 1\n<b>\"q\"</b>")"},
      {R"({"type":"string"})", "line\n</parameter> is literal\nend",
       R"("line\n</parameter> is literal\nend")"},
      {R"({"type":"string"})",
       "line\n</parameter> <parameter=other>literal\nend",
       R"("line\n</parameter> <parameter=other>literal\nend")"},
      {R"({"type":"string"})", "line\n</parameter> </function>literal\nend",
       R"("line\n</parameter> </function>literal\nend")"},
  };
  for (const auto& item : cases) {
    for (
        const auto* root :
        {R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"})",
         R"({"type":"object","additionalProperties":true})"}) {
      for (const bool stream : {false, true}) {
        auto parameters = gufo::json::parse(root);
        parameters["properties"]["v"] = gufo::json::parse(item.schema);
        parameters["required"] = gufo::json::parse(R"(["v"])");
        auto body = gufo::json::parse(R"({"model":"test-model",
          "messages":[{"role":"user","content":"call record"}],
          "tools":[],"tool_choice":"required"})");
        auto tool = gufo::json::Value::object();
        tool["type"] = "function";
        tool["function"]["name"] = "record";
        tool["function"]["parameters"] = parameters;
        body["tools"].push_back(std::move(tool));
        body["stream"] = stream;
        FakeBackend backend;
        backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
        const auto raw = std::string(
                             "<tool_call>\n<function=record>\n"
                             "<parameter=v>\n") +
                         item.raw + "\n</parameter>\n</function>\n</tool_call>";
        if (stream)
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
        else
          backend.pieces = {raw};
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        if (response.status != 200)
          std::cerr << "native schema=" << item.schema << " raw=" << item.raw
                    << " status=" << response.status
                    << " body=" << response.body << '\n';
        Expect(response.status == 200, "native typed call succeeds");
        std::string arguments;
        std::string content;
        if (stream) {
          response.streaming_body([&](std::string_view part) {
            if (part == "data: [DONE]\n\n")
              return true;
            const auto event = gufo::json::parse(part.substr(6));
            for (const auto& choice : event.find("choices")->items())
              if (const auto* delta = choice.find("delta")) {
                content += delta->member_str("content");
                if (const auto* calls = delta->find("tool_calls"))
                  for (const auto& call : calls->items())
                    arguments += call.find("function")->member_str("arguments");
              }
            return true;
          });
        } else {
          const auto output = gufo::json::parse(response.body);
          const auto& message =
              *output.find("choices")->items()[0].find("message");
          content = message.member_str("content");
          if (const auto* calls = message.find("tool_calls"))
            arguments =
                calls->items()[0].find("function")->member_str("arguments");
        }
        const auto expected = std::string(R"({"v":)") + item.expected + "}";
        if (arguments.empty() || gufo::json::parse(arguments).dump() !=
                                     gufo::json::parse(expected).dump())
          std::cerr << "typing schema=" << item.schema << " raw=" << item.raw
                    << " stream=" << stream << " got=" << arguments << '\n';
        Expect(!arguments.empty() && gufo::json::parse(arguments).dump() ==
                                         gufo::json::parse(expected).dump(),
               "native arguments are typed as llama.cpp types them");
        Expect(content.empty(), "a native call leaves no visible framing");
      }
    }
  }
  // Parseable JSON and Python-literal recovery must both respect a known
  // argument type. A malformed call must not invoke a tool with another type.
  for (const auto& [type, raw] :
       std::vector<std::pair<std::string, std::string>>{{"integer", "true"},
                                                        {"integer", "True"},
                                                        {"boolean", "1"},
                                                        {"array", "{}"},
                                                        {"object", "null"}}) {
    auto body = gufo::json::parse(R"({"model":"test-model",
      "messages":[{"role":"user","content":"call record"}],
      "tools":[{"type":"function","function":{"name":"record",
        "parameters":{"type":"object","properties":{"v":{"type":"integer"}},
        "required":["v"],"additionalProperties":false}}}]})");
    auto tool = body["tools"].items()[0];
    tool["function"]["parameters"]["properties"]["v"]["type"] = type;
    body["tools"] = gufo::json::Value::array();
    body["tools"].push_back(std::move(tool));
    FakeBackend backend;
    backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
    backend.pieces = {"<tool_call>\n<function=record>\n<parameter=v>\n" + raw +
                      "\n</parameter>\n</function>\n</tool_call>"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "invalid native output is handled safely");
    const auto output = gufo::json::parse(response.body);
    const auto& message = *output.find("choices")->items()[0].find("message");
    Expect(!message.find("tool_calls"),
           "a value with the wrong declared type is not invoked");
  }
  // A declared name with surrounding spaces remains exact. llama.cpp's
  // JSON mapper trims it despite matching the exact key in its PEG grammar.
  for (const bool stream : {false, true}) {
    auto body = gufo::json::parse(R"({"model":"test-model",
      "messages":[{"role":"user","content":"call record"}],
      "tools":[{"type":"function","function":{"name":"record","strict":true,
        "parameters":{"type":"object","properties":{" value ":{"type":"string"}},
        "required":[" value "],"additionalProperties":false}}}],
      "tool_choice":"required"})");
    body["stream"] = stream;
    FakeBackend backend;
    backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
    backend.pieces = {
        "<tool_call>\n<function=record>\n<parameter= value >\nx\n</parameter>\n"
        "</function>\n</tool_call>"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "spaced parameter call succeeds");
    std::string arguments;
    if (stream) {
      response.streaming_body([&](std::string_view part) {
        if (part == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(part.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            if (const auto* calls = delta->find("tool_calls"))
              for (const auto& call : calls->items())
                arguments += call.find("function")->member_str("arguments");
        return true;
      });
    } else {
      const auto output = gufo::json::parse(response.body);
      arguments = output.find("choices")
                      ->items()[0]
                      .find("message")
                      ->find("tool_calls")
                      ->items()[0]
                      .find("function")
                      ->member_str("arguments");
    }
    Expect(arguments == R"({" value ":"x"})",
           "a declared spaced name keeps its exact spelling");
  }
}

void TestNativeUnionToolTypes() {
  // Non-strict unions stay in Qwen's native syntax (#383). The parser tries
  // their typed alternatives first, as llama.cpp's qwen3-coder parser does,
  // while plain strings keep quotes and literal JSON text.
  auto body = gufo::json::parse(R"({"model":"test-model",
    "messages":[{"role":"user","content":"call record"}],
    "tools":[{"type":"function","function":{"name":"record","parameters":{
      "type":"object","properties":{
        "content":{"type":"string"},
        "provider":{"anyOf":[{"type":"string","const":"brave"},
                             {"type":"string","const":"exa"}]},
        "args":{"anyOf":[{"type":"string"},
                         {"type":"object","additionalProperties":true}]},
        "label":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "note":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}},
      "required":["content"]}}}],
    "tool_choice":"auto"})");
  FakeBackend backend;
  backend.pieces = {
      "<tool_call>\n<function=record>\n"
      "<parameter=content>\n#include \"a.h\"\n42\n</parameter>\n"
      "<parameter=provider>\nexa\n</parameter>\n"
      "<parameter=args>\n{\"key\": 1}\n</parameter>\n"
      "<parameter=label>\nnull\n</parameter>\n"
      "<parameter=note>\nplain text\n</parameter>\n"
      "<parameter=limit>\n3\n</parameter>\n"
      "</function>\n</tool_call>"};
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(response.status == 200, "native union call succeeds");
  const auto output = gufo::json::parse(response.body);
  const auto arguments = gufo::json::parse(output.find("choices")
                                               ->items()[0]
                                               .find("message")
                                               ->find("tool_calls")
                                               ->items()[0]
                                               .find("function")
                                               ->member_str("arguments"));
  Expect(arguments.dump() ==
             gufo::json::parse(R"({"content":"#include \"a.h\"\n42",
               "provider":"exa","args":{"key":1},"label":null,
               "note":"plain text","limit":3})")
                 .dump(),
         "native union arguments try typed alternatives before text");
}

void TestWildcardToolTypes() {
  using gufo::json::Value;
  using Constraint = gufo::sampling::JsonConstraint;
  using Format = Constraint::ToolFormat;
  const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"value":{"type":"string"}},"required":["value"],
    "patternProperties":{"^x_":{"type":"integer"}}})");
  const auto arguments = gufo::json::parse(
      R"({"value":"alpha","x_n":1,"x_b":true,"x_z":null,"x_a":[1],"x_o":{"n":1},"x_s":"1"})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters = Constraint::ToolParameters(schema, false, format);
    Expect(parameters != nullptr, "wildcard tools keep native framing");
    const auto grammar = Constraint::WithTools(
        nullptr, {{"record", parameters}}, true, false, format);
    // Qwen tags carry no type flag, so as in llama.cpp only declared
    // parameters are generated; DeepSeek's flag keeps typed wildcards.
    const auto expected = format == Format::kQwen
                              ? gufo::json::parse(R"({"value":"alpha"})")
                              : arguments;
    std::string call;
    if (format == Format::kQwen) {
      std::string extras;
      for (const auto& [name, value] : arguments.members())
        if (name != "value")
          extras += "<parameter=" + name + ">\n" +
                    (value.is_string() ? value.str() : value.dump()) +
                    "\n</parameter>\n";
      const std::string head =
          "<tool_call>\n<function=record>\n<parameter=value>\nalpha\n"
          "</parameter>\n";
      const std::string tail = "</function>\n</tool_call>";
      auto rejected = grammar->Start();
      for (unsigned char byte : head + extras + tail)
        rejected = grammar->Advance(rejected, byte);
      Expect(rejected.empty() || !grammar->Complete(rejected),
             "undeclared Qwen wildcards are not generated");
      call = head + tail;
    } else {
      call = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n";
      for (const auto& [name, value] : arguments.members())
        call += "<｜DSML｜parameter name=\"" + name + "\" string=\"" +
                (value.is_string() ? "true" : "false") + "\">" +
                (value.is_string() ? value.str() : value.dump()) +
                "</｜DSML｜parameter>\n";
      call += "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    }
    auto state = grammar->Start();
    for (unsigned char byte : call)
      state = grammar->Advance(state, byte);
    Expect(grammar->Complete(state), "round-trip call is grammar-admissible");
    for (bool responses : {false, true}) {
      auto body = gufo::json::parse(R"({"model":"test-model",
        "messages":[{"role":"user","content":"call record"}],
        "tools":[{"type":"function","function":{"name":"record","strict":false}}],
        "tool_choice":"auto"})");
      auto tool = body["tools"].items()[0];
      tool["function"]["parameters"] = schema;
      body["tools"] = Value::array();
      body["tools"].push_back(std::move(tool));
      FakeBackend backend;
      backend.pieces = {call};
      gufo::server::HttpResponse response;
      if (responses) {
        gufo::server::ChatRequest chat;
        Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
               "wildcard Responses controls are accepted");
        response = gufo::server::CreateOpenAiResponse(
            Request(body.dump()), backend, chat, 256, {}, false);
      } else {
        response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      }
      Expect(response.status == 200, "wildcard call succeeds");
      const auto output = gufo::json::parse(response.body);
      const auto encoded =
          responses
              ? output.find("output")->items().back().member_str("arguments")
              : output.find("choices")
                    ->items()[0]
                    .find("message")
                    ->find("tool_calls")
                    ->items()[0]
                    .find("function")
                    ->member_str("arguments");
      Expect(encoded == expected.dump(),
             "wildcard JSON types survive transport");
    }
  }
}

void TestToolMetadataAndFraming() {
  using gufo::json::Value;
  auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"call f"}],
    "tools":[{"type":"function","function":{"name":"f","strict":true,
      "parameters":{"type":"object","properties":{"text":{"type":"string"}},
        "required":["text"],"additionalProperties":false}}}],
    "tool_choice":{"type":"function","name":"f"},
    "parallel_tool_calls":null})");
  for (const bool stream : {false, true}) {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts nullable parallel_tool_calls");
    Expect(chat.forced_tool_name == "f" && !chat.parallel_tool_calls,
           "named tools retain their identity and single-call semantics");
    FakeBackend backend;
    backend.pieces = {
        "<tool_call>\n<function=f>\n<parameter=text>\r\nliteral\r\r\n"
        "</parameter>\n</function>\n</tool_call>"};
    const auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 256, {}, stream);
    const auto inspect = [&](const Value& result) {
      const auto* choice = result.find("tool_choice");
      Expect(choice && choice->member_str("type") == "function" &&
                 choice->member_str("name") == "f",
             "Responses echoes named function choices, including lifecycle "
             "events");
      if (result.member_str("status") == "completed") {
        const auto arguments = gufo::json::parse(
            result.find("output")->items().back().member_str("arguments"));
        Expect(arguments.member_str("text") == "literal\r",
               "CRLF framing retains a literal trailing carriage return");
      }
    };
    if (stream)
      response.streaming_body([&](std::string_view chunk) {
        const auto event =
            gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
        if (const auto* result = event.find("response"))
          inspect(*result);
        return true;
      });
    else
      inspect(gufo::json::parse(response.body));
  }
  body["tool_choice"] = "auto";
  gufo::server::ChatRequest chat;
  Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat) &&
             chat.parallel_tool_calls && chat.forced_tool_name.empty(),
         "null Responses parallel control retains default");
  chat.forced_tool_name = "f";
  Expect(!gufo::server::ParseOpenAiResponseControls(
             gufo::json::parse(R"({"tool_choice":"none"})"), &chat) &&
             chat.forced_tool_name.empty() &&
             chat.tool_choice == gufo::server::ChatRequest::ToolChoice::kNone,
         "an explicit choice clears inherited named-tool metadata");
  FakeBackend backend;
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             400,
         "Chat Completions parallel control still requires a boolean");
}

void TestResponsesOutput() {
  using Backend = gufo::server::TextGenerationBackend;
  for (const bool reasoning : {false, true}) {
    for (const bool limited : {false, true}) {
      FakeBackend backend;
      backend.pieces =
          reasoning ? std::vector<std::string>{"Check ", "\xE2\x94",
                                               "\x8C</think>", "\nAnswer"}
                    : std::vector<std::string>{"Answer ", "\xE2\x94", "\x8C"};
      backend.finish_reason = limited ? Backend::FinishReason::kLength
                                      : Backend::FinishReason::kStop;
      backend.reasoning_tokens = reasoning ? 3 : 0;
      gufo::server::ChatRequest chat;
      chat.reasoning.enabled = reasoning;
      const auto buffered = gufo::server::CreateOpenAiResponse(
          Request("{}"), backend, chat, 0, {}, false);
      const auto body = gufo::json::parse(buffered.body);
      const auto stream = gufo::server::CreateOpenAiResponse(
          Request("{}"), backend, chat, 0, {}, true);
      std::vector<gufo::json::Value> events;
      std::string text, thought;
      stream.streaming_body([&](std::string_view chunk) {
        const auto begin = chunk.find("\ndata: ");
        Expect(begin != std::string::npos, "Responses SSE has event and data");
        auto event = gufo::json::parse(chunk.substr(begin + 7));
        const auto type = event.member_str("type");
        Expect(chunk.starts_with("event: " + type + "\n"),
               "SSE name matches the semantic event type");
        Expect(event.member_size("sequence_number") == events.size(),
               "Responses events have consecutive sequence numbers");
        if (type == "response.output_text.delta")
          text += event.member_str("delta");
        if (type == "response.reasoning_summary_text.delta")
          thought += event.member_str("delta");
        if (const auto* response = event.find("response")) {
          Expect(response->find("parallel_tool_calls") &&
                     !response->find("parallel_tool_calls")->as_bool() &&
                     response->member_str("tool_choice") == "none" &&
                     response->find("tools")->items().empty(),
                 "Lifecycle events declare the supported tool policy");
        }
        events.push_back(std::move(event));
        return true;
      });
      Expect(events.front().member_str("type") == "response.created" &&
                 events[1].member_str("type") == "response.in_progress",
             "Responses starts with the lifecycle events");
      Expect(events.back().member_str("type") ==
                 (limited ? "response.incomplete" : "response.completed"),
             "Responses reports its terminal status");
      const auto& terminal = *events.back().find("response");
      Expect(text == (reasoning ? "Answer" : "Answer ┌") &&
                 thought == (reasoning ? "Check ┌" : ""),
             "Responses preserves UTF-8 and separates reasoning from text");
      const auto& items = terminal.find("output")->items();
      const auto& buffered_items = body.find("output")->items();
      Expect(items.size() == buffered_items.size(),
             "Buffered and streamed Responses have the same output items");
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto field =
            items[i].member_str("type") == "reasoning" ? "summary" : "content";
        Expect(items[i].find(field)->dump() ==
                   buffered_items[i].find(field)->dump(),
               "Buffered and streamed text/reasoning agree");
      }
      Expect(terminal.find("usage")
                     ->find("input_tokens_details")
                     ->member_size("cached_tokens") == 5,
             "Responses retains prompt-cache usage");
      for (const auto* response : {&body, &terminal}) {
        Expect(response->find("parallel_tool_calls") &&
                   !response->find("parallel_tool_calls")->as_bool() &&
                   response->member_str("tool_choice") == "none" &&
                   response->find("tools")->items().empty(),
               "Responses includes SDK-required tool fields");
        const auto& usage = *response->find("usage");
        Expect(usage.find("input_tokens_details")
                           ->member_size("cache_write_tokens") == 2 &&
                   usage.find("output_tokens_details")
                           ->member_size("reasoning_tokens") ==
                       backend.reasoning_tokens &&
                   usage.member_size("output_tokens") == backend.pieces.size(),
               "Responses includes new-cache and actual reasoning usage");
      }
    }
  }
}

void TestGemmaNativeToolCalls() {
  using Backend = gufo::server::TextGenerationBackend;
  FakeBackend backend;
  backend.markup = {
      .reasoning_start = "<|channel>thought\n",
      .reasoning_end = "<channel|>",
      .tool_syntax = Backend::OutputMarkup::ToolSyntax::kGemma4,
  };
  const std::string body = R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"weather in Rome"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"get_weather",
            "parameters":{
              "type":"object",
              "properties":{"city":{"type":"string"}},
              "required":["city"]
            }
          }
        }]
      })";
  const auto expect_call = [](const gufo::server::HttpResponse& response,
                              std::string_view message) {
    Expect(
        response.status == 200 &&
            response.body.find("tool_calls") != std::string::npos &&
            response.body.find(R"(\"city\":\"Rome\")") != std::string::npos &&
            response.body.find("call:") == std::string::npos,
        message);
  };
  backend.pieces = {"<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>}",
                    "<tool_call|>"};
  auto response = gufo::server::HandleOpenAiChat(Request(body), backend);
  Expect(backend.last_request.constrained_tools,
         "Gemma 4 calls are constrained like other models' native calls");
  expect_call(response, "Constrained native Gemma 4 calls are parsed");

  // With thinking on, the output may open the thought channel before a call.
  backend.initial_output_state_override = Backend::InitialOutputState::kAuto;
  backend.pieces = {"<|channel>thought\nUse the tool.<channel|>",
                    "<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>}",
                    "<tool_call|>"};
  for (const bool stream : {false, true}) {
    auto request = gufo::json::parse(body);
    request["stream"] = stream;
    request["tool_choice"] = "required";
    response = gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
    const auto output = stream ? std::string() : response.body;
    std::string streamed;
    if (stream)
      response.streaming_body([&](std::string_view chunk) {
        streamed += chunk;
        return true;
      });
    const auto& text = stream ? streamed : output;
    Expect(text.find("Use the tool.") != std::string::npos &&
               text.find(R"("reasoning_content")") != std::string::npos &&
               text.find("<|channel>") == std::string::npos &&
               text.find("call:") == std::string::npos &&
               text.find(R"(\"city\":\"Rome\")") != std::string::npos,
           "A Gemma 4 thought before a required call stays reasoning");
  }

  // An agent turn: automatic, streamed token by token. Gemma 4 opens an empty
  // thought channel after tool responses even without thinking.
  backend.initial_output_state_override = Backend::InitialOutputState::kAuto;
  const std::string call =
      "<|channel>thought\n<channel|><|tool_call>call:get_weather{city:<|\"|>"
      "Rome<|\"|>,note:<|\"|>a, "
      "{b}: \"c\"\n<|\"|>}<tool_call|>";
  backend.pieces.clear();
  for (std::size_t i = 0; i < call.size(); i += 3)
    backend.pieces.push_back(call.substr(i, 3));
  auto request = gufo::json::parse(body);
  request["stream"] = true;
  request["parallel_tool_calls"] = false;
  response = gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
  std::string streamed;
  response.streaming_body([&](std::string_view chunk) {
    streamed += chunk;
    return true;
  });
  std::string arguments;
  for (std::size_t at = streamed.find("data: {"); at != std::string::npos;
       at = streamed.find("data: {", at + 1)) {
    const auto end = streamed.find("\n\n", at);
    const auto event = gufo::json::parse(streamed.substr(at + 6, end - at - 6));
    if (const auto* choices = event.find("choices"))
      for (const auto& choice : choices->items())
        if (const auto* calls = choice.find("delta")->find("tool_calls"))
          for (const auto& call : calls->items())
            arguments += call.find("function")->member_str("arguments");
  }
  Expect(streamed.find("error") == std::string::npos &&
             streamed.find("channel") == std::string::npos &&
             arguments == R"({"city":"Rome","note":"a, {b}: \"c\"\n"})",
         "A streamed Gemma 4 call is parsed: " + streamed);
}

void TestResponsesLiveAndCancellation() {
  FakeBackend backend;
  backend.pieces = {"<tool_call>first", "second"};
  backend.block_after_first_piece = true;
  auto response = gufo::server::CreateOpenAiResponse(Request("{}"), backend, {},
                                                     0, {}, true);
  std::atomic<bool> first{false};
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view chunk) {
      if (chunk.find("event: response.output_text.delta") != std::string::npos)
        first = true;
      return true;
    });
  });
  Expect(backend.WaitForFirstPiece() && first && !backend.completed,
         "Responses sends text before generation completes");
  backend.Release();
  writer.join();

  backend.block_after_first_piece = false;
  backend.completed = false;
  auto cancelled = gufo::server::CreateOpenAiResponse(Request("{}"), backend,
                                                      {}, 0, {}, true);
  bool terminal = false;
  cancelled.streaming_body([&](std::string_view chunk) {
    terminal |= chunk.find("event: response.completed") != std::string::npos;
    return chunk.find("event: response.output_text.delta") == std::string::npos;
  });
  Expect(!terminal && !backend.completed,
         "Responses disconnect cancels generation without a completed event");
}

}  // namespace

void TestStreamingPromptProgress() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.progress = {{.total = 7, .cache = 5, .processed = 5, .time_ms = 0},
                      {.total = 7, .cache = 5, .processed = 7, .time_ms = 3}};
  const auto stream = [&](std::string_view options) {
    auto response = gufo::server::HandleOpenAiChat(
        Request(
            R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],)"
            R"("stream":true)" +
            std::string(options) + "}"),
        backend);
    Expect(response.status == 200 && response.streaming_body,
           "progress request streams");
    std::string output;
    response.streaming_body([&](std::string_view chunk) {
      output.append(chunk);
      return true;
    });
    return output;
  };

  const auto output = stream(R"(,"return_progress":true)");
  const auto first = output.find(
      R"("delta":{},"finish_reason":null}],)"
      R"("prompt_progress":{"total":7,"cache":5,"processed":5,"time_ms":0}})");
  const auto last = output.find(
      R"("delta":{},"finish_reason":null}],)"
      R"("prompt_progress":{"total":7,"cache":5,"processed":7,"time_ms":3}})");
  const auto content = output.find(R"("content":"Hel")");
  Expect(first != std::string::npos && last != std::string::npos &&
             first < last && last < content,
         "empty-delta prompt progress chunks precede streamed content");

  for (const auto* options :
       {"", R"(,"return_progress":false)", R"(,"return_progress":null)"}) {
    Expect(stream(options).find("prompt_progress") == std::string::npos,
           "prompt progress is opt-in");
  }

  const auto invalid = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],)"
          R"("stream":true,"return_progress":"yes"})"),
      backend);
  Expect(invalid.status == 400 &&
             invalid.body.find("invalid_return_progress") != std::string::npos,
         "return_progress must be a boolean");
}

void TestResponsesPromptProgress() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.progress = {{.total = 7, .cache = 5, .processed = 5, .time_ms = 0},
                      {.total = 7, .cache = 5, .processed = 7, .time_ms = 3}};
  for (const auto* setting : {"true", "false", "null"}) {
    gufo::server::ChatRequest chat;
    const auto body = gufo::json::parse(std::string(R"({"return_progress":)") +
                                        setting + "}");
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts boolean and null progress settings");
    const auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 8, {}, true);
    std::size_t sequence = 0, progress_count = 0;
    bool content_seen = false;
    response.streaming_body([&](std::string_view chunk) {
      const auto offset = chunk.find("data: ");
      Expect(offset != std::string_view::npos, "Responses SSE contains data");
      const auto event = gufo::json::parse(chunk.substr(offset + 6));
      Expect(event.member_size("sequence_number") == sequence++,
             "progress preserves contiguous Responses sequence numbers");
      if (const auto* progress = event.find("prompt_progress")) {
        Expect(
            !content_seen &&
                event.member_str("type") == "response.in_progress" &&
                event.find("response")->member_str("status") == "in_progress",
            "Responses progress precedes content in an in_progress event");
        Expect(
            progress->member_size("processed") == (progress_count == 0 ? 5 : 7),
            "Responses progress preserves cached and processed counts");
        ++progress_count;
      }
      content_seen |= event.member_str("type") == "response.output_text.delta";
      return true;
    });
    Expect(content_seen && progress_count == (chat.return_progress ? 2 : 0),
           "Responses progress is opt-in and preserves content");
  }
  gufo::server::ChatRequest chat;
  Expect(gufo::server::ParseOpenAiResponseControls(
             gufo::json::parse(R"({"return_progress":1})"), &chat)
             .has_value(),
         "Responses rejects non-boolean progress settings");
}

// Issue #383: the model repeats the closing framing of a call it just made, and
// a client that stores assistant content replays that markup as history on
// every later turn. Issue #266: the same framing folded into an argument value
// or an unfinished call silently dropped. Both are parse-path defects, so they
// are covered here for buffered and streamed transports.
void TestToolClosingFraming() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model", "messages":[{"role":"user","content":"use f"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"}},
      "required":["text"],"additionalProperties":false}}}]
  })");
  const std::string call =
      "<tool_call><function=f><parameter=text>\n42\n</parameter></function></"
      "tool_call>";
  const std::string inline_call =
      "<tool_call><function=f><parameter=text>42</parameter></function></"
      "tool_call>";
  const std::string envelope =
      "<invoke name=\"f\"><parameter name=\"text\">x</parameter></invoke>";
  const std::string raw_xml =
      "<invoke name=\"documentation\"><parameter "
      "name=\"text\">x</parameter></invoke>";
  struct Case {
    std::string text;
    std::size_t calls;
    std::string argument;
    std::string content;
  };
  const std::vector<Case> qwen_cases{
      {"I'll update `config.py\n" + call, 1, R"({"text":"42"})",
       "I'll update `config.py\n"},
      {"Let`s write it.\n" + call, 1, R"({"text":"42"})", "Let`s write it.\n"},
      {"looks like `" + call + "` and no call.", 0, "",
       "looks like `" + call + "` and no call."},
      {"looks like ``" + call + "`` and no call.", 0, "",
       "looks like ``" + call + "`` and no call."},
      {"looks like `" + call, 1, R"({"text":"42"})", "looks like `"},
      {"looks like ``" + call, 1, R"({"text":"42"})", "looks like ``"},
      {"`example\n\n" + call, 1, R"({"text":"42"})", "`example\n\n"},
      {"looks like `literal <think>\n" + call + "`", 0, "",
       "looks like `literal <think>\n" + call + "`"},
      {call + "\n</invoke>\n</parameter>\n</function>\n", 1, R"({"text":"42"})",
       ""},
      {call + "\n</function>\n" + call, 2, R"({"text":"42"})", ""},
      {call + "\n" + envelope, 1, R"({"text":"42"})", ""},
      {call + "\n</invoke>\n<|im_end|>", 1, R"({"text":"42"})",
       std::string(kImEnd)},
      {"</invoke>\n" + call, 1, R"({"text":"42"})", "</invoke>\n"},
      {"</invoke>", 0, "", "</invoke>"},
      {"Text </parameter>", 0, "", "Text </parameter>"},
      {"The token is <|im_end|>", 0, "", "The token is <|im_end|>"},
      {"<parameter>x</parameter>", 0, "", "<parameter>x</parameter>"},
      {raw_xml, 0, "", raw_xml},
      {"Raw XML: " + raw_xml, 0, "", "Raw XML: " + raw_xml},
      {envelope, 0, "", ""},
      {"Planning.\n" + envelope, 0, "", "Planning."},
      {"<invoke name=\"f\"><parameter name=\"text\">x", 0, "", ""},
      {"Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</"
       "tool_call>",
       0, "",
       "Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</"
       "tool_call>"},
      {"<function=f><parameter=text>x</parameter></function>", 0, "",
       "<function=f><parameter=text>x</parameter></function>"},
      {"```xml\n" + envelope + "\n```", 0, "", "```xml\n" + envelope + "\n```"},
      {"`" + envelope + "` and prose", 0, "", "`" + envelope + "` and prose"},
      {"```python\nprint(1)\n" + call, 1, R"({"text":"42"})",
       "```python\nprint(1)\n"},
      {"```xml\n" + call + "\n```", 0, "", "```xml\n" + call + "\n```"},
      {"```xml\n" + call + "\n```\n" + call, 1, R"({"text":"42"})",
       "```xml\n" + call + "\n```\n"},
      {"`" + inline_call + "`", 0, "", "`" + inline_call + "`"},
      {"``" + inline_call + "``", 0, "", "``" + inline_call + "``"},
      {"```xml\n<tool_call><function=unknown></function></tool_call>", 0, "",
       "```xml\n<tool_call><function=unknown></function></tool_call>"},
      {"```xml\n<tool_call><function=f></function></tool_call>", 0, "",
       "```xml\n<tool_call><function=f></function></tool_call>"},
      {"```xml\n<tool_call><function=f><parameter=extra>x</parameter></"
       "function></tool_call>",
       0, "",
       "```xml\n<tool_call><function=f><parameter=extra>x</parameter></"
       "function></tool_call>"},
      {"<tool_call><function=f><parameter=text>\nEOS = "
       "\"<|im_end|>\"\n</parameter></function></tool_call>",
       1, R"({"text":"EOS = \"<|im_end|>\""})", ""},
      {"<tool_call><function=f><parameter=text>\n<|im_start|><|endoftext|>"
       "<|not_a_vocab_entry|>\n</parameter></function></tool_call>",
       1, R"({"text":"<|im_start|><|endoftext|><|not_a_vocab_entry|>"})", ""},
      {"<tool_call><function=f><parameter=text>\n</invoke>\n</parameter></"
       "function></tool_call>",
       1, R"({"text":"</invoke>"})", ""},
  };
  // As in llama.cpp, DeepSeek output is content outside its native block,
  // which ends the output: client envelopes and closers are never removed.
  const std::string dsml =
      "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
      "<｜DSML｜parameter name=\"text\" string=\"true\">42"
      "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
  const std::vector<Case> deepseek_cases{
      {envelope, 0, "", envelope},
      {"Planning.\n" + envelope, 0, "", "Planning.\n" + envelope},
      {"<invoke name=\"f\"><parameter name=\"text\">x", 0, "",
       "<invoke name=\"f\"><parameter name=\"text\">x"},
      {"</invoke>\n</parameter>", 0, "", "</invoke>\n</parameter>"},
      {call, 0, "", call},
      {"</invoke>" + dsml, 1, R"({"text":"42"})", "</invoke>"},
      {"Planning.\n" + envelope + dsml, 1, R"({"text":"42"})",
       "Planning.\n" + envelope},
  };
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  for (const auto& [format, cases] :
       {std::pair{Format::kQwen, &qwen_cases},
        std::pair{Format::kDeepSeek, &deepseek_cases}})
    for (const auto& item : *cases) {
      for (bool stream : {false, true}) {
        for (bool bytewise : {false, true}) {
          auto body = schema;
          body["stream"] = stream;
          body["tool_choice"] = "auto";
          FakeBackend backend;
          backend.tool_format = format;
          if (bytewise)
            for (const char byte : item.text)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces.push_back(item.text);
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200, "framing request succeeds");
          std::vector<Value> calls;
          std::string content;
          std::string finish;
          if (!stream) {
            const auto output = gufo::json::parse(response.body);
            const auto& choice = output.find("choices")->items()[0];
            const auto& message = *choice.find("message");
            content = message.member_str("content");
            finish = choice.member_str("finish_reason");
            if (const auto* found = message.find("tool_calls"))
              calls.assign(found->items().begin(), found->items().end());
          } else {
            response.streaming_body([&](std::string_view part) {
              if (part == "data: [DONE]\n\n")
                return true;
              const auto event = gufo::json::parse(part.substr(6));
              if (event.contains("error"))
                std::cerr << "Framing input: " << item.text
                          << "\nStream error: " << event.dump() << '\n';
              Expect(!event.contains("error"),
                     "quoted examples do not start tentative tool calls");
              for (const auto& choice : event.find("choices")->items()) {
                finish += choice.member_str("finish_reason");
                if (const auto* delta = choice.find("delta")) {
                  content += delta->member_str("content");
                  if (const auto* found = delta->find("tool_calls"))
                    AppendToolDeltas(calls, *found, stream);
                }
              }
              return true;
            });
          }
          if (calls.size() != item.calls || content != item.content) {
            std::cerr << "Framing input: " << item.text << "\nstream=" << stream
                      << " bytewise=" << bytewise
                      << "\nExpected content: " << item.content
                      << "\nActual content: " << content
                      << "\nExpected calls: " << item.calls
                      << ", actual: " << calls.size() << '\n';
          }
          Expect(calls.size() == item.calls,
                 "only complete declared calls are emitted");
          Expect(content == item.content,
                 "literal content and framing are distinguished using request "
                 "context");
          Expect(finish == (item.calls ? "tool_calls" : "stop"),
                 "finish reason agrees with parsed calls");
          if (!calls.empty())
            Expect(calls.back().find("function")->member_str("arguments") ==
                       item.argument,
                   "literal arguments survive exactly");
        }
      }
    }
}

void TestUnconstrainedQuoteFallbackChecksSchema() {
  const std::string prefix = "```python\nprint(1)\n";
  const std::string valid =
      "<tool_call>\n<function=f>\n<parameter=text>42</parameter>\n</"
      "function>\n</tool_call>";
  const std::string wrong =
      "<tool_call><function=f><parameter=text>43</parameter></function></"
      "tool_call>";
  const std::string missing = "<tool_call><function=f></function></tool_call>";
  struct Case {
    std::string text;
    bool call;
    std::string content;
  };
  for (const auto& item : std::vector<Case>{
           {prefix + valid, true, prefix},
           {prefix + wrong, false, prefix + wrong},
           {prefix + missing, false, prefix + missing},
           {"looks like `" + valid + "`", false, "looks like `" + valid + "`"},
           {"looks like ``" + valid + "``", false,
            "looks like ``" + valid + "``"},
           {"looks like `" + valid, true, "looks like `"},
           {"I'll update `config.py\n" + valid, true,
            "I'll update `config.py\n"},
           {"Let`s write it.\n" + valid, true, "Let`s write it.\n"},
           {"looks like ``" + valid, true, "looks like ``"},
           {"looks like `" + wrong, false, "looks like `" + wrong},
           {"looks like `" + missing, false, "looks like `" + missing},
           {"looks like `" + wrong + "\n\n", false,
            "looks like `" + wrong + "\n\n"},
       }) {
    for (const bool constrained : {false, true}) {
      for (const bool stream : {false, true}) {
        for (const bool bytewise : {false, true}) {
          gufo::server::ChatRequest chat;
          chat.reasoning.enabled = false;
          Expect(
              !gufo::server::ParseOpenAiResponseControls(
                  gufo::json::parse(
                      R"({"tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"text":{"type":"string","const":"42"}},"required":["text"],"additionalProperties":false}}]})"),
                  &chat),
              "fallback fixture has valid tool controls");
          chat.constrained_tools = constrained;
          FakeBackend backend;
          backend.tool_format =
              gufo::sampling::JsonConstraint::ToolFormat::kQwen;
          if (bytewise)
            for (const char byte : item.text)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces.push_back(item.text);
          const auto response = gufo::server::CreateOpenAiResponse(
              Request("{}"), backend, chat, 512, {}, stream);
          gufo::json::Value output;
          std::string streamed;
          if (!stream)
            output = gufo::json::parse(response.body);
          else
            response.streaming_body([&](std::string_view chunk) {
              const auto offset = chunk.find("data: ");
              const auto event = gufo::json::parse(chunk.substr(offset + 6));
              if (event.member_str("type") == "response.output_text.delta")
                streamed += event.member_str("delta");
              if (event.member_str("type") == "response.completed")
                output = *event.find("response");
              return true;
            });
          std::string content;
          std::size_t calls = 0;
          for (const auto& part : output.find("output")->items()) {
            if (part.member_str("type") == "function_call") {
              ++calls;
              Expect(part.member_str("name") == "f" &&
                         part.member_str("arguments") == R"({"text":"42"})",
                     "fallback emits exactly the declared schema-valid call");
            }
            if (part.member_str("type") == "message") {
              for (const auto& text : part.find("content")->items())
                content += text.member_str("text");
            }
          }
          Expect(calls == (item.call ? 1u : 0u),
                 "both parser paths apply unfinished-quote schema checks and "
                 "protect "
                 "inline quotations");
          Expect(content == item.content,
                 "quoted documentation and invalid fence examples stay "
                 "byte-exact");
          if (stream)
            Expect(streamed == content,
                   "streamed documentation agrees with the final response");
        }
      }
    }
  }
}

void TestUnfinishedInlineReasoningFallback() {
  const std::string prefix = "Check `foo then\n";
  const std::string valid =
      "<tool_call><function=f><parameter=text>42</parameter></function></"
      "tool_call>";
  const std::string wrong =
      "<tool_call><function=f><parameter=text>43</parameter></function></"
      "tool_call>";
  const std::string missing = "<tool_call><function=f></function></tool_call>";
  struct Case {
    std::string text;
    bool call;
    std::string reasoning;
    std::string content;
  };
  for (const auto& item : std::vector<Case>{
           {prefix + valid, true, prefix, ""},
           {prefix + valid + "</think>", true, prefix, ""},
           {prefix + wrong, false, prefix + wrong, ""},
           {prefix + missing, false, prefix + missing, ""},
           {prefix + wrong + "</think>Answer", false, prefix + wrong, "Answer"},
           {prefix + valid + "`</think>Answer", false, prefix + valid + "`",
            "Answer"},
           {prefix + valid + "`</think>" + valid, true, prefix + valid + "`",
            ""},
       }) {
    for (const bool stream : {false, true})
      for (const bool bytewise : {false, true}) {
        gufo::server::ChatRequest chat;
        chat.reasoning.enabled = true;
        Expect(
            !gufo::server::ParseOpenAiResponseControls(
                gufo::json::parse(
                    R"({"tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"text":{"type":"string","const":"42"}},"required":["text"],"additionalProperties":false}}]})"),
                &chat),
            "reasoning fallback fixture has valid controls");
        // Exercise the legacy implicit transition before </think>. Constrained
        // reasoning deliberately requires the explicit phase delimiter.
        chat.constrained_tools = false;
        FakeBackend backend;
        backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
        const auto& raw = item.text;
        if (bytewise)
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
        else
          backend.pieces = {raw};
        const auto response = gufo::server::CreateOpenAiResponse(
            Request("{}"), backend, chat, 512, {}, stream);
        gufo::json::Value output;
        std::string streamed_content, streamed_reasoning;
        if (!stream)
          output = gufo::json::parse(response.body);
        else
          response.streaming_body([&](std::string_view chunk) {
            const auto event =
                gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
            if (event.member_str("type") == "response.output_text.delta")
              streamed_content += event.member_str("delta");
            if (event.member_str("type") ==
                "response.reasoning_summary_text.delta")
              streamed_reasoning += event.member_str("delta");
            if (event.member_str("type") == "response.completed")
              output = *event.find("response");
            return true;
          });
        std::string content, reasoning;
        std::size_t calls = 0;
        for (const auto& part : output.find("output")->items()) {
          if (part.member_str("type") == "function_call") {
            ++calls;
            Expect(part.member_str("name") == "f" &&
                       part.member_str("arguments") == R"({"text":"42"})",
                   "recovered reasoning call satisfies its complete schema");
          }
          for (const auto* field : {"content", "summary"})
            if (const auto* parts = part.find(field)) {
              for (const auto& text : parts->items())
                (std::string_view(field) == "summary" ? reasoning : content) +=
                    text.member_str("text");
            }
        }
        if (calls != (item.call ? 1u : 0u) || content != item.content ||
            Trimmed(reasoning) != Trimmed(item.reasoning)) {
          std::cerr << "Reasoning fallback: " << raw << "\nstream=" << stream
                    << " bytewise=" << bytewise << "\ncontent=" << content
                    << "\nreasoning=" << reasoning << "\ncalls=" << calls
                    << '\n';
        }
        Expect(calls == (item.call ? 1u : 0u),
               "an unfinished reasoning quote recovers only a declared "
               "schema-valid call");
        Expect(content == item.content &&
                   Trimmed(reasoning) == Trimmed(item.reasoning),
               "quoted or rejected examples remain in their original phase");
        if (stream)
          Expect(streamed_content == content &&
                     Trimmed(streamed_reasoning) == Trimmed(reasoning),
                 "held reasoning and content agree with final output");
      }
  }
}

// Issue #383 follow-up: a model answering *about* the dialect names its tags,
// and that prose must survive as content. Captured from the live session whose
// reply was truncated at the first inline mention (assistant row 103597) while
// a phantom `terminal` call carrying "..." reached the client and ran.
void TestProseAboutTheDialectIsNotACall() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model",
    "messages":[{"role":"user","content":"why was it a bug"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"}},
      "required":["text"]}}}]
  })");
  const std::string message =
      "Got it — no comment posted. Here's the explanation, just for you.\n"
      "\n"
      "## Why it was a bug\n"
      "\n"
      "**The setup.** The Qwen chat model was trained to signal tool calls "
      "with special markup: `<tool_call>`. The server's job is to act as a "
      "**translator**: it watches the raw token stream, converts that markup "
      "into the structured tool_calls field of the OpenAI API, and makes sure "
      "the *rest* of the markup never appears as text.\n"
      "\n"
      "1. **Closing tags escaped into content.** When the call parsed, gufo "
      "removed the opening envelope but let the tail — `</parameter>`, "
      "`</invoke>`, `</function>` — escape into content.\n"
      "\n"
      "The client repeats `<invoke name=\"terminal\">` in history, so the "
      "model mimics that dialect. Quoted, the shape is:\n"
      "\n"
      "```\n"
      "<invoke name=\"f\">\n"
      "<parameter name=\"text\">x</parameter>\n"
      "</invoke>\n"
      "```\n"
      "\n"
      "The same holds for `<|tool_call|>` and for `<function=f>`. Nothing "
      "above is a call being made, and none of it may be cut.";
  for (bool stream : {false, true}) {
    for (bool bytewise : {false, true}) {
      auto body = schema;
      body["stream"] = stream;
      body["tool_choice"] = "auto";
      FakeBackend backend;
      if (bytewise) {
        for (char byte : message)
          backend.pieces.emplace_back(1, byte);
      } else {
        backend.pieces.push_back(message);
      }
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "an answer about the dialect is served");
      std::vector<Value> calls;
      std::string content;
      if (!stream) {
        const auto output = gufo::json::parse(response.body);
        const auto& choice =
            *output.find("choices")->items()[0].find("message");
        if (const auto* text = choice.find("content");
            text && text->is_string())
          content = text->get_str();
        if (const auto* found = choice.find("tool_calls"))
          calls.assign(found->items().begin(), found->items().end());
      } else {
        std::string output;
        response.streaming_body([&](std::string_view part) {
          output += part;
          return true;
        });
        std::size_t cursor = 0;
        while ((cursor = output.find("data: ", cursor)) != std::string::npos) {
          const auto begin = cursor + 6;
          cursor = output.find('\n', begin);
          const auto payload = output.substr(begin, cursor - begin);
          if (payload == "[DONE]")
            break;
          const auto event = gufo::json::parse(payload);
          const auto* choices = event.find("choices");
          if (!choices || choices->empty())
            continue;
          const auto* delta = choices->items()[0].find("delta");
          if (!delta)
            continue;
          content += delta->member_str("content");
          if (const auto* found = delta->find("tool_calls"))
            AppendToolDeltas(calls, *found, stream);
        }
      }
      if (calls.empty() && Trimmed(content) != Trimmed(message))
        std::cerr << "Prose about the dialect, stream=" << stream
                  << " bytewise=" << bytewise
                  << "\nExpected content: " << message
                  << "\nActual content: " << content << '\n';
      Expect(calls.empty(), "prose that names the dialect is not a call");
      Expect(Trimmed(content) == Trimmed(message),
             "an answer about the dialect survives in full");
    }
  }
}

/// A bracket-dense line, in units that never split across pieces.
std::string AngleUnits(std::size_t units) {
  std::string text;
  text.reserve(units * 3);
  for (std::size_t i = 0; i < units; ++i)
    text += "<a>";
  return text;
}

/// Bracket-dense content is ordinary text: every '<' in it is one probe for the
/// parser, and a held or dropped run shows up as text that lost its brackets.
///
/// This is the deterministic form of the live case. Asked to repeat such a
/// line, a model runs away into a repetition loop and never stops, so the live
/// case reports a model runaway rather than a framing failure.
void TestBracketDenseContent() {
  constexpr std::size_t kUnits = 2000;
  constexpr std::size_t kUnitsPerPiece = 30;
  const std::string dense = AngleUnits(kUnits);
  // The tail is what the framing rules act on: a pipe-shaped spelling the
  // vocabulary does not own is prose, whether it is complete or still arriving.
  const std::vector<std::string> tails{"", "<|not_a_vocab|>",
                                       "<|not_a_vocab_entry"};
  for (const std::string& tail : tails) {
    FakeBackend backend;
    for (std::size_t offset = 0; offset < dense.size();
         offset += kUnitsPerPiece * 3)
      backend.pieces.push_back(dense.substr(offset, kUnitsPerPiece * 3));
    if (!tail.empty())
      backend.pieces.push_back(tail);

    auto body = gufo::json::parse(
        R"({"model":"test-model","messages":[{"role":"user","content":"echo it"}],)"
        R"("reasoning_effort":"none"})");
    const auto buffered =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    const auto parsed = gufo::json::parse(buffered.body);
    const auto* message = parsed.find("choices")->items()[0].find("message");
    Expect(message->member_str("content") == dense + tail,
           "bracket-dense prose and its tail survive verbatim");

    body["stream"] = true;
    const auto streamed =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string content;
    streamed.streaming_body([&](std::string_view chunk) {
      if (chunk == "data: [DONE]\n\n")
        return true;
      const auto event = gufo::json::parse(chunk.substr(6));
      for (const auto& choice : event.find("choices")->items()) {
        const auto* delta = choice.find("delta");
        if (delta)
          content += delta->member_str("content");
      }
      return true;
    });
    Expect(content == dense + tail,
           "the streaming path retains bracket-dense prose and its tail");
  }
}

struct ToolBoundaryOutput {
  std::string content, reasoning, finish, error_message;
  std::vector<std::string> arguments, names;
  std::size_t incomplete_items{0};  // Responses streams: items closed as incomplete
  bool failed{false};
};

/// Token-level reasoning end for a fixture: untracked (the text rule), or the
/// index of the piece that is the generated tag token (none: never closed).
struct TokenReasoningEnd {
  bool tracked{false};
  std::optional<std::size_t> piece;
};

// Exercise public serializers with independent fixtures. Validate event
// ownership too: a delta cannot precede its header or follow its block's close.
ToolBoundaryOutput ToolBoundaryRequest(
    const std::vector<std::string>& pieces,
    gufo::sampling::JsonConstraint::ToolFormat format, int api, bool stream,
    FakeBackend::InitialOutputState initial =
        FakeBackend::InitialOutputState::kContent,
    FakeBackend::FinishReason finish = FakeBackend::FinishReason::kStop,
    FakeBackend::TerminationCause cause = FakeBackend::TerminationCause::kNone,
    const TokenReasoningEnd& token_end = {}, int shape = 0) {
  using gufo::json::Value;
  FakeBackend backend;
  backend.pieces = pieces;
  backend.track_reasoning_end = token_end.tracked;
  backend.reasoning_end_piece = token_end.piece;
  backend.tool_format = format;
  backend.initial_output_state_override = initial;
  backend.finish_reason = finish;
  backend.termination_cause = cause;
  backend.stop_sequence = "HALT";
  auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"test framing"}],
    "tool_choice":"auto","tools":[
      {"type":"function","function":{"name":"record","parameters":{
        "type":"object","properties":{"text":{"type":"string"},
          "data":{"type":"object"}},"additionalProperties":false}}},
      {"type":"function","function":{"name":"noop","parameters":{
        "type":"object","properties":{},"additionalProperties":false}}}
    ]})");
  body["stream"] = stream;
  body["reasoning_effort"] =
      initial == FakeBackend::InitialOutputState::kContent ? "none" : "low";
  // Shape 1: no tools (the unconstrained parser); 2: a JSON response format.
  if (shape != 0) {
    body.erase_member("tools");
    body.erase_member("tool_choice");
  }
  if (shape == 2)
    body["response_format"] = gufo::json::parse(R"({"type":"json_schema",
      "json_schema":{"name":"answer","strict":true,"schema":{"type":"object",
      "properties":{"a":{"type":"integer"}},"required":["a"],
      "additionalProperties":false}}})");
  gufo::server::HttpResponse response;
  if (api == 0)
    response = gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  else {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "tag boundary controls parse");
    response = api == 1 ? gufo::server::CreateOpenAiResponse(
                              Request("{}"), backend, chat, 4096, {}, stream)
                        : gufo::server::CreateAnthropicMessage(
                              Request("{}"), backend, chat, 4096, {}, stream);
  }
  Expect(response.status == 200, "tag boundary request is accepted");
  ToolBoundaryOutput out;
  std::vector<Value> chat_calls;
  std::map<std::size_t, bool> open;
  std::map<std::size_t, std::size_t> tools;
  bool terminal = false;
  const auto header = [&](std::size_t index, const Value& item) {
    Expect(!tools.contains(index), "tool header occurs once");
    Expect(!item.member_str("call_id", item.member_str("id")).empty(),
           "tool header has an ID");
    tools[index] = out.arguments.size();
    out.arguments.emplace_back();
    out.names.push_back(item.member_str("name"));
  };
  const auto item = [&](const Value& value) {
    const auto type = value.member_str("type");
    if (type == "function_call" || type == "tool_use") {
      out.names.push_back(value.member_str("name"));
      out.arguments.push_back(type == "function_call"
                                  ? value.member_str("arguments")
                                  : value.find("input")->dump());
    } else if (type == "thinking")
      out.reasoning += value.member_str("thinking");
    else if (type == "text")
      out.content += value.member_str("text");
    else
      for (const auto* key : {"summary", "content"})
        if (const auto* parts = value.find(key))
          for (const auto& part : parts->items())
            (std::string_view(key) == "summary" ? out.reasoning
                                                : out.content) +=
                part.member_str("text");
  };
  const auto collect = [&](const Value& event) {
    const auto type = event.member_str("type");
    const auto* error = event.find("error");
    if ((error && !error->is_null()) || type == "response.failed") {
      out.failed = true;
      if (error && error->is_object())
        out.error_message = error->member_str("message");
      else if (const auto* response_error =
                   event.find("response") ? event.find("response")->find("error")
                                          : nullptr)
        out.error_message = response_error->member_str("message");
      return;
    }
    if (api == 0) {
      for (const auto& choice : event.find("choices")->items()) {
        const auto* message = choice.find(stream ? "delta" : "message");
        out.content += message->member_str("content");
        out.reasoning += message->member_str("reasoning_content");
        if (const auto* calls = message->find("tool_calls")) {
          Expect(!terminal, "tool deltas precede the terminal event");
          AppendToolDeltas(chat_calls, *calls, stream);
        }
        if (!choice.member_str("finish_reason").empty()) {
          Expect(!terminal, "one terminal choice");
          terminal = true;
          out.finish = choice.member_str("finish_reason");
        }
      }
    } else if (!stream) {
      for (const auto& value :
           event.find(api == 1 ? "output" : "content")->items())
        item(value);
      out.finish = event.member_str(api == 1 ? "status" : "stop_reason");
      terminal = true;
    } else {
      const auto index = event.member_size(api == 1 ? "output_index" : "index");
      if (type == "response.output_item.added" ||
          type == "content_block_start") {
        if (terminal || open.contains(index)) {
          std::cerr << "duplicate block api=" << api
                    << " event=" << event.dump() << "\nraw=";
          for (const auto& piece : pieces)
            std::cerr << piece;
          std::cerr << '\n';
        }
        Expect(!terminal && !open.contains(index),
               "each output block starts exactly once");
        open[index] = true;
        const auto& value = *event.find(api == 1 ? "item" : "content_block");
        if (value.member_str("type") == "function_call" ||
            value.member_str("type") == "tool_use")
          header(index, value);
      } else if (type == "response.output_item.done" ||
                 type == "content_block_stop") {
        Expect(open.contains(index) && open[index],
               "each output block closes exactly once");
        if (api == 1 && tools.contains(index))
          Expect(event.find("item")->member_str("arguments") ==
                     out.arguments[tools.at(index)],
                 "done tool item equals its accumulated deltas");
        if (api == 1 && event.find("item")->member_str("status") == "incomplete")
          ++out.incomplete_items;
        open[index] = false;
      } else if (type == "response.function_call_arguments.delta" ||
                 type == "content_block_delta") {
        Expect(!terminal && open.contains(index) && open[index],
               "deltas belong to an open block");
        const auto* delta = event.find("delta");
        if (api == 1 || delta->member_str("type") == "input_json_delta") {
          Expect(tools.contains(index), "arguments follow their tool header");
          out.arguments[tools.at(index)] +=
              api == 1 ? event.member_str("delta")
                       : delta->member_str("partial_json");
        } else if (delta->member_str("type") == "thinking_delta")
          out.reasoning += delta->member_str("thinking");
        else if (delta->member_str("type") == "text_delta")
          out.content += delta->member_str("text");
      } else if (type == "response.function_call_arguments.done") {
        Expect(
            open.contains(index) && open[index] && tools.contains(index) &&
                event.member_str("arguments") == out.arguments[tools.at(index)],
            "argument done event matches its live tool");
      } else if (type == "response.output_text.delta")
        out.content += event.member_str("delta");
      else if (type == "response.reasoning_summary_text.delta")
        out.reasoning += event.member_str("delta");
      else if (type == "response.completed" || type == "response.incomplete" ||
               type == "message_delta") {
        Expect(!terminal, "one terminal event");
        terminal = true;
        for (const auto& [_, active] : open)
          Expect(!active, "all blocks close before the terminal event");
        out.finish = api == 1 ? event.find("response")->member_str("status")
                              : event.find("delta")->member_str("stop_reason");
      }
    }
  };
  if (stream)
    response.streaming_body([&](std::string_view chunk) {
      if (chunk != "data: [DONE]\n\n")
        collect(gufo::json::parse(chunk.substr(chunk.find("data: ") + 6)));
      return true;
    });
  else
    collect(gufo::json::parse(response.body));
  for (const auto& call : chat_calls) {
    Expect(!call.member_str("id").empty(), "Chat tools have stable identities");
    out.names.push_back(call.find("function")->member_str("name"));
    out.arguments.push_back(call.find("function")->member_str("arguments"));
  }
  Expect(terminal || out.failed, "stream terminates or reports its error");
  return out;
}

// L-REL-8: reasoning ends at the tag the model generated as its tag token. The
// same characters written as ordinary text tokens (a model quoting code that
// contains "</think>") stay inside reasoning, on every route and transport.
void TestTokenLevelReasoningEnd() {
  using Initial = FakeBackend::InitialOutputState;
  struct Case {
    const char* name;
    std::vector<std::string> pieces;
    TokenReasoningEnd end;
    Initial initial;
    std::string reasoning, content;
    std::vector<std::string> names{};
    // 0: with tools (Chat's structured parser), 1: without tools
    // (the unconstrained parser), 2: a JSON response format (Chat only).
    std::vector<int> shapes{0, 1};
  };
  const std::vector<Case> cases = {
      {"tag token, then a tool call",
       {"Plan.", "</think>",
        "\n\n<tool_call>\n<function=noop>\n</function>\n</tool_call>"},
       {true, 1}, Initial::kReasoning, "Plan.", "", {"noop"}, {0}},
      {"quoted tag, then a tool call with no tag token",
       {"Pass \"", "</think>", "\" on.\n\n",
        "<tool_call>\n<function=noop>\n</function>\n</tool_call>"},
       {true, std::nullopt}, Initial::kReasoning, "Pass \"</think>\" on.", "",
       {"noop"}, {0}},
      // The JSON grammar's gate ends at the first textual tag and then masks
      // the tag token, so a response format keeps the text rule (L-REL-8b).
      {"response format keeps the text rule",
       {"Quote ", "</think>", "{\"a\":1}"}, {true, std::nullopt},
       Initial::kReasoning, "Quote", "{\"a\":1}", {}, {2}},
      {"quoted tag, then the tag token",
       {"Pass the literal \"", "</think>", "\" through.", "</think>",
        "\n\nDone."},
       {true, 3}, Initial::kReasoning,
       "Pass the literal \"</think>\" through.", "Done."},
      // An invalid byte decodes to U+FFFD (three bytes), so the tracked offset
      // lands two bytes before the tag in the decoded text.
      {"invalid byte before a quoted tag and the tag token",
       {"Bad \xFF byte, quoted ", "</think>", " text.", "</think>",
        "\n\nDone."},
       {true, 3}, Initial::kReasoning,
       "Bad \xEF\xBF\xBD byte, quoted </think> text.", "Done."},
      {"quoted tag split over pieces",
       {"Write ", "</th", "ink>", " as text.", "</think>", "\n\nDone."},
       {true, 4}, Initial::kReasoning, "Write </think> as text.", "Done."},
      {"tag token only (unchanged)",
       {"Short plan.", "</think>", "\n\nDone."}, {true, 1},
       Initial::kReasoning, "Short plan.", "Done."},
      {"quoted tag, no tag token: reasoning stays open",
       {"Quote ", "</think>", " here."}, {true, std::nullopt},
       Initial::kReasoning, "Quote </think> here.", ""},
      {"model-opened reasoning, quoted tag, then the tag token",
       {"<think>", "Use ", "</think>", " verbatim.", "</think>", "\n\nDone."},
       {true, 4}, Initial::kAuto, "Use </think> verbatim.", "Done."},
      {"untracked backend keeps the text rule",
       {"Quote ", "</think>", " here."}, {false, std::nullopt},
       Initial::kReasoning, "Quote", "here."},
  };
  for (const auto& item : cases)
    for (const int shape : item.shapes)
      for (const int api : {0, 1, 2})
        for (const bool stream : {false, true}) {
          if (shape == 2 && api != 0)
            continue;
          const auto out = ToolBoundaryRequest(
              item.pieces, gufo::sampling::JsonConstraint::ToolFormat::kQwen,
              api, stream, item.initial, FakeBackend::FinishReason::kStop,
              FakeBackend::TerminationCause::kNone, item.end, shape);
          const bool ok = !out.failed &&
                          Trimmed(out.reasoning) == item.reasoning &&
                          Trimmed(out.content) == item.content &&
                          out.names == item.names;
          if (!ok)
            std::cerr << "case=" << item.name << " shape=" << shape
                      << " api=" << api << " stream=" << stream
                      << " reasoning=[" << out.reasoning << "] content=["
                      << out.content << "] calls=" << out.names.size() << "\n";
          Expect(ok, "reasoning ends at the generated tag token");
        }
}

void TestToolTagBoundaries() {
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  using Initial = FakeBackend::InitialOutputState;
  using Finish = FakeBackend::FinishReason;
  const auto bytewise = [](const std::string& raw) {
    std::vector<std::string> pieces;
    for (const char byte : raw)
      pieces.emplace_back(1, byte);
    return pieces;
  };
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const bool qwen = format == Format::kQwen;
    for (const auto initial : {Initial::kReasoning, Initial::kAuto})
      for (const auto finish : {Finish::kLength, Finish::kStopSequence}) {
        const std::string phase = "<think>Check.</think>\n";
        for (const int api : {0, 1, 2})
          for (std::size_t cut = 0; cut <= phase.size(); ++cut) {
            const auto raw = phase.substr(0, cut);
            const auto buffered =
                ToolBoundaryRequest({raw}, format, api, false, initial, finish);
            const auto streamed = ToolBoundaryRequest(
                bytewise(raw), format, api, true, initial, finish);
            Expect(
                !streamed.failed && !buffered.failed &&
                    streamed.arguments.empty() &&
                    streamed.content == buffered.content &&
                    streamed.reasoning == buffered.reasoning &&
                    streamed.reasoning.find("<think>") == std::string::npos &&
                    streamed.reasoning.find("</think>") == std::string::npos,
                "interruption within a thinking tag preserves the admitted "
                "phase without opening a tool");
          }
      }
    for (const std::string frame : {"\n", "\r\n"}) {
      if (!qwen && frame == "\r\n")
        continue;
      const std::string opening =
          qwen ? "<tool_call>\n<function=record>\n<parameter=text>" + frame
               : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
                 "<｜DSML｜parameter name=\"text\" string=\"true\">";
      const std::string closing =
          qwen ? frame + "</parameter>" + frame + "</function>\n</tool_call>"
               : "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</"
                 "｜DSML｜tool_calls>";
      for (const std::string value :
           {"", "\r", "\n", " \t\r\n", "café 🍵 \"quoted\" \\",
            "literal <think></think> <tool_call></tool_call> <|im_end|>",
            "line\n</parameter> is data\n</parameter>\rX\nlast"}) {
        const auto raw = opening + value + closing;
        auto expected = gufo::json::Value::object();
        expected["text"] = value;
        for (const int api : {0, 1, 2}) {
          const auto buffered = ToolBoundaryRequest({raw}, format, api, false);
          if (buffered.failed ||
              buffered.arguments != std::vector<std::string>{expected.dump()} ||
              !buffered.content.empty() || !buffered.reasoning.empty()) {
            std::cerr << "buffered boundary api=" << api << " raw=" << raw
                      << " content=" << buffered.content
                      << " reasoning=" << buffered.reasoning << '\n';
            for (const auto& argument : buffered.arguments)
              std::cerr << "arguments=" << argument << '\n';
          }
          Expect(!buffered.failed &&
                     buffered.arguments ==
                         std::vector<std::string>{expected.dump()} &&
                     buffered.content.empty() && buffered.reasoning.empty(),
                 "buffered fixture retains the independent literal value");
          // Every two-piece split, plus a stream that splits every byte.
          for (std::size_t cut = 0; cut <= raw.size() + 1; ++cut) {
            const auto streamed = ToolBoundaryRequest(
                cut > raw.size() ? bytewise(raw)
                                 : std::vector<std::string>{raw.substr(0, cut),
                                                            raw.substr(cut)},
                format, api, true);
            if (streamed.failed || streamed.arguments != buffered.arguments)
              std::cerr << "tag split format=" << qwen << " api=" << api
                        << " cut=" << cut << " raw=" << raw << '\n';
            Expect(!streamed.failed &&
                       streamed.arguments == buffered.arguments &&
                       streamed.names == buffered.names &&
                       streamed.content.empty() && streamed.reasoning.empty(),
                   "all tag splits preserve values and hide native framing");
          }
        }
      }
      const std::string call = opening + "value" + closing;
      const std::string noop =
          qwen ? "<tool_call>\n<function=noop>\n</function>\n</tool_call>"
               : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"noop\">\n"
                 "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
      for (const auto initial :
           {Initial::kContent, Initial::kReasoning, Initial::kAuto})
        for (const bool implicit : {false, true}) {
          if (implicit && initial != Initial::kReasoning)
            continue;
          const auto prefix =
              initial == Initial::kContent ? std::string{}
              : implicit                   ? std::string("Check tags.\n\n")
                         : (initial == Initial::kAuto ? "<think>" : "") +
                               std::string("Check tags.</think>\n");
          const auto raw = prefix + call + "\n" + noop;
          for (const int api : {0, 1, 2}) {
            const auto buffered =
                ToolBoundaryRequest({raw}, format, api, false, initial);
            for (std::size_t cut = 0; cut <= raw.size(); ++cut) {
              const auto streamed =
                  ToolBoundaryRequest({raw.substr(0, cut), raw.substr(cut)},
                                      format, api, true, initial);
              if (streamed.failed || streamed.arguments != buffered.arguments ||
                  Trimmed(streamed.reasoning) != Trimmed(buffered.reasoning))
                std::cerr << "phase split api=" << api << " cut=" << cut
                          << " initial=" << static_cast<int>(initial)
                          << " implicit=" << implicit << " raw=" << raw
                          << "\nstreamed reasoning=" << streamed.reasoning
                          << "\nbuffered reasoning=" << buffered.reasoning
                          << '\n';
              Expect(
                  !streamed.failed && !buffered.failed &&
                      streamed.arguments == buffered.arguments &&
                      streamed.names ==
                          std::vector<std::string>{"record", "noop"} &&
                      Trimmed(streamed.reasoning) ==
                          Trimmed(buffered.reasoning) &&
                      Trimmed(streamed.reasoning) ==
                          (initial == Initial::kContent ? "" : "Check tags.") &&
                      streamed.content == buffered.content &&
                      Trimmed(streamed.content).empty(),
                  "reasoning boundaries and successive calls agree across "
                  "transports");
            }
          }
        }
      for (const int api : {0, 1, 2})
        for (const bool enabled : {false, true}) {
          const auto raw = "<think>Check tags.</think>\n" + call;
          const auto initial =
              enabled ? Initial::kReasoning : Initial::kContent;
          for (const bool stream : {false, true}) {
            const auto output = ToolBoundaryRequest(bytewise(raw), format, api,
                                                    stream, initial);
            Expect(!output.failed &&
                       Trimmed(output.reasoning) ==
                           (enabled ? "Check tags." : "") &&
                       output.content ==
                           (enabled ? "" : "<think>Check tags.</think>\n") &&
                       output.arguments ==
                           std::vector<std::string>{R"({"text":"value"})"},
                   "an opening think tag follows the admitted initial phase");
          }
        }
      // Stop/length at every byte of both calls, including their opening and
      // closing tags. A partial JSON prefix must never be repaired into a call.
      for (const auto finish : {Finish::kLength, Finish::kStopSequence})
        for (const int api : {0, 1, 2})
          for (std::size_t cut = 0; cut <= call.size(); ++cut) {
            const auto raw =
                "Check.</think>\n" + noop + "\n" + call.substr(0, cut);
            const auto buffered = ToolBoundaryRequest(
                {raw}, format, api, false, Initial::kReasoning, finish);
            const auto streamed = ToolBoundaryRequest(
                bytewise(raw), format, api, true, Initial::kReasoning, finish);
            Expect(!streamed.failed && !buffered.failed &&
                       streamed.arguments.size() >= buffered.arguments.size() &&
                       streamed.arguments.size() <=
                           buffered.arguments.size() + 1 &&
                       streamed.content == buffered.content &&
                       Trimmed(streamed.reasoning) == "Check.",
                   "interrupted tags retain preceding calls and reasoning");
            for (std::size_t i = 0; i < buffered.arguments.size(); ++i)
              Expect(streamed.arguments[i] == buffered.arguments[i],
                     "calls completed before interruption remain unchanged");
            if (streamed.arguments.size() > buffered.arguments.size()) {
              bool invalid = false;
              try {
                (void)gufo::json::parse(streamed.arguments.back());
              } catch (const std::exception&) {
                invalid = true;
              }
              Expect(invalid, "an interrupted prefix is not executable JSON");
            }
          }
    }
  }
}

void TestStreamingQuotedAndMalformedTools() {
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const bool qwen = format == Format::kQwen;
    const std::string opening =
        qwen ? "<tool_call>\n<function=record>\n"
             : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n";
    const auto parameter = [&](const std::string& value) {
      return qwen ? "<parameter=text>\n" + value + "\n</parameter>\n"
                  : "<｜DSML｜parameter name=\"text\" string=\"true\">" +
                        value + "</｜DSML｜parameter>\n";
    };
    const std::string closing =
        qwen ? "</function>\n</tool_call>"
             : "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const auto call = opening + parameter("value") + closing;
    for (const int api : {0, 1, 2}) {
      {
        const std::string data =
            R"({"literal":"</parameter></｜DSML｜parameter>","phase":"<think></think>","nested":"<tool_call><function=noop>"})";
        const auto raw =
            opening +
            (qwen ? "<parameter=data>\n" + data + "\n</parameter>\n"
                  : "<｜DSML｜parameter name=\"data\" string=\"false\">" +
                        data + "</｜DSML｜parameter>\n") +
            closing;
        std::vector<std::string> pieces;
        for (const char byte : raw)
          pieces.emplace_back(1, byte);
        const auto output = ToolBoundaryRequest(pieces, format, api, true);
        auto expected = gufo::json::Value::object();
        expected["data"] = gufo::json::parse(data);
        Expect(
            !output.failed &&
                output.arguments == std::vector<std::string>{expected.dump()} &&
                output.content.empty() && output.reasoning.empty(),
            "JSON string delimiters inside typed parameters remain argument "
            "data");
      }
      {
        const auto raw = opening + parameter("unmatched ` in data") + closing +
                         "\n" + opening + parameter("later ` in data") +
                         closing;
        std::vector<std::string> pieces;
        for (const char byte : raw)
          pieces.emplace_back(1, byte);
        for (const bool stream : {false, true}) {
          const auto output = ToolBoundaryRequest(pieces, format, api, stream);
          Expect(
              !output.failed && Trimmed(output.content).empty() &&
                  output.reasoning.empty() &&
                  output.arguments ==
                      std::vector<std::string>{
                          R"({"text":"unmatched ` in data"})",
                          R"({"text":"later ` in data"})"},
              "backticks inside arguments cannot quote a subsequent tool call");
        }
      }
      for (const auto& quote : {std::string("`"), std::string("```\n")}) {
        const auto example = "Example: " + quote + call + quote + "\n";
        for (const bool reasoning : {false, true}) {
          const auto raw = example + (reasoning ? "</think>\n" : "") + call;
          std::vector<std::string> pieces;
          for (const char byte : raw)
            pieces.emplace_back(1, byte);
          const auto initial = reasoning
                                   ? FakeBackend::InitialOutputState::kReasoning
                                   : FakeBackend::InitialOutputState::kContent;
          const auto buffered =
              ToolBoundaryRequest({raw}, format, api, false, initial);
          const auto streamed =
              ToolBoundaryRequest(pieces, format, api, true, initial);
          Expect(!streamed.failed && !buffered.failed &&
                     streamed.names == std::vector<std::string>{"record"} &&
                     streamed.arguments ==
                         std::vector<std::string>{R"({"text":"value"})"} &&
                     streamed.content == buffered.content &&
                     Trimmed(streamed.reasoning) == Trimmed(buffered.reasoning),
                 "closing quotes never turn a streamed example into a phantom "
                 "call");
        }
      }
      // An EMPTY value shares the frame's newline with the closing delimiter
      // ("<parameter=text>\n</parameter>\n"). A partial parse must hold back
      // the delimiter there too: streaming "</", "</parameter>" as argument
      // data (production, 10-21 Stage 1b, args {"command":"</parameter>) ends
      // the stream with "streamed tool call did not complete safely".
      if (qwen) {
        const auto empty_call =
            opening + "<parameter=text>\n</parameter>\n" + closing;
        std::vector<std::string> bytes;
        for (const char byte : empty_call)
          bytes.emplace_back(1, byte);
        const auto cut =
            opening.size() +
            std::string("<parameter=text>\n</parameter>").size();
        const std::vector<std::string> split = {empty_call.substr(0, cut),
                                                empty_call.substr(cut)};
        for (const auto& pieces : {bytes, split}) {
          const auto streamed = ToolBoundaryRequest(pieces, format, api, true);
          Expect(!streamed.failed &&
                     streamed.arguments ==
                         std::vector<std::string>{R"({"text":""})"} &&
                     streamed.content.empty(),
                 "an empty parameter value never streams its closing tag as "
                 "argument data");
        }
      }
      // Keep existing duplicate-parameter semantics. Once conflicting bytes
      // have been streamed, report failure instead of completing an unsafe
      // call.
      for (const bool conflicting : {false, true}) {
        const auto raw = opening + parameter("value") +
                         parameter(conflicting ? "changed" : "value") + closing;
        std::vector<std::string> pieces;
        for (const char byte : raw)
          pieces.emplace_back(1, byte);
        const auto streamed = ToolBoundaryRequest(pieces, format, api, true);
        if (!conflicting) {
          Expect(!streamed.failed &&
                     streamed.arguments ==
                         std::vector<std::string>{R"({"text":"value"})"},
                 "identical duplicate parameters do not duplicate argument "
                 "deltas");
        } else {
          Expect(streamed.failed && streamed.finish.empty() &&
                     !streamed.arguments.empty(),
                 "conflicting parameters fail without a successful terminal "
                 "event");
          bool invalid = false;
          try {
            (void)gufo::json::parse(streamed.arguments.back());
          } catch (const std::exception&) {
            invalid = true;
          }
          Expect(
              invalid,
              "conflicting tool data is never completed into executable JSON");
        }
      }
    }
  }
}

// A tool-argument loop stop (fix (b)) ends the request while a call is open.
// Arguments stream before the stop is known. When part of the cut call's
// arguments is already on the wire, the client cannot take it back, so every
// streamed route ends exactly like an ordinary EOS inside a started call: an
// error with a fixed message and no stop, completed, end_turn or tool_use
// terminal event. Without such a delta, and without streaming, the request
// ends with the normal stop and the cut call is dropped.
void TestToolArgumentLoopStopClosesStreamedCall() {
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  using Cause = FakeBackend::TerminationCause;
  using Finish = FakeBackend::FinishReason;
  const std::string open = "<tool_call>\n<function=record>\n<parameter=text>\n";
  const std::string complete =
      open + "first\n</parameter>\n</function>\n</tool_call>\n";
  const std::string period = "</function>\n</tool_call>\n" + open;
  for (const bool preceding : {false, true}) {
    std::string raw = preceding ? complete : std::string();
    raw += open;
    for (int repeat = 0; repeat < 16; ++repeat)
      raw += period;
    std::vector<std::string> pieces;
    for (const char byte : raw)
      pieces.emplace_back(1, byte);
    const std::size_t complete_calls = preceding ? 1 : 0;
    for (const int api : {0, 1, 2}) {
      // (a) Streamed, with deltas of the cut call already sent: the error
      // ending, the fixed message, and no terminal event of a normal stop.
      const auto stopped = ToolBoundaryRequest(
          pieces, Format::kQwen, api, true,
          FakeBackend::InitialOutputState::kContent, Finish::kStop,
          Cause::kToolArgumentLoop);
      Expect(stopped.failed && stopped.finish.empty(),
             "a loop stop inside streamed arguments ends with an error and no stop");
      Expect(stopped.error_message ==
                 "generation stopped: the tool call repeated its arguments (loop breaker)",
             "the loop stop error carries the fixed message");
      Expect(stopped.incomplete_items == 0,
             "no item is closed as incomplete by the error ending");
      Expect(stopped.names.size() >= complete_calls &&
                 stopped.arguments.size() == stopped.names.size() &&
                 stopped.names.size() <= complete_calls + 1,
             "only the complete call and the cut call were streamed");
      if (preceding)
        Expect(stopped.arguments.front() == R"({"text":"first"})",
               "the call before the loop stays complete as streamed");
      if (stopped.arguments.size() > complete_calls) {
        Expect(stopped.arguments.back().starts_with(R"({"text":")"),
               "the cut call streamed its argument prefix");
        bool executable = true;
        try {
          (void)gufo::json::parse(stopped.arguments.back());
        } catch (const std::exception&) {
          executable = false;
        }
        Expect(!executable, "the call cut by the loop stop is not executable JSON");
      }

      // (c) Without streaming the cut call is dropped and the turn stops.
      const std::string expected = api == 0   ? "stop"
                                   : api == 1 ? "completed"
                                   : preceding ? "tool_use"
                                               : "end_turn";
      const auto buffered = ToolBoundaryRequest(
          {raw}, Format::kQwen, api, false,
          FakeBackend::InitialOutputState::kContent, Finish::kStop,
          Cause::kToolArgumentLoop);
      Expect(!buffered.failed && buffered.finish == expected,
             "a buffered loop stop ends with a normal stop on every route");
      Expect(buffered.names ==
                     std::vector<std::string>(complete_calls, "record") &&
                 buffered.arguments.size() == complete_calls,
             "the buffered loop stop drops the cut call and keeps complete calls");
      Expect(Trimmed(buffered.content).empty() && buffered.reasoning.empty(),
             "tool markup of the cut call does not leak into buffered content");

      // (b) Streamed, with no partial call on the wire: the normal endings.
      // Complete calls only, then text only.
      const std::string only_complete = preceding ? complete : "Hello.";
      std::vector<std::string> clean;
      for (const char byte : only_complete)
        clean.emplace_back(1, byte);
      const auto clean_stop = ToolBoundaryRequest(
          clean, Format::kQwen, api, true,
          FakeBackend::InitialOutputState::kContent, Finish::kStop,
          Cause::kToolArgumentLoop);
      Expect(!clean_stop.failed,
             "a loop stop without a partial call on the wire is not an error");
      Expect(clean_stop.finish == expected,
             "a loop stop without a partial call ends with the normal stop");
      Expect(clean_stop.incomplete_items == 0,
             "no item is incomplete when no partial call was sent");
      if (preceding)
        Expect(clean_stop.names == std::vector<std::string>{"record"} &&
                   clean_stop.arguments.front() == R"({"text":"first"})",
               "the complete call is delivered before the normal stop");
      else
        Expect(Trimmed(clean_stop.content) == "Hello." &&
                   clean_stop.names.empty(),
               "plain text is delivered before the normal stop");

      const auto ordinary = ToolBoundaryRequest(
          pieces, Format::kQwen, api, true,
          FakeBackend::InitialOutputState::kContent, Finish::kStop,
          Cause::kNone);
      Expect(ordinary.failed && ordinary.finish.empty(),
             "ordinary EOS inside a streamed call is not reported as a stop");
    }
  }
}

void TestIncrementalToolArguments() {
  using gufo::json::Value;
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  const std::string value =
      "first line\nliteral </parameter> is data\n"
      "<tool_call> and <|im_end|> stay literal; \"quoted\" \\ café";
  for (const auto format : {Format::kQwen, Format::kDeepSeek})
    for (const int api : {0, 1, 2})
      for (const bool reasoning : {false, true}) {
        FakeBackend backend;
        backend.tool_format = format;
        backend.block_after_first_piece = true;
        const bool qwen = format == Format::kQwen;
        const std::string opening =
            qwen ? "<tool_call>\n<function=record>\n<parameter=text>\n"
                 : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
                   "<｜DSML｜parameter name=\"text\" string=\"true\">";
        const std::string closing =
            qwen ? "\n</parameter>\n</function>\n</tool_call>"
                 : "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
                   "</｜DSML｜tool_calls>";
        backend.pieces = {(reasoning ? "Check briefly.</think>\n" : "") +
                          opening + value.substr(0, 10)};
        // Exercise every byte boundary, including UTF-8 and escaped JSON.
        for (const char byte : value.substr(10) + closing)
          backend.pieces.emplace_back(1, byte);
        auto body = gufo::json::parse(R"({
          "model":"test-model","stream":true,
          "messages":[{"role":"user","content":"record the literal text"}],
          "tools":[{"type":"function","function":{"name":"record",
            "parameters":{"type":"object","properties":{"text":{"type":"string"}},
              "required":["text"],"additionalProperties":false}}}],
          "tool_choice":"required"})");
        if (reasoning)
          body["reasoning_effort"] = "low";
        gufo::server::HttpResponse response;
        if (api == 0) {
          response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        } else {
          gufo::server::ChatRequest chat;
          Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                 "stream fixture controls parse");
          chat.reasoning.enabled = reasoning;
          response = api == 1
                         ? gufo::server::CreateOpenAiResponse(
                               Request("{}"), backend, chat, 512, {}, true)
                         : gufo::server::CreateAnthropicMessage(
                               Request("{}"), backend, chat, 512, {}, true);
        }
        std::mutex mutex;
        std::condition_variable condition;
        std::string arguments, name, id, content;
        std::size_t fragments = 0, headers = 0;
        bool failed = false;
        std::jthread writer([&] {
          response.streaming_body([&](std::string_view chunk) {
            if (chunk == "data: [DONE]\n\n")
              return true;
            const auto event =
                gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
            const std::lock_guard lock(mutex);
            failed |= event.contains("error") ||
                      event.member_str("type") == "response.failed";
            const auto append = [&](std::string piece) {
              if (!piece.empty()) {
                arguments += piece;
                ++fragments;
              }
            };
            const auto header = [&](const Value& item) {
              name = item.member_str("name");
              id = item.member_str("call_id", item.member_str("id"));
              ++headers;
            };
            if (api == 0) {
              if (const auto* choices = event.find("choices"))
                for (const auto& choice : choices->items()) {
                  const auto* delta = choice.find("delta");
                  content += delta->member_str("content");
                  if (const auto* calls = delta->find("tool_calls"))
                    for (const auto& call : calls->items()) {
                      Expect(call.member_size("index") == 0,
                             "argument deltas keep their tool index");
                      if (!call.member_str("id").empty()) {
                        id = call.member_str("id");
                        name = call.find("function")->member_str("name");
                        ++headers;
                      }
                      append(call.find("function")->member_str("arguments"));
                    }
                }
            } else if (api == 1) {
              if (event.member_str("type") == "response.output_item.added" &&
                  event.find("item")->member_str("type") == "function_call")
                header(*event.find("item"));
              if (event.member_str("type") ==
                  "response.function_call_arguments.delta")
                append(event.member_str("delta"));
              if (event.member_str("type") == "response.output_text.delta")
                content += event.member_str("delta");
            } else {
              if (event.member_str("type") == "content_block_start" &&
                  event.find("content_block")->member_str("type") == "tool_use")
                header(*event.find("content_block"));
              if (const auto* delta = event.find("delta")) {
                if (delta->member_str("type") == "input_json_delta")
                  append(delta->member_str("partial_json"));
                if (delta->member_str("type") == "text_delta")
                  content += delta->member_str("text");
              }
            }
            condition.notify_all();
            return true;
          });
        });
        Expect(backend.WaitForFirstPiece(), "model reaches the partial call");
        bool live = false;
        {
          std::unique_lock lock(mutex);
          live = condition.wait_for(lock, 2s, [&] {
            return arguments.find("first line") != std::string::npos;
          });
          Expect(!backend.completed, "generation is still blocked");
        }
        backend.Release();
        writer.join();
        Expect(live, "tool argument bytes arrive before generation completes");
        Expect(!failed, "fragmented tool stream completes without an error");
        Expect(headers == 1 && !id.empty() && name == "record",
               "one stable tool header is emitted");
        Expect(fragments > 2, "arguments arrive incrementally");
        Expect(gufo::json::parse(arguments).member_str("text") == value,
               "fragment assembly preserves literal argument bytes");
        Expect(content.empty(), "tool markup does not leak into content");
        // Disconnect while the same backend is still producing arguments.
        // The compatibility serializers must propagate a failed writer too.
        backend.completed = false;
        backend.block_after_first_piece = false;
        gufo::server::ChatRequest cancelled_chat;
        Expect(
            !gufo::server::ParseOpenAiResponseControls(body, &cancelled_chat),
            "cancelled tool controls parse");
        cancelled_chat.reasoning.enabled = reasoning;
        auto cancelled_response =
            api == 0
                ? gufo::server::HandleOpenAiChat(Request(body.dump()), backend)
            : api == 1
                ? gufo::server::CreateOpenAiResponse(
                      Request("{}"), backend, cancelled_chat, 512, {}, true)
                : gufo::server::CreateAnthropicMessage(
                      Request("{}"), backend, cancelled_chat, 512, {}, true);
        bool disconnected = false;
        cancelled_response.streaming_body([&](std::string_view chunk) {
          if (chunk == "data: [DONE]\n\n")
            return true;
          const auto event =
              gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
          if (const auto* choices = event.find("choices"))
            for (const auto& choice : choices->items())
              disconnected |= choice.find("delta")->contains("tool_calls");
          disconnected |= event.member_str("type") ==
                          "response.function_call_arguments.delta";
          if (const auto* delta = event.find("delta");
              delta && delta->is_object())
            disconnected |= delta->member_str("type") == "input_json_delta";
          return !disconnected;
        });
        Expect(disconnected && !backend.completed,
               "disconnecting a partial tool cancels backend generation");
      }
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--tool-loop-stop") {
    TestToolArgumentLoopStopOutput();
    TestToolArgumentLoopStopClosesStreamedCall();
    std::cout << "Tool-loop API tests passed\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--debug-hooks") {
    DebugHooksRawOutputTest();
    DebugHooksSettingsTest();
    DebugHooksSettingsEnvironmentTest();
    return 0;
  }

  DebugHooksRawOutputTest();
  DebugHooksSettingsTest();
  DebugHooksSettingsEnvironmentTest();
  TestStderrGateIsLineIndependent();
  TestToolCallRenderParseRoundTrip();
  TestToolTagBoundaries();
  TestTokenLevelReasoningEnd();
  TestStreamingQuotedAndMalformedTools();
  TestIncrementalToolArguments();
  TestHistoricalTypedArgumentsUseTojson();
  TestHistoricalNumberArgumentsKeepSourceText();
  TestBehaviourRequestOptions();
  TestNullReasoningHistory();
  TestProductionSamplingSelection();
  TestProductionParserRecovery();
  TestStreamingPromptProgress();
  TestResponsesPromptProgress();
  TestToolMarkersInsideConstrainedReasoning();
  TestNativeToolImplicitReasoningEnd();
  TestToolMarkersWhenToolsDisabled();
  TestBracketDenseContent();
  TestStopSequencesAndDefaultFields();
  TestStructuredResponseFormat();
  TestStructuredToolTruncation();
  TestStrictToolSchema();
  TestExplicitStopOutputFraming();
  TestStopInsideToolArguments();
  TestResponsesOutput();
  TestNativeToolTransports();
  TestJsonToolStringOwnership();
  TestNativeToolDialectSelection();
  TestNativeToolTextOutsideEnvelopes();
  TestNativeReferencedArgumentTypes();
  TestWildcardToolTypes();
  TestNativeUnionToolTypes();
  TestNativeArgumentTypingMatrix();
  TestToolMetadataAndFraming();
  TestGemmaNativeToolCalls();
  TestResponsesLiveAndCancellation();
  TestCachePromptOption();
  TestToolChoiceEnforcement();
  TestPrefillDiagnosticUsage();
  TestReasoningTokensInUsage();
  TestStreamingIsLive();
  TestLessThanProseStreamsBeforeCompletion();
  TestStreamingWithoutUsage();
  TestUtf8Output();
  TestCachedPrefillMetrics();
  TestBackendSamplingDefaults();
  TestModelSamplingDefaults();
  TestCompleteToolDefinitionsReachTemplate();
  TestFlatToolFieldsReachTemplate();
  TestAllSamplingControlsReachBackend();
  TestUnsupportedSamplingControlsAreRejected();
  TestSamplingRanges();
  TestAssistantReasoningContentReachesBackend();
  TestPiReasoningControlsAndOutputFraming();
  TestPiNativeDeepSeekThinkingObject();
  TestStreamingPromptOpenedReasoning();
  TestInitialOutputPhases();
  TestReasoningAnswerSeparator();
  TestConflictingReasoningControlsAreRejected();
  TestToolCallsAreStructured();
  TestToolParameterCompatibility();
  TestToolArgumentLoopStopOutput();
  TestToolArgumentLoopStopClosesStreamedCall();
  TestInvalidToolsFailBeforeGeneration();
  TestResponsesClientCompatTolerances();
  TestMidConversationSystemMessagesKeepOrder();
  TestToolNameCharacters();
  TestMalformedHistoricalFunctions();
  TestToolClosingFraming();
  TestUnconstrainedQuoteFallbackChecksSchema();
  TestUnfinishedInlineReasoningFallback();
  TestProseAboutTheDialectIsNotACall();
  TestQwenToolBoundariesAndSchema();
  TestDeepSeekToolCallsAreStructured();
  TestDeepSeekRepeatedToolParameters();
  TestWrongModelIsRejected();
  TestClientIdentityReachesBackend();
  TestStreamingOverloadIsRejectedBeforeHeaders();
  TestImagePartsRetainOrderAndIdentity();
  TestTextOnlyBackendRefusesImagesBeforeReading();
  TestImageCountAndByteBudget();
  TestToolImagesRetainOrderAndIdentity();
  std::cout << "All OpenAI chat protocol tests passed\n";
  return 0;
}
