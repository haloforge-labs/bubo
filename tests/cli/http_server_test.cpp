#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/debug_request.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/prompt_scoring.hpp"
#include "src/cli/serve/trace.hpp"
#include "src/core/platform/memory_guard.hpp"
#include "src/core/platform/socket.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::Logger;
using gufo::server::LogLevel;
using gufo::server::LogLevelFromName;
using gufo::server::LogLevelName;
using gufo::server::TextGenerationBackend;

/// Reports an optional admission and fixed prompt progress before delegating
/// token generation.
class ProgressRequest final : public TextGenerationBackend::GenerationRequest {
public:
  ProgressRequest(std::shared_ptr<GenerationRequest> inner,
                  std::vector<TextGenerationBackend::PromptProgress> progress,
                  bool admit = false)
      : inner_(std::move(inner)),
        progress_(std::move(progress)),
        admit_(admit) {}

  TextGenerationBackend::Result Wait(
      const TextGenerationBackend::TokenCallback& on_token,
      const TextGenerationBackend::ProgressCallback& on_progress,
      const TextGenerationBackend::StartCallback& on_start) override {
    if (admit_ && on_start && !on_start())
      inner_->Cancel();
    for (const auto& value : progress_) {
      if (on_progress && !on_progress(value)) {
        inner_->Cancel();
        break;
      }
    }
    return inner_->Wait(on_token, on_progress, on_start);
  }
  void Cancel() noexcept override { inner_->Cancel(); }

private:
  std::shared_ptr<GenerationRequest> inner_;
  std::vector<TextGenerationBackend::PromptProgress> progress_;
  bool admit_;
};

class DebugHooksScoreRequest final
    : public TextGenerationBackend::GenerationRequest {
public:
  explicit DebugHooksScoreRequest(TextGenerationBackend::Result result)
      : result_(std::move(result)) {}
  TextGenerationBackend::Result Wait(
      const TextGenerationBackend::TokenCallback&,
      const TextGenerationBackend::ProgressCallback&,
      const TextGenerationBackend::StartCallback& started) override {
    if (started && !started())
      cancelled_ = true;
    result_.cancelled = cancelled_;
    return result_;
  }
  void Cancel() noexcept override { cancelled_ = true; }

private:
  TextGenerationBackend::Result result_;
  std::atomic<bool> cancelled_{false};
};

class LiveScoredFakeRequest final
    : public TextGenerationBackend::GenerationRequest {
public:
  using Run = std::function<TextGenerationBackend::Result(
      const TextGenerationBackend::ScoredTokenCallback&)>;
  explicit LiveScoredFakeRequest(Run run) : run_(std::move(run)) {}
  TextGenerationBackend::Result Wait(
      const TextGenerationBackend::TokenCallback&,
      const TextGenerationBackend::ProgressCallback&,
      const TextGenerationBackend::StartCallback&) override {
    return run_({});
  }
  TextGenerationBackend::Result WaitScored(
      const TextGenerationBackend::ScoredTokenCallback& callback) override {
    return run_(
        [&](std::string_view text, const gufo::sampling::TokenLogprob& score) {
          return !cancelled_ && callback(text, score);
        });
  }
  void Cancel() noexcept override { cancelled_ = true; }

private:
  Run run_;
  bool cancelled_{false};
};

class FakeBackend final : public TextGenerationBackend {
public:
  struct Call {
    std::string prompt;
    gufo::server::ChatRequest chat;
    std::size_t max_tokens = 0;
    gufo::sampling::SamplingConfig sampling;
    std::string client_id;
    std::vector<std::string> stop_sequences;
    std::string trace_request;
  };
  Call LastCall() {
    const std::lock_guard lock(mutex_);
    return last_;
  }
  void SetOutput(std::string text) {
    const std::lock_guard lock(mutex_);
    output_ = std::move(text);
  }
  std::string served_name = "test";
  std::string model_id() const override { return served_name; }
  bool ready() const override { return true; }
  bool device_lost() const override { return lost.load(); }
  bool supports_images() const override { return image_support.load(); }
  bool supports_logprobs() const override { return logprob_support.load(); }
  bool supports_diffusion_reads() const override { return diffusion_support; }
  std::size_t diffusion_canvas_length() const override { return 8; }
  std::size_t vocab_size() const override { return 1024; }
  std::string token_piece(gufo::tokenization::TokenId token) const override {
    if (token == 0)
      return "\xC3\xA9";
    if (token == 1)
      return "x";
    throw std::invalid_argument("unknown synthetic token");
  }
  std::shared_ptr<GenerationRequest> start_score(
      std::string_view prompt, const gufo::sampling::SamplingConfig& config,
      std::size_t max_score_tokens,
      const CancellationCheck& cancellation) override {
    if (!logprob_support)
      throw std::invalid_argument("prompt logprobs are not supported for this model");
    gufo::server::CheckPromptScoreLimits(prompt.size(), max_score_tokens);
    if (score_bad_alloc)
      gufo::server::GuardPromptScoringStage("session creation", 0,
                                            [] { throw std::bad_alloc(); });
    ++score_calls;
    Result result;
    result.text = std::string(prompt);
    result.prompt_tokens = prompt.size();
    result.tokens.assign(prompt.size(), 1);
    result.prompt_logprobs.resize(prompt.size());
    result.finish_reason = FinishReason::kLength;
    gufo::sampling::SamplerState sampler(config);
    const std::array<float, 2> row{2, 1};
    for (std::size_t i = 1; i < prompt.size(); ++i) {
      result.prompt_logprobs[i] = sampler.RawLogprobs(
          row, 1, static_cast<std::size_t>(config.logprobs));
      sampler.Accept(1);
    }
    if (cancellation && cancellation())
      result.cancelled = true;
    return std::make_shared<DebugHooksScoreRequest>(std::move(result));
  }
  gufo::json::Value debug_prompt(const gufo::server::ChatRequest* chat,
                                 std::string_view text) const override {
    if (chat)
      text = chat->messages.front().content;
    auto response = gufo::json::Value::object();
    response["text"] = std::string(text);
    response["token_ids"] = gufo::json::Value::array();
    for (unsigned char c : text)
      response["token_ids"].push_back(static_cast<std::size_t>(c));
    return response;
  }
  std::atomic<bool> image_support{false};
  std::atomic<bool> logprob_support{false};
  std::atomic<bool> diffusion_support{false};
  std::atomic<bool> synthetic_eos{false};
  bool short_scores{false};
  std::atomic<int> score_calls{0};
  std::atomic<bool> score_bad_alloc{false};
  std::uint32_t max_context() const override { return 65536; }
  std::vector<SessionState> session_states() const override { return sessions; }
  std::vector<SessionState> sessions;
  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancellation, bool stream, bool ignore_eos,
      std::string_view client_id,
      const std::vector<std::string>& stop_sequences, bool return_progress,
      bool debug_timeline = false, bool cache_prompt = true) override {
    if (debug_timeline)
      throw std::invalid_argument("timeline is not supported for this backend");
    if (sampling.logprobs >= 0) {
      return std::make_shared<LiveScoredFakeRequest>(
          [this, prompt = std::string(prompt), max_tokens, sampling,
           cancellation, client = std::string(client_id), stop_sequences,
           cache_prompt](const ScoredTokenCallback& score) {
            return CompleteImpl(prompt, max_tokens, sampling, cancellation, {},
                                client, stop_sequences, score, cache_prompt);
          });
    }
    last_ignore_eos = ignore_eos;
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_complete(
            prompt, max_tokens, sampling, cancellation, stream, false,
            client_id, stop_sequences, return_progress, debug_timeline,
            cache_prompt),
        progress, admit.load());
  }
  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancellation, bool stream) override {
    if (request.diffusion_read && request.diffusion_read->read_only) {
      {
        const std::lock_guard lock(mutex_);
        last_.chat = request;
        last_.max_tokens = max_tokens;
      }
      return std::make_shared<DebugHooksScoreRequest>(ScriptedRead(
          request.diffusion_read->canvas_length, 7, synthetic_eos.load()));
    }
    if (sampling.logprobs >= 0) {
      return std::make_shared<LiveScoredFakeRequest>(
          [this, max_tokens, sampling, cancellation,
           cache_prompt =
               request.cache_prompt](const ScoredTokenCallback& score) {
            return CompleteImpl("x", max_tokens, sampling, cancellation, {},
                                "anonymous", {}, score, cache_prompt);
          });
    }
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                          cancellation, stream),
        std::vector<PromptProgress>{}, admit.load());
  }
  std::shared_ptr<GenerationRequest> start_complete_tokens(
      const gufo::server::TokenCompletionRequest& request,
      std::size_t max_tokens, const gufo::sampling::SamplingConfig&,
      const CancellationCheck&, bool) override {
    {
      const std::lock_guard lock(mutex_);
      last_token_request_ = request;
      last_token_max_tokens_ = max_tokens;
    }
    if (request.diffusion_read && request.diffusion_read->read_only)
      return std::make_shared<DebugHooksScoreRequest>(
          ScriptedRead(request.diffusion_read->canvas_length,
                       request.prompt.size(), synthetic_eos.load()));
    Result result;
    result.prompt_tokens = request.prompt.size();
    result.tokens = {11, 12};
    result.logprobs = {{11, -0.1, {}}, {12, -0.2, {}}};
    result.completion_tokens = 2;
    result.text = "two tokens";
    return std::make_shared<DebugHooksScoreRequest>(std::move(result));
  }
  std::pair<gufo::server::TokenCompletionRequest, std::size_t>
  LastTokenRequest() {
    const std::lock_guard lock(mutex_);
    return {last_token_request_, last_token_max_tokens_};
  }
  static Result ScriptedRead(std::size_t width, std::size_t prompt_tokens,
                             bool eos = false) {
    Result result;
    result.prompt_tokens = prompt_tokens;
    result.diffusion_rows.emplace();
    for (std::size_t pos = 0; pos < width; ++pos) {
      gufo::server::DiffusionReadRow row;
      row.token =
          static_cast<gufo::sampling::TokenId>(eos && pos == 0 ? 0 : 10 + pos);
      row.logprob = -0.1;
      for (std::size_t rank = 0; rank < 20; ++rank)
        row.top_logprobs.push_back(
            {static_cast<gufo::sampling::TokenId>(
                 eos && pos == 0 && rank == 0 ? 0 : 10 + pos + rank),
             -0.1 - static_cast<double>(rank) / 10});
      row.label_logprobs.push_back({900, -9.0 - static_cast<double>(pos)});
      result.diffusion_rows->push_back(std::move(row));
    }
    return result;
  }
  SamplingDefaults sampling_defaults() const override { return defaults; }
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning;
  }
  InitialOutputState initial_output_state(
      const gufo::server::ChatRequest& request) const override {
    return initial_output_state_override.value_or(
        TextGenerationBackend::initial_output_state(request));
  }
  std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  std::optional<std::size_t> count_chat_tokens(
      const gufo::server::ChatRequest& request) const override {
    counted_messages = request.messages.size();
    counted_tools = request.tools.size();
    return chat_token_count;
  }
  std::optional<std::size_t> chat_token_count;
  mutable std::size_t counted_messages{0};
  mutable std::size_t counted_tools{0};
  Result complete(std::string_view prompt, std::size_t limit,
                  const gufo::sampling::SamplingConfig& sampling,
                  const CancellationCheck& cancel, const TokenCallback& token,
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true) override {
    return CompleteImpl(prompt, limit, sampling, cancel, token, client_id,
                        stop_sequences, {}, cache_prompt);
  }
  Result CompleteImpl(std::string_view prompt, std::size_t limit,
                      const gufo::sampling::SamplingConfig& sampling,
                      const CancellationCheck& cancel,
                      const TokenCallback& token, std::string_view client_id,
                      const std::vector<std::string>& stop_sequences,
                      const ScoredTokenCallback& on_score,
                      bool cache_prompt = true) {
    ++calls;
    last_cache_prompt = cache_prompt;
    if (failure == 20)
      throw gufo::server::TextGenerationError(gufo::server::TextGenerationErrorCode::kInsufficientMemory, "not enough free memory",
                                              gufo::server::MemoryRefusal{std::uint64_t{2} << 30, std::uint64_t{1} << 30});
    if (failure == 21)
      throw gufo::server::TextGenerationError(gufo::server::TextGenerationErrorCode::kInsufficientMemoryPermanent, "cannot ever fit",
                                              gufo::server::MemoryRefusal{std::uint64_t{200} << 30, std::uint64_t{1} << 30});
    if (failure == 22)
      throw gufo::server::TextGenerationError(gufo::server::TextGenerationErrorCode::kMemoryPressure, "stopped on low memory");
    if (failure == 1)
      throw std::length_error("context exceeded");
    if (failure == 2)
      throw std::invalid_argument("invalid prompt");
    // A failed probe: the scheduler reports device loss from then on.
    if (failure == 3) {
      lost = true;
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kDeviceLost,
          gufo::server::kDeviceLostMessage);
    }
    // A failure on a device that still accepts work.
    if (failure == 4)
      throw std::runtime_error("unspecified launch failure");
    if (failure == 5)
      throw std::runtime_error("");
    Result result;
    if (wait_for_disconnect) {
      entered.release();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!(disconnected = cancel && cancel()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished.release();
      result.cancelled = disconnected;
      return result;
    }
    {
      const std::lock_guard lock(mutex_);
      last_ = {.prompt = std::string(prompt),
               .max_tokens = limit,
               .sampling = sampling,
               .client_id = std::string(client_id),
               .stop_sequences = stop_sequences,
               .trace_request = gufo::server::Trace::CurrentRequest()};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_rounds = 2;
    result.draft_tokens = 8;
    result.prefill_tokens = 2;
    result.prefill_ms = 4;
    result.completion_tokens = 1;
    result.decode_ms = 2;
    result.finish_reason =
        limit == 1 ? FinishReason::kLength : FinishReason::kStop;
    if (logprob_support) {
      gufo::sampling::SamplerState sampler(sampling);
      const std::array<std::array<float, 2>, 2> rows{{{3, 1}, {-1, 3}}};
      result.text.clear();
      for (const auto& row : rows) {
        auto scoring = sampler;
        const auto selected = sampler.Sample(row);
        result.tokens.push_back(selected);
        result.text += token_piece(selected);
        if (sampling.logprobs >= 0)
          result.logprobs.push_back(scoring.RawLogprobs(
              row, selected, static_cast<std::size_t>(sampling.logprobs)));
        if (on_score && !result.logprobs.empty() &&
            !on_score(token_piece(selected), result.logprobs.back())) {
          result.cancelled = true;
          return result;
        }
        sampler.Accept(selected);
      }
      result.completion_tokens = result.tokens.size();
    }
    if (short_scores && !result.logprobs.empty())
      result.logprobs.pop_back();
    if (!forced_stop_sequence.empty()) {
      result.finish_reason = FinishReason::kStopSequence;
      result.stop_sequence = forced_stop_sequence;
    }
    if (token && failure != 8) {
      std::this_thread::sleep_for(token_delay);
      (void)token(result.text);
    }
    if (failure == 6)
      throw std::runtime_error("context exceeded");
    if (failure == 7) {
      lost = true;
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kDeviceLost,
          gufo::server::kDeviceLostMessage);
    }
    return result;
  }
  Result chat(const gufo::server::ChatRequest& request, std::size_t limit,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancel,
              const TokenCallback& token) override {
    auto result =
        complete("", limit, sampling, cancel, token, request.client_id,
                 request.stop_sequences, request.cache_prompt);
    {
      const std::lock_guard lock(mutex_);
      last_.chat = request;
    }
    return result;
  }
  std::atomic<int> calls{0};
  std::atomic<bool> last_ignore_eos{false};
  std::atomic<bool> last_cache_prompt{true};
  std::vector<PromptProgress> progress;
  // Report a scheduler admission before generation, as the HIP backend does.
  std::atomic<bool> admit{false};
  std::chrono::milliseconds token_delay{0};
  std::atomic<int> failure{0};
  std::atomic<bool> lost{false};
  std::string forced_stop_sequence;
  gufo::ReasoningOptions reasoning;
  std::optional<InitialOutputState> initial_output_state_override;
  bool wait_for_disconnect{false};
  std::atomic<bool> disconnected{false};
  std::binary_semaphore entered{0};
  std::binary_semaphore finished{0};

private:
  std::mutex mutex_;
  Call last_;
  gufo::server::TokenCompletionRequest last_token_request_;
  std::size_t last_token_max_tokens_{0};
  std::string output_{"ok"};
};

class RunningServer {
public:
  explicit RunningServer(gufo::server::HttpServerOptions options = {},
                         bool handle_signals = false,
                         std::string host = "127.0.0.1")
      : backend(std::make_shared<FakeBackend>()),
        server(host, 0, backend, nullptr, nullptr, nullptr, std::move(options)),
        ipv6_(host.find(':') != std::string::npos) {
    server.add("POST", "/echo", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.body};
    });
    server.add("POST", "/stream-error", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .streaming_body = [](const auto& write) {
            (void)write("first chunk");
            throw std::runtime_error("injected stream failure");
          }};
    });
    server.add("POST", "/stream", [](const auto& request, auto&) {
      auto log = std::make_shared<gufo::server::HttpResponse::StreamLog>();
      return gufo::server::HttpResponse{
          .streaming_body =
              [log, fail = !request.body.empty(),
               reported = request.body == "reported"](const auto& write) {
                (void)write(std::string_view("a\0b", 3));
                (void)write("");
                (void)write("end");
                if (fail)
                  log->error_code = "injected";
                log->error_event_sent = reported;
              },
          .stream_log = log,
      };
    });
    server.add("POST", "/sse-idle", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .headers = {{"Content-Type", "text/event-stream"}},
          .streaming_body =
              [](const auto& write) {
                (void)write("data: first\n\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(90));
                (void)write("data: last\n\n");
                (void)write("data: [DONE]\n\n");
              },
      };
    });
    server.add("POST", "/sse-finish", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{
          .headers = {{"Content-Type", "text/event-stream"}},
          .streaming_body = [fail = !request.body.empty()](const auto& write) {
            (void)write("data: first\n\n");
            if (fail)
              throw std::runtime_error("injected SSE failure");
            (void)write("data: [DONE]\n\n");
          }};
    });
    std::string error;
    assert(server.start(&error));
    worker = std::jthread([this, handle_signals] {
      server.run(handle_signals);
      run_finished.release();
    });
  }
  ~RunningServer() {
    server.stop();
    worker.join();
  }

  int Connect() {
    const int fd = ::socket(ipv6_ ? AF_INET6 : AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    assert(gufo::platform::SetSocketTimeout(fd, SO_RCVTIMEO,
                                            std::chrono::seconds{3}) == 0);
    if (ipv6_) {
      sockaddr_in6 address{};
      address.sin6_family = AF_INET6;
      address.sin6_port = htons(server.port());
      assert(::inet_pton(AF_INET6, "::1", &address.sin6_addr) == 1);
      assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0);
    } else {
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons(server.port());
      assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
      assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0);
    }
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false) {
    const int fd = Connect();
    while (!request.empty()) {
      const auto count =
          ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      ::shutdown(fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = gufo::platform::SocketRead(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    gufo::platform::CloseSocket(fd);
    return response;
  }

  std::string Post(std::string_view path, std::string_view body) {
    return Send("POST " + std::string(path) + " HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  }

  std::shared_ptr<FakeBackend> backend;
  HttpServer server;
  bool ipv6_;
  std::binary_semaphore run_finished{0};
  std::jthread worker;
};

void ExpectStatus(const std::string& response, int status) {
  if (!response.starts_with("HTTP/1.1 " + std::to_string(status) + " ")) {
    std::cerr << response << '\n';
    std::abort();
  }
}

void TestAuthorization() {
  RunningServer secured({.api_key = "test-secret"});
  for (const std::string path :
       {"/health", "/ready", "/v1/models", "/v1/chat/completions",
        "/v1/responses", "/v1/messages", "/v1/audio/speech",
        "/v1/audio/transcriptions", "/v1/video/generations", "/echo"}) {
    ExpectStatus(secured.Send("POST " + path + " HTTP/1.1\r\n\r\n"), 401);
  }
  for (const std::string header :
       {"", "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer "
        "test-secret\r\n"}) {
    const auto response =
        secured.Send("GET /v1/models HTTP/1.1\r\n" + header + "\r\n");
    ExpectStatus(response, 401);
    assert(response.find("WWW-Authenticate: Bearer") != std::string::npos);
    assert(response.find("test-secret") == std::string::npos);
  }
  assert(secured.backend->calls == 0);
  ExpectStatus(secured.Send("GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  "
                            "test-secret\r\n\r\n"),
               200);
  ExpectStatus(secured.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"),
               204);
  const std::string body = R"({"prompt":"hi","max_tokens":1})";
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nAuthorization: "
                            "Bearer test-secret\r\n"
                            "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body),
               200);
  assert(secured.backend->calls == 1);
}

// Drains the asynchronous logger before a test swaps or restores a sink. The
// call stays outside assert() so a build without asserts still flushes.
void DrainLogs() {
  const bool flushed = Logger::FlushSync();
  assert(flushed);
  (void)flushed;
}

struct CapturedLogs {
  std::string text;
  std::string request_id;
};

// Drives one pass of traffic at a fixed verbosity. The process-wide filter is
// shared by every test in this binary, so the previous level is restored.
CapturedLogs CaptureTraffic(LogLevel level) {
  const LogLevel previous = Logger::Level();
  Logger::SetLevel(level);
  // Drain queued writes before replacing their process-wide destination.
  DrainLogs();
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  CapturedLogs captured;
  {
    RunningServer server;
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
    const auto response = server.Post(
        "/v1/chat/completions?private-query",
        R"({"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true})");
    ExpectStatus(response, 200);
    const auto header = response.find("X-Request-ID: ");
    assert(header != std::string::npos);
    const auto begin = header + std::string("X-Request-ID: ").size();
    captured.request_id =
        response.substr(begin, response.find("\r\n", begin) - begin);

    const auto failed = server.Post("/stream-error", "");
    ExpectStatus(failed, 200);
    assert(failed.find("first chunk") != std::string::npos);
    assert(failed.find("HTTP/1.1", 1) == std::string::npos);

    ExpectStatus(server.Send("GET /not-a-route HTTP/1.1\r\n\r\n"), 404);
    // Rejected requests: a model name may be logged (filtered), request
    // content never. The Responses `include` value is echoed in the response
    // body under a code that is not on the log allow-list.
    ExpectStatus(
        server.Post("/v1/chat/completions",
                    R"({"model":"zz-named-model","messages":[{"role":"user","content":"SENTINEL-CONTENT-1"}]})"),
        404);
    (void)server.Post(
        "/v1/responses",
        R"({"model":"test","input":"SENTINEL-CONTENT-2","include":["SENTINEL-INCLUDE-3"]})");
    ExpectStatus(
        server.Post("/v1/chat/completions",
                    R"({"model":"test","messages":[{"role":"SENTINEL-ROLE-4","content":"SENTINEL-CONTENT-4"}]})"),
        400);
  }
  Logger::Info("test", "escaped\n\x1b[31m");
  DrainLogs();  // the async log sink writes later: drain before the capture ends
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous);
  captured.text = output.str();
  return captured;
}

void TestRequestLogging() {
  const CapturedLogs info = CaptureTraffic(LogLevel::kInfo);
  const std::string& log = info.text;
  assert(log.find("request=" + info.request_id + " event=received") !=
         std::string::npos);
  assert(log.find("request=" + info.request_id + " event=completed") !=
         std::string::npos);
  assert(log.find("cache=memory") != std::string::npos);
  assert(log.find("acceptance_pct=50.0") != std::string::npos);
  assert(log.find("rss_mib=") != std::string::npos);
  assert(log.find("error_code=server_exception") != std::string::npos);
  // A successful poll is quiet at the default level, and so is the received
  // line for a GET that is not on the inference list.
  assert(log.find("path=/health") == std::string::npos);
  // Statuses never are: 4xx keeps escalating to WARN under any filter.
  assert(log.find("path=/not-a-route status=404") != std::string::npos);
  assert(log.find("[WARN]") != std::string::npos);
  // The peer address and, for a refused request, the error text close the
  // line's own keys.
  const auto refused = log.find("path=/not-a-route status=404");
  const auto refused_end = log.find('\n', refused);
  const auto refused_line = log.substr(refused, refused_end - refused);
  assert(refused_line.find(" client=") != std::string::npos);
  assert(refused_line.find(" error=\"") != std::string::npos);
  assert(refused_line.find(" error_code=") < refused_line.find(" client="));
  assert(refused_line.find(" client=") < refused_line.find(" error=\""));
  const auto served = log.find("request=" + info.request_id + " event=completed");
  assert(log.substr(served, log.find('\n', served) - served).find(" client=") !=
         std::string::npos);
  assert(log.substr(served, log.find('\n', served) - served).find(" error=") ==
         std::string::npos);
  // Default deny: a model name is logged (as a filtered name), no request
  // content ever is, whatever the response body says.
  assert(log.find("zz-named-model") != std::string::npos);
  assert(
      log.find(
          "model 'zz-named-model' is not served: this process serves 'test'") !=
      std::string::npos);
  assert(log.find("SENTINEL-") == std::string::npos);
  assert(log.find("private-query") == std::string::npos);
  assert(log.find("private-prompt") == std::string::npos);
  assert(log.find("escaped\\x0a\\x1b[31m") != std::string::npos);
  // The startup confirmation is INFO-tier, so the default level reports it.
  // TestQuietTiersSuppressLifecycle covers the tiers that do not.
  assert(log.find("event=listening") != std::string::npos);

  const CapturedLogs debug = CaptureTraffic(LogLevel::kDebug);
  assert(debug.text.find("[DEBUG]") != std::string::npos);
  assert(debug.text.find("event=received method=GET path=/health") !=
         std::string::npos);
  assert(debug.text.find("path=/health status=200") != std::string::npos);
  // Debug adds lines; it must not downgrade an inference completion to DEBUG,
  // nor pull a refused 4xx back under the threshold.
  assert(debug.text.find("[INFO] [http] request=" + debug.request_id +
                         " event=completed") != std::string::npos);
  assert(debug.text.find("path=/not-a-route status=404") != std::string::npos);
  // The higher tier must still hide every byte of the request.
  assert(debug.text.find("private-query") == std::string::npos);
  assert(debug.text.find("private-prompt") == std::string::npos);
  // Options are echoed by `gufo serve`, not by the HTTP layer itself.
  assert(debug.text.find("event=options") == std::string::npos);
}

std::string ResponseRequestId(const std::string& response) {
  const auto header = response.find("X-Request-ID: ");
  assert(header != std::string::npos);
  const auto begin = header + std::string("X-Request-ID: ").size();
  return response.substr(begin, response.find("\r\n", begin) - begin);
}

// The body of a chunked HTTP/1.1 response, as the client reassembles it.
std::string ChunkedBody(const std::string& response) {
  std::size_t cursor = response.find("\r\n\r\n");
  assert(cursor != std::string::npos);
  cursor += 4;
  std::string body;
  for (;;) {
    const auto line_end = response.find("\r\n", cursor);
    assert(line_end != std::string::npos);
    const auto size =
        std::stoul(response.substr(cursor, line_end - cursor), nullptr, 16);
    if (size == 0) {
      return body;
    }
    body += response.substr(line_end + 2, size);
    cursor = line_end + 2 + size + 2;
  }
}

// `--trace` keeps what a text request carried in and out under the id the
// logs name. Polls, other routes and unauthenticated requests stay out.
void TestContentTrace() {
  namespace fs = std::filesystem;
  using gufo::server::Trace;
  const auto path =
      fs::temp_directory_path() /
      ("gufo-trace-test-" + std::to_string(::getpid()) + ".jsonl");
  fs::remove(path);
  assert(!Trace::Open(path.string()).has_value());
  std::string plain_id;
  std::string stream_id;
  std::string stream_wire;
  std::string invalid_id;
  {
    RunningServer server(
        {.api_key = "test-secret",
         .sse_heartbeat_interval = std::chrono::milliseconds(5)});
    server.backend->SetOutput("<tool_call>leak");
    const auto post = [&](std::string_view route, std::string_view body,
                          bool authorized = true) {
      return server.Send(
          "POST " + std::string(route) + " HTTP/1.1\r\n" +
          (authorized ? "Authorization: Bearer test-secret\r\n" : "") +
          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" +
          std::string(body));
    };
    const auto plain = post("/v1/completions",
                            R"({"prompt":"private-prompt","max_tokens":8})");
    ExpectStatus(plain, 200);
    plain_id = ResponseRequestId(plain);
    // The id is bound where the backend generates, so the scheduler's record
    // can name it.
    assert(server.backend->LastCall().trace_request == plain_id);
    // Admission commits the stream before a slow first token, so keepalives
    // reach the client and must reach the trace too.
    server.backend->admit = true;
    server.backend->token_delay = std::chrono::milliseconds(40);
    const auto streamed =
        post("/v1/completions",
             R"({"prompt":"streamed-prompt","max_tokens":8,"stream":true})");
    ExpectStatus(streamed, 200);
    stream_id = ResponseRequestId(streamed);
    stream_wire = ChunkedBody(streamed);
    assert(stream_wire.find(": ping\n\n") != std::string::npos);
    assert(server.backend->LastCall().trace_request == stream_id);
    const auto invalid = post("/v1/completions", "{");
    ExpectStatus(invalid, 400);
    invalid_id = ResponseRequestId(invalid);
    ExpectStatus(
        post("/v1/completions", R"({"prompt":"unauthorized-prompt"})", false),
        401);
    ExpectStatus(post("/echo", "echo-body"), 200);
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\nAuthorization: Bearer "
                             "test-secret\r\n\r\n"),
                 200);
  }
  Trace::Close();

#ifndef _WIN32  // POSIX file modes do not exist on Windows (fs::status reports all permissions)
  assert((fs::status(path).permissions() & fs::perms::all) ==
         (fs::perms::owner_read | fs::perms::owner_write));
#endif
  std::vector<gufo::json::Value> records;
  {
    std::ifstream input(path);
    for (std::string line; std::getline(input, line);) {
      assert(line.find("unauthorized-prompt") == std::string::npos);
      assert(line.find("echo-body") == std::string::npos);
      records.push_back(gufo::json::parse(line));
    }
  }
  fs::remove(path);
  assert(records.size() == 6);
  const auto record = [&](std::string_view event,
                          const std::string& id) -> const gufo::json::Value& {
    for (const auto& candidate : records) {
      if (candidate.member_str("event") == event &&
          candidate.member_str("request") == id) {
        assert(!candidate.member_str("time").empty());
        return candidate;
      }
    }
    std::abort();
  };

  const auto& request = record("request", plain_id);
  assert(request.member_str("method") == "POST");
  assert(request.member_str("path") == "/v1/completions");
  assert(request.member_str("body") ==
         R"({"prompt":"private-prompt","max_tokens":8})");
  const auto& reply = record("response", plain_id);
  assert(reply.member_size("status") == 200);
  assert(reply.member_str("outcome") == "completed");
  assert(!reply.find("stream")->as_bool());
  const auto body = gufo::json::parse(reply.member_str("body"));
  assert(body.find("choices")->items()[0].member_str("text") ==
         "<tool_call>leak");

  assert(
      record("request", stream_id).member_str("body").find("streamed-prompt") !=
      std::string::npos);
  const auto& streamed = record("response", stream_id);
  assert(streamed.find("stream")->as_bool());
  assert(streamed.member_str("body") == stream_wire);
  assert(stream_wire.find("<tool_call>leak") != std::string::npos);

  assert(record("request", invalid_id).member_str("body") == "{");
  assert(record("response", invalid_id).member_size("status") == 400);
}

// The threshold covers the lifecycle lines too: `--log-level=warn|error` boots
// and stops without a word, which docs/SERVER.md states, while an escalation
// keeps its own tier and a filtered INFO receipt line never reaches the log.
void TestQuietTiersSuppressLifecycle() {
  const CapturedLogs warn = CaptureTraffic(LogLevel::kWarn);
  assert(warn.text.find("event=listening") == std::string::npos);
  assert(warn.text.find("event=received") == std::string::npos);
  assert(warn.text.find("path=/not-a-route status=404") != std::string::npos);

  const CapturedLogs error = CaptureTraffic(LogLevel::kError);
  assert(error.text.find("event=listening") == std::string::npos);
  assert(error.text.find("event=received") == std::string::npos);
  assert(error.text.find("path=/not-a-route status=404") == std::string::npos);
}

void TestLogLevelFilter() {
  const LogLevel previous = Logger::Level();
  assert(previous == LogLevel::kInfo);

  DrainLogs();
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  Logger::SetLevel(LogLevel::kError);
  Logger::Debug("probe", "hidden-debug");
  Logger::Info("probe", "hidden-info");
  Logger::Warn("probe", "hidden-warn");
  Logger::Error("probe", "visible-error");
  // The printf-style entry point the imported DeepSeek runtime uses has to
  // filter before it formats: an INFO loader line stays silent at an absolute
  // threshold while an ERROR line survives.
  Logger::LogFormatted(LogLevel::kInfo, "ds4", "hidden-ds4-info %d", 1);
  Logger::LogFormatted(LogLevel::kError, "ds4", "visible-ds4-error %d", 2);
  Logger::LogRequest("r-quiet", "GET", "/health", 200, 1.0, "", "completed",
                     LogLevel::kDebug);
  Logger::LogRequest("r-404", "GET", "/missing", 404, 1.0);
  Logger::LogRequest("r-500", "POST", "/boom", 500, 1.0);
  Logger::SetLevel(LogLevel::kWarn);
  Logger::Debug("probe", "still-hidden-debug");
  Logger::Warn("probe", "visible-at-warn");
  Logger::LogFormatted(LogLevel::kInfo, "ds4", "still-hidden-ds4-info");
  Logger::LogFormatted(LogLevel::kWarn, "ds4", "visible-ds4-warn");
  Logger::LogRequest("r-404b", "GET", "/missing", 404, 1.0);
  DrainLogs();
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous);

  const std::string log = output.str();
  assert(log.find("hidden-debug") == std::string::npos);
  assert(log.find("hidden-info") == std::string::npos);
  assert(log.find("hidden-warn") == std::string::npos);
  assert(log.find("still-hidden-debug") == std::string::npos);
  assert(log.find("hidden-ds4-info") == std::string::npos);
  assert(log.find("still-hidden-ds4-info") == std::string::npos);
  assert(log.find("visible-error") != std::string::npos);
  assert(log.find("visible-at-warn") != std::string::npos);
  assert(log.find("visible-ds4-error 2") != std::string::npos);
  assert(log.find("visible-ds4-warn") != std::string::npos);
  assert(log.find("[DEBUG]") == std::string::npos);
  // A poll that is quiet by default stays filtered, while an escalation keeps
  // its own tier: `--log-level error` means errors only, not "hide failures".
  assert(log.find("request=r-quiet event=completed") == std::string::npos);
  assert(log.find("request=r-404 event=completed") == std::string::npos);
  assert(log.find("request=r-500 event=completed") != std::string::npos);
  assert(log.find("request=r-404b event=completed") != std::string::npos);
  assert(Logger::Level() == LogLevel::kInfo);
}

void TestLogLevelNames() {
  assert(LogLevelFromName("debug") == LogLevel::kDebug);
  assert(LogLevelFromName("info") == LogLevel::kInfo);
  assert(LogLevelFromName("warn") == LogLevel::kWarn);
  assert(LogLevelFromName("error") == LogLevel::kError);
  assert(!LogLevelFromName("trace").has_value());
  assert(!LogLevelFromName("DEBUG").has_value());
  assert(!LogLevelFromName("").has_value());
  assert(LogLevelName(LogLevel::kDebug) == "debug");
  assert(LogLevelName(LogLevel::kError) == "error");
}

void TestFramingAndMetrics() {
  RunningServer server({.max_request_body_bytes = 8192});
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  for (const std::string header :
       {"X!#$%&'*+-.^_`|~: one", "cOnTeNt-LeNgTh: 0"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 200);
  }
  for (const std::string header :
       {"Content-Length: nope", "Content-Length: -1", "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3", "Transfer-Encoding: chunked",
        "broken-header", "Transfer-Encoding : chunked", "Content-Length\t: 5",
        "X-Test: one\r\n folded", "Bad Name: x"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 400);
  }
  ExpectStatus(server.Send("POST /echo\r\n\r\n"), 400);
  ExpectStatus(server.Send("POST /echo HTTP/1.1 extra\r\n\r\n"), 400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
      400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 8193\r\n\r\n"), 413);
  const std::string payload(5000, 'x');
  const auto echo = server.Send(
      "POST /echo HTTP/1.1\r\nContent-Length: 5000\r\nContent-Length: "
      "5000\r\n\r\n" +
      payload);
  ExpectStatus(echo, 200);
  assert(echo.substr(echo.find("\r\n\r\n") + 4) == payload);

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  const auto response =
      server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body);
  ExpectStatus(response, 200);
  const auto parsed =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  const auto* timings = parsed.find("timings");
  assert(timings != nullptr);
  assert(timings->member_double("prompt_n") == 2);
  assert(timings->member_double("cache_n") == 8);
  assert(timings->member_double("prompt_per_second") == 500);
  assert(timings->member_double("prompt_per_token_ms") == 2);
}

void TestFallbackBackendMetrics() {
  namespace metrics = gufo::server::detail;
  RunningServer server;
  struct Endpoint {
    const char* path;
    const char* body;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})"},
           Endpoint{"/v1/completions", R"({"prompt":"hi"})"},
           Endpoint{"/v1/responses", R"({"input":"hi"})"},
       }) {
    for (const bool stream : {false, true}) {
      const auto prompt_before = metrics::TotalPromptTokens().load();
      const auto generated_before = metrics::TotalGenTokens().load();
      const auto cached_before = metrics::TotalCachedPromptTokens().load();
      const auto rounds_before = metrics::TotalDraftRounds().load();
      const auto drafts_before = metrics::TotalDraftTokens().load();
      const auto accepted_before = metrics::TotalDraftAcceptedTokens().load();
      const auto prompt_seconds_before = metrics::TotalPromptSeconds().load();
      const auto gen_seconds_before = metrics::TotalGenSeconds().load();
      auto body = gufo::json::parse(endpoint.body);
      body["model"] = "test";
      body["stream"] = stream;
      ExpectStatus(server.Post(endpoint.path, body.dump()), 200);
      const auto response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
      ExpectStatus(response, 200);
      // The backend reports 10 prompt tokens, of which 8 were cached.
      assert(response.find("llamacpp:prompt_tokens_total " +
                           std::to_string(prompt_before + 2) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:prompt_tokens_cached_total " +
                           std::to_string(cached_before + 8) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_drafts_total " +
                           std::to_string(rounds_before + 2) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_draft_tokens_total " +
                           std::to_string(drafts_before + 8) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_accepted_tokens_total " +
                           std::to_string(accepted_before + 4) + "\n") !=
             std::string::npos);
      assert(metrics::TotalPromptSeconds().load() - prompt_seconds_before >
                 0.0039 &&
             metrics::TotalGenSeconds().load() - gen_seconds_before > 0.0019);
      assert(metrics::MaxSequenceTokens().load() >= 11);
      assert(response.find("llamacpp:tokens_predicted_total " +
                           std::to_string(generated_before + 1) + "\n") !=
             std::string::npos);
    }
  }
  struct Case {
    std::size_t prefill, cached;
    bool cancelled;
    std::size_t expected_prompt;
  };
  for (const auto& test : {
           Case{6, 4, false, 6},
           Case{0, 4, false, 6},
           Case{0, 10, false, 0},
           Case{0, 20, false, 0},
           Case{0, 0, true, 0},
           Case{3, 0, true, 3},
       }) {
    const auto prompt_before = metrics::TotalPromptTokens().load();
    const auto generated_before = metrics::TotalGenTokens().load();
    TextGenerationBackend::Result result;
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = test.cached;
    result.prefill_tokens = test.prefill;
    result.completion_tokens = 2;
    result.cancelled = test.cancelled;
    gufo::server::RecordServerMetrics(result);
    assert(metrics::TotalPromptTokens().load() ==
           prompt_before + test.expected_prompt);
    assert(metrics::TotalGenTokens().load() == generated_before + 2);
  }
}

void TestLlamaSlotsAndMetrics() {
  RunningServer server;
  const auto get = [&](const std::string& path) {
    const auto response = server.Send("GET " + path + " HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 200);
    return response.substr(response.find("\r\n\r\n") + 4);
  };
  const auto expect_slot = [](const gufo::json::Value& slot, std::size_t id,
                              bool processing) {
    assert(slot.member_size("id", 99) == id);
    assert(slot.member_size("n_ctx") == 65536);
    assert(slot.find("speculative")->is_bool());
    assert(slot.find("is_processing")->as_bool(!processing) == processing);
    assert(slot.find("state")->as_double() == (processing ? 1 : 0));
    assert(slot.find("task_id")->as_double() ==
           slot.find("id_task")->as_double());
    for (const char* key : {"n_prompt_tokens", "n_prompt_tokens_cache",
                            "n_prompt_tokens_processed"}) {
      assert(slot.find(key)->is_number());
    }
    assert(slot.find("prompt")->is_string() &&
           slot.member_str("prompt").empty());
    assert(slot.member_str("model") == "test");
    const auto* next = slot.find("next_token");
    assert(next != nullptr && next->is_array() && next->size() == 1);
    const auto& token = next->items().front();
    assert(token.find("has_next_token")->as_bool(!processing) == processing);
    assert(token.find("has_new_line")->is_bool());
    assert(token.find("n_remain")->is_number() &&
           token.find("n_decoded")->is_number());
  };

  // Backends without a session pool report one idle slot.
  auto slots = gufo::json::parse(get("/slots"));
  assert(slots.is_array() && slots.size() == 1);
  const auto& fallback = slots.items().front();
  expect_slot(fallback, 0, false);
  assert(fallback.find("id_task")->as_double() == -1);
  assert(fallback.member_size("n_prompt_tokens", 99) == 0);
  assert(fallback.find("next_token")
             ->items()
             .front()
             .find("n_remain")
             ->as_double() == -1);
  assert(get("/metrics").find("llamacpp:kv_cache_usage_ratio 0\n") !=
         std::string::npos);

  server.backend->sessions.resize(2);
  server.backend->sessions[1] = {
      .processing = true,
      .speculative = true,
      .request_id = 42,
      .prompt_tokens = 10,
      .cached_prompt_tokens = 8,
      .processed_prompt_tokens = 2,
      .generated_tokens = 3,
      .remaining_tokens = 5,
  };
  slots = gufo::json::parse(get("/v1/slots"));
  assert(slots.size() == 2);
  expect_slot(slots.items()[0], 0, false);
  const auto& busy = slots.items()[1];
  expect_slot(busy, 1, true);
  assert(busy.find("speculative")->as_bool() &&
         busy.member_size("id_task") == 42 &&
         busy.member_size("n_prompt_tokens") == 10 &&
         busy.member_size("n_prompt_tokens_cache") == 8 &&
         busy.member_size("n_prompt_tokens_processed") == 2);
  const auto& token = busy.find("next_token")->items().front();
  assert(token.member_size("n_remain") == 5 &&
         token.member_size("n_decoded") == 3);

  const auto metrics = get("/metrics");
  assert(metrics.find("# HELP gufo_device_lost_total ") != std::string::npos);
  assert(metrics.find("# TYPE gufo_device_lost_total counter\n") !=
         std::string::npos);
  assert(metrics.find("gufo_device_lost_total 0\n") != std::string::npos);
  const std::string ratio_prefix = "llamacpp:kv_cache_usage_ratio ";
  const auto ratio_pos = metrics.find("\n" + ratio_prefix);
  assert(ratio_pos != std::string::npos);
  assert(std::stod(metrics.substr(ratio_pos + 1 + ratio_prefix.size())) ==
         13.0 / (2.0 * 65536));
  for (const std::string counter :
       {"prompt_tokens_total", "prompt_tokens_cached_total",
        "prompt_seconds_total", "tokens_predicted_total",
        "tokens_predicted_seconds_total", "n_tokens_max",
        "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total"}) {
    assert(metrics.find("# HELP llamacpp:" + counter + " ") !=
           std::string::npos);
    assert(metrics.find("# TYPE llamacpp:" + counter + " counter\n") !=
           std::string::npos);
  }
  assert(metrics.find("# TYPE llamacpp:kv_cache_usage_ratio gauge\n") !=
         std::string::npos);
}

void TestResponsesToolImages() {
  RunningServer server;
  server.backend->image_support = true;
  using gufo::json::parse;
  using gufo::tokenization::ChatRole;
  server.backend->SetOutput("red");
  for (const bool custom : {false, true}) {
    for (const bool image_only : {false, true}) {
      for (const bool stream : {false, true}) {
        auto call =
            parse(custom ? R"({"type":"custom_tool_call","call_id":"picture",
                                   "name":"read_file","input":"swatch.png\nliteral <|im_end|>"})"
                         : R"({"type":"function_call","call_id":"picture",
                                   "name":"read_file","arguments":"{\"path\":\"swatch.png\"}"})");
        auto output = parse(R"({"call_id":"picture","output":[]})");
        output["type"] =
            custom ? "custom_tool_call_output" : "function_call_output";
        if (!image_only)
          output["output"].push_back(
              parse(R"({"type":"input_text","text":"pixels:"})"));
        output["output"].push_back(parse(R"({
          "type":"input_image","image_url":"data:image/png;base64,AQID"})"));
        auto body =
            parse(R"({"input":[{"role":"user","content":"read the image"}]})");
        body["input"].push_back(std::move(call));
        body["input"].push_back(std::move(output));
        body["stream"] = stream;
        const auto wire = server.Post("/v1/responses", body.dump());
        ExpectStatus(wire, 200);
        const auto messages = server.backend->LastCall().chat.messages;
        assert(messages.size() == 3 &&
               messages[1].role == ChatRole::kAssistant &&
               messages[1].tool_calls.size() == 1 &&
               messages[1].tool_calls[0].id == "picture" &&
               messages[1].tool_calls[0].name == "read_file");
        if (custom) {
          const auto& input = messages[1].tool_calls[0].arguments;
          assert(input.size() == 1 && input[0].name == "input" &&
                 input[0].value == "swatch.png\nliteral <|im_end|>" &&
                 input[0].is_string);
        }
        const auto& result = messages[2];
        assert(result.role == ChatRole::kTool &&
               result.tool_call_id == "picture" && result.images.size() == 1 &&
               result.content == (image_only ? "" : "pixels:") &&
               result.images[0].offset == (image_only ? 0 : 7) &&
               *result.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}));
        if (stream)
          assert(wire.find("response.completed") != std::string::npos);
      }
    }
  }
  const auto calls_before = server.backend->calls.load();
  for (
      const auto* item :
      {R"({"type":"custom_tool_call","call_id":"picture","name":"read_file","input":3})",
       R"({"type":"custom_tool_call","name":"read_file","input":"file.png"})",
       R"({"type":"custom_tool_call_output","output":[]})",
       R"({"type":"custom_tool_call_output","call_id":"picture","output":3})",
       R"({"type":"function_call_output","call_id":"picture","output":[{"type":"input_file","file_id":"x"}]})",
       R"({"type":"custom_tool_call_output","call_id":"picture","output":[{"type":"input_image","file_id":"x"}]})"}) {
    auto body = parse(R"({"input":[]})");
    body["input"].push_back(parse(item));
    ExpectStatus(server.Post("/v1/responses", body.dump()), 400);
  }
  assert(server.backend->calls == calls_before);
}

// A text-only model refuses image blocks before any image is read, so a
// client cannot make the server fetch a URL for a request that cannot succeed.
void TestTextOnlyBackendRefusesImagesBeforeReading() {
  RunningServer server;
  server.backend->image_support = false;
  const int calls = server.backend->calls;
  // An unresolvable host or invalid base64 would produce a different error
  // if the image were fetched or decoded.
  for (const auto* url : {"https://image-fetch-must-not-happen.invalid/x.png",
                          "data:image/png;base64,!!!not-base64!!!"}) {
    const std::string messages =
        std::string(R"({"messages":[{"role":"user","content":[{"type":"image",)"
                    R"("source":{"type":"url","url":")") +
        url + R"("}}]}]})";
    const std::string responses =
        std::string(
            R"({"input":[{"role":"user","content":[{"type":"input_image",)"
            R"("image_url":")") +
        url + R"("}]}]})";
    for (const auto& [path, body] :
         {std::pair<std::string, std::string>{"/v1/messages", messages},
          {"/v1/messages/count_tokens", messages},
          {"/v1/responses", responses}}) {
      const auto rejected = server.Post(path, body);
      ExpectStatus(rejected, 400);
      if (path == "/v1/responses") {
        assert(rejected.find("does not support image input") !=
               std::string::npos);
      } else {
        // Bubo has no Anthropic image-block support (#510). These routes
        // reject the block without inspecting or reading its image source.
        assert(rejected.find("use /v1/chat/completions for images") !=
               std::string::npos);
      }
    }
  }
  assert(server.backend->calls == calls);
}

void TestCompatibilityRequests() {
  RunningServer server;
  server.backend->image_support = true;
  using gufo::json::parse;
  const auto response_body = [](const std::string& response) {
    ExpectStatus(response, 200);
    return parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  struct Endpoint {
    const char *path, *body, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi"})", "max_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi"})", "max_output_tokens"},
           Endpoint{"/v1/messages",
                    R"({"messages":[{"role":"user","content":"hi"}]})",
                    "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi"})", "n_predict"},
       }) {
    auto body = parse(endpoint.body);
    response_body(server.Post(endpoint.path, body.dump()));
    assert(server.backend->LastCall().max_tokens == 0);
    server.backend->defaults.model = gufo::sampling::TextModelPreset::kQwen38;
    server.backend->defaults.supplied = {};
    server.backend->reasoning.enabled = false;
    response_body(server.Post(endpoint.path, body.dump()));
    const auto preset = server.backend->LastCall().sampling;
    assert(preset.temperature == 0.7F && preset.top_p == 0.8F &&
           preset.top_k == 20 && preset.presence_penalty == 1.5F);
    server.backend->defaults.sampling.top_k = 0;
    server.backend->defaults.supplied.top_k = true;
    body["temperature"] = 0;
    body["presence_penalty"] = 0;
    response_body(server.Post(endpoint.path, body.dump()));
    const auto overridden = server.backend->LastCall().sampling;
    assert(overridden.temperature == 0 && overridden.top_k == 0 &&
           overridden.top_p == 0.8F && overridden.presence_penalty == 0);
    server.backend->defaults = {};
    server.backend->reasoning = {};
    body["model"] = "test";
    body[endpoint.limit] = 1;
    body["temperature"] = 0.6;
    body["top_k"] = 40;
    body["top_p"] = 0.9;
    body["seed"] = 123;
    body["repeat_penalty"] = 1.1;
    const auto output = response_body(server.Post(endpoint.path, body.dump()));
    const auto last = server.backend->LastCall();
    for (int failure : {1, 2}) {
      server.backend->failure = failure;
      const auto rejected = server.Post(endpoint.path, body.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find(failure == 1
                               ? "context_length_exceeded"
                               : "invalid_prompt") != std::string::npos);
    }
    server.backend->failure = 0;
    assert(last.client_id == "127.0.0.1");
    assert(last.max_tokens == 1 && last.sampling.temperature == 0.6F &&
           last.sampling.top_k == 40 && last.sampling.top_p == 0.9F &&
           last.sampling.seed == 123 && last.sampling.repeat_penalty == 1.1F);
    if (std::string_view(endpoint.path) == "/v1/responses") {
      assert(output.member_str("status") == "incomplete");
      assert(output.find("incomplete_details")->member_str("reason") ==
             "max_output_tokens");
      assert(output.find("usage")->contains("input_tokens_details"));
    } else if (std::string_view(endpoint.path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "max_tokens");
      // input_tokens excludes cache reads; together they cover the prompt.
      assert(output.find("usage")->member_size("input_tokens") == 2 &&
             output.find("usage")->member_size("cache_read_input_tokens") == 8);
    } else if (std::string_view(endpoint.path) == "/completion") {
      assert(output.find("stopped_length")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "length");
    }
    const int calls = server.backend->calls;
    for (const auto& [field, value] :
         {std::pair{"temperature", 2.01}, std::pair{"presence_penalty", 2.01},
          std::pair{"frequency_penalty", -2.01}}) {
      auto invalid = body;
      invalid[field] = value;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto value : {"0", "-2", "1.5", "1e100", "\"1\"", "true"}) {
      auto invalid = body;
      invalid[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto field :
         {"stream", "echo", "store", "background", "tools", "stop", "reasoning",
          "output_config", "logit_bias", "ignore_eos"}) {
      const std::string_view path(endpoint.path);
      const std::string_view name(field);
      if ((path == "/v1/responses" || path == "/v1/completions" ||
           path == "/v1/messages") &&
          name == "stream")
        continue;
      if (path == "/v1/completions" && name == "ignore_eos")
        continue;
      auto invalid = body;
      invalid[field] = true;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    const bool streams = std::string_view(endpoint.path) != "/completion";
    if (streams) {
      auto invalid = body;
      invalid["stream"] = "true";
      const auto rejected = server.Post(endpoint.path, invalid.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find("'stream' must be a boolean") != std::string::npos);
    }
    auto invalid = body;
    invalid["n"] = 1.4;
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    invalid = body;
    invalid["model"] = "wrong";
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 404);
    ExpectStatus(server.Post(endpoint.path, "[]"), 400);
    ExpectStatus(server.Post(endpoint.path, "{"), 400);
    assert(server.backend->calls == calls);
    for (const auto value : {"-1", "null"}) {   // v0.1.1: both leave the server default in place
      auto unset = body;
      unset[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, unset.dump()), 200);
      assert(server.backend->LastCall().max_tokens == 0);
    }
    if (streams) {
      // A null stream is unset, as on Chat Completions.
      auto unset = body;
      unset["stream"] = gufo::json::Value();
      const auto buffered = server.Post(endpoint.path, unset.dump());
      assert(buffered.find("text/event-stream") == std::string::npos);
      response_body(buffered);
    }
  }
  const int calls = server.backend->calls;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":["one","two"]})"),
               400);
  ExpectStatus(server.Post("/v1/responses",
                           R"({"input":[{"role":"assistant","content":[
                           {"type":"input_text","text":"describe"},
                           {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"tool","content":"hi"}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages", R"({"messages":[]})"), 400);
  ExpectStatus(
      server.Post("/infill", R"({"input_prefix":"one","input_suffix":"two"})"),
      501);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               501);
  assert(server.backend->calls == calls);

  // With a backend that can count, count_tokens reports the prompt size of
  // the same parsed request Messages would generate from, without generating.
  server.backend->chat_token_count = 1234;
  const auto counted = response_body(
      server.Post("/v1/messages/count_tokens", R"({"system":"Be brief.",
      "messages":[{"role":"user","content":[
        {"type":"text","text":"what is it?"}]}],
      "tools":[{"name":"look","input_schema":{"type":"object"}}]})"));
  assert(counted.find("input_tokens") != nullptr &&
         counted.find("input_tokens")->as_double() == 1234.0);
  assert(server.backend->counted_messages == 2 &&
         server.backend->counted_tools == 1);
  assert(server.backend->calls == calls);
  for (const auto* invalid :
       {R"({"stream":true,"messages":[{"role":"user","content":"hi"}]})",
        R"({"messages":[]})",
        R"({"messages":[{"role":"user","content":"hi"}],"thinking":true})"})
    ExpectStatus(server.Post("/v1/messages/count_tokens", invalid), 400);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"model":"other","messages":[
                             {"role":"user","content":"hi"}]})"),
               404);
  server.backend->chat_token_count.reset();

  for (const auto& [url, expected] :
       {std::pair{"data:image/gif;base64,AA==", "image/gif"},
        std::pair{"data:image/png,AA==", "base64"},
        std::pair{"data:image/png;base64,A===", "base64"}}) {
    const auto input = parse(
        R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":)" +
        gufo::json::Value(url).dump() +
        R"(},{"type":"input_text","text":"describe"}]}]})");
    const auto rejected = server.Post("/v1/responses", input.dump());
    ExpectStatus(rejected, 400);
    const auto parsed = parse(rejected.substr(rejected.find("\r\n\r\n") + 4));
    const auto& error = *parsed.find("error");
    assert(error.member_str("code") == "invalid_request");
    assert(error.member_str("message").find(expected) != std::string::npos);
  }
  assert(server.backend->calls == calls);

  const auto structured =
      response_body(server.Post("/v1/responses",
                                R"({"input":[{"role":"user","content":[
        {"type":"input_text","text":"describe"},
        {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],
        "reasoning":{"effort":"none"},
        "text":{"format":{"type":"json_schema","name":"answer","strict":true,
          "schema":{"type":"object","properties":{"score":{"type":"integer","minimum":1,"maximum":5}},
          "required":["score"],"additionalProperties":false}}}})"));
  const auto request = server.backend->LastCall().chat;
  assert(request.response_format && request.reasoning.enabled == false &&
         request.messages.back().images.size() == 1 &&
         request.messages.back().images[0].offset == 8);

  const auto response = response_body(server.Post(
      "/v1/responses",
      R"({"instructions":"Be concise.","input":[{"role":"user","content":[
          {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false})"));
  assert(response.member_str("status") == "completed");
  const auto messages = server.backend->LastCall().chat.messages;
  assert(messages.size() == 2);
  assert(messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         messages[0].content == "Be concise." && messages[1].content == "hi");
  for (const auto limit : {1, 2}) {
    const auto streaming = server.Post(
        "/v1/responses",
        std::string(R"({"input":"hi","stream":true,"max_output_tokens":)") +
            std::to_string(limit) + "}");
    ExpectStatus(streaming, 200);
    assert(streaming.find("text/event-stream") != std::string::npos);
    assert(streaming.find("event: response.created") != std::string::npos);
    assert(streaming.find("event: response.output_text.delta") !=
           std::string::npos);
    assert(streaming.find(limit == 1 ? "event: response.incomplete"
                                     : "event: response.completed") !=
           std::string::npos);
  }
  server.backend->failure = 6;
  const auto failed_stream =
      server.Post("/v1/responses", R"({"input":"hi","stream":true})");
  ExpectStatus(failed_stream, 200);  // Fake backend fails after headers.
  assert(failed_stream.find("event: response.failed") != std::string::npos);
  assert(failed_stream.find("event: response.completed") == std::string::npos);
  const auto failed_data = failed_stream.find(
      "data: ", failed_stream.find("event: response.failed"));
  assert(failed_data != std::string::npos);
  const auto failed_event = gufo::json::parse(
      std::string_view(failed_stream)
          .substr(failed_data + 6,
                  failed_stream.find("\n\n", failed_data) - failed_data - 6));
  assert(failed_event.find("response")->find("error")->member_str("code") ==
         "server_error");
  assert(failed_stream.ends_with("0\r\n\r\n"));
  server.backend->failure = 0;
  const auto continued = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}]},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]},
        {"role":"user","content":"Next question"}]})"));
  assert(continued.member_str("status") == "completed");
  const auto replay_messages = server.backend->LastCall().chat.messages;
  assert(replay_messages.size() == 3 &&
         replay_messages[1].thought == "Thoughts" &&
         replay_messages[1].content == "Answer");

  // Codex replays reasoning items with the null content and encrypted_content
  // fields it serialized from the response; only non-null payload is rejected.
  const auto null_replay = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}],
         "content":null,"encrypted_content":null},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]}]})"));
  assert(null_replay.member_str("status") == "completed");
  const auto null_replay_messages = server.backend->LastCall().chat.messages;
  assert(null_replay_messages.size() == 2 &&
         null_replay_messages[1].thought == "Thoughts" &&
         null_replay_messages[1].content == "Answer");
  const auto opaque_replay =
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"question"},
        {"type":"reasoning","id":"rs_1",
         "summary":[],"encrypted_content":"gAAAA"}]})");
  ExpectStatus(opaque_replay, 400);

  // Codex re-sends developer items mid-conversation. They stay in place, so
  // the prompt before them is unchanged and reusable.
  const auto in_place =
      response_body(server.Post("/v1/responses",
                                R"({"instructions":"Base rules.","input":[
        {"role":"user","content":"first"},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"noted","annotations":[]}]},
        {"type":"message","role":"developer","content":[
          {"type":"input_text","text":"Compacted state"}]},
        {"role":"user","content":"second"}]})"));
  assert(in_place.member_str("status") == "completed");
  const auto in_place_messages = server.backend->LastCall().chat.messages;
  assert(in_place_messages.size() == 5 &&
         in_place_messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         in_place_messages[0].content == "Base rules." &&
         in_place_messages[1].content == "first" &&
         in_place_messages[2].content == "noted" &&
         in_place_messages[3].role ==
             gufo::tokenization::ChatRole::kDeveloper &&
         in_place_messages[3].content == "Compacted state" &&
         in_place_messages[4].content == "second");

  // A developer item inside a replayed reasoning/function_call group waits
  // for the group to end instead of splitting the assistant turn.
  const auto replay_items = gufo::json::parse(R"([
      {"role":"user","content":"Check state."},
      {"type":"reasoning","summary":[
        {"type":"summary_text","text":"I will inspect."}]},
      {"type":"function_call","call_id":"call_1","name":"lookup",
       "arguments":"{\"key\":\"state\"}"},
      {"type":"function_call_output","call_id":"call_1","output":"OK"}])");
  const auto developer =
      gufo::json::parse(R"({"role":"developer","content":"Follow policy."})");
  for (const std::size_t position : {2U, 3U}) {
    auto input = gufo::json::Value::array();
    for (std::size_t i = 0; i < replay_items.size(); ++i) {
      if (i == position)
        input.push_back(developer);
      input.push_back(replay_items.items()[i]);
    }
    auto body = gufo::json::Value::object();
    body["input"] = std::move(input);
    const auto replay =
        response_body(server.Post("/v1/responses", body.dump()));
    assert(replay.member_str("status") == "completed");
    const auto grouped = server.backend->LastCall().chat.messages;
    assert(grouped.size() == 4);
    assert(grouped[0].role == gufo::tokenization::ChatRole::kUser &&
           grouped[1].role == gufo::tokenization::ChatRole::kAssistant &&
           grouped[1].content.empty() &&
           grouped[1].thought == "I will inspect." &&
           grouped[1].tool_calls.size() == 1 &&
           grouped[1].tool_calls[0].id == "call_1" &&
           grouped[2].role == gufo::tokenization::ChatRole::kDeveloper &&
           grouped[2].content == "Follow policy." &&
           grouped[3].role == gufo::tokenization::ChatRole::kTool &&
           grouped[3].tool_call_id == "call_1" && grouped[3].content == "OK");
  }

  // The Responses API carries request-only fields with no native effect here
  // (hosted tool types, include, reasoning.summary, text.verbosity). Accept
  // them and keep every executable function tool, flattening the client-side
  // namespace grouping. Codex is one such client.
  const auto hosted = response_body(server.Post(
      "/v1/responses",
      R"({"model":"test","instructions":"You are a coding agent.","input":[
          {"type":"message","role":"developer","content":[
            {"type":"input_text","text":"AGENTS instructions"}]},
          {"type":"message","role":"user","content":[
            {"type":"input_text","text":"say hi"}]}],
        "reasoning":{"effort":"medium","summary":"auto"},
        "text":{"verbosity":"low"},
        "tool_choice":"auto","parallel_tool_calls":true,
        "store":false,"stream":false,
        "include":["reasoning.encrypted_content"],
        "prompt_cache_key":"cache-1","client_metadata":{"thread_id":"t-1"},
        "tools":[
          {"type":"function","name":"exec_command","strict":false,
           "parameters":{"type":"object",
             "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
          {"type":"namespace","name":"multi_agent_v1","tools":[
            {"type":"function","name":"close_agent","strict":false,
             "parameters":{"type":"object","properties":{}}}]},
          {"type":"web_search","external_web_access":false}]})"));
  assert(hosted.member_str("status") == "completed");
  const auto hosted_call = server.backend->LastCall();
  assert(hosted_call.chat.tools.size() == 2 &&
         hosted_call.chat.tools[0].name == "exec_command" &&
         hosted_call.chat.tools[1].name == "close_agent");
  assert(hosted_call.chat.reasoning.enabled == true &&
         hosted_call.chat.reasoning.effort == gufo::ReasoningEffort::kMedium);
  assert(hosted_call.chat.messages.size() == 3 &&
         hosted_call.chat.messages[0].role ==
             gufo::tokenization::ChatRole::kSystem &&
         hosted_call.chat.messages[0].content == "You are a coding agent." &&
         hosted_call.chat.messages[2].content == "say hi");

  // Responses replays function calls and their outputs between turns under the
  // same call_id; the adapter folds them back into the prompt.
  const auto tool_replay = response_body(server.Post("/v1/responses",
                                                     R"({"input":[
          {"type":"message","role":"user","content":[
            {"type":"input_text","text":"list files"}]},
          {"type":"function_call","id":"fc_1","call_id":"call_1",
           "name":"exec_command","arguments":"{\"cmd\":\"ls\"}",
           "status":"completed"},
          {"type":"function_call_output","call_id":"call_1","output":"file.txt"},
          {"type":"message","role":"assistant","content":[
            {"type":"output_text","text":"Here are the files.",
             "annotations":[]}]}]})"));
  assert(tool_replay.member_str("status") == "completed");
  const auto tool_messages = server.backend->LastCall().chat.messages;
  assert(tool_messages.size() == 4);
  assert(tool_messages[1].role == gufo::tokenization::ChatRole::kAssistant &&
         tool_messages[1].tool_calls.size() == 1 &&
         tool_messages[1].tool_calls[0].id == "call_1" &&
         tool_messages[1].tool_calls[0].name == "exec_command");
  assert(tool_messages[2].content == "file.txt" &&
         tool_messages[3].content == "Here are the files.");

  // The include leniency is Responses-only: endpoints that run the shared
  // compatibility validator (Messages) still reject it.
  ExpectStatus(
      server.Post(
          "/v1/messages",
          R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":2,
                      "include":["x"]})"),
      400);

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");

  // Messages reports reasoning in its own block and restores replayed blocks
  // as the assistant thought.
  server.backend->SetOutput("<think>plan</think>answer");
  const auto thinking = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"enabled","budget_tokens":1024}})"));
  const auto blocks = thinking.find("content")->items();
  assert(blocks.size() == 2 && blocks[0].member_str("type") == "thinking" &&
         blocks[0].member_str("thinking") == "plan" &&
         blocks[0].contains("signature") &&
         blocks[1].member_str("type") == "text" &&
         blocks[1].member_str("text") == "answer");
  assert(server.backend->LastCall().chat.reasoning.enabled == true);
  response_body(server.Post("/v1/messages", R"({"messages":[
      {"role":"user","content":"hi"},
      {"role":"assistant","content":[
        {"type":"thinking","thinking":"plan","signature":""},
        {"type":"text","text":"answer"}]},
      {"role":"user","content":"next"}],
    "thinking":{"type":"disabled"}})"));
  const auto replayed = server.backend->LastCall().chat;
  assert(replayed.reasoning.enabled == false);
  assert(replayed.messages.size() == 3 &&
         replayed.messages[1].thought == "plan" &&
         replayed.messages[1].content == "answer");

  // An explicit content phase preserves requested literal reasoning tags.
  // The default automatic phase above still recognizes reasoning blocks.
  server.backend->initial_output_state_override =
      FakeBackend::InitialOutputState::kContent;
  const std::string literal_thinking = "<think>literal example</think>";
  server.backend->SetOutput(literal_thinking);
  const auto literal = response_body(
      server.Post("/v1/messages", R"({"messages":[{"role":"user","content":
        "Copy this XML exactly: <think>literal example</think>"}],
        "thinking":{"type":"disabled"}})"));
  assert(server.backend->LastCall().chat.reasoning.enabled == false);
  const auto literal_blocks = literal.find("content")->items();
  assert(literal_blocks.size() == 1 &&
         literal_blocks[0].member_str("type") == "text" &&
         literal_blocks[0].member_str("text") == literal_thinking &&
         "disabled thinking preserves literal tags as one text block");
  server.backend->initial_output_state_override.reset();

  server.backend->SetOutput("answer");
  const auto plain = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}]})"));
  assert(plain.find("content")->items().size() == 1 &&
         plain.find("content")->items()[0].member_str("text") == "answer");
  for (const auto* invalid :
       {R"({"messages":[{"role":"user","content":[
           {"type":"thinking","thinking":"plan"}]}]})",
        R"({"messages":[{"role":"user","content":"hi"}],"thinking":true})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"enabled","budget_tokens":0}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"adaptive","display":"full"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"adaptive"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"minimal"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"format":{"type":"json_schema"}}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "reasoning_effort":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":true}})"})
    ExpectStatus(server.Post("/v1/messages", invalid), 400);

  // The reasoning fields Claude Code sends on every request. Adaptive keeps
  // the server's thinking default and the reasoning is still returned.
  server.backend->SetOutput("<think>plan</think>answer");
  for (const bool server_thinking : {false, true}) {
    server.backend->reasoning.enabled = server_thinking;
    server.backend->reasoning.effort = gufo::ReasoningEffort::kLow;
    const auto adaptive = response_body(
        server.Post("/v1/messages", R"({"max_tokens":256,"stream":false,
            "thinking":{"type":"adaptive","display":"omitted"},
            "output_config":{"effort":"xhigh"},
            "messages":[{"role":"user","content":"hi"}]})"));
    if (server_thinking)
      assert(adaptive.find("content")->items()[0].member_str("thinking") ==
             "plan");
    const auto reasoning = server.backend->LastCall().chat.reasoning;
    assert(reasoning.enabled == server_thinking);
    assert(reasoning.effort == gufo::ReasoningEffort::kXHigh);
  }
  server.backend->reasoning = {};
  const auto updates = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"enabled","display":"updates"}})"));
  assert(updates.find("content")->items()[0].member_str("thinking") == "plan");
  // Effort never enables thinking.
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"disabled"},"output_config":{"effort":"low"}})"));
  assert(server.backend->LastCall().chat.reasoning.enabled == false);
  assert(server.backend->LastCall().chat.reasoning.effort ==
         gufo::ReasoningEffort::kLow);

  // Messages tools map onto the Chat tool path: declarations, tool_choice,
  // tool_use output and replayed tool_use/tool_result history.
  const std::string weather_tool = R"([{"name":"get_weather",
      "description":"Get weather","input_schema":{"type":"object",
      "properties":{"city":{"type":"string"}},"required":["city"]}}])";
  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto called = response_body(server.Post(
      "/v1/messages", R"({"max_tokens":64,"tool_choice":{"type":"any"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}"));
  assert(called.member_str("stop_reason") == "tool_use");
  const auto call_blocks = called.find("content")->items();
  assert(call_blocks.size() == 1 &&
         call_blocks[0].member_str("type") == "tool_use" &&
         !call_blocks[0].member_str("id").empty() &&
         call_blocks[0].member_str("name") == "get_weather" &&
         call_blocks[0].find("input")->member_str("city") == "Rome");
  const auto declared = server.backend->LastCall().chat;
  assert(declared.tools.size() == 1 &&
         declared.tools[0].name == "get_weather" &&
         declared.tools[0].description == "Get weather" &&
         declared.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kRequired &&
         declared.constrained_tools);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"tool","name":"get_weather"},
          "tools":)" + weather_tool +
                                "}"));
  assert(server.backend->LastCall().chat.forced_tool_name == "get_weather" &&
         !server.backend->LastCall().chat.parallel_tool_calls);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"auto","disable_parallel_tool_use":true},
          "tools":)" + weather_tool +
                                "}"));
  assert(server.backend->LastCall().chat.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kAuto &&
         !server.backend->LastCall().chat.parallel_tool_calls);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"none"},"tools":)" +
                                weather_tool + "}"));
  assert(server.backend->LastCall().chat.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kNone &&
         !server.backend->LastCall().chat.constrained_tools);

  server.backend->SetOutput("It is sunny.");
  const auto answered = response_body(server.Post("/v1/messages", R"({
      "tools":)" + weather_tool + R"(,"messages":[
        {"role":"user","content":"weather in Rome?"},
        {"role":"assistant","content":[
          {"type":"thinking","thinking":"look it up","signature":""},
          {"type":"text","text":"Checking."},
          {"type":"tool_use","id":"toolu_1","name":"get_weather",
           "input":{"city":"Rome"}}]},
        {"role":"user","content":[
          {"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
          {"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
           "content":[{"type":"text","text":"timeout"}]},
          {"type":"text","text":"thanks"}]}]})"));
  assert(answered.member_str("stop_reason") == "end_turn" &&
         answered.find("content")->items()[0].member_str("text") ==
             "It is sunny.");
  const auto history = server.backend->LastCall().chat.messages;
  assert(history.size() == 5);
  assert(history[1].role == gufo::tokenization::ChatRole::kAssistant &&
         history[1].thought == "look it up" &&
         history[1].content == "Checking." &&
         history[1].tool_calls.size() == 1 &&
         history[1].tool_calls[0].id == "toolu_1" &&
         history[1].tool_calls[0].name == "get_weather" &&
         history[1].tool_calls[0].arguments.size() == 1 &&
         history[1].tool_calls[0].arguments[0].name == "city" &&
         history[1].tool_calls[0].arguments[0].value == "Rome");
  assert(history[2].role == gufo::tokenization::ChatRole::kTool &&
         history[2].tool_call_id == "toolu_1" && history[2].content == "sunny");
  assert(history[3].role == gufo::tokenization::ChatRole::kTool &&
         history[3].tool_call_id == "toolu_2" &&
         history[3].content == "timeout");
  assert(history[4].role == gufo::tokenization::ChatRole::kUser &&
         history[4].content == "thanks");

  const auto calls_before = server.backend->calls.load();
  for (const auto& invalid : std::vector<std::string>{
           R"({"messages":[{"role":"user","content":"hi"}],"tools":{}})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tools":[{"type":"web_search_20250305","name":"web_search"}]})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tools":[{"name":"f"}]})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":{"type":"any"}})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":"auto","tools":)" +
               weather_tool + "}",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":{"type":"auto","disable_parallel_tool_use":1},
               "tools":)" +
               weather_tool + "}",
           R"({"messages":[{"role":"assistant","content":[{"type":"tool_use",
               "id":"toolu_1","name":"f","input":"{}"}]}]})",
           R"({"messages":[{"role":"assistant","content":[{"type":"tool_use",
               "name":"f","input":{}}]}]})",
           R"({"messages":[{"role":"user","content":[{"type":"tool_result",
               "content":"x"}]}]})",
           R"({"messages":[{"role":"user","content":[{"type":"tool_result",
               "tool_use_id":"toolu_1","content":[{"type":"image"}]}]}]})",
       })
    ExpectStatus(server.Post("/v1/messages", invalid), 400);
  assert(server.backend->calls == calls_before);
  // A turn cut by max_tokens reports the cut, as Chat finish_reason does,
  // even when a complete call precedes it.
  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto truncated = response_body(server.Post(
      "/v1/messages", R"({"max_tokens":1,"tool_choice":{"type":"auto"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}"));
  assert(truncated.member_str("stop_reason") == "max_tokens");
  // Without declared tools, call markup stays visible text as before.
  server.backend->SetOutput("<tool_call>leak");
  const auto undeclared = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}]})"));
  assert(undeclared.member_str("stop_reason") == "end_turn" &&
         undeclared.find("content")->items().size() == 1 &&
         undeclared.find("content")->items()[0].member_str("text") ==
             "<tool_call>leak");

  // Streamed Messages use Anthropic SSE events over the same filters.
  const auto sse_events = [](const std::string& wire) {
    std::vector<gufo::json::Value> events;
    for (auto data = wire.find("data: "); data != std::string::npos;
         data = wire.find("data: ", data + 6)) {
      const auto end = wire.find("\n\n", data);
      events.push_back(gufo::json::parse(
          std::string_view(wire).substr(data + 6, end - data - 6)));
    }
    return events;
  };
  server.backend->SetOutput("<think>plan</think>answer");
  const auto streamed_text =
      server.Post("/v1/messages", R"({"stream":true,"max_tokens":64,
          "thinking":{"type":"enabled","budget_tokens":1024},
          "messages":[{"role":"user","content":"hi"}]})");
  ExpectStatus(streamed_text, 200);
  assert(streamed_text.find("text/event-stream") != std::string::npos &&
         streamed_text.find("event: message_start\n") != std::string::npos);
  std::vector<std::string> types;
  std::string thought, answer;
  const auto text_events = sse_events(streamed_text);
  for (const auto& event : text_events) {
    types.push_back(event.member_str("type"));
    if (const auto* delta = event.find("delta");
        delta && delta->member_str("type") == "thinking_delta")
      thought += delta->member_str("thinking");
    else if (delta && delta->member_str("type") == "text_delta")
      answer += delta->member_str("text");
  }
  assert(thought == "plan" && answer == "answer");
  // Thinking blocks keep the trimmed reasoning Messages reported before
  // streaming, buffered and streamed alike.
  server.backend->SetOutput("<think>\nplan\n\nmore\n</think>\n\nanswer");
  for (const bool stream : {false, true}) {
    const auto framed =
        server.Post("/v1/messages",
                    std::string(R"({"max_tokens":64,"stream":)") +
                        (stream ? "true" : "false") +
                        R"(,"thinking":{"type":"enabled","budget_tokens":1024},
            "messages":[{"role":"user","content":"hi"}]})");
    ExpectStatus(framed, 200);
    std::string framed_thought;
    if (stream) {
      for (const auto& event : sse_events(framed))
        if (const auto* delta = event.find("delta");
            delta && delta->member_str("type") == "thinking_delta")
          framed_thought += delta->member_str("thinking");
    } else {
      framed_thought =
          response_body(framed).find("content")->items()[0].member_str(
              "thinking");
    }
    assert(framed_thought == "plan\n\nmore");
  }
  assert(types.front() == "message_start" && types.back() == "message_stop");
  assert(text_events[1].find("content_block")->member_str("type") ==
             "thinking" &&
         text_events[1].member_size("index") == 0);
  const auto& text_done = text_events[text_events.size() - 2];
  assert(text_done.member_str("type") == "message_delta" &&
         text_done.find("delta")->member_str("stop_reason") == "end_turn" &&
         text_done.find("usage")->member_size("input_tokens") == 2 &&
         text_done.find("usage")->member_size("cache_read_input_tokens") == 8);
  assert(std::ranges::count(types, "content_block_start") == 2 &&
         std::ranges::count(types, "content_block_stop") == 2);

  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto streamed_tool = server.Post(
      "/v1/messages", R"({"stream":true,"tool_choice":{"type":"any"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}");
  ExpectStatus(streamed_tool, 200);
  assert(streamed_tool.find("<tool_call>") == std::string::npos);
  bool tool_started = false;
  std::string partial_json, stop_reason;
  for (const auto& event : sse_events(streamed_tool)) {
    if (const auto* block = event.find("content_block"))
      tool_started |= block->member_str("type") == "tool_use" &&
                      block->member_str("name") == "get_weather";
    if (const auto* delta = event.find("delta")) {
      if (delta->member_str("type") == "input_json_delta")
        partial_json += delta->member_str("partial_json");
      if (event.member_str("type") == "message_delta")
        stop_reason = delta->member_str("stop_reason");
    }
  }
  assert(tool_started && stop_reason == "tool_use" &&
         gufo::json::parse(partial_json).member_str("city") == "Rome");

  server.backend->failure = 6;
  const auto failed_messages = server.Post(
      "/v1/messages",
      R"({"stream":true,"messages":[{"role":"user","content":"hi"}]})");
  ExpectStatus(failed_messages, 200);  // Fake backend fails after headers.
  const auto failed_events = sse_events(failed_messages);
  assert(failed_events.back().member_str("type") == "error" &&
         failed_events.back().find("error")->member_str("type") ==
             "api_error" &&
         failed_messages.find("event: message_stop") == std::string::npos);
  server.backend->failure = 0;
  server.backend->SetOutput("ok");
}

void TestModelInputModalities() {
  RunningServer server;
  for (const auto* endpoint : {"/v1/chat/completions", "/v1/completions",
                               "/v1/responses", "/v1/messages"}) {
    for (const auto* name : {"test", "bubo", "some-other-model"}) {
      auto body = gufo::json::Value::object();
      body["model"] = name;
      body[std::string_view(endpoint) == "/v1/responses" ? "max_output_tokens"
                                                         : "max_tokens"] = 4;
      if (std::string_view(endpoint) == "/v1/completions")
        body["prompt"] = "hello";
      else if (std::string_view(endpoint) == "/v1/responses")
        body["input"] = "hello";
      else
        body["messages"] =
            gufo::json::parse(R"([{"role":"user","content":"hello"}])");
      const auto response = server.Post(endpoint, body.dump());
      const bool rejected = std::string_view(name) == "some-other-model";
      ExpectStatus(response, rejected ? 404 : 200);
      if (rejected) {
        assert(response.find("model_not_found") != std::string::npos);
        assert(response.find("this process serves 'test'") !=
               std::string::npos);
        assert(response.find("; this process serves") == std::string::npos);
      }
    }
  }
  // Keep the same model ID: capability follows the loaded backend, not its
  // name.
  for (const bool images : {false, true, false}) {
    server.backend->image_support = images;
    const auto response = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 200);
    const auto listing =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(listing.member_str("object") == "list");
    const auto& models = listing.find("data")->items();
    assert(models.size() == 2);
    assert(models[1].member_str("id") == "bubo");
    assert(models[1].member_str("owned_by") == "bubo");
    const auto& model = models[0];
    assert(model.member_str("id") == "test");
    assert(model.member_str("owned_by") == "bubo");
    assert(model.member_size("context_length") == 65536);
    assert(model.member_size("max_model_len") == 65536);
    assert(model.find("meta") != nullptr &&
           model.find("meta")->member_size("n_ctx") == 65536);
    const auto* architecture = model.find("architecture");
    assert(architecture != nullptr);
    const auto* modalities = architecture->find("input_modalities");
    assert(modalities != nullptr);
    assert(modalities->dump() ==
           (images ? R"(["text","image"])" : R"(["text"])"));
  }
  server.backend->served_name = "bubo";
  const auto response = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  const auto listing =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  assert(listing.find("data")->size() == 1);
  assert(listing.find("data")->items()[0].member_str("id") == "bubo");
}

void TestLlamaProps() {
  RunningServer server;
  const auto body_of = [](const std::string& response) {
    return gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  // No `?model=`: the server describes the model it serves.
  const auto plain = server.Send("GET /props HTTP/1.1\r\n\r\n");
  ExpectStatus(plain, 200);
  const auto props = body_of(plain);
  assert(props.member_str("model") == "test");
  assert(props.find("template") != nullptr);
  assert(props.find("model_info") != nullptr);
  const auto* settings = props.find("default_generation_settings");
  assert(settings != nullptr && settings->member_size("n_ctx") == 65536);
  assert(props.member_size("total_slots") == 1);
  const auto* modalities = props.find("modalities");
  assert(modalities != nullptr && modalities->find("vision") != nullptr &&
         !modalities->find("vision")->as_bool(true));

  // A named model is echoed as before.
  const auto named = server.Send("GET /props?model=other HTTP/1.1\r\n\r\n");
  ExpectStatus(named, 200);
  assert(body_of(named).member_str("model") == "other");

  // The values follow the backend.
  server.backend->image_support = true;
  server.backend->sessions.resize(3);
  const auto wide = server.Send("GET /props HTTP/1.1\r\n\r\n");
  ExpectStatus(wide, 200);
  const auto wide_props = body_of(wide);
  assert(wide_props.member_size("total_slots") == 3);
  assert(wide_props.find("modalities")->find("vision")->as_bool(false));
}

void TestRawCompletionStreaming() {
  RunningServer server;
  using gufo::json::parse;
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  ExpectStatus(models, 200);
  const auto listing = parse(models.substr(models.find("\r\n\r\n") + 4));
  assert(listing.find("data")->items()[0].member_size("context_length") ==
         65536);

  const auto response = server.Post(
      "/v1/completions",
      R"({"prompt":"hello","max_tokens":256,"stream":true,"stream_options":{"include_usage":true},"ignore_eos":true})");
  ExpectStatus(response, 200);
  assert(response.find("text/event-stream") != std::string::npos);
  std::vector<gufo::json::Value> events;
  std::size_t offset = 0;
  while ((offset = response.find("data: ", offset)) != std::string::npos) {
    const auto begin = offset + 6;
    const auto end = response.find("\n\n", begin);
    assert(end != std::string::npos);
    const auto data = std::string_view(response).substr(begin, end - begin);
    if (data != "[DONE]")
      events.push_back(parse(data));
    offset = end + 2;
  }
  assert(events.size() == 3);
  const auto& content = events[0];
  assert(content.member_str("object") == "text_completion");
  assert(content.find("choices")->items().size() == 1);
  assert(content.find("choices")->items()[0].member_str("text") == "ok");
  assert(content.find("choices")->items()[0].find("finish_reason")->is_null());
  assert(content.find("usage") == nullptr);
  const auto& terminal = events[1];
  assert(terminal.find("choices")->items().size() == 1);
  assert(terminal.find("choices")->items()[0].member_str("text").empty());
  assert(terminal.find("choices")->items()[0].member_str("finish_reason") ==
         "stop");
  assert(terminal.find("usage") == nullptr);
  const auto* timings = terminal.find("timings");
  assert(timings != nullptr);
  assert(timings->member_size("prompt_n") == 2);
  assert(timings->member_double("prompt_ms") == 4);
  assert(timings->member_size("predicted_n") == 1);
  assert(timings->member_size("cache_n") == 8);
  const auto& usage = events[2];
  assert(usage.find("choices")->items().empty());
  assert(usage.find("usage")->member_size("completion_tokens") == 1);
  assert(usage.find("usage")->member_size("cached_tokens") == 8);
  assert(response.find("data: [DONE]\n\n") != std::string::npos);
  assert(server.backend->last_ignore_eos);
  assert(server.backend->LastCall().max_tokens == 256);

  const auto without_usage =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(without_usage, 200);
  assert(without_usage.find("\"timings\":") != std::string::npos);
  assert(without_usage.find("\"usage\":") == std::string::npos);

  for (const int failure : {1, 5, 6}) {
    server.backend->failure = failure;
    const std::string message =
        failure == 5 ? "generation failed" : "context exceeded";
    for (
        const auto& [path, body] : {
            std::pair{
                "/v1/completions",
                R"({"prompt":"hello","stream":true,"stream_options":{"include_usage":true}})"},
            std::pair{
                "/v1/chat/completions",
                R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
            std::pair{"/v1/responses", R"({"input":"hello","stream":true})"},
        }) {
      const auto failed = server.Post(path, body);
      ExpectStatus(failed, failure == 6 ? 200 : 500);
      assert(failed.find("\"message\":\"" + message + "\"") !=
             std::string::npos);
      if (failure != 6) {
        assert(failed.find("Content-Type: application/json") !=
               std::string::npos);
        assert(failed.find("data: ") == std::string::npos);
        assert(failed.find("\"code\":\"server_exception\"") !=
               std::string::npos);
        continue;
      }
      const bool responses = std::string_view(path) == "/v1/responses";
      assert(failed.find(responses ? "\"code\":\"server_error\""
                                   : "\"code\":\"generation_failed\"") !=
             std::string::npos);
      assert(failed.find(responses ? "event: response.failed"
                                   : "data: [DONE]\n\n") != std::string::npos);
      assert(failed.find("event: response.completed") == std::string::npos);
      assert(failed.ends_with("0\r\n\r\n"));
    }
  }
  server.backend->failure = 0;

  for (
      const auto* body : {
          R"({"prompt":"hello","stream_options":{"include_usage":true}})",
          R"({"prompt":"hello","stream":true,"stream_options":{"include_usage":"true"}})",
          R"({"prompt":"hello","stream":true,"stream_options":{"other":true}})",
          R"({"prompt":"hello","ignore_eos":1})",
          R"({"prompt":"hello","stream":true,"text":{"format":{"type":"json_object"}}})",
          R"({"prompt":"hello","stream":true,"reasoning":{"effort":"none"}})",
      })
    ExpectStatus(server.Post("/v1/completions", body), 400);

  // Raw Completions owns the fixed-length contract; every other text endpoint
  // rejects the field instead of silently generating a shorter run.
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"ignore_eos":true})"),
      400);
}

// A generic failure keeps its cause in the server log, not only in the
// client's response, whether it happens before or after streaming starts.
void TestGenerationFailureLogsCause() {
  DrainLogs();
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    RunningServer server;
    for (const int failure : {1, 6}) {
      server.backend->failure = failure;
      for (const auto* path :
           {"/v1/chat/completions", "/v1/responses", "/v1/messages"}) {
        const std::string body =
            std::string_view(path) == "/v1/chat/completions"
                ? R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"
            : std::string_view(path) == "/v1/responses"
                ? R"({"input":"hello","stream":true})"
                : R"({"messages":[{"role":"user","content":"hello"}],"stream":true})";
        ExpectStatus(server.Post(path, body), failure == 6 ? 200 : 500);
      }
    }
  }
  DrainLogs();
  std::clog.rdbuf(previous_sink);
  const std::string log = output.str();
  const auto count = [&log](std::string_view needle) {
    std::size_t found = 0;
    for (auto at = log.find(needle); at != std::string::npos;
         at = log.find(needle, at + 1))
      ++found;
    return found;
  };
  // Failures are logged by exception type, never by their text: the fake's
  // "context exceeded" stands in for request content an exception can quote.
  // Before streaming, the request line names the type with the request ID.
  assert(count("error_code=server_exception error_type=std::length_error") ==
         3);
  // Each affected adapter logs the type after streaming starts.
  assert(count("generation failed after streaming error_type=") == 3);
  assert(count("context exceeded") == 0);
}

void TestDeviceLoss() {
  std::atomic<int> hook_calls{0};
  RunningServer server({.on_device_lost = [&] { ++hook_calls; }});
  const std::string chat =
      R"({"model":"test","messages":[{"role":"user","content":"hi"}]})";

  // A failure while the device still accepts work keeps the existing contract.
  server.backend->failure = 4;
  const auto failed = server.Post("/v1/completions", R"({"prompt":"hello"})");
  ExpectStatus(failed, 500);
  assert(failed.find("\"code\":\"server_exception\"") != std::string::npos);
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  ExpectStatus(server.Send("GET /ready HTTP/1.1\r\n\r\n"), 200);
  assert(hook_calls == 0);

  // A stream has committed its status; the terminal event names the loss.
  server.backend->failure = 7;
  const auto stream =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(stream, 200);
  assert(stream.find("\"code\":\"device_lost\"") != std::string::npos);
  assert(stream.find(gufo::server::kDeviceLostMessage) != std::string::npos);
  assert(hook_calls == 1);

  server.backend->failure = 0;
  for (const std::string path : {"/health", "/v1/health", "/healthz", "/ready",
                                 "/v1/ready", "/readyz"}) {
    const auto response = server.Send("GET " + path + " HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 503);
    assert(response.find("\"status\":\"device_lost\"") != std::string::npos);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
  }
  const int calls = server.backend->calls;
  for (const auto& [path, body] :
       {std::pair<std::string, std::string>{"/v1/chat/completions", chat},
        {"/v1/completions", R"({"prompt":"hello","stream":true})"},
        {"/v1/embeddings", R"({"input":"hello"})"}}) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find("Retry-After") == std::string::npos);
  }
  assert(server.backend->calls == calls);
  // Routes that never reach the device keep answering.
  ExpectStatus(server.Send("GET /v1/models HTTP/1.1\r\n\r\n"), 200);
  ExpectStatus(server.Send("GET /metrics HTTP/1.1\r\n\r\n"), 200);
  assert(hook_calls == 1);

  // Before a response starts, the failing request itself gets the 503.
  std::atomic<int> first_hook_calls{0};
  RunningServer first({.on_device_lost = [&] { ++first_hook_calls; }});
  first.backend->failure = 3;
  for (const auto& [path, body] :
       {std::pair<std::string, std::string>{"/v1/completions",
                                            R"({"prompt":"hello"})"},
        {"/v1/chat/completions", chat}}) {
    const auto response = first.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find(gufo::server::kDeviceLostMessage) !=
           std::string::npos);
    assert(response.find("Retry-After") == std::string::npos);
  }
  assert(first.backend->calls == 1);
  assert(first_hook_calls == 1);
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions", R"({"prompt":"hello","stream":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
       std::pair{"/v1/responses", R"({"input":"hello","stream":true})"}}) {
    RunningServer early;
    early.backend->failure = 3;
    const auto response = early.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find("data: ") == std::string::npos);
  }
}

void TestDeviceLossWhileWriterBlocked() {
  std::atomic<int> hook_calls{0};
  std::binary_semaphore hook_called{0};
  std::binary_semaphore writing{0};
  std::atomic<bool> write_finished{false};
  RunningServer server({.sse_heartbeat_interval = std::chrono::milliseconds(0),
                        .on_device_lost = [&] {
                          ++hook_calls;
                          hook_called.release();
                        }});
  server.server.add("POST", "/blocked", [&](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body = [&](const auto& write) {
          const std::string chunk(8 * 1024 * 1024, 'x');
          writing.release();
          (void)write(chunk);
          write_finished = true;
        }};
  });
  const int fd = server.Connect();
  const int receive_buffer = 1024;
  assert(gufo::platform::SetSocketOption(fd, SOL_SOCKET, SO_RCVBUF,
                                         receive_buffer) == 0);
  const std::string request = "POST /blocked HTTP/1.1\r\n\r\n";
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(writing.try_acquire_for(std::chrono::seconds(2)));
  // The peer never drains the response. No health request or completed write
  // is available to trigger shutdown; the listener must observe the loss.
  server.backend->lost = true;
  assert(hook_called.try_acquire_for(std::chrono::seconds(2)));
  assert(hook_calls == 1 && !write_finished);
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
}

void TestInvalidBindSettings() {
  for (const int port : {-1, 65536}) {
    HttpServer server("127.0.0.1", port, nullptr);
    std::string error;
    assert(!server.start(&error));
    assert(!error.empty());
  }
  HttpServer invalid("bad.address", 0, nullptr);
  std::string error;
  assert(!invalid.start(&error));
  assert(!error.empty());
}

void TestCompatibilityUtf8() {
  RunningServer server;
  server.backend->SetOutput("é中😀\xE2\x94!\xF0\x9F");
  const std::string expected = "é中😀\xEF\xBF\xBD!\xEF\xBF\xBD";
  for (const auto& [path, input] : {
           std::pair{"/v1/completions", R"({"prompt":"hi"})"},
           std::pair{"/v1/responses", R"({"input":"hi"})"},
           std::pair{"/v1/messages",
                     R"({"messages":[{"role":"user","content":"hi"}]})"},
           std::pair{"/completion", R"({"prompt":"hi"})"},
       }) {
    const auto response = server.Post(path, input);
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    std::string text;
    if (std::string_view(path) == "/v1/completions")
      text = output.find("choices")->items()[0].member_str("text");
    else if (std::string_view(path) == "/v1/responses")
      text = output.find("output")
                 ->items()[0]
                 .find("content")
                 ->items()[0]
                 .member_str("text");
    else if (std::string_view(path) == "/v1/messages")
      text = output.find("content")->items()[0].member_str("text");
    else
      text = output.member_str("content");
    assert(text == expected);
  }
}

void TestQueryParameters() {
  gufo::server::HttpRequest request;
  request.query = "notafter=wrong&note=after=wrong&after=right+value%26x";
  assert(request.query_param("after") == "right value&x");
  request.query = "notafter=wrong&note=after=wrong";
  assert(request.query_param("after").empty());
  request.query = "%61fter=encoded&broken=%xz&empty";
  assert(request.query_param("after") == "encoded");
  assert(request.query_param("broken") == "%xz");
  assert(request.query_param("empty").empty());
}

void TestPeerDisconnect() {
  RunningServer server;
  server.backend->wait_for_disconnect = true;
  const int fd = server.Connect();
  const std::string body = R"({"prompt":"hi","max_tokens":128})";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
  gufo::platform::CloseSocket(fd);
  assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
  assert(server.backend->disconnected);
}

void TestCompatibilityStopSequences() {
  RunningServer server;
  server.backend->forced_stop_sequence = "END";
  for (
      const auto& [path, body] : {
          std::pair{"/v1/completions", R"({"prompt":"hello","stop":"END"})"},
          std::pair{"/completion", R"({"prompt":"hello","stop":["END"]})"},
          std::pair{
              "/v1/messages",
              R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"]})"},
      }) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    assert(server.backend->LastCall().stop_sequences ==
           std::vector<std::string>{"END"});
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    if (std::string_view(path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "stop_sequence");
      assert(output.member_str("stop_sequence") == "END");
    } else if (std::string_view(path) == "/completion") {
      assert(output.find("stopped_word")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
      assert(output.member_str("stopping_word") == "END");
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "stop");
    }
  }
  const auto before = server.backend->calls.load();
  for (const auto* stop : {"null", "\"END\"", "[null]", "[\"\"]", "true"}) {
    auto body = gufo::json::parse(
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})");
    body["stop_sequences"] = gufo::json::parse(stop);
    ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  }
  ExpectStatus(server.Post("/v1/responses", R"({"input":"hi","stop":"END"})"),
               400);
  assert(server.backend->calls == before);
  auto body = gufo::json::parse(
      R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":[]})");
  for (int i = 0; i < 65; ++i)
    body["stop_sequences"].push_back(std::to_string(i));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  body["stop_sequences"] = gufo::json::Value::array();
  for (int i = 0; i < 5; ++i)
    body["stop_sequences"].push_back(std::string(4096, 'x'));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  server.backend->forced_stop_sequence.clear();
  for (const int limit : {1, 32}) {
    body["stop_sequences"] = gufo::json::Value::array();
    body["max_tokens"] = limit;
    const auto response = server.Post("/v1/messages", body.dump());
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(output.member_str("stop_reason") ==
           (limit == 1 ? "max_tokens" : "end_turn"));
    assert(output.find("stop_sequence")->is_null());
  }
}

void TestCompatibilityThinkingDefaults() {
  RunningServer server;
  for (const bool enabled : {false, true}) {
    server.backend->reasoning = {
        .enabled = enabled,
        .effort = gufo::ReasoningEffort::kHigh,
        .preserve_thinking = true,
    };
    for (
        const auto& [path, body] : {
            std::pair{"/v1/responses", R"({"input":"hello"})"},
            std::pair{
                "/v1/messages",
                R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})"},
        }) {
      ExpectStatus(server.Post(path, body), 200);
      const auto reasoning = server.backend->LastCall().chat.reasoning;
      assert(reasoning.enabled == enabled);
      assert(reasoning.effort == gufo::ReasoningEffort::kHigh);
      assert(reasoning.preserve_thinking == true);
    }
  }
}

void TestResponseSamplingDefaults() {
  RunningServer server;
  using gufo::sampling::TextModelPreset;
  for (const auto model :
       {TextModelPreset::kQwen38, TextModelPreset::kDeepSeekV4Flash}) {
    server.backend->defaults.model = model;
    for (const bool server_thinking : {false, true}) {
      server.backend->reasoning.enabled = server_thinking;
      for (const char* effort :
           {"null", "\"none\"", "\"minimal\"", "\"low\"", "\"medium\"",
            "\"high\"", "\"xhigh\"", "\"max\""}) {
        auto body = gufo::json::parse(
            R"({"input":"hello","max_output_tokens":1,"temperature":null,"top_p":null,"presence_penalty":null,"reasoning":{},"text":{"format":{"type":"json_object"}}})");
        body["reasoning"]["effort"] = gufo::json::parse(effort);
        const bool thinking = std::string_view(effort) == "null"
                                  ? server_thinking
                                  : std::string_view(effort) != "\"none\"";
        const bool qwen_off = model == TextModelPreset::kQwen38 && !thinking;
        server.backend->defaults.supplied = {};
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        auto call = server.backend->LastCall();
        assert(call.chat.reasoning.enabled == thinking);
        assert(call.chat.response_format);
        assert(call.sampling.temperature == (qwen_off ? 0.7F : 1.0F));
        assert(call.sampling.top_p == (qwen_off ? 0.8F : 0.95F));
        assert(call.sampling.presence_penalty == (qwen_off ? 1.5F : 0.0F));
        assert(call.sampling.top_k ==
               (model == TextModelPreset::kQwen38 ? 20 : 0));
        // Explicit CLI values survive reasoning changes and SDK nulls.
        server.backend->defaults.sampling.temperature = 0.25F;
        server.backend->defaults.sampling.top_k = 0;
        server.backend->defaults.supplied = {.temperature = true,
                                             .top_k = true};
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        call = server.backend->LastCall();
        assert(call.sampling.temperature == 0.25F && call.sampling.top_k == 0);
        assert(call.sampling.top_p == (qwen_off ? 0.8F : 0.95F));
        // Per-request zero overrides both CLI and model values, also streamed.
        body["temperature"] = 0;
        body["presence_penalty"] = 0;
        body["top_k"] = 3;
        body["stream"] = true;
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        call = server.backend->LastCall();
        assert(call.sampling.temperature == 0 && call.sampling.top_k == 3);
        assert(call.sampling.presence_penalty == 0);
      }
    }
  }
  // Responses must preserve the same request-owned cache control as Chat.
  auto body = gufo::json::parse(R"({"input":"hello","max_output_tokens":1})");
  ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
  assert(server.backend->LastCall().chat.cache_prompt);
  for (const bool stream : {false, true}) {
    body["stream"] = stream;
    for (const bool cache : {false, true}) {
      body["cache_prompt"] = cache;
      ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
      assert(server.backend->LastCall().chat.cache_prompt == cache);
    }
  }
  // As in llama-server, null keeps the default.
  body["cache_prompt"] = false;
  ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
  body["cache_prompt"] = gufo::json::Value();
  ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
  assert(server.backend->LastCall().chat.cache_prompt);
  for (const char* invalid : {"0", "\"false\""}) {
    body["cache_prompt"] = gufo::json::parse(invalid);
    const int calls = server.backend->calls;
    ExpectStatus(server.Post("/v1/responses", body.dump()), 400);
    assert(server.backend->calls == calls);
  }
}

void TestStreamingFraming() {
  RunningServer server;
  const std::string chunks = std::string("3\r\na\0b\r\n", 8) + "3\r\nend\r\n";
  for (const std::string body : {"", "fail", "reported"}) {
    const auto response = server.Post("/stream", body);
    ExpectStatus(response, 200);
    assert(response.find("Transfer-Encoding: chunked\r\n") !=
           std::string::npos);
    assert(response.substr(response.find("\r\n\r\n") + 4) ==
           chunks + (body == "fail" ? "" : "0\r\n\r\n"));
  }
  const auto thrown = server.Post("/stream-error", "");
  assert(thrown.substr(thrown.find("\r\n\r\n") + 4) == "b\r\nfirst chunk\r\n");
  // Existing HTTP/1.0 clients retain close-delimited framing.
  const auto legacy = server.Send("POST /stream HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.substr(legacy.find("\r\n\r\n") + 4) ==
         std::string("a\0bend", 6));
}

void TestSseHeartbeat() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(20)});
  const auto response = server.Post("/sse-idle", "");
  ExpectStatus(response, 200);
  const auto first = response.find("data: first\n\n");
  const auto ping = response.find(": ping\n\n", first);
  const auto second_ping = response.find(": ping\n\n", ping + 1);
  const auto last = response.find("data: last\n\n", second_ping);
  assert(first != std::string::npos);
  assert(ping != std::string::npos);
  assert(second_ping != std::string::npos);
  assert(last != std::string::npos);
  assert(response.find("data: [DONE]\n\n", last) != std::string::npos);
  assert(response.ends_with("0\r\n\r\n"));

  const auto legacy = server.Send("POST /sse-idle HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.find(": ping\n\n") != std::string::npos);
  assert(legacy.ends_with("data: [DONE]\n\n"));

  RunningServer disabled({.sse_heartbeat_interval = {}});
  assert(disabled.Post("/sse-idle", "").find(": ping") == std::string::npos);
}

void TestDeferredStreamHeaders() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(5)});
  server.server.add("POST", "/early-failure", [](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body =
            [](const auto&) {
              std::this_thread::sleep_for(std::chrono::milliseconds(30));
              throw std::runtime_error("early generation failure");
            },
        .defer_stream_headers = true};
  });
  const auto failed = server.Post("/early-failure", "");
  ExpectStatus(failed, 500);
  assert(failed.find(": ping") == std::string::npos);
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions", R"({"prompt":"hello","stream":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
       std::pair{"/v1/responses", R"({"input":"hello","stream":true})"}}) {
    server.backend->failure =
        8;  // Successful termination without a token callback.
    server.backend->SetOutput("");
    const auto empty = server.Post(path, body);
    ExpectStatus(empty, 200);
    assert(empty.ends_with("0\r\n\r\n"));
  }
  // Opt-in progress deliberately starts the stream before tokens exist.
  server.backend->progress = {{.total = 100, .processed = 10}};
  server.backend->failure = 4;
  const auto progress =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":true})");
  ExpectStatus(progress, 200);
  assert(progress.find("prompt_progress") != std::string::npos);
  assert(progress.find("\"code\":\"generation_failed\"") != std::string::npos);
  server.backend->progress.clear();
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions",
                 R"({"prompt":"hello","stream":true,"return_progress":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"return_progress":true})"},
       std::pair{
           "/v1/responses",
           R"({"input":"hello","stream":true,"return_progress":true})"}}) {
    const auto failure = server.Post(path, body);
    ExpectStatus(failure, 200);
    assert(failure.find(std::string_view(path) == "/v1/responses"
                            ? "\"code\":\"server_error\""
                            : "\"code\":\"generation_failed\"") !=
           std::string::npos);
    assert(failure.ends_with("0\r\n\r\n"));
  }
}

void TestAdmittedStreamHeaders() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(5)});
  server.backend->admit = true;
  const std::pair<const char*, const char*> requests[] = {
      {"/v1/completions", R"({"prompt":"hello","stream":true})"},
      {"/v1/chat/completions",
       R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
      {"/v1/responses", R"({"input":"hello","stream":true})"}};
  // Admission commits headers, so keepalives cover prefill before a token.
  server.backend->token_delay = std::chrono::milliseconds(40);
  for (const auto& [path, body] : requests) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    // Responses streams keep alive with a response.in_progress event (the lifecycle one is the first); the others with the comment.
    const bool responses = std::string_view(path) == "/v1/responses";
    const auto lifecycle = responses ? response.find("event: response.in_progress") : std::string::npos;
    const auto ping = responses ? response.find("event: response.in_progress", lifecycle + 1) : response.find(": ping\n\n");
    assert(ping != std::string::npos);
    assert(!responses || response.find(": ping") == std::string::npos);
    const auto token = response.find(R"("ok")");
    assert(token != std::string::npos);
    assert(ping < token);
    assert(response.ends_with("0\r\n\r\n"));
  }
  // After admission, failures before any token are terminal SSE errors.
  for (const auto& [path, body] : requests) {
    RunningServer lost;
    lost.backend->admit = true;
    lost.backend->failure = 3;
    const auto failed = lost.Post(path, body);
    ExpectStatus(failed, 200);
    assert(failed.find(gufo::server::kDeviceLostMessage) != std::string::npos);
    assert(failed.find("data: ") != std::string::npos);
    assert(failed.ends_with("0\r\n\r\n"));
  }
}

void TestSseHeartbeatShutdown() {
  // Quick completion and exceptions can request stop while the heartbeat
  // thread is entering its wait. Cleanup must not wait for this deadline:
  // Send() has a three-second socket timeout.
  RunningServer server({.sse_heartbeat_interval = std::chrono::seconds(30)});
  std::vector<std::jthread> clients;
  for (int client = 0; client < 8; ++client) {
    clients.emplace_back([&] {
      for (int request = 0; request < 16; ++request) {
        const bool fail = request % 2 != 0;
        const auto response = server.Post("/sse-finish", fail ? "fail" : "");
        ExpectStatus(response, 200);
        assert(response.find("data: first\n\n") != std::string::npos);
        assert(response.find(": ping") == std::string::npos);
        assert(response.ends_with("0\r\n\r\n") == !fail);
        assert((response.find("data: [DONE]\n\n") != std::string::npos) ==
               !fail);
      }
    });
  }
}

void TestSignalShutdown() {
  // Process signals must never terminate the test runner itself. Prove that
  // both idle listeners and active generation return through normal cleanup.
  for (const int signal : {SIGINT, SIGTERM}) {
    for (const bool active : {false, true}) {
      const auto run_case = [&] {
        RunningServer server({}, true);
        // Accepting a request proves run() installed its handlers.
        ExpectStatus(server.Post("/echo", "ready"), 200);
        int fd = -1;
        if (active) {
          server.backend->wait_for_disconnect = true;
          fd = server.Connect();
          const std::string body = R"({"prompt":"hello"})";
          const std::string request =
              "POST /v1/completions HTTP/1.1\r\nContent-Length: " +
              std::to_string(body.size()) + "\r\n\r\n" + body;
          assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
                 static_cast<ssize_t>(request.size()));
          assert(
              server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
        }
#ifdef _WIN32
        assert(::raise(signal) == 0);
#else
        assert(::kill(::getpid(), signal) == 0);
#endif
        assert(server.run_finished.try_acquire_for(std::chrono::seconds(2)));
        if (active) {
          assert(server.backend->disconnected);
          gufo::platform::CloseSocket(fd);
        }
      };
#ifdef _WIN32
      // No fork() on Windows: run in-process. The CRT calls the handler
      // synchronously on this thread, and without one SIGINT/SIGTERM would
      // end the runner, which fails the test just as loudly.
      run_case();
#else
      const pid_t child = ::fork();
      assert(child >= 0);
      if (child == 0) {
        ::alarm(5);
        run_case();
        ::_exit(0);
      }
      int status = 0;
      assert(::waitpid(child, &status, 0) == child);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
    }
  }
}

}  // namespace

void TestRawCompletionCachePrompt() {
  RunningServer server;
  for (const auto* path : {"/v1/completions", "/completion"}) {
    ExpectStatus(
        server.Post(path, R"({"prompt":"hello","cache_prompt":false})"), 200);
    assert(!server.backend->last_cache_prompt);
    ExpectStatus(server.Post(path, R"({"prompt":"hello"})"), 200);
    assert(server.backend->last_cache_prompt);
    ExpectStatus(
        server.Post(path, R"({"prompt":"hello","cache_prompt":false})"), 200);
    ExpectStatus(server.Post(path, R"({"prompt":"hello","cache_prompt":true})"),
                 200);
    assert(server.backend->last_cache_prompt);
    // As in llama-server, null keeps the default.
    ExpectStatus(
        server.Post(path, R"({"prompt":"hello","cache_prompt":false})"), 200);
    ExpectStatus(server.Post(path, R"({"prompt":"hello","cache_prompt":null})"),
                 200);
    assert(server.backend->last_cache_prompt);
    const auto before = server.backend->calls.load();
    for (const auto* value : {"\"false\"", "0"}) {
      const auto rejected = server.Post(
          path,
          std::string(R"({"prompt":"hello","cache_prompt":)") + value + "}");
      ExpectStatus(rejected, 400);
      assert(rejected.find("'cache_prompt' must be a boolean") !=
             std::string::npos);
    }
    assert(server.backend->calls == before);
  }
  server.backend->logprob_support = true;
  for (const bool stream : {false, true}) {
    auto scored = gufo::json::parse(
        R"({"prompt":"hello","max_tokens":2,"logprobs":2,"cache_prompt":false})");
    scored["stream"] = stream;
    ExpectStatus(server.Post("/v1/completions", scored.dump()), 200);
    assert(!server.backend->last_cache_prompt);
  }
}

void TestRawCompletionPromptProgress() {
  RunningServer server;
  server.backend->progress = {
      {.total = 4, .cache = 1, .processed = 4, .time_ms = 2}};
  const auto response =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":true})");
  ExpectStatus(response, 200);
  const auto progress = response.find(
      R"("choices":[{"text":"","index":0,"logprobs":null,"finish_reason":null}],)"
      R"("prompt_progress":{"total":4,"cache":1,"processed":4,"time_ms":2}})");
  assert(progress != std::string::npos);
  assert(progress < response.find(R"("text":"ok")"));

  const auto plain =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(plain, 200);
  assert(plain.find("prompt_progress") == std::string::npos);
  const auto null_progress =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":null})");
  ExpectStatus(null_progress, 200);
  assert(null_progress.find("prompt_progress") == std::string::npos);
  const auto buffered = server.Post(
      "/v1/completions", R"({"prompt":"hello","return_progress":true})");
  ExpectStatus(buffered, 200);
  assert(buffered.find("prompt_progress") == std::string::npos);
  ExpectStatus(server.Post("/v1/completions",
                           R"({"prompt":"hello","return_progress":1})"),
               400);
}

std::vector<gufo::json::Value> DebugHooksAllSseData(
    const std::string& response) {
  std::vector<gufo::json::Value> frames;
  for (std::size_t start = 0;
       (start = response.find("data: ", start)) != std::string::npos;) {
    start += 6;
    const auto end = response.find('\n', start);
    const auto payload = response.substr(start, end - start);
    if (payload != "[DONE]")
      frames.push_back(gufo::json::parse(payload));
    start = end;
  }
  return frames;
}

gufo::json::Value DebugHooksHttpResponseBody(const std::string& response) {
  return gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
}

gufo::json::Value DebugHooksHttpSseChunk(const std::string& response) {
  const auto start = response.find("data: ");
  assert(start != std::string::npos);
  const auto end = response.find("\n\n", start);
  assert(end != std::string::npos);
  return gufo::json::parse(response.substr(start + 6, end - start - 6));
}

gufo::json::Value DebugHooksHttpResponseEvent(const std::string& response,
                                            const std::string& event) {
  const auto wire = ChunkedBody(response);
  const auto position = wire.find("event: " + event + "\n");
  assert(position != std::string::npos);
  const auto start = wire.find("data: ", position);
  const auto end = wire.find("\n\n", start);
  assert(start != std::string::npos && end != std::string::npos);
  return gufo::json::parse(wire.substr(start + 6, end - start - 6));
}

void DebugHooksHttpTestSettingsEcho() {
  auto settings = std::make_shared<gufo::json::Value>(gufo::json::Value::object());
  (*settings)["version"] = "fake-settings";
  (*settings)["revision"] = "fake-revision";
  (*settings)["resolved"]["context"] = 512;
  RunningServer enabled({.debug_hooks = true, .debug_settings = settings});
  const auto get = [&](const std::string& path) {
    return enabled.Send("GET " + path + " HTTP/1.1\r\n\r\n");
  };
  const auto diagnostic = get("/debug/settings");
  ExpectStatus(diagnostic, 200);
  assert(DebugHooksHttpResponseBody(diagnostic).dump() == settings->dump());
  const auto models = get("/v1/models");
  ExpectStatus(models, 200);
  assert(DebugHooksHttpResponseBody(models).find("data")->items()[0]
             .find("gufo")->find("settings")->dump() == settings->dump());
  for (
      const auto& [path, input] :
      {std::pair{"/v1/completions", R"({"prompt":"x","max_tokens":2})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":2})"},
       std::pair{"/v1/responses", R"({"input":"x","max_output_tokens":2})"}}) {
    const auto response = enabled.Post(path, input);
    ExpectStatus(response, 200);
    assert(DebugHooksHttpResponseBody(response).find("gufo")->find("settings")->dump() ==
           settings->dump());
    auto body = gufo::json::parse(input);
    body["stream"] = true;
    const auto streaming = enabled.Post(path, body.dump());
    ExpectStatus(streaming, 200);
    const auto chunk = DebugHooksHttpSseChunk(streaming);
    assert(chunk.find("gufo")->find("settings")->dump() == settings->dump());
    if (std::string_view(path) == "/v1/responses")
      assert(chunk.find("response")->find("gufo")->find("settings")->dump() == settings->dump());
  }
  RunningServer disabled;
  ExpectStatus(disabled.Send("GET /debug/settings HTTP/1.1\r\n\r\n"), 404);
  const auto uninstrumented = DebugHooksHttpResponseBody(
      disabled.Post("/v1/completions", R"({"prompt":"x","max_tokens":2})"));
  assert(!uninstrumented.find("gufo") || !uninstrumented.find("gufo")->contains("settings"));
  RunningServer nonloopback({.debug_hooks = true, .debug_settings = settings},
                           false, "0.0.0.0");
  ExpectStatus(nonloopback.Send("GET /debug/settings HTTP/1.1\r\n\r\n"), 404);
}

void DebugHooksHttpTestResponsesLogprobs() {
  RunningServer server;
  const auto input = gufo::json::parse(
      R"({"input":"x","max_output_tokens":2,"temperature":0,
          "reasoning":{"effort":"none"},
          "include":["message.output_text.logprobs"],"top_logprobs":2})");
  // Base ReadCompatibilityOptions rejects top_logprobs before examining include.
  const auto unsupported = server.Post("/v1/responses", input.dump());
  ExpectStatus(unsupported, 400);
  assert(unsupported.substr(unsupported.find("\r\n\r\n") + 4) ==
         R"({"error":{"message":"request field 'top_logprobs' is not supported on this endpoint","type":"invalid_request_error","code":"invalid_request"}})");
  assert(server.backend->calls == 0);
  server.backend->logprob_support = true;
  server.backend->initial_output_state_override = TextGenerationBackend::InitialOutputState::kContent;
  const auto response = server.Post("/v1/responses", input.dump());
  ExpectStatus(response, 200);
  const auto body = DebugHooksHttpResponseBody(response);
  const auto& message = body.find("output")->items()[0];
  const auto& part = message.find("content")->items()[0];
  assert(message.member_str("type") == "message" && part.member_str("type") == "output_text");
  assert(part.member_str("text") == "\xC3\xA9x");
  const auto& scores = part.find("logprobs")->items();
  assert(scores.size() == 2 && scores[0].find("top_logprobs")->size() == 2);
  assert(scores[0].member_str("token") == "\xC3\xA9");
  assert(scores[0].find("bytes")->items()[0].as_size() == 195);
  assert(std::abs(scores[0].member_double("logprob") + std::log(1 + std::exp(-2.0))) < 1e-12);
  auto streaming_input = input;
  streaming_input["stream"] = true;
  const auto streaming = server.Post("/v1/responses", streaming_input.dump());
  ExpectStatus(streaming, 200);
  const auto delta = DebugHooksHttpResponseEvent(streaming, "response.output_text.delta");
  const auto done = DebugHooksHttpResponseEvent(streaming, "response.output_text.done");
  const auto part_done = DebugHooksHttpResponseEvent(streaming, "response.content_part.done");
  assert(delta.member_str("delta") == "\xC3\xA9" &&
         delta.find("logprobs")->size() == 1);
  std::size_t deltas = 0;
  std::string streamed_text;
  auto streamed_scores = gufo::json::Value::array();
  for (const auto& event : DebugHooksAllSseData(streaming)) {
    if (event.member_str("type") != "response.output_text.delta")
      continue;
    if (event.find("logprobs")->empty()) {
      assert(event.member_str("delta").empty());
      continue;
    }
    ++deltas;
    streamed_text += event.member_str("delta");
    assert(event.find("logprobs")->size() == 1);
    streamed_scores.push_back(event.find("logprobs")->items()[0]);
  }
  assert(deltas == 2 && streamed_text == part.member_str("text") &&
         streamed_scores.dump() == part.find("logprobs")->dump());
  assert(done.member_str("text") == "\xC3\xA9x" && done.find("logprobs")->dump() == part.find("logprobs")->dump());
  assert(part_done.find("part")->find("logprobs")->dump() == part.find("logprobs")->dump());
  const auto completed = DebugHooksHttpResponseEvent(streaming,
      body.member_str("status") == "incomplete" ? "response.incomplete" : "response.completed");
  assert(completed.find("response")->find("output")->items()[0]
             .find("content")->items()[0].find("logprobs")->dump() == part.find("logprobs")->dump());
  for (const auto* count : {"-1", "21", "1.5", "true", "null", "\"2\""}) {
    auto invalid = input;
    invalid["top_logprobs"] = gufo::json::parse(count);
    ExpectStatus(server.Post("/v1/responses", invalid.dump()), 400);
  }
  for (const auto* include : {
           R"(["message.output_text.logprobs","unknown"])",
           R"(["message.output_text.logprobs",1])", R"("message.output_text.logprobs")",
           R"(["reasoning.encrypted_content"])", R"(null)"}) {
    auto invalid = input;
    invalid["include"] = gufo::json::parse(include);
    ExpectStatus(server.Post("/v1/responses", invalid.dump()), 400);
  }
  auto implicit_top = gufo::json::parse(
      R"({"input":"x","max_output_tokens":2,"reasoning":{"effort":"none"},
          "include":["message.output_text.logprobs"]})");
  const auto defaults = DebugHooksHttpResponseBody(server.Post("/v1/responses", implicit_top.dump()));
  assert(defaults.find("output")->items()[0].find("content")->items()[0]
             .find("logprobs")->items()[0].find("top_logprobs")->empty());
  server.backend->initial_output_state_override = TextGenerationBackend::InitialOutputState::kReasoning;
  ExpectStatus(server.Post("/v1/responses", input.dump()), 400);
}

void DebugHooksHttpTestNullableLegacyLimit() {
  // v0.1.1 HTTP fix: null (and -1) leave the server default in place on the compatibility routes (35c386a answered 400 for null). Every other bad
  // value keeps its exact base bytes, and hooks OFF changes nothing else.
  RunningServer server;
  server.backend->defaults.max_tokens = 7;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":"x","max_tokens":null})"), 200);
  assert(server.backend->LastCall().max_tokens == 7);
  const auto response = server.Post("/v1/completions", R"({"prompt":"x","max_tokens":"x"})");
  ExpectStatus(response, 400);
  const auto start = response.find("\r\n\r\n");
  assert(start != std::string::npos);
  assert(response.substr(start + 4) ==
         R"({"error":{"message":"'max_tokens' must be an integer between 1 and 4294967295","type":"invalid_request_error","code":"invalid_request"}})");
  assert(server.backend->calls == 1 && server.backend->score_calls == 0);
}

void DebugHooksHttpTestDisabledAndProbeErrors() {
  RunningServer disabled;
  for (const auto* value :
       {"null", "42", "true", "[]", "{}", R"({"unknown":true})",
        R"({"timeline":true})", R"({"timeline":"bad","cache_probe":true})"}) {
    // Base 35c386a ReadCompatibilityOptions ignores debug, then rejects suffix.
    const auto response = disabled.Post(
        "/v1/completions",
        std::string("{\"prompt\":\"hello\",\"suffix\":\"x\",\"debug\":") +
            value + "}");
    ExpectStatus(response, 400);
    const auto start = response.find("\r\n\r\n");
    assert(
        response.substr(start + 4) ==
        R"({"error":{"message":"request field 'suffix' is not supported on this endpoint","type":"invalid_request_error","code":"invalid_request"}})");
  }
  for (const auto* path : {"/v1/responses", "/v1/messages", "/completion"}) {
    for (const auto* value :
         {"null", "17", "[]", R"({"timeline":true})", R"({"unknown":true})"}) {
      const auto response = disabled.Post(
          path, std::string("{\"suffix\":\"x\",\"debug\":") + value + "}");
      ExpectStatus(response, 400);
      const auto body = response.substr(response.find("\r\n\r\n") + 4);
      if (std::string_view(path) == "/v1/messages") {
        // v0.1.1: /v1/messages errors use the Anthropic shape.
        const auto parsed = gufo::json::parse(body);
        assert(parsed.member_str("type") == "error");
        assert(parsed.find("error")->member_str("type") == "invalid_request_error");
        assert(parsed.find("error")->member_str("message") == "request field 'suffix' is not supported on this endpoint");
      } else {
        assert(body ==
               R"({"error":{"message":"request field 'suffix' is not supported on this endpoint","type":"invalid_request_error","code":"invalid_request"}})");
      }
    }
  }
  assert(disabled.backend->calls == 0);
  RunningServer enabled({.debug_hooks = true});
  for (const auto& response :
       {enabled.Post("/v1/completions",
                     R"({"prompt":"hello","debug":{"cache_probe":true}})"),
        enabled.Send("GET /debug/cache HTTP/1.1\r\n\r\n")}) {
    ExpectStatus(response, 400);
    assert(DebugHooksHttpResponseBody(response).find("error")->member_str(
               "message") == "cache inspection is not implemented");
  }
  assert(enabled.backend->calls == 0);
}

void DebugHooksHttpTestLogprobValidation() {
  RunningServer server;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":"x","logprobs":2})"), 400);
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprobs":true})"),
      400);
  const auto unsupported_prompt = server.Post(
      "/v1/completions", R"({"prompt":"xx","echo":true,"max_tokens":0,"logprobs":2})");
  ExpectStatus(unsupported_prompt, 400);
  assert(DebugHooksHttpResponseBody(unsupported_prompt)
             .find("error")
             ->member_str("message") == "'echo' must be false");
  assert(server.backend->calls == 0);
  server.backend->logprob_support = true;
  server.backend->initial_output_state_override = TextGenerationBackend::InitialOutputState::kContent;
  for (const auto* count : {"-1", "21", "1.5", "true", "\"2\""}) {
    ExpectStatus(server.Post("/v1/completions",
                            std::string("{\"prompt\":\"x\",\"logprobs\":") + count + "}"), 400);
    ExpectStatus(server.Post(
        "/v1/chat/completions",
        std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],") +
            "\"logprobs\":true,\"top_logprobs\":" + count + "}"), 400);
  }
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"x"}],"top_logprobs":2})"),
      400);
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprobs":true,"stop":"END"})"),
      400);
  server.backend->initial_output_state_override = TextGenerationBackend::InitialOutputState::kReasoning;
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprobs":true})"),
      400);
  assert(server.backend->calls == 0);
}

void DebugHooksHttpTestRawScoredOutput() {
  RunningServer server({.debug_hooks = true});
  server.backend->logprob_support = true;
  server.backend->initial_output_state_override =
      TextGenerationBackend::InitialOutputState::kContent;
  for (const bool stream : {false, true}) {
    auto input = gufo::json::parse(R"({
      "model":"test","messages":[{"role":"user","content":"x"}],
      "temperature":0,"max_tokens":2,"logprobs":true,
      "debug":{"raw_output":true}})");
    input["stream"] = stream;
    const auto response = server.Post("/v1/chat/completions", input.dump());
    ExpectStatus(response, 200);
    const auto events =
        stream ? DebugHooksAllSseData(response)
               : std::vector{DebugHooksHttpResponseBody(response)};
    const auto& last = events.back();
    assert(last.contains("usage") && last.contains("gufo_debug"));
    const auto* raw = last.find("gufo_debug")->find("raw_output");
    assert(raw && raw->member_str("text") == "\xC3\xA9x" &&
           raw->member_size("tokens") == 2);
    const auto* parse = raw->find("parse");
    assert(parse->member_size("content_chars") == 2 &&
           parse->member_size("reasoning_chars") == 0 &&
           parse->member_size("parsed_tool_calls") == 0 &&
           parse->member_size("unfinished_call_bytes") == 0);
    for (std::size_t i = 0; i + 1 < events.size(); ++i)
      assert(!events[i].contains("gufo_debug"));
  }
}

void DebugHooksHttpTestGeneratedLogprobs() {
  RunningServer server;
  server.backend->logprob_support = true;
  server.backend->initial_output_state_override = TextGenerationBackend::InitialOutputState::kContent;
  const auto chat = server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"temperature":0,"max_tokens":2,"logprobs":true,"top_logprobs":2})");
  ExpectStatus(chat, 200);
  const auto body = DebugHooksHttpResponseBody(chat);
  const auto& choice = body.find("choices")->items()[0];
  assert(choice.find("message")->member_str("content") == "\xC3\xA9x");
  const auto& records = choice.find("logprobs")->find("content")->items();
  assert(records.size() == 2);
  const auto& bytes = records[0].find("bytes")->items();
  assert(bytes.size() == 2 && bytes[0].as_size() == 195 && bytes[1].as_size() == 169);
  assert(std::abs(records[0].member_double("logprob") + std::log(1 + std::exp(-2.0))) < 1e-12);
  assert(records[0].find("top_logprobs")->items()[0].member_str("token") == "\xC3\xA9");
  const auto streamed = server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"temperature":0,"max_tokens":2,"stream":true,"logprobs":true,"top_logprobs":0})");
  ExpectStatus(streamed, 200);
  const auto chunk = DebugHooksHttpSseChunk(streamed);
  const auto& streamed_choice = chunk.find("choices")->items()[0];
  assert(chunk.member_str("object") == "chat.completion.chunk");
  assert(streamed_choice.find("delta")->member_str("content") == "\xC3\xA9");
  assert(streamed_choice.find("logprobs")->find("content")->size() == 1);
  const auto chat_frames = DebugHooksAllSseData(streamed);
  assert(chat_frames.size() == 3);
  std::string chat_text;
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& c = chat_frames[i].find("choices")->items()[0];
    assert(c.find("finish_reason")->is_null() &&
           c.find("logprobs")->find("content")->size() == 1);
    assert(!chat_frames[i].contains("usage") &&
           !chat_frames[i].contains("timings"));
    chat_text += c.find("delta")->member_str("content");
  }
  assert(chat_text == choice.find("message")->member_str("content"));
  assert(streamed_choice.find("logprobs")->find("content")->items()[0]
             .find("top_logprobs")->empty());
  assert(streamed.find("data: [DONE]") != std::string::npos);
  const auto legacy = server.Post(
      "/v1/completions",
      R"({"prompt":"x","temperature":0,"max_tokens":2,"stream":true,"logprobs":2})");
  ExpectStatus(legacy, 200);
  const auto legacy_chunk = DebugHooksHttpSseChunk(legacy);
  const auto& legacy_choice = legacy_chunk.find("choices")->items()[0];
  assert(legacy_choice.member_str("text") == "\xC3\xA9");
  const auto* legacy_scores = legacy_choice.find("logprobs");
  assert(legacy_scores->find("tokens")->size() == 1);
  assert(legacy_scores->find("token_logprobs")->size() == 1);
  assert(legacy_scores->find("top_logprobs")->size() == 1);
  const auto legacy_frames = DebugHooksAllSseData(legacy);
  assert(legacy_frames.size() == 3);
  const auto& second = legacy_frames[1].find("choices")->items()[0];
  assert(second.member_str("text") == "x" &&
         second.find("logprobs")->find("tokens")->size() == 1 &&
         second.find("logprobs")->find("text_offset")->items()[0].as_size() ==
             1);
  const auto ordinary = server.Post(
      "/v1/completions", R"({"prompt":"x","temperature":0.7,"seed":123,"max_tokens":2})");
  const auto scored = server.Post(
      "/v1/completions", R"({"prompt":"x","temperature":0.7,"seed":123,"max_tokens":2,"logprobs":2})");
  ExpectStatus(ordinary, 200);
  ExpectStatus(scored, 200);
  assert(DebugHooksHttpResponseBody(ordinary).find("choices")->items()[0].member_str("text") ==
         DebugHooksHttpResponseBody(scored).find("choices")->items()[0].member_str("text"));
}

void DebugHooksHttpTestPromptScoreErrors() {
  {
    RunningServer server({.max_score_tokens = 2});
    server.backend->logprob_support = true;
    const auto over = server.Post(
        "/v1/completions",
        R"({"prompt":"xxx","echo":true,"max_tokens":0,"logprobs":2})");
    ExpectStatus(over, 400);
    assert(over.find("prompt has 3 tokens") != std::string::npos);
    assert(over.find("--max-score-tokens 2") != std::string::npos);
    assert(server.backend->score_calls == 0);
    // Allocation failure: a stage and token, a 503, never a bare 500.
    server.backend->score_bad_alloc = true;
    const auto oom = server.Post(
        "/v1/completions",
        R"({"prompt":"xx","echo":true,"max_tokens":0,"logprobs":2})");
    ExpectStatus(oom, 503);
    assert(oom.find("ran out of host memory during session creation at token 0") !=
           std::string::npos);
    assert(oom.find("scoring_out_of_memory") != std::string::npos);
    assert(oom.find("bad allocation") == std::string::npos);
    assert(oom.find("Retry-After") != std::string::npos);
    server.backend->score_bad_alloc = false;
    ExpectStatus(server.Post("/v1/completions",
                             R"({"prompt":"xx","echo":true,"max_tokens":0,"logprobs":2})"),
                 200);
  }
  {
    // The response cap answers 413 even when --max-score-tokens allows more.
    RunningServer server({.max_score_tokens = 100000});
    server.backend->logprob_support = true;
    const auto big = server.Post(
        "/v1/completions",
        R"({"prompt":")" +
            std::string(gufo::server::kMaxPromptScoreResponseTokens + 1, 'x') +
            R"(","echo":true,"max_tokens":0,"logprobs":0})");
    ExpectStatus(big, 413);
    assert(big.find("response_too_large") != std::string::npos);
    assert(big.find("65537") != std::string::npos);
    assert(server.backend->score_calls == 0);
  }
}

void DebugHooksHttpTestPromptScores() {
  RunningServer server({.max_score_tokens = 2});
  server.backend->logprob_support = true;
  const auto scored = server.Post(
      "/v1/completions", R"({"prompt":"xx","echo":true,"max_tokens":0,"logprobs":2,"temperature":0})");
  ExpectStatus(scored, 200);
  const auto body = DebugHooksHttpResponseBody(scored);
  const auto& choice = body.find("choices")->items()[0];
  assert(choice.member_str("text") == "xx");
  const auto* scores = choice.find("logprobs");
  assert(scores->find("token_logprobs")->items()[0].is_null());
  assert(scores->find("top_logprobs")->items()[0].is_null());
  assert(std::abs(scores->find("token_logprobs")->items()[1].as_double() +
                  1 + std::log(1 + std::exp(-1.0))) < 1e-12);
  assert(body.find("usage")->member_size("completion_tokens") == 0);
  assert(body.find("usage")->member_size("prompt_tokens") == 2);
  assert(server.backend->calls == 0 && server.backend->score_calls == 1);
  ExpectStatus(server.Post("/v1/completions",
                          R"({"prompt":"xxx","echo":true,"max_tokens":0,"logprobs":2})"), 400);
  assert(server.backend->score_calls == 1);
  const auto stream = server.Post(
      "/v1/completions", R"({"prompt":"xx","echo":true,"max_tokens":0,"logprobs":0,"stream":true})");
  ExpectStatus(stream, 200);
  const auto stream_body = DebugHooksHttpSseChunk(stream);
  assert(stream_body.find("choices")->items()[0].find("logprobs")
             ->find("token_logprobs")->items()[0].is_null());
  assert(stream_body.find("usage")->member_size("completion_tokens") == 0);
  assert(server.backend->calls == 0 && server.backend->score_calls == 2);
}

void DebugHooksBaseGoldenBytesAndExposure() {
  RunningServer off;
  // Golden bodies transcribed from git show 35c386a:openai_chat.cpp
  // ParseRequest/Error and http_server.cpp
  // ReadCompatibilityOptions/InvalidCompatibilityRequest/Err.
  const std::string chat_input =
      R"({"model":"test","messages":[{"role":"user","content":"x"}]})";
  for (const auto* field : {"logprobs", "top_logprobs"}) {
    auto body = gufo::json::parse(chat_input);
    body[field] = field == std::string_view("logprobs")
                      ? gufo::json::Value(true)
                      : gufo::json::Value(2);
    const auto response = off.Post("/v1/chat/completions", body.dump());
    ExpectStatus(response, 400);
    const auto expected =
        std::string(R"({"error":{"message":"request field ')") + field +
        R"(' is not implemented","type":"invalid_request_error","code":"unsupported_field"}})";
    assert(response.substr(response.find("\r\n\r\n") + 4) == expected);
  }
  for (const auto* value :
       {"null", "42", "true", "[]", "{}", R"({"unknown":true})",
        R"({"timeline":true})", R"({"cache_probe":true})",
        R"({"timeline":"bad"})"}) {
    const auto response = off.Post("/v1/chat/completions",
                                   std::string("{\"debug\":") + value + "}");
    ExpectStatus(response, 400);
    assert(
        response.substr(response.find("\r\n\r\n") + 4) ==
        R"({"error":{"message":"'model' is required","type":"invalid_request_error","code":"missing_model"}})");
  }
  for (const auto* path : {"/v1/completions", "/v1/responses"}) {
    for (const auto* field : {"logprobs", "top_logprobs"}) {
      const auto response =
          off.Post(path, std::string("{\"prompt\":\"x\",\"input\":\"x\",\"") +
                             field + "\":2}");
      ExpectStatus(response, 400);
      const auto expected =
          std::string(R"({"error":{"message":"request field ')") + field +
          R"(' is not supported on this endpoint","type":"invalid_request_error","code":"invalid_request"}})";
      assert(response.substr(response.find("\r\n\r\n") + 4) == expected);
    }
  }
  // Base OpenAiResponses ignores every include value. ResponsesOutput::Part
  // supplies these exact deterministic bytes (IDs/time are independently random).
  for (const auto* include : {
           R"(["message.output_text.logprobs"])",
           R"(["message.output_text.logprobs",1,"unknown"])",
           R"("message.output_text.logprobs")", "null", "42", "{}"}) {
    const auto response = off.Post("/v1/responses",
        std::string(R"({"input":"x","reasoning":{"effort":"none"},"include":)") + include + "}");
    ExpectStatus(response, 200);
    const auto body = DebugHooksHttpResponseBody(response);
    const auto& output = body.find("output")->items();
    assert(output.size() == 1 && output[0].member_str("type") == "message");
    assert(output[0].find("content")->dump() ==
           R"([{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}])");
    assert(off.backend->LastCall().sampling.logprobs == -1);
    const auto error = off.Post("/v1/responses",
        std::string(R"({"include":)") + include + "}");
    ExpectStatus(error, 400);
    assert(error.substr(error.find("\r\n\r\n") + 4) ==
           R"({"error":{"message":"'input' must contain text, message items with text/images, reasoning items, function calls or function outputs","type":"invalid_request_error","code":"invalid_request"}})");
  }
  // Full generation body transcribed from 35c386a LlamaCompletion,
  // UsageJson, GenerationTimings and the original FakeBackend fixed result.
  const std::string completion_golden =
      R"({"content":"ok","stop":true,"stopped_eos":true,"stopped_length":false,"stopped_word":false,"stopped_limit":false,"stopping_word":"","tokens_predicted":1,"tokens_evaluated":10,"tokens_cached":8,"timings":{"prompt_n":2,"prompt_ms":4,"prompt_per_token_ms":2,"prompt_per_second":500,"predicted_n":1,"predicted_ms":2,"predicted_per_token_ms":2,"predicted_per_second":500,"cache_n":8,"cache_restore_ms":0,"cache_snapshot_ms":0,"cache_disk_enqueue_ms":0,"draft_rounds":2,"draft_n":8,"draft_n_accepted":4},"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":8},"cached_tokens":8,"prompt_tokens_per_second":500,"completion_tokens_per_second":500,"draft_tokens":8,"draft_tokens_accepted":4}})";
  for (const auto* debug : {"null", "17", "false", "[]", "{}",
                            R"({"timeline":true})", R"({"unknown":true})",
                            R"({"cache_probe":true})", R"({"timeline":"bad"})"}) {
    const auto success = off.Post("/completion",
        std::string("{\"prompt\":\"hello\",\"n_predict\":2,\"debug\":") + debug + "}");
    ExpectStatus(success, 200);
    assert(success.substr(success.find("\r\n\r\n") + 4) == completion_golden);
  }
  const auto success = off.Post("/completion", R"({"prompt":"hello","n_predict":2})");
  ExpectStatus(success, 200);
  assert(success.substr(success.find("\r\n\r\n") + 4) == completion_golden);
  // Deterministic successful response golden from base handle_request /health.
  const auto health = off.Send("GET /health HTTP/1.1\r\n\r\n");
  assert(health.substr(health.find("\r\n\r\n") + 4) == R"({"status":"ok"})");
  for (const auto* host :
       {"127.0.0.1", "127.7.8.9", "127.255.255.254", "::1", "localhost"})
    assert(gufo::server::DebugLoopbackListener(host));
  for (const auto* host : {"0.0.0.0", "::", "192.168.1.7", "128.0.0.1",
                           "127.0.0.999", "localhost.example"})
    assert(!gufo::server::DebugLoopbackListener(host));
  auto settings = std::make_shared<const gufo::json::Value>(
      gufo::json::parse(R"({"secret":"local"})"));
  RunningServer remote({.debug_hooks = true, .debug_settings = settings}, false,
                       "0.0.0.0");
  for (const auto* path :
       {"/v1/completions", "/v1/responses", "/v1/chat/completions",
        "/v1/messages", "/completion"}) {
    const auto response = remote.Post(path, R"({"debug":{"timeline":true}})");
    ExpectStatus(response, 400);
    assert(DebugHooksHttpResponseBody(response).find("error")->member_str(
               "message") == "debug hooks require a loopback listener");
  }
  ExpectStatus(remote.Send("GET /debug/cache HTTP/1.1\r\n\r\n"), 404);
  ExpectStatus(remote.Post("/debug/prompt", R"({"text":"x"})"), 404);
  const auto models = DebugHooksHttpResponseBody(
      remote.Send("GET /v1/models HTTP/1.1\r\n\r\n"));
  assert(!models.find("data")->items()[0].contains("gufo"));
  const auto completion = DebugHooksHttpResponseBody(
      remote.Post("/v1/completions", R"({"prompt":"x"})"));
  assert(!completion.contains("gufo"));
  RunningServer local({.debug_hooks = true}, false, "localhost");
  for (const auto* path : {"/v1/messages", "/completion"})
    ExpectStatus(local.Post(path, R"({"debug":{}})"), 400);
  ExpectStatus(local.Post("/v1/completions",
                          R"({"prompt":"x","debug":{"timeline":true}})"),
               400);
  const auto before_inspection = local.backend->calls.load();
  const auto inspected = DebugHooksHttpResponseBody(
      local.Post("/debug/prompt", R"({"text":"xyz"})"));
  assert(inspected.member_str("text") == "xyz" &&
         inspected.find("token_ids")->size() == 3);
  const auto chat_prompt = DebugHooksHttpResponseBody(local.Post(
      "/debug/prompt",
      R"({"model":"test","messages":[{"role":"user","content":"xyz"}]})"));
  assert(chat_prompt.dump() == inspected.dump() &&
         local.backend->calls.load() == before_inspection);
  RunningServer ipv6_local({.debug_hooks = true, .debug_settings = settings},
                           false, "::1");
  ExpectStatus(ipv6_local.Send("GET /debug/settings HTTP/1.1\r\n\r\n"), 200);
  RunningServer ipv6_remote({.debug_hooks = true, .debug_settings = settings},
                            false, "::");
  ExpectStatus(ipv6_remote.Send("GET /debug/settings HTTP/1.1\r\n\r\n"), 404);
  ExpectStatus(ipv6_remote.Post("/v1/completions",
                                R"({"prompt":"x","debug":{"timeline":true}})"),
               400);
  RunningServer broken;
  broken.backend->logprob_support = true;
  broken.backend->short_scores = true;
  broken.backend->initial_output_state_override =
      TextGenerationBackend::InitialOutputState::kContent;
  for (const auto* path :
       {"/v1/chat/completions", "/v1/completions", "/v1/responses"}) {
    const auto response = broken.Post(
        path,
        path == std::string_view("/v1/chat/completions")
            ? R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprobs":true})"
        : path == std::string_view("/v1/completions")
            ? R"({"prompt":"x","logprobs":2})"
            : R"({"input":"x","include":["message.output_text.logprobs"]})");
    ExpectStatus(response, 500);
    assert(DebugHooksHttpResponseBody(response).find("error")->member_str(
               "message") == "generated token and logprob counts differ");
  }
  const auto late = broken.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprobs":true,"stream":true})");
  ExpectStatus(late, 200);
  assert(late.find("generated token and logprob counts differ") !=
             std::string::npos &&
         late.find("\"finish_reason\":\"error\"") != std::string::npos);
}

void TestDiffusionReadContract() {
  using gufo::json::parse;
  RunningServer server;
  server.backend->diffusion_support = true;
  const auto body_of = [](const std::string& wire) {
    ExpectStatus(wire, 200);
    return parse(wire.substr(wire.find("\r\n\r\n") + 4));
  };
  const std::string xargs =
      R"({"diffusion_seed_canvas":[1,2],"diffusion_canvas_length":2,"diffusion_max_steps":2,"diffusion_read_only":true,"diffusion_constrained":true,"diffusion_pinned":[0]})";
  const std::string chat =
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1,"logprobs":true,"top_logprobs":2,"logprob_token_ids":[900,900],"return_tokens_as_token_ids":true,"vllm_xargs":)" +
      xargs + "}";
  const auto response = body_of(server.Post("/v1/chat/completions", chat));
  assert(response.member_str("object") == "chat.completion");
  const auto& choice = response.find("choices")->items().front();
  assert(choice.find("message")->member_str("role") == "assistant");
  assert(choice.find("message")->member_str("content").empty());
  assert(choice.member_str("finish_reason") == "length");
  assert(response.find("usage")->member_size("prompt_tokens") == 7);
  const auto& rows = choice.find("logprobs")->find("content")->items();
  assert(rows.size() == 2);
  const auto fixture = parse(
      R"({"token":"token_id:10","logprob":-0.1,"bytes":[116,111,107,101,110,95,105,100,58,49,48],"top_logprobs":[{"token":"token_id:10","logprob":-0.1,"bytes":[116,111,107,101,110,95,105,100,58,49,48]},{"token":"token_id:11","logprob":-0.2,"bytes":[116,111,107,101,110,95,105,100,58,49,49]},{"token":"token_id:900","logprob":-9,"bytes":[116,111,107,101,110,95,105,100,58,57,48,48]}]})");
  assert(rows[0].member_str("token") == fixture.member_str("token"));
  assert(rows[0].find("bytes")->dump() == fixture.find("bytes")->dump());
  const auto& expected_tops = fixture.find("top_logprobs")->items();
  const auto& actual_tops = rows[0].find("top_logprobs")->items();
  assert(actual_tops.size() == expected_tops.size());
  for (std::size_t i = 0; i < expected_tops.size(); ++i) {
    assert(actual_tops[i].member_str("token") ==
           expected_tops[i].member_str("token"));
    assert(actual_tops[i].find("logprob")->as_double() ==
           expected_tops[i].find("logprob")->as_double());
    assert(actual_tops[i].find("bytes")->dump() ==
           expected_tops[i].find("bytes")->dump());
  }
  assert(rows[1].find("top_logprobs")->items().back().member_str("token") ==
         "token_id:900");
  assert(server.backend->LastCall().max_tokens == 1);
  assert(server.backend->LastCall().chat.diffusion_read->read_only);
  assert(server.backend->LastCall().chat.diffusion_read->label_ids.size() == 1);

  const std::string completion =
      R"({"model":"test","prompt":[7,8,9],"max_tokens":1,"logprobs":2,"logprob_token_ids":[900],"return_tokens_as_token_ids":true,"vllm_xargs":)" +
      xargs + "}";
  const auto completed = body_of(server.Post("/v1/completions", completion));
  assert(completed.member_str("object") == "text_completion");
  const auto& legacy =
      *completed.find("choices")->items().front().find("logprobs");
  const auto legacy_fixture = parse(
      R"({"tokens":["token_id:10","token_id:11"],"token_logprobs":[-0.1,-0.1],"top_logprobs":[{"token_id:10":-0.1,"token_id:11":-0.2,"token_id:900":-9},{"token_id:11":-0.1,"token_id:12":-0.2,"token_id:900":-10}],"text_offset":[0,11]})");
  assert(completed.find("choices")->items().front().member_str("text").empty());
  assert(legacy.find("tokens")->dump() ==
         legacy_fixture.find("tokens")->dump());
  assert(legacy.find("token_logprobs")->size() ==
         legacy_fixture.find("token_logprobs")->size());
  assert(legacy.find("text_offset")->dump() ==
         legacy_fixture.find("text_offset")->dump());
  const auto& tops = legacy.find("top_logprobs")->items();
  const auto& expected_legacy_tops =
      legacy_fixture.find("top_logprobs")->items();
  assert(tops.size() == expected_legacy_tops.size());
  for (std::size_t i = 0; i < tops.size(); ++i)
    for (const auto& [token, value] : expected_legacy_tops[i].members())
      assert(tops[i].find(token)->as_double() == value.as_double());
  assert(tops[0].size() == 3 &&
         tops[0].find("token_id:900")->as_double() == -9);
  assert(tops[1].size() == 3 &&
         tops[1].find("token_id:900")->as_double() == -10);
  assert(completed.find("usage")->member_size("prompt_tokens") == 3);
  auto [last, clipped] = server.backend->LastTokenRequest();
  assert(clipped == 1 && last.prompt.size() == 3);
  assert(last.diffusion_read->pinned_positions == std::vector<std::size_t>{0});
  const auto integer_flags = body_of(server.Post(
      "/v1/completions",
      R"({"prompt":[7],"logprobs":0,"return_tokens_as_token_ids":true,"vllm_xargs":{"diffusion_canvas_length":2,"diffusion_seed_canvas":[1,2],"diffusion_read_only":1,"diffusion_constrained":0}})"));
  assert(integer_flags.find("choices")
             ->items()
             .front()
             .find("logprobs")
             ->find("tokens")
             ->size() == 2);
  assert(server.backend->LastTokenRequest().second == 2);

  // A requested label beyond the top 20 is present at every canvas position.
  auto wide = parse(chat);
  wide["top_logprobs"] = 20;
  const auto all = body_of(server.Post("/v1/chat/completions", wide.dump()));
  for (const auto& row : all.find("choices")
                             ->items()
                             .front()
                             .find("logprobs")
                             ->find("content")
                             ->items()) {
    const auto& top = row.find("top_logprobs")->items();
    assert(top.size() == 21 &&
           top.back().member_str("token") == "token_id:900");
  }
  server.backend->synthetic_eos = true;
  auto long_read = parse(chat);
  long_read["max_tokens"] = 100;
  const auto eos_read =
      body_of(server.Post("/v1/chat/completions", long_read.dump()));
  const auto& eos_rows = eos_read.find("choices")
                             ->items()
                             .front()
                             .find("logprobs")
                             ->find("content")
                             ->items();
  assert(eos_rows.size() == 2 &&
         eos_rows.front().member_str("token") == "token_id:0");
  assert(server.backend->LastCall().max_tokens == 2);
  server.backend->synthetic_eos = false;

  const auto generated = body_of(server.Post(
      "/v1/completions",
      R"({"model":"test","prompt":[1,2,3],"max_tokens":4,"logprobs":0,"return_tokens_as_token_ids":true,"stop_token_ids":[12]})"));
  const auto& generated_logprobs =
      *generated.find("choices")->items().front().find("logprobs");
  assert(generated_logprobs.find("tokens")->dump() ==
         R"(["token_id:11","token_id:12"])");
  assert(server.backend->LastTokenRequest().first.stop_token_ids ==
         std::vector<gufo::sampling::TokenId>{12});
  body_of(server.Post("/v1/completions", R"({"prompt":"ordinary"})"));
  body_of(server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"ordinary"}]})"));

  const auto bad = [&](std::string_view extra, std::string_view message) {
    const std::string request =
        R"({"model":"test","messages":[{"role":"user","content":"x"}],"vllm_xargs":)" +
        std::string(extra) + "}";
    const auto wire = server.Post("/v1/chat/completions", request);
    ExpectStatus(wire, 400);
    assert(wire.find(message) != std::string::npos);
    assert(parse(wire.substr(wire.find("\r\n\r\n") + 4))
               .find("error")
               ->member_str("type") == "invalid_request_error");
  };
  bad(R"({"diffusion_seed_canvas":"abc"})", "list of token ids");
  bad(R"({"diffusion_seed_canvas":[1,true]})", "list of token ids");
  bad(R"({"diffusion_seed_canvas":[1,2.5]})", "list of token ids");
  bad(R"({"diffusion_seed_canvas":[1,-1]})", "ids must be in");
  bad(R"({"diffusion_seed_canvas":[1,1024]})", "ids must be in");
  bad(R"({"diffusion_canvas_length":2,"diffusion_seed_canvas":[1]})",
      "exactly 2 ids");
  bad(R"({"diffusion_max_steps":0})", "positive integer");
  bad(R"({"diffusion_max_steps":true})", "positive integer");
  bad(R"({"diffusion_max_steps":"1"})", "positive integer");
  bad(R"({"diffusion_read_only":"yes"})", "boolean");
  bad(R"({"diffusion_read_only":2})", "boolean");
  bad(R"({"diffusion_read_only":[1]})", "boolean");
  bad(R"({"diffusion_constrained":"yes"})", "boolean");
  bad(R"({"diffusion_constrained":2})", "boolean");
  bad(R"({"diffusion_constrained":[1]})", "boolean");
  bad(R"({"diffusion_constrained":true})", "needs logprob_token_ids");
  bad(R"({"diffusion_pinned":"0,1"})", "list of canvas positions");
  bad(R"({"diffusion_pinned":[0]})", "needs a diffusion_seed_canvas");
  bad(R"({"diffusion_seed_canvas":[0,0,0,0,0,0,0,0],"diffusion_pinned":[true]})",
      "list of canvas positions");
  bad(R"({"diffusion_seed_canvas":[0,0,0,0,0,0,0,0],"diffusion_pinned":[8]})",
      "inside the canvas");
  bad(R"({"diffusion_canvas_length":0})", "positive integer");
  bad(R"({"diffusion_canvas_length":true})", "positive integer");
  bad(R"({"diffusion_canvas_length":"2"})", "positive integer");
  bad(R"({"diffusion_canvas_length":9})", "no larger");
  bad(R"({"diffusion_typo":true})", "unknown vllm_xargs field");
  const auto bad_labels = [&](const std::string& value,
                              std::string_view message) {
    const std::string request =
        R"({"model":"test","messages":[{"role":"user","content":"x"}],"logprob_token_ids":)" +
        value + "}";
    const auto wire = server.Post("/v1/chat/completions", request);
    ExpectStatus(wire, 400);
    assert(wire.find(message) != std::string::npos);
  };
  bad_labels(R"("1")", "list of token ids");
  bad_labels("[true]", "list of token ids");
  bad_labels("[1.0]", "list of token ids");
  bad_labels("[-1]", "ids must be in");
  bad_labels("[1024]", "ids must be in");
  auto too_many = gufo::json::Value::array();
  for (int i = 0; i < 513; ++i)
    too_many.push_back(i);
  bad_labels(too_many.dump(), "at most 512");
  ExpectStatus(
      server.Post(
          "/v1/completions",
          R"({"prompt":[1],"logprobs":0,"vllm_xargs":{"diffusion_canvas_length":9}})"),
      400);
  ExpectStatus(server.Post("/v1/completions",
                           R"({"prompt":[1],"logprob_token_ids":[1024]})"),
               400);
  // Non-diffusion xargs have no effect.
  body_of(server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"vllm_xargs":{"other":1}})"));
  body_of(server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"vllm_xargs":{"diffusion_canvas_length":null}})"));
  server.backend->diffusion_support = false;
  const auto unsupported_chat = server.Post("/v1/chat/completions", chat);
  ExpectStatus(unsupported_chat, 400);
  assert(unsupported_chat.find("this model does not support diffusion reads") !=
         std::string::npos);
  const auto unsupported_typo = server.Post(
      "/v1/chat/completions",
      R"({"model":"test","messages":[{"role":"user","content":"x"}],"vllm_xargs":{"diffusion_typo":1}})");
  ExpectStatus(unsupported_typo, 400);
  assert(unsupported_typo.find("this model does not support diffusion reads") !=
         std::string::npos);
  const auto unsupported_completion =
      server.Post("/v1/completions", completion);
  ExpectStatus(unsupported_completion, 400);
  assert(unsupported_completion.find(
             "this model does not support diffusion reads") !=
         std::string::npos);
}

// Memory guard: the three guard codes finish a request row as "refused" with the numbers, every row carries status and error_code,
// and POST /debug/memory (--debug-hooks, loopback only) forces the guard's readings.
void TestMemoryGuardRowsAndDebugRoute() {
  constexpr std::uint64_t GiB = std::uint64_t{1} << 30;
  std::mutex rows_mutex;
  std::vector<gufo::json::Value> rows;
  gufo::server::HttpServerOptions options;
  options.on_request_completed = [&](gufo::json::Value row) {
    const std::lock_guard<std::mutex> lock(rows_mutex);
    rows.push_back(std::move(row));
  };
  RunningServer server(std::move(options));
  const auto last_row = [&] {
    const std::lock_guard<std::mutex> lock(rows_mutex);
    assert(!rows.empty());
    return rows.back();
  };

  server.backend->failure = 20;   // transient: 503 insufficient_memory
  auto response = server.Post("/v1/completions", R"({"prompt":"x","max_tokens":1})");
  ExpectStatus(response, 503);
  assert(response.find("Retry-After: 5") != std::string::npos);
  assert(response.find("\"code\":\"insufficient_memory\"") != std::string::npos);
  auto row = last_row();
  assert(row.member_str("outcome") == "refused" && row.member_size("status") == 503 && row.member_str("error_code") == "insufficient_memory");
  assert(row.find("refusal")->member_str("code") == "insufficient_memory" && row.find("refusal")->member_size("needed_bytes") == 2 * GiB &&
         row.find("refusal")->member_size("free_bytes") == 1 * GiB);

  server.backend->failure = 21;   // permanent: 507
  response = server.Post("/v1/completions", R"({"prompt":"x","max_tokens":1})");
  ExpectStatus(response, 507);
  assert(response.find("Retry-After") == std::string::npos);
  row = last_row();
  assert(row.member_str("outcome") == "refused" && row.member_size("status") == 507 && row.member_str("error_code") == "insufficient_memory_permanent");

  server.backend->failure = 22;  // a running request stopped on low memory
  response = server.Post("/v1/completions", R"({"prompt":"x","max_tokens":1})");
  ExpectStatus(response, 503);
  assert(response.find("Retry-After: 5") != std::string::npos);
  row = last_row();
  assert(row.member_str("outcome") == "refused" && row.member_str("error_code") == "memory_pressure");

  server.backend->failure = 1;   // any other failure keeps "failed" and now names its status and code
  response = server.Post("/v1/completions", R"({"prompt":"x","max_tokens":1})");
  row = last_row();
  assert(row.member_str("outcome") == "failed" && row.member_size("status") >= 400 && !row.member_str("error_code").empty());
  assert(!row.contains("refusal"));

  server.backend->failure = 0;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":"x","max_tokens":1})"), 200);
  row = last_row();
  assert(row.member_str("outcome") == "completed" && row.member_size("status") == 200 && row.find("error_code")->is_null());

  // The debug route: refused without --debug-hooks, and with the guard off.
  ExpectStatus(server.Send("GET /debug/memory HTTP/1.1\r\n\r\n"), 404);
  struct Source final : gufo::platform::MemoryReadingSource {
    gufo::platform::MemoryReadings Read() noexcept override {
      gufo::platform::MemoryReadings r;
      r.ram_ok = true;
      r.ram_available = 13 * (std::uint64_t{1} << 30);
      r.ram_total = 31 * (std::uint64_t{1} << 30);
      return r;
    }
  } source;
  gufo::platform::MemoryGuard guard(&source, 4 * GiB);
  RunningServer off({.debug_hooks = true}, false, "localhost");
  ExpectStatus(off.Send("GET /debug/memory HTTP/1.1\r\n\r\n"), 404);   // no process guard yet
  gufo::platform::SetProcessMemoryGuard(&guard);
  RunningServer remote({.debug_hooks = true}, false, "0.0.0.0");
  ExpectStatus(remote.Send("GET /debug/memory HTTP/1.1\r\n\r\n"), 404);   // not a loopback listener
  RunningServer local({.debug_hooks = true}, false, "localhost");
  auto state = DebugHooksHttpResponseBody(local.Send("GET /debug/memory HTTP/1.1\r\n\r\n"));
  assert(state.member_str("level") == "ok" && !state.find("overridden")->as_bool() && state.member_size("ram_available_mib") == 13 * 1024);
  state = DebugHooksHttpResponseBody(local.Post("/debug/memory", R"({"ram_available_mib":500,"low_memory":true})"));
  assert(state.find("overridden")->as_bool() && state.member_size("ram_available_mib") == 500 && state.find("low_memory_signal")->as_bool());
  assert(guard.Read().ram_available == 500 * (std::uint64_t{1} << 20) && guard.Read().low_memory_signal);
  ExpectStatus(local.Post("/debug/memory", "{"), 400);
  state = DebugHooksHttpResponseBody(local.Post("/debug/memory", R"({"extra_need_mib":2048})"));
  assert(state.member_size("extra_need_mib") == 2048 && guard.debug_need() == 2 * GiB);
  state = DebugHooksHttpResponseBody(local.Post("/debug/memory", R"({"clear":true})"));
  assert(!state.find("overridden")->as_bool() && state.member_size("ram_available_mib") == 13 * 1024 && guard.debug_need() == 0);
  gufo::platform::SetProcessMemoryGuard(nullptr);
}

// ---------------------------------------------------------------------------
// v0.1.1 HTTP fixes (multislot 2026-10-20): CORS preflight, max_tokens -1 / null, Anthropic error shape, event keepalive for Responses streams
// ---------------------------------------------------------------------------

// Header value of the first header named `name` in a raw response, or "" (case-sensitive name as sent by the server).
std::string RawHeader(const std::string& response, std::string_view name) {
  const auto head = response.substr(0, response.find("\r\n\r\n"));
  const auto key = "\r\n" + std::string(name) + ": ";
  const auto at = head.find(key);
  if (at == std::string::npos)
    return "";
  const auto begin = at + key.size();
  return head.substr(begin, head.find("\r\n", begin) - begin);
}

std::size_t CountOccurrences(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
    ++count;
  return count;
}

void TestCorsPreflightEchoesRequestedHeaders() {
  RunningServer server;
  const auto asked = server.Send(
      "OPTIONS /v1/messages HTTP/1.1\r\nOrigin: http://localhost:3000\r\nAccess-Control-Request-Method: POST\r\n"
      "Access-Control-Request-Headers: x-api-key, anthropic-version, x-stainless-os, content-type\r\n\r\n");
  ExpectStatus(asked, 204);
  assert(RawHeader(asked, "Access-Control-Allow-Headers") == "x-api-key, anthropic-version, x-stainless-os, content-type");
  assert(RawHeader(asked, "Access-Control-Allow-Origin") == "*");
  assert(RawHeader(asked, "Access-Control-Allow-Methods") == "GET, POST, DELETE, OPTIONS");
  assert(CountOccurrences(asked, "Access-Control-Allow-Headers:") == 1);   // never twice
  assert(CountOccurrences(asked, "Access-Control-Allow-Origin:") == 1);
  // No request header: the fixed list, as before.
  const auto plain = server.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n");
  ExpectStatus(plain, 204);
  assert(RawHeader(plain, "Access-Control-Allow-Headers") == "Content-Type, Authorization, Range");
  // Anything that is not a list of header names is not echoed.
  for (const std::string& bad : std::vector<std::string>{"x-a;b", "x-a\"b", "<script>", std::string(2000, 'a')}) {
    const auto hostile = server.Send("OPTIONS /v1/models HTTP/1.1\r\nAccess-Control-Request-Headers: " + bad + "\r\n\r\n");
    ExpectStatus(hostile, 204);
    assert(RawHeader(hostile, "Access-Control-Allow-Headers") == "Content-Type, Authorization, Range");
  }
  // Ordinary responses keep the fixed CORS headers.
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  assert(RawHeader(models, "Access-Control-Allow-Origin") == "*");
}

void TestMaxTokensMinusOneAndNull() {
  RunningServer server;
  struct Endpoint {
    const char *path, *body_prefix, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi")", "max_tokens"},
           Endpoint{"/v1/chat/completions", R"({"model":"test","messages":[{"role":"user","content":"hi"}])", "max_tokens"},
           Endpoint{"/v1/chat/completions", R"({"model":"test","messages":[{"role":"user","content":"hi"}])", "max_completion_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi")", "max_output_tokens"},
           Endpoint{"/v1/messages", R"({"messages":[{"role":"user","content":"hi"}])", "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi")", "n_predict"},
       }) {
    for (const std::string value : {"-1", "null"}) {
      const auto response =
          server.Post(endpoint.path, std::string(endpoint.body_prefix) + ",\"" + endpoint.limit + "\":" + value + "}");
      ExpectStatus(response, 200);
      // The server default stays in place (the fake backend's default is 0 = unset).
      assert(server.backend->LastCall().max_tokens == 0);
    }
    // An explicit integer takes the unchanged path: it is passed on, and the checks still reject the rest.
    ExpectStatus(server.Post(endpoint.path, std::string(endpoint.body_prefix) + ",\"" + endpoint.limit + "\":3}"), 200);
    assert(server.backend->LastCall().max_tokens == 3);
    for (const std::string bad : {"-2", "0", "1.5", "\"x\"", "true"}) {
      ExpectStatus(server.Post(endpoint.path, std::string(endpoint.body_prefix) + ",\"" + endpoint.limit + "\":" + bad + "}"), 400);
    }
  }
}

void TestMessagesErrorsUseTheAnthropicShape() {
  RunningServer server({.api_key = "test-secret"});
  const auto fields = [](const std::string& response) {
    const auto body = gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    return body;
  };
  const auto expect = [&](const std::string& response, int status, std::string_view type) {
    ExpectStatus(response, status);
    const auto body = fields(response);
    assert(body.member_str("type") == "error");
    const auto* error = body.find("error");
    assert(error != nullptr && error->is_object());
    assert(error->member_str("type") == type);
    assert(!error->member_str("message").empty());
    assert(body.member_str("request_id") == ResponseRequestId(response));
    assert(error->contains("code") && !error->member_str("code").empty());   // the stable code stays as an extra field
    assert(!body.contains("code"));
  };
  const std::string auth = "Authorization: Bearer test-secret\r\n";
  const auto post = [&](std::string_view path, std::string_view body, bool authorized = true) {
    return server.Send("POST " + std::string(path) + " HTTP/1.1\r\n" + (authorized ? auth : "") + "Content-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  };
  expect(post("/v1/messages", "{"), 400, "invalid_request_error");
  expect(post("/v1/messages", R"({"model":"nope","messages":[{"role":"user","content":"hi"}],"max_tokens":1})"), 404, "not_found_error");
  expect(post("/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":"x"})"), 400, "invalid_request_error");
  expect(post("/v1/messages/count_tokens", "{"), 400, "invalid_request_error");
  const auto unauthorized = post("/v1/messages", R"({"messages":[]})", false);
  expect(unauthorized, 401, "authentication_error");
  assert(unauthorized.find("WWW-Authenticate: Bearer") != std::string::npos);
  assert(unauthorized.find("test-secret") == std::string::npos);
  // Success and the OpenAI routes are untouched.
  ExpectStatus(post("/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":2})"), 200);
  const auto openai = post("/v1/chat/completions", "{");
  ExpectStatus(openai, 400);
  const auto openai_body = fields(openai);
  assert(openai_body.find("error") != nullptr && openai_body.find("type") == nullptr);
  assert(openai_body.find("error")->contains("code"));
  const auto openai_denied = post("/v1/chat/completions", "{}", false);
  ExpectStatus(openai_denied, 401);
  assert(fields(openai_denied).find("type") == nullptr);
}

// Splits a chunked SSE reply into (event name, parsed data) pairs; the [DONE] sentinel and comment frames are skipped.
std::vector<std::pair<std::string, gufo::json::Value>> SseEvents(const std::string& response) {
  std::vector<std::pair<std::string, gufo::json::Value>> events;
  const auto wire = ChunkedBody(response);
  for (std::size_t at = 0; at < wire.size();) {
    const auto end = wire.find("\n\n", at);
    if (end == std::string::npos)
      break;
    const auto frame = wire.substr(at, end - at);
    at = end + 2;
    const auto data = frame.find("data: ");
    if (data == std::string::npos || frame.find("data: [DONE]") != std::string::npos)
      continue;
    std::string name;
    if (const auto event = frame.find("event: "); event != std::string::npos)
      name = frame.substr(event + 7, frame.find('\n', event) - event - 7);
    events.emplace_back(name, gufo::json::parse(frame.substr(data + 6)));
  }
  return events;
}

void TestResponsesStreamKeepaliveIsAnEvent() {
  RunningServer server({.sse_heartbeat_interval = std::chrono::milliseconds(5)});
  server.backend->admit = true;
  server.backend->token_delay = std::chrono::milliseconds(60);
  const auto stream = server.Post("/v1/responses", R"({"input":"hi","stream":true,"max_output_tokens":4})");
  ExpectStatus(stream, 200);
  const auto wire = ChunkedBody(stream);
  // The keepalive is an event a client counts as activity, not a comment.
  assert(wire.find(": ping") == std::string::npos);
  const auto events = SseEvents(stream);
  assert(!events.empty());
  std::size_t in_progress = 0;
  std::string id;
  std::uint64_t expected = 0;
  for (const auto& [name, data] : events) {
    // One increasing sequence over every event, keepalives included: no duplicate, no gap, wire order.
    assert(data.find("sequence_number") != nullptr);
    assert(data.member_size("sequence_number") == expected);
    ++expected;
    if (name == "response.created")
      id = data.find("response")->member_str("id");
    if (name == "response.in_progress") {
      ++in_progress;
      assert(data.member_str("type") == "response.in_progress");
      // A valid in_progress event: the response object with its id, status in_progress.
      assert(data.find("response")->member_str("id") == id);
      assert(data.find("response")->member_str("status") == "in_progress");
    }
  }
  assert(!id.empty());
  assert(in_progress >= 3);   // the lifecycle one plus at least two keepalives during the 4 x 60 ms of generation
  // The stream still ends normally with the full output.
  assert(wire.find("event: response.completed") != std::string::npos || wire.find("event: response.incomplete") != std::string::npos);
  assert(wire.find("event: response.output_text.delta") != std::string::npos);
  assert(stream.ends_with("0\r\n\r\n"));
  // Chat Completions and Messages streams keep the comment frame.
  const auto chat = server.Post("/v1/chat/completions", R"({"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":4})");
  ExpectStatus(chat, 200);
  assert(ChunkedBody(chat).find(": ping\n\n") != std::string::npos);
  const auto messages = server.Post("/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":4})");
  ExpectStatus(messages, 200);
  assert(ChunkedBody(messages).find(": ping\n\n") != std::string::npos);
}

void TestHeartbeatFrameIsPerResponse() {
  RunningServer server({.sse_heartbeat_interval = std::chrono::milliseconds(20)});
  server.server.add("POST", "/sse-event", [](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body =
            [](const auto& write) {
              (void)write("data: first\n\n");
              std::this_thread::sleep_for(std::chrono::milliseconds(90));
              (void)write("data: last\n\n");
            },
        .heartbeat_frame = [] { return std::string("event: tick\ndata: {}\n\n"); }};
  });
  const auto response = server.Post("/sse-event", "");
  ExpectStatus(response, 200);
  assert(CountOccurrences(ChunkedBody(response), "event: tick\ndata: {}\n\n") >= 2);
  assert(ChunkedBody(response).find(": ping") == std::string::npos);
  // An empty frame writes nothing this round, and the stream is not harmed.
  server.server.add("POST", "/sse-skip", [](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body =
            [](const auto& write) {
              (void)write("data: first\n\n");
              std::this_thread::sleep_for(std::chrono::milliseconds(90));
              (void)write("data: last\n\n");
            },
        .heartbeat_frame = [] { return std::string(); }};
  });
  const auto skipped = server.Post("/sse-skip", "");
  ExpectStatus(skipped, 200);
  assert(ChunkedBody(skipped) == "data: first\n\ndata: last\n\n");
  // The default is still the comment.
  const auto idle = server.Post("/sse-idle", "");
  assert(ChunkedBody(idle).find(": ping\n\n") != std::string::npos);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--debug-hooks") {
    DebugHooksBaseGoldenBytesAndExposure();
    DebugHooksHttpTestSettingsEcho();
    DebugHooksHttpTestResponsesLogprobs();
    DebugHooksHttpTestNullableLegacyLimit();
    DebugHooksHttpTestDisabledAndProbeErrors();
    DebugHooksHttpTestLogprobValidation();
    DebugHooksHttpTestRawScoredOutput();
    DebugHooksHttpTestGeneratedLogprobs();
    DebugHooksHttpTestPromptScores();
    DebugHooksHttpTestPromptScoreErrors();
    return 0;
  }

  DebugHooksBaseGoldenBytesAndExposure();
  // The log assertions below match "[LEVEL] [component]" text written to a
  // redirected stderr sink, so the real stderr's TTY state must not add ANSI
  // tint around the level tag.
  ::setenv("NO_COLOR", "1", 1);
  DebugHooksHttpTestSettingsEcho();
  DebugHooksHttpTestResponsesLogprobs();
  DebugHooksHttpTestNullableLegacyLimit();
  DebugHooksHttpTestDisabledAndProbeErrors();
  DebugHooksHttpTestLogprobValidation();
  DebugHooksHttpTestRawScoredOutput();
  DebugHooksHttpTestGeneratedLogprobs();
  DebugHooksHttpTestPromptScores();
  DebugHooksHttpTestPromptScoreErrors();
  TestMemoryGuardRowsAndDebugRoute();
  TestRequestLogging();
  TestContentTrace();
  TestQuietTiersSuppressLifecycle();
  TestLogLevelFilter();
  TestLogLevelNames();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestFramingAndMetrics();
  TestFallbackBackendMetrics();
  TestLlamaSlotsAndMetrics();
  TestCompatibilityRequests();
  TestDiffusionReadContract();
  TestTextOnlyBackendRefusesImagesBeforeReading();
  TestResponsesToolImages();
  TestModelInputModalities();
  TestLlamaProps();
  TestRawCompletionStreaming();
  TestGenerationFailureLogsCause();
  TestDeviceLoss();
  TestDeviceLossWhileWriterBlocked();
  TestRawCompletionPromptProgress();
  TestRawCompletionCachePrompt();
  TestCompatibilityStopSequences();
  TestCompatibilityThinkingDefaults();
  TestResponseSamplingDefaults();
  TestCompatibilityUtf8();
  TestPeerDisconnect();
  TestStreamingFraming();
  TestSseHeartbeat();
  TestCorsPreflightEchoesRequestedHeaders();
  TestMaxTokensMinusOneAndNull();
  TestMessagesErrorsUseTheAnthropicShape();
  TestResponsesStreamKeepaliveIsAnEvent();
  TestHeartbeatFrameIsPerResponse();
  TestSseHeartbeatShutdown();
  TestDeferredStreamHeaders();
  TestAdmittedStreamHeaders();
  TestSignalShutdown();
  std::cout << "HTTP transport checks passed.\n";
}
