#ifndef GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
#define GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/debug_timeline.hpp"
#include "src/cli/serve/diffusion_reads.hpp"
#include "src/core/reasoning.hpp"
#include "src/core/sampling.hpp"
#include "src/core/text_sampling_defaults.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace gufo::server {

enum class TextGenerationErrorCode : std::uint8_t {
  kQueueFull,
  kClientQueueFull,
  kDeadlineExceeded,
  kOutputLimit,
  kOutputBackpressure,
  kSchedulerStopping,
  kToolChoiceUnsatisfied,
  kDeviceLost,
  /// Prompt scoring needed more host memory than was available.
  kScoringOutOfMemory,
  /// A prompt-scoring response would exceed the response size cap.
  kResponseTooLarge,
  /// Memory guard: a request needs memory it cannot run without and the free memory is short now (retry later).
  kInsufficientMemory,
  /// Memory guard: the request needs more memory than this machine could ever give (do not retry).
  kInsufficientMemoryPermanent,
  /// Memory guard: a running request was stopped because Windows ran low on memory.
  kMemoryPressure,
};

/// The numbers behind a memory-guard refusal, for the request row and the error body.
struct MemoryRefusal {
  std::uint64_t needed_bytes{0};
  std::uint64_t free_bytes{0};
};

/// Client-facing message for a GPU context that can no longer execute work.
inline constexpr const char* kDeviceLostMessage =
    "GPU context lost; restart required";

class TextGenerationError final : public std::runtime_error {
public:
  TextGenerationError(TextGenerationErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}
  TextGenerationError(TextGenerationErrorCode code, const std::string& message, MemoryRefusal refusal)
      : std::runtime_error(message), code_(code), refusal_(refusal) {}

  [[nodiscard]] TextGenerationErrorCode code() const noexcept { return code_; }
  [[nodiscard]] int http_status() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
      case TextGenerationErrorCode::kClientQueueFull:
        return 429;
      case TextGenerationErrorCode::kDeadlineExceeded:
        return 408;
      case TextGenerationErrorCode::kOutputLimit:
      case TextGenerationErrorCode::kOutputBackpressure:
      case TextGenerationErrorCode::kSchedulerStopping:
      case TextGenerationErrorCode::kDeviceLost:
      case TextGenerationErrorCode::kScoringOutOfMemory:
      case TextGenerationErrorCode::kInsufficientMemory:
      case TextGenerationErrorCode::kMemoryPressure:
        return 503;
      case TextGenerationErrorCode::kInsufficientMemoryPermanent:
        return 507;
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return 502;
      case TextGenerationErrorCode::kResponseTooLarge:
        return 413;
    }
    return 500;
  }
  [[nodiscard]] const char* stable_code() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
        return "queue_full";
      case TextGenerationErrorCode::kClientQueueFull:
        return "client_queue_full";
      case TextGenerationErrorCode::kDeadlineExceeded:
        return "deadline_exceeded";
      case TextGenerationErrorCode::kOutputLimit:
        return "output_limit";
      case TextGenerationErrorCode::kOutputBackpressure:
        return "output_backpressure";
      case TextGenerationErrorCode::kSchedulerStopping:
        return "scheduler_stopping";
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return "tool_choice_unsatisfied";
      case TextGenerationErrorCode::kDeviceLost:
        return "device_lost";
      case TextGenerationErrorCode::kScoringOutOfMemory:
        return "scoring_out_of_memory";
      case TextGenerationErrorCode::kResponseTooLarge:
        return "response_too_large";
      case TextGenerationErrorCode::kInsufficientMemory:
        return "insufficient_memory";
      case TextGenerationErrorCode::kInsufficientMemoryPermanent:
        return "insufficient_memory_permanent";
      case TextGenerationErrorCode::kMemoryPressure:
        return "memory_pressure";
    }
    return "generation_error";
  }
  [[nodiscard]] bool retryable() const noexcept {
    return code_ != TextGenerationErrorCode::kOutputLimit &&
           code_ != TextGenerationErrorCode::kToolChoiceUnsatisfied &&
           code_ != TextGenerationErrorCode::kDeviceLost &&
           code_ != TextGenerationErrorCode::kResponseTooLarge &&
           code_ != TextGenerationErrorCode::kInsufficientMemoryPermanent;
  }
  /// Seconds for the Retry-After header of a retryable error.
  [[nodiscard]] int retry_after_seconds() const noexcept {
    return code_ == TextGenerationErrorCode::kInsufficientMemory ||
                   code_ == TextGenerationErrorCode::kMemoryPressure
               ? 5
               : 1;
  }
  /// True for the three memory-guard codes (request rows report outcome "refused").
  [[nodiscard]] bool memory_guard() const noexcept {
    return code_ == TextGenerationErrorCode::kInsufficientMemory ||
           code_ == TextGenerationErrorCode::kInsufficientMemoryPermanent ||
           code_ == TextGenerationErrorCode::kMemoryPressure;
  }
  [[nodiscard]] const std::optional<MemoryRefusal>& refusal() const noexcept { return refusal_; }

private:
  TextGenerationErrorCode code_;
  std::optional<MemoryRefusal> refusal_;
};

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<tokenization::ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<tokenization::ChatMessage> messages;
  std::vector<tokenization::ChatTool> tools;
  std::string client_id{"anonymous"};
  ToolChoice tool_choice{ToolChoice::kAuto};
  std::string forced_tool_name;
  /// Responses namespace of each flattened function, echoed on its calls.
  std::map<std::string, std::string> tool_namespaces;
  bool constrained_tools{false};
  bool parallel_tool_calls{true};
  ReasoningOptions reasoning;
  bool add_vision_id{false};
  /// chat_template_kwargs.tool_call_format == "json". Only a model whose own
  /// chat template offers JSON tool calls (Cyber-Tiel) acts on it.
  bool json_tool_calls{false};
  /// Bypass prompt reuse for this request; its completed state may be retained.
  bool cache_prompt{true};
  /// Optional streaming prompt progress; never changes prompt or cache
  /// identity.
  bool return_progress{false};
  bool debug_timeline{false};
  std::vector<std::string> stop_sequences;
  std::shared_ptr<const sampling::JsonConstraint> response_format;
  std::string response_format_description;
  std::optional<DiffusionReadParams> diffusion_read;
};

/// Model-agnostic text generation boundary used by the HTTP transport.
///
/// Implementations retain ownership of tokenization, templates, complete
/// continuation state, and execution resources. The transport observes only
/// generated text pieces and aggregate request metrics.
class TextGenerationBackend {
public:
  using CancellationCheck = std::function<bool()>;
  using TokenCallback = std::function<bool(std::string_view)>;
  using ScoredTokenCallback =
      std::function<bool(std::string_view, const sampling::TokenLogprob&)>;

  /// llama-server `prompt_progress`; `processed` includes cached tokens.
  struct PromptProgress {
    std::size_t total{0};
    std::size_t cache{0};
    std::size_t processed{0};
    std::int64_t time_ms{0};
  };
  using ProgressCallback = std::function<bool(const PromptProgress&)>;
  /// Called at most once, before progress or output, when a streaming request
  /// starts work or has waited in the queue for a bounded time.
  using StartCallback = std::function<bool()>;

  /// One execution session for llama-server `/slots`. Idle sessions report
  /// zero counts rather than the previous request's.
  struct SessionState {
    bool processing{false};
    bool speculative{false};
    std::uint64_t request_id{0};
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    /// Prompt tokens prefilled so far, excluding cached tokens.
    std::size_t processed_prompt_tokens{0};
    std::size_t generated_tokens{0};
    std::size_t remaining_tokens{0};
  };

  enum class FinishReason : std::uint8_t {
    kStop,
    kStopSequence,
    kLength,
    kCancelled,
  };

  enum class TerminationCause : std::uint8_t {
    kNone,
    kToolArgumentLoop,
  };

  enum class InitialOutputState : std::uint8_t {
    kAuto,
    kReasoning,
    kContent,
  };

  /// Model-specific markup of generated text: the reasoning delimiters and
  /// the tool-call syntax the HTTP adapters parse.
  struct OutputMarkup {
    enum class ToolSyntax : std::uint8_t { kQwenOrDsml, kGemma4 };
    std::string_view reasoning_start{"<think>"};
    std::string_view reasoning_end{"</think>"};
    ToolSyntax tool_syntax{ToolSyntax::kQwenOrDsml};
  };

  /// Where the model's own end-of-reasoning tag begins in the generated text,
  /// found from token ids. A literal tag the model writes as ordinary text
  /// tokens (for example while quoting code) is not an end. Backends that do
  /// not track tokens leave `tracked` false, and the HTTP adapters search the
  /// text for the tag as before.
  struct ReasoningEnd {
    bool tracked{false};
    /// Byte offset of the first generated tag; empty while none was generated.
    std::optional<std::size_t> byte;
  };

  struct SamplingDefaults {
    /// Zero means generate until EOS or the remaining context is exhausted.
    std::size_t max_tokens{0};
    sampling::SamplingConfig sampling;
    sampling::TextModelPreset model{sampling::TextModelPreset::kUnspecified};
    sampling::SamplingOverrides supplied{sampling::SamplingOverrides::All()};

    [[nodiscard]] sampling::SamplingConfig Resolve(
        std::optional<bool> thinking) const {
      return sampling::ResolveTextSampling(model, thinking, sampling, supplied);
    }
  };

  struct Result {
    std::string text;
    std::vector<tokenization::TokenId> tokens;
    std::vector<sampling::TokenLogprob> logprobs;
    std::vector<std::optional<sampling::TokenLogprob>> prompt_logprobs;
    /// Present only for a read-only diffusion request; one row per position.
    std::optional<std::vector<DiffusionReadRow>> diffusion_rows;
    std::shared_ptr<DebugTimeline> debug_timeline;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    std::size_t cache_restore_bytes{0};
    std::size_t cache_snapshot_bytes{0};
    /// Bytes admitted to background persistence; completion is logged by cache.
    std::size_t cache_disk_queued_bytes{0};
    std::size_t cache_shared_bytes{0};
    /// Shared-prefix snapshots queued while prefilling this request.
    std::size_t cache_shared_prefix_snapshots{0};
    std::size_t cache_shared_prefix_bytes{0};
    std::size_t completion_tokens{0};
    /// Generated reasoning tokens, excluding the closing template delimiter.
    std::size_t reasoning_tokens{0};
    ReasoningEnd reasoning_end;
    /// Number of speculative verification rounds actually executed.
    std::size_t draft_rounds{0};
    std::size_t draft_tokens{0};
    std::size_t draft_accepted_tokens{0};
    std::size_t lookup_tokens{0};
    std::size_t lookup_accepted_tokens{0};
    std::size_t prefill_tokens{0};
    std::size_t prefill_chunks{0};
    std::size_t active_decode_prefill_chunks{0};
    std::size_t max_prefill_chunk_tokens{0};
    std::size_t max_consecutive_active_prefill_chunks{0};
    std::size_t configured_active_prefill_tokens{0};
    std::size_t queue_depth_at_submit{0};
    std::size_t client_queue_depth_at_submit{0};
    std::size_t resident_requests_at_admission{0};
    std::size_t requested_logical_concurrency{1};
    std::size_t physical_execution_width{1};
    std::size_t max_buffered_output_bytes{0};
    double queue_ms{0.0};
    /// Part of queue_ms spent waiting for a concurrent request to publish
    /// the prompt prefix both share.
    double shared_prefix_wait_ms{0.0};
    double cache_restore_ms{0.0};
    double cache_snapshot_ms{0.0};
    double cache_disk_enqueue_ms{0.0};
    double cache_shared_prefix_ms{0.0};
    double ttft_ms{0.0};
    double mean_inter_token_ms{0.0};
    double max_inter_token_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::string client_id{"anonymous"};
    std::string execution_plan{"serial-c1"};
    std::string prefill_fallback_reason;
    std::string cache_miss_reason;
    /// Snapshot copies this request wanted that the cache refused, and why (GitHub #9).
    std::size_t cache_snapshots_skipped{0};
    std::string cache_snapshot_skip_reason;
    /// Retained snapshots when the request ended: the payload sum the byte budget charges, and the device and host bytes they occupy.
    std::size_t cache_retained_bytes{0};
    std::size_t cache_retained_device_bytes{0};
    std::size_t cache_retained_host_bytes{0};
    std::size_t cache_common_prefix_tokens{0};
    std::size_t cache_checkpoint_tokens{0};
    FinishReason finish_reason{FinishReason::kStop};
    TerminationCause termination_cause{TerminationCause::kNone};
    std::string stop_sequence;
    bool incremental_prefill_supported{false};
    bool cache_hit{false};
    bool cache_disk_hit{false};
    bool cancelled{false};
    /// Internal: the scheduler recorded this request's `/metrics` counters.
    bool token_metrics_recorded{false};
    std::optional<models::qwen38_flash_next::PrefillDiagTotals> prefill_diag;
    /// Actual scheduler slot; backends that do not expose it leave it null.
    std::optional<std::size_t> slot_id;
  };

  class GenerationRequest {
  public:
    GenerationRequest() = default;
    virtual ~GenerationRequest() = default;

    GenerationRequest(const GenerationRequest&) = delete;
    GenerationRequest& operator=(const GenerationRequest&) = delete;
    GenerationRequest(GenerationRequest&&) = delete;
    GenerationRequest& operator=(GenerationRequest&&) = delete;

    /// Streaming requests may report their start and prompt progress before
    /// any token. Backends without a queue may never report the start.
    virtual Result Wait(const TokenCallback& on_token = {},
                        const ProgressCallback& on_progress = {},
                        const StartCallback& on_start = {}) = 0;
    virtual Result WaitScored(const ScoredTokenCallback&) {
      throw std::logic_error("backend does not implement scored streaming");
    }
    virtual void Cancel() noexcept = 0;
    /// The reasoning end generated so far. A piece delivered to Wait's
    /// callback is already covered, so a streaming adapter can ask while it
    /// handles the piece that carries the tag.
    [[nodiscard]] virtual ReasoningEnd CurrentReasoningEnd() const {
      return {};
    }
    /// Effective constrained tool format, including any schema fallback.
    /// Available before Wait and stable for this admitted request. Backends
    /// without this metadata retain the adapter's legacy format detection.
    [[nodiscard]] virtual std::optional<sampling::JsonConstraint::ToolFormat>
    ToolFormat() const {
      return std::nullopt;
    }
  };

  TextGenerationBackend() = default;
  virtual ~TextGenerationBackend() = default;

  TextGenerationBackend(const TextGenerationBackend&) = delete;
  TextGenerationBackend& operator=(const TextGenerationBackend&) = delete;
  TextGenerationBackend(TextGenerationBackend&&) = delete;
  TextGenerationBackend& operator=(TextGenerationBackend&&) = delete;

  [[nodiscard]] virtual std::string model_id() const = 0;
  [[nodiscard]] virtual bool ready() const = 0;
  /// True once a failed generation showed that the execution device can no
  /// longer run work, for example after a GPU reset. The loss is permanent.
  [[nodiscard]] virtual bool device_lost() const { return false; }
  /// Maximum tokens accepted by the loaded model under the configured context.
  /// Zero when no text model is loaded.
  [[nodiscard]] virtual std::uint32_t max_context() const { return 0; }
  /// Live execution sessions; empty when the backend has no session pool.
  [[nodiscard]] virtual std::vector<SessionState> session_states() const {
    return {};
  }
  /// Whether the loaded backend accepts image inputs in chat requests.
  [[nodiscard]] virtual bool supports_images() const { return false; }
  /// False when the backend cannot constrain output to a JSON schema
  /// (response_format / text.format); such requests are rejected with 400.
  [[nodiscard]] virtual bool supports_response_format() const { return true; }
  [[nodiscard]] virtual bool supports_logprobs() const { return false; }
  [[nodiscard]] virtual bool supports_diffusion_reads() const { return false; }
  [[nodiscard]] virtual std::size_t diffusion_canvas_length() const {
    return 0;
  }
  [[nodiscard]] virtual std::size_t vocab_size() const { return 0; }
  [[nodiscard]] virtual bool supports_timeline() const { return false; }
  [[nodiscard]] virtual json::Value debug_prompt(const ChatRequest*,
                                                 std::string_view) const {
    throw std::invalid_argument(
        "prompt inspection is not supported for this model");
  }
  [[nodiscard]] virtual std::string token_piece(tokenization::TokenId) const {
    throw std::invalid_argument("logprobs are not supported for this model");
  }
  virtual std::shared_ptr<GenerationRequest> start_score(
      std::string_view, const sampling::SamplingConfig&, std::size_t,
      const CancellationCheck& = {}) {
    throw std::invalid_argument(
        "prompt logprobs are not supported for this model");
  }
  [[nodiscard]] virtual SamplingDefaults sampling_defaults() const {
    return {};
  }
  [[nodiscard]] virtual ReasoningOptions reasoning_defaults() const {
    return {};
  }
  /// The loaded tokenizer's control tokens, or null when it owns none. The
  /// parser treats a pipe-wrapped spelling as call framing only when this trie
  /// knows it; every other lookalike stays literal argument data (#383).
  [[nodiscard]] virtual InitialOutputState initial_output_state(
      const ChatRequest&) const {
    return InitialOutputState::kAuto;
  }
  [[nodiscard]] virtual OutputMarkup output_markup() const { return {}; }

  virtual Result complete(std::string_view prompt, std::size_t max_tokens,
                          const sampling::SamplingConfig& sampling,
                          const CancellationCheck& is_cancelled = {},
                          const TokenCallback& on_token = {},
                          std::string_view client_id = "anonymous",
                          const std::vector<std::string>& stop_sequences = {},
                          bool cache_prompt = true) = 0;

  virtual Result chat(const ChatRequest& request, std::size_t max_tokens,
                      const sampling::SamplingConfig& sampling,
                      const CancellationCheck& is_cancelled = {},
                      const TokenCallback& on_token = {}) = 0;

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature, const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return complete(prompt, max_tokens, config, is_cancelled, on_token,
                    "anonymous", stop_sequences, cache_prompt);
  }

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              float temperature, const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return chat(request, max_tokens, config, is_cancelled, on_token);
  }

  /// Reserves admission before a streaming response commits successful headers.
  ///
  /// Backends with an asynchronous scheduler override this. The default
  /// request defers the existing blocking chat call until Wait().
  virtual std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false);

  /// Raw prompt generation with admission reserved before SSE headers.
  virtual std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool debug_timeline = false,
      bool cache_prompt = true);

  /// Token-ID prompt for DiffusionGemma reads and prefix generation.
  virtual std::shared_ptr<GenerationRequest> start_complete_tokens(
      const TokenCompletionRequest&, std::size_t,
      const sampling::SamplingConfig&, const CancellationCheck& = {},
      bool = false) {
    throw std::invalid_argument(
        "this model does not support token-ID completions");
  }

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens, float temperature,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return start_chat(request, max_tokens, config, is_cancelled, stream_output);
  }

  [[nodiscard]] virtual std::size_t count_tokens(
      std::string_view text) const = 0;

  /// Prompt tokens a chat request would use after template rendering, without
  /// generating. Empty when the backend cannot tell.
  [[nodiscard]] virtual std::optional<std::size_t> count_chat_tokens(
      const ChatRequest& request) const = 0;
};

inline json::Value SerializeDebugTimeline(
    const TextGenerationBackend::Result& result) {
  if (!result.debug_timeline)
    throw std::invalid_argument("timeline was not requested");
  return result.debug_timeline->Serialize();
}

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool debug_timeline, bool cache_prompt) {
  (void)stream_output;
  (void)return_progress;
  if (debug_timeline)
    throw std::invalid_argument("timeline is not supported by this backend");
  if (ignore_eos)
    throw std::invalid_argument("backend does not support ignore_eos");
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              std::string prompt, std::size_t max_tokens,
                              sampling::SamplingConfig sampling,
                              CancellationCheck cancellation,
                              std::string client_id,
                              std::vector<std::string> stop_sequences,
                              bool cache_prompt)
        : backend_(backend),
          prompt_(std::move(prompt)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          cancellation_(std::move(cancellation)),
          client_id_(std::move(client_id)),
          stop_sequences_(std::move(stop_sequences)),
          cache_prompt_(cache_prompt) {}

    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel))
        throw std::logic_error("generation request was already consumed");
      return backend_.complete(
          prompt_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (cancellation_ && cancellation_());
          },
          on_token, client_id_, stop_sequences_, cache_prompt_);
    }
    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    std::string prompt_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck cancellation_;
    std::string client_id_;
    std::vector<std::string> stop_sequences_;
    bool cache_prompt_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };
  return std::make_shared<DeferredGenerationRequest>(
      *this, std::string(prompt), max_tokens, sampling, is_cancelled,
      std::string(client_id), stop_sequences, cache_prompt);
}

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_chat(const ChatRequest& request,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  const CancellationCheck& is_cancelled,
                                  bool stream_output) {
  (void)stream_output;
  if (request.debug_timeline)
    throw std::invalid_argument("timeline is not supported by this backend");
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              ChatRequest chat_request, std::size_t token_limit,
                              sampling::SamplingConfig sampling_config,
                              CancellationCheck external_cancellation)
        : backend_(backend),
          request_(std::move(chat_request)),
          max_tokens_(token_limit),
          sampling_(sampling_config),
          external_cancellation_(std::move(external_cancellation)) {}

    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel)) {
        throw std::logic_error("generation request was already consumed");
      }
      return backend_.chat(
          request_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (external_cancellation_ && external_cancellation_());
          },
          on_token);
    }

    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    ChatRequest request_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck external_cancellation_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };

  return std::make_shared<DeferredGenerationRequest>(*this, request, max_tokens,
                                                     sampling, is_cancelled);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
