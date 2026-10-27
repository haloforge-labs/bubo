#ifndef GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_
#define GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/diffusion_gemma/engine.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"

namespace gufo::server {

/// HTTP text backend for DiffusionGemma's block diffusion. Up to the
/// engine's session count of requests run at once, their denoising steps
/// batched; output arrives a canvas (256 tokens) at a time. The diffusion
/// sampler replaces token sampling, so temperature/top-k/top-p and penalties
/// do not apply; `seed` makes a request reproducible. Thinking is off unless
/// the request enables it (`chat_template_kwargs.enable_thinking` or
/// `reasoning`); the thinking channel is then returned as reasoning, and
/// otherwise removed. Raw completions keep it as `<think>` text. Tools are
/// rendered in the checkpoint's native format and its calls are returned to
/// the adapter as Qwen XML call blocks, which it parses into tool calls.
/// JSON-constrained output and speculative decoding are not available.
class DiffusionGemmaBackend final : public TextGenerationBackend {
public:
  using TextGenerationBackend::chat;
  using TextGenerationBackend::complete;
  using TextGenerationBackend::start_chat;

  DiffusionGemmaBackend(const std::filesystem::path& root,
                        const models::diffusion_gemma::EngineOptions& options,
                        std::string model_id,
                        const std::filesystem::path& mmproj = {},
                        bool vision_disabled = false);

  [[nodiscard]] std::string model_id() const override { return model_id_; }
  [[nodiscard]] bool ready() const override { return true; }
  [[nodiscard]] bool supports_images() const override {
    return vision_ != nullptr;
  }
  /// The diffusion sampler cannot apply a JSON grammar token by token.
  [[nodiscard]] bool supports_response_format() const override { return false; }
  [[nodiscard]] bool supports_diffusion_reads() const override { return true; }
  [[nodiscard]] std::size_t diffusion_canvas_length() const override {
    return static_cast<std::size_t>(engine_->config().canvas);
  }
  [[nodiscard]] std::size_t vocab_size() const override {
    return static_cast<std::size_t>(engine_->config().vocab);
  }
  [[nodiscard]] std::string token_piece(
      tokenization::TokenId id) const override;
  [[nodiscard]] std::uint32_t max_context() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  /// A thinking reply starts with `<think>` (the thinking channel).
  [[nodiscard]] InitialOutputState initial_output_state(
      const ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kAuto
               : InitialOutputState::kContent;
  }

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true) override;

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) override;

  /// Validates and expands the prompt before a stream header is sent.
  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      bool stream_output = false) override;

  /// Raw completions, with `ignore_eos` (fixed-length benchmarks).
  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool debug_timeline = false,
      bool cache_prompt = true) override;

  std::shared_ptr<GenerationRequest> start_complete_tokens(
      const TokenCompletionRequest&, std::size_t,
      const sampling::SamplingConfig&, const CancellationCheck& = {},
      bool stream_output = false) override;

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;
  [[nodiscard]] std::optional<std::size_t> count_chat_tokens(
      const ChatRequest& request) const override;

  void set_default_max_tokens(std::size_t tokens) {
    default_max_tokens_ = tokens;
  }
  void set_max_image_tokens(std::uint32_t tokens) {
    max_image_tokens_ = tokens;
  }
  /// See Engine::Warmup.
  double Warmup() { return engine_->Warmup(); }

private:
  /// The same text/tool/reasoning prompt for generation and token counting.
  [[nodiscard]] models::gemma4::vision::Prompt PrepareChat(
      const ChatRequest& request) const;
  /// How the thinking channel appears in the reply.
  enum class Thinking { kRemove, kTagged };
  Result CompleteRaw(std::string_view prompt, std::size_t max_tokens,
                     const sampling::SamplingConfig& sampling,
                     const CancellationCheck& is_cancelled,
                     const TokenCallback& on_token, std::string_view client_id,
                     const std::vector<std::string>& stop_sequences,
                     bool ignore_eos, bool cache_prompt);
  Result Run(const models::gemma4::vision::Prompt& prompt,
             std::size_t max_tokens, const sampling::SamplingConfig& sampling,
             const CancellationCheck& is_cancelled,
             const TokenCallback& on_token, std::string_view client_id,
             const std::vector<std::string>& stop_sequences, Thinking thinking,
             bool ignore_eos, bool cache_prompt = true);
  /// Memory guard: refuses an image request whose vision memory does not fit (see the definition).
  void AdmitVisionMemory(const models::gemma4::vision::Prompt& prompt);
  Result RunRead(const models::gemma4::vision::Prompt& prompt,
                 const DiffusionReadParams& params,
                 const sampling::SamplingConfig& sampling,
                 const CancellationCheck& is_cancelled,
                 std::string_view client_id, bool cache_prompt);

  std::unique_ptr<models::diffusion_gemma::Engine> engine_;
  std::shared_ptr<models::gemma4::vision::Encoder> vision_;
  mutable std::mutex vision_mutex_;
  std::atomic<bool> vision_resident_{false};
  std::string model_id_;
  std::size_t default_max_tokens_{0};
  std::uint32_t max_image_tokens_{0};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_
