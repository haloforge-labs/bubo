#include "src/cli/serve/diffusion_gemma_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

#include "src/cli/serve/diffusion_gemma_options.hpp"
#include "src/cli/serve/diffusion_gemma_read.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/platform/memory_guard.hpp"
#include "src/models/diffusion_gemma/chat.hpp"
#include "src/models/diffusion_gemma/image_chat.hpp"
#include "src/models/gemma4/chat_template.hpp"

namespace gufo::server {
namespace dg = models::diffusion_gemma;

namespace {

// Bytes at the end of `text` that start an incomplete UTF-8 sequence.
std::size_t IncompleteUtf8Tail(std::string_view text) {
  const std::size_t n = text.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, n); ++back) {
    const auto byte = static_cast<unsigned char>(text[n - back]);
    if ((byte & 0xC0) == 0x80)
      continue;  // Continuation byte: keep looking for the lead.
    std::size_t length = 1;
    if ((byte & 0xE0) == 0xC0)
      length = 2;
    else if ((byte & 0xF0) == 0xE0)
      length = 3;
    else if ((byte & 0xF8) == 0xF0)
      length = 4;
    return length > back ? back : 0;
  }
  return 0;
}

// The generated tokens as text, the thinking channel
// (`<|channel>thought\n...<channel|>`) as `<think>...</think>`.
std::string DecodeTaggedThinking(const dg::Tokenizer& tokenizer,
                                 std::span<const std::int32_t> tokens,
                                 std::int32_t open, std::int32_t close) {
  std::string text;
  std::size_t start = 0;
  const auto flush = [&](std::size_t end) {
    text += tokenizer.Decode(tokens.subspan(start, end - start), false);
    start = end + 1;
  };
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == open) {
      flush(i);
      text += "<think>";
    } else if (tokens[i] == close) {
      flush(i);
      text += "</think>";
    }
  }
  flush(tokens.size());
  // The channel's name is not part of the reasoning.
  for (std::size_t at = text.find("<think>thought"); at != std::string::npos;
       at = text.find("<think>thought", at + 7)) {
    std::size_t end = at + 14;
    if (end < text.size() && text[end] == '\n')
      ++end;
    text.erase(at + 7, end - at - 7);
  }
  return text;
}

}  // namespace

DiffusionGemmaBackend::DiffusionGemmaBackend(
    const std::filesystem::path& root, const dg::EngineOptions& options,
    std::string model_id, const std::filesystem::path& mmproj,
    bool vision_disabled)
    : engine_(std::make_unique<dg::Engine>(root, options)),
      vision_(ShouldBindDiffusionGemmaVision(mmproj, vision_disabled)
                  ? std::make_shared<models::gemma4::vision::Encoder>(
                        mmproj,
                        static_cast<std::uint32_t>(engine_->config().hidden))
                  : nullptr),
      model_id_(std::move(model_id)) {}

std::uint32_t DiffusionGemmaBackend::max_context() const {
  return static_cast<std::uint32_t>(engine_->max_context());
}

TextGenerationBackend::SamplingDefaults
DiffusionGemmaBackend::sampling_defaults() const {
  SamplingDefaults defaults;
  defaults.max_tokens = default_max_tokens_;
  return defaults;
}

std::size_t DiffusionGemmaBackend::count_tokens(std::string_view text) const {
  return engine_->tokenizer().Encode(text).size();
}

std::string DiffusionGemmaBackend::token_piece(tokenization::TokenId id) const {
  if (id >= static_cast<tokenization::TokenId>(engine_->config().vocab))
    throw std::invalid_argument(
        "token id is outside DiffusionGemma vocabulary");
  const std::int32_t token = static_cast<std::int32_t>(id);
  return engine_->tokenizer().Decode(std::span(&token, 1), false);
}

std::optional<std::size_t> DiffusionGemmaBackend::count_chat_tokens(
    const ChatRequest& request) const {
  return PrepareChat(request).tokens.size();
}

TextGenerationBackend::Result DiffusionGemmaBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt) {
  return CompleteRaw(prompt, max_tokens, sampling, is_cancelled, on_token,
                     client_id, stop_sequences, false, cache_prompt);
}

TextGenerationBackend::Result DiffusionGemmaBackend::CompleteRaw(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool ignore_eos, bool cache_prompt) {
  models::gemma4::vision::Prompt prepared;
  prepared.tokens = engine_->tokenizer().Encode(prompt);
  if (prepared.tokens.empty() ||
      prepared.tokens.front() != engine_->config().bos_id)
    prepared.tokens.insert(prepared.tokens.begin(), engine_->config().bos_id);
  return Run(prepared, max_tokens, sampling, is_cancelled, on_token, client_id,
             stop_sequences, Thinking::kTagged, ignore_eos, cache_prompt);
}

std::shared_ptr<TextGenerationBackend::GenerationRequest>
DiffusionGemmaBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool debug_timeline, bool cache_prompt) {
  (void)stream_output;
  (void)return_progress;
  if (debug_timeline)
    throw std::invalid_argument("timeline is not supported by this backend");
  // Runs on Wait, like the default deferred request, with ignore_eos.
  class Request final : public GenerationRequest {
  public:
    Request(DiffusionGemmaBackend& backend, std::string prompt,
            std::size_t max_tokens, sampling::SamplingConfig sampling,
            CancellationCheck cancellation, std::string client_id,
            std::vector<std::string> stop_sequences, bool ignore_eos,
            bool cache_prompt)
        : backend_(backend),
          prompt_(std::move(prompt)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          cancellation_(std::move(cancellation)),
          client_id_(std::move(client_id)),
          stop_sequences_(std::move(stop_sequences)),
          ignore_eos_(ignore_eos),
          cache_prompt_(cache_prompt) {}
    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel))
        throw std::logic_error("generation request was already consumed");
      return backend_.CompleteRaw(
          prompt_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (cancellation_ && cancellation_());
          },
          on_token, client_id_, stop_sequences_, ignore_eos_, cache_prompt_);
    }
    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    DiffusionGemmaBackend& backend_;
    std::string prompt_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck cancellation_;
    std::string client_id_;
    std::vector<std::string> stop_sequences_;
    bool ignore_eos_;
    bool cache_prompt_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };
  return std::make_shared<Request>(
      *this, std::string(prompt), max_tokens, sampling, is_cancelled,
      std::string(client_id), stop_sequences, ignore_eos, cache_prompt);
}

std::shared_ptr<TextGenerationBackend::GenerationRequest>
DiffusionGemmaBackend::start_complete_tokens(const TokenCompletionRequest&,
                                             std::size_t,
                                             const sampling::SamplingConfig&,
                                             const CancellationCheck&, bool) {
  throw std::invalid_argument(
      "DiffusionGemma token-ID completions are not supported yet");
}

std::shared_ptr<TextGenerationBackend::GenerationRequest>
DiffusionGemmaBackend::start_chat(const ChatRequest& request,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  const CancellationCheck& is_cancelled,
                                  bool stream_output) {
  if (request.diffusion_read) {
    if (stream_output)
      throw std::invalid_argument(
          "streaming diffusion reads are not supported");
    if (request.debug_timeline)
      throw std::invalid_argument("timeline is not supported by this backend");
    if (!request.stop_sequences.empty())
      throw std::invalid_argument(
          "stop sequences are not supported for diffusion reads");
    const auto& params = *request.diffusion_read;
    (void)PlanDiffusionGemmaRead(
        params, engine_->config().canvas, engine_->config().vocab,
        std::max(sampling.logprobs, 0), 0, request.cache_prompt);
    auto prompt = PrepareChat(request);
    class ReadRequest final : public GenerationRequest {
    public:
      ReadRequest(DiffusionGemmaBackend& backend,
                  models::gemma4::vision::Prompt prompt,
                  DiffusionReadParams params, sampling::SamplingConfig sampling,
                  CancellationCheck cancellation, std::string client_id,
                  bool cache_prompt)
          : backend_(backend),
            prompt_(std::move(prompt)),
            params_(std::move(params)),
            sampling_(sampling),
            cancellation_(std::move(cancellation)),
            client_id_(std::move(client_id)),
            cache_prompt_(cache_prompt) {}
      Result Wait(const TokenCallback&, const ProgressCallback&,
                  const StartCallback&) override {
        if (waited_.exchange(true, std::memory_order_acq_rel))
          throw std::logic_error("generation request was already consumed");
        return backend_.RunRead(
            prompt_, params_, sampling_,
            [this] {
              return cancelled_.load(std::memory_order_acquire) ||
                     (cancellation_ && cancellation_());
            },
            client_id_, cache_prompt_);
      }
      void Cancel() noexcept override {
        cancelled_.store(true, std::memory_order_release);
      }

    private:
      DiffusionGemmaBackend& backend_;
      models::gemma4::vision::Prompt prompt_;
      DiffusionReadParams params_;
      sampling::SamplingConfig sampling_;
      CancellationCheck cancellation_;
      std::string client_id_;
      bool cache_prompt_;
      std::atomic<bool> waited_{false};
      std::atomic<bool> cancelled_{false};
    };
    return std::make_shared<ReadRequest>(
        *this, std::move(prompt), params, sampling, is_cancelled,
        request.client_id, request.cache_prompt);
  }
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  if (!has_images)
    return TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                             is_cancelled, stream_output);
  (void)stream_output;
  if (request.debug_timeline)
    throw std::invalid_argument("timeline is not supported by this backend");
  auto prompt = PrepareChat(request);
  class Request final : public GenerationRequest {
  public:
    Request(DiffusionGemmaBackend& backend,
            models::gemma4::vision::Prompt prompt, std::size_t max_tokens,
            sampling::SamplingConfig sampling, CancellationCheck cancellation,
            std::string client_id, std::vector<std::string> stop_sequences,
            bool thinking, bool cache_prompt)
        : backend_(backend),
          prompt_(std::move(prompt)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          cancellation_(std::move(cancellation)),
          client_id_(std::move(client_id)),
          stop_sequences_(std::move(stop_sequences)),
          thinking_(thinking),
          cache_prompt_(cache_prompt) {}
    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      if (waited_.exchange(true, std::memory_order_acq_rel))
        throw std::logic_error("generation request was already consumed");
      return backend_.Run(
          prompt_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (cancellation_ && cancellation_());
          },
          on_token, client_id_, stop_sequences_,
          thinking_ ? Thinking::kTagged : Thinking::kRemove, false,
          cache_prompt_);
    }
    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    DiffusionGemmaBackend& backend_;
    models::gemma4::vision::Prompt prompt_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck cancellation_;
    std::string client_id_;
    std::vector<std::string> stop_sequences_;
    bool thinking_;
    bool cache_prompt_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };
  return std::make_shared<Request>(
      *this, std::move(prompt), max_tokens, sampling, is_cancelled,
      request.client_id, request.stop_sequences,
      request.reasoning.enabled.value_or(false), request.cache_prompt);
}

models::gemma4::vision::Prompt DiffusionGemmaBackend::PrepareChat(
    const ChatRequest& request) const {
  std::vector<dg::Message> messages;
  for (const auto& message : request.messages) {
    dg::Message out;
    switch (message.role) {
      case tokenization::ChatRole::kSystem:
        out.role = "system";
        break;
      case tokenization::ChatRole::kDeveloper:
        out.role = "developer";
        break;
      case tokenization::ChatRole::kAssistant:
        out.role = "assistant";
        break;
      case tokenization::ChatRole::kTool:
        out.role = "tool";
        break;
      default:
        out.role = "user";
        break;
    }
    out.content = message.content;
    out.tool_call_id = message.tool_call_id;
    out.name = message.name;
    for (const auto& call : message.tool_calls) {
      dg::ToolCall converted;
      converted.id = call.id;
      converted.name = call.name;
      for (const auto& argument : call.arguments) {
        json::Value value(argument.value);
        if (!argument.is_string) {
          try {
            value = json::parse(argument.value);
          } catch (const std::exception&) {
          }
        }
        converted.arguments[argument.name] = std::move(value);
      }
      out.tool_calls.push_back(std::move(converted));
    }
    messages.push_back(std::move(out));
  }
  std::vector<json::Value> tools;
  for (const auto& tool : request.tools) {
    json::Value definition;
    try {
      definition = json::parse(tool.definition_json);
    } catch (const std::exception&) {
    }
    if (!definition.is_object() || definition.find("function") == nullptr) {
      json::Value function = json::Value::object();
      function["name"] = tool.name;
      function["description"] = tool.description;
      function["parameters"] = json::parse(tool.parameters_json);
      definition = json::Value::object();
      definition["type"] = "function";
      definition["function"] = std::move(function);
    }
    tools.push_back(std::move(definition));
  }
  const bool thinking = request.reasoning.enabled.value_or(false);
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  if (!has_images) {
    models::gemma4::vision::Prompt prompt;
    prompt.tokens =
        engine_->tokenizer().Encode(dg::RenderChat(messages, tools, thinking));
    return prompt;
  }
  ValidateDiffusionGemmaImageRequest(has_images, vision_ != nullptr);
  const auto rendered = dg::RenderImageChat(
      request.messages, std::move(messages), tools, thinking);
  const auto& tokenizer = engine_->tokenizer();
  const models::gemma4::vision::ImageTokens ids{tokenizer.TokenId("<|image>"),
                                                tokenizer.TokenId("<|image|>"),
                                                tokenizer.TokenId("<image|>")};
  if (ids.start != 255999 || ids.soft != 258880 || ids.end != 258882)
    throw std::invalid_argument(
        "DiffusionGemma vocabulary has unexpected image IDs");
  auto prompt = models::gemma4::vision::PrepareRendered(
      rendered.text, rendered.image_offsets, request.messages,
      [&](std::string_view text) { return tokenizer.Encode(text); }, ids,
      vision_->identity(), max_context());
  std::vector<dg::ImageSpan> spans;
  spans.reserve(prompt.images.size());
  for (const auto& image : prompt.images)
    spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  dg::ValidateImagePrompt(prompt.tokens, engine_->prefill_chunk(),
                          engine_->max_context(), engine_->config().canvas,
                          spans);
  std::uint64_t image_tokens = 0;
  for (const auto& image : prompt.images)
    image_tokens += image.span.rows;
  if (max_image_tokens_ && image_tokens > max_image_tokens_)
    throw std::invalid_argument(
        "image tokens exceed the per-request limit (GUFO_MAX_IMAGE_TOKENS)");
  return prompt;
}

TextGenerationBackend::Result DiffusionGemmaBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
  if (request.diffusion_read &&
      (request.debug_timeline || !request.stop_sequences.empty()))
    throw std::invalid_argument(
        "timeline and stop sequences are not supported for diffusion reads");
  const auto prompt = PrepareChat(request);
  if (request.diffusion_read)
    return RunRead(prompt, *request.diffusion_read, sampling, is_cancelled,
                   request.client_id, request.cache_prompt);
  const bool thinking = request.reasoning.enabled.value_or(false);
  return Run(prompt, max_tokens, sampling, is_cancelled, on_token,
             request.client_id, request.stop_sequences,
             thinking ? Thinking::kTagged : Thinking::kRemove, false,
             request.cache_prompt);
}

/// Memory guard (see core/platform/memory_guard.hpp): an image request needs the vision tower and the encoder scratch, committed on the
/// first image. Refused only when that mandatory memory is short; a text request needs nothing new and is never refused here. The guard
/// holds no snapshot cache for this engine, so there is nothing to drop; the check and the readings are the same as for the other runners.
void DiffusionGemmaBackend::AdmitVisionMemory(const models::gemma4::vision::Prompt& prompt) {
  auto* guard = gufo::platform::ProcessMemoryGuard();
  if (guard == nullptr || !guard->enabled() || prompt.images.empty() || !vision_)
    return;
  constexpr std::size_t kScratchBytes = std::size_t{640} << 20;
  gufo::platform::MemoryNeed need;
  need.mandatory_device_bytes = vision_resident_.load(std::memory_order_relaxed) ? kScratchBytes : kDiffusionGemmaVisionReserveBytes;
  const auto result = guard->Admit(need);
  if (result.verdict == gufo::platform::AdmitVerdict::kAdmit)
    return;
  const bool permanent = result.verdict == gufo::platform::AdmitVerdict::kRefusePermanent;
  const auto mib = [](std::uint64_t bytes) { return std::to_string(bytes >> 20); };
  Logger::Warn("memory_guard",
               "event=memory_guard action=refuse kind=" + std::string(permanent ? "permanent" : "transient") +
                   " needed_mib=" + mib(result.needed_bytes) + " free_mib=" + mib(result.free_bytes) +
                   " floor_mib=" + mib(guard->floor_bytes()) + " prompt_tokens=" + std::to_string(prompt.tokens.size()) +
                   " level=" + gufo::platform::GuardLevelName(guard->level()) + " runner=diffusion_gemma");
  throw TextGenerationError(permanent ? TextGenerationErrorCode::kInsufficientMemoryPermanent : TextGenerationErrorCode::kInsufficientMemory,
                            result.message, MemoryRefusal{result.needed_bytes, result.free_bytes});
}

TextGenerationBackend::Result DiffusionGemmaBackend::RunRead(
    const models::gemma4::vision::Prompt& prompt,
    const DiffusionReadParams& params, const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, std::string_view client_id,
    bool cache_prompt) {
  if (prompt.tokens.size() + diffusion_canvas_length() > max_context())
    throw std::invalid_argument(
        "prompt leaves no room for a 256-token read within the context");
  AdmitVisionMemory(prompt);
  const std::uint64_t seed = sampling.seed >= 0
                                 ? static_cast<std::uint64_t>(sampling.seed)
                                 : std::random_device{}() * 0x100000001ULL;
  auto options = PlanDiffusionGemmaRead(
      params, engine_->config().canvas, engine_->config().vocab,
      std::max(sampling.logprobs, 0), seed, cache_prompt);
  options.cancelled = [&is_cancelled] {
    return is_cancelled && is_cancelled();
  };
  std::vector<std::shared_ptr<const models::gemma4::vision::Encoder::Embedding>>
      encoded;
  std::vector<dg::ImageInput> image_inputs;
  encoded.reserve(prompt.images.size());
  image_inputs.reserve(prompt.images.size());
  for (const auto& image : prompt.images) {
    if (!vision_)
      throw std::invalid_argument(
          "image input requires a matching --mmproj BF16 sidecar");
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      encoded.push_back(vision_->Encode(image.pixels, {}, is_cancelled));
    }
    vision_resident_.store(true, std::memory_order_relaxed);
    if (encoded.back()->rows() != image.span.rows ||
        encoded.back()->width() !=
            static_cast<std::uint32_t>(engine_->config().hidden))
      throw std::runtime_error(
          "DiffusionGemma image encoder row shape mismatch");
    image_inputs.push_back(
        {{image.span.offset, image.span.rows, image.prefix_identity},
         encoded.back()->data()});
  }
  const auto read = engine_->Read(prompt.tokens, image_inputs, options);
  return DiffusionGemmaReadResult(read, client_id);
}

TextGenerationBackend::Result DiffusionGemmaBackend::Run(
    const models::gemma4::vision::Prompt& prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    Thinking thinking, bool ignore_eos, bool cache_prompt) {
  const auto& tokenizer = engine_->tokenizer();
  const int room =
      engine_->max_context() - static_cast<int>(prompt.tokens.size());
  if (room < engine_->config().canvas)
    throw std::invalid_argument(
        "prompt of " + std::to_string(prompt.tokens.size()) +
        " tokens leaves no room for a reply within "
        "the " +
        std::to_string(engine_->max_context()) + "-token context");
  if (max_tokens == 0)
    max_tokens = default_max_tokens_;
  if (max_tokens == 0 || max_tokens > static_cast<std::size_t>(room))
    max_tokens = static_cast<std::size_t>(room);
  AdmitVisionMemory(prompt);

  dg::GenerationOptions options;
  options.diffusion = engine_->diffusion_defaults();
  options.max_new_tokens = static_cast<int>(max_tokens);
  options.suppress_end = ignore_eos;
  // Without prompt caching the request encodes its whole prompt.
  options.reuse_prompt = cache_prompt;
  options.seed = sampling.seed >= 0 ? static_cast<std::uint64_t>(sampling.seed)
                                    : std::random_device{}() * 0x100000001ULL;

  options.cancelled = [&is_cancelled] {
    return is_cancelled && is_cancelled();
  };
  const std::int32_t open = tokenizer.TokenId("<|channel>");
  const std::int32_t close = tokenizer.TokenId("<channel|>");
  std::size_t hold = 0;
  for (const auto& stop : stop_sequences)
    hold = std::max(hold, stop.size() - 1);

  Result result;
  result.client_id = std::string(client_id);
  result.execution_plan = "diffusion";
  result.prompt_tokens = prompt.tokens.size();

  std::vector<std::shared_ptr<const models::gemma4::vision::Encoder::Embedding>>
      encoded;
  std::vector<dg::ImageInput> image_inputs;
  encoded.reserve(prompt.images.size());
  image_inputs.reserve(prompt.images.size());
  for (const auto& image : prompt.images) {
    if (!vision_)
      throw std::invalid_argument(
          "image input requires a matching --mmproj BF16 sidecar");
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      encoded.push_back(vision_->Encode(image.pixels, {}, is_cancelled));
    }
    vision_resident_.store(true, std::memory_order_relaxed);
    if (encoded.back()->rows() != image.span.rows ||
        encoded.back()->width() !=
            static_cast<std::uint32_t>(engine_->config().hidden))
      throw std::runtime_error(
          "DiffusionGemma image encoder row shape mismatch");
    image_inputs.push_back(
        {{image.span.offset, image.span.rows, image.prefix_identity},
         encoded.back()->data()});
  }

  std::vector<std::int32_t> generated;
  std::string visible;
  // The decoded reply and where an unterminated tool call starts in it.
  std::string decoded;
  std::size_t held = std::string::npos;
  std::size_t emitted = 0;
  bool stopped = false, consumer_cancelled = false;
  const auto emit = [&](std::size_t end) {
    if (end <= emitted)
      return true;
    const std::string piece = visible.substr(emitted, end - emitted);
    emitted = end;
    if (on_token && !on_token(piece)) {
      consumer_cancelled = true;
      return false;
    }
    return true;
  };

  const auto stats = engine_->Generate(
      prompt.tokens, image_inputs, options,
      [&](std::span<const std::int32_t> block) {
        if (is_cancelled && is_cancelled()) {
          consumer_cancelled = true;
          return false;
        }
        generated.insert(generated.end(), block.begin(), block.end());
        // Specials stay in the text: native tool calls use them as framing.
        decoded =
            thinking == Thinking::kTagged
                ? DecodeTaggedThinking(tokenizer, generated, open, close)
                : tokenizer.Decode(
                      dg::StripThinkingTokens(generated, open, close), false);
        visible = dg::NativeToolCallsToXml(decoded, &held);
        std::size_t cut = std::string::npos;
        for (const auto& stop : stop_sequences) {
          const std::size_t from =
              emitted > stop.size() ? emitted - stop.size() : 0;
          const auto found = visible.find(stop, from);
          if (found != std::string::npos && found < cut) {
            cut = found;
            result.stop_sequence = stop;
          }
        }
        if (cut != std::string::npos) {
          visible.resize(cut);
          stopped = true;
          (void)emit(visible.size());
          return false;
        }
        // A call still being generated is withheld until it closes.
        std::size_t end = visible.size() > hold ? visible.size() - hold : 0;
        end -= IncompleteUtf8Tail(std::string_view(visible).substr(0, end));
        return emit(end);
      });
  // A matched stop sequence ends generation through the block callback,
  // which the engine reports as cancellation.
  consumer_cancelled = consumer_cancelled || (stats.cancelled && !stopped);
  if (!stopped && !consumer_cancelled) {
    // A call left unterminated at the end of the reply stays visible as text
    // rather than vanishing.
    if (held != std::string::npos)
      visible += decoded.substr(held);
    (void)emit(visible.size());
  }

  {
    char line[320];
    std::snprintf(line, sizeof(line),
                  "event=generation prompt=%d reused=%d generated=%d blocks=%d "
                  "steps=%d prefill_ms=%.1f denoise_ms=%.1f commit_ms=%.1f "
                  "ms_per_step=%.1f",
                  stats.prompt_tokens, stats.reused_tokens,
                  stats.generated_tokens, stats.blocks, stats.denoising_steps,
                  stats.prefill_ms, stats.denoise_ms, stats.commit_ms,
                  stats.denoise_ms / std::max(1, stats.denoising_steps));
    Logger::Info("diffusion", line);
  }
  result.text = visible;
  result.tokens.assign(generated.begin(), generated.end());
  result.completion_tokens = generated.size();
  result.prefill_ms = stats.prefill_ms;
  result.cached_prompt_tokens = static_cast<std::size_t>(stats.reused_tokens);
  // Only the part after the reused prefix is encoded.
  result.prefill_tokens = prompt.tokens.size() - result.cached_prompt_tokens;
  result.decode_ms = stats.denoise_ms + stats.commit_ms;
  result.ttft_ms = stats.first_block_ms;
  if (consumer_cancelled) {
    result.cancelled = true;
    result.finish_reason = FinishReason::kCancelled;
  } else if (stopped) {
    result.finish_reason = FinishReason::kStopSequence;
  } else if (stats.stopped_by_eos) {
    result.finish_reason = FinishReason::kStop;
  } else {
    result.finish_reason = FinishReason::kLength;
  }
  return result;
}

}  // namespace gufo::server
