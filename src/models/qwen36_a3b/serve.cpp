// gufo-a3b --serve (see serve.hpp).
#include "src/models/qwen36_a3b/serve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen36_a3b/prompt_lookup.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

using server::ChatRequest;
using server::TextGenerationBackend;
using Clock = std::chrono::steady_clock;
using Dist = std::vector<std::pair<std::int32_t, double>>;

double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/// Bytes at the end of `s` that start an incomplete UTF-8 sequence (held
/// back from streaming until the next token completes them).
std::size_t IncompleteUtf8Tail(std::string_view s) {
  const std::size_t limit = std::min<std::size_t>(3, s.size());
  for (std::size_t i = 1; i <= limit; ++i) {
    const auto c = static_cast<unsigned char>(s[s.size() - i]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte
    const std::size_t need = (c & 0x80) == 0      ? 1
                             : (c & 0xE0) == 0xC0 ? 2
                             : (c & 0xF0) == 0xE0 ? 3
                             : (c & 0xF8) == 0xF0 ? 4
                                                  : 1;
    return need > i ? i : 0;
  }
  return 0;
}

class A3bBackend final : public TextGenerationBackend {
 public:
  A3bBackend(Engine& engine, const tokenization::QwenTokenizer& tok,
             ServeOptions options)
      : engine_(engine), tok_(tok), o_(std::move(options)) {
    im_end_ = tok_.FindSpecialToken("<|im_end|>");
    eos_ = tok_.GetEosTokenId();
    if (o_.mtp > 0 && !engine_.HasMtp()) o_.mtp = 0;
    if (!engine_.HasDFlash()) o_.dflash_n = 0;
  }

  std::string model_id() const override { return o_.model_id; }
  bool ready() const override { return true; }
  SamplingDefaults sampling_defaults() const override {
    return {o_.max_tokens, o_.sampling};
  }
  ReasoningOptions reasoning_defaults() const override {
    ReasoningOptions r;
    r.enabled = o_.think;
    r.preserve_thinking = o_.preserve_thinking;
    return r;
  }
  InitialOutputState initial_output_state(
      const ChatRequest& request) const override {
    return Options(request).enable_thinking ? InitialOutputState::kReasoning
                                            : InitialOutputState::kContent;
  }

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled,
                  const TokenCallback& on_token,
                  std::string_view client_id) override {
    Result r = Generate(Encode(prompt), 0, max_tokens, sampling, is_cancelled,
                        on_token);
    r.client_id = std::string(client_id);
    return r;
  }

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled,
              const TokenCallback& on_token) override {
    const auto options = Options(request);
    std::string error;
    const auto text = tokenization::QwenChatTemplate::Render(
        request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        options, &error);
    if (!text) throw std::invalid_argument(error);
    const auto ids = Encode(*text);
    // Checkpoint before the generation suffix: clients re-send the previous
    // assistant turn in many forms (without its reasoning, reformatted), and
    // each differs right where the suffix was.
    const auto suffix =
        Encode(tokenization::GenerationPrompt(options.enable_thinking));
    std::size_t prefix = 0;
    if (ids.size() > suffix.size() &&
        std::equal(suffix.begin(), suffix.end(),
                   ids.end() - static_cast<std::ptrdiff_t>(suffix.size())))
      prefix = ids.size() - suffix.size();
    Result r = Generate(ids, request.cache_prompt ? prefix : 0, max_tokens,
                        sampling, is_cancelled, on_token);
    r.client_id = request.client_id;
    return r;
  }

  std::size_t count_tokens(std::string_view text) const override {
    return Encode(text).size();
  }

 private:
  tokenization::ChatTemplateOptions Options(const ChatRequest& request) const {
    auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                        request.add_vision_id);
    // Qwen3.6's template: no reasoning-effort instruction, and earlier turns'
    // reasoning is dropped unless the client asks to preserve it.
    options.reasoning_effort = tokenization::QwenReasoningEffort::kMedium;
    options.preserve_thinking = request.reasoning.preserve_thinking.value_or(
        o_.preserve_thinking.value_or(false));
    options.require_tool_call =
        request.tool_choice == ChatRequest::ToolChoice::kRequired;
    return options;
  }

  std::vector<std::int32_t> Encode(std::string_view text) const {
    const auto e = tok_.Encode(
        text, {.add_bos = false, .add_eos = false, .parse_special_tokens = true});
    return {e.begin(), e.end()};
  }

  bool IsStop(std::int32_t t) const {
    return t == static_cast<std::int32_t>(eos_) ||
           (im_end_ && t == static_cast<std::int32_t>(*im_end_));
  }

  Result Generate(const std::vector<std::int32_t>& ids, std::size_t prefix,
                  std::size_t max_tokens, const sampling::SamplingConfig& sc,
                  const CancellationCheck& is_cancelled,
                  const TokenCallback& on_token);

  Engine& engine_;
  const tokenization::QwenTokenizer& tok_;
  ServeOptions o_;
  std::optional<tokenization::TokenId> im_end_;
  tokenization::TokenId eos_{0};
  std::mutex mutex_;  // one engine slot: requests run one at a time
  std::vector<std::int32_t> ckpt_;  // tokens of the engine's checkpoint
};

TextGenerationBackend::Result A3bBackend::Generate(
    const std::vector<std::int32_t>& ids, std::size_t prefix,
    std::size_t max_tokens, const sampling::SamplingConfig& sc,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
  const std::lock_guard lock(mutex_);
  Result r;
  r.execution_plan = "a3b-serial";
  r.prompt_tokens = ids.size();
  const std::size_t ctx = engine_.max_context();
  if (ids.empty() || ids.size() + 16 > ctx)
    throw std::invalid_argument("prompt of " + std::to_string(ids.size()) +
                                " tokens does not fit the " +
                                std::to_string(ctx) + "-token context");
  max_tokens = std::min(max_tokens == 0 ? o_.max_tokens : max_tokens,
                        ctx - ids.size() - 16);
  std::string error;
  const auto fail = [&](const char* what) {
    ckpt_.clear();
    engine_.Reset();
    throw std::runtime_error(std::string(what) + ": " + error);
  };

  // Prompt cache: resume from the checkpoint when this prompt extends it.
  const auto p0 = Clock::now();
  std::size_t start = 0;
  if (!ckpt_.empty() && prefix >= ckpt_.size() &&
      std::equal(ckpt_.begin(), ckpt_.end(), ids.begin()) &&
      engine_.RestoreCheckpoint(&error))
    start = ckpt_.size();
  error.clear();
  if (start == 0) {
    engine_.Reset();
    ckpt_.clear();
  }
  Candidates last{};
  if (prefix > start) {
    if (!engine_.SpecPrefill({ids.data() + start, prefix - start}, &last,
                             &error) ||
        !engine_.SaveCheckpoint(&error))
      fail("prefill");
    ckpt_.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(prefix));
  }
  const std::size_t rest = std::max(start, prefix);
  if (!engine_.SpecPrefill({ids.data() + rest, ids.size() - rest}, &last,
                           &error))
    fail("prefill");
  r.cached_prompt_tokens = start;
  r.cache_hit = start > 0;
  r.prefill_tokens = ids.size() - start;
  r.prefill_chunks = 1;
  r.prefill_ms = Ms(p0, Clock::now());

  // Sampling over each row's top-64 candidates (exact for top-k <= 64);
  // top-1 drafts are accepted with probability p(draft), else resampled
  // from p without the draft.
  std::mt19937_64 rng(sc.seed >= 0 ? static_cast<std::uint64_t>(sc.seed)
                                   : std::random_device{}());
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const auto dist = [&](const Candidates& c) {
    Dist d;
    if (sc.temperature <= 0.0F) {
      d.push_back({c.ids[0], 1.0});
      return d;
    }
    const int k = sc.top_k > 0 ? std::min<int>(sc.top_k, Candidates::kCount)
                               : static_cast<int>(Candidates::kCount);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      const double w = std::exp((c.logits[i] - c.logits[0]) / sc.temperature);
      d.push_back({c.ids[i], w});
      sum += w;
    }
    double cum = 0.0;
    std::size_t keep = d.size();
    for (std::size_t i = 0; i < d.size(); ++i) {
      if (sc.min_p > 0.0F && d[i].second < sc.min_p * d[0].second) {
        keep = std::max<std::size_t>(i, 1);
        break;
      }
      cum += d[i].second / sum;
      if (cum >= sc.top_p) {
        keep = i + 1;
        break;
      }
    }
    d.resize(keep);
    double total = 0.0;
    for (const auto& e : d) total += e.second;
    for (auto& e : d) e.second /= total;
    return d;
  };
  const auto sample = [&](const Dist& d, std::int32_t exclude) {
    double total = 0.0;
    for (const auto& e : d)
      if (e.first != exclude) total += e.second;
    double u = uniform(rng) * total;
    for (const auto& e : d) {
      if (e.first == exclude) continue;
      if ((u -= e.second) <= 0.0) return e.first;
    }
    for (auto it = d.rbegin(); it != d.rend(); ++it)
      if (it->first != exclude) return it->first;
    return d.front().first;
  };
  const auto prob = [](const Dist& d, std::int32_t t) {
    for (const auto& e : d)
      if (e.first == t) return e.second;
    return 0.0;
  };

  std::vector<std::int32_t> hist(ids.begin(), ids.end());
  PromptLookup index(static_cast<std::size_t>(std::max(o_.lookup, 1)));
  std::string pending;  // streamed text held back mid UTF-8 sequence
  bool cancelled = false;
  const auto emit = [&](std::int32_t t) {
    hist.push_back(t);
    r.tokens.push_back(static_cast<tokenization::TokenId>(t));
    ++r.completion_tokens;
    const std::string piece = tok_.DecodeTokenCopy(t);
    r.text += piece;
    if (!on_token) return;
    pending += piece;
    const std::size_t tail = IncompleteUtf8Tail(pending);
    if (pending.size() > tail) {
      if (!on_token(std::string_view(pending).substr(0, pending.size() - tail)))
        cancelled = true;
      pending.erase(0, pending.size() - tail);
    }
  };

  const auto d0 = Clock::now();
  std::int32_t x = sample(dist(last), -1);
  bool stop = IsStop(x);
  std::vector<std::int32_t> drafts;
  std::vector<Candidates> rows;
  r.finish_reason = FinishReason::kStop;
  while (!stop) {
    if (r.completion_tokens >= max_tokens) {
      r.finish_reason = FinishReason::kLength;
      break;
    }
    if (cancelled || (is_cancelled && is_cancelled())) {
      r.cancelled = true;
      r.finish_reason = FinishReason::kCancelled;
      break;
    }
    emit(x);
    if (r.completion_tokens == 1) r.ttft_ms = r.prefill_ms + Ms(d0, Clock::now());
    if (r.completion_tokens >= max_tokens) {
      r.finish_reason = FinishReason::kLength;
      break;
    }
    const auto left = static_cast<int>(max_tokens - r.completion_tokens);
    const auto wide = static_cast<std::uint32_t>(
        std::min<int>(Engine::kMaxRows - 1, std::max(0, left)));
    std::size_t copied = SIZE_MAX;
    const auto fill = [&] {
      if (o_.lookup == 0 || drafts.size() >= wide) return false;
      const auto m = index.Find(hist, drafts);
      if (m.length == 0) return false;
      const std::size_t count =
          std::min<std::size_t>(wide - drafts.size(), hist.size() - m.start);
      copied = drafts.size();
      for (std::size_t i = 0; i < count; ++i) drafts.push_back(hist[m.start + i]);
      return count != 0;
    };
    drafts.clear();
    if (o_.lookup > 0) index.Extend(hist);
    bool ok = true;
    if (o_.dflash_n > 0) {
      std::vector<float> dprob;
      ok = engine_.DFlashDraft(
          x, std::min<std::uint32_t>(static_cast<std::uint32_t>(o_.dflash_n), wide),
          &drafts, &dprob, &error);
      if (ok && o_.survival > 0) {
        double alive = 1.0;
        for (std::size_t j = 0; j < drafts.size(); ++j) {
          alive *= dprob[j];
          if (j > 0 && alive < o_.survival) {
            drafts.resize(j);
            break;
          }
        }
      }
      if (ok && o_.lookup > 0) (void)fill();
      ok = ok && engine_.Verify(x, drafts, &rows, &error);
    } else if (o_.mtp == 0 || left <= 0) {
      if (o_.lookup > 0 && fill()) {
        ok = engine_.Verify(x, drafts, &rows, &error);
      } else {
        ok = engine_.SpecStep(x, 0, &drafts, &rows, &error);
      }
    } else {
      // One MTP draft at a time; survival stops the chain once the product
      // of the drafts' top probabilities falls below the floor. Prompt
      // lookup copies follow a kept draft (Flash-Next's order).
      const int cap = std::min(o_.mtp, left);
      Candidates q{};
      ok = engine_.DraftFirst(x, &q, &error);
      double alive = 1.0;
      while (ok) {
        const Dist qd = dist(q);
        const std::int32_t d = q.ids[0];
        alive *= qd.front().second;
        if (!drafts.empty() && alive < o_.survival) break;
        drafts.push_back(d);
        if (fill() || static_cast<int>(drafts.size()) >= cap) break;
        ok = engine_.DraftNext(d, &q, &error);
      }
      ok = ok && engine_.Verify(x, drafts, &rows, &error);
    }
    if (!ok) fail("decode");
    r.draft_tokens += drafts.size();
    if (copied < drafts.size()) r.lookup_tokens += drafts.size() - copied;
    std::uint32_t keep = 0;
    std::int32_t next = -1;
    for (std::uint32_t j = 0; j < drafts.size(); ++j) {
      const Dist d = dist(rows[j]);
      if (uniform(rng) < prob(d, drafts[j])) {
        ++r.draft_accepted_tokens;
        if (j >= copied) ++r.lookup_accepted_tokens;
        if (IsStop(drafts[j])) {
          keep = j + 1;
          stop = true;
          break;
        }
        emit(drafts[j]);
        if (r.completion_tokens >= max_tokens) {
          keep = j + 1;
          stop = true;
          r.finish_reason = FinishReason::kLength;
          break;
        }
        continue;
      }
      keep = j + 1;
      next = sample(d, drafts[j]);
      break;
    }
    if (!stop && next < 0) {
      keep = static_cast<std::uint32_t>(drafts.size()) + 1;
      next = sample(dist(rows[drafts.size()]), -1);
    }
    if (!engine_.SpecCommit(keep, &error)) fail("commit");
    x = next;
    stop = stop || IsStop(x);
  }
  if (on_token && !pending.empty() && !cancelled) (void)on_token(pending);
  r.decode_ms = Ms(d0, Clock::now());
  if (r.completion_tokens > 1)
    r.mean_inter_token_ms = r.decode_ms / static_cast<double>(r.completion_tokens);
  std::fprintf(stderr,
               "a3b: prompt %zu (%zu cached) %.0f t/s | gen %zu %.1f t/s | "
               "accept %zu/%zu | %s\n",
               r.prompt_tokens, r.cached_prompt_tokens,
               r.prefill_ms > 0 ? r.prefill_tokens * 1000.0 / r.prefill_ms : 0.0,
               r.completion_tokens,
               r.decode_ms > 0 ? r.completion_tokens * 1000.0 / r.decode_ms : 0.0,
               r.draft_accepted_tokens, r.draft_tokens,
               r.finish_reason == FinishReason::kStop     ? "stop"
               : r.finish_reason == FinishReason::kLength ? "length"
                                                          : "cancelled");
  server::RecordServerMetrics(r);
  return r;
}

}  // namespace

int RunServe(Engine& engine, const tokenization::QwenTokenizer& tokenizer,
             const ServeOptions& options) {
  auto backend = std::make_shared<A3bBackend>(engine, tokenizer, options);
  server::HttpServer http(options.host, options.port, backend, nullptr,
                          nullptr, nullptr,
                          server::HttpServerOptions{
                              .max_request_body_bytes =
                                  options.max_request_bytes,
                              .max_connections = options.max_connections,
                              .api_key = options.api_key,
                          });
  std::string error;
  if (!http.start(&error)) {
    std::fprintf(stderr, "a3b: cannot start the server: %s\n", error.c_str());
    return 1;
  }
  std::fprintf(stderr,
               "a3b: serving '%s' at http://%s:%d/v1 (OpenAI API; one request "
               "at a time, %u-token context)\n",
               options.model_id.c_str(), options.host.c_str(), http.port(),
               engine.max_context());
  std::fflush(stderr);
  http.run();
  return 0;
}

}  // namespace gufo::models::qwen36_a3b
