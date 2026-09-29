// gufo serve's text runner for Qwen3.6-35B-A3B (see text_runner.hpp).
//
// Speculative steps keep the output exact the way Flash-Next's MTP does for
// deterministic drafts: every verified row is sampled with the request's own
// sampler, a draft is kept when the sample equals it, and on a mismatch the
// RNG is rewound so the next step draws that token again from the same row.
// The text is therefore the plain decode's, draw for draw, whatever drafted.
#include "src/models/qwen36_a3b/text_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen36_a3b/prompt_lookup.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

using server::TextRunnerToken;

constexpr double kMtpSurvivalFloor = 0.6;
constexpr double kDFlashSurvivalFloor = 0.2;
constexpr std::size_t kLookupMinMatch = 12;
/// Prompt tokens per Prefill call (a cancellation point).
constexpr std::size_t kPrefillStep = 16384;
constexpr std::uint32_t kDefaultContext = 131072;
constexpr std::string_view kStateAbi = "qwen36-a3b-gfx1151-state-v1";

/// Flash-Next's DraftVocabulary::kLatinText rule (engine.cpp there):
/// special tokens, tokens of printable ASCII (or tab / newline / carriage
/// return) and tokens that are complete UTF-8 of ASCII plus common
/// typographic marks.
std::vector<std::int32_t> LatinTextVocabulary(
    const tokenization::QwenTokenizer& tokenizer) {
  constexpr char32_t kTypographic[] = {
      0x2014, 0x2013, 0x201C, 0x201D, 0x2018, 0x2019, 0x2026, 0x2022, 0x2192,
      0x2190, 0x2248, 0x00D7, 0x00F7, 0x00B1, 0x00B0, 0x00B7, 0x00E9};
  const auto ascii = [](unsigned char b) {
    return (b >= 0x20 && b < 0x7F) || b == '\t' || b == '\n' || b == '\r';
  };
  std::vector<std::int32_t> ids;
  for (std::size_t id = 0; id < tokenizer.GetVocabSize(); ++id) {
    const auto token = static_cast<tokenization::TokenId>(id);
    if (tokenizer.IsSpecialToken(token)) {
      ids.push_back(static_cast<std::int32_t>(id));
      continue;
    }
    const std::string text = tokenizer.DecodeTokenCopy(token);
    bool keep = !text.empty();
    for (std::size_t i = 0; keep && i < text.size();) {
      const auto b = static_cast<unsigned char>(text[i]);
      if (b < 0x80) {
        keep = ascii(b);
        ++i;
        continue;
      }
      const std::size_t length = (b & 0xE0) == 0xC0   ? 2
                                 : (b & 0xF0) == 0xE0 ? 3
                                                      : 0;
      if (length == 0 || i + length > text.size()) {
        keep = false;
        break;
      }
      char32_t cp = length == 2 ? (b & 0x1F) : (b & 0x0F);
      for (std::size_t j = 1; j < length; ++j) {
        const auto c = static_cast<unsigned char>(text[i + j]);
        if ((c & 0xC0) != 0x80) {
          keep = false;
          break;
        }
        cp = (cp << 6) | (c & 0x3F);
      }
      keep = keep && std::find(std::begin(kTypographic), std::end(kTypographic),
                               cp) != std::end(kTypographic);
      i += length;
    }
    if (keep)
      ids.push_back(static_cast<std::int32_t>(id));
  }
  return ids;
}

struct Model {
  std::unique_ptr<Engine> engine;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  std::optional<tokenization::TokenId> im_end;
  ServeConfig config;
  std::uint32_t limit{0};  // usable context (verify headroom kept free)
  std::string model_id;
};

class State final : public server::TextRunnerState {
public:
  explicit State(std::shared_ptr<Model> model) : model(std::move(model)) {}

  void Invalidate() noexcept override {
    model->engine->Reset();
    kv.clear();
    has_frontier = false;
    lookup.Clear();
  }

  Engine& engine() const { return *model->engine; }
  std::uint32_t position() const { return model->engine->position(); }
  std::span<const std::int32_t> history() const {
    return std::span<const std::int32_t>(kv).first(position());
  }
  /// Replaces the tokens behind KV rows [pos, ...) with `tokens`.
  void Wrote(std::uint32_t pos, std::span<const std::int32_t> tokens) {
    kv.resize(pos);
    kv.insert(kv.end(), tokens.begin(), tokens.end());
  }

  std::shared_ptr<Model> model;
  /// The tokens whose KV rows the caches hold, in order; may run past
  /// position() after a restore (rows that are still valid).
  std::vector<std::int32_t> kv;
  Candidates frontier{};  // candidates of the next token
  std::uint32_t frontier_row{0};
  bool has_frontier{false};
  PromptLookup lookup{kLookupMinMatch};
};

class StateSnapshot final : public server::TextRunnerSnapshot {
public:
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return state.size() + tokens.size() * sizeof(std::int32_t) +
           logits.size() * sizeof(float);
  }
  const Model* owner{nullptr};
  std::vector<std::uint8_t> state;
  std::vector<std::int32_t> tokens;
  Candidates frontier{};
  std::vector<float> logits;  // the frontier row
};

State& Require(server::TextRunnerState& state) {
  auto* s = dynamic_cast<State*>(&state);
  if (s == nullptr)
    throw std::logic_error("text runner state is not A3B");
  return *s;
}
const State& Require(const server::TextRunnerState& state) {
  const auto* s = dynamic_cast<const State*>(&state);
  if (s == nullptr)
    throw std::logic_error("text runner state is not A3B");
  return *s;
}

void Check(bool ok, const char* what, const std::string& error) {
  if (!ok)
    throw std::runtime_error(std::string("A3B ") + what + ": " + error);
}

class Runner final : public server::TextModelRunner {
public:
  explicit Runner(std::shared_ptr<Model> model) : m_(std::move(model)) {}

  [[nodiscard]] server::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = m_->model_id,
        .state_abi = std::string(kStateAbi),
        .max_context = m_->limit,
        .capabilities =
            server::TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = true,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] server::TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = std::nullopt,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = server::HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<server::TextExecutionPlan> SupportedPlans()
      const override {
    return {
        {.kind = server::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return m_->tokenizer->Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const server::ChatRequest& request) const override {
    return Prepare(request).tokens;
  }

  [[nodiscard]] std::optional<server::TextPreparedPrompt> PreparePrompt(
      const server::ChatRequest& request) const override {
    for (const auto& message : request.messages)
      if (!message.images.empty())
        throw std::invalid_argument("Qwen3.6-35B-A3B has no image input");
    auto prompt = Prepare(request);
    return server::TextPreparedPrompt{
        std::move(prompt.tokens), {}, prompt.stable_prefix_tokens};
  }

  [[nodiscard]] server::TextGenerationBackend::InitialOutputState
  InitialOutputState(const server::ChatRequest& request) const override {
    return ChatOptions(request).enable_thinking
               ? server::TextGenerationBackend::InitialOutputState::kReasoning
               : server::TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return m_->tokenizer->Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<server::TextRunnerState> CreateState()
      const override {
    if (states_.fetch_add(1) != 0)
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B serves one session (--sessions 1)");
    return std::make_unique<State>(m_);
  }

  void PreparePrefixReuse(
      server::TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    if (Require(state).position() != prefix.size())
      throw std::logic_error("A3B reused prefix does not match its state");
  }

  [[nodiscard]] server::TextPrefillStep Prefill(
      server::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& s = Require(state);
    if (offset != s.position())
      throw std::logic_error("A3B prefill offset does not match its state");
    if (offset >= prompt.size())
      throw std::logic_error("A3B prefill has no remaining input");
    if (prompt.size() > m_->limit)
      throw std::invalid_argument("prompt exceeds the context");
    const std::size_t n = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, kPrefillStep});
    std::vector<std::int32_t> chunk(prompt.begin() + offset,
                                    prompt.begin() + offset + n);
    std::string error;
    const auto pos = s.position();
    if (!s.engine().SpecPrefill(chunk, &s.frontier, &error)) {
      s.Invalidate();
      Check(false, "prefill", error);
    }
    s.Wrote(pos, chunk);
    s.frontier_row = 0;
    s.has_frontier = true;
    return {.consumed_tokens = n, .decode_ready = offset + n == prompt.size()};
  }

  [[nodiscard]] server::TextDecodeSelection SelectNext(
      server::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& s = Require(state);
    if (s.position() >= m_->limit)
      return {.stop = true, .piece = {}};
    const auto token = SampleRow(s, s.frontier, s.frontier_row, sampler);
    if (IsStop(token))
      return {.stop = true, .token = 0, .piece = {}};
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = m_->tokenizer->DecodeTokenCopy(
                static_cast<tokenization::TokenId>(token))};
  }

  [[nodiscard]] std::optional<server::TextDecodeSelection> PreviewFirstToken(
      server::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  void Advance(server::TextRunnerState& state,
               TextRunnerToken token) const override {
    auto& s = Require(state);
    const auto x = static_cast<std::int32_t>(token);
    std::vector<Candidates> rows;
    std::string error;
    const auto pos = s.position();
    const bool ok = s.engine().Verify(x, {}, &rows, &error, MtpCatchUp()) &&
                    s.engine().SpecCommit(1, &error);
    if (!ok) {
      s.Invalidate();
      Check(false, "decode", error);
    }
    s.Wrote(pos, std::span<const std::int32_t>(&x, 1));
    s.frontier = rows[0];
    s.frontier_row = 0;
  }

  [[nodiscard]] server::TextDecodeStep DecodeStep(
      server::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override;

  [[nodiscard]] std::size_t CheckpointPosition(
      const server::TextRunnerState& state) const override {
    return Require(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const server::TextRunnerState& state) const override {
    const auto& s = Require(state);
    return s.engine().StateBytes() +
           std::size_t{s.position()} * sizeof(std::int32_t) +
           std::size_t{s.engine().config().vocab} * sizeof(float);
  }

  [[nodiscard]] std::unique_ptr<server::TextRunnerSnapshot> Snapshot(
      const server::TextRunnerState& state) const override {
    const auto& s = Require(state);
    if (!s.has_frontier)
      throw std::logic_error("A3B snapshot before the prompt");
    auto snapshot = std::make_unique<StateSnapshot>();
    snapshot->owner = m_.get();
    snapshot->state.resize(s.engine().StateBytes());
    const auto history = s.history();
    snapshot->tokens.assign(history.begin(), history.end());
    snapshot->frontier = s.frontier;
    snapshot->logits.resize(s.engine().config().vocab);
    std::string error;
    Check(s.engine().SaveState(snapshot->state, &error) &&
              s.engine().RowLogits(s.frontier_row, snapshot->logits.data(),
                                   &error),
          "snapshot", error);
    return snapshot;
  }

  void RestoreOrFork(
      server::TextRunnerState& state,
      const server::TextRunnerSnapshot& snapshot) const override {
    const auto* snap = dynamic_cast<const StateSnapshot*>(&snapshot);
    if (snap == nullptr || snap->owner != m_.get() || snap->tokens.empty())
      throw std::invalid_argument("snapshot does not belong to this model");
    auto& s = Require(state);
    std::string error;
    s.lookup.Clear();
    // The caches hold KV rows for s.kv; the snapshot's recurrent state fits
    // them only over the same tokens. Otherwise rebuild it by prefill.
    const bool rows_valid =
        s.kv.size() >= snap->tokens.size() &&
        std::equal(snap->tokens.begin(), snap->tokens.end(), s.kv.begin());
    if (rows_valid) {
      if (!s.engine().LoadState(snap->state, &error) ||
          !s.engine().SetRowLogits(snap->logits.data(), &error)) {
        s.Invalidate();
        Check(false, "snapshot restore", error);
      }
      s.frontier = snap->frontier;
    } else {
      s.engine().Reset();
      s.kv.clear();
      if (!s.engine().SpecPrefill(snap->tokens, &s.frontier, &error)) {
        s.Invalidate();
        Check(false, "snapshot rebuild", error);
      }
      s.kv = snap->tokens;
    }
    s.frontier_row = 0;
    s.has_frontier = true;
  }

private:
  tokenization::ChatTemplateOptions ChatOptions(
      const server::ChatRequest& request) const {
    auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                        request.add_vision_id);
    // Qwen3.6's template: no reasoning-effort line, and earlier turns'
    // reasoning is dropped unless the client asks to keep it.
    options.reasoning_effort = tokenization::QwenReasoningEffort::kMedium;
    options.preserve_thinking =
        request.reasoning.preserve_thinking.value_or(false);
    options.require_tool_call =
        request.tool_choice == server::ChatRequest::ToolChoice::kRequired;
    options.max_output_bytes =
        tokenization::RenderedPromptBoundBytes(m_->limit);
    return options;
  }

  qwen::vision::Prompt Prepare(const server::ChatRequest& request) const {
    return qwen::vision::Prepare(
        *m_->tokenizer, request.messages,
        request.tool_choice == server::ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        ChatOptions(request), {}, m_->limit);
  }

  /// Steps without MTP drafts still feed the MTP block when it drafts.
  bool MtpCatchUp() const {
    return m_->config.drafting == ServeConfig::Drafting::kMtp &&
           m_->engine->HasMtp();
  }

  bool IsStop(std::int32_t token) const {
    const auto t = static_cast<tokenization::TokenId>(token);
    return m_->tokenizer->IsStopToken(t) || (m_->im_end && t == *m_->im_end);
  }

  /// Samples row `row` of the last pass: from its top-64 candidates when
  /// that is provably exact, otherwise from the full logits row.
  std::int32_t SampleRow(State& s, const Candidates& c, std::uint32_t row,
                         sampling::SamplerState& sampler) const {
    std::array<sampling::TokenId, Candidates::kCount> ids{};
    for (std::size_t i = 0; i < ids.size(); ++i)
      ids[i] = static_cast<sampling::TokenId>(c.ids[i]);
    const auto vocab = s.engine().config().vocab;
    if (const auto t = sampler.SampleFromTop(c.logits, ids, vocab))
      return static_cast<std::int32_t>(*t);
    std::vector<float> logits(vocab);
    std::string error;
    Check(s.engine().RowLogits(row, logits.data(), &error), "logits", error);
    return static_cast<std::int32_t>(sampler.Sample(logits));
  }

  /// The probability `sampler` gives the top candidate of `c`.
  static double TopProbability(const Candidates& c,
                               const sampling::SamplerState& sampler) {
    std::array<sampling::TokenId, Candidates::kCount> ids{};
    for (std::size_t i = 0; i < ids.size(); ++i)
      ids[i] = static_cast<sampling::TokenId>(c.ids[i]);
    return sampler.Distribution(c.logits, ids).probability(0);
  }

  std::shared_ptr<Model> m_;
  mutable std::atomic<int> states_{0};
};

server::TextDecodeStep Runner::DecodeStep(
    server::TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  if (max_tokens == 0)
    throw std::invalid_argument("A3B decode budget must be at least one");
  auto& s = Require(state);
  auto& engine = s.engine();
  const auto& config = m_->config;
  server::TextDecodeStep step;
  const std::uint32_t pos = s.position();
  if (pos >= m_->limit) {
    step.stop = true;
    return step;
  }
  sampling::SamplerState working = sampler;
  const std::int32_t x = SampleRow(s, s.frontier, s.frontier_row, working);
  if (IsStop(x)) {
    sampler.CopyDrawStateFrom(working);
    step.stop = true;
    return step;
  }
  working.Accept(static_cast<sampling::TokenId>(x));

  // Drafts: at most the budget and the context allow, and 7 (verify rows).
  const std::size_t room =
      std::min<std::size_t>({max_tokens, m_->limit - pos, Engine::kMaxRows});
  const std::size_t cap =
      std::min<std::size_t>(room - 1, config.max_draft_tokens);
  std::vector<std::int32_t> drafts;
  std::size_t copied = std::numeric_limits<std::size_t>::max();
  std::vector<std::int32_t> history;
  if (config.prompt_lookup && cap > 0) {
    const auto h = s.history();
    history.assign(h.begin(), h.end());
    history.push_back(x);
    s.lookup.Extend(history);
  }
  // Prompt lookup: copies after a long enough match of the context plus the
  // drafts so far fill the step (Flash-Next's order: after an MTP draft).
  const auto fill = [&] {
    if (!config.prompt_lookup || drafts.size() >= cap)
      return false;
    const auto match = s.lookup.Find(history, drafts);
    if (match.length == 0)
      return false;
    const std::size_t count =
        std::min(cap - drafts.size(), history.size() - match.start);
    copied = std::min(copied, drafts.size());
    for (std::size_t i = 0; i < count; ++i)
      drafts.push_back(history[match.start + i]);
    return count != 0;
  };

  std::string error;
  std::vector<Candidates> rows;
  bool ok = true;
  bool catch_up = MtpCatchUp();
  if (cap > 0 && config.drafting == ServeConfig::Drafting::kMtp &&
      engine.HasMtp()) {
    Candidates q{};
    ok = engine.DraftFirst(x, &q, &error);
    catch_up = false;
    double alive = 1.0;
    while (ok) {
      const std::int32_t d = q.ids[0];
      if (config.survival) {
        alive *= TopProbability(q, working);
        if (!drafts.empty() && alive < kMtpSurvivalFloor)
          break;
      }
      drafts.push_back(d);
      if (fill() || drafts.size() >= cap)
        break;
      ok = engine.DraftNext(d, &q, &error);
    }
  } else if (cap > 0 && config.drafting == ServeConfig::Drafting::kDFlash) {
    std::vector<float> probs;
    ok = engine.DFlashDraft(x,
                            static_cast<std::uint32_t>(std::min<std::size_t>(
                                cap, engine.DFlashMaxDrafts())),
                            &drafts, &probs, &error);
    if (ok && config.survival) {
      double alive = 1.0;
      for (std::size_t j = 0; j < drafts.size(); ++j) {
        alive *= probs[j];
        if (j > 0 && alive < kDFlashSurvivalFloor) {
          drafts.resize(j);
          break;
        }
      }
    }
    if (ok)
      (void)fill();
  } else if (cap > 0) {
    (void)fill();
  }
  ok = ok && engine.Verify(x, drafts, &rows, &error, catch_up);
  if (!ok) {
    s.Invalidate();
    Check(false, "decode", error);
  }

  step.selections.push_back({.stop = false,
                             .token = static_cast<TextRunnerToken>(x),
                             .piece = m_->tokenizer->DecodeTokenCopy(
                                 static_cast<tokenization::TokenId>(x))});
  std::vector<std::int32_t> kept{x};
  std::uint32_t keep = static_cast<std::uint32_t>(drafts.size()) + 1;
  for (std::uint32_t j = 0; j < drafts.size(); ++j) {
    const auto rng = working.rng_state();
    const std::int32_t t = SampleRow(s, rows[j], j, working);
    if (IsStop(t)) {
      keep = j + 1;
      step.stop = true;
      break;
    }
    if (t != drafts[j]) {
      // The next step samples this row again with the same draw.
      working.SetRngState(rng);
      keep = j + 1;
      break;
    }
    working.Accept(static_cast<sampling::TokenId>(t));
    kept.push_back(t);
    ++step.draft_accepted_tokens;
    if (j >= copied)
      ++step.lookup_accepted_tokens;
    step.selections.push_back({.stop = false,
                               .token = static_cast<TextRunnerToken>(t),
                               .piece = m_->tokenizer->DecodeTokenCopy(
                                   static_cast<tokenization::TokenId>(t))});
  }
  step.draft_tokens = drafts.size();
  if (copied < drafts.size())
    step.lookup_tokens = drafts.size() - copied;
  if (!engine.SpecCommit(keep, &error)) {
    s.Invalidate();
    Check(false, "commit", error);
  }
  s.Wrote(pos, kept);
  s.frontier = rows[keep - 1];
  s.frontier_row = keep - 1;
  sampler.CopyDrawStateFrom(working);
  return step;
}

}  // namespace

std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    const std::string& model_path, const ServeConfig& config,
    std::string* error) {
  auto model = std::make_shared<Model>();
  model->config = config;
  if (model->config.max_draft_tokens + 1 > Engine::kMaxRows)
    model->config.max_draft_tokens = Engine::kMaxRows - 1;
  const std::uint32_t context =
      config.max_context != 0 ? config.max_context : kDefaultContext;
  Options options;
  // Room for a full verify pass past the last usable position.
  options.max_context = context + Engine::kMaxRows;
  if (config.latin_draft_vocab) {
    // The vocabulary comes from the tokenizer, which needs the GGUF: read
    // it first, then load the engine with the list.
    auto reader = core::GgufReader::OpenFile(model_path, error);
    if (reader == nullptr)
      return nullptr;
    auto tokenizer =
        tokenization::QwenTokenizer::CreateFromGguf(*reader, error);
    if (tokenizer == nullptr)
      return nullptr;
    options.draft_vocab_ids = LatinTextVocabulary(*tokenizer);
  }
  model->engine = Engine::Load(model_path, options, error);
  if (model->engine == nullptr)
    return nullptr;
  if (config.drafting == ServeConfig::Drafting::kMtp &&
      !model->engine->HasMtp()) {
    if (error != nullptr)
      *error = "--speculative mtp: this GGUF has no MTP block";
    return nullptr;
  }
  if (config.drafting == ServeConfig::Drafting::kDFlash &&
      !model->engine->LoadDFlash(config.dflash_model_path, error))
    return nullptr;
  model->tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(model->engine->gguf(), error);
  if (model->tokenizer == nullptr)
    return nullptr;
  model->im_end = model->tokenizer->FindSpecialToken("<|im_end|>");
  model->limit = context;
  model->model_id = model->engine->gguf()
                        .GetMetadataString("general.name")
                        .value_or("qwen3.6-35b-a3b");
  return std::make_shared<Runner>(std::move(model));
}

}  // namespace gufo::models::qwen36_a3b
