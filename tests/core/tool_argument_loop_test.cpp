#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace {
namespace s = gufo::sampling;
namespace q = gufo::models::qwen38_flash_next;
using Breaker = s::LoopBreaker;

#define CHECK(condition)                                             \
  do {                                                               \
    if (!(condition))                                                \
      throw std::runtime_error(std::string(#condition) +             \
                               " line=" + std::to_string(__LINE__)); \
  } while (false)

// Dense synthetic IDs for the real Qwen35 BPE pieces of the measured loop.
// Offline IDs: 510,1628,29,198,248059,198,248058,198,27,1628,28,9951,
// 853,956,29,198,27,15704,28,5454,29,198. No model/external fixture is needed.
const std::vector<s::TokenId> kOpen{1, 3, 328, 11,  329, 330, 331, 332,
                                    9, 3, 328, 333, 329, 334, 9,   3};
const std::vector<s::TokenId> kLoop{10,  11,  9,   3,   2,   3,   1, 3,
                                    328, 11,  329, 330, 331, 332, 9, 3,
                                    328, 333, 329, 334, 9,   3};
const std::vector<s::TokenId> kClose{3, 13, 6, 10, 11, 6, 2};

struct Fixture {
  std::shared_ptr<s::BehaviourConfig> behaviour =
      std::make_shared<s::BehaviourConfig>();
  std::shared_ptr<std::vector<std::string>> pieces =
      std::make_shared<std::vector<std::string>>(
          std::initializer_list<std::string>{"",
                                             "<tool_call>",
                                             "</tool_call>",
                                             "\n",
                                             "<function=",
                                             "execute_bash",
                                             ">\n",
                                             "<parameter",
                                             "=command",
                                             ">",
                                             "</",
                                             "function",
                                             "same line\n",
                                             "</parameter",
                                             "Before.\n",
                                             "<think>",
                                             "</think>",
                                             "[",
                                             "{\"v\":1}",
                                             ",",
                                             "]",
                                             "<functions>",
                                             "<tool_",
                                             "call>"});
  s::SamplingConfig config;

  explicit Fixture(bool array = false, bool required = false) {
    pieces->push_back("cat <<'EOF'\n");  // 24
    pieces->push_back("EOF\n");          // 25
    for (int line = 0; line < 300; ++line)
      pieces->push_back("coherent heredoc line " + std::to_string(line) + "\n");
    pieces->push_back("same,csv,row\n");          // 326
    pieces->push_back("assert(value == 42);\n");  // 327
    for (const auto* piece :
         {"<", "=", "execute", "_b", "ash", "parameter", "command"})
      pieces->push_back(piece);            // 328..334: measured BPE pieces
    pieces->push_back("{\"command\":\"");  // 335: JSON answer, outside any call
    pieces->push_back("\"}");              // 336
    pieces->push_back("cat > out.xml <<'EOF'\n<root>\n");  // 337
    pieces->push_back("</root>\nEOF\n");                   // 338
    pieces->push_back("documentation examples:\n");        // 339
    pieces->push_back("cat > fixture.txt <<'EOF'\n");      // 340
    const auto schema = gufo::json::parse(
        array
            ? R"({"type":"object","properties":{"command":{"type":"array","items":{"type":"object","properties":{"v":{"type":"integer"}},"required":["v"],"additionalProperties":false}}},"required":["command"],"additionalProperties":false})"
            : R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false})");
    auto constraint = std::make_shared<s::TokenConstraint>();
    constraint->grammar = s::JsonConstraint::WithTools(
        nullptr,
        {{"execute_bash",
          s::JsonConstraint::ToolParameters(
              schema, true, s::JsonConstraint::ToolFormat::kQwen)}},
        required, true, s::JsonConstraint::ToolFormat::kQwen);
    constraint->vocabulary = std::make_shared<s::ConstraintVocabulary>(
        static_cast<std::uint32_t>(pieces->size()), [&](std::uint32_t id) {
          return s::ConstraintVocabulary::Piece{pieces->at(id), id == 0};
        });
    behaviour->loop_breaker = true;
    behaviour->loop_exempt = true;
    behaviour->eog_ids = {0};
    behaviour->think_start = {15};
    behaviour->think_end = {16};
    behaviour->token_pieces = pieces;
    behaviour->tool_framing_ids = {1, 2};
    config.behaviour = behaviour;
    config.constraint = constraint;
    config.seed = 42;
  }

  std::vector<float> Row(s::TokenId winner = 12) const {
    std::vector<float> row(pieces->size(), -10);
    row[winner] = 10;
    return row;
  }
};

void ThresholdsAndFraming() {
  for (const int period : {1, 8, 13, 64, 128}) {
    Breaker breaker;
    CHECK(breaker.cfg.tool_repeats == 16);
    CHECK(breaker.cfg.tool_tokens == 128);
    CHECK(breaker.cfg.tool_max_period == 128);
    breaker.accept(999, Breaker::THINK);
    const auto answer_think_tail = breaker.tail;
    const int stop_at = std::max(128, 16 * period);
    for (int i = 0; i < stop_at; ++i) {
      const auto event =
          breaker.accept(1000 + i % period, Breaker::TOOL, i % period == 0);
      CHECK(event.what ==
            (i + 1 == stop_at ? Breaker::STOP_TOOL : Breaker::NONE));
      if (event.what != Breaker::NONE) {
        CHECK(event.period == period);
        CHECK(event.repeats == stop_at / period);
      }
    }
    CHECK(breaker.stopped_tools == 1 && breaker.nudges == 0);
    CHECK(breaker.current == Breaker::THINK && breaker.count == 1);
    CHECK(breaker.tail == answer_think_tail);
    CHECK(breaker.accept(1000, Breaker::TOOL, true).what == Breaker::NONE);
  }
  Breaker no_framing;
  // Framing in the prefix is insufficient: it must be in the repeated period.
  no_framing.accept(7, Breaker::TOOL, true);
  for (int i = 0; i < 4096; ++i)
    CHECK(no_framing.accept(1000 + i % 13, Breaker::TOOL).what ==
          Breaker::NONE);
  CHECK(no_framing.stopped_tools == 0);
  Breaker too_long;
  for (int i = 0; i < 4096; ++i)
    CHECK(too_long.accept(1000 + i % 129, Breaker::TOOL, i % 129 == 0).what ==
          Breaker::NONE);
}

s::SamplerState Stopped(Fixture& fixture) {
  s::SamplerState sampler(fixture.config);
  sampler.Accept(kOpen);
  CHECK(sampler.NeedsConstraintMask());
  CHECK(sampler.SaveDrawState().behaviour->tool_call_open);
  int accepted = 0;
  for (; accepted < 16 * static_cast<int>(kLoop.size()); ++accepted) {
    const auto token = kLoop[accepted % kLoop.size()];
    CHECK(sampler.Sample(fixture.Row(token)) == token);
    sampler.Accept(token);
    if (sampler.SaveDrawState().behaviour->loop.stopped_tools)
      break;
  }
  CHECK(accepted < 16 * static_cast<int>(kLoop.size()));
  const auto state = sampler.SaveDrawState().behaviour;
  CHECK(state->loop.tool.stop.period == static_cast<int>(kLoop.size()));
  CHECK(state->loop.tool.stop.repeats == 16);
  CHECK(state->loop.tool.count == 352);
  CHECK(state->loop.detections == 1 && state->loop.nudges == 0);
  CHECK(state->loop.count == 0);
  return sampler;
}

void SamplingPaths() {
  for (const bool required : {false, true}) {
    Fixture fixture(false, required);
    auto stopped = Stopped(fixture);
    const auto row = fixture.Row();
    CHECK(stopped.WithoutBehaviour().Distribution(row).probability(0) == 0);
    CHECK(stopped.Distribution(row).probability(0) == 1);
    auto ar = stopped;
    CHECK(ar.Sample(row) == 0);
    CHECK(ar.RawLogprobs(row, 0, 1).logprob == 0);
    auto top = stopped;
    const std::vector<s::TokenId> ids{12, 0, 3};
    const std::vector<float> values{10, -10, -10};
    // An active grammar declines the device list, then uses the forced full
    // row.
    CHECK(!top.SampleFromTop(values, ids, row.size()));
    CHECK(top.Sample(row) == 0);
    auto lookup = stopped;
    CHECK(q::VerifyDraft(row, 12, lookup, [](int token) {
            return token == 0;
          }) == q::DraftDecision::kStop);
    CHECK(lookup.SaveDrawState().behaviour->accepted_count ==
          stopped.SaveDrawState().behaviour->accepted_count);
    auto mtp = stopped;
    q::MtpProposal proposal;
    proposal.size = 1;
    proposal.token = proposal.ids[0] = 12;
    proposal.probability = proposal.probabilities[0] = 1;
    const auto verified = q::VerifyMtpProposal(row, proposal, mtp);
    CHECK(!verified.accepted && verified.token == 0);
    auto residual = stopped;
    const std::vector<s::TokenId> proposed{12};
    const std::vector<float> probabilities{1};
    CHECK(residual.SampleResidual(row, proposed, probabilities) == 0);
    for (const auto pending : {0u, 12u}) {
      auto deferred = stopped;
      deferred.DeferSample(pending);
      CHECK(deferred.Sample(row) == 0);
    }
    // Defensive acceptance of terminal EOG leaves the partial grammar intact.
    ar.Accept(0);
    CHECK(ar.NeedsConstraintMask());
    CHECK(ar.Sample(row) == 0);
    auto grammar = ar.WithoutBehaviour();
    CHECK(grammar.Distribution(row).probability(0) == 0);
    grammar.Accept(kClose);
    CHECK(grammar.Sample(fixture.Row(0)) == 0);
    const auto summary = ar.BehaviourSummary();
    CHECK(std::count_if(summary.begin(), summary.end(), [](const auto& line) {
            return line.find(
                       "name=loop_breaker taken=1 fallback=0 "
                       "reason=tool_argument_loop period=22 repeats=16") !=
                   std::string::npos;
          }) == 1);
  }
}

void LegitimateArgumentsAndBoundaries() {
  Fixture fixture;
  s::SamplerState lines(fixture.config);
  lines.Accept(kOpen);
  for (int i = 0; i < 50; ++i)
    lines.Accept(12);
  // Many tokens per identical line: pass both the repeat and minimum-token
  // gates.
  for (int i = 0; i < 1024; ++i)
    lines.Accept(12);
  CHECK(lines.SaveDrawState().behaviour->loop.stopped_tools == 0);
  lines.Accept(kClose);
  CHECK(!lines.SaveDrawState().behaviour->tool_call_open);
  CHECK(lines.SaveDrawState().behaviour->loop.tool.count == 0);
  CHECK(lines.Sample(fixture.Row(0)) == 0);
  // Complete canonical calls cannot accumulate a TOOL history across calls.
  for (int i = 0; i < 50; ++i) {
    lines.Accept(kOpen);
    lines.Accept(12);
    lines.Accept(kClose);
  }
  CHECK(lines.SaveDrawState().behaviour->loop.stopped_tools == 0);
  s::SamplerState heredoc(fixture.config);
  heredoc.Accept(kOpen);
  heredoc.Accept(24);
  for (s::TokenId line = 26; line < 326; ++line)
    heredoc.Accept(line);
  heredoc.Accept(25);
  heredoc.Accept(kClose);
  CHECK(heredoc.SaveDrawState().behaviour->loop.stopped_tools == 0);
  for (const s::TokenId content : {326u, 327u}) {
    s::SamplerState repeated(fixture.config);
    repeated.Accept(kOpen);
    for (int i = 0; i < 1024; ++i)
      repeated.Accept(content);
    CHECK(repeated.SaveDrawState().behaviour->loop.stopped_tools == 0);
  }
  // Common words/punctuation that are individual framing pieces are harmless.
  for (const s::TokenId content : {10u, 11u, 328u, 333u}) {
    s::SamplerState repeated(fixture.config);
    repeated.Accept(kOpen);
    for (int i = 0; i < 1024; ++i)
      repeated.Accept(content);
    CHECK(repeated.SaveDrawState().behaviour->loop.stopped_tools == 0);
  }
  for (const auto& spelling :
       std::vector<std::vector<s::TokenId>>{{10, 11}, {328, 333}}) {
    s::SamplerState framing_only(fixture.config);
    framing_only.Accept(kOpen);
    for (int i = 0; i < 128; ++i)
      framing_only.Accept(spelling[i % spelling.size()]);
    CHECK(framing_only.SaveDrawState().behaviour->loop.stopped_tools == 0);
  }
  Fixture array(true);
  s::SamplerState objects(array.config);
  objects.Accept(kOpen);
  objects.Accept(17);
  for (int i = 0; i < 1024; ++i) {
    if (i)
      objects.Accept(19);
    objects.Accept(18);
  }
  objects.Accept(20);
  objects.Accept(kClose);
  CHECK(objects.SaveDrawState().behaviour->loop.stopped_tools == 0);
  auto plain_config = fixture.config;
  plain_config.constraint.reset();
  s::SamplerState plain(plain_config);
  fixture.behaviour->initial_thinking = true;
  s::SamplerState thinking(fixture.config);
  for (int i = 0; i < 4096; ++i) {
    plain.Accept(kLoop[i % kLoop.size()]);
    thinking.Accept(kLoop[i % kLoop.size()]);
  }
  CHECK(plain.SaveDrawState().behaviour->loop.detections == 0);
  CHECK(thinking.SaveDrawState().behaviour->loop.detections == 0);
  CHECK(!thinking.NeedsConstraintMask());
  fixture.behaviour->initial_thinking = false;
  s::SamplerState split(fixture.config);
  split.Accept(22);
  CHECK(!split.SaveDrawState().behaviour->tool_call_open);
  split.Accept(23);
  CHECK(split.SaveDrawState().behaviour->tool_call_open);
  s::SamplerState pseudo(fixture.config);
  for (int i = 0; i < 4096; ++i) {
    const auto token = kLoop[i % kLoop.size()];
    pseudo.Accept(token == 1 ? 21 : token);
  }
  CHECK(!pseudo.NeedsConstraintMask());
  CHECK(pseudo.SaveDrawState().behaviour->loop.detections == 0);
  // A restrictive JSON answer grammar may contain literal tool framing in a
  // string; those accepted tokens are not on the native call branch.
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false})");
  auto constrained_answer = std::make_shared<s::TokenConstraint>();
  constrained_answer->vocabulary = fixture.config.constraint->vocabulary;
  constrained_answer->grammar = s::JsonConstraint::WithTools(
      s::JsonConstraint::Compile(schema, true),
      {{"execute_bash",
        s::JsonConstraint::ToolParameters(
            schema, true, s::JsonConstraint::ToolFormat::kQwen)}},
      false, true, s::JsonConstraint::ToolFormat::kQwen);
  auto answer_config = fixture.config;
  answer_config.constraint = constrained_answer;
  s::SamplerState quoted(answer_config);
  quoted.Accept(335);
  CHECK(quoted.NeedsConstraintMask());
  for (int i = 0; i < 4096; ++i) {
    const auto token = kLoop[i % kLoop.size()];
    if (token != 3)
      quoted.Accept(token);  // JSON strings contain no raw newline
  }
  quoted.Accept(336);
  CHECK(quoted.SaveDrawState().behaviour->loop.detections == 0);
  CHECK(quoted.SaveDrawState().behaviour->loop.tool.count == 0);
}

void EnvelopeScopeRegressions() {
  Fixture fixture;
  const auto script = [&](std::span<const s::TokenId> block, int repeats,
                          s::TokenId prefix, s::TokenId suffix) {
    s::SamplerState sampler(fixture.config);
    sampler.Accept(kOpen);
    sampler.Accept(prefix);
    for (int i = 0; i < repeats; ++i)
      sampler.Accept(block);
    sampler.Accept(suffix);
    sampler.Accept(kClose);
    CHECK(!sampler.ToolArgumentLoopDetected());
    CHECK(sampler.Sample(fixture.Row(0)) == 0);
  };
  // Valid heredoc XML with 50 empty function elements, and fixture/document
  // examples of parameter openings, contain no envelope special IDs.
  const std::vector<s::TokenId> xml{328, 11, 9, 3, 10, 11, 9, 3};
  const std::vector<s::TokenId> examples{328, 333, 329, 334, 9, 3};
  script(xml, 50, 337, 338);
  script(examples, 50, 339, 3);
  // Documented known limit: an argument literally writing >=16 whole tool-call
  // envelopes still stops. The intended 30-copy fixture must not execute.
  s::SamplerState literal(fixture.config);
  literal.Accept(kOpen);
  literal.Accept(340);
  int accepted = 0;
  while (!literal.ToolArgumentLoopDetected() &&
         accepted < 30 * static_cast<int>(kLoop.size())) {
    literal.Accept(kLoop[accepted++ % kLoop.size()]);
  }
  CHECK(literal.ToolArgumentLoopDetected());
  CHECK(accepted < 30 * static_cast<int>(kLoop.size()));
  CHECK(literal.SaveDrawState().behaviour->loop.tool.stop.period == 22);
  CHECK(literal.Sample(fixture.Row()) == 0);
}

namespace server = gufo::server;
class LoopRunner final : public server::TextModelRunner {
  struct State final : server::TextRunnerState {
    std::size_t prompt = 0, generated = 0;
    void Invalidate() noexcept override { prompt = generated = 0; }
  };
  const Fixture& fixture_;
  int mode_;
  bool final_advance_;

public:
  mutable std::size_t advances = 0;
  LoopRunner(const Fixture& fixture, int mode, bool final_advance)
      : fixture_(fixture), mode_(mode), final_advance_(final_advance) {}
  server::TextRunnerDescriptor Descriptor() const override {
    server::TextRunnerDescriptor descriptor;
    descriptor.model_id = "cpu-tool-loop";
    descriptor.state_abi = "cpu-tool-loop-v1";
    descriptor.max_context = 8192;
    descriptor.capabilities.incremental_prefill = true;
    descriptor.capabilities.incremental_text_is_exact = true;
    descriptor.capabilities.final_token_advance_required = final_advance_;
    descriptor.capabilities.multi_token_decode = mode_ != 0;
    descriptor.capabilities.batched_multi_token_decode = mode_ == 2;
    return descriptor;
  }
  server::TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 4096,
            .per_request_state_bytes = 128,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = 0};
  }
  std::vector<server::TextExecutionPlan> SupportedPlans() const override {
    return {{server::TextExecutionPlanKind::kSerial, 1}};
  }
  std::vector<server::TextRunnerToken> Tokenize(
      std::string_view) const override {
    return {14};
  }
  std::optional<std::vector<server::TextRunnerToken>> RenderAndTokenize(
      const server::ChatRequest&) const override {
    return std::vector<server::TextRunnerToken>{14};
  }
  std::string Decode(
      std::span<const server::TextRunnerToken> tokens) const override {
    std::string text;
    for (const auto token : tokens)
      text += fixture_.pieces->at(token);
    return text;
  }
  std::unique_ptr<server::TextRunnerState> CreateState() const override {
    return std::make_unique<State>();
  }
  server::TextPrefillStep Prefill(
      server::TextRunnerState& state,
      std::span<const server::TextRunnerToken> prompt, std::size_t offset,
      std::size_t budget) const override {
    auto& current = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    current.prompt = offset + count;
    current.generated = 0;
    return {.consumed_tokens = count,
            .decode_ready = current.prompt == prompt.size()};
  }
  s::TokenId Next(const State& state) const {
    return state.generated < kOpen.size()
               ? kOpen[state.generated]
               : kLoop[(state.generated - kOpen.size()) % kLoop.size()];
  }
  server::TextDecodeSelection SelectNext(
      server::TextRunnerState& state, s::SamplerState& sampler) const override {
    const auto token =
        sampler.Sample(fixture_.Row(Next(dynamic_cast<State&>(state))));
    return {.stop = token == 0,
            .token = token,
            .piece = fixture_.pieces->at(token)};
  }
  void Advance(server::TextRunnerState& state,
               server::TextRunnerToken) const override {
    ++dynamic_cast<State&>(state).generated;
    ++advances;
  }
  std::size_t CheckpointPosition(
      const server::TextRunnerState& state) const override {
    const auto& current = dynamic_cast<const State&>(state);
    return current.prompt + current.generated;
  }
  server::TextDecodeStep DecodeStep(server::TextRunnerState& state,
                                    std::size_t max_tokens,
                                    s::SamplerState& sampler) const override {
    if (mode_ == 0)
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    auto working = sampler;
    server::TextDecodeStep step;
    // Seven-row cycles put the exact 353-token boundary on an MTP verification
    // row. As in the engine, acceptance occurs on the working sampler and the
    // caller later replays returned tokens. No final-row stop is fabricated
    // here.
    for (std::size_t i = 0; i < std::min<std::size_t>(7, max_tokens); ++i) {
      const auto proposed = Next(dynamic_cast<State&>(state));
      s::TokenId token;
      if (i == 0)
        token = working.Sample(fixture_.Row(proposed));
      else {
        q::MtpProposal proposal;
        proposal.size = 1;
        proposal.token = proposal.ids[0] = proposed;
        proposal.probability = proposal.probabilities[0] = 1;
        token = q::VerifyMtpProposal(fixture_.Row(proposed), proposal, working)
                    .token;
      }
      if (token == 0) {
        step.stop = true;
        break;
      }
      step.selections.push_back(
          {.token = token, .piece = fixture_.pieces->at(token)});
      Advance(state, token);
      working.Accept(token);
    }
    sampler.CopyDrawStateFrom(working);
    return step;
  }
};

void ExactBoundaryCompletion() {
  Fixture fixture;
  // The canonical opener isn't TOOL history: 353 emitted tokens = 352 TOOL
  // tokens. Detection on that last permitted row outranks ordinary length.
  for (const int mode : {0, 1, 2}) {
    for (const bool final_advance : {false, true}) {
      for (const std::size_t cap : {352u, 353u, 354u}) {
        auto runner =
            std::make_shared<LoopRunner>(fixture, mode, final_advance);
        server::TextRunnerRamCacheOptions cache;
        cache.disabled = true;
        auto pool = std::make_shared<server::TextRunnerPool>(
            runner, 1, std::nullopt, cache);
        server::TextGenerationScheduler scheduler(pool);
        const auto result = scheduler.Submit({14}, cap, fixture.config).Wait();
        const bool detected = cap >= 353;
        CHECK(result.tokens.size() == std::min<std::size_t>(cap, 353));
        CHECK(result.finish_reason ==
              (detected
                   ? server::TextGenerationBackend::FinishReason::kStop
                   : server::TextGenerationBackend::FinishReason::kLength));
        CHECK(result.termination_cause ==
              (detected
                   ? server::TextGenerationBackend::TerminationCause::
                         kToolArgumentLoop
                   : server::TextGenerationBackend::TerminationCause::kNone));
        CHECK(std::ranges::find(result.tokens, 0) == result.tokens.end());
        const auto expected_advances = mode == 0 && !final_advance && cap <= 353
                                           ? result.tokens.size() - 1
                                           : result.tokens.size();
        CHECK(runner->advances == expected_advances);
      }
    }
  }
}

void DisabledIdentityAndRollback() {
  Fixture fixture;
  Fixture array_fixture(true);
  std::vector<std::vector<s::TokenId>> streams(6);
  streams[0] = kOpen;
  for (int i = 0; i < 4096; ++i)
    streams[0].push_back(kLoop[i % kLoop.size()]);
  streams[1] = kOpen;
  streams[1].insert(streams[1].end(), 1024, 12);
  streams[1].insert(streams[1].end(), kClose.begin(), kClose.end());
  streams[2] = kOpen;
  streams[2].push_back(24);
  for (s::TokenId line = 26; line < 326; ++line)
    streams[2].push_back(line);
  streams[2].push_back(25);
  streams[2].insert(streams[2].end(), kClose.begin(), kClose.end());
  streams[3] = kOpen;
  streams[3].push_back(17);
  for (int i = 0; i < 1024; ++i) {
    if (i)
      streams[3].push_back(19);
    streams[3].push_back(18);
  }
  streams[3].push_back(20);
  streams[3].insert(streams[3].end(), kClose.begin(), kClose.end());
  for (int i = 0; i < 4096; ++i)
    streams[4].push_back(kLoop[i % kLoop.size()]);
  streams[5] = streams[4];
  for (const bool master_off : {false, true}) {
    for (std::size_t scenario = 0; scenario < streams.size(); ++scenario) {
      auto baseline_policy =
          std::make_shared<s::BehaviourConfig>(*fixture.behaviour);
      baseline_policy->initial_thinking = scenario == 5;
      // Keep the pre-existing think phase policy active with the master loop
      // switch off, as production does; initial_thinking alone is not a policy.
      baseline_policy->think_eog_guard = scenario == 5;
      baseline_policy->loop_tool_args = false;
      auto disabled_policy =
          std::make_shared<s::BehaviourConfig>(*baseline_policy);
      disabled_policy->loop_tool_args = master_off;
      disabled_policy->loop_breaker = !master_off;
      auto baseline_config =
          scenario == 3 ? array_fixture.config : fixture.config;
      if (scenario == 4)
        baseline_config.constraint.reset();
      auto disabled_config = baseline_config;
      baseline_config.behaviour = baseline_policy;
      disabled_config.behaviour = disabled_policy;
      baseline_config.temperature = disabled_config.temperature = 1;
      s::SamplerState baseline(baseline_config), disabled(disabled_config);
      for (const auto token : streams[scenario]) {
        auto row = fixture.Row(token);
        for (auto& value : row)
          if (value != 10)
            value = -INFINITY;
        const auto before = baseline.Distribution(row),
                   after = disabled.Distribution(row);
        CHECK(before.entries().size() == after.entries().size());
        for (std::size_t i = 0; i < before.entries().size(); ++i) {
          CHECK(before.entries()[i].token == after.entries()[i].token);
          CHECK(before.entries()[i].value == after.entries()[i].value);
        }
        CHECK(baseline.Sample(row) == disabled.Sample(row));
        CHECK(baseline.rng_state() == disabled.rng_state());
        baseline.Accept(token);
        disabled.Accept(token);
        CHECK(std::ranges::equal(baseline.history(), disabled.history()));
      }
      CHECK(baseline.SaveDrawState().behaviour->loop.detections == 0);
      CHECK(baseline.SaveDrawState().behaviour->counters.empty());
      if (const auto state = disabled.SaveDrawState().behaviour) {
        CHECK(state->loop.detections == 0);
        CHECK(state->counters.empty());
      }
    }
  }
  s::SamplerState committed(fixture.config);
  committed.Accept(kOpen);
  auto draft = committed.WithoutConstraint();
  for (int i = 0; i < 4096; ++i)
    draft.Accept(kLoop[i % kLoop.size()]);
  CHECK(draft.SaveDrawState().behaviour->loop.tool.count ==
        committed.SaveDrawState().behaviour->loop.tool.count);
  CHECK(draft.SaveDrawState().behaviour->loop.stopped_tools == 0);
  while (true) {
    auto tentative = committed;
    const auto ordinal = committed.SaveDrawState().behaviour->loop.tool.count;
    CHECK(ordinal < 4096);
    const auto token = kLoop[(ordinal - (kOpen.size() - 1)) % kLoop.size()];
    tentative.Accept(token);
    if (tentative.SaveDrawState().behaviour->loop.stopped_tools) {
      // A discarded speculative acceptance cannot change the committed
      // detector.
      CHECK(committed.SaveDrawState().behaviour->loop.stopped_tools == 0);
      auto restored = tentative;
      restored.RestoreDrawState(committed.SaveDrawState());
      CHECK(restored.SaveDrawState().behaviour->loop.stopped_tools == 0);
      auto rejected = committed;
      CHECK(q::VerifyDraft(fixture.Row(token), 12, rejected, [](int t) {
              return t == 0;
            }) == q::DraftDecision::kReject);
      CHECK(rejected.SaveDrawState().behaviour->loop.tool.count == ordinal);
      auto mtp_rejected = committed;
      q::MtpProposal proposal;
      proposal.size = 1;
      proposal.token = proposal.ids[0] = 12;
      proposal.probability = proposal.probabilities[0] = 1;
      const auto residual =
          q::VerifyMtpProposal(fixture.Row(token), proposal, mtp_rejected);
      CHECK(!residual.accepted && residual.token == token);
      CHECK(mtp_rejected.SaveDrawState().behaviour->loop.tool.count == ordinal);
      mtp_rejected.DeferSample(residual.token);
      CHECK(mtp_rejected.Sample(fixture.Row(token)) == token);
      CHECK(mtp_rejected.SaveDrawState().behaviour->loop.tool.count == ordinal);
      mtp_rejected.Accept(token);
      CHECK(mtp_rejected.Sample(fixture.Row()) == 0);
      auto verified = committed;
      CHECK(q::VerifyDraft(fixture.Row(token), static_cast<int>(token),
                           verified, [](int t) { return t == 0; }) ==
            q::DraftDecision::kAccept);
      committed.CopyDrawStateFrom(verified);
      committed.Accept(token);
      CHECK(committed.SaveDrawState().behaviour->loop.stopped_tools == 1);
      CHECK(committed.Sample(fixture.Row()) == 0);
      break;
    }
    committed.Accept(token);
  }
  committed.ResetHistory({});
  CHECK(!committed.SaveDrawState().behaviour->tool_call_open);
  CHECK(committed.SaveDrawState().behaviour->loop.stopped_tools == 0);
}
}  // namespace

int main() {
  try {
    ThresholdsAndFraming();
    SamplingPaths();
    LegitimateArgumentsAndBoundaries();
    EnvelopeScopeRegressions();
    ExactBoundaryCompletion();
    DisabledIdentityAndRollback();
    std::cout << "tool_argument_loop_test: 6 groups PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
