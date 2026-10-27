#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckLengthController() {
  qfn::MtpLengthController controller(7);
  const auto initial = controller.Choose(7);
  Require(controller.Choose(0) == 0, "empty draft budget was ignored");
  Require(controller.Choose(1) == 1, "one-draft budget was ignored");
  controller.Observe(0, 0);
  Require(controller.Choose(7) == initial, "empty cycle changed policy");

  for (unsigned i = 0; i < 12; ++i) {
    const auto length = controller.Choose(7);
    controller.Observe(length, length);
  }
  Require(controller.Choose(7) == 7,
          "fully accepted prefixes did not grow to the maximum length");
  Require(controller.Choose(2) == 2, "remaining output budget was ignored");
  // After a long run of fully accepted chains the evidence window (about ten
  // observations) needs 13 rejected cycles to back off to AR; the older 0.75
  // weight needed 6.
  unsigned rejected_cycles = 0;
  while (controller.Choose(7) != 0 && rejected_cycles < 20) {
    controller.Observe(0, controller.Choose(7));
    ++rejected_cycles;
  }
  Require(controller.Choose(7) == 0,
          "unprofitable speculation did not back off to AR");
  Require(rejected_cycles >= 8 && rejected_cycles <= 16,
          "back-off to AR took an unexpected number of rejected cycles");
  qfn::MtpLengthController backed_off(7);
  Require(backed_off.Restore(controller.State()), "backoff restore failed");
  for (unsigned i = 0; i < 16; ++i) {
    Require(controller.Choose(7) == 0 && backed_off.Choose(7) == 0,
            "AR interval ended early");
    controller.ObserveArToken();
    backed_off.ObserveArToken();
  }
  Require(controller.Choose(7) == 1 && backed_off.Choose(7) == 1,
          "AR backoff did not retry a proposal");
  controller.Reset();
  Require(controller.Choose(7) == initial, "reset retained old acceptance");
  controller.Observe(5, 6);
  qfn::MtpLengthController restored(7);
  Require(restored.Restore(controller.State()), "controller restore failed");
  for (unsigned budget = 0; budget <= 8; ++budget) {
    Require(restored.Choose(budget) == controller.Choose(budget),
            "restored policy changes draft widths");
  }
  auto invalid = controller.State();
  invalid.successes[0] = -1.0F;
  Require(!restored.Restore(invalid) && !restored.Restore({}),
          "invalid controller state accepted");

  // The unverified suffix is not additional evidence of failure.
  qfn::MtpLengthController short_tail(7);
  qfn::MtpLengthController long_tail(7);
  for (unsigned accepted : {2U, 0U, 1U, 2U, 1U, 0U}) {
    short_tail.Observe(accepted, accepted + 1);
    long_tail.Observe(accepted, 7);
    for (unsigned budget = 0; budget <= 9; ++budget) {
      Require(short_tail.Choose(budget) == long_tail.Choose(budget),
              "unverified suffix biased the acceptance estimate");
    }
  }
  for (unsigned limit = 1; limit <= 7; ++limit) {
    qfn::MtpLengthController a(limit);
    qfn::MtpLengthController b(limit);
    for (unsigned cycle = 0; cycle < 32; ++cycle) {
      const auto budget = cycle % 10;
      const auto length = a.Choose(budget);
      Require(length == b.Choose(budget) && length <= budget && length <= limit,
              "controller violated a limit or changed on replay");
      const unsigned accepted = length == 0 ? 0 : cycle % (length + 1);
      a.Observe(accepted, length);
      b.Observe(accepted, length);
      if (length == 0) {
        a.ObserveArToken();
        b.ObserveArToken();
      }
    }
  }
  qfn::MtpLengthController early_failure(7), late_failure(7);
  for (unsigned i = 0; i < 12; ++i) {
    early_failure.Observe(0, 7);
    late_failure.Observe(4, 7);
  }
  Require(early_failure.Choose(7) == 0 && late_failure.Choose(7) > 1,
          "conditional acceptance by depth was collapsed to one probability");
}

// The evidence window could stay too short; the near-tie threshold could be
// too strict or too loose; the tie could override AR back-off or a better
// width of three, or exceed the budget or controller limit. Rejected chains
// could stop backing off or fail to retry. The larger evidence mass could be
// lost in State(), copying, Restore(), or Reset(). A fully accepted chain
// could stop probing after its delay, and open-text acceptance could still
// favor one draft.
void CheckDraftLengthTuning() {
  constexpr unsigned context = 8192;
  qfn::MtpLengthController tied(7);
  auto state = tied.State();
  state.successes.fill(1.0F);
  state.failures.fill(9.0F);
  state.successes[0] = 7.0F;
  state.failures[0] = 3.0F;
  state.successes[1] = 4.3F;
  state.failures[1] = 5.7F;
  state.explored_depth = 7;
  state.failed_depths = 127;
  Require(tied.Restore(state) && tied.Choose(7, context) == 2,
          "near-tied second draft was not selected");
  qfn::MtpLengthController below_tie(7);
  auto below = state;
  below.successes[1] = 4.0F;
  below.failures[1] = 6.0F;
  Require(below_tie.Restore(below) && tied.Choose(7, context) == 2 &&
              below_tie.Choose(7, context) == 1,
          "second draft crossed the near-tie threshold");

  qfn::MtpLengthController unprofitable(7);
  auto weak = state;
  weak.successes[0] = 2.55F;
  weak.failures[0] = 7.45F;
  weak.successes[1] = 10.0F;
  weak.failures[1] = 0.0F;
  Require(unprofitable.Restore(weak), "low-acceptance state was rejected");
  unprofitable.Observe(0, 1, context);
  Require(tied.Choose(7, context) == 2 &&
              unprofitable.State().retry_tokens == 16 &&
              unprofitable.Choose(7, context) == 0,
          "tie overrode the better AR back-off");

  qfn::MtpLengthController deeper(7);
  auto deep = state;
  deep.successes[0] = 8.0F;
  deep.failures[0] = 2.0F;
  deep.successes[1] = 8.0F;
  deep.failures[1] = 2.0F;
  deep.successes[2] = 6.5F;
  deep.failures[2] = 3.5F;
  Require(deeper.Restore(deep) && tied.Choose(7, context) == 2 &&
              deeper.Choose(7, context) == 3,
          "tie displaced a better chain of three");

  qfn::MtpLengthController limited(1);
  auto short_state = state;
  short_state.explored_depth = 1;
  short_state.failed_depths = 1;
  Require(limited.Restore(short_state) && tied.Choose(7, context) == 2 &&
              tied.Choose(1, context) == 1 && tied.Choose(0, context) == 0 &&
              limited.Choose(7, context) == 1,
          "tie exceeded the budget or controller limit");

  qfn::MtpLengthController rejected(7);
  for (unsigned i = 0; i < 12; ++i)
    rejected.Observe(0, 1, context);
  Require(rejected.State().failures[0] > 7.0F &&
              rejected.State().retry_tokens == 16 &&
              rejected.Choose(7, context) == 0,
          "rejected chains did not retain evidence and back off");
  for (unsigned i = 0; i < 16; ++i) {
    Require(rejected.Choose(7, context) == 0,
            "AR interval ended before the recovery probe");
    rejected.ObserveArToken();
  }
  Require(rejected.State().failures[0] > 7.0F &&
              rejected.Choose(7, context) == 1,
          "AR back-off did not retry after sixteen tokens");
  rejected.Observe(0, 1, context);
  Require(rejected.State().failures[0] > 7.0F &&
              rejected.Choose(7, context) == 0,
          "failed recovery probe did not restore AR back-off");

  qfn::MtpLengthController evidence(7);
  for (unsigned i = 0; i < 80; ++i)
    evidence.Observe(i % 2, 1, context);
  const auto saved = evidence.State();
  const float mass = saved.successes[0] + saved.failures[0];
  const auto same_state = [](const qfn::MtpLengthState& a,
                             const qfn::MtpLengthState& b) {
    return a.successes == b.successes && a.failures == b.failures &&
           a.retry_tokens == b.retry_tokens && a.probe_depth == b.probe_depth &&
           a.explored_depth == b.explored_depth &&
           a.probe_delay == b.probe_delay && a.failed_depths == b.failed_depths;
  };
  auto copied = evidence;
  qfn::MtpLengthController restored(7);
  const bool restored_ok = restored.Restore(saved);
  Require(mass > 9.99F && mass < 10.01F &&
              same_state(saved, copied.State()) && restored_ok &&
              same_state(saved, restored.State()),
          "ten-observation evidence did not survive copying and Restore");
  evidence.Reset();
  copied.Reset();
  const bool reset_restored_ok = restored.Restore(evidence.State());
  Require(mass > 9.99F && same_state(evidence.State(), copied.State()) &&
              evidence.State().successes[0] + evidence.State().failures[0] ==
                  2.0F &&
              reset_restored_ok &&
              same_state(evidence.State(), restored.State()),
          "Reset or restored reset state retained old evidence");

  qfn::MtpLengthController delayed(7);
  auto probe = state;
  probe.successes[0] = 6.5F;
  probe.failures[0] = 3.5F;
  probe.successes[1] = 2.0F;
  probe.failures[1] = 8.0F;
  probe.probe_delay = 1;
  Require(delayed.Restore(probe), "probe-delay state was rejected");
  delayed.Observe(1, 1, context);
  Require(delayed.State().probe_delay == 0 &&
              delayed.State().probe_depth == 0,
          "probe delay was not consumed by an accepted chain");
  delayed.Observe(1, 1, context);
  Require(delayed.State().successes[0] > 7.0F &&
              delayed.State().probe_depth == 2 &&
              delayed.State().probe_delay == 16 &&
              delayed.Choose(7, context) == 2,
          "accepted chains stopped probing deeper after the delay");

  // Fixed acceptance stream: first/second/third conditional acceptance is
  // about 80/60/40%, including misses; the stream is independent of width.
  std::array<unsigned, 320> accepted_stream{};
  constexpr std::array<unsigned, 7> accept_percent{80, 60, 40, 40, 40, 40, 40};
  std::uint32_t seed = 8;
  const auto next = [&seed] {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  for (auto& accepted : accepted_stream)
    while (accepted < accept_percent.size() &&
           next() % 100 < accept_percent[accepted])
      ++accepted;
  qfn::MtpLengthController replay(7);
  std::array<unsigned, 4> widths{};  // 0, 1, 2, deeper; skip warm-up.
  for (unsigned cycle = 0; cycle < accepted_stream.size(); ++cycle) {
    const unsigned length = replay.Choose(7, context);
    if (cycle >= 64)
      ++widths[std::min(length, 3U)];
    if (length == 0)
      replay.ObserveArToken();
    else
      replay.Observe(std::min(accepted_stream[cycle], length), length, context);
  }
  Require(widths == std::array<unsigned, 4>{0, 18, 233, 5},
          "open-text replay did not prefer two drafts over one");
}

void CheckCompactProposals() {
  const std::array<float, 5> target_logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    sampling::SamplerState sampler(test.config, history);
    const auto p = sampler.Distribution(target_logits);
    for (std::uint64_t seed = 0; seed < 16; ++seed) {
      auto rng = seed;
      auto replay_rng = seed;
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &rng);
      const auto replay =
          qfn::SampleMtpProposal(candidates, sampler, &replay_rng);
      Require(q.token == replay.token && q.ids == replay.ids &&
                  q.probabilities == replay.probabilities && rng == replay_rng,
              "compact proposal is not reproducible");
      double sum = 0.0;
      std::array<double, 5> draft{};
      for (std::size_t i = 0; i < q.size; ++i) {
        const double mass = q.probabilities[i] * double(1U << 24);
        Require(mass >= 0.0 && mass == std::floor(mass),
                "proposal probability is not an exact discrete mass");
        draft[q.ids[i]] += q.probabilities[i];
        sum += q.probabilities[i];
      }
      Require(
          sum == 1.0 && q.probability > 0.0F && draft[q.token] == q.probability,
          "sampled q differs from the exported distribution");
      double accepted = 0.0, residual = 0.0;
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        accepted += std::min(probability, draft[i]);
        residual += std::max(probability - draft[i], 0.0);
      }
      Require(std::abs(accepted + residual - 1.0) < 1e-12,
              "acceptance and residual mass do not conserve the target");
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        const double correction =
            residual > 0.0
                ? (1.0 - accepted) * std::max(probability - draft[i], 0.0) /
                      residual
                : 0.0;
        Require(std::abs(std::min(probability, draft[i]) + correction -
                         probability) < 1e-12,
                "bounded draft support changes target output probability");
      }
    }
  }
  sampling::SamplerState counted({.temperature = 1,
                                  .repeat_penalty = 1.5F,
                                  .repeat_last_n = 1,
                                  .frequency_penalty = 0.25F,
                                  .presence_penalty = 0.5F},
                                 std::array<sampling::TokenId, 2>{1, 1});
  counted.Accept(std::array<sampling::TokenId, 2>{3, 3});
  counted.Accept(std::vector<sampling::TokenId>(80, 0));
  counted.Accept(4);
  std::uint64_t rng = 17;
  const auto proposal = qfn::SampleMtpProposal(candidates, counted, &rng);
  // Token 3 is outside the repetition window but keeps both generated counts.
  // Token 1 occurs only in the prompt. Token 4 gets all three penalties.
  const std::array<double, 3> adjusted{1.2 - 2 * 0.25 - 0.5, 0.1,
                                       -0.7 * 1.5 - 0.25 - 0.5};
  double sum = 0;
  for (auto logit : adjusted)
    sum += std::exp(logit);
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto index = std::find(candidates.ids.begin(),
                                 candidates.ids.begin() + 3, proposal.ids[i]) -
                       candidates.ids.begin();
    Require(index < 3 &&
                std::abs(proposal.probabilities[i] -
                         std::exp(adjusted[index]) / sum) < 2.0 / (1U << 24),
            "compact MTP proposal lost full generated penalty counts");
  }
}

void CheckCalibratedCosts() {
  qfn::MtpLengthController single(7, 1), concurrent(7, 8);
  auto state = single.State();
  state.successes.fill(6.0F);
  state.failures.fill(4.0F);
  state.explored_depth = 7;
  state.failed_depths = 127;
  Require(single.Restore(state) && concurrent.Restore(state),
          "calibrated controller state rejected");
  Require(single.Choose(7, 0) > concurrent.Choose(7, 0),
          "concurrent verification cost did not shorten the draft chain");
  single.Observe(0, 1);
  concurrent.Observe(0, 1);
  Require(single.Choose(7) > 0 && concurrent.Choose(7) == 0,
          "measured verification cost did not change the AR break-even point");

  for (unsigned c = 1; c <= 8; ++c) {
    qfn::MtpLengthController perfect(7, c);
    for (unsigned cycle = 0; cycle < 8; ++cycle) {
      const auto length = perfect.Choose(7);
      perfect.Observe(length, length);
    }
    Require(perfect.Choose(7) == 7,
            "perfect acceptance did not retain the fastest measured width");
    for (unsigned depth : {0, 4096, 32768, 131072, 262144}) {
      const auto costs = qfn::MtpCycleCosts(depth, c);
      Require(costs[0] > 0, "AR baseline cost is not positive");
      for (std::size_t i = 1; i < costs.size(); ++i)
        Require(std::isfinite(costs[i]) && costs[i] > costs[i - 1],
                "calibrated costs do not increase with verification work");
    }
  }
}

void CheckBatchProfitability() {
  qfn::MtpLengthController a(7), b(7);
  for (unsigned i = 0; i < 12; ++i) {
    a.Observe(7, 7);
    b.Observe(7, 7);
  }
  qfn::MtpBatchController policy;
  std::array<qfn::MtpBatchController::Row, 2> rows{{{&a, 7}, {&b, 7}}};
  Require(policy.Choose(rows, 0) > 0,
          "transient occupancy paid an unnecessary plain control");
  for (unsigned i = 0; i < 2; ++i)
    policy.Observe(2, 0, policy.Choose(rows, 0), 100.0F);
  Require(policy.Choose(rows, 0) == 0,
          "stable occupancy omitted its plain control");
  // Perfect proposals are unprofitable when measured cycle cost is too high.
  for (unsigned w = 0; w <= 7; ++w)
    for (unsigned repeat = 0; repeat < 3; ++repeat)
      policy.Observe(2, 0, w, w == 0 ? 50.0F : 1000.0F);
  Require(policy.Choose(rows, 0) == 0,
          "perfect acceptance overrode measured unprofitable execution");
  policy.Observe(2, 0, 0, 1000.0F);  // Previous speculative catch-up debt.
  policy.Observe(2, 0, 0, 50.0F);
  Require(policy.Choose(rows, 0) == 0,
          "catch-up debt inflated the measured plain baseline");
  // Context and occupancy samples must not contaminate each other.
  Require(policy.Choose(rows, 32768) > 0,
          "deep batch inherited unprofitable shallow timings");
  for (unsigned i = 0; i < 32; ++i)
    policy.Observe(2, 0, 3, 60.0F);
  Require(policy.Choose(rows, 0) == 3,
          "batch did not select the profitable measured width");
  policy.Observe(2, 0, 7, 500.0F);
  policy.Observe(2, 0, 3, 2000.0F);
  Require(policy.Choose(rows, 0) == 3,
          "a width transition charged the preceding chain to the new width");
  rows[1].budget = 1;
  Require(policy.Choose(rows, 0) <= 1, "batch exceeded one member's budget");
  policy.Observe(2, 0, 0, std::numeric_limits<float>::quiet_NaN());
  Require(policy.Choose(rows, 0) <= 1, "invalid timing corrupted policy");
}

void CheckSampledOutputFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  constexpr unsigned trials = 100000;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 73});
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &draft_rng);
      const auto token = qfn::VerifyMtpProposal(logits, q, sampler).token;
      ++counts[token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      Require(std::abs(double(counts[token]) / trials - p.probability(token)) <
                  0.008,
              "proposal and correction draws distort target frequencies");
    }
  }
}

void CheckVerificationBoundaries() {
  const std::array<float, 3> logits{0, -1, -2};
  const std::array<sampling::TokenId, 3> history{1, 1, 2};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    auto config = test.config;
    config.temperature = 0.5F;
    config.seed = 6;
    // top-p and min-p cutoffs immediately around the finite FP64 boundary.
    for (const float top_p : {0.8F, 0.8668133F, 0.8668134F, 1.0F}) {
      config.top_p = top_p;
      for (const float min_p : {0.0F, 0.13533527F, 0.13533530F}) {
        config.min_p = min_p;
        sampling::SamplerState sampler(config, history);
        sampler.Accept(1);
        const auto target = sampler.Distribution(logits);
        for (unsigned draft = 0; draft < logits.size(); ++draft) {
          qfn::MtpProposal proposal;
          proposal.size = 1;
          proposal.ids[0] = proposal.token = draft;
          proposal.probabilities[0] = proposal.probability = 1;
          for (unsigned draw = 0; draw < 16; ++draw) {
            auto reference = sampler;
            auto replay = sampler;
            const bool accepted =
                reference.Uniform() < target.probability(draft);
            const auto expected =
                accepted ? draft
                         : target.SampleResidual(
                               std::span(proposal.ids).first(1),
                               std::span(proposal.probabilities).first(1),
                               reference.mutable_rng_state());
            const auto actual =
                qfn::VerifyMtpProposal(logits, proposal, sampler);
            const auto repeated =
                qfn::VerifyMtpProposal(logits, proposal, replay);
            Require(actual.accepted == accepted && actual.token == expected &&
                        sampler.rng_state() == reference.rng_state() &&
                        actual.token == repeated.token &&
                        actual.accepted == repeated.accepted &&
                        sampler.rng_state() == replay.rng_state(),
                    "MTP acceptance/residual must use the canonical AR "
                    "distribution exactly");
          }
        }
      }
    }
  }
}

// Survival floor (decode thread 2026-10-09): zero acceptance drives the
// controller into its back-off; survival then takes plain steps (0 drafts) for
// the 16-token AR interval and probes periodically; healthy acceptance and the
// floor switched off keep the old upper-bound width.
void CheckSurvivalFloor() {
  qfn::MtpLengthController controller(7);
  Require(qfn::SurvivalDrafts(controller, 7, 0, true) == 7,
          "healthy survival width changed");
  Require(qfn::SurvivalDrafts(controller, 3, 0, true) == 3,
          "survival ignored the remaining budget");
  Require(qfn::SurvivalDrafts(controller, 0, 0, true) == 0,
          "empty budget drafted");
  unsigned cycles = 0;
  while (qfn::SurvivalDrafts(controller, 7, 0, true) != 0 && cycles < 32) {
    controller.Observe(0, qfn::SurvivalDrafts(controller, 7, 0, true));
    ++cycles;
  }
  Require(cycles < 32, "zero acceptance never reached the survival floor");
  Require(qfn::SurvivalDrafts(controller, 7, 0, false) == 7,
          "floor off changed the survival width");
  for (unsigned round = 0; round < 3; ++round) {
    for (unsigned i = 0; i < 16; ++i) {
      Require(qfn::SurvivalDrafts(controller, 7, 0, true) == 0,
              "plain interval ended early");
      controller.ObserveArToken();
    }
    const auto probe = qfn::SurvivalDrafts(controller, 7, 0, true);
    Require(probe >= 1, "no periodic probe after the plain interval");
    controller.Observe(0, probe);
    Require(qfn::SurvivalDrafts(controller, 7, 0, true) == 0,
            "a failed probe did not restore the plain interval");
  }
  // Recovery: once proposals land again, survival returns to its width.
  unsigned stable = 0;
  for (unsigned i = 0; i < 200 && stable < 8; ++i) {
    const auto drafts = qfn::SurvivalDrafts(controller, 7, 0, true);
    if (drafts == 0) {
      controller.ObserveArToken();
      stable = 0;
      continue;
    }
    controller.Observe(drafts, drafts);
    stable = drafts == 7 ? stable + 1 : 0;
  }
  Require(stable >= 8, "survival did not recover after acceptance returned");
}

void CheckDraftHeadSwitch() {
  qfn::DraftHeadSwitchConfig config;
  Require(qfn::ParseDraftHeadSwitch("", &config) && config.enter == 2 &&
              config.window == 16 && config.leave == 64,
          "empty setting changed the defaults");
  Require(qfn::ParseDraftHeadSwitch("1,1,1", &config) && config.enter == 1 &&
              config.window == 1 && config.leave == 1,
          "minimal setting rejected");
  Require(qfn::ParseDraftHeadSwitch("3,64,200", &config) &&
              config.enter == 3 && config.window == 64 && config.leave == 200,
          "64-token window rejected");
  for (const char* bad : {"0,16,64", "17,16,64", "2,65,64", "2,16,0", "2,16",
                          "2,16,64,1", ",16,64", "2,,64", "2,16,", "a,16,64",
                          "2 ,16,64", "-1,16,64", "2,16,9999999999"}) {
    qfn::DraftHeadSwitchConfig kept;
    Require(!qfn::ParseDraftHeadSwitch(bad, &kept) && kept.enter == 2 &&
                kept.window == 16 && kept.leave == 64,
            "malformed GUFO_DRAFT_VOCAB_AUTO accepted or partially applied");
  }

  qfn::DraftHeadSwitch head;  // enter 2 of 16, leave after 64
  for (int i = 0; i < 100; ++i)
    Require(!head.Observe(true) && !head.full(), "latin output switched");
  // A lone stray token (an accented name, an emoji) does not switch, and it
  // ages out of the window.
  Require(!head.Observe(false) && !head.full(), "one stray token switched");
  for (int i = 0; i < 15; ++i)
    Require(!head.Observe(true), "latin after a stray token switched");
  Require(!head.Observe(false) && !head.full(),
          "a stray token outside the window was still counted");
  // Two non-latin tokens inside the window switch on the second one.
  for (int i = 0; i < 14; ++i)
    (void)head.Observe(true);
  Require(head.Observe(false) && head.full() && head.to_full() == 1,
          "two of the last 16 did not switch to full");
  // Full mode: latin runs shorter than 64 keep it; 64 in a row leave.
  for (int i = 0; i < 63; ++i)
    Require(!head.Observe(true) && head.full(), "left full mode early");
  Require(!head.Observe(false) && head.full(), "non-latin left full mode");
  for (int i = 0; i < 63; ++i)
    Require(!head.Observe(true), "latin run did not restart");
  Require(head.Observe(true) && !head.full() && head.to_latin() == 1,
          "64 latin tokens did not return to latin");
  // Back on latin the window starts empty: one token cannot re-enter.
  Require(!head.Observe(false) && !head.full(), "stale window re-entered");
  Require(head.Observe(false) && head.full() && head.to_full() == 2,
          "second switch to full missed");
  // Alternating script: every second token non-latin stays in full.
  for (int i = 0; i < 200; ++i)
    Require(!head.Observe(i % 2 == 0), "alternating output flipped modes");
  Require(head.full(), "alternating output left full mode");
  head.Reset();
  Require(!head.full() && head.to_full() == 2 && head.to_latin() == 1,
          "reset changed the switch counters or kept full mode");
  qfn::DraftHeadSwitch eager({.enter = 1, .window = 1, .leave = 1});
  Require(eager.Observe(false) && eager.full() && eager.Observe(true) &&
              !eager.full(),
          "1,1,1 setting did not follow each token");
  qfn::DraftHeadSwitch wide({.enter = 2, .window = 64, .leave = 64});
  (void)wide.Observe(false);
  for (int i = 0; i < 62; ++i)
    (void)wide.Observe(true);
  Require(wide.Observe(false) && wide.full(),
          "64-token window lost its oldest token");

  // Graph key: the latin / full (non-auto) keys are the pre-switch layout,
  // and the full-head bit gives an auto session's full mode its own graph.
  const auto old_key = [](std::uint32_t n, bool row, bool token,
                          bool candidates, bool sparse, bool kv_only) {
    return static_cast<std::uint64_t>(n) |
           (static_cast<std::uint64_t>(row) << 32) | (std::uint64_t{1} << 40) |
           (std::uint64_t{token} << 41) | (std::uint64_t{candidates} << 43) |
           (std::uint64_t{sparse} << 44) | (std::uint64_t{kv_only} << 45);
  };
  for (std::uint32_t n = 1; n <= 8; ++n) {
    for (unsigned bits = 0; bits < 32; ++bits) {
      const bool row = bits & 1, token = bits & 2, candidates = bits & 4,
                 sparse = bits & 8, kv_only = bits & 16;
      const auto latin =
          qfn::MtpGraphKey(n, row, token, candidates, sparse, kv_only, false);
      const auto full =
          qfn::MtpGraphKey(n, row, token, candidates, sparse, kv_only, true);
      Require(latin == old_key(n, row, token, candidates, sparse, kv_only),
              "graph key changed for latin / full sessions");
      Require(full != latin && (full ^ latin) == (std::uint64_t{1} << 46),
              "full-head graph key does not differ in its own bit");
    }
  }
}

int main() {
  try {
    CheckVerificationBoundaries();
    CheckLengthController();
    CheckDraftLengthTuning();
    CheckSurvivalFloor();
    CheckDraftHeadSwitch();
    CheckCalibratedCosts();
    CheckBatchProfitability();
    CheckCompactProposals();
    CheckSampledOutputFrequencies();
    const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
    const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      if (test.config.uses_random_sampling())
        continue;
      for (unsigned accepted = 0; accepted < 4; ++accepted) {
        sampling::SamplerState ar(test.config, history);
        auto speculative = ar;
        for (unsigned i = 0; i < accepted; ++i) {
          const auto token = ar.Sample(logits);
          ar.Accept(token);
          Require(qfn::VerifyDraft(logits, static_cast<std::int32_t>(token),
                                   speculative, [](auto) { return false; }) ==
                      qfn::DraftDecision::kAccept,
                  "matching target sample was rejected");
        }
        const auto before = speculative.rng_state();
        const auto saved_history = std::vector(speculative.history().begin(),
                                               speculative.history().end());
        const auto token = ar.Sample(logits);
        Require(qfn::VerifyDraft(logits, (token + 1) % logits.size(),
                                 speculative, [](auto) { return false; }) ==
                    qfn::DraftDecision::kReject,
                "mismatching target sample was accepted");
        Require(speculative.rng_state() == before &&
                    std::vector(speculative.history().begin(),
                                speculative.history().end()) == saved_history,
                "rejected suffix consumed RNG or history");
        Require(speculative.Sample(logits) == token,
                "deferred rejection did not reproduce the AR token");
        ar.Accept(token);
        speculative.Accept(token);
        Require(ar.rng_state() == speculative.rng_state() &&
                    std::vector(ar.history().begin(), ar.history().end()) ==
                        std::vector(speculative.history().begin(),
                                    speculative.history().end()),
                "accepted prefix differs from AR state");
        auto stop_reference = speculative;
        const auto stop_token = stop_reference.Sample(logits);
        Require(qfn::VerifyDraft(
                    logits, (stop_token + 1) % logits.size(), speculative,
                    [&](auto t) {
                      return t == static_cast<std::int32_t>(stop_token);
                    }) == qfn::DraftDecision::kStop,
                "target stop must end verification even when draft differs");
        Require(speculative.rng_state() == stop_reference.rng_state(),
                "stop draw was discarded");
      }
    }
    std::cout
        << "MTP policy replay, greedy prefix/RNG/stops and compact proposal "
           "probabilities pass all sampling configurations\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
