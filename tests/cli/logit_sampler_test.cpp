#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/logprobs.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestGreedySelectsFiniteArgmaxWithoutAdvancingRng() {
  const std::array<float, 4> logits = {
      -1.0F, 5.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  std::uint64_t rng_state = 1234;

  const auto token = gufo::sampling::SampleLogits(logits, 0.0F, &rng_state);

  Expect(token == 1, "greedy sampling selects the finite argmax");
  Expect(rng_state == 1234, "greedy sampling does not consume RNG state");
}

void TestDefaultConfigPreservesGreedyDecoding() {
  const std::array<float, 4> logits = {-1.0F, 5.0F, 3.0F, 2.0F};
  gufo::sampling::SamplerState sampler;

  const auto before = sampler.rng_state();
  const auto token = sampler.Sample(logits);

  Expect(token == 1, "default sampler is greedy");
  Expect(sampler.rng_state() == before,
         "default sampler does not consume RNG state");
}

void TestExternalArgmaxRequiresAppliedPenalties() {
  using gufo::sampling::SamplerState;
  const std::array<float, 2> logits{1.0F, 0.75F};
  SamplerState sampler({.presence_penalty = 1.5F});
  sampler.Accept(0);
  Expect(sampler.Sample(logits) == 1, "penalty changes the greedy winner");
  Expect(!sampler.CanSelectArgmax(0), "raw argmax cannot bypass penalties");
  Expect(sampler.CanSelectArgmax(1, /*penalties_applied=*/true),
         "an exactly penalized argmax avoids downloading verification logits");
  SamplerState sampled({.temperature = .7F, .presence_penalty = 1.5F});
  Expect(!sampled.CanSelectArgmax(1, /*penalties_applied=*/true),
         "applied penalties never bypass random sampling");
}

void TestTemperatureSamplingIsDeterministicAndNonGreedy() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  bool saw_first = false;
  bool saw_non_first = false;

  for (std::uint64_t seed = 1; seed <= 128; ++seed) {
    auto left_state = seed;
    auto right_state = seed;
    const auto left = gufo::sampling::SampleLogits(logits, 1.0F, &left_state);
    const auto right = gufo::sampling::SampleLogits(logits, 1.0F, &right_state);
    Expect(left == right && left_state == right_state,
           "temperature sampling is reproducible from the same RNG state");
    saw_first = saw_first || left == 0;
    saw_non_first = saw_non_first || left != 0;
  }

  Expect(saw_first && saw_non_first,
         "temperature sampling does not collapse to greedy argmax");
}

void TestTopKFiltersTheCandidateSet() {
  const std::array<float, 5> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-k retains exactly k candidates");
  Expect(distribution.probability(0) > 0.0 && distribution.probability(1) > 0.0,
         "top-k retains the highest-logit candidates");
  Expect(distribution.probability(2) == 0.0,
         "top-k removes lower-logit candidates");
}

void TestTopPFiltersByCumulativeProbability() {
  const std::array<float, 3> logits = {3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_p = 0.7F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-p retains the smallest cumulative-probability prefix");
  Expect(distribution.probability(2) == 0.0,
         "top-p removes the low-probability tail");
  config.top_p = 0.0F;
  for (const auto floor : {0U, 1U, 2U}) {
    config.min_keep = floor;
    const auto zero = gufo::sampling::BuildDistribution(logits, config);
    Expect(zero.entries().size() == std::max(floor, 1U),
           "top-p zero preserves the minimum token count");
    Expect(zero.best_token() == 0, "top-p zero keeps the highest probability");
  }
}

void TestMinPFiltersRelativeToTheBestToken() {
  const std::array<float, 3> logits = {3.0F, 2.4F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "min-p retains candidates above the relative threshold");
  Expect(distribution.probability(2) == 0.0,
         "min-p removes candidates far below the best token");
}

void TestMinKeepProvidesAFilterFloor() {
  const std::array<float, 4> logits = {5.0F, 4.0F, 3.0F, 2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 1;
  config.top_p = 0.1F;
  config.min_p = 0.99F;
  config.min_keep = 3;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 3,
         "min-keep prevents enabled filters from shrinking below its floor");
}

void TestTemperatureBeforeProbabilityFilters() {
  // Independent closed-form expectations for softmax([0,-1,-2] / T).
  // Also exercise the bounded (>1024 vocabulary) candidate selector.
  for (const std::size_t size : {3U, 2048U}) {
    std::vector<float> logits(size, -1000.0F);
    logits[0] = 0.0F;
    logits[1] = -1.0F;
    logits[2] = -2.0F;
    for (const bool nucleus : {false, true}) {
      gufo::sampling::SamplingConfig config{.temperature = 0.5F, .seed = 42};
      if (nucleus)
        config.top_p = 0.8F;
      else
        config.min_p = 0.2F;
      gufo::sampling::SamplerState sampler(config);
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            sampler.Distribution(logits)}) {
        Expect(distribution.entries().size() == 1 &&
                   distribution.probability(0) == 1.0,
               "temperature sharpens probabilities before filtering");
      }
      for (unsigned draw = 0; draw < 32; ++draw)
        Expect(sampler.Sample(logits) == 0,
               "ordinary decode uses the tempered candidate set");

      config.temperature = 2.0F;
      config.top_p = nucleus ? 0.8F : 1.0F;
      config.min_p = nucleus ? 0.0F : 0.5F;
      const double first = 1.0 / (1.0 + std::exp(-0.5));
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            gufo::sampling::SamplerState(config).Distribution(logits)}) {
        Expect(
            distribution.entries().size() == 2 &&
                std::abs(distribution.probability(0) - first) < 1e-12 &&
                std::abs(distribution.probability(1) - (1.0 - first)) < 1e-12,
            "temperature flattens probabilities before filtering");
      }
    }
  }
}

void TestRepeatPenaltyUsesCommittedHistory() {
  const std::array<float, 3> logits = {2.0F, 1.0F, 0.0F};
  const std::array<gufo::sampling::TokenId, 1> history = {0};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 3.0F;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 1,
         "repeat penalty can demote a recently used token");
}

void TestFrequencyAndPresencePenaltiesUseCounts() {
  const std::array<float, 2> logits = {2.0F, 1.4F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 0};

  gufo::sampling::SamplingConfig frequency_config;
  frequency_config.frequency_penalty = 0.4F;
  const auto frequency_distribution =
      gufo::sampling::BuildDistribution(logits, frequency_config, {}, history);
  Expect(frequency_distribution.best_token() == 1,
         "frequency penalty scales with occurrence count");

  gufo::sampling::SamplingConfig presence_config;
  presence_config.presence_penalty = 0.7F;
  const auto presence_distribution =
      gufo::sampling::BuildDistribution(logits, presence_config, {}, history);
  Expect(presence_distribution.best_token() == 1,
         "presence penalty applies once when a token is present");
}

void TestRepeatWindowIsBounded() {
  const std::array<float, 2> logits = {2.0F, 2.1F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 1};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 1;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 0,
         "repeat-last-n excludes tokens outside the configured window");
}

void TestFastGreedyMatchesNormalizedPenaltyRoute() {
  const std::array<float, 5> logits = {3.0F, 2.9F, 2.8F, 2.7F, 2.6F};
  const std::array<gufo::sampling::TokenId, 4> history = {0, 0, 1, 3};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 1.5F;
  config.frequency_penalty = 0.2F;
  config.presence_penalty = 0.1F;
  gufo::sampling::SamplerState sampler(config, history);

  const auto expected =
      gufo::sampling::BuildDistribution(logits, config, history).best_token();
  const auto actual = sampler.Sample(logits);

  Expect(actual == expected,
         "linear greedy route matches the normalized penalty route");
}

void TestFastTopKSamplingNeverEscapesSelectedSet() {
  const std::array<float, 6> logits = {8.0F, 7.0F, 1.0F, 0.0F, -1.0F, -2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  bool saw_first = false;
  bool saw_second = false;
  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    const auto token = sampler.Sample(logits);
    Expect(token <= 1, "top-k fast route retains only selected candidates");
    saw_first = saw_first || token == 0;
    saw_second = saw_second || token == 1;
  }
  Expect(saw_first && saw_second,
         "top-k fast route samples the retained distribution");
}

void TestFastMinPSamplingNeverEscapesRelativeThreshold() {
  const std::array<float, 4> logits = {3.0F, 2.5F, 1.0F, -4.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    Expect(sampler.Sample(logits) <= 1,
           "min-p linear route rejects tokens below the relative threshold");
  }
}

void TestSeedAndStateAreRequestLocal() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.seed = 42;
  config.repeat_last_n = 2;
  gufo::sampling::SamplerState left(config);
  gufo::sampling::SamplerState right(config);

  for (int sample = 0; sample < 16; ++sample) {
    Expect(left.Sample(logits) == right.Sample(logits),
           "equal seeds produce equal token sequences");
  }
  left.Accept(1);
  left.Accept(2);
  left.Accept(0);
  Expect(left.history().size() == 2 && left.history()[0] == 2 &&
             left.history()[1] == 0,
         "sampler retains only its bounded committed history");
  Expect(right.history().empty(), "sampler histories are request-local");
}

void TestDistributionIsNormalized() {
  const std::array<float, 4> logits = {2.0F, 1.0F, 0.0F, -1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.8F;
  const auto distribution = gufo::sampling::BuildDistribution(logits, config);
  double sum = 0.0;
  for (const auto& entry : distribution.entries()) {
    sum += entry.value;
  }
  Expect(std::abs(sum - 1.0) < 1e-12,
         "post-filter probabilities are normalized");
}

void TestGreedySparsePenaltyWalk() {
  using namespace gufo::sampling;
  std::vector<float> logits(16385);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = float(int((i * 7919) % 65521) - 32768) / 32768;
  logits[0] = logits[128] = std::numeric_limits<float>::quiet_NaN();
  logits[777] = std::numeric_limits<float>::infinity();
  for (const float repeat : {.7F, 1.F, 1.3F}) {
    for (const float penalty : {-1.5F, 1.5F}) {
      SamplingConfig config{.seed = 73,
                            .repeat_penalty = repeat,
                            .repeat_last_n = 3,
                            .frequency_penalty = penalty / 3,
                            .presence_penalty = penalty};
      const std::vector<TokenId> history{0, 128, 777, 16384, UINT32_MAX};
      SamplerState sampler(config, history);
      std::vector<TokenId> generated;
      for (unsigned step = 0; step < 16; ++step) {
        const auto expected =
            BuildDistribution(logits, config, history, generated);
        Expect(
            sampler.Sample(logits) == expected.best_token(),
            "sparse greedy walk matches independent distribution construction");
        const TokenId token =
            step % 3 == 0 ? 128 : (step * 977) % logits.size();
        generated.push_back(token);
        sampler.Accept(token);
      }
    }
  }
}

void TestSamplingFailsClosedOnInvalidInputs() {
  bool empty_rejected = false;
  try {
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits({}, 1.0F, &rng_state);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  Expect(empty_rejected, "empty logits are rejected");

  bool missing_rng_rejected = false;
  try {
    const std::array<float, 2> logits = {0.0F, 1.0F};
    (void)gufo::sampling::SampleLogits(logits, 1.0F, nullptr);
  } catch (const std::invalid_argument&) {
    missing_rng_rejected = true;
  }
  Expect(missing_rng_rejected, "sampled decoding requires RNG state");

  bool non_finite_rejected = false;
  try {
    const std::array<float, 2> logits = {
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::infinity()};
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits(logits, 1.0F, &rng_state);
  } catch (const std::runtime_error&) {
    non_finite_rejected = true;
  }
  Expect(non_finite_rejected, "an all-non-finite distribution is rejected");

  bool invalid_config_rejected = false;
  try {
    gufo::sampling::SamplingConfig config;
    config.top_p = -0.01F;
    config.Validate();
  } catch (const std::invalid_argument&) {
    invalid_config_rejected = true;
  }
  Expect(invalid_config_rejected, "invalid sampling controls are rejected");
}

void TestZeroDrawAndNonFiniteCandidates() {
  // Inverse xorshift state whose next 53-bit uniform is exactly zero.
  constexpr std::uint64_t zero_draw_state = UINT64_C(0x98d76a164d99a710);
  const std::array<float, 3> cold_logits{-1000.0F, 0.0F, -1.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 0});
  sampler.SetRngState(zero_draw_state);
  Expect(sampler.Sample(cold_logits) == 1,
         "zero draw must not select underflowed zero probability");
  const std::array<float, 5> masked_logits{
      std::numeric_limits<float>::quiet_NaN(), 4.0F,
      std::numeric_limits<float>::infinity(), 0.0F,
      -std::numeric_limits<float>::infinity()};
  for (const auto seed : {0, 73, 808}) {
    gufo::sampling::SamplerState masked({.temperature = 1.0F, .seed = seed});
    auto rng = masked.rng_state();
    const auto expected =
        gufo::sampling::BuildDistribution(masked_logits, masked.config())
            .Sample(&rng);
    Expect(masked.Sample(masked_logits) == expected,
           "linear CPU sampling must ignore nonfinite candidates");
  }
}

void TestStrategyReplayAgainstReference() {
  const std::array<float, 5> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (const auto seed : {0, 1, 73, 808}) {
      auto config = test.config;
      config.seed = seed;
      std::vector<gufo::sampling::TokenId> history{1, 1, 2, 3};
      gufo::sampling::SamplerState sampler(config, history);
      std::vector<gufo::sampling::TokenId> generated;
      auto replay = sampler;
      for (int step = 0; step < 16; ++step) {
        const auto distribution = gufo::sampling::BuildDistribution(
            logits, config, history, generated);
        std::vector<gufo::sampling::Probability> ordered(
            distribution.entries().begin(), distribution.entries().end());
        if (config.top_k == 0 && config.top_p == 1.0F &&
            !(config.min_p > 0.0F && config.min_keep > 1)) {
          std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
        }
        auto rng = sampler.rng_state();
        double threshold = gufo::sampling::Uniform(&rng);
        auto expected = distribution.best_token();
        if (config.temperature > 0.0F) {
          for (const auto& entry : ordered) {
            if (entry.value > 0.0 && threshold < entry.value) {
              expected = entry.token;
              break;
            }
            threshold -= entry.value;
          }
        }
        const auto token = sampler.Sample(logits);
        Expect(token == expected && token == replay.Sample(logits), test.name);
        generated.push_back(token);
        sampler.Accept(token);
        replay.Accept(token);
      }
    }
  }
}

void TestFilteredDistributionAgainstFullSort() {
  std::vector<float> logits(4099);
  const std::array<gufo::sampling::TokenId, 6> history{8, 8, 8, 11, 17, 29};
  for (bool concentrated : {false, true}) {
    for (std::size_t i = 0; i < logits.size(); ++i)
      logits[i] = concentrated ? -(float)((i * 73) % logits.size()) * 0.2F
                               : std::sin((float)i * 0.7F);
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      gufo::sampling::SamplerState sampler(test.config, history);
      const auto rng = sampler.rng_state();
      const auto expected =
          gufo::sampling::BuildDistribution(logits, test.config, history);
      const auto actual = sampler.Distribution(logits);
      const auto positive = std::ranges::count_if(
          expected.entries(), [](const auto& e) { return e.value > 0; });
      Expect(actual.entries().size() == static_cast<std::size_t>(positive),
             test.name);
      for (const auto& entry : expected.entries())
        Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
               test.name);
      Expect(sampler.rng_state() == rng,
             "materializing probabilities consumes no random draws");
    }
  }
}

void TestBlockSkippedTopKAgainstFullSort() {
  // Large rows exercise block skipping: ties straddling the top-k boundary,
  // non-finite entries and penalties that lift a low token into the set.
  std::vector<float> logits(70001);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = std::floor(std::sin(static_cast<float>(i) * 0.37F) * 8.0F);
  logits[5] = std::numeric_limits<float>::quiet_NaN();
  logits[640] = std::numeric_limits<float>::infinity();
  logits[641] = -std::numeric_limits<float>::infinity();
  logits[69999] = 9.0F;
  const std::array<gufo::sampling::TokenId, 7> history{
      3, 700, 701, 33333, 50000, 69999, 70000};
  for (const gufo::sampling::SamplingConfig config :
       {gufo::sampling::SamplingConfig{
            .temperature = 1.0F, .top_k = 64, .top_p = 0.95F},
        gufo::sampling::SamplingConfig{.temperature = 1.0F, .top_k = 1},
        gufo::sampling::SamplingConfig{.temperature = 0.7F, .top_k = 300},
        gufo::sampling::SamplingConfig{.temperature = 1.0F,
                                       .top_k = 64,
                                       .repeat_penalty = 1.3F,
                                       .presence_penalty = -9.0F},
        gufo::sampling::SamplingConfig{.temperature = 0.8F,
                                       .top_k = 20,
                                       .repeat_penalty = 0.5F,
                                       .frequency_penalty = 2.0F}}) {
    gufo::sampling::SamplerState sampler(config, history);
    sampler.Accept(std::span(history).first(3));
    const auto expected = gufo::sampling::BuildDistribution(
        logits, config, history, std::span(history).first(3));
    const auto actual = sampler.Distribution(logits);
    Expect(actual.entries().size() == expected.entries().size(),
           "block-skipped top-k keeps the reference candidate count");
    for (const auto& entry : expected.entries())
      Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
             "block-skipped top-k keeps the reference distribution");
  }
}

void TestDeferredResidualReplay() {
  const std::array<float, 3> logits{100.0F, -100.0F, -100.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 7});
  // A retained residual must win over a fresh draw, even when that fresh
  // draw would almost certainly select a different token.
  sampler.DeferSample(2);
  const auto rng = sampler.rng_state();
  auto copied = sampler;
  gufo::sampling::SamplerState assigned;
  assigned = sampler;
  bool duplicate_rejected = false;
  try {
    sampler.DeferSample(1);
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  Expect(duplicate_rejected, "pending residual cannot be overwritten");
  gufo::sampling::SamplerState published({.temperature = 1.0F});
  published.Accept(1);
  published.CopyDrawStateFrom(sampler);
  Expect(published.history().size() == 1 && published.history()[0] == 1,
         "publishing draw state leaves committed history unchanged");
  for (auto* state : {&sampler, &copied, &assigned, &published}) {
    Expect(state->Sample(logits) == 2 && state->rng_state() == rng,
           "copied residual is consumed without another RNG draw");
    Expect(state->Sample(logits) == 0 && state->rng_state() != rng,
           "residual is consumed only once");
  }
  sampler.DeferSample(2);
  sampler.ResetHistory({});
  Expect(sampler.Sample(logits) == 0, "new history discards pending residual");
  sampler.DeferSample(2);
  sampler.SetRngState(rng);
  Expect(sampler.Sample(logits) == 0, "RNG reset discards pending residual");
}

void TestCanonicalTargetDistribution() {
  std::vector<float> logits(4099, -1000.0F);
  logits[0] = 0;
  logits[17] = -1;
  logits[4098] = -2;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (float temperature : {0.5F, 0.99999994F, 2.0F}) {
      auto config = test.config;
      config.temperature = temperature;
      config.seed = 19;
      gufo::sampling::SamplerState sampler(config);
      for (unsigned step = 0; step < 16; ++step) {
        const auto distribution = sampler.Distribution(logits);
        auto rng = sampler.rng_state();
        const auto expected = distribution.Sample(&rng);
        Expect(sampler.Sample(logits) == expected && sampler.rng_state() == rng,
               "AR and verification share exact support, CDF and RNG draws");
        sampler.Accept(expected);
      }
    }
  }
}

void TestResponsePenaltyScope() {
  using gufo::sampling::SamplerState;
  using gufo::sampling::TokenId;
  const std::array<float, 3> logits{3, 2, 1};
  SamplerState sampler({.temperature = 1,
                        .seed = 7,
                        .repeat_last_n = 0,
                        .frequency_penalty = .5F,
                        .presence_penalty = .25F},
                       std::vector<TokenId>{0, 0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "frequency/presence must not penalize prompt tokens");
  sampler.Accept(std::vector<TokenId>{0, 0});
  sampler.Accept(std::vector<TokenId>(80, 2));
  const std::array<float, 3> expected_logits{1.75F, 2, -39.25F};
  const auto expected =
      gufo::sampling::BuildDistribution(expected_logits, {.temperature = 1});
  for (const auto token : {0U, 1U, 2U})
    Expect(std::abs(sampler.Distribution(logits).probability(token) -
                    expected.probability(token)) < 1e-12,
           "full response counts must outlive the repetition window");
  auto rejected = sampler;
  rejected.Accept(1);
  Expect(sampler.Distribution(logits).best_token() == 1 &&
             rejected.Distribution(logits).best_token() == 0,
         "tentative penalty counts must not leak through speculative rollback");
  auto replay = sampler;
  Expect(replay.Sample(logits) == sampler.Sample(logits),
         "penalty copies preserve RNG replay");
  sampler.ResetHistory(std::vector<TokenId>{0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "new request resets generated counts");
}

void TestDebugHooksRawLogprobs() {
  using gufo::sampling::SamplerState;
  using gufo::sampling::SamplingConfig;
  using gufo::sampling::TokenId;
  const std::array<float, 4> row{4, 3, 3, -2};
  SamplingConfig config;
  config.temperature = .2F;
  config.top_k = 1;
  config.top_p = .2F;
  config.repeat_penalty = 2;
  config.frequency_penalty = .5F;
  config.presence_penalty = .25F;
  config.seed = 47;
  SamplerState sampler(config);
  sampler.Accept(0);
  const auto state = sampler.rng_state();
  const auto history = std::vector<TokenId>(sampler.history().begin(),
                                           sampler.history().end());
  const auto score = sampler.RawLogprobs(row, 0, 4);
  const std::array<double, 4> adjusted{1.25, 3, 3, -2};
  double sum = 0;
  for (const auto value : adjusted)
    sum += std::exp(value - 3);
  const double norm = std::log(sum);
  Expect(std::abs(score.logprob - (adjusted[0] - 3 - norm)) < 1e-12,
         "debug logprobs equal CPU raw penalized log-softmax");
  Expect(score.top_logprobs.size() == row.size() &&
             score.top_logprobs[0].token == 1 &&
             score.top_logprobs[1].token == 2 &&
             score.top_logprobs[2].token == 0,
         "debug top logprobs sorted by raw score with token-ID ties");
  for (const auto& top : score.top_logprobs)
    Expect(std::abs(top.value - (adjusted[top.token] - 3 - norm)) < 1e-12,
           "debug top logprobs use raw unfiltered normalization");
  Expect(sampler.rng_state() == state &&
             std::ranges::equal(sampler.history(), history),
         "debug scoring leaves RNG and history untouched");
  Expect(sampler.RawLogprobs(row, 0, 0).top_logprobs.empty(),
         "zero top logprobs retains selected token score only");
  bool rejected = false;
  try {
    (void)sampler.RawLogprobs(row, 0, 21);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "debug scoring rejects top logprobs above twenty");
  const std::array<float, 2> extreme{1000000, 1000000};
  Expect(std::abs(SamplerState().RawLogprobs(extreme, 1, 2).logprob +
                  std::log(2.0)) < 1e-12,
         "debug log-softmax is stable for large equal logits");
}

void TestDebugHooksLogprobsPreserveSeededDraws() {
  using gufo::sampling::SamplerState;
  gufo::sampling::SamplingConfig config;
  config.temperature = .7F;
  config.top_k = 3;
  config.top_p = .9F;
  config.seed = 71;
  const std::array<float, 5> row{1, 2, 3, 0, -1};
  SamplerState ordinary(config);
  SamplerState candidates(config);
  config.logprobs = 5;
  SamplerState scored(config);
  const std::array<float, 4> candidate_values{3, 2, 1, 0};
  const std::array<gufo::sampling::TokenId, 4> candidate_ids{2, 1, 0, 3};
  for (std::size_t step = 0; step < 32; ++step) {
    const auto baseline = ordinary.Sample(row);
    auto pre_draw = scored;
    const auto token = scored.Sample(row);
    (void)pre_draw.RawLogprobs(row, token, 5);
    Expect(token == baseline && scored.rng_state() == ordinary.rng_state(),
           "debug raw logprobs preserve seeded sampling and RNG draws");
    const auto candidate = candidates.SampleFromTop(candidate_values,
                                                    candidate_ids, row.size());
    Expect(candidate && *candidate == token &&
               candidates.rng_state() == scored.rng_state(),
           "debug full-row sampling preserves candidate-path draws");
    ordinary.Accept(token);
    scored.Accept(token);
    candidates.Accept(token);
  }
}

void TestGreedyToolWinnerDoesNotBuildVocabularyMask() {
  using namespace gufo::sampling;
  using Format = JsonConstraint::ToolFormat;
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false})");
  const auto grammar = JsonConstraint::WithTools(
      nullptr, {{"f", JsonConstraint::ToolParameters(schema, false, Format::kQwen)}},
      false, true, Format::kQwen);
  const auto binding = [&] {
    auto c = std::make_shared<TokenConstraint>();
    c->grammar = grammar;
    c->vocabulary = std::make_shared<ConstraintVocabulary>(257, [](std::uint32_t i) {
      return ConstraintVocabulary::Piece{
          i < 256 ? std::string(1, static_cast<char>(i)) : "", i == 256};
    });
    return c;
  };
  auto policy = std::make_shared<BehaviourConfig>();
  policy->eog_boundary = true;
  policy->eog_min_prob = .01;
  policy->eog_ids = {256};
  const std::string prefix = "<tool_call>\n<function=f>\n<parameter=text>\n";
  for (bool illegal_raw_winner : {false, true}) {
    auto c = binding();
    SamplingConfig config;
    config.constraint = c;
    config.behaviour = policy;
    SamplerState sampler(config);
    for (unsigned char ch : prefix)
      sampler.Accept(ch);
    Expect(c->CachedMaskCount() == 0, "setup has no materialized masks");
    Expect(!c->vocabulary->Allows(*grammar, sampler.constraint_state(), 256),
           "unfinished tool call forbids EOG and proves a restrictive grammar");
    std::vector<float> row(257, -INFINITY);
    row['a'] = row['b'] = 2;
    row[256] = illegal_raw_winner ? 9 : -8;
    // Independent full-mask oracle; this does not populate TokenConstraint's cache.
    const auto mask = c->vocabulary->Allowed(*grammar, sampler.constraint_state());
    std::uint32_t expected = 0;
    float best = -INFINITY;
    for (std::uint32_t id = 0; id < row.size(); ++id)
      if (mask[id] && row[id] > best) { best = row[id]; expected = id; }
    if (!illegal_raw_winner) {
      Expect(sampler.CanSelectArgmax('a', true), "legal GPU winner remains eligible");
      Expect(c->CachedMaskCount() == 0, "winner eligibility avoids full mask");
    }
    const auto before = sampler.rng_state();
    const auto distribution = sampler.Distribution(row);
    Expect(distribution.best_token() == expected, "lazy greedy equals full-mask oracle");
    Expect(sampler.Sample(row) == expected && sampler.rng_state() == before,
           "lazy greedy preserves selected token, token-ID ties and RNG");
    if (!illegal_raw_winner) {
      Expect(c->CachedMaskCount() == 0, "legal anchor avoids full mask");
      Expect(distribution.behaviour() && distribution.behaviour()->stages.empty(),
             "restrictive grammar keeps EOG interventions inactive");
    }
  }
}

void TestDebugHooksBiasAndGrammarLogprobs() {
  using namespace gufo::sampling;
  const std::array<std::string, 5> pieces{"x", "{", " ", "[", "\""};
  auto constraint = std::make_shared<TokenConstraint>();
  constraint->grammar = JsonConstraint::Object();
  constraint->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t token) {
        return ConstraintVocabulary::Piece{pieces[token], false};
      });
  auto behaviour = std::make_shared<BehaviourConfig>();
  behaviour->logit_bias_enabled = true;
  behaviour->logit_bias = {{1, 2}};
  SamplingConfig config;
  config.constraint = constraint;
  config.behaviour = behaviour;
  config.temperature = .25F;
  config.top_k = 1;
  const std::array<float, 5> row{20, 1, 0, 8, 9};
  const auto mask = constraint->Allowed(constraint->grammar->Start());
  double maximum = -INFINITY;
  for (std::size_t i = 0; i < row.size(); ++i)
    if ((*mask)[i])
      maximum = std::max(maximum, static_cast<double>(row[i] + (i == 1 ? 2 : 0)));
  double sum = 0;
  for (std::size_t i = 0; i < row.size(); ++i)
    if ((*mask)[i])
      sum += std::exp(row[i] + (i == 1 ? 2 : 0) - maximum);
  const auto score = SamplerState(config).RawLogprobs(row, 1, 5);
  Expect(std::abs(score.logprob - (3 - maximum - std::log(sum))) < 1e-12,
         "debug scores apply explicit bias and grammar before normalization");
  for (const auto& top : score.top_logprobs)
    Expect((*mask)[top.token] != 0,
           "debug top logprobs exclude grammar-masked tokens");
}

void TestDebugHooksTeacherForcedFirstTokenPhase() {
  using namespace gufo::sampling;
  auto behaviour = std::make_shared<BehaviourConfig>();
  behaviour->think_start = {0};
  behaviour->think_end = {1};
  behaviour->eog_ids = {2};
  behaviour->think_eog_guard = true;
  SamplingConfig config;
  config.behaviour = behaviour;
  config.logprobs = 0;
  const std::array<float, 4> row{1, 2, 3, 4};
  SamplerState sampler(config);
  Expect(std::isfinite(sampler.RawLogprobs(row, 2, 0).logprob),
         "prompt scorer starts outside thinking before the first token");
  sampler.Accept(TokenId{0});
  Expect(sampler.RawLogprobs(row, 2, 0).logprob == -INFINITY,
         "teacher-forced first token updates thinking masks for the second row");
  sampler.Accept(TokenId{1});
  Expect(std::isfinite(sampler.RawLogprobs(row, 2, 0).logprob),
         "teacher-forced closing tag restores the content distribution");
}

void TestDebugHooksLogprobTransport() {
  using gufo::sampling::TokenId;
  using gufo::sampling::TokenLogprob;
  const std::array<TokenLogprob, 2> scores{
      TokenLogprob{0, -.25, {{0, -.25}, {1, -2}}},
      TokenLogprob{1, -INFINITY, {}}};
  const auto text = [](TokenId token) {
    return token == 0 ? std::string("\xC3\xA9") : std::string("x");
  };
  const auto chat = gufo::server::ChatLogprobs(scores, text);
  const auto& content = chat.find("content")->items();
  const auto& bytes = content[0].find("bytes")->items();
  Expect(content.size() == 2 && content[0].member_str("token") == text(0) &&
             bytes.size() == 2 && bytes[0].as_size() == 195 &&
             bytes[1].as_size() == 169,
         "debug chat logprobs retain multibyte token bytes as unsigned values");
  Expect(content[0].find("top_logprobs")->size() == 2 &&
             content[1].find("top_logprobs")->empty() &&
             content[1].member_double("logprob") == -9999,
         "debug chat top0 shape and masked-token sentinel are explicit");
  const std::array<TokenId, 2> tokens{0, 1};
  const std::array<std::optional<TokenLogprob>, 2> prompt{std::nullopt, scores[1]};
  const auto legacy = gufo::server::LegacyLogprobs(tokens, prompt, text);
  Expect(legacy.find("token_logprobs")->items()[0].is_null() &&
             legacy.find("top_logprobs")->items()[0].is_null() &&
             legacy.find("tokens")->size() == 2 &&
             legacy.find("text_offset")->items()[1].as_size() == 1,
         "debug prompt scoring first token is null and offsets count characters");
  const auto split_text = [](TokenId token) {
    return token == 0 ? std::string("\xC3") : std::string("\xA9");
  };
  const auto split = gufo::server::ChatLogprobs(scores, split_text);
  for (std::size_t i = 0; i < scores.size(); ++i) {
    const auto& record = split.find("content")->items()[i];
    Expect(record.member_str("token") == "\xEF\xBF\xBD" &&
               record.find("bytes")->size() == 1 &&
               record.find("bytes")->items()[0].as_size() == (i == 0 ? 195 : 169),
           "debug split UTF8 token strings use replacement and retain raw bytes");
  }
  const auto split_legacy = gufo::server::LegacyLogprobs(tokens, prompt, split_text);
  Expect(split_legacy.find("tokens")->items()[0].str() == "\xEF\xBF\xBD" &&
             split_legacy.find("tokens")->items()[1].str() == "\xEF\xBF\xBD",
         "debug legacy split token strings contain valid UTF8");
  const auto generated_legacy = gufo::server::LegacyLogprobs(scores, split_text);
  for (const auto& top : generated_legacy.find("top_logprobs")->items())
    for (const auto& [key, value] : top.members())
      Expect(key == "\xEF\xBF\xBD",
             "debug legacy top logprob keys sanitize split codepoints");
}

void TestScoringPreservesMtpAcceptReject() {
  using namespace gufo::sampling;
  using namespace gufo::models::qwen38_flash_next;
  bool accepted = false, rejected = false;
  const std::array<float, 4> row{2.0F, 0.0F, -1.0F, -2.0F};
  const std::array<float, 3> values{2.0F, 0.0F, -1.0F};
  const std::array<TokenId, 3> ids{0, 1, 2};
  for (float temperature : {0.0F, 1.0F}) {
    for (std::uint64_t seed = 1; seed <= 128; ++seed) {
      SamplingConfig off;
      off.temperature = temperature;
      off.top_k = 2;
      off.seed = seed;
      auto on = off;
      on.logprobs = 2;
      SamplerState plain(off), scored(on);
      MtpProposal proposal;
      proposal.token = 1;
      proposal.size = 2;
      proposal.ids[0] = 0;
      proposal.ids[1] = 1;
      proposal.probabilities[0] = .5F;
      proposal.probabilities[1] = .5F;
      proposal.probability = .5F;
      for (int cycle = 0; cycle < 4; ++cycle) {
        auto before = scored;
        const auto candidates =
            temperature > 0 ? plain.DistributionFromTop(values, ids, row.size())
                            : std::optional<SamplingDistribution>{};
        if (temperature > 0)
          Expect(candidates.has_value(),
                 "compact candidate support is proven exact");
        const auto a = candidates ? VerifyMtpProposal(*candidates, row.size(),
                                                      proposal, plain)
                                  : VerifyMtpProposal(row, proposal, plain);
        const auto b = VerifyMtpProposal(row, proposal, scored);
        const auto record = before.RawLogprobs(row, b.token, 2);
        Expect(a.token == b.token && a.accepted == b.accepted &&
                   plain.rng_state() == scored.rng_state() &&
                   record.token == b.token,
               "scoring preserves production MTP acceptance, residual token "
               "and RNG frontier");
        accepted |= a.accepted;
        rejected |= !a.accepted;
        plain.Accept(a.token);
        scored.Accept(b.token);
      }
    }
  }
  Expect(accepted && rejected,
         "speculative proof exercises both accept and reject decisions");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--debug-hooks") {
    TestDebugHooksRawLogprobs();
    TestDebugHooksLogprobsPreserveSeededDraws();
    TestScoringPreservesMtpAcceptReject();
    TestDebugHooksBiasAndGrammarLogprobs();
    TestDebugHooksTeacherForcedFirstTokenPhase();
    TestDebugHooksLogprobTransport();
    return 0;
  }

  bool sub_float_resolution = false;
  for (std::uint64_t seed = 1; seed < 100; ++seed) {
    auto state = seed;
    auto reference = seed;
    const auto bits = gufo::sampling::NextRandom(&reference);
    const double draw = gufo::sampling::Uniform(&state);
    Expect(draw >= 0 && draw < 1 && state == reference,
           "uniform stays half-open and consumes one RNG word");
    Expect(draw == static_cast<double>(bits >> 11) * 0x1.0p-53,
           "uniform retains all 53 random mantissa bits");
    sub_float_resolution |=
        draw != static_cast<double>(static_cast<float>(draw));
  }
  Expect(sub_float_resolution, "uniform must exceed float resolution");
  TestCanonicalTargetDistribution();
  TestScoringPreservesMtpAcceptReject();
  TestDebugHooksRawLogprobs();
  TestDebugHooksLogprobsPreserveSeededDraws();
  TestDebugHooksBiasAndGrammarLogprobs();
  TestDebugHooksTeacherForcedFirstTokenPhase();
  TestDebugHooksLogprobTransport();
  TestResponsePenaltyScope();
  TestGreedySelectsFiniteArgmaxWithoutAdvancingRng();
  TestDefaultConfigPreservesGreedyDecoding();
  TestExternalArgmaxRequiresAppliedPenalties();
  TestGreedyToolWinnerDoesNotBuildVocabularyMask();
  TestTemperatureSamplingIsDeterministicAndNonGreedy();
  TestTopKFiltersTheCandidateSet();
  TestTopPFiltersByCumulativeProbability();
  TestMinPFiltersRelativeToTheBestToken();
  TestMinKeepProvidesAFilterFloor();
  TestTemperatureBeforeProbabilityFilters();
  TestRepeatPenaltyUsesCommittedHistory();
  TestFrequencyAndPresencePenaltiesUseCounts();
  TestRepeatWindowIsBounded();
  TestFastGreedyMatchesNormalizedPenaltyRoute();
  TestFastTopKSamplingNeverEscapesSelectedSet();
  TestFastMinPSamplingNeverEscapesRelativeThreshold();
  TestSeedAndStateAreRequestLocal();
  TestDistributionIsNormalized();
  TestGreedySparsePenaltyWalk();
  TestSamplingFailsClosedOnInvalidInputs();
  TestZeroDrawAndNonFiniteCandidates();
  TestStrategyReplayAgainstReference();
  TestFilteredDistributionAgainstFullSort();
  TestBlockSkippedTopKAgainstFullSort();
  TestDeferredResidualReplay();
  std::cout << "All logit sampler tests passed.\n";
  return 0;
}
