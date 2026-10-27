#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/cli/serve/behaviour_options.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace s = gufo::sampling;
namespace q = gufo::models::qwen38_flash_next;
namespace {
static_assert(!std::is_trivially_copy_constructible_v<
              decltype(s::BehaviourConfig{}.marker_entropy_gate)>);
static_assert(!std::is_trivially_copy_assignable_v<
              decltype(s::BehaviourConfig{}.marker_entropy_gate)>);
static_assert(!std::is_trivially_copy_constructible_v<
              std::optional<s::ThinkMarkerCounts>>);
static_assert(
    !std::is_trivially_copy_assignable_v<std::optional<s::ThinkMarkerCounts>>);
static_assert(
    !std::is_trivially_copy_constructible_v<std::optional<s::ThinkMarkerRow>>);
static_assert(
    !std::is_trivially_copy_assignable_v<std::optional<s::ThinkMarkerRow>>);
#define CHECK(condition)                                             \
  do {                                                               \
    if (!(condition))                                                \
      throw std::runtime_error(std::string(#condition) +             \
                               " line=" + std::to_string(__LINE__)); \
  } while (false)

std::uint64_t Bits(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

std::shared_ptr<s::BehaviourConfig> Behaviour(bool thinking = true) {
  auto b = std::make_shared<s::BehaviourConfig>();
  b->initial_thinking = thinking;
  b->marker_ids = {2, 5};
  b->marker_bias = -2;
  b->think_start = {10};
  b->think_end = {11, 12};
  return b;
}

s::SamplingConfig Config(std::shared_ptr<s::BehaviourConfig> b,
                         float temperature = 1, int top_k = 3) {
  s::SamplingConfig c;
  c.behaviour = std::move(b);
  c.temperature = temperature;
  c.top_k = top_k;
  c.seed = 42;
  return c;
}

void Same(const s::SamplingDistribution& a, const s::SamplingDistribution& b) {
  CHECK(a.entries().size() == b.entries().size());
  for (std::size_t i = 0; i < a.entries().size(); ++i) {
    CHECK(a.entries()[i].token == b.entries()[i].token);
    CHECK(Bits(a.entries()[i].value) == Bits(b.entries()[i].value));
  }
}

bool Stage(const s::SamplingDistribution& d, std::string_view name) {
  return std::ranges::any_of(d.behaviour()->stages, [&](const auto& stage) {
    return stage.name == name;
  });
}

const s::ThinkMarkerRow& Observation(const s::SamplingDistribution& d) {
  CHECK(d.behaviour() && d.behaviour()->think_marker_row);
  return *d.behaviour()->think_marker_row;
}

s::ThinkMarkerCounts Counts(const s::SamplerState& sampler) {
  const auto state = sampler.SaveDrawState();
  CHECK(state.behaviour);
  return state.behaviour->think_markers.value_or(s::ThinkMarkerCounts{});
}

std::uint64_t MarkerTaken(const s::SamplerState& sampler) {
  std::uint64_t total = 0;
  for (const auto& line : sampler.BehaviourSummary())
    if (line.find("name=marker_bias ") != std::string::npos)
      total += std::stoull(line.substr(line.find("taken=") + 6));
  return total;
}

void SameCounts(const s::ThinkMarkerCounts& a, const s::ThinkMarkerCounts& b) {
  CHECK(a.think_tokens == b.think_tokens && a.markers == b.markers);
  CHECK(a.gate_rows == b.gate_rows && a.gate_open == b.gate_open);
  CHECK(a.hist_all == b.hist_all && a.hist_marker == b.hist_marker);
}

std::vector<s::TokenId> TopIds(const std::vector<float>& row,
                               std::size_t keep) {
  std::vector<s::TokenId> ids(row.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::ranges::sort(ids, [&](auto a, auto b) {
    return row[a] == row[b] ? a < b : row[a] > row[b];
  });
  ids.resize(keep);
  // The transport need not be ordered by logits or global ID.
  std::ranges::reverse(ids);
  return ids;
}

std::vector<float> Values(const std::vector<float>& row,
                          const std::vector<s::TokenId>& ids) {
  std::vector<float> values;
  for (const auto id : ids)
    values.push_back(row[id]);
  return values;
}

// Independent shipped marker loop, optionally skipped, then the native sampler.
// The existing frozen behaviour_fast_equivalence test additionally pins all
// pre-gate metadata/state against the unchanged c847e6df source fixture.
s::SamplingDistribution Legacy(const s::SamplerState& sampler,
                               std::span<const float> raw,
                               std::span<const s::TokenId> ids = {},
                               bool global = false, bool marker_stage = true) {
  const auto& b = *sampler.config().behaviour;
  std::vector<float> changed(raw.begin(), raw.end());
  if (marker_stage && b.initial_thinking && b.marker_bias != 0) {
    if (ids.empty()) {
      for (auto token : b.marker_ids)
        if (token < changed.size())
          changed[token] += static_cast<float>(b.marker_bias);
    } else {
      for (std::size_t i = 0; i < changed.size(); ++i)
        if (std::ranges::find(b.marker_ids, ids[i]) != b.marker_ids.end())
          changed[i] += static_cast<float>(b.marker_bias);
    }
  }
  const auto native = sampler.WithoutBehaviour();
  if (ids.empty())
    return native.Distribution(changed);
  if (!global)
    return native.Distribution(changed, ids);
  std::vector<std::size_t> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](auto a, auto b) { return ids[a] < ids[b]; });
  std::vector<s::TokenId> ordered_ids;
  std::vector<float> ordered_values;
  for (auto i : order) {
    ordered_ids.push_back(ids[i]);
    ordered_values.push_back(changed[i]);
  }
  const auto local = native.Distribution(ordered_values, ordered_ids);
  std::vector<s::Probability> entries;
  for (auto e : local.entries())
    entries.push_back({ordered_ids[e.token], e.value});
  return s::SamplingDistribution(std::move(entries));
}

void UnsetAndCalibrationIdentity() {
  for (bool thinking : {false, true}) {
    for (int top_k : {0, 1, 3, 20}) {
      for (float temperature : {0.0F, 0.7F, 1.0F}) {
        for (int pattern = 0; pattern < 4; ++pattern) {
          std::vector<float> row(32);
          for (std::size_t i = 0; i < row.size(); ++i)
            row[i] = -100 - static_cast<float>(i);
          for (unsigned i = 0; i < 8; ++i)
            row[i] = pattern == 0
                         ? 4.0F
                         : 6.0F - static_cast<float>(i) * (pattern * .5F);
          if (pattern == 3)
            row[2] = -INFINITY;
          auto b = Behaviour(thinking);
          auto gated = std::make_shared<s::BehaviourConfig>(*b);
          gated->marker_entropy_gate = 0;
          s::SamplerState shipped(Config(b, temperature, top_k));
          s::SamplerState calibration(Config(gated, temperature, top_k));
          auto actual = shipped.Distribution(row);
          const auto expected = Legacy(shipped, row);
          Same(actual, expected);
          Same(actual, calibration.Distribution(row));
          CHECK(!actual.behaviour()->think_marker_row);
          for (std::uint64_t seed = 0; seed < 32; ++seed) {
            auto a = shipped, z = calibration;
            a.SetRngState(seed);
            z.SetRngState(seed);
            auto legacy_rng = seed;
            const auto token = temperature == 0 ? expected.best_token()
                                                : expected.Sample(&legacy_rng);
            CHECK(a.Sample(row) == token && z.Sample(row) == token);
            CHECK(a.rng_state() == legacy_rng && z.rng_state() == legacy_rng);
          }
          const auto ids = TopIds(row, row.size());
          const auto values = Values(row, ids);
          Same(shipped.Distribution(values, ids), Legacy(shipped, values, ids));
          Same(shipped.Distribution(values, ids),
               calibration.Distribution(values, ids));
          const auto local = shipped.Distribution(values, ids);
          const auto local_calibration = calibration.Distribution(values, ids);
          const auto local_legacy = Legacy(shipped, values, ids);
          for (std::uint64_t seed = 0; seed < 32; ++seed) {
            auto a = seed, z = seed, expected_rng = seed;
            const auto token = local_legacy.Sample(&expected_rng);
            CHECK(local.Sample(&a) == token &&
                  local_calibration.Sample(&z) == token);
            CHECK(a == expected_rng && z == expected_rng);
          }
          if (top_k > 0 && temperature > 0) {
            const auto full =
                shipped.DistributionFromTop(values, ids, row.size());
            const auto on =
                calibration.DistributionFromTop(values, ids, row.size());
            CHECK(full && on);
            Same(*full, Legacy(shipped, values, ids, true));
            Same(*full, *on);
            for (std::uint64_t seed = 0; seed < 32; ++seed) {
              auto a = shipped, z = calibration;
              a.SetRngState(seed);
              z.SetRngState(seed);
              auto expected_rng = seed;
              const auto token = full->Sample(&expected_rng);
              CHECK(a.SampleFromTop(values, ids, row.size()) == token);
              CHECK(z.SampleFromTop(values, ids, row.size()) == token);
              CHECK(a.rng_state() == expected_rng &&
                    z.rng_state() == expected_rng);
            }
          }
          for (const auto& line : shipped.BehaviourSummary())
            CHECK(!line.starts_with("THINK_MARKERS "));
          CHECK(!shipped.SaveDrawState().behaviour->think_markers);
        }
      }
    }
  }
}

void ClosedIdentityAndRawEntropy() {
  std::vector<float> row(100, -50);
  row[2] = 3;
  row[5] = 2.75F;
  row[7] = 2.5F;
  row[8] = 2;
  const auto ids = TopIds(row, 8);
  const auto values = Values(row, ids);
  for (bool thinking : {false, true}) {
    auto b = Behaviour(thinking);
    b->marker_entropy_gate = 12;
    auto zero = std::make_shared<s::BehaviourConfig>(*b);
    zero->marker_bias = 0;
    zero->marker_entropy_gate.reset();
    s::SamplerState gated(Config(b)), off(Config(zero));
    CHECK(gated.BehaviourEnabled() && !off.BehaviourEnabled());
    const auto full = gated.Distribution(row);
    const auto local = gated.Distribution(values, ids);
    Same(full,
         Legacy(gated, row, {}, /*global=*/false, /*marker_stage=*/false));
    Same(local, Legacy(gated, values, ids, /*global=*/false,
                       /*marker_stage=*/false));
    Same(full, off.Distribution(row));
    Same(local, off.Distribution(values, ids));
    const auto a = gated.DistributionFromTop(values, ids, row.size());
    const auto z = off.DistributionFromTop(values, ids, row.size());
    CHECK(a && z);
    // Zero bias disables this fixture's behaviour layer. Its native global
    // candidate shortcut normalizes once; shipped behaviour uses the local
    // sampler's normalized SoftmaxWeights and renormalizes in the public
    // distribution constructors. Skip only the marker stage in that same
    // path so this remains a bit-exact test, without a tolerance.
    const auto skipped = Legacy(gated, values, ids, /*global=*/true,
                                /*marker_stage=*/false);
    Same(*a, skipped);
    CHECK(a->entries().size() == z->entries().size());
    for (std::size_t i = 0; i < a->entries().size(); ++i)
      CHECK(a->entries()[i].token == z->entries()[i].token);
    CHECK(!Stage(full, "marker_bias") && !Stage(local, "marker_bias") &&
          !Stage(*a, "marker_bias"));
    if (thinking) {
      CHECK(!Observation(full).gate_open && !Observation(local).gate_open &&
            !Observation(*a).gate_open);
    }
    for (unsigned i = 0; i < 64; ++i) {
      CHECK(gated.Sample(row) == off.Sample(row));
      CHECK(gated.rng_state() == off.rng_state());
      auto candidate = gated;
      candidate.SetRngState(i);
      auto expected_rng = static_cast<std::uint64_t>(i);
      const auto drawn = candidate.SampleFromTop(values, ids, row.size());
      CHECK(drawn == skipped.Sample(&expected_rng));
      CHECK(candidate.rng_state() == expected_rng);
      auto native_rng = static_cast<std::uint64_t>(i);
      CHECK(drawn == z->Sample(&native_rng));
      CHECK(candidate.rng_state() == native_rng);
    }
  }
  auto b = Behaviour();
  b->marker_entropy_gate = 0;
  auto c = Config(b, .25F, 3);
  c.min_keep = 5;  // Entropy uses exactly top_k, not the sampling floor.
  c.top_p = .7F;
  c.min_p = .1F;
  const double entropy =
      Observation(s::SamplerState(c).Distribution(row)).entropy;
  const double sum = 1 + std::exp(-.25) + std::exp(-.5);
  const double expected =
      std::log(sum) - (std::exp(-.25) / sum * -.25 + std::exp(-.5) / sum * -.5);
  CHECK(std::abs(entropy - expected) < 1e-14);
  // Each earlier transform would drastically change post-transform entropy.
  b->logit_bias_enabled = true;
  b->logit_bias = {{2, 20}, {9, 100}};
  b->model_suppress_tokens = true;
  b->suppress_ids = {5};
  b->eog_min_prob_exact = true;
  c.temperature = 2;
  CHECK(Bits(Observation(s::SamplerState(c).Distribution(row)).entropy) ==
        Bits(entropy));
  c.top_k = 0;
  CHECK(Observation(s::SamplerState(c).Distribution(row)).entropy > entropy);
}

void FullCompactEntropyBits() {
  for (int top_k : {1, 3, 20}) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      std::vector<float> row(100, -5);
      for (unsigned i = 0; i < 24; ++i)
        row[i] =
            pattern == 0 ? 4.0F : 4 - static_cast<float>(i) * (.01F * pattern);
      if (pattern == 3)
        row[4] = -INFINITY;
      auto b = Behaviour();
      b->marker_entropy_gate = 1;
      b->marker_bias = -.1;
      s::SamplerState sampler(Config(b, .7F, top_k));
      const auto ids = TopIds(row, 32);
      const auto values = Values(row, ids);
      const auto full = sampler.Distribution(row);
      const auto local = sampler.Distribution(values, ids);
      const auto global = sampler.DistributionFromTop(values, ids, row.size());
      CHECK(global);
      for (const auto* compact : {&local, &*global}) {
        CHECK(Bits(Observation(full).entropy) ==
              Bits(Observation(*compact).entropy));
        CHECK(Observation(full).gate_open == Observation(*compact).gate_open);
        CHECK(Stage(full, "marker_bias") == Stage(*compact, "marker_bias"));
      }
      for (s::TokenId i = 0; i < row.size(); ++i)
        CHECK(std::abs(full.probability(i) - global->probability(i)) < 1e-14);
    }
  }
  // An incomplete raw top-k never supplies a candidate decision.
  auto b = Behaviour();
  b->marker_entropy_gate = 1;
  s::SamplerState sampler(Config(b, 1, 3));
  const std::vector<float> values = {3, 2, 1};
  const std::vector<s::TokenId> ids = {2, 5, 7};
  CHECK(!sampler.DistributionFromTop(values, ids, 100));
  CHECK(Counts(sampler).gate_rows == 0);
}

void ThresholdOnlyMarkerStage() {
  auto b = Behaviour();
  b->marker_entropy_gate = 0;
  b->logit_bias_enabled = true;
  b->logit_bias = {{4, 0}};
  b->model_suppress_tokens = true;
  b->suppress_ids = {6};
  b->think_eog_guard = true;
  b->eog_ids = {0};
  std::vector<float> row(16, -20);
  row[2] = 1;
  row[3] = 0;
  const double entropy =
      Observation(s::SamplerState(Config(b, 0, 2)).Distribution(row)).entropy;
  b->marker_entropy_gate = entropy;
  s::SamplerState equal(Config(b, 0, 2));
  const auto open = equal.Distribution(row);
  CHECK(Observation(open).gate_open && Stage(open, "marker_bias"));
  CHECK(equal.Sample(row) == 3);
  CHECK(MarkerTaken(equal) == 1);
  b = std::make_shared<s::BehaviourConfig>(*b);
  // The row lies exactly one representable double below this threshold.
  b->marker_entropy_gate = std::nextafter(entropy, INFINITY);
  s::SamplerState below(Config(b, 0, 2));
  const auto closed = below.Distribution(row);
  CHECK(!Observation(closed).gate_open && !Stage(closed, "marker_bias"));
  CHECK(below.Sample(row) == 2);
  CHECK(MarkerTaken(below) == 0);
  std::vector<std::string> open_stages, closed_stages;
  for (const auto& stage : open.behaviour()->stages)
    if (stage.name != "marker_bias")
      open_stages.push_back(stage.name + ":" + stage.reason);
  for (const auto& stage : closed.behaviour()->stages)
    closed_stages.push_back(stage.name + ":" + stage.reason);
  CHECK(open_stages == closed_stages);
  // Fix tau and move the raw row by one float ULP in either direction.
  b->marker_entropy_gate = entropy;
  auto less_certain = row, more_certain = row;
  less_certain[2] = std::nextafter(row[2], -INFINITY);
  more_certain[2] = std::nextafter(row[2], INFINITY);
  const auto high = s::SamplerState(Config(b, 0, 2)).Distribution(less_certain);
  const auto low = s::SamplerState(Config(b, 0, 2)).Distribution(more_certain);
  CHECK(Observation(low).entropy < entropy &&
        Observation(high).entropy > entropy);
  CHECK(!Stage(low, "marker_bias") && Stage(high, "marker_bias"));
  // Removing only marker_bias is the final-policy counterfactual.
  for (const auto& stage : open.behaviour()->stages)
    if (stage.name == "marker_bias")
      Same(stage.before, closed);
}

struct EnvironmentScope {
  std::vector<std::pair<std::string, std::optional<std::string>>> saved;
  static void Set(const char* name, const char* value) {
#ifdef _WIN32
    CHECK(_putenv_s(name, value ? value : "") == 0);
#else
    CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
#endif
  }
  EnvironmentScope() {
    for (const char* name :
         {"GUFO_THINK_EOG_GUARD", "GUFO_THINK_GUARD_ESCALATE",
          "GUFO_EOG_BOUNDARY", "GUFO_EOG_MIN_PROB_EXACT", "GUFO_EOG_MIN_PROB",
          "GUFO_EOG_DEFER_P", "GUFO_SPEC_EOG_STOP", "GUFO_LOOP_BREAKER",
          "GUFO_LOOP_BREAKER_EOG_CLOSE", "GUFO_LOOP_BREAKER_TRANSITION",
          "GUFO_REASONING_BUDGET", "GUFO_LOGIT_BIAS",
          "GUFO_MODEL_SUPPRESS_TOKENS", "GUFO_PRODUCTION_SAMPLING",
          "GUFO_THINK_MARKER_BIAS", "GUFO_THINK_MARKER_BIAS_BY_EFFORT",
          "GUFO_THINK_MARKER_ENTROPY_GATE"}) {
      const char* value = std::getenv(name);
      saved.emplace_back(
          name, value ? std::optional<std::string>(value) : std::nullopt);
      Set(name, nullptr);
    }
  }
  ~EnvironmentScope() {
    for (const auto& [name, value] : saved)
      Set(name.c_str(), value ? value->c_str() : nullptr);
  }
};

void EnvironmentValidation() {
  EnvironmentScope environment;
  const auto parse =
      [](std::shared_ptr<const s::BehaviourConfig> previous = {}) {
        return gufo::server::ParseBehaviourOptions(gufo::json::Value::object(),
                                                   previous);
      };
  CHECK(!parse());
  for (const char* value : {"0", "0.25", "12"}) {
    EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", value);
    CHECK(!parse());  // The gate alone must not enable behaviour or suspend
                      // grammar.
    EnvironmentScope::Set("GUFO_THINK_EOG_GUARD", "1");
    const auto b = parse();
    CHECK(b && b->enabled() && b->marker_entropy_gate == std::stod(value));
    EnvironmentScope::Set("GUFO_THINK_EOG_GUARD", nullptr);
  }
  for (const char* value :
       {"-1", "12.000001", "nan", "NaN", "inf", "not-a-number", "1x"}) {
    EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", value);
    bool rejected = false;
    try {
      (void)parse();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
  }
  EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", nullptr);
  auto previous = Behaviour();
  previous->marker_entropy_gate = 1;
  EnvironmentScope::Set("GUFO_THINK_MARKER_BIAS", "-2");
  CHECK(!parse(previous)->marker_entropy_gate);
  EnvironmentScope::Set("GUFO_THINK_MARKER_ENTROPY_GATE", "");
  CHECK(!parse()->marker_entropy_gate);
}

void GateOnlyConstrainedThinkingIdentity() {
  const auto answer = s::JsonConstraint::Compile(gufo::json::parse(R"({
    "type":"object","properties":{"x":{"type":"boolean"}},
    "required":["x"],"additionalProperties":false})"),
                                                 true);
  const auto grammar = s::JsonConstraint::WithReasoning(answer);
  const std::vector<std::string> pieces = {
      "thinking", "</thi", "nk>{\"x\":", "true}", "nk>INVALID", ""};
  auto binding = std::make_shared<s::TokenConstraint>();
  binding->grammar = grammar;
  binding->vocabulary = std::make_shared<s::ConstraintVocabulary>(
      static_cast<std::uint32_t>(pieces.size()), [&](auto id) {
        return s::ConstraintVocabulary::Piece{pieces[id], id == 5};
      });
  auto plain = Behaviour();
  plain->marker_bias = 0;
  const auto config = [&](std::shared_ptr<s::BehaviourConfig> b) {
    auto c = Config(std::move(b), 0, 0);
    c.constraint = binding;
    return c;
  };
  auto unset_config = config(plain);
  unset_config.behaviour.reset();
  s::SamplerState unset(unset_config);
  auto open = std::make_shared<s::BehaviourConfig>(*plain);
  open->marker_entropy_gate = 0;
  auto closed = std::make_shared<s::BehaviourConfig>(*plain);
  closed->marker_entropy_gate = 12;
  s::SamplerState tau0(config(open)), tau12(config(closed));
  CHECK(!plain->enabled() && !open->enabled() && !closed->enabled());
  CHECK(!unset.BehaviourEnabled() && !tau0.BehaviourEnabled() &&
        !tau12.BehaviourEnabled());
  auto expected_state = grammar->Start();
  const auto check_state = [&] {
    CHECK(unset.constraint_state() == expected_state);
    CHECK(tau0.constraint_state() == expected_state);
    CHECK(tau12.constraint_state() == expected_state);
    CHECK(*binding->Allowed(unset.constraint_state()) ==
          *binding->Allowed(tau0.constraint_state()));
    CHECK(*binding->Allowed(unset.constraint_state()) ==
          *binding->Allowed(tau12.constraint_state()));
  };
  check_state();
  CHECK(!unset.CanSelectArgmax(5) && !tau0.CanSelectArgmax(5) &&
        !tau12.CanSelectArgmax(5));
  // Pending validation must still reject a premature stop with a gate alone.
  for (const auto* source : {&unset, &tau0, &tau12}) {
    auto pending = *source;
    pending.DeferSample(5);
    bool rejected = false;
    try {
      (void)pending.Sample(std::vector<float>(pieces.size(), 0));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    CHECK(rejected && pending.rng_state() == source->rng_state());
    CHECK(pending.constraint_state() == source->constraint_state());
  }
  const auto ids = std::vector<s::TokenId>{5, 4, 3, 2, 1, 0};
  for (s::TokenId expected : {0u, 1u, 2u, 3u, 5u}) {
    std::vector<float> row(pieces.size(), -INFINITY);
    row[5] = 101;  // Premature stop outranks every legal continuation.
    row[expected] = expected == 5 ? 101 : 0;
    if (expected == 2)
      row[4] = 100;  // Invalid text after the split close.
    const auto native = unset.Distribution(row);
    Same(native, tau0.Distribution(row));
    Same(native, tau12.Distribution(row));
    CHECK(native.probability(expected) == 1);
    if (expected != 5)
      CHECK(native.probability(5) == 0);
    if (expected == 2)
      CHECK(native.probability(4) == 0);
    const auto values = Values(row, ids);
    Same(unset.Distribution(values, ids), tau0.Distribution(values, ids));
    Same(unset.Distribution(values, ids), tau12.Distribution(values, ids));
    CHECK(!unset.DistributionFromTop(values, ids, row.size()));
    CHECK(!tau0.DistributionFromTop(values, ids, row.size()));
    CHECK(!tau12.DistributionFromTop(values, ids, row.size()));
    CHECK(unset.Sample(row) == expected && tau0.Sample(row) == expected &&
          tau12.Sample(row) == expected);
    CHECK(unset.rng_state() == 42 && tau0.rng_state() == 42 &&
          tau12.rng_state() == 42);
    expected_state =
        binding->vocabulary->Accept(*grammar, expected_state, expected);
    unset.Accept(expected);
    tau0.Accept(expected);
    tau12.Accept(expected);
    check_state();
  }
  CHECK(grammar->Complete(expected_state));
  CHECK(unset.BehaviourSummary().empty() && tau0.BehaviourSummary().empty() &&
        tau12.BehaviourSummary().empty());
}

void ObservationCountersAndRollback() {
  auto b = Behaviour(false);
  b->marker_bias = 0;
  b->think_eog_guard = true;  // Independently enable behaviour at bias zero.
  b->marker_entropy_gate = .5;
  s::SamplerState sampler(Config(b, 0, 0));
  CHECK(!sampler.SaveDrawState().behaviour->think_markers);
  const auto emit = [&](s::TokenId token, const std::vector<float>& row) {
    CHECK(sampler.Sample(row) == token);
    sampler.Accept(token);
  };
  std::vector<float> point(16, -INFINITY);
  point[10] = 0;
  emit(10, point);  // Opening token was outside think.
  CHECK(Counts(sampler).think_tokens == 0);
  std::vector<float> confident(16, -INFINITY);
  confident[2] = 0;
  emit(2, confident);  // Marker, H=0, closed.
  const auto saved = sampler.SaveDrawState();
  auto tentative = sampler;
  confident[2] = -INFINITY;
  confident[5] = 0;
  CHECK(q::VerifyDraft(confident, 7, tentative, [](auto) { return false; }) ==
        q::DraftDecision::kReject);
  SameCounts(Counts(tentative), Counts(sampler));
  // A sampled but discarded row rolls back aggregates exactly like Taken.
  CHECK(sampler.Sample(confident) == 5);
  CHECK(Counts(sampler).think_tokens == 2);
  sampler.RestoreDrawState(saved);
  CHECK(Counts(sampler).think_tokens == 1);
  std::vector<float> uniform(16, 0);
  emit(0, uniform);  // Nonmarker, H=log(16)=2.77, bin 27, open.
  // Worker has already counted the draw; publish totals and replay once.
  auto worker = sampler;
  CHECK(worker.Sample(confident) == 5);
  worker.Accept(5);
  sampler.CopyDrawStateFrom(worker);
  CHECK(Counts(sampler).think_tokens == 3);
  sampler.Accept(5);
  SameCounts(Counts(sampler), Counts(worker));
  point.assign(16, -INFINITY);
  point[11] = 0;
  emit(11, point);
  point[11] = -INFINITY;
  point[12] = 0;
  emit(12, point);  // Both rows were thinking before Accept closed the span.
  point[12] = -INFINITY;
  point[2] = 0;
  emit(2, point);  // Answer marker is excluded.
  const auto counts = Counts(sampler);
  CHECK(counts.think_tokens == 5 && counts.markers == 2);
  CHECK(counts.gate_rows == 5 && counts.gate_open == 1);
  CHECK(counts.hist_all[0] == 4 && counts.hist_all[27] == 1);
  CHECK(counts.hist_marker[0] == 2);
  CHECK(std::accumulate(counts.hist_all.begin(), counts.hist_all.end(), 0ULL) ==
        5);
  CHECK(std::accumulate(counts.hist_marker.begin(), counts.hist_marker.end(),
                        0ULL) == 2);
  bool summary = false;
  for (const auto& line : sampler.BehaviourSummary())
    if (line ==
        "THINK_MARKERS tau=0.5 bias=0 think_tokens=5 markers=2 gate_rows=5 "
        "gate_open=1 "
        "hist_all=4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,"
        "0 "
        "hist_marker=2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0 scope=request")
      summary = true;
  CHECK(summary);
  auto clone = sampler;
  clone.Accept(10);
  std::vector<float> tail(32, 0);
  CHECK(clone.Sample(tail) == 0);
  clone.Accept(0);
  CHECK(Counts(clone).hist_all[30] == 1);
  SameCounts(Counts(sampler), counts);
  sampler.ResetHistory({});
  SameCounts(Counts(sampler), {});
  CHECK(!sampler.SaveDrawState().behaviour->think_markers);
}

void VerificationAndResidualRows() {
  auto b = Behaviour();
  b->marker_bias = 0;
  b->think_eog_guard = true;
  b->marker_entropy_gate = 0;
  const auto c = Config(b, 1, 2);
  std::vector<float> row(16, -INFINITY);
  row[2] = row[5] = 0;
  const std::vector<s::TokenId> ids = {5, 2, 7};
  const auto values = Values(row, ids);
  s::SamplerState ar(c), candidate(c), draft(c), mtp(c), residual(c);
  const auto token = ar.Sample(row);
  CHECK(Counts(ar).think_tokens == 1 && Counts(ar).hist_all[6] == 1);
  ar.Accept(token);
  CHECK(candidate.SampleFromTop(values, ids, row.size()) == token);
  candidate.Accept(token);
  CHECK(q::VerifyDraft(row, token, draft, [](auto) { return false; }) ==
        q::DraftDecision::kAccept);
  q::MtpProposal p;
  p.size = 2;
  p.ids[0] = 2;
  p.ids[1] = 5;
  p.probabilities[0] = p.probabilities[1] = .5F;
  p.token = 2;
  p.probability = .5F;
  const auto verified = q::VerifyMtpProposal(row, p, mtp);
  CHECK(verified.accepted);
  mtp.Accept(verified.token);
  // A rejected proposal counts the actual target residual on its worker.
  p.size = 1;
  p.ids[0] = p.token = 7;
  p.probabilities[0] = p.probability = 1;
  auto rejected_worker = residual;
  const auto rejected = q::VerifyMtpProposal(row, p, rejected_worker);
  CHECK(!rejected.accepted && Counts(rejected_worker).gate_rows == 1);
  CHECK(Counts(rejected_worker).markers == 1);
  CHECK(Counts(residual).gate_rows == 0);
  rejected_worker.Accept(rejected.token);
  SameCounts(Counts(ar), Counts(rejected_worker));
  const auto corrected = residual.SampleResidual(
      row, std::span(p.ids).first(1), std::span(p.probabilities).first(1));
  CHECK(Counts(residual).gate_rows == 1 && Counts(residual).markers == 1);
  residual.DeferSample(corrected);
  CHECK(residual.Sample(row) == corrected);
  CHECK(Counts(residual).gate_rows == 1);
  residual.Accept(corrected);
  for (const auto* state : {&candidate, &draft, &mtp, &residual})
    SameCounts(Counts(ar), Counts(*state));
  const auto raw_proposal = q::SampleMtpProposal(
      q::MtpCandidateLogits{{2, 5}, {0, 0}, 2}, ar, ar.mutable_rng_state());
  CHECK(raw_proposal.size == 2 && raw_proposal.probabilities[0] == .5F);
  // H metadata survives the inert-EOG and direct argmax opportunities at bias
  // 0.
  s::SamplerState greedy(Config(b, 0, 2));
  CHECK(!greedy.CanSelectArgmax(2));
  CHECK(Observation(greedy.Distribution(row)).entropy > 0);
  greedy.Accept(11);
  greedy.Accept(12);
  CHECK(greedy.CanSelectArgmax(2));
}

void StopCutCountersFollowTaken() {
  auto b = Behaviour();
  b->marker_entropy_gate = 0;
  s::SamplerState sampler(Config(b, 0, 2));
  const auto saved = sampler.SaveDrawState();
  auto worker = sampler;
  std::vector<float> row(16, -20);
  row[2] = 1;
  row[3] = 0;
  CHECK(q::VerifyDraft(row, 3, worker, [](auto token) { return token == 3; }) ==
        q::DraftDecision::kStop);
  CHECK(Counts(worker).think_tokens == 1 && Counts(worker).gate_rows == 1);
  CHECK(Counts(worker).hist_all[5] == 1 && Counts(worker).markers == 0);
  CHECK(worker.SaveDrawState().behaviour->accepted_count == 0);
  CHECK(MarkerTaken(worker) == 1 && MarkerTaken(sampler) == 0);
  CHECK(Counts(sampler).think_tokens == 0);  // Discarded worker has no effect.
  auto restored = worker;
  restored.RestoreDrawState(saved);
  CHECK(Counts(restored).think_tokens == 0 && MarkerTaken(restored) == 0);
  sampler.CopyDrawStateFrom(worker);
  SameCounts(Counts(sampler), Counts(worker));
  CHECK(MarkerTaken(sampler) == 1);  // Publish even though stop cuts this row.
  sampler.Accept(3);
  SameCounts(Counts(sampler), Counts(worker));  // Accept replay does not count.
}

void EntropyEndpointsAndForcedRows() {
  auto b = Behaviour();
  b->marker_entropy_gate = 12;
  // tau=12 is closed in production top_k=20, but not universally: the
  // specified full-row entropy can exceed 12 for vocabularies > exp(12).
  std::vector<float> uniform(163000, 0);
  const auto full = s::SamplerState(Config(b, 0, 0)).Distribution(uniform);
  CHECK(std::abs(Observation(full).entropy - std::log(163000.0)) < 1e-13);
  CHECK(Observation(full).gate_open && Stage(full, "marker_bias"));
  const auto top20 = s::SamplerState(Config(b, 0, 20)).Distribution(uniform);
  CHECK(!Observation(top20).gate_open && !Stage(top20, "marker_bias"));
  // A pre-existing mandatory force can rescue an empty raw finite row.
  b->marker_entropy_gate = 0;
  b->reasoning_budget_enabled = true;
  b->reasoning_budget = 0;
  b->budget_forced = {11, 12};
  s::SamplerState forced(Config(b, 0, 2));
  const std::vector<float> empty(16, -INFINITY);
  const auto distribution = forced.Distribution(empty);
  CHECK(Observation(distribution).entropy == 0);
  CHECK(Observation(distribution).gate_open &&
        !Stage(distribution, "marker_bias"));
  CHECK(forced.Sample(empty) == 11);
  forced.Accept(11);
  CHECK(Counts(forced).hist_all[0] == 1 && Counts(forced).gate_open == 1);
}
}  // namespace

int main() {
  try {
    const auto run = [](const char* name, auto test) {
      std::cout << "RUN " << name << std::endl;
      test();
    };
    run("UnsetAndCalibrationIdentity", UnsetAndCalibrationIdentity);
    run("ClosedIdentityAndRawEntropy", ClosedIdentityAndRawEntropy);
    run("FullCompactEntropyBits", FullCompactEntropyBits);
    run("ThresholdOnlyMarkerStage", ThresholdOnlyMarkerStage);
    run("EnvironmentValidation", EnvironmentValidation);
    run("GateOnlyConstrainedThinkingIdentity",
        GateOnlyConstrainedThinkingIdentity);
    run("ObservationCountersAndRollback", ObservationCountersAndRollback);
    run("VerificationAndResidualRows", VerificationAndResidualRows);
    run("StopCutCountersFollowTaken", StopCutCountersFollowTaken);
    run("EntropyEndpointsAndForcedRows", EntropyEndpointsAndForcedRows);
    std::cout << "think_marker_entropy_test: 10 groups PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
