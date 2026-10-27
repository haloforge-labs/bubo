// L-REL-6: SamplerState::RawLogprobs must return bit-identical values to the
// implementation it replaced (kept as RawLogprobsReference), and must not
// allocate a vocabulary-sized buffer per row. CPU only.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"

namespace {

// Largest single allocation seen while armed.
std::atomic<bool> g_armed{false};
std::atomic<std::size_t> g_largest{0};

}  // namespace

void* operator new(std::size_t size) {
  if (g_armed.load(std::memory_order_relaxed)) {
    auto seen = g_largest.load(std::memory_order_relaxed);
    while (size > seen &&
           !g_largest.compare_exchange_weak(seen, size,
                                            std::memory_order_relaxed)) {
    }
  }
  if (void* p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

using namespace gufo::sampling;

int g_checks = 0;

void Expect(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    std::exit(1);
  }
}

std::uint64_t Bits(double v) {
  std::uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

struct Outcome {
  bool threw{false};
  std::string what;
  TokenLogprob score;
};

template <typename Fn>
Outcome Run(Fn&& fn) {
  Outcome out;
  try {
    out.score = fn();
  } catch (const std::exception& e) {
    out.threw = true;
    out.what = e.what();
  }
  return out;
}

// Exact comparison: same exception text, or the same bits everywhere.
void ExpectSame(const SamplerState& state, std::span<const float> row,
                TokenId token, std::size_t top_n, const std::string& label) {
  const auto fresh = Run([&] { return state.RawLogprobs(row, token, top_n); });
  const auto old =
      Run([&] { return state.RawLogprobsReference(row, token, top_n); });
  const auto tag = label + " token=" + std::to_string(token) +
                   " top_n=" + std::to_string(top_n);
  Expect(fresh.threw == old.threw && fresh.what == old.what,
         tag + ": same exception");
  if (fresh.threw)
    return;
  Expect(fresh.score.token == old.score.token, tag + ": token");
  Expect(Bits(fresh.score.logprob) == Bits(old.score.logprob),
         tag + ": logprob bits");
  Expect(fresh.score.top_logprobs.size() == old.score.top_logprobs.size(),
         tag + ": top size");
  for (std::size_t i = 0; i < old.score.top_logprobs.size(); ++i) {
    Expect(fresh.score.top_logprobs[i].token == old.score.top_logprobs[i].token,
           tag + ": top token " + std::to_string(i));
    Expect(Bits(fresh.score.top_logprobs[i].value) ==
               Bits(old.score.top_logprobs[i].value),
           tag + ": top value bits " + std::to_string(i));
  }
}

std::vector<float> MakeRow(std::size_t vocab, std::uint32_t seed,
                           bool ties, bool specials) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-12.0F, 12.0F);
  std::vector<float> row(vocab);
  for (auto& v : row) {
    v = dist(rng);
    if (ties)
      v = std::round(v * 2.0F) / 2.0F;  // many equal values
  }
  if (specials) {
    row[1] = -std::numeric_limits<float>::infinity();
    row[3] = std::numeric_limits<float>::quiet_NaN();
    row[5] = std::numeric_limits<float>::infinity();
    row[vocab - 1] = -std::numeric_limits<float>::infinity();
  }
  return row;
}

void CheckAll(const SamplerState& state, std::span<const float> row,
              const std::string& label) {
  const std::array<TokenId, 6> tokens{
      0, 1, 3, 5, static_cast<TokenId>(row.size() / 2),
      static_cast<TokenId>(row.size() - 1)};
  for (const auto token : tokens)
    for (const std::size_t top_n : {0U, 1U, 2U, 5U, 20U})
      ExpectSame(state, row, token, top_n, label);
}

void TestPlainAndTies() {
  for (const bool ties : {false, true})
    for (const bool specials : {false, true}) {
      const auto row = MakeRow(257, 11, ties, specials);
      SamplerState state{SamplingConfig{}};
      CheckAll(state, row,
               std::string("plain ties=") + (ties ? "1" : "0") +
                   " specials=" + (specials ? "1" : "0"));
    }
  // Fewer finite entries than top_n.
  const std::array<float, 3> tiny{1.0F, 1.0F, 1.0F};
  SamplerState tiny_state{SamplingConfig{}};
  for (TokenId token = 0; token < 3; ++token)
    for (const std::size_t top_n : {0U, 1U, 3U, 20U})
      ExpectSame(tiny_state, tiny, token, top_n, "tiny all-equal");
}

void TestPenalties() {
  SamplingConfig config;
  config.repeat_penalty = 1.35F;
  config.frequency_penalty = 0.4F;
  config.presence_penalty = 0.2F;
  config.repeat_last_n = 40;
  SamplerState state(config);
  std::mt19937 rng(5);
  for (int i = 0; i < 300; ++i)
    state.Accept(static_cast<TokenId>(rng() % 257));
  for (const bool ties : {false, true}) {
    const auto row = MakeRow(257, 21, ties, true);
    CheckAll(state, row, "penalties");
  }
  // Penalties on the very first and last vocabulary entries.
  SamplerState edge(config);
  edge.Accept(TokenId{0});
  edge.Accept(TokenId{256});
  edge.Accept(TokenId{0});
  CheckAll(edge, MakeRow(257, 22, true, false), "penalties at the edges");
}

void TestBehaviour() {
  for (const bool production : {false, true}) {
    auto behaviour = std::make_shared<BehaviourConfig>();
    behaviour->think_start = {7};
    behaviour->think_end = {8};
    behaviour->eog_ids = {9, 10};
    behaviour->think_eog_guard = true;
    behaviour->logit_bias_enabled = true;
    behaviour->logit_bias = {{4, 2.5F}, {6, -1.0F}};
    behaviour->production_sampling = production;
    SamplingConfig config;
    config.behaviour = behaviour;
    config.repeat_penalty = 1.2F;
    config.frequency_penalty = 0.1F;
    SamplerState state(config);
    const auto row = MakeRow(257, 31, true, true);
    const std::string base = production ? "behaviour+production" : "behaviour";
    CheckAll(state, row, base + " outside think");
    state.Accept(TokenId{7});  // opens thinking: the eog guard now masks
    CheckAll(state, row, base + " in think");
    state.Accept(TokenId{3});
    state.Accept(TokenId{4});
    CheckAll(state, row, base + " in think after tokens");
    state.Accept(TokenId{8});
    CheckAll(state, row, base + " after think");
  }
  // A configured behaviour that is not enabled.
  auto inert = std::make_shared<BehaviourConfig>();
  SamplingConfig config;
  config.behaviour = inert;
  config.repeat_penalty = 1.1F;
  SamplerState state(config);
  state.Accept(TokenId{2});
  CheckAll(state, MakeRow(64, 41, false, true), "inert behaviour");
}

void TestConstraintMask() {
  const std::array<std::string, 5> pieces{"x", "{", " ", "[", "\""};
  auto constraint = std::make_shared<TokenConstraint>();
  constraint->grammar = JsonConstraint::Object();
  constraint->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t token) {
        return ConstraintVocabulary::Piece{pieces[token], false};
      });
  const std::array<float, 5> row{20, 1, 0, 8, 9};
  {
    SamplingConfig config;
    config.constraint = constraint;
    SamplerState state(config);
    for (TokenId token = 0; token < 5; ++token)
      for (const std::size_t top_n : {0U, 1U, 5U})
        ExpectSame(state, row, token, top_n, "constraint");
  }
  {
    auto behaviour = std::make_shared<BehaviourConfig>();
    behaviour->think_start = {3};
    behaviour->think_end = {4};
    behaviour->eog_ids = {2};
    behaviour->think_eog_guard = true;
    behaviour->logit_bias_enabled = true;
    behaviour->logit_bias = {{1, 2.0F}};
    SamplingConfig config;
    config.constraint = constraint;
    config.behaviour = behaviour;
    config.repeat_penalty = 1.3F;
    SamplerState state(config);
    for (TokenId token = 0; token < 5; ++token)
      for (const std::size_t top_n : {0U, 1U, 5U})
        ExpectSame(state, row, token, top_n, "constraint+behaviour");
    state.Accept(TokenId{3});  // thinking: the constraint is suspended
    for (TokenId token = 0; token < 5; ++token)
      for (const std::size_t top_n : {0U, 1U, 5U})
        ExpectSame(state, row, token, top_n, "constraint suspended in think");
  }
  {
    // Constraint vocabulary differs from the row.
    SamplingConfig config;
    config.constraint = constraint;
    SamplerState state(config);
    const std::array<float, 4> wrong{1, 2, 3, 4};
    ExpectSame(state, wrong, 1, 2, "constraint size mismatch");
  }
}

void TestDegenerateRows() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  SamplerState state{SamplingConfig{}};
  const std::array<float, 4> all_nan{nan, nan, nan, nan};
  const std::array<float, 4> all_inf{-inf, -inf, inf, -inf};
  ExpectSame(state, all_nan, 1, 2, "all NaN");
  ExpectSame(state, all_inf, 1, 2, "no finite value");
  const std::array<float, 4> one{nan, 2.0F, -inf, nan};
  for (TokenId token = 0; token < 4; ++token)
    ExpectSame(state, one, token, 3, "one finite value");
  // Invalid arguments keep the same errors.
  ExpectSame(state, one, 1, 21, "top_n above 20");
  ExpectSame(state, one, 9, 1, "token out of range");
  const std::array<float, 0> empty{};
  const auto rejected =
      Run([&] { return state.RawLogprobs(empty, 0, 1); });
  Expect(rejected.threw, "empty row is rejected");
}

void TestAllocationBound() {
  constexpr std::size_t kVocab = 248320;
  SamplingConfig config;
  config.repeat_penalty = 1.1F;
  config.frequency_penalty = 0.05F;
  config.repeat_last_n = 30000;
  SamplerState state(config);
  std::mt19937 rng(9);
  std::vector<TokenId> history(25000);
  for (auto& t : history)
    t = static_cast<TokenId>(rng() % kVocab);
  state.Accept(std::span<const TokenId>(history));
  const auto row = MakeRow(kVocab, 77, true, false);

  g_largest = 0;
  g_armed = true;
  const auto fresh = state.RawLogprobs(row, 12345, 20);
  g_armed = false;
  const auto fresh_largest = g_largest.load();

  g_largest = 0;
  g_armed = true;
  const auto old = state.RawLogprobsReference(row, 12345, 20);
  g_armed = false;
  const auto old_largest = g_largest.load();

  Expect(Bits(fresh.logprob) == Bits(old.logprob), "248K row: logprob bits");
  Expect(fresh.top_logprobs.size() == 20 && old.top_logprobs.size() == 20,
         "248K row: 20 entries");
  for (std::size_t i = 0; i < 20; ++i)
    Expect(fresh.top_logprobs[i].token == old.top_logprobs[i].token &&
               Bits(fresh.top_logprobs[i].value) ==
                   Bits(old.top_logprobs[i].value),
           "248K row: top entries");
  std::cout << "largest single allocation per row: new " << fresh_largest
            << " bytes, old " << old_largest << " bytes\n";
  Expect(fresh_largest <= (std::size_t{1} << 20),
         "new RawLogprobs allocates at most 1 MiB at a time");
  Expect(old_largest > (std::size_t{1} << 20),
         "the measurement sees the old vocabulary-sized buffer");
}

}  // namespace

int main() {
  TestPlainAndTies();
  TestPenalties();
  TestBehaviour();
  TestConstraintMask();
  TestDegenerateRows();
  TestAllocationBound();
  std::cout << "raw logprobs equivalence: " << g_checks << " checks passed\n";
  return 0;
}
