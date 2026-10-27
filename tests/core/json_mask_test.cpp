#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace gufo::sampling {
class ConstraintVocabularyTestAccess {
public:
  struct Profile {
    double advance_ms{0};
    double intern_ms{0};
    double state_ms{0};
    std::size_t advances{0};
    std::size_t direct_nodes{0};
    std::size_t cached_nodes{0};
    std::size_t states{0};
  };

  static JsonConstraint::State CanonicalState(
      const ConstraintVocabulary& vocab, const JsonConstraint& grammar,
      const JsonConstraint::State& state) {
    return grammar.CanonicalMaskState(state, vocab.max_token_bytes_);
  }

  static unsigned ThreadCount(const ConstraintVocabulary& vocab) {
    return vocab.mask_threads_;
  }

  // Frozen copy of ConstraintVocabulary::Allowed from 2af4fd45. Keep this
  // implementation independent of production changes to the trie walk.
  static std::vector<std::uint8_t> ReferenceAllowed(
      const ConstraintVocabulary& vocab, const JsonConstraint& grammar,
      const JsonConstraint::State& state, Profile* profile = nullptr) {
    constexpr std::uint32_t kLexeme = 1U << 30;
    constexpr std::size_t kMaxWork = 2000000;
    using Clock = std::chrono::steady_clock;
    auto record = [](double* total, Clock::time_point start) {
      if (total)
        *total +=
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    };
    if (std::ranges::any_of(state, [&](const auto& stack) {
          return !stack.symbols.empty() && (stack.symbols.back() & kLexeme) &&
                 grammar.lexemes_.at(stack.symbols.back() & ~kLexeme)
                     ->PlainReasoning(stack.lexeme);
        })) {
      std::vector<std::uint8_t> mask(vocab.pieces_.size(), 1);
      for (const auto token : vocab.reasoning_boundary_tokens_)
        mask[token] = vocab.Allows(grammar, state, token);
      return mask;
    }
    std::vector<std::uint8_t> mask(vocab.pieces_.size());
    if (grammar.Complete(state)) {
      for (std::size_t i = 0; i < vocab.pieces_.size(); ++i)
        mask[i] = vocab.pieces_[i].stop ||
                  (grammar.automatic_tools_ && vocab.pieces_[i].text.empty());
      if (grammar.stop_only_when_complete_)
        return mask;
    }
    struct CachedState {
      JsonConstraint::State state;
      std::array<std::uint32_t, 256> next;
      explicit CachedState(JsonConstraint::State s) : state(std::move(s)) {
        next.fill(UINT32_MAX);
      }
    };
    std::deque<CachedState> states;
    states.emplace_back(JsonConstraint::State{});
    states.emplace_back(state);
    std::map<JsonConstraint::State, std::uint32_t> intern{{{}, 0}, {state, 1}};
    std::size_t work = 0;
    auto direct = [&](auto&& self, std::uint32_t node,
                      const JsonConstraint::State& current) -> void {
      if (++work > kMaxWork)
        throw std::runtime_error("JSON token mask work limit exceeded");
      if (profile)
        ++profile->direct_nodes;
      for (auto token = vocab.trie_[node].token; token != UINT32_MAX;
           token = vocab.next_token_[token])
        mask[token] = 1;
      for (auto child = vocab.trie_[node].child; child != UINT32_MAX;
           child = vocab.trie_[child].sibling) {
        const auto start = profile ? Clock::now() : Clock::time_point{};
        auto next = grammar.Advance(current, vocab.trie_[child].byte);
        if (profile) {
          record(&profile->advance_ms, start);
          ++profile->advances;
        }
        if (!next.empty())
          self(self, child, next);
      }
    };
    auto walk = [&](auto&& self, std::uint32_t node,
                    std::uint32_t current) -> void {
      if (std::ranges::any_of(states[current].state, [&](const auto& stack) {
            return !stack.symbols.empty() && (stack.symbols.back() & kLexeme) &&
                   !grammar.lexemes_.at(stack.symbols.back() & ~kLexeme)
                        ->CacheTransitions();
          })) {
        direct(direct, node, states[current].state);
        return;
      }
      if (++work > kMaxWork)
        throw std::runtime_error("JSON token mask work limit exceeded");
      if (profile)
        ++profile->cached_nodes;
      for (auto token = vocab.trie_[node].token; token != UINT32_MAX;
           token = vocab.next_token_[token])
        mask[token] = 1;
      for (auto child = vocab.trie_[node].child; child != UINT32_MAX;
           child = vocab.trie_[child].sibling) {
        const auto byte = vocab.trie_[child].byte;
        auto& transition = states[current].next[byte];
        if (transition == UINT32_MAX) {
          const auto advance_start =
              profile ? Clock::now() : Clock::time_point{};
          auto next = grammar.Advance(states[current].state, byte);
          if (profile) {
            record(&profile->advance_ms, advance_start);
            ++profile->advances;
          }
          const auto intern_start =
              profile ? Clock::now() : Clock::time_point{};
          if (states.size() >= 8192 && !intern.contains(next)) {
            if (profile)
              record(&profile->intern_ms, intern_start);
            if (!next.empty())
              direct(direct, child, next);
            continue;
          }
          auto [found, inserted] = intern.emplace(next, states.size());
          if (profile)
            record(&profile->intern_ms, intern_start);
          transition = found->second;
          if (inserted) {
            const auto state_start =
                profile ? Clock::now() : Clock::time_point{};
            states.emplace_back(std::move(next));
            if (profile)
              record(&profile->state_ms, state_start);
          }
        }
        if (transition != 0)
          self(self, child, transition);
      }
    };
    walk(walk, 0, 1);
    if (profile)
      profile->states = states.size();
    if (std::ranges::none_of(mask, [](auto value) { return value != 0; }))
      throw std::runtime_error("JSON constraint has no valid token");
    return mask;
  }

  static void PrintRootStats(const ConstraintVocabulary& vocab,
                             const JsonConstraint& grammar,
                             const JsonConstraint::State& state) {
    const auto mask = ReferenceAllowed(vocab, grammar, state);
    std::array<std::size_t, 256> allowed{};
    for (std::size_t id = 0; id < mask.size(); ++id)
      if (mask[id] && !vocab.pieces_[id].text.empty())
        ++allowed[static_cast<unsigned char>(vocab.pieces_[id].text[0])];
    for (auto root = vocab.trie_[0].child; root != UINT32_MAX;
         root = vocab.trie_[root].sibling) {
      std::size_t nodes = 0;
      std::vector<std::uint32_t> pending{root};
      while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        ++nodes;
        for (auto child = vocab.trie_[node].child; child != UINT32_MAX;
             child = vocab.trie_[child].sibling)
          pending.push_back(child);
      }
      const auto byte = vocab.trie_[root].byte;
      if (allowed[byte])
        std::cout << "root=" << static_cast<unsigned>(byte)
                  << " nodes=" << nodes << " allowed=" << allowed[byte] << '\n';
    }
  }
};
}  // namespace gufo::sampling

using gufo::json::parse;
using gufo::sampling::ConstraintVocabulary;
using gufo::sampling::ConstraintVocabularyTestAccess;
using gufo::sampling::JsonConstraint;
using gufo::sampling::TokenConstraint;

namespace {
using Clock = std::chrono::steady_clock;
volatile std::size_t observed_mask = 0;

void SetThreads(int count) {
  const auto text = std::to_string(count);
#ifdef _WIN32
  if (_putenv_s("GUFO_MASK_THREADS", text.c_str()))
    throw std::runtime_error("cannot set GUFO_MASK_THREADS");
#else
  if (setenv("GUFO_MASK_THREADS", text.c_str(), 1))
    throw std::runtime_error("cannot set GUFO_MASK_THREADS");
#endif
}

std::string Alpha(std::uint32_t value) {
  std::string result(5, 'a');
  for (char& byte : result) {
    byte = static_cast<char>('a' + value % 26);
    value /= 26;
  }
  return result;
}

std::vector<ConstraintVocabulary::Piece> MakePieces(std::size_t size) {
  std::vector<ConstraintVocabulary::Piece> pieces;
  pieces.reserve(size);
  for (unsigned byte = 0; byte < 128; ++byte)
    pieces.push_back({std::string(1, static_cast<char>(byte)), false});
  pieces.push_back({"", true});
  constexpr std::array<std::string_view, 8> prefixes{
      " ", " word", "query", "\"", "<", "{", "\n", "x"};
  for (std::uint32_t id = 129; id < size; ++id)
    pieces.push_back(
        {std::string(prefixes[id % prefixes.size()]) + Alpha(id), false});
  return pieces;
}

constexpr std::array<int, 3> kThreadCounts{0, 1, 8};
using VocabularyModes = std::array<std::unique_ptr<ConstraintVocabulary>, 3>;

template<typename Reader>
VocabularyModes MakeModes(std::uint32_t size, Reader reader) {
  VocabularyModes modes;
  for (std::size_t i = 0; i < modes.size(); ++i) {
    SetThreads(kThreadCounts[i]);
    modes[i] = std::make_unique<ConstraintVocabulary>(size, reader);
    if (ConstraintVocabularyTestAccess::ThreadCount(*modes[i]) !=
        static_cast<unsigned>(kThreadCounts[i]))
      throw std::runtime_error("mask thread environment setting ignored");
  }
  return modes;
}

template<typename F>
double Milliseconds(F&& operation, int repeats) {
  std::vector<double> samples;
  samples.reserve(repeats);
  (void)operation();
  for (int i = 0; i < repeats; ++i) {
    const auto start = Clock::now();
    auto mask = operation();
    observed_mask = observed_mask + mask.size() + mask.front();
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count());
  }
  std::ranges::sort(samples);
  return samples[samples.size() / 2];
}

struct Checkpoint {
  std::string name;
  JsonConstraint::State state;
};

std::vector<Checkpoint> ScriptedStates(const JsonConstraint& grammar) {
  const std::string script =
      R"(<tool_call>{"name":"search","arguments":{"query":"orbital mechanics","limit":3}}</tool_call>)";
  auto state = grammar.Start();
  std::vector<Checkpoint> states{{"start", state}};
  for (std::size_t i = 0; i < script.size(); ++i) {
    state = grammar.Advance(state, static_cast<unsigned char>(script[i]));
    if (state.empty())
      throw std::runtime_error("scripted tool call rejected at byte " +
                               std::to_string(i));
    std::string label = "step-" + std::to_string(i + 1);
    const auto prefix = script.substr(0, i + 1);
    if (prefix == "<tool_call>")
      label = "tool-open";
    if (prefix == R"(<tool_call>{"name":")")
      label = "name";
    if (prefix == R"(<tool_call>{"name":"search","arguments":{)")
      label = "arguments";
    if (prefix == R"(<tool_call>{"name":"search","arguments":{"query":")")
      label = "string";
    if (prefix ==
        R"(<tool_call>{"name":"search","arguments":{"query":"orbital)")
      label = "string-body";
    if (prefix ==
        R"(<tool_call>{"name":"search","arguments":{"query":"orbital mechanics","limit":)")
      label = "number";
    if (prefix ==
        R"(<tool_call>{"name":"search","arguments":{"query":"orbital mechanics","limit":3)")
      label = "number-body";
    if (i + 1 == script.size())
      label = "complete";
    states.push_back({std::move(label), state});
  }
  if (!grammar.Complete(state))
    throw std::runtime_error("scripted tool call did not complete");
  return states;
}

void PrintProfile(const ConstraintVocabulary& vocab,
                  const JsonConstraint& grammar,
                  const std::vector<Checkpoint>& states) {
  std::cout << "class original_ms advance_ms intern_ms state_ms "
               "advances direct_nodes cached_nodes states\n";
  for (const auto& point : states) {
    if (point.name.starts_with("step-"))
      continue;
    const auto state = ConstraintVocabularyTestAccess::CanonicalState(
        vocab, grammar, point.state);
    const auto original = Milliseconds(
        [&] {
          return ConstraintVocabularyTestAccess::ReferenceAllowed(
              vocab, grammar, state);
        },
        3);
    ConstraintVocabularyTestAccess::Profile profile;
    (void)ConstraintVocabularyTestAccess::ReferenceAllowed(vocab, grammar,
                                                           state, &profile);
    std::cout << point.name << ' ' << std::fixed << std::setprecision(3)
              << original << ' ' << profile.advance_ms << ' '
              << profile.intern_ms << ' ' << profile.state_ms << ' '
              << profile.advances << ' ' << profile.direct_nodes << ' '
              << profile.cached_nodes << ' ' << profile.states << '\n';
  }
}

void CheckAll(const VocabularyModes& modes, const JsonConstraint& grammar,
              const std::vector<Checkpoint>& states) {
  const auto& vocab = *modes[0];
  std::mt19937 engine(0x5a3f11);
  std::uniform_int_distribution<std::uint32_t> ids(0, vocab.size() - 1);
  std::size_t checked = 0;
  for (const auto& point : states) {
    for (const auto& state :
         {point.state, ConstraintVocabularyTestAccess::CanonicalState(
                           vocab, grammar, point.state)}) {
      const auto original = ConstraintVocabularyTestAccess::ReferenceAllowed(
          vocab, grammar, state);
      for (std::size_t i = 0; i < modes.size(); ++i) {
        const auto candidate = modes[i]->Allowed(grammar, state);
        if (original != candidate)
          throw std::runtime_error("mask mismatch: " + point.name +
                                   " N=" + std::to_string(kThreadCounts[i]));
      }
      for (int sample = 0; sample < 5000; ++sample) {
        const auto id = ids(engine);
        if (bool(original[id]) != vocab.Allows(grammar, state, id))
          throw std::runtime_error("scalar mismatch: " + point.name +
                                   " token=" + std::to_string(id));
        ++checked;
      }
    }
  }
  std::cout << "equality states=" << states.size()
            << " forms=raw,canonical threads=0,1,8 scalar_samples=" << checked
            << " PASS\n";
}

void CheckSmall(const JsonConstraint& grammar,
                const std::vector<Checkpoint>& states) {
  auto pieces = MakePieces(129);
  pieces.push_back({"", false});
  const auto modes =
      MakeModes(pieces.size(), [&](std::uint32_t id) { return pieces[id]; });
  const auto& vocab = *modes[0];
  for (const auto& point :
       {states.front(), states[states.size() / 2], states.back()}) {
    const auto original = ConstraintVocabularyTestAccess::ReferenceAllowed(
        vocab, grammar, point.state);
    for (const auto& mode : modes) {
      if (original != mode->Allowed(grammar, point.state))
        throw std::runtime_error("small mask mismatch");
    }
    for (std::uint32_t id = 0; id < vocab.size(); ++id)
      if (bool(original[id]) != vocab.Allows(grammar, point.state, id))
        throw std::runtime_error("small scalar mismatch");
  }
  std::cout << "small states=3 all_tokens=130 PASS\n";
}

void CheckNoValid() {
  const auto grammar = JsonConstraint::Compile(
      parse(
          R"({"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false})"),
      true);
  const auto modes = MakeModes(5000, [](std::uint32_t id) {
    return ConstraintVocabulary::Piece{"{x" + Alpha(id), false};
  });
  for (const auto& mode : modes) {
    std::string old_error, new_error;
    try {
      (void)ConstraintVocabularyTestAccess::ReferenceAllowed(*mode, *grammar,
                                                             grammar->Start());
    } catch (const std::runtime_error& error) {
      old_error = error.what();
    }
    try {
      (void)mode->Allowed(*grammar, grammar->Start());
    } catch (const std::runtime_error& error) {
      new_error = error.what();
    }
    if (old_error != "JSON constraint has no valid token" ||
        new_error != old_error)
      throw std::runtime_error("no valid token exception mismatch");
  }
  std::cout << "no-valid-token N=0,1,8 PASS\n";
}

void CheckWorkerException() {
  const auto grammar = JsonConstraint::Object();
  const auto modes = MakeModes(5000, [](std::uint32_t id) {
    return ConstraintVocabulary::Piece{"x{" + Alpha(id), false};
  });
  // The first byte is valid; expanding the second reveals an invalid rule.
  // In parallel mode that exception must cross the worker future.
  const JsonConstraint::State state{
      {{999999U, 0x80000000U | '{', 0x80000000U | 'x'}, ""}};
  for (const auto& mode : modes) {
    bool thrown = false;
    try {
      (void)mode->Allowed(*grammar, state);
    } catch (const std::out_of_range&) {
      thrown = true;
    }
    if (!thrown)
      throw std::runtime_error("worker exception not propagated");
  }
  std::cout << "worker-exception N=0,1,8 PASS\n";
}

// TokenConstraint keeps at most kMaskCacheEntries finished masks, evicts the
// least recently used one, never evicts the initial state, and returns the
// same bytes as an uncached ConstraintVocabulary::Allowed.
void CheckMaskCachePolicy(const std::shared_ptr<const JsonConstraint>& grammar,
                          const std::vector<Checkpoint>& states) {
  constexpr std::size_t kCap = TokenConstraint::kMaskCacheEntries;
  auto pieces = MakePieces(1024);
  auto vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t id) { return pieces[id]; });
  // Distinct canonical states in script order; [0] is the initial state.
  std::vector<JsonConstraint::State> distinct;
  for (const auto& point : states) {
    auto canonical = ConstraintVocabularyTestAccess::CanonicalState(
        *vocabulary, *grammar, point.state);
    if (std::ranges::find(distinct, canonical) == distinct.end())
      distinct.push_back(std::move(canonical));
  }
  if (distinct.size() < kCap + 4)
    throw std::runtime_error("mask cache test needs more distinct states: " +
                             std::to_string(distinct.size()));
  const auto same = [](const auto& a, const auto& b) { return a.get() == b.get(); };
  TokenConstraint binding;
  binding.grammar = grammar;
  binding.vocabulary = vocabulary;
  const auto initial = binding.Allowed(distinct[0]);
  const auto recent = binding.Allowed(distinct[1]);
  const auto oldest = binding.Allowed(distinct[2]);
  for (std::size_t i = 3; i < kCap; ++i)
    (void)binding.Allowed(distinct[i]);
  if (binding.CachedMaskCount() != kCap)
    throw std::runtime_error("mask cache not filled to its cap");
  // Touch distinct[1]: distinct[2] is now the least recently used non-initial.
  if (!same(binding.Allowed(distinct[1]), recent))
    throw std::runtime_error("mask cache hit returned a new mask");
  (void)binding.Allowed(distinct[kCap]);
  if (binding.CachedMaskCount() != kCap)
    throw std::runtime_error("mask cache grew past its cap");
  if (!same(binding.Allowed(distinct[0]), initial) ||
      !same(binding.Allowed(distinct[1]), recent))
    throw std::runtime_error("mask cache evicted a recent or initial entry");
  const auto rebuilt = binding.Allowed(distinct[2]);
  if (same(rebuilt, oldest) || *rebuilt != *oldest)
    throw std::runtime_error("least recently used entry was not evicted");
  // Flood past the cap; the initial mask stays and every mask stays exact.
  for (int round = 0; round < 2; ++round)
    for (std::size_t i = 1; i < distinct.size(); ++i) {
      const auto mask = binding.Allowed(distinct[i]);
      if (binding.CachedMaskCount() > kCap)
        throw std::runtime_error("mask cache exceeded its cap");
      if (*mask != vocabulary->Allowed(*grammar, distinct[i]))
        throw std::runtime_error("cached mask differs from a fresh mask");
    }
  if (!same(binding.Allowed(distinct[0]), initial))
    throw std::runtime_error("initial mask was evicted");
  std::cout << "mask-cache-policy states=" << distinct.size() << " cap=" << kCap
            << " PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
  const auto schema = parse(R"({
    "type":"object","properties":{
      "query":{"type":"string","minLength":1,"maxLength":80},
      "limit":{"type":"integer","minimum":1,"maximum":10}},
    "required":["query","limit"],"additionalProperties":false})");
  const auto parameters = JsonConstraint::ToolParameters(
      schema, true, JsonConstraint::ToolFormat::kJson);
  const auto grammar =
      JsonConstraint::WithTools(nullptr, {{"search", parameters}}, true, false,
                                JsonConstraint::ToolFormat::kJson);
  const auto states = ScriptedStates(*grammar);
  auto pieces = MakePieces(151936);
  const auto modes =
      MakeModes(pieces.size(), [&](std::uint32_t id) { return pieces[id]; });
  const auto& vocab = *modes[0];
  if (argc > 1 && std::string_view(argv[1]) == "--baseline") {
    PrintProfile(vocab, *grammar, states);
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--serial") {
    std::cout << "class original_ms serial_ms\n";
    for (const auto& point : states) {
      if (point.name.starts_with("step-"))
        continue;
      const auto state = ConstraintVocabularyTestAccess::CanonicalState(
          vocab, *grammar, point.state);
      const auto original = Milliseconds(
          [&] {
            return ConstraintVocabularyTestAccess::ReferenceAllowed(
                vocab, *grammar, state);
          },
          3);
      const auto serial =
          Milliseconds([&] { return vocab.Allowed(*grammar, state); }, 3);
      std::cout << point.name << ' ' << std::fixed << std::setprecision(3)
                << original << ' ' << serial << '\n';
    }
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--roots") {
    for (const auto& point : states) {
      if (point.name == "string" || point.name == "complete") {
        std::cout << "class=" << point.name << '\n';
        ConstraintVocabularyTestAccess::PrintRootStats(vocab, *grammar,
                                                       point.state);
      }
    }
    return 0;
  }
  CheckMaskCachePolicy(grammar, states);
  CheckAll(modes, *grammar, states);
  CheckSmall(*grammar, states);
  CheckNoValid();
  CheckWorkerException();
  std::cout << "class original_ms serial_ms threads1_ms threads8_ms\n";
  for (const auto& point : states) {
    if (point.name.starts_with("step-"))
      continue;
    const auto state = ConstraintVocabularyTestAccess::CanonicalState(
        vocab, *grammar, point.state);
    const auto original = Milliseconds(
        [&] {
          return ConstraintVocabularyTestAccess::ReferenceAllowed(
              vocab, *grammar, state);
        },
        9);
    const auto timed = [&](std::size_t index) {
      return Milliseconds(
          [&] { return modes[index]->Allowed(*grammar, state); }, 9);
    };
    std::cout << point.name << ' ' << std::fixed << std::setprecision(3)
              << original << ' ' << timed(0) << ' ' << timed(1) << ' '
              << timed(2) << '\n';
  }
  std::cout << "json_mask_test PASS\n";
}
