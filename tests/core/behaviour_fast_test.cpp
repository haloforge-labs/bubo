// Compiled separately against the exact a40d37f sampler and the candidate.
// The transcript contains every entry's double bits, stages, RNG and state.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "tests/fixtures/behaviour_fast_real.hpp"

namespace s = gufo::sampling;
namespace q = gufo::models::qwen38_flash_next;
namespace {
constexpr std::size_t kVocab = 248320;
struct Transcript {
  std::ofstream out;
  std::ofstream index;
  std::map<std::string, std::uint64_t> stages_seen, taken_max, fallback_max;
  unsigned fast_top{0}, residual{0}, accepted{0};
  std::array<unsigned, 3> draft_decisions{};
  explicit Transcript(const char* path) : out(path, std::ios::binary) {
    if (!out)
      throw std::runtime_error("cannot open transcript");
    index.open(std::string(path) + ".index.tsv");
    if (!index)
      throw std::runtime_error("cannot open transcript index");
  }
  void Mark(unsigned case_id, unsigned step) {
    index << out.tellp() << '\t' << case_id << '\t' << step << '\n';
  }
  template<typename T>
  void Put(T value) {
    const auto encoded = static_cast<std::uint64_t>(value);
    out.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
  }
  void Double(double value) { Put(std::bit_cast<std::uint64_t>(value)); }
  void String(std::string_view text) {
    Put(text.size());
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  template<typename T>
  void Vector(const T& values) {
    Put(values.size());
    for (const auto& value : values)
      Put(value);
  }
  void Distribution(const s::SamplingDistribution& d, bool metadata = true) {
    Put(d.entries().size());
    for (const auto& e : d.entries()) {
      Put(e.token);
      Double(e.value);
    }
    if (!metadata)
      return;
    Put(static_cast<bool>(d.behaviour()));
    if (!d.behaviour())
      return;
    const auto& m = *d.behaviour();
    Put(m.defer_stop);
    Put(m.close_eog);
    Put(m.random);
    Put(m.stages.size());
    for (const auto& stage : m.stages) {
      ++stages_seen[stage.name];
      String(stage.name);
      String(stage.reason);
      Put(stage.before_random);
      Put(stage.after_random);
      Put(stage.before_valid);
      Distribution(stage.before, false);
      Distribution(stage.after, false);
    }
  }
  void State(const s::SamplerState& sampler) {
    const auto draw = sampler.SaveDrawState();
    Put(draw.rng);
    Put(draw.pending.has_value());
    if (draw.pending)
      Put(*draw.pending);
    Vector(sampler.history());
    try {
      Put(sampler.NeedsConstraintMask());
      String("mask_valid");
    } catch (const std::runtime_error& error) {
      // Exhausted grammar support is an observable state, not a serializer
      // crash. Preserve the exact diagnostic in the old/new transcript.
      Put(false);
      String(error.what());
    }
    Put(draw.behaviour.has_value());
    if (!draw.behaviour)
      return;
    const auto& b = *draw.behaviour;
    Put(b.thinking);
    Put(b.prefix_used);
    Put(b.waiting_utf8);
    Put(b.close_decision);
    Put(b.accepted_count);
    Put(b.remaining);
    Put(b.force_pos);
    Vector(b.deferred_at);
    Vector(b.close_at);
    Vector(b.match_tail);
    Vector(b.forced);
    Vector(b.matched_end);
    String(b.force_reason);
    String(b.utf8_tail);
    const auto& e = b.boundary;
    String(e.tail);
    Put(e.has_text);
    Put(e.fence_open);
    Put(e.inline_code_open);
    Put(e.dollar_math_open);
    Put(e.display_math_open);
    Put(e.latex_math);
    Put(e.pending_backslash);
    Put(e.pending_backticks);
    Put(e.pending_dollars);
    Put(e.deferred_stop);
    Put(e.allowed());
    const auto& l = b.loop;
    Put(l.cfg.think_repeats);
    Put(l.cfg.think_tokens);
    Put(l.cfg.answer_repeats);
    Put(l.cfg.answer_tokens);
    Vector(l.tail);
    Vector(l.matching);
    Put(l.count);
    Put(l.current);
    Put(l.nudge_left);
    Put(l.watch_left);
    Put(l.detections);
    Put(l.nudges);
    Put(l.forced_thinks);
    Put(l.stopped_answers);
    Put(l.pending_head);
    Put(l.pending_size);
    Put(l.terminal);
    for (const auto& event : l.pending) {
      Put(event.what);
      Put(event.where);
      Put(event.period);
      Put(event.repeats);
    }
    Put(b.counters.size());
    for (const auto& [key, count] : b.counters) {
      taken_max[key.first] = std::max(taken_max[key.first], count.taken);
      fallback_max[key.first] =
          std::max(fallback_max[key.first], count.fallback);
      String(key.first);
      String(key.second);
      Put(count.taken);
      Put(count.fallback);
    }
    const auto summary = sampler.BehaviourSummary();
    Put(summary.size());
    for (const auto& line : summary)
      String(line);
  }
};

std::vector<float> RealRow(unsigned index) {
  const auto& recorded = behaviour_fast_fixture::rows.at(index);
  const auto floor = std::bit_cast<float>(recorded.back().bits);
  std::uint64_t seed = 0xa17c39 + index;
  std::vector<float> row(kVocab);
  for (auto& v : row)
    v = floor - 2.0F - static_cast<float>(s::Uniform(&seed) * 8);
  for (const auto& e : recorded)
    row[e.id] = std::bit_cast<float>(e.bits);
  return row;
}

std::shared_ptr<s::BehaviourConfig> Behaviour(std::size_t vocab, int effort,
                                              bool thinking) {
  auto b = std::make_shared<s::BehaviourConfig>();
  b->initial_thinking = thinking;
  b->think_eog_guard = true;
  b->eog_boundary = true;
  b->eog_min_prob = .2;
  b->eog_defer_p = .9;
  b->spec_eog_stop = true;
  b->loop_breaker = true;
  b->loop_eog_close = true;
  b->loop_transition = 2;
  const auto eog = static_cast<s::TokenId>(vocab > 256 ? vocab - 5 : 0);
  b->eog_ids = {eog, eog + 1, static_cast<s::TokenId>(vocab + 7), eog};
  b->think_start = {10};
  b->think_end = {11, 12};
  b->loop_prefix = {4, 5};
  b->marker_bias_by_effort = "low:-2,medium:-1,xhigh:0";
  b->marker_bias = effort == 0 ? -2 : effort == 1 ? -1 : 0;
  for (s::TokenId i = 0; i < 200; ++i)
    b->marker_ids.push_back(20 + i * (vocab > 256 ? 1231 : 1));
  b->marker_ids.push_back(static_cast<s::TokenId>(vocab + 3));
  auto pieces = std::make_shared<std::vector<std::string>>(vocab, "a");
  (*pieces)[eog] = "";
  (*pieces)[eog + 1] = "";
  (*pieces)[3] = ".";
  (*pieces)[4] = "<tool_";
  (*pieces)[5] = "call>";
  (*pieces)[6] = "{}";
  (*pieces)[7] = "\n";
  (*pieces)[8] = "<function=f>";
  (*pieces)[9] = "</function>";
  (*pieces)[10] = "<think>";
  (*pieces)[11] = "</";
  (*pieces)[12] = "think>";
  (*pieces)[13] = "</tool_call>";
  b->token_pieces = pieces;
  return b;
}

s::SamplingConfig Config(std::shared_ptr<s::BehaviourConfig> b,
                         float temperature, int top_k = 20) {
  s::SamplingConfig c;
  c.behaviour = std::move(b);
  c.temperature = temperature;
  c.top_k = top_k;
  c.top_p = .95F;
  c.seed = 119;
  return c;
}

void Grammar(s::SamplingConfig& c) {
  const auto pieces = c.behaviour->token_pieces;
  auto vocabulary = std::make_shared<s::ConstraintVocabulary>(
      pieces->size(), [pieces](std::uint32_t id) {
        return s::ConstraintVocabulary::Piece{(*pieces)[id], id < 2};
      });
  auto constraint = std::make_shared<s::TokenConstraint>();
  constraint->grammar = s::JsonConstraint::WithTools(
      nullptr, {{"f", s::JsonConstraint::Object()}}, false, false,
      s::JsonConstraint::ToolFormat::kQwen);
  constraint->vocabulary = vocabulary;
  c.constraint = constraint;
}

template<typename F>
void Attempt(Transcript& out, F operation) {
  // Both exception type and text are part of the equivalence contract.
  try {
    operation();
    out.String("OK");
  } catch (const std::invalid_argument& e) {
    out.String("invalid_argument");
    out.String(e.what());
  } catch (const std::runtime_error& e) {
    out.String("runtime_error");
    out.String(e.what());
  }
}

struct CandidateList {
  std::vector<s::TokenId> ids;
  std::vector<float> values;
};
CandidateList TopList(const std::vector<float>& row, bool short_list) {
  std::vector<s::TokenId> ids(row.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto keep = std::min<std::size_t>(short_list ? 21 : 64, row.size());
  std::partial_sort(
      ids.begin(), ids.begin() + keep, ids.end(), [&](auto a, auto z) {
        const float av = std::isfinite(row[a]) ? row[a] : -INFINITY;
        const float zv = std::isfinite(row[z]) ? row[z] : -INFINITY;
        return av == zv ? a < z : av > zv;
      });
  ids.resize(keep);
  std::vector<float> values;
  for (auto id : ids)
    values.push_back(row[id]);
  return {std::move(ids), std::move(values)};
}
std::optional<s::SamplingDistribution> TopTarget(Transcript& out,
                                                 s::SamplerState& sampler,
                                                 const std::vector<float>& row,
                                                 bool short_list) {
  const auto list = TopList(row, short_list);
  auto target = sampler.DistributionFromTop(list.values, list.ids, row.size());
  out.Put(target.has_value());
  if (target) {
    ++out.fast_top;
    out.Distribution(*target);
  }
  return target;
}
void Candidates(Transcript& out, s::SamplerState& sampler,
                const std::vector<float>& row, bool short_list) {
  auto [ids, values] = TopList(row, short_list);
  const auto saved = sampler.SaveDrawState();
  Attempt(out, [&] {
    auto d = sampler.DistributionFromTop(values, ids, row.size());
    out.Put(d.has_value());
    if (d) {
      ++out.fast_top;
      out.Distribution(*d);
    }
    auto token = sampler.SampleFromTop(values, ids, row.size());
    out.Put(token.has_value());
    if (token)
      out.Put(*token);
  });
  out.State(sampler);
  sampler.RestoreDrawState(saved);
  out.State(sampler);
  // Duplicate/unsorted compact IDs exercise the original local-index semantics.
  ids = {2, 0, 2, 1, 20, static_cast<s::TokenId>(row.size() + 3)};
  values = {2, 4, 1, -INFINITY, std::nanf(""), 0};
  Attempt(out, [&] { out.Distribution(sampler.Distribution(values, ids)); });
}

unsigned Run(Transcript& out, unsigned case_id, s::SamplingConfig c,
             std::vector<float> row, unsigned steps, unsigned mode) {
  out.Put(case_id);
  out.Put(steps);
  out.Put(mode);
  c.seed += case_id;
  const auto eog = c.behaviour->eog_ids.front();
  s::SamplerState sampler(c);
  if (!c.behaviour->initial_thinking)
    sampler.Accept(2);
  if (mode ==
      2) {  // Pending deferred stop: removing boundary can activate defer.
    auto draw = sampler.SaveDrawState();
    draw.behaviour->boundary.deferred_stop = true;
    sampler.RestoreDrawState(draw);
  }
  if (mode == 3) {  // Direct close redirection, then natural committed close.
    auto draw = sampler.SaveDrawState();
    draw.behaviour->prefix_used = true;
    sampler.RestoreDrawState(draw);
  }
  if (mode == 9) {
    sampler.Accept(11);
    sampler.Accept(12);
    sampler.Accept(4);
    sampler.Accept(5);
  }
  for (unsigned step = 0; step < steps; ++step) {
    out.Mark(case_id, step);
    out.Put(step);
    out.State(sampler);
    if (mode == 4 && step == 160) {
      sampler.Accept(11);
      sampler.Accept(12);
    }
    if (mode == 4 && step == 200)
      sampler.Accept(10);
    if (mode == 2 && step == 2)
      sampler.Accept(3);
    if (mode == 10 && step == 2) {
      sampler.Accept(11);
      sampler.Accept(12);
    }
    if (mode == 10 && step == 4) {
      sampler.Accept(4);
      sampler.Accept(5);
    }
    if (mode == 10 && step == 10)
      sampler.ResetHistory({});
    Attempt(out, [&] { out.Distribution(sampler.Distribution(row)); });
    if (step % 61 == 0 || (mode == 10 && step % 3 == 0)) {
      Attempt(out, [&] {
        const auto lp = sampler.RawLogprobs(row, 2, 5);
        out.Put(lp.token);
        out.Double(lp.logprob);
        out.Put(lp.top_logprobs.size());
        for (const auto& e : lp.top_logprobs) {
          out.Put(e.token);
          out.Double(e.value);
        }
      });
      Candidates(out, sampler, row, step % 2 == 0);
    }
    if (step % 17 == 0) {
      // Actual VerifyDraft copy/restore and stop, both rollback and acceptance.
      const auto main_before = sampler;
      auto tentative = sampler;
      Attempt(out, [&] {
        const auto decision =
            q::VerifyDraft(row, 2, tentative, [eog](std::int32_t t) {
              return static_cast<s::TokenId>(t) == eog;
            });
        out.Put(decision);
        ++out.draft_decisions[static_cast<unsigned>(decision)];
      });
      out.State(tentative);
      const auto draw = sampler.SaveDrawState();
      Attempt(out, [&] {
        q::MtpProposal proposal;
        proposal.size = 2;
        proposal.ids[0] = eog;
        proposal.ids[1] = 2;
        proposal.probabilities[0] = .5F;
        proposal.probabilities[1] = .5F;
        proposal.token = step % 34 == 0 ? eog : 2;
        proposal.probability = .5F;
        for (unsigned j = 0; j < 3; ++j) {
          auto from_top = j == 0 ? std::optional<s::SamplingDistribution>{}
                                 : TopTarget(out, sampler, row, j == 1);
          auto target =
              from_top ? std::move(*from_top) : sampler.Distribution(row);
          out.Distribution(target);
          const auto state = sampler.SaveDrawState();
          const double scale =
              j == 2 && state.behaviour && state.behaviour->thinking ? 1.5
                                                                     : 1.0;
          auto result = q::VerifyMtpProposal(target, row.size(), proposal,
                                             sampler, scale);
          if (result.accepted)
            ++out.accepted;
          else
            ++out.residual;
          out.Put(result.token);
          out.Put(result.accepted);
          out.State(sampler);
          sampler.Accept(result.token);
          out.State(sampler);
        }
      });
      sampler.RestoreDrawState(
          draw);  // Journals/counters and RNG restore together.
      out.State(sampler);
      sampler =
          main_before;  // Engine rollback also restores tentative history.
      out.State(sampler);
      Attempt(out, [&] {
        auto speculative = sampler;
        auto published = sampler;
        const auto token = speculative.Sample(row);
        speculative.DeferSample(token);
        published.CopyDrawStateFrom(speculative);
        out.State(published);
        const auto pending_draw = published.SaveDrawState();
        out.Put(published.Sample(row));
        out.State(published);
        published.RestoreDrawState(pending_draw);
        out.State(published);
      });
    }
    // Main sequence token-for-token draw + accept; caller controls transitions.
    Attempt(out, [&] {
      const auto token = sampler.Sample(row);
      out.Put(token);
      out.State(sampler);
      sampler.Accept(token);
      out.State(sampler);
    });
  }
  return steps;
}

int Dump(const char* path) {
  Transcript out(path);
  unsigned cases = 0, steps = 0;
  // Production effort map, temperatures, states on real-derived full rows.
  for (unsigned r = 0; r < 4; ++r)
    for (int effort = 0; effort < 4; ++effort)
      for (bool thinking : {false, true})
        for (float temperature : {0.0F, 1.0F}) {
          auto row = RealRow(r);
          auto b = Behaviour(row.size(), effort, thinking);
          if (r == 0 && effort == 1 && !thinking)
            b->eog_min_prob_exact = true;
          if (r == 1)
            row[b->eog_ids[0]] = row[behaviour_fast_fixture::rows[r][0].id] + 2;
          if (r == 2) {
            row[b->eog_ids[0]] = -INFINITY;
            row[b->eog_ids[1]] = std::nanf("");
          }
          if (r == 3) {
            b->marker_bias = 8;
            // A real-derived tail marker enters top-k after its boost.
            row[20] = row[behaviour_fast_fixture::rows[r][63].id] - .5F;
            const auto cut = row[behaviour_fast_fixture::rows[r][19].id];
            b->marker_bias = static_cast<double>(cut - row[20]) + 1.0;
          }
          const int top_k =
              r == 0 && effort == 3 && !thinking && temperature == 0 ? 0 : 20;
          steps += Run(out, cases++, Config(b, temperature, top_k), row, 12,
                       r == 1 ? 2 : 0);
        }
  // Small adversarial sequences, including 220+ steps across think -> answer.
  for (unsigned variant = 0; variant < 16; ++variant)
    for (int effort = 0; effort < 4; ++effort)
      for (float temperature : {0.0F, 1.0F}) {
        auto b = Behaviour(256, effort, variant % 2 == 0);
        std::vector<float> row(256, -20);
        row[0] = 9;
        row[1] = 8;
        row[2] = 10;
        row[20] = 11;
        auto c = Config(b, temperature, variant == 5 ? 0 : 20);
        unsigned mode = 0;
        switch (variant) {
          case 0:
            row[0] = -INFINITY;
            row[1] = -INFINITY;
            row[2] =
                40;  // Stable recurrence through nudge and forced transition.
            mode = 4;
            break;
          case 1:
            // A pending stop forces low-ranked EOG at the visible boundary;
            // its counterfactual instead picks token 2. Exercise taken > 0.
            row[2] = 40;
            mode = 2;
            break;
          case 2:
            mode = 3;
            row[0] = 40;
            break;
          case 3:
            b->eog_min_prob_exact = true;
            break;
          case 4:
            b->reasoning_budget_enabled = true;
            b->reasoning_budget = 3;
            break;
          case 5:
            c.top_p = 1;
            break;
          case 6:
            std::fill(row.begin(), row.end(), 1);
            break;
          case 7:
            row[0] = std::nanf("");
            row[1] = -INFINITY;
            row[9] = INFINITY;
            break;
          case 8:
            std::fill(row.begin(), row.end(), -INFINITY);
            break;
          case 9:
            Grammar(c);
            b->loop_exempt = true;
            mode = 9;
            break;
          case 10:
            b->logit_bias_enabled = true;
            b->logit_bias = {{0, 5}, {20, -3}, {999, 4}};
            break;
          case 11:
            b->model_suppress_tokens = true;
            b->suppress_ids = {0, 2, 999};
            break;
          case 12:
            b->production_sampling = true;
            b->native_temperature = .7F;
            b->native_top_k = 32;
            b->native_presence_penalty = 1.5F;
            c.repeat_penalty = 1.1F;
            c.frequency_penalty = .1F;
            break;
          case 13:
            b->eog_defer_p = 0;
            b->eog_boundary = false;
            b->eog_min_prob = .9;
            break;
          case 14:
            b->reasoning_budget_enabled = true;
            b->reasoning_budget = 0;
            std::fill(row.begin(), row.end(), -INFINITY);
            row[0] = 10;
            break;
          case 15:
            c.min_keep = 25;
            c.min_p = .2F;
            c.top_p = .7F;
            break;
        }
        steps += Run(out, cases++, c, row, variant == 0 ? 240 : 24, mode);
      }
  for (int effort = 0; effort < 4; ++effort)
    for (float temperature : {0.0F, 1.0F}) {
      auto b = Behaviour(256, effort, true);
      auto c = Config(b, temperature);
      b->loop_exempt = true;
      Grammar(c);
      std::vector<float> row(256, -10);
      row[2] = 10;
      row[6] = 9;
      if (effort == 3 && temperature == 0) {
        // Removing the last behaviour switch changes NeedsConstraintMask's
        // thinking exemption. Never lazify a guard's grammar evaluation.
        b->eog_boundary = false;
        b->eog_min_prob = b->eog_defer_p = 0;
        b->spec_eog_stop = b->loop_breaker = b->loop_eog_close = false;
        b->marker_bias = 0;
        b->marker_bias_by_effort.clear();
        row[0] = row[1] = -INFINITY;
      }
      steps += Run(out, cases++, c, row, 24, 10);
    }
  for (const auto name :
       {"think_eog_guard", "eog_boundary", "eog_min_prob", "eog_defer",
        "marker_bias", "loop_breaker", "loop_transition", "loop_eog_close",
        "reasoning_budget", "logit_bias", "model_suppress_tokens",
        "production_sampling"}) {
    if (out.stages_seen[name] == 0)
      throw std::runtime_error(std::string("stage fixture never exercised: ") +
                               name);
    std::cout << "COVERAGE name=" << name
              << " stage_rows=" << out.stages_seen[name]
              << " max_taken=" << out.taken_max[name]
              << " max_fallback=" << out.fallback_max[name] << '\n';
  }
  if (!out.fast_top || !out.residual || !out.accepted ||
      !out.taken_max["eog_defer"] || !out.fallback_max["candidate_path"] ||
      std::ranges::any_of(out.draft_decisions,
                          [](unsigned n) { return n == 0; }))
    throw std::runtime_error(
        "missing speculative fast/fallback/accept/residual/draft-decision "
        "coverage");
  std::cout << "SPEC fast_top=" << out.fast_top << " residual=" << out.residual
            << " accepted=" << out.accepted
            << " draft_accept=" << out.draft_decisions[0]
            << " draft_reject=" << out.draft_decisions[1]
            << " draft_stop=" << out.draft_decisions[2] << '\n';
  out.out.flush();
  out.index.flush();
  if (!out.out || !out.index)
    throw std::runtime_error("incomplete transcript write");
  std::cout << "DUMP cases=" << cases << " steps=" << steps << " PASS\n";
  return 0;
}

int Bench() {
  volatile std::uint64_t checksum = 0;
  for (unsigned mode = 0; mode < 7; ++mode) {
    auto row = RealRow(mode % 4);
    auto b = Behaviour(row.size(), mode == 1 ? 1 : 3, mode < 2);
    // Benchmark production-like valid list sizes; bounds/duplicates belong to
    // adversarial equivalence cases, not the measured speed gain.
    b->eog_ids.resize(2);
    b->marker_ids.pop_back();
    // Non-finite EOG covers zero probability without denominator
    // materialization.
    if (mode == 3) {
      row[b->eog_ids[0]] = -INFINITY;
      row[b->eog_ids[1]] = -INFINITY;
    }
    if (mode == 4) {
      b->production_sampling = true;
      b->native_temperature = .7F;
      b->native_top_k = 32;
    }
    if (mode == 5)
      b->eog_min_prob_exact = true;
    // Mode 6: greedy answer token with top_k 0 (a benchmark client's request);
    // the EOG denominator then covers the whole row.
    s::SamplerState sampler(mode == 6 ? Config(b, 0, 0) : Config(b, 1));
    if (mode >= 2)
      sampler.Accept(2);
    constexpr unsigned iterations = 48;
    for (unsigned i = 0; i < 4; ++i)
      checksum = checksum ^ sampler.Sample(row);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i)
      checksum = checksum ^ sampler.Sample(row);
    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    std::cout << "BENCH mode=" << mode << " ms_per_token=" << ms / iterations
              << " iterations=" << iterations << " checksum=" << checksum
              << '\n';
  }
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  try {
#ifdef GUFO_BEHAVIOUR_BASELINE_SHA
    std::cout << "ORACLE a40d37f sampling.cpp sha256="
              << GUFO_BEHAVIOUR_BASELINE_SHA << '\n';
#endif
    if (argc == 2 && std::string_view(argv[1]) == "--bench")
      return Bench();
    if (argc == 3 && std::string_view(argv[1]) == "--dump")
      return Dump(argv[2]);
    throw std::invalid_argument("use --dump PATH or --bench");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
