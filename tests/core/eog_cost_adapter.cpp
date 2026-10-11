// Compiled twice, once with sampling=sampling_eog_old and the pinned B6.2
// headers. Both implementations link the actual sampler sources.
#include <bit>
#include <cstring>
#include <sstream>

#include "src/cli/serve/behaviour_options.hpp"
#include "tests/core/eog_cost_api.hpp"

namespace s = gufo::sampling;
namespace {
struct Bytes {
  std::ostringstream out;
  template<class T>
  void Put(T value) {
    const auto n = static_cast<std::uint64_t>(value);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  }
  void String(std::string_view text) {
    Put(text.size());
    out.write(text.data(), text.size());
  }
  template<class T>
  void Vector(const T& v) {
    Put(v.size());
    for (const auto& x : v)
      Put(x);
  }
  void Dist(const s::SamplingDistribution& d, bool meta = true) {
    Put(d.entries().size());
    for (auto e : d.entries()) {
      Put(e.token);
      Put(std::bit_cast<std::uint64_t>(e.value));
    }
    if (!meta)
      return;
    Put(bool(d.behaviour()));
    if (!d.behaviour())
      return;
    const auto& b = *d.behaviour();
    Put(b.defer_stop);
    Put(b.close_eog);
    Put(b.random);
    Put(b.stages.size());
    for (const auto& st : b.stages) {
      String(st.name);
      String(st.reason);
      Put(st.before_random);
      Put(st.after_random);
      Put(st.before_valid);
      Dist(st.before, false);
      Dist(st.after, false);
    }
  }
  void State(const s::SamplerState& sampler) {
    const auto d = sampler.SaveDrawState();
    Put(d.rng);
    Put(d.pending.has_value());
    if (d.pending)
      Put(*d.pending);
    Vector(sampler.history());
    Put(sampler.NeedsConstraintMask());
    Put(d.behaviour.has_value());
    if (!d.behaviour)
      return;
    const auto& b = *d.behaviour;
    Put(b.thinking);
    Put(b.prefix_used);
    Put(b.waiting_utf8);
    Put(b.close_decision);
    Put(b.guard_escalated);
    Put(b.guard_escalation_taken);
    Put(b.guard_blocked_attempts);
    Put(b.accepted_count);
    Put(b.remaining);
    Put(b.force_pos);
    Vector(b.deferred_at);
    Vector(b.close_at);
    Vector(b.guard_blocked_at);
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
    Put(l.cfg.think_max_period);
    Put(l.cfg.answer_max_period);
    Vector(l.tail);
    Vector(l.matching);
    Put(l.count);
    Put(l.current);
    Put(l.nudge_left);
    Put(l.watch_left);
    Put(l.nudges_in_span);
    Put(l.detections);
    Put(l.nudges);
    Put(l.forced_thinks);
    Put(l.stopped_answers);
    Put(l.pending_head);
    Put(l.pending_size);
    Put(l.terminal);
    for (const auto& ev : l.pending) {
      Put(ev.what);
      Put(ev.where);
      Put(ev.period);
      Put(ev.repeats);
    }
    Put(b.counters.size());
    for (const auto& [key, n] : b.counters) {
      String(key.first);
      String(key.second);
      Put(n.taken);
      Put(n.fallback);
    }
    const auto lines = sampler.BehaviourSummary();
    Put(lines.size());
    for (const auto& line : lines)
      String(line);
  }
};
s::SamplingConfig Config(const EogCostConfig& input) {
  // Environment is installed by the caller, once, before either factory.
  auto b = gufo::server::ParseBehaviourOptions(gufo::json::Value::object());
  if (!b)
    throw std::runtime_error("production environment not installed");
  b->think_eog_guard = input.flags & 1;
  b->eog_boundary = input.flags & 2;
  if (!(input.flags & 4))
    b->eog_min_prob = -1;
  if (!(input.flags & 8))
    b->eog_defer_p = 0;
  b->spec_eog_stop = input.flags & 16;
  if (!input.extras) {
    b->loop_breaker = b->loop_eog_close = false;
    b->marker_bias_by_effort.clear();
  }
  b->eog_ids = input.eog;
  b->think_start = {248068};
  b->think_end = {248069};
  b->budget_forced = b->think_end;
  static const auto pieces = [&] {
    auto result = std::make_shared<std::vector<std::string>>(input.vocab);
    (*result)[3] = "word";
    (*result)[4] = "done.";
    return result;
  }();
  b->token_pieces = pieces;
  s::SamplingConfig c;
  c.temperature = input.temperature;
  c.top_k = input.top_k;
  c.top_p = input.top_p;
  c.seed = 123;
  c.behaviour = b;
  if (input.permissive_tools) {
    static const auto constraint = [&] {
      const auto cached_pieces = pieces;
      auto result = std::make_shared<s::TokenConstraint>();
      result->grammar = s::JsonConstraint::WithTools(
          nullptr, {{"f", s::JsonConstraint::Object()}}, false, false,
          s::JsonConstraint::ToolFormat::kQwen);
      result->vocabulary = std::make_shared<s::ConstraintVocabulary>(
          input.vocab, [cached_pieces, eog = input.eog](std::uint32_t id) {
            return s::ConstraintVocabulary::Piece{
                (*cached_pieces)[id], std::ranges::find(eog, id) != eog.end()};
          });
      return result;
    }();
    c.constraint = constraint;
    b->loop_exempt = true;
  }
  return c;
}
class Adapter final : public EogCostSampler {
  s::SamplerState sampler_;
  s::SamplerState::DrawState base_;
  std::vector<float> pinned_, downloaded_;

public:
  explicit Adapter(const EogCostConfig& cfg)
      : sampler_(Config(cfg)), base_(sampler_.SaveDrawState()) {}
  void Reset(unsigned state, std::uint64_t seed) override {
    auto draw = base_;
    draw.rng = seed;
    if (draw.behaviour) {
      auto& b = *draw.behaviour;
      b.thinking = state == 0;
      b.boundary.accept_piece(state == 1 ? "word" : "done.");
      b.boundary.deferred_stop = state == 3;
      if (state == 4) {
        b.thinking = true;
        b.guard_escalated = true;
      }
      if (state == 5)
        b.Force({248069}, "reasoning_budget");
    }
    sampler_.ResetHistory({});
    sampler_.RestoreDrawState(std::move(draw));
  }
  std::uint32_t Row(std::span<const float> row, std::span<const float> values,
                    std::span<const std::uint32_t> ids,
                    bool candidates) override {
    std::optional<s::TokenId> token;
    if (candidates)
      token = sampler_.SampleFromTop(values, ids, row.size());
    if (!token && candidates) {
      // DownloadVerification: D2H into pinned storage, then into the retained
      // host row. Measure CPU copies of the exact size; no HIP delay is faked.
      pinned_.resize(row.size());
      downloaded_.resize(row.size());
      std::memcpy(pinned_.data(), row.data(), row.size_bytes());
      std::copy(pinned_.begin(), pinned_.end(), downloaded_.begin());
      asm volatile(""
                   :
                   : "r"(pinned_.data()), "r"(downloaded_.data())
                   : "memory");
      row = downloaded_;
    }
    if (!token)
      token = sampler_.Sample(row);
    return *token;
  }
  void Commit(std::uint32_t token) override { sampler_.Accept(token); }
  std::string Snapshot() const override {
    Bytes b;
    b.State(sampler_);
    return b.out.str();
  }
  std::string Distribution(std::span<const float> row) const override {
    Bytes b;
    b.Dist(sampler_.Distribution(row));
    return b.out.str();
  }
  std::vector<std::string> Counters() const override {
    return sampler_.BehaviourSummary();
  }
};
}  // namespace
#ifdef EOG_COST_OLD
std::unique_ptr<EogCostSampler> EogCostOld(const EogCostConfig& c) {
  return std::make_unique<Adapter>(c);
}
#else
std::unique_ptr<EogCostSampler> EogCostNew(const EogCostConfig& c) {
  return std::make_unique<Adapter>(c);
}
#endif
