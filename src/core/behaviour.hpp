#ifndef GUFO_CORE_BEHAVIOUR_HPP_
#define GUFO_CORE_BEHAVIOUR_HPP_
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace gufo::sampling {
// Value-state ports of production common/eog-boundary.h:16 and
// common/loop-breaker.h:13. No llama context/model/GPU is required.
struct EogBoundaryState {
    std::string tail;
    bool has_text = false;
    bool fence_open = false;
    bool inline_code_open = false;
    bool dollar_math_open = false;
    bool display_math_open = false;
    int latex_math = 0; // 1 = \\( ... \\), 2 = \\[ ... \\]
    bool pending_backslash = false;
    int pending_backticks = 0;
    int pending_dollars = 0;
    // LLAMA_EOG_DEFER_P: a confident EOG was masked at a non-boundary; end at the next allowed boundary
    bool deferred_stop = false;

    void reset() { *this = {}; }

    void flush_backticks() {
        if (pending_backticks >= 3) fence_open = !fence_open;
        else if (pending_backticks > 0 && !fence_open) inline_code_open = !inline_code_open;
        pending_backticks = 0;
    }

    void flush_dollars() {
        if (!fence_open && !inline_code_open) {
            if (pending_dollars >= 2) display_math_open = !display_math_open;
            else if (pending_dollars == 1) dollar_math_open = !dollar_math_open;
        }
        pending_dollars = 0;
    }

    void accept_piece(std::string_view piece) {
        for (const unsigned char c : piece) {
            if (c != '`') flush_backticks();
            if (c != '$') flush_dollars();
            if (c == '`') {
                ++pending_backticks;
                pending_backslash = false;
                continue;
            }
            if (c == '$' && !pending_backslash && !fence_open && !inline_code_open) {
                ++pending_dollars;
                pending_backslash = false;
                continue;
            }
            if (pending_backslash && !fence_open && !inline_code_open) {
                if (c == '(') latex_math = 1;
                else if (c == '[') latex_math = 2;
                else if ((c == ')' && latex_math == 1) || (c == ']' && latex_math == 2)) latex_math = 0;
            }
            pending_backslash = c == '\\';
        }
        if (!piece.empty()) {
            has_text = true;
            tail.append(piece.data(), piece.size());
            if (tail.size() > 128) tail.erase(0, tail.size() - 128);
        }
    }

    static bool last_codepoint(std::string_view text, uint32_t & cp, size_t & start) {
        if (text.empty()) return false;
        start = text.size() - 1;
        const unsigned char last = (unsigned char) text[start];
        if (last < 0x80) { cp = last; return true; }
        while (start > 0 && ((unsigned char) text[start] & 0xC0) == 0x80) --start;
        const unsigned char lead = (unsigned char) text[start];
        const size_t length = lead >= 0xF0 && lead <= 0xF4 ? 4 :
                              lead >= 0xE0 && lead <= 0xEF ? 3 :
                              lead >= 0xC2 && lead <= 0xDF ? 2 : 0;
        if (length == 0 || text.size() - start != length) return false;
        cp = lead & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
        for (size_t i = start + 1; i < text.size(); ++i) {
            const unsigned char b = (unsigned char) text[i];
            if ((b & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (b & 0x3F);
        }
        return true;
    }

    bool allowed() const {
        if (!has_text) return true;
        auto state = *this;
        state.flush_backticks();
        state.flush_dollars();
        if (state.fence_open || state.inline_code_open || state.dollar_math_open ||
            state.display_math_open || state.latex_math != 0) return false;

        std::string_view text(tail);
        while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
        if (text.empty()) return false;

        uint32_t cp = 0;
        size_t start = 0;
        if (!last_codepoint(text, cp, start)) return false; // incomplete multi-token UTF-8
        if (cp == '\n' || cp == '\r') return true;
        if (cp == '>') {
            const size_t open = text.substr(0, start).rfind('<');
            if (open == std::string_view::npos || open + 1 == start) return false;
            const unsigned char first = (unsigned char) text[open + 1];
            return std::isalpha(first) || first == '/' || first == '!' || first == '?';
        }
        if (cp == '.' || cp == '!' || cp == '?' || cp == ':' || cp == ';' ||
            cp == ')' || cp == ']' || cp == '}' || cp == '`' || cp == '$') return true;
        // Common non-ASCII sentence punctuation and closing quotes/brackets.
        if (cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF1A || cp == 0xFF1B ||
            cp == 0x2019 || cp == 0x201D || cp == 0x00BB || cp == 0x203A ||
            cp == 0xFF09 || cp == 0x3011 || cp == 0x300B || cp == 0x300D || cp == 0x300F) return true;
        if (cp == '\'' || cp == '"' || cp == '*' || cp == '_' || cp == '~') {
            // Opening quotes and emphasis after whitespace are not endings.
            if (start == 0) return false;
            const unsigned char prev = (unsigned char) text[start - 1];
            return !std::isspace(prev) && prev != '(' && prev != '[' && prev != '{' && prev != '<';
        }
        return false; // includes commas and ordinary word characters
    }
};


struct LoopBreaker {
    enum channel { ANSWER, THINK };
    enum action { NONE, NUDGE, FORCE_THINK, STOP_ANSWER };
    static constexpr int kMaxPeriod = 128;

    struct config {
        int think_repeats = 16;
        int think_tokens = 128;
        int answer_repeats = 32;
        int answer_tokens = 512;
        int think_max_period = kMaxPeriod;
        int answer_max_period = 16;
    } cfg;

    struct event {
        action what = NONE;
        channel where = ANSWER;
        int period = 0;
        int repeats = 0;
    };

    std::array<int32_t, kMaxPeriod> tail {};
    std::array<uint32_t, kMaxPeriod> matching {};
    uint64_t count = 0;
    channel current = ANSWER;
    int nudge_left = 0;
    int watch_left = 0;
    int nudges_in_span = 0;
    uint64_t detections = 0;
    uint64_t nudges = 0;
    uint64_t forced_thinks = 0;
    uint64_t stopped_answers = 0;
    std::array<event, 16> pending {};
    uint8_t pending_head = 0;
    uint8_t pending_size = 0;
    bool terminal = false;

    void reset() {
        const config saved = cfg;
        *this = LoopBreaker{};
        cfg = saved;
    }

    void boundary(channel next) {
        tail = {};
        matching = {};
        count = 0;
        current = next;
        nudge_left = 0;
        watch_left = 0;
        nudges_in_span = 0;
        terminal = false;
    }

    bool nudging() const { return nudge_left > 0; }

    // Called once for each target token accepted by the sampler.
    // This includes each token in a multi-token speculative verification pass.
    event accept(int32_t token, channel where) {
        if (where != current) boundary(where);
        if (terminal) return {};
        if (nudge_left > 0) --nudge_left;
        if (watch_left > 0) --watch_left;

        int period = 0;
        int repeats = 0;
        const int min_repeats = where == THINK ? cfg.think_repeats : cfg.answer_repeats;
        const int min_tokens  = where == THINK ? cfg.think_tokens  : cfg.answer_tokens;
        const int max_period = std::clamp(
            where == THINK ? cfg.think_max_period : cfg.answer_max_period,
            0, kMaxPeriod);
        for (int p = 1; p <= max_period; ++p) {
            auto & run = matching[p - 1];
            if (count >= (uint64_t) p && tail[(count - p) % tail.size()] == token) {
                if (run != std::numeric_limits<uint32_t>::max()) ++run;
            } else {
                run = 0;
            }
            const uint64_t pure = (uint64_t) run + p;
            if (!period && run >= (uint32_t) p && pure >= (uint64_t) min_tokens &&
                pure / p >= (uint64_t) min_repeats) {
                period = p;
                repeats = (int) std::min<uint64_t>(pure / p, (uint64_t) std::numeric_limits<int32_t>::max());
            }
        }
        tail[count % tail.size()] = token;
        ++count;

        event found {};
        if (period && nudges_in_span == 0) {
            found = { NUDGE, where, period, repeats };
            nudge_left = 16;
            watch_left = 16 + 256;
            ++detections;
            ++nudges;
            ++nudges_in_span;
        } else if (period && nudges_in_span >= 1 && !nudging()) {
            found = { where == THINK ? FORCE_THINK : STOP_ANSWER, where, period, repeats };
            nudge_left = 0;
            watch_left = 0;
            terminal = true;
            ++detections;
            if (where == THINK) ++forced_thinks; else ++stopped_answers;
        }
        if (found.what != NONE) {
            if (pending_size == pending.size()) {
                pending_head = (uint8_t) ((pending_head + 1) % pending.size());
                --pending_size;
            }
            pending[(pending_head + pending_size) % pending.size()] = found;
            ++pending_size;
        }
        return found;
    }

    void replace_last_action(action what) {
        if (pending_size) pending[(pending_head + pending_size - 1) % pending.size()].what = what;
    }

    event take_event() {
        if (!pending_size) return {};
        const event result = pending[pending_head];
        pending_head = (uint8_t) ((pending_head + 1) % pending.size());
        --pending_size;
        return result;
    }
};

struct BehaviourConfig {
  std::string marker_bias_by_effort, reasoning_budget_message;
  std::string reasoning_start_tag{"<think>"};
  std::vector<std::string> reasoning_end_tags{"</think>"};
  std::optional<double> request_marker_bias;
  bool production_sampling{false}, model_suppress_tokens{false};
  std::vector<std::uint32_t> suppress_ids;
  float native_temperature{0}, native_top_p{1}, native_min_p{0};
  int native_top_k{0};
  float native_repeat_penalty{1}, native_frequency_penalty{0}, native_presence_penalty{0};
  std::size_t native_min_keep{0}, native_repeat_last_n{64};
  bool think_eog_guard{false}, eog_boundary{false}, eog_min_prob_exact{false};
  bool loop_breaker{false}, loop_eog_close{false}, loop_exempt{false};
  bool reasoning_budget_enabled{false}, logit_bias_enabled{false};
  bool spec_eog_stop{false}, initial_thinking{false};
  double eog_min_prob{0}, eog_defer_p{0}, marker_bias{0};
  int loop_transition{2};
  int think_guard_escalate{16};
  std::int64_t reasoning_budget{-1};
  std::vector<std::uint32_t> eog_ids, think_start, think_end;
  std::vector<std::vector<std::uint32_t>> think_starts, think_ends;
  std::vector<std::uint32_t> budget_forced, loop_prefix, marker_ids;
  std::vector<std::pair<std::uint32_t,float>> logit_bias;
  std::vector<std::pair<std::string,float>> text_logit_bias;
  std::shared_ptr<const std::vector<std::string>> token_pieces;
  [[nodiscard]] bool enabled() const noexcept {
    return production_sampling || model_suppress_tokens || think_eog_guard || eog_boundary || eog_min_prob > 0 ||
      eog_defer_p > 0 || loop_breaker || loop_eog_close ||
      reasoning_budget_enabled || logit_bias_enabled || marker_bias != 0 ||
      !marker_bias_by_effort.empty() || request_marker_bias.has_value() ||
      spec_eog_stop;
  }
};
struct BehaviourCount { std::uint64_t taken{0}, fallback{0}; };
struct BehaviourState {
  bool thinking{false}, prefix_used{false}, waiting_utf8{false};
  bool close_decision{false}, guard_escalated{false}, guard_escalation_taken{false};
  std::uint64_t guard_blocked_attempts{0};
  std::uint64_t accepted_count{0};
  std::vector<std::uint64_t> deferred_at, close_at, guard_blocked_at;
  std::int64_t remaining{-1};
  std::vector<std::uint32_t> match_tail, forced, matched_end;
  std::size_t force_pos{0};
  std::string force_reason;
  EogBoundaryState boundary;
  LoopBreaker loop;
  std::string utf8_tail;
  std::map<std::pair<std::string,std::string>, BehaviourCount> counters;

  void Reset(const BehaviourConfig& cfg) {
    *this = {};
    thinking = cfg.initial_thinking;
    remaining = cfg.reasoning_budget;
    if (thinking && cfg.reasoning_budget_enabled && remaining == 0)
      Force(cfg.budget_forced,
            "reasoning_budget");
  }
  void Force(const std::vector<std::uint32_t>& seq, std::string reason) {
    if (seq.empty()) return;
    forced=seq; force_pos=0; force_reason=std::move(reason);
  }
  [[nodiscard]] bool forcing() const { return force_pos < forced.size(); }
  [[nodiscard]] std::string_view Piece(const BehaviourConfig& cfg,
                                      std::uint32_t token) const {
    if (!cfg.token_pieces || token >= cfg.token_pieces->size()) return {};
    return (*cfg.token_pieces)[token];
  }
  [[nodiscard]] bool Ends(const std::vector<std::uint32_t>& seq) const {
    return !seq.empty() && match_tail.size() >= seq.size() &&
      std::equal(seq.rbegin(),seq.rend(),match_tail.rbegin());
  }
  void Accept(const BehaviourConfig& cfg, std::uint32_t token,
              bool grammar_active) {
    ++accepted_count;
    if(std::ranges::find(deferred_at,accepted_count)!=deferred_at.end()) boundary.deferred_stop=true;
    if(std::ranges::find(close_at,accepted_count)!=close_at.end()) close_decision=true;
    // Sampling/draft positions journal Taken events; only acceptance consumes them.
    const bool blocked=std::ranges::find(guard_blocked_at,accepted_count)!=
      guard_blocked_at.end();
    // Keep ordinals for CopyDrawStateFrom replay, as with close_at.
    if (thinking && blocked && !guard_escalated) {
      ++guard_blocked_attempts;
      if (cfg.think_guard_escalate>0 &&
          guard_blocked_attempts>=static_cast<std::uint64_t>(cfg.think_guard_escalate)) {
        guard_escalated=true;
        guard_escalation_taken=true;
      }
    }
    const bool was_thinking=thinking, was_forcing=forcing();
    const bool prefix_forcing=was_forcing&&force_reason=="loop_transition"&&cfg.loop_transition==2;
    if(prefix_forcing && remaining>0)--remaining;
    // The redirect already selected the first close token. Continue only its
    // remaining tokens; this is not an arbitrary forced think close.
    const bool redirected_close=close_decision;
    if (redirected_close) {
      forced=cfg.think_end;force_pos=1;force_reason="loop_eog_close";
      close_decision=false;
    }
    const auto piece=Piece(cfg,token);
    match_tail.push_back(token);
    auto max_match=std::max(cfg.think_start.size(),cfg.think_end.size());
    for(const auto& seq:cfg.think_starts)max_match=std::max(max_match,seq.size());
    for(const auto& seq:cfg.think_ends)max_match=std::max(max_match,seq.size());
    if (max_match==0) match_tail.clear();
    else if (match_tail.size()>max_match)
      match_tail.erase(match_tail.begin(),match_tail.end()-max_match);
    std::vector<std::uint32_t> end_match;
    if(Ends(cfg.think_end))end_match=cfg.think_end;
    for(const auto& seq:cfg.think_ends)if(seq.size()>end_match.size()&&Ends(seq))end_match=seq;
    if (!end_match.empty() && thinking) {
      matched_end=std::move(end_match);
      thinking=false; prefix_used=false; waiting_utf8=false;
      guard_blocked_attempts=0;guard_escalated=false;
      forced.clear();force_pos=0;boundary.reset();loop.boundary(LoopBreaker::ANSWER);
      return;
    }
    if (!thinking && (Ends(cfg.think_start)||std::ranges::any_of(cfg.think_starts,[&](const auto& seq){return Ends(seq);}))) {
      matched_end.clear();
      thinking=true;prefix_used=false;remaining=cfg.reasoning_budget;
      guard_blocked_attempts=0;guard_escalated=false;
      boundary.reset();loop.boundary(LoopBreaker::THINK);
      if(cfg.reasoning_budget_enabled && remaining==0)
        Force(cfg.budget_forced,"reasoning_budget");
      return;
    }
    if (forcing() && !redirected_close) {
      ++force_pos;
      if (!forcing()) {
        forced.clear();force_pos=0;
        if(force_reason=="loop_transition" && cfg.loop_transition==2) {
          prefix_used=true;loop.boundary(LoopBreaker::THINK);
          if(cfg.reasoning_budget_enabled && remaining==0)
            Force(cfg.budget_forced,"reasoning_budget");
        }
      }
    }
    if (!thinking && !was_thinking) boundary.accept_piece(piece);
    if (thinking && cfg.reasoning_budget_enabled && !was_forcing) {
      utf8_tail.append(piece);
      if (utf8_tail.size()>8) utf8_tail.erase(0,utf8_tail.size()-8);
      std::uint32_t cp;std::size_t start;
      const bool complete=utf8_tail.empty() || EogBoundaryState::last_codepoint(utf8_tail,cp,start);
      if(remaining>0) --remaining;
      if(remaining==0) {
        waiting_utf8=!complete;
        if(complete) Force(cfg.budget_forced,"reasoning_budget");
      }
    }
    if (cfg.loop_breaker && !grammar_active && !was_forcing && !forcing()) {
      auto event=loop.accept(static_cast<std::int32_t>(token),thinking?LoopBreaker::THINK:LoopBreaker::ANSWER);
      const bool backstop=cfg.loop_transition==2 && prefix_used && thinking && event.what!=LoopBreaker::NONE;
      if(event.what==LoopBreaker::FORCE_THINK || backstop) {
        if(backstop || cfg.loop_transition==0 || cfg.loop_prefix.empty())
          Force(cfg.budget_forced,"loop_breaker");
        else {
          auto seq=cfg.loop_prefix;
          if(cfg.loop_transition!=2) {
            if(cfg.budget_forced.size()>=cfg.think_end.size() &&
               std::equal(cfg.think_end.rbegin(),cfg.think_end.rend(),cfg.budget_forced.rbegin()))
              seq.insert(seq.begin(),cfg.budget_forced.begin(),cfg.budget_forced.end()-cfg.think_end.size());
            seq.insert(seq.end(),cfg.think_end.begin(),cfg.think_end.end());
          }
          Force(seq,"loop_transition");
        }
      } else if(event.what==LoopBreaker::STOP_ANSWER && !cfg.eog_ids.empty())
        Force({cfg.eog_ids.front()},"loop_breaker");
    }
  }
  void Taken(std::string name,std::string reason) { ++counters[{std::move(name),std::move(reason)}].taken; }
  void Fallback(std::string name,std::string reason) { ++counters[{std::move(name),std::move(reason)}].fallback; }
  [[nodiscard]] std::vector<std::string> Summary() const {
    std::vector<std::string> out;
    for(const auto& [key,count]:counters)
      out.push_back("BEHAVIOUR_ENGAGE name="+key.first+" taken="+std::to_string(count.taken)+
        " fallback="+std::to_string(count.fallback)+" reason="+key.second+" scope=request");
    return out;
  }
};
[[nodiscard]] inline bool eog_closes_think(const BehaviourConfig& cfg,
                                         const BehaviourState& state) {
  return cfg.loop_eog_close && (state.prefix_used || state.guard_escalated);
}
} // namespace gufo::sampling
#endif
