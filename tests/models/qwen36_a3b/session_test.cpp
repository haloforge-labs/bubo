// A3B engine session tests (GPU): state round trip, restore over a different
// token history, prefill chunk independence and greedy MTP equivalence.
//
//   a3b_session_test <model.gguf> [--quick]
//
// Without a model path the test prints SKIP and exits 0, so a stray ctest run
// does not fail. Each check is independent; the exit code is 1 if any fails.
// The engine has no session API: one Engine is loaded and every check starts
// from Reset(), so the checks do not depend on each other.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

#ifndef _WIN32
extern char** environ;
#endif

namespace {

namespace a3b = gufo::models::qwen36_a3b;
using a3b::Candidates;
using a3b::Engine;
using Tokens = std::vector<std::int32_t>;
using TokenSpan = std::span<const std::int32_t>;

// Chunk-split logits may differ by rounding when the two splits take
// different kernel shapes. A split fails when its top-1 token changes or the
// largest absolute logit difference exceeds this bound.
constexpr double kLogitTolerance = 5e-2;
constexpr std::uint32_t kMaxContext = 4096;
constexpr std::uint32_t kWideChunk = 2048;
constexpr std::size_t kCorpusTokens = 2400;

std::string Fmt(const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list copy;
  va_copy(copy, args);
  const int n = std::vsnprintf(nullptr, 0, format, copy);
  va_end(copy);
  std::string out(n > 0 ? static_cast<std::size_t>(n) : 0, '\0');
  if (n > 0)
    std::vsnprintf(out.data(), out.size() + 1, format, args);
  va_end(args);
  return out;
}

void Need(bool ok, const std::string& error, const char* what) {
  if (!ok)
    throw std::runtime_error(std::string(what) + ": " + error);
}

class Verdict {
public:
  void Note(const std::string& text) const {
    std::printf("  %s\n", text.c_str());
  }
  void Expect(bool condition, const std::string& text) {
    std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", text.c_str());
    if (!condition)
      ok_ = false;
  }
  bool ok() const { return ok_; }

private:
  bool ok_{true};
};

// --- Prompts -----------------------------------------------------------

constexpr std::size_t kSentenceCount = 10;
const char* const kTextA[kSentenceCount] = {
    "The compiler lowers each kernel to a wave-level program and schedules "
    "its memory loads ahead of the arithmetic that needs them.",
    "A hybrid trunk alternates linear recurrent layers with a few full "
    "attention layers, so most of the context lives in a fixed-size state.",
    "Quantised weights are stored in blocks with a shared scale, and the "
    "matrix kernels dequantise each block in registers before the dot product.",
    "The scheduler keeps the graph small by replaying one captured launch for "
    "every decode step instead of issuing a thousand kernels again.",
    "Speculative decoding drafts a few tokens cheaply, verifies them in one "
    "wide pass and rolls the recurrent state back to the last accepted row.",
    "Prompt processing batches many rows together, which turns the narrow "
    "vector kernels into wide matrix multiplies that fill the whole chip.",
    "A prompt checkpoint stores only the recurrent state, because the cached "
    "keys and values below the saved position are never rewritten.",
    "Each expert holds its own gate, up and down matrices, and the router "
    "picks the eight best experts for every token it sees.",
    "Memory bandwidth, not arithmetic, limits the decode rate, so the plan "
    "counts bytes read per token before it counts operations.",
    "A regression test pins the output of a fixed prompt, then fails loudly "
    "when a change to a kernel moves a single logit by more than rounding."};
const char* const kTextB[kSentenceCount] = {
    "The old bakery on the corner opens before sunrise and sells out of rye "
    "bread by nine in the morning.",
    "A narrow path climbs through the pine forest to a meadow where goats "
    "graze beside a cold stream.",
    "To make a good broth, simmer the bones for hours, skim the surface "
    "often and add the salt only at the end.",
    "The night train crosses the mountains slowly, and the conductor "
    "announces each village in three languages.",
    "Harvest in the valley starts in late September, when the grapes taste "
    "of honey and the mornings smell of smoke.",
    "She kept a notebook of every bird she saw from the window, with the "
    "date, the weather and a small drawing.",
    "The market sells copper pots, woven baskets, dried figs and maps of "
    "cities that changed their names long ago.",
    "A lighthouse keeper once painted the tower white every spring, and the "
    "fishermen set their calendars by it.",
    "In winter the harbour freezes at the edges, and the ferries run only "
    "when the wind turns toward the south.",
    "He repaired clocks in a shop with no windows, and every one of them "
    "struck the hour a few seconds apart."};

struct Corpus {
  Tokens a;
  Tokens b;
  std::string source;
};

Tokens PseudoRandomTokens(std::uint32_t seed, std::size_t count,
                          std::uint32_t vocab) {
  const std::uint32_t limit = std::min<std::uint32_t>(vocab, 100000);
  Tokens out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    out.push_back(static_cast<std::int32_t>(1000 + seed % (limit - 1000)));
  }
  return out;
}

Tokens TextTokens(const gufo::tokenization::QwenTokenizer& tokenizer,
                  const char* const* sentences, std::size_t count) {
  std::string text;
  std::size_t entry = 0;
  std::size_t chars = count * 6;
  for (int doublings = 0;; ++doublings) {
    if (doublings >= 16)
      throw std::runtime_error(
          Fmt("TextTokens: the tokenizer produced fewer than %zu tokens after "
              "%d doublings of the input text (%zu characters)",
              count, doublings, text.size()));
    while (text.size() < chars) {
      text += Fmt("Entry %zu: %s ", entry, sentences[entry % kSentenceCount]);
      ++entry;
    }
    const auto encoded = tokenizer.Encode(text);
    if (encoded.size() >= count) {
      Tokens out;
      out.reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        out.push_back(static_cast<std::int32_t>(encoded[i]));
      return out;
    }
    chars *= 2;
  }
}

TokenSpan Prefix(const Tokens& t, std::size_t n) {
  if (n > t.size())
    throw std::logic_error("prompt prefix longer than the corpus");
  return TokenSpan(t.data(), n);
}

struct Session {
  Engine* engine{nullptr};
  Corpus corpus;
  bool quick{false};
};

// --- Comparison helpers --------------------------------------------------

struct FloatDiff {
  bool bitwise{true};
  double max_abs{0};
};

FloatDiff CompareFloats(const float* a, const float* b, std::size_t n) {
  FloatDiff diff;
  diff.bitwise = std::memcmp(a, b, n * sizeof(float)) == 0;
  if (diff.bitwise)
    return diff;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0)
      continue;
    double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (std::isnan(d))
      d = std::numeric_limits<double>::infinity();
    diff.max_abs = std::max(diff.max_abs, d);
  }
  return diff;
}

FloatDiff CompareVectors(const std::vector<float>& a,
                         const std::vector<float>& b) {
  if (a.size() != b.size()) {
    FloatDiff diff;
    diff.bitwise = false;
    diff.max_abs = std::numeric_limits<double>::infinity();
    return diff;
  }
  return CompareFloats(a.data(), b.data(), a.size());
}

std::size_t ArgMax(const std::vector<float>& v) {
  return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) -
                                  v.begin());
}

double Margin(const std::vector<float>& v) {
  double best = -std::numeric_limits<double>::infinity();
  double second = best;
  for (float f : v) {
    if (f > best) {
      second = best;
      best = f;
    } else if (f > second) {
      second = f;
    }
  }
  return best - second;
}

// Log-softmax of float32 logits, accumulated in double.
std::vector<double> LogSoftmax(const std::vector<float>& x) {
  double top = -std::numeric_limits<double>::infinity();
  for (float f : x)
    top = std::max(top, static_cast<double>(f));
  double sum = 0;
  for (float f : x)
    sum += std::exp(static_cast<double>(f) - top);
  const double log_sum = top + std::log(sum);
  std::vector<double> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = static_cast<double>(x[i]) - log_sum;
  return out;
}

// Ids of the 5 largest logits (ties: lower id first), largest first.
std::vector<std::size_t> Top5(const std::vector<float>& x) {
  std::vector<std::size_t> idx(x.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    idx[i] = i;
  const std::size_t k = std::min<std::size_t>(5, idx.size());
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
                    idx.end(), [&](std::size_t a, std::size_t b) {
                      return x[a] > x[b] || (x[a] == x[b] && a < b);
                    });
  idx.resize(k);
  return idx;
}

std::string IdList(const std::vector<std::size_t>& ids) {
  std::string out;
  for (std::size_t i = 0; i < ids.size(); ++i)
    out += Fmt(i == 0 ? "%zu" : " %zu", ids[i]);
  return out;
}

// One line of distribution-level metrics between the last-row logits of a
// reference and a variant: KL(p_ref || p_var) in nats over the full
// vocabulary, the largest absolute probability difference, the token with the
// largest absolute LOGIT difference (with its reference probability and
// rank), both top-5 id lists with a set-equality flag, and both top-1
// probabilities. Information only; no pass criterion uses it.
std::string DistributionLine(const std::vector<float>& ref,
                             const std::vector<float>& var) {
  if (ref.size() != var.size() || ref.empty())
    return "distribution metrics: logit vectors differ in size";
  const std::vector<double> lp = LogSoftmax(ref);
  const std::vector<double> lq = LogSoftmax(var);
  double kl = 0;
  double max_dp = 0;
  double max_dl = -1;
  std::size_t at = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double p = std::exp(lp[i]);
    const double q = std::exp(lq[i]);
    if (p > 0)
      kl += p * (lp[i] - lq[i]);
    max_dp = std::max(max_dp, std::fabs(p - q));
    double dl = std::fabs(static_cast<double>(ref[i]) -
                          static_cast<double>(var[i]));
    if (std::isnan(dl))
      dl = std::numeric_limits<double>::infinity();
    if (dl > max_dl) {
      max_dl = dl;
      at = i;
    }
  }
  std::size_t rank = 1;
  for (float f : ref)
    rank += f > ref[at] ? 1 : 0;
  const std::vector<std::size_t> top_ref = Top5(ref);
  const std::vector<std::size_t> top_var = Top5(var);
  std::vector<std::size_t> a = top_ref;
  std::vector<std::size_t> b = top_var;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return Fmt("distribution: KL(ref||variant) %.6g nats, max abs prob diff "
             "%.6g, largest abs logit diff %.6g at token %zu (ref prob %.6g, "
             "ref rank %zu), top-5 ref [%s] variant [%s] sets %s, top-1 prob "
             "ref %.6g variant %.6g",
             kl, max_dp, max_dl, at, std::exp(lp[at]), rank,
             IdList(top_ref).c_str(), IdList(top_var).c_str(),
             a == b ? "equal" : "DIFFER", std::exp(lp[top_ref[0]]),
             std::exp(lq[top_var[0]]));
}

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

std::size_t FirstDifference(const Tokens& a, const Tokens& b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i)
    if (a[i] != b[i])
      return i;
  return a.size() == b.size() ? kNone : n;
}

// --- Engine drivers --------------------------------------------------------

struct Decoded {
  Tokens tokens;                   // the token each step predicts
  std::vector<float> margin;       // top-1 minus top-2 logit of each step
  std::vector<float> first_logits;  // logits of the first step (optional)
  // Full logits of the first `keep_steps` steps (optional, one row per step).
  std::vector<std::vector<float>> step_logits;
};

// Plain greedy decoding: SpecStep with no drafts, then SpecCommit(1). `x` is
// the token at position(), not yet processed. `keep_steps` > 0 also keeps the
// full logits of that many leading steps in step_logits.
Decoded DecodeAr(Engine& e, std::int32_t x, std::size_t steps,
                 bool keep_logits, std::size_t keep_steps = 0) {
  Decoded d;
  std::string err;
  std::vector<std::int32_t> drafts;
  std::vector<Candidates> rows;
  if (keep_logits)
    d.first_logits.resize(e.config().vocab);
  for (std::size_t i = 0; i < steps; ++i) {
    Need(e.SpecStep(x, 0, &drafts, &rows, &err), err, "SpecStep");
    if (i == 0 && keep_logits)
      Need(e.RowLogits(0, d.first_logits.data(), &err), err, "RowLogits");
    if (i < keep_steps) {
      d.step_logits.emplace_back(e.config().vocab);
      Need(e.RowLogits(0, d.step_logits.back().data(), &err), err,
           "RowLogits");
    }
    const std::int32_t next = rows[0].ids[0];
    d.tokens.push_back(next);
    d.margin.push_back(rows[0].logits[0] - rows[0].logits[1]);
    Need(e.SpecCommit(1, &err), err, "SpecCommit");
    x = next;
  }
  return d;
}

struct MtpStats {
  std::size_t cycles{0};
  std::size_t drafted{0};
  std::size_t accepted{0};
};

// Greedy MTP decoding. Returns exactly `total` tokens starting with `x`
// (x counts as the first). A draft is accepted when it equals the verified
// row's top-1; the first mismatch is replaced by that row's top-1.
Tokens DecodeMtp(Engine& e, std::int32_t x, std::size_t total,
                 std::uint32_t draft, MtpStats* stats) {
  std::string err;
  std::vector<std::int32_t> drafts;
  std::vector<Candidates> rows;
  Tokens out{x};
  while (out.size() < total) {
    Need(e.SpecStep(x, draft, &drafts, &rows, &err), err, "SpecStep");
    ++stats->cycles;
    stats->drafted += drafts.size();
    std::uint32_t keep = static_cast<std::uint32_t>(drafts.size()) + 1;
    std::int32_t next = rows[drafts.size()].ids[0];
    bool done = false;
    for (std::size_t j = 0; j < drafts.size(); ++j) {
      if (rows[j].ids[0] != drafts[j]) {
        keep = static_cast<std::uint32_t>(j) + 1;
        next = rows[j].ids[0];
        break;
      }
      ++stats->accepted;
      out.push_back(drafts[j]);
      if (out.size() >= total) {
        keep = static_cast<std::uint32_t>(j) + 1;
        done = true;
        break;
      }
    }
    Need(e.SpecCommit(keep, &err), err, "SpecCommit");
    if (done)
      break;
    out.push_back(next);
    x = next;
  }
  return out;
}

struct Prefilled {
  std::vector<float> logits;  // last prompt row, full vocabulary
  std::vector<std::uint8_t> state;
  std::int32_t top{0};  // Candidates top-1 of the final SpecPrefill
};

// Reset, then prefill `length` tokens in one piece (split 0) or in two
// SpecPrefill calls cut at `split`.
Prefilled PrefillSplit(Engine& e, const Tokens& t, std::size_t length,
                       std::size_t split) {
  std::string err;
  Prefilled p;
  e.Reset();
  const TokenSpan all = Prefix(t, length);
  if (split > 0) {
    Candidates unused{};
    Need(e.SpecPrefill(all.first(split), &unused, &err), err, "SpecPrefill");
  }
  Candidates last{};
  Need(e.SpecPrefill(all.subspan(split), &last, &err), err, "SpecPrefill");
  p.top = last.ids[0];
  p.logits.resize(e.config().vocab);
  Need(e.RowLogits(0, p.logits.data(), &err), err, "RowLogits");
  p.state.resize(e.StateBytes());
  Need(e.SaveState(p.state, &err), err, "SaveState");
  return p;
}

FloatDiff CompareState(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) {
    FloatDiff diff;
    diff.bitwise = false;
    diff.max_abs = std::numeric_limits<double>::infinity();
    return diff;
  }
  // The state is the header, then F32 rows, then one int32 row; comparing it
  // as floats is exact for equal bits and a close bound otherwise.
  return CompareFloats(reinterpret_cast<const float*>(a.data()),
                       reinterpret_cast<const float*>(b.data()),
                       a.size() / sizeof(float));
}

void ExpectSameDecode(Verdict& v, const std::string& label, const Decoded& ref,
                      const Decoded& got) {
  const std::size_t at = FirstDifference(ref.tokens, got.tokens);
  if (at == kNone) {
    v.Expect(true, Fmt("%s: %zu continuation tokens identical", label.c_str(),
                       ref.tokens.size()));
  } else {
    v.Expect(false,
             Fmt("%s: continuation differs at step %zu (reference token %d, "
                 "got %d; reference top1-top2 margin %.4g)",
                 label.c_str(), at,
                 at < ref.tokens.size() ? ref.tokens[at] : -1,
                 at < got.tokens.size() ? got.tokens[at] : -1,
                 at < ref.margin.size() ? static_cast<double>(ref.margin[at])
                                        : 0.0));
  }
  if (!ref.first_logits.empty() && !got.first_logits.empty()) {
    const FloatDiff d = CompareVectors(ref.first_logits, got.first_logits);
    v.Expect(d.bitwise,
             Fmt("%s: first-step logits %s (max abs diff %.6g)", label.c_str(),
                 d.bitwise ? "bit-identical" : "DIFFER", d.max_abs));
  }
}

// The first decode step after a Reset may build or capture the step graph for
// key (0,0) (eager first use, graph replay afterwards). To keep that first
// use out of the compared arms, run 3 throwaway DecodeAr steps here; the
// caller then does its own Reset and prefill again.
void WarmStepGraph(Engine& e, TokenSpan prompt) {
  std::string err;
  Candidates last{};
  e.Reset();
  Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
  (void)DecodeAr(e, last.ids[0], 3, false);
}

std::string TokenList(const Tokens& t, std::size_t n) {
  std::string out;
  for (std::size_t i = 0; i < std::min(n, t.size()); ++i)
    out += Fmt(i == 0 ? "%d" : " %d", t[i]);
  return out;
}

// Max abs logit difference at each of the first `n` steps, as text.
std::string StepDiffs(const Decoded& ref, const Decoded& got, std::size_t n) {
  std::string out;
  const std::size_t m =
      std::min({n, ref.step_logits.size(), got.step_logits.size()});
  for (std::size_t i = 0; i < m; ++i)
    out += Fmt(i == 0 ? "%.4g" : " %.4g",
               CompareVectors(ref.step_logits[i], got.step_logits[i]).max_abs);
  return out;
}

// The SpecPrefill chunks one call of `n` tokens is cut into: wide chunks of
// kWideChunk rows ("2048+252"), or 8-row groups ("287 x 8-row + 4-row").
std::string Partition(std::uint32_t chunk, std::size_t n) {
  if (n == 0)
    return "0";
  if (chunk == 0) {
    std::string out = Fmt("%zu x 8-row", n / 8);
    if (n % 8 != 0)
      out += Fmt(" + %zu-row", n % 8);
    return out;
  }
  std::string out;
  for (std::size_t rest = n; rest > 0;) {
    const std::size_t take = std::min<std::size_t>(rest, chunk);
    out += Fmt(out.empty() ? "%zu" : "+%zu", take);
    rest -= take;
  }
  return out;
}

// --- Check 1 ---------------------------------------------------------------

void CheckStateRoundTrip(Session& s, Verdict& v) {
  Engine& e = *s.engine;
  std::string err;
  const std::size_t L = s.quick ? 96 : 400;
  const std::size_t N = s.quick ? 16 : 48;
  const std::size_t K = N / 2;
  const TokenSpan prompt = Prefix(s.corpus.a, L);
  const std::size_t vocab = e.config().vocab;

  // Warm-up: 3 throwaway decode steps, so the first use of the step graph key
  // (0,0) (eager run, then capture) is not inside the compared arms.
  WarmStepGraph(e, prompt);

  // Reference: prefill, then N greedy steps.
  Candidates last{};
  e.Reset();
  Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
  std::vector<float> frontier(vocab);
  Need(e.RowLogits(0, frontier.data(), &err), err, "RowLogits");
  const std::int32_t x0 = last.ids[0];
  const Decoded ref = DecodeAr(e, x0, N, true);
  v.Note(Fmt("prompt %zu tokens, %zu greedy steps, first token %d", L, N, x0));

  // Same start: prefill, SaveState, decode K tokens, LoadState over the same
  // history, decode N again.
  e.Reset();
  Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
  std::vector<float> frontier2(vocab);
  Need(e.RowLogits(0, frontier2.data(), &err), err, "RowLogits");
  const FloatDiff fd = CompareVectors(frontier, frontier2);
  v.Expect(fd.bitwise && last.ids[0] == x0,
           Fmt("prefill is repeatable: frontier logits %s (max abs diff %.6g)",
               fd.bitwise ? "bit-identical" : "DIFFER", fd.max_abs));
  std::vector<std::uint8_t> saved(e.StateBytes());
  Need(e.SaveState(saved, &err), err, "SaveState");
  v.Note(Fmt("state snapshot %zu bytes at position %u", saved.size(),
             e.position()));
  const Decoded wander = DecodeAr(e, x0, K, false);
  v.Expect(FirstDifference(
               wander.tokens,
               Tokens(ref.tokens.begin(),
                      ref.tokens.begin() + static_cast<std::ptrdiff_t>(K))) ==
               kNone,
           "decode after SaveState follows the reference path");
  Need(e.LoadState(saved, &err), err, "LoadState");
  v.Expect(e.position() == L,
           Fmt("LoadState returns position() to %zu (got %u)", L, e.position()));
  std::vector<std::uint8_t> again(saved.size());
  Need(e.SaveState(again, &err), err, "SaveState");
  v.Expect(again == saved, "SaveState right after LoadState gives the same bytes");
  Need(e.SetRowLogits(frontier.data(), &err), err, "SetRowLogits");
  std::vector<float> readback(vocab);
  Need(e.RowLogits(0, readback.data(), &err), err, "RowLogits");
  v.Expect(CompareVectors(frontier, readback).bitwise,
           "SetRowLogits / RowLogits round trip is bit-identical");
  const Decoded restored = DecodeAr(e, x0, N, true);
  ExpectSameDecode(v, "LoadState over the same history", ref, restored);

  // Checkpoint: prefill the head, SaveCheckpoint, prefill the tail and decode;
  // RestoreCheckpoint, prefill the tail again and decode again.
  const std::size_t cut = L * 2 / 3;
  e.Reset();
  Candidates unused{};
  Need(e.SpecPrefill(Prefix(s.corpus.a, cut), &unused, &err), err,
       "SpecPrefill");
  Need(e.SaveCheckpoint(&err), err, "SaveCheckpoint");
  const TokenSpan tail(s.corpus.a.data() + cut, L - cut);
  Candidates first{};
  Need(e.SpecPrefill(tail, &first, &err), err, "SpecPrefill");
  std::vector<float> frontier_a(vocab);
  Need(e.RowLogits(0, frontier_a.data(), &err), err, "RowLogits");
  const Decoded run_a = DecodeAr(e, first.ids[0], N, true);
  Need(e.RestoreCheckpoint(&err), err, "RestoreCheckpoint");
  v.Expect(e.position() == cut,
           Fmt("RestoreCheckpoint returns position() to %zu (got %u)", cut,
               e.position()));
  Candidates second{};
  Need(e.SpecPrefill(tail, &second, &err), err, "SpecPrefill");
  std::vector<float> frontier_b(vocab);
  Need(e.RowLogits(0, frontier_b.data(), &err), err, "RowLogits");
  const FloatDiff cd = CompareVectors(frontier_a, frontier_b);
  v.Expect(cd.bitwise && first.ids[0] == second.ids[0],
           Fmt("checkpoint restore + tail prefill: frontier logits %s (max abs "
               "diff %.6g)",
               cd.bitwise ? "bit-identical" : "DIFFER", cd.max_abs));
  const Decoded run_b = DecodeAr(e, second.ids[0], N, true);
  ExpectSameDecode(v, "RestoreCheckpoint over the same history", run_a, run_b);
}

// --- Check 2 ---------------------------------------------------------------

// engine.hpp: the state holds no KV, so it restores only over the token
// history it was saved with. LoadState has no history to compare, so the
// engine may refuse (error return) or reproduce the saved behaviour; anything
// else is a silent wrong continuation.
void CheckRestoreDivergentHistory(Session& s, Verdict& v) {
  Engine& e = *s.engine;
  std::string err;
  const std::size_t L = s.quick ? 96 : 400;
  const std::size_t N = s.quick ? 16 : 48;
  const std::size_t shift = s.quick ? 17 : 37;
  const Tokens h1(s.corpus.a.begin(),
                  s.corpus.a.begin() + static_cast<std::ptrdiff_t>(L));
  const std::size_t vocab = e.config().vocab;

  // Warm-up: 3 throwaway decode steps, so the first use of the step graph key
  // (0,0) (eager run, then capture) is not inside the compared arms.
  WarmStepGraph(e, TokenSpan(h1.data(), h1.size()));

  e.Reset();
  Candidates last{};
  Need(e.SpecPrefill(h1, &last, &err), err, "SpecPrefill");
  const std::int32_t x0 = last.ids[0];
  std::vector<std::uint8_t> saved(e.StateBytes());
  Need(e.SaveState(saved, &err), err, "SaveState");
  constexpr std::size_t kStepLogits = 8;
  const Decoded ref = DecodeAr(e, x0, N, true, kStepLogits);
  v.Note(Fmt("saved state of a %zu-token history; reference continues %zu "
             "steps from token %d (vocab %zu)",
             L, N, x0, vocab));

  const auto edited = [&](std::size_t index) {
    Tokens t = h1;
    t[index] = static_cast<std::int32_t>((t[index] + 7919) % 100000);
    return t;
  };
  struct Scenario {
    std::string name;
    Tokens history;
    bool control;
    Tokens stale;  // prefilled (then Reset) before `history`; empty = none
  };
  std::vector<Scenario> scenarios;
  scenarios.push_back({"control: the same history again", h1, true, {}});
  scenarios.push_back(
      {Fmt("A1 same length %zu, different text", L),
       Tokens(s.corpus.b.begin(),
              s.corpus.b.begin() + static_cast<std::ptrdiff_t>(L)),
       false,
       {}});
  scenarios.push_back({Fmt("A2 same length %zu, one token changed at %zu", L,
                           L / 2),
                       edited(L / 2), false, {}});
  scenarios.push_back({Fmt("A3 same length %zu, one token changed at %zu", L,
                           L - 2),
                       edited(L - 2), false, {}});
  scenarios.push_back(
      {Fmt("B longer (%zu tokens), different text", L + shift),
       Tokens(s.corpus.b.begin(),
              s.corpus.b.begin() + static_cast<std::ptrdiff_t>(L + shift)),
       false,
       {}});
  // Scenario C: the rows L-shift .. L of the engine must hold stale KV from a
  // DIFFERENT, earlier history, not scenario B's text (that would make C a
  // copy of A1). So a third history (pseudo-random ids, unlike h1 and corpus
  // b) of L tokens is prefilled first; after Reset the shorter history
  // overwrites only its own rows and the rows above stay stale.
  scenarios.push_back(
      {Fmt("C shorter (%zu tokens), different text, rows %zu..%zu stale from a "
           "third earlier history",
           L - shift, L - shift, L),
       Tokens(s.corpus.b.begin(),
              s.corpus.b.begin() + static_cast<std::ptrdiff_t>(L - shift)),
       false,
       PseudoRandomTokens(0xC0FFEE11u, L, static_cast<std::uint32_t>(vocab))});

  std::size_t refused = 0;
  std::size_t reproduced = 0;
  std::size_t diverged = 0;
  for (const Scenario& sc : scenarios) {
    Candidates unused{};
    if (!sc.stale.empty()) {
      e.Reset();
      Need(e.SpecPrefill(sc.stale, &unused, &err), err, "SpecPrefill");
    }
    e.Reset();
    Need(e.SpecPrefill(sc.history, &unused, &err), err, "SpecPrefill");
    std::string load_error;
    if (!e.LoadState(saved, &load_error)) {
      ++refused;
      v.Note(Fmt("%s: LoadState refused (%s)", sc.name.c_str(),
                 load_error.c_str()));
      v.Expect(!sc.control,
               Fmt("%s: engine REFUSED LoadState (%s)", sc.name.c_str(),
                   load_error.c_str()));
      continue;
    }
    v.Note(Fmt("%s: LoadState returned ok", sc.name.c_str()));
    if (e.position() != L) {
      v.Expect(false, Fmt("%s: LoadState accepted but position() is %u, "
                          "expected %zu",
                          sc.name.c_str(), e.position(), L));
      ++diverged;
      continue;
    }
    const Decoded got = DecodeAr(e, x0, N, true, kStepLogits);
    const std::size_t at = FirstDifference(ref.tokens, got.tokens);
    const FloatDiff d = CompareVectors(ref.first_logits, got.first_logits);
    // Where does it diverge? Per-step logit differences (steps after the first
    // differing token follow different inputs, so they are large by nature),
    // the leading tokens of both sequences and the reference margin at the
    // first differing step.
    v.Note(Fmt("%s: max abs logit diff per step (first %zu steps): %s",
               sc.name.c_str(), kStepLogits,
               StepDiffs(ref, got, kStepLogits).c_str()));
    v.Note(Fmt("%s: first 16 tokens, saved-history run: [%s]", sc.name.c_str(),
               TokenList(ref.tokens, 16).c_str()));
    v.Note(Fmt("%s: first 16 tokens, restored run:      [%s]", sc.name.c_str(),
               TokenList(got.tokens, 16).c_str()));
    if (at == kNone)
      v.Note(Fmt("%s: token sequences identical over %zu steps (no first "
                 "differing step)",
                 sc.name.c_str(), N));
    else
      v.Note(Fmt("%s: token sequences first differ at step %zu; reference "
                 "top1-top2 margin there %.4g",
                 sc.name.c_str(), at,
                 at < ref.margin.size() ? static_cast<double>(ref.margin[at])
                                        : 0.0));
    if (at == kNone && d.bitwise) {
      ++reproduced;
      v.Expect(true, Fmt("%s: engine accepted LoadState and REPRODUCED the "
                         "saved behaviour exactly (%zu tokens, first-step "
                         "logits bit-identical)",
                         sc.name.c_str(), N));
      continue;
    }
    ++diverged;
    std::size_t differing = 0;
    for (std::size_t i = 0; i < std::min(ref.tokens.size(), got.tokens.size());
         ++i)
      differing += ref.tokens[i] != got.tokens[i] ? 1 : 0;
    std::string where;
    if (at != kNone)
      where = Fmt("%zu of %zu continuation tokens differ, first at step %zu "
                  "(saved-history token %d, got %d); ",
                  differing, N, at, at < ref.tokens.size() ? ref.tokens[at] : -1,
                  at < got.tokens.size() ? got.tokens[at] : -1);
    else
      where = Fmt("all %zu continuation tokens match but ", N);
    v.Expect(false,
             Fmt("%s: engine ACCEPTED LoadState (no error) and continued "
                 "WRONG: %sfirst-step logits max abs diff %.6g (top-1 %s); "
                 "the recurrent state came from the %zu-token saved history, "
                 "the KV rows from the %zu-token history it replaced%s",
                 sc.name.c_str(), where.c_str(), d.max_abs,
                 ArgMax(ref.first_logits) == ArgMax(got.first_logits)
                     ? "same"
                     : "CHANGED",
                 L, sc.history.size(),
                 sc.control ? " [control failed: the comparison is not valid]"
                            : ""));
  }
  v.Note(Fmt("engine behaviour over %zu scenarios: refused %zu, reproduced "
             "%zu, silent divergence %zu (control counts as reproduced)",
             scenarios.size(), refused, reproduced, diverged));
  v.Note("LoadState checks only size, magic and position bound; the caller "
         "must match the token history (text_runner.cpp rows_valid does)");
}

// --- Check 3 ---------------------------------------------------------------

struct ChunkRestore {
  Engine& e;
  ~ChunkRestore() { e.SetPrefillChunk(kWideChunk); }
};

void CheckPrefillChunks(Session& s, Verdict& v) {
  Engine& e = *s.engine;
  ChunkRestore restore{e};
  // Splits are chosen per route so that no split equals the partition the
  // one-piece reference already uses: wide chunks are 2048 rows (so 2048 is
  // not a split for 2300 tokens: 2047 and 2049 are), and the 8-row route
  // groups rows by 8 (so multiples of 8 such as 256, 512 and 2048 would give
  // the same groups: 257, 513 and 2049 are used; 94, 135 and 1025 are not
  // multiples of 8).
  struct Case {
    std::size_t length;
    std::vector<std::size_t> splits[2];  // [route]: wide, 8-row
  };
  std::vector<Case> cases;
  if (s.quick) {
    cases.push_back({300, {{94, 135, 256}, {94, 135, 257}}});
  } else {
    cases.push_back({1200, {{94, 135, 512, 1025}, {94, 135, 513, 1025}}});
    cases.push_back({2300, {{1025, 2047, 2049}, {1025, 2049}}});
  }
  struct Route {
    const char* name;
    std::uint32_t chunk;
  };
  const Route routes[] = {{"wide", kWideChunk}, {"8-row", 0}};
  v.Note(Fmt("pass bound: top-1 unchanged and max abs logit diff <= %.3g "
             "(kLogitTolerance); bitwise equality is reported, not required",
             kLogitTolerance));
  std::vector<Prefilled> one_piece[2];
  for (std::size_t r = 0; r < 2; ++r) {
    e.SetPrefillChunk(routes[r].chunk);
    for (const Case& c : cases) {
      const Prefilled ref = PrefillSplit(e, s.corpus.a, c.length, 0);
      one_piece[r].push_back(ref);
      const std::size_t ref_top = ArgMax(ref.logits);
      v.Note(Fmt("route %s, %zu tokens, one piece: top-1 token %zu (candidates "
                 "%d), top1-top2 margin %.4g",
                 routes[r].name, c.length, ref_top, ref.top, Margin(ref.logits)));
      v.Note(Fmt("route %s, %zu tokens: ref chunks %s", routes[r].name,
                 c.length, Partition(routes[r].chunk, c.length).c_str()));
      for (std::size_t split : c.splits[r]) {
        const Prefilled got = PrefillSplit(e, s.corpus.a, c.length, split);
        const FloatDiff ld = CompareVectors(ref.logits, got.logits);
        const FloatDiff sd = CompareState(ref.state, got.state);
        const bool same_top =
            ArgMax(ref.logits) == ArgMax(got.logits) && ref.top == got.top;
        const bool ok = same_top && ld.max_abs <= kLogitTolerance;
        const std::string ref_chunks = Partition(routes[r].chunk, c.length);
        const std::string split_chunks =
            "[" + Partition(routes[r].chunk, split) + "] [" +
            Partition(routes[r].chunk, c.length - split) + "]";
        v.Expect(ok, Fmt("route %s, %zu tokens split at %zu + %zu (ref chunks "
                         "%s, split chunks %s): logits %s, "
                         "max abs diff %.6g, top-1 %s, state max abs diff %.6g",
                         routes[r].name, c.length, split, c.length - split,
                         ref_chunks.c_str(), split_chunks.c_str(),
                         ld.bitwise ? "bit-identical" : "not bitwise",
                         ld.max_abs, same_top ? "same" : "CHANGED",
                         sd.max_abs));
        v.Note(Fmt("route %s, %zu tokens split at %zu: %s", routes[r].name,
                   c.length, split,
                   DistributionLine(ref.logits, got.logits).c_str()));
      }
    }
  }
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const FloatDiff d = CompareVectors(one_piece[0][i].logits,
                                       one_piece[1][i].logits);
    v.Note(Fmt("cross-route (information only), %zu tokens one piece, wide vs "
               "8-row: max abs logit diff %.6g, top-1 %s",
               cases[i].length, d.max_abs,
               ArgMax(one_piece[0][i].logits) == ArgMax(one_piece[1][i].logits)
                   ? "same"
                   : "CHANGED"));
    v.Note(Fmt("cross-route (information only), %zu tokens one piece, ref = "
               "wide, variant = 8-row: %s",
               cases[i].length,
               DistributionLine(one_piece[0][i].logits, one_piece[1][i].logits)
                   .c_str()));
  }
}

// --- Check 4 ---------------------------------------------------------------

void CheckMtpGreedyEquivalence(Session& s, Verdict& v) {
  Engine& e = *s.engine;
  if (!e.HasMtp()) {
    v.Note("SKIP: this model file has no MTP block");
    return;
  }
  std::string err;
  const std::size_t L = s.quick ? 64 : 200;
  const std::size_t G = s.quick ? 32 : 128;
  const TokenSpan prompt = Prefix(s.corpus.a, L);

  Candidates last{};
  e.Reset();
  Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
  const std::int32_t x0 = last.ids[0];
  Decoded ar = DecodeAr(e, x0, G - 1, false);
  Tokens ref{x0};
  ref.insert(ref.end(), ar.tokens.begin(), ar.tokens.end());
  v.Note(Fmt("prompt %zu tokens, %zu greedy tokens by plain decoding", L, G));

  const auto compare = [&](const std::string& label, const Tokens& got,
                           const MtpStats& st) {
    const std::size_t at = FirstDifference(ref, got);
    const std::string stats =
        Fmt("cycles %zu, drafted %zu, accepted %zu (%.1f%%), %.2f tokens per "
            "cycle",
            st.cycles, st.drafted, st.accepted,
            st.drafted ? 100.0 * st.accepted / st.drafted : 0.0,
            st.cycles ? static_cast<double>(got.size()) / st.cycles : 0.0);
    // Without a rejected draft the SpecCommit keep < n path never ran, and
    // the equivalence above says nothing about rollback.
    v.Expect(st.drafted > st.accepted,
             st.drafted > st.accepted
                 ? Fmt("%s: rollback exercised (drafted %zu > accepted %zu)",
                       label.c_str(), st.drafted, st.accepted)
                 : Fmt("%s: rollback not exercised: acceptance 100 %% "
                       "(drafted %zu, accepted %zu)",
                       label.c_str(), st.drafted, st.accepted));
    if (at == kNone) {
      v.Expect(true, Fmt("%s: %zu tokens identical to plain decoding; %s",
                         label.c_str(), got.size(), stats.c_str()));
      return;
    }
    v.Expect(false,
             Fmt("%s: first difference at index %zu (plain decoding %d, MTP "
                 "%d; plain-decoding top1-top2 margin there %.4g); %s",
                 label.c_str(), at, at < ref.size() ? ref[at] : -1,
                 at < got.size() ? got[at] : -1,
                 at >= 1 && at - 1 < ar.margin.size()
                     ? static_cast<double>(ar.margin[at - 1])
                     : 0.0,
                 stats.c_str()));
  };

  for (std::uint32_t draft : {2u, 4u}) {
    e.Reset();
    Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
    MtpStats st;
    const Tokens got = DecodeMtp(e, last.ids[0], G, draft, &st);
    compare(Fmt("draft length %u", draft), got, st);
  }

  // After LoadState of a saved start state: wander with another draft length
  // (this rewrites KV and MTP rows above the start), restore, run again.
  e.Reset();
  Need(e.SpecPrefill(prompt, &last, &err), err, "SpecPrefill");
  std::vector<std::uint8_t> saved(e.StateBytes());
  Need(e.SaveState(saved, &err), err, "SaveState");
  {
    MtpStats wander_stats;
    (void)DecodeMtp(e, last.ids[0], G / 2, 2, &wander_stats);
  }
  Need(e.LoadState(saved, &err), err, "LoadState");
  v.Expect(e.position() == L,
           Fmt("LoadState returns position() to %zu (got %u)", L, e.position()));
  MtpStats st;
  const Tokens got = DecodeMtp(e, last.ids[0], G, 3, &st);
  compare("draft length 3 after LoadState of the start state", got, st);
}

// --- Driver ----------------------------------------------------------------

bool RunCheck(const char* name, Session& s,
              const std::function<void(Session&, Verdict&)>& check) {
  std::printf("CHECK %s\n", name);
  std::fflush(stdout);
  Verdict v;
  try {
    check(s, v);
  } catch (const std::exception& error) {
    v.Expect(false, std::string("exception: ") + error.what());
  }
  std::printf("%s %s\n", v.ok() ? "PASS" : "FAIL", name);
  std::fflush(stdout);
  return v.ok();
}

struct NamedCheck {
  const char* short_name;  // for --only
  const char* name;
  void (*run)(Session&, Verdict&);
};
const NamedCheck kChecks[] = {
    {"state", "CheckStateRoundTrip", CheckStateRoundTrip},
    {"divergent", "CheckRestoreDivergentHistory", CheckRestoreDivergentHistory},
    {"prefill", "CheckPrefillChunks", CheckPrefillChunks},
    {"mtp", "CheckMtpGreedyEquivalence", CheckMtpGreedyEquivalence},
};

// "NAME=value" of every set A3B_* environment variable, sorted by name.
std::vector<std::string> A3bEnvironment() {
  std::vector<std::string> out;
#ifdef _WIN32
  char** const vars = _environ;
#else
  char** const vars = environ;
#endif
  for (char** p = vars; p != nullptr && *p != nullptr; ++p)
    if (std::strncmp(*p, "A3B_", 4) == 0)
      out.emplace_back(*p);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  // Unbuffered output: detail lines must reach the log even when the process
  // crashes or is killed.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  const char* const usage =
      "usage: a3b_session_test <model.gguf> [--quick] "
      "[--only=state,divergent,prefill,mtp]";
  std::string path;
  bool quick = false;
  std::vector<std::string> only;  // short names; empty = all checks
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else if (arg.rfind("--only=", 0) == 0) {
      const std::string list = arg.substr(7);
      for (std::size_t at = 0; at <= list.size();) {
        const std::size_t comma = list.find(',', at);
        const std::string name = list.substr(
            at, comma == std::string::npos ? std::string::npos : comma - at);
        const bool known =
            std::any_of(std::begin(kChecks), std::end(kChecks),
                        [&](const NamedCheck& c) { return name == c.short_name; });
        if (!known) {
          std::fprintf(stderr, "unknown check '%s' in --only\n%s\n",
                       name.c_str(), usage);
          return 2;
        }
        only.push_back(name);
        if (comma == std::string::npos)
          break;
        at = comma + 1;
      }
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "%s\n", usage);
      return 2;
    } else {
      path = arg;
    }
  }
  if (path.empty()) {
    std::printf("SKIP a3b_session_test: no model path given (%s)\n", usage);
    return 0;
  }

  // Self-watchdog: a hung GPU call must not hold the test forever.
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::minutes(35));
    std::fflush(stdout);
    std::fprintf(stderr, "a3b_session_test: watchdog expired\n");
    std::fflush(stderr);
    std::_Exit(3);
  }).detach();

  a3b::Options options;
  options.max_context = kMaxContext;
  std::string error;
  auto engine = Engine::Load(path, options, &error);
  if (!engine) {
    std::fprintf(stderr, "FAIL a3b_session_test: load failed: %s\n",
                 error.c_str());
    return 1;
  }

  Session session;
  session.engine = engine.get();
  session.quick = quick;
  std::string tokenizer_error;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromGguf(
      engine->gguf(), &tokenizer_error);
  try {
    if (tokenizer) {
      session.corpus.a = TextTokens(*tokenizer, kTextA, kCorpusTokens);
      session.corpus.b = TextTokens(*tokenizer, kTextB, kCorpusTokens);
      session.corpus.source = "fixed English text through the model tokenizer";
    }
  } catch (const std::exception& failure) {
    tokenizer_error = failure.what();
    session.corpus.a.clear();
  }
  if (session.corpus.a.empty()) {
    session.corpus.a = PseudoRandomTokens(0x2545F491u, kCorpusTokens,
                                          engine->config().vocab);
    session.corpus.b = PseudoRandomTokens(0x9E3779B9u, kCorpusTokens,
                                          engine->config().vocab);
    session.corpus.source = "pseudo-random token ids (tokenizer: " +
                            tokenizer_error + ")";
    const char* allow = std::getenv("A3B_TEST_ALLOW_RANDOM_IDS");
    if (allow == nullptr || std::strcmp(allow, "1") != 0) {
      std::printf("FAIL prompt_source: tokenizer unavailable, pseudo-random ids "
                  "(tokenizer: %s); set A3B_TEST_ALLOW_RANDOM_IDS=1 to run "
                  "with pseudo-random ids anyway\n",
                  tokenizer_error.c_str());
      return 1;
    }
    std::printf("WARN prompt_source: tokenizer unavailable, pseudo-random ids "
                "(tokenizer: %s); continuing because "
                "A3B_TEST_ALLOW_RANDOM_IDS=1\n",
                tokenizer_error.c_str());
  }
  std::printf("a3b_session_test: model %s\n", path.c_str());
  std::printf("a3b_session_test: mode %s, prompts from %s, MTP %s\n",
              quick ? "quick" : "full", session.corpus.source.c_str(),
              engine->HasMtp() ? "present" : "absent");

  // Every A3B_* environment variable that is set, so logs record the switches.
  const std::vector<std::string> env = A3bEnvironment();
  if (env.empty())
    std::printf("a3b_session_test: env none\n");
  for (const std::string& entry : env)
    std::printf("a3b_session_test: env %s\n", entry.c_str());

  int failed = 0;
  int ran = 0;
  for (const NamedCheck& c : kChecks) {
    if (!only.empty() &&
        std::find(only.begin(), only.end(), c.short_name) == only.end())
      continue;
    ++ran;
    failed += RunCheck(c.name, session, c.run) ? 0 : 1;
  }
  std::printf("a3b_session_test: %d of %d checks failed\n", failed, ran);
  return failed == 0 ? 0 : 1;
}
