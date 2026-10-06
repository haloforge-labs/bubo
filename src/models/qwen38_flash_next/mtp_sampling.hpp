#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

enum class DraftDecision { kAccept, kReject, kStop };

inline constexpr std::size_t kMtpCandidates = 64;

struct MtpCandidateLogits {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> logits{};
  std::size_t size{0};
};

struct MtpProposal {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> probabilities{};
  std::size_t size{0};
  sampling::TokenId token{0};
  float probability{0.0F};
};

/// The proposal has bounded support; the target distribution stays complete.
/// Integer probability masses sum to 2^24, so the exported F32 q sums to
/// exactly one and is exactly the distribution used to draw the proposal.
inline MtpProposal SampleMtpProposal(const MtpCandidateLogits& candidates,
                                     const sampling::SamplerState& sampler,
                                     std::uint64_t* rng) {
  if (candidates.size == 0 || candidates.size > kMtpCandidates) {
    throw std::invalid_argument("invalid MTP candidate count");
  }
  const auto distribution =
      sampler.WithoutBehaviour().Distribution(std::span(candidates.logits).first(candidates.size),
                           std::span(candidates.ids).first(candidates.size));
  constexpr std::uint32_t units = 1U << 24;
  std::array<std::uint32_t, kMtpCandidates> mass{};
  MtpProposal proposal;
  proposal.size = distribution.entries().size();
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto& entry = distribution.entries()[i];
    proposal.ids[i] = candidates.ids[entry.token];
    mass[i] = static_cast<std::uint32_t>(std::floor(entry.value * units));
    total += mass[i];
  }
  mass[0] += units - total;
  const auto draw = static_cast<std::uint32_t>(sampling::Uniform(rng) * units);
  std::uint32_t cumulative = 0;
  bool selected = false;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    proposal.probabilities[i] = static_cast<float>(mass[i]) / units;
    cumulative += mass[i];
    if (!selected && draw < cumulative) {
      proposal.token = proposal.ids[i];
      proposal.probability = proposal.probabilities[i];
      selected = true;
    }
  }
  return proposal;
}

struct MtpVerification {
  sampling::TokenId token;
  bool accepted;
};

/// Lenient (bounded-relaxation) acceptance of sampled MTP drafts, default OFF (decode thread, lead 3, patch v4).
/// GUFO_LENIENT=bounded (literal value only; anything else or unset = off and the verifier is bit-for-bit the exact one) accepts a draft token x
/// whenever u * q(x) < (1 + lambda) * t(x), u = the SAME single uniform draw the exact test uses (u * q < t): RNG consumption is unchanged, lambda -> 0
/// reduces to the exact test, a rejection draws the residual from max(0, t - q) as before. The emitted distribution E satisfies E <= (1 + lambda) t
/// pointwise, so KL(E||t) <= ln(1 + lambda) and TV(E, t) <= lambda / (1 + lambda) (asserted below). GUFO_LENIENT_LAMBDA: default 0.5 only when the variable
/// is UNSET; a set value must parse and satisfy 0 < lambda <= 2 (1e-12 is valid = a null arm that runs the new code), otherwise lenient acceptance stays OFF
/// and a '[lenient] REJECTED' line is printed (no silent default). GUFO_LENIENT_THINK_ONLY (default on; the literal 0 turns the gate off) restricts the
/// relaxation to positions inside the reasoning span: the most recent of <think> (248068) / </think> (248069) / <tool_call> (248058) in the committed chain
/// is <think>. A draft token that is an end-of-turn token, </think> or <tool_call> is never relaxed.
struct MtpLenientConfig {
  bool enabled{false};
  double lambda{0.5};
  bool think_only{true};
};

inline constexpr std::int32_t kMtpThinkStartToken = 248068;  // Qwen3.8 vocabulary (248,320 entries)
inline constexpr std::int32_t kMtpThinkEndToken = 248069;
inline constexpr std::int32_t kMtpToolCallToken = 248058;

struct MtpLenientCounters {
  std::atomic<std::uint64_t> checked{0};           // positions verified with the relaxation active
  std::atomic<std::uint64_t> taken{0};             // exact test rejected, relaxed test accepted
  std::atomic<std::uint64_t> skipped_answer{0};    // outside the thinking gate
  std::atomic<std::uint64_t> skipped_special{0};   // draft is end-of-turn / </think> / <tool_call>
  std::atomic<std::uint64_t> bound_violations{0};  // KL > ln(1+lambda) or TV > lambda/(1+lambda) (must stay 0)
  std::atomic<std::int64_t> kl_micro{0};           // sum of KL(E||t) in micro-nats
  std::atomic<std::int64_t> dh_micro{0};           // sum of H(E) - H(t) in micro-nats
  std::atomic<std::int64_t> tail_micro{0};         // sum of E(tail) - t(tail), tail = tokens with t < 5 % of the peak, in micro-units
  std::atomic<std::uint64_t> tv_micro_max{0};
  void Print(const char* tag) const;
  ~MtpLenientCounters();
};

inline const MtpLenientConfig& MtpLenient() {
  static const MtpLenientConfig config = [] {
    MtpLenientConfig c;
    const char* mode = std::getenv("GUFO_LENIENT");
    if (mode != nullptr && std::strcmp(mode, "bounded") == 0) {
      bool valid = true;
      if (const char* l = std::getenv("GUFO_LENIENT_LAMBDA")) {
        char* end = nullptr;
        const double v = std::strtod(l, &end);
        if (end != l && *end == '\0' && std::isfinite(v) && v > 0.0 && v <= 2.0) {
          c.lambda = v;
        } else {
          valid = false;
          std::fprintf(stderr, "[lenient] REJECTED GUFO_LENIENT_LAMBDA='%s' (need 0 < lambda <= 2): lenient acceptance is OFF\n", l);
        }
      }
      if (const char* t = std::getenv("GUFO_LENIENT_THINK_ONLY")) {
        c.think_only = !(t[0] == '0' && t[1] == '\0');
      }
      c.enabled = valid;
      if (valid) {
        std::fprintf(stderr, "[lenient] active mode=bounded lambda=%.12g think_only=%d (new MtpLenient helpers in use)\n", c.lambda, c.think_only ? 1 : 0);
      }
    }
    return c;
  }();
  return config;
}

inline MtpLenientCounters& MtpLenientStats() {
  static MtpLenientCounters counters;
  return counters;
}

inline void MtpLenientCounters::Print(const char* tag) const {
  const auto& c = MtpLenient();
  const double n = static_cast<double>(checked.load());
  char line[640];
  std::snprintf(line, sizeof line,
                "[lenient] %s mode=bounded lambda=%.12g think_only=%d checked=%llu taken=%llu skipped_answer=%llu skipped_special=%llu taken_frac=%.5f kl_mean=%.6f dH_mean=%.6f tail_shift_mean=%.6f tv_max=%.5f bound_violations=%llu\n",
                tag, c.lambda, c.think_only ? 1 : 0, static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(taken.load()),
                static_cast<unsigned long long>(skipped_answer.load()), static_cast<unsigned long long>(skipped_special.load()), n > 0 ? static_cast<double>(taken.load()) / n : 0.0,
                n > 0 ? 1e-6 * static_cast<double>(kl_micro.load()) / n : 0.0, n > 0 ? 1e-6 * static_cast<double>(dh_micro.load()) / n : 0.0,
                n > 0 ? 1e-6 * static_cast<double>(tail_micro.load()) / n : 0.0, 1e-6 * static_cast<double>(tv_micro_max.load()),
                static_cast<unsigned long long>(bound_violations.load()));
  std::fputs(line, stderr);
}

inline MtpLenientCounters::~MtpLenientCounters() {
  if (MtpLenient().enabled) Print("final");
}

/// Acceptance scale for the draft at this position: 1.0 (exact) unless GUFO_LENIENT is on and, with the gate, the position is inside thinking.
[[nodiscard]] inline double MtpLenientScale(bool thinking) {
  const auto& c = MtpLenient();
  if (!c.enabled) return 1.0;
  if (c.think_only && !thinking) {
    MtpLenientStats().skipped_answer.fetch_add(1, std::memory_order_relaxed);
    return 1.0;
  }
  return 1.0 + c.lambda;
}

struct MtpLenientShift {
  double tv{0}, kl{0}, dh{0}, tail{0};
};

/// Exact distribution shift of the relaxed verifier at one position. With A = sum_i min(q_i, scale t_i), resid(y) = max(0, t(y) - q(y)), Z = sum_y resid(y):
/// E(y) = min(q(y), scale t(y)) [if y is a candidate] + (1 - A) resid(y) / Z. Returns TV(E,t), KL(E||t), H(E) - H(t) (nats) and E(tail) - t(tail) with
/// tail = tokens whose target probability is below 5 % of the peak. E <= scale * t pointwise, so KL <= ln(scale) and TV <= (scale - 1) / scale.
[[nodiscard]] inline MtpLenientShift MtpLenientDistributionShift(const sampling::SamplingDistribution& target, const MtpProposal& proposal, double scale) {
  MtpLenientShift out;
  const auto entries = target.entries();
  double accept = 0.0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    accept += std::min<double>(proposal.probabilities[i], scale * target.probability(proposal.ids[i]));
  }
  double peak = 0.0, z = 0.0;
  std::vector<double> q_of(entries.size(), 0.0);
  for (std::size_t k = 0; k < entries.size(); ++k) {
    for (std::size_t i = 0; i < proposal.size; ++i) {
      if (proposal.ids[i] == entries[k].token) q_of[k] = proposal.probabilities[i];
    }
    peak = std::max(peak, entries[k].value);
    z += std::max(0.0, entries[k].value - q_of[k]);
  }
  if (z <= 0.0) return out;
  const double rej = 1.0 - accept;
  for (std::size_t k = 0; k < entries.size(); ++k) {
    const double t = entries[k].value;
    const double q = q_of[k];
    const double e = std::min(q, scale * t) + rej * std::max(0.0, t - q) / z;
    out.tv += std::abs(e - t);
    if (e > 0.0 && t > 0.0) out.kl += e * std::log(e / t);
    if (e > 0.0) out.dh -= e * std::log(e);
    if (t > 0.0) out.dh += t * std::log(t);
    if (t < 0.05 * peak) out.tail += e - t;
  }
  out.tv *= 0.5;
  return out;
}

/// The same FP64 target distribution drives AR draws, support, acceptance
/// and residual correction. Only proposal q uses the compact F32 masses.
/// `target` is sampler.Distribution(logits) for the verified row, however
/// it was computed (the full row or an exact candidate list).
inline MtpVerification VerifyMtpProposal(
    const sampling::SamplingDistribution& target, std::size_t vocab,
    const MtpProposal& proposal, sampling::SamplerState& sampler,
    double accept_scale = 1.0) {
  if (proposal.size == 0 || proposal.size > kMtpCandidates ||
      proposal.token >= vocab || !std::isfinite(proposal.probability) ||
      proposal.probability <= 0 || proposal.probability > 1)
    throw std::invalid_argument("invalid MTP proposal");
  double total = 0;
  double selected = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    if (proposal.ids[i] >= vocab || !std::isfinite(proposal.probabilities[i]) ||
        proposal.probabilities[i] < 0)
      throw std::invalid_argument("invalid MTP proposal mass");
    total += proposal.probabilities[i];
    if (proposal.ids[i] == proposal.token)
      selected += proposal.probabilities[i];
  }
  if (total != 1 || selected != proposal.probability)
    throw std::invalid_argument("MTP proposal masses do not match the draw");
  const auto rng_before = sampler.rng_state();
  const auto draw = sampler.Uniform() * proposal.probability;
  const auto target_probability = target.probability(proposal.token);
  bool accepted = draw < target_probability;
  if (accept_scale != 1.0) {
    // Lenient relaxation (see MtpLenientConfig): the same draw, a larger acceptance threshold.
    auto& stats = MtpLenientStats();
    const auto exact = accepted;
    const auto relaxed = draw < target_probability * accept_scale;
    const auto n = stats.checked.fetch_add(1, std::memory_order_relaxed) + 1;
    if (relaxed && !exact) stats.taken.fetch_add(1, std::memory_order_relaxed);
    const auto shift = MtpLenientDistributionShift(target, proposal, accept_scale);
    stats.kl_micro.fetch_add(static_cast<std::int64_t>(std::llround(1e6 * shift.kl)), std::memory_order_relaxed);
    stats.dh_micro.fetch_add(static_cast<std::int64_t>(std::llround(1e6 * shift.dh)), std::memory_order_relaxed);
    stats.tail_micro.fetch_add(static_cast<std::int64_t>(std::llround(1e6 * shift.tail)), std::memory_order_relaxed);
    const auto tv_micro = static_cast<std::uint64_t>(1e6 * shift.tv);
    auto seen = stats.tv_micro_max.load(std::memory_order_relaxed);
    while (tv_micro > seen && !stats.tv_micro_max.compare_exchange_weak(seen, tv_micro, std::memory_order_relaxed)) {
    }
    if (shift.kl > std::log(accept_scale) + 1e-9 || shift.tv > (accept_scale - 1.0) / accept_scale + 1e-9) {
      stats.bound_violations.fetch_add(1, std::memory_order_relaxed);
    }
    if (n == 1 || (n & 127) == 0) stats.Print("periodic");
    accepted = relaxed;
  }
  const auto token = accepted ? proposal.token : target.SampleResidual(
      std::span(proposal.ids).first(proposal.size),
      std::span(proposal.probabilities).first(proposal.size),
      sampler.mutable_rng_state());
  sampler.ObserveBehaviour(target,token,rng_before,
      std::span(proposal.ids).first(proposal.size),
      std::span(proposal.probabilities).first(proposal.size),
      proposal.token,proposal.probability);
  return {token,accepted};
}

inline MtpVerification VerifyMtpProposal(std::span<const float> logits,
                                         const MtpProposal& proposal,
                                         sampling::SamplerState& sampler,
                                         double accept_scale = 1.0) {
  return VerifyMtpProposal(sampler.Distribution(logits), logits.size(),
                           proposal, sampler, accept_scale);
}

/// The draft is deterministic, so an exact target sample decides acceptance.
/// A rejected draw is deferred to the next cycle: restore its RNG so that
/// stopping at this prefix consumes exactly the draws made by ordinary decode.
template<typename IsStop>
DraftDecision VerifyDraft(std::span<const float> logits, std::int32_t draft,
                          sampling::SamplerState& sampler, IsStop is_stop) {
  const auto before = sampler;
  const auto token = sampler.Sample(logits);
  if (is_stop(static_cast<std::int32_t>(token))) {
    return DraftDecision::kStop;
  }
  if (token != static_cast<sampling::TokenId>(draft)) {
    sampler = before;
    return DraftDecision::kReject;
  }
  sampler.Accept(token);
  return DraftDecision::kAccept;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
