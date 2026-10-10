#include "src/core/sampling.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace gufo::sampling {
namespace {

struct Candidate {
  TokenId token;
  double logit;
};

[[nodiscard]] bool IsBetterCandidate(const Candidate& left,
                                     const Candidate& right) noexcept {
  if (left.logit == right.logit) {
    return left.token < right.token;
  }
  return left.logit > right.logit;
}

[[nodiscard]] bool IsBetterProbability(const Probability& left,
                                       const Probability& right) noexcept {
  if (left.value == right.value) {
    return left.token < right.token;
  }
  return left.value > right.value;
}

[[nodiscard]] std::size_t MinimumKept(const SamplingConfig& config,
                                      std::size_t size) {
  return std::min(size, std::max<std::size_t>(config.min_keep, 1));
}

void SortCandidates(std::vector<Candidate>* candidates) {
  std::ranges::sort(*candidates, IsBetterCandidate);
}

[[nodiscard]] std::vector<double> SoftmaxWeights(
    std::span<const Candidate> candidates, double temperature) {
  if (candidates.empty()) {
    throw std::runtime_error("sampling filters removed every token");
  }
  const double maximum = candidates.front().logit;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  double sum = 0.0;
  for (const auto& candidate : candidates) {
    const double weight = std::exp((candidate.logit - maximum) / temperature);
    weights.push_back(weight);
    sum += weight;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }
  for (double& weight : weights) {
    weight /= sum;
  }
  return weights;
}

double Penalize(double logit, const SamplingConfig& config,
                const TokenPenalty& penalty) {
  if (penalty.repeated && config.repeat_penalty != 1.0F) {
    logit = logit <= 0 ? logit * config.repeat_penalty
                       : logit / config.repeat_penalty;
  }
  logit -=
      static_cast<double>(config.frequency_penalty) * penalty.generated_count;
  if (penalty.generated_count != 0)
    logit -= config.presence_penalty;
  return logit;
}

void ApplyPenalties(std::vector<Candidate>* candidates,
                    const SamplingConfig& config,
                    std::span<const TokenPenalty> penalties) {
  if (penalties.empty())
    return;
  for (auto& candidate : *candidates) {
    const auto found = std::ranges::lower_bound(penalties, candidate.token, {},
                                                &TokenPenalty::token);
    if (found != penalties.end() && found->token == candidate.token)
      candidate.logit = Penalize(candidate.logit, config, *found);
  }
}

void ApplyTopK(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_k <= 0) {
    SortCandidates(candidates);
    return;
  }
  const auto configured = static_cast<std::size_t>(config.top_k);
  const std::size_t keep =
      std::min(candidates->size(),
               std::max(configured, MinimumKept(config, candidates->size())));
  if (keep < candidates->size()) {
    std::partial_sort(candidates->begin(),
                      candidates->begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates->end(), IsBetterCandidate);
  } else {
    SortCandidates(candidates);
  }
  candidates->resize(keep);
}

void ApplyTopP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_p >= 1.0F || candidates->size() <= 1) {
    return;
  }
  const auto probabilities =
      SoftmaxWeights(*candidates, static_cast<double>(config.temperature));
  const std::size_t minimum = MinimumKept(config, candidates->size());
  double cumulative = 0.0;
  std::size_t keep = 0;
  while (keep < candidates->size()) {
    cumulative += probabilities[keep];
    ++keep;
    if (keep >= minimum && cumulative >= static_cast<double>(config.top_p)) {
      break;
    }
  }
  candidates->resize(keep);
}

void ApplyMinP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.min_p <= 0.0F || candidates->size() <= 1) {
    return;
  }
  const double threshold = static_cast<double>(config.temperature) *
                           std::log(static_cast<double>(config.min_p));
  const double maximum = candidates->front().logit;
  std::size_t keep = 0;
  while (keep < candidates->size() &&
         candidates->at(keep).logit - maximum >= threshold) {
    ++keep;
  }
  keep = std::max(keep, MinimumKept(config, candidates->size()));
  candidates->resize(std::min(keep, candidates->size()));
}

[[nodiscard]] std::uint64_t InitialRngState(std::int64_t seed) {
  if (seed >= 0) {
    return static_cast<std::uint64_t>(seed);
  }
  std::random_device random_device;
  return (static_cast<std::uint64_t>(random_device()) << 32U) ^
         static_cast<std::uint64_t>(random_device());
}

constexpr std::size_t kSkipBlock = 64;

/// Largest value in a block, ignoring NaN; +inf is returned as is.
[[nodiscard]] float BlockMaximum(std::span<const float> block) noexcept {
  if (block.size() == kSkipBlock) {
    // A fixed trip count with independent lanes lets the compiler vectorize.
    std::array<float, 8> lanes;
    lanes.fill(-std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < kSkipBlock; i += lanes.size())
      for (std::size_t lane = 0; lane < lanes.size(); ++lane)
        lanes[lane] =
            block[i + lane] > lanes[lane] ? block[i + lane] : lanes[lane];
    float maximum = lanes[0];
    for (const float lane : lanes)
      maximum = lane > maximum ? lane : maximum;
    return maximum;
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (const float value : block)
    maximum = value > maximum ? value : maximum;
  return maximum;
}

}  // namespace

void SamplingConfig::Validate() const {
  if (constraint && (!constraint->grammar || !constraint->vocabulary ||
                     constraint->vocabulary->size() == 0))
    throw std::invalid_argument("sampling constraint is incomplete");
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "sampling temperature must be finite and nonnegative");
  }
  if (top_k < 0) {
    throw std::invalid_argument("sampling top-k must be nonnegative");
  }
  if (!std::isfinite(top_p) || top_p < 0.0F || top_p > 1.0F) {
    throw std::invalid_argument("sampling top-p must be in [0, 1]");
  }
  if (!std::isfinite(min_p) || min_p < 0.0F || min_p > 1.0F) {
    throw std::invalid_argument("sampling min-p must be in [0, 1]");
  }
  if (seed < -1) {
    throw std::invalid_argument("sampling seed must be -1 or nonnegative");
  }
  if (!std::isfinite(repeat_penalty) || repeat_penalty <= 0.0F) {
    throw std::invalid_argument(
        "sampling repeat penalty must be finite and positive");
  }
  if (!std::isfinite(frequency_penalty)) {
    throw std::invalid_argument("sampling frequency penalty must be finite");
  }
  if (!std::isfinite(presence_penalty)) {
    throw std::invalid_argument("sampling presence penalty must be finite");
  }
}

bool SamplingConfig::penalties_enabled() const noexcept {
  return repeat_penalty != 1.0F || frequency_penalty != 0.0F ||
         presence_penalty != 0.0F;
}

bool SamplingConfig::uses_random_sampling() const noexcept {
  return temperature > 0.0F && (top_k != 1 || min_keep > 1);
}

bool SamplingConfig::can_use_unmodified_argmax() const noexcept {
  return !constraint && (!behaviour || !behaviour->enabled()) && temperature == 0.0F && !penalties_enabled();
}

bool SamplerState::CanSelectArgmax(TokenId token,
                                   bool penalties_applied) const {
  if (BehaviourEnabled()) {
    const auto& b=*config_.behaviour;
    const auto& st=behaviour_state_;
    const auto fail=[&](const char* reason){behaviour_state_.Fallback("candidate_path",reason);return false;};
    const bool eog=std::ranges::find(b.eog_ids,token)!=b.eog_ids.end();
    if(st.forcing() && token!=st.forced[st.force_pos])return fail("forced_greedy_choice_needs_full_row");
    if(b.production_sampling && (b.native_temperature!=config_.temperature ||
       b.native_top_k!=config_.top_k || b.native_top_p!=config_.top_p ||
       b.native_presence_penalty!=config_.presence_penalty ||
       b.native_frequency_penalty!=config_.frequency_penalty ||
       b.native_repeat_penalty!=config_.repeat_penalty))return fail("preset_counterfactual_needs_full_row");
    if(b.model_suppress_tokens && std::ranges::find(b.suppress_ids,token)!=b.suppress_ids.end())
      return fail("suppressed_greedy_choice_needs_full_row");
    if(b.logit_bias_enabled)for(auto [id,bias]:b.logit_bias)
      if(bias>0||(id==token&&bias!=0))return fail("biased_greedy_choice_needs_full_row");
    if(st.thinking && b.marker_bias!=0 && (b.marker_bias>0||
       std::ranges::find(b.marker_ids,token)!=b.marker_ids.end()))return fail("marker_greedy_choice_needs_full_row");
    if(b.loop_breaker&&st.loop.nudging())return fail("loop_nudge_needs_probability_row");
    if(!st.forcing() && eog && st.thinking && (b.think_eog_guard||
       eog_closes_think(b,st)))return fail("thinking_eog_greedy_needs_full_row");
    if(!st.thinking && !st.forcing() && !NeedsConstraintMask()) {
      if(st.boundary.deferred_stop && !eog)return fail("deferred_eog_greedy_needs_full_row");
      if(eog && ((b.eog_boundary&&!st.boundary.allowed())||b.eog_min_prob>0))
        return fail("answer_eog_greedy_needs_full_row");
    }
  }
  if (pending_sample_ || config_.temperature != 0.0F ||
      (!penalties_applied && config_.penalties_enabled()))
    return false;
  if (!config_.constraint || (BehaviourEnabled() && behaviour_state_.thinking))
    return true;
  if (token >= config_.constraint->vocabulary->size())
    throw std::out_of_range("target argmax exceeds the constraint vocabulary");
  // Keep full masks for closed JSON outputs: subsequent sampled requests
  // reuse them. Tool/prose grammars can reuse an already legal target argmax.
  return !config_.constraint->grammar->stop_only_when_complete_ &&
         config_.constraint->vocabulary->Allows(*config_.constraint->grammar,
                                                constraint_state_, token);
}

SamplingDistribution::SamplingDistribution(std::vector<Probability> entries)
    : entries_(std::move(entries)) {
  if (entries_.empty()) {
    throw std::invalid_argument("sampling distribution cannot be empty");
  }
  std::unordered_set<TokenId> tokens;
  double sum = 0.0;
  for (const auto& entry : entries_) {
    if (!std::isfinite(entry.value) || entry.value < 0.0 ||
        !tokens.insert(entry.token).second) {
      throw std::invalid_argument("sampling distribution is malformed");
    }
    sum += entry.value;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::invalid_argument("sampling distribution has no probability");
  }
  for (auto& entry : entries_) {
    entry.value /= sum;
  }
  std::erase_if(entries_, [](const auto& entry) { return entry.value == 0; });
  std::ranges::sort(entries_,
                    [](const Probability& left, const Probability& right) {
                      if (left.value == right.value) {
                        return left.token < right.token;
                      }
                      return left.value > right.value;
                    });
}

SamplingDistribution::SamplingDistribution(std::vector<Probability> entries,
                                           double total)
    : entries_(std::move(entries)) {
  if (!(total > 0) || !std::isfinite(total))
    throw std::runtime_error("logit softmax normalization failed");
  for (auto& entry : entries_)
    entry.value /= total;
  std::erase_if(entries_, [](const auto& entry) { return entry.value == 0; });
}

std::span<const Probability> SamplingDistribution::entries() const noexcept {
  return entries_;
}

TokenId SamplingDistribution::best_token() const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  return std::ranges::max_element(entries_,
                                  [](const auto& a, const auto& b) {
                                    return IsBetterProbability(b, a);
                                  })
      ->token;
}

double SamplingDistribution::probability(TokenId token) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.token == token) {
      return entry.value;
    }
  }
  return 0.0;
}

TokenId SamplingDistribution::Sample(std::uint64_t* rng_state) const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  if (entries_.size() == 1) {
    return entries_.front().token;
  }
  double sample = gufo::sampling::Uniform(rng_state);
  for (const auto& entry : entries_) {
    sample -= entry.value;
    if (sample < 0.0) {
      return entry.token;
    }
  }
  return entries_.back().token;
}

TokenId SamplingDistribution::SampleResidual(
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities,
    std::uint64_t* rng_state) const {
  if (candidate_ids.size() != candidate_probabilities.size()) {
    throw std::invalid_argument("sparse probability row is malformed");
  }
  std::unordered_map<TokenId, double> draft;
  for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
    const float probability_value = candidate_probabilities[index];
    if (!std::isfinite(probability_value) || probability_value < 0.0F) {
      throw std::invalid_argument(
          "sparse probability row contains an invalid value");
    }
    draft[candidate_ids[index]] += static_cast<double>(probability_value);
  }

  std::vector<Probability> residual;
  residual.reserve(entries_.size());
  double sum = 0.0;
  for (const auto& entry : entries_) {
    const auto found = draft.find(entry.token);
    const double q = found == draft.end() ? 0.0 : found->second;
    const double value = std::max(entry.value - q, 0.0);
    if (value > 0.0) {
      residual.push_back({.token = entry.token, .value = value});
      sum += value;
    }
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return Sample(rng_state);
  }
  return SamplingDistribution(std::move(residual), sum).Sample(rng_state);
}

static SamplingDistribution DistributionWithPenalties(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenPenalty> penalties) {
  config.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (std::isfinite(logits[index])) {
      candidates.push_back({
          .token = static_cast<TokenId>(index),
          .logit = static_cast<double>(logits[index]),
      });
    }
  }
  if (candidates.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  ApplyPenalties(&candidates, config, penalties);
  ApplyTopK(&candidates, config);
  if (config.temperature == 0.0F) {
    return SamplingDistribution(
        {{.token = candidates.front().token, .value = 1.0}});
  }
  ApplyTopP(&candidates, config);
  ApplyMinP(&candidates, config);
  const auto probabilities =
      SoftmaxWeights(candidates, static_cast<double>(config.temperature));
  std::vector<Probability> entries;
  entries.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    entries.push_back(
        {.token = candidates[index].token, .value = probabilities[index]});
  }
  return SamplingDistribution(std::move(entries));
}

SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens,
    std::span<const TokenId> generated_tokens) {
  if (config.constraint) {
    SamplerState sampler(config, prompt_tokens);
    sampler.Accept(generated_tokens);
    return sampler.Distribution(logits);
  }
  std::unordered_map<TokenId, TokenPenalty> counts;
  if (config.frequency_penalty != 0 || config.presence_penalty != 0) {
    for (const auto token : generated_tokens) {
      auto& p = counts[token];
      p.token = token;
      ++p.generated_count;
    }
  }
  if (config.repeat_penalty != 1 && config.repeat_last_n != 0) {
    const auto tail = generated_tokens.last(
        std::min(generated_tokens.size(), config.repeat_last_n));
    const auto prompt_tail = prompt_tokens.last(
        std::min(prompt_tokens.size(), config.repeat_last_n - tail.size()));
    for (auto sequence : {prompt_tail, tail}) {
      for (const auto token : sequence) {
        auto& p = counts[token];
        p.token = token;
        p.repeated = 1;
      }
    }
  }
  std::vector<TokenPenalty> penalties;
  for (const auto& [token, penalty] : counts)
    penalties.push_back(penalty);
  std::ranges::sort(penalties, {}, &TokenPenalty::token);
  return DistributionWithPenalties(logits, config, penalties);
}

SamplerState::SamplerState(SamplingConfig config,
                           std::span<const TokenId> initial_history)
    : config_(config),
      logprobs_requested_(config.logprobs >= 0),
      history_(initial_history.begin(), initial_history.end()),
      rng_state_(InitialRngState(config.seed)) {
  config_.Validate();
  if (config_.behaviour) behaviour_state_.Reset(*config_.behaviour);
  if (config_.constraint)
    constraint_state_ = config_.constraint->grammar->Start();
  TrimHistory();
  RebuildPenaltyCounts();
}

SamplerState::SamplerState(const SamplerState& other)
    : behaviour_state_(other.behaviour_state_),
      observe_behaviour_(other.observe_behaviour_),
      config_(other.config_),
      logprobs_requested_(other.logprobs_requested_),
      constraint_state_(other.constraint_state_),
      constraint_mask_(other.constraint_mask_),
      constraint_restricts_(other.constraint_restricts_),
      history_(other.history_),
      penalty_counts_(other.penalty_counts_),
      rng_state_(other.rng_state_),
      pending_sample_(other.pending_sample_) {}

SamplerState& SamplerState::operator=(const SamplerState& other) {
  if (this == &other) {
    return *this;
  }
  behaviour_state_ = other.behaviour_state_;
  observe_behaviour_ = other.observe_behaviour_;
  config_ = other.config_;
  logprobs_requested_ = other.logprobs_requested_;
  constraint_state_ = other.constraint_state_;
  constraint_mask_ = other.constraint_mask_;
  constraint_restricts_ = other.constraint_restricts_;
  history_ = other.history_;
  penalty_counts_ = other.penalty_counts_;
  candidate_scratch_.clear();
  rng_state_ = other.rng_state_;
  pending_sample_ = other.pending_sample_;
  return *this;
}

SamplerState SamplerState::WithoutConstraint() const {
  auto copy = *this;
  copy.config_.constraint.reset();
  copy.constraint_state_.clear();
  copy.constraint_mask_.reset();
  return copy;
}

bool SamplerState::NeedsConstraintMask() const {
  if (BehaviourEnabled() && behaviour_state_.thinking) return false;
  if (!config_.constraint)
    return false;
  if (!constraint_mask_) {
    constraint_mask_ = config_.constraint->Allowed(constraint_state_);
    constraint_restricts_ = std::ranges::any_of(
        *constraint_mask_, [](auto value) { return value == 0; });
  }
  return constraint_restricts_;
}

std::vector<float> SamplerState::ConstrainedLogits(
    std::span<const float> logits, std::span<const TokenId> ids) const {
  const auto mask = config_.constraint->Allowed(constraint_state_);
  if ((!ids.empty() && ids.size() != logits.size()) ||
      (ids.empty() && logits.size() != mask->size()))
    throw std::invalid_argument(
        "JSON constraint vocabulary differs from logits");
  std::vector<float> masked(logits.begin(), logits.end());
  for (std::size_t i = 0; i < masked.size(); ++i) {
    const auto token = ids.empty() ? i : ids[i];
    if (token >= mask->size())
      throw std::invalid_argument("invalid compact token ID");
    if (!(*mask)[token])
      masked[i] = -std::numeric_limits<float>::infinity();
  }
  return masked;
}

const SamplingConfig& SamplerState::config() const noexcept {
  return config_;
}

std::span<const TokenId> SamplerState::history() const noexcept {
  return history_;
}

std::uint64_t SamplerState::rng_state() const noexcept {
  return rng_state_;
}

std::uint64_t* SamplerState::mutable_rng_state() noexcept {
  return &rng_state_;
}

void SamplerState::SetRngState(std::uint64_t state) noexcept {
  rng_state_ = state;
  pending_sample_.reset();
}

void SamplerState::CopyDrawStateFrom(const SamplerState& other) {
  rng_state_ = other.rng_state_;
  pending_sample_ = other.pending_sample_;
  behaviour_state_.counters = other.behaviour_state_.counters;
  behaviour_state_.boundary.deferred_stop = other.behaviour_state_.boundary.deferred_stop;
  behaviour_state_.close_decision = false;
  behaviour_state_.deferred_at = other.behaviour_state_.deferred_at;
  behaviour_state_.close_at = other.behaviour_state_.close_at;
  behaviour_state_.guard_blocked_at = other.behaviour_state_.guard_blocked_at;
}

void SamplerState::ResetHistory(std::span<const TokenId> tokens) {
  if (config_.behaviour) behaviour_state_.Reset(*config_.behaviour);
  if (config_.constraint)
    constraint_state_ = config_.constraint->grammar->Start();
  constraint_mask_.reset();
  pending_sample_.reset();
  penalty_counts_.clear();
  history_.assign(tokens.begin(), tokens.end());
  TrimHistory();
  RebuildPenaltyCounts();
}

void SamplerState::Accept(TokenId token) {
  Accept(std::span<const TokenId>(&token, 1));
}

void SamplerState::Accept(std::span<const TokenId> tokens) {
  // Preserve the native all-unset span transaction, including failures.
  if(!BehaviourEnabled()&&config_.constraint) {
    auto next=constraint_state_;
    for(auto token:tokens)next=config_.constraint->vocabulary->Accept(
      *config_.constraint->grammar,next,token);
    if(next!=constraint_state_) {constraint_state_=std::move(next);constraint_mask_.reset();}
  }
  // O/common/sampling.cpp:940/965: lazy tool grammar is suspended in think,
  // and the end sequence is replayed so a closing sequence can own a trigger.
  for(auto token:tokens) {
    const bool before_think=BehaviourEnabled()&&behaviour_state_.thinking;
    if(BehaviourEnabled())behaviour_state_.Accept(*config_.behaviour,token,
      config_.constraint!=nullptr||config_.behaviour->loop_exempt);
    const bool after_think=BehaviourEnabled()&&behaviour_state_.thinking;
    if(!config_.constraint||!BehaviourEnabled())continue;
    auto next=constraint_state_;
    if(!before_think&&!after_think)
      next=config_.constraint->vocabulary->Accept(*config_.constraint->grammar,next,token);
    else if(before_think&&!after_think&&config_.constraint->grammar->automatic_tools_)
      for(auto end_token:behaviour_state_.matched_end)
        next=config_.constraint->vocabulary->Accept(*config_.constraint->grammar,next,end_token);
    if(next!=constraint_state_) {constraint_state_=std::move(next);constraint_mask_.reset();}
  }
  if (config_.frequency_penalty != 0 || config_.presence_penalty != 0 ||
      (BehaviourEnabled() && config_.behaviour->production_sampling)) {
    for (const auto token : tokens) {
      auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                            &TokenPenalty::token);
      if (found == penalty_counts_.end() || found->token != token)
        found = penalty_counts_.insert(found, TokenPenalty{.token = token});
      if (found->generated_count == std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("generated token count overflow");
      ++found->generated_count;
    }
  }
  if (config_.repeat_last_n != 0) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    TrimHistory();
  }
  RebuildPenaltyCounts();
}

SamplingDistribution SamplerState::Distribution(
    std::span<const float> logits) const {
  if (BehaviourEnabled()) return BehaviourDistributionFor(logits);
  if (NeedsConstraintMask()) {
    if (config_.temperature == 0) {
      config_.Validate();
      return SamplingDistribution({{SampleConstrainedGreedy(logits), 1.0}},
                                  1.0);
    }
    const auto masked = ConstrainedLogits(logits);
    return WithoutConstraint().Distribution(masked);
  }
  config_.Validate();
  if (logits.empty() || logits.size() > std::numeric_limits<TokenId>::max())
    throw std::invalid_argument("invalid sampling vocabulary size");
  if (config_.temperature == 0)
    return SamplingDistribution({{SampleGreedy(logits), 1.0}}, 1.0);
  const bool needs_floor = config_.min_p > 0 && config_.min_keep > 1;
  if (config_.top_k == 0 && config_.top_p == 1 && !needs_floor)
    return LinearDistribution(logits);
  auto working = *this;
  working.PrepareSelected(logits);
  auto& candidates = working.candidate_scratch_;
  const double maximum = candidates.front().value;
  double total = 0;
  for (auto& candidate : candidates) {
    candidate.value =
        std::exp((candidate.value - maximum) / config_.temperature);
    total += candidate.value;
  }
  return SamplingDistribution(std::move(candidates), total);
}

SamplingDistribution SamplerState::Distribution(
    std::span<const float> logits, std::span<const TokenId> token_ids) const {
  if (BehaviourEnabled()) return BehaviourDistributionFor(logits,token_ids);
  if (NeedsConstraintMask()) {
    const auto masked = ConstrainedLogits(logits, token_ids);
    return WithoutConstraint().Distribution(masked, token_ids);
  }
  if (logits.size() != token_ids.size())
    throw std::invalid_argument("compact logits and token IDs differ in size");
  std::vector<TokenPenalty> penalties;
  penalties.reserve(token_ids.size());
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    const auto found = std::ranges::lower_bound(penalty_counts_, token_ids[i],
                                                {}, &TokenPenalty::token);
    if (found != penalty_counts_.end() && found->token == token_ids[i]) {
      auto penalty = *found;
      penalty.token = static_cast<TokenId>(i);
      penalties.push_back(penalty);
    }
  }
  return DistributionWithPenalties(logits, config_, penalties);
}

std::optional<SamplingDistribution> SamplerState::DistributionFromTop(
    std::span<const float> values, std::span<const TokenId> ids,
    std::size_t vocab) const {
  if (BehaviourEnabled()) {
    if (!BehaviourTopSafe(values,ids,vocab)) return std::nullopt;
    return BehaviourDistributionFor(values,ids,vocab);
  }
  if (values.size() != ids.size() || values.empty() ||
      config_.temperature == 0 || config_.top_k <= 0 ||
      !penalty_counts_.empty() || NeedsConstraintMask()) {
    // The unmasked top list cannot prove the constrained top-k: allowed
    // tokens below the list may outrank every remaining listed token.
    // Callers retain the full-row path, which applies the grammar mask.
    return std::nullopt;
  }
  std::vector<Probability> candidates;
  candidates.reserve(values.size());
  bool complete = false;  // the list holds every finite logit of the row
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      complete = true;  // non-finite entries sort last: finite ones ran out
      continue;
    }
    if (ids[i] >= vocab) {
      return std::nullopt;
    }
    candidates.push_back(
        {.token = ids[i], .value = static_cast<double>(values[i])});
  }
  if (candidates.empty()) {
    return std::nullopt;
  }
  // The same strict order select_best produces (value, then token).
  std::ranges::sort(candidates, IsBetterProbability);
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i].token == candidates[i - 1].token) {
      return std::nullopt;
    }
  }
  const std::size_t keep = std::min(
      vocab, std::max<std::size_t>(static_cast<std::size_t>(config_.top_k),
                                   std::max<std::size_t>(config_.min_keep, 1)));
  if (!complete) {
    // Every token outside the list scores at most the list's last value, so
    // a strictly higher k-th value means the list holds the whole top-k,
    // ties included.
    if (keep >= candidates.size() ||
        !(candidates[keep - 1].value > candidates.back().value)) {
      return std::nullopt;
    }
  }
  if (candidates.size() > keep) {
    candidates.resize(keep);
  }
  auto working = *this;
  working.candidate_scratch_ = std::move(candidates);
  working.FinishSelected(0.0, false);
  auto& kept = working.candidate_scratch_;
  const double maximum = kept.front().value;
  double total = 0;
  for (auto& candidate : kept) {
    candidate.value =
        std::exp((candidate.value - maximum) / config_.temperature);
    total += candidate.value;
  }
  return SamplingDistribution(std::move(kept), total);
}

std::optional<TokenId> SamplerState::SampleFromTop(
    std::span<const float> values, std::span<const TokenId> ids,
    std::size_t vocab) {
  config_.Validate();
  // Keep pending-sample constraint validation in the full-row Sample path.
  if (NeedsConstraintMask())
    return std::nullopt;
  if (pending_sample_) {
    if (*pending_sample_ >= vocab) {
      throw std::invalid_argument("pending sample exceeds vocabulary");
    }
    const auto token = *pending_sample_;
    pending_sample_.reset();
    return token;
  }
  auto distribution = DistributionFromTop(values, ids, vocab);
  if (!distribution) {
    return std::nullopt;
  }
  const auto before=rng_state_;
  const auto token=distribution->Sample(&rng_state_);
  ObserveBehaviour(*distribution,token,before);
  return token;
}

void SamplerState::DeferSample(TokenId token) {
  if (pending_sample_) {
    throw std::logic_error("a sampled token is already pending");
  }
  pending_sample_ = token;
}

TokenId SamplerState::Sample(std::span<const float> logits) {
  config_.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }
  if (pending_sample_) {
    if (*pending_sample_ >= logits.size()) {
      throw std::invalid_argument("pending sample exceeds vocabulary");
    }
    const auto token = *pending_sample_;
    if (config_.constraint && !(BehaviourEnabled()&&behaviour_state_.thinking) &&
        !config_.constraint->Allowed(constraint_state_)->at(token))
      throw std::runtime_error("pending sample violates JSON constraint");
    pending_sample_.reset();
    return token;
  }
  if (BehaviourEnabled()) {
    const auto distribution = Distribution(logits);
    const auto before=rng_state_;
    const auto token=distribution.behaviour()->random ? distribution.Sample(&rng_state_) : distribution.best_token();
    ObserveBehaviour(distribution,token,before);
    return token;
  }
  if (config_.constraint) {
    const auto distribution = Distribution(logits);
    return config_.temperature == 0 ? distribution.best_token()
                                    : distribution.Sample(&rng_state_);
  }
  if (config_.temperature == 0.0F) {
    return SampleGreedy(logits);
  }
  return Distribution(logits).Sample(&rng_state_);
}

TokenId SamplerState::SampleResidual(
    std::span<const float> target_logits,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities) {
  return Distribution(target_logits)
      .SampleResidual(candidate_ids, candidate_probabilities, &rng_state_);
}

double SamplerState::Uniform() {
  return gufo::sampling::Uniform(&rng_state_);
}

void SamplerState::TrimHistory() {
  if (config_.repeat_last_n == 0) {
    history_.clear();
    return;
  }
  if (history_.size() > config_.repeat_last_n) {
    history_.erase(
        history_.begin(),
        history_.end() - static_cast<std::ptrdiff_t>(config_.repeat_last_n));
  }
}

void SamplerState::RebuildPenaltyCounts() {
  if (!config_.penalties_enabled() && !(BehaviourEnabled() && config_.behaviour->production_sampling))
    return;
  for (auto& penalty : penalty_counts_)
    penalty.repeated = 0;
  if (config_.repeat_penalty != 1) {
    for (const auto token : history_) {
      auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                            &TokenPenalty::token);
      if (found == penalty_counts_.end() || found->token != token)
        found = penalty_counts_.insert(found, TokenPenalty{.token = token});
      found->repeated = 1;
    }
  }
  std::erase_if(penalty_counts_, [](const auto& p) {
    return p.generated_count == 0 && p.repeated == 0;
  });
}

double SamplerState::AdjustedLogit(TokenId token, float logit) const noexcept {
  const auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                              &TokenPenalty::token);
  return found == penalty_counts_.end() || found->token != token
             ? static_cast<double>(logit)
             : Penalize(logit, config_, *found);
}

TokenId SamplerState::SampleGreedy(std::span<const float> logits) const {
  double best_logit = -std::numeric_limits<double>::infinity();
  TokenId best_token = 0;
  bool found = false;
  auto penalty = penalty_counts_.begin();
  for (std::size_t begin = 0; begin < logits.size(); begin += kSkipBlock) {
    const std::size_t end = std::min(begin + kSkipBlock, logits.size());
    while (penalty != penalty_counts_.end() && penalty->token < begin)
      ++penalty;
    // An unpenalized block that cannot exceed the best value is skipped; an
    // equal later value never replaces the earlier token.
    if (found && (penalty == penalty_counts_.end() || penalty->token >= end) &&
        static_cast<double>(BlockMaximum(logits.subspan(begin, end - begin))) <=
            best_logit)
      continue;
    for (std::size_t index = begin; index < end; ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      // Both vocab IDs and sparse penalties are sorted. Walk them together
      // instead of binary-searching the history for every vocabulary entry.
      while (penalty != penalty_counts_.end() && penalty->token < index)
        ++penalty;
      const double adjusted =
          penalty != penalty_counts_.end() && penalty->token == index
              ? Penalize(logits[index], config_, *penalty)
              : static_cast<double>(logits[index]);
      if (!std::isfinite(adjusted)) {
        throw std::runtime_error(
            "sampling penalties produced a non-finite logit");
      }
      if (!found || adjusted > best_logit) {
        best_logit = adjusted;
        best_token = static_cast<TokenId>(index);
        found = true;
      }
    }
  }
  if (!found) {
    throw std::runtime_error("logit distribution contains no finite values");
  }
  return best_token;
}

TokenId SamplerState::SampleConstrainedGreedy(
    std::span<const float> logits) const {
  const auto mask = config_.constraint->Allowed(constraint_state_);
  if (logits.size() != mask->size())
    throw std::invalid_argument(
        "JSON constraint vocabulary differs from logits");
  // Read the mask directly: greedy constraints need neither a copied vocabulary
  // of masked logits nor a cloned sampler/history. Preserve token-ID tie order.
  double best_logit = -std::numeric_limits<double>::infinity();
  TokenId best_token = 0;
  bool found = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!(*mask)[index] || !std::isfinite(logits[index]))
      continue;
    const double value = penalty_counts_.empty()
                             ? static_cast<double>(logits[index])
                             : AdjustedLogit(index, logits[index]);
    if (!std::isfinite(value))
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    if (value > best_logit) {
      best_logit = value;
      best_token = static_cast<TokenId>(index);
      found = true;
    }
  }
  if (!found)
    throw std::runtime_error("logit distribution contains no finite values");
  return best_token;
}

SamplingDistribution SamplerState::LinearDistribution(
    std::span<const float> logits) const {
  std::vector<Probability> candidates;
  candidates.reserve(logits.size());
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i]))
      continue;
    const double value =
        penalty_counts_.empty()
            ? static_cast<double>(logits[i])
            : AdjustedLogit(static_cast<TokenId>(i), logits[i]);
    if (!std::isfinite(value))
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    maximum = std::max(maximum, value);
    candidates.push_back({static_cast<TokenId>(i), value});
  }
  if (candidates.empty())
    throw std::runtime_error("logit distribution contains no finite values");
  const double threshold =
      config_.min_p > 0
          ? maximum + static_cast<double>(config_.temperature) *
                          std::log(static_cast<double>(config_.min_p))
          : -std::numeric_limits<double>::infinity();
  double total = 0;
  for (auto& candidate : candidates) {
    candidate.value =
        candidate.value >= threshold
            ? std::exp((candidate.value - maximum) / config_.temperature)
            : 0;
    total += candidate.value;
  }
  return SamplingDistribution(std::move(candidates), total);
}

void SamplerState::PrepareSelected(std::span<const float> logits) {
  candidate_scratch_.clear();
  // Vocabulary IDs and sparse penalties are both sorted. Walk them together
  // instead of searching the penalties for every vocabulary entry.
  const auto adjusted_at = [&](std::size_t index, auto& penalty) {
    while (penalty != penalty_counts_.end() && penalty->token < index)
      ++penalty;
    if (penalty == penalty_counts_.end() || penalty->token != index)
      return static_cast<double>(logits[index]);
    const double adjusted = Penalize(logits[index], config_, *penalty);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    return adjusted;
  };
  const auto for_each_adjusted = [&](auto&& visit) {
    auto penalty = penalty_counts_.begin();
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (std::isfinite(logits[index]))
        visit(index, adjusted_at(index, penalty));
    }
  };
  const auto select_best = [&](std::size_t limit) {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(limit);
    auto penalty = penalty_counts_.begin();
    const auto offer = [&](std::size_t index) {
      if (!std::isfinite(logits[index])) {
        return;
      }
      const Probability candidate{
          .token = static_cast<TokenId>(index),
          .value = adjusted_at(index, penalty),
      };
      if (candidate_scratch_.size() < limit) {
        candidate_scratch_.push_back(candidate);
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      } else if (IsBetterProbability(candidate, candidate_scratch_.front())) {
        std::pop_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                      IsBetterProbability);
        candidate_scratch_.back() = candidate;
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      }
    };
    for (std::size_t begin = 0; begin < logits.size(); begin += kSkipBlock) {
      const std::size_t end = std::min(begin + kSkipBlock, logits.size());
      while (penalty != penalty_counts_.end() && penalty->token < begin)
        ++penalty;
      // Once the heap is full, skip an unpenalized block whose largest logit
      // cannot beat the weakest kept entry. Every kept entry has a lower token
      // ID, so an equal later value loses the tie as well. NaN never raises the
      // block maximum and +inf keeps the block for the exact per-token path.
      if (candidate_scratch_.size() == limit &&
          (penalty == penalty_counts_.end() || penalty->token >= end)) {
        const float maximum = BlockMaximum(logits.subspan(begin, end - begin));
        if (static_cast<double>(maximum) <= candidate_scratch_.front().value)
          continue;
      }
      for (std::size_t index = begin; index < end; ++index)
        offer(index);
    }
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };
  const auto select_all = [&]() {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(logits.size());
    for_each_adjusted([&](std::size_t index, double adjusted) {
      candidate_scratch_.push_back({
          .token = static_cast<TokenId>(index),
          .value = adjusted,
      });
    });
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };

  double full_softmax_sum = 0.0;
  bool has_full_softmax_sum = false;
  if (config_.top_k > 0) {
    const std::size_t keep = std::min(
        logits.size(),
        std::max<std::size_t>(static_cast<std::size_t>(config_.top_k),
                              std::max<std::size_t>(config_.min_keep, 1)));
    select_best(keep);
  } else if (config_.top_p < 1.0F && logits.size() > 1024) {
    double maximum = -std::numeric_limits<double>::infinity();
    bool found = false;
    for_each_adjusted([&](std::size_t, double adjusted) {
      maximum = std::max(maximum, adjusted);
      found = true;
    });
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }
    for_each_adjusted([&](std::size_t, double adjusted) {
      full_softmax_sum += std::exp((adjusted - maximum) / config_.temperature);
    });
    if (!(full_softmax_sum > 0.0) || !std::isfinite(full_softmax_sum)) {
      throw std::runtime_error("logit softmax normalization failed");
    }
    has_full_softmax_sum = true;

    constexpr std::size_t initial_top_p_candidates = 256;
    const std::size_t initial_keep = std::min(
        logits.size(), std::max(initial_top_p_candidates,
                                std::max<std::size_t>(config_.min_keep, 1)));
    select_best(initial_keep);

    double selected_mass = 0.0;
    for (const auto& candidate : candidate_scratch_) {
      selected_mass +=
          std::exp((candidate.value - maximum) / config_.temperature);
    }
    if (selected_mass < static_cast<double>(config_.top_p) * full_softmax_sum) {
      select_all();
    }
  } else {
    select_all();
  }
  FinishSelected(full_softmax_sum, has_full_softmax_sum);
}

void SamplerState::FinishSelected(double full_softmax_sum,
                                  bool has_full_softmax_sum) {
  if (candidate_scratch_.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  std::size_t keep = candidate_scratch_.size();
  const std::size_t minimum = MinimumKept(config_, candidate_scratch_.size());
  if (config_.top_p < 1.0F && keep > 1) {
    const double maximum = candidate_scratch_.front().value;
    double sum = full_softmax_sum;
    if (!has_full_softmax_sum) {
      sum = 0.0;
      for (const auto& candidate : candidate_scratch_) {
        sum += std::exp((candidate.value - maximum) / config_.temperature);
      }
    }
    const double target = static_cast<double>(config_.top_p) * sum;
    double cumulative = 0.0;
    std::size_t top_p_keep = 0;
    while (top_p_keep < keep) {
      cumulative += std::exp((candidate_scratch_[top_p_keep].value - maximum) /
                             config_.temperature);
      ++top_p_keep;
      if (top_p_keep >= minimum && cumulative >= target) {
        break;
      }
    }
    keep = top_p_keep;
  }
  if (config_.min_p > 0.0F && keep > 1) {
    const double threshold = candidate_scratch_.front().value +
                             static_cast<double>(config_.temperature) *
                                 std::log(static_cast<double>(config_.min_p));
    std::size_t min_p_keep = 0;
    while (min_p_keep < keep &&
           candidate_scratch_[min_p_keep].value >= threshold) {
      ++min_p_keep;
    }
    keep = std::min(keep, std::max(min_p_keep, minimum));
  }
  candidate_scratch_.resize(keep);
}


SamplerState SamplerState::WithoutBehaviour() const {
  auto copy=*this; copy.config_.behaviour.reset(); return copy;
}
bool SamplerState::BehaviourEnabled() const noexcept {
  return config_.behaviour && config_.behaviour->enabled();
}
std::vector<std::string> SamplerState::BehaviourSummary() const {
  auto lines=behaviour_state_.Summary();
  if(!BehaviourEnabled()) return lines;
  const auto& b=*config_.behaviour;
  lines.push_back("BEHAVIOUR_ENGAGE name=guard_escalation taken="+std::string(behaviour_state_.guard_escalation_taken?"1":"0")+
    " fallback=0 reason="+(behaviour_state_.guard_escalation_taken?"blocked_attempts_reached_k":"none")+" scope=request");
  std::vector<std::pair<std::string,bool>> enabled={
    {"think_eog_guard",b.think_eog_guard},{"eog_boundary",b.eog_boundary},
    {"eog_min_prob",b.eog_min_prob>0},{"eog_defer",b.eog_defer_p>0},
    {"spec_eog_stop",b.spec_eog_stop},{"loop_breaker",b.loop_breaker},
    {"loop_transition",b.loop_breaker&&b.loop_transition!=0},
    {"loop_eog_close",b.loop_eog_close},{"reasoning_budget",b.reasoning_budget_enabled},
    {"marker_bias",b.marker_bias!=0||!b.marker_bias_by_effort.empty()||b.request_marker_bias.has_value()},{"logit_bias",b.logit_bias_enabled},
    {"production_sampling",b.production_sampling},{"model_suppress_tokens",b.model_suppress_tokens}};
  for(const auto& [name,on]:enabled) {
    if(!on) continue;
    const bool seen=std::ranges::any_of(behaviour_state_.counters,[&](const auto& kv){return kv.first.first==name;});
    if(!seen) {
      const bool exempt=(config_.constraint||b.loop_exempt)&&
        (name=="loop_breaker"||name=="loop_transition"||name=="loop_eog_close");
      lines.push_back("BEHAVIOUR_ENGAGE name="+name+" taken=0 fallback=0 reason="+
        (exempt?"grammar_or_tool_request":"none")+" scope=request");
    }
  }
  return lines;
}
void SamplerState::RecordSpecEogStop(std::size_t skipped) {
  // Native gufo already truncates at a target EOG, accepted or residual.
  // Preserve that contract: skipped native rows are not a new intervention.
  (void)skipped;
}
namespace {
bool Contains(std::span<const TokenId> ids,TokenId id) {
  return std::ranges::find(ids,id)!=ids.end();
}
// A lease owns its vector until all users of that row are finished. Nested
// counterfactual / candidate evaluations borrow separate buffers. No scratch
// is copied with a sampler or shared between threads or speculative states.
struct BehaviourScratch {
  static std::vector<std::vector<float>>& Pool() {
    thread_local std::vector<std::vector<float>> pool;
    return pool;
  }
  std::vector<float> logits;
  BehaviourScratch() {
    auto& pool = Pool();
    if (!pool.empty()) {
      logits = std::move(pool.back());
      pool.pop_back();
    }
  }
  BehaviourScratch(const BehaviourScratch&) = delete;
  BehaviourScratch& operator=(const BehaviourScratch&) = delete;
  BehaviourScratch(BehaviourScratch&&) noexcept = default;
  BehaviourScratch& operator=(BehaviourScratch&& other) noexcept {
    logits.swap(other.logits);
    return *this;
  }
  ~BehaviourScratch() {
    if (logits.capacity() == 0) return;
    logits.clear();
    // Recycling must never change exception behaviour (including unwinding).
    try { Pool().push_back(std::move(logits)); } catch (...) {}
  }
};
struct BehaviourLogits : BehaviourScratch {
  struct Stage {
    std::string name, reason;
    bool counterfactual_identical{false};
  };
  std::vector<Stage> stages;
  float temperature{0};
  bool defer{false};
};
// O/common/sampling.cpp:1096-1188: raw T1 canonical top-k denominator,
// explicit EOG bias exemptions, grammar/channel exemptions and deferred stop.
BehaviourLogits TransformBehaviour(std::span<const float> raw,
    std::span<const TokenId> ids,const SamplingConfig& c,
    const BehaviourState& st,bool grammar_active) {
  const auto& b=*c.behaviour;
  BehaviourLogits result;
  auto& current = result.logits;
  current.assign(raw.begin(), raw.end());
  float temperature=c.temperature;
  const auto id=[&](std::size_t i){return ids.empty()?static_cast<TokenId>(i):ids[i];};
  const auto stage = [&](std::string name, std::string reason, bool identical = false) {
    result.stages.push_back({std::move(name), std::move(reason), identical});
  };
  const auto mask = [&](TokenId token) {
    bool changed_finite = false;
    const auto at = [&](std::size_t i) {
      changed_finite |= std::isfinite(current[i]);
      current[i] = -INFINITY;
    };
    if (ids.empty()) {
      if (token < current.size()) at(token);
    } else {
      for (std::size_t i = 0; i < current.size(); ++i)
        if (id(i) == token) at(i);
    }
    return changed_finite;
  };
  const auto exempt=[&](TokenId token){return b.logit_bias_enabled&&std::ranges::any_of(b.logit_bias,[&](auto v){return v.first==token;});};
  // raw top-k is independent of a device list's capacity, and identical on AR
  // and every target verification row. This subset defines the conservative
  // upper-bound policy; exact mode uses the complete raw vocabulary instead.
  BehaviourScratch denominator;
  auto& denom = denominator.logits;
  bool denominator_ready = false;
  double maximum = -INFINITY, sum = 0;
  // Nothing else consumes this raw denominator. Preserve its arithmetic and
  // ordering, but materialize it only on the first finite probability query.
  const auto ensure_denominator = [&] {
    if (denominator_ready) return;
    denom.assign(raw.begin(), raw.end());
    std::erase_if(denom, [](float v) { return !std::isfinite(v); });
    if (!b.eog_min_prob_exact && c.top_k > 0 &&
        denom.size() > static_cast<std::size_t>(c.top_k)) {
      std::partial_sort(denom.begin(), denom.begin() + c.top_k, denom.end(),
                        std::greater<>{});
      denom.resize(c.top_k);
    } else std::ranges::sort(denom, std::greater<>{});
    maximum = denom.empty() ? -INFINITY : denom.front();
    for (auto v : denom) sum += std::exp(static_cast<double>(v) - maximum);
    denominator_ready = true;
  };
  const auto raw_probability = [&](TokenId token) {
    if (ids.empty()) {
      if (token >= raw.size() || !std::isfinite(raw[token])) return 0.0;
      ensure_denominator();
      if (sum <= 0) return 0.0;
      return std::min(1.0, std::exp(static_cast<double>(raw[token]) - maximum) / sum);
    }
    for (std::size_t i = 0; i < raw.size(); ++i) {
      if (id(i) == token && std::isfinite(raw[i])) {
        ensure_denominator();
        if (sum > 0)
          return std::min(1.0, std::exp(static_cast<double>(raw[i]) - maximum) / sum);
      }
    }
    return 0.0;
  };
  if(b.logit_bias_enabled && !b.logit_bias.empty()) {
    for(auto [token,bias]:b.logit_bias)for(std::size_t i=0;i<current.size();++i)
      if(id(i)==token)current[i]+=bias;
    stage("logit_bias","explicit_token_bias_changed_choice");
  }
  // O/common/sampling.cpp:664-680 merges GGUF suppression (-inf) after
  // user bias, so an explicit finite bias cannot revive a suppressed token.
  if(b.model_suppress_tokens&&!b.suppress_ids.empty()) {
    for(auto token:b.suppress_ids)mask(token);
    stage("model_suppress_tokens","gguf_model_suppressed_token_changed_choice");
  }
  if(st.thinking&&b.marker_bias!=0&&!b.marker_ids.empty()&&!st.forcing()) {
    if (ids.empty()) {
      // The request builder creates this list from a std::set: once per token.
      assert(std::unordered_set<TokenId>(b.marker_ids.begin(), b.marker_ids.end()).size() ==
             b.marker_ids.size());
      for (auto token : b.marker_ids)
        if (token < current.size()) current[token] += static_cast<float>(b.marker_bias);
    } else {
      for (std::size_t i = 0; i < current.size(); ++i)
        if (Contains(b.marker_ids, id(i))) current[i] += static_cast<float>(b.marker_bias);
    }
    stage("marker_bias","thinking_marker_bias_changed_choice");
  }
  if(!st.thinking && !st.forcing() && !grammar_active) {
    const bool boundary_ok=!b.eog_boundary||st.boundary.allowed();
    if(b.eog_boundary&&!boundary_ok) {
      if(b.eog_defer_p>0&&!st.boundary.deferred_stop)
        for(auto token:b.eog_ids)if(!exempt(token)&&raw_probability(token)>=b.eog_defer_p)result.defer=true;
      bool changed_finite = false;
      for (auto token : b.eog_ids)
        if (!exempt(token)) changed_finite |= mask(token);
      // Removing boundary can activate eog_defer when a stop is pending.
      stage("eog_boundary", "visible_text_not_at_boundary",
            !changed_finite && !st.boundary.deferred_stop);
    }
    if(boundary_ok&&st.boundary.deferred_stop&&!b.eog_ids.empty() &&
       [&] {
         if (ids.empty()) {
           for (auto token : b.eog_ids)
             if (token < current.size() && std::isfinite(current[token])) return true;
         } else {
           for (std::size_t i = 0; i < current.size(); ++i)
             if (Contains(b.eog_ids, id(i)) && std::isfinite(current[i])) return true;
         }
         return false;
       }()) {
      if (ids.empty()) {
        // Save before filling; duplicates are restored from the same original
        // value. Non-finite EOG bits, including NaNs, survive unchanged.
        std::vector<std::pair<TokenId, float>> saved;
        for (auto token : b.eog_ids)
          if (token < current.size()) saved.push_back({token, current[token]});
        std::ranges::fill(current, -INFINITY);
        for (auto [token, value] : saved) current[token] = value;
      } else {
        for (std::size_t i = 0; i < current.size(); ++i)
          if (!Contains(b.eog_ids, id(i))) current[i] = -INFINITY;
      }
      stage("eog_defer","confident_eog_reached_visible_boundary");
    } else if(b.eog_min_prob>0) {
      bool changed_finite = false;
      for (auto token : b.eog_ids)
        if (!exempt(token) && raw_probability(token) < b.eog_min_prob)
          changed_finite |= mask(token);
      stage("eog_min_prob", b.eog_min_prob_exact ? "raw_full_probability_below_threshold" :
            "raw_topk_upper_bound_below_threshold", !changed_finite);
    }
  }
  if(st.thinking&&b.think_eog_guard&&!st.forcing()&&!eog_closes_think(b,st)) {
    bool changed_finite = false;
    for (auto token : b.eog_ids) changed_finite |= mask(token);
    stage("think_eog_guard", "eog_inside_open_think", !changed_finite);
  }
  if(b.loop_breaker&&st.loop.nudging()&&!c.constraint&&!b.loop_exempt&&!st.forcing()) {
    temperature=c.temperature+0.5F;
    stage("loop_breaker","cycle_nudge_temperature_plus_0.5");
  }
  if(st.forcing()) {
    const auto forced=st.forced[st.force_pos];
    for(std::size_t i=0;i<current.size();++i)current[i]=id(i)==forced?0.0F:-INFINITY;
    stage(st.force_reason,"forced_transition_token_"+std::to_string(forced));
  }
  result.temperature=temperature;
  return result;
}
}
TokenLogprob SamplerState::RawLogprobs(std::span<const float> logits,
                                     TokenId token, std::size_t top_n) const {
  if (logits.empty() || token >= logits.size() || top_n > 20)
    throw std::invalid_argument("invalid logprob vocabulary or top_logprobs");
  auto native = WithoutBehaviour();
  std::optional<BehaviourLogits> transformed;
  if (BehaviourEnabled()) {
    if (behaviour_state_.thinking)
      native.config_.constraint.reset();
    transformed.emplace(TransformBehaviour(logits, {}, config_, behaviour_state_,
                                           NeedsConstraintMask()));
    logits = transformed->logits;
  }
  const auto mask = native.config_.constraint
                        ? native.config_.constraint->Allowed(constraint_state_)
                        : nullptr;
  if (mask && mask->size() != logits.size())
    throw std::invalid_argument("JSON constraint vocabulary differs from logits");
  std::vector<Probability> scores;
  scores.reserve(logits.size());
  double maximum = -std::numeric_limits<double>::infinity();
  double selected = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i]) || (mask && !(*mask)[i]))
      continue;
    const auto id = static_cast<TokenId>(i);
    const double value = native.AdjustedLogit(id, logits[i]);
    if (!std::isfinite(value))
      throw std::runtime_error("sampling penalties produced a non-finite logit");
    maximum = std::max(maximum, value);
    if (id == token)
      selected = value;
    scores.push_back({id, value});
  }
  if (scores.empty())
    throw std::runtime_error("logit distribution contains no finite values");
  double sum = 0;
  for (const auto& score : scores)
    sum += std::exp(score.value - maximum);
  // Subtract the peak first to preserve small logprobs on large logit rows.
  const double normalization = std::log(sum);
  TokenLogprob result{token, selected - maximum - normalization, {}};
  if (top_n != 0) {
    const auto keep = std::min(top_n, scores.size());
    std::partial_sort(scores.begin(), scores.begin() + keep, scores.end(),
                      IsBetterProbability);
    scores.resize(keep);
    for (auto& score : scores)
      score.value = score.value - maximum - normalization;
    result.top_logprobs = std::move(scores);
  }
  return result;
}

SamplerState SamplerState::BehaviourWithoutIntervention(std::string_view name) const {
  const auto& b=*config_.behaviour;
  auto variant=*this;variant.observe_behaviour_=false;
  auto disabled=std::make_shared<BehaviourConfig>(b);
  if(name=="production_sampling") {
    disabled->production_sampling=false;
    variant.config_.temperature=b.native_temperature;variant.config_.top_k=b.native_top_k;
    variant.config_.top_p=b.native_top_p;variant.config_.min_p=b.native_min_p;
    variant.config_.repeat_penalty=b.native_repeat_penalty;
    variant.config_.frequency_penalty=b.native_frequency_penalty;
    variant.config_.presence_penalty=b.native_presence_penalty;
    variant.config_.min_keep=b.native_min_keep;variant.config_.repeat_last_n=b.native_repeat_last_n;
    variant.RebuildPenaltyCounts();
  } else if(name=="logit_bias") disabled->logit_bias_enabled=false;
  else if(name=="model_suppress_tokens")disabled->model_suppress_tokens=false;
  else if(name=="marker_bias")disabled->marker_bias=0;
  else if(name=="think_eog_guard")disabled->think_eog_guard=false;
  else if(name=="eog_boundary")disabled->eog_boundary=false;
  else if(name=="eog_min_prob")disabled->eog_min_prob=0;
  else if(name=="eog_defer") {
    disabled->eog_defer_p=0;variant.behaviour_state_.boundary.deferred_stop=false;
  } else if(name=="loop_eog_close")disabled->loop_eog_close=false;
  else if(name=="reasoning_budget")disabled->reasoning_budget_enabled=false;
  else if(name=="loop_breaker") {
    disabled->loop_breaker=false;variant.behaviour_state_.loop.nudge_left=0;
  } else if(name=="loop_transition")disabled->loop_transition=0;
  if(variant.behaviour_state_.forcing()&&variant.behaviour_state_.force_reason==name) {
    if(name=="loop_transition")variant.behaviour_state_.Force(b.budget_forced,"loop_breaker");
    else {variant.behaviour_state_.forced.clear();variant.behaviour_state_.force_pos=0;}
  }
  variant.config_.behaviour=std::move(disabled);
  return variant;
}
SamplingDistribution SamplerState::BehaviourDistributionFor(
    std::span<const float> logits,std::span<const TokenId> ids,std::size_t vocab) const {
  if(logits.empty()||(!ids.empty()&&logits.size()!=ids.size()))
    throw std::invalid_argument("invalid behaviour sampling row");
  const bool global=!ids.empty()&&vocab>0;
  auto native=WithoutBehaviour();
  if(behaviour_state_.thinking)native.config_.constraint.reset();
  const auto build_with=[&](SamplerState working,std::span<const float> row){
    if(global) {
      std::vector<std::size_t> order(ids.size());
      for(std::size_t i=0;i<order.size();++i)order[i]=i;
      std::ranges::sort(order,[&](auto a,auto z){return ids[a]<ids[z];});
      std::vector<TokenId> ordered_ids;std::vector<float> ordered_logits;
      for(auto i:order){ordered_ids.push_back(ids[i]);ordered_logits.push_back(row[i]);}
      auto local=working.Distribution(ordered_logits,ordered_ids);
      std::vector<Probability> entries;
      for(auto e:local.entries())entries.push_back({ordered_ids[e.token],e.value});
      return SamplingDistribution(std::move(entries));
    }
    return ids.empty()?working.Distribution(row):working.Distribution(row,ids);
  };
  const auto build=[&](std::span<const float> row,float temperature){
    auto working=native;working.config_.temperature=temperature;
    return build_with(std::move(working),row);
  };
  auto metadata=std::make_shared<BehaviourDistribution>();
  const auto& b=*config_.behaviour;
  const auto transforms=TransformBehaviour(logits,ids,config_,behaviour_state_,NeedsConstraintMask());
  // Filtering is applied once to FINAL transformed logits. Intermediates can
  // legitimately have empty support that a later mandatory force rescues.
  auto current=build(transforms.logits,transforms.temperature);
  metadata->random=transforms.temperature>0;
  if(b.production_sampling)
    metadata->stages.push_back({"production_sampling","production_profile_changed_choice",{},{}});
  for(const auto& stage:transforms.stages)
    metadata->stages.push_back({stage.name,stage.reason,{},{}});
  metadata->defer_stop=transforms.defer;
  // O/reasoning-budget.cpp:402 is a selected-token rewrite. Its exact port is
  // a push-forward of the entire filtered target probability mass BEFORE
  // min(1,p/q) and residual sampling. Draft q can remain the native proposal.
  if(eog_closes_think(b,behaviour_state_)&&behaviour_state_.thinking&&
     !behaviour_state_.forcing()&&!b.think_end.empty()) {
    const auto end=b.think_end.front();
    std::vector<Probability> entries;double moved=0;
    for(auto e:current.entries()) {
      const auto token=ids.empty()||global?e.token:ids[e.token];
      if(Contains(b.eog_ids,token))moved+=e.value;else entries.push_back(e);
    }
    if(moved>0) {
      TokenId destination=end;
      if(!ids.empty()&&!global) {
        const auto found=std::ranges::find(ids,end);
        if(found==ids.end())throw std::runtime_error("compact EOG close target not representable");
        destination=static_cast<TokenId>(found-ids.begin());
      }
      auto found=std::ranges::find(entries,destination,&Probability::token);
      if(found==entries.end())entries.push_back({destination,moved});else found->value+=moved;
      auto next=SamplingDistribution(std::move(entries));
      metadata->stages.push_back({"loop_eog_close","mode2_open_think_eog_mass_redirected",current,next,metadata->random,metadata->random});
      current=std::move(next);metadata->close_eog=true;
    }
  }
  if(observe_behaviour_) {
    // Counters compare FINAL configured policy against the same final policy
    // with precisely this intervention removed, with every other gate/bias
    // retained. An earlier intermediate choice does not establish engagement.
    for(auto& stage:metadata->stages) {
      const bool identical = std::ranges::any_of(transforms.stages, [&](const auto& transform) {
        return transform.name == stage.name && transform.counterfactual_identical;
      });
      if (identical && (stage.name != "think_eog_guard" || !config_.constraint)) {
        // Finite support and every subsequent operation are unchanged. Keep
        // the stage and its exact before/after distributions for observation.
        stage.before = current;
        stage.before.behaviour_.reset();
        stage.before_random = metadata->random;
        stage.after = current;
        stage.after_random = metadata->random;
        continue;
      }
      auto variant=BehaviourWithoutIntervention(stage.name);
      try {
        stage.before=variant.BehaviourDistributionFor(logits,ids,vocab);
        stage.before_random=stage.before.behaviour()->random;
        stage.before.behaviour_.reset();
      } catch(const std::runtime_error& error) {
        const std::string_view message=error.what();
        if(message!="logit distribution contains no finite values" &&
           message!="sampling filters removed every token" &&
           message!="JSON constraint has no valid token")throw;
        stage.before_valid=false;
        stage.reason+="_counterfactual_no_legal_token";
      }
      stage.after=current;stage.after_random=metadata->random;
    }
  } else metadata->stages.clear();
  current.behaviour_=std::move(metadata);return current;
}
bool SamplerState::BehaviourTopSafe(std::span<const float> values,
    std::span<const TokenId> ids,std::size_t vocab) const {
  const auto fail=[&](std::string reason){behaviour_state_.Fallback("candidate_path",std::move(reason));return false;};
  if(values.size()!=ids.size()||values.empty()||config_.top_k<=0||config_.temperature<=0||
     (observe_behaviour_&&config_.penalties_enabled()))return fail("native_candidate_policy_not_applicable");
  if(NeedsConstraintMask())return fail("grammar_mask_needs_full_row");
  const auto& b=*config_.behaviour;
  if(observe_behaviour_) {
    const auto transforms=TransformBehaviour(values,ids,config_,behaviour_state_,false);
    std::vector<std::string> names;
    for(const auto& stage:transforms.stages)names.push_back(stage.name);
    if(b.production_sampling)names.push_back("production_sampling");
    if(eog_closes_think(b,behaviour_state_)&&behaviour_state_.thinking)
      names.push_back("loop_eog_close");
    for(const auto& name:names) {
      auto variant=BehaviourWithoutIntervention(name);
      if(!variant.BehaviourTopSafe(values,ids,vocab))
        return fail("counterfactual_"+name+"_topk_not_proven");
    }
  }
  if(b.eog_min_prob_exact&&(b.eog_min_prob>0||b.eog_defer_p>0))return fail("exact_eog_needs_full_row");
  for(auto id:ids)if(id>=vocab)return fail("invalid_device_candidate_id");
  std::unordered_set<TokenId> unique;
  for(auto id:ids)if(!unique.insert(id).second)return fail("duplicate_device_candidate_id");
  for(auto value:values)if(std::isnan(value)||value==INFINITY)return fail("invalid_device_candidate_logit");
  const bool complete=values.size()>=vocab||std::ranges::any_of(values,[](float v){return !std::isfinite(v);});
  const std::size_t keep=std::min(vocab,std::max<std::size_t>(config_.top_k,std::max<std::size_t>(config_.min_keep,1)));
  // The raw top-k denominator must itself be complete, even after a mask or
  // bias changes the transformed ranking.
  auto raw=std::vector<float>(values.begin(),values.end());std::ranges::sort(raw,std::greater<>{});
  if(!complete&&(keep>=raw.size()||!(raw[keep-1]>raw.back())))return fail("raw_topk_not_proven");
  if(b.production_sampling) {
    auto baseline=WithoutBehaviour();
    baseline.config_.temperature=b.native_temperature;baseline.config_.top_k=b.native_top_k;
    baseline.config_.top_p=b.native_top_p;baseline.config_.min_p=b.native_min_p;
    baseline.config_.repeat_penalty=b.native_repeat_penalty;
    baseline.config_.frequency_penalty=b.native_frequency_penalty;
    baseline.config_.presence_penalty=b.native_presence_penalty;
    if(b.native_repeat_last_n!=config_.repeat_last_n || b.native_top_k<=0)
      return fail("preset_counterfactual_topk_needs_full_row");
    baseline.RebuildPenaltyCounts();
    std::vector<double> ranked_native;
    double unseen_native=complete?-INFINITY:raw.back();
    double largest_boost=0;
    for(const auto& penalty:baseline.penalty_counts_)
      largest_boost=std::max(largest_boost,
        std::max(0.0,-static_cast<double>(b.native_frequency_penalty)*penalty.generated_count)+
        std::max(0.0,-static_cast<double>(b.native_presence_penalty)));
    if(b.native_repeat_penalty!=1) {
      // Include either sign at the boundary: repetition scales positive and
      // negative logits differently, and unseen raw values are bounded above.
      unseen_native=std::max(unseen_native,
        unseen_native<=0?unseen_native*b.native_repeat_penalty:unseen_native/b.native_repeat_penalty);
    }
    unseen_native+=largest_boost;
    for(std::size_t i=0;i<values.size();++i)if(std::isfinite(values[i]))
      ranked_native.push_back(baseline.AdjustedLogit(ids[i],values[i]));
    std::ranges::sort(ranked_native,std::greater<>{});
    const auto native_keep=std::min(vocab,std::max<std::size_t>(b.native_top_k,std::max<std::size_t>(b.native_min_keep,1)));
    if(!complete&&(ranked_native.size()<native_keep||!(ranked_native[native_keep-1]>unseen_native)))
      return fail("preset_counterfactual_topk_not_proven");
  }
  if(behaviour_state_.forcing()) {
    if(!Contains(ids,behaviour_state_.forced[behaviour_state_.force_pos]))return fail("forced_token_not_in_device_candidates");
    return true;
  }
  if((behaviour_state_.boundary.deferred_stop&&behaviour_state_.boundary.allowed())||
     (eog_closes_think(b,behaviour_state_)&&behaviour_state_.thinking)) {
    for(auto token:b.eog_ids)if(!Contains(ids,token))return fail("forced_eog_not_in_device_candidates");
    if(eog_closes_think(b,behaviour_state_)&&behaviour_state_.thinking&&
       !b.think_end.empty()&&!Contains(ids,b.think_end.front()))return fail("forced_token_not_in_device_candidates");
  }
  double unseen_extra=0;
  if(b.logit_bias_enabled)for(auto [token,bias]:b.logit_bias)if(bias>0) {
    if(!Contains(ids,token))return fail("positive_bias_token_not_in_device_candidates");
    // All positively biased IDs are represented; unseen tokens receive zero
    // positive user bias. Negative bias can only lower their upper bound.
  }
  if(behaviour_state_.thinking&&b.marker_bias>0) {
    for(auto token:b.marker_ids)if(!Contains(ids,token))return fail("positive_bias_token_not_in_device_candidates");
    // All marker IDs are represented, so unseen marker boost is also zero.
  }
  const auto transformed=TransformBehaviour(values,ids,config_,behaviour_state_,false);
  auto ranked_float=transformed.logits;
  std::vector<double> ranked(ranked_float.begin(),ranked_float.end());
  double unseen=raw.back()+unseen_extra;
  if(config_.penalties_enabled()) {
    if(config_.repeat_penalty!=1)
      unseen=std::max(unseen,unseen<=0?unseen*config_.repeat_penalty:unseen/config_.repeat_penalty);
    double boost=0;
    for(const auto& penalty:penalty_counts_)
      boost=std::max(boost,std::max(0.0,-static_cast<double>(config_.frequency_penalty)*penalty.generated_count)+
        std::max(0.0,-static_cast<double>(config_.presence_penalty)));
    unseen+=boost;
    for(std::size_t i=0;i<ranked.size();++i)if(std::isfinite(ranked[i]))
      ranked[i]=AdjustedLogit(ids[i],ranked_float[i]);
  }
  std::ranges::sort(ranked,std::greater<>{});
  std::erase_if(ranked,[](double v){return !std::isfinite(v);});
  if(!complete&&(ranked.size()<keep||!(ranked[keep-1]>unseen)))return fail("transformed_topk_not_proven");
  return true;
}
void SamplerState::ObserveBehaviour(const SamplingDistribution& distribution,
    TokenId actual,std::uint64_t rng_before,std::span<const TokenId> proposal_ids,
    std::span<const float> proposal_probabilities,TokenId proposed,double q) {
  const auto& metadata=distribution.behaviour();if(!metadata)return;
  const auto decide=[&](const SamplingDistribution& d,bool random){
    auto rng=rng_before;
    if(q>0) {
      if(gufo::sampling::Uniform(&rng)*q<d.probability(proposed))return proposed;
      return d.SampleResidual(proposal_ids,proposal_probabilities,&rng);
    }
    return random?d.Sample(&rng):d.best_token();
  };
  for(const auto& stage:metadata->stages) {
    if (!stage.before_valid) {
      // An undefined baseline cannot prove that a different token was chosen.
      // Report the unavailable counterfactual rather than fabricate engagement.
      behaviour_state_.Fallback("counterfactual",stage.name+"_no_legal_token");
    } else if(decide(stage.after,stage.after_random)==actual&&
              decide(stage.before,stage.before_random)!=actual) {
      behaviour_state_.Taken(stage.name,stage.reason);
      if(stage.name=="think_eog_guard")
        behaviour_state_.guard_blocked_at.push_back(behaviour_state_.accepted_count+1);
    }
  }
  if(metadata->defer_stop) {
    behaviour_state_.boundary.deferred_stop=true;
    behaviour_state_.deferred_at.push_back(behaviour_state_.accepted_count+1);
  }
  if(metadata->close_eog && config_.behaviour && !config_.behaviour->think_end.empty() &&
      actual==config_.behaviour->think_end.front())
  {
    // Acceptance journal, rather than an early shared boolean, controls
    // the exact token ordinal at which a multi-token close starts.
    behaviour_state_.close_at.push_back(behaviour_state_.accepted_count+1);
  }
}

std::uint64_t NextRandom(std::uint64_t* state) {
  if (state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  std::uint64_t value = *state;
  if (value == 0) {
    value = UINT64_C(0x9e3779b97f4a7c15);
  }
  value ^= value >> 12U;
  value ^= value << 25U;
  value ^= value >> 27U;
  *state = value;
  return value * UINT64_C(0x2545f4914f6cdd1d);
}

double Uniform(std::uint64_t* state) {
  const std::uint64_t value = NextRandom(state);
  return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

TokenId SampleLogits(std::span<const float> logits, float temperature,
                     std::uint64_t* rng_state) {
  if (temperature > 0.0F && rng_state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  SamplingConfig config;
  config.temperature = temperature;
  config.seed = 0;
  SamplerState sampler(config);
  if (rng_state != nullptr) {
    sampler.SetRngState(*rng_state);
  }
  const TokenId token = sampler.Sample(logits);
  if (rng_state != nullptr) {
    *rng_state = sampler.rng_state();
  }
  return token;
}

double TokenProbability(std::span<const float> logits, float temperature,
                        TokenId token) {
  if (token >= logits.size()) {
    throw std::out_of_range("sampled token exceeds target vocabulary");
  }
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(logits, config).probability(token);
}

TokenId SampleResidual(std::span<const float> target_logits, float temperature,
                       std::span<const TokenId> candidate_ids,
                       std::span<const float> candidate_probabilities,
                       std::uint64_t* rng_state) {
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(target_logits, config)
      .SampleResidual(candidate_ids, candidate_probabilities, rng_state);
}

}  // namespace gufo::sampling
