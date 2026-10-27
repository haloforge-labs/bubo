#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "src/core/json_constraint.hpp"
#include "src/core/behaviour.hpp"

namespace gufo::sampling {

using TokenId = std::uint32_t;

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, temperature, top-k, top-p, min-p.
/// The defaults preserve Gufo's existing greedy decoding behavior.
struct SamplingConfig {
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
  std::shared_ptr<const TokenConstraint> constraint;
  std::shared_ptr<const BehaviourConfig> behaviour;
  // Negative leaves scoring off; otherwise collect the requested top 0..20.
  std::int32_t logprobs{-1};

  void Validate() const;

  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
};

/// Unique token counts: repetition window and full generated response.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

struct TokenLogprob {
  TokenId token{0};
  double logprob{0};
  std::vector<Probability> top_logprobs;
};

struct BehaviourDistribution;

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

public:
  [[nodiscard]] const std::shared_ptr<const BehaviourDistribution>& behaviour() const { return behaviour_; }
private:
  friend class SamplerState;
  std::shared_ptr<const BehaviourDistribution> behaviour_;
  // Entries generated internally have unique IDs; retain their stable order.
  SamplingDistribution(std::vector<Probability> entries, double total);
  std::vector<Probability> entries_;
};

struct BehaviourStage {
  std::string name, reason;
  SamplingDistribution before, after;
  bool before_random{true}, after_random{true}, before_valid{true};
};
struct BehaviourDistribution {
  std::vector<BehaviourStage> stages;
  bool defer_stop{false}, close_eog{false}, random{false};
  std::optional<ThinkMarkerRow> think_marker_row;
};

/// Builds the exact distribution from prompt and request-generated tokens.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens = {},
    std::span<const TokenId> generated_tokens = {});

/// Request-owned controls, repetition window, full generated counts and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  [[nodiscard]] bool LogprobsRequested() const noexcept {
    return logprobs_requested_;
  }
  /// Draft q may remain unconstrained; target p always uses this request's
  /// grammar.
  [[nodiscard]] SamplerState WithoutConstraint() const;
  [[nodiscard]] SamplerState WithoutBehaviour() const;
  [[nodiscard]] bool BehaviourEnabled() const noexcept;
  /// Accepted TOOL terminal event; tentative draws alone cannot set it.
  [[nodiscard]] bool ToolArgumentLoopDetected() const noexcept {
    return BehaviourEnabled() && behaviour_state_.loop.stopped_tools != 0;
  }
  [[nodiscard]] std::vector<std::string> BehaviourSummary() const;
  void RecordSpecEogStop(std::size_t skipped);
  void ObserveBehaviour(const SamplingDistribution& distribution,
      TokenId actual, std::uint64_t rng_before,
      std::span<const TokenId> proposal_ids = {},
      std::span<const float> proposal_probabilities = {},
      TokenId proposed = 0, double q = 0);
  /// A finite argmax can bypass downloading logits if it satisfies the grammar.
  /// Set penalties_applied only for a winner computed with this state's exact
  /// penalty arithmetic and history; raw model argmaxes leave it false.
  [[nodiscard]] bool CanSelectArgmax(TokenId token,
                                     bool penalties_applied = false) const;
  /// A lazy tool grammar can admit the entire vocabulary before a call.
  /// Retain ordinary CPU/GPU sampling while its mask makes no restriction.
  [[nodiscard]] bool NeedsConstraintMask() const;
  [[nodiscard]] const JsonConstraint::State& constraint_state() const noexcept {
    return constraint_state_;
  }
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  [[nodiscard]] std::span<const TokenPenalty> penalties() const noexcept {
    return penalty_counts_;
  }
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;
  /// Publish RNG and any deferred draw without accepting tentative history.
  void CopyDrawStateFrom(const SamplerState& other);
  struct DrawState {
    std::uint64_t rng;
    std::optional<TokenId> pending;
    std::optional<BehaviourState> behaviour;
  };
  [[nodiscard]] DrawState SaveDrawState() const {
    return {rng_state_, pending_sample_, BehaviourEnabled() ? std::optional(behaviour_state_) : std::nullopt};
  }
  void RestoreDrawState(DrawState state) {
    rng_state_ = state.rng;
    pending_sample_ = state.pending;
    if(state.behaviour) behaviour_state_ = std::move(*state.behaviour);
  }

  /// Starts a request: tokens initialize repetition only; generated counts
  /// reset.
  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  /// Raw log-softmax after deterministic bias/masks and history penalties,
  /// before temperature or truncation. Does not draw or accept a token.
  [[nodiscard]] TokenLogprob RawLogprobs(std::span<const float> logits,
                                       TokenId token, std::size_t top_n) const;
  /// The RawLogprobs implementation from before L-REL-6 (a SamplerState copy,
  /// a vocabulary-sized candidate vector, a partial sort). Test oracle only:
  /// RawLogprobs must return bit-identical values.
  [[nodiscard]] TokenLogprob RawLogprobsReference(std::span<const float> logits,
                                                  TokenId token,
                                                  std::size_t top_n) const;
  /// Compact logits use token_ids for penalties; returned IDs index logits.
  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits, std::span<const TokenId> token_ids) const;
  /// Distribution(logits) computed from a candidate list instead of the full
  /// row: `values`/`ids` are a correct top-N of the row's logits (by value,
  /// any tie order). Exact whenever it returns a value: it needs top_k > 0,
  /// temperature > 0 and no active penalties, and a list that provably holds
  /// every token of the top-k (the k-th value strictly above the list's
  /// last, or a list that already contains every finite logit). Otherwise it
  /// returns nothing and callers use the full row.
  [[nodiscard]] std::optional<SamplingDistribution> DistributionFromTop(
      std::span<const float> values, std::span<const TokenId> ids,
      std::size_t vocab) const;
  /// Sample(logits) from a candidate list (see DistributionFromTop), with the
  /// same pending-draw handling and RNG use. Nothing when not applicable.
  [[nodiscard]] std::optional<TokenId> SampleFromTop(
      std::span<const float> values, std::span<const TokenId> ids,
      std::size_t vocab);
  /// Retain a speculative residual draw for the next Sample call. Copies
  /// preserve this draw along with the RNG; resetting history discards it.
  void DeferSample(TokenId token);
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  [[nodiscard]] bool ToolArgumentStopping() const;
  [[nodiscard]] SamplerState BehaviourWithoutIntervention(std::string_view name) const;
  [[nodiscard]] SamplingDistribution BehaviourDistributionFor(
      std::span<const float> logits, std::span<const TokenId> ids = {},
      std::size_t vocab = 0,
      std::optional<ThinkMarkerRow> think_marker_row = {}) const;
  void ObserveThinkMarker(const ThinkMarkerRow& row, TokenId actual);
  [[nodiscard]] bool BehaviourTopSafe(
      std::span<const float> values, std::span<const TokenId> ids,
      std::size_t vocab, std::optional<ThinkMarkerRow>& think_marker_row) const;
  mutable BehaviourState behaviour_state_;
  bool observe_behaviour_{true};
  [[nodiscard]] std::vector<float> ConstrainedLogits(
      std::span<const float> logits, std::span<const TokenId> ids = {}) const;
  void TrimHistory();
  void RebuildPenaltyCounts();
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId SampleConstrainedGreedy(
      std::span<const float> logits) const;
  [[nodiscard]] SamplingDistribution LinearDistribution(
      std::span<const float> logits) const;
  struct SelectionProof {
    Probability topk_last;
    double full_sum{0}, previous_mass{0};
    std::size_t top_p_keep{0}, minimum{0};
  };
  [[nodiscard]] std::optional<SamplingDistribution> InertEogDistribution(
      std::span<const float> logits, SamplerState& native) const;
  void PrepareSelected(std::span<const float> logits,
                       SelectionProof* proof = nullptr);
  /// top-p and min-p over the sorted candidate_scratch_ (PrepareSelected's
  /// second half).
  void FinishSelected(double full_softmax_sum, bool has_full_softmax_sum,
                      SelectionProof* proof = nullptr);

  SamplingConfig config_;
  // Resolve request opt-in once; speculative sampler copies retain it.
  bool logprobs_requested_{false};
  JsonConstraint::State constraint_state_;
  mutable std::shared_ptr<const std::vector<std::uint8_t>> constraint_mask_;
  mutable bool constraint_restricts_{false};
  std::vector<TokenId> history_;
  std::vector<TokenPenalty> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t rng_state_{0};
  std::optional<TokenId> pending_sample_;
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
