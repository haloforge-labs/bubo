#ifndef GUFO_MODELS_DIFFUSION_GEMMA_READ_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_READ_HPP_

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "src/models/diffusion_gemma/engine.hpp"

namespace gufo::models::diffusion_gemma {

/// Host decisions shared by the worker and CPU tests. A read always runs to
/// its request-local step cap, ignores EOS, and never schedules a cache commit.
struct ReadStepPlan {
  bool self_condition{false};
  bool make_probabilities{false};
  bool finish{false};
};
[[nodiscard]] ReadStepPlan PlanReadStep(int completed_steps, int max_steps);

void ValidateReadOptions(const ReadOptions& options, int canvas, int vocab);

/// Mutates the next canvas only when another denoise step follows. Pins take
/// their original seed token; free positions use the normal accept/renoise
/// decision. `renoise` supplies independently drawn vocabulary IDs.
void AdvanceReadCanvas(std::span<std::int32_t> current,
                       std::span<const std::int32_t> seed,
                       std::span<const std::uint8_t> pins,
                       std::span<const std::uint8_t> accepted,
                       std::span<const std::int32_t> sampled,
                       const std::function<std::int32_t()>& renoise);

/// CPU reference for the score kernel. Inputs are raw FP32 LM-head logits;
/// the result uses temperature-1 log-softmax of soft-capped logits.
[[nodiscard]] ReadRow ScoreLogitsRow(std::span<const float> logits,
                                     float softcap, int top_k,
                                     std::span<const std::int32_t> labels);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_READ_HPP_
