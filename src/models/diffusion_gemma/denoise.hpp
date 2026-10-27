#ifndef GUFO_MODELS_DIFFUSION_GEMMA_DENOISE_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_DENOISE_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/models/diffusion_gemma/config.hpp"

namespace gufo::models::diffusion_gemma {

// Host-side decisions of the block-diffusion sampler. Each denoising step
// runs the decoder over the canvas and reads back, per row, the entropy of
// the sampling distribution, its argmax and a sample; these functions turn
// that into the next canvas and the stop decision.

/// Sampling temperature of a step. Steps count down from
/// max_denoising_steps to 1, so the temperature falls linearly from t_max
/// towards t_min.
[[nodiscard]] float StepTemperature(const DiffusionDefaults& defaults,
                                    int step);

/// Entropy-bound acceptance: rows taken in order of increasing entropy are
/// accepted while the summed entropy of the rows before them stays within
/// `bound`. The lowest-entropy row is therefore always accepted. Ties keep
/// row order.
[[nodiscard]] std::vector<bool> AcceptRows(std::span<const float> entropy,
                                           float bound);

/// A block is finished when its argmax canvas did not change since the
/// previous step (unless stability_threshold is 0) and the mean row entropy
/// is below confidence_threshold. `previous_argmax` is empty on the first
/// step.
[[nodiscard]] bool BlockConverged(const DiffusionDefaults& defaults,
                                  std::span<const std::int32_t> previous_argmax,
                                  std::span<const std::int32_t> argmax,
                                  std::span<const float> entropy);

/// The part of a finished canvas that is emitted: up to (not including)
/// the first end-of-sequence token and at most `budget` tokens. The reply
/// stops at the end token only when the budget does not cut it first.
struct BlockReply {
  int tokens{0};
  bool end_of_sequence{false};
};
[[nodiscard]] BlockReply TakeReply(std::span<const std::int32_t> canvas,
                                   std::span<const int> eos_ids, int budget);

/// SplitMix64 finaliser, used to derive independent seeds.
[[nodiscard]] std::uint64_t MixSeed(std::uint64_t x);

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_DENOISE_HPP_
