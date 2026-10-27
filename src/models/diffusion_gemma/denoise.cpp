#include "src/models/diffusion_gemma/denoise.hpp"

#include <algorithm>
#include <numeric>

namespace gufo::models::diffusion_gemma {

float StepTemperature(const DiffusionDefaults& defaults, int step) {
  return defaults.t_min +
         (defaults.t_max - defaults.t_min) *
             (static_cast<float>(step) / defaults.max_denoising_steps);
}

std::vector<bool> AcceptRows(std::span<const float> entropy, float bound) {
  std::vector<int> order(entropy.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return entropy[a] < entropy[b]; });
  std::vector<bool> accepted(entropy.size());
  double cumulative = 0;
  for (const int row : order) {
    cumulative += entropy[row];
    accepted[row] = cumulative - entropy[row] <= bound;
  }
  return accepted;
}

bool BlockConverged(const DiffusionDefaults& defaults,
                    std::span<const std::int32_t> previous_argmax,
                    std::span<const std::int32_t> argmax,
                    std::span<const float> entropy) {
  const bool stable =
      defaults.stability_threshold == 0 ||
      (!previous_argmax.empty() && std::ranges::equal(previous_argmax, argmax));
  double mean = 0;
  for (const float e : entropy)
    mean += e;
  mean /= static_cast<double>(entropy.size());
  return stable && mean < defaults.confidence_threshold;
}

BlockReply TakeReply(std::span<const std::int32_t> canvas,
                     std::span<const int> eos_ids, int budget) {
  const auto end = std::ranges::find_if(canvas, [&](std::int32_t token) {
    return std::ranges::find(eos_ids, token) != eos_ids.end();
  });
  const int reply = static_cast<int>(end - canvas.begin());
  return {.tokens = std::min(reply, budget),
          .end_of_sequence = end != canvas.end() && reply <= budget};
}

std::uint64_t MixSeed(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace gufo::models::diffusion_gemma
