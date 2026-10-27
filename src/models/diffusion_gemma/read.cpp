#include "src/models/diffusion_gemma/read.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gufo::models::diffusion_gemma {

ReadStepPlan PlanReadStep(int completed_steps, int max_steps) {
  if (max_steps < 1 || completed_steps < 0 || completed_steps >= max_steps)
    throw std::invalid_argument("invalid diffusion read step");
  const bool final = completed_steps + 1 == max_steps;
  return {.self_condition = completed_steps != 0,
          .make_probabilities = !final,
          .finish = final};
}

void ValidateReadOptions(const ReadOptions& options, int canvas, int vocab) {
  if (canvas != 256)
    throw std::invalid_argument("diffusion read width is not supported yet");
  if (!options.read_only)
    throw std::invalid_argument(
        "diffusion reads without diffusion_read_only are not supported yet");
  if (options.max_steps < 1 || options.max_steps > 8)
    throw std::invalid_argument("diffusion_max_steps supports 1..8 for now");
  if (options.top_k < 0 || options.top_k > 20)
    throw std::invalid_argument("top_logprobs must be in 0..20");
  if (!options.seed_canvas.empty() &&
      options.seed_canvas.size() != static_cast<std::size_t>(canvas))
    throw std::invalid_argument("diffusion_seed_canvas needs exactly 256 ids");
  for (const auto id : options.seed_canvas)
    if (id < 0 || id >= vocab)
      throw std::invalid_argument(
          "diffusion_seed_canvas id is outside vocabulary");
  if (!options.pinned_positions.empty() && options.seed_canvas.empty())
    throw std::invalid_argument(
        "diffusion_pinned needs a diffusion_seed_canvas");
  for (const int position : options.pinned_positions)
    if (position < 0 || position >= canvas)
      throw std::invalid_argument(
          "diffusion_pinned position is outside canvas");
  if (options.label_ids.size() > 512)
    throw std::invalid_argument("logprob_token_ids allows at most 512 ids");
  for (const auto id : options.label_ids)
    if (id < 0 || id >= vocab)
      throw std::invalid_argument("logprob_token_ids id is outside vocabulary");
}

void AdvanceReadCanvas(std::span<std::int32_t> current,
                       std::span<const std::int32_t> seed,
                       std::span<const std::uint8_t> pins,
                       std::span<const std::uint8_t> accepted,
                       std::span<const std::int32_t> sampled,
                       const std::function<std::int32_t()>& renoise) {
  const auto n = current.size();
  if (seed.size() != n || pins.size() != n || accepted.size() != n ||
      sampled.size() != n)
    throw std::invalid_argument("read canvas update shape mismatch");
  for (std::size_t i = 0; i < n; ++i)
    current[i] = pins[i] ? seed[i] : accepted[i] ? sampled[i] : renoise();
}

ReadRow ScoreLogitsRow(std::span<const float> logits, float softcap, int top_k,
                       std::span<const std::int32_t> labels) {
  if (logits.empty() || softcap <= 0 || top_k < 0 || top_k > 20)
    throw std::invalid_argument("invalid read score shape");
  std::vector<float> capped(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i)
    capped[i] = softcap * std::tanh(logits[i] / softcap);
  const float maximum = *std::max_element(capped.begin(), capped.end());
  double sum = 0;
  for (const float value : capped)
    sum += std::exp(static_cast<double>(value - maximum));
  const double lse = static_cast<double>(maximum) + std::log(sum);
  std::vector<std::int32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const auto better = [&](std::int32_t a, std::int32_t b) {
    return capped[a] > capped[b] || (capped[a] == capped[b] && a < b);
  };
  const auto take = std::min<std::size_t>(
      logits.size(), static_cast<std::size_t>(std::max(1, top_k)));
  std::partial_sort(order.begin(), order.begin() + take, order.end(), better);
  ReadRow row;
  row.token = order.front();
  row.logprob = static_cast<double>(capped[row.token]) - lse;
  for (int i = 0; i < top_k && i < static_cast<int>(logits.size()); ++i)
    row.top_logprobs.push_back(
        {order[i], static_cast<double>(capped[order[i]]) - lse});
  std::vector<std::int32_t> seen;
  for (const auto id : labels) {
    if (id < 0 || static_cast<std::size_t>(id) >= logits.size())
      throw std::invalid_argument("label id is outside vocabulary");
    if (std::find(seen.begin(), seen.end(), id) != seen.end())
      continue;
    seen.push_back(id);
    if (std::find_if(row.top_logprobs.begin(), row.top_logprobs.end(),
                     [id](const ReadCandidate& candidate) {
                       return candidate.token == id;
                     }) == row.top_logprobs.end())
      row.label_logprobs.push_back({id, static_cast<double>(capped[id]) - lse});
  }
  return row;
}

}  // namespace gufo::models::diffusion_gemma
