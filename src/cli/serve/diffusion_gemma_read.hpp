#ifndef GUFO_SERVER_DIFFUSION_GEMMA_READ_HPP_
#define GUFO_SERVER_DIFFUSION_GEMMA_READ_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/diffusion_gemma/engine.hpp"
#include "src/models/diffusion_gemma/read.hpp"

namespace gufo::server {

/// The CPU-testable boundary between the serve contract and the GPU engine.
inline models::diffusion_gemma::ReadOptions PlanDiffusionGemmaRead(
    const DiffusionReadParams& params, int served_width, int vocab,
    int requested_top_k, std::uint64_t seed, bool cache_prompt) {
  if (params.canvas_length != static_cast<std::size_t>(served_width) ||
      served_width != 256)
    throw std::invalid_argument(
        "diffusion_canvas_length other than 256 is not supported yet");
  if (params.constrained)
    throw std::invalid_argument("diffusion_constrained is not supported yet");
  if (!params.read_only)
    throw std::invalid_argument(
        "diffusion reads without diffusion_read_only are not supported yet");
  if (params.max_steps >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("diffusion_max_steps is too large");
  models::diffusion_gemma::ReadOptions options;
  options.max_steps = static_cast<int>(params.max_steps);
  options.top_k = requested_top_k;
  options.seed = seed;
  options.reuse_prompt = cache_prompt;
  if (params.seed_canvas)
    for (const auto id : *params.seed_canvas)
      options.seed_canvas.push_back(static_cast<std::int32_t>(id));
  for (const auto position : params.pinned_positions) {
    if (position > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("diffusion_pinned position is too large");
    options.pinned_positions.push_back(static_cast<int>(position));
  }
  for (const auto id : params.label_ids)
    options.label_ids.push_back(static_cast<std::int32_t>(id));
  models::diffusion_gemma::ValidateReadOptions(options, served_width, vocab);
  return options;
}

inline TextGenerationBackend::Result DiffusionGemmaReadResult(
    const models::diffusion_gemma::ReadResult& read,
    std::string_view client_id) {
  TextGenerationBackend::Result result;
  result.client_id = std::string(client_id);
  result.execution_plan = "diffusion-read";
  result.prompt_tokens = read.stats.prompt_tokens;
  result.cached_prompt_tokens = read.stats.reused_tokens;
  result.prefill_tokens = read.stats.prompt_tokens - read.stats.reused_tokens;
  result.prefill_ms = read.stats.prefill_ms;
  result.decode_ms = read.stats.denoise_ms;
  result.ttft_ms = read.stats.first_block_ms;
  result.completion_tokens = read.rows.size();
  result.finish_reason = TextGenerationBackend::FinishReason::kLength;
  result.diffusion_rows.emplace();
  result.diffusion_rows->reserve(read.rows.size());
  for (const auto& source : read.rows) {
    DiffusionReadRow row;
    row.token = static_cast<sampling::TokenId>(source.token);
    row.logprob = source.logprob;
    for (const auto& top : source.top_logprobs)
      row.top_logprobs.push_back(
          {static_cast<sampling::TokenId>(top.token), top.logprob});
    for (const auto& label : source.label_logprobs)
      row.label_logprobs.push_back(
          {static_cast<sampling::TokenId>(label.token), label.logprob});
    result.diffusion_rows->push_back(std::move(row));
  }
  return result;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_GEMMA_READ_HPP_
