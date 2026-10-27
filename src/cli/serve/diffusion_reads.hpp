#ifndef GUFO_SERVER_DIFFUSION_READS_HPP_
#define GUFO_SERVER_DIFFUSION_READS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::server {

/// Request fields understood by the DiffusionGemma read path. A missing seed
/// leaves canvas initialization to the backend.
struct DiffusionReadParams {
  std::optional<std::vector<sampling::TokenId>> seed_canvas;
  std::size_t canvas_length{0};
  std::size_t max_steps{1};
  bool read_only{false};
  std::vector<std::size_t> pinned_positions;
  bool constrained{false};
  std::vector<sampling::TokenId> label_ids;
};

/// One canvas position. The backend returns exact scores for every requested
/// label in either top_logprobs or label_logprobs. Probability::value is the
/// natural-log probability in both vectors.
struct DiffusionReadRow {
  sampling::TokenId token{0};
  double logprob{0};
  std::vector<sampling::Probability> top_logprobs;
  std::vector<sampling::Probability> label_logprobs;
};

struct TokenCompletionRequest {
  std::vector<sampling::TokenId> prompt;
  std::optional<DiffusionReadParams> diffusion_read;
  std::vector<sampling::TokenId> stop_token_ids;
  std::string client_id{"anonymous"};
  bool cache_prompt{true};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_READS_HPP_
