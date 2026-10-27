#ifndef GUFO_SERVER_DIFFUSION_READ_API_HPP_
#define GUFO_SERVER_DIFFUSION_READ_API_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/diffusion_reads.hpp"
#include "src/cli/serve/logprobs.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

struct DiffusionReadOptions {
  std::optional<DiffusionReadParams> params;
  bool return_tokens_as_token_ids{false};
  std::vector<sampling::TokenId> stop_token_ids;
};

inline bool DiffusionInteger(const json::Value& value) {
  if (!value.is_number() || !std::isfinite(value.as_double()) ||
      std::floor(value.as_double()) != value.as_double())
    return false;
  const std::string spelling = value.dump();
  return !spelling.empty() &&
         std::all_of(spelling.begin() + (spelling.front() == '-'),
                     spelling.end(),
                     [](char c) { return c >= '0' && c <= '9'; });
}

inline std::optional<std::string> DiffusionTokenIds(
    const json::Value& value, std::string_view field, std::size_t vocab,
    std::vector<sampling::TokenId>* out) {
  if (!value.is_array())
    return "'" + std::string(field) + "' must be a list of token ids";
  for (const auto& id : value.items()) {
    if (!DiffusionInteger(id))
      return "'" + std::string(field) + "' must be a list of token ids";
    if (id.as_double() < 0 || id.as_double() >= static_cast<double>(vocab))
      return "'" + std::string(field) + "' ids must be in [0, " +
             std::to_string(vocab) + ")";
    out->push_back(static_cast<sampling::TokenId>(id.as_size()));
  }
  return std::nullopt;
}

inline std::optional<std::string> ParseDiffusionReadOptions(
    const json::Value& body, const TextGenerationBackend& backend,
    bool completions, DiffusionReadOptions* out) {
  const auto* xargs = body.find("vllm_xargs");
  if (xargs && !xargs->is_object())
    return "'vllm_xargs' must be an object";
  bool has_diffusion_field = false;
  std::string unknown_diffusion_field;
  if (xargs) {
    for (const auto& [key, value] : xargs->members()) {
      (void)value;
      if (!key.starts_with("diffusion_"))
        continue;
      has_diffusion_field = true;
      if (key != "diffusion_seed_canvas" && key != "diffusion_canvas_length" &&
          key != "diffusion_max_steps" && key != "diffusion_read_only" &&
          key != "diffusion_constrained" && key != "diffusion_pinned")
        unknown_diffusion_field = key;
    }
  }
  const auto* labels = body.find("logprob_token_ids");
  const auto* as_ids = body.find("return_tokens_as_token_ids");
  const auto* stop_ids = body.find("stop_token_ids");
  if ((has_diffusion_field || labels || as_ids || stop_ids) &&
      !backend.supports_diffusion_reads())
    return "this model does not support diffusion reads";
  if (!unknown_diffusion_field.empty())
    return "unknown vllm_xargs field '" + unknown_diffusion_field + "'";
  if (stop_ids && !completions)
    return "'stop_token_ids' is only supported on completions";
  if (as_ids) {
    if (!as_ids->is_bool())
      return "'return_tokens_as_token_ids' must be a boolean";
    out->return_tokens_as_token_ids = as_ids->as_bool();
  }
  if (stop_ids) {
    if (auto error =
            DiffusionTokenIds(*stop_ids, "stop_token_ids", backend.vocab_size(),
                              &out->stop_token_ids))
      return error;
  }
  if (!has_diffusion_field && !labels)
    return std::nullopt;

  DiffusionReadParams params;
  params.canvas_length = backend.diffusion_canvas_length();
  if (params.canvas_length == 0 || backend.vocab_size() == 0)
    return "diffusion read model has no canvas length or vocabulary";
  if (const auto* width =
          xargs ? xargs->find("diffusion_canvas_length") : nullptr) {
    if (!width->is_null()) {
      if (!DiffusionInteger(*width) || width->as_double() <= 0)
        return "'diffusion_canvas_length' must be a positive integer";
      if (width->as_double() > static_cast<double>(params.canvas_length))
        return "'diffusion_canvas_length' must be no larger than the served "
               "canvas length";
      params.canvas_length = width->as_size();
    }
  }
  if (const auto* steps =
          xargs ? xargs->find("diffusion_max_steps") : nullptr) {
    if (!DiffusionInteger(*steps) || steps->as_double() <= 0 ||
        steps->as_size() == 0)
      return "'diffusion_max_steps' must be a positive integer";
    params.max_steps = steps->as_size();
  }
  const auto boolean = [&](std::string_view key,
                           bool* target) -> std::optional<std::string> {
    const auto* value = xargs ? xargs->find(std::string(key)) : nullptr;
    if (!value)
      return std::nullopt;
    if (value->is_bool()) {
      *target = value->as_bool();
      return std::nullopt;
    }
    if (DiffusionInteger(*value) &&
        (value->as_double() == 0 || value->as_double() == 1)) {
      *target = value->as_double() == 1;
      return std::nullopt;
    }
    return "'" + std::string(key) + "' must be a boolean";
  };
  if (auto error = boolean("diffusion_read_only", &params.read_only))
    return error;
  if (auto error = boolean("diffusion_constrained", &params.constrained))
    return error;
  if (const auto* seed =
          xargs ? xargs->find("diffusion_seed_canvas") : nullptr) {
    params.seed_canvas.emplace();
    if (auto error =
            DiffusionTokenIds(*seed, "diffusion_seed_canvas",
                              backend.vocab_size(), &*params.seed_canvas))
      return error;
    if (params.seed_canvas->size() != params.canvas_length)
      return "'diffusion_seed_canvas' needs exactly " +
             std::to_string(params.canvas_length) + " ids, got " +
             std::to_string(params.seed_canvas->size());
  }
  if (const auto* pinned = xargs ? xargs->find("diffusion_pinned") : nullptr) {
    if (!pinned->is_array())
      return "'diffusion_pinned' must be a list of canvas positions";
    if (!params.seed_canvas)
      return "'diffusion_pinned' needs a diffusion_seed_canvas";
    for (const auto& position : pinned->items()) {
      if (!DiffusionInteger(position))
        return "'diffusion_pinned' must be a list of canvas positions";
      if (position.as_double() < 0 ||
          position.as_double() >= static_cast<double>(params.canvas_length))
        return "'diffusion_pinned' positions must be inside the canvas";
      params.pinned_positions.push_back(position.as_size());
    }
  }
  if (labels) {
    if (auto error = DiffusionTokenIds(*labels, "logprob_token_ids",
                                       backend.vocab_size(), &params.label_ids))
      return error;
    if (params.label_ids.size() > 512)
      return "'logprob_token_ids' allows at most 512 ids";
    std::sort(params.label_ids.begin(), params.label_ids.end());
    params.label_ids.erase(
        std::unique(params.label_ids.begin(), params.label_ids.end()),
        params.label_ids.end());
  }
  if (params.constrained && params.label_ids.empty())
    return "'diffusion_constrained' needs logprob_token_ids";
  out->params = std::move(params);
  return std::nullopt;
}

inline std::string DiffusionTokenText(sampling::TokenId token) {
  return "token_id:" + std::to_string(token);
}

inline std::vector<sampling::TokenLogprob> MergeDiffusionRows(
    const std::vector<DiffusionReadRow>& rows,
    const DiffusionReadParams& params, std::size_t top_k) {
  if (rows.size() != params.canvas_length)
    throw std::invalid_argument(
        "diffusion read returned the wrong canvas row count");
  std::vector<sampling::TokenLogprob> merged;
  merged.reserve(rows.size());
  for (const auto& row : rows) {
    sampling::TokenLogprob score{row.token, row.logprob, {}};
    for (const auto& top : row.top_logprobs) {
      if (score.top_logprobs.size() >= top_k)
        break;
      if (std::none_of(
              score.top_logprobs.begin(), score.top_logprobs.end(),
              [&](const auto& item) { return item.token == top.token; }))
        score.top_logprobs.push_back(top);
    }
    for (const auto label : params.label_ids) {
      if (std::any_of(score.top_logprobs.begin(), score.top_logprobs.end(),
                      [&](const auto& item) { return item.token == label; }))
        continue;
      const auto found =
          std::find_if(row.label_logprobs.begin(), row.label_logprobs.end(),
                       [&](const auto& item) { return item.token == label; });
      if (found == row.label_logprobs.end())
        throw std::invalid_argument(
            "diffusion read omitted a requested label id");
      score.top_logprobs.push_back(*found);
    }
    std::stable_sort(
        score.top_logprobs.begin(), score.top_logprobs.end(),
        [](const auto& a, const auto& b) { return a.value > b.value; });
    merged.push_back(std::move(score));
  }
  return merged;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_READ_API_HPP_
