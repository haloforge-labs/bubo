#ifndef GUFO_MODELS_QWEN36_A3B_TEXT_RUNNER_HPP_
#define GUFO_MODELS_QWEN36_A3B_TEXT_RUNNER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "src/cli/serve/text_model_runner.hpp"

/// `gufo serve` for Qwen3.6-35B-A3B (qwen35moe) GGUFs: the A3B engine as a
/// TextModelRunner, so it runs under gufo's scheduler, prompt cache and HTTP
/// layer like every other text model.
namespace gufo::models::qwen36_a3b {

/// Serve settings, mapped from gufo serve's flags (inference_backend.cpp).
struct ServeConfig {
  enum class Drafting { kOff, kMtp, kDFlash };
  /// Context tokens (0 = 131072).
  std::uint32_t max_context{0};
  /// --speculative mtp uses the MTP block inside the model GGUF.
  Drafting drafting{Drafting::kOff};
  std::string dflash_model_path;
  /// Most drafts per step (the verify pass takes at most 7).
  std::uint32_t max_draft_tokens{6};
  /// --mtp-policy survival: stop drafting once the chain's estimated
  /// survival falls below a floor (MTP 0.6, DFlash2 0.2).
  bool survival{false};
  /// --mtp-draft-vocab latin: the draft head over Latin-text tokens only.
  bool latin_draft_vocab{false};
  /// --prompt-lookup: drafts copied from the context (12-token matches).
  bool prompt_lookup{false};
  /// --mmproj: the Qwen3.6 vision sidecar (BF16, projection 2048). Empty
  /// looks for mmproj-BF16.gguf beside the model, as the Qwen loader does.
  std::string vision_model_path;
};

/// True for the GGUF architecture this runtime serves.
[[nodiscard]] inline bool IsA3bArchitecture(std::string_view architecture) {
  return architecture == "qwen35moe";
}

/// Loads the model and returns its runner (one session).
[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    const std::string& model_path, const ServeConfig& config,
    std::string* error);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_TEXT_RUNNER_HPP_
