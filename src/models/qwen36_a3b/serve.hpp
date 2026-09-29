#ifndef GUFO_MODELS_QWEN36_A3B_SERVE_HPP_
#define GUFO_MODELS_QWEN36_A3B_SERVE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/sampling.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

/// `gufo-a3b --serve`: an OpenAI-compatible server (gufo's HTTP transport and
/// chat handlers, used read-only) over one A3B engine slot. Requests run one
/// at a time. The prompt is rendered with gufo's Qwen formatter at Qwen3.6's
/// defaults (no reasoning-effort line, earlier turns' reasoning dropped), and
/// a checkpoint before the generation suffix lets the next turn of the same
/// conversation prefill only its new tokens.
namespace gufo::models::qwen36_a3b {

struct ServeOptions {
  std::string host{"127.0.0.1"};
  int port{8080};
  std::string model_id{"qwen3.6-35b-a3b"};
  std::string api_key;
  std::size_t max_connections{16};
  std::size_t max_request_bytes{64ULL << 20};
  // Reasoning defaults for requests that leave them unset (unset: thinking
  // on, earlier turns' reasoning dropped, as Qwen3.6's template).
  std::optional<bool> think;
  std::optional<bool> preserve_thinking;
  // Speculative policy (as the benches): MTP drafts with a survival floor,
  // prompt lookup after each kept MTP draft, or DFlash block drafts.
  int mtp{6};
  double survival{0.6};
  int lookup{12};
  int dflash_n{0};  // > 0 with a loaded DFlash draft
  std::size_t max_tokens{32768};
  sampling::SamplingConfig sampling;
};

int RunServe(Engine& engine, const tokenization::QwenTokenizer& tokenizer,
             const ServeOptions& options);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_SERVE_HPP_
