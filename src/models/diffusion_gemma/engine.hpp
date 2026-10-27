#ifndef GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/models/diffusion_gemma/config.hpp"
#include "src/models/diffusion_gemma/image_prompt.hpp"
#include "src/models/diffusion_gemma/tokenizer.hpp"

namespace gufo::models::diffusion_gemma {

struct EngineOptions {
  /// Largest prompt plus generated length; sizes the global-layer KV cache.
  int max_context{131072};
  /// Prompt tokens per encoder pass. Larger chunks give the expert products
  /// fuller row groups; each prompt row costs about 390 KB of scratch and
  /// sliding-layer cache.
  int prefill_chunk{8192};
  /// Concurrent requests. Each session holds its own KV caches; the
  /// denoising steps of active requests run as one batched forward pass.
  int sessions{1};
  /// Print per-phase GPU time for each request to stderr.
  bool profile{false};
};

struct GenerationOptions {
  int max_new_tokens{256};
  /// Fixed-length generation: the end-of-sequence tokens are never sampled,
  /// so the reply runs to max_new_tokens (benchmarks).
  bool suppress_end{false};
  DiffusionDefaults diffusion;
  std::uint64_t seed{0};
  /// Reuse the longest prompt prefix a free session's cache holds; false
  /// encodes the whole prompt.
  bool reuse_prompt{true};
  /// Polled while the request runs (from the calling thread); true stops it
  /// (GenerationStats::cancelled).
  std::function<bool()> cancelled;
};

struct GenerationStats {
  int prompt_tokens{0};
  int reused_tokens{0};  ///< Prompt prefix reused from the previous cache.
  int generated_tokens{0};
  int blocks{0};
  int denoising_steps{0};
  double prefill_ms{0};      ///< Prompt encoding.
  double denoise_ms{0};      ///< All denoising steps.
  double commit_ms{0};       ///< Encoding accepted canvases into the cache.
  double first_block_ms{0};  ///< Prompt submission to the first canvas.
  bool stopped_by_eos{false};
  bool cancelled{false};
};

struct ReadOptions {
  /// Empty starts from the engine's random canvas; otherwise exactly one
  /// vocabulary ID per canvas position.
  std::vector<std::int32_t> seed_canvas;
  int max_steps{1};
  std::vector<int> pinned_positions;
  std::vector<std::int32_t> label_ids;
  int top_k{20};
  bool read_only{true};
  bool reuse_prompt{true};
  std::uint64_t seed{0};
  std::function<bool()> cancelled;
};

struct ReadCandidate {
  std::int32_t token{0};
  double logprob{0};
};
struct ReadRow {
  std::int32_t token{0};
  double logprob{0};
  std::vector<ReadCandidate> top_logprobs;
  std::vector<ReadCandidate> label_logprobs;
};
struct ReadResult {
  GenerationStats stats;
  std::vector<ReadRow> rows;
};

/// DiffusionGemma block-diffusion engine on gfx1151. Generate may be
/// called from several threads: requests wait for a free session, prompt
/// chunks run one at a time between denoising steps, and the denoising steps
/// of all active requests run as one batched forward pass.
class Engine {
public:
  /// Called once per accepted canvas with the tokens it contributes (EOS and
  /// everything after it removed). Return false to cancel.
  using BlockCallback = std::function<bool(std::span<const std::int32_t>)>;

  Engine(const std::filesystem::path& root, const EngineOptions& options);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  [[nodiscard]] const Config& config() const;
  [[nodiscard]] const Tokenizer& tokenizer() const;
  [[nodiscard]] const DiffusionDefaults& diffusion_defaults() const;
  [[nodiscard]] int max_context() const;
  [[nodiscard]] int prefill_chunk() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  /// Generates a reply to `prompt` in a free session: the one whose cache
  /// (a previous prompt and its reply's committed canvases) shares the
  /// longest reusable prefix with it, else the least recently used. The
  /// prefix is reused when the sliding layers still hold the window before
  /// it; only the rest is encoded. `on_block` runs on the calling thread.
  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           const GenerationOptions& options,
                           const BlockCallback& on_block);
  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           std::span<const ImageInput> images,
                           const GenerationOptions& options,
                           const BlockCallback& on_block);

  /// Scores one seeded canvas at the request-local denoise step cap. The
  /// prompt is cacheable, while the canvas is never appended to the cache.
  ReadResult Read(std::span<const std::int32_t> prompt,
                  const ReadOptions& options);
  ReadResult Read(std::span<const std::int32_t> prompt,
                  std::span<const ImageInput> images,
                  const ReadOptions& options);

  /// Runs an encoder chunk and two denoising steps so that the first
  /// request does not pay for loading kernels. Leaves the cache empty;
  /// returns elapsed milliseconds.
  double Warmup();

  // Benchmark primitives, on the first session while no request runs.
  // Reset() empties every cache; Encode() appends tokens through the causal
  // encoder and returns elapsed milliseconds.
  void Reset();
  double Encode(std::span<const std::int32_t> tokens);
  [[nodiscard]] int cached_tokens() const;

  /// Reference check: encodes `prompt`, runs the denoiser on `canvas`
  /// without self-conditioning, then on `canvas2` conditioned on the first
  /// step's logits (temperature 1). Returns both steps' softcapped logits,
  /// row-major [2][canvas][vocab].
  std::vector<float> DebugLogits(std::span<const std::int32_t> prompt,
                                 const std::vector<std::int32_t>& canvas,
                                 const std::vector<std::int32_t>& canvas2);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_
