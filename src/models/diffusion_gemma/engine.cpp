#include "src/models/diffusion_gemma/engine.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/models/diffusion_gemma/cache_state.hpp"
#include "src/models/diffusion_gemma/denoise.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/profiler.hpp"
#include "src/models/diffusion_gemma/read.hpp"
#include "src/models/diffusion_gemma/weights.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

/// Owning device allocation, zero-initialised.
class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0)
      return;
    HipCheck(hipMalloc(&data_, bytes), "allocation");
    // The memset runs on the null stream, which the engine's non-blocking
    // stream does not wait for: finish it before the buffer is used.
    HipCheck(hipMemset(data_, 0, bytes), "memset");
    HipCheck(hipStreamSynchronize(nullptr), "memset");
  }
  ~Buffer() {
    if (data_)
      (void)hipFree(data_);
  }
  Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }
  template<typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

struct Engine::Impl {
  // Most key-range splits of a canvas attention pass (merged afterwards).
  static constexpr int kMaxSplits = 16;
  // Vocabulary splits of the self-conditioning product: 16 x 22 column
  // tiles = 352 workgroups, enough to fill the GPU (see soft_embed.hip).
  static constexpr int kSoftEmbedSplits = 16;
  // Sliding positions kept beyond the attention window, so that a prompt
  // diverging from the cache up to this far before its end (an edited or
  // re-tokenised reply) still reuses the cache (CacheState::Reuse).
  static constexpr int kReuseHistory = 4096;
  // Per-block seed offset: an odd constant, so blocks of one request draw
  // from distinct streams.
  static constexpr std::uint64_t kBlockSeedStride = 0x1000193ULL;

  std::filesystem::path root;
  EngineOptions options;
  Config config;
  DiffusionDefaults defaults;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<DeviceWeights> device_weights;
  hipStream_t stream{};
  std::unique_ptr<Profiler> profiler;

  int rows{0};  // Scratch rows: max(prefill chunk, sessions x canvas).
  float embed_scale{0};

  // Layer pass scratch, `rows` rows (an encoder chunk or the canvases):
  // residual stream, normed input, attention and dense MLP intermediates.
  Buffer ids, h, xn, qkv, q, attn, o, ff, ff2, rout, act, mlp_out;
  // Experts: router logits, top-k choice, the dispatch grouping rows by
  // expert (and its list of 64-row groups), per-assignment expert
  // intermediates and outputs.
  Buffer router_logits, top_index, top_weight, counts, offsets, source, slot,
      groups, group_count, expert_act, expert_out;
  // Canvas attention split-key partials.
  Buffer split_out, split_stats;
  // Denoiser head and sampler, the batch's canvas rows: logits, per-row
  // entropy, argmax and sample.
  Buffer logits, entropy, argmax, sample;
  // Compact read scores. Scratch is shared by read jobs on the worker's
  // stream; only the final step of a read uses it.
  Buffer score_tile_max, score_tile_sum, score_tile_ids, score_tile_values;
  Buffer score_top_ids, score_top_values, score_labels, score_lse, score_argmax;
  Buffer score_label_ids;
  // Self-conditioning: token embeddings, soft embedding (and its split
  // partials), normed soft embedding, the conditioning signal.
  Buffer embeds, soft, soft_partials, soft_norm, signal;

  // One request's state on the device: KV caches per layer, [kv heads]
  // [capacity][head dim] (see CacheState for the slot layout), and its
  // canvas's self-conditioning probabilities (blocked for
  // SoftEmbedProduct).
  struct Session {
    CacheState cache{0, kReuseHistory};
    std::vector<Buffer> k_cache, v_cache;
    Buffer probs;
    Buffer pinned;
    bool busy{false};
    std::uint64_t last_used{0};
  };
  std::vector<Session> sessions;
  std::uint64_t session_clock{0};
  int global_capacity{0};
  // Sliding caches hold the kept positions plus one pass. With one
  // session that pass may be a prompt chunk; with several, sessions hold
  // room for a canvas only, and prompt chunks run in the shared `stage`
  // caches (sliding layers only), copied in before and out after.
  int sliding_capacity{0};
  int stage_capacity{0};
  std::vector<Buffer> k_stage, v_stage;
  // Moves the sliding caches' kept positions to the front after a pass.
  Buffer shift_scratch;

  // The sessions of the current layer pass and their rows of the scratch
  // buffers: the attention blocks read and write each one's caches.
  // Decoder segments (canvases) see their cache plus the whole canvas;
  // Text prompt chunks and committed canvases are causal; each image span
  // is a separate bidirectional encoder segment.
  struct Segment {
    int row0{0};
    int rows{0};
    Session* session{nullptr};
    bool decoder{true};
    bool image{false};
    bool staged{false};  // Sliding layers use the stage caches.
  };
  std::vector<Segment> segments;
  Buffer sliding_freq, global_freq;  // RoPE inverse frequencies.

  // The LM head is the tied embedding, which stays row-major for lookups
  // and the self-conditioning product: kernels::Linear reads this blocked
  // copy. `linear_partials` holds split-reduction partials.
  Buffer lm_head, linear_partials;

  // Profile mode: experts with at least one row, over canvas passes.
  std::size_t active_experts{0}, expert_passes{0};

  Impl(const std::filesystem::path& r, const EngineOptions& o)
      : root(r), options(o) {
    config = LoadConfig(root);
    defaults = LoadDiffusionDefaults(root);
    tokenizer = std::make_unique<Tokenizer>(root);
    options.sessions = std::max(1, options.sessions);
    sessions = std::vector<Session>(options.sessions);
    for (auto& session : sessions)
      session.cache = CacheState(config.sliding_window, kReuseHistory);
    options.prefill_chunk =
        std::max(256, (options.prefill_chunk + 63) / 64 * 64);
    // At least a canvas and the 64-token warm-up prompt.
    options.max_context = std::clamp(options.max_context, config.canvas + 64,
                                     config.max_positions);
    HipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    device_weights = std::make_unique<DeviceWeights>(root, config);
    profiler = std::make_unique<Profiler>(stream, options.profile);
    // The embedding scale is a checkpoint-dtype (BF16) constant.
    {
      const float exact = std::sqrt(static_cast<float>(config.hidden));
      std::uint32_t bits;
      std::memcpy(&bits, &exact, 4);
      bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
      std::memcpy(&embed_scale, &bits, 4);
    }
    Allocate();
    StartWorker();
  }

  ~Impl() {
    StopWorker();
    if (stream) {
      (void)hipStreamSynchronize(stream);
      profiler.reset();
      (void)hipStreamDestroy(stream);
    }
  }

  void Allocate() {
    const Config& c = config;
    const int batch = options.sessions * c.canvas;
    rows = std::max(options.prefill_chunk, batch);
    const std::size_t R = rows, H = c.hidden, V = c.vocab, B = batch;
    const std::size_t qkv_width =
        static_cast<std::size_t>(c.heads) * c.global_head_dim +
        static_cast<std::size_t>(c.global_kv_heads) * c.global_head_dim;
    const std::size_t qkv_width_sliding =
        static_cast<std::size_t>(c.heads + 2 * c.sliding_kv_heads) *
        c.sliding_head_dim;
    const std::size_t attn_width =
        static_cast<std::size_t>(c.heads) * c.global_head_dim;
    const std::size_t A = R * c.top_k;
    ids = Buffer(R * 4);
    h = Buffer(R * H * 4);
    embeds = Buffer(B * H * 4);
    xn = Buffer(R * H * 2);
    qkv = Buffer(R * std::max(qkv_width, qkv_width_sliding) * 4);
    q = Buffer(R * attn_width * 2);
    attn = Buffer(R * attn_width * 2);
    o = Buffer(R * H * 4);
    ff = Buffer(R * H * 2);
    ff2 = Buffer(R * H * 2);
    rout = Buffer(R * H * 2);
    act = Buffer(R * c.intermediate * 2);
    mlp_out = Buffer(R * H * 4);
    router_logits = Buffer(R * c.experts * 4);
    top_index = Buffer(A * 4);
    top_weight = Buffer(A * 4);
    counts = Buffer(c.experts * 4);
    offsets = Buffer(c.experts * 4);
    source = Buffer(A * 4);
    slot = Buffer(A * 4);
    groups = Buffer(
        std::size_t(kernels::ExpertGroupBound(rows, c.experts, c.top_k, 64)) *
        8);
    group_count = Buffer(4);
    expert_act = Buffer(A * c.expert_intermediate * 2);
    expert_out = Buffer(A * H * 2);
    logits = Buffer(B * V * 4);
    soft = Buffer(B * H * 4);
    soft_partials = Buffer(std::size_t(kSoftEmbedSplits) * c.canvas * H * 4);
    soft_norm = Buffer(B * H * 2);
    signal = Buffer(B * H * 4);
    entropy = Buffer(B * 4);
    argmax = Buffer(B * 4);
    sample = Buffer(B * 4);
    const std::size_t score_tiles = (V + 4095) / 4096;
    score_tile_max = Buffer(std::size_t(c.canvas) * score_tiles * 4);
    score_tile_sum = Buffer(std::size_t(c.canvas) * score_tiles * 4);
    score_tile_ids = Buffer(std::size_t(c.canvas) * score_tiles * 20 * 4);
    score_tile_values = Buffer(std::size_t(c.canvas) * score_tiles * 20 * 4);
    score_top_ids = Buffer(std::size_t(c.canvas) * 20 * 4);
    score_top_values = Buffer(std::size_t(c.canvas) * 20 * 4);
    score_labels = Buffer(std::size_t(c.canvas) * 512 * 4);
    score_lse = Buffer(std::size_t(c.canvas) * 4);
    score_argmax = Buffer(std::size_t(c.canvas) * 4);
    score_label_ids = Buffer(512 * 4);
    // Split-key attention partials for canvas passes.
    split_out = Buffer(std::size_t(kMaxSplits) * c.canvas * attn_width * 4);
    split_stats = Buffer(std::size_t(kMaxSplits) * c.canvas * c.heads * 2 * 4);

    // Global layers hold every position; sliding layers the window, the
    // reuse history and one pass (see sliding_capacity).
    const int keep = sessions[0].cache.sliding_keep();
    global_capacity = options.max_context + c.canvas;
    const bool shared = options.sessions > 1;
    sliding_capacity = keep + (shared ? c.canvas : rows);
    stage_capacity = shared ? keep + rows : 0;
    shift_scratch = Buffer(static_cast<std::size_t>(c.sliding_kv_heads) * keep *
                           c.sliding_head_dim * 2);
    const auto cache_bytes = [&](int l, int capacity) {
      return static_cast<std::size_t>(c.kv_heads(l)) * capacity *
             c.head_dim(l) * 2;
    };
    for (auto& session : sessions) {
      for (int l = 0; l < c.layers; ++l) {
        const std::size_t bytes =
            cache_bytes(l, c.is_global[l] ? global_capacity : sliding_capacity);
        session.k_cache.emplace_back(bytes);
        session.v_cache.emplace_back(bytes);
      }
      session.probs = Buffer(std::size_t(c.canvas) * V * 2);
      session.pinned = Buffer(c.canvas);
    }
    for (int l = 0; l < c.layers; ++l) {
      const bool staged = shared && !c.is_global[l];
      k_stage.emplace_back(staged ? cache_bytes(l, stage_capacity) : 0);
      v_stage.emplace_back(staged ? cache_bytes(l, stage_capacity) : 0);
    }

    // RoPE inverse frequencies, as transformers computes them in FP32.
    std::vector<float> sliding(c.sliding_head_dim / 2),
        global(c.global_head_dim / 2, 0.0F);
    for (int i = 0; i < c.sliding_head_dim / 2; ++i)
      sliding[i] = 1.0F / std::pow(c.sliding_theta,
                                   static_cast<float>(2 * i) /
                                       static_cast<float>(c.sliding_head_dim));
    const int rotated =
        static_cast<int>(c.global_rotary_fraction * c.global_head_dim / 2);
    for (int i = 0; i < rotated; ++i)
      global[i] = 1.0F / std::pow(c.global_theta,
                                  static_cast<float>(2 * i) /
                                      static_cast<float>(c.global_head_dim));
    sliding_freq = Buffer(sliding.size() * 4);
    global_freq = Buffer(global.size() * 4);
    HipCheck(hipMemcpy(sliding_freq.as<void>(), sliding.data(),
                       sliding.size() * 4, hipMemcpyHostToDevice),
             "rope upload");
    HipCheck(hipMemcpy(global_freq.as<void>(), global.data(), global.size() * 4,
                       hipMemcpyHostToDevice),
             "rope upload");

    lm_head = Buffer(V * H * 2);
    kernels::BlockLinearWeights(device_weights->view().embed,
                                lm_head.as<bf16_t>(), c.vocab, c.hidden,
                                stream);
    // Partials for the largest split reduction of any pass's products.
    std::size_t partials = kernels::LinearPartialFloats(c.canvas, c.vocab, H);
    const std::pair<int, int> shapes[] = {
        {static_cast<int>(qkv_width), c.hidden},
        {static_cast<int>(qkv_width_sliding), c.hidden},
        {c.hidden, static_cast<int>(attn_width)},
        {c.hidden, c.heads * c.sliding_head_dim},
        {2 * c.intermediate, c.hidden},
        {c.hidden, c.intermediate},
        {c.experts, c.hidden}};
    for (int m = 1; m <= rows; ++m)
      for (const auto& [n, k] : shapes)
        partials = std::max(partials, kernels::LinearPartialFloats(m, n, k));
    if (partials > 0)
      linear_partials = Buffer(partials * 4);
    HipCheck(hipStreamSynchronize(stream), "blocking the LM head");
  }

  /// out (fp32) = a (row-major bf16, `t` rows) * W^T, W blocked.
  void Dense(const void* a, const bf16_t* w, float* out, int t, int n, int k) {
    kernels::Linear(static_cast<const bf16_t*>(a), w,
                    linear_partials.as<float>(), out, t, n, k, stream);
  }

  /// Input norm, QKV, Q/K norms and RoPE (writing this pass's keys and
  /// values to each segment's caches), attention and the output projection;
  /// leaves the three feed-forward inputs (ff, ff2, rout).
  void AttentionBlock(int l, int t) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    const bool global = c.is_global[l];
    const int H = c.hidden, hd = c.head_dim(l), kv = c.kv_heads(l);
    const int qkv_rows = (c.heads + (global ? 1 : 2) * kv) * hd;
    profiler->Phase(global ? "qkv_global" : "qkv_sliding");
    kernels::RmsNormToBf16(h.as<float>(), w.input_norm, xn.as<bf16_t>(), t, H,
                           c.rms_eps, stream);
    profiler->Phase(global ? "qkv_dense_global" : "qkv_dense_sliding");
    Dense(xn.as<void>(), w.qkv, qkv.as<float>(), t, qkv_rows, H);
    for (const auto& segment : segments) {
      const CacheState& cache = segment.session->cache;
      const bool staged = segment.staged && !global;
      bf16_t* k_cache =
          (staged ? k_stage[l] : segment.session->k_cache[l]).as<bf16_t>();
      bf16_t* v_cache =
          (staged ? v_stage[l] : segment.session->v_cache[l]).as<bf16_t>();
      const int capacity = global   ? global_capacity
                           : staged ? stage_capacity
                                    : sliding_capacity;
      const auto pass = global ? cache.GlobalPass() : cache.SlidingPass();
      const std::size_t skip_elements =
          static_cast<std::size_t>(pass.skip) * hd;
      const std::size_t q_offset =
          static_cast<std::size_t>(segment.row0) * c.heads * hd;
      profiler->Phase(global ? "rope_global" : "rope_sliding");
      kernels::QkvNormRope(
          qkv.as<float>() + static_cast<std::size_t>(segment.row0) * qkv_rows,
          w.q_norm, w.k_norm,
          kernels::RopeTable{global ? global_freq.as<float>()
                                    : sliding_freq.as<float>()},
          q.as<bf16_t>() + q_offset, k_cache, v_cache, segment.rows, c.heads,
          kv, hd, global, cache.length(), pass.first_slot, capacity, c.rms_eps,
          stream);
      // Text encoder passes are causal; image rows see their full span.
      // Sliding layers retain their window in either case.
      profiler->Phase(global ? "attn_global" : "attn_sliding");
      kernels::Attention(
          q.as<bf16_t>() + q_offset, k_cache + skip_elements,
          v_cache + skip_elements, attn.as<bf16_t>() + q_offset, segment.rows,
          c.heads, kv, hd, capacity, pass.first_slot + segment.rows - pass.skip,
          cache.length(), pass.first_position,
          /*causal=*/!segment.decoder && !segment.image,
          (!segment.decoder && !global) ? c.sliding_window : 0, stream,
          kernels::SplitWorkspace{.out = split_out.as<float>(),
                                  .stats = split_stats.as<float>(),
                                  .capacity = split_out.bytes() / 4,
                                  .max_splits = kMaxSplits});
    }
    profiler->Phase(global ? "o_global" : "o_sliding");
    Dense(attn.as<void>(), w.o, o.as<float>(), t, H, c.heads * hd);
    profiler->Phase("post_attention");
    kernels::PostAttention(h.as<float>(), o.as<float>(), w.post_attention_norm,
                           w.pre_ff_norm, w.pre_ff_norm_2, w.router_scale,
                           ff.as<bf16_t>(), ff2.as<bf16_t>(), rout.as<bf16_t>(),
                           t, H, c.rms_eps, stream);
  }

  /// Dense MLP on `ff` into `mlp_out`.
  void DenseMlpBlock(int l, int t) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    profiler->Phase("dense_mlp");
    kernels::LinearGeluMul(ff.as<bf16_t>(), w.mlp_gate_up, act.as<bf16_t>(), t,
                           c.intermediate, c.hidden, stream);
    Dense(act.as<void>(), w.mlp_down, mlp_out.as<float>(), t, c.hidden,
          c.intermediate);
  }

  /// Rows per expert group for a pass of `t` rows: 128 once experts average
  /// at least 32 rows (t * top_k / experts: a prompt chunk, or two or more
  /// batched canvases), so that each widened weight tile serves more rows;
  /// else 64 (a canvas averages 16), and always 64 for Q8_0, whose
  /// per-block scaling makes the larger tile slower.
  int ExpertGroupRows(int t) const {
    const bool q8 =
        device_weights->view().layers[0].expert_gate_up_q8 != nullptr;
    return t * config.top_k >= 32 * config.experts && !q8 ? 128 : 64;
  }

  /// Mixture of experts on `ff2` (routed by `rout`): per-assignment outputs
  /// in `expert_out`, combined by FinishLayer.
  void ExpertBlock(int l, int t) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    const int H = c.hidden;
    profiler->Phase("router");
    Dense(rout.as<void>(), w.router_proj, router_logits.as<float>(), t,
          c.experts, H);
    kernels::Route(router_logits.as<float>(), w.per_expert_scale,
                   top_index.as<std::int32_t>(), top_weight.as<float>(), t,
                   c.experts, c.top_k, stream);
    const int group_rows = ExpertGroupRows(t);
    kernels::Dispatch(top_index.as<std::int32_t>(), counts.as<std::int32_t>(),
                      offsets.as<std::int32_t>(), source.as<std::int32_t>(),
                      slot.as<std::int32_t>(), groups.as<std::int32_t>(),
                      group_count.as<std::int32_t>(), t, c.experts, c.top_k,
                      group_rows, stream);
    const kernels::ExpertRouting routing{
        .counts = counts.as<std::int32_t>(),
        .offsets = offsets.as<std::int32_t>(),
        .source = source.as<std::int32_t>(),
        .groups = groups.as<std::int32_t>(),
        .group_count = group_count.as<std::int32_t>(),
        .max_groups =
            kernels::ExpertGroupBound(t, c.experts, c.top_k, group_rows),
        .group_rows = group_rows};
    if (profiler->enabled() && t == c.canvas)
      CountActiveExperts();
    profiler->Phase("expert_gate_up");
    kernels::ExpertGateUp(
        ff2.as<bf16_t>(), w.expert_gate_up, routing, expert_act.as<bf16_t>(), H,
        c.expert_intermediate, stream,
        kernels::ExpertQ8{w.expert_gate_up_q8, w.expert_gate_up_scale},
        kernels::ExpertFp8{w.expert_gate_up_fp8, w.expert_gate_up_fp8_scale},
        kernels::ExpertFp4{w.expert_gate_up_fp4,
                           w.expert_gate_up_fp4_block_scale,
                           w.expert_gate_up_fp4_tensor_scale});
    profiler->Phase("expert_down");
    kernels::ExpertDown(
        expert_act.as<bf16_t>(), w.expert_down, routing,
        expert_out.as<bf16_t>(), H, c.expert_intermediate, stream,
        kernels::ExpertQ8{w.expert_down_q8, w.expert_down_scale},
        kernels::ExpertFp8{w.expert_down_fp8, w.expert_down_fp8_scale},
        kernels::ExpertFp4{w.expert_down_fp4, w.expert_down_fp4_block_scale,
                           w.expert_down_fp4_tensor_scale});
  }

  /// Profile mode: experts with at least one row in this canvas pass.
  void CountActiveExperts() {
    std::vector<std::int32_t> host(config.experts);
    HipCheck(hipMemcpyAsync(host.data(), counts.as<void>(), host.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "profile counts");
    HipCheck(hipStreamSynchronize(stream), "profile counts");
    for (const auto n : host)
      active_experts += n > 0 ? 1 : 0;
    ++expert_passes;
  }

  /// Runs every layer over the `t` rows in `h`, each segment's rows against
  /// its own session's caches.
  void Layers(int t) {
    const Config& c = config;
    for (int l = 0; l < c.layers; ++l) {
      const auto& w = device_weights->view().layers[l];
      AttentionBlock(l, t);
      DenseMlpBlock(l, t);
      ExpertBlock(l, t);
      profiler->Phase("finish");
      kernels::FinishLayer(
          h.as<float>(), mlp_out.as<float>(),
          kernels::ExpertOutputs{.expert_out = expert_out.as<bf16_t>(),
                                 .slot = slot.as<std::int32_t>(),
                                 .weight = top_weight.as<float>(),
                                 .k = c.top_k},
          w.post_ff_norm_1, w.post_ff_norm_2, w.post_ff_norm, w.layer_scalar, t,
          c.hidden, c.rms_eps, stream);
    }
  }

  /// Copies `count` positions of a sliding layer's K or V cache, every head,
  /// between caches of different capacities.
  void CopyPositions(const Buffer& from, int from_capacity, int from_slot,
                     const Buffer& to, int to_capacity, int to_slot,
                     int count) {
    if (count == 0)
      return;
    const Config& c = config;
    const std::size_t row = static_cast<std::size_t>(c.sliding_head_dim) * 2;
    HipCheck(
        hipMemcpy2DAsync(to.as<char>() + to_slot * row, to_capacity * row,
                         from.as<char>() + from_slot * row, from_capacity * row,
                         count * row, c.sliding_kv_heads,
                         hipMemcpyDeviceToDevice, stream),
        "sliding cache copy");
  }

  /// One text encoder pass over `t` tokens appended to `session`'s cache.
  void EncodeChunk(Session& session, const std::int32_t* tokens, int t) {
    const EncodeWork work{.session = &session, .tokens = tokens, .rows = t};
    Forward({}, std::span<const EncodeWork>(&work, 1));
  }

  /// Encodes `tokens` into `session`'s cache in prompt chunks of at most
  /// `chunk` tokens; returns elapsed milliseconds.
  double Encode(Session& session, std::span<const std::int32_t> tokens) {
    if (session.cache.length() + static_cast<int>(tokens.size()) >
        options.max_context)
      throw std::length_error(
          "prompt of " + std::to_string(session.cache.length() + tokens.size()) +
          " tokens does not fit the context of " +
          std::to_string(options.max_context) +
          " tokens; raise --context or shorten the prompt");
    const auto start = Clock::now();
    for (std::size_t done = 0; done < tokens.size();) {
      const int t = static_cast<int>(
          std::min<std::size_t>(options.prefill_chunk, tokens.size() - done));
      EncodeChunk(session, tokens.data() + done, t);
      done += t;
    }
    HipCheck(hipStreamSynchronize(stream), "encode");
    return Ms(start, Clock::now());
  }

  struct ScoreHost {
    std::vector<std::int32_t> top_ids;
    std::vector<float> top_logprobs;
    std::vector<float> label_logprobs;
    std::vector<float> logsumexp;
    std::vector<std::int32_t> argmax;
  };

  /// One canvas of a batched denoising step: its session, tokens, sampler
  /// settings, and where its per-row results go.
  struct CanvasStep {
    Session* session{nullptr};
    const std::int32_t* tokens{nullptr};
    bool self_condition{false};
    bool make_probabilities{true};
    bool has_pins{false};
    float temperature{1.0F};
    std::uint64_t seed{0};
    kernels::SuppressedTokens suppressed;
    float* entropy{nullptr};
    std::int32_t* argmax{nullptr};
    std::int32_t* sample{nullptr};
    ScoreHost* scores{nullptr};
    const std::int32_t* label_ids{nullptr};
    int label_count{0};
  };

  /// Causal rows appended to a session's cache: a prompt chunk or a
  /// committed canvas.
  struct EncodeWork {
    Session* session{nullptr};
    const std::int32_t* tokens{nullptr};
    int rows{0};
    const ImageInput* image{nullptr};
  };

  void DenoiseStep(std::span<const CanvasStep> batch) { Forward(batch, {}); }

  /// One forward pass over the stacked rows of `canvases` (a denoising step
  /// each, after its own session's cache) and then `encodes` (prompt rows
  /// appended to their sessions' caches; at most one may need the stage
  /// caches). Leaves each canvas's per-row entropy, argmax and sample in its
  /// host arrays and its self-conditioning probabilities in its session.
  void Forward(std::span<const CanvasStep> canvases,
               std::span<const EncodeWork> encodes) {
    const Config& c = config;
    const auto& weights = device_weights->view();
    const int n = c.canvas, H = c.hidden;
    const int decode_rows = n * static_cast<int>(canvases.size());
    int t = decode_rows;
    for (const auto& work : encodes)
      t += work.rows;
    const auto row = [&](std::size_t b, int width) { return b * n * width; };
    std::vector<std::int32_t> pass_ids;
    pass_ids.reserve(t);
    for (const auto& canvas : canvases)
      pass_ids.insert(pass_ids.end(), canvas.tokens, canvas.tokens + n);
    for (const auto& work : encodes)
      pass_ids.insert(pass_ids.end(), work.tokens, work.tokens + work.rows);
    HipCheck(hipMemcpyAsync(ids.as<void>(), pass_ids.data(), t * 4,
                            hipMemcpyHostToDevice, stream),
             "token upload");
    segments.clear();
    if (!canvases.empty()) {
      profiler->Phase("self_condition");
      kernels::Embed(ids.as<std::int32_t>(), weights.embed, embed_scale,
                     embeds.as<float>(), decode_rows, H, stream);
      bool any_self_condition = false;
      for (std::size_t b = 0; b < canvases.size(); ++b) {
        if (!canvases[b].self_condition)
          continue;
        any_self_condition = true;
        kernels::SoftEmbedProduct(canvases[b].session->probs.as<bf16_t>(),
                                  weights.embed, soft_partials.as<float>(),
                                  soft.as<float>() + row(b, H), n, c.vocab, H,
                                  kSoftEmbedSplits, stream);
        if (canvases[b].has_pins)
          kernels::ZeroPinnedSoftRows(
              soft.as<float>() + row(b, H),
              canvases[b].session->pinned.as<std::uint8_t>(), n, H, stream);
      }
      if (any_self_condition) {
        // Over every canvas row: canvases without self-conditioning ignore
        // theirs.
        kernels::SoftEmbedNorm(soft.as<float>(), weights.sc_pre_norm,
                               embed_scale, soft_norm.as<bf16_t>(), decode_rows,
                               H, c.rms_eps, stream);
        kernels::LinearGeluMul(soft_norm.as<bf16_t>(), weights.sc_gate_up,
                               act.as<bf16_t>(), decode_rows, c.intermediate, H,
                               stream);
        Dense(act.as<void>(), weights.sc_down, signal.as<float>(), decode_rows,
              H, c.intermediate);
      }
      for (std::size_t b = 0; b < canvases.size(); ++b) {
        kernels::SelfConditionCombine(
            embeds.as<float>() + row(b, H),
            canvases[b].self_condition ? signal.as<float>() + row(b, H)
                                       : nullptr,
            h.as<float>() + row(b, H), n, H, c.rms_eps, stream);
        segments.push_back(Segment{.row0 = static_cast<int>(row(b, 1)),
                                   .rows = n,
                                   .session = canvases[b].session,
                                   .decoder = true});
      }
    }
    int row0 = decode_rows;
    bool stage_used = false;
    for (const auto& work : encodes) {
      Session& session = *work.session;
      const bool staged =
          session.cache.sliding_valid() + work.rows > sliding_capacity;
      if (staged) {
        if (stage_used)
          throw std::logic_error("two prompt chunks need the stage caches");
        stage_used = true;
        for (int l = 0; l < c.layers; ++l)
          if (!c.is_global[l]) {
            CopyPositions(session.k_cache[l], sliding_capacity, 0, k_stage[l],
                          stage_capacity, 0, session.cache.sliding_valid());
            CopyPositions(session.v_cache[l], sliding_capacity, 0, v_stage[l],
                          stage_capacity, 0, session.cache.sliding_valid());
          }
      }
      profiler->Phase("encode_embed");
      if (work.image) {
        HipCheck(hipMemcpyAsync(
                     h.as<float>() + static_cast<std::size_t>(row0) * H,
                     work.image->embedding,
                     static_cast<std::size_t>(work.rows) * H * sizeof(float),
                     hipMemcpyDeviceToDevice, stream),
                 "image embedding copy");
      } else {
        kernels::Embed(ids.as<std::int32_t>() + row0, weights.embed,
                       embed_scale,
                       h.as<float>() + static_cast<std::size_t>(row0) * H,
                       work.rows, H, stream);
      }
      segments.push_back(Segment{.row0 = row0,
                                 .rows = work.rows,
                                 .session = &session,
                                 .decoder = false,
                                 .image = work.image != nullptr,
                                 .staged = staged});
      row0 += work.rows;
    }
    Layers(t);
    if (!canvases.empty()) {
      profiler->Phase("lm_head");
      kernels::RmsNormToBf16(h.as<float>(), weights.final_norm, xn.as<bf16_t>(),
                             decode_rows, H, c.rms_eps, stream);
      Dense(xn.as<void>(), lm_head.as<bf16_t>(), logits.as<float>(),
            decode_rows, c.vocab, H);
      profiler->Phase("sampler");
      for (std::size_t b = 0; b < canvases.size(); ++b)
        kernels::SampleRows(logits.as<float>() + row(b, c.vocab),
                            canvases[b].make_probabilities
                                ? canvases[b].session->probs.as<bf16_t>()
                                : nullptr,
                            entropy.as<float>() + row(b, 1),
                            argmax.as<std::int32_t>() + row(b, 1),
                            sample.as<std::int32_t>() + row(b, 1), n, c.vocab,
                            c.final_softcap, canvases[b].temperature,
                            canvases[b].seed, canvases[b].suppressed, stream);
      for (std::size_t b = 0; b < canvases.size(); ++b) {
        const auto& canvas = canvases[b];
        if (!canvas.scores)
          continue;
        const auto score = kernels::ScoreWorkspace{
            .tile_max = score_tile_max.as<float>(),
            .tile_sum = score_tile_sum.as<float>(),
            .tile_ids = score_tile_ids.as<std::int32_t>(),
            .tile_values = score_tile_values.as<float>(),
            .top_ids = score_top_ids.as<std::int32_t>(),
            .top_logprobs = score_top_values.as<float>(),
            .label_logprobs = score_labels.as<float>(),
            .logsumexp = score_lse.as<float>(),
            .argmax = score_argmax.as<std::int32_t>()};
        if (canvas.label_count)
          HipCheck(hipMemcpyAsync(score_label_ids.as<void>(), canvas.label_ids,
                                  canvas.label_count * 4, hipMemcpyHostToDevice,
                                  stream),
                   "read label upload");
        kernels::ScoreRows(logits.as<float>() + row(b, c.vocab), n, c.vocab,
                           c.final_softcap, score_label_ids.as<std::int32_t>(),
                           canvas.label_count, score, stream);
        HipCheck(hipMemcpyAsync(canvas.scores->top_ids.data(),
                                score_top_ids.as<void>(), n * 20 * 4,
                                hipMemcpyDeviceToHost, stream),
                 "read top ids");
        HipCheck(hipMemcpyAsync(canvas.scores->top_logprobs.data(),
                                score_top_values.as<void>(), n * 20 * 4,
                                hipMemcpyDeviceToHost, stream),
                 "read top scores");
        HipCheck(hipMemcpyAsync(canvas.scores->logsumexp.data(),
                                score_lse.as<void>(), n * 4,
                                hipMemcpyDeviceToHost, stream),
                 "read logsumexp");
        HipCheck(hipMemcpyAsync(canvas.scores->argmax.data(),
                                score_argmax.as<void>(), n * 4,
                                hipMemcpyDeviceToHost, stream),
                 "read argmax");
        if (canvas.label_count)
          HipCheck(hipMemcpyAsync(canvas.scores->label_logprobs.data(),
                                  score_labels.as<void>(),
                                  n * canvas.label_count * 4,
                                  hipMemcpyDeviceToHost, stream),
                   "read label scores");
      }
    }
    // The encoded rows join their caches; the sliding caches keep the
    // newest positions at the front.
    profiler->Phase("cache_shift");
    for (std::size_t e = 0; e < encodes.size(); ++e) {
      const Segment& segment = segments[canvases.size() + e];
      Session& session = *segment.session;
      const auto shift = session.cache.Append(
          std::span<const std::int32_t>(encodes[e].tokens, encodes[e].rows),
          encodes[e].image
              ? std::span<const ImageSpan>(
                    static_cast<const ImageSpan*>(encodes[e].image), 1)
              : std::span<const ImageSpan>{});
      for (int l = 0; l < c.layers; ++l) {
        if (c.is_global[l])
          continue;
        if (segment.staged) {
          CopyPositions(k_stage[l], stage_capacity, shift.from,
                        session.k_cache[l], sliding_capacity, 0, shift.count);
          CopyPositions(v_stage[l], stage_capacity, shift.from,
                        session.v_cache[l], sliding_capacity, 0, shift.count);
          continue;
        }
        for (auto* buffer : {&session.k_cache[l], &session.v_cache[l]})
          kernels::ShiftCache(buffer->as<bf16_t>(), shift_scratch.as<bf16_t>(),
                              c.sliding_kv_heads, sliding_capacity,
                              c.sliding_head_dim, shift.from, shift.count,
                              stream);
      }
    }
    if (!canvases.empty()) {
      profiler->Phase("readback");
      for (std::size_t b = 0; b < canvases.size(); ++b) {
        HipCheck(
            hipMemcpyAsync(canvases[b].entropy, entropy.as<float>() + row(b, 1),
                           n * 4, hipMemcpyDeviceToHost, stream),
            "readback");
        HipCheck(hipMemcpyAsync(canvases[b].argmax,
                                argmax.as<std::int32_t>() + row(b, 1), n * 4,
                                hipMemcpyDeviceToHost, stream),
                 "readback");
        HipCheck(hipMemcpyAsync(canvases[b].sample,
                                sample.as<std::int32_t>() + row(b, 1), n * 4,
                                hipMemcpyDeviceToHost, stream),
                 "readback");
      }
    }
    HipCheck(hipStreamSynchronize(stream), "forward pass");
  }

  // ------------------------------------------------------------ requests

  /// One request: the caller's inputs, the worker's progress through it,
  /// and the finished canvases waiting for the caller.
  struct Job {
    std::vector<std::int32_t> prompt;
    std::vector<ImageInput> images;
    std::vector<ImageSpan> image_identities;
    GenerationOptions options;  // `cancelled` stays with the caller.
    std::optional<ReadOptions> read;
    ReadResult read_result;
    GenerationStats stats;
    Clock::time_point start;
    std::atomic<bool> cancel{false};
    // Shared with the caller.
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<std::int32_t>> blocks;
    bool finished{false};
    std::exception_ptr error;
    // Worker only.
    enum class Phase { kPrefill, kDenoise, kDone } phase{Phase::kPrefill};
    bool commit_pending{false};  // A finished canvas waits to be encoded.
    Session* session{nullptr};
    std::size_t encoded{0};  // Prompt tokens in the session's cache.
    std::mt19937_64 rng;
    kernels::SuppressedTokens suppressed;
    int max_blocks{0};
    int block{0};
    // The canvas being denoised.
    std::uint64_t block_seed{0};
    int steps{0};
    Clock::time_point block_start;
    std::vector<std::int32_t> current, argmax_canvas, previous_argmax;
    std::vector<float> row_entropy;
    std::vector<std::int32_t> row_argmax, row_sample;
    std::vector<std::int32_t> read_seed;
    std::vector<std::uint8_t> pins;
    ScoreHost scores;
  };

  /// Prompt chunk size while other requests are denoising, so that they
  /// wait for at most one such chunk between steps.
  static constexpr int kInterleavedChunk = 2048;

  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<std::shared_ptr<Job>> queue;
  bool stopping{false};
  std::vector<std::shared_ptr<Job>> active;  // Worker only.
  // Held by the worker around GPU work and by the benchmark primitives.
  std::mutex gpu_mutex;
  std::thread worker;

  void CancelAndWait(const std::shared_ptr<Job>& job) {
    job->cancel = true;
    queue_ready.notify_all();
    std::unique_lock<std::mutex> lock(job->mutex);
    job->ready.wait(lock, [&] { return job->finished; });
  }

  void StartWorker() {
    worker = std::thread([this] { Work(); });
  }

  void StopWorker() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    queue_ready.notify_all();
    if (worker.joinable())
      worker.join();
  }

  /// The free session for `job`: the one that reuses the longest prompt
  /// prefix, else the least recently used; null when all are busy.
  Session* PickSession(const Job& job) {
    Session* best = nullptr;
    int best_reuse = -1;
    for (auto& session : sessions) {
      if (session.busy)
        continue;
      const int reuse =
          job.options.reuse_prompt
              ? session.cache.ReusableLength(job.prompt, job.image_identities)
              : 0;
      if (reuse > best_reuse ||
          (reuse == best_reuse && session.last_used < best->last_used)) {
        best = &session;
        best_reuse = reuse;
      }
    }
    return best;
  }

  void Admit(Job& job, Session& session) {
    job.session = &session;
    session.busy = true;
    session.last_used = ++session_clock;
    if (job.options.reuse_prompt) {
      job.stats.reused_tokens =
          session.cache.Reuse(job.prompt, job.image_identities);
    } else {
      session.cache.Clear();
      job.stats.reused_tokens = 0;
    }
    job.encoded = static_cast<std::size_t>(job.stats.reused_tokens);
  }

  void Work() {
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_ready.wait(lock, [&] {
          return stopping || !active.empty() ||
                 (!queue.empty() && PickSession(*queue.front()) != nullptr);
        });
        if (stopping)
          break;
        while (!queue.empty()) {
          Session* session = PickSession(*queue.front());
          if (session == nullptr)
            break;
          auto job = std::move(queue.front());
          queue.pop_front();
          Admit(*job, *session);
          active.push_back(std::move(job));
        }
      }
      std::lock_guard<std::mutex> gpu(gpu_mutex);
      try {
        Advance();
      } catch (...) {
        const auto error = std::current_exception();
        for (auto& job : active) {
          job->error = error;
          Finish(*job);
        }
      }
      std::erase_if(active, [](const auto& job) {
        return job->phase == Job::Phase::kDone;
      });
    }
    // Shutting down: nothing more runs.
    std::lock_guard<std::mutex> lock(queue_mutex);
    const auto error = std::make_exception_ptr(
        std::runtime_error("DiffusionGemma engine stopped"));
    for (auto& job : active) {
      job->error = error;
      Finish(*job);
    }
    for (auto& job : queue) {
      job->error = error;
      Finish(*job);
    }
  }

  /// One forward pass over every request's pending work: the canvases being
  /// denoised (one step each), then finished canvases to commit and prompt
  /// chunks (oldest request first), within the pass's row budget.
  void Advance() {
    for (auto& job : active)
      if (job->cancel.load() && job->phase != Job::Phase::kDone) {
        job->stats.cancelled = true;
        Finish(*job);
      }
    const Config& c = config;
    std::vector<Job*> stepping, committing, prefilling;
    std::vector<int> prefill_rows;
    for (auto& job : active)
      if (job->phase == Job::Phase::kDenoise && !job->commit_pending)
        stepping.push_back(job.get());
    // Prompt rows this pass may add: a whole chunk on its own, else
    // at most kInterleavedChunk so that denoising requests are not held up.
    int budget = stepping.empty()
                     ? options.prefill_chunk
                     : std::min(options.prefill_chunk, kInterleavedChunk);
    budget =
        std::min(budget, rows - c.canvas * static_cast<int>(stepping.size()));
    std::vector<EncodeWork> encodes;
    for (auto& job : active)
      if (job->phase == Job::Phase::kDenoise && job->commit_pending &&
          budget >= c.canvas) {
        encodes.push_back(EncodeWork{.session = job->session,
                                     .tokens = job->argmax_canvas.data(),
                                     .rows = c.canvas});
        committing.push_back(job.get());
        budget -= c.canvas;
      }
    bool staged = false;
    for (auto& job : active) {
      if (job->phase != Job::Phase::kPrefill)
        continue;
      const int remaining = static_cast<int>(job->prompt.size() - job->encoded);
      const PromptChunk chunk =
          job->images.empty()
              ? PromptChunk{std::min(remaining, budget), -1}
              : PlanPromptChunk(job->prompt.size(), job->encoded, budget,
                                options.prefill_chunk, options.max_context,
                                job->image_identities);
      const int take = chunk.rows;
      if (remaining > 0 && take <= 0)
        break;
      // Only one chunk per pass can use the stage caches.
      const bool needs_stage =
          job->session->cache.sliding_valid() + take > sliding_capacity;
      if (take > 0 && needs_stage && staged)
        continue;
      staged = staged || (take > 0 && needs_stage);
      if (take > 0)
        encodes.push_back(EncodeWork{
            .session = job->session,
            .tokens = job->prompt.data() + job->encoded,
            .rows = take,
            .image = chunk.image >= 0 ? &job->images[chunk.image] : nullptr});
      prefilling.push_back(job.get());
      prefill_rows.push_back(take);
      budget -= take;
    }
    std::vector<CanvasStep> batch;
    for (Job* job : stepping) {
      const DiffusionDefaults& d = job->options.diffusion;
      const ReadStepPlan plan =
          job->read ? PlanReadStep(job->steps, job->read->max_steps)
                    : ReadStepPlan{};
      // Each step draws its Gumbel noise from its own stream.
      batch.push_back(CanvasStep{
          .session = job->session,
          .tokens = job->current.data(),
          .self_condition = job->read ? plan.self_condition : job->steps > 0,
          .make_probabilities = !job->read || plan.make_probabilities,
          .has_pins = job->read && !job->read->pinned_positions.empty(),
          .temperature = StepTemperature(d, d.max_denoising_steps - job->steps),
          .seed = MixSeed(job->block_seed ^
                          (static_cast<std::uint64_t>(job->steps) << 40)),
          .suppressed = job->suppressed,
          .entropy = job->row_entropy.data(),
          .argmax = job->row_argmax.data(),
          .sample = job->row_sample.data(),
          .scores = job->read && plan.finish ? &job->scores : nullptr,
          .label_ids = job->read ? job->read->label_ids.data() : nullptr,
          .label_count =
              job->read ? static_cast<int>(job->read->label_ids.size()) : 0});
    }
    const auto start = Clock::now();
    if (!batch.empty() || !encodes.empty())
      Forward(batch, encodes);
    const double ms = Ms(start, Clock::now());
    for (std::size_t i = 0; i < prefilling.size(); ++i) {
      Job& job = *prefilling[i];
      job.encoded += prefill_rows[i];
      job.stats.prefill_ms += ms;
      if (job.encoded == job.prompt.size())
        StartGeneration(job);
    }
    for (Job* job : committing) {
      job->stats.commit_ms += ms;
      job->commit_pending = false;
      ++job->block;
      if (!StartBlock(*job))
        Finish(*job);
    }
    std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
    for (Job* job : stepping) {
      const DiffusionDefaults& d = job->options.diffusion;
      ++job->steps;
      if (job->read) {
        job->argmax_canvas = job->row_argmax;
        if (job->steps >= job->read->max_steps) {
          job->stats.denoise_ms += Ms(job->block_start, Clock::now());
          job->stats.denoising_steps = job->steps;
          job->stats.blocks = 1;
          job->stats.first_block_ms = Ms(job->start, Clock::now());
          job->read_result.stats = job->stats;
          job->read_result.rows.reserve(c.canvas);
          const int top_k = job->read->top_k;
          const int labels = static_cast<int>(job->read->label_ids.size());
          for (int row = 0; row < c.canvas; ++row) {
            ReadRow result;
            result.token = job->scores.argmax[row];
            result.logprob = job->scores.top_logprobs[row * 20];
            for (int rank = 0; rank < top_k; ++rank)
              result.top_logprobs.push_back(
                  {job->scores.top_ids[row * 20 + rank],
                   job->scores.top_logprobs[row * 20 + rank]});
            for (int label = 0; label < labels; ++label) {
              const int id = job->read->label_ids[label];
              if (std::none_of(result.top_logprobs.begin(),
                               result.top_logprobs.end(),
                               [id](const ReadCandidate& top) {
                                 return top.token == id;
                               }))
                result.label_logprobs.push_back(
                    {id, job->scores.label_logprobs[row * labels + label]});
            }
            job->read_result.rows.push_back(std::move(result));
          }
          Finish(*job);
          continue;
        }
        const auto accepted = AcceptRows(job->row_entropy, d.entropy_bound);
        std::vector<std::uint8_t> mask(c.canvas);
        for (int i = 0; i < c.canvas; ++i)
          mask[i] = accepted[i];
        AdvanceReadCanvas(job->current, job->read_seed, job->pins, mask,
                          job->row_sample, [&] { return uniform(job->rng); });
        continue;
      }
      const auto accepted = AcceptRows(job->row_entropy, d.entropy_bound);
      for (int i = 0; i < c.canvas; ++i)
        job->current[i] = accepted[i] ? job->row_sample[i] : uniform(job->rng);
      job->argmax_canvas = job->row_argmax;
      const bool done = BlockConverged(d, job->previous_argmax, job->row_argmax,
                                       job->row_entropy);
      job->previous_argmax = job->row_argmax;
      if (done || job->steps >= d.max_denoising_steps)
        FinishBlock(*job);
    }
  }

  /// The prompt is in the cache: set up the sampler and the first canvas.
  void StartGeneration(Job& job) {
    profiler->Report("prefill");
    const Config& c = config;
    job.rng.seed(MixSeed(job.options.seed));
    if (job.options.suppress_end)
      for (const int id : c.eos_ids)
        job.suppressed.ids[job.suppressed.count++] = id;
    job.max_blocks =
        std::max(1, (job.options.max_new_tokens + c.canvas - 1) / c.canvas);
    job.block = 0;
    if (StartBlock(job)) {
      job.phase = Job::Phase::kDenoise;
      if (job.read) {
        if (!job.read->seed_canvas.empty())
          job.current = job.read->seed_canvas;
        job.read_seed = job.current;
        job.pins.assign(c.canvas, 0);
        for (const int position : job.read->pinned_positions)
          job.pins[position] = 1;
        if (!job.read->pinned_positions.empty())
          HipCheck(
              hipMemcpyAsync(job.session->pinned.as<void>(), job.pins.data(),
                             c.canvas, hipMemcpyHostToDevice, stream),
              "read pins upload");
        job.scores.top_ids.resize(std::size_t(c.canvas) * 20);
        job.scores.top_logprobs.resize(std::size_t(c.canvas) * 20);
        job.scores.label_logprobs.resize(std::size_t(c.canvas) *
                                         job.read->label_ids.size());
        job.scores.logsumexp.resize(c.canvas);
        job.scores.argmax.resize(c.canvas);
      }
    } else
      Finish(job);
  }

  /// Starts the job's next canvas; false when the context is full.
  bool StartBlock(Job& job) {
    const Config& c = config;
    if (job.session->cache.length() + c.canvas > options.max_context)
      return false;
    job.block_seed =
        MixSeed(job.options.seed + kBlockSeedStride * (job.block + 1));
    // The canvas starts as random tokens; rows not accepted at a step are
    // drawn again.
    std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
    job.current.resize(c.canvas);
    for (auto& token : job.current)
      token = uniform(job.rng);
    job.row_entropy.resize(c.canvas);
    job.row_argmax.resize(c.canvas);
    job.row_sample.resize(c.canvas);
    job.previous_argmax.clear();
    job.steps = 0;
    job.block_start = Clock::now();
    return true;
  }

  void FinishBlock(Job& job) {
    const Config& c = config;
    GenerationStats& stats = job.stats;
    stats.denoise_ms += Ms(job.block_start, Clock::now());
    stats.denoising_steps += job.steps;
    ++stats.blocks;
    const auto reply =
        TakeReply(job.argmax_canvas, c.eos_ids,
                  job.options.max_new_tokens - stats.generated_tokens);
    stats.stopped_by_eos = reply.end_of_sequence;
    stats.generated_tokens += reply.tokens;
    if (job.block == 0)
      stats.first_block_ms = Ms(job.start, Clock::now());
    {
      std::lock_guard<std::mutex> lock(job.mutex);
      job.blocks.emplace_back(job.argmax_canvas.begin(),
                              job.argmax_canvas.begin() + reply.tokens);
    }
    job.ready.notify_all();
    if (stats.stopped_by_eos ||
        stats.generated_tokens >= job.options.max_new_tokens ||
        job.block + 1 == job.max_blocks || job.cancel.load()) {
      Finish(job);
      return;
    }
    // Reached only when the whole canvas belongs to the reply: it is
    // encoded into the cache in the next pass, then the next canvas starts.
    job.commit_pending = true;
  }

  void Finish(Job& job) {
    if (job.phase == Job::Phase::kDone)
      return;
    job.phase = Job::Phase::kDone;
    if (job.session != nullptr) {
      job.session->busy = false;
      job.session->last_used = ++session_clock;
    }
    if (profiler->enabled() && expert_passes > 0)
      std::fprintf(stderr, "profile active experts per canvas layer: %.1f\n",
                   static_cast<double>(active_experts) / expert_passes);
    profiler->Report("denoise and commit");
    {
      std::lock_guard<std::mutex> lock(job.mutex);
      job.finished = true;
    }
    job.ready.notify_all();
    // A freed session may admit a queued request.
    queue_ready.notify_all();
  }

  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           std::span<const ImageInput> images,
                           const GenerationOptions& request,
                           const BlockCallback& on_block) {
    const Config& c = config;
    if (prompt.empty())
      throw std::invalid_argument("empty prompt");
    if (request.suppress_end &&
        c.eos_ids.size() > kernels::SuppressedTokens::kMax)
      throw std::invalid_argument("too many end-of-sequence tokens");
    auto job = std::make_shared<Job>();
    job->prompt.assign(prompt.begin(), prompt.end());
    job->images.assign(images.begin(), images.end());
    job->image_identities.reserve(images.size());
    for (const auto& image : images) {
      if (!image.embedding)
        throw std::invalid_argument("image span has no encoder output");
      job->image_identities.push_back(image);
    }
    ValidateImagePrompt(prompt, options.prefill_chunk, options.max_context,
                        c.canvas, job->image_identities);
    job->options = request;
    job->options.cancelled = nullptr;
    job->stats.prompt_tokens = static_cast<int>(prompt.size());
    job->start = Clock::now();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (stopping)
        throw std::runtime_error("DiffusionGemma engine stopped");
      queue.push_back(job);
    }
    queue_ready.notify_all();
    bool consumer_stopped = false;
    try {
      for (;;) {
        std::deque<std::vector<std::int32_t>> ready;
        bool finished = false;
        {
          std::unique_lock<std::mutex> lock(job->mutex);
          job->ready.wait_for(lock, std::chrono::milliseconds(20), [&] {
            return !job->blocks.empty() || job->finished;
          });
          ready.swap(job->blocks);
          finished = job->finished;
        }
        if (request.cancelled && request.cancelled())
          job->cancel = true;
        for (const auto& block : ready) {
          if (consumer_stopped)
            break;
          if (on_block && !on_block(block)) {
            consumer_stopped = true;
            job->cancel = true;
          }
        }
        if (finished)
          break;
      }
    } catch (...) {
      CancelAndWait(job);
      throw;
    }
    if (job->error)
      std::rethrow_exception(job->error);
    GenerationStats stats = job->stats;
    stats.cancelled = stats.cancelled || consumer_stopped;
    return stats;
  }

  ReadResult Read(std::span<const std::int32_t> prompt,
                  std::span<const ImageInput> images,
                  const ReadOptions& request) {
    const Config& c = config;
    ValidateReadOptions(request, c.canvas, c.vocab);
    if (prompt.empty())
      throw std::invalid_argument("empty prompt");
    auto job = std::make_shared<Job>();
    job->prompt.assign(prompt.begin(), prompt.end());
    job->images.assign(images.begin(), images.end());
    job->image_identities.reserve(images.size());
    for (const auto& image : images) {
      if (!image.embedding)
        throw std::invalid_argument("image span has no encoder output");
      job->image_identities.push_back(image);
    }
    ValidateImagePrompt(prompt, options.prefill_chunk, options.max_context,
                        c.canvas, job->image_identities);
    job->options.diffusion = defaults;
    job->options.seed = request.seed;
    job->options.reuse_prompt = request.reuse_prompt;
    job->read = request;
    job->read->cancelled = nullptr;
    job->stats.prompt_tokens = static_cast<int>(prompt.size());
    job->start = Clock::now();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (stopping)
        throw std::runtime_error("DiffusionGemma engine stopped");
      queue.push_back(job);
    }
    queue_ready.notify_all();
    try {
      for (;;) {
        bool finished = false;
        {
          std::unique_lock<std::mutex> lock(job->mutex);
          job->ready.wait_for(lock, std::chrono::milliseconds(20),
                              [&] { return job->finished; });
          finished = job->finished;
        }
        if (request.cancelled && request.cancelled())
          job->cancel = true;
        if (finished)
          break;
      }
    } catch (...) {
      CancelAndWait(job);
      throw;
    }
    if (job->error)
      std::rethrow_exception(job->error);
    if (job->stats.cancelled)
      throw std::runtime_error("diffusion read cancelled");
    return std::move(job->read_result);
  }
};

Engine::Engine(const std::filesystem::path& root, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(root, options)) {}
Engine::~Engine() = default;

const Config& Engine::config() const {
  return impl_->config;
}
const Tokenizer& Engine::tokenizer() const {
  return *impl_->tokenizer;
}
const DiffusionDefaults& Engine::diffusion_defaults() const {
  return impl_->defaults;
}
int Engine::max_context() const {
  return impl_->options.max_context;
}
int Engine::prefill_chunk() const {
  return impl_->options.prefill_chunk;
}
std::size_t Engine::weight_bytes() const {
  return impl_->device_weights->view().device_bytes;
}

GenerationStats Engine::Generate(std::span<const std::int32_t> prompt,
                                 const GenerationOptions& options,
                                 const BlockCallback& on_block) {
  return impl_->Generate(prompt, {}, options, on_block);
}

GenerationStats Engine::Generate(std::span<const std::int32_t> prompt,
                                 std::span<const ImageInput> images,
                                 const GenerationOptions& options,
                                 const BlockCallback& on_block) {
  return impl_->Generate(prompt, images, options, on_block);
}

ReadResult Engine::Read(std::span<const std::int32_t> prompt,
                        const ReadOptions& options) {
  return impl_->Read(prompt, {}, options);
}

ReadResult Engine::Read(std::span<const std::int32_t> prompt,
                        std::span<const ImageInput> images,
                        const ReadOptions& options) {
  return impl_->Read(prompt, images, options);
}

double Engine::Warmup() {
  auto& impl = *impl_;
  const auto start = Clock::now();
  const int filler = impl.tokenizer->TokenId("\xE2\x96\x81the");
  // A context below the prefill chunk (--context 4096 with the default 8192
  // chunk) must still load: the warm-up chunk fits the context.
  std::vector<std::int32_t> tokens(
      std::min(impl.options.prefill_chunk, impl.options.max_context),
      filler >= 0 ? filler : 1);
  tokens[0] = impl.config.bos_id;
  // An encoder chunk, then one short request per session: their two
  // denoising steps (with and without self-conditioning) and commit load
  // every kernel, at every batch size up to all sessions, before the first
  // request.
  Reset();
  (void)Encode(tokens);
  GenerationOptions options;
  options.diffusion = impl.defaults;
  options.diffusion.max_denoising_steps = 2;
  options.max_new_tokens = 2 * impl.config.canvas;
  std::vector<std::thread> requests;
  for (int i = 0; i < impl.options.sessions; ++i)
    requests.emplace_back([&impl, &tokens, &options, i] {
      // Distinct prompts, so that each takes its own session.
      std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 64);
      prompt[1] = impl.config.bos_id + 1 + i;
      (void)impl.Generate(prompt, {}, options, {});
    });
  for (auto& request : requests)
    request.join();
  Reset();
  return Ms(start, Clock::now());
}

void Engine::Reset() {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  for (auto& session : impl_->sessions)
    session.cache.Clear();
}

double Engine::Encode(std::span<const std::int32_t> tokens) {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  const double ms = impl_->Encode(impl_->sessions[0], tokens);
  impl_->profiler->Report();
  return ms;
}

int Engine::cached_tokens() const {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  return impl_->sessions[0].cache.length();
}

std::vector<float> Engine::DebugLogits(
    std::span<const std::int32_t> prompt,
    const std::vector<std::int32_t>& canvas,
    const std::vector<std::int32_t>& canvas2) {
  Reset();
  auto& impl = *impl_;
  std::lock_guard<std::mutex> gpu(impl.gpu_mutex);
  auto& session = impl.sessions[0];
  const std::size_t rows = impl.config.canvas, vocab = impl.config.vocab;
  (void)impl.Encode(session, prompt);
  std::vector<float> out(2 * rows * vocab);
  std::vector<float> e(rows);
  std::vector<std::int32_t> a(rows), b(rows);
  const float cap = impl.config.final_softcap;
  for (int step = 0; step < 2; ++step) {
    const Impl::CanvasStep canvas_step{
        .session = &session,
        .tokens = (step == 0 ? canvas : canvas2).data(),
        .self_condition = step == 1,
        .temperature = 1.0F,
        .seed = static_cast<std::uint64_t>(step + 1),
        .suppressed = {},
        .entropy = e.data(),
        .argmax = a.data(),
        .sample = b.data()};
    impl.DenoiseStep(std::span<const Impl::CanvasStep>(&canvas_step, 1));
    float* dst = out.data() + step * rows * vocab;
    HipCheck(hipMemcpy(dst, impl.logits.as<void>(), rows * vocab * 4,
                       hipMemcpyDeviceToHost),
             "debug readback");
    for (std::size_t i = 0; i < rows * vocab; ++i)
      dst[i] = cap * std::tanh(dst[i] / cap);
  }
  return out;
}

}  // namespace gufo::models::diffusion_gemma
