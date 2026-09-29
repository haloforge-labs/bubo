#ifndef GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_
#define GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_a3b/decode_kernels.hpp"

/// Qwen3.6-35B-A3B (qwen35moe) exploration runtime: hybrid Gated DeltaNet /
/// gated attention trunk with a 256-expert top-8 MoE and a gated shared
/// expert. Built from Flash-Next's kernel launchers, never modifying them.
namespace gufo::models::qwen36_a3b {

struct DFlash;  // dflash.hpp

struct Config {
  std::uint32_t layers{0};
  std::uint32_t hidden{0};
  std::uint32_t vocab{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t rotary_dim{0};
  float rope_theta{0};
  float eps{0};
  std::uint32_t full_attention_interval{0};
  std::uint32_t ssm_conv_kernel{0};
  std::uint32_t ssm_head_dim{0};
  std::uint32_t ssm_k_heads{0};
  std::uint32_t ssm_v_heads{0};
  std::uint32_t experts{0};
  std::uint32_t experts_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t nextn{0};  // MTP blocks after the trunk (not in `layers`)

  bool IsLinear(std::uint32_t layer) const {
    return (layer + 1) % full_attention_interval != 0;
  }
  std::uint32_t ConvChannels() const {
    return (2 * ssm_k_heads + ssm_v_heads) * ssm_head_dim;
  }
  std::uint32_t SsmValueDim() const { return ssm_v_heads * ssm_head_dim; }
  std::uint32_t QDim() const { return heads * head_dim; }
  std::uint32_t KvDim() const { return kv_heads * head_dim; }
};

/// A device tensor in its GGUF encoding. rows = output dim, cols = input dim.
struct Tensor {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  const float* f32() const { return static_cast<const float*>(data); }
};

struct Layer {
  bool linear{false};
  Tensor attn_norm, post_norm;
  // Gated DeltaNet
  Tensor ssm_qkv, ssm_z, ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out;
  float* alpha_beta{nullptr};  // stacked [alpha ; beta] F32 rows
  // Gated attention
  Tensor attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_out;
  // MoE
  // Stacked [ffn_gate_inp ; ffn_gate_inp_shexp] rows, F32 or BF16.
  const void* router{nullptr};
  WeightKind router_type{WeightKind::kF32};
  Tensor gate_exps, up_exps, down_exps, sh_gate, sh_up, sh_down;
};

/// The nextn (MTP) block: a gated-attention layer with the MoE, fed by
/// eh_proj([enorm(embed(token)) ; hnorm(hidden)]) and read out through
/// shared_head_norm and the trunk's head.
struct MtpLayer {
  Layer block;
  Tensor eh_proj, enorm, hnorm, head_norm;
};

/// Top candidates of one logits row, descending by logit.
struct Candidates {
  static constexpr std::uint32_t kCount = 64;
  std::int32_t ids[kCount];
  float logits[kCount];
};

struct Options {
  std::uint32_t max_context{32768};
  /// Diagnostic: keep Flash-Next's sigmoid(z) GDN gate instead of silu(z).
  bool fn_gate{false};
  /// Diagnostic: route with Flash-Next's SmallGemm instead of F32Gemv.
  bool fn_router{false};
  /// Diagnostic: Flash-Next's per-head attention in the fused path instead
  /// of A3B's one-read-per-KV-head kernel (attention.hip).
  bool fn_attention{false};
  /// Replay the one-token decode step from a captured HIP graph.
  bool graphs{true};
  /// The A3B fused decode kernels (decode_kernels.hip); false runs the
  /// original chain of Flash-Next launchers, kept for A/B.
  bool fused{true};
  /// Diagnostic: fused parts (1 GDN mixer, 2 attention, 4 MoE); the others
  /// run the original launchers.
  std::uint32_t fused_parts{7};
  /// Diagnostic: time every phase with events (eager; disables graphs).
  bool profile{false};
  /// With profile: synchronize at every mark and use the host clock, so a
  /// phase's time cannot leak into its neighbour (adds sync latency).
  bool profile_sync{false};
  /// Diagnostic ablation: bit (1 << Phase) skips that phase's kernels, so a
  /// graph-replayed decode measures what the phase costs (output is garbage).
  std::uint32_t skip{0};
  /// Diagnostic: time the speculative step's parts (eager, host-synced).
  bool spec_profile{false};
  /// Token IDs (one per line) the MTP draft head is restricted to. Drafts
  /// only: verification always uses the full head, so output stays exact.
  std::string draft_vocab;
  /// The same restriction as a list of token IDs (used when non-empty).
  std::vector<std::int32_t> draft_vocab_ids;
  /// SpecPrefill chunk (tokens) for the wide prefill (prefill.cpp: Flash-
  /// Next's WMMA GEMMs, routed expert GEMMs and causal attention); 0 runs
  /// the prompt through the decode kernels 8 rows at a time.
  std::uint32_t prefill_chunk{2048};
};

enum Phase : int {
  kPhEmbed,
  kPhNorm,
  kPhGdnProj,
  kPhGdnCore,
  kPhGdnOut,
  kPhAttnProj,
  kPhAttnCore,
  kPhAttnOut,
  kPhRouter,
  kPhShared,
  kPhGateUp,
  kPhDown,
  kPhEpilogue,
  kPhHead,
  kPhCount
};

class Engine {
public:
  static constexpr std::uint32_t kMaxRows = 8;

  static std::unique_ptr<Engine> Load(const std::filesystem::path& path,
                                      const Options& options,
                                      std::string* error);
  ~Engine();

  const Config& config() const { return c_; }
  const core::GgufReader& gguf() const { return *gguf_; }
  std::uint32_t position() const { return pos_; }
  std::uint32_t max_context() const { return options_.max_context; }

  /// Clears the recurrent state and KV cache.
  void Reset();
  /// Runs up to kMaxRows tokens at the current position and advances it.
  /// When `logits` is non-null it receives the last token's logits, or with
  /// `all_rows` every token's logits ([tokens][vocab]).
  bool Forward(std::span<const std::int32_t> tokens, float* logits,
               std::string* error, bool all_rows = false);
  /// Blocks until the stream is idle.
  bool Sync(std::string* error);
  /// With Options::profile, prints the per-phase split of the profiled
  /// one-token forwards so far.
  void PrintProfile() const;
  /// Changes the ablation mask and drops the captured decode graph.
  void SetSkip(std::uint32_t mask);

  // --- Speculative decoding with the file's MTP block (fused path only). ---
  bool HasMtp() const { return c_.nextn > 0; }
  /// Prefills `tokens` at the current position, filling the MTP block's KV
  /// cache alongside, and returns the last row's candidates.
  bool SpecPrefill(std::span<const std::int32_t> tokens, Candidates* last,
                   std::string* error);
  /// One draft-and-verify step. `x` is the token at position(), not yet
  /// processed. Drafts `draft` tokens (0 = plain decode) with the MTP block,
  /// runs [x, drafts] through the trunk and returns the drafts and the
  /// candidates of all draft + 1 rows. Follow with SpecCommit.
  bool SpecStep(std::int32_t x, std::uint32_t draft,
                std::vector<std::int32_t>* drafts,
                std::vector<Candidates>* rows, std::string* error);
  /// Keeps the first `keep` verified rows (1..draft + 1) and rolls the
  /// recurrent state back to them.
  bool SpecCommit(std::uint32_t keep, std::string* error);
  // Host-driven steps (the caller owns the proposal policy): DraftFirst runs
  // the MTP catch-up and returns candidates for d1; DraftNext(d_j) returns
  // candidates for d_{j+1}; Verify runs [x, drafts] through the trunk; then
  // SpecCommit. Drafts may be any tokens (MTP, sampled, prompt lookup).
  // Verify with `catch_up` and no DraftFirst this step (prompt lookup
  // drafts) runs the MTP catch-up in the same graph, without the head, so
  // the MTP cache stays current for later steps.
  bool DraftFirst(std::int32_t x, Candidates* q, std::string* error);
  bool DraftNext(std::int32_t token, Candidates* q, std::string* error);
  bool Verify(std::int32_t x, std::span<const std::int32_t> drafts,
              std::vector<Candidates>* rows, std::string* error,
              bool catch_up = false);
  /// With Options::spec_profile, prints the speculative step's split.
  void PrintSpecProfile() const;
  /// Diagnostic: the wide prefill over `tokens` (at most one chunk) at the
  /// current position with every row's logits ([tokens][vocab]) to the host.
  bool WideLogits(std::span<const std::int32_t> tokens, float* logits,
                  std::string* error);
  /// Prompt checkpoint (checkpoint.cpp): SaveCheckpoint after a SpecPrefill
  /// keeps the recurrent state and position; RestoreCheckpoint returns to it
  /// (KV below it is intact), so a prompt extending that prefix prefills
  /// only its new tokens with SpecPrefill.
  bool SaveCheckpoint(std::string* error);
  bool RestoreCheckpoint(std::string* error);
  // --- Host state snapshots (state.cpp), for gufo serve's prompt cache. ---
  /// Bytes of SaveState: the recurrent (GDN and conv) state, the MTP
  /// catch-up rows and the DFlash context ring. The KV caches are not
  /// included: rows below position() stay intact until later tokens rewrite
  /// them, so a state restores only over the token history it was saved
  /// with (the caller keeps track).
  std::size_t StateBytes() const;
  /// Saves the state between steps (not inside a speculative step).
  bool SaveState(std::span<std::uint8_t> out, std::string* error);
  /// Restores SaveState's bytes; position() returns to the saved one.
  bool LoadState(std::span<const std::uint8_t> in, std::string* error);
  /// Full logits of row `row` of the last SpecPrefill (row 0) or verify pass,
  /// valid until the next trunk pass.
  bool RowLogits(std::uint32_t row, float* out, std::string* error);
  /// Writes a logits row into row 0 (a restored frontier for RowLogits).
  bool SetRowLogits(const float* in, std::string* error);
  // --- DFlash-2 drafting (dflash.cpp). ---
  /// Loads a DFlash-2 draft GGUF for this target; from then on every prefill
  /// and commit feeds the draft's context (call before any request).
  bool LoadDFlash(const std::filesystem::path& path, std::string* error);
  bool HasDFlash() const { return df_ != nullptr; }
  /// Most drafts one DFlashDraft returns (block size - 1, verify width - 1).
  std::uint32_t DFlashMaxDrafts() const;
  /// Proposes up to `count` tokens after `x` (the token at position(), not
  /// yet processed) and, with `probs`, each one's probability under the
  /// draft head; follow with Verify(x, drafts) and SpecCommit.
  bool DFlashDraft(std::int32_t x, std::uint32_t count,
                   std::vector<std::int32_t>* drafts, std::vector<float>* probs,
                   std::string* error);
  /// Diagnostic: Flash-Next's decode attention in the fused path (eager
  /// passes only: captured graphs keep the kernel they were built with).
  void SetFnAttention(bool on) { options_.fn_attention = on; }
  /// Switches SpecPrefill between the wide path (chunk > 0) and 8-row chunks.
  void SetPrefillChunk(std::uint32_t chunk) {
    if (wide_.rows == 0)
      options_.prefill_chunk = chunk;
    else
      options_.prefill_chunk = chunk == 0 ? 0 : wide_.rows;
  }

private:
  Engine() = default;
  bool Bind(std::string* error);
  bool Upload(std::string* error);
  bool Allocate(std::string* error);
  const void* Device(const core::GgufTensorInfo& t) const;
  bool Get(const std::string& name, Tensor* out, std::string* error);
  /// Stacks F32 or BF16 matrices (one type) into one device allocation.
  void* StackRows(std::initializer_list<const Tensor*> parts,
                  std::string* error);

  void Dense(const Tensor& w, const float* x, float* out, std::uint32_t n,
             const void* xq = nullptr);
  const void* Quantize(const float* x, std::uint32_t n, std::uint32_t k);
  void LinearAttention(const Layer& l, std::uint32_t idx, std::uint32_t n);
  void Attention(const Layer& l, std::uint32_t idx, std::uint32_t n);
  void Moe(const Layer& l, std::uint32_t n);
  /// Embedding, trunk and (for rows > 0) the head over the last `rows`.
  void Body(std::uint32_t n, std::uint32_t rows);
  /// The fused trunk over `n` rows of `tokens` at `pos`; the head runs on the
  /// last `rows`. `capture` keeps every row's final normed hidden in
  /// target_h_ (the MTP seed); `snapshots` records GDN rollback rows.
  void BodyFused(std::uint32_t n, std::uint32_t rows,
                 const std::int32_t* tokens, const std::uint32_t* pos,
                 bool capture, bool snapshots);
  void FusedGdn(const Layer& l, std::uint32_t idx, std::uint32_t n,
                bool snapshots);
  void FusedAttention(const Layer& l, __half* k_cache, __half* v_cache,
                      __half* vt_cache, const std::uint32_t* pos, float* res,
                      std::uint32_t n);
  void FusedMoe(const Layer& l, float* res, std::uint32_t n);
  /// The MTP block over `n` rows at positions pos..: tokens and seed hiddens
  /// in device memory. With `draft_out`, the last row's argmax goes there and
  /// its normed hidden to mtp_hn_ (the next chained seed).
  void MtpPass(const std::int32_t* tokens, const float* hidden, std::uint32_t n,
               const std::uint32_t* pos, std::int32_t* draft_out,
               bool candidates);
  bool LoadDraftVocab(std::string* error);
  bool BuildDraftHead(std::vector<std::int32_t> ids, std::string* error);
  /// Copies trunk layer `layer`'s output rows into their feature slot (no-op
  /// unless the layer is a DFlash tap); `wide` selects the prefill buffer.
  void DFlashTap(std::uint32_t layer, const float* res, std::uint32_t n,
                 bool wide);
  /// Feeds committed positions pos0..pos0 + n - 1 (their tapped features)
  /// into the draft's context ring; rows before end - window are skipped.
  /// pos_dev (optional) holds pos0 on the device for a single row group.
  bool DFlashInject(std::uint32_t n, std::uint32_t pos0, std::uint32_t end,
                    bool wide, const std::uint32_t* pos_dev,
                    std::string* error);
  void DFlashReset();
  /// The draft context ring with the prompt checkpoint (checkpoint.cpp).
  bool DFlashSaveCheckpoint(std::string* error);
  bool DFlashRestoreCheckpoint(std::string* error);
  bool DFlashDraftPass(std::int32_t x, std::uint32_t count,
                       std::vector<std::int32_t>* drafts,
                       std::vector<float>* probs, std::string* error);
  std::shared_ptr<DFlash> df_;
  bool BindMtp(std::string* error);

  // --- Wide prefill (prefill.cpp). ---
  bool SpecPrefillWide(std::span<const std::int32_t> tokens, Candidates* last,
                       std::string* error);
  bool AllocateWide(std::string* error);
  void FreeWide();
  /// The trunk over wide_.tokens (n rows at pos_); leaves wide_.res.
  bool WideTrunk(std::uint32_t n, std::string* error);
  /// out = x W^T over n rows: Q8_0 through the F16 or int8 WMMA GEMM, F32 and
  /// BF16 through hipBLAS. The staged F16 / tiled-Q8 / BF16 copy of x is
  /// reused while wide_.staged == x (Unstage() after rewriting x).
  bool WideDense(const Tensor& w, const float* x, float* out, std::uint32_t n,
                 std::string* error);
  void Unstage() {
    wide_.staged = nullptr;
    wide_.staged_kinds = 0;
  }
  bool WideGdn(const Layer& l, std::uint32_t idx, std::uint32_t n,
               std::string* error);
  bool WideAttention(const Layer& l, __half* k_cache, __half* v_cache,
                     __half* vt_cache, float* res, std::uint32_t n,
                     std::uint32_t start_pos, std::string* error);
  bool WideMoe(const Layer& l, float* res, std::uint32_t n, std::string* error);
  /// out = x W^T for BF16 weights; with x_lo the activations are the BF16
  /// split x + x_lo (a second, accumulating GEMM).
  bool BlasBf16(const void* w, const void* x, const void* x_lo, float* out,
                std::uint32_t m, std::uint32_t n, std::uint32_t k,
                std::string* error);
  // A3B_WIDE_PROFILE: per-phase event times of each SpecPrefill.
  enum WidePhase {
    kWGdnProj,
    kWGdnCore,
    kWGdnOut,
    kWAttnProj,
    kWAttnCore,
    kWAttnOut,
    kWRouter,
    kWShared,
    kWRouteHost,
    kWGateUp,
    kWDown,
    kWEpilogue,
    kWOther,
    kWMtp,
    kWBf16Exp,
    kWTopK,
    kWPhases
  };
  void WideMark(int phase);
  void WideReport(std::size_t tokens);
  struct Wide {
    std::uint32_t rows{0};  // allocated chunk
    void* blas{nullptr};    // hipblasHandle_t
    const float* staged{nullptr};
    std::uint32_t staged_kinds{0};  // 1 F16, 2 tiled Q8, 4 BF16
    float *res{}, *xn{}, *blk{}, *h{}, *qkv{}, *z{}, *ab{}, *conv{}, *qn{},
        *kn{}, *raw{}, *gdn{}, *q{}, *gate{}, *k{}, *v{}, *q24{}, *g24{},
        *o24{}, *router{}, *weights{}, *gate_e{}, *shg{}, *shu{}, *sh_out{},
        *gc{}, *uc{}, *dc{}, *emb{}, *cat{}, *mh{};
    __half *half{}, *up_half{}, *down_half{};
    void *bf{}, *bf_lo{}, *q8t{}, *xc{}, *xc_lo{}, *ac{}, *ac_lo{};
    std::int32_t *ids{}, *bounds{}, *cursors{}, *rows_token{}, *rows_slot{},
        *tiles{}, *tiles_bf{}, *tokens{};
    std::uint32_t* counts{};
    // Pinned host mirrors.
    std::uint32_t* counts_host{};
    std::int32_t *tiles_host{}, *tiles_bf_host{}, *tokens_host{};
    hipEvent_t counts_ready{nullptr};
    float* logits{};  // WideLogits rows
    std::uint32_t logits_rows{0};
    std::uint32_t* pos_list{};  // A3B_WIDE_DEBUG group positions
    std::vector<hipEvent_t> events;
    std::vector<double> stamps;
    std::vector<int> marks;
  } wide_;
  bool AllocateSpec(std::string* error);
  /// Top candidates of the first `rows` logits rows into cand_out_*.
  void TopCandidates(std::uint32_t rows);
  bool Run(const std::function<void()>& body, std::uint64_t key,
           std::string* error);
  hipGraphExec_t decode_exec_{nullptr};
  bool decode_warm_{false};
  // Speculative graphs by shape key; a shape runs eagerly once first.
  std::unordered_map<std::uint64_t, hipGraphExec_t> spec_graphs_;
  std::unordered_set<std::uint64_t> spec_warm_;
  void Mark(Phase phase);
  bool Skip(Phase phase) const { return (options_.skip >> phase) & 1u; }
  void Collect();
  std::vector<hipEvent_t> events_;
  std::vector<double> host_ms_;  // profile_sync timestamps
  std::vector<Phase> marks_;
  bool profiling_{false};  // this forward records marks
  double phase_ms_[kPhCount]{};
  double phase_bytes_[kPhCount]{};  // weight bytes read per token
  std::uint64_t profiled_{0};

  Config c_;
  Options options_;
  std::unique_ptr<core::GgufReader> gguf_;
  std::vector<void*> regions_;  // device copies of the GGUF's mapped regions
  std::vector<void*> owned_;    // other device allocations
  Tensor token_embd_, output_norm_, output_;
  std::vector<Layer> layers_;
  MtpLayer mtp_;
  hipStream_t stream_{nullptr};

  // Per-layer state (indexed by layer; unused slots stay null).
  std::vector<float*> conv_state_, ssm_state_;
  // vt_cache_: V transposed in 16-position tiles (see StoreVt), read by the
  // WMMA decode attention.
  std::vector<__half*> k_cache_, v_cache_, vt_cache_;
  // GDN rollback rows per layer (row r = after r + 1 tokens).
  std::vector<std::array<float*, 7>> state_snap_, conv_snap_;

  // Scratch for up to kMaxRows rows.
  float *res_{}, *xn_{}, *blk_{}, *qkv_{}, *z_{}, *ab_{}, *conv_scratch_{},
      *qn_{}, *kn_{}, *raw_{}, *gdn_out_{}, *qg_{}, *q_{}, *gate_{}, *k_{},
      *v_{}, *ctx_{}, *partials_{}, *router_{}, *weights_{}, *sh_act_{},
      *sh_tmp_{}, *sh_out_{}, *e_act_{}, *e_up_{}, *e_down_{}, *logits_{};
  void* xq_{};
  // Fused path: Q8_1 activations, F32 gated activations for BF16 consumers,
  // and the packed [q/gate | k | v] attention projection.
  Q8_1Block* aq_{};
  Q8_1Block* act_q_{};  // routed + shared activations for the down GEMV
  float *xf_{}, *packed_{};
  std::int32_t *ids_{}, *tokens_dev_{};
  std::uint32_t* pos_dev_{};
  // Pinned staging for tokens and position.
  std::int32_t* tokens_host_{};
  std::uint32_t* pos_host_{};
  std::uint32_t pos_{0};

  // Speculative state (AllocateSpec).
  __half *mtp_k_{}, *mtp_v_{}, *mtp_vt_{};
  float *target_h_{}, *mtp_h_{}, *mtp_pending_{}, *mtp_hn_{}, *res_m_{},
      *emb_m_{}, *cat_f_{}, *logits_m_{};
  Q8_1Block* cat_q_{};
  std::int32_t *vtok_{}, *mtp_tok_{};
  void* argmax_scratch_{};
  // Top-candidate selection workspaces; the first 64 ids are followed by
  // their 64 logits, the Candidates layout.
  std::uint32_t *cand_ws_{}, *cand_scratch_{};
  float* cand_out_{};  // [rows][Candidates]
  // Pinned staging: [0] the new token, [1..] positions; and the downloads
  // ([0..7] verify tokens, then the rows' Candidates).
  std::uint32_t *stage_host_{}, *stage_dev_{};
  std::uint8_t* down_host_{};
  enum SpecSlot {
    kSpecCatchUp,
    kSpecChain,
    kSpecVerify,
    kSpecCandidates,
    kSpecCommit,
    kSpecSlots
  };
  void SpecTick(int slot);
  double spec_ms_[kSpecSlots]{};
  std::uint64_t spec_steps_{0};
  std::chrono::steady_clock::time_point spec_mark_{};
  std::uint32_t mtp_rows_{0};     // rows of the next MTP catch-up pass
  bool step_mtp_{false};          // the pending step ran the MTP block
  std::uint32_t draft_depth_{0};  // drafts proposed in the pending step
  // Staging layout (words): [0] x, [1] trunk position, [2] catch-up
  // position, [3] chain position, [kStageChainToken], [kStageVerify..] the
  // verify tokens.
  static constexpr std::uint32_t kStageWords = 32;
  static constexpr std::uint32_t kStageChainToken = 8;
  static constexpr std::uint32_t kStageVerify = 16;
  void* draft_head_{nullptr};
  std::int32_t* draft_map_{nullptr};
  std::uint32_t draft_vocab_{0};
  std::uint8_t* draft_host_{};
  std::uint32_t verify_rows_{0};  // rows of the last SpecStep
  // Prompt checkpoint (checkpoint.cpp).
  std::vector<float*> ckpt_ssm_, ckpt_conv_;
  float* ckpt_pending_{nullptr};
  std::uint32_t ckpt_pos_{0};
  bool ckpt_valid_{false};
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_
