#ifndef GUFO_MODELS_QWEN36_A3B_DFLASH_HPP_
#define GUFO_MODELS_QWEN36_A3B_DFLASH_HPP_

#include <cstdint>
#include <vector>

#include "src/models/qwen36_a3b/engine.hpp"

/// DFlash-2 block-diffusion drafter for the A3B engine (dflash.cpp), the
/// architecture of gufo's 27B DFlash-2 path (src/models/qwen/hip/dflash):
/// 6 draft layers with dynamic grouped convolutions around attention and the
/// FFN, attention over a sliding window of injected target context plus the
/// non-causal block [anchor, mask x k], the target's embedding and LM head,
/// and the low-rank selector that picks each draft given its predecessor.
///
/// Context: every committed target position contributes the outputs of 8
/// trunk layers (dflash.target_layers, taken as layer-input indices, so
/// layer N - 1's output), concatenated, projected by fc, normed, then each
/// draft layer's K (normed, rotated) and V go to a ring of `capacity` slots.
///
/// Draft matrices are converted to Q8_0 at load (the file is F16) and run
/// through the A3B decode GEMV; the selector codebooks become BF16 (the 27B
/// selector kernel's format). The drafts only propose tokens: verification
/// uses the target alone, so the output does not depend on them.
namespace gufo::models::qwen36_a3b {

struct DFlashLayer {
  Tensor q, k, v, o, gate, up, down, attn_conv_proj, ffn_conv_proj;
  const float *attn_norm{}, *ffn_norm{}, *q_norm{}, *k_norm{},
      *attn_conv_base{}, *ffn_conv_base{};
  float *ring_k{}, *ring_v{};  // [capacity][kv_heads * head_dim] F32
};

struct DFlash {
  ~DFlash();

  // Topology (from the draft GGUF).
  std::uint32_t hidden{0}, heads{0}, kv_heads{0}, head_dim{0}, rotary{0},
      ff{0}, block_size{0}, mask_token{0}, conv_k{0}, conv_g{0},
      sel_rank{0}, sel_topk{0}, window{0}, capacity{0};
  float theta{0}, eps{0};
  std::vector<std::uint32_t> taps;  // trunk layer outputs, zero-based
  std::vector<int> tap_slot;        // by trunk layer: feature slot or -1

  std::vector<DFlashLayer> layers;
  Tensor fc, sel_hidden;
  const float *enc_norm{}, *out_norm{};
  const void *pred{}, *succ{};  // [vocab][sel_rank] BF16
  // With the engine's draft vocab (--draft-vocab): the codebooks gathered to
  // its V' tokens; pred_p has one more row, the current anchor's.
  void *pred_p{}, *succ_p{};

  // Context features: decode / verify rows, and one wide prefill chunk.
  float *feat{}, *wide_feat{};
  std::uint32_t injected{0};  // positions [0, injected) are in the ring
  // Prompt checkpoint copies of the rings (DFlashSaveCheckpoint).
  std::vector<float*> ckpt_k, ckpt_v;
  std::uint32_t ckpt_injected{0};

  // Block scratch (up to Engine::kMaxRows rows).
  float *h{}, *xn{}, *dyn{}, *conv{}, *q{}, *k{}, *v{}, *att{}, *o{}, *f{},
      *sel{}, *logits{}, *conf{}, *ones{}, *partial_scores{};
  std::uint32_t *partial_ids{}, *tok{}, *pos{}, *pos_list{};
  Q8_1Block* aq{};
  std::uint32_t* host{};  // pinned: [0, 16) tokens, [16] position
  std::uint32_t* pos_host{};  // pinned injection group positions
  std::uint32_t max_groups{0};

  std::vector<void*> owned;  // device allocations
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_DFLASH_HPP_
