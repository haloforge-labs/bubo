#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Optional host destinations for independent predictor qualification.
/// Every populated span contains the final complete row of its stage.
struct MtpTrace {
  std::span<float> normalized_hidden;
  std::span<float> fused;
  std::span<float> attention;
  std::span<float> hidden;
  std::span<float> head;
  std::span<float> ffn_input;
  std::span<float> ffn_output;
  [[nodiscard]] bool Valid(std::size_t H, std::size_t D) const noexcept {
    for (const auto row : {normalized_hidden, fused, attention, hidden})
      if (!row.empty() && row.size() != D)
        return false;
    for (const auto row : {head, ffn_input, ffn_output})
      if (!row.empty() && row.size() != H)
        return false;
    return true;
  }
};

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Compact routed stack: router row e addresses stack entry e in all three
/// projections. Allocation-file original expert indices are provenance only.
struct RoutedWeights {
  TensorRef router, gate, up, down;
  [[nodiscard]] static std::optional<RoutedWeights> Bind(
      const core::GgufReader& reader, const Config& config, std::uint32_t layer,
      std::string* error_msg = nullptr);
};

/// Two hyper-connection mixers per layer plus the output head mixer share
/// this shape; `inject` is absent on the head mixer.
struct HcMixer {
  TensorRef norm;    ///< [hc_dim] grouped RMSNorm gamma, (1 + w) folded.
  TensorRef down;    ///< [hc_dim -> low_rank]
  TensorRef up;      ///< [low_rank -> hc_dim]
  TensorRef inject;  ///< [hc_dim -> hc_count], empty on the head mixer.
};

struct LayerWeights {
  bool linear{false};
  HcMixer hc_attn;
  HcMixer hc_ffn;

  // Gated DeltaNet (linear layers).
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, channels]
  TensorRef ssm_alpha;   ///< [hidden -> v_heads]
  TensorRef ssm_beta;    ///< [hidden -> v_heads]
  TensorRef ssm_dt;      ///< [v_heads] softplus bias
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log)
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim -> hidden]

  // Gated GQA + QSA indexer (attention layers).
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim], q|gate per head
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;     ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]
  TensorRef indexer_q;    ///< [hidden -> indexer_heads * indexer_dim]
  TensorRef indexer_k;    ///< [hidden -> indexer_dim]
  TensorRef indexer_q_norm;
  TensorRef indexer_k_norm;

  // PLE n-gram injection (one linear layer).
  TensorRef ple_key;         ///< [ple_dim -> hc_dim]
  TensorRef ple_value;       ///< [ple_dim -> hidden]
  TensorRef ple_norm_key;    ///< [hc_dim]
  TensorRef ple_norm_query;  ///< [hc_dim]
  TensorRef ple_norm_conv;   ///< [hc_dim]
  TensorRef ple_conv1d;      ///< [conv_kernel, hc_dim]

  // Mixture of experts.
  TensorRef router;          ///< [hidden -> num_experts] F32
  TensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  TensorRef shexp_gate;      ///< [hidden -> shared_ff]
  TensorRef shexp_up;        ///< [hidden -> shared_ff]
  TensorRef shexp_down;      ///< [shared_ff -> hidden]

  // Speculative `nextn` block only.
  TensorRef nextn_enorm;  ///< [hidden]
  TensorRef nextn_hnorm;  ///< [hc_dim], one RMS denominator across all streams
  TensorRef nextn_eh_proj;  ///< GGUF rows: [fc_embedding | fc_hidden]
  HcMixer nextn_head;       ///< Output mixer of the draft block.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;  ///< [hidden -> vocab]
  TensorRef output;      ///< [hidden -> vocab]
  HcMixer hc_head;       ///< Final mixer, doubles as the output norm.
  TensorRef ple_table;   ///< [ple_head_dim, ple_rows], read from disk only.
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Dense replacements for decode and verification passes of at most eight rows.
/// Shapes match the corresponding trunk tensor. Native quantized and floating
/// trunk tensors are supported; prefill continues to use the trunk. Supported
/// replacements are Q8_0, Q6_K, Q5_K, Q5_1, Q4_K, IQ4_XS, IQ4_NL, IQ3_S,
/// BF16, F16 and F32. No tensor is converted during twin upload.
///   linear: attn_qkv, attn_gate (both or neither), ssm_out
///   attention: attn_q, attn_k, attn_v (all or none), attn_output
///   all layers: hc_attn_down/up, hc_ffn_down/up, ffn_gate/up/down_shexp
/// Output HC down/up are optional. Norms, MTP, embeddings, routers, PLE,
/// experts and the target output projection retain their existing paths.
/// Missing Q8_0 block tensors require a legacy gufo.dense_twin.keep_q8 entry.
/// Listed names must be absent from the twin and retain their native trunk
/// encoding. Missing non-Q8 block tensors also retain their trunk encoding.
struct TwinWeights {
  std::vector<LayerWeights> layers;  ///< only the twinned TensorRefs are populated
  HcMixer hc_head;  ///< optional output HC projections; norm stays in the trunk
  std::size_t tensors{0};
  std::size_t bytes_4bit{0};  ///< encoded bytes of the twin tensors
  std::size_t bytes_8bit{0};  ///< encoded bytes of the trunk tensors they mirror
  std::size_t kept_tensors{
      0};  ///< native trunk tensors explicitly retained by the legacy keep list
  std::size_t kept_bytes{0};

  [[nodiscard]] static std::optional<TwinWeights> Bind(
      const core::GgufReader& reader, const ModelWeights& trunk,
      std::string* error_msg = nullptr);
};

/// DRAFT OUTPUT HEAD (GUFO_DRAFT_OUT_MODEL, default OFF, needs MTP): a small GGUF holding ONLY `output.weight` (hidden -> vocab) as Q4_K or Q5_K. The MTP draft steps score their candidates with it
/// instead of the trunk's Q8_0 output head (a draft step reads the whole vocabulary projection, 675 MB at Q8_0); every verify / trunk logits pass keeps the Q8_0 head, so under exact acceptance the text is
/// unchanged and only acceptance / speed can move. The trunk file is untouched; any problem is fatal (no fallback to the 8-bit head).
[[nodiscard]] std::optional<TensorRef> BindDraftOutput(const core::GgufReader& reader, const TensorRef& trunk_output, std::string* error_msg = nullptr);

/// The MTP draft block from the `shared-*` sidecar: one attention layer with
/// its own hyper-connection mixers and experts; token embedding and LM head
/// are borrowed from the trunk.
struct MtpWeights {
  Config config;
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
