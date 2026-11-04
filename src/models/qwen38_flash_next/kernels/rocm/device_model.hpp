#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

#ifdef GUFO_NPU
namespace gufo::npu {
struct LinearZWeights;
struct ProjectionWeights;
class ServerPrefill;
}  // namespace gufo::npu
#endif

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  std::string name;
  bool decode_twin{false};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceMixer {
  DeviceTensor norm;
  DeviceTensor down;
  DeviceTensor up;
  DeviceTensor inject;
};

struct DeviceLayer {
  bool linear{false};
  DeviceMixer hc_attn;
  DeviceMixer hc_ffn;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
#ifdef GUFO_NPU
  std::shared_ptr<const npu::LinearZWeights> npu_z;
  std::shared_ptr<const npu::ProjectionWeights> npu_gate, npu_shared;
#endif
  /// Alpha and beta rows stacked without changing native weight encoding.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1], retaining the native router encoding.
  DeviceTensor router;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

/// The trunk (and optionally the MTP draft block) uploaded to the GPU. The
/// n-gram table is never uploaded: it is read from disk per token.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr
#ifdef GUFO_NPU
      ,
      std::shared_ptr<npu::ServerPrefill> prefill = {}
#endif
  );

#ifdef GUFO_NPU
  const std::shared_ptr<npu::ServerPrefill>& npu_prefill() const noexcept {
    return npu_prefill_;
  }
#endif
  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  /// The leading output() rows the MTP draft head scores in place (see
  /// draft_vocab.hpp); empty when the draft projects the full vocabulary.
  const DeviceTensor& draft_output() const noexcept { return draft_output_; }
  /// Special tokens above the prefix: the first rows of each session's
  /// draft table.
  std::span<const std::int32_t> draft_specials() const noexcept {
    return draft_specials_;
  }
  /// Bytes of one output() row.
  std::size_t output_row_bytes() const noexcept {
    return output_.rows == 0 ? 0 : output_bytes_ / output_.rows;
  }
  /// Drafts score output() rows [0, prefix_rows) in place plus a
  /// per-session table that starts with `specials`. Draft proposals only
  /// steer speculation: the target still verifies every token, so greedy
  /// text is unchanged. Sampled text can differ: a smaller head changes
  /// acceptance, and so the later random draws.
  [[nodiscard]] bool SetDraftVocab(std::uint32_t prefix_rows,
                                   std::vector<std::int32_t> specials,
                                   std::string* error_msg);
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const DeviceMixer& twin_hc_head() const noexcept { return twin_hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// GUFO_HC_LOAD_Q8 narrowed every BF16 HC projection to Q8_0 at upload
  /// (and widened BF16 inject weights to F32); see hc_load_q8_policy.hpp.
  [[nodiscard]] bool hc_load_q8_active() const noexcept {
    return hc_load_q8_active_;
  }
  /// GUFO_HC_LOAD_Q8_VERIFY (debug): reads the trunk's HC tensors, the token
  /// embedding and output head and EVERY tensor of the MTP block back from the
  /// device and compares them byte for byte (zero tail included) with the
  /// file; converted tensors are compared with a fresh serial conversion.
  /// Prints one "[hcq8-verify] ok|MISMATCH phase=<phase> ..." line and returns
  /// false (with `error_msg`) on a mismatch or a read-back failure.
  [[nodiscard]] bool VerifyContent(const ModelWeights& weights,
                                   const MtpWeights* mtp, const char* phase,
                                   std::string* error_msg) const;
  /// Decode twin (see TwinWeights): uploads the 4-bit dense copy next to the trunk and builds twin_layers(), copies of layers() whose dense projections point at
  /// the 4-bit tensors (every other tensor shares the trunk layer's device memory). Any failure is an error; there is no fallback.
  [[nodiscard]] bool UploadTwin(const TwinWeights& twin,
                                const core::GgufReader& twin_reader,
                                std::string* error_msg = nullptr);
  [[nodiscard]] bool has_twin() const noexcept { return !twin_layers_.empty(); }
  /// Draft-only 4-bit copy of the output head (see BindDraftOutput); the trunk head (output()) is untouched.
  [[nodiscard]] bool UploadDraftOut(const TensorRef& ref, const core::GgufReader& reader, std::string* error_msg);
  [[nodiscard]] bool has_draft_out() const noexcept { return !draft_out_.empty(); }
  const DeviceTensor& draft_out() const noexcept { return draft_out_; }
  /// DEBUG (GUFO_DENSE_TWIN_SHADOW=<fake-quant twin file>, needs GUFO_DENSE_TWIN_ALLOW_Q8=1): uploads a second, Q8_0 twin whose values are the dequantised 4-bit values; the executor then runs every 4-bit dense
  /// projection a second time through the Q8_0 path on the SAME activations and accumulates the per-tensor difference (real-activation kernel check).
  [[nodiscard]] bool UploadShadow(const TwinWeights& shadow, const core::GgufReader& shadow_reader, std::string* error_msg);
  [[nodiscard]] bool has_shadow() const noexcept { return !shadow_map_.empty(); }
  struct ShadowEntry {
    const DeviceTensor* shadow{nullptr};
    std::string label;
  };
  [[nodiscard]] const ShadowEntry* ShadowFor(const void* twin_data) const {
    const auto it = shadow_map_.find(twin_data);
    return it == shadow_map_.end() ? nullptr : &it->second;
  }
  [[nodiscard]] bool UploadTwinImpl(const TwinWeights& twin, const core::GgufReader& twin_reader, std::string* error_msg, std::vector<DeviceLayer>& out_layers, bool primary);
  const std::vector<DeviceLayer>& twin_layers() const noexcept { return twin_layers_; }
  [[nodiscard]] std::size_t twin_bytes() const noexcept { return twin_bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }
  /// Widest K and most rows among the tensors NativeWmmaGemm routes (the
  /// zero-padded 96-row input and output copies of a short prefill pass).
  [[nodiscard]] std::size_t max_native_gemm_cols() const noexcept {
    return max_native_gemm_cols_;
  }
  [[nodiscard]] std::size_t max_native_gemm_rows() const noexcept {
    return max_native_gemm_rows_;
  }

private:
  DeviceModel() = default;

#ifdef GUFO_NPU
  std::shared_ptr<npu::ServerPrefill> npu_prefill_;
#endif
  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  std::size_t output_bytes_{0};
  DeviceTensor draft_output_;
  std::vector<std::int32_t> draft_specials_;
  DeviceMixer hc_head_;
  DeviceMixer twin_hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  bool hc_load_q8_active_{false};
  std::vector<DeviceLayer> twin_layers_;
  DeviceTensor draft_out_;
  std::vector<DeviceLayer> shadow_layers_;
  std::unordered_map<const void*, ShadowEntry> shadow_map_;
  std::size_t twin_bytes_{0};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
  std::size_t max_native_gemm_cols_{0};
  std::size_t max_native_gemm_rows_{0};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
