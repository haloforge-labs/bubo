#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};

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
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16.
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
      std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
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

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<DeviceLayer> twin_layers_;
  DeviceTensor draft_out_;
  std::vector<DeviceLayer> shadow_layers_;
  std::unordered_map<const void*, ShadowEntry> shadow_map_;
  std::size_t twin_bytes_{0};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
