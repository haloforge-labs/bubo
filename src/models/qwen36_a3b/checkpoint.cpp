// A3B prompt checkpoint: the recurrent state after a prompt prefix, so a
// follow-up request that extends the prefix prefills only its new tokens.
// The KV caches need no copy: positions below the checkpoint are never
// rewritten by later tokens. The DFlash draft's context is a ring, which
// later tokens do overwrite: it is copied with the checkpoint.
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

namespace fn = gufo::models::qwen38_flash_next::rocm;

}  // namespace

bool Engine::SaveCheckpoint(std::string* error) {
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  const std::size_t state = static_cast<std::size_t>(c_.ssm_v_heads) *
                            c_.ssm_head_dim * c_.ssm_head_dim;
  if (verify_rows_ != 0) {
    if (error != nullptr)
      *error = "checkpoint inside a speculative step";
    return false;
  }
  if (ckpt_ssm_.empty()) {
    ckpt_ssm_.assign(c_.layers, nullptr);
    ckpt_conv_.assign(c_.layers, nullptr);
    for (std::uint32_t i = 0; i < c_.layers; ++i) {
      if (!layers_[i].linear)
        continue;
      void* a = nullptr;
      void* b = nullptr;
      if (hipMalloc(&a, state * sizeof(float)) != hipSuccess ||
          hipMalloc(&b, conv * sizeof(float)) != hipSuccess) {
        if (error != nullptr)
          *error = "checkpoint alloc failed";
        return false;
      }
      owned_.push_back(a);
      owned_.push_back(b);
      ckpt_ssm_[i] = static_cast<float*>(a);
      ckpt_conv_[i] = static_cast<float*>(b);
    }
    void* p = nullptr;
    if (hipMalloc(&p, static_cast<std::size_t>(c_.hidden) * sizeof(float)) !=
        hipSuccess) {
      if (error != nullptr)
        *error = "checkpoint alloc failed";
      return false;
    }
    owned_.push_back(p);
    ckpt_pending_ = static_cast<float*>(p);
  }
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    fn::CopyDevice(ssm_state_[i], ckpt_ssm_[i], state, stream_);
    fn::CopyDevice(conv_state_[i], ckpt_conv_[i], conv, stream_);
  }
  if (mtp_pending_ != nullptr)
    fn::CopyDevice(mtp_pending_, ckpt_pending_, c_.hidden, stream_);
  if (HasDFlash() && !DFlashSaveCheckpoint(error))
    return false;
  ckpt_pos_ = pos_;
  ckpt_valid_ = true;
  return Sync(error);
}

bool Engine::RestoreCheckpoint(std::string* error) {
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  const std::size_t state = static_cast<std::size_t>(c_.ssm_v_heads) *
                            c_.ssm_head_dim * c_.ssm_head_dim;
  if (!ckpt_valid_) {
    if (error != nullptr)
      *error = "no prompt checkpoint";
    return false;
  }
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    fn::CopyDevice(ckpt_ssm_[i], ssm_state_[i], state, stream_);
    fn::CopyDevice(ckpt_conv_[i], conv_state_[i], conv, stream_);
  }
  if (mtp_pending_ != nullptr)
    fn::CopyDevice(ckpt_pending_, mtp_pending_, c_.hidden, stream_);
  if (HasDFlash() && !DFlashRestoreCheckpoint(error))
    return false;
  pos_ = ckpt_pos_;
  verify_rows_ = 0;
  step_mtp_ = false;
  mtp_rows_ = 0;  // the next SpecPrefill re-seeds the MTP catch-up
  return Sync(error);
}

}  // namespace gufo::models::qwen36_a3b
