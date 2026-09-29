// A3B host state snapshots (engine.hpp: StateBytes / SaveState / LoadState)
// and logits row access for gufo serve's text runner (text_runner.cpp).
#include <cstring>

#include "src/models/qwen36_a3b/dflash.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

struct StateHeader {
  std::uint32_t magic;
  std::uint32_t pos;
  std::uint32_t mtp_rows;
  std::uint32_t dflash_injected;
};
constexpr std::uint32_t kStateMagic = 0x33423341;  // "A3B3"

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
  return false;
}

}  // namespace

std::size_t Engine::StateBytes() const {
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  const std::size_t state = static_cast<std::size_t>(c_.ssm_v_heads) *
                            c_.ssm_head_dim * c_.ssm_head_dim;
  std::size_t bytes = sizeof(StateHeader);
  for (std::uint32_t i = 0; i < c_.layers; ++i)
    if (layers_[i].linear)
      bytes += (conv + state) * sizeof(float);
  if (mtp_pending_ != nullptr)
    bytes +=
        (1 + kMaxRows) * static_cast<std::size_t>(c_.hidden) * sizeof(float) +
        kMaxRows * sizeof(std::int32_t);
  if (df_)
    bytes += df_->layers.size() * 2 * static_cast<std::size_t>(df_->capacity) *
             df_->kv_heads * df_->head_dim * sizeof(float);
  return bytes;
}

bool Engine::SaveState(std::span<std::uint8_t> out, std::string* error) {
  if (out.size() != StateBytes())
    return Fail(error, "state snapshot size mismatch");
  if (verify_rows_ != 0)
    return Fail(error, "state snapshot inside a speculative step");
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  const std::size_t state = static_cast<std::size_t>(c_.ssm_v_heads) *
                            c_.ssm_head_dim * c_.ssm_head_dim;
  const StateHeader header{kStateMagic, pos_, mtp_rows_,
                           df_ ? df_->injected : 0};
  std::memcpy(out.data(), &header, sizeof(header));
  std::uint8_t* at = out.data() + sizeof(header);
  const auto down = [&](const void* src, std::size_t bytes) {
    const bool ok = hipMemcpyAsync(at, src, bytes, hipMemcpyDeviceToHost,
                                   stream_) == hipSuccess;
    at += bytes;
    return ok;
  };
  bool ok = true;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    ok = ok && down(ssm_state_[i], state * sizeof(float)) &&
         down(conv_state_[i], conv * sizeof(float));
  }
  if (mtp_pending_ != nullptr) {
    const std::size_t H = c_.hidden;
    ok = ok && down(mtp_pending_, H * sizeof(float)) &&
         down(mtp_h_, kMaxRows * H * sizeof(float)) &&
         down(mtp_tok_, kMaxRows * sizeof(std::int32_t));
  }
  if (df_) {
    const std::size_t ring = static_cast<std::size_t>(df_->capacity) *
                             df_->kv_heads * df_->head_dim * sizeof(float);
    for (const auto& layer : df_->layers)
      ok = ok && down(layer.ring_k, ring) && down(layer.ring_v, ring);
  }
  if (!ok)
    return Fail(error, "state snapshot download failed");
  return Sync(error);
}

bool Engine::LoadState(std::span<const std::uint8_t> in, std::string* error) {
  StateHeader header{};
  if (in.size() != StateBytes() || in.size() < sizeof(header))
    return Fail(error, "state snapshot size mismatch");
  std::memcpy(&header, in.data(), sizeof(header));
  if (header.magic != kStateMagic || header.pos > options_.max_context)
    return Fail(error, "invalid state snapshot");
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  const std::size_t state = static_cast<std::size_t>(c_.ssm_v_heads) *
                            c_.ssm_head_dim * c_.ssm_head_dim;
  const std::uint8_t* at = in.data() + sizeof(header);
  const auto up = [&](void* dst, std::size_t bytes) {
    const bool ok = hipMemcpyAsync(dst, at, bytes, hipMemcpyHostToDevice,
                                   stream_) == hipSuccess;
    at += bytes;
    return ok;
  };
  bool ok = true;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    ok = ok && up(ssm_state_[i], state * sizeof(float)) &&
         up(conv_state_[i], conv * sizeof(float));
  }
  if (mtp_pending_ != nullptr) {
    const std::size_t H = c_.hidden;
    ok = ok && up(mtp_pending_, H * sizeof(float)) &&
         up(mtp_h_, kMaxRows * H * sizeof(float)) &&
         up(mtp_tok_, kMaxRows * sizeof(std::int32_t));
  }
  if (df_) {
    const std::size_t ring = static_cast<std::size_t>(df_->capacity) *
                             df_->kv_heads * df_->head_dim * sizeof(float);
    for (const auto& layer : df_->layers)
      ok = ok && up(layer.ring_k, ring) && up(layer.ring_v, ring);
    df_->injected = header.dflash_injected;
  }
  if (!ok)
    return Fail(error, "state snapshot upload failed");
  pos_ = header.pos;
  mtp_rows_ = header.mtp_rows;
  verify_rows_ = 0;
  step_mtp_ = false;
  draft_depth_ = 0;
  // The pageable source must stay alive until the copies ran.
  return Sync(error);
}

bool Engine::RowLogits(std::uint32_t row, float* out, std::string* error) {
  if (row >= kMaxRows)
    return Fail(error, "logits row out of range");
  if (hipMemcpyAsync(out, logits_ + static_cast<std::size_t>(row) * c_.vocab,
                     c_.vocab * sizeof(float), hipMemcpyDeviceToHost,
                     stream_) != hipSuccess)
    return Fail(error, "logits download failed");
  return Sync(error);
}

bool Engine::SetRowLogits(const float* in, std::string* error) {
  if (hipMemcpyAsync(logits_, in, c_.vocab * sizeof(float),
                     hipMemcpyHostToDevice, stream_) != hipSuccess)
    return Fail(error, "logits upload failed");
  return Sync(error);
}

}  // namespace gufo::models::qwen36_a3b
