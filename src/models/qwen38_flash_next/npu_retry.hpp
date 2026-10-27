#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NPU_RETRY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NPU_RETRY_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Session state a chunk adds to instead of overwriting, copied before an NPU
/// prefill chunk so the chunk can be run again on the GPU (L-REL-2): per linear
/// layer the conv state and the recurrent state, then the PLE conv history.
/// K/V rows, the indexer ring, pooled blocks and logits are rewritten by the
/// rerun and are not part of it.
struct NpuRetryLayout {
  std::size_t linear_layers{0};
  std::size_t conv_floats{0};
  std::size_t state_floats{0};
  std::size_t ple_floats{0};
  [[nodiscard]] std::size_t Floats() const noexcept {
    return linear_layers * (conv_floats + state_floats) + ple_floats;
  }
};

[[nodiscard]] inline NpuRetryLayout NpuRetryLayoutFor(
    const Config& c) noexcept {
  NpuRetryLayout layout;
  for (std::uint32_t il = 0; il < c.num_layers; ++il)
    if (c.IsLinearLayer(il))
      ++layout.linear_layers;
  layout.conv_floats = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  layout.state_floats =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  if (c.ple_layer >= 0)
    layout.ple_floats = std::size_t{c.PleConvHistory()} * c.HcDim();
  return layout;
}

/// One piece of the backup, in the order of the buffer: the conv state and the
/// recurrent state of each linear layer, then the PLE history. offset and count
/// are in floats from the start of the buffer.
enum class NpuRetryKind { kConv, kState, kPle };
struct NpuRetryRegion {
  NpuRetryKind kind{NpuRetryKind::kConv};
  std::uint32_t layer{0};  ///< model layer (unused for kPle)
  std::size_t offset{0};
  std::size_t count{0};
  /// A 16-byte copy needs the buffer offset on a 16-byte boundary (the session
  /// buffers come from hipMalloc and are aligned).
  [[nodiscard]] bool Aligned16() const noexcept { return offset % 4 == 0; }
};
[[nodiscard]] inline std::vector<NpuRetryRegion> NpuRetryRegions(
    const Config& c) {
  const auto layout = NpuRetryLayoutFor(c);
  std::vector<NpuRetryRegion> regions;
  std::size_t offset = 0;
  const auto add = [&](NpuRetryKind kind, std::uint32_t layer,
                       std::size_t count) {
    regions.push_back({kind, layer, offset, count});
    offset += count;
  };
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!c.IsLinearLayer(il))
      continue;
    add(NpuRetryKind::kConv, il, layout.conv_floats);
    add(NpuRetryKind::kState, il, layout.state_floats);
  }
  if (layout.ple_floats != 0)
    add(NpuRetryKind::kPle, 0, layout.ple_floats);
  return regions;
}
/// The regions go to the device in launches of at most this many (the table is
/// a kernel argument; kCopyRegionsPerLaunch in kernels.hpp is the same number).
inline constexpr std::size_t kNpuRetryRegionsPerLaunch = 64;
inline constexpr std::size_t kNpuRetryMaxLaunches = 4;
/// Launches for `regions` regions.
[[nodiscard]] inline std::size_t NpuRetryLaunches(
    std::size_t regions) noexcept {
  return (regions + kNpuRetryRegionsPerLaunch - 1) / kNpuRetryRegionsPerLaunch;
}
/// Whether that is few enough; if not, the backup is unavailable.
[[nodiscard]] inline bool NpuRetryFits(std::size_t regions) noexcept {
  return NpuRetryLaunches(regions) <= kNpuRetryMaxLaunches;
}

/// Device bytes of the retry backup buffer; 0 when NPU prefill is disabled
/// (the buffer is then never allocated).
[[nodiscard]] inline std::size_t NpuRetryBackupBytes(
    const Config& c, bool npu_prefill_enabled) noexcept {
  return npu_prefill_enabled ? NpuRetryLayoutFor(c).Floats() * sizeof(float)
                             : 0;
}

/// How the backup buffer and its two timing events reach the device. The
/// executor passes the HIP calls; a test passes fakes. clear_error drops the
/// device runtime's sticky last error, which a failed allocation leaves behind
/// and the next unrelated check would otherwise report.
struct NpuRetryBackupOps {
  std::function<void*(std::size_t bytes)> alloc;  ///< null on failure
  std::function<void(void*)> free;
  std::function<void*()> create_event;  ///< null on failure
  std::function<void(void*)> destroy_event;
  std::function<void()> clear_error;
};

/// The buffer and events of the retry backup. The first failure to set them up
/// is remembered: retry stays off for this owner and nothing is allocated
/// again. Release gives everything back (the NPU is off for the process).
class NpuRetryBackupResources {
public:
  NpuRetryBackupResources() = default;
  explicit NpuRetryBackupResources(NpuRetryBackupOps ops)
      : ops_(std::move(ops)) {}
  NpuRetryBackupResources(const NpuRetryBackupResources&) = delete;
  NpuRetryBackupResources& operator=(const NpuRetryBackupResources&) = delete;
  NpuRetryBackupResources& operator=(NpuRetryBackupResources&& other) noexcept {
    if (this != &other) {
      Release();
      ops_ = std::move(other.ops_);
      buffer_ = std::exchange(other.buffer_, nullptr);
      begin_ = std::exchange(other.begin_, nullptr);
      end_ = std::exchange(other.end_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
      failed_ = other.failed_;
    }
    return *this;
  }
  ~NpuRetryBackupResources() { Release(); }

  /// True when the buffer and events exist (allocated now if needed).
  [[nodiscard]] bool Ensure(std::size_t bytes) {
    if (buffer_ != nullptr)
      return true;
    if (failed_ || !ops_.alloc)
      return false;
    begin_ = ops_.create_event();
    end_ = begin_ ? ops_.create_event() : nullptr;
    buffer_ = end_ ? ops_.alloc(bytes) : nullptr;
    if (buffer_ == nullptr) {
      Release();
      ops_.clear_error();
      failed_ = true;
      return false;
    }
    bytes_ = bytes;
    return true;
  }
  void Release() noexcept {
    if (buffer_ != nullptr)
      ops_.free(buffer_);
    if (begin_ != nullptr)
      ops_.destroy_event(begin_);
    if (end_ != nullptr)
      ops_.destroy_event(end_);
    buffer_ = begin_ = end_ = nullptr;
    bytes_ = 0;
  }
  [[nodiscard]] bool failed() const noexcept { return failed_; }
  [[nodiscard]] bool allocated() const noexcept { return buffer_ != nullptr; }
  [[nodiscard]] void* buffer() const noexcept { return buffer_; }
  [[nodiscard]] void* begin_event() const noexcept { return begin_; }
  [[nodiscard]] void* end_event() const noexcept { return end_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  NpuRetryBackupOps ops_;
  void* buffer_{nullptr};
  void* begin_{nullptr};
  void* end_{nullptr};
  std::size_t bytes_{0};
  bool failed_{false};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_NPU_RETRY_HPP_
