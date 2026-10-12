// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_LINEAR_Z_HPP_
#define GUFO_NPU_LINEAR_Z_HPP_
#include <functional>
#include <future>
#include <memory>

#include "bfp.hpp"
#include "staging.hpp"

namespace gufo::npu {
// Immutable, independent CPU B streams. No GGUF/HIP/XRT alias survives staging.
struct LinearZWeights {
  unsigned gpu_rows = 16384;
  std::vector<std::vector<std::uint8_t>> b;
};
std::shared_ptr<const LinearZWeights> PrepareLinearZ(const Q8Matrix& qkv,
                                                     const Q8Matrix& gate,
                                                     unsigned slices);
bool LinearZQualified(unsigned tokens, unsigned capacity, bool prefill,
                      bool speculative, bool projections_ready,
                      const LinearZWeights* weights) noexcept;

struct BorrowedActivations {
  std::vector<std::span<const std::uint8_t>> groups;
  std::shared_ptr<void> lifetime;
  std::function<void()> ready;
  // HIP transfers defer the prepared command until the stream produced A.
  std::function<void(std::function<void()>)> submit;
};
struct LayerInputs {
  std::vector<std::vector<std::uint8_t>> owned;
  BorrowedActivations borrowed;
  std::size_t size() const {
    return borrowed.groups.empty() ? owned.size() : borrowed.groups.size();
  }
  std::span<const std::uint8_t> operator[](std::size_t g) const {
    return borrowed.groups.empty() ? std::span<const std::uint8_t>(owned[g])
                                   : borrowed.groups[g];
  }
};
using LinearDispatch = std::function<NpuResult(
    std::span<const std::byte>, std::span<const std::byte>, DispatchInfo)>;
using BatchDispatch = std::function<bool(
    std::span<const std::byte>, const std::vector<std::vector<std::uint8_t>>&,
    const std::function<void(std::size_t, std::span<const std::byte>)>&)>;
using PackedConsumer =
    std::function<void(std::size_t, std::size_t, std::span<const std::byte>)>;
struct LinearZResult {
  bool valid = false;
  bool dma_uncertain = false;
  std::string error;
  // [tokens][tail columns], z plus optional trailing QKV for slices 7..12.
  std::vector<float> tail;
  bool packed = false;
};
// The persistent worker owns inputs, weights and the dispatch capability.
class LinearZWork {
public:
  LinearZWork(std::shared_ptr<const LinearZWeights> weights,
              std::vector<std::vector<std::uint8_t>> a, int layer,
              LinearDispatch dispatch,
              std::function<void(std::string)> reject = {},
              BatchDispatch batch = {}, PackedConsumer packed = {},
              BorrowedActivations borrowed = {});
  LinearZResult Finish(std::chrono::milliseconds deadline =
                           std::chrono::milliseconds(30000)) const;
  const std::shared_future<LinearZResult>& Result() const { return result_; }

private:
  std::shared_future<LinearZResult> result_;
};

// The same orchestration is exercised by CPU mocks and the HIP implementation.
// CPU transfers complete synchronously. HIP transfers enqueue a stream wait
// followed by decode; completion and invalid results are checked per chunk.
class LinearZTransfer {
public:
  virtual ~LinearZTransfer() = default;
  virtual std::vector<std::vector<std::uint8_t>> FenceInput() = 0;
  virtual BorrowedActivations BorrowInput() { return {}; }
  virtual PackedConsumer PackedOutputConsumer() { return {}; }
  virtual bool WaitOnStream(std::shared_future<LinearZResult>,
                            std::chrono::milliseconds) {
    return false;
  }
  virtual void FencePackedOutput(unsigned) {
    throw std::logic_error("packed output unsupported");
  }
  virtual void FenceOutput(std::span<const float> tail, unsigned columns) = 0;
};
std::unique_ptr<LinearZWork> BeginLinearZ(
    std::shared_ptr<const LinearZWeights> weights, LinearZTransfer& transfer,
    int layer, LinearDispatch dispatch,
    std::function<void(std::string)> reject = {}, BatchDispatch batch = {});
// HIP: enqueue the wait and decode before the first consumer, without waiting
// on the CPU. A callback failure rejects the request at chunk completion.
LinearZResult FinishLinearZ(
    const LinearZWork& work, LinearZTransfer& transfer, unsigned columns,
    std::chrono::milliseconds deadline = std::chrono::milliseconds(30000));
}  // namespace gufo::npu
#endif
