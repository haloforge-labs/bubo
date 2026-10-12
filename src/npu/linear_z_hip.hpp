// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_LINEAR_Z_HIP_HPP_
#define GUFO_NPU_LINEAR_Z_HIP_HPP_
#include <hip/hip_runtime.h>

#include "linear_z.hpp"
#include "server_prefill.hpp"

namespace gufo::npu {
// Stream-ordered activation and result transfer bridge.
// Mapped pages live with the service; the stream orders producer, worker wait,
// decode and consumers. The host checks completion once per chunk.
class HipLinearZTransfer final : public LinearZTransfer {
public:
  HipLinearZTransfer(std::shared_ptr<ServerPrefill> service,
                     const std::uint16_t* input, float* output, unsigned tokens,
                     hipStream_t stream, unsigned pitch = 16384,
                     unsigned offset = 0xffffffffU, float* second = nullptr);
  ~HipLinearZTransfer() override;
  std::vector<std::vector<std::uint8_t>> FenceInput() override;
  BorrowedActivations BorrowInput() override;
  PackedConsumer PackedOutputConsumer() override;
  bool WaitOnStream(std::shared_future<LinearZResult> result,
                    std::chrono::milliseconds deadline) override;
  void FencePackedOutput(unsigned columns) override;
  void FenceOutput(std::span<const float> tail, unsigned columns) override;

private:
  void Check(hipError_t status);
  std::shared_ptr<ServerPrefill> service_;
  const std::uint16_t* input_;
  float* output_;
  unsigned tokens_, pitch_, offset_;
  float* second_;
  hipStream_t stream_;
  std::uint8_t *a_ = nullptr, *host_a_ = nullptr;
  unsigned* invalid_ = nullptr;
  std::shared_ptr<HipLinearZStorage> storage_;
};
// Call only after the executor's existing forward wait; no HIP calls here.
bool CheckHipPrefillChunk(ServerPrefill& service, std::string* error);
}  // namespace gufo::npu
#endif
