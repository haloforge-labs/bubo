// SPDX-License-Identifier: MIT
#include "linear_z_hip.hpp"

#include <algorithm>
#include <cstring>

#include "bfp_hip.hpp"
#include "chunk_retry.hpp"
#include "fault_telemetry.hpp"
#include "hal_fault_monitor.hpp"
#include "layer_worker.hpp"
#include "projection_geometry.hpp"
#include "request_stats.hpp"
#include "src/cli/serve/logging.hpp"

namespace gufo::npu {
struct HipLinearZStorage {
  std::uint8_t* host_a = nullptr;
  std::uint8_t* device_a = nullptr;
  unsigned* host_invalid = nullptr;
  unsigned* device_invalid = nullptr;
  std::vector<std::uint8_t*> host_c, device_c;
  std::vector<std::uint8_t*> extra_a, extra_a_device;
  ~HipLinearZStorage() {
    for (auto* p : extra_a)
      (void)hipHostFree(p);
    for (auto* p : host_c)
      (void)hipHostFree(p);
    if (host_invalid)
      (void)hipHostFree(host_invalid);
    if (host_a)
      (void)hipHostFree(host_a);
  }
};
HipLinearZTransfer::HipLinearZTransfer(std::shared_ptr<ServerPrefill> service,
                                       const std::uint16_t* input,
                                       float* output, unsigned tokens,
                                       hipStream_t stream, unsigned pitch,
                                       unsigned offset, float* second)
    : service_(std::move(service)),
      input_(input),
      output_(output),
      tokens_(tokens),
      pitch_(pitch),
      offset_(offset),
      second_(second),
      stream_(stream) {
  if (!service_ || !input || !output || !QualifiedPrefill(tokens, 4) ||
      !service_->lane_ready())
    throw std::invalid_argument("NPU prefill HIP bridge not qualified/ready");
  auto& shared = service_->TransferStorage();
  if (!shared) {
    auto storage = std::make_shared<HipLinearZStorage>();
    Check(hipHostMalloc(reinterpret_cast<void**>(&storage->host_a), bfp::ABytes,
                        hipHostMallocMapped | hipHostMallocNonCoherent));
    Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&storage->device_a),
                                  storage->host_a, 0));
    Check(hipHostMalloc(reinterpret_cast<void**>(&storage->host_invalid),
                        sizeof(unsigned), hipHostMallocMapped));
    Check(hipHostGetDevicePointer(
        reinterpret_cast<void**>(&storage->device_invalid),
        storage->host_invalid, 0));
    shared = std::move(storage);
  }
  storage_ = shared;
  while (storage_->extra_a.size() + 1 < tokens_ / kTokens) {
    std::uint8_t* host = nullptr;
    void* device = nullptr;
    Check(hipHostMalloc(reinterpret_cast<void**>(&host), bfp::ABytes,
                        hipHostMallocMapped | hipHostMallocNonCoherent));
    const auto status = hipHostGetDevicePointer(&device, host, 0);
    if (status != hipSuccess) {
      (void)hipHostFree(host);
      Check(status);
    }
    storage_->extra_a.push_back(host);
    storage_->extra_a_device.push_back(static_cast<std::uint8_t*>(device));
  }

  while (storage_->host_c.size() < std::size_t{tokens_ / kTokens} * 12) {
    std::uint8_t* host = nullptr;
    void* device = nullptr;
    Check(hipHostMalloc(reinterpret_cast<void**>(&host), bfp::CBytes,
                        hipHostMallocMapped | hipHostMallocNonCoherent));
    const auto status = hipHostGetDevicePointer(&device, host, 0);
    if (status != hipSuccess) {
      (void)hipHostFree(host);
      Check(status);
    }
    storage_->host_c.push_back(host);
    storage_->device_c.push_back(static_cast<std::uint8_t*>(device));
  }
  host_a_ = storage_->host_a;
  a_ = storage_->device_a;
  invalid_ = storage_->device_invalid;
  if (service_->BeginTransfer())
    Check(hipMemsetAsync(invalid_, 0, sizeof(unsigned), stream_));
}
HipLinearZTransfer::~HipLinearZTransfer() = default;
PackedConsumer HipLinearZTransfer::PackedOutputConsumer() {
  return [storage = storage_](std::size_t g, std::size_t s,
                              std::span<const std::byte> c) {
    if (g * 12 + s >= storage->host_c.size() || c.size() != bfp::CBytes)
      throw std::invalid_argument("NPU packed output geometry");
    std::memcpy(storage->host_c[g * 12 + s], c.data(), c.size());
  };
}
void HipLinearZTransfer::FencePackedOutput(unsigned columns) {
  const auto width =
      ProjectionUploadWidth(pitch_, offset_, second_ != nullptr, columns);
  if (!width)
    throw std::invalid_argument("NPU packed output geometry");
  const unsigned offset = offset_ == 0xffffffffU ? 16384 - columns : offset_;
  for (unsigned g = 0; g < tokens_ / kTokens; ++g) {
    const unsigned slices = second_ ? 2 : columns / kN;
    for (unsigned s = 0; s < slices; ++s) {
      auto* dst = (second_ && s == 1 ? second_ : output_) +
                  std::size_t{g} * kTokens * pitch_;
      const unsigned col = second_ ? 0 : offset + s * kN;
      const unsigned valid = second_ ? width : kN;
      Check(bfp::DecodeCHip({storage_->device_c[g * 12 + s], bfp::CBytes},
                            {dst, std::size_t{kTokens} * pitch_}, pitch_, col,
                            valid, false, invalid_, stream_));
    }
  }
}
void HipLinearZTransfer::Check(hipError_t status) {
  if (status == hipSuccess)
    return;
  service_->Hold("NPU prefill HIP launch failed");
  throw std::runtime_error(std::string("NPU prefill HIP: ") +
                           hipGetErrorString(status));
}
BorrowedActivations HipLinearZTransfer::BorrowInput() {
  BorrowedActivations input;
  input.lifetime = storage_;
  for (unsigned g = 0; g < tokens_ / kTokens; ++g) {
    auto* dst = g == 0 ? a_ : storage_->extra_a_device[g - 1];
    const auto* host = g == 0 ? host_a_ : storage_->extra_a[g - 1];
    Check(bfp::EncodeAHip(
        {input_ + std::size_t{g} * kTokens * kK, std::size_t{kTokens} * kK}, kK,
        {dst, bfp::ABytes}, invalid_, stream_));
    input.groups.emplace_back(host, bfp::ABytes);
  }
  // Initialize the persistent worker on the CPU, before any host callback.
  auto* worker = &PersistentLayerWorker();
  input.submit = [stream = stream_, worker](std::function<void()> command) {
    struct Call {
      LayerWorker* worker;
      std::function<void()> command;
      std::uint64_t enqueued;
    };
    auto call =
        std::make_unique<Call>(Call{worker, std::move(command), NowNs()});
    const auto status = hipLaunchHostFunc(
        stream,
        [](void* data) noexcept {
          std::unique_ptr<Call> c(static_cast<Call*>(data));
          Stats().hostfunc_lat_ns.fetch_add(NowNs() - c->enqueued,
                                            std::memory_order_relaxed);
          try {
            c->worker->Enqueue(std::move(c->command));
          } catch (...) {
            FailProcess("NPU stream submit failed");
          }
        },
        call.get());
    if (status != hipSuccess) {
      FailProcess("NPU stream submit launch failed");
      throw std::runtime_error(std::string("NPU stream submit: ") +
                               hipGetErrorString(status));
    }
    (void)call.release();
  };
  return input;
}
std::vector<std::vector<std::uint8_t>> HipLinearZTransfer::FenceInput() {
  throw std::logic_error("NPU HIP input requires stream-ordered Begin");
}
void HipLinearZTransfer::FenceOutput(std::span<const float>, unsigned) {
  throw std::logic_error("NPU HIP output requires packed mapped pages");
}
bool HipLinearZTransfer::WaitOnStream(std::shared_future<LinearZResult> result,
                                      std::chrono::milliseconds deadline) {
  struct Wait {
    std::shared_future<LinearZResult> result;
    std::chrono::milliseconds deadline;
    int lane;
  };
  auto wait = std::make_unique<Wait>(
      Wait{std::move(result), std::min(deadline, service_->session().Timeout()),
           WaitLane()});
  Check(hipLaunchHostFunc(
      stream_,
      [](void* data) noexcept {
        std::unique_ptr<Wait> w(static_cast<Wait*>(data));
        auto& stats = Stats();
        stats.finish_calls.fetch_add(1, std::memory_order_relaxed);
        try {
          if (w->deadline.count() <= 0)
            throw std::runtime_error("NPU stream result deadline");
          if (w->result.wait_for(std::chrono::milliseconds(0)) !=
              std::future_status::ready) {
            const auto t0 = NowNs();
            const bool ready =
                w->result.wait_for(w->deadline) == std::future_status::ready;
            stats.finish_not_ready.fetch_add(1, std::memory_order_relaxed);
            const auto waited = NowNs() - t0;
            stats.finish_wait_ns.fetch_add(waited, std::memory_order_relaxed);
            stats.lane_wait_ns[w->lane].fetch_add(waited, std::memory_order_relaxed);
            stats.lane_blocked[w->lane].fetch_add(1, std::memory_order_relaxed);
            if (!ready)
              throw std::runtime_error("NPU stream result timeout");
          }
          const auto& r = w->result.get();
          if (!r.valid)
            FailProcess(r.error);
          if (!r.valid)
            stats.failures.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
          stats.failures.fetch_add(1, std::memory_order_relaxed);
          FailProcess(e.what());
        } catch (...) {
          stats.failures.fetch_add(1, std::memory_order_relaxed);
          FailProcess("NPU stream wait failed");
        }
      },
      wait.get()));
  (void)wait.release();
  return true;
}
bool CheckHipPrefillChunk(ServerPrefill& service, std::string* error) {
  if (!service.ChunkUsed())
    return true;
  const auto& storage = service.TransferStorage();
  unsigned cause = storage ? *storage->host_invalid : 0;
  // Internal test switch (debug hooks only, see ServerPrefill::Start): report
  // this chunk as an invalid result and take the normal failure path below.
  if (!cause &&
      TestFailChunkHits(service.knobs().test_fail_chunk,
                        Stats().chunks.load(std::memory_order_relaxed)))
    cause = kInvalidDecoderExp255;
  if (cause) {
    Stats().failures.fetch_add(1, std::memory_order_relaxed);
    Faults().invalid_chunks.fetch_add(1, std::memory_order_relaxed);
  }
  // Chunk context: stamps, idle gap, memory, per-chunk counters. The only HAL
  // monitor access is the sequence number read inside EndChunkTelemetry.
  // Telemetry must never throw out of this function: it did not throw before.
  const auto& start = service.chunk();
  std::string detail;
  std::uint64_t hal_new = 0;
  try {
    const auto end = EndChunkTelemetry(start);
    NoteRequestChunk(start, end);
    hal_new = end.hal_new;  // HAL Id 15 events only; WHEA is counted apart
    if (cause || hal_new)
      detail = FormatChunkFailure(start, end, cause);
    // G1: the power rule sees this chunk's mean. A trip only holds later
    // chunks; it never fails this request.
    if (!cause && !ProcessFailed()) {
      try {
        if (auto* guard = service.guard())
          guard->OnChunkEnd(start, end);
      } catch (...) {
      }
    }
  } catch (...) {
    detail.clear();
  }
  if (cause) {
    FailProcess("NPU prefill chunk has invalid activations or packed output" +
                (detail.empty() ? std::string() : ": " + detail));
  } else if (hal_new) {
    // A fault with no Bubo failure (finite garbage would pass the check).
    try {
      server::Logger::Warn(
          "npu",
          "HAL fault event during an NPU chunk with a valid result: " + detail);
    } catch (...) {
    }
  }
  if (!ProcessFailed())
    return true;
  if (error)
    *error =
        "NPU prefill chunk failed (timeout, XRT error or invalid result); "
        "NPU disabled; later requests use GPU";
  return false;
}
}  // namespace gufo::npu
