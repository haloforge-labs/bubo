#pragma once

#include <memory>
#include <mutex>

#include "src/cli/serve/lifecycle.hpp"
#include "src/npu/runtime_status.hpp"

namespace bubo::cli {
// Owned by the serve runner, whose lifecycle outlives it. A producer may have
// copied the callback before it is cleared; the shared sink makes that late
// callback harmless. No lifecycle lock is held while calling the NPU API.
class NpuStatusPublisher {
public:
  explicit NpuStatusPublisher(ServeLifecycle& lifecycle)
      : sink_(std::make_shared<Sink>(&lifecycle)) {
    gufo::npu::SetNpuRuntimeStatusPublisher(
        [sink = sink_](const gufo::npu::NpuRuntimeStatus& status) {
          PublishTo(sink, status);
        });
    try {
      Publish(gufo::npu::CurrentNpuRuntimeStatus());
    } catch (...) {
      Clear();
      throw;
    }
  }
  ~NpuStatusPublisher() { Clear(); }
  NpuStatusPublisher(const NpuStatusPublisher&) = delete;
  NpuStatusPublisher& operator=(const NpuStatusPublisher&) = delete;

  // Publish already-observed producer values without querying the NPU path.
  void Publish(const gufo::npu::NpuRuntimeStatus& status) {
    PublishTo(sink_, status);
  }

private:
  struct Sink {
    explicit Sink(ServeLifecycle* target) : lifecycle(target) {}
    std::mutex mutex;
    ServeLifecycle* lifecycle;
  };
  static void PublishTo(const std::shared_ptr<Sink>& sink,
                        const gufo::npu::NpuRuntimeStatus& status) {
    auto details = Value::object();
    details["npu_prefill_runtime"] = gufo::npu::ToJson(status);
    std::lock_guard lock(sink->mutex);
    if (sink->lifecycle)
      sink->lifecycle->SetDetails(std::move(details));
  }
  void Clear() {
    gufo::npu::SetNpuRuntimeStatusPublisher({});
    std::lock_guard lock(sink_->mutex);
    sink_->lifecycle = nullptr;
  }
  std::shared_ptr<Sink> sink_;
};
}  // namespace bubo::cli
