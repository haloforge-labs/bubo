#ifndef GUFO_CORE_HIP_DEVICE_PROBE_HPP_
#define GUFO_CORE_HIP_DEVICE_PROBE_HPP_

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cli/serve/logging.hpp"
#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip {

// Own the probe resources from model load: context-loss detection must not
// depend on allocating resources after the failed work unit.
class DeviceUsabilityProbe {
public:
  DeviceUsabilityProbe() {
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    if (const auto error = hipMalloc(&buffer_, sizeof(std::uint32_t));
        error != hipSuccess) {
      LogCleanupError(hipStreamDestroy(stream_));
      throw std::runtime_error(std::string("device probe allocation: ") +
                               hipGetErrorString(error));
    }
  }
  ~DeviceUsabilityProbe() {
    LogCleanupError(hipFree(buffer_));
    LogCleanupError(hipStreamDestroy(stream_));
  }
  DeviceUsabilityProbe(const DeviceUsabilityProbe&) = delete;
  DeviceUsabilityProbe& operator=(const DeviceUsabilityProbe&) = delete;
  DeviceUsabilityProbe(DeviceUsabilityProbe&&) = delete;
  DeviceUsabilityProbe& operator=(DeviceUsabilityProbe&&) = delete;

  [[nodiscard]] bool DeviceUsable() const {
    constexpr auto kTimeout = std::chrono::seconds(5);
    (void)hipGetLastError();
    if (hipMemsetAsync(buffer_, 0, sizeof(std::uint32_t), stream_) != hipSuccess ||
        hipGetLastError() != hipSuccess)
      return false;
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    for (;;) {
      const auto status = hipStreamQuery(stream_);
      if (status == hipSuccess)
        return true;
      if (status != hipErrorNotReady)
        return false;
      if (std::chrono::steady_clock::now() >= deadline) {
        server::Logger::Warn("scheduler", "event=device_probe_timeout timeout_s=" +
                                              std::to_string(kTimeout.count()));
        return true; // Pending work on busy hardware is not a lost context.
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

private:
  hipStream_t stream_{};
  void* buffer_{};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_DEVICE_PROBE_HPP_
