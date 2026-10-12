// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_FABRIC_PEAK_HPP_
#define GUFO_NPU_FABRIC_PEAK_HPP_
#include <memory>
#include <optional>
#include <string>

#include "src/core/experimental_authorization.hpp"

namespace gufo::npu {
enum class RestoreReason {
  Exit,
  Abort,
  CrashHandler,
  WatcherDeath,
  SetFailure
};
struct ClockSnapshot {
  std::optional<unsigned> exact_mode;
  std::string reason;
};
// This is OpenCL clock policy, not another XRT layer. A qualified driver MUST
// supply an exact mode query and a separately armed restorer before mutation.
class ClockDriver {
public:
  virtual ~ClockDriver() = default;
  virtual ClockSnapshot Query() = 0;
  virtual bool IndependentRestorerReady() const noexcept = 0;
  virtual bool SetPeak() = 0;
  virtual bool Restore(unsigned mode, RestoreReason reason) noexcept = 0;
};
class FabricPeak {
public:
  FabricPeak(core::ExperimentalAuthorization authorization, ClockDriver& driver)
      : authorization_(std::move(authorization)), driver_(driver) {}
  ~FabricPeak();
  FabricPeak(const FabricPeak&) = delete;
  FabricPeak& operator=(const FabricPeak&) = delete;
  void Start();
  bool Restore(RestoreReason reason) noexcept;
  bool restore_pending() const noexcept { return dirty_; }

private:
  core::ExperimentalAuthorization authorization_;
  ClockDriver& driver_;
  std::optional<unsigned> previous_;
  bool attempted_ = false, dirty_ = false;
};
// Lazy System32 OpenCL load, explicit fabric authorization only. No device
// query/set is attempted without a qualified mode ABI and independent recovery.
std::unique_ptr<ClockDriver> InstalledOpenClClockDriver(
    const core::ExperimentalAuthorization& authorization);
}  // namespace gufo::npu
#endif
