// SPDX-License-Identifier: MIT
#include "fabric_peak.hpp"

#include <Windows.h>

#include <stdexcept>

namespace gufo::npu {
void FabricPeak::Start() {
  authorization_.Require(core::ExperimentalFeature::FabricPeak);
  if (attempted_)
    throw std::runtime_error(
        "fabric-peak cannot retry in this process lifetime");
  attempted_ = true;
  auto snapshot = driver_.Query();
  if (!snapshot.exact_mode)
    throw std::runtime_error("fabric-peak refused: " + snapshot.reason);
  previous_ = snapshot.exact_mode;
  if (!driver_.IndependentRestorerReady())
    throw std::runtime_error(
        "fabric-peak refused: independent crash/death restorer unavailable");
  // A setter may report failure AFTER changing hardware. Mark dirty first;
  // every failure path must restore the saved exact mode, not assume DEFAULT.
  dirty_ = true;
  try {
    if (!driver_.SetPeak())
      throw std::runtime_error("fabric-peak setter failed");
  } catch (...) {
    (void)Restore(RestoreReason::SetFailure);
    throw;
  }
}
bool FabricPeak::Restore(RestoreReason reason) noexcept {
  if (!dirty_)
    return true;
  if (!driver_.Restore(*previous_, reason))
    return false;
  dirty_ = false;
  return true;
}
FabricPeak::~FabricPeak() {
  (void)Restore(RestoreReason::Exit);
}
namespace {
class OpenClClockDriver final : public ClockDriver {
public:
  OpenClClockDriver() {
    wchar_t system[MAX_PATH];
    const auto size = GetSystemDirectoryW(system, MAX_PATH);
    if (!size || size >= MAX_PATH)
      throw std::runtime_error("fabric-peak System32 path unavailable");
    std::wstring path(system, size);
    path += L"\\OpenCL.dll";
    module_ =
        LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module_)
      throw std::runtime_error(
          "fabric-peak installed OpenCL loader unavailable");
  }
  ~OpenClClockDriver() override { FreeLibrary(module_); }
  ClockSnapshot Query() override {
    // Symbol discovery is not an exact previous-mode query. Do not guess a
    // vendor ABI, treat clock ratios as a mode, or restore a guessed DEFAULT.
    const auto extension_lookup =
        GetProcAddress(module_, "clGetExtensionFunctionAddressForPlatform");
    return {{},
            extension_lookup ? "installed OpenCL lacks a qualified exact "
                               "previous-mode query; no clock setter called"
                             : "installed OpenCL lacks extension lookup; no "
                               "clock setter called"};
  }
  bool IndependentRestorerReady() const noexcept override { return false; }
  bool SetPeak() override {
    throw std::runtime_error(
        "unqualified clSetDeviceClockModeAMD PEAK is disabled");
  }
  bool Restore(unsigned, RestoreReason) noexcept override { return false; }

private:
  HMODULE module_ = nullptr;
};
}  // namespace
std::unique_ptr<ClockDriver> InstalledOpenClClockDriver(
    const core::ExperimentalAuthorization& authorization) {
  authorization.Require(core::ExperimentalFeature::FabricPeak);
  return std::make_unique<OpenClClockDriver>();
}
}  // namespace gufo::npu
