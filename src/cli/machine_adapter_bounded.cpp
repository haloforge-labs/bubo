// The bounded wrapper around a machine probe. It is a separate file so a test
// driver that supplies its own CreateSystemMachineProbe does not link the real
// probe (machine_adapter_system.cpp) through status.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_output.hpp"

namespace bubo::cli {
namespace {
std::atomic<bool> g_abandoned_call{false};

// Runs each slow call of `inner` on its own thread and waits for it up to a
// limit. A driver call can block for a minute on a cold start; the status
// record must still arrive.
class BoundedMachineProbe final : public MachineProbe {
public:
  BoundedMachineProbe(std::shared_ptr<MachineProbe> inner,
                      std::chrono::milliseconds each,
                      std::chrono::milliseconds total)
      : inner_(std::move(inner)),
        each_(each),
        deadline_(std::chrono::steady_clock::now() + total) {}

  MachineReading OverlayScheme() override {
    return Call<MachineReading>(
        &MachineProbe::OverlayScheme,
        "powrprof.dll PowerGetEffectiveOverlayScheme",
        [](const std::string& source, const std::string& why) {
          return UnavailableReading(source, "Windows 10 1709 or later", "guid",
                                    UtcTime(), why);
        });
  }
  MachineReading EffectivePowerMode() override {
    return Call<MachineReading>(
        &MachineProbe::EffectivePowerMode,
        "powrprof.dll PowerRegisterForEffectivePowerModeNotifications",
        [](const std::string& source, const std::string& why) {
          return UnavailableReading(source, "EFFECTIVE_POWER_MODE_V2", "enum",
                                    UtcTime(), why);
        });
  }
  MachineReading PowerSource() override { return inner_->PowerSource(); }
  MachineGpuReadings Gpu() override {
    return Call<MachineGpuReadings>(
        &MachineProbe::Gpu, "ADLX GPU metrics",
        [](const std::string& source, const std::string& why) {
          MachineGpuReadings out;
          out.power = UnavailableReading(source, "ADLX", "W", UtcTime(), why);
          out.clock = UnavailableReading(source, "ADLX", "MHz", UtcTime(), why);
          return out;
        });
  }
  MachineVgmReadings Vgm() override {
    return Call<MachineVgmReadings>(
        &MachineProbe::Vgm, "ADLX variable graphics memory",
        [](const std::string& source, const std::string& why) {
          MachineVgmReadings out;
          out.supported =
              UnavailableReading(source, "ADLX", "bool", UtcTime(), why);
          out.carved_raw_gb =
              UnavailableReading(source, "ADLX", "GB", UtcTime(), why);
          out.remaining_raw_gb =
              UnavailableReading(source, "ADLX", "GB", UtcTime(), why);
          return out;
        });
  }
  MachineReading HostPhysicalMemory() override {
    return inner_->HostPhysicalMemory();
  }
  MachineReading HostAvailableMemory() override {
    return inner_->HostAvailableMemory();
  }
  MachineReading InstalledPhysicalMemory() override {
    return inner_->InstalledPhysicalMemory();
  }
  MachineReading OemManufacturer() override {
    return inner_->OemManufacturer();
  }
  MachineReading OemModel() override { return inner_->OemModel(); }
  MachineReading BiosVendor() override { return inner_->BiosVendor(); }
  MachineReading BiosVersion() override { return inner_->BiosVersion(); }
  MachineReading BiosDate() override { return inner_->BiosDate(); }
  MachineDxgiReadings Dxgi() override {
    return Call<MachineDxgiReadings>(
        &MachineProbe::Dxgi, "DXGI adapter enumeration",
        [](const std::string& source, const std::string& why) {
          return DxgiReadingsFromAdapters({}, source + ": " + why, UtcTime());
        });
  }

private:
  template <class R>
  R Call(R (MachineProbe::*method)(), const char* source,
         const std::function<R(const std::string&, const std::string&)>&
             unavailable) {
    struct State {
      std::mutex mutex;
      std::condition_variable done;
      std::optional<R> result;
      bool failed = false;
    };
    const auto left = deadline_ - std::chrono::steady_clock::now();
    const auto wait =
        std::min<std::chrono::steady_clock::duration>(each_, left);
    if (wait <= std::chrono::steady_clock::duration::zero())
      return unavailable(source,
                         "Skipped: an earlier call did not answer in time.");
    auto state = std::make_shared<State>();
    std::thread([state, inner = inner_, method] {
      std::optional<R> value;
      bool failed = false;
      try {
        value = ((*inner).*method)();
      } catch (...) {
        failed = true;
      }
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->result = std::move(value);
        state->failed = failed;
      }
      state->done.notify_all();
    }).detach();
    std::unique_lock<std::mutex> lock(state->mutex);
    const bool finished = state->done.wait_for(
        lock, wait, [&] { return state->result.has_value() || state->failed; });
    if (finished && state->result)
      return std::move(*state->result);
    if (!finished)
      g_abandoned_call = true;
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(wait).count();
    return unavailable(source, finished
                                   ? "The call failed."
                                   : "The call did not answer in " +
                                         std::to_string(ms) + " ms.");
  }

  std::shared_ptr<MachineProbe> inner_;
  std::chrono::milliseconds each_;
  std::chrono::steady_clock::time_point deadline_;
};
}  // namespace

std::unique_ptr<MachineProbe> MakeBoundedMachineProbe(
    std::shared_ptr<MachineProbe> inner, std::chrono::milliseconds each,
    std::chrono::milliseconds total) {
  return std::make_unique<BoundedMachineProbe>(std::move(inner), each, total);
}

std::unique_ptr<MachineProbe> CreateBoundedSystemMachineProbe(
    std::chrono::milliseconds each, std::chrono::milliseconds total) {
  return MakeBoundedMachineProbe(
      std::shared_ptr<MachineProbe>(CreateSystemMachineProbe()), each, total);
}

bool MachineProbeHasAbandonedCall() { return g_abandoned_call.load(); }
}  // namespace bubo::cli
