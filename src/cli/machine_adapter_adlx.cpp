// ADLX read-only pass. The ADLX SDK headers are not part of this repository.
// The build compiles this code only when BUBO_ADLX_SDK_DIR names an installed
// SDK. The AMD runtime library (amdadlx64.dll) loads at run time through the
// SDK helper. Nothing here adds an AMD file to the package.

#include "src/cli/machine_adapter_adlx.hpp"

#include <string>
#include <utility>

#include "src/cli/machine_output.hpp"

#if defined(_WIN32) && defined(BUBO_HAS_ADLX)
#include "ADLXHelper/Windows/Cpp/ADLXHelper.h"
#include "Include/IPerformanceMonitoring.h"
#include "Include/ISystem3.h"
#endif

namespace bubo::cli {
namespace {
constexpr const char* kPowerSource =
    "ADLX IADLXGPUMetrics::GPUPower (IADLXPerformanceMonitoringServices)";
constexpr const char* kClockSource =
    "ADLX IADLXGPUMetrics::GPUClockSpeed (IADLXPerformanceMonitoringServices)";
constexpr const char* kVgmSource =
    "ADLX IADLXSystem3::GetVariableGraphicsMemory";

AdlxSample AllUnavailable(const std::string& version,
                          const std::string& reason) {
  const auto time = UtcTime();
  AdlxSample sample;
  sample.gpu.power =
      UnavailableReading(kPowerSource, version, "W", time, reason);
  sample.gpu.clock =
      UnavailableReading(kClockSource, version, "MHz", time, reason);
  sample.vgm.supported = UnavailableReading(
      std::string(kVgmSource) + " IsSupported", version, "bool", time, reason);
  sample.vgm.carved_raw_gb = UnavailableReading(
      std::string(kVgmSource) + " GetOption MemoryCarved", version, "GB", time,
      reason);
  sample.vgm.remaining_raw_gb = UnavailableReading(
      std::string(kVgmSource) + " GetOption MemoryRemaining", version, "GB",
      time, reason);
  return sample;
}
}  // namespace

#if defined(_WIN32) && defined(BUBO_HAS_ADLX)
namespace {
using namespace adlx;
std::string Code(ADLX_RESULT result) {
  return "ADLX result " + std::to_string(static_cast<int>(result)) + ".";
}
bool Supported(ADLX_RESULT (ADLX_STD_CALL IADLXGPUMetricsSupport::*fn)(
                   adlx_bool*),
               IADLXGPUMetricsSupportPtr& support) {
  adlx_bool value = false;
  return support && ADLX_SUCCEEDED(((*support).*fn)(&value)) && value;
}
}  // namespace

AdlxSample SampleAdlx() {
  ADLXHelper helper;
  if (ADLX_FAILED(helper.Initialize()))
    return AllUnavailable("ADLX (not loaded)",
                          "ADLX is not available. The AMD driver library did "
                          "not load.");
  const char* version_text = helper.QueryVersion();
  const std::string version =
      std::string("ADLX runtime ") + (version_text ? version_text : "unknown");
  const auto time = UtcTime();
  AdlxSample sample = AllUnavailable(version, "Not read.");
  {
    IADLXPerformanceMonitoringServicesPtr perf;
    IADLXGPUListPtr gpus;
    IADLXGPUPtr gpu;
    IADLXSystem* system = helper.GetSystemServices();
    if (!system ||
        ADLX_FAILED(system->GetPerformanceMonitoringServices(&perf)) ||
        ADLX_FAILED(system->GetGPUs(&gpus)) ||
        ADLX_FAILED(gpus->At(gpus->Begin(), &gpu))) {
      sample = AllUnavailable(version,
                              "The ADLX GPU or performance service is "
                              "not available.");
    } else {
      const char* name = nullptr;
      if (ADLX_SUCCEEDED(gpu->Name(&name)) && name)
        sample.gpu.gpu_name = name;
      IADLXGPUMetricsSupportPtr support;
      perf->GetSupportedGPUMetrics(gpu, &support);
      sample.gpu.power_supported =
          Supported(&IADLXGPUMetricsSupport::IsSupportedGPUPower, support);
      sample.gpu.clock_supported =
          Supported(&IADLXGPUMetricsSupport::IsSupportedGPUClockSpeed, support);
      IADLXGPUMetricsPtr metrics;
      const ADLX_RESULT got = perf->GetCurrentGPUMetrics(gpu, &metrics);
      adlx_double watts = 0;
      adlx_int mhz = 0;
      if (!sample.gpu.power_supported)
        sample.gpu.power =
            UnavailableReading(kPowerSource, version, "W", time,
                               "ADLX reports GPU power as not supported.");
      else if (ADLX_FAILED(got) || ADLX_FAILED(metrics->GPUPower(&watts)))
        sample.gpu.power = UnavailableReading(
            kPowerSource, version, "W", time,
            "The GPU power metric read failed. " + Code(got));
      else
        sample.gpu.power =
            AvailableReading(static_cast<double>(watts), kPowerSource, version,
                             "W", time);
      if (!sample.gpu.clock_supported)
        sample.gpu.clock =
            UnavailableReading(kClockSource, version, "MHz", time,
                               "ADLX reports GPU clock as not supported.");
      else if (ADLX_FAILED(got) || ADLX_FAILED(metrics->GPUClockSpeed(&mhz)))
        sample.gpu.clock = UnavailableReading(
            kClockSource, version, "MHz", time,
            "The GPU clock metric read failed. " + Code(got));
      else
        sample.gpu.clock =
            AvailableReading(static_cast<double>(mhz), kClockSource, version,
                             "MHz", time);

      // Variable Graphics Memory: IsSupported and GetOption only.
      const std::string supported_source =
          std::string(kVgmSource) + " IsSupported";
      IADLXSystem3Ptr system3(system);
      IADLXVariableGraphicsMemoryPtr vgm;
      adlx_bool vgm_supported = false;
      if (!system3 ||
          ADLX_FAILED(system3->GetVariableGraphicsMemory(&vgm)) || !vgm) {
        const std::string reason =
            "This ADLX runtime has no Variable Graphics Memory interface.";
        sample.vgm.supported = UnavailableReading(supported_source, version,
                                                  "bool", time, reason);
        sample.vgm.carved_raw_gb.reason = reason;
        sample.vgm.remaining_raw_gb.reason = reason;
      } else if (ADLX_FAILED(vgm->IsSupported(&vgm_supported))) {
        const std::string reason = "The VGM support query failed.";
        sample.vgm.supported = UnavailableReading(supported_source, version,
                                                  "bool", time, reason);
        sample.vgm.carved_raw_gb.reason = reason;
        sample.vgm.remaining_raw_gb.reason = reason;
      } else {
        sample.vgm.supported = AvailableReading(
            static_cast<bool>(vgm_supported), supported_source, version,
            "bool", time);
        IADLXVariableGraphicsMemoryOptionPtr option;
        adlx_double carved = 0;
        adlx_double remaining = 0;
        if (!vgm_supported) {
          const std::string reason = "ADLX reports VGM as not supported.";
          sample.vgm.carved_raw_gb.reason = reason;
          sample.vgm.remaining_raw_gb.reason = reason;
        } else if (ADLX_FAILED(vgm->GetOption(&option)) || !option) {
          const std::string reason = "The VGM option could not be read.";
          sample.vgm.carved_raw_gb.reason = reason;
          sample.vgm.remaining_raw_gb.reason = reason;
        } else {
          if (ADLX_SUCCEEDED(option->MemoryCarved(&carved)))
            sample.vgm.carved_raw_gb = AvailableReading(
                static_cast<double>(carved),
                std::string(kVgmSource) + " GetOption MemoryCarved", version,
                "GB", time);
          else
            sample.vgm.carved_raw_gb.reason = "MemoryCarved read failed.";
          if (ADLX_SUCCEEDED(option->MemoryRemaining(&remaining)))
            sample.vgm.remaining_raw_gb = AvailableReading(
                static_cast<double>(remaining),
                std::string(kVgmSource) + " GetOption MemoryRemaining",
                version, "GB", time);
          else
            sample.vgm.remaining_raw_gb.reason = "MemoryRemaining read failed.";
        }
      }
    }
  }
  helper.Terminate();
  return sample;
}
#else
AdlxSample SampleAdlx() {
  return AllUnavailable(
      "ADLX (not built in)",
      "This build has no ADLX support. Build with BUBO_ADLX_SDK_DIR to read "
      "AMD driver telemetry.");
}
#endif
}  // namespace bubo::cli
