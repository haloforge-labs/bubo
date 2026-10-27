#pragma once

#include "src/cli/machine_adapter.hpp"

namespace bubo::cli {
// One read-only ADLX pass: supported GPU power and clock metrics and the
// Variable Graphics Memory reading. ADLX loads at run time from the AMD
// driver. A build without BUBO_ADLX_SDK_DIR, a missing driver library or an
// unsupported metric gives unavailable readings with a reason.
struct AdlxSample {
  MachineGpuReadings gpu;
  MachineVgmReadings vgm;
};
AdlxSample SampleAdlx();
}  // namespace bubo::cli
