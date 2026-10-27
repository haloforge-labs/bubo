// SPDX-License-Identifier: MIT
#include "server_prefill.hpp"

#include "bo_address_table.hpp"
#include "chip_power.hpp"
#include "chunk_retry.hpp"
#include "hal_fault_monitor.hpp"
#include "mem_trace.hpp"
#include "platform_info.hpp"
#include "request_stats.hpp"
#include "runtime_status.hpp"
#include "src/cli/serve/logging.hpp"
namespace gufo::npu {
std::unique_ptr<NpuGuard> ServerPrefill::MakeGuard() {
  GuardHooks hooks;
  hooks.now_ns = [] { return SteadyNowNs(); };
  hooks.memory = [](std::int64_t t0, std::int64_t t1, double clamp) {
    auto* sampler = GlobalChipPower();
    return sampler ? sampler->MemWindow(t0, t1, clamp) : MemoryWindow{};
  };
  hooks.power = [](std::int64_t t0, std::int64_t t1) {
    auto* sampler = GlobalChipPower();
    return sampler ? sampler->Window(t0, t1) : PowerWindow{};
  };
  hooks.process_failed = [] { return ProcessFailed(); };
  hooks.log = [](bool warn, const std::string& text) {
    server::Logger::Log(
        warn ? server::LogLevel::kWarn : server::LogLevel::kInfo, "npu", text);
  };
  hooks.publish = [](const GuardTripInfo& info) {
    NpuHoldInfo hold;
    hold.held = info.held;
    hold.reason = info.reason;
    hold.trips = info.trips;
    hold.sticky = info.sticky;
    hold.w_mean = info.power_w_mean;
    hold.min_available_mib = info.min_available_mib;
    hold.max_commit_pct = info.max_commit_pct;
    NoteNpuHold(hold);
  };
  hooks.set_keepalive_paused = [this](bool paused) {
    session_.SetKeepalivePaused(paused);
  };
  hooks.warm_up = [this](std::string* error) {
    return session_.WarmUpBeat(error);
  };
  return std::make_unique<NpuGuard>(knobs_.guard, knobs_.guard_test,
                                    std::move(hooks));
}
void ServerPrefill::BeginChunk(unsigned tokens) noexcept {
  ++chunk_serial_;
  chunk_used_ = false;
  reset_invalid_ = true;
  chunk_hold_ = false;
  // BO device addresses: one line at the first chunk (weights are loaded by
  // then) and again whenever a BO was created later.
  try {
    if (const auto generation = BoTableGeneration();
        generation != logged_bo_generation_) {
      logged_bo_generation_ = generation;
      if (const auto table = CurrentBoTable())
        server::Logger::Info("npu", FormatBoTable(*table));
    }
  } catch (...) {
  }
  Stats().last_layer.store(-1, std::memory_order_relaxed);
  if (!ready())
    return;
  // The guard decides once per chunk; everything inside the chunk reads the
  // latched value (lane_ready()). A guard failure never holds anything.
  try {
    if (guard_)
      chunk_hold_ = guard_->OnBeginChunk();
  } catch (...) {
    chunk_hold_ = false;
  }
  if (chunk_hold_)
    return;  // held: no NPU use, so no telemetry, no memory probe, no check
  chunk_ = ChunkStart{};  // a throwing telemetry must not leave the last
                          // chunk's marks
  try {
    chunk_ = BeginChunkTelemetry(tokens);  // clock, memory, counters, HAL seq
  } catch (...) {
  }
}
bool ServerPrefill::Start() {
  ready_ = session_.Start().healthy;
#ifndef GUFO_NPU_TESTING  // unit tests never touch the Windows event log
  // Detection only; a failure to subscribe is logged once and changes nothing.
  // A monitor failure must never reach the startup catch that disables the NPU.
  if (ready_) {
    try {
      InstallGlobalHalMonitor(MakeWindowsHalFaultSource(), knobs_.hal_monitor);
    } catch (...) {
    }
    // G0 chip power: logging only, no gate; same rule, never throws into
    // startup.
    try {
      // The sampler also feeds the guard's memory rule: with
      // GUFO_NPU_CHIP_POWER=0 and the guard on it runs without opening RAPL.
      const bool sampler = knobs_.chip_power || knobs_.guard != GuardMode::kOff;
      InstallGlobalChipPower(
          MakeWindowsChipPowerSource(), sampler, MakeWindowsPagesSource(),
          knobs_.mem_trace ? MakeWindowsGpuMemSource() : nullptr,
          /*with_thread=*/true, /*power_source=*/knobs_.chip_power);
      const auto scheme = ReadPowerScheme();
      server::Logger::Info("npu", FormatPowerScheme(scheme));
      if (scheme.ok)
        NoteNpuScheme(scheme.guid, scheme.name);
    } catch (...) {
    }
    // BIOS and board, read once from the registry (no COM): one line, and the
    // same fields on every NPU failure line. Fail-open.
    try {
      auto source = MakeRegistryPlatformSource();
      std::string problem;
      const auto& platform = LoadPlatformInfo(source.get(), &problem);
      server::Logger::Info("npu", "platform " + FormatPlatformFields(platform));
      if (!problem.empty())
        server::Logger::Info("npu", "platform info incomplete: " + problem);
    } catch (...) {
    }
  }
  // G0b, log only, fail open: buffer-page residency sample at chunk begin/end.
  if (ready_ && knobs_.mem_trace) {
    try {
      InstallGlobalMemProbe(
          MakeWindowsMemProbe([this] { return session_.SamplePages(); }));
    } catch (...) {
    }
  }
#endif
#ifndef GUFO_NPU_TESTING
  if (ready_) {
    try {
      server::Logger::Info(
          "npu", std::string("NPU prefill guard: mode ") +
                     GuardModeName(knobs_.guard) +
                     " (envelope guard for power and memory, see the NPU "
                     "README)");
      if (knobs_.guard_test.power_w || knobs_.guard_test.hold_s)
        server::Logger::Warn(
            "npu",
            "npu guard: GUFO_NPU_GUARD_TEST override active (internal test "
            "setting, not for normal use): power_w=" +
                (knobs_.guard_test.power_w
                     ? std::to_string(*knobs_.guard_test.power_w)
                     : std::string("default")) +
                " hold_s=" +
                (knobs_.guard_test.hold_s
                     ? std::to_string(*knobs_.guard_test.hold_s)
                     : std::string("default")));
      if (knobs_.test_fail_chunk)
        server::Logger::Warn("npu",
                             TestFailChunkActiveText(*knobs_.test_fail_chunk));
    } catch (...) {
    }
  }
#endif
  // Status: armed, or off with the start-up reason. Authorization and build
  // denials never reach here (the server reports those itself).
  NoteNpuStartup(ready_, ready_ ? std::string() : "NPU startup failed.");
  return ready_;
}
void ServerPrefill::Reject(std::string error) {
  ready_ = false;
  session_.RejectResult(std::move(error));
}
void ServerPrefill::Hold(std::string error) noexcept {
  try {
    Reject(std::move(error));
  } catch (...) {
  }
}
ServerPrefill::~ServerPrefill() {
  StopGlobalHalMonitor();
  StopGlobalChipPower();
  StopGlobalMemProbe();
  session_.Shutdown();
}
}  // namespace gufo::npu
