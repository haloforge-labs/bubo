// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_CHIP_POWER_HPP_
#define GUFO_NPU_CHIP_POWER_HPP_
// G0: chip power LOGGING only. There is no gate and nothing here changes what
// the NPU does. A 1 s background thread reads the cumulative RAPL package
// energy counter (Windows PDH, loaded at run time) into a small ring of (time,
// energy) samples. The chunk code asks for the mean and maximum power over its
// own interval from that ring: no PDH call on the prefill path, only a short
// mutex-guarded copy. Any failure turns the source "unavailable": it is logged
// once and the log fields print "na".
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "fault_state.hpp"
#include "mem_trace.hpp"

namespace gufo::npu {



struct PowerSample {
  std::int64_t ts_ns = 0;
  double energy_pwh = 0;
};
inline constexpr double kJoulesPerPwh = 3.6e-9;
inline constexpr double kMaxPlausibleWatts = 400.0;

class IChipPowerSource {
public:
  virtual ~IChipPowerSource() = default;
  virtual bool Open(std::string* error) = 0;
  virtual bool Read(PowerSample* sample, std::string* error) = 0;
  virtual void Close() noexcept {}
};

struct PowerWindow {
  bool ok = false;
  bool approx =
      false;  // interval shorter than the 1 s sampling: nearest second
  double mean_w = 0, max_w = 0, span_s = 0;
  unsigned intervals = 0;
};

// One memory reading taken by the same 1 s tick (cheap: GlobalMemoryStatusEx
// and, when it exists, the Pages/sec counter). Used for the minimum available
// memory and maximum commit % seen while the NPU was active.
struct MemoryPoint {
  std::int64_t ts_ns = 0;
  std::uint64_t avail_mib = 0, commit_mib = 0, commit_limit_mib = 0;
  double pages_per_s = -1;  // -1: counter not available
  GpuMemPoint gpu;          // PDH GPU memory counters (ok flags inside)
};
struct MemoryWindow {
  bool ok = false;
  std::uint64_t min_avail_mib = 0;
  double max_commit_pct = 0;
  double max_pages_per_s = -1;  // -1: not available
  // GPU memory counters over the interval (min/max of the 1 s points).
  bool gpu_ok = false;
  double gpu_local_min = 0, gpu_local_max = 0;
  double gpu_dedicated_min = 0, gpu_dedicated_max = 0;
  double gpu_shared_min = 0, gpu_shared_max = 0;
  // Guard inputs (G1): how well the window is covered, and the extremes the
  // memory rule needs. pages_mean averages the per-sample values after
  // clamping each to the caller's pages_clamp.
  unsigned samples = 0;
  double span_s = 0, max_gap_s = 0, min_commit_pct = 0;
  unsigned pages_samples = 0;
  double pages_mean = 0;
  bool gpu_adapter_ok = false;
  double gpu_adapter_shared_max = 0;
};
// Memory\Pages/sec (PDH English counter). Fail open: errors mean "na".
class IPagesSource {
public:
  virtual ~IPagesSource() = default;
  virtual bool Open(std::string* error) = 0;
  virtual bool Read(double* pages_per_s, std::string* error) = 0;
  virtual void Close() noexcept {}
};

class ChipPowerMonitor {
public:
  using Log = std::function<void(bool warn, const std::string&)>;
  ChipPowerMonitor(std::unique_ptr<IChipPowerSource> source, FaultState& state,
                   Log log,
                   std::chrono::milliseconds period = std::chrono::seconds(1),
                   std::function<std::int64_t()> now_ns = {});
  // Optional; call before Start().
  void SetPagesSource(std::unique_ptr<IPagesSource> pages) {
    pages_source_ = std::move(pages);
  }
  // Optional; call before Start(). Read only on the sampler tick.
  void SetGpuMemSource(std::unique_ptr<IGpuMemSource> gpu) {
    gpu_source_ = std::move(gpu);
  }
  ~ChipPowerMonitor();
  ChipPowerMonitor(const ChipPowerMonitor&) = delete;
  ChipPowerMonitor& operator=(const ChipPowerMonitor&) = delete;
  // enabled=false: nothing is opened. with_thread=false: the caller drives
  // SampleOnce() (tests). Returns true when the source is usable.
  // enabled=false: nothing at all. power_source=false: the memory sampler runs
  // but the RAPL source is never opened (GUFO_NPU_CHIP_POWER=0 with the guard).
  bool Start(bool enabled, bool with_thread = true, bool power_source = true);
  void Stop() noexcept;
  // One validated read. False once the monitor is unavailable.
  bool SampleOnce();
  // One tick of the sampler: power read (if available) and a memory point.
  void Tick();
  // pages_clamp > 0: every Pages/sec sample is clamped to it before averaging.
  MemoryWindow MemWindow(std::int64_t t0_ns, std::int64_t t1_ns,
                         double pages_clamp = -1) const;
  // A failed power read is tolerated; this many in a row make it unavailable.
  static constexpr unsigned kFailuresToUnavailable = 5;
  bool available() const noexcept;
  bool thread_running() const noexcept { return thread_.joinable(); }
  // Mean/max power over [t0_ns, t1_ns] (steady_clock ns) from cached samples.
  PowerWindow Window(std::int64_t t0_ns, std::int64_t t1_ns) const;
  static constexpr std::size_t kRing = 512;

private:
  void MarkUnavailable(const std::string& why);
  void MarkPagesUnavailable(const std::string& why);
  void MarkGpuUnavailable(const std::string& why);
  void SampleMemoryPoint();
  std::unique_ptr<IChipPowerSource> source_;
  FaultState& state_;
  Log log_;
  std::chrono::milliseconds period_;
  mutable std::mutex mutex_;
  std::deque<PowerSample> ring_;
  std::deque<MemoryPoint> mem_ring_;
  std::unique_ptr<IPagesSource> pages_source_;
  std::function<std::int64_t()> now_ns_;
  bool pages_logged_ = false, gpu_logged_ = false;
  std::unique_ptr<IGpuMemSource> gpu_source_;
  bool unavailable_logged_ = false, opened_ = false;
  unsigned consecutive_failures_ = 0;
  bool PowerFailure(const std::string& why);
  std::mutex wake_mutex_;
  std::condition_variable wake_;
  bool stop_ = false;
  std::thread thread_;
};

// Process-wide monitor, read lock-free by the chunk code.
ChipPowerMonitor* GlobalChipPower() noexcept;
ChipPowerMonitor& InstallGlobalChipPower(
    std::unique_ptr<IChipPowerSource> source, bool enabled,
    std::unique_ptr<IPagesSource> pages = nullptr,
    std::unique_ptr<IGpuMemSource> gpu = nullptr, bool with_thread = true,
    bool power_source = true);
void StopGlobalChipPower() noexcept;
// Windows PDH source (pdh.dll loaded at run time). Null on other platforms.
std::unique_ptr<IChipPowerSource> MakeWindowsChipPowerSource();
std::unique_ptr<IPagesSource> MakeWindowsPagesSource();

// Read-only: active Windows power scheme and effective overlay.
struct PowerScheme {
  bool ok = false;
  std::string guid, name, overlay_guid;
};
PowerScheme ReadPowerScheme();
std::string FormatPowerScheme(const PowerScheme& scheme);
// User + kernel CPU time of this process in ms (GetProcessTimes); 0 elsewhere.
double ProcessCpuMs();
}  // namespace gufo::npu
#endif
