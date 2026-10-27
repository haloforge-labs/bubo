// SPDX-License-Identifier: MIT
#include "chip_power.hpp"
#include "fault_telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"

#ifdef _WIN32
#include <pdh.h>
#include <pdhmsg.h>
#include <powrprof.h>
#include <windows.h>
#endif

namespace gufo::npu {
ChipPowerMonitor::ChipPowerMonitor(std::unique_ptr<IChipPowerSource> source,
                                   FaultState& state, Log log,
                                   std::chrono::milliseconds period,
                                   std::function<std::int64_t()> now_ns)
    : source_(std::move(source)),
      state_(state),
      log_(std::move(log)),
      period_(period),
      now_ns_(std::move(now_ns)) {
  if (!now_ns_)
    now_ns_ = [] { return SteadyNowNs(); };
}
ChipPowerMonitor::~ChipPowerMonitor() {
  Stop();
}
bool ChipPowerMonitor::available() const noexcept {
  return state_.chip_power_state.load(std::memory_order_relaxed) == 1;
}
void ChipPowerMonitor::MarkUnavailable(const std::string& why) {
  state_.chip_power_state.store(3, std::memory_order_relaxed);
  bool first = false;
  {
    std::lock_guard lock(mutex_);
    first = !unavailable_logged_;
    unavailable_logged_ = true;
    ring_.clear();
  }
  if (first && log_) {
    try {
      log_(false, "chip power unavailable (logging only, no gate): " + why);
    } catch (...) {
    }
  }
}
void ChipPowerMonitor::MarkPagesUnavailable(const std::string& why) {
  state_.pages_state.store(3, std::memory_order_relaxed);
  bool first = false;
  {
    std::lock_guard lock(mutex_);
    first = !pages_logged_;
    pages_logged_ = true;
  }
  if (first && log_) {
    try {
      log_(false, "Memory Pages/sec unavailable (logging only): " + why);
    } catch (...) {
    }
  }
}
void ChipPowerMonitor::MarkGpuUnavailable(const std::string& why) {
  state_.gpu_mem_state.store(3, std::memory_order_relaxed);
  bool first = false;
  {
    std::lock_guard lock(mutex_);
    first = !gpu_logged_;
    gpu_logged_ = true;
  }
  if (first && log_) {
    try {
      log_(false, "GPU memory counters unavailable (logging only): " + why);
    } catch (...) {
    }
  }
}
void ChipPowerMonitor::SampleMemoryPoint() {
  const MemorySample m = SampleMemory(/*precise_commit=*/true);
  if (!m.ok)
    return;
  MemoryPoint point;
  point.ts_ns = now_ns_();
  point.avail_mib = m.available_mib;
  point.commit_mib = m.commit_mib;
  point.commit_limit_mib = m.commit_limit_mib;
  if (pages_source_ &&
      state_.pages_state.load(std::memory_order_relaxed) == 1) {
    double pages = 0;
    std::string error;
    if (pages_source_->Read(&pages, &error))
      point.pages_per_s = pages;
    else
      MarkPagesUnavailable(error.empty() ? "read failed" : error);
  }
  if (gpu_source_ &&
      state_.gpu_mem_state.load(std::memory_order_relaxed) == 1) {
    GpuMemPoint gpu;
    std::string error;
    if (gpu_source_->Read(&gpu, &error))
      point.gpu = gpu;
    else if (!error.empty())
      MarkGpuUnavailable(
          error);  // no instances yet (empty error) is not a failure
  }
  std::lock_guard lock(mutex_);
  mem_ring_.push_back(point);
  if (mem_ring_.size() > kRing)
    mem_ring_.pop_front();
}
void ChipPowerMonitor::Tick() {
  try {
    if (opened_)
      SampleOnce();  // also while unavailable: a transient failure can recover
    SampleMemoryPoint();
  } catch (...) {
  }
}
MemoryWindow ChipPowerMonitor::MemWindow(std::int64_t t0_ns, std::int64_t t1_ns,
                                         double pages_clamp) const {
  MemoryWindow out;
  std::lock_guard lock(mutex_);
  std::int64_t first_ts = 0, previous_ts = 0;
  double pages_sum = 0;
  for (const auto& p : mem_ring_) {
    if (p.ts_ns <= t0_ns || p.ts_ns > t1_ns)
      continue;
    const double pct = p.commit_limit_mib
                           ? 100.0 * static_cast<double>(p.commit_mib) /
                                 static_cast<double>(p.commit_limit_mib)
                           : 0.0;
    if (out.samples == 0) {
      first_ts = p.ts_ns;
      out.min_commit_pct = pct;
    } else {
      out.max_gap_s = std::max(
          out.max_gap_s, static_cast<double>(p.ts_ns - previous_ts) / 1e9);
      out.min_commit_pct = std::min(out.min_commit_pct, pct);
    }
    ++out.samples;
    previous_ts = p.ts_ns;
    out.span_s = static_cast<double>(p.ts_ns - first_ts) / 1e9;
    if (p.pages_per_s >= 0) {
      ++out.pages_samples;
      pages_sum += pages_clamp > 0 ? std::min(p.pages_per_s, pages_clamp)
                                   : p.pages_per_s;
    }
    if (!out.ok) {
      out.ok = true;
      out.min_avail_mib = p.avail_mib;
      out.max_commit_pct = pct;
    } else {
      out.min_avail_mib = std::min(out.min_avail_mib, p.avail_mib);
      out.max_commit_pct = std::max(out.max_commit_pct, pct);
    }
    out.max_pages_per_s = std::max(out.max_pages_per_s, p.pages_per_s);
    if (p.gpu.proc_ok) {
      if (!out.gpu_ok) {
        out.gpu_ok = true;
        out.gpu_local_min = out.gpu_local_max = p.gpu.proc_local_mib;
        out.gpu_dedicated_min = out.gpu_dedicated_max =
            p.gpu.proc_dedicated_mib;
        out.gpu_shared_min = out.gpu_shared_max = p.gpu.proc_shared_mib;
      } else {
        out.gpu_local_min = std::min(out.gpu_local_min, p.gpu.proc_local_mib);
        out.gpu_local_max = std::max(out.gpu_local_max, p.gpu.proc_local_mib);
        out.gpu_dedicated_min =
            std::min(out.gpu_dedicated_min, p.gpu.proc_dedicated_mib);
        out.gpu_dedicated_max =
            std::max(out.gpu_dedicated_max, p.gpu.proc_dedicated_mib);
        out.gpu_shared_min =
            std::min(out.gpu_shared_min, p.gpu.proc_shared_mib);
        out.gpu_shared_max =
            std::max(out.gpu_shared_max, p.gpu.proc_shared_mib);
      }
    }
    if (p.gpu.adapter_ok) {
      out.gpu_adapter_ok = true;
      out.gpu_adapter_shared_max =
          std::max(out.gpu_adapter_shared_max, p.gpu.adapter_shared_mib);
    }
  }
  if (out.pages_samples)
    out.pages_mean = pages_sum / out.pages_samples;
  return out;
}
bool ChipPowerMonitor::Start(bool enabled, bool with_thread,
                             bool power_source) {
  if (!enabled || !source_) {
    state_.chip_power_state.store(0, std::memory_order_relaxed);
    state_.pages_state.store(0, std::memory_order_relaxed);
    return false;
  }
  try {
    // Power, Pages/sec and the memory points are independent: one failing
    // never stops the others. Everything here is logging only.
    std::string error;
    if (!power_source) {
      state_.chip_power_state.store(0, std::memory_order_relaxed);
    } else if (!source_->Open(&error)) {
      MarkUnavailable(error);
    } else {
      opened_ = true;
      state_.chip_power_state.store(1, std::memory_order_relaxed);
      if (SampleOnce() && log_)  // priming read
        log_(false, "chip power logging on (RAPL package energy, no gate)");
    }
    if (pages_source_) {
      std::string pages_error;
      if (pages_source_->Open(&pages_error))
        state_.pages_state.store(1, std::memory_order_relaxed);
      else
        MarkPagesUnavailable(pages_error);
    }
    if (gpu_source_) {
      std::string gpu_error;
      if (gpu_source_->Open(&gpu_error))
        state_.gpu_mem_state.store(1, std::memory_order_relaxed);
      else
        MarkGpuUnavailable(gpu_error);
    }
    SampleMemoryPoint();
    if (with_thread) {
      try {
        thread_ = std::thread([this] {
          std::unique_lock lock(wake_mutex_);
          while (!wake_.wait_for(lock, period_, [this] { return stop_; })) {
            lock.unlock();
            Tick();
            lock.lock();
          }
        });
      } catch (...) {
        MarkUnavailable("could not create the sampling thread");
        return false;
      }
    }
    return available();
  } catch (...) {
    MarkUnavailable("exception while starting");
    return false;
  }
}
void ChipPowerMonitor::Stop() noexcept {
  {
    std::lock_guard lock(wake_mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable())
    thread_.join();
  if (source_)
    source_->Close();
  if (pages_source_)
    pages_source_->Close();
  if (gpu_source_)
    gpu_source_->Close();
}
// A bad read or sample clears the ring (no interval may span the gap) and
// counts; only kFailuresToUnavailable in a row make the power unavailable.
bool ChipPowerMonitor::PowerFailure(const std::string& why) {
  unsigned failures = 0;
  {
    std::lock_guard lock(mutex_);
    ring_.clear();
    failures = ++consecutive_failures_;
  }
  if (failures >= kFailuresToUnavailable)
    MarkUnavailable(why);
  return false;
}
bool ChipPowerMonitor::SampleOnce() {
  if (!opened_)
    return false;
  try {
    PowerSample sample;
    std::string error;
    if (!source_->Read(&sample, &error))
      return PowerFailure(error.empty() ? "read failed" : error);
    const char* bad = nullptr;
    bool recovered = false;
    {
      std::lock_guard lock(mutex_);
      if (!ring_.empty()) {
        const auto& last = ring_.back();
        if (sample.ts_ns <= last.ts_ns) {
          bad = "non-monotonic sample time";
        } else if (sample.energy_pwh < last.energy_pwh) {
          bad = "energy counter went backwards (wrap or reset)";
        } else {
          const double seconds =
              static_cast<double>(sample.ts_ns - last.ts_ns) / 1e9;
          const double watts =
              (sample.energy_pwh - last.energy_pwh) * kJoulesPerPwh / seconds;
          if (watts < 0 || watts > kMaxPlausibleWatts + 1e-6)
            bad = "power outside 0-400 W";
        }
      }
      if (!bad) {
        ring_.push_back(sample);
        if (ring_.size() > kRing)
          ring_.pop_front();
        consecutive_failures_ = 0;
        recovered = !available();
        if (recovered)
          unavailable_logged_ = false;
      }
    }
    if (bad)
      return PowerFailure(bad);
    if (recovered) {
      state_.chip_power_state.store(1, std::memory_order_relaxed);
      if (log_)
        log_(false, "chip power recovered (logging and guard input resume)");
    }
    return true;
  } catch (...) {
    return PowerFailure("exception while sampling");
  }
}
PowerWindow ChipPowerMonitor::Window(std::int64_t t0_ns,
                                     std::int64_t t1_ns) const {
  PowerWindow out;
  if (!available())
    return out;
  // Walk the ring under the lock (about 500 samples at most, no copy).
  std::lock_guard lock(mutex_);
  const auto& copy = ring_;
  if (copy.size() < 2)
    return out;
  double energy_j = 0, seconds = 0, max_w = 0;
  unsigned used = 0;
  struct Interval {
    std::int64_t end;
    double joules, secs;
  };
  Interval latest{};
  bool have_latest = false;
  for (std::size_t i = 1; i < copy.size(); ++i) {
    const double secs =
        static_cast<double>(copy[i].ts_ns - copy[i - 1].ts_ns) / 1e9;
    const double joules =
        (copy[i].energy_pwh - copy[i - 1].energy_pwh) * kJoulesPerPwh;
    if (copy[i].ts_ns <= t1_ns) {
      latest = {copy[i].ts_ns, joules, secs};
      have_latest = true;
    }
    if (copy[i].ts_ns > t0_ns && copy[i].ts_ns <= t1_ns) {
      energy_j += joules;
      seconds += secs;
      max_w = std::max(max_w, joules / secs);
      ++used;
    }
  }
  if (used == 0) {
    if (!have_latest)
      return out;
    out.approx = true;  // chunk shorter than the sampling period
    energy_j = latest.joules;
    seconds = latest.secs;
    max_w = latest.joules / latest.secs;
    used = 1;
  }
  out.ok = true;
  out.intervals = used;
  out.span_s = seconds;
  out.mean_w = energy_j / seconds;
  out.max_w = max_w;
  return out;
}

namespace {
std::atomic<ChipPowerMonitor*> global_chip{nullptr};
std::mutex global_chip_mutex;
std::unique_ptr<ChipPowerMonitor>& GlobalOwner() {
  static std::unique_ptr<ChipPowerMonitor> owner;
  return owner;
}
void ChipLog(bool warn, const std::string& text) {
  server::Logger::Log(warn ? server::LogLevel::kWarn : server::LogLevel::kInfo,
                      "npu", text);
}
}  // namespace
ChipPowerMonitor* GlobalChipPower() noexcept {
  return global_chip.load(std::memory_order_acquire);
}
ChipPowerMonitor& InstallGlobalChipPower(
    std::unique_ptr<IChipPowerSource> source, bool enabled,
    std::unique_ptr<IPagesSource> pages, std::unique_ptr<IGpuMemSource> gpu,
    bool with_thread, bool power_source) {
  std::lock_guard lock(global_chip_mutex);
  global_chip.store(nullptr, std::memory_order_release);
  if (GlobalOwner())
    GlobalOwner()->Stop();
  GlobalOwner() =
      std::make_unique<ChipPowerMonitor>(std::move(source), Faults(), ChipLog);
  GlobalOwner()->SetPagesSource(std::move(pages));
  GlobalOwner()->SetGpuMemSource(std::move(gpu));
  GlobalOwner()->Start(enabled, with_thread, power_source);
  global_chip.store(GlobalOwner().get(), std::memory_order_release);
  return *GlobalOwner();
}
void StopGlobalChipPower() noexcept {
  try {
    std::lock_guard lock(global_chip_mutex);
    global_chip.store(nullptr, std::memory_order_release);
    if (GlobalOwner())
      GlobalOwner()->Stop();
  } catch (...) {
  }
}

std::string FormatPowerScheme(const PowerScheme& scheme) {
  if (!scheme.ok)
    return "power scheme unavailable";
  return "power scheme " + scheme.guid + " name=" + scheme.name +
         " overlay=" + scheme.overlay_guid;
}

#ifdef _WIN32
namespace {
std::string GuidText(const GUID& g) {
  char text[48];
  std::snprintf(text, sizeof(text),
                "{%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                static_cast<unsigned long>(g.Data1), g.Data2, g.Data3,
                g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4],
                g.Data4[5], g.Data4[6], g.Data4[7]);
  return text;
}
std::string Narrow(const wchar_t* text) {
  const int bytes =
      WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  if (bytes <= 1)
    return {};
  std::string out(static_cast<std::size_t>(bytes), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), bytes, nullptr,
                      nullptr);
  out.resize(static_cast<std::size_t>(bytes) - 1);
  return out;
}

// pdh.dll is loaded at run time, like wevtapi: a missing library makes Open()
// fail, which is the "unavailable" state, never a build or start-up failure.
class PdhChipPowerSource final : public IChipPowerSource {
public:
  ~PdhChipPowerSource() override { Close(); }
  bool Open(std::string* error) override {
    library_ =
        LoadLibraryExW(L"pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library_) {
      *error = "pdh.dll not loadable, error " + std::to_string(GetLastError());
      return false;
    }
    open_ = reinterpret_cast<decltype(&PdhOpenQueryW)>(
        GetProcAddress(library_, "PdhOpenQueryW"));
    add_ = reinterpret_cast<decltype(&PdhAddEnglishCounterW)>(
        GetProcAddress(library_, "PdhAddEnglishCounterW"));
    collect_ = reinterpret_cast<decltype(&PdhCollectQueryData)>(
        GetProcAddress(library_, "PdhCollectQueryData"));
    raw_ = reinterpret_cast<decltype(&PdhGetRawCounterValue)>(
        GetProcAddress(library_, "PdhGetRawCounterValue"));
    formatted_ = reinterpret_cast<decltype(&PdhGetFormattedCounterValue)>(
        GetProcAddress(library_, "PdhGetFormattedCounterValue"));
    close_ = reinterpret_cast<decltype(&PdhCloseQuery)>(
        GetProcAddress(library_, "PdhCloseQuery"));
    if (!open_ || !add_ || !collect_ || !raw_ || !formatted_ || !close_) {
      *error = "pdh.dll lacks the counter functions";
      return false;
    }
    if (open_(nullptr, 0, &query_) != ERROR_SUCCESS) {
      query_ = nullptr;
      *error = "PdhOpenQuery failed";
      return false;
    }
    // Cumulative energy (picowatt-hours): mean power over any interval is
    // dE/dt. If only the Power counter exists, integrate it.
    PDH_STATUS energy =
        add_(query_, L"\\Energy Meter(RAPL_Package0_PKG)\\Energy", 0,
             &energy_counter_);
    if (energy != ERROR_SUCCESS) {
      energy_counter_ = nullptr;
      PDH_STATUS power =
          add_(query_, L"\\Energy Meter(RAPL_Package0_PKG)\\Power", 0,
               &power_counter_);
      if (power != ERROR_SUCCESS) {
        power_counter_ = nullptr;
        char text[96];
        std::snprintf(text, sizeof(text),
                      "RAPL package counters missing (PDH 0x%08lX)",
                      static_cast<unsigned long>(energy));
        *error = text;
        return false;
      }
    }
    collect_(query_);  // priming collect
    return true;
  }
  bool Read(PowerSample* sample, std::string* error) override {
    if (!query_ || collect_(query_) != ERROR_SUCCESS) {
      *error = "PdhCollectQueryData failed";
      return false;
    }
    const auto now = SteadyNowNs();
    DWORD type = 0;
    if (energy_counter_) {
      PDH_RAW_COUNTER raw{};
      if (raw_(energy_counter_, &type, &raw) != ERROR_SUCCESS ||
          (raw.CStatus != PDH_CSTATUS_VALID_DATA &&
           raw.CStatus != PDH_CSTATUS_NEW_DATA)) {
        *error = "invalid energy counter data";
        return false;
      }
      sample->ts_ns = now;
      sample->energy_pwh = static_cast<double>(raw.FirstValue);
      return true;
    }
    PDH_FMT_COUNTERVALUE value{};
    if (formatted_(power_counter_, PDH_FMT_DOUBLE, &type, &value) !=
            ERROR_SUCCESS ||
        (value.CStatus != PDH_CSTATUS_VALID_DATA &&
         value.CStatus != PDH_CSTATUS_NEW_DATA)) {
      *error = "invalid power counter data";
      return false;
    }
    if (have_last_) {
      const double seconds = static_cast<double>(now - last_ns_) / 1e9;
      integrated_pwh_ += value.doubleValue / 1000.0 * seconds / kJoulesPerPwh;
    }
    have_last_ = true;
    last_ns_ = now;
    sample->ts_ns = now;
    sample->energy_pwh = integrated_pwh_;
    return true;
  }
  void Close() noexcept override {
    if (query_ && close_)
      close_(query_);
    query_ = nullptr;
  }

private:
  HMODULE library_ = nullptr;
  decltype(&PdhOpenQueryW) open_ = nullptr;
  decltype(&PdhAddEnglishCounterW) add_ = nullptr;
  decltype(&PdhCollectQueryData) collect_ = nullptr;
  decltype(&PdhGetRawCounterValue) raw_ = nullptr;
  decltype(&PdhGetFormattedCounterValue) formatted_ = nullptr;
  decltype(&PdhCloseQuery) close_ = nullptr;
  PDH_HQUERY query_ = nullptr;
  PDH_HCOUNTER energy_counter_ = nullptr, power_counter_ = nullptr;
  bool have_last_ = false;
  std::int64_t last_ns_ = 0;
  double integrated_pwh_ = 0;
};
}  // namespace
namespace {
// Memory\Pages/sec: a rate counter, so Open() primes it and the first Read()
// is the second collect.
class PdhPagesSource final : public IPagesSource {
public:
  ~PdhPagesSource() override { Close(); }
  bool Open(std::string* error) override {
    library_ =
        LoadLibraryExW(L"pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library_) {
      *error = "pdh.dll not loadable, error " + std::to_string(GetLastError());
      return false;
    }
    auto open = reinterpret_cast<decltype(&PdhOpenQueryW)>(
        GetProcAddress(library_, "PdhOpenQueryW"));
    auto add = reinterpret_cast<decltype(&PdhAddEnglishCounterW)>(
        GetProcAddress(library_, "PdhAddEnglishCounterW"));
    collect_ = reinterpret_cast<decltype(&PdhCollectQueryData)>(
        GetProcAddress(library_, "PdhCollectQueryData"));
    formatted_ = reinterpret_cast<decltype(&PdhGetFormattedCounterValue)>(
        GetProcAddress(library_, "PdhGetFormattedCounterValue"));
    close_ = reinterpret_cast<decltype(&PdhCloseQuery)>(
        GetProcAddress(library_, "PdhCloseQuery"));
    if (!open || !add || !collect_ || !formatted_ || !close_) {
      *error = "pdh.dll lacks the counter functions";
      return false;
    }
    if (open(nullptr, 0, &query_) != ERROR_SUCCESS) {
      query_ = nullptr;
      *error = "PdhOpenQuery failed";
      return false;
    }
    const PDH_STATUS status = add(query_, L"\\Memory\\Pages/sec", 0, &counter_);
    if (status != ERROR_SUCCESS) {
      char text[64];
      std::snprintf(text, sizeof(text), "counter missing (PDH 0x%08lX)",
                    static_cast<unsigned long>(status));
      *error = text;
      return false;
    }
    collect_(query_);  // priming collect
    return true;
  }
  bool Read(double* pages_per_s, std::string* error) override {
    DWORD type = 0;
    PDH_FMT_COUNTERVALUE value{};
    if (!query_ || collect_(query_) != ERROR_SUCCESS ||
        formatted_(counter_, PDH_FMT_DOUBLE, &type, &value) != ERROR_SUCCESS ||
        (value.CStatus != PDH_CSTATUS_VALID_DATA &&
         value.CStatus != PDH_CSTATUS_NEW_DATA) ||
        !(value.doubleValue >= 0) || value.doubleValue > 1e8) {
      *error = "invalid Pages/sec data";
      return false;
    }
    *pages_per_s = value.doubleValue;
    return true;
  }
  void Close() noexcept override {
    if (query_ && close_)
      close_(query_);
    query_ = nullptr;
  }

private:
  HMODULE library_ = nullptr;
  decltype(&PdhCollectQueryData) collect_ = nullptr;
  decltype(&PdhGetFormattedCounterValue) formatted_ = nullptr;
  decltype(&PdhCloseQuery) close_ = nullptr;
  PDH_HQUERY query_ = nullptr;
  PDH_HCOUNTER counter_ = nullptr;
};
}  // namespace
std::unique_ptr<IPagesSource> MakeWindowsPagesSource() {
  return std::make_unique<PdhPagesSource>();
}
std::unique_ptr<IChipPowerSource> MakeWindowsChipPowerSource() {
  return std::make_unique<PdhChipPowerSource>();
}
PowerScheme ReadPowerScheme() {
  PowerScheme out;
  HMODULE lib =
      LoadLibraryExW(L"powrprof.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!lib)
    return out;
  using GetActive = DWORD(WINAPI*)(HKEY, GUID**);
  using ReadName = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*,
                                  PUCHAR, LPDWORD);
  using GetOverlay = DWORD(WINAPI*)(GUID*);
  auto get_active =
      reinterpret_cast<GetActive>(GetProcAddress(lib, "PowerGetActiveScheme"));
  auto read_name =
      reinterpret_cast<ReadName>(GetProcAddress(lib, "PowerReadFriendlyName"));
  auto get_overlay = reinterpret_cast<GetOverlay>(
      GetProcAddress(lib, "PowerGetEffectiveOverlayScheme"));
  GUID* scheme = nullptr;
  if (!get_active || get_active(nullptr, &scheme) != ERROR_SUCCESS || !scheme)
    return out;
  out.guid = GuidText(*scheme);
  if (read_name) {
    DWORD size = 0;
    read_name(nullptr, scheme, nullptr, nullptr, nullptr, &size);
    if (size > 2 && size < 4096) {
      std::vector<UCHAR> buffer(size + 2, 0);
      if (read_name(nullptr, scheme, nullptr, nullptr, buffer.data(), &size) ==
          ERROR_SUCCESS)
        out.name = Narrow(reinterpret_cast<const wchar_t*>(buffer.data()));
    }
  }
  LocalFree(scheme);
  if (get_overlay) {
    GUID overlay{};
    if (get_overlay(&overlay) == ERROR_SUCCESS)
      out.overlay_guid = GuidText(overlay);
  }
  out.ok = true;
  return out;
}
double ProcessCpuMs() {
  FILETIME created, exited, kernel, user;
  if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
    return 0;
  const auto to_ms = [](const FILETIME& f) {
    return static_cast<double>(
               (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) |
               f.dwLowDateTime) /
           10000.0;
  };
  return to_ms(kernel) + to_ms(user);
}
#else
std::unique_ptr<IChipPowerSource> MakeWindowsChipPowerSource() {
  return nullptr;
}
std::unique_ptr<IPagesSource> MakeWindowsPagesSource() {
  return nullptr;
}
PowerScheme ReadPowerScheme() {
  return {};
}
double ProcessCpuMs() {
  return 0;
}
#endif
}  // namespace gufo::npu
