// SPDX-License-Identifier: MIT
#include "fault_telemetry.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
// clang-format off
#include <psapi.h>
// clang-format on
#endif

#include "bo_address_table.hpp"
#include "mem_trace.hpp"
#include "platform_info.hpp"
#include "request_stats.hpp"
#include "runtime_status.hpp"

namespace gufo::npu {
namespace {
std::function<std::chrono::system_clock::time_point()>& ClockOverride() {
  static std::function<std::chrono::system_clock::time_point()> fn;
  return fn;
}
std::chrono::system_clock::time_point WallNow() {
  const auto& fn = ClockOverride();
  return fn ? fn() : std::chrono::system_clock::now();
}
std::atomic<IMemorySource*> memory_override{nullptr};

std::string Fmt(const char* format, ...) {
  char text[1536];
  va_list args;
  va_start(args, format);
  std::vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  return text;
}
unsigned long long U(std::uint64_t v) {
  return static_cast<unsigned long long>(v);
}
std::string MemText(const MemorySample& m, bool commit) {
  if (!m.ok)
    return "na";
  return std::to_string(commit ? m.commit_mib : m.available_mib);
}

// chip_power_w_* come from 1 Hz RAPL package samples (approx=1: the chunk was
// shorter than a sample, so the nearest second is shown). Logging only.
double CommitPct(const MemorySample& m) {
  return m.ok && m.commit_limit_mib
             ? 100.0 * static_cast<double>(m.commit_mib) /
                   static_cast<double>(m.commit_limit_mib)
             : 0.0;
}
std::string MemoryText(const ChunkEnd& end) {
  const std::string limit =
      end.memory.ok ? std::to_string(end.memory.commit_limit_mib) : "na";
  const std::string pages =
      end.max_pages_per_s >= 0 ? Fmt("%.0f", end.max_pages_per_s) : "na";
  return Fmt(
      " commit_limit_mib=%s commit_pct_end=%.1f min_avail_mib=%llu "
      "max_commit_pct=%.1f pages_per_s_max=%s pages=%s",
      limit.c_str(), CommitPct(end.memory), U(end.min_avail_mib),
      end.max_commit_pct, pages.c_str(), PagesStateName());
}
std::string NaU(bool ok, std::uint64_t v) {
  return ok ? std::to_string(v) : "na";
}
std::string NaF(bool ok, double v) {
  return ok ? Fmt("%.0f", v) : "na";
}
// G0b: min/max of the PDH GPU memory points over the NPU-active interval.
struct GpuAgg {
  bool ok = false, adapter_ok = false;
  double lmin = 0, lmax = 0, dmin = 0, dmax = 0, smin = 0, smax = 0, amax = 0;
  void Add(const MemoryWindow& w) {
    if (w.gpu_ok) {
      if (!ok) {
        ok = true;
        lmin = w.gpu_local_min;
        lmax = w.gpu_local_max;
        dmin = w.gpu_dedicated_min;
        dmax = w.gpu_dedicated_max;
        smin = w.gpu_shared_min;
        smax = w.gpu_shared_max;
      } else {
        lmin = std::min(lmin, w.gpu_local_min);
        lmax = std::max(lmax, w.gpu_local_max);
        dmin = std::min(dmin, w.gpu_dedicated_min);
        dmax = std::max(dmax, w.gpu_dedicated_max);
        smin = std::min(smin, w.gpu_shared_min);
        smax = std::max(smax, w.gpu_shared_max);
      }
    }
    if (w.gpu_adapter_ok) {
      adapter_ok = true;
      amax = std::max(amax, w.gpu_adapter_shared_max);
    }
  }
};
std::string GpuText(const GpuAgg& g) {
  return Fmt(
      " gpu_proc_local_mib_min=%s gpu_proc_local_mib_max=%s "
      "gpu_proc_dedicated_mib_min=%s gpu_proc_dedicated_mib_max=%s "
      "gpu_proc_shared_mib_min=%s gpu_proc_shared_mib_max=%s "
      "gpu_adapter_shared_mib_max=%s",
      NaF(g.ok, g.lmin).c_str(), NaF(g.ok, g.lmax).c_str(),
      NaF(g.ok, g.dmin).c_str(), NaF(g.ok, g.dmax).c_str(),
      NaF(g.ok, g.smin).c_str(), NaF(g.ok, g.smax).c_str(),
      NaF(g.adapter_ok, g.amax).c_str());
}
// Per chunk: buffer-page residency (begin/end) and process memory.
std::string TraceText(const ChunkStart& s, const ChunkEnd& e) {
  const char* state = !GlobalMemProbe()            ? "off"
                      : (s.trace.ok && e.trace.ok) ? "on"
                                                   : "na";
  const bool b = s.trace.ok, f = e.trace.ok;
  GpuAgg gpu;
  gpu.Add(e.gpu_window);
  return Fmt(" bo_pages_sampled=%s bo_pages_valid_begin=%s "
             "bo_pages_valid_end=%s bo_pages_shared_end=%s "
             "bo_pages_locked=%s bo_pages_large=%s process_ws_mib_begin=%s "
             "process_ws_mib_end=%s process_private_mib_end=%s mem_trace=%s",
             NaU(f, e.trace.pages_sampled).c_str(),
             NaU(b, s.trace.valid).c_str(), NaU(f, e.trace.valid).c_str(),
             NaU(f, e.trace.shared).c_str(),
             NaU(f, std::max(s.trace.locked, e.trace.locked)).c_str(),
             NaU(f, e.trace.large).c_str(), NaU(b, s.trace.ws_mib).c_str(),
             NaU(f, e.trace.ws_mib).c_str(),
             NaU(f, e.trace.private_mib).c_str(), state) +
         GpuText(gpu) + Fmt(" gpu_mem=%s", GpuMemStateName()) +
         Fmt(" first_chunk_after_rearm=%d", s.first_after_rearm ? 1 : 0);
}
std::string PowerText(const PowerWindow& power, double cpu_ms) {
  if (!power.ok)
    return Fmt(
        " chip_power_w_mean=na chip_power_w_max=na cpu_ms=%.0f chip_power=%s",
        cpu_ms, ChipPowerStateName());
  return Fmt(
      " chip_power_w_mean=%.1f chip_power_w_max=%.1f chip_power_n=%u "
      "chip_power_approx=%d cpu_ms=%.0f chip_power=%s",
      power.mean_w, power.max_w, power.intervals, power.approx ? 1 : 0, cpu_ms,
      ChipPowerStateName());
}

struct TraceAgg {
  bool have = false;
  std::uint64_t pages = 0, valid_min = 0, locked_max = 0;
  std::uint64_t ws_first = 0, ws_min = 0, ws_last = 0, private_max = 0;
  void Add(const ChunkStart& s, const ChunkEnd& e) {
    for (const MemTraceSample* t : {&s.trace, &e.trace}) {
      if (!t->ok)
        continue;
      if (!have) {
        have = true;
        pages = t->pages_sampled;
        valid_min = t->valid;
        ws_first = ws_min = t->ws_mib;
      }
      valid_min = std::min<std::uint64_t>(valid_min, t->valid);
      locked_max = std::max<std::uint64_t>(locked_max, t->locked);
      ws_min = std::min(ws_min, t->ws_mib);
      ws_last = t->ws_mib;
      private_max = std::max(private_max, t->private_mib);
    }
  }
};
// Mean chip power of each NPU chunk of a request (first 16), for calibration of
// the guard threshold: "~" marks a sub-second estimate, "na" no power data.
constexpr std::size_t kChunkPowerListMax = 16;
struct RequestAccumulator {
  std::vector<std::string> chunk_w;
  std::vector<std::string> chunk_backup;  // retry backup copy ms per chunk
  std::vector<std::string> chunk_ms;      // NPU chunk wall time per chunk
  double backup_total = 0, backup_max = 0;
  unsigned chunk_n = 0;
  bool first_after_rearm = false;
  TraceAgg trace;
  GpuAgg gpu;
  double power_joules = 0, power_seconds = 0, power_max = 0, cpu_ms = 0;
  double active_s = 0;  // NPU-active seconds (sum of chunk durations)
  std::uint64_t min_avail_mib = 0, commit_limit_mib = 0;
  double max_commit_pct = 0, max_pages = -1;
  std::mutex mutex;
  bool any = false;
  WallStamp first, last_end;
  std::int64_t first_idle_gap_ms = -1;
  MemorySample first_memory, last_memory;
  std::uint64_t hal_faults_taken = 0, whea_taken = 0;
};
RequestAccumulator& Request() {
  static RequestAccumulator accumulator;
  return accumulator;
}
}  // namespace

std::string DescribeInvalidBits(unsigned bits) {
  if (bits == 0)
    return "none";
  struct Name {
    unsigned bit;
    const char* text;
  };
  static constexpr Name names[] = {
      {kInvalidEncoderNonFinite, "A-nonfinite"},
      {kInvalidEncoderSaturated, "A-saturated"},
      {kInvalidDecoderExp255, "C-exp255"},
      {kInvalidDecoderExp254Mant128, "C-exp254-mant128"}};
  std::string text;
  for (const auto& name : names) {
    if (bits & name.bit) {
      if (!text.empty())
        text += '|';
      text += name.text;
      bits &= ~name.bit;
    }
  }
  if (bits) {
    if (!text.empty())
      text += '|';
    text += Fmt("unknown-0x%x", bits);
  }
  return text;
}

void SetWallClockForTest(
    std::function<std::chrono::system_clock::time_point()> fn) {
  ClockOverride() = std::move(fn);
}

WallStamp FormatWallStamp(std::chrono::system_clock::time_point time) {
  using namespace std::chrono;
  const auto since = time.time_since_epoch();
  auto seconds = duration_cast<std::chrono::seconds>(since);
  auto ms = duration_cast<milliseconds>(since - seconds).count();
  if (ms < 0) {  // before 1970: keep the fraction positive
    ms += 1000;
    seconds -= std::chrono::seconds(1);
  }
  const std::time_t t = static_cast<std::time_t>(seconds.count());
  std::tm utc{}, local{};
#ifdef _WIN32
  gmtime_s(&utc, &t);
  localtime_s(&local, &t);
#else
  gmtime_r(&t, &utc);
  localtime_r(&t, &local);
#endif
  WallStamp stamp;
  stamp.utc = Fmt("%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ", utc.tm_year + 1900,
                  utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                  utc.tm_sec, static_cast<long long>(ms));
  stamp.local =
      Fmt("%04d-%02d-%02d %02d:%02d:%02d.%03lld", local.tm_year + 1900,
          local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min,
          local.tm_sec, static_cast<long long>(ms));
  return stamp;
}

IMemorySource* SetMemorySourceForTest(IMemorySource* source) {
  return memory_override.exchange(source);
}
MemorySample SampleMemory(bool precise_commit) {
  if (auto* source = memory_override.load())
    return source->Sample();
  MemorySample sample;
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    sample.ok = true;
    sample.available_mib = status.ullAvailPhys >> 20;
    // ullTotalPageFile is the smaller of the system and the process/job limit
    // (skewed under a job object): kept only as a fallback.
    sample.commit_limit_mib = status.ullTotalPageFile >> 20;
    sample.commit_mib =
        (status.ullTotalPageFile -
         std::min(status.ullTotalPageFile, status.ullAvailPageFile)) >>
        20;
    // System-wide commit charge and limit (what the guard's memory rule uses).
    using PerfFn = BOOL(WINAPI*)(PPERFORMANCE_INFORMATION, DWORD);
    static const PerfFn perf = reinterpret_cast<PerfFn>(GetProcAddress(
        GetModuleHandleW(L"kernel32.dll"), "K32GetPerformanceInfo"));
    PERFORMANCE_INFORMATION info{};
    info.cb = sizeof(info);
    if (precise_commit && perf && perf(&info, sizeof(info)) &&
        info.CommitLimit > 0) {
      const std::uint64_t page = info.PageSize;
      sample.commit_limit_mib = (info.CommitLimit * page) >> 20;
      sample.commit_mib = (info.CommitTotal * page) >> 20;
    }
  }
#endif
  return sample;
}

CounterSnapshot CounterSnapshot::Take() {
  auto& s = Stats();
  CounterSnapshot c;
  c.linear_begins = s.linear_begins.load(std::memory_order_relaxed);
  c.projection_begins = s.projection_begins.load(std::memory_order_relaxed);
  c.finish_calls = s.finish_calls.load(std::memory_order_relaxed);
  c.failures = s.failures.load(std::memory_order_relaxed);
  c.hostfunc_lat_ns = s.hostfunc_lat_ns.load(std::memory_order_relaxed);
  return c;
}

ChunkStart BeginChunkTelemetry(unsigned tokens) {
  ChunkStart start;
  start.tokens = tokens;
  start.wall = WallNow();
  start.stamp = FormatWallStamp(start.wall);
  start.steady_ns = SteadyNowNs();
  start.idle_gap_ms = IdleGapMs();
  start.memory = SampleMemory();
  start.counters = CounterSnapshot::Take();
  start.cpu_ms = ProcessCpuMs();
  SampleMemTrace(&start.trace);  // fail-open; fields print na
  start.hal_seq = Faults().hal_seq.load(std::memory_order_acquire);
  return start;
}

ChunkEnd EndChunkTelemetry(const ChunkStart& start) {
  ChunkEnd end;
  end.stamp = FormatWallStamp(WallNow());
  end.duration_ms = static_cast<double>(SteadyNowNs() - start.steady_ns) / 1e6;
  end.memory = SampleMemory();
  const auto now = CounterSnapshot::Take();
  end.counters.linear_begins = now.linear_begins - start.counters.linear_begins;
  end.counters.projection_begins =
      now.projection_begins - start.counters.projection_begins;
  end.counters.finish_calls = now.finish_calls - start.counters.finish_calls;
  end.counters.failures = now.failures - start.counters.failures;
  end.counters.hostfunc_lat_ns =
      now.hostfunc_lat_ns - start.counters.hostfunc_lat_ns;
  end.cpu_ms = ProcessCpuMs() - start.cpu_ms;
  SampleMemTrace(&end.trace);
  end.min_avail_mib =
      std::min(start.memory.available_mib, end.memory.available_mib);
  end.max_commit_pct = std::max(CommitPct(start.memory), CommitPct(end.memory));
  if (auto* sampler = GlobalChipPower()) {
    const auto window = sampler->MemWindow(start.steady_ns, SteadyNowNs());
    if (window.ok) {
      end.min_avail_mib = std::min(end.min_avail_mib, window.min_avail_mib);
      // One definition: the sampler's (GetPerformanceInfo), as the guard uses.
      end.max_commit_pct = window.max_commit_pct;
      end.max_pages_per_s = window.max_pages_per_s;
    }
    end.gpu_window = window;
  }
  if (auto* power = GlobalChipPower())  // cached samples; no PDH call here
    end.power = power->Window(start.steady_ns, SteadyNowNs());
  end.index = Stats().chunks.load(std::memory_order_relaxed);
  end.hal_new =
      Faults().hal_seq.load(std::memory_order_acquire) - start.hal_seq;
  return end;
}

std::string FormatChunkFailure(const ChunkStart& start, const ChunkEnd& end,
                               unsigned cause_bits) {
  auto& faults = Faults();
  std::string text =
      Fmt("cause=0x%x(%s) chunk=%llu tokens=%u", cause_bits,
          DescribeInvalidBits(cause_bits).c_str(), U(end.index), start.tokens);
  text += Fmt(
      " start_utc=%s start_local=%s end_utc=%s end_local=%s duration_ms=%.1f",
      start.stamp.utc.c_str(), start.stamp.local.c_str(), end.stamp.utc.c_str(),
      end.stamp.local.c_str(), end.duration_ms);
  text += Fmt(" idle_gap_ms=%lld", static_cast<long long>(start.idle_gap_ms));
  text += Fmt(
      " avail_mib_start=%s avail_mib_end=%s commit_mib_start=%s "
      "commit_mib_end=%s",
      MemText(start.memory, false).c_str(), MemText(end.memory, false).c_str(),
      MemText(start.memory, true).c_str(), MemText(end.memory, true).c_str());
  text +=
      Fmt(" linear_begins=%llu projection_begins=%llu finish_calls=%llu "
          "failures=%llu hostfunc_lat_ms=%.1f",
          U(end.counters.linear_begins), U(end.counters.projection_begins),
          U(end.counters.finish_calls), U(end.counters.failures),
          static_cast<double>(end.counters.hostfunc_lat_ns) / 1e6);
  text += Fmt(" hal_faults_total=%llu hal_in_chunk=%llu monitor=%s",
              U(faults.hal_faults.load()), U(end.hal_new), MonitorStateName());
  text += PowerText(end.power, end.cpu_ms);
  text += MemoryText(end);
  text += TraceText(start, end);
  // Which buffer the latest IOMMU fault hit: when one arrived during this
  // chunk, or within the last 3 s.
  {
    const auto resolved =
        RecentHalResolution(end.hal_new ? 60'000'000'000LL : 3'000'000'000LL);
    if (!resolved.empty())
      text += " hal_" + resolved;
  }
  // Which BIOS/board this fault happened on (read once at NPU start).
  text += " " + FormatPlatformFields(CachedPlatformInfo());
  return text;
}

static std::atomic<std::uint64_t> g_chunks_noted{0};

std::uint64_t RequestChunksNoted() noexcept {
  return g_chunks_noted.load(std::memory_order_relaxed);
}

void NoteRequestChunk(const ChunkStart& start, const ChunkEnd& end) {
  g_chunks_noted.fetch_add(1, std::memory_order_relaxed);
  auto& r = Request();
  std::lock_guard lock(r.mutex);
  if (!r.any) {
    r.any = true;
    r.first = start.stamp;
    r.first_idle_gap_ms = start.idle_gap_ms;
    r.first_memory = start.memory;
    r.power_joules = r.power_seconds = r.power_max = r.cpu_ms = r.active_s = 0;
    r.min_avail_mib = end.min_avail_mib;
    r.max_commit_pct = end.max_commit_pct;
    r.max_pages = -1;
    r.trace = TraceAgg{};
    r.gpu = GpuAgg{};
    r.chunk_w.clear();
    r.chunk_backup.clear();
    r.chunk_ms.clear();
    r.backup_total = r.backup_max = 0;
    r.chunk_n = 0;
    r.first_after_rearm = false;
  }
  ++r.chunk_n;
  if (r.chunk_ms.size() < kChunkPowerListMax) {
    r.chunk_ms.push_back(Fmt("%.1f", end.duration_ms));
    r.chunk_backup.push_back("na");  // NoteChunkBackup fills it in
  }
  if (start.first_after_rearm)
    r.first_after_rearm = true;
  if (r.chunk_w.size() < kChunkPowerListMax)
    r.chunk_w.push_back(end.power.ok
                            ? std::string(end.power.approx ? "~" : "") +
                                  Fmt("%.1f", end.power.mean_w)
                            : "na");
  r.trace.Add(start, end);
  r.gpu.Add(end.gpu_window);
  if (end.power.ok) {
    r.power_joules += end.power.mean_w * end.power.span_s;
    r.power_seconds += end.power.span_s;
    r.power_max = std::max(r.power_max, end.power.max_w);
  }
  r.cpu_ms += end.cpu_ms;
  r.active_s += end.duration_ms / 1000.0;
  r.min_avail_mib = std::min(r.min_avail_mib, end.min_avail_mib);
  r.max_commit_pct = std::max(r.max_commit_pct, end.max_commit_pct);
  r.max_pages = std::max(r.max_pages, end.max_pages_per_s);
  if (end.memory.ok)
    r.commit_limit_mib = end.memory.commit_limit_mib;
  r.last_end = end.stamp;
  r.last_memory = end.memory;
}

void NoteChunkBackup(double ms) {
  auto& r = Request();
  std::lock_guard lock(r.mutex);
  if (!r.any)
    return;
  if (r.chunk_n >= 1 && r.chunk_n <= r.chunk_backup.size() && ms >= 0)
    r.chunk_backup[r.chunk_n - 1] = Fmt("%.2f", ms);
  if (ms >= 0) {
    r.backup_total += ms;
    r.backup_max = std::max(r.backup_max, ms);
  }
}

std::string TakeRequestTelemetrySuffix() {
  auto& r = Request();
  auto& faults = Faults();
  std::lock_guard lock(r.mutex);
  const auto hal = faults.hal_faults.load();
  const auto whea = faults.whea_events.load();
  std::string text;
  if (r.any) {
    text = Fmt(
        " t_first_utc=%s t_first_local=%s t_last_end_utc=%s "
        "t_last_end_local=%s idle_gap_ms=%lld avail_mib_first=%s "
        "avail_mib_last=%s commit_mib_last=%s",
        r.first.utc.c_str(), r.first.local.c_str(), r.last_end.utc.c_str(),
        r.last_end.local.c_str(), static_cast<long long>(r.first_idle_gap_ms),
        MemText(r.first_memory, false).c_str(),
        MemText(r.last_memory, false).c_str(),
        MemText(r.last_memory, true).c_str());
    if (r.power_seconds > 0)
      text += Fmt(" chip_power_w_mean=%.1f chip_power_w_max=%.1f",
                  r.power_joules / r.power_seconds, r.power_max);
    else
      text += " chip_power_w_mean=na chip_power_w_max=na";
    text += Fmt(" cpu_ms=%.0f", r.cpu_ms);
    text +=
        Fmt(" commit_limit_mib=%llu min_avail_mib=%llu max_commit_pct=%.1f "
            "pages_per_s_max=%s pages=%s",
            U(r.commit_limit_mib), U(r.min_avail_mib), r.max_commit_pct,
            r.max_pages >= 0 ? Fmt("%.0f", r.max_pages).c_str() : "na",
            PagesStateName());
    text += Fmt(
        " bo_pages_sampled=%s bo_pages_valid_min=%s bo_pages_locked_max=%s "
        "process_ws_mib_first=%s process_ws_mib_min=%s process_ws_mib_last=%s "
        "process_private_mib_max=%s mem_trace=%s",
        NaU(r.trace.have, r.trace.pages).c_str(),
        NaU(r.trace.have, r.trace.valid_min).c_str(),
        NaU(r.trace.have, r.trace.locked_max).c_str(),
        NaU(r.trace.have, r.trace.ws_first).c_str(),
        NaU(r.trace.have, r.trace.ws_min).c_str(),
        NaU(r.trace.have, r.trace.ws_last).c_str(),
        NaU(r.trace.have, r.trace.private_max).c_str(),
        !GlobalMemProbe() ? "off"
        : r.trace.have    ? "on"
                          : "na");
    text += GpuText(r.gpu) + Fmt(" gpu_mem=%s", GpuMemStateName());
    std::string list;
    for (std::size_t i = 0; i < r.chunk_w.size(); ++i)
      list += (i ? "," : "") + r.chunk_w[i];
    text += " chunk_power_w=[" + list + "]" +
            Fmt(" chunk_power_n=%u first_chunk_after_rearm=%d", r.chunk_n,
                r.first_after_rearm ? 1 : 0);
    if (!r.chunk_backup.empty()) {
      std::string backup;
      for (std::size_t i = 0; i < r.chunk_backup.size(); ++i)
        backup += (i ? "," : "") + r.chunk_backup[i];
      std::string times;
      for (std::size_t i = 0; i < r.chunk_ms.size(); ++i)
        times += (i ? "," : "") + r.chunk_ms[i];
      text += " chunk_ms=[" + times + "] chunk_backup_ms=[" + backup + "]" +
              Fmt(" backup_ms_total=%.2f backup_ms_max=%.2f", r.backup_total,
                  r.backup_max);
    }
    text += Fmt(" npu_chunk_gpu_retries=%llu", U(faults.gpu_retries.load()));
  }
  text += Fmt(" hal_faults=%llu whea_events=%llu monitor=%s",
              U(hal - r.hal_faults_taken), U(whea - r.whea_taken),
              MonitorStateName());
  text += Fmt(" chip_power=%s", ChipPowerStateName());
  if (r.any) {  // values of the request that just ended, for the runtime status
    LastRequestSummary last;
    if (r.power_seconds > 0)
      last.power_w_mean = r.power_joules / r.power_seconds;
    last.window_s = r.active_s;
    last.min_available_mib = static_cast<double>(r.min_avail_mib);
    last.max_commit_pct = r.max_commit_pct;
    NoteLastRequest(last);
  }
  r.any = false;
  r.hal_faults_taken = hal;
  r.whea_taken = whea;
  return text;
}
}  // namespace gufo::npu
