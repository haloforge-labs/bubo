// SPDX-License-Identifier: MIT
#include "mem_trace.hpp"

#include <atomic>
#include <mutex>

#include "src/cli/serve/logging.hpp"

#ifdef _WIN32
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <windows.h>
#endif

namespace gufo::npu {
namespace {
std::atomic<IMemProbe*> global_probe{nullptr};
std::mutex global_mutex;
std::unique_ptr<IMemProbe>& Owner() {
  static std::unique_ptr<IMemProbe> owner;
  return owner;
}
std::atomic<bool> failure_logged{false};
std::function<void(const std::string&)>& LogOverride() {
  static std::function<void(const std::string&)> fn;
  return fn;
}
}  // namespace

IMemProbe* GlobalMemProbe() noexcept {
  return global_probe.load(std::memory_order_acquire);
}
void InstallGlobalMemProbe(std::unique_ptr<IMemProbe> probe) {
  std::lock_guard lock(global_mutex);
  global_probe.store(nullptr, std::memory_order_release);
  Owner() = std::move(probe);
  failure_logged.store(false);
  global_probe.store(Owner().get(), std::memory_order_release);
}
void StopGlobalMemProbe() noexcept {
  try {
    std::lock_guard lock(global_mutex);
    global_probe.store(nullptr, std::memory_order_release);
    Owner().reset();
  } catch (...) {
  }
}
void SetMemTraceLogForTest(std::function<void(const std::string&)> log) {
  LogOverride() = std::move(log);
  failure_logged.store(false);
}
bool SampleMemTrace(MemTraceSample* sample) noexcept {
  IMemProbe* probe = GlobalMemProbe();
  if (!probe)
    return false;
  try {
    std::string error;
    *sample = MemTraceSample{};
    if (probe->Sample(sample, &error) && sample->ok)
      return true;
    sample->ok = false;
    if (!failure_logged.exchange(true)) {
      const std::string text =
          "memory trace unavailable (logging only): " +
          (error.empty() ? std::string("probe failed") : error);
      if (LogOverride())
        LogOverride()(text);
      else
        server::Logger::Info("npu", text);
    }
  } catch (...) {
    sample->ok = false;
  }
  return false;
}

#ifdef _WIN32
namespace {
class WindowsMemProbe final : public IMemProbe {
public:
  explicit WindowsMemProbe(PageProvider provider)
      : provider_(std::move(provider)) {
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    if (kernel) {
      info_ = reinterpret_cast<InfoFn>(
          GetProcAddress(kernel, "K32GetProcessMemoryInfo"));
      query_ = reinterpret_cast<QueryFn>(
          GetProcAddress(kernel, "K32QueryWorkingSetEx"));
    }
  }
  bool Sample(MemTraceSample* out, std::string* error) override {
    if (!info_ || !query_) {
      *error = "kernel32 lacks K32GetProcessMemoryInfo / K32QueryWorkingSetEx";
      return false;
    }
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!info_(GetCurrentProcess(),
               reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),
               sizeof(counters))) {
      *error = "GetProcessMemoryInfo failed, error " +
               std::to_string(GetLastError());
      return false;
    }
    out->ws_mib = counters.WorkingSetSize >> 20;
    out->private_mib = counters.PrivateUsage >> 20;
    if (pages_.empty() && provider_) {
      // Same addresses for the whole process once the list exists.
      const auto list = provider_();
      for (const void* page : list)
        pages_.push_back({const_cast<void*>(page), {}});
    }
    out->pages_sampled = static_cast<unsigned>(pages_.size());
    if (!pages_.empty()) {
      for (auto& p : pages_)
        p.VirtualAttributes.Flags = 0;
      if (!query_(GetCurrentProcess(), pages_.data(),
                  static_cast<DWORD>(pages_.size() * sizeof(pages_[0])))) {
        *error =
            "QueryWorkingSetEx failed, error " + std::to_string(GetLastError());
        return false;
      }
      for (const auto& p : pages_) {
        const auto& a = p.VirtualAttributes;
        // Shared/Locked/LargePage only mean something for a valid page: for an
        // invalid page the same bits are other fields (Invalid layout).
        if (!a.Valid)
          continue;
        ++out->valid;
        out->shared += a.Shared ? 1U : 0U;
        out->locked += a.Locked ? 1U : 0U;
        out->large += a.LargePage ? 1U : 0U;
      }
    }
    out->ok = true;
    return true;
  }

private:
  using InfoFn = BOOL(WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
  using QueryFn = BOOL(WINAPI*)(HANDLE, PVOID, DWORD);
  PageProvider provider_;
  InfoFn info_ = nullptr;
  QueryFn query_ = nullptr;
  std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages_;
};

// PDH "GPU Process Memory" (our pid) and "GPU Adapter Memory" counters. The
// per-process instances appear only after the process uses the GPU, so the
// counters are added again when none were found and every 30 reads (instance
// names change). Sampler thread only.
class PdhGpuMemSource final : public IGpuMemSource {
public:
  ~PdhGpuMemSource() override { Close(); }
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
    array_ = reinterpret_cast<decltype(&PdhGetFormattedCounterArrayW)>(
        GetProcAddress(library_, "PdhGetFormattedCounterArrayW"));
    close_ = reinterpret_cast<decltype(&PdhCloseQuery)>(
        GetProcAddress(library_, "PdhCloseQuery"));
    if (!open_ || !add_ || !collect_ || !array_ || !close_) {
      *error = "pdh.dll lacks the counter functions";
      return false;
    }
    if (!Build()) {
      *error = "GPU memory counter sets not found";
      return false;
    }
    return true;
  }
  bool Read(GpuMemPoint* point, std::string* error) override {
    if (!query_) {
      *error = "no query";
      return false;
    }
    if (reads_ % 30 == 0 && reads_ != 0)
      Rebuild();
    ++reads_;
    if (collect_(query_) != ERROR_SUCCESS) {
      *error = "PdhCollectQueryData failed";
      return false;
    }
    double values[3] = {0, 0, 0};
    bool proc = true;
    for (int i = 0; i < 3; ++i) {
      double sum = 0;
      if (!process_[i] || !Sum(process_[i], &sum))
        proc = false;
      values[i] = sum;
    }
    double adapter = 0;
    point->adapter_ok = adapter_ && Sum(adapter_, &adapter);
    point->adapter_shared_mib = adapter / 1048576.0;
    point->proc_ok = proc;
    point->proc_local_mib = values[0] / 1048576.0;
    point->proc_dedicated_mib = values[1] / 1048576.0;
    point->proc_shared_mib = values[2] / 1048576.0;
    if (!proc && !instances_found_)
      Rebuild();  // instances appear after the first GPU use
    return point->proc_ok || point->adapter_ok;
  }
  void Close() noexcept override {
    if (query_ && close_)
      close_(query_);
    query_ = nullptr;
  }

private:
  bool Build() {
    if (open_(nullptr, 0, &query_) != ERROR_SUCCESS) {
      query_ = nullptr;
      return false;
    }
    static const wchar_t* kinds[3] = {L"Local Usage", L"Dedicated Usage",
                                      L"Shared Usage"};
    int added = 0;
    for (int i = 0; i < 3; ++i) {
      std::wstring path = L"\\GPU Process Memory(pid_" +
                          std::to_wstring(GetCurrentProcessId()) + L"*)\\" +
                          kinds[i];
      process_[i] = nullptr;
      if (add_(query_, path.c_str(), 0, &process_[i]) == ERROR_SUCCESS)
        ++added;
      else
        process_[i] = nullptr;
    }
    adapter_ = nullptr;
    if (add_(query_, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &adapter_) !=
        ERROR_SUCCESS)
      adapter_ = nullptr;
    else
      ++added;
    collect_(query_);  // priming
    return added > 0;
  }
  void Rebuild() {
    Close();
    instances_found_ = false;
    Build();
  }
  // Sum of the valid instances of a wildcard counter (bytes).
  bool Sum(PDH_HCOUNTER counter, double* sum) {
    DWORD bytes = 0, count = 0;
    PDH_STATUS status = array_(counter, PDH_FMT_LARGE, &bytes, &count, nullptr);
    if (status != static_cast<PDH_STATUS>(PDH_MORE_DATA) || bytes == 0)
      return false;
    std::vector<char> buffer(bytes);
    auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
    status = array_(counter, PDH_FMT_LARGE, &bytes, &count, items);
    if (status != ERROR_SUCCESS || count == 0)
      return false;
    bool any = false;
    for (DWORD i = 0; i < count; ++i) {
      if (items[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA ||
          items[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) {
        *sum += static_cast<double>(items[i].FmtValue.largeValue);
        any = true;
      }
    }
    if (any)
      instances_found_ = true;
    return any;
  }
  HMODULE library_ = nullptr;
  decltype(&PdhOpenQueryW) open_ = nullptr;
  decltype(&PdhAddEnglishCounterW) add_ = nullptr;
  decltype(&PdhCollectQueryData) collect_ = nullptr;
  decltype(&PdhGetFormattedCounterArrayW) array_ = nullptr;
  decltype(&PdhCloseQuery) close_ = nullptr;
  PDH_HQUERY query_ = nullptr;
  PDH_HCOUNTER process_[3] = {nullptr, nullptr, nullptr};
  PDH_HCOUNTER adapter_ = nullptr;
  unsigned reads_ = 0;
  bool instances_found_ = false;
};
}  // namespace
std::unique_ptr<IMemProbe> MakeWindowsMemProbe(PageProvider provider) {
  return std::make_unique<WindowsMemProbe>(std::move(provider));
}
std::unique_ptr<IGpuMemSource> MakeWindowsGpuMemSource() {
  return std::make_unique<PdhGpuMemSource>();
}
#else
std::unique_ptr<IMemProbe> MakeWindowsMemProbe(PageProvider) {
  return nullptr;
}
std::unique_ptr<IGpuMemSource> MakeWindowsGpuMemSource() {
  return nullptr;
}
#endif
}  // namespace gufo::npu
