// SPDX-License-Identifier: MIT
#include "hal_fault_monitor.hpp"

#include <algorithm>
#include <cctype>
#include <thread>

#include "bo_address_table.hpp"
#include "fault_telemetry.hpp"
#include "src/cli/serve/logging.hpp"

#ifdef _WIN32
#include <windows.h>
#include <winevt.h>
#endif

namespace gufo::npu {
namespace {
std::string Lower(std::string_view text) {
  std::string out(text);
  for (auto& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}
std::string Unescape(std::string_view text) {
  static constexpr std::pair<std::string_view, char> table[] = {
      {"&amp;", '&'},
      {"&lt;", '<'},
      {"&gt;", '>'},
      {"&quot;", '"'},
      {"&apos;", '\''}};
  std::string out;
  for (std::size_t i = 0; i < text.size();) {
    bool replaced = false;
    if (text[i] == '&') {
      for (const auto& [name, ch] : table) {
        if (text.substr(i, name.size()) == name) {
          out += ch;
          i += name.size();
          replaced = true;
          break;
        }
      }
    }
    if (!replaced)
      out += text[i++];
  }
  return out;
}
// Value of attr='..' or attr=".." inside one tag's attribute text.
std::string Attribute(std::string_view tag, std::string_view attr) {
  for (std::size_t at = 0; (at = tag.find(attr, at)) != std::string_view::npos;
       at += attr.size()) {
    if (at > 0 && !std::isspace(static_cast<unsigned char>(tag[at - 1])))
      continue;
    std::size_t i = at + attr.size();
    while (i < tag.size() && std::isspace(static_cast<unsigned char>(tag[i])))
      ++i;
    if (i >= tag.size() || tag[i] != '=')
      continue;
    ++i;
    while (i < tag.size() && std::isspace(static_cast<unsigned char>(tag[i])))
      ++i;
    if (i >= tag.size() || (tag[i] != '\'' && tag[i] != '"'))
      continue;
    const char quote = tag[i++];
    const auto end = tag.find(quote, i);
    if (end == std::string_view::npos)
      return {};
    return Unescape(tag.substr(i, end - i));
  }
  return {};
}
// Text of the first "<name ...>" (attributes only) at or after from.
std::string_view OpenTag(std::string_view xml, std::string_view name,
                         std::size_t from, std::size_t* after) {
  for (std::size_t at = from;
       (at = xml.find(name, at)) != std::string_view::npos; at += name.size()) {
    if (at == 0 || xml[at - 1] != '<')
      continue;
    const std::size_t end = at + name.size();
    if (end < xml.size() &&
        !std::isspace(static_cast<unsigned char>(xml[end])) &&
        xml[end] != '>' && xml[end] != '/')
      continue;  // longer tag name
    const auto close = xml.find('>', end);
    if (close == std::string_view::npos)
      return {};
    if (after)
      *after = close + 1;
    return xml.substr(end, close - end);
  }
  return {};
}
}  // namespace

bool ParseEventXml(std::string_view xml, HalFaultEvent* event) {
  std::size_t after = 0;
  const auto provider = OpenTag(xml, "Provider", 0, &after);
  event->provider = Attribute(provider, "Name");
  const auto time = OpenTag(xml, "TimeCreated", 0, &after);
  event->event_time = Attribute(time, "SystemTime");
  const auto id_tag = OpenTag(xml, "EventID", 0, &after);
  if (id_tag.data() == nullptr)
    return false;
  const auto id_end = xml.find('<', after);
  try {
    event->id = std::stoi(std::string(xml.substr(after, id_end - after)));
  } catch (...) {
    return false;
  }
  // EventData items: <Data Name='X'>value</Data>, or unnamed
  // <Data>value</Data>.
  std::size_t data_from = xml.find("<EventData");
  if (data_from == std::string_view::npos)
    data_from = xml.find("<UserData");
  event->data.clear();
  event->reason.clear();
  event->info.clear();
  event->device.clear();
  if (data_from == std::string_view::npos)
    return true;
  unsigned index = 0;
  for (std::size_t from = data_from;;) {
    const auto tag = OpenTag(xml, "Data", from, &after);
    if (tag.data() == nullptr)
      break;
    std::string value;
    if (tag.empty() || tag.back() != '/') {
      const auto close = xml.find("</Data>", after);
      if (close == std::string_view::npos)
        break;
      value = Unescape(xml.substr(after, close - after));
      from = close + 7;
    } else {
      from = after;
    }
    const auto name = Attribute(tag, "Name");
    const auto key = Lower(name);
    if (!event->data.empty())
      event->data += ' ';
    event->data +=
        (name.empty() ? "item" + std::to_string(index) : name) + "=" + value;
    ++index;
    if (key.find("reason") != std::string::npos)
      event->reason = value;
    else if (key.find("information") != std::string::npos ||
             key.find("address") != std::string::npos)
      event->info = value;
    else if (key.find("device") != std::string::npos ||
             key.find("bdf") != std::string::npos)
      event->device = value;
  }
  return true;
}

std::string FormatHalEvent(const HalFaultEvent& event) {
  const auto field = [](const std::string& value) {
    return value.empty() ? std::string("?") : value;
  };
  std::string data = event.data.substr(0, 300);
  if (event.kind == "HAL")
    return "HAL-fault event: " + field(event.event_time) +
           " id=" + std::to_string(event.id) +
           " reason=" + field(event.reason) + " info=" + field(event.info) +
           " device=" + field(event.device) + " recv=" + event.received_utc +
           " seq=" + std::to_string(event.seq) +
           (event.resolves.empty() ? std::string() : " " + event.resolves) +
           (event.reason.empty() || event.info.empty() ? " data=[" + data + "]"
                                                       : std::string());
  return "WHEA event: " + field(event.event_time) +
         " id=" + std::to_string(event.id) + " provider=" + event.provider +
         " recv=" + event.received_utc + " seq=" + std::to_string(event.seq) +
         " data=[" + data + "]";
}

HalFaultMonitor::HalFaultMonitor(std::unique_ptr<IHalFaultSource> source,
                                 FaultState& state, Log log,
                                 std::function<std::int64_t()> now_ms)
    : source_(std::move(source)),
      state_(state),
      log_(std::move(log)),
      now_ms_(std::move(now_ms)) {
  if (!now_ms_)
    now_ms_ = [] { return SteadyNowNs() / 1000000; };
}
HalFaultMonitor::~HalFaultMonitor() {
  Stop();
}
bool HalFaultMonitor::Start(bool enabled) {
  if (!enabled || !source_ || running_) {
    if (!enabled)
      state_.monitor_state.store(0, std::memory_order_relaxed);
    return running_;
  }
  // Detection only: nothing in here may throw into the caller.
  try {
    {
      std::lock_guard lock(mutex_);
      baseline_ = next_seq_;
    }
    source_->SetBrokenHandler([this](const std::string& why) {
      if (broken_.exchange(true))
        return;  // log once
      state_.monitor_state.store(2, std::memory_order_relaxed);
      if (log_)
        log_(true, "HAL fault monitor stopped (monitor=broken): " + why);
    });
    std::string error;
    if (!source_->Start([this](HalFaultEvent event) { Push(std::move(event)); },
                        &error)) {
      start_failed_ = true;  // logged once, here; the monitor stays a no-op
      state_.monitor_state.store(3, std::memory_order_relaxed);
      if (log_)
        log_(false, "HAL fault monitor unavailable: " + error);
      return false;
    }
  } catch (...) {
    start_failed_ = true;
    state_.monitor_state.store(3, std::memory_order_relaxed);
    try {
      if (log_)
        log_(false, "HAL fault monitor unavailable: exception while starting");
    } catch (...) {
    }
    return false;
  }
  running_ = true;
  state_.monitor_state.store(1, std::memory_order_relaxed);
  if (log_)
    log_(false, "HAL fault monitor on (detection only)");
  return true;
}
void HalFaultMonitor::Stop() noexcept {
  if (source_)
    source_->Stop();  // joins the delivery thread before this returns
  running_ = false;
}
void HalFaultMonitor::Push(HalFaultEvent event) {
  try {
    if (event.kind.empty())
      event.kind =
          event.provider.find("WHEA") != std::string::npos ? "WHEA" : "HAL";
    event.received_utc = FormatWallStamp(std::chrono::system_clock::now()).utc;
    if (event.kind == "HAL") {
      // Which buffer did the IOMMU fault hit? Fail-open: any problem = na.
      try {
        const auto address = ParseFaultAddress(event.info);
        const auto table = CurrentBoTable();
        event.resolves =
            !address || !table ? "resolves=na" : ResolveText(*table, *address);
        NoteHalResolution(event.resolves);
      } catch (...) {
        event.resolves = "resolves=na";
      }
    }
    bool do_log = false;
    std::uint64_t reported_suppressed = 0;
    {
      std::lock_guard lock(mutex_);
      event.seq = ++next_seq_;
      ring_[event.seq % kRing] = event;
      // Log rate limit: at most kMaxLogPerMinute lines per minute. Events are
      // still stored and counted; the next window reports how many were
      // skipped.
      const auto now = now_ms_();
      if (window_logged_ == 0 || now - window_start_ms_ >= 60000) {
        reported_suppressed = suppressed_;
        suppressed_ = 0;
        window_start_ms_ = now;
        window_logged_ = 0;
      }
      if (window_logged_ < kMaxLogPerMinute) {
        ++window_logged_;
        do_log = true;
      } else {
        ++suppressed_;
      }
    }
    const bool hal = event.kind == "HAL";
    if (hal)
      state_.hal_faults.fetch_add(1, std::memory_order_relaxed);
    else
      state_.whea_events.fetch_add(1, std::memory_order_relaxed);
    if (log_) {
      if (reported_suppressed)
        log_(true, "suppressed " + std::to_string(reported_suppressed) +
                       " HAL/WHEA events in the previous minute (log limit " +
                       std::to_string(kMaxLogPerMinute) +
                       " per minute; counters and ring hold them)");
      if (do_log)
        log_(true, FormatHalEvent(event));
    }
    // Publish last: a reader that sees the new sequence sees the ring entry.
    // Only HAL events move the sequence the chunk code reads.
    if (hal)
      state_.hal_seq.fetch_add(1, std::memory_order_release);  // one producer
  } catch (...) {
  }
}
std::uint64_t HalFaultMonitor::ring_sequence() const {
  std::lock_guard lock(mutex_);
  return next_seq_;
}
std::vector<HalFaultEvent> HalFaultMonitor::EventsSince(
    std::uint64_t after) const {
  std::lock_guard lock(mutex_);
  const auto last = next_seq_;
  std::vector<HalFaultEvent> out;
  const auto first =
      std::max<std::uint64_t>(after + 1, last > kRing ? last - kRing + 1 : 1);
  for (auto seq = first; seq <= last; ++seq)
    if (ring_[seq % kRing].seq == seq)
      out.push_back(ring_[seq % kRing]);
  return out;
}

namespace {
std::mutex global_mutex;
std::unique_ptr<HalFaultMonitor>& Global() {
  static std::unique_ptr<HalFaultMonitor> monitor;
  return monitor;
}
void DefaultLog(bool warn, const std::string& text) {
  server::Logger::Log(warn ? server::LogLevel::kWarn : server::LogLevel::kInfo,
                      "npu", text);
}
}  // namespace
HalFaultMonitor* GlobalHalMonitor() {
  std::lock_guard lock(global_mutex);
  return Global().get();
}
HalFaultMonitor& InstallGlobalHalMonitor(
    std::unique_ptr<IHalFaultSource> source, bool enabled) {
  std::unique_ptr<HalFaultMonitor> old;
  std::lock_guard lock(global_mutex);
  old = std::move(Global());
  if (old)
    old->Stop();
  Global() = std::make_unique<HalFaultMonitor>(std::move(source), Faults(),
                                               DefaultLog);
  Global()->Start(enabled);
  return *Global();
}
void StopGlobalHalMonitor() noexcept {
  try {
    std::lock_guard lock(global_mutex);
    if (Global())
      Global()->Stop();
  } catch (...) {
  }
}

#ifdef _WIN32
namespace {
// wevtapi.dll is loaded at run time: a missing or blocked library makes
// Start() fail quietly instead of breaking the build or the server.
class WindowsHalFaultSource final : public IHalFaultSource {
public:
  ~WindowsHalFaultSource() override { Stop(); }
  bool Start(std::function<void(HalFaultEvent)> sink,
             std::string* error) override {
    sink_ = std::move(sink);
    library_ =
        LoadLibraryExW(L"wevtapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library_) {
      *error =
          "wevtapi.dll not loadable, error " + std::to_string(GetLastError());
      return false;
    }
    subscribe_ = reinterpret_cast<decltype(&EvtSubscribe)>(
        GetProcAddress(library_, "EvtSubscribe"));
    next_ = reinterpret_cast<decltype(&EvtNext)>(
        GetProcAddress(library_, "EvtNext"));
    render_ = reinterpret_cast<decltype(&EvtRender)>(
        GetProcAddress(library_, "EvtRender"));
    close_ = reinterpret_cast<decltype(&EvtClose)>(
        GetProcAddress(library_, "EvtClose"));
    if (!subscribe_ || !next_ || !render_ || !close_) {
      *error = "wevtapi.dll lacks the event subscription functions";
      return false;
    }
    signal_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    stop_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!signal_ || !stop_) {
      *error = "CreateEvent failed, error " + std::to_string(GetLastError());
      return false;
    }
    // New events only (no replay): HAL IOMMU fault Id 15 and any WHEA-Logger
    // event, from the System channel.
    static constexpr wchar_t query[] =
        L"*[System[(Provider[@Name='Microsoft-Windows-HAL'] and (EventID=15)) "
        L"or Provider[@Name='Microsoft-Windows-WHEA-Logger']]]";
    subscription_ = subscribe_(nullptr, signal_, L"System", query, nullptr,
                               nullptr, nullptr, EvtSubscribeToFutureEvents);
    if (!subscription_) {
      *error = "EvtSubscribe failed, error " + std::to_string(GetLastError());
      return false;
    }
    try {
      thread_ = std::thread([this] { Run(); });
    } catch (...) {
      Stop();  // close the subscription and handles; no thread exists
      *error = "could not create the event thread";
      return false;
    }
    return true;
  }
  void SetBrokenHandler(std::function<void(const std::string&)> h) override {
    broken_ = std::move(h);
  }
  bool Running() const noexcept override { return thread_.joinable(); }
  void Stop() noexcept override {
    if (stop_)
      SetEvent(stop_);
    if (thread_.joinable())
      thread_.join();  // the thread is gone before the handles close
    if (subscription_ && close_)
      close_(subscription_);
    subscription_ = nullptr;
    if (signal_)
      CloseHandle(signal_);
    if (stop_)
      CloseHandle(stop_);
    signal_ = stop_ = nullptr;
  }

private:
  void Break(const std::string& why) noexcept {
    try {
      if (broken_)
        broken_(why);
    } catch (...) {
    }
  }
  void Run() noexcept {
    try {
      RunLoop();
    } catch (...) {
      Break("exception in the event thread");
    }
  }
  void RunLoop() {
    const HANDLE wait[2] = {signal_, stop_};
    bool first = true;  // drain once before the first wait (no missed signal)
    for (;;) {
      if (!first) {
        const auto result = WaitForMultipleObjects(2, wait, FALSE, INFINITE);
        if (result == WAIT_OBJECT_0 + 1)
          return;  // Stop() requested
        if (result != WAIT_OBJECT_0) {
          Break("wait failed, error " + std::to_string(GetLastError()));
          return;
        }
      }
      first = false;
      for (;;) {
        EVT_HANDLE events[16];
        DWORD returned = 0;
        if (!next_(subscription_, 16, events, 0, 0, &returned)) {
          const DWORD error = GetLastError();
          if (error != ERROR_NO_MORE_ITEMS && error != ERROR_TIMEOUT) {
            Break("EvtNext failed, error " + std::to_string(error));
            return;  // subscription broke
          }
          break;
        }
        for (DWORD i = 0; i < returned; ++i) {
          Deliver(events[i]);
          close_(events[i]);
        }
      }
    }
  }
  void Deliver(EVT_HANDLE handle) noexcept {
    try {
      std::vector<wchar_t> buffer(8192);
      DWORD used = 0, count = 0;
      if (!render_(nullptr, handle, EvtRenderEventXml,
                   static_cast<DWORD>(buffer.size() * sizeof(wchar_t)),
                   buffer.data(), &used, &count)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
          return;
        buffer.resize(used / sizeof(wchar_t) + 1);
        if (!render_(nullptr, handle, EvtRenderEventXml,
                     static_cast<DWORD>(buffer.size() * sizeof(wchar_t)),
                     buffer.data(), &used, &count))
          return;
      }
      const int bytes = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1,
                                            nullptr, 0, nullptr, nullptr);
      if (bytes <= 1)
        return;
      std::string xml(static_cast<std::size_t>(bytes), '\0');
      WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, xml.data(), bytes,
                          nullptr, nullptr);
      xml.resize(static_cast<std::size_t>(bytes) - 1);
      HalFaultEvent event;
      if (ParseEventXml(xml, &event) && sink_)
        sink_(std::move(event));
    } catch (...) {
    }
  }
  std::function<void(HalFaultEvent)> sink_;
  std::function<void(const std::string&)> broken_;
  HMODULE library_ = nullptr;
  decltype(&EvtSubscribe) subscribe_ = nullptr;
  decltype(&EvtNext) next_ = nullptr;
  decltype(&EvtRender) render_ = nullptr;
  decltype(&EvtClose) close_ = nullptr;
  EVT_HANDLE subscription_ = nullptr;
  HANDLE signal_ = nullptr, stop_ = nullptr;
  std::thread thread_;
};
}  // namespace
std::unique_ptr<IHalFaultSource> MakeWindowsHalFaultSource() {
  return std::make_unique<WindowsHalFaultSource>();
}
#else
std::unique_ptr<IHalFaultSource> MakeWindowsHalFaultSource() {
  return nullptr;
}
#endif
}  // namespace gufo::npu
