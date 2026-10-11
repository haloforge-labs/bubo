#include "src/cli/serve/logging.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "src/cli/serve/logging_sink.hpp"

namespace gufo::server {
namespace {

void WriteStderr(std::string_view batch) {
  // One stream write and flush per batch, to the same stderr stream buffer as
  // before. This also preserves existing CPU tests that capture clog.
  std::clog.write(batch.data(), static_cast<std::streamsize>(batch.size()));
  std::clog.flush();
  if (!std::clog.good())
    throw std::runtime_error("stderr write failed");
}

AsyncLogSink& ProductionSink() {
  // Intentionally retain state through CRT teardown, including a stuck writer.
  static auto* sink = [] {
    const char* sync = std::getenv("GUFO_LOG_SYNC");
    auto* result =
        new AsyncLogSink(WriteStderr, sync && std::string_view(sync) == "1");
    std::atexit([] { Logger::Shutdown(); });
    return result;
  }();
  return *sink;
}

#ifdef GUFO_LOG_TESTING
AsyncLogSink* test_sink = nullptr;
#endif

AsyncLogSink& Sink() {
#ifdef GUFO_LOG_TESTING
  if (test_sink)
    return *test_sink;
#endif
  return ProductionSink();
}

std::string FormatRaw(const char* format, va_list args) {
  va_list copy;
  va_copy(copy, args);
  const int size = std::vsnprintf(nullptr, 0, format, copy);
  va_end(copy);
  if (size < 0)
    return {};
  std::string line(static_cast<std::size_t>(size) + 1, '\0');
  std::vsnprintf(line.data(), line.size(), format, args);
  line.resize(static_cast<std::size_t>(size));
  return line;
}

// Process-wide verbosity threshold. Connection workers, the text scheduler and
// video workers all log through Logger, so the filter has to be shared state
// that is safe to read while another thread is still starting up.
std::atomic<LogLevel>& LevelStorage() {
  static std::atomic<LogLevel> level{LogLevel::kInfo};
  return level;
}

std::string SafeLine(std::string_view input, std::size_t limit = 8192) {
  std::string output;
  for (const unsigned char c : input.substr(0, limit)) {
    if (c < 0x20 || c == 0x7f) {
      constexpr char hex[] = "0123456789abcdef";
      output += "\\x";
      output += hex[c >> 4];
      output += hex[c & 15];
    } else {
      output += static_cast<char>(c);
    }
  }
  if (input.size() > limit)
    output += "...";
  return output;
}

std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm calendar{};
  ::localtime_r(&seconds, &calendar);
  std::array<char, 32> buffer{};
  const auto written = std::strftime(buffer.data(), buffer.size(),
                                     "%Y-%m-%d %H:%M:%S", &calendar);
  return {buffer.data(), written};
}

[[maybe_unused]] std::size_t ReadMemoryKiB(const char* path,
                                           std::string_view field) {
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    if (line.starts_with(field)) {
      std::istringstream value(line.substr(field.size()));
      std::size_t kib = 0;
      if (value >> kib)
        return kib;
    }
  }
  return 0;
}

}  // namespace

struct AsyncLogSink::State {
  explicit State(Writer output, bool sync)
      : writer(std::move(output)), synchronous(sync) {}
  struct Entry {
    std::string line;
    std::uint64_t sequence;
  };
  // Charge the entry and deque bookkeeping as well as its bytes. Otherwise
  // empty/tiny protected messages could grow without reaching the hard cap.
  static constexpr std::size_t kEntryCharge = sizeof(Entry) + 24;
  Writer writer;
  const bool synchronous;
  std::mutex mutex, io_mutex;
  std::condition_variable wake, changed;
  std::deque<Entry> queue;
  std::uint64_t accepted = 0, delivered = 0, dropped = 0, reported = 0;
  std::uint64_t active_lines = 0;
  std::size_t bytes = 0;
  bool stopping = false, done = false, failed = false;
  std::string timeout_report;

  void Run() {
    std::unique_lock lock(mutex);
    for (;;) {
      wake.wait(lock, [&] {
        return stopping || !queue.empty() || dropped != reported;
      });
      if (queue.empty() && dropped == reported && timeout_report.empty() &&
          stopping) {
        done = true;
        changed.notify_all();
        return;
      }
      std::string batch;
      batch.reserve(65536);
      std::uint64_t sequence = delivered, count = 0;
      std::size_t consumed = 0;
      // Cap fragments as well as bytes. Start sparse or burst batches as soon
      // as the writer is free, then flush each batch without a 100 ms delay.
      while (!queue.empty() && count < 64 && batch.size() < 65536) {
        auto& entry = queue.front();
        batch += entry.line;
        consumed += entry.line.size() + kEntryCharge;
        sequence = entry.sequence;
        queue.pop_front();
        ++count;
      }
      const auto drop_target = dropped;
      if (drop_target != reported)
        batch += "[log] dropped " + std::to_string(drop_target - reported) +
                 " lines\n";
      if (!timeout_report.empty()) {
        batch += timeout_report;
        timeout_report.clear();
      }
      active_lines = count;
      lock.unlock();
      bool ok = true;
      try {
        const std::lock_guard io_lock(io_mutex);
        writer(batch);
      } catch (...) {
        ok = false;
      }
      lock.lock();
      bytes -= consumed;
      active_lines = 0;
      if (ok) {
        delivered = sequence;
        reported = drop_target;
      } else {
        // Stop after a sink error: retrying a broken descriptor would spin.
        failed = done = true;
        changed.notify_all();
        return;
      }
      changed.notify_all();
    }
  }
};

AsyncLogSink::AsyncLogSink(Writer writer, bool synchronous)
    : state_(std::make_shared<State>(std::move(writer), synchronous)) {
  if (!synchronous)
    std::thread([state = state_] {
#ifdef _WIN32
      (void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
#else
      sched_param priority{};
      (void)pthread_setschedparam(pthread_self(), SCHED_OTHER, &priority);
#endif
      state->Run();
    }).detach();
}

AsyncLogSink::~AsyncLogSink() {
  (void)Shutdown();
}

void AsyncLogSink::Push(std::string line, LogLevel level) {
  const auto state = state_;
  if (state->synchronous) {
    const std::lock_guard io_lock(state->io_mutex);
    state->writer(line);
    const std::lock_guard lock(state->mutex);
    ++state->accepted;
    ++state->delivered;
    return;
  }
  std::unique_lock lock(state->mutex);
  const auto lines = state->queue.size() + state->active_lines;
  const auto charge = line.size() > 16 * 1024 * 1024
                          ? 16 * 1024 * 1024 + 1
                          : line.size() + State::kEntryCharge;
  if (level < LogLevel::kWarn &&
      (state->stopping || state->failed || lines >= 16384 ||
       charge > 4 * 1024 * 1024 || state->bytes > 4 * 1024 * 1024 - charge)) {
    ++state->dropped;
    state->wake.notify_one();
    return;
  }
  if (state->stopping || state->failed || charge > 16 * 1024 * 1024 ||
      state->bytes > 16 * 1024 * 1024 - charge) {
    // Only WARN/ERROR can reach this deliberate last-resort synchronous path.
    // Preserve the already accepted prefix before writing this fatal message.
    lock.unlock();
    (void)FlushSync();
    const std::lock_guard io_lock(state->io_mutex);
    state->writer(line);
    return;
  }
  state->bytes += charge;
  state->queue.push_back({std::move(line), ++state->accepted});
  state->wake.notify_one();
}

bool AsyncLogSink::FlushSync(std::chrono::milliseconds timeout) {
  const auto state = state_;
  std::unique_lock lock(state->mutex);
  const auto target = state->accepted, drop_target = state->dropped;
  state->wake.notify_one();
  const auto ready = [&] {
    return state->failed ||
           (state->delivered >= target && state->reported >= drop_target);
  };
  if (timeout == std::chrono::milliseconds::max())
    state->changed.wait(lock, ready);
  else if (!state->changed.wait_for(lock, timeout, ready))
    return false;
  return !state->failed;
}

AsyncLogSink::Stats AsyncLogSink::GetStats() const {
  const std::lock_guard lock(state_->mutex);
  return {state_->accepted,
          state_->delivered,
          state_->dropped,
          state_->reported,
          state_->accepted - state_->delivered,
          state_->bytes};
}

AsyncLogSink::Stats AsyncLogSink::Shutdown() {
  const auto state = state_;
  std::unique_lock lock(state->mutex);
  if (!state->stopping) {
    // Reserve 50 ms of the 2 s exit budget for accounting and the loss report.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1950);
    state->stopping = true;
    state->wake.notify_one();
    if (!state->synchronous)
      state->changed.wait_until(lock, deadline, [&] { return state->done; });
    const auto pending = state->accepted - state->delivered;
    if (pending) {
      state->timeout_report = "[log] shutdown: undelivered " +
                              std::to_string(pending) +
                              " lines (2 s shutdown budget)\n";
#ifdef _WIN32
      // A blocked stderr cannot receive a report within the deadline. Also
      // expose it to the Windows debugger; the writer emits it if it resumes.
      OutputDebugStringA(state->timeout_report.c_str());
#endif
    }
  }
  return {state->accepted,
          state->delivered,
          state->dropped,
          state->reported,
          state->accepted - state->delivered,
          state->bytes};
}

void Logger::Raw(const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto line = FormatRaw(format, args);
  va_end(args);
  Sink().Push(std::move(line), LogLevel::kError);
}

void Logger::Raw(LogLevel level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto line = FormatRaw(format, args);
  va_end(args);
  Sink().Push(std::move(line), level);
}

bool Logger::FlushSync(std::chrono::milliseconds timeout) {
  return Sink().FlushSync(timeout);
}

void Logger::Shutdown() {
  (void)Sink().Shutdown();
}

#ifdef GUFO_LOG_TESTING
void Logger::SetSinkForTesting(AsyncLogSink* sink) {
  test_sink = sink;
}
#endif

std::optional<LogLevel> LogLevelFromName(std::string_view name) {
  if (name == "debug")
    return LogLevel::kDebug;
  if (name == "info")
    return LogLevel::kInfo;
  if (name == "warn")
    return LogLevel::kWarn;
  if (name == "error")
    return LogLevel::kError;
  return std::nullopt;
}

std::string_view LogLevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kError:
      return "error";
  }
  return "info";
}

void Logger::SetLevel(LogLevel level) {
  LevelStorage().store(level, std::memory_order_relaxed);
}

LogLevel Logger::Level() {
  return LevelStorage().load(std::memory_order_relaxed);
}

bool Logger::Enabled(LogLevel level) {
  return level >= Level();
}

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view message) {
  if (!Enabled(level)) {
    return;
  }
  static const bool color =
      std::getenv("NO_COLOR") == nullptr && ::isatty(STDERR_FILENO) != 0;
  const char* tag = level == LogLevel::kError   ? "ERROR"
                    : level == LogLevel::kWarn  ? "WARN"
                    : level == LogLevel::kDebug ? "DEBUG"
                                                : "INFO";
  const char* tint = level == LogLevel::kError   ? "\033[31m"
                     : level == LogLevel::kWarn  ? "\033[33m"
                     : level == LogLevel::kDebug ? "\033[2;36m"
                                                 : "\033[32m";
  std::ostringstream output;
  output << CurrentTimestamp() << ' ';
  if (color)
    output << tint;
  output << '[' << tag << ']';
  if (color)
    output << "\033[0m";
  output << " [" << SafeLine(component, 64) << "] " << SafeLine(message)
         << '\n';
  Sink().Push(output.str(), level);
}

void Logger::Diagnostic(std::string_view component, std::string_view message) {
  std::ostringstream output;
  output << CurrentTimestamp() << " [INFO] [" << SafeLine(component, 64) << "] "
         << SafeLine(message, message.size()) << '\n';
  Sink().Push(output.str(), LogLevel::kInfo);
}

void Logger::LogFormatted(LogLevel level, std::string_view component,
                          const char* format, ...) {
  if (!Enabled(level)) {
    return;
  }
  // A fixed buffer is enough for these process-authored lines and avoids an
  // allocation that a filtered call site would otherwise pay for. vsnprintf
  // reports the untruncated length, so clamp it to what was written.
  char buffer[1024];
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  const std::size_t length =
      written < 0 ? 0
                  : std::min<std::size_t>(static_cast<std::size_t>(written),
                                          sizeof(buffer) - 1);
  Log(level, component, std::string_view(buffer, length));
}

std::string Logger::MemoryStatus() {
  std::ostringstream output;
#ifdef _WIN32
  output << "rss_mib=" << gufo_resident_bytes() / (1024 * 1024)
         << " host_available_mib="
         << gufo_available_physical_bytes() / (1024 * 1024);
#else
  output << "rss_mib=" << ReadMemoryKiB("/proc/self/status", "VmRSS:") / 1024
         << " host_available_mib="
         << ReadMemoryKiB("/proc/meminfo", "MemAvailable:") / 1024;
#endif
  return output.str();
}

void Logger::LogRequest(std::string_view id, std::string_view method,
                        std::string_view path, int status_code,
                        double duration_ms, std::string_view details,
                        std::string_view outcome, LogLevel success_level) {
  // Escalation stays upward-only: a debug-tier poll that failed still logs at
  // WARN/ERROR under the default threshold.
  const LogLevel level =
      status_code >= 500 || outcome == "stream_error"   ? LogLevel::kError
      : status_code >= 400 || outcome == "disconnected" ? LogLevel::kWarn
                                                        : success_level;
  // A filtered line must not pay for MemoryStatus(): two /proc reads per
  // request add up on every quiet poll the tier suppresses.
  if (!Enabled(level)) {
    return;
  }
  std::ostringstream message;
  message << "request=" << id << " event=completed method=" << method
          << " path=" << path << " status=" << status_code
          << " duration_ms=" << std::fixed << std::setprecision(1)
          << duration_ms << " outcome=" << outcome;
  if (!details.empty())
    message << ' ' << details;
  message << ' ' << MemoryStatus();
  Log(level, "http", message.str());
}

}  // namespace gufo::server
