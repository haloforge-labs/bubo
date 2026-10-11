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
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
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

bool WriteStderr(std::string_view batch) noexcept {
  try {
    std::clog.write(batch.data(), static_cast<std::streamsize>(batch.size()));
    std::clog.flush();
    if (std::clog.good())
      return true;
  } catch (...) {
  }
  // badbit is sticky. Clear it even when stream exceptions are enabled, so
  // a transient failure does not poison later batches.
  try {
    std::clog.clear();
  } catch (...) {
  }
  return false;
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
  using Clock = std::chrono::steady_clock;
  explicit State(Writer output, bool sync)
      : writer(std::move(output)), synchronous(sync) {}
  struct Entry {
    std::string line;
    std::uint64_t sequence;
  };
  static constexpr std::size_t kEntryCharge = sizeof(Entry) + 24;
  Writer writer;
  const bool synchronous;
  std::mutex mutex;
  std::condition_variable wake, changed;
  std::deque<Entry> queue;
  // Only one caller-owned hard-cap record may bypass queue admission. It is
  // submitted after the accepted prefix retires and still written by Run.
  std::optional<Entry> fallback;
  std::uint64_t accepted = 0, completed = 0, delivered = 0;
  std::uint64_t dropped = 0, reported = 0, failures = 0, failed_lines = 0;
  std::uint64_t active_lines = 0;
  std::size_t bytes = 0;
  bool stopping = false, done = false, destroying = false;
  bool discard = false, failure_reported = false;
  std::string timeout_report;

  void DiscardAfterTimeout() {
    if (discard)
      return;
    discard = true;
    changed.notify_all();
    // Never take a stream lock or write a real handle on the caller. A broken
    // or full pipe may stall this single detached notice, but not the server.
    try {
      std::thread([] {
        constexpr char notice[] =
            "[log] sink timeout: subsequent lines discarded and counted\n";
#ifdef _WIN32
        const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
        if (handle != nullptr && handle != INVALID_HANDLE_VALUE &&
            GetFileType(handle) != FILE_TYPE_UNKNOWN) {
          DWORD written = 0;
          (void)WriteFile(handle, notice, sizeof(notice) - 1, &written,
                          nullptr);
        }
#else
        // A closed stderr pipe must not terminate the process with SIGPIPE.
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGPIPE);
        (void)pthread_sigmask(SIG_BLOCK, &signals, nullptr);
        pollfd handle{STDERR_FILENO, POLLOUT, 0};
        if (poll(&handle, 1, 0) > 0 && (handle.revents & POLLOUT))
          (void)::write(STDERR_FILENO, notice, sizeof(notice) - 1);
#endif
      }).detach();
    } catch (...) {
      // Thread creation failure cannot escape a logging/teardown caller.
    }
  }

  void Run() {
    std::unique_lock lock(mutex);
    for (;;) {
      wake.wait(lock, [&] {
        return destroying || !queue.empty() || fallback ||
               dropped != reported || !timeout_report.empty() ||
               (stopping && !done);
      });
      if (queue.empty() && !fallback && dropped == reported &&
          timeout_report.empty()) {
        if (stopping && failures && !failure_reported) {
          timeout_report = "[log] sink failures " + std::to_string(failures) +
                           " writes, discarded " +
                           std::to_string(failed_lines) + " lines\n";
          failure_reported = true;
        } else {
          done = stopping;
          changed.notify_all();
          if (destroying)
            return;
          continue;
        }
      }
      std::string batch;
      batch.reserve(65536);
      std::uint64_t sequence = completed, count = 0;
      std::size_t consumed = 0;
      if (fallback &&
          (queue.empty() || fallback->sequence < queue.front().sequence)) {
        batch = std::move(fallback->line);
        sequence = fallback->sequence;
        fallback.reset();
        count = 1;
      } else {
        while (!queue.empty() && count < 64 && batch.size() < 65536 &&
               (!fallback || queue.front().sequence < fallback->sequence)) {
          auto& entry = queue.front();
          batch += entry.line;
          consumed += entry.line.size() + kEntryCharge;
          sequence = entry.sequence;
          queue.pop_front();
          ++count;
        }
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
      bool ok = false;
      try {
        ok = writer(batch);
      } catch (...) {
        // Injected/third-party writers may throw. No sink exception escapes.
      }
      lock.lock();
      bytes -= consumed;
      active_lines = 0;
      completed = sequence;
      reported = drop_target;
      if (ok)
        delivered += count;
      else {
        ++failures;
        failed_lines += count;
      }
      // A failure retires that batch, then waits for fresh work. Never retry
      // a broken sink in a spin loop or permanently kill the writer.
      changed.notify_all();
    }
  }
};

AsyncLogSink::AsyncLogSink(Writer writer, bool synchronous)
    : state_(std::make_shared<State>(std::move(writer), synchronous)) {
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
  const std::lock_guard lock(state_->mutex);
  state_->destroying = true;
  state_->wake.notify_one();
}

void AsyncLogSink::Push(std::string line, LogLevel level) {
  const auto state = state_;
  std::unique_lock lock(state->mutex);
  if (state->discard) {
    ++state->dropped;
    state->wake.notify_one();
    return;
  }
  const auto lines = state->queue.size() + state->active_lines;
  const auto charge = line.size() > 16 * 1024 * 1024
                          ? 16 * 1024 * 1024 + 1
                          : line.size() + State::kEntryCharge;
  if (!state->synchronous && level < LogLevel::kWarn &&
      (state->stopping || lines >= 16384 || charge > 4 * 1024 * 1024 ||
       state->bytes > 4 * 1024 * 1024 - charge)) {
    ++state->dropped;
    state->wake.notify_one();
    return;
  }
  const auto deadline = State::Clock::now() + std::chrono::milliseconds(1950);
  bool fallback_write = false;
  if (state->stopping || charge > 16 * 1024 * 1024 ||
      state->bytes > 16 * 1024 * 1024 - charge) {
    // One total 2 s budget covers both the prefix and the fallback write.
    // After its first expiry every later call is an immediate counted discard.
    if (!state->changed.wait_until(lock, deadline,
                                   [&] {
                                     return state->discard ||
                                            state->completed == state->accepted;
                                   }) ||
        state->discard) {
      state->DiscardAfterTimeout();
      ++state->dropped;
      state->wake.notify_one();
      return;
    }
    state->fallback = State::Entry{std::move(line), ++state->accepted};
    fallback_write = true;
  } else {
    state->bytes += charge;
    state->queue.push_back({std::move(line), ++state->accepted});
  }
  state->done = false;
  const auto target = state->accepted;
  state->wake.notify_one();
  if (state->synchronous || fallback_write) {
    if (!state->changed.wait_until(lock, deadline, [&] {
          return state->discard || state->completed >= target;
        }))
      state->DiscardAfterTimeout();
  }
}

bool AsyncLogSink::FlushSync(std::chrono::milliseconds timeout) {
  const auto state = state_;
  std::unique_lock lock(state->mutex);
  const auto target = state->accepted, drop_target = state->dropped;
  state->wake.notify_one();
  const auto ready = [&] {
    return state->discard ||
           (state->completed >= target && state->reported >= drop_target);
  };
  // Even an omitted timeout is bounded on a caller path.
  timeout = std::min(timeout, std::chrono::milliseconds(1950));
  if (!state->changed.wait_for(lock, timeout, ready))
    return false;
  return !state->discard && state->failures == 0;
}

AsyncLogSink::Stats AsyncLogSink::GetStats() const {
  const std::lock_guard lock(state_->mutex);
  return {state_->accepted,
          state_->delivered,
          state_->dropped,
          state_->reported,
          state_->accepted - state_->completed,
          state_->bytes,
          state_->failures,
          state_->failed_lines};
}

AsyncLogSink::Stats AsyncLogSink::Shutdown() {
  const auto state = state_;
  std::unique_lock lock(state->mutex);
  if (!state->stopping) {
    const auto deadline = State::Clock::now() + std::chrono::milliseconds(1950);
    state->stopping = true;
    state->wake.notify_one();
    if (!state->changed.wait_until(lock, deadline,
                                   [&] { return state->done; })) {
      const auto queued = state->queue.size() + (state->fallback ? 1 : 0);
      state->timeout_report = "[log] shutdown: queued at deadline " +
                              std::to_string(queued) + " lines, in-flight " +
                              std::to_string(state->active_lines) +
                              " lines (2 s shutdown budget)\n";
      state->DiscardAfterTimeout();
      state->wake.notify_one();
    }
  }
  return {state->accepted,
          state->delivered,
          state->dropped,
          state->reported,
          state->accepted - state->completed,
          state->bytes,
          state->failures,
          state->failed_lines};
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
