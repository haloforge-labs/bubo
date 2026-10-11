#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/logging_sink.hpp"

using gufo::server::AsyncLogSink;
using gufo::server::Logger;
using gufo::server::LogLevel;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace {
void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
double Ms(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}
struct Capture {
  std::mutex mutex;
  std::condition_variable changed;
  std::string text;
  std::size_t writes = 0;
  void Append(std::string_view batch) {
    const std::lock_guard lock(mutex);
    text += batch;
    ++writes;
    changed.notify_all();
  }
  std::string Text() {
    const std::lock_guard lock(mutex);
    return text;
  }
};
struct Hook {
  explicit Hook(AsyncLogSink& sink) { Logger::SetSinkForTesting(&sink); }
  ~Hook() { Logger::SetSinkForTesting(nullptr); }
};
void SlowSink() {
  auto capture = std::make_shared<Capture>();
  AsyncLogSink sink([capture](std::string_view batch) {
    std::this_thread::sleep_for(20ms);
    capture->Append(batch);
  });
  Hook hook(sink);
  std::array<std::vector<double>, 4> times;
  std::vector<std::thread> producers;
  const auto start = Clock::now();
  for (int t = 0; t < 4; ++t) {
    producers.emplace_back([&, t] {
      times[t].reserve(2500);
      for (int i = 0; i < 2500; ++i) {
        const auto before = Clock::now();
        Logger::Info("slow", "thread=" + std::to_string(t) +
                                 " line=" + std::to_string(i));
        times[t].push_back(Ms(Clock::now() - before) * 1000);
      }
    });
  }
  for (auto& producer : producers)
    producer.join();
  const auto producer_ms = Ms(Clock::now() - start);
  Expect(Logger::FlushSync(), "slow-sink flush succeeds");
  const auto sink_ms = Ms(Clock::now() - start);
  std::vector<double> sorted;
  for (const auto& part : times)
    sorted.insert(sorted.end(), part.begin(), part.end());
  std::sort(sorted.begin(), sorted.end());
  const auto p99 = sorted[9899];
  std::istringstream lines(capture->Text());
  std::array<int, 4> next{};
  int delivered = 0;
  for (std::string line; std::getline(lines, line);) {
    const auto at = line.find("thread=");
    int t = -1, i = -1;
    Expect(at != std::string::npos &&
               std::sscanf(line.c_str() + at, "thread=%d line=%d", &t, &i) == 2,
           "slow sink retains exact producer identifiers");
    Expect(t >= 0 && t < 4 && i == next[t]++,
           "slow sink preserves per-thread order");
    ++delivered;
  }
  const auto stats = sink.GetStats();
  std::cout << "(a) p99_us=" << p99 << " max_us=" << sorted.back()
            << " producer_ms=" << producer_ms << " sink_ms=" << sink_ms
            << " writes=" << capture->writes << " delivered=" << delivered
            << " dropped=" << stats.dropped << '\n';
  Expect(p99 < 200 && producer_ms < 1000 && sink_ms > 2000,
         "slow sink meets the producer and sink timing gates");
  Expect(delivered == 10000 && stats.dropped == 0 && stats.pending == 0,
         "slow sink flush delivers every line");
}
struct Blocked {
  std::binary_semaphore entered{0}, resume{0};
  std::shared_ptr<Capture> capture = std::make_shared<Capture>();
  std::atomic<bool> first{true};
  void Write(std::string_view batch) {
    if (first.exchange(false)) {
      entered.release();
      resume.acquire();
    }
    capture->Append(batch);
  }
};
void Overflow(std::size_t payload_size, const char* bound) {
  auto blocked = std::make_shared<Blocked>();
  AsyncLogSink sink(
      [blocked](std::string_view batch) { blocked->Write(batch); });
  Hook hook(sink);
  Logger::Raw("seed\n");
  Expect(blocked->entered.try_acquire_for(2s),
         "overflow sink is blocked inside its first write");
  const std::string payload(payload_size, 'x');
  const auto start = Clock::now();
  double max_us = 0;
  for (int i = 0; i < 50000; ++i) {
    const auto before = Clock::now();
    Logger::Raw(LogLevel::kInfo, "info=%d %s\n", i, payload.c_str());
    max_us = std::max(max_us, Ms(Clock::now() - before) * 1000);
  }
  const auto producer_ms = Ms(Clock::now() - start);
  const auto full = sink.GetStats();
  Expect(full.accepted > 1 && full.dropped == 50000 - (full.accepted - 1),
         "drop count equals pushed minus kept, including active bytes/lines");
  for (int i = 0; i < 100; ++i) {
    Logger::Raw("error=%d\n", i);
    Logger::Warn("overflow", "warn=" + std::to_string(i));
  }
  Expect(sink.GetStats().dropped == full.dropped,
         "WARN and ERROR never increment drops");
  blocked->resume.release();
  Expect(Logger::FlushSync(), "overflow flush finishes once the sink resumes");
  const auto text = blocked->capture->Text();
  for (int i = 0; i < 100; ++i) {
    Expect(text.find("error=" + std::to_string(i) + "\n") != std::string::npos,
           "each raw ERROR is delivered");
    Expect(text.find("warn=" + std::to_string(i) + "\n") != std::string::npos,
           "each WARN is delivered");
  }
  const auto summary =
      "[log] dropped " + std::to_string(full.dropped) + " lines\n";
  const auto at = text.find(summary);
  Expect(at != std::string::npos &&
             text.find("[log] dropped", at + 1) == std::string::npos,
         "one summary reports the overflow after resumption");
  const auto stats = sink.GetStats();
  std::cout << "(b) bound=" << bound
            << " pushed=50000 kept=" << full.accepted - 1
            << " dropped=" << full.dropped << " max_call_us=" << max_us
            << " max_call_bound_us=50000 producer_ms=" << producer_ms
            << " errors=100 warns=100 delivered=" << stats.delivered << '\n';
  Expect(max_us < 50000 && producer_ms < 1000 && stats.pending == 0,
         "overflow callers meet the stated 50 ms maximum scheduling-inclusive "
         "bound");
}
void HardCap() {
  auto blocked = std::make_shared<Blocked>();
  AsyncLogSink sink(
      [blocked](std::string_view batch) { blocked->Write(batch); });
  sink.Push("seed\n", LogLevel::kInfo);
  Expect(blocked->entered.try_acquire_for(2s), "hard-cap sink is blocked");
  const std::string large(8 * 1024 * 1024, 'E');
  sink.Push(large, LogLevel::kError);
  auto fallback = std::async(std::launch::async,
                             [&] { sink.Push(large, LogLevel::kWarn); });
  Expect(fallback.wait_for(50ms) == std::future_status::timeout,
         "WARN over the 16 MiB cap uses the permitted synchronous last resort");
  Expect(sink.GetStats().pending_bytes <= 16 * 1024 * 1024,
         "fallback never adds bytes beyond the hard queue cap");
  blocked->resume.release();
  Expect(fallback.wait_for(2s) == std::future_status::ready,
         "last resort completes after the accepted prefix drains");
  fallback.get();
  Expect(sink.FlushSync(), "hard-cap final prefix flush succeeds");
  const auto text = blocked->capture->Text();
  Expect(text.size() == 5 + 2 * large.size() && text.starts_with("seed\n"),
         "both large ERROR/WARN messages arrive without truncation or loss");
  std::cout << "extra hard_cap_bytes=16777216 delivered_bytes=" << text.size()
            << " dropped=" << sink.GetStats().dropped << " fallback=1\n";
}

void EmptyProtectedBound() {
  auto blocked = std::make_shared<Blocked>();
  AsyncLogSink sink(
      [blocked](std::string_view batch) { blocked->Write(batch); });
  sink.Push("seed\n", LogLevel::kInfo);
  Expect(blocked->entered.try_acquire_for(2s), "empty-record sink is blocked");
  sink.Push(std::string(16 * 1024 * 1024 - 512, 'E'), LogLevel::kError);
  auto producer = std::async(std::launch::async, [&] {
    for (int i = 0; i < 1024; ++i)
      sink.Push({}, LogLevel::kError);
  });
  const bool bounded = producer.wait_for(50ms) == std::future_status::timeout;
  const auto full = sink.GetStats();
  blocked->resume.release();
  producer.get();
  Expect(sink.FlushSync(), "empty protected records flush after resumption");
  Expect(
      bounded && full.pending_bytes <= 16 * 1024 * 1024 && full.accepted < 1026,
      "empty protected records consume storage and trigger the hard cap");
  std::cout << "extra empty_protected_bound=1 pending_bytes="
            << full.pending_bytes << " accepted_at_cap=" << full.accepted
            << " dropped=" << full.dropped << '\n';
}

void Ordering() {
  auto capture = std::make_shared<Capture>();
  AsyncLogSink sink(
      [capture](std::string_view batch) { capture->Append(batch); });
  Hook hook(sink);
  Logger::Info("order", "first");
  Logger::Raw("raw=%d\n", 2);
  Logger::Diagnostic("order", "third");
  Logger::Error("order", "fourth");
  Logger::Raw("fragment");
  Logger::Raw(" end\n");
  Expect(Logger::FlushSync(), "mixed ordering flush succeeds");
  const auto text = capture->Text();
  const auto a = text.find("[order] first\n"), b = text.find("raw=2\n");
  const auto c = text.find("[order] third\n"),
             d = text.find("[order] fourth\n");
  Expect(a < b && b < c && c < d && text.ends_with("fragment end\n"),
         "Logger, Diagnostic and Raw preserve order and raw fragments");
  std::cout << "(c) ordered_calls=6 delivered=" << sink.GetStats().delivered
            << " dropped=" << sink.GetStats().dropped << '\n';
}
void FlushUnderLoad() {
  auto capture = std::make_shared<Capture>();
  AsyncLogSink sink([capture](std::string_view batch) {
    std::this_thread::sleep_for(1ms);
    capture->Append(batch);
  });
  Hook hook(sink);
  std::atomic<bool> running{true};
  std::atomic<int> pushed{0};
  std::thread producer([&] {
    for (int i = 0; i < 2000; ++i) {
      Logger::Raw(LogLevel::kInfo, "load=%d\n", i);
      pushed.store(i + 1);
      std::this_thread::sleep_for(100us);
    }
    running.store(false);
  });
  int concurrent_flushes = 0;
  double max_ms = 0;
  while (running.load()) {
    const auto prefix = sink.GetStats().accepted;
    const auto start = Clock::now();
    Expect(Logger::FlushSync(2s), "flush completes while a producer is active");
    max_ms = std::max(max_ms, Ms(Clock::now() - start));
    Expect(sink.GetStats().delivered >= prefix,
           "flush delivered its accepted prefix");
    concurrent_flushes += running.load() && pushed.load() > 0;
    std::this_thread::sleep_for(1ms);
  }
  producer.join();
  Expect(Logger::FlushSync(), "final load flush completes");
  const auto stats = sink.GetStats();
  Expect(
      stats.delivered == 2000 && stats.dropped == 0 && concurrent_flushes > 0,
      "load test delivers all lines with overlapping flush calls");
  std::cout << "(d) concurrent_flushes=" << concurrent_flushes
            << " max_flush_ms=" << max_ms << " delivered=" << stats.delivered
            << " dropped=" << stats.dropped << '\n';
}
void StuckShutdown() {
  auto blocked = std::make_shared<Blocked>();
  AsyncLogSink sink(
      [blocked](std::string_view batch) { blocked->Write(batch); });
  sink.Push("stuck\n", LogLevel::kError);
  Expect(blocked->entered.try_acquire_for(2s),
         "shutdown sink is stuck inside write");
  const auto start = Clock::now();
  const auto stats = sink.Shutdown();
  const auto shutdown_ms = Ms(Clock::now() - start);
  Expect(shutdown_ms >= 1900 && shutdown_ms < 2000 && stats.pending == 1,
         "shutdown waits at most 2 s plus scheduling allowance and reports "
         "pending lines");
  const auto again = Clock::now();
  sink.Shutdown();
  Expect(Ms(Clock::now() - again) < 50,
         "repeated shutdown does not wait another 2 seconds");
  blocked->resume.release();
  std::unique_lock lock(blocked->capture->mutex);
  Expect(blocked->capture->changed.wait_for(
             lock, 2s,
             [&] {
               return blocked->capture->text.find(
                          "shutdown: undelivered 1 lines") != std::string::npos;
             }),
         "shutdown loss report reaches stderr when the blocked sink resumes");
  std::cout << "(e) shutdown_ms=" << shutdown_ms << " wait_bound_ms=2000"
            << " drain_wait_ms=1950 undelivered=" << stats.pending
            << " report_after_resume=1\n";
}
void SyncSmoke() {
  std::ostringstream capture;
  auto* previous = std::clog.rdbuf(capture.rdbuf());
  Logger::Raw("synchronous=%d\n", 1);
  Expect(capture.str() == "synchronous=1\n",
         "GUFO_LOG_SYNC=1 writes before returning");
  Logger::FlushSync();
  std::clog.rdbuf(previous);
  std::cout << "sync-smoke immediate=1 PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--sync-smoke") {
    SyncSmoke();
    return 0;
  }
  SlowSink();
  Overflow(0, "16K-lines");
  Overflow(1024, "4MiB-bytes");
  HardCap();
  EmptyProtectedBound();
  Ordering();
  FlushUnderLoad();
  StuckShutdown();
  std::cout << "PASS: log_async_test (CPU only, no model)\n";
}
