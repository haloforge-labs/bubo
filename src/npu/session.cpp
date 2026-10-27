// SPDX-License-Identifier: MIT
// Persistent context and 1000 ms / 2000 ms keepalive semantics from the MIT
// fork.
#include "session.hpp"

#include <atomic>
#include <charconv>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "fault_state.hpp"
#include "runtime_status.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"
namespace gufo::npu {
namespace {
std::atomic<bool> process_attempted{false}, process_failed{false};
unsigned Strict(std::string_view text, unsigned fallback, unsigned low,
                unsigned high) {
  if (text.empty())
    return fallback;
  unsigned value = 0;
  auto r = std::from_chars(text.data(), text.data() + text.size(), value);
  if (r.ec != std::errc{} || r.ptr != text.data() + text.size() ||
      value < low || value > high)
    throw AdapterError(ErrorKind::Validation, "invalid NPU timing value");
  return value;
}
}  // namespace
bool ProcessFailed() noexcept {
  return process_failed.load();
}
void FailProcess(std::string_view message) noexcept {
  if (process_failed.exchange(true))
    return;
  try {
    std::string error(message);
    for (auto& c : error)
      if (c == '\n' || c == '\r' || c == '\t')
        c = ' ';
    server::Logger::Warn(
        "npu", error + "; NPU disabled for process; GPU path");
  } catch (...) {
  }
  NoteNpuFaulted();  // status: faulted, reason "fault" (no-op before start-up)
}
Timing ParseTiming(std::string_view timeout, std::string_view keepalive,
                   std::string_view interval) {
  if (!keepalive.empty() && keepalive != "0" && keepalive != "1")
    throw AdapterError(ErrorKind::Validation,
                       "expected literal NPU keepalive 0|1");
  Timing t;
  t.timeout = std::chrono::milliseconds(Strict(timeout, 8000, 1, 8000));
  t.keepalive_interval = std::chrono::milliseconds(
      Strict(keepalive, 1, 0, 1) ? Strict(interval, 1000, 0, 60000) : 0);
  if (t.keepalive_interval.count() != 0 && t.keepalive_interval.count() < 100)
    throw AdapterError(ErrorKind::Validation, "invalid NPU keepalive interval");
  return t;
}
struct NpuSession::State {
  SessionOptions options;
  std::mutex mutex, wake_mutex;
  std::condition_variable wake;
  std::thread keepalive;
  std::unique_ptr<PrivateBoAdapter> adapter;
  std::atomic<bool> healthy{false}, stop{false}, paused{false};
  bool attempted = false;
  std::uint64_t beats = 0, dispatches = 0;
  std::chrono::steady_clock::time_point last_dispatch;
  std::vector<std::byte> heartbeat_a, heartbeat_b;
  explicit State(SessionOptions o) : options(std::move(o)) {}
  NpuResult Fail(std::string error) {
    healthy = false;
    FailProcess(error);
    return {false, false, false, false, false, std::move(error), {}};
  }
  NpuResult Run(std::span<const std::byte> a, std::span<const std::byte> b,
                const DispatchInfo& info) {
    NpuResult r;
    try {
      if (info.heartbeat) {
        adapter->Heartbeat();
        r.submitted = r.executed = true;
        ++beats;
      } else {
        // The single-dispatch API accepts mutable input spans. Residency is
        // reserved for ExecuteBatch's immutable prepared model weights.
        adapter->SetInputs(a, b);
        adapter->Submit();
        r.submitted = true;
        adapter->Finish();
        r.executed = true;
        r.output.resize(kCBytes);
        adapter->CopyOutput(r.output);
        ++dispatches;
        NoteRealDispatch();
      }
      r.healthy = r.result_valid = true;
      last_dispatch = std::chrono::steady_clock::now();
    } catch (const std::exception& e) {
      return Fail(e.what());
    }
    return r;
  }
};
NpuSession::NpuSession(SessionOptions o)
    : state_(std::make_shared<State>(std::move(o))) {}
NpuSession::~NpuSession() {
  Shutdown();
}
bool NpuSession::Healthy() const noexcept {
  return state_->healthy && !process_failed;
}
std::chrono::milliseconds NpuSession::Timeout() const noexcept {
  return state_->options.timing.timeout;
}
NpuResult NpuSession::Start() {
  auto& s = *state_;
  if (!s.options.authorization.Allows(core::ExperimentalFeature::Prefill))
    return {};
  if (ProcessFailed() || s.attempted || process_attempted.exchange(true))
    return {};
  s.attempted = true;
  try {
    if (s.options.timing.timeout.count() < 1 ||
        s.options.timing.timeout.count() > 8000)
      throw AdapterError(ErrorKind::Validation, "invalid NPU timeout");
    s.heartbeat_a.resize(kABytes);
    s.heartbeat_b.resize(kBBytes);
    s.adapter = PrivateBoAdapter::Create(
        s.options.authorization, s.options.artifacts, s.heartbeat_a,
        s.heartbeat_b, s.options.timing.timeout);
    DispatchInfo beat;
    beat.heartbeat = true;
    auto first = s.Run(s.heartbeat_a, s.heartbeat_b, beat);
    s.healthy = first.healthy;
    if (s.healthy && s.options.timing.keepalive_interval.count()) {
      s.keepalive = std::thread([state = state_] {
        std::unique_lock lock(state->wake_mutex);
        while (!state->wake.wait_for(lock,
                                     state->options.timing.keepalive_interval,
                                     [&] { return state->stop.load(); })) {
          lock.unlock();
          {
            std::lock_guard serial(state->mutex);
            if (!state->paused.load(std::memory_order_relaxed) &&
                state->healthy && !process_failed &&
                std::chrono::steady_clock::now() - state->last_dispatch >=
                    std::chrono::milliseconds(2000)) {
              DispatchInfo beat;
              beat.heartbeat = true;
              state->Run(state->heartbeat_a, state->heartbeat_b, beat);
            }
          }
          lock.lock();
        }
      });
    }
    if (s.healthy) {
      try {
        const auto platform = json::parse(s.adapter->PowerMode());
        server::Logger::Info("npu", "power mode (read-only)=" + platform.dump());
      } catch (...) {
        // Optional platform information must not disable a working session.
      }
    }
    return first;
  } catch (const std::exception& e) {
    return s.Fail(e.what());
  } catch (...) {
    return s.Fail("unknown startup failure");
  }
}
void NpuSession::SetKeepalivePaused(bool paused) noexcept {
  state_->paused.store(paused, std::memory_order_relaxed);
}
bool NpuSession::WarmUpBeat(std::string* error) {
  std::lock_guard lock(state_->mutex);
  if (!Healthy() || !state_->adapter) {
    *error = "NPU session not healthy";
    return false;
  }
  DispatchInfo beat;
  beat.heartbeat = true;
  const auto result =
      state_->Run(state_->heartbeat_a, state_->heartbeat_b, beat);
  if (!result.healthy) {
    *error = result.error;
    return false;
  }
  return true;
}
std::vector<const void*> NpuSession::SamplePages() {
  std::lock_guard lock(state_->mutex);
  if (!state_->adapter)
    return {};
  try {
    return state_->adapter->SamplePages();
  } catch (...) {
    return {};
  }
}
void NpuSession::PrepareWeight(std::span<const std::byte> b) {
  std::lock_guard lock(state_->mutex);
  if (Healthy()) {
    try {
      state_->adapter->PrepareWeight(b);
    } catch (const std::exception& e) {
      state_->Fail(e.what());
    }
  }
}
bool NpuSession::DispatchBatch(
    std::span<const std::byte> a,
    const std::vector<std::vector<std::uint8_t>>& weights,
    const std::function<void(std::size_t, std::span<const std::byte>)>&
        consume) {
  std::lock_guard lock(state_->mutex);
  if (!Healthy())
    return false;
  try {
    const auto repairs_before = state_->adapter->StaleWeightRepairs();
    state_->adapter->ExecuteBatch(a, weights, consume);
    state_->dispatches += weights.size();
    state_->last_dispatch = std::chrono::steady_clock::now();
    NoteRealDispatch();
    if (const auto repairs = state_->adapter->StaleWeightRepairs();
        repairs != repairs_before)
      server::Logger::Warn(
          "npu",
          "weight guard: " + std::to_string(repairs - repairs_before) +
              " weight buffer(s) changed under a reused address and were "
              "re-sent (GUFO_NPU_WEIGHT_GUARD=1)");
    return true;
  } catch (const std::exception& e) {
    state_->Fail(e.what());
    return false;
  }
}
NpuResult NpuSession::Dispatch(std::span<const std::byte> a,
                               std::span<const std::byte> b,
                               DispatchInfo info) {
  if (a.size() != kABytes || b.size() != kBBytes)
    return {false, false, false, false, false, "NPU input size rejected", {}};
  std::lock_guard lock(state_->mutex);
  if (!Healthy())
    return {};
  return state_->Run(a, b, info);
}
NpuResult NpuSession::RejectResult(std::string error) {
  // May be called by a bounded stream wait while XRT owns the session mutex.
  return state_->Fail(std::move(error));
}
NpuResult NpuSession::Shutdown() {
  state_->stop = true;
  state_->healthy = false;
  state_->wake.notify_all();
  if (state_->keepalive.joinable())
    state_->keepalive.join();
  return {};
}
}  // namespace gufo::npu
