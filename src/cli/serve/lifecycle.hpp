#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/status_throughput.hpp"

namespace bubo::cli {
// State is published before output is queued. The inference and control
// threads never write machine stdout and never wait for its consumer.
class ServeLifecycle {
public:
  explicit ServeLifecycle(bool machine = true);
  ~ServeLifecycle();
  ServeLifecycle(const ServeLifecycle&) = delete;
  ServeLifecycle& operator=(const ServeLifecycle&) = delete;
  const InstanceIdentity& Identity() const { return identity_; }
  bool MachineEnabled() const { return writer_ != nullptr; }
  void Configure(std::string endpoint, std::string served_model, Value settings,
                 std::optional<std::filesystem::path> log_path = {},
                 bool listener_bound = false);
  void Transition(std::string type, Value payload = Value::object());
  void SetDetails(Value details);
  Value Snapshot() const;
  void SetSampler(std::function<Value()> sampler);
  void SetStop(std::function<void()> stop);
  bool StopRequested() const { return stop_requested_.load(); }
  void OpenControl();
  void Finish();
  void RecordAbruptExit();

private:
  void RequestStop();
  void Drain();
  void SampleLocked();
  void SampleLoop();
  void MachineLoop();
  void PublishLocked(bool tombstone = false);
  InstanceIdentity identity_;
  std::unique_ptr<MachineWriter> writer_;
  std::unique_ptr<ControlServer> control_;
  mutable std::mutex mutex_;
  mutable std::mutex sample_mutex_;
  mutable ThroughputBaseline throughput_;
  std::chrono::steady_clock::time_point sampled_at_{};
  std::condition_variable sample_wake_;
  std::condition_variable machine_wake_;
  std::mutex machine_mutex_;
  std::atomic<bool> samples_closing_{false};
  std::thread sample_thread_;
  std::thread machine_thread_;
  Value snapshot_;
  std::function<Value()> sampler_;
  std::function<void()> stop_;
  std::atomic<bool> stop_requested_{false};
  std::condition_variable wake_;
  struct Event {
    std::string type;
    Value payload;
  };
  std::deque<Event> events_;
  std::size_t queued_bytes_ = 0;
  bool output_failed_ = false;
  bool closing_ = false;
  bool finished_ = false;
  bool published_ = false;
  std::thread output_thread_;
};
}  // namespace bubo::cli
