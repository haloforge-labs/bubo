// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gufo::npu::fake_hip {
// CPU stream: kernels and HIP host functions execute in order on one thread.
class Stream {
public:
  Stream();
  ~Stream();
  hipStream_t get() { return reinterpret_cast<hipStream_t>(this); }
  void Enqueue(std::function<void()> call);
  void Drain();
  void Record(std::string event);
  std::vector<std::string> Events();
  std::atomic<unsigned> hostfuncs{0}, resets{0}, decodes{0}, syncs{0};
  std::atomic<bool> invalidate_next_decode{false}, check_c{true};

private:
  std::mutex mutex_, events_mutex_;
  std::condition_variable wake_;
  std::deque<std::function<void()>> queue_;
  std::vector<std::string> events_;
  bool stop_ = false;
  std::thread thread_;
};
}  // namespace gufo::npu::fake_hip
