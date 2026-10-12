// SPDX-License-Identifier: MIT
#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
namespace gufo::npu {
// One persistent worker and one queue command per layer, as in the fork.
class LayerWorker {
public:
  LayerWorker()
      : thread_([this] {
          std::unique_lock lock(mutex_);
          for (;;) {
            wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty())
              return;
            auto call = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();
            call();
            lock.lock();
          }
        }) {}
  ~LayerWorker() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }
  void Enqueue(std::function<void()> call) {
    {
      std::lock_guard lock(mutex_);
      queue_.push_back(std::move(call));
    }
    wake_.notify_one();
  }

private:
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::function<void()>> queue_;
  bool stop_ = false;
  std::thread thread_;
};
inline LayerWorker& PersistentLayerWorker() {
  static LayerWorker worker;
  return worker;
}
inline void EnqueueLayerWork(std::function<void()> call) {
  PersistentLayerWorker().Enqueue(std::move(call));
}
}  // namespace gufo::npu
