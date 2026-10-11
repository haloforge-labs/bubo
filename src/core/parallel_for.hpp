#ifndef GUFO_CORE_PARALLEL_FOR_HPP_
#define GUFO_CORE_PARALLEL_FOR_HPP_

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace gufo::core {

struct ParallelForOptions {
  unsigned max_threads = 0;
  std::size_t min_parallel_size = 32;
};

// Each callback owns its output row/slice. All workers finish before return or
// rethrow; callbacks must not depend on iteration order or shared scratch.
template<typename Function>
void ParallelFor(std::size_t count, Function&& function,
                 ParallelForOptions options = {}) {
#if defined(_OPENMP)
  (void)options;
  // Preserve the upstream OpenMP static scheduling when explicitly enabled.
  std::exception_ptr error;
  std::mutex error_mutex;
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < count; ++i) {
    try {
      function(i);
    } catch (...) {
      std::lock_guard lock(error_mutex);
      if (!error)
        error = std::current_exception();
    }
  }
  if (error)
    std::rethrow_exception(error);
#else
  if (count < options.min_parallel_size || count < 2) {
    for (std::size_t i = 0; i < count; ++i)
      function(i);
    return;
  }
  const unsigned requested = options.max_threads == 0
                                 ? std::thread::hardware_concurrency()
                                 : options.max_threads;
  const std::size_t workers =
      std::min(count, static_cast<std::size_t>(std::clamp(requested, 1U, 8U)));
  if (workers == 1) {
    for (std::size_t i = 0; i < count; ++i)
      function(i);
    return;
  }
  std::vector<std::exception_ptr> errors(workers);
  std::vector<std::jthread> threads;
  threads.reserve(workers - 1);
  const auto run = [&](std::size_t worker) {
    // Balanced contiguous chunks, without overflowing count * worker.
    const std::size_t width = count / workers;
    const std::size_t extra = count % workers;
    const std::size_t begin = worker * width + std::min(worker, extra);
    const std::size_t end = begin + width + (worker < extra ? 1 : 0);
    try {
      for (std::size_t i = begin; i < end; ++i)
        function(i);
    } catch (...) {
      errors[worker] = std::current_exception();
    }
  };
  // jthread also joins already-started workers if thread creation throws.
  for (std::size_t worker = 1; worker < workers; ++worker)
    threads.emplace_back(run, worker);
  run(0);
  for (auto& thread : threads)
    thread.join();
  for (const auto& error : errors)
    if (error)
      std::rethrow_exception(error);
#endif
}

}  // namespace gufo::core

#endif  // GUFO_CORE_PARALLEL_FOR_HPP_
