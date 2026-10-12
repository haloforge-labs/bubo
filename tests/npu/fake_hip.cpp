// SPDX-License-Identifier: MIT
// Uses the SDK's declarations but defines every used HIP symbol locally.
// No HIP import library, runtime DLL, device calls or ports.
#include "fake_hip.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <future>

#include "src/npu/bfp_hip.hpp"

namespace gufo::npu::fake_hip {
thread_local bool in_callback = false;
Stream& Get(hipStream_t stream) {
  assert(!in_callback && stream);
  return *reinterpret_cast<Stream*>(stream);
}
Stream::Stream()
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
Stream::~Stream() {
  Drain();
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_one();
  thread_.join();
}
void Stream::Enqueue(std::function<void()> call) {
  {
    std::lock_guard lock(mutex_);
    queue_.push_back(std::move(call));
  }
  wake_.notify_one();
}
void Stream::Drain() {
  auto done = std::make_shared<std::promise<void>>();
  auto ready = done->get_future();
  Enqueue([done] { done->set_value(); });
  ready.wait();
}
void Stream::Record(std::string event) {
  std::lock_guard lock(events_mutex_);
  events_.push_back(std::move(event));
}
std::vector<std::string> Stream::Events() {
  std::lock_guard lock(events_mutex_);
  return events_;
}
}  // namespace gufo::npu::fake_hip

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned) {
  assert(!gufo::npu::fake_hip::in_callback);
  *ptr = std::calloc(1, size);
  return *ptr ? hipSuccess : hipErrorOutOfMemory;
}
hipError_t hipHostFree(void* ptr) {
  assert(!gufo::npu::fake_hip::in_callback);
  std::free(ptr);
  return hipSuccess;
}
hipError_t hipHostGetDevicePointer(void** device, void* host, unsigned) {
  assert(!gufo::npu::fake_hip::in_callback);
  *device = host;
  return hipSuccess;
}
const char* hipGetErrorString(hipError_t) {
  assert(!gufo::npu::fake_hip::in_callback);
  return "fake HIP error";
}
hipError_t hipMemsetAsync(void* ptr, int value, size_t bytes,
                          hipStream_t stream) {
  auto& s = gufo::npu::fake_hip::Get(stream);
  ++s.resets;
  s.Enqueue([=] { std::memset(ptr, value, bytes); });
  return hipSuccess;
}
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* data) {
  auto& s = gufo::npu::fake_hip::Get(stream);
  const unsigned index = s.hostfuncs++;
  s.Enqueue([&s, fn, data, index] {
    s.Record("host.begin." + std::to_string(index));
    gufo::npu::fake_hip::in_callback = true;
    fn(data);
    gufo::npu::fake_hip::in_callback = false;
    s.Record("host.end." + std::to_string(index));
  });
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t stream) {
  auto& s = gufo::npu::fake_hip::Get(stream);
  ++s.syncs;
  s.Drain();
  return hipSuccess;
}
namespace gufo::npu::bfp {
hipError_t EncodeAHip(std::span<const std::uint16_t> input, std::size_t pitch,
                      std::span<std::uint8_t> output, unsigned*,
                      hipStream_t stream) {
  auto& s = fake_hip::Get(stream);
  assert(pitch == K && input.size() == std::size_t{M} * K);
  assert(output.size() == ABytes);
  s.Enqueue([=, &s] {
    s.Record("encode");
    std::memset(output.data(), input.front(), output.size());
  });
  return hipSuccess;
}
hipError_t DecodeCHip(std::span<const std::uint8_t> input,
                      std::span<float> output, std::size_t pitch,
                      std::size_t col0, std::size_t valid_cols, bool accumulate,
                      unsigned* invalid, hipStream_t stream) {
  auto& s = fake_hip::Get(stream);
  assert(input.size() == CBytes && output.size() == M * pitch);
  assert(valid_cols && col0 + valid_cols <= pitch && !accumulate);
  s.Enqueue([=, &s] {
    s.Record("decode");
    if (s.check_c)
      assert(input.front() == 0x42 && input.back() == 0x42);
    output[col0] = input.front();
    if (s.invalidate_next_decode.exchange(false))
      *invalid = 1;
    ++s.decodes;
  });
  return hipSuccess;
}
}  // namespace gufo::npu::bfp
