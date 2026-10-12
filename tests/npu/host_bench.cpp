// SPDX-License-Identifier: MIT
// This source also builds against the frozen 5c71d457 adapter outside the repo.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>

#include "fake_xrt.hpp"
#include "src/npu/bfp.hpp"
#include "src/npu/private_bo_adapter.hpp"
using namespace gufo::npu;
using Clock = std::chrono::steady_clock;
template<class F>
double Measure(unsigned n, F call) {
  const auto start = Clock::now();
  for (unsigned i = 0; i < n; ++i) {
    call();
    std::atomic_signal_fence(std::memory_order_seq_cst);
  }
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
             .count() /
         n;
}
#ifdef NPU_HEAVY_BASELINE
struct Sink : EventSink {
  EventEnvelope Stamp() noexcept override { return {}; }
  void Record(const CallOutcome&) noexcept override {}
};
#endif
int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  std::vector<std::byte> a(kABytes), b(kBBytes), c(kCBytes), stage_a(kABytes),
      stage_b(kBBytes);
  std::vector<float> decoded(std::size_t{4096} * 1024);
  std::vector<std::vector<std::uint8_t>> weights(
      6, std::vector<std::uint8_t>(kBBytes));
  fake::Reset();
  fake::control.expected_a = 0;
#ifdef NPU_HEAVY_BASELINE
  Sink sink;
  auto adapter = PrivateBoAdapter::Create(
      gufo::core::ExperimentalAuthorization::FakePrefill(), argv[1], a, b,
      sink);
  const auto dispatch = [&] {
    // These are the actual removed NpuSession::Dispatch staging copies.
    std::memcpy(stage_a.data(), a.data(), a.size());
    std::memcpy(stage_b.data(), b.data(), b.size());
    adapter->SetInputs(stage_a, stage_b);
    adapter->Submit();
    adapter->Finish();
    adapter->CopyOutput(c);
  };
  dispatch();
  std::cout << "heavy original adapter + session staging, null trace sink: "
            << Measure(100, dispatch) << " us/dispatch\n";
  std::cout << "heavy with original CPU C decode: " << Measure(100, [&] {
    dispatch();
    bfp::DecodeC({reinterpret_cast<const std::uint8_t*>(c.data()), c.size()},
                 decoded, 1024, 0, 1024, false);
  }) << " us/dispatch\n";
#else
  auto adapter = PrivateBoAdapter::Create(
      gufo::core::ExperimentalAuthorization::FakePrefill(), argv[1], a, b);
  for (const auto& w : weights)
    adapter->PrepareWeight(std::as_bytes(std::span(w)));
  const auto dispatch = [&] {
    adapter->ExecuteBatch(a, weights,
                          [&](std::size_t, std::span<const std::byte> output) {
                            std::memcpy(c.data(), output.data(), output.size());
                          });
  };
  dispatch();
  std::cout << "lean resident batch + packed C copy: "
            << Measure(100, dispatch) / 6 << " us/dispatch\n";
#endif
  std::cout << "A memcpy once: " << Measure(100, [&] {
    std::memcpy(stage_a.data(), a.data(), a.size());
  }) << " us (lean amortizes /6)\n";
  std::cout << "B memcpy: " << Measure(100, [&] {
    std::memcpy(stage_b.data(), b.data(), b.size());
  }) << " us (lean load only)\n";
  volatile int equal = 0;
  std::cout << "B memcmp: " << Measure(100, [&] {
    equal = std::memcmp(stage_b.data(), b.data(), b.size());
  }) << " us (lean none)\n";
  std::cout << "CPU C decode: " << Measure(100, [&] {
    bfp::DecodeC({reinterpret_cast<const std::uint8_t*>(c.data()), c.size()},
                 decoded, 1024, 0, 1024, false);
  }) << " us (lean GPU decode)\n";
  std::cout << "C packed copy: " << Measure(100, [&] {
    std::memcpy(c.data(), stage_a.data(), c.size());
  }) << " us\n";
  std::cout << "buffers A=" << a.size() << " B=" << b.size()
            << " C=" << c.size() << " bytes\n";
}
