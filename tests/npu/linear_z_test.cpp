// SPDX-License-Identifier: MIT
#include "src/npu/linear_z.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>

#include "fake_hip.hpp"
#include "fake_xrt.hpp"
#include "file_io_hook.hpp"
#include "src/npu/linear_z_geometry.hpp"
#include "src/npu/linear_z_hip.hpp"

using namespace gufo::npu;
using namespace std::chrono_literals;
namespace {
auto Weights(unsigned slices = 1) {
  auto w = std::make_shared<LinearZWeights>();
  w->gpu_rows = 16384 - slices * kN;
  w->b.resize(slices, std::vector<std::uint8_t>(bfp::BBytes));
  return w;
}
struct Transfer : LinearZTransfer {
  unsigned groups = 1, uploads = 0;
  bool invalid = false, nonstandard_upload = false;
  bool input_done = false, output_done = false;
  std::vector<float> tail;
  std::vector<std::vector<std::uint8_t>> FenceInput() override {
    if (invalid)
      throw std::runtime_error("encoder invalid flag");
    input_done = true;
    return {groups, std::vector<std::uint8_t>(bfp::ABytes)};
  }
  void FenceOutput(std::span<const float> v, unsigned columns) override {
    assert(input_done && v.size() == std::size_t{groups} * kTokens * columns);
    if (nonstandard_upload)
      throw 7;
    ++uploads;
    tail.assign(v.begin(), v.end());
    output_done = true;
  }
};
NpuResult Good(unsigned value = 1) {
  NpuResult r{true, true, true, true, false, {}, {}};
  r.output.resize(bfp::CBytes);
  for (std::size_t i = 0; i < r.output.size(); i += 9) {
    r.output[i] = std::byte{128};
    std::fill_n(r.output.begin() + i + 1, 8, std::byte(value * 8));
  }
  return r;
}
template<class F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}
}  // namespace
int main(int argc, char** argv) {
  assert(argc == 2);
  {
    Transfer transfer;
    transfer.nonstandard_upload = true;
    auto work = BeginLinearZ(Weights(), transfer, 0,
                             [](auto, auto, auto) { return Good(); });
    auto result = FinishLinearZ(*work, transfer, 1024);
    assert(!result.valid && result.tail.empty() && !transfer.output_done);
    std::cout << "PASS: non-standard output callback exception rejects linear "
                 "result\n";
  }
  assert(LinearZLaunchRows(0, 16384, 10240) == 16384);
  for (unsigned slices = 1; slices <= 12; ++slices) {
    const auto rows = 16384 - slices * 1024;
    assert(LinearZLaunchRows(rows, 16384, 10240) == rows);
    // Independent launch-order oracle: every prefix row tile occurs once,
    // no tail tile runs. Token grouping and full 16384 output pitch stay fixed.
    unsigned count[64]{};
    for (unsigned y = 0; y < rows / 256; ++y)
      for (unsigned x = 0; x < 32; ++x) {
        const auto row_tile = (y / 2) * 2 + ((y % 2) * 32 + x) % 2;
        ++count[row_tile];
      }
    for (unsigned tile = 0; tile < 64; ++tile)
      assert(count[tile] == (tile < rows / 256 ? 32U : 0U));
  }
  for (auto rows : {1U, 1024U, 2048U, 3072U, 4095U, 4097U, 9215U, 10239U,
                    10241U, 15361U, 16385U, 17408U})
    assert(LinearZLaunchRows(rows, 16384, 10240) == 0);
  std::cout << "PASS: shared host launch guard and independent GPU prefix/tile "
               "coverage oracle\n";
  assert(LinearZLaunchRows(4096, 8192, 10240) == 0);
  assert(LinearZLaunchRows(4096, 16384, 10241) == 0);
  auto w = Weights();
  for (auto n : {4096U, 8192U, 12288U, 16384U})
    assert(LinearZQualified(n, 16384, true, false, false, w.get()));
  for (auto n : {0U, 1U, 1024U, 4095U, 4097U, 8193U, 16385U})
    assert(!LinearZQualified(n, 16384, true, false, false, w.get()));
  assert(!LinearZQualified(4096, 4095, true, false, false, w.get()));
  assert(!LinearZQualified(4096, 8192, false, false, false, w.get()));
  assert(!LinearZQualified(4096, 8192, true, true, false, w.get()));
  assert(!LinearZQualified(4096, 8192, true, false, true, w.get()));
  assert(!LinearZQualified(4096, 8192, true, false, false, nullptr));
  assert(!LinearZQualified(20480, 32768, true, false, false, w.get()));
  auto forged = Weights();
  forged->gpu_rows = 14336;
  assert(!LinearZQualified(4096, 8192, true, false, false, forged.get()));
  std::cout << "PASS: eager prefill only; "
               "token/capacity/speculative/ready/null fallback\n";
  constexpr std::size_t row = 80 * 34;
  std::vector<std::byte> qkv(10240 * row), gate(6144 * row);
  Q8Matrix q{gufo::core::GgmlType::kQ8_0, 10240, 2560, qkv};
  Q8Matrix z{gufo::core::GgmlType::kQ8_0, 6144, 2560, gate};
  auto staged = PrepareLinearZ(q, z, 1);
  assert(staged && staged->gpu_rows == 15360 && staged->b.size() == 1);
  assert(staged->b[0].size() == bfp::BBytes);
  gate.back() = std::byte{7};
  assert(staged->b[0].back() == 0);
  auto all = PrepareLinearZ(q, z, 6);
  assert(all && all->gpu_rows == 10240 && all->b.size() == 6);
  auto twelve = PrepareLinearZ(q, z, 12);
  assert(twelve && twelve->gpu_rows == 4096 && twelve->b.size() == 12);
  Reject([&] { PrepareLinearZ(q, z, 0); });
  Reject([&] { PrepareLinearZ(q, z, 13); });
  auto short_q = q;
  short_q.bytes = short_q.bytes.first(short_q.bytes.size() - 1);
  assert(!PrepareLinearZ(short_q, z, 1));
  auto short_z = z;
  short_z.bytes = short_z.bytes.first(short_z.bytes.size() - 1);
  assert(!PrepareLinearZ(q, short_z, 1));
  auto bad = z;
  bad.rows = 6143;
  assert(!PrepareLinearZ(q, bad, 6));
  bad.bytes = bad.bytes.first(6143 * row);
  assert(!PrepareLinearZ(q, bad, 6));
  bad = z;
  bad.type = gufo::core::GgmlType::kF16;
  assert(!PrepareLinearZ(q, bad, 6));
  gate[gate.size() - row + 1] = std::byte{0x7c};
  Reject([&] { PrepareLinearZ(q, z, 1); });
  std::cout
      << "PASS: owned Q8 z staging, 1/6/12 slices, trailing QKV at 7..12, "
         "strict slice bounds and finite scales\n";
  LinearDispatch forbidden = [](auto, auto, auto) {
    std::abort();
    return NpuResult{};
  };
  Reject([&] { LinearZWork(nullptr, {{1}}, 0, forbidden); });
  Reject([&] { LinearZWork(w, {}, 0, forbidden); });
  Reject([&] { LinearZWork(w, {{1}}, 0, forbidden); });
  auto short_b = Weights();
  short_b->b[0].pop_back();
  Reject([&] {
    LinearZWork(short_b, {std::vector<std::uint8_t>(bfp::ABytes)}, 0,
                forbidden);
  });
  auto bad_rows = Weights();
  bad_rows->gpu_rows = 14336;
  Reject([&] {
    LinearZWork(bad_rows, {std::vector<std::uint8_t>(bfp::ABytes)}, 0,
                forbidden);
  });
  auto good_a = std::vector<std::uint8_t>(bfp::ABytes);
  Reject([&] { LinearZWork(w, {good_a}, 0, {}); });
  Reject([&] {
    LinearZWork(w, {good_a, good_a, good_a, good_a, good_a}, 0, forbidden);
  });
  std::cout << "PASS: forged work dimensions, empty/short A/B, dispatch "
               "capability and max groups rejected before worker\n";
  Transfer transfer;
  transfer.groups = 4;
  std::atomic<unsigned> calls = 0;
  auto work = BeginLinearZ(
      Weights(6), transfer, 46, [&](auto a, auto b, DispatchInfo info) {
        assert(transfer.input_done && a.size() == kABytes &&
               b.size() == kBBytes);
        assert(info.layer == 46 && info.slice >= 0 && info.slice < 6);
        const auto group = calls++ / 6;
        assert(info.lane == "linear-z" &&
               info.group == static_cast<int>(group));
        return Good(1 + group + info.slice);
      });
  auto result = FinishLinearZ(*work, transfer, 6144);
  assert(result.valid && transfer.output_done && transfer.uploads == 1 &&
         calls == 24);
  assert(!work->Finish(0ms).valid && !work->Finish(-1ms).valid);
  for (unsigned t : {0U, 4095U, 4096U, 8191U, 8192U, 12287U, 12288U, 16383U})
    for (unsigned s = 0; s < 6; ++s)
      for (unsigned c : {0U, 1023U})
        assert(transfer.tail[std::size_t{t} * 6144 + s * 1024 + c] ==
               (1 + t / 4096 + s) / 4.0F);
  std::cout << "PASS: four groups/six slices, trace layer/lane/group/slice and "
               "independent tail oracle\n";
  Transfer wide;
  auto wide_work =
      BeginLinearZ(Weights(12), wide, 44, [](auto, auto, DispatchInfo info) {
        assert(info.slice >= 0 && info.slice < 12 && info.group == 0);
        return Good(1 + info.slice % 6);
      });
  auto wide_result = FinishLinearZ(*wide_work, wide, 12288);
  assert(wide_result.valid && wide.output_done &&
         wide.tail.size() == std::size_t{4096} * 12288);
  for (unsigned slice = 0; slice < 12; ++slice)
    for (unsigned t : {0U, 4095U})
      assert(wide.tail[std::size_t{t} * 12288 + slice * 1024 + 1023] ==
             (1 + slice % 6) / 4.0F);
  std::cout << "PASS: all twelve tail slices including QKV are "
               "completed/decoded/fenced before any consumer\n";
  for (unsigned fault = 0; fault < 8; ++fault) {
    Transfer t;
    std::atomic<unsigned> rejections = 0;
    auto task = BeginLinearZ(
        w, t, 0,
        [=](auto, auto, auto) {
          auto r = Good();
          if (fault == 0)
            r.submitted = false;
          if (fault == 1)
            r.executed = false;
          if (fault == 2)
            r.healthy = false;
          if (fault == 3)
            r.result_valid = false;
          if (fault == 4)
            r.owner_recovery = true;
          if (fault == 5)
            r.output.pop_back();
          if (fault == 6)
            r.output[0] = std::byte{255};
          if (fault == 7)
            throw std::runtime_error("dispatch exception");
          return r;
        },
        [&](std::string error) {
          assert(!error.empty());
          ++rejections;
        });
    auto r = FinishLinearZ(*task, t, 1024);
    assert(!r.valid && !r.error.empty() && t.uploads == 0 && r.tail.empty() &&
           rejections == 1);
  }
  Transfer invalid;
  invalid.invalid = true;
  Reject([&] {
    BeginLinearZ(w, invalid, 0, [](auto, auto, auto) {
      std::abort();
      return NpuResult{};
    });
  });
  std::cout << "PASS: "
               "submitted/executed/healthy/valid/recovery/size/nonfinite/"
               "exception refuse consumer; encoder flag before dispatch\n";
  std::promise<void> entered, release, finished;
  auto gate_wait = release.get_future().share();
  std::weak_ptr<const LinearZWeights> lifetime;
  Transfer delayed;
  {
    auto owned = Weights();
    lifetime = owned;
    auto task = BeginLinearZ(owned, delayed, 0, [&](auto a, auto b, auto) {
      entered.set_value();
      gate_wait.wait();
      assert(a.size() == kABytes && b.size() == kBBytes);
      finished.set_value();
      return Good();
    });
    entered.get_future().wait();
    auto r = FinishLinearZ(*task, delayed, 1024, 10ms);
    assert(!r.valid && delayed.uploads == 0);
  }
  assert(!lifetime.expired());
  release.set_value();
  finished.get_future().wait();
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!lifetime.expired()) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(1ms);
  }
  std::cout << "PASS: timed-out/destroyed caller retains A/B/weight/capability "
               "until worker completes; no upload/reuse\n";
  {
    fake::Reset();
    fake::control.expected_a = 0;
    SetEnvironmentVariableA("GUFO_NPU_RUN_DIR", nullptr);
    SetEnvironmentVariableA("GUFO_NPU_WATCHER_HEARTBEAT", nullptr);
    SessionOptions options;
    options.authorization =
        gufo::core::ExperimentalAuthorization::FakePrefill();
    options.artifacts = argv[1];
    options.timing.keepalive_interval = 0ms;
    auto service = std::make_shared<ServerPrefill>(options);
    assert(service->Start());
    service->BeginChunk();
    fake_hip::Stream stream;
    std::vector<std::uint16_t> input(std::size_t{8192} * kK);
    std::vector<float> output(std::size_t{8192} * 16384);
    auto t = std::make_unique<HipLinearZTransfer>(
        service, input.data(), output.data(), 8192, stream.get());
    std::promise<void> submitted, complete;
    auto completion = complete.get_future().share();
    auto submission = submitted.get_future();
    std::atomic<unsigned> groups{0};
    auto task =
        BeginLinearZ(Weights(2), *t, 0, forbidden, {},
                     [&](auto a, const auto& b, const auto& consume) {
                       assert(a.size() == bfp::ABytes && b.size() == 2);
                       if (groups++ == 0) {
                         submitted.set_value();
                         completion.wait();
                       }
                       std::vector<std::byte> c(bfp::CBytes, std::byte{0x42});
                       for (unsigned s = 0; s < 2; ++s)
                         consume(s, c);
                       return true;
                     });
    assert(submission.wait_for(2s) == std::future_status::ready);
    auto finish = std::async(std::launch::async,
                             [&] { return FinishLinearZ(*task, *t, 2048); });
    const bool returned = finish.wait_for(500ms) == std::future_status::ready;
    if (!returned)
      complete.set_value();
    assert(returned && finish.get().valid && stream.decodes == 0);
    assert(stream.syncs == 0);
    task.reset();
    t.reset();
    complete.set_value();
    stream.Enqueue([&] {
      assert(output[14336] == 0x42);
      assert(output[std::size_t{4096} * 16384 + 14336] == 0x42);
      stream.Record("consumer");
    });
    assert(hipStreamSynchronize(stream.get()) == hipSuccess);
    std::string error;
    assert(CheckHipPrefillChunk(*service, &error));
    assert(groups == 2 && stream.hostfuncs == 2 && stream.decodes == 4);
    assert(stream.resets == 1 && stream.syncs == 1);
    const auto events = stream.Events();
    auto at = [&](const char* value) {
      return std::find(events.begin(), events.end(), value) - events.begin();
    };
    assert(at("encode") < at("host.begin.0"));
    assert(at("host.end.1") < at("decode") && at("decode") < at("consumer"));
    std::cout
        << "PASS: two-group stream Begin/Wait/Decode/consumer ordering; "
           "Finish returns before NPU completion; caller destruction safe\n";
  }
}
