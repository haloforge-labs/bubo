// SPDX-License-Identifier: MIT
#include "src/npu/session.hpp"

#include <windows.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>

#include "fake_hip.hpp"
#include "fake_xrt.hpp"
#include "file_io_hook.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/npu/linear_z_hip.hpp"
#include "src/npu/projection.hpp"
#include "src/npu/request_stats.hpp"
#include "xrt/xrt_kernel.h"
using namespace gufo::npu;
using namespace std::chrono_literals;
int main(int argc, char** argv) {
  assert(argc == 3);
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
  fake::Reset();
  fake::control.expected_a = 0;
  const std::string mode = argv[1];
  SessionOptions o;
  o.authorization = gufo::core::ExperimentalAuthorization::FakePrefill();
  o.artifacts = argv[2];
  o.timing.keepalive_interval = 0ms;
  if (mode == "stream_timeout")
    o.timing.timeout = 20ms;
  if (mode == "denied") {
    o.authorization = {};
    auto denied = std::make_shared<ServerPrefill>(o);
    assert(!denied->Start() && !denied->ready() && !ProcessFailed());
    assert(fake::control.calls.empty() && fake::control.devices == 0 &&
           fake::control.allocations == 0 && fake::control.starts == 0);
    gufo::server::Logger::FlushSync();
    return 0;
  }
  if (mode.starts_with("startup_")) {
    std::filesystem::path temporary;
    if (mode == "startup_missing_dir") {
      o.artifacts = std::filesystem::temp_directory_path() /
                    ("npu-absent-" + std::to_string(GetCurrentProcessId()) +
                     "-" + std::to_string(GetTickCount64()));
      assert(!std::filesystem::exists(o.artifacts));
    } else if (mode == "startup_hash" || mode == "startup_missing_insts" ||
               mode == "startup_missing_xclbin") {
      temporary = std::filesystem::temp_directory_path() /
                  ("npu-session-" + std::to_string(GetCurrentProcessId()) +
                   "-" + std::to_string(GetTickCount64()));
      std::filesystem::create_directory(temporary);
      for (const auto* name : {"insts_4096x2560x1024_128x64x128.txt",
                               "final_4096x2560x1024_128x64x128.xclbin"})
        std::filesystem::copy_file(o.artifacts / name, temporary / name);
      o.artifacts = temporary;
      if (mode == "startup_hash") {
        std::fstream instructions(
            temporary / "insts_4096x2560x1024_128x64x128.txt",
            std::ios::in | std::ios::out | std::ios::binary);
        char byte = 0;
        instructions.read(&byte, 1);
        byte ^= 1;
        instructions.seekp(0);
        instructions.write(&byte, 1);
      } else {
        std::filesystem::remove(
            temporary / (mode == "startup_missing_insts"
                             ? "insts_4096x2560x1024_128x64x128.txt"
                             : "final_4096x2560x1024_128x64x128.xclbin"));
      }
    } else if (mode == "startup_pre_failed") {
      FailProcess("invalid NPU startup configuration");
    } else if (mode == "startup_version") {
      fake::control.runtime_mismatch = true;
    } else if (mode == "startup_runtime_missing") {
      fake::control.fail_call = "runtime.verify";
    } else {
      assert(mode == "startup_device");
      fake::control.fail_call = "device.create";
    }
    auto failed = std::make_shared<ServerPrefill>(o);
    assert(!failed->Start() && !failed->ready() && ProcessFailed());
    const auto calls = fake::control.calls;
    assert(!failed->Start() && fake::control.calls == calls);
    auto retry = std::make_shared<ServerPrefill>(o);
    assert(!retry->Start() && fake::control.calls == calls);
    assert(fake::control.allocations == 0 && fake::control.starts == 0);
    if (mode != "startup_device")
      assert(fake::control.devices == 0);
    failed->session().RejectResult("second failure must not log");
    if (!temporary.empty())
      std::filesystem::remove_all(temporary);
    gufo::server::Logger::FlushSync();
    return 0;
  }
  if (mode == "power_unavailable")
    fake::control.fail_call = "device.get_info.platform";
  if (mode == "power_turbo")
    fake::control.power_mode = "Turbo";
  auto service = std::make_shared<ServerPrefill>(o);
  auto& session = service->session();
  assert(service->Start());
  if (mode == "power_unavailable" || mode == "power_turbo") {
    assert(service->ready() && !ProcessFailed());
    assert(std::count(fake::control.calls.begin(), fake::control.calls.end(),
                      "device.get_info.platform") == 1);
    gufo::server::Logger::FlushSync();
    return 0;
  }
  std::vector<std::byte> a(kABytes);
  auto weights = std::make_shared<LinearZWeights>();
  weights->gpu_rows = 10240;
  weights->b.assign(6, std::vector<std::uint8_t>(kBBytes));
  auto& b = weights->b;
  for (const auto& w : b)
    session.PrepareWeight(std::as_bytes(std::span(w)));
  const auto allocations = fake::control.allocations;
  const auto writes = npu_test_writes.load(), flushes = npu_test_flushes.load();
  auto start = std::chrono::steady_clock::now();
  if (mode == "single") {
    for (unsigned char value : {1, 2}) {
      b[0][0] = value;
      fake::control.expected_b = value;
      auto result = session.Dispatch(a, std::as_bytes(std::span(b[0])), {});
      assert(result.result_valid && session.Healthy());
    }
  } else {
    fake_hip::Stream stream;
    std::vector<std::uint16_t> input(std::size_t{kTokens} * kK);
    std::vector<float> output(std::size_t{kTokens} * 16384);
    auto dispatch = [&](auto a, auto b, auto info) {
      return session.Dispatch(a, b, std::move(info));
    };
    auto batch = [&](auto a, const auto& b, const auto& consume) {
      return session.DispatchBatch(a, b, consume);
    };
    auto reject = [service](auto error) { service->Reject(std::move(error)); };
    auto linear = [&](unsigned layer) {
      HipLinearZTransfer transfer(service, input.data(), output.data(), kTokens,
                                  stream.get());
      auto work =
          BeginLinearZ(weights, transfer, layer, dispatch, reject, batch);
      auto result = FinishLinearZ(*work, transfer, 6144);
      assert(result.valid && result.packed && result.tail.empty());
    };
    service->BeginChunk();
    if (mode == "clean") {
      // All 834 layers are enqueued before the single chunk completion wait.
      for (unsigned i = 0; i < 834; ++i)
        linear(i);
      assert(hipStreamSynchronize(stream.get()) == hipSuccess);
      std::string error;
      assert(CheckHipPrefillChunk(*service, &error) && error.empty());
      assert(fake::control.starts == 5005 && stream.decodes == 5004);
      assert(stream.hostfuncs == 1668 && stream.syncs == 1 &&
             stream.resets == 1);
      assert(session.Healthy() && fake::control.allocations == allocations);
      unsigned a_syncs = 0, b_syncs = 0, c_syncs = 0;
      for (const auto& call : fake::control.calls) {
        if (call == "bo.sync.1.TO_DEVICE")
          ++a_syncs;
        if (call == "bo.sync.2.TO_DEVICE")
          ++b_syncs;
        if (call.ends_with(".FROM_DEVICE"))
          ++c_syncs;
      }
      assert(a_syncs == 835 && b_syncs == 1 && c_syncs == 5004);
      const auto micros = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start)
                              .count();
      std::cout << "5004 stream-ordered lean dispatches: " << micros / 5004
                << " us/dispatch (CPU fake C memset/copy included)\n";
    } else if (mode == "layer" || mode == "stream_timeout") {
      std::promise<void> entered, release;
      auto released = release.get_future().share();
      auto entered_future = entered.get_future();
      std::atomic<bool> completed{false};
      bool first = true;
      fake::control.on_call = [&](const std::string& name) {
        if (name == "run.wait" && first) {
          first = false;
          entered.set_value();
          released.wait();
          completed = true;
        }
      };
      HipLinearZTransfer transfer(service, input.data(), output.data(), kTokens,
                                  stream.get());
      auto work = BeginLinearZ(weights, transfer, 0, dispatch, reject, batch);
      assert(entered_future.wait_for(2s) == std::future_status::ready);
      auto finish = std::async(std::launch::async, [&] {
        return FinishLinearZ(*work, transfer, 6144);
      });
      const bool returned = finish.wait_for(500ms) == std::future_status::ready;
      if (!returned)
        release.set_value();
      assert(returned && !completed && finish.get().valid);
      assert(stream.syncs == 0);
      if (mode == "stream_timeout") {
        stream.check_c = false;
        // The wait must time out even though XRT still owns the session mutex.
        assert(hipStreamSynchronize(stream.get()) == hipSuccess);
        std::string error;
        assert(!CheckHipPrefillChunk(*service, &error));
        assert(error.find("NPU prefill chunk failed") != std::string::npos);
        assert(ProcessFailed() && !service->ready() && !completed);
        release.set_value();
        work->Result().wait();
        service->BeginChunk();
        assert(CheckHipPrefillChunk(*service, &error));
        session.RejectResult("second failure must not log");
      } else {
        release.set_value();
        assert(hipStreamSynchronize(stream.get()) == hipSuccess);
        std::string error;
        assert(CheckHipPrefillChunk(*service, &error));
        const auto events = stream.Events();
        auto pos = [&](const char* name) {
          return std::find(events.begin(), events.end(), name) - events.begin();
        };
        assert(pos("encode") < pos("host.begin.0"));
        assert(pos("host.end.0") < pos("host.begin.1"));
        assert(pos("host.end.1") < pos("decode"));
        assert(output[10240] == 0x42 && completed);
        assert(Stats().finish_not_ready == 1 && Stats().finish_wait_ns > 0);
        assert(Stats().hostfunc_lat_ns > 0);
        // Both projection lanes reuse the same pages and sticky invalid word.
        for (auto lane : {Lane::QueryGate, Lane::SharedGate}) {
          auto w = std::make_shared<ProjectionWeights>();
          w->lane = lane;
          w->rows = lane == Lane::QueryGate ? 1024 : 9;
          w->b.assign(lane == Lane::QueryGate ? 1 : 2, b[0]);
          const auto pitch = lane == Lane::QueryGate ? 13312U : 9U;
          HipLinearZTransfer t(service, input.data(), output.data(), kTokens,
                               stream.get(), pitch, 0,
                               lane == Lane::SharedGate
                                   ? output.data() + std::size_t{kTokens} * 9
                                   : nullptr);
          auto task = BeginProjection(w, t, 1, dispatch, reject, batch);
          assert(FinishProjection(
                     *task, t, lane == Lane::QueryGate ? w->rows : 2 * w->rows)
                     .valid);
        }
        stream.Enqueue([&] {
          stream.Record("consumer");
          assert(output[0] == 0x42);
        });
        assert(hipStreamSynchronize(stream.get()) == hipSuccess);
        assert(CheckHipPrefillChunk(*service, &error));
        assert(stream.hostfuncs == 6 && stream.resets == 1 &&
               stream.decodes == 9);
        const auto line = TakeRequestLine();
        assert(line.find("chunks=1 ") != std::string::npos);
        assert(line.find("qgate_begins=1 shared_begins=1") !=
               std::string::npos);
        assert(line.find("hostfunc_lat_ms=") != std::string::npos);
        service->BeginChunk();
        linear(0);
        assert(hipStreamSynchronize(stream.get()) == hipSuccess);
        assert(CheckHipPrefillChunk(*service, &error) && stream.resets == 2);
      }
      fake::control.on_call = {};
    } else {
      stream.check_c = mode == "invalid";
      if (mode == "timeout")
        fake::control.state = ERT_CMD_STATE_TIMEOUT;
      else if (mode == "error")
        fake::control.fail_call = "run.start";
      else {
        assert(mode == "invalid");
        stream.invalidate_next_decode = true;
      }
      linear(0);
      if (mode == "invalid")
        linear(1);
      assert(hipStreamSynchronize(stream.get()) == hipSuccess);
      std::string error;
      assert(!CheckHipPrefillChunk(*service, &error) && !error.empty());
      assert(ProcessFailed() && !service->ready() && !session.Healthy());
      const auto starts = fake::control.starts;
      const auto hostfuncs = stream.hostfuncs.load();
      unsigned gpu_fallbacks = 0;
      for (unsigned i = 0; i < 3; ++i) {
        service->BeginChunk();
        if (!service->ready())
          ++gpu_fallbacks;
        assert(CheckHipPrefillChunk(*service, &error));
      }
      assert(gpu_fallbacks == 3 && fake::control.starts == starts);
      assert(stream.hostfuncs == hostfuncs && stream.resets == 1);
      session.RejectResult("second failure must not log");
    }
  }
  assert(npu_test_writes == writes && npu_test_flushes == flushes);
  std::cout << "PASS " << mode
            << ": stream-ordered fake HIP/XRT; zero dispatch "
               "file writes/flushes; allocations="
            << allocations << "\n";
  gufo::server::Logger::FlushSync();
}
