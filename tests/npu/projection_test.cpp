// SPDX-License-Identifier: MIT
#include "src/npu/projection.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>

#include "fake_hip.hpp"
#include "fake_xrt.hpp"
#include "file_io_hook.hpp"
#include "src/npu/linear_z_hip.hpp"
using namespace gufo::npu;
using namespace std::chrono_literals;
namespace {
auto Weights(Lane lane = Lane::QueryGate, unsigned rows = 1024) {
  auto w = std::make_shared<ProjectionWeights>();
  w->lane = lane;
  w->rows = rows;
  w->b.resize(lane == Lane::QueryGate ? rows / 1024 : 2,
              std::vector<std::uint8_t>(bfp::BBytes));
  return w;
}
struct Transfer : LinearZTransfer {
  bool nonstandard_upload = false;
  unsigned groups = 1, uploads = 0;
  bool input_done = false, output_done = false, invalid = false,
       upload_error = false;
  std::vector<float> output;
  std::vector<std::vector<std::uint8_t>> FenceInput() override {
    if (invalid)
      throw std::runtime_error("encoder invalid");
    input_done = true;
    return {groups, std::vector<std::uint8_t>(bfp::ABytes)};
  }
  void FenceOutput(std::span<const float> data, unsigned cols) override {
    assert(input_done && data.size() == std::size_t{groups} * 4096 * cols);
    if (nonstandard_upload)
      throw 7;
    if (upload_error)
      throw std::runtime_error("upload fence failed");
    ++uploads;
    output.assign(data.begin(), data.end());
    output_done = true;
  }
};
NpuResult Good(unsigned slice = 0, unsigned group = 0) {
  NpuResult r{true, true, true, true, false, {}, {}};
  r.output.resize(bfp::CBytes);
  // Independent physical tiling formula; varying every token/column catches
  // query head offsets, token groups, gate/up reversal and padding reads.
  for (std::size_t g = 0; g < bfp::CBytes / 9; ++g) {
    const auto tile = g / 8192, i = g % 8192;
    const auto row =
        (tile / 8) * 512 + (i / 256) * 16 + ((i / 16) % 2) * 8 + i % 8;
    const auto col = (tile % 8) * 128 + ((i / 32) % 8) * 16 + ((i / 8) % 2) * 8;
    r.output[g * 9] = std::byte{127};
    for (unsigned j = 0; j < 8; ++j)
      r.output[g * 9 + j + 1] =
          std::byte(1 + (row + col + j + slice * 5 + group * 7) % 63);
  }
  return r;
}
template<class F>
void Reject(F f) {
  bool yes = false;
  try {
    f();
  } catch (const std::exception&) {
    yes = true;
  }
  assert(yes);
}
}  // namespace
int main(int argc, char** argv) {
  assert(argc == 2);
  {
    Transfer transfer;
    transfer.nonstandard_upload = true;
    auto work = BeginProjection(Weights(), transfer, 0,
                                [](auto, auto, auto) { return Good(); });
    auto result = FinishProjection(*work, transfer, 1024);
    assert(!result.valid && result.tail.empty() && !transfer.output_done);
    std::cout << "PASS: non-standard output callback exception rejects "
                 "projection result\n";
  }
  for (unsigned s = 1; s <= 12; ++s) {
    const auto first = s * 1024;
    assert(QueryGpuRows(first) == 13312 - first);
    unsigned counts[52]{};
    for (unsigned y = 0; y < QueryGpuRows(first) / 256; ++y)
      for (unsigned x = 0; x < 32; ++x)
        ++counts[first / 256 + y];
    for (unsigned y = 0; y < 52; ++y)
      assert(counts[y] == (y < first / 256 ? 0U : 32U));
    assert(ProjectionUploadWidth(13312, 0, false, first) == first);
  }
  for (auto first : {0U, 1U, 1023U, 1025U, 12289U, 13312U, 14336U, 0xffffffffU})
    assert(!QueryGpuRows(first));
  for (auto rows : {1U, 7U, 8U, 9U, 512U, 1024U})
    assert(ProjectionUploadWidth(rows, 0, true, 2 * rows) == rows);
  for (auto cols : {0U, 1U, 1023U, 12289U, 13312U})
    assert(!ProjectionUploadWidth(13312, 0, false, cols));
  assert(!ProjectionUploadWidth(13312, 1, false, 1024));
  assert(!ProjectionUploadWidth(13313, 0, false, 1024));
  assert(!ProjectionUploadWidth(0, 0, true, 0));
  assert(!ProjectionUploadWidth(1025, 0, true, 2050));
  assert(!ProjectionUploadWidth(512, 0, true, 1023));
  assert(!ProjectionUploadWidth(512, 1, true, 1024));
  assert(!ProjectionUploadWidth(512, 0, false, 1024));
  std::cout << "PASS: 1..12 head pairs and GPU suffix/KV coverage; "
               "query/shared upload bounds\n";
  auto w = Weights();
  for (auto n : {4096U, 8192U, 12288U, 16384U})
    assert(ProjectionQualified(n, 16384, true, false, false, false, w.get()));
  for (auto n : {0U, 1U, 4095U, 4097U, 20480U})
    assert(!ProjectionQualified(n, 32768, true, false, false, false, w.get()));
  assert(!ProjectionQualified(4096, 4095, true, false, false, false, w.get()));
  assert(!ProjectionQualified(4096, 8192, false, false, false, false, w.get()));
  assert(!ProjectionQualified(4096, 8192, true, true, false, false, w.get()));
  assert(!ProjectionQualified(4096, 8192, true, false, true, false, w.get()));
  assert(!ProjectionQualified(4096, 8192, true, false, false, true, w.get()));
  assert(!ProjectionQualified(4096, 8192, true, false, false, false, nullptr));
  for (auto lane : {Lane::LinearZ, Lane::SharedUp}) {
    auto f = Weights();
    f->lane = lane;
    assert(!ProjectionGeometry(*f));
  }
  auto zero_shared = Weights(Lane::SharedGate, 1);
  zero_shared->rows = 0;
  assert(!ProjectionGeometry(*zero_shared));
  auto high_shared = Weights(Lane::SharedGate, 1024);
  high_shared->rows = 1025;
  assert(!ProjectionGeometry(*high_shared));
  auto wrongpairs = Weights(Lane::SharedGate, 512);
  wrongpairs->b.pop_back();
  assert(!ProjectionGeometry(*wrongpairs));
  auto forged = Weights();
  forged->rows = 2048;
  assert(!ProjectionGeometry(*forged));
  std::cout << "PASS: eager full-group prefill only; "
               "speculative/prepared/last-only/capacity refusal\n";
  constexpr unsigned rb = 80 * 34;
  std::vector<std::byte> q(12288 * rb), k(512 * rb), v(512 * rb), gate(9 * rb),
      up(9 * rb);
  Q8Matrix qm{gufo::core::GgmlType::kQ8_0, 12288, 2560, q},
      km{gufo::core::GgmlType::kQ8_0, 512, 2560, k}, vm = km;
  vm.bytes = v;
  assert(!PrepareQueryGate(qm, km, vm, 0, 24, 2, 256));
  Reject([&] { PrepareQueryGate(qm, km, vm, 13, 24, 2, 256); });
  assert(!PrepareQueryGate(qm, km, vm, 1, 23, 2, 256));
  assert(!PrepareQueryGate(qm, km, vm, 1, 24, 3, 256));
  assert(!PrepareQueryGate(qm, km, vm, 1, 24, 2, 128));
  auto wrongcols = qm;
  wrongcols.cols = 2561;
  assert(!PrepareQueryGate(wrongcols, km, vm, 1, 24, 2, 256));
  auto wrongrows = qm;
  wrongrows.rows = 12287;
  assert(!PrepareQueryGate(wrongrows, km, vm, 1, 24, 2, 256));
  auto shortq = qm;
  shortq.bytes = shortq.bytes.first(shortq.bytes.size() - 1);
  assert(!PrepareQueryGate(shortq, km, vm, 1, 24, 2, 256));
  shortq = qm;
  shortq.type = gufo::core::GgmlType::kF16;
  assert(!PrepareQueryGate(shortq, km, vm, 1, 24, 2, 256));
  auto shortk = km;
  shortk.rows = 513;
  assert(!PrepareQueryGate(qm, shortk, vm, 1, 24, 2, 256));
  auto staged = PrepareQueryGate(qm, km, vm, 12, 24, 2, 256);
  assert(staged && staged->b.size() == 12 && staged->rows == 12288);
  auto saved = staged->b[0];
  q[0] = std::byte{0};
  q[1] = std::byte{0x7c};
  assert(staged->b[0] == saved);
  Reject([&] { PrepareQueryGate(qm, km, vm, 1, 24, 2, 256); });
  Q8Matrix gm{gufo::core::GgmlType::kQ8_0, 9, 2560, gate}, um = gm;
  um.bytes = up;
  auto sw = PrepareSharedExpert(gm, um, true);
  assert(sw && sw->rows == 9 && sw->b.size() == 2);
  assert(!PrepareSharedExpert(gm, um, false));
  auto bad = um;
  bad.rows = 8;
  assert(!PrepareSharedExpert(gm, bad, true));
  bad = um;
  bad.bytes = bad.bytes.first(bad.bytes.size() - 1);
  assert(!PrepareSharedExpert(gm, bad, true));
  bad = um;
  bad.rows = 1025;
  assert(!PrepareSharedExpert(gm, bad, true));
  up[1] = std::byte{0x7c};
  Reject([&] { PrepareSharedExpert(gm, um, true); });
  std::cout << "PASS: owned qualified Q8 B streams; "
               "head/type/extent/finite-scale guards and SHEXP=0\n";
  for (auto rows : {9U, 512U, 1024U}) {
    auto weights = Weights(Lane::SharedGate, rows);
    Transfer t;
    t.groups = rows == 9 ? 4 : 1;
    unsigned calls = 0;
    auto work = BeginProjection(weights, t, 47, [&](auto a, auto b, auto info) {
      assert(t.input_done && a.size() == bfp::ABytes &&
             b.size() == bfp::BBytes);
      unsigned group = calls / 2, slice = calls++ % 2;
      assert(info.layer == 47 && info.group == int(group) &&
             info.slice == int(slice));
      assert(info.lane == (slice == 0 ? "shared-gate" : "shared-up"));
      return Good(slice, group);
    });
    auto r = FinishProjection(*work, t, 2 * rows);
    assert(r.valid && t.output_done && t.uploads == 1 && calls == 2 * t.groups);
    for (unsigned g = 0; g < t.groups; ++g)
      for (unsigned token = 0; token < 4096; ++token)
        for (unsigned s = 0; s < 2; ++s)
          for (unsigned c = 0; c < rows; ++c)
            assert(t.output[(std::size_t{g} * 4096 + token) * 2 * rows +
                            s * rows + c] ==
                   float(1 + (token + c + s * 5 + g * 7) % 63) / 64);
  }
  for (auto slices : {1U, 8U, 12U}) {
    auto weights = Weights(Lane::QueryGate, slices * 1024);
    Transfer t;
    unsigned calls = 0;
    auto work = BeginProjection(weights, t, 3, [&](auto, auto, auto info) {
      assert(info.lane == "attention-q-gate" && info.layer == 3 &&
             info.group == 0 && info.slice == int(calls));
      return Good(calls++);
    });
    auto r = FinishProjection(*work, t, slices * 1024);
    assert(r.valid && t.output_done && calls == slices);
    for (unsigned token : {0U, 7U, 8U, 511U, 512U, 4095U})
      for (unsigned s = 0; s < slices; ++s)
        for (unsigned c = 0; c < 1024; ++c)
          assert(t.output[std::size_t{token} * slices * 1024 + s * 1024 + c] ==
                 float(1 + (token + c + s * 5) % 63) / 64);
  }
  std::cout << "PASS: independent C oracle, 1/8/12 query slices, shared "
               "9/512/1024 rows, four groups and lane trace tags\n";
  for (unsigned field = 0; field < 8; ++field) {
    Transfer t;
    unsigned calls = 0, rejected = 0;
    auto work = BeginProjection(
        Weights(Lane::SharedGate, 9), t, 0,
        [&](auto, auto, auto) {
          ++calls;
          auto r = Good();
          if (field == 0)
            r.submitted = false;
          if (field == 1)
            r.executed = false;
          if (field == 2)
            r.healthy = false;
          if (field == 3)
            r.result_valid = false;
          if (field == 4)
            r.owner_recovery = true;
          if (field == 5)
            r.output.clear();
          if (field == 6)
            r.output[0] = std::byte{255};
          if (field == 7)
            throw std::runtime_error("planted");
          return r;
        },
        [&](auto) { ++rejected; });
    auto r = FinishProjection(*work, t, 18);
    assert(!r.valid && r.tail.empty() && !t.uploads &&
           (calls == 1 || (field == 6 && calls == 2)) && rejected == 1);
    assert(r.owner_recovery == (field == 4));
  }
  Transfer invalid;
  invalid.invalid = true;
  unsigned calls = 0;
  Reject([&] {
    BeginProjection(w, invalid, 0, [&](auto, auto, auto) {
      ++calls;
      return Good();
    });
  });
  assert(!calls);
  Transfer upload;
  upload.upload_error = true;
  auto work =
      BeginProjection(w, upload, 0, [](auto, auto, auto) { return Good(); });
  assert(!FinishProjection(*work, upload, 1024).valid && !upload.output_done);
  assert(!work->Finish(0ms).valid);
  std::cout
      << "PASS: "
         "submission/execution/health/validity/recovery/size/decode/exception "
         "refusal; stop slices, reject and input/output fences\n";
  auto a = std::vector<std::vector<std::uint8_t>>(
      1, std::vector<std::uint8_t>(bfp::ABytes));
  Reject([&] {
    ProjectionWork x({}, a, 0, [](auto, auto, auto) { return Good(); });
  });
  Reject([&] {
    ProjectionWork x(forged, a, 0, [](auto, auto, auto) { return Good(); });
  });
  Reject([&] {
    ProjectionWork x(w, {}, 0, [](auto, auto, auto) { return Good(); });
  });
  Reject([&] {
    ProjectionWork x(w, {5, std::vector<std::uint8_t>(bfp::ABytes)}, 0,
                     [](auto, auto, auto) { return Good(); });
  });
  Reject([&] {
    ProjectionWork x(w, {{1}}, 0, [](auto, auto, auto) { return Good(); });
  });
  Reject([&] { ProjectionWork x(w, a, 0, {}); });
  auto shortb = Weights();
  shortb->b[0].pop_back();
  Reject([&] {
    ProjectionWork x(shortb, a, 0, [](auto, auto, auto) { return Good(); });
  });
  std::atomic<bool> entered = false, release = false, done = false;
  auto owned = Weights();
  std::weak_ptr<const ProjectionWeights> weak = owned;
  auto cap = std::make_shared<int>(7);
  std::weak_ptr<int> weakcap = cap;
  {
    ProjectionWork x(owned, std::move(a), 0,
                     [cap, &entered, &release, &done](auto aa, auto bb, auto) {
                       entered = true;
                       while (!release)
                         std::this_thread::sleep_for(1ms);
                       assert(aa.size() == bfp::ABytes &&
                              bb.size() == bfp::BBytes && *cap == 7);
                       done = true;
                       return Good();
                     });
    while (!entered)
      std::this_thread::sleep_for(1ms);
    owned.reset();
    cap.reset();
    assert(!x.Finish(1ms).valid);
  }
  assert(!weak.expired() && !weakcap.expired());
  release = true;
  while (!done)
    std::this_thread::sleep_for(1ms);
  for (unsigned i = 0; i < 5000 && (!weak.expired() || !weakcap.expired()); ++i)
    std::this_thread::sleep_for(1ms);
  assert(weak.expired() && weakcap.expired());
  std::cout << "PASS: forged work/input bounds; bounded deadline and worker "
               "ownership after caller timeout/destruction\n";
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
    std::vector<std::uint16_t> input(std::size_t{kTokens} * kK);
    std::vector<float> output(std::size_t{kTokens} * 13312),
        second(kTokens * 9);
    for (auto lane : {Lane::QueryGate, Lane::SharedGate}) {
      const auto rows = lane == Lane::QueryGate ? 12288U : 9U;
      const auto pitch = lane == Lane::QueryGate ? 13312U : 9U;
      auto t = std::make_unique<HipLinearZTransfer>(
          service, input.data(), output.data(), kTokens, stream.get(), pitch, 0,
          lane == Lane::SharedGate ? second.data() : nullptr);
      std::promise<void> submitted, complete;
      auto completion = complete.get_future().share();
      auto submission = submitted.get_future();
      auto task = BeginProjection(
          Weights(lane, rows), *t, 0,
          [](auto, auto, auto) {
            std::abort();
            return NpuResult{};
          },
          {},
          [&](auto a, const auto& b, const auto& consume) {
            assert(a.size() == bfp::ABytes);
            submitted.set_value();
            completion.wait();
            std::vector<std::byte> c(bfp::CBytes, std::byte{0x42});
            for (unsigned s = 0; s < b.size(); ++s)
              consume(s, c);
            return true;
          });
      assert(submission.wait_for(2s) == std::future_status::ready);
      auto finish = std::async(std::launch::async, [&] {
        return FinishProjection(*task, *t,
                                lane == Lane::QueryGate ? rows : 2 * rows);
      });
      const bool returned = finish.wait_for(500ms) == std::future_status::ready;
      if (!returned)
        complete.set_value();
      assert(returned && finish.get().valid);
      task.reset();
      t.reset();
      complete.set_value();
      stream.Enqueue([&] {
        assert(output[0] == 0x42);
        if (lane == Lane::QueryGate)
          assert(output[11 * kN] == 0x42);
        else
          assert(second[0] == 0x42);
      });
      assert(hipStreamSynchronize(stream.get()) == hipSuccess);
      std::string error;
      assert(CheckHipPrefillChunk(*service, &error));
    }
    assert(stream.hostfuncs == 4 && stream.decodes == 14 && stream.resets == 1);
    std::cout
        << "PASS: q/gate twelve slices and shared partial gate/up consume "
           "mapped C after stream wait; both Finish calls return early\n";
  }
}
