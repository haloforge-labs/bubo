// SPDX-License-Identifier: MIT
#include "src/npu/staging.hpp"

#include <windows.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "fake_xrt.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/npu/bfp.hpp"

using namespace gufo::npu;
static_assert(bfp::ABytes == kABytes && bfp::BBytes == kBBytes &&
              bfp::CBytes == kCBytes);
namespace {
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}
constexpr std::size_t row_bytes = 80 * 34;
std::vector<std::byte> Fixture(unsigned rows) {
  std::vector<std::byte> bytes(rows * row_bytes);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned block = 0; block < 80; ++block) {
      const auto base = row * row_bytes + block * 34;
      bytes[base] = std::byte{0};
      bytes[base + 1] = std::byte{0x3c};
      for (unsigned q = 0; q < 32; ++q)
        bytes[base + 2 + q] = std::byte((row + block + q) % 256);
      bytes[base + 2] = std::byte(row & 255);
      bytes[base + 3] = std::byte(row >> 8);
    }
  }
  return bytes;
}
}  // namespace
int main() {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
  fake::Reset();
  assert(IsKnob("GUFO_NPU_DIR"));
  assert(!IsKnob("GUFO_NPU_") && !IsKnob("GUFO_NPU_LAYERS"));
  // Actual resolver precedence, without model/header/DLL/device activity.
  for (unsigned origin = 0; origin < 3; ++origin) {
    gufo::server::ReleaseInputs inputs;
    inputs.model_family = "qwen3.8-flash-next";
    inputs.explicit_flags["--npu-prefill"] = "on";
    if (origin != 1) {
      inputs.environment["GUFO_NPU_DIR"] = "0";
      inputs.environment["GUFO_NPU_SLICES"] = "12";
    }
    if (origin != 0) {
      inputs.manifest_defaults["GUFO_NPU_DIR"] = "manifest-artifacts";
      inputs.manifest_defaults["GUFO_NPU_SLICES"] = "1";
    }
    auto resolved = gufo::server::ResolveReleaseSettings(inputs);
    std::map<std::string, std::string> selected;
    for (const auto& setting : resolved.settings)
      if (IsKnob(setting.name) && setting.value)
        selected[setting.name] = *setting.value;
    auto parsed = ParseKnobs(resolved.prefill_authorization, selected);
    assert(parsed.slices == (origin == 1 ? 1u : 12u));
    assert(parsed.artifacts == (origin == 1 ? "manifest-artifacts" : "0"));
    inputs.explicit_flags["--npu-prefill"] = "off";
    inputs.environment["GUFO_NPU_SLICES"] = "malformed";
    resolved = gufo::server::ResolveReleaseSettings(inputs);
    bool called = false;
    assert(!gufo::core::CreateAuthorizedPrefill(
        resolved.prefill_authorization, [&](auto token) {
          called = true;
          (void)ParseKnobs(token, selected);
        }));
    assert(!called);
  }
  const auto token = gufo::core::ExperimentalAuthorization::FakePrefill();
  for (const auto* malformed : {"01", "00", "2", "-1", "+1", "1x", " 1"}) {
    Reject([&] { ParseTiming("", malformed, ""); });
  }
  std::map<std::string, std::string> values{{"GUFO_NPU_DIR", "0"}};
  auto knobs = ParseKnobs(token, values);
  assert(knobs.artifacts == "0" && knobs.slices == 6 && knobs.gate_slices == 0);
  assert(knobs.shared_expert && knobs.hal_monitor);  // HAL monitor default on
  assert(IsKnob("GUFO_NPU_HAL_MONITOR") && IsKnob("GUFO_NPU_CHIP_POWER"));
  assert(knobs.chip_power);  // chip power logging default on
  assert(IsKnob("GUFO_NPU_MEM_TRACE") && knobs.mem_trace);
  // G1 guard: on|off|log (default on), strict; and the internal test override.
  assert(IsKnob("GUFO_NPU_GUARD") && IsKnob("GUFO_NPU_GUARD_TEST"));
  assert(knobs.guard == GuardMode::kOn && !knobs.guard_test.power_w &&
         !knobs.guard_test.hold_s);
  for (const auto& [text, mode] :
       {std::pair<const char*, GuardMode>{"on", GuardMode::kOn},
        {"off", GuardMode::kOff},
        {"log", GuardMode::kLog}}) {
    auto guarded = values;
    guarded["GUFO_NPU_GUARD"] = text;
    assert(ParseKnobs(token, guarded).guard == mode);
  }
  for (const auto* bad : {"", "ON", "1", "0", "yes", "on ", "enforce"}) {
    auto invalid = values;
    invalid["GUFO_NPU_GUARD"] = bad;
    Reject([&] { ParseKnobs(token, invalid); });
  }
  {
    auto tested = values;
    tested["GUFO_NPU_GUARD_TEST"] = "power_w=30,hold_s=60";
    const auto parsed = ParseKnobs(token, tested);
    assert(parsed.guard_test.power_w == 30 && parsed.guard_test.hold_s == 60);
  }
  for (const auto* bad : {"power_w=x", "volts=3", "hold_s=-1", "power_w"}) {
    auto invalid = values;
    invalid["GUFO_NPU_GUARD_TEST"] = bad;
    Reject([&] { ParseKnobs(token, invalid); });
  }
  // L-REL-2: the token-count part of the lane qualification, shared with the
  // linear and projection lanes and with the backup decision in Session::Feed.
  assert(QualifiedChunk(4096, 4096) && QualifiedChunk(8192, 8192) &&
         QualifiedChunk(16384, 16384));
  assert(!QualifiedChunk(100, 4096) && !QualifiedChunk(1024, 4096) &&
         !QualifiedChunk(4095, 4096) && !QualifiedChunk(4097, 8192) &&
         !QualifiedChunk(8192, 4096) && !QualifiedChunk(20480, 20480) &&
         !QualifiedChunk(0, 4096));
  // L-REL-2 internal failure injection: a chunk number, strict.
  assert(IsKnob("GUFO_NPU_TEST_FAIL_CHUNK") && !knobs.test_fail_chunk);
  for (const auto* value : {"0", "1", "12"}) {
    auto failing = values;
    failing["GUFO_NPU_TEST_FAIL_CHUNK"] = value;
    assert(ParseKnobs(token, failing).test_fail_chunk ==
           std::stoull(std::string(value)));
  }
  for (const auto* bad : {"", "-1", "x", "1.5", "1 ", "1000001"}) {
    auto invalid = values;
    invalid["GUFO_NPU_TEST_FAIL_CHUNK"] = bad;
    Reject([&] { ParseKnobs(token, invalid); });
  }
  for (const auto* value : {"0", "1"}) {
    auto traced = values;
    traced["GUFO_NPU_MEM_TRACE"] = value;
    assert(ParseKnobs(token, traced).mem_trace == (std::string(value) == "1"));
  }
  for (const auto* value : {"0", "1"}) {
    auto powered = values;
    powered["GUFO_NPU_CHIP_POWER"] = value;
    assert(ParseKnobs(token, powered).chip_power ==
           (std::string(value) == "1"));
  }
  for (const auto* value : {"0", "1"}) {
    auto monitored = values;
    monitored["GUFO_NPU_HAL_MONITOR"] = value;
    assert(ParseKnobs(token, monitored).hal_monitor ==
           (std::string(value) == "1"));
  }
  assert(knobs.timing.timeout.count() == 8000 &&
         knobs.timing.keepalive_interval.count() == 1000);
  Reject([&] { ParseKnobs({}, values); });
  const auto packaged = ParseKnobs(token, {});
  std::array<wchar_t, 32768> executable{};
  const auto length = GetModuleFileNameW(nullptr, executable.data(),
                                         static_cast<DWORD>(executable.size()));
  assert(length && length < executable.size());
  assert(packaged.artifacts ==
         (std::filesystem::path(executable.data()).parent_path() /
          ".." / "tools" / "npu").lexically_normal());
  assert(packaged.slices == 6 && packaged.gate_slices == 0 &&
         packaged.shared_expert);
  for (auto directory : {std::string{}, std::string("path\0suffix", 11)}) {
    auto bad_path = values;
    bad_path["GUFO_NPU_DIR"] = directory;
    Reject([&] { ParseKnobs(token, bad_path); });
  }
  for (const auto* name :
       {"GUFO_NPU_SHEXP", "GUFO_NPU_KEEPALIVE", "GUFO_NPU_HAL_MONITOR",
        "GUFO_NPU_CHIP_POWER", "GUFO_NPU_MEM_TRACE"}) {
    for (const auto* bad :
         {"", "01", "00", "2", "-1", "+1", "1x", " 1", "1 ", "on"}) {
      auto invalid = values;
      invalid[name] = bad;
      Reject([&] { ParseKnobs(token, invalid); });
    }
  }
  for (const auto* name : {"GUFO_NPU_SLICES", "GUFO_NPU_GATE_SLICES"}) {
    for (const auto* bad :
         {"", "13", "-1", "+1", "1x", " 1", "1 ", "999999999999999999999"}) {
      auto invalid = values;
      invalid[name] = bad;
      Reject([&] { ParseKnobs(token, invalid); });
    }
  }
  for (const auto* name : {"GUFO_NPU_TIMEOUT_MS", "GUFO_NPU_KEEPALIVE_MS"}) {
    for (const auto* bad :
         {"", "-1", "+1", "1x", " 1", "999999999999999999999"}) {
      auto invalid = values;
      invalid[name] = bad;
      Reject([&] { ParseKnobs(token, invalid); });
    }
  }
  auto invalid = values;
  invalid["GUFO_NPU_SLICES"] = "0";
  Reject([&] { ParseKnobs(token, invalid); });
  for (const auto* bad : {"0", "8001"}) {
    invalid = values;
    invalid["GUFO_NPU_TIMEOUT_MS"] = bad;
    Reject([&] { ParseKnobs(token, invalid); });
  }
  invalid = values;
  invalid["GUFO_NPU_KEEPALIVE_MS"] = "99";
  Reject([&] { ParseKnobs(token, invalid); });
  invalid["GUFO_NPU_KEEPALIVE_MS"] = "60001";
  Reject([&] { ParseKnobs(token, invalid); });
  values["GUFO_NPU_SLICES"] = "12";
  for (unsigned slices = 0; slices <= 12; ++slices) {
    values["GUFO_NPU_GATE_SLICES"] = std::to_string(slices);
    assert(ParseKnobs(token, values).gate_slices == slices);
  }
  values["GUFO_NPU_GATE_SLICES"] = "0";
  values["GUFO_NPU_SHEXP"] = "0";
  values["GUFO_NPU_KEEPALIVE"] = "0";
  knobs = ParseKnobs(token, values);
  assert(knobs.slices == 12 && knobs.gate_slices == 0 && !knobs.shared_expert &&
         knobs.timing.keepalive_interval.count() == 0);
  assert(token.Allows(gufo::core::ExperimentalFeature::Prefill));
  std::cout << "PASS: strict knobs, DIR=0 path, "
               "lane/keepalive controls retain authorization\n";

  auto linear_bytes = Fixture(16384);
  Q8Matrix linear{gufo::core::GgmlType::kQ8_0, 16384, 2560, linear_bytes};
  for (unsigned count = 1; count <= 12; ++count) {
    auto plan = LinearPlan(linear, count);
    assert(plan.size() == count && plan.front().row0 == 16384 - count * 1024);
    assert(plan.back().row0 == 15360);
  }
  auto staged = StageQ8(linear, LinearPlan(linear, 6).front());
  const auto source_start = (16384 - 6144) * row_bytes;
  assert(std::equal(staged.begin(), staged.end(),
                    linear_bytes.begin() + source_start));
  linear_bytes[source_start + 2] = std::byte{77};
  assert(staged[2] != std::byte{77});  // independent lifetime
  auto gate_bytes = Fixture(13312);
  Q8Matrix gate{gufo::core::GgmlType::kQ8_0, 13312, 2560, gate_bytes};
  auto gate_plan = GatePlan(gate, 12, 24, 2, 256);
  assert(gate_plan.size() == 12 && gate_plan.back().first_head == 22 &&
         gate_plan.back().head_count == 2 &&
         gate_plan.back().row0 + 1024 == 12288);
  assert(GatePlan({}, 0, 0, 0, 0).empty());
  auto shared_bytes = Fixture(768);
  Q8Matrix shared{gufo::core::GgmlType::kQ8_0, 768, 2560, shared_bytes};
  auto shared_plan = SharedPlan(shared, shared, true);
  staged = StageQ8(shared, shared_plan.front());
  assert(std::equal(shared_bytes.begin(), shared_bytes.end(), staged.begin()));
  assert(std::all_of(staged.begin() + shared_bytes.size(), staged.end(),
                     [](auto byte) { return byte == std::byte{0}; }));
  assert(SharedPlan({}, {}, false).empty());
  const auto bfp_weights = bfp::EncodeB(staged);
  assert(bfp_weights.size() == bfp::BBytes);
  for (std::size_t g = 0; g < bfp::BBytes / 9; ++g) {
    if (bfp::Position(bfp::Layout::B, g).col >= 768)
      assert(std::all_of(bfp_weights.begin() + g * 9,
                         bfp_weights.begin() + g * 9 + 9,
                         [](auto byte) { return byte == 0; }));
  }
  std::cout << "PASS: owned padded staging encodes to independent full B BO "
               "bytes\n";
  Reject([&] { LinearPlan(linear, 0); });
  Reject([&] { LinearPlan(linear, 13); });
  Reject([&] { GatePlan(gate, 13, 24, 2, 256); });
  Reject([&] { GatePlan(gate, 8, 16, 2, 256); });
  Reject([&] { GatePlan(gate, 8, 24, 2, 128); });
  Reject([&] { GatePlan(gate, 8, 24, 4, 256); });
  Reject([&] { LinearPlan(gate, 6); });
  auto bad = linear;
  bad.rows = 16383;
  Reject([&] { LinearPlan(bad, 6); });
  bad = linear;
  bad.cols = 2559;
  Reject([&] { LinearPlan(bad, 6); });
  bad = linear;
  bad.type = gufo::core::GgmlType::kF16;
  Reject([&] { LinearPlan(bad, 6); });
  bad = linear;
  bad.bytes = bad.bytes.first(bad.bytes.size() - 1);
  Reject([&] { LinearPlan(bad, 6); });
  bad = linear;
  bad.rows = std::numeric_limits<std::uint64_t>::max();
  Reject([&] { LinearPlan(bad, 6); });
  bad = shared;
  bad.rows = 767;
  Reject([&] { SharedPlan(shared, bad, true); });
  Reject([&] { SharedPlan(linear, linear, true); });
  Reject([&] { StageQ8(linear, {Lane::LinearZ, 16384, 1024}); });
  Reject([&] { StageQ8(linear, {Lane::LinearZ, UINT32_MAX, 1024}); });
  Reject([&] { StageQ8(linear, {Lane::LinearZ, 0, 1024}); });
  Reject([&] { StageQ8(gate, {Lane::QueryGate, 12288, 1024, 24, 2}); });
  Reject([&] { StageQ8(gate, {Lane::QueryGate, 0, 1024, 0, 1}); });
  shared_bytes[1] = std::byte{0x7c};
  Reject([&] { StageQ8(shared, shared_plan.front()); });
  assert(!QualifiedPrefill(0, 4) && !QualifiedPrefill(1, 4) &&
         !QualifiedPrefill(4095, 4) && !QualifiedPrefill(4097, 4) &&
         !QualifiedPrefill(4096, 0) && !QualifiedPrefill(20480, 4) &&
         !QualifiedPrefill(UINT64_MAX, 4) && QualifiedPrefill(16384, 4));
  std::cout << "PASS: Q8 row copies, owned storage, shared zero padding, "
               "two-head slices, K/V exclusion, shape/size/scale/overflow "
               "rejects, prefill capacity\n";
  assert(fake::control.devices == 0 && fake::control.starts == 0);
  std::cout << "PASS: actual resolver env/manifest/both precedence; malformed "
               "OFF knobs never parsed; fake device/start counters zero\n";
}
