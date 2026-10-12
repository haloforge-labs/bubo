// SPDX-License-Identifier: MIT
#include "src/npu/staging.hpp"

#include <windows.h>

#include <algorithm>
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
  assert(IsKnob("GUFO_NPU_DIR") && !IsKnob("GUFO_NPU_TRACE"));
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
    inputs.explicit_flags.clear();
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
  assert(knobs.artifacts == "0" && knobs.slices == 6 && knobs.gate_slices == 8);
  assert(knobs.shared_expert && !knobs.isolate);
  assert(knobs.timing.timeout.count() == 8000 &&
         knobs.timing.keepalive_interval.count() == 1000);
  Reject([&] { ParseKnobs({}, values); });
  Reject([&] { ParseKnobs(token, {}); });
  for (auto directory : {std::string{}, std::string("path\0suffix", 11)}) {
    auto bad_path = values;
    bad_path["GUFO_NPU_DIR"] = directory;
    Reject([&] { ParseKnobs(token, bad_path); });
  }
  for (const auto* name :
       {"GUFO_NPU_SHEXP", "GUFO_NPU_ISOLATE", "GUFO_NPU_KEEPALIVE"}) {
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
  values["GUFO_NPU_GATE_SLICES"] = "0";
  values["GUFO_NPU_SHEXP"] = "0";
  values["GUFO_NPU_KEEPALIVE"] = "0";
  values["GUFO_NPU_ISOLATE"] = "0";
  knobs = ParseKnobs(token, values);
  assert(knobs.slices == 12 && knobs.gate_slices == 0 && !knobs.shared_expert &&
         !knobs.isolate && knobs.timing.keepalive_interval.count() == 0);
  assert(token.Allows(gufo::core::ExperimentalFeature::Prefill));
  values["GUFO_NPU_ISOLATE"] = "1";
  assert(ParseKnobs(token, values).isolate);
  std::cout << "PASS: strict knobs, DIR=0 path, ISOLATE=0 false, "
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
  std::cout << "PASS: P5 owned padded staging encodes to independent full B BO "
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
