// SPDX-License-Identifier: MIT
#include "src/npu/private_bo_adapter.hpp"

#include <cassert>
#include <iostream>

#include "fake_xrt.hpp"
using namespace gufo::npu;
int main(int argc, char** argv) {
  assert(argc >= 2);
  fake::Reset();
  fake::control.expected_a = 0;
  assert(InstructionWordCount(22032) == 5508);
  for (auto n : {0ull, 1ull, 3ull, 22033ull, 17179869184ull}) {
    bool rejected = false;
    try {
      InstructionWordCount(n);
    } catch (const AdapterError&) {
      rejected = true;
    }
    assert(rejected);
  }
  std::vector<std::byte> a(kABytes), b(kBBytes), c(kCBytes);
  auto adapter = PrivateBoAdapter::Create(
      gufo::core::ExperimentalAuthorization::FakePrefill(), argv[1], a, b);
  adapter->Submit();
  adapter->Finish();
  adapter->CopyOutput(c);
  assert(c.front() == std::byte{0x42} && c.back() == std::byte{0x42});
  bool rejected = false;
  try {
    adapter->SetInputs(std::span(a).first(1), b);
  } catch (const AdapterError&) {
    rejected = true;
  }
  assert(rejected);
  adapter.reset();
  assert(fake::control.live_bos == 0);
  std::cout << "PASS adapter sizes, ABI, output and clean lifetime\n";
}
