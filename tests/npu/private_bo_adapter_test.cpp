// SPDX-License-Identifier: MIT
#include "src/npu/private_bo_adapter.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fake_xrt.hpp"
#include "src/npu/bo_address_table.hpp"
#include "xrt/xrt_kernel.h"
using namespace gufo::npu;
namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
}  // namespace
int main(int argc, char** argv) {
  assert(argc >= 2);
  std::cout << std::unitbuf;  // keep progress visible if an assertion aborts
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

  // S0c-1: the weights map is keyed by the vector's data pointer only.
  const std::filesystem::path artifacts = argv[1];
  const auto create = [&] {
    return PrivateBoAdapter::Create(
        gufo::core::ExperimentalAuthorization::FakePrefill(), artifacts, a, b);
  };
  const auto consume = [](std::size_t, std::span<const std::byte>) {};
  const auto make = [](unsigned char first) {
    return std::vector<std::vector<std::uint8_t>>(
        1, std::vector<std::uint8_t>(kBBytes, first));
  };
  {
    // Current behaviour (guard off): same address, new bytes. ExecuteBatch
    // finds the address in the map and does NOT re-send. The fake device checks
    // the bytes in the weight buffer, so the stale buffer is caught here; on
    // real hardware the NPU would silently multiply by the old weights.
    fake::Reset();
    fake::control.expected_a = 0;
    auto stale = create();
    auto weights = make(0x11);
    fake::control.expected_b = 0x11;
    stale->ExecuteBatch(a, weights, consume);
    weights[0].assign(kBBytes, 0x22);  // different weights, same data pointer
    fake::control.expected_b = 0x22;
    bool stale_used = false;
    try {
      stale->ExecuteBatch(a, weights, consume);
    } catch (const std::runtime_error& e) {
      stale_used =
          std::string(e.what()).find("B must match") != std::string::npos;
    }
    assert(stale_used && stale->StaleWeightRepairs() == 0);
    std::cout << "S0c current behaviour: reused address keeps the OLD weight "
                 "buffer (no re-send) unless PrepareWeight is called\n";
  }
  {
    // PrepareWeight on a known address does refresh the buffer.
    fake::Reset();
    fake::control.expected_a = 0;
    auto adapter2 = create();
    auto weights = make(0x11);
    fake::control.expected_b = 0x11;
    adapter2->ExecuteBatch(a, weights, consume);
    weights[0].assign(kBBytes, 0x22);
    adapter2->PrepareWeight(std::as_bytes(std::span(weights[0])));
    fake::control.expected_b = 0x22;
    adapter2->ExecuteBatch(a, weights, consume);
  }
  {
    // Guard on (GUFO_NPU_WEIGHT_GUARD=1): changed first/last 4 KiB is detected
    // and the buffer is re-sent; unchanged bytes are not re-sent.
    _putenv_s("GUFO_NPU_WEIGHT_GUARD", "1");
    fake::Reset();
    fake::control.expected_a = 0;
    auto guarded = create();
    auto weights = make(0x11);
    fake::control.expected_b = 0x11;
    guarded->ExecuteBatch(a, weights, consume);
    guarded->ExecuteBatch(a, weights, consume);
    assert(guarded->StaleWeightRepairs() == 0);
    weights[0].assign(kBBytes, 0x22);
    fake::control.expected_b = 0x22;
    guarded->ExecuteBatch(a, weights, consume);
    assert(guarded->StaleWeightRepairs() == 1);
    guarded->ExecuteBatch(a, weights, consume);
    assert(guarded->StaleWeightRepairs() == 1);
    // A change only in the middle is outside the fingerprint (documented
    // limit).
    weights[0][kBBytes / 2] = 0x33;
    guarded->ExecuteBatch(a, weights, consume);
    assert(guarded->StaleWeightRepairs() == 1);
    _putenv_s("GUFO_NPU_WEIGHT_GUARD", "");
    std::cout << "PASS S0c weight guard (first/last 4 KiB fingerprint)\n";
  }

  // S0c-2: 4-run ring. Slice k starts only after slice k-4 was reaped; slices
  // are reaped (consumed) in order; every start is paired with one wait.
  for (const std::size_t count : {1u, 3u, 4u, 5u, 12u}) {
    fake::Reset();
    fake::control.expected_a = 0;
    auto ring = create();
    std::vector<std::vector<std::uint8_t>> weights(
        count, std::vector<std::uint8_t>(kBBytes));
    fake::control.expected_b = 0;
    int starts_seen = 0, max_outstanding = 0;
    fake::control.on_call = [&](const std::string& name) {
      if (name == "run.start") {
        const int outstanding = fake::control.starts - fake::control.waits;
        assert(outstanding < 4);  // slot reuse needs a prior reap
        if (starts_seen >= 4)
          assert(fake::control.waits >= starts_seen - 3);
        max_outstanding = std::max(max_outstanding, outstanding + 1);
        ++starts_seen;
      }
    };
    std::vector<std::size_t> order;
    ring->ExecuteBatch(
        a, weights, [&](std::size_t slice, std::span<const std::byte> c) {
          order.push_back(slice);
          assert(c.size() == kCBytes && c.front() == std::byte{0x42});
        });
    assert(order.size() == count && starts_seen == static_cast<int>(count));
    for (std::size_t i = 0; i < order.size(); ++i)
      assert(order[i] == i);  // in-order reaping
    assert(fake::control.starts == fake::control.waits &&
           max_outstanding ==
               static_cast<int>(std::min<std::size_t>(count, 4)));
    assert(!ring->DmaPossible() && !ring->Failed());
    fake::control.on_call = {};
  }
  {
    // Geometry limits and a failed completion leave the adapter failed with DMA
    // possibly outstanding (pages are kept until process exit).
    fake::Reset();
    fake::control.expected_a = 0;
    auto bad = create();
    std::vector<std::vector<std::uint8_t>> none,
        too_many(13, std::vector<std::uint8_t>(kBBytes));
    bool rejected_empty = false, rejected_many = false;
    try {
      bad->ExecuteBatch(a, none, consume);
    } catch (const AdapterError&) {
      rejected_empty = true;
    }
    try {
      bad->ExecuteBatch(a, too_many, consume);
    } catch (const AdapterError&) {
      rejected_many = true;
    }
    assert(rejected_empty && rejected_many);
    auto weights = make(0);
    fake::control.expected_b = 0;
    fake::control.state = ERT_CMD_STATE_TIMEOUT;
    bool timed_out = false;
    try {
      bad->ExecuteBatch(a, weights, consume);
    } catch (const AdapterError& e) {
      timed_out = e.kind == ErrorKind::Timeout;
    }
    assert(timed_out && bad->Failed() && bad->DmaPossible());
    bool refused = false;
    try {
      bad->ExecuteBatch(a, weights, consume);
    } catch (const AdapterError& e) {
      refused = e.kind == ErrorKind::State;
    }
    assert(refused);
    std::cout << "PASS S0c 4-run ring invariants and failure state\n";
  }

  // BO device-address table (fake XRT hands out known, back-to-back addresses).
  {
    fake::Reset();
    fake::control.expected_a = 0;
    ResetBoRegistryForTest();
    auto adapter3 = create();
    const auto table = CurrentBoTable();
    assert(table && table->entries.size() == 12 && BoTableGeneration() == 1);
    std::map<std::string, int> kinds;
    for (const auto& e : table->entries) {
      ++kinds[e.kind];
      assert(e.known && e.address >= 0x2D000000ULL && e.size > 0);
    }
    assert(kinds["instr"] == 1 && kinds["a"] == 1 && kinds["b"] == 1 &&
           kinds["c"] == 1 && kinds["scratch"] == 2 &&
           kinds["heartbeat_a"] == 1 && kinds["heartbeat_b"] == 1 &&
           kinds["output"] == 4);
    // The A BO is the second BO created: resolve inside it, by name.
    const BoEntry* a_bo = nullptr;
    for (const auto& e : table->entries)
      if (std::string(e.kind) == "a")
        a_bo = &e;
    assert(a_bo && a_bo->size == kABytes);
    assert(ResolveText(*table, a_bo->address + 5) == "resolves=a#0+0x5");
    // Weights: PrepareWeight adds weight#0.. and republishes the snapshot.
    std::vector<std::vector<std::uint8_t>> w3(
        3, std::vector<std::uint8_t>(kBBytes));
    const auto before = BoTableGeneration();
    for (auto& w : w3)
      adapter3->PrepareWeight(std::as_bytes(std::span(w)));
    const auto with_weights = CurrentBoTable();
    assert(BoTableGeneration() == before + 3 &&
           with_weights->entries.size() == 15);
    assert(Has(FormatBoTable(*with_weights), "weight n=3 min=") &&
           Has(FormatBoTable(*with_weights), "weight#0-2 0x"));
    // The earlier snapshot is immutable: still 12 entries.
    assert(table->entries.size() == 12);
    // A weight created lazily (unseen pointer in ExecuteBatch) updates the
    // table.
    auto lazy = make(0);
    fake::control.expected_b = 0;
    adapter3->ExecuteBatch(a, lazy, consume);
    const auto after_lazy = CurrentBoTable();
    assert(after_lazy->entries.size() == 16 &&
           Has(FormatBoTable(*after_lazy), "weight#0-3 0x"));
    // Re-sending a known weight creates no new entry.
    adapter3->PrepareWeight(std::as_bytes(std::span(w3[0])));
    assert(CurrentBoTable()->entries.size() == 16);
    std::cout << "BO-ADDRESS-LINE: " << FormatBoTable(*after_lazy) << "\n";
  }
  {  // addresses that cannot be read are "na", never an error
    for (const bool zero : {true, false}) {
      fake::Reset();
      fake::control.expected_a = 0;
      ResetBoRegistryForTest();
      fake::control.address_zero = zero;
      fake::control.address_throw = !zero;
      auto adapter4 = create();
      std::vector<std::uint8_t> w(kBBytes);
      adapter4->PrepareWeight(std::as_bytes(std::span(w)));
      const auto na = CurrentBoTable();
      assert(na && na->entries.size() == 13);
      for (const auto& e : na->entries)
        assert(!e.known);
      assert(ResolveText(*na, 0x2DEF0000) == "resolves=na");
      assert(Has(FormatBoTable(*na), "weight n=1 na=1 runs=[]"));
    }
    ResetBoRegistryForTest();
  }
  std::cout << "PASS BO address table: fixed, weights, lazy, na\n";
}
