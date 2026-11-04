#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

qfn::PrefillDiagPass MakePass(bool moe_bench) {
  qfn::PrefillDiagPass pass;
  pass.pass = 3;
  pass.start = 4096;
  pass.totals.tokens = 4096;
  pass.totals.moe_bench = moe_bench;
  pass.layers.push_back({0, 4096, 250, 1000});
  pass.layers.push_back({1, 4096, 251, 2000});
  pass.layers[0].moe_us = 123.4;
  pass.layers[0].gate = 18;
  pass.layers[0].up = 18;
  pass.layers[0].down = 20;
  pass.layers[0].route = 0;
  pass.layers[1].moe_us = 456.5;
  pass.layers[1].gate = 21;
  pass.layers[1].up = 21;
  pass.layers[1].down = 42;
  pass.layers[1].route = 2;
  const double phase0[4] = {10.4, 60.2, 50.1, 2.7}, phase1[4] = {0, 300.4, 150.3, 6.2};
  for (int i = 0; i < 4; ++i) {
    pass.layers[0].phase_us[i] = phase0[i];
    pass.layers[1].phase_us[i] = phase1[i];
  }
  pass.layers[0].gap_us = 40.4;
  pass.layers[0].host_wall_us = 2080.4;
  pass.layers[1].gap_us = 31.6;
  pass.layers[1].host_wall_us = 3099.5;
  return pass;
}

std::string LayersOf(const std::string& line) {
  return line.substr(line.find(" layers=") + 8);
}
}  // namespace

int main() {
  try {
    // GUFO_PREFILL_DIAG alone (moe_bench=false): the old four-field format.
    const auto plain = qfn::PrefillDiagLine(MakePass(false));
    Require(LayersOf(plain) == "0:4096:250:1000,1:4096:251:2000",
            "moe_bench=false layers changed");
    Require(plain.find("moe_us=") == std::string::npos,
            "moe_bench=false printed bench fields");

    // GUFO_MOE_BENCH: fourteen fields after the layer id: us, three types,
    // route, four phase times, the GPU gap before the span and the RouteHints
    // host wall time.
    const auto bench = qfn::PrefillDiagLine(MakePass(true));
    const auto layers = LayersOf(bench);
    Require(layers == "0:4096:250:1000:123:18:18:20:wmma:10:60:50:3:40:2080,"
                      "1:4096:251:2000:457:21:21:42:mmq:0:300:150:6:32:3100",
            "moe_bench=true layers mismatch");
    std::istringstream tuples(layers);
    std::string tuple;
    int count = 0;
    while (std::getline(tuples, tuple, ',')) {
      std::size_t colons = 0;
      for (char c : tuple)
        colons += c == ':';
      Require(colons == 14, "tuple must have 14 fields after the layer id");
      ++count;
    }
    Require(count == 2, "tuple count mismatch");
    Require(qfn::PrefillDiagRouteName(1) == std::string("lanerow"),
            "route name mismatch");
    // GUFO_ROUTED_WMMA_IQ=2: route 3 is the register-fed IQ pair kernel; the
    // other codes keep their names.
    Require(qfn::PrefillDiagRouteName(0) == std::string("wmma") &&
                qfn::PrefillDiagRouteName(2) == std::string("mmq") &&
                qfn::PrefillDiagRouteName(3) == std::string("iq-pair2"),
            "iq-pair2 route name mismatch");
    auto pair2 = MakePass(true);
    pair2.layers[0].route = 3;
    Require(LayersOf(qfn::PrefillDiagLine(pair2))
                    .rfind("0:4096:250:1000:123:18:18:20:iq-pair2:", 0) == 0,
            "iq-pair2 tuple mismatch");
    // Its layer counters print only when set, so value 0/1 lines are unchanged.
    Require(bench.find("moe_iq_pair2") == std::string::npos,
            "pair2 counters printed without pair2 layers");
    pair2.totals.moe_iq_pair2_layers = 20;
    pair2.totals.moe_iq_pair2_fallback_layers = 28;
    Require(qfn::PrefillDiagLine(pair2).find(
                " moe_iq_pair2_layers=20 moe_iq_pair2_fallback_layers=28") !=
                std::string::npos,
            "pair2 counters missing");
    // Totals aggregate the phase split, the gap and the host wall time.
    qfn::PrefillDiagTotals sum, one;
    one.moe_bench = true;
    one.moe_phase_us[1] = 5;
    one.moe_gap_us = 7;
    one.moe_host_wall_us = 9;
    one.moe_iq_pair2_layers = 3;
    one.moe_iq_pair2_fallback_layers = 4;
    sum.Add(one);
    sum.Add(one);
    Require(sum.moe_phase_us[1] == 10 && sum.moe_gap_us == 14 &&
                sum.moe_host_wall_us == 18,
            "phase totals not aggregated");
    Require(sum.moe_iq_pair2_layers == 6 && sum.moe_iq_pair2_fallback_layers == 8,
            "pair2 counters not aggregated");
    std::cout << "PASS prefill-diag layers: byte-identical without moe_bench, "
                 "14 fields after the layer id with it, phase totals add up\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
