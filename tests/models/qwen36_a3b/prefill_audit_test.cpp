#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen36_a3b/prefill_failure.hpp"
#include "src/models/qwen38_flash_next/routed_expert_audit.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace a3b = gufo::models::qwen36_a3b;
void Check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  using qfn::ObserveRoutedExpertRow;
  using qfn::RoutedExpertDroppedRows;
  const auto vector = RoutedExpertDroppedRows::kZeroOrUntouched;
  for (auto bits : {0u, 0x80000000u}) {
    const auto row = ObserveRoutedExpertRow(false, false, bits, vector);
    Check(row.executed == 0 && row.invalid == 0, "dropped zero accepted");
    Check(ObserveRoutedExpertRow(false, false, bits).invalid == 1,
          "strict dropped zero rejected");
  }
  Check(ObserveRoutedExpertRow(false, true, 0x7fc01357u, vector).invalid == 0,
        "dropped sentinel accepted");
  Check(ObserveRoutedExpertRow(false, true, 0x7fc01357u, vector).executed == 0,
        "dropped sentinel not counted");
  for (auto bits : {0x3f800000u, 1u, 0x7f800000u, 0x7fc00000u})
    Check(ObserveRoutedExpertRow(false, false, bits, vector).invalid == 1,
          "dropped nonzero rejected");
  const auto live = ObserveRoutedExpertRow(true, false, 0u, vector);
  Check(live.executed == 1 && live.invalid == 0, "live zero write counted");
  Check(ObserveRoutedExpertRow(true, true, 0x7fc01357u, vector).invalid == 1,
        "missing live write rejected");
  Check(ObserveRoutedExpertRow(true, false, 0x7f800000u, vector).invalid == 1,
        "live nonfinite rejected");
  std::uint32_t writes = 0, invalid = 0;
  for (int row = 0; row < 8; ++row) {
    const auto observed = ObserveRoutedExpertRow(row < 3, false, 0u, vector);
    writes += observed.executed;
    invalid += observed.invalid;
  }
  Check(writes == 3 && invalid == 0,
        "k=3 live-write total excludes dropped zeros");
  std::cout
      << "PASS audit dropped zero/sentinel, nonzero, strict and live writes\n";
  std::string diagnostic;
  try {
    a3b::FailPrefillExperts("A3B prefill expert counters disagree",
                            [&](auto detail) { diagnostic = detail; });
  } catch (const std::exception& e) {
    Check(std::string(e.what()) == "inference execution failed",
          "HTTP-carried error must be generic");
    Check(diagnostic == "A3B prefill expert counters disagree",
          "private detail retained in diagnostic");
    std::cout << "PASS HTTP-carried message: " << e.what() << '\n';
    return 0;
  }
  throw std::runtime_error("failure lost failed status");
}
