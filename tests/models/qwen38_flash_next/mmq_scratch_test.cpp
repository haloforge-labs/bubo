// Failure cases were recorded in decode-coder256/NOTES.md before this test.
#include <array>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/scratch_plan.hpp"

namespace m = gufo::models::qwen38_flash_next::rocm::mmq;
using T = gufo::core::GgmlType;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
struct PoolDouble {
  std::size_t capacity = 0;
  unsigned allocations = 0;
  bool capturing = false;
  void Request(std::size_t bytes) {
    if (bytes <= capacity)
      return;
    m::RequirePoolGrowthAllowed(capturing);
    capacity = m::ScratchBlockCapacity(bytes);
    ++allocations;
  }
};
using Combination = std::array<T, 3>;
constexpr std::array production{Combination{T::kQ4_K, T::kQ4_K, T::kQ5_1},
                                Combination{T::kQ4_K, T::kQ4_K, T::kQ8_0},
                                Combination{T::kQ5_K, T::kQ5_K, T::kQ8_0}};
// Seven combinations logged in the real prompt, plus an adversarial
// IQ4_XS/Q2_0 combination to cover eight combinations in this test matrix.
constexpr std::array coder{Combination{T::kIQ3_XXS, T::kIQ3_XXS, T::kIQ4_NL},
                           Combination{T::kIQ2_S, T::kIQ2_S, T::kQ2_0},
                           Combination{T::kIQ2_S, T::kIQ2_S, T::kIQ4_NL},
                           Combination{T::kIQ3_XXS, T::kIQ3_XXS, T::kQ2_0},
                           Combination{T::kIQ3_S, T::kIQ3_S, T::kIQ4_NL},
                           Combination{T::kIQ3_S, T::kIQ3_S, T::kQ2_0},
                           Combination{T::kIQ4_XS, T::kIQ4_XS, T::kIQ4_NL},
                           Combination{T::kIQ4_XS, T::kIQ4_XS, T::kQ2_0}};
constexpr std::array<Combination, 1> q2{
    Combination{T::kQ2_0, T::kQ2_0, T::kQ2_0}};

m::PoolScratchPlan Plan(std::span<const Combination> combinations,
                        std::size_t experts, bool iq_enabled = true) {
  m::PoolScratchPlan plan;
  for (std::size_t n = 1; n <= 4096; ++n) {
    for (bool prefill : {false, true}) {
      for (auto types : combinations) {
        const bool q4 = types[0] == T::kQ4_K;
        const bool standard = q4 || types[0] == T::kQ5_K;
        const bool iq = types[0] != T::kQ2_0 && !standard;
        // Fixture eligibility: H=2560,F=640, paired gates, and down
        // Q5_1/Q8_0/IQ4_NL/Q2_0; shipped IQ WMMA threshold is 256 rows.
        const bool wmma =
            (prefill || n > 32) && (standard || (iq && iq_enabled && n >= 256));
        const bool v2k = q4 && experts == 512;
        plan.Include(n, prefill,
                     m::RoutedPoolScratchBytes(
                         m::RoutedPoolUse<T>{types[0], types[1], types[2],
                                             experts, 10, 2560, 640},
                         n, prefill, wmma, v2k));
      }
    }
  }
  return plan;
}
std::size_t Align(std::size_t n) {
  return (n + 255) / 256 * 256;
}
std::size_t IndependentTiled(std::size_t n, std::size_t experts) {
  return 2 * Align(n * 10 * 4) + Align((experts + 1) * 4) +
         Align(n * 10 * 2560 / 128 * 144 + 80 * 144);
}
void Check() {
  for (std::size_t experts : {256, 512}) {
    const auto p = Plan(production, experts);
    // Q4 wide passes bypass the pool. The 32-row down projection uses the
    // largest remaining tiled request. The pinned Q5_K layer cannot use V2K,
    // so it determines the alternate bound even at E=512.
    const auto q4_primary = 2 * Align(320 * 4) + Align((experts + 1) * 4) +
                            Align(320 * 1024 / 128 * 144 + 80 * 144);
    Require(p.primary == q4_primary, "production primary reservation inflated");
    Require(p.alternate == Align(80 * 1024 / 32 * 36 + 640),
            "production alternate missed the pinned Q5_K pool user");
    const auto q4_only = Plan(std::span(production).first(2), experts);
    Require(q4_only.alternate == (experts == 512
                                      ? Align(10 * 1024 / 32 * 36 + 80)
                                      : Align(80 * 1024 / 32 * 36 + 640)),
            "Q4-only alternate missed V2K or scalar pool user");
    const auto c = Plan(coder, experts);
    Require(c.primary == IndependentTiled(255, experts),
            "mixed-format maximum is not the last pre-IQ-WMMA width");
    Require(c.alternate == Align(80 * 1024 / 32 * 36),
            "mixed-format alternate reservation wrong");
    const auto all_q2 = Plan(q2, experts);
    Require(all_q2.primary == IndependentTiled(4096, experts),
            "Q2_0 tiled maximum wrong");
    Require(Plan(coder, experts, false).primary == all_q2.primary,
            "disabled IQ route omitted actual MMQ users");
    const std::array later_user{production[0], q2[0]};
    const std::array earlier_user{q2[0], production[0]};
    Require(Plan(later_user, experts).primary == all_q2.primary &&
                Plan(earlier_user, experts).primary == all_q2.primary,
            "pool planning depends on the first layer's format");
    const auto none = Plan({}, experts);
    Require(none.primary == 0 && none.alternate == 0 &&
                m::ScratchBlockCapacity(0) == 0,
            "model with only caller-owned dense staging allocated an arena");
    for (auto bound : {p.primary, p.alternate, c.primary, c.alternate,
                       all_q2.primary, all_q2.alternate, std::size_t{0}}) {
      PoolDouble pool;
      pool.Request(bound);
      const auto before = pool.allocations;
      pool.capturing = true;
      pool.Request(bound);
      bool refused = false;
      try {
        pool.Request(pool.capacity + 1);
      } catch (const m::PoolCaptureError& error) {
        refused = std::string(error.what()).find("device allocation refused") !=
                  std::string::npos;
      }
      Require(refused && pool.allocations == before,
              "capture reached device allocator");
    }
    std::cout << "E=" << experts << " Q4=" << p.primary << '/' << p.alternate
              << " IQ=" << c.primary << '/' << c.alternate
              << " Q2=" << all_q2.primary << '/' << all_q2.alternate << '\n';
  }
  // Positive requests retain the original first block and geometric sequence.
  constexpr std::size_t mib = 1U << 20;
  Require(m::ScratchBlockCapacity(1) == 16 * mib &&
              m::ScratchBlockCapacity(16 * mib) == 16 * mib &&
              m::ScratchBlockCapacity(16 * mib + 1) == 32 * mib &&
              m::ScratchBlockCapacity(64 * mib + 1) == 128 * mib,
          "existing positive allocation sequence changed");
  bool overflow = false;
  try {
    (void)m::TiledScratchBytes(std::numeric_limits<std::size_t>::max(), 512, 10,
                               2560);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  Require(overflow, "scratch size overflow accepted");
}
}  // namespace
int main() {
  try {
    Check();
    std::cout << "MMQ scratch bounds and capture refusal passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
