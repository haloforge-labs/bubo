#include "src/models/qwen38_flash_next/scratch_bytes.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "src/cli/serve/snapshot_capacity.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
int main() {
  assert(qfn::AllocationRequestBytes(0, 4) == 4);
  assert(qfn::AllocationRequestBytes(8, 4) == 32);
  for (auto width : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
    bool failed = false;
    try {
      (void)qfn::AllocationRequestBytes(2, width);
    } catch (const std::exception&) {
      failed = true;
    }
    assert(failed);
  }
  std::size_t device = 0, host = 0;
  qfn::AddAllocationBytes(device, 32);
  qfn::AddAllocationBytes(device, 64);
  qfn::AddAllocationBytes(host, 128);
  assert(qfn::ResidentWithScratch(1024, device, host) == 1248);
  auto overflow = std::numeric_limits<std::size_t>::max();
  bool rejected = false;
  try {
    qfn::AddAllocationBytes(overflow, 1);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  assert(rejected && overflow == std::numeric_limits<std::size_t>::max());
  // Independent nominal fixture from the allocation audit: 343775.5 device
  // and 10251.5 pinned/mapped bytes per extra aligned row. Real accounting
  // records the actual allocations, including format-specific maxima/fixed
  // buffers; it must never multiply this shared term by session count.
  const std::size_t d4 = 1452107712, h4 = 43302336;
  const auto scratch4 = qfn::ResidentWithScratch(0, d4, h4);
  const auto scratch8 =
      qfn::ResidentWithScratch(0, d4 + 1408104448, h4 + 41990144);
  assert(scratch8 - scratch4 == 1450094592);
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  // At the edge: the legacy footprint admitted a cache for 8192, while
  // counting its shared scratch leaves less than the 512 MiB cache minimum.
  const auto plan = [](std::uint64_t footprint) {
    return gufo::server::PlanSnapshotCapacity(
        {.host_total_bytes = 128ull << 30,
         .model_footprint_bytes = footprint,
         .device_total_bytes = 70ull << 30});
  };
  constexpr std::size_t legacy4 = 68219611988ull, legacy8 = 68274137940ull;
  assert(plan(legacy8).capacity_bytes > 0);
  assert(plan(qfn::ResidentWithScratch(legacy4, d4, h4)).capacity_bytes > 0);
  const auto wide = plan(legacy8 + scratch8);
  assert(wide.capacity_bytes == 0 &&
         wide.clamp == "disabled_ceiling_below_minimum");
  assert(legacy8 + scratch8 > 66 * gib - (512ull << 20));
  std::cout << "PASS allocation accounting, overflow, shared UMA footprint and "
               "8192 edge budget\n";
}
