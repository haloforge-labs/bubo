#ifndef GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
#define GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

#include <cstdint>
#include <cstdlib>

namespace gufo::hip {

/// Keep this much free in the device heap for driver command submission.
inline constexpr std::uint64_t kDispatchReserve = 4ull << 30;

/// The reserve in force: kDispatchReserve unless GUFO_DISPATCH_RESERVE_MIB
/// names another size in MiB (0 to 1,048,576), read once. A smaller reserve
/// lets pinned cache backing and weights use more of the device heap on
/// machines that need it (two long sessions); the default stays 4 GiB.
[[nodiscard]] inline std::uint64_t DispatchReserveBytes() {
  static const std::uint64_t reserve = []() -> std::uint64_t {
    if (const char* text = std::getenv("GUFO_DISPATCH_RESERVE_MIB")) {
      char* end = nullptr;
      const auto mib = std::strtoull(text, &end, 10);
      if (end != text && *end == '\0' && mib <= (1ull << 20))
        return mib << 20;
    }
    return kDispatchReserve;
  }();
  return reserve;
}

/// True when `bytes` can be taken from `free_bytes` and still leave
/// DispatchReserveBytes().
[[nodiscard]] inline bool HeapFitsAboveReserve(std::uint64_t free_bytes,
                                               std::uint64_t bytes) {
  return bytes <= free_bytes && free_bytes - bytes >= DispatchReserveBytes();
}

enum class Placement { kMalloc, kManaged };

[[nodiscard]] inline Placement ChooseWeightPlacement(
    std::uint64_t free_bytes, std::uint64_t total_weight_bytes) {
  return HeapFitsAboveReserve(free_bytes, total_weight_bytes)
             ? Placement::kMalloc
             : Placement::kManaged;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
