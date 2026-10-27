#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace gufo::server {
/// RAM always left to the OS and other processes by an explicit cache limit.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;
inline constexpr std::size_t kAutomaticSnapshotMaxBytes = std::size_t{32} << 30;
/// Smallest automatic or explicit RAM cache that is still worth keeping. A
/// ceiling under this disables the cache (capacity 0) with a startup warning.
inline constexpr std::uint64_t kMinSnapshotCacheBytes = std::uint64_t{512}
                                                        << 20;
/// The device total is rounded down to this step before planning. The WDDM
/// budget behind it moved by 786 KB between two starts on 2026-10-14, which
/// would move an automatic capacity; whole GiB keep the plan identical.
inline constexpr std::uint64_t kDeviceTotalStepBytes = std::uint64_t{1} << 30;
/// Inputs of the deterministic snapshot-cache capacity. None of them is a
/// free-memory reading, so the result is the same on a busy and an idle host.
struct SnapshotCapacityInputs {
  std::uint64_t host_total_bytes{0};
  /// Model bytes held in host RAM (host weight copies, NPU buffers).
  std::uint64_t host_resident_bytes{0};
  /// Resident weights + sessions * (per-session state + scratch). Device
  /// memory when device_total_bytes is set, host RAM otherwise.
  std::uint64_t model_footprint_bytes{0};
  std::uint64_t reserve_bytes{kHostSnapshotHeadroomBytes};
  /// Device total; absent when the backend cannot report it.
  std::optional<std::uint64_t> device_total_bytes;
  /// 0 selects the automatic capacity.
  std::uint64_t explicit_bytes{0};
};

struct SnapshotCapacityPlan {
  std::size_t automatic_bytes{0};
  std::size_t ceiling_bytes{0};
  std::size_t capacity_bytes{0};
  /// The smaller side of the ceiling: "host" or "device".
  std::string limit{"host"};
  /// Empty, "explicit_over_ceiling" or "disabled_ceiling_below_minimum".
  std::string clamp;
};

/// A snapshot keeps a host payload and may hold frozen device storage, so the
/// ceiling is the smaller of two budgets:
///   host   = host_total - host_resident - reserve
///   device = device_total (whole GiB) - footprint - reserve
/// A carve-out APU keeps its weights out of host RAM (Windows 96 GiB carve:
/// 32 GiB host, 90 GiB model), so they count on the device side only. Where
/// device memory is host RAM (GTT, small carve) the device total is at most
/// the host total and the device side is the smaller one. Without a device
/// total the footprint counts against the host. A ceiling under
/// kMinSnapshotCacheBytes gives ceiling and capacity 0 (cache disabled).
/// automatic = min(32 GiB, ceiling / 2). An explicit limit is exact up to the
/// ceiling and clamps to it otherwise.
[[nodiscard]] inline SnapshotCapacityPlan PlanSnapshotCapacity(
    const SnapshotCapacityInputs& inputs) {
  const auto SaturatingAdd = [](std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
  };
  const auto ClampToSize = [](std::uint64_t bytes) {
    return static_cast<std::size_t>(std::min<std::uint64_t>(
        bytes, std::numeric_limits<std::size_t>::max()));
  };
  const auto remaining = [](std::uint64_t total, std::uint64_t taken) {
    return total > taken ? total - taken : std::uint64_t{0};
  };
  auto host_taken =
      SaturatingAdd(inputs.host_resident_bytes, inputs.reserve_bytes);
  if (!inputs.device_total_bytes)
    host_taken = SaturatingAdd(host_taken, inputs.model_footprint_bytes);
  auto ceiling = remaining(inputs.host_total_bytes, host_taken);
  SnapshotCapacityPlan plan;
  if (inputs.device_total_bytes) {
    const auto device_total =
        *inputs.device_total_bytes -
        *inputs.device_total_bytes % kDeviceTotalStepBytes;
    const auto device = remaining(
        device_total,
        SaturatingAdd(inputs.model_footprint_bytes, inputs.reserve_bytes));
    if (device < ceiling) {
      ceiling = device;
      plan.limit = "device";
    }
  }
  if (ceiling < kMinSnapshotCacheBytes) {
    plan.clamp = "disabled_ceiling_below_minimum";
    return plan;
  }
  plan.ceiling_bytes = ClampToSize(ceiling);
  plan.automatic_bytes =
      std::min(kAutomaticSnapshotMaxBytes, ClampToSize(ceiling / 2));
  if (inputs.explicit_bytes == 0) {
    plan.capacity_bytes = plan.automatic_bytes;
  } else if (inputs.explicit_bytes > plan.ceiling_bytes) {
    plan.capacity_bytes = plan.ceiling_bytes;
    plan.clamp = "explicit_over_ceiling";
  } else {
    plan.capacity_bytes = ClampToSize(inputs.explicit_bytes);
  }
  return plan;
}

}  // namespace gufo::server
