#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace bubo::cli {
inline constexpr std::uint64_t k96GbGpuReservationBytes = 64000000000ULL;
inline constexpr const char* k96GbMemoryAdvice =
    "On a 96 GB computer, use the lower-memory model. Set Variable Graphics "
    "Memory to the "
    "largest value offered (72 GB if available, at least 64 GB), then restart "
    "Windows.";
inline constexpr const char* k96GbReservationPass =
    "At least 64 GB is reserved for graphics, enough for the lower-memory "
    "model.";
inline constexpr const char* k96GbSetupMemoryRefusal =
    "Not enough memory is available for this setup. Set Variable Graphics "
    "Memory to the "
    "largest value offered (at least 64 GB), close other memory-heavy apps, "
    "and check again.";
inline std::string FullSize96GbMemoryRefusal(
    std::optional<std::uint64_t> needed_bytes) {
  return "Does not fit this computer: needs " +
         (needed_bytes ? "about " +
                             std::to_string(static_cast<std::uint64_t>(
                                 std::llround(*needed_bytes / 1073741824.0))) +
                             " GiB of GPU memory. "
                       : "more GPU memory than is available. ") +
         "Use the lower-memory model.";
}
}  // namespace bubo::cli
