#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <string_view>

namespace gufo::core {
inline constexpr std::uint32_t kAmdPciVendorId = 0x1002;
inline constexpr std::array<std::uint32_t, 3> kStrixHaloDeviceIds{
    0x1586, 0x1587, 0x1588};
inline bool IsStrixHaloPciId(std::uint32_t vendor, std::uint32_t device) {
  if (vendor != kAmdPciVendorId)
    return false;
  for (const auto id : kStrixHaloDeviceIds)
    if (device == id)
      return true;
  return false;
}
inline bool IsStrixHaloPciId(std::string_view text) {
  const auto start = text.find("1002:");
  if (start == std::string_view::npos || text.size() < start + 9)
    return false;
  const auto digits = text.substr(start + 5, 4);
  std::uint32_t device = 0;
  const auto parsed =
      std::from_chars(digits.data(), digits.data() + digits.size(), device, 16);
  return parsed.ec == std::errc{} &&
         parsed.ptr == digits.data() + digits.size() &&
         IsStrixHaloPciId(kAmdPciVendorId, device);
}
}  // namespace gufo::core
