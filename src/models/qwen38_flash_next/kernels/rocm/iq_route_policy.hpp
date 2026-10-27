#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_ROUTE_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_ROUTE_POLICY_HPP_

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace gufo::models::qwen38_flash_next::rocm {


inline constexpr std::uint32_t kIqRouteMinRows = 256;

inline constexpr bool UseIqWmmaRoute(bool enabled, std::uint32_t pass_rows,
                                     std::uint32_t min_rows) {
  if (!enabled || pass_rows < min_rows)
    return false;
  return true;
}

inline bool ParseIqRouteMinRows(std::string_view value,
                                std::uint32_t* min_rows) {
  if (min_rows == nullptr || value.empty())
    return false;
  std::uint32_t parsed = 0;
  const char* end = value.data() + value.size();
  const auto result = std::from_chars(value.data(), end, parsed);
  if (result.ec != std::errc{} || result.ptr != end)
    return false;
  *min_rows = parsed;
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_ROUTE_POLICY_HPP_
