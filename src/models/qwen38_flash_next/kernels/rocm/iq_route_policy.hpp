#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_ROUTE_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_ROUTE_POLICY_HPP_

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace gufo::models::qwen38_flash_next::rocm {

// D1 lost to MMQ at 512 pass rows and won by 1024; ABBA will test 256.
inline constexpr std::uint32_t kIqRouteMinRows = 256;

inline constexpr bool UseIqWmmaRoute(bool enabled, std::uint32_t pass_rows,
                                     std::uint32_t min_rows) {
  if (!enabled || pass_rows < min_rows)
    return false;
  return true;
}

// Rows that choose the route of one forward pass. A prefill pass counts the
// rows of the whole prompt it belongs to (0 when the session did not say), so
// the short pass that follows a prefix-cache restore runs the route the cold
// passes of the same prompt ran; any other pass counts its own rows.
inline constexpr std::uint32_t IqRouteRows(bool prefill,
                                           std::uint32_t pass_rows,
                                           std::uint32_t prompt_rows) {
  return prefill && prompt_rows > pass_rows ? prompt_rows : pass_rows;
}

// GUFO_ROUTED_WMMA_IQ: 0 selects the MMQ fallback, 1 the IQ WMMA route with
// the shared-memory decode, 2 (or unset) the same route with the register-fed
// IQ gate/up pair kernel (pf-iq-pair) on the layers IqPair2Covers admits. The
// pair kernel decodes the same F16 weights as route 1, so the output is
// bit-identical. Any other value warns and keeps the default, 2.
enum class IqWmmaMode : std::uint8_t { kOff, kOn, kPair2 };

struct IqWmmaModeValue {
  IqWmmaMode mode = IqWmmaMode::kPair2;
  bool unsupported = false;
};

inline constexpr IqWmmaModeValue ParseIqWmmaMode(const char* value) {
  if (value == nullptr)
    return {};
  const std::string_view text(value);
  if (text == "0")
    return {IqWmmaMode::kOff, false};
  if (text == "1")
    return {IqWmmaMode::kOn, false};
  if (text == "2")
    return {IqWmmaMode::kPair2, false};
  return {IqWmmaMode::kPair2, true};
}

// GGML type ids of the gate/up types RoutedIqPairKernel covers.
inline constexpr bool IqPair2Type(std::uint32_t ggml_type) {
  return ggml_type == 22      // IQ2_S
         || ggml_type == 18   // IQ3_XXS
         || ggml_type == 21   // IQ3_S
         || ggml_type == 17   // IQ2_XS
         || ggml_type == 16;  // IQ2_XXS
}

// Value 2 takes the pair kernel for a gate/up layer whose gate and up share a
// covered type, on passes with the wide 128-row pair tiles; every other layer
// (narrow 64-row pair tiles included) keeps route 1.
inline constexpr bool IqPair2Covers(IqWmmaMode mode, std::uint32_t gate_type,
                                    std::uint32_t up_type,
                                    std::uint32_t pair_tile_rows,
                                    std::uint64_t k, std::uint64_t m) {
  return mode == IqWmmaMode::kPair2 && gate_type == up_type &&
         IqPair2Type(gate_type) && pair_tile_rows == 128 && k != 0 &&
         k % 256 == 0 && m != 0;
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
