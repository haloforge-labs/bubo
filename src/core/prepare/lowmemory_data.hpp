#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace bubo::prepare {
// Published release repository.
inline constexpr std::string_view kLowMemoryRepository =
    "HaloForgeLabs/bubo-qwen3.8-flash-next-lower-memory";
// Final published revision includes README, NOTICE and LICENSE.
inline constexpr std::string_view kLowMemoryCommit =
    "2f4f068840c7d85c7b0d8ac044db9e410687d636";
inline constexpr std::string_view kLowMemoryRecipeId =
    "huihui-lowmemory-iq4xs-nofullattn-nohc-q4k-v1";
inline constexpr std::string_view kLowMemoryIstaRevision =
    "ed59f92082b1e93c0e96d60a8b11aab089b52f09";
inline constexpr std::string_view kLowMemoryComposeManifestSha256 =
    "d87bc832f3b2bdae7eb65007a7a6f704d3f9674c13204fe1e5a1414ecf512084";
// Decode supplies the measured main resident bytes. Until then there is no
// lowmemory memory verdict. This slot never uses the download size as a proxy.
inline constexpr std::optional<std::uint64_t>
    kLowMemoryMeasuredMainResidentBytes = std::nullopt;
struct LowMemoryShardPin {
  std::string_view file_name;
  std::string_view local_source_name;
  std::uint64_t size;
  std::string_view sha256;
  std::string_view fingerprint_slot;
};

inline constexpr std::array<LowMemoryShardPin, 4> kLowMemoryShards{{
    {"Qwen3.8-Flash-Next-Huihui-LowMemory-00001-of-00004.gguf",
     "Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00001-of-00004.gguf", 10947264ULL,
     "d9a9945bf6c622a5725156f4d3f23ba56006e597012edbfee851ffcbaa874d30",
     "fingerprints/lowmemory_main_1.json"},
    {"Qwen3.8-Flash-Next-Huihui-LowMemory-00002-of-00004.gguf",
     "Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00002-of-00004.gguf",
     42270514496ULL,
     "a27c6eaff43a7f46970efa4382c01ae758747fcf4469576cbeb12d0ead5a6e7b",
     "fingerprints/lowmemory_main_2.json"},
    {"Qwen3.8-Flash-Next-Huihui-LowMemory-00003-of-00004.gguf",
     "Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00003-of-00004.gguf",
     33568858464ULL,
     "5587334e6b35795fc7259a889b958b401d53a613a649d1dec4e023a8028b92fb",
     "fingerprints/lowmemory_main_3.json"},
    {"Qwen3.8-Flash-Next-Huihui-LowMemory-00004-of-00004.gguf",
     "Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00004-of-00004.gguf",
     8758754880ULL,
     "42c2450068305ef710375b043a42dc3a7ce42de490f11a8cf154c00eaf6ca67b",
     "fingerprints/lowmemory_main_4.json"},
}};
}  // namespace bubo::prepare
