#ifndef GUFO_CLI_KV_CACHE_OPTIONS_HPP_
#define GUFO_CLI_KV_CACHE_OPTIONS_HPP_

#include <string>
#include <string_view>

#include "src/cli/arg_parser.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace gufo::cli {

inline void RegisterKvCacheTypeOption(
    ArgParser& parser, models::qwen38_flash_next::KvCacheType* type) {
  parser.AddCustomOption(
      "", "--kv-cache-type", "TYPE",
      "Ordinary K/V storage: f16 (default) or q8_0 "
      "(q8 is lossy; Flash-Next only)",
      "Model",
      [type](std::string_view, std::string_view value, std::string* error) {
        using models::qwen38_flash_next::KvCacheType;
        if (value == "f16")
          *type = KvCacheType::kF16;
        else if (value == "q8_0")
          *type = KvCacheType::kQ8_0;
        else if (value == "q8_0-rotated")
          *type = KvCacheType::kQ8_0Rotated;
        else {
          if (error)
            *error = "--kv-cache-type must be f16, q8_0 or q8_0-rotated";
          return false;
        }
        return true;
      });
}

// Run after reading architecture metadata, before loading device weights. Never
// silently fall back to F16 for an unsupported model family.
inline bool ValidateKvCacheTypeForArchitecture(
    models::qwen38_flash_next::KvCacheType type, std::string_view architecture,
    std::string* error) {
  using models::qwen38_flash_next::KvCacheType;
  if (models::qwen38_flash_next::IsKnownKvCacheType(type) &&
      (type == KvCacheType::kF16 || architecture == "qwen4exp"))
    return true;
  if (error)
    *error =
        "--kv-cache-type q8_0 or q8_0-rotated requires Qwen3.8-Flash-Next "
        "(GGUF architecture qwen4exp)";
  return false;
}

}  // namespace gufo::cli
#endif  // GUFO_CLI_KV_CACHE_OPTIONS_HPP_
