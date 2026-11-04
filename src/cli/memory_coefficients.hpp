#pragma once

#include <cstdint>
#include <string_view>

namespace bubo::cli {







inline constexpr std::uint64_t kLowMemoryCalibratedMainResidentBytes =
    55774419200ULL;
// The 256-expert Flash Next Coder (catalogue id, resident main weights and the
// size of its MTP file). The catalogue row states the same two numbers; the
// models contract test compares them.
inline constexpr std::string_view kCoderPackageId =
    "qwen38-flash-next-coder-ista-gsq-rco-256";
inline constexpr std::uint64_t kCoderMeasuredMainResidentBytes = 29200000000ULL;
inline constexpr std::uint64_t kCoderMtpBytes = 2786568256ULL;
}  // namespace bubo::cli
