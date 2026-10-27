#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

namespace bubo::cli {
struct StopOptions {
  std::string instance_id;
  std::optional<std::uint32_t> requested_wait;
};
StopOptions ParseStopOptions(std::span<const char* const> args);

int RunStop(std::span<const char* const> args, std::string_view local_app_data,
            std::ostream& out, std::ostream& err);
}  // namespace bubo::cli
