#ifndef GUFO_CLI_MODEL_FAMILIES_HPP_
#define GUFO_CLI_MODEL_FAMILIES_HPP_
#include <string_view>
namespace gufo::cli {
inline constexpr std::string_view kModelFamilies[] = {
#include "src/cli/serve/release_families.inc"
};
}  // namespace gufo::cli
#endif
