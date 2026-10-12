#pragma once

#include <string_view>
#include <vector>

namespace bubo::prepare {
struct RuntimeNotice {
  std::string_view path, sha256, text;
};
const std::vector<RuntimeNotice>& RuntimeNotices();
}  // namespace bubo::prepare
