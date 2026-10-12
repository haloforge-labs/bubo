#pragma once

#include <string_view>
#include <vector>

namespace bubo::prepare {
// Approved pinned model legal texts (LICENSE, NOTICE, PROVENANCE.md), embedded
// byte for byte. An empty list means the bundle is still OPEN.
struct ModelNotice {
  std::string_view name, sha256, text;
};
const std::vector<ModelNotice>& ModelNotices();
}  // namespace bubo::prepare
