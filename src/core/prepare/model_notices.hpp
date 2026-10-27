#pragma once

#include <string_view>
#include <vector>

#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare {
// Approved pinned model legal texts (LICENSE, NOTICE, PROVENANCE.md), embedded
// byte for byte. An empty list means the bundle is still OPEN.
struct ModelNotice {
  std::string_view name, sha256, text;
};
// Huihui set (the default). Existing callers keep this overload.
const std::vector<ModelNotice>& ModelNotices();
// Per-set bundle: tools/release/gen_model_notices.py embeds each bundle.
const std::vector<ModelNotice>& ModelNotices(ModelSet set);
}  // namespace bubo::prepare
