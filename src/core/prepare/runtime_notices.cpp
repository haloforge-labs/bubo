#include "src/core/prepare/runtime_notices.hpp"

namespace bubo::prepare {
namespace {
#include "src/core/prepare/runtime_notices.inc"
}  // namespace
const std::vector<RuntimeNotice>& RuntimeNotices() {
  return kRuntimeNotices;
}
}  // namespace bubo::prepare
