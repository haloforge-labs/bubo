#include "src/core/prepare/model_notices.hpp"

namespace bubo::prepare {
namespace {
#include "src/core/prepare/model_notices.inc"
}  // namespace
const std::vector<ModelNotice>& ModelNotices() {
  return kModelNotices;
}
}  // namespace bubo::prepare
