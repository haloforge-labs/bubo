#include "src/core/prepare/model_notices.hpp"

namespace bubo::prepare {
namespace {
#include "src/core/prepare/model_notices.inc"
#include "src/core/prepare/model_notices_lowmemory.inc"
#include "src/core/prepare/model_notices_original.inc"
}  // namespace
const std::vector<ModelNotice>& ModelNotices() {
  return kModelNotices;
}
const std::vector<ModelNotice>& ModelNotices(ModelSet set) {
  if (set == ModelSet::LowMemory)
    return kLowMemoryModelNotices;
  return set == ModelSet::Original ? kOriginalModelNotices : kModelNotices;
}
}  // namespace bubo::prepare
