#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_PRODUCER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_PRODUCER_HPP_

#include "src/models/qwen38_flash_next/kv_cache.hpp"
#include "src/models/qwen38_flash_next/kv_rotation.hpp"

namespace gufo::models::qwen38_flash_next::rocm::detail {

// Private executor sequencing boundary for separately projected cache-only
// rows. Prepare performs K RMS/RoPE; Store is the canonical codec-only store.
// Callbacks keep this exact production sequence testable without HIP. They do
// not constitute a numerical CPU implementation of the device operations.
template<class Prepare, class Rotate, class Store>
bool ProduceKvRows(KvCacheType type, Prepare&& prepare, Rotate&& rotate,
                   Store&& store) {
  prepare();
  if (KvIsRotated(type) && (!rotate(false, KvRotationKind::kKeyQuery) ||
                            !rotate(true, KvRotationKind::kValueOutput)))
    return false;
  store(false);
  store(true);
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::detail
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_PRODUCER_HPP_
