#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next::rocm::mmq {

inline std::size_t ScratchAdd(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("MMQ scratch size overflow");
  return a + b;
}
inline std::size_t ScratchMul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("MMQ scratch size overflow");
  return a * b;
}
inline std::size_t ScratchPad(std::size_t bytes, std::size_t alignment) {
  return ScratchAdd(bytes, alignment - 1) / alignment * alignment;
}

// These are activation layouts, independent of the weight type. The adapter
// asserts their geometry against the vendored definitions at compile time.
inline constexpr std::size_t kMmqBlockBytes = 144;
inline constexpr std::size_t kMmqValues = 128;
inline constexpr std::size_t kMmqMaxColumns = 80;
inline constexpr std::size_t kMmqRowPadding = 512;

inline std::size_t MmqActivationBytes(std::size_t rows, std::size_t cols) {
  return ScratchAdd(
      ScratchMul(ScratchMul(rows, ScratchPad(cols, kMmqRowPadding)),
                 kMmqBlockBytes) /
          kMmqValues,
      kMmqMaxColumns * kMmqBlockBytes);
}

// The four simultaneously live allocations in qfn_mmq_moe_impl.
inline std::size_t TiledScratchBytes(std::size_t tokens, std::size_t experts,
                                     std::size_t used, std::size_t cols) {
  if (!tokens || !experts || !used || used > experts || !cols)
    throw std::invalid_argument("invalid MMQ scratch shape");
  const auto slots = ScratchMul(tokens, used);
  const auto ids = ScratchPad(ScratchMul(slots, 4), 256);
  const auto bounds = ScratchPad(ScratchMul(ScratchAdd(experts, 1), 4), 256);
  return ScratchAdd(ScratchAdd(ScratchMul(ids, 2), bounds),
                    ScratchPad(MmqActivationBytes(slots, cols), 256));
}

inline std::size_t VectorScratchBytes(std::size_t rows, std::size_t used,
                                      std::size_t cols, bool gated,
                                      bool sorted_down) {
  std::size_t groups = 0;
  if (sorted_down && rows > 1)
    groups = ScratchMul(rows, 8);
  else if (gated && rows > 1) {
    const auto slots = ScratchMul(rows, used);
    groups = rows > 8
                 ? ScratchAdd(4, ScratchMul(slots, 72))
                 : ScratchMul(ScratchAdd(ScratchMul(slots, rows + 1),
                                         ScratchMul(ScratchAdd(slots, 1), 2)),
                              4);
  }
  return ScratchPad(
      ScratchAdd(ScratchMul(ScratchMul(rows, ScratchPad(cols, 512)), 36) / 32,
                 groups),
      256);
}

// Keep this header usable by the C++17 vendored adapter as well as C++20 host
// code; only host callers instantiate the weight-type-dependent planner.
template<class Type>
struct RoutedPoolUse {
  Type gate, up, down;
  std::size_t experts, used, hidden, ff;
};

// wmma and v2k come from the executor's actual route eligibility. Dense Q8,
// twin/head prequantized vectors and unquantized projections use executor-owned
// staging, not this pool. A model without routed pool users therefore needs 0.
template<class Type>
inline std::size_t RoutedPoolScratchBytes(const RoutedPoolUse<Type>& w,
                                          std::size_t tokens, bool prefill,
                                          bool wmma, bool v2k) {
  if (wmma || (v2k && !prefill && tokens >= 2 && tokens <= 8))
    return 0;
  const bool matrix = prefill || tokens > 8;
  const bool expert_matrix = prefill || tokens > 32;
  const bool same = w.gate == w.up;
  const bool gated =
      !matrix && same && w.used <= 32 &&
      (w.gate == Type::kQ4_K || w.gate == Type::kQ5_K || w.gate == Type::kQ8_0);
  // Conservatively include F32 input even if ReadyQ8 can avoid its staging.
  const auto gate =
      expert_matrix
          ? TiledScratchBytes(tokens, w.experts, w.used, w.hidden)
          : VectorScratchBytes(tokens, w.used, w.hidden, gated, false);
  const auto slots = ScratchMul(tokens, w.used);
  const bool tiled_down = (prefill || slots > 32) && matrix;
  const bool sorted =
      w.ff == 640 && (w.down == Type::kQ5_1 || w.down == Type::kQ8_0);
  const auto down = tiled_down
                        ? TiledScratchBytes(slots, w.experts, 1, w.ff)
                        : VectorScratchBytes(slots, 1, w.ff, false, sorted);
  return std::max(gate, down);
}

inline std::size_t ScratchBlockCapacity(std::size_t bytes) {
  if (!bytes)
    return 0;
  std::size_t capacity = std::size_t{16} << 20;
  while (capacity < bytes)
    capacity = ScratchMul(capacity, 2);
  return capacity;
}

struct PoolScratchPlan {
  std::size_t primary = 0, alternate = 0;
  void Include(std::size_t rows, bool prefill, std::size_t bytes) {
    primary = std::max(primary, bytes);
    if (!prefill && rows <= 8)
      alternate = std::max(alternate, bytes);
  }
};

class PoolCaptureError : public std::runtime_error {
public:
  PoolCaptureError()
      : std::runtime_error(
            "MMQ scratch pool exhausted during graph capture; reserve the full "
            "shape on this stream before capture (device allocation refused)") {
  }
};

inline void RequirePoolGrowthAllowed(bool capturing) {
  if (capturing)
    throw PoolCaptureError();
}

}  // namespace gufo::models::qwen38_flash_next::rocm::mmq
