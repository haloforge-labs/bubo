#include "src/models/qwen38_flash_next/kv_rotation.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

float Finite(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("nonfinite Flash-Next KV rotation");
  return value;
}
}  // namespace

void RotateKvHead(std::span<float> head, KvRotationKind kind) {
  if (head.size() != 256 || std::fegetround() != FE_TONEAREST)
    throw std::invalid_argument("KV rotation requires D256 and F32 RNE");
  std::size_t width;
  float scale;
  switch (kind) {
    case KvRotationKind::kKeyQuery:
      width = 256;
      scale = 0.0625F;
      break;
    case KvRotationKind::kValueOutput:
      width = 64;
      scale = 0.125F;
      break;
    default:
      throw std::invalid_argument("unsupported Flash-Next KV rotation kind");
  }
  // Bounded transactional workspace: reject before changing the caller's head.
  std::array<float, 256> work;
  for (std::size_t i = 0; i < work.size(); ++i) {
    const float x = Finite(head[i]);
    volatile float normalized = x * scale;
    work[i] = Finite(normalized);
  }
  for (std::size_t base = 0; base < work.size(); base += width) {
    for (std::size_t distance = 1; distance < width; distance *= 2) {
      for (std::size_t block = 0; block < width; block += 2 * distance) {
        for (std::size_t i = 0; i < distance; ++i) {
          const auto low = base + block + i;
          const auto high = low + distance;
          const float u = work[low];
          const float v = work[high];
          volatile float sum = u + v;
          volatile float difference = u - v;
          work[low] = Finite(sum);
          work[high] = Finite(difference);
        }
      }
    }
  }
  std::ranges::copy(work, head.begin());
}
}  // namespace gufo::models::qwen38_flash_next
