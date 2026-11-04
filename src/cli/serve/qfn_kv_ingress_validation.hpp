#ifndef GUFO_SERVER_QFN_KV_INGRESS_VALIDATION_HPP_
#define GUFO_SERVER_QFN_KV_INGRESS_VALIDATION_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/models/qwen38_flash_next/continuation_layout.hpp"

namespace gufo::server::detail {
// Private model-owned admission at the existing host-buffer finish boundary.
// External callers must finish loading/checksumming the COMPLETE payload before
// invoking finish. This does not validate trusted native borrow callbacks.
inline void InstallQfnPackedKvIngressValidation(
    ComponentCacheResources& resources,
    const models::qwen38_flash_next::Config& config, bool mtp,
    std::uint32_t context, models::qwen38_flash_next::KvCacheType type) {
  using namespace models::qwen38_flash_next;
  std::uint32_t version;
  switch (type) {
    case KvCacheType::kF16:
      return;  // F16 retains its original allocator/finish.
    case KvCacheType::kQ8_0:
      version = 2;
      break;
    case KvCacheType::kQ8_0Rotated:
      version = 3;
      break;
    default:
      throw std::invalid_argument("unsupported packed KV ingress type");
  }
  if (!resources.allocate)
    throw std::invalid_argument("packed KV ingress requires an allocator");

  // kept_rows only sizes a non-KV private component. Its value does not change
  // IDs or the ordinary/draft classification; no private geometry is inferred.
  const ContinuationLayout layout(config, mtp, context, 1, type);
  const auto elements = config.AttentionKvDim();
  const auto row = KvRowBytes(elements, type);
  struct Rule {
    cache::ComponentId id;
    bool packed;
  };
  std::vector<Rule> rules;
  bool has_kv = false;
  for (const auto& component : layout.Components()) {
    const auto part = component.part;
    const bool packed = part == ContinuationPart::kKey ||
                        part == ContinuationPart::kValue ||
                        part == ContinuationPart::kDraftKey ||
                        part == ContinuationPart::kDraftValue;
    if (packed && (component.descriptor.row_bytes != row ||
                   component.descriptor.layout_version != version || !row))
      throw std::invalid_argument("invalid packed KV ingress geometry");
    rules.push_back({component.descriptor.id, packed});
    has_kv |= packed;
  }
  if (!has_kv)
    throw std::invalid_argument("packed KV ingress has no KV components");

  auto allocate = resources.allocate;
  auto owner = resources.owner;
  resources.allocate = [allocate = std::move(allocate),
                        owner = std::move(owner), rules = std::move(rules),
                        elements, row, context,
                        type](const cache::PayloadRequest& request) {
    const auto rule = std::ranges::find(rules, request.component, &Rule::id);
    if (rule == rules.end())
      throw std::invalid_argument("unknown packed KV ingress component");
    if (!rule->packed)
      return allocate(request);  // Tokens/indexer/private: verbatim finish.
    if ((request.category != cache::ResourceCategory::kBackingAssigned &&
         request.category != cache::ResourceCategory::kPrivateTail) ||
        request.first >= request.end || request.end > context ||
        request.end - request.first >
            std::numeric_limits<std::size_t>::max() / row ||
        request.bytes != (request.end - request.first) * row)
      throw std::invalid_argument("invalid packed KV ingress row request");
    auto buffer = allocate(request);
    if (!buffer.finish || buffer.bytes.size() < request.bytes ||
        buffer.bytes.data() == nullptr)
      throw std::invalid_argument("incomplete packed KV ingress buffer");
    const auto bytes =
        std::span<const std::byte>(buffer.bytes).first(request.bytes);
    auto finish = std::move(buffer.finish);
    // Copies of std::function must not convert the same backing twice, or
    // retry a failed capture after mutating its bytes. Publication owns the
    // host-serialized call; this is not a concurrent-finish API.
    auto consumed = std::make_shared<bool>(false);
    buffer.finish = [finish = std::move(finish), owner, bytes, consumed,
                     elements, row, type]() mutable {
      if (std::exchange(*consumed, true))
        throw std::logic_error("packed KV ingress finish already consumed");
      for (std::size_t offset = 0; offset < bytes.size(); offset += row)
        ValidateKvRow(bytes.subspan(offset, row), elements, type);
      return finish();  // Ownership conversion ONLY after every row passed.
    };
    return buffer;  // In particular, borrow is neither wrapped nor invoked.
  };
}
}  // namespace gufo::server::detail
#endif
