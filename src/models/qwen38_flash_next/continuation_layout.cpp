#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

namespace gufo::models::qwen38_flash_next {
namespace {
bool IsDraft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool IsPool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
std::uint32_t KvComponentLayoutVersion(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return 1;
    case KvCacheType::kQ8_0:
      return 2;
    case KvCacheType::kQ8_0Rotated:
      return 3;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}
}  // namespace

std::string KvCacheCompatibilityIdentity(KvCacheType type) {
  // Bind the attention implementation and packed reconstruction
  // path independently of other builds. Equal row widths are not an ABI.
  std::string identity = "ordinary_kv_dtype=";
  identity += KvCacheTypeName(type);
  identity += "\nordinary_kv_scope=all-trunk-and-mtp\n";
  switch (type) {
    case KvCacheType::kF16:
      identity +=
          "ordinary_kv_layout=row-major-f16-le-v1\n"
          "ordinary_kv_block_elements=1\n"
          "ordinary_kv_scale_dtype=none\n"
          "ordinary_kv_rounding_saturation_revision=f16-rne-v1\n"
          "ordinary_kv_reconstruction=f16-identity-v1\n";
      break;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      identity +=
          "ordinary_kv_layout=ggml-q8_0-f16le-scale-then-i8x32-v1\n"
          "ordinary_kv_block_elements=32\n"
          "ordinary_kv_scale_dtype=f16\n"
          "ordinary_kv_rounding_saturation_revision="
          "f32-absmax-div127-roundf-away-clamp127-scale-f16-rne-v1\n"
          "ordinary_kv_reconstruction=f16-rne-of-f32-scale-times-i8-v1\n"
          "ordinary_kv_gpu_reconstruction_equality=ieee-values-zero-sign-"
          "equivalent-v1\n"
          "ordinary_kv_zero_or_f32_scale_underflow=scale0-codes0\n"
          "ordinary_kv_f16_scale_underflow=preserve-codes-value-zero-v1\n";
      break;
    default:
      throw std::invalid_argument("unsupported Flash-Next KV cache type");
  }
  if (KvIsRotated(type)) {
    identity +=
        "ordinary_kv_basis=hadamard-rotated-v1\n"
        "ordinary_kv_key_query_transform=h256-per-head\n"
        "ordinary_kv_value_output_transform=4-contiguous-h64-per-head\n"
        "ordinary_kv_transform_order=natural-sylvester-no-signs-no-"
        "permutation\n"
        "ordinary_kv_normalization=first-f32-rne-h256-1/16-h64-1/8\n"
        "ordinary_kv_butterfly=ascending-distance-f32-rne-low-u+v-high-u-v-v1\n"
        "ordinary_kv_transform_placement=post-rms-post-rope-pre-store\n"
        "ordinary_kv_output_inverse=after-complete-attention-before-sigmoid-"
        "gate\n"
        "ordinary_kv_indexer_transform=none\n"
        "ordinary_kv_gpu_transform_equality=nonzero-bit-exact-zero-sign-"
        "equivalent\n";
  }
  identity +=
      "ordinary_kv_key_insertion=post-f32-rope\n"
      "ordinary_kv_value_insertion=f32-projection\n"
      "indexer_pool_dtype=f16\n"
      "indexer_query_raw_dtype=f32\n"
      "gdn_state_dtype=f32\n";
  const auto epoch = std::to_string(ExpectedKvPayloadVersion(type));
  identity += "numerical_kernel_epoch=" + epoch +
              "\nnumerical_build_lane=rocm-qfn-q8-wmma-v" + epoch + "\n";
  return identity;
}

ContinuationLayout::ContinuationLayout(const Config& c, bool mtp,
                                       std::uint32_t context,
                                       std::uint32_t kept_rows,
                                       KvCacheType type)
    : context_(context), ratio_(c.compress_ratio), mtp_(mtp) {
  if (!context || context > c.MaxContextLength() || !ratio_ ||
      !c.full_attention_interval || !c.num_layers || !c.AttentionKvDim() ||
      !c.indexer_head_dim || !c.indexer_top_k || !c.ssm_conv_kernel ||
      !c.ssm_head_dim || !c.ssm_num_v_heads || !c.HcDim() || !c.vocab_size ||
      (mtp && !kept_rows))
    throw std::invalid_argument("invalid Flash-Next continuation geometry");
  if (KvIsRotated(type) && (c.head_dim != 256 || c.rotary_dim != 64 ||
                            c.num_heads != 24 || c.num_kv_heads != 2))
    throw std::invalid_argument(
        "rotated Flash-Next KV requires D256/R64/24Q/2KV");
  const auto kv_row_bytes = KvRowBytes(c.AttentionKvDim(), type);
  raw_capacity_ = std::min(context, std::max(c.indexer_top_k, ratio_ - 1));
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0,
                       std::uint32_t version = 1) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            version,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            row ? cache::Rows{256} : 0,
                            state},
                           part,
                           layer});
  };
  add(ContinuationPart::kTokens, sizeof(std::int32_t), 0);
  add(ContinuationPart::kLogits, 0, std::size_t{c.vocab_size} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if (c.IsLinearLayer(layer)) {
      add(ContinuationPart::kConvolution, 0,
          std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels() *
              sizeof(float),
          layer);
      add(ContinuationPart::kRecurrent, 0,
          std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim *
              sizeof(float),
          layer);
    } else {
      add(ContinuationPart::kKey, kv_row_bytes, 0, layer,
          KvComponentLayoutVersion(type));
      add(ContinuationPart::kValue, kv_row_bytes, 0, layer,
          KvComponentLayoutVersion(type));
      add(ContinuationPart::kPooledKeys, std::size_t{c.indexer_head_dim} * 2, 0,
          layer);
      add(ContinuationPart::kRawKeys, 0,
          std::size_t{raw_capacity_} * c.indexer_head_dim * sizeof(float),
          layer);
    }
  }
  if (c.ple_layer >= 0)
    add(ContinuationPart::kPleHistory, 0,
        std::size_t{c.PleConvHistory()} * c.HcDim() * sizeof(float));
  if (mtp) {
    add(ContinuationPart::kDraftKey, kv_row_bytes, 0, 0,
        KvComponentLayoutVersion(type));
    add(ContinuationPart::kDraftValue, kv_row_bytes, 0, 0,
        KvComponentLayoutVersion(type));
    add(ContinuationPart::kDraftPooledKeys, std::size_t{c.indexer_head_dim} * 2,
        0);
    add(ContinuationPart::kDraftRawKeys, 0,
        std::size_t{raw_capacity_} * c.indexer_head_dim * sizeof(float));
    add(ContinuationPart::kDraftResidual, 0,
        std::size_t{c.HcDim()} * sizeof(float));
    add(ContinuationPart::kKeptHidden, 0,
        std::size_t{kept_rows} * c.HcDim() * sizeof(float));
  }
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t target, std::uint32_t draft, std::uint32_t blocks,
    std::uint32_t draft_blocks) const {
  if (target > context_ || draft > target || blocks > target / ratio_ ||
      draft_blocks > draft / ratio_ ||
      target - blocks * ratio_ > raw_capacity_ ||
      draft - draft_blocks * ratio_ > raw_capacity_ ||
      (!mtp_ && (draft || draft_blocks)))
    throw std::invalid_argument("invalid Flash-Next continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_) {
    const bool draft_part = IsDraft(component.part);
    // Private state belongs to this target checkpoint, including the lagging
    // predictor's residual and raw ring. Its execution frontier is recorded
    // by the draft row components and private metadata.
    positions.push_back(
        {component.descriptor.id,
         component.descriptor.kind == cache::ComponentKind::kPrivateState
             ? target
         : IsPool(component.part) ? (draft_part ? draft_blocks : blocks)
                                  : (draft_part ? draft : target)});
  }
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}
std::vector<ContinuationBackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram, std::uint32_t concurrency) const {
  if (!concurrency || concurrency > 8)
    throw std::invalid_argument("invalid continuation backing concurrency");
  // Private frontiers, rows and tails share the same committed pages. Fixed
  // per-component classes can exhaust private backing while rows remain free.
  // Contiguous page groups retain one payload owner and charge all slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t target,
                                         std::uint32_t draft,
                                         std::uint32_t blocks,
                                         std::uint32_t draft_blocks) const {
  const auto positions = Positions(target, draft, blocks, draft_blocks);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}
}  // namespace gufo::models::qwen38_flash_next
