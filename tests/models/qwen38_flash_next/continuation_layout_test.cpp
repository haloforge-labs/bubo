#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;

namespace {
// Frozen production geometry, independently of the layout's descriptor sum:
// logits + 256-byte metadata + 36 GDN layers + 12 raw indexer rings + PLE.
constexpr std::size_t kPrivateArBytes =
    248320ULL * 4 + 256 +
    36ULL * ((4 - 1) * (2 * 16 * 128 + 48 * 128) + 48 * 128 * 128) * 4 +
    12ULL * 2048 * 128 * 4 + (4 - 1) * 3ULL * (4 * 2560) * 4;
constexpr std::size_t kPrivateMtpBytes =
    kPrivateArBytes + 2048ULL * 128 * 4 + (1 + 8) * (4 * 2560ULL) * 4;
static_assert(qfn::kContinuationMetadataBytes == 256);
static_assert(kPrivateArBytes == 131614976 && kPrivateMtpBytes == 133032192);

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

qfn::Config ProductionGeometry() {
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.context_length = 262144;
  c.hidden_size = 2560;
  c.hc_count = 4;
  c.vocab_size = 248320;
  c.num_kv_heads = 2;
  c.num_heads = 24;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}

template<class F>
void Reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid geometry or frontier accepted");
}

void Check(bool mtp) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout layout(c, mtp, 100000, 8);
  Require(layout.Components().size() == (mtp ? 130 : 124),
          "state inventory omitted a component");
  Require(layout.PrivateBytes() == (mtp ? kPrivateMtpBytes : kPrivateArBytes),
          "private byte claim differs from independent geometry");
  Require(layout.RawCapacity() == 2048, "raw ring reserve is not bounded");
  for (const auto target : {32768U, 100000U}) {
    const auto draft = mtp ? target - 1 : 0;
    const auto positions =
        layout.Positions(target, draft, target / 4, draft / 4);
    // Twelve target attention layers, thirteen with the predictor. Two
    // 512-element f16 KV rows each; one 128-element f16 pool per four tokens;
    // host token history has four bytes per target token.
    const auto expected =
        std::size_t{target} * 25348 +
        (mtp ? std::size_t{draft} * 2048 + std::size_t{draft / 4} * 256 : 0);
    Require(layout.RowBytes(target, draft, target / 4, draft / 4) == expected,
            "row byte claim used target position for the draft or pools");
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const auto& component = layout.Components()[i];
      Require(component.descriptor.id.value == i + 1,
              "component IDs are not unique and stable");
      if (component.descriptor.kind == cache::ComponentKind::kPrivateState)
        Require(positions[i].valid_rows == target,
                "private state does not attest the checkpoint boundary");
      if (component.part == qfn::ContinuationPart::kDraftKey)
        Require(positions[i].valid_rows == draft,
                "predictor KV lost its independent frontier");
      if (component.part == qfn::ContinuationPart::kDraftPooledKeys)
        Require(positions[i].valid_rows == draft / 4,
                "predictor pools are counted in tokens");
    }
  }
  // Before sparse attention starts, every raw row is private. Afterwards a
  // checkpoint can hold an unfinished pool and a lagging predictor together.
  (void)layout.Positions(2048, mtp ? 2047 : 0, 0, 0);
  (void)layout.Positions(2051, mtp ? 2049 : 0, 512, mtp ? 512 : 0);
  Reject([&] { (void)layout.Positions(2049, 0, 0, 0); });
  Reject([&] { (void)layout.Positions(100001, 0, 25000, 0); });
  Reject([&] { (void)layout.Positions(10, 11, 2, 2); });
  Reject([&] { (void)layout.Positions(10, 0, 3, 0); });
  Reject([&] { (void)layout.Positions(10, mtp ? 9 : 0, 2, 3); });
  if (!mtp)
    Reject([&] { (void)layout.Positions(10, 1, 2, 0); });
  const qfn::ContinuationLayout short_layout(c, mtp, 1000, 8);
  Require(short_layout.RawCapacity() == 1000,
          "short contexts reserve an unreachable raw frontier");
  auto no_ple = c;
  no_ple.ple_layer = -1;
  const qfn::ContinuationLayout no_ple_layout(no_ple, mtp, 100000, 8);
  Require(layout.PrivateBytes() - no_ple_layout.PrivateBytes() == 368640,
          "PLE history inventory is incorrect");
}
void CheckQuantized(bool mtp) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout f16(c, mtp, 200000, 8);
  const qfn::ContinuationLayout q8(c, mtp, 200000, 8, qfn::KvCacheType::kQ8_0);
  Require(q8.Components().size() == f16.Components().size(),
          "q8 changed the component inventory");
  Require(q8.PrivateBytes() == f16.PrivateBytes(),
          "q8 changed GDN, indexer rings or other private state");
  std::size_t packed_components = 0;
  for (std::size_t i = 0; i < q8.Components().size(); ++i) {
    const auto& packed = q8.Components()[i];
    const auto& original = f16.Components()[i];
    Require(packed.part == original.part && packed.layer == original.layer &&
                packed.descriptor.id == original.descriptor.id,
            "q8 changed stable component identities");
    const bool ordinary = packed.part == qfn::ContinuationPart::kKey ||
                          packed.part == qfn::ContinuationPart::kValue ||
                          packed.part == qfn::ContinuationPart::kDraftKey ||
                          packed.part == qfn::ContinuationPart::kDraftValue;
    if (ordinary) {
      ++packed_components;
      Require(packed.descriptor.row_bytes == 544 &&
                  original.descriptor.row_bytes == 1024 &&
                  packed.descriptor.layout_version == 2 &&
                  original.descriptor.layout_version == 1,
              "ordinary KV did not use sixteen canonical 34-byte blocks");
    } else {
      const auto& p = packed.descriptor;
      const auto& o = original.descriptor;
      Require(p.kind == o.kind && p.layout_version == o.layout_version &&
                  p.row_bytes == o.row_bytes &&
                  p.state_bytes == o.state_bytes &&
                  p.rows_per_chunk == o.rows_per_chunk,
              "q8 changed tokens, pooled indexer keys or private descriptors");
    }
  }
  Require(packed_components == (mtp ? 26 : 24),
          "not all trunk and MTP ordinary K/V components are packed");
  for (const auto target :
       {1U, 2U, 3U, 255U, 256U, 257U, 2051U, 100000U, 200000U}) {
    for (const auto lag : {0U, 1U, 3U}) {
      const auto draft = mtp && target > lag ? target - lag : 0;
      const auto blocks = target / 4, draft_blocks = draft / 4;
      Require(q8.Positions(target, draft, blocks, draft_blocks) ==
                  f16.Positions(target, draft, blocks, draft_blocks),
              "q8 changed target/draft/pool frontiers");
      const std::size_t expected =
          std::size_t{target} * 13060 + std::size_t{blocks} * 3072 +
          std::size_t{draft} * 1088 + std::size_t{draft_blocks} * 256;
      Require(q8.RowBytes(target, draft, blocks, draft_blocks) == expected,
              "packed byte arithmetic lost a tail, pool or lagging draft");
      if (mtp && !lag && target % 4 == 0)
        Require(q8.PrivateBytes() + expected ==
                    kPrivateMtpBytes + 14980ULL * target,
                "q8 caught-up payload disagrees with the study");
    }
  }
  auto bad = c;
  bad.head_dim = 255;
  Reject([&] {
    qfn::ContinuationLayout layout(bad, mtp, 1000, 8, qfn::KvCacheType::kQ8_0);
  });
  Reject([&] {
    qfn::ContinuationLayout layout(c, mtp, 1000, 8,
                                   static_cast<qfn::KvCacheType>(99));
  });
}

void CheckRotated(bool mtp) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout plain(c, mtp, 200000, 8,
                                      qfn::KvCacheType::kQ8_0);
  const qfn::ContinuationLayout rotated(c, mtp, 200000, 8,
                                        qfn::KvCacheType::kQ8_0Rotated);
  Require(plain.PrivateBytes() == rotated.PrivateBytes(),
          "rotation changed private capacity");
  Require(plain.Components().size() == rotated.Components().size(),
          "rotation changed inventory");
  for (std::size_t i = 0; i < plain.Components().size(); ++i) {
    auto p = plain.Components()[i].descriptor;
    const auto r = rotated.Components()[i].descriptor;
    if (p.layout_version == 2) {
      Require(r.layout_version == 3 && r.row_bytes == 544,
              "rotated component is not v3");
      p.layout_version = 3;
    }
    Require(p.id == r.id && p.kind == r.kind && p.row_bytes == r.row_bytes &&
                p.rows_per_chunk == r.rows_per_chunk &&
                p.state_bytes == r.state_bytes &&
                p.layout_version == r.layout_version,
            "rotation changed component geometry");
  }
  for (auto target : {1U, 255U, 256U, 257U, 2048U, 2051U, 100000U, 200000U}) {
    const auto draft = mtp ? target - 1 : 0;
    Require(plain.Positions(target, draft, target / 4, draft / 4) ==
                    rotated.Positions(target, draft, target / 4, draft / 4) &&
                plain.RowBytes(target, draft, target / 4, draft / 4) ==
                    rotated.RowBytes(target, draft, target / 4, draft / 4),
            "rotation changed frontiers/capacity");
  }
  for (unsigned field = 0; field < 4; ++field) {
    auto bad = c;
    if (field == 0)
      bad.head_dim = 128;
    if (field == 1)
      bad.rotary_dim = 128;
    if (field == 2)
      bad.num_heads = 12;
    if (field == 3)
      bad.num_kv_heads = 1;
    Reject([&] {
      qfn::ContinuationLayout layout(bad, mtp, 1000, 8,
                                     qfn::KvCacheType::kQ8_0Rotated);
    });
  }
}

void CheckIdentity() {
  static_assert(qfn::Session::kSnapshotPayloadVersion == 21);
  Require(qfn::ModelOptions{}.kv_cache_type == qfn::KvCacheType::kF16,
          "model KV default is not F16");
  for (const auto expected :
       {qfn::KvCacheType::kF16, qfn::KvCacheType::kQ8_0}) {
    Require(qfn::Session::IsCompatibleSnapshotPayload(
                qfn::kKvCacheNumericsEpoch, expected, expected),
            "current matching snapshot ABI was rejected");
    for (const auto version : {0U, 17U, 18U, 19U, 20U, 22U, 23U})
      Require(!qfn::Session::IsCompatibleSnapshotPayload(version, expected,
                                                         expected),
              "old or unknown snapshot epoch was accepted");
    const auto other = expected == qfn::KvCacheType::kF16
                           ? qfn::KvCacheType::kQ8_0
                           : qfn::KvCacheType::kF16;
    Require(!qfn::Session::IsCompatibleSnapshotPayload(
                qfn::kKvCacheNumericsEpoch, other, expected),
            "cross-dtype snapshot was accepted");
    Require(!qfn::Session::IsCompatibleSnapshotPayload(
                qfn::kKvCacheNumericsEpoch, static_cast<qfn::KvCacheType>(99),
                expected),
            "unknown snapshot dtype was accepted");
  }
  Require(!qfn::Session::IsCompatibleSnapshotPayload(
              qfn::kKvCacheNumericsEpoch, static_cast<qfn::KvCacheType>(99),
              static_cast<qfn::KvCacheType>(99)),
          "matching unknown dtypes bypassed the snapshot gate");
  const auto f16 = qfn::KvCacheCompatibilityIdentity(qfn::KvCacheType::kF16);
  const auto q8 = qfn::KvCacheCompatibilityIdentity(qfn::KvCacheType::kQ8_0);
  // Independent current-format byte expectations: no normalization or source
  // extraction can hide changes to ordering, spelling, arithmetic or basis.
  const std::string f16_storage =
      "ordinary_kv_dtype=f16\n"
      "ordinary_kv_scope=all-trunk-and-mtp\n"
      "ordinary_kv_layout=row-major-f16-le-v1\n"
      "ordinary_kv_block_elements=1\n"
      "ordinary_kv_scale_dtype=none\n"
      "ordinary_kv_rounding_saturation_revision=f16-rne-v1\n"
      "ordinary_kv_reconstruction=f16-identity-v1\n";
  const std::string q8_storage =
      "ordinary_kv_scope=all-trunk-and-mtp\n"
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
  const std::string rotation =
      "ordinary_kv_basis=hadamard-rotated-v1\n"
      "ordinary_kv_key_query_transform=h256-per-head\n"
      "ordinary_kv_value_output_transform=4-contiguous-h64-per-head\n"
      "ordinary_kv_transform_order=natural-sylvester-no-signs-no-permutation\n"
      "ordinary_kv_normalization=first-f32-rne-h256-1/16-h64-1/8\n"
      "ordinary_kv_butterfly=ascending-distance-f32-rne-low-u+v-high-u-v-v1\n"
      "ordinary_kv_transform_placement=post-rms-post-rope-pre-store\n"
      "ordinary_kv_output_inverse=after-complete-attention-before-sigmoid-"
      "gate\n"
      "ordinary_kv_indexer_transform=none\n"
      "ordinary_kv_gpu_transform_equality=nonzero-bit-exact-zero-sign-"
      "equivalent\n";
  const std::string state =
      "ordinary_kv_key_insertion=post-f32-rope\n"
      "ordinary_kv_value_insertion=f32-projection\n"
      "indexer_pool_dtype=f16\n"
      "indexer_query_raw_dtype=f32\n"
      "gdn_state_dtype=f32\n";
  const std::string plain_version =
      "numerical_kernel_epoch=21\n"
      "numerical_build_lane=rocm-qfn-q8-wmma-v21\n";
  const std::string rotated_version =
      "numerical_kernel_epoch=22\n"
      "numerical_build_lane=rocm-qfn-q8-wmma-v22\n";
  Require(f16 == f16_storage + state + plain_version,
          "F16 identity differs from current expected bytes");
  Require(q8 == "ordinary_kv_dtype=q8_0\n" + q8_storage + state + plain_version,
          "plain q8 identity differs from current expected bytes");
  Require(qfn::KvCacheCompatibilityIdentity(qfn::KvCacheType::kQ8_0Rotated) ==
              "ordinary_kv_dtype=q8_0-rotated\n" + q8_storage + rotation +
                  state + rotated_version,
          "rotated q8 identity differs from current expected bytes");
  Require(f16 != q8 &&
              f16.find("ordinary_kv_dtype=f16\n") != std::string::npos &&
              q8.find("ordinary_kv_dtype=q8_0\n") != std::string::npos,
          "numeric identity does not distinguish storage dtypes");
  for (const auto* identity : {&f16, &q8}) {
    for (const auto field :
         {"ordinary_kv_layout=", "ordinary_kv_block_elements=",
          "ordinary_kv_scale_dtype=",
          "ordinary_kv_rounding_saturation_revision=",
          "ordinary_kv_reconstruction=",
          "ordinary_kv_key_insertion=post-f32-rope\n",
          "indexer_pool_dtype=f16\n", "gdn_state_dtype=f32\n",
          "numerical_kernel_epoch=21\n", "numerical_build_lane="})
      Require(identity->find(field) != std::string::npos,
              "numeric identity omitted a storage/arithmetic contract");
  }
  Require(q8.find("roundf-away-clamp127-scale-f16-rne-v1") != std::string::npos,
          "q8 identity omitted canonical rounding and saturation");
  const std::string equality =
      "ordinary_kv_gpu_reconstruction_equality=ieee-values-zero-sign-"
      "equivalent-v1\n";
  const std::string underflow =
      "ordinary_kv_f16_scale_underflow=preserve-codes-value-zero-v1\n";
  Require(q8.find(equality) != std::string::npos &&
              q8.find(underflow) != std::string::npos &&
              q8.find("preserve-codes-signed-zero-v1") == std::string::npos &&
              f16.find("ordinary_kv_gpu_reconstruction_equality=") ==
                  std::string::npos,
          "GPU value equality contract is missing or changed F16 identity");
  // Correcting the old zero-sign promise changes the q8 identity input to
  // component/disk fingerprints even without an epoch change. Never alias it.
  auto legacy_q8 = q8;
  legacy_q8.erase(legacy_q8.find(equality), equality.size());
  legacy_q8.replace(legacy_q8.find(underflow), underflow.size(),
                    "ordinary_kv_f16_scale_underflow="
                    "preserve-codes-signed-zero-v1\n");
  const auto fingerprint = [](const std::string& value) {
    return gufo::crypto::Sha256Hex(
        {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
  };
  Require(fingerprint(q8) != fingerprint(legacy_q8),
          "corrected q8 contract did not change its numeric fingerprint");
  Reject([&] {
    (void)qfn::KvCacheCompatibilityIdentity(static_cast<qfn::KvCacheType>(99));
  });
}

void CheckExtendedContext(bool mtp, qfn::KvCacheType type) {
  const auto native = ProductionGeometry();
  Reject([&] {
    qfn::ContinuationLayout layout(native, mtp, native.context_length + 1, 8);
  });
  for (const auto context : {409600U, 655360U}) {
    auto scaled = native;
    scaled.rope_scaling = {.factor = context == 409600 ? 1.5625F : 2.5F,
                           .original_context = 262144};
    const qfn::ContinuationLayout layout(scaled, mtp, context, 8, type);
    Require(scaled.context_length == 262144,
            "continuation scaling rewrote the native context");
    Require(layout.Components().size() == (mtp ? 130 : 124),
            "extended context changed the state inventory");
    Require(layout.RawCapacity() == 2048,
            "extended context inflated the bounded raw ring");
    for (const auto target : {262145U, context}) {
      const auto draft = mtp ? target - 1 : 0;
      // Twelve pairs of canonical KV rows plus unchanged pool/token bytes.
      const std::size_t pair_bytes =
          type == qfn::KvCacheType::kF16 ? 2048 : 1088;
      const auto expected =
          std::size_t{target} * (12 * pair_bytes + 4) +
          std::size_t{target / 4} * 3072 +
          (mtp ? std::size_t{draft} * pair_bytes + std::size_t{draft / 4} * 256
               : 0);
      Require(layout.RowBytes(target, draft, target / 4, draft / 4) == expected,
              "extended frontier lost target or predictor rows");
    }
    Reject([&] {
      qfn::ContinuationLayout too_large(scaled, mtp, context + 1, 8);
    });
    Reject([&] { (void)layout.Positions(context + 1, 0, context / 4, 0); });
  }
}
void CheckBacking(bool mtp, std::uint32_t context, std::uint32_t concurrency,
                  std::size_t ram, std::size_t checkpoints,
                  std::size_t independent_rows, qfn::KvCacheType type) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout layout(c, mtp, context, 8, type);
  std::size_t available = 0, required = 0;
  std::size_t bytes = 0;
  for (const auto& allocation : layout.BackingPlan(ram, concurrency)) {
    Require(allocation.block_bytes % 4096 == 0 && allocation.blocks != 0,
            "invalid physical backing class");
    Require(allocation.metadata_bytes >= 4096 + allocation.blocks * 1024,
            "backing plan omitted allocator headroom");
    Require(allocation.block_bytes == 65536,
            "backing pages cannot serve all components");
    available += allocation.blocks;
    bytes +=
        allocation.block_bytes * allocation.blocks + allocation.metadata_bytes;
  }
  Require(bytes <= (ram - (4ULL << 20)) / 10 * 9,
          "physical backing exceeds the startup budget");
  for (const auto& component : layout.Components()) {
    const auto& d = component.descriptor;
    const auto payload = d.kind == cache::ComponentKind::kPrivateState
                             ? d.state_bytes
                             : d.rows_per_chunk * d.row_bytes;
    const auto pages = (payload + 65535) / 65536;
    if (d.kind == cache::ComponentKind::kPrivateState)
      required += pages * checkpoints;
    else {
      const bool pooled =
          component.part == qfn::ContinuationPart::kPooledKeys ||
          component.part == qfn::ContinuationPart::kDraftPooledKeys;
      const auto rows =
          pooled ? independent_rows / c.compress_ratio : independent_rows;
      required += pages * ((rows + d.rows_per_chunk - 1) / d.rows_per_chunk +
                           checkpoints);
    }
  }
  // Independent physical demands, including shared-class collisions and a
  // partial tail per checkpoint. A row-heavy pool must not strand most of its
  // budget while private state already refuses feasible histories.
  Require(available >= required,
          "feasible capture envelope does not fit committed pages");
  Require(layout.BackingPlan(1 << 20, concurrency).empty(),
          "tiny budgets should leave cache metadata headroom");
  Reject([&] { (void)layout.BackingPlan(ram, 0); });
  Reject([&] { (void)layout.BackingPlan(ram, 9); });
}
}  // namespace

int main() {
  try {
    Check(false);
    Check(true);
    CheckIdentity();
    for (const auto mtp : {false, true}) {
      CheckQuantized(mtp);
      CheckRotated(mtp);
      for (const auto type :
           {qfn::KvCacheType::kF16, qfn::KvCacheType::kQ8_0}) {
        CheckExtendedContext(mtp, type);
        CheckBacking(mtp, 32768, 2, 8ULL << 30, 32, 2 * 32768, type);
        CheckBacking(mtp, 32768, 2, 2ULL << 30, 8, 2 * 4096, type);
        CheckBacking(mtp, 4096, 8, 8ULL << 30, 8, 8 * 4096, type);
        CheckBacking(mtp, 133760, 1, 8ULL << 30, 8, 2 * 100000, type);
      }
    }
    const auto c = ProductionGeometry();
    Reject([&] { qfn::ContinuationLayout layout(c, false, 0, 8); });
    Reject([&] { qfn::ContinuationLayout layout(c, true, 100000, 0); });
    auto bad = c;
    bad.compress_ratio = 0;
    Reject([&] { qfn::ContinuationLayout layout(bad, false, 100000, 8); });
    std::cout << "Flash-Next continuation geometry: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
