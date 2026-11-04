// Native real-profile YaRN regression for snapshot_test's explicit mode.
// Keep kv_rotation_producer_native.cpp's P0/default-YaRN fixture unchanged.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "tests/models/qwen38_flash_next/separate_mtp_test_support.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace q = qfn::rocm;
namespace {
constexpr unsigned kThreshold = 2048, kKept = 8, kBatch = 64;
constexpr unsigned kTrunkPosition = kThreshold + kKept;
constexpr unsigned kContext = kTrunkPosition;
constexpr float kFactor = 1.5625F;
constexpr const char* kBuild = "ON";
using Bytes = std::vector<std::uint8_t>;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
void Hip(hipError_t error) {
  Require(error == hipSuccess, hipGetErrorString(error));
}
unsigned Epoch(qfn::KvCacheType type) {
  return qfn::ExpectedKvPayloadVersion(type);
}

// Named wire probe bound to actual captured geometry, byte length and scaling.
// Position 2056 and kept-hidden count 8 are deliberately different quantities.
struct Header {
  std::array<char, 8> magic;
  std::uint32_t layers, interval, conv, state, kv_row, type, epoch;
  std::uint32_t index_dim, ratio, hc_dim, ple, has_mtp;
  std::uint32_t position, blocks, mtp_position, mtp_blocks, residual, hidden;
  std::uint32_t images;
  float rope_factor;
  std::uint32_t rope_context;
  float rope_fast, rope_slow;
  std::array<std::int32_t, qfn::Config::kMaxPleNgram - 1> ngram;
  std::uint64_t bytes;
};
static_assert(sizeof(Header) == 120 && offsetof(Header, bytes) == 112);
enum class Part { kKey, kValue, kRaw, kPool };
struct Region {
  bool draft;
  unsigned layer;
  Part part;
  std::size_t offset, size;
};
struct Frame {
  Bytes bytes;
  Header header;
  std::vector<Region> regions;
  std::span<const std::uint8_t> Data(const Region& region) const {
    return std::span(bytes).subspan(region.offset, region.size);
  }
  std::span<const std::uint8_t> Draft(Part part) const {
    for (const auto& region : regions)
      if (region.draft && region.part == part)
        return Data(region);
    throw std::runtime_error("missing captured draft region");
  }
};
void Exact(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b,
           const std::string& what) {
  Require(a.size() == b.size(), what + ": size mismatch");
  const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
  if (mismatch.first != a.end())
    throw std::runtime_error(what + ": first different byte=" +
                             std::to_string(mismatch.first - a.begin()));
}
Frame Capture(q::Executor& exec, q::Session& session, qfn::KvCacheType type,
              unsigned position = kTrunkPosition) {
  std::string error;
  Frame frame;
  frame.bytes.resize(exec.SnapshotBytes(session, kKept));
  Require(exec.SaveSnapshot(session, kKept, frame.bytes, &error), error);
  Require(frame.bytes.size() >= sizeof(Header), "short native frame");
  auto& h = frame.header;
  std::memcpy(&h, frame.bytes.data(), sizeof(h));
  const auto& c = exec.config();
  const auto rope = qfn::NormalizedRopeScaling(c.rope_scaling);
  Require(
      h.magic == std::array<char, 8>{'Q', 'F', 'N', 'S', 'N', 'A', 'P', '8'} &&
          h.layers == c.num_layers && h.interval == c.full_attention_interval &&
          h.conv == (c.ssm_conv_kernel - 1) * c.SsmConvChannels() &&
          h.state == c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim &&
          h.kv_row == c.AttentionKvDim() && h.type == unsigned(type) &&
          h.epoch == Epoch(type) && h.index_dim == c.indexer_head_dim &&
          h.ratio == c.compress_ratio && h.hc_dim == c.HcDim() &&
          h.ple == (c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0) &&
          h.has_mtp == 1 && h.position == position && h.hidden == kKept &&
          h.images == 0 && h.bytes == frame.bytes.size() &&
          h.blocks * h.ratio <= h.position &&
          h.mtp_blocks * h.ratio <= h.mtp_position &&
          h.mtp_position <= h.position && h.residual <= 1 &&
          h.rope_factor == rope.factor &&
          h.rope_context == rope.original_context &&
          h.rope_fast == rope.beta_fast && h.rope_slow == rope.beta_slow,
      "native YaRN wire probe/geometry/scaling drift");
  std::size_t offset = sizeof(Header);
  const auto skip = [&](std::size_t size) {
    Require(offset <= frame.bytes.size() && size <= frame.bytes.size() - offset,
            "native frame region outside payload");
    offset += size;
  };
  const auto add = [&](bool draft, unsigned layer, Part part,
                       std::size_t size) {
    const auto first = offset;
    skip(size);
    frame.regions.push_back({draft, layer, part, first, size});
    const auto data = frame.Data(frame.regions.back());
    if (part == Part::kKey || part == Part::kValue) {
      const auto row_bytes = qfn::KvRowBytes(h.kv_row, type);
      Require(size % row_bytes == 0, "partial captured KV row");
      for (std::size_t row = 0; row < size; row += row_bytes)
        qfn::ValidateKvRow(std::as_bytes(data.subspan(row, row_bytes)),
                           h.kv_row, type);
    } else if (part == Part::kPool) {
      Require(size % 2 == 0, "partial captured F16 pool value");
      for (std::size_t i = 0; i < size; i += 2)
        Require(
            std::isfinite(qfn::KvHalfBitsToFloat(
                std::uint16_t(data[i]) | (std::uint16_t(data[i + 1]) << 8))),
            "nonfinite captured pooled indexer value");
    } else {
      Require(size % sizeof(float) == 0, "partial raw indexer value");
      for (std::size_t i = 0; i < size; i += sizeof(float)) {
        float value;
        std::memcpy(&value, data.data() + i, sizeof(value));
        Require(std::isfinite(value), "nonfinite captured raw indexer value");
      }
    }
  };
  for (unsigned layer = 0; layer < h.layers; ++layer)
    if (c.IsLinearLayer(layer))
      skip(std::size_t(h.conv + h.state) * sizeof(float));
  skip(std::size_t(h.ple) * sizeof(float));
  const auto kv_row = qfn::KvRowBytes(h.kv_row, type);
  for (unsigned layer = 0; layer < h.layers; ++layer) {
    if (c.IsLinearLayer(layer))
      continue;
    add(false, layer, Part::kKey, h.position * kv_row);
    add(false, layer, Part::kValue, h.position * kv_row);
    add(false, layer, Part::kRaw,
        std::size_t(h.position - h.blocks * h.ratio) * h.index_dim *
            sizeof(float));
    add(false, layer, Part::kPool,
        std::size_t(h.blocks) * h.index_dim * sizeof(std::uint16_t));
  }
  add(true, h.layers, Part::kRaw,
      std::size_t(h.mtp_position - h.mtp_blocks * h.ratio) * h.index_dim *
          sizeof(float));
  add(true, h.layers, Part::kPool,
      std::size_t(h.mtp_blocks) * h.index_dim * sizeof(std::uint16_t));
  add(true, h.layers, Part::kKey, h.mtp_position * kv_row);
  add(true, h.layers, Part::kValue, h.mtp_position * kv_row);
  skip(std::size_t(h.residual ? h.hc_dim : 0) * sizeof(float));
  skip(std::size_t(h.hidden) * h.hc_dim * sizeof(float));
  Require(offset == frame.bytes.size() && frame.regions.size() == 52,
          "native YaRN region inventory drift");
  return frame;
}
constexpr bool EqualHalfBits(std::uint16_t a, std::uint16_t b) {
  return a == b || ((a & 0x7fffU) == 0 && (b & 0x7fffU) == 0);
}
static_assert(EqualHalfBits(0, 0x8000) && EqualHalfBits(0x8000, 0));
static_assert(!EqualHalfBits(0, 1) && !EqualHalfBits(0x3c00, 0xbc00));
void HalfValues(std::span<const std::uint8_t> a,
                std::span<const std::uint8_t> b, const std::string& what) {
  Require(a.size() == b.size() && a.size() % 2 == 0,
          what + ": F16 size mismatch");
  for (std::size_t i = 0; i < a.size(); i += 2) {
    const auto x = std::uint16_t(a[i]) | (std::uint16_t(a[i + 1]) << 8);
    const auto y = std::uint16_t(b[i]) | (std::uint16_t(b[i + 1]) << 8);
    Require(EqualHalfBits(x, y),
            what + ": different F16 value at byte=" + std::to_string(i));
  }
}
void SameComponents(const Frame& a, const Frame& b, bool ordinary_only,
                    const std::string& what) {
  Require(a.regions.size() == b.regions.size(), "region inventory mismatch");
  for (std::size_t i = 0; i < a.regions.size(); ++i) {
    const auto& x = a.regions[i];
    const auto& y = b.regions[i];
    Require(x.draft == y.draft && x.layer == y.layer && x.part == y.part,
            "region ordering mismatch");
    if (ordinary_only && x.draft)
      continue;
    const auto label = what + (x.draft ? ":draft:" : ":ordinary:") +
                       std::to_string(x.layer) +
                       ":part=" + std::to_string(unsigned(x.part));
    // Packed q8 bytes stay strict. Existing GPU F16 value equality admits
    // either zero sign, but requires every nonzero bit pattern unchanged.
    if (x.part == Part::kPool ||
        (a.header.type == unsigned(qfn::KvCacheType::kF16) &&
         (x.part == Part::kKey || x.part == Part::kValue)))
      HalfValues(a.Data(x), b.Data(y), label);
    else
      Exact(a.Data(x), b.Data(y), label);
  }
}
void Frontier(const Frame& frame, unsigned draft_position) {
  const auto& h = frame.header;
  Require(h.position == kTrunkPosition && h.hidden == kKept &&
              h.blocks == 514 && h.mtp_position == draft_position &&
              draft_position > kThreshold &&
              h.mtp_blocks == draft_position / 4 && h.mtp_blocks >= 512 &&
              frame.Draft(Part::kPool).size() ==
                  std::size_t(h.mtp_blocks) * 128 * 2 &&
              frame.Draft(Part::kRaw).size() ==
                  std::size_t(draft_position % 4) * 128 * 4,
          "actual 2048-threshold pooling frontier/count mismatch");
  const auto pool = frame.Draft(Part::kPool);
  Require(std::any_of(pool.begin(), pool.end(),
                      [](auto byte) { return byte != 0; }),
          "vacuous all-zero pooled indexer fixture");
}
std::unique_ptr<q::Session> NewSession(q::Executor& exec) {
  std::string error;
  auto session = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                    kContext, &error);
  Require(bool(session), error);
  return session;
}
std::unique_ptr<q::Session> Fork(q::Executor& exec, const Frame& frame) {
  auto session = NewSession(exec);
  q::Executor::SnapshotInfo info;
  std::string error;
  Require(exec.RestoreSnapshot(*session, frame.bytes, &info, &error), error);
  Require(info.position == kTrunkPosition && info.hidden_rows == kKept,
          "restore confused token position with kept hidden count");
  return session;
}
void FreshRestore(q::Executor& exec, std::unique_ptr<q::Session>& session,
                  const Frame& frame, qfn::KvCacheType type) {
  session.reset();
  session = Fork(exec, frame);
  const auto restored = Capture(exec, *session, type);
  Exact(frame.bytes, restored.bytes, "fresh native restore payload");
}
Frame MakeBase(q::Executor& exec, qfn::KvCacheType type,
               std::span<const std::int32_t> tokens) {
  auto session = NewSession(exec);
  std::string error;
  for (unsigned first = 0; first < kThreshold; first += kBatch) {
    if (first != 0)
      Require(exec.MtpForward(*session, tokens.subspan(first, 1), kKept - 1,
                              {.kv_only = true}, &error),
              error);
    Require(exec.MtpPosition(*session) == first,
            "prefill MTP catch-up did not reach trunk frontier");
    Require(exec.Forward(*session, tokens.subspan(first, kBatch), 0, nullptr,
                         q::Executor::ForwardMode::kPrefill, &error),
            error);
    Require(session->position() == first + kBatch &&
                exec.MtpPosition(*session) == first + kBatch - 1,
            "production prefill did not retain shifted MTP frontier");
  }
  Require(exec.MtpForward(*session, tokens.subspan(kThreshold, 1), kKept - 1,
                          {.kv_only = true}, &error),
          error);
  {
    const auto threshold = Capture(exec, *session, type, kThreshold);
    Require(threshold.header.position == kThreshold &&
                threshold.header.blocks == 0 &&
                threshold.header.mtp_position == kThreshold &&
                threshold.header.mtp_blocks == 0 &&
                threshold.Draft(Part::kPool).empty() &&
                threshold.Draft(Part::kRaw).size() == kThreshold * 128 * 4,
            "end==top_k unexpectedly pooled before candidate execution");
  }
  // Keep only these eight real trunk hidden rows. No 2048-row verifier/logit
  // allocation and no profile/top_k override or fabricated snapshot frontier.
  Require(exec.Forward(*session, tokens.subspan(kThreshold, kKept), 0, nullptr,
                       q::Executor::ForwardMode::kDecode, &error),
          error);
  auto base = Capture(exec, *session, type);
  Require(base.header.position == kTrunkPosition &&
              base.header.hidden == kKept && base.header.blocks == 514 &&
              base.header.mtp_position == kThreshold &&
              base.header.mtp_blocks == 0 && base.Draft(Part::kPool).empty() &&
              base.Draft(Part::kRaw).size() == kThreshold * 128 * 4,
          "real production base does not expose the intended pooling backlog");
  return base;
}

// Supplemental sensitivity check only: the PRIMARY proof below calls real
// scalar/full/cache-only/batched executors. Replay their real frozen raw keys
// through the EXISTING pool op to confirm scaled vs unscaled nonidentity on
// identical inputs, without uploading a second model or inventing numerics.
struct DeviceBytes {
  void* data = nullptr;
  explicit DeviceBytes(std::size_t size) { Hip(hipMalloc(&data, size)); }
  ~DeviceBytes() {
    if (hipFree(data) != hipSuccess)
      std::abort();
  }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
};
void PoolSensitivity(q::Executor& exec, const q::DeviceModel& model,
                     const Frame& base, const Frame& produced) {
  const auto& c = exec.config();
  const auto yarn =
      qfn::MakeYarnRope(c.rope_scaling, c.rotary_dim, c.rope_theta);
  const auto raw = base.Draft(Part::kRaw);
  constexpr std::size_t pool_bytes = 512 * 128 * sizeof(std::uint16_t);
  Require(raw.size() == kThreshold * 128 * sizeof(float),
          "raw sensitivity input");
  DeviceBytes keys(raw.size()), scaled(pool_bytes), off(pool_bytes), control(8);
  const std::array<std::uint32_t, 2> controls{0, kThreshold};
  Hip(hipMemcpyAsync(keys.data, raw.data(), raw.size(), hipMemcpyHostToDevice,
                     exec.stream()));
  Hip(hipMemcpyAsync(control.data, controls.data(), sizeof(controls),
                     hipMemcpyHostToDevice, exec.stream()));
  const auto launch = [&](void* out, const qfn::YarnRope& scaling) {
    q::PoolIndexerBlocks(
        static_cast<const float*>(keys.data), model.mtp().indexer_k_norm.f32(),
        static_cast<__half*>(out),
        static_cast<const std::uint32_t*>(control.data),
        static_cast<const std::uint32_t*>(control.data) + 1, 1, 512, 4,
        c.indexer_head_dim, c.rotary_dim, c.rope_theta, c.rms_eps, kThreshold,
        exec.stream(), nullptr, scaling);
  };
  launch(scaled.data, yarn);
  launch(off.data, {});
  Bytes a(pool_bytes), b(pool_bytes);
  Hip(hipMemcpyAsync(a.data(), scaled.data, a.size(), hipMemcpyDeviceToHost,
                     exec.stream()));
  Hip(hipMemcpyAsync(b.data(), off.data, b.size(), hipMemcpyDeviceToHost,
                     exec.stream()));
  Hip(hipStreamSynchronize(exec.stream()));
  Exact(a, produced.Draft(Part::kPool).first(pool_bytes),
        "real executor pooled backlog vs scaled sensitivity replay");
  const auto block_one = std::span(a).subspan(128 * 2, 128 * 2);
  Require(
      !std::equal(block_one.begin(), block_one.end(), b.begin() + 128 * 2),
      "scaled/off pool sensitivity is vacuous at nonzero absolute position 4");
}
}  // namespace

void CheckYarnCacheOnlyProducers(const char* model_path, const char* mtp_path,
                                 qfn::KvCacheType type, bool require_stacked) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Require(bool(reader), error);
  auto weights = qfn::ModelWeights::Bind(*reader, &error);
  Require(bool(weights), error);
  auto mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
  Require(bool(mtp_reader), error);
  auto mtp = qfn::MtpWeights::Bind(*mtp_reader, weights->config, &error);
  Require(bool(mtp), error);
  auto& config = weights->config;
  Require(config.indexer_top_k == kThreshold && config.compress_ratio == 4 &&
              config.indexer_head_dim == 128 &&
              config.context_length >= kContext,
          "requires the real production 2048/4 indexer profile");
  config.rope_scaling = {.factor = kFactor,
                         .original_context = config.context_length};
  Require(qfn::ValidateRopeScaling(config.rope_scaling, config.context_length,
                                   &error),
          error);
  const auto yarn = qfn::MakeYarnRope(config.rope_scaling, config.rotary_dim,
                                      config.rope_theta);
  Require(yarn.enabled == 1 && yarn.inv_factor != 1 && yarn.mscale > 1,
          "nondefault static YaRN was not enabled");
  auto device = q::DeviceModel::Upload(*weights, *reader, &*mtp,
                                       mtp_reader.get(), &error);
  Require(bool(device), error);
  const auto& draft = device->mtp();
  if (require_stacked) {
    Require(!draft.attn_qkv.empty(), "requires real stacked-Q8 MTP asset");
  } else {
    gufo::test::UseSeparateMtpViews(*device);
  }
  std::unique_ptr<qfn::NgramTable> ngram;
  if (config.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = qfn::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, config.ple_head_dim, t.type, &error);
    Require(bool(ngram), error);
  }
  auto exec = q::Executor::Create(*device, ngram.get(),
                                  {.max_batch = kBatch,
                                   .max_logit_rows = kKept,
                                   .max_speculative = kKept,
                                   .max_context = kContext,
                                   .kv_cache_type = type},
                                  &error);
  Require(bool(exec), error);
  const std::array<std::int32_t, 8> pattern{42, 57, 81, 63, 44, 90, 72, 51};
  std::vector<std::int32_t> tokens(kTrunkPosition + 1);
  for (unsigned i = 0; i < tokens.size(); ++i)
    tokens[i] = pattern[i % pattern.size()];
  const auto base = MakeBase(*exec, type, tokens);
  const auto tail = std::span(tokens).subspan(kThreshold + 1, kKept);
  std::optional<Frame> canonical;
  const auto check = [&](const Frame& f, const Frame& k, unsigned position) {
    Frontier(f, position);
    Frontier(k, position);
    SameComponents(f, base, true,
                   "full MTP changed frozen ordinary components");
    SameComponents(k, base, true,
                   "cache-only MTP changed frozen ordinary components");
    SameComponents(f, k, false, "full/cache-only K/V/raw/pooled exact bytes");
  };
  for (unsigned chunk : {1U, 3U, 8U}) {
    auto full = Fork(*exec, base), cache = Fork(*exec, base);
    for (unsigned first = 0; first < kKept; first += chunk) {
      const auto n = std::min(chunk, kKept - first);
      const auto part = tail.subspan(first, n);
      Require(exec->MtpForward(*full, part, first, {}, &error), error);
      Require(exec->MtpForward(*cache, part, first, {.kv_only = true}, &error),
              error);
      auto f = Capture(*exec, *full, type);
      const auto k = Capture(*exec, *cache, type);
      check(f, k, kThreshold + first + n);
      FreshRestore(*exec, full, f, type);
      FreshRestore(*exec, cache, k, type);
      if (first + n == kKept) {
        if (canonical)
          SameComponents(*canonical, f, false,
                         "scalar/chunk terminal components");
        else {
          PoolSensitivity(*exec, *device, base, f);
          canonical = std::move(f);
        }
      }
    }
  }
  // Ragged cohorts keep total rows <=8 and both peers finish at position 2056.
  std::array<std::unique_ptr<q::Session>, 2> full{Fork(*exec, base),
                                                  Fork(*exec, base)};
  std::array<std::unique_ptr<q::Session>, 2> cache{Fork(*exec, base),
                                                   Fork(*exec, base)};
  std::array<unsigned, 2> first{};
  for (auto counts : {std::array<unsigned, 2>{3, 4}, {3, 1}, {2, 3}}) {
    const std::array<q::Executor::MtpBatchItem, 2> f{
        {{full[0].get(), tail.subspan(first[0], counts[0]), int(first[0]),
          false},
         {full[1].get(), tail.subspan(first[1], counts[1]), int(first[1]),
          false}}};
    const std::array<q::Executor::MtpBatchItem, 2> k{
        {{cache[0].get(), tail.subspan(first[0], counts[0]), int(first[0]),
          true},
         {cache[1].get(), tail.subspan(first[1], counts[1]), int(first[1]),
          true}}};
    Require(exec->MtpForwardBatch(f, &error), error);
    Require(exec->MtpForwardBatch(k, &error), error);
    for (unsigned peer = 0; peer < 2; ++peer) {
      first[peer] += counts[peer];
      const auto fb = Capture(*exec, *full[peer], type);
      const auto kb = Capture(*exec, *cache[peer], type);
      check(fb, kb, kThreshold + first[peer]);
      FreshRestore(*exec, full[peer], fb, type);
      FreshRestore(*exec, cache[peer], kb, type);
      if (first[peer] == kKept)
        SameComponents(*canonical, fb, false,
                       "scalar/ragged-batch terminal components");
    }
  }
  // Actual native component hashes for coordinator-matched ON/OFF receipts;
  // neither source hashes nor CPU serializer proxy bytes stand in for these.
  for (const auto& region : canonical->regions)
    std::cout << "YARN_NATIVE_COMPONENT rotation_build=" << kBuild
              << " type=" << qfn::KvCacheTypeName(type)
              << " stacked=" << require_stacked << " draft=" << region.draft
              << " layer=" << region.layer << " part=" << unsigned(region.part)
              << " bytes=" << region.size
              << " sha256=" << gufo::crypto::Sha256Hex(canonical->Data(region))
              << '\n';
  std::cout
      << "YARN_PRODUCER_REAL2048 rotation_build=" << kBuild
      << " type=" << qfn::KvCacheTypeName(type)
      << " layout=" << (require_stacked ? "stacked-q8" : "separate-q8-views")
      << " static_yarn_test_override=1.5625 geometry_override=none"
         " weight_bytes=unaltered"
         " trunk=2056 hidden=8 draft_base=2048 pooled_base=0 pooled_final=514"
         " scalar/chunk/batch/full/cache-only=packed_strict_f16_nonzero_exact"
         " f16_zero_sign=equivalent restored_payload_bytes=exact"
         " nonzero_pool_sensitivity=PASS model_quality=NOT_TESTED\n";
}
