// Optional real-model mode of kv_rotation_ops_test. Uses supported stacked
// Q8 MTP weights through test-only separate row views, with the production
// binding/upload/executor factories and actual native cache captures.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "tests/models/qwen38_flash_next/separate_mtp_test_support.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace q = qfn::rocm;
namespace {
constexpr auto kType = qfn::KvCacheType::kQ8_0Rotated;
constexpr unsigned kContext = 64, kHidden = 8;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

// Named QFNSNAP8 wire probe, checked against an actual capture's geometry,
// epoch and total size before using any region. Layout drift is a failure.
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
static_assert(offsetof(Header, bytes) == 112 && sizeof(Header) == 120);
struct KvRows {
  std::vector<std::vector<std::uint8_t>> ordinary;
  std::array<std::vector<std::uint8_t>, 2> draft;
  bool operator==(const KvRows&) const = default;
};
std::vector<std::uint8_t> Save(q::Executor& exec, q::Session& session) {
  std::string error;
  std::vector<std::uint8_t> bytes(exec.SnapshotBytes(session, kHidden));
  Require(exec.SaveSnapshot(session, kHidden, bytes, &error), error);
  return bytes;
}
void Restore(q::Executor& exec, q::Session& session,
             const std::vector<std::uint8_t>& bytes) {
  std::string error;
  q::Executor::SnapshotInfo info;
  Require(exec.RestoreSnapshot(session, bytes, &info, &error), error);
  Require(info.position == kHidden && info.hidden_rows == kHidden,
          "producer fixture lost trunk/hidden frontier on restore");
}
KvRows ReadKv(const std::vector<std::uint8_t>& bytes, const qfn::Config& c) {
  Require(bytes.size() >= sizeof(Header), "short native producer capture");
  Header h{};
  std::memcpy(&h, bytes.data(), sizeof(h));
  Require(
      h.magic == std::array<char, 8>{'Q', 'F', 'N', 'S', 'N', 'A', 'P', '8'} &&
          h.layers == c.num_layers && h.interval == c.full_attention_interval &&
          h.conv == (c.ssm_conv_kernel - 1) * c.SsmConvChannels() &&
          h.state == c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim &&
          h.kv_row == c.AttentionKvDim() && h.type == unsigned(kType) &&
          h.epoch == qfn::kKvRotatedNumericsEpoch &&
          h.index_dim == c.indexer_head_dim && h.ratio == c.compress_ratio &&
          h.hc_dim == c.HcDim() && h.has_mtp == 1 &&
          h.ple == (c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0) &&
          h.position == kHidden && h.hidden == kHidden && h.images == 0 &&
          h.bytes == bytes.size() && h.blocks * h.ratio <= h.position &&
          h.mtp_blocks * h.ratio <= h.mtp_position &&
          h.mtp_position <= h.position,
      "native producer capture/wire probe drift");
  std::size_t offset = sizeof(Header);
  const auto take = [&](std::size_t size) {
    Require(offset <= bytes.size() && size <= bytes.size() - offset,
            "native producer region outside capture");
    const auto span = std::span(bytes).subspan(offset, size);
    offset += size;
    return span;
  };
  const auto packed = [&](unsigned rows) {
    const auto row_bytes = qfn::KvRowBytes(h.kv_row, kType);
    const auto span = take(rows * row_bytes);
    for (unsigned row = 0; row < rows; ++row)
      qfn::ValidateKvRow(
          std::as_bytes(span.subspan(row * row_bytes, row_bytes)), h.kv_row,
          kType);
    return std::vector<std::uint8_t>(span.begin(), span.end());
  };
  for (unsigned layer = 0; layer < h.layers; ++layer)
    if (c.IsLinearLayer(layer))
      (void)take(std::size_t(h.conv + h.state) * sizeof(float));
  (void)take(std::size_t(h.ple) * sizeof(float));
  KvRows result;
  for (unsigned layer = 0; layer < h.layers; ++layer) {
    if (c.IsLinearLayer(layer))
      continue;
    result.ordinary.push_back(packed(h.position));
    result.ordinary.push_back(packed(h.position));
    (void)take(std::size_t(h.position - h.blocks * h.ratio) * h.index_dim * 4);
    (void)take(std::size_t(h.blocks) * h.index_dim * 2);
  }
  (void)take(std::size_t(h.mtp_position - h.mtp_blocks * h.ratio) *
             h.index_dim * 4);
  (void)take(std::size_t(h.mtp_blocks) * h.index_dim * 2);
  for (auto& rows : result.draft)
    rows = packed(h.mtp_position);
  (void)take(std::size_t(h.residual ? h.hc_dim : 0) * sizeof(float));
  (void)take(std::size_t(h.hidden) * h.hc_dim * sizeof(float));
  Require(offset == bytes.size() && !result.ordinary.empty(),
          "native producer capture region inventory drift");
  return result;
}
}  // namespace

void CheckKvCacheOnlyProducers(const char* model_path, const char* mtp_path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Require(bool(reader), error);
  auto weights = qfn::ModelWeights::Bind(*reader, &error);
  Require(bool(weights), error);
  auto mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
  Require(bool(mtp_reader), error);
  auto mtp = qfn::MtpWeights::Bind(*mtp_reader, weights->config, &error);
  Require(bool(mtp), error);
  // P0 only: do not fold the separately reserved batched YaRN issue into this
  // gate. A supplied scaling model must fail rather than silently normalize.
  Require(weights->config.rope_scaling.factor == 1.0F,
          "P0 producer regression requires unscaled RoPE");
  auto device = q::DeviceModel::Upload(*weights, *reader, &*mtp,
                                       mtp_reader.get(), &error);
  Require(bool(device), error);
  gufo::test::UseSeparateMtpViews(*device);
  std::unique_ptr<qfn::NgramTable> ngram;
  if (weights->config.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = qfn::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, weights->config.ple_head_dim, t.type, &error);
    Require(bool(ngram), error);
  }
  auto exec = q::Executor::Create(*device, ngram.get(),
                                  {.max_batch = 64,
                                   .max_logit_rows = kHidden,
                                   .max_speculative = kHidden,
                                   .max_context = kContext,
                                   .kv_cache_type = kType},
                                  &error);
  Require(bool(exec), error);
  const auto make = [&] {
    auto session = exec->CreateSession(gufo::core::SessionMode::kSpeculative,
                                       kContext, &error);
    Require(bool(session), error);
    return session;
  };
  const std::array<std::int32_t, kHidden> tokens{42, 57, 81, 63,
                                                 44, 90, 72, 51};
  auto origin = make();
  // Decode mode deliberately avoids prefill's automatic cache-only catch-up.
  Require(exec->Forward(*origin, tokens, 0, nullptr,
                        q::Executor::ForwardMode::kDecode, &error),
          error);
  Require(exec->MtpPosition(*origin) == 0, "base unexpectedly ran MTP");
  const auto base = Save(*exec, *origin);
  const auto ordinary = ReadKv(base, exec->config()).ordinary;
  origin.reset();
  const auto equal = [&](q::Session& full, q::Session& cache) {
    const auto f = ReadKv(Save(*exec, full), exec->config());
    const auto k = ReadKv(Save(*exec, cache), exec->config());
    Require(f == k && f.ordinary == ordinary && !f.draft[0].empty() &&
                !f.draft[1].empty(),
            "full/cache-only ordinary or draft K/V packed bytes differ");
    for (const auto& rows : f.draft)
      Require(std::any_of(rows.begin(), rows.end(),
                          [](auto byte) { return byte != 0; }),
              "vacuous all-zero draft producer fixture");
  };
  for (unsigned chunk : {1U, 3U, 8U}) {
    auto full = make(), cache = make();
    Restore(*exec, *full, base);
    Restore(*exec, *cache, base);
    for (unsigned first = 0; first < tokens.size(); first += chunk) {
      const auto part = std::span(tokens).subspan(
          first, std::min<unsigned>(chunk, tokens.size() - first));
      Require(exec->MtpForward(*full, part, first, {}, &error), error);
      Require(exec->MtpForward(*cache, part, first, {.kv_only = true}, &error),
              error);
      equal(*full, *cache);
      // Materialize/restore after real producer writes, then continue through
      // the same branch. Residual validity differs intentionally; KV must not.
      const auto f = Save(*exec, *full), k = Save(*exec, *cache);
      full = make();
      cache = make();
      Restore(*exec, *full, f);
      Restore(*exec, *cache, k);
      Require(Save(*exec, *full) == f && Save(*exec, *cache) == k,
              "fresh-session producer restore changed payload bytes");
      equal(*full, *cache);
    }
  }
  std::array<std::unique_ptr<q::Session>, 2> full{make(), make()};
  std::array<std::unique_ptr<q::Session>, 2> cache{make(), make()};
  for (unsigned i = 0; i < 2; ++i) {
    Restore(*exec, *full[i], base);
    Restore(*exec, *cache[i], base);
  }
  // Ragged peers expose RowScratch offsets and per-session device positions.
  // Full and cache-only use identical projection cohort shapes, not scalar
  // arithmetic as an oracle for a differently shaped batch.
  for (unsigned round = 0; round < 2; ++round) {
    const auto a = std::span(tokens).subspan(round * 3, 3);
    const auto b = std::span(tokens).subspan(round * 4, 4);
    const std::array<q::Executor::MtpBatchItem, 2> f{
        {{full[0].get(), a, int(round * 3), false},
         {full[1].get(), b, int(round * 4), false}}};
    const std::array<q::Executor::MtpBatchItem, 2> k{
        {{cache[0].get(), a, int(round * 3), true},
         {cache[1].get(), b, int(round * 4), true}}};
    Require(exec->MtpForwardBatch(f, &error), error);
    Require(exec->MtpForwardBatch(k, &error), error);
    for (unsigned i = 0; i < 2; ++i) {
      equal(*full[i], *cache[i]);
      const auto fb = Save(*exec, *full[i]), kb = Save(*exec, *cache[i]);
      full[i] = make();
      cache[i] = make();
      Restore(*exec, *full[i], fb);
      Restore(*exec, *cache[i], kb);
      Require(Save(*exec, *full[i]) == fb && Save(*exec, *cache[i]) == kb,
              "fresh-session batched producer restore changed payload bytes");
      equal(*full[i], *cache[i]);
    }
  }
  std::cout
      << "ROTATED_KV_PRODUCER: separate-weight scalar/chunked/ragged-batch/"
         "restored ordinary+draft K/V full-vs-cache-only packed bytes exact; "
         "not model quality or cross-shape bit identity\n";
}
