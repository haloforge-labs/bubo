// CPU-only proof of the ACTUAL private finish wrapper, not a native/model or
// transfer/disk publication proof. Fake backing uses real ledger/Payloads.
#include "src/cli/serve/qfn_kv_ingress_validation.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;
namespace server = gufo::server;
namespace {
constexpr std::array kPackedTypes{
    qfn::KvCacheType::kQ8_0,
    qfn::KvCacheType::kQ8_0Rotated,
};
constexpr std::uint32_t kContext = 256, kElements = 512;
constexpr std::size_t kRow =
    qfn::KvRowBytes(kElements, qfn::KvCacheType::kQ8_0);
void Require(bool ok, const char* what) {
  if (!ok)
    throw std::runtime_error(what);
}
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid ingress accepted or wrong exception");
}
template<class F>
void Consumed(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::logic_error& e) {
    rejected =
        std::string(e.what()).find("already consumed") != std::string::npos;
  }
  Require(rejected, "finish retried or converted twice");
}
qfn::Config Geometry() {
  // Production attention geometry and all 48 layers/26 ordinary+draft K/V
  // components. Small unrelated private dimensions keep this CPU fixture tiny.
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.context_length = kContext;
  c.hidden_size = 8;
  c.hc_count = 1;
  c.vocab_size = 32;
  c.num_kv_heads = 2;
  c.num_heads = 24;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.indexer_head_dim = 8;
  c.indexer_top_k = 16;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 2;
  c.ssm_head_dim = 4;
  c.ssm_num_k_heads = c.ssm_num_v_heads = 1;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}
bool Packed(qfn::ContinuationPart part) {
  return part == qfn::ContinuationPart::kKey ||
         part == qfn::ContinuationPart::kValue ||
         part == qfn::ContinuationPart::kDraftKey ||
         part == qfn::ContinuationPart::kDraftValue;
}
std::vector<std::byte> Canonical(qfn::KvCacheType type, std::size_t rows = 2) {
  std::vector<std::byte> bytes(rows * kRow);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<float> values(kElements);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = (static_cast<int>(i % 17) - 8) / 16.0F;
    values[0] = 127;
    values[1] = 0.5F;
    values[2] = -0.5F;
    values[3] = 1.5F;
    values[4] = -1.5F;
    std::fill(values.begin() + 32, values.begin() + 64, 0);
    values[32] = 1e-7F;  // Stored-scale underflow with preserved nonzero codes.
    qfn::EncodeKvRow(values, type, std::span(bytes).subspan(row * kRow, kRow));
  }
  return bytes;
}
void Fault(std::span<std::byte> bytes, unsigned fault) {
  auto block = bytes.last(sizeof(qfn::KvBlockQ8_0));
  const auto code = offsetof(qfn::KvBlockQ8_0, values);
  if (fault == 0) {
    block[code] = std::byte{0x80};
    return;
  }
  const auto scale = std::array<std::uint16_t, 5>{0xbc00, 0x8000, 0x7c00,
                                                  0x7e00, 0x7bff}[fault - 1];
  block[0] = std::byte(scale & 255);
  block[1] = std::byte(scale >> 8);
  std::fill(block.begin() + code, block.end(), std::byte{0});
  if (fault == 5)
    block[code] = std::byte{127};
}
struct Stats {
  std::size_t allocations{}, finishes{};
  bool short_buffer{}, no_finish{}, finish_throws{};
  std::weak_ptr<std::vector<std::byte>> last;
};
struct BorrowSentinel {
  std::shared_ptr<const void> owner;
  std::shared_ptr<cache::ResourceReservation> reservation;
  cache::Payload operator()(cache::SlotLease&,
                            std::span<const std::byte>) const {
    throw std::logic_error(
        "trusted borrow callback must not be invoked by this test");
  }
};
struct PhysicalBacking {
  cache::ResourceCharge pool;
  std::shared_ptr<std::vector<std::byte>>
      data;  // Destroy storage before charge.
};
struct Fixture {
  cache::ResourceLedger ledger{{1ULL << 20, 1ULL << 20, 0, 0}};
  std::shared_ptr<Stats> stats = std::make_shared<Stats>();
  server::ComponentCacheResources resources;
  Fixture() {
    resources.owner = std::make_shared<int>(7);
    resources.allocate = [this,
                          stats = stats](const cache::PayloadRequest& request) {
      ++stats->allocations;
      Require(request.bytes <= 65536, "unbounded CPU fixture allocation");
      auto data = std::make_shared<std::vector<std::byte>>(request.bytes + 64,
                                                           std::byte{0xA5});
      stats->last = data;
      auto physical = std::make_shared<PhysicalBacking>(PhysicalBacking{
          ledger.Reserve(cache::ResourceCategory::kBackingFree, data->size())
              .Convert(),
          data});
      auto reservation = std::make_shared<cache::ResourceReservation>(
          physical->pool.ReserveBacking(request.category));
      auto owner = std::shared_ptr<const void>(physical, data->data());
      server::ComponentCaptureBuffer buffer;
      buffer.bytes = *data;
      if (stats->short_buffer)
        buffer.bytes = buffer.bytes.first(request.bytes - 1);
      if (!stats->no_finish)
        buffer.finish = [stats, reservation, owner] {
          ++stats->finishes;
          if (stats->finish_throws)
            throw std::runtime_error("original finish failure");
          return cache::Payload::Committed(reservation->Convert(), owner);
        };
      buffer.borrow = BorrowSentinel{owner, reservation};
      return buffer;
    };
  }
  void Install(const qfn::Config& c = Geometry(), bool mtp = true,
               qfn::KvCacheType type = qfn::KvCacheType::kQ8_0,
               std::uint32_t context = kContext) {
    server::detail::InstallQfnPackedKvIngressValidation(resources, c, mtp,
                                                        context, type);
  }
  void Released() {
    Require(stats->last.expired(), "backing owner leaked");
    Require(ledger.Snapshot().total_bytes == 0, "reservation/charge leaked");
  }
};
cache::PayloadRequest Request(cache::ComponentId id) {
  return {id, cache::ResourceCategory::kPrivateTail, 5, 7, 2 * kRow};
}
void Fill(server::ComponentCaptureBuffer& buffer,
          const std::vector<std::byte>& bytes) {
  Require(buffer.bytes.size() >= bytes.size(), "test fill capacity");
  std::copy(bytes.begin(), bytes.end(), buffer.bytes.begin());
}
void PlainRejectsBeforeOwnership() {
  const auto type = qfn::KvCacheType::kQ8_0;
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  Fixture f;
  f.Install(Geometry(), true, type);
  const auto kv = std::ranges::find_if(
      layout.Components(), [](const auto& c) { return Packed(c.part); });
  {
    auto request = Request(kv->descriptor.id);
    auto buffer = f.resources.allocate(request);
    Fill(buffer, Canonical(type));
    Fault(buffer.bytes.first(request.bytes), 0);
    Reject([&] { (void)buffer.finish(); });
    Require(f.stats->finishes == 0,
            "plain malformed payload converted ownership before validation");
  }
  f.Released();
}
void CodecAndClassification(qfn::KvCacheType type) {
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  std::size_t packed = 0, untouched = 0;
  const auto canonical = Canonical(type);
  for (const auto& component : layout.Components()) {
    Fixture f;
    f.Install(Geometry(), true, type);
    if (!Packed(component.part)) {
      auto request =
          component.descriptor.kind == cache::ComponentKind::kPrivateState
              ? cache::PayloadRequest{component.descriptor.id,
                                      cache::ResourceCategory::kPrivateState, 0,
                                      0, component.descriptor.state_bytes}
              : cache::PayloadRequest{component.descriptor.id,
                                      cache::ResourceCategory::kPrivateTail, 0,
                                      2, component.descriptor.row_bytes * 2};
      {
        auto buffer = f.resources.allocate(request);
        // Deliberately poison-looking bytes: these are NOT ordinary/draft KV.
        if (request.bytes >= 34)
          Fault(buffer.bytes.first(request.bytes), 0);
        auto payload = buffer.finish();
        Require(payload.IsValid() && f.stats->finishes == 1,
                "private/indexer finish changed");
        Require(buffer.borrow.target<BorrowSentinel>() != nullptr,
                "borrow callback replaced");
      }
      f.Released();
      ++untouched;
      continue;
    }
    ++packed;
    const auto request = Request(component.descriptor.id);
    {
      auto buffer = f.resources.allocate(request);
      Fill(buffer, canonical);
      auto copied_finish = buffer.finish;
      auto payload = buffer.finish();
      Require(payload.IsValid() && f.stats->finishes == 1,
              "valid finish did not convert once");
      Require(
          std::equal(canonical.begin(), canonical.end(), buffer.bytes.begin()),
          "finish changed canonical codec bytes");
      Require(
          std::all_of(buffer.bytes.begin() + request.bytes, buffer.bytes.end(),
                      [](auto b) { return b == std::byte{0xA5}; }),
          "finish touched padding");
      Consumed([&] { (void)copied_finish(); });
      Require(f.stats->finishes == 1, "copied finish converted twice");
      Require(buffer.borrow.target<BorrowSentinel>() != nullptr,
              "trusted borrow callback changed");
    }
    f.Released();
    for (unsigned fault = 0; fault < 6; ++fault) {
      const auto before = f.stats->finishes;
      {
        auto buffer = f.resources.allocate(request);
        Fill(buffer, canonical);
        Fault(buffer.bytes.first(request.bytes),
              fault);  // Last block of row TWO.
        auto copied_finish = buffer.finish;
        Reject([&] { (void)buffer.finish(); });
        Require(f.stats->finishes == before,
                "original finish called before all rows validated");
        Fill(buffer, canonical);
        Consumed([&] { (void)copied_finish(); });
        Require(f.stats->finishes == before,
                "failed finish retried conversion");
      }
      f.Released();
      {
        auto retry = f.resources.allocate(request);
        Fill(retry, canonical);
        auto payload = retry.finish();
        Require(payload.IsValid() && f.stats->finishes == before + 1,
                "fresh valid retry failed");
      }
      f.Released();
    }
  }
  Require(packed == 26 && untouched > 0,
          "actual ordinary/draft layout classification incomplete");
}
void RequestsAndLifetime(qfn::KvCacheType type) {
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  const auto kv = std::ranges::find_if(layout.Components(), [](const auto& c) {
                    return Packed(c.part);
                  })->descriptor.id;
  const auto valid = Request(kv);
  Fixture f;
  f.Install(Geometry(), true, type);
  std::vector<cache::PayloadRequest> bad;
  for (auto id : {0U, 0xffffffffU}) {
    auto r = valid;
    r.component = {id};
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.bytes = 0;
    bad.push_back(r);
  }
  {
    auto r = valid;
    --r.bytes;
    bad.push_back(r);
  }
  {
    auto r = valid;
    ++r.bytes;
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.first = r.end;
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.first = r.end + 1;
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.end = kContext + 1;
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.end = std::numeric_limits<cache::Rows>::max();
    bad.push_back(r);
  }
  {
    auto r = valid;
    r.category = cache::ResourceCategory::kPrivateState;
    bad.push_back(r);
  }
  for (const auto& r : bad)
    Reject([&] { (void)f.resources.allocate(r); });
  Require(f.stats->allocations == 0 && f.stats->finishes == 0,
          "bad request reached original allocator");
  for (bool missing_finish : {false, true}) {
    f.stats->short_buffer = !missing_finish;
    f.stats->no_finish = missing_finish;
    Reject([&] { (void)f.resources.allocate(valid); });
    Require(f.stats->finishes == 0,
            "incomplete buffer reached original finish");
    f.Released();
  }
  f.stats->short_buffer = f.stats->no_finish = false;
  {
    auto chunk = valid;
    chunk.category = cache::ResourceCategory::kBackingAssigned;
    auto buffer = f.resources.allocate(chunk);
    Fill(buffer, Canonical(type));
    auto payload = buffer.finish();
    Require(
        payload.IsValid() &&
            payload.Category() == cache::ResourceCategory::kBackingMaterialized,
        "valid full-chunk category was not preserved");
  }
  f.Released();
  const auto prior_finishes = f.stats->finishes;
  {
    auto buffer = f.resources.allocate(valid);
    Fill(buffer, Canonical(type));
    f.stats->finish_throws = true;
    bool failed = false;
    try {
      (void)buffer.finish();
    } catch (const std::runtime_error&) {
      failed = true;
    }
    Require(failed && f.stats->finishes == prior_finishes + 1,
            "original finish exception lost");
    Consumed([&] { (void)buffer.finish(); });
  }
  f.Released();
  f.stats->finish_throws = false;
  std::weak_ptr<void> resource_owner = f.resources.owner;
  {
    auto buffer = f.resources.allocate(valid);
    Fill(buffer, Canonical(type));
    f.resources.owner.reset();
    f.resources.allocate = {};
    Require(!resource_owner.expired(), "finish lost resource-owner lifetime");
    auto payload = buffer.finish();
    Require(payload.IsValid(), "finish lost backing after resources dropped");
  }
  Require(resource_owner.expired(),
          "resource owner retained after finish/buffer release");
  f.Released();
}
void InstallationAndLegacy(qfn::KvCacheType type) {
  Fixture legacy;
  legacy.Install(qfn::Config{}, false, qfn::KvCacheType::kF16, 0);
  {
    auto buffer = legacy.resources.allocate(Request({999999}));
    auto payload = buffer.finish();  // F16 skips packed geometry/byte checks.
    Require(payload.IsValid() && legacy.stats->finishes == 1,
            "F16 allocator/finish altered");
  }
  legacy.Released();
  if (type == qfn::KvCacheType::kQ8_0Rotated) {
    for (unsigned fault = 0; fault < 4; ++fault) {
      Fixture f;
      auto c = Geometry();
      if (fault == 0)
        c.head_dim = 128;
      if (fault == 1)
        c.rotary_dim = 32;
      if (fault == 2)
        c.num_kv_heads = 1;
      if (fault == 3)
        c.num_heads = 23;
      Reject([&] { f.Install(c, true, type); });
      Require(f.stats->allocations == 0, "bad installation allocated");
      {
        auto buffer = f.resources.allocate(Request({999999}));
        auto payload = buffer.finish();  // Failed install is transactional.
      }
      f.Released();
    }
  }
  Fixture f;
  Reject(
      [&] { f.Install(Geometry(), true, static_cast<qfn::KvCacheType>(99)); });
  Reject([&] { f.Install(Geometry(), true, type, 0); });
  auto no_layers = Geometry();
  no_layers.num_layers = 0;
  Reject([&] { f.Install(no_layers, true, type); });
  auto no_kv = Geometry();
  no_kv.num_layers = 3;
  Reject([&] { f.Install(no_kv, false, type); });
  f.resources.allocate = {};
  Reject([&] { f.Install(Geometry(), true, type); });
  Fixture ar;
  ar.Install(Geometry(), false, type);
  const qfn::ContinuationLayout mtp(Geometry(), true, kContext, 1, type);
  const auto draft =
      std::ranges::find(mtp.Components(), qfn::ContinuationPart::kDraftKey,
                        &qfn::ContinuationComponent::part);
  Reject([&] { (void)ar.resources.allocate(Request(draft->descriptor.id)); });
  Require(ar.stats->allocations == 0,
          "AR classifier accepted nonexistent draft component");
}
void FailedCaptureRollback(qfn::KvCacheType type) {
  Fixture f;
  f.Install(Geometry(), true, type);
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  std::vector<cache::ComponentDescriptor> descriptors;
  cache::ComponentId bad_id{};
  for (const auto& component : layout.Components()) {
    descriptors.push_back(component.descriptor);
    if (component.part == qfn::ContinuationPart::kValue && !bad_id.value)
      bad_id = component.descriptor.id;
  }
  auto history = cache::ExecutionHistory::Cold(f.ledger, descriptors, {7, 8});
  const auto baseline = f.ledger.Snapshot();
  std::vector<server::ComponentCaptureBuffer> buffers;
  std::shared_ptr<const cache::Checkpoint> retained;
  const std::vector<cache::Token> tokens{10, 11};
  const auto positions = layout.Positions(2, 2, 0, 0);
  std::size_t next = 0;
  Reject([&] {
    retained = history.Capture(
        {tokens, cache::InputIdentity(tokens.size(), {}), positions,
         cache::CheckpointPurpose::kPrompt, 0},
        [&](const auto&) { return buffers.at(next++).finish(); },
        [&](std::span<const cache::PayloadRequest> requests) {
          for (const auto& request : requests) {
            buffers.push_back(f.resources.allocate(request));
            if (std::ranges::any_of(layout.Components(), [&](const auto& c) {
                  return c.descriptor.id == request.component && Packed(c.part);
                })) {
              Fill(buffers.back(),
                   Canonical(type, request.end - request.first));
              if (request.component == bad_id)
                Fault(buffers.back().bytes.first(request.bytes), 0);
            }
          }
        });
  });
  Require(!retained && f.stats->finishes > 0,
          "partial capture retained or did not exercise prior finishes");
  buffers.clear();
  Require(f.ledger.Snapshot().bytes == baseline.bytes &&
              f.ledger.Snapshot().reserved_bytes == baseline.reserved_bytes,
          "failed complete capture leaked committed payloads/reservations");
}
}  // namespace
int main() {
  try {
    PlainRejectsBeforeOwnership();
    for (auto type : kPackedTypes) {
      CodecAndClassification(type);
      RequestsAndLifetime(type);
      InstallationAndLegacy(type);
      FailedCaptureRollback(type);
    }
    std::cout
        << "packed q8 ingress validation: " << kPackedTypes.size()
        << " modes x 26 KV components x six faults; unchanged non-KV/F16, "
           "rollback/lifetimes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qfn ingress CPU FAIL: " << e.what() << '\n';
    return 1;
  }
}
