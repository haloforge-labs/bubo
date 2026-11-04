#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/cli/component_sequence_trace.hpp"

namespace {
using namespace gufo;
using namespace server;
using namespace cache;
using cache::testing::FakeStream;
namespace trace = gufo::tests::component_sequence_trace;

void Require(bool condition, const std::string& reason) {
  if (!condition)
    throw std::runtime_error(reason);
}

struct RegistrationLayout {
  // The logs do not contain native virtual addresses. These variants model
  // possible layouts at the same byte budget, not the OS allocator's response
  // to free RAM. Growth completes before each simulated prefill chunk; copies
  // still complete asynchronously through FakeStream::Wait.
  bool check{}, gaps{}, grow{}, trace{};
  // Boundary before this segment is adjacent even in an otherwise gapped
  // layout. These are simulated addresses, not addresses from a GPU log.
  std::size_t adjacent_boundary = std::numeric_limits<std::size_t>::max();
};

// Real byte geometry and ledger charges, sparse contents: only one marker per
// 64 KiB backing block is touched. No HIP runtime or model is linked or used.
// The uninitialized array supplies valid address arithmetic without populating
// the multi-gigabyte payload. Completion callbacks perform all marker accesses.
struct Backing : std::enable_shared_from_this<Backing> {
  struct Mark {
    std::uint64_t generation{};
    bool assigned{}, written{};
    ComponentId component;
    Rows first{}, end{}, private_position{};
    std::size_t allocation_first{};
  };
  ResourceLedger& ledger;
  ResourceCharge metadata;
  std::vector<ResourceCharge> blocks;
  std::vector<Mark> marks;
  std::unique_ptr<std::byte[]> storage;
  std::size_t capacity{}, segment{}, next{}, next_private{};
  std::vector<std::size_t> segment_offsets;
  std::size_t storage_bytes{};
  std::size_t submissions{}, restores{}, completed{}, released{}, reused{},
      full{};
  RegistrationLayout layout;
  std::string failure;

  Backing(ResourceLedger& ledger, std::size_t budget, RegistrationLayout layout)
      : ledger(ledger), layout(layout) {
    const auto backing_budget = (budget - (4ULL << 20)) / 10 * 9;
    const auto count = (backing_budget - 4096) / (65536 + 1024);
    capacity = count * 65536;
    const auto parts = (count + 2047) / 2048;
    segment = (count + parts - 1) / parts;
    metadata = ledger.Reserve(ResourceCategory::kMetadata, 4096 + count * 1024)
                   .Convert();
    for (std::size_t first = 0; first < count; first += segment) {
      segment_offsets.push_back(storage_bytes);
      storage_bytes += std::min(segment, count - first) * 65536;
      if (layout.gaps && first + segment < count &&
          first / segment + 1 != layout.adjacent_boundary)
        storage_bytes += 65536;
    }
    storage.reset(new std::byte[storage_bytes]);
    marks.resize(count);
    blocks.reserve(count);
    if (layout.grow)
      GrowAhead(2ULL << 30);
    else
      while (Grow()) {
      }
    next_private = blocks.size();
    if (layout.grow)
      std::cout << "registration-layout segment_blocks=" << segment
                << " initial_committed_bytes=" << blocks.size() * 65536
                << " gaps=" << layout.gaps
                << " adjacent_boundary=" << layout.adjacent_boundary << '\n';
  }
  bool Grow() {
    if (blocks.size() == marks.size())
      return false;
    const auto end = std::min(marks.size(), blocks.size() + segment);
    while (blocks.size() < end)
      blocks.push_back(
          ledger.Reserve(ResourceCategory::kBackingFree, 65536).Convert());
    return true;
  }
  void GrowAhead(std::size_t reserve) {
    while (ledger.Snapshot().bytes[static_cast<std::size_t>(
               ResourceCategory::kBackingFree)] < std::min(reserve, capacity))
      if (!Grow())
        break;
  }
  std::byte* Address(std::size_t index) const {
    return storage.get() + segment_offsets[index / segment] +
           index % segment * 65536;
  }
  std::size_t Index(std::size_t offset) const {
    const auto s = static_cast<std::size_t>(
        std::upper_bound(segment_offsets.begin(), segment_offsets.end(),
                         offset) -
        segment_offsets.begin() - 1);
    const auto within = offset - segment_offsets[s];
    Require(within < std::min(segment, marks.size() - s * segment) * 65536,
            "transfer names a virtual-address gap");
    return s * segment + within / 65536;
  }
  std::size_t RegisteredBytes(const void* pointer) const {
    const auto [first, end] = Range(pointer, 1);
    (void)end;
    const auto s = first / segment;
    const auto offset =
        static_cast<const std::byte*>(pointer) - Address(s * segment);
    return std::min(segment, blocks.size() - s * segment) * 65536 - offset;
  }
  void Refresh(std::size_t first, std::size_t end) {
    for (auto i = first; i < end; ++i) {
      const auto info = blocks[i].Info();
      if (marks[i].assigned &&
          info.category == ResourceCategory::kBackingFree && !info.reserved) {
        marks[i].assigned = marks[i].written = false;
        *Address(i) = std::byte{0xdd};
        ++released;
      }
    }
  }
  std::pair<std::size_t, std::size_t> Range(const void* pointer,
                                            std::size_t bytes) const {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base = reinterpret_cast<std::uintptr_t>(storage.get());
    Require(bytes && address >= base && address - base < storage_bytes &&
                bytes <= storage_bytes - (address - base),
            "transfer names unallocated backing");
    const auto offset = address - base;
    const auto first = Index(offset), end = Index(offset + bytes - 1) + 1;
    Require(end <= blocks.size(), "transfer names an uncommitted segment");
    Require(Address(end - 1) - Address(first) == (end - first - 1) * 65536,
            "transfer spans an unallocated virtual-address gap");
    return {first, end};
  }
  std::vector<std::uint64_t> Pin(const void* pointer, std::size_t bytes) {
    const auto [first, end] = Range(pointer, bytes);
    Refresh(first, end);
    std::vector<std::uint64_t> result;
    for (auto i = first; i < end; ++i) {
      Require(marks[i].assigned, "submission names released backing");
      result.push_back(marks[i].generation);
    }
    return result;
  }
  Completion Transfer(bool load, ComponentId id, Rows first, Rows end,
                      const void* pointer, std::size_t bytes, Stream& stream,
                      std::function<void(bool)> settle, Rows frontier) {
    const auto [a, b] = Range(pointer, bytes);
    const auto registered_bytes = RegisteredBytes(pointer);
    if (layout.trace && load && a / segment != (b - 1) / segment)
      std::cout << "registration-crossing P=" << frontier
                << " component=" << id.value << " rows=" << first << ".." << end
                << " blocks=" << a << ".." << b << " segments=" << a / segment
                << ".." << (b - 1) / segment << '\n';
    auto generations = Pin(pointer, bytes);
    ++submissions;
    restores += load;
    return dynamic_cast<FakeStream&>(stream).Submit(
        [this, load, id, first, end, a, b, frontier, bytes, registered_bytes,
         generations = std::move(generations), settle = std::move(settle)] {
          Refresh(a, b);
          bool good = true;
          for (auto i = a; i < b; ++i) {
            good &=
                marks[i].assigned && marks[i].generation == generations[i - a];
            if (load)
              good &= marks[i].written && *Address(i) != std::byte{0xdd};
            good &= marks[i].component == id;
            const auto row = trace::kComponents[id.value - 1].row_bytes;
            if (load && row)
              good &= first + (i - a) * 65536 / row ==
                      marks[i].first +
                          (i - marks[i].allocation_first) * 65536 / row;
            if (load && !row)
              good &= marks[i].private_position == frontier;
          }
          // A native alias belongs to one registration. Host-address adjacency
          // alone does not establish adjacency of independent device aliases.
          const bool alias_ok = !layout.check || bytes <= registered_bytes;
          good &= alias_ok;
          if (!good && failure.empty())
            failure = std::string(load ? "restore" : "capture/preserve") +
                      " component=" + std::to_string(id.value) +
                      " rows=" + std::to_string(first) + ".." +
                      std::to_string(end) + " blocks=" + std::to_string(a) +
                      ".." + std::to_string(b) +
                      " P=" + std::to_string(frontier) +
                      " bytes=" + std::to_string(bytes) +
                      " registered_bytes=" + std::to_string(registered_bytes) +
                      (alias_ok ? " released/reassigned/unwritten"
                                : " crosses simulated registration alias");
          if (good && !load)
            for (auto i = a; i < b; ++i) {
              marks[i].written = true;
              marks[i].private_position = frontier;
              *Address(i) = std::byte{0x3c};
            }
          ++completed;
          settle(good);
          return good ? TransferResult::kSucceeded : TransferResult::kFailed;
        });
  }
  ComponentCaptureBuffer Allocate(const PayloadRequest& request) {
    const auto count =
        std::max<std::size_t>(1, (request.bytes + 65535) / 65536);
    const bool backwards = request.category == ResourceCategory::kPrivateState;
    auto& cursor = backwards ? next_private : next;
    std::optional<std::size_t> selected;
    for (std::size_t n = 0; n < blocks.size(); ++n) {
      const auto i = backwards
                         ? (cursor + blocks.size() - 1 - n) % blocks.size()
                         : (cursor + n) % blocks.size();
      if (i + count > blocks.size() || i / segment != (i + count - 1) / segment)
        continue;
      if (std::none_of(marks.begin() + i, marks.begin() + i + count,
                       [](const auto& mark) { return mark.assigned; })) {
        selected = i;
        break;
      }
    }
    if (!selected) {
      if (layout.grow && Grow())
        return Allocate(request);
      ++full;
      throw ResourceExhausted();
    }
    auto reservation =
        std::make_shared<ResourceReservation>(ResourceCharge::ReserveBacking(
            std::span<const ResourceCharge>(blocks).subspan(*selected, count),
            request.category));
    for (auto i = *selected; i < *selected + count; ++i) {
      Require(!marks[i].assigned, "double backing assignment");
      reused += marks[i].generation != 0;
      ++marks[i].generation;
      marks[i].assigned = true;
      marks[i].written = false;
      marks[i].component = request.component;
      marks[i].first = request.first;
      marks[i].end = request.end;
      marks[i].allocation_first = *selected;
      *Address(i) = std::byte{0xa5};
    }
    cursor = backwards ? *selected : *selected + count;
    auto bytes = std::span(Address(*selected), count * 65536);
    auto owner = std::shared_ptr<const void>(shared_from_this(), bytes.data());
    return {bytes,
            [reservation, owner] {
              return Payload::Committed(reservation->Convert(), owner);
            },
            this,
            [reservation, owner, bytes, request](SlotLease& slot, auto source) {
              return Payload::Borrowed(
                  slot.Borrow(request.component, request.first, request.end,
                              std::move(*reservation), owner,
                              bytes.first(request.bytes), source));
            }};
  }
};

class AsyncAdapter final : public Adapter {
public:
  struct State final : Slot {
    MutationGuard& guard;
    Rows target{};
    std::size_t pending{};
    bool restoring{}, failed{};
    explicit State(MutationGuard& guard) : guard(guard) {}
    bool IsValid() const noexcept override { return !restoring && !failed; }
  };
  std::shared_ptr<Backing> pool;
  std::vector<ComponentDescriptor> descriptors;
  std::unique_ptr<std::byte[]> device_rows{new std::byte[200000ULL * 1024]};
  explicit AsyncAdapter(std::shared_ptr<Backing> pool) : pool(std::move(pool)) {
    for (const auto& c : trace::kComponents)
      descriptors.push_back(
          {{static_cast<std::uint32_t>(descriptors.size() + 1)},
           1,
           c.row_bytes ? ComponentKind::kAppendRows
                       : ComponentKind::kPrivateState,
           c.row_bytes,
           c.rows_per_chunk,
           c.state_bytes});
  }
  static State& As(Slot& slot) { return dynamic_cast<State&>(slot); }
  Capabilities GetCapabilities() const override { return {true, true}; }
  std::span<const ComponentDescriptor> Components() const override {
    return descriptors;
  }
  Identity CompatibilityIdentity() const override { return {91, 11}; }
  std::unique_ptr<Slot> CreateSlot(MutationGuard& guard) override {
    return std::make_unique<State>(guard);
  }
  std::vector<ComponentPosition> At(Rows target) const {
    std::vector<ComponentPosition> result;
    for (std::size_t i = 0; i < trace::kComponents.size(); ++i) {
      const auto& c = trace::kComponents[i];
      auto rows = c.draft && target ? target - 1 : target;
      if (!c.row_bytes)
        rows = target;
      else if (c.pooled)
        rows = rows > 2048 ? rows / 4 : 0;
      result.push_back({descriptors[i].id, rows});
    }
    return result;
  }
  std::vector<ComponentPosition> Positions(const Slot& slot) const override {
    return At(dynamic_cast<const State&>(slot).target);
  }
  std::vector<Rows> PlanPrefill(const Slot&, Rows, Rows end) const override {
    return {end};
  }
  void Guard(State& state, bool release) const {
    for (const auto& d : descriptors)
      if (d.kind == ComponentKind::kAppendRows) {
        if (release)
          state.guard.BeforeRelease(d.id, 0, 200000);
        else
          state.guard.BeforeOverwrite(d.id, 0, 200000);
      }
  }
  void BeginRestore(Slot& slot,
                    std::span<const ComponentPosition> expected) override {
    auto& state = As(slot);
    Require(!state.pending && !state.failed,
            "restore starts with pending work");
    Guard(state, false);
    state.target = expected.front().valid_rows;
    state.restoring = true;
  }
  Completion Copy(Slot& slot, ComponentId id, Rows first, Rows end,
                  const void* bytes, std::size_t size, Stream& stream,
                  bool load) {
    auto& state = As(slot);
    ++state.pending;
    return pool->Transfer(
        load, id, first, end, bytes, size, stream,
        [&state, load](bool good) {
          --state.pending;
          if (load && !good)
            state.failed = true;
        },
        state.target);
  }
  Completion CapturePrivate(const Slot& slot, ComponentId id,
                            std::span<std::byte> bytes,
                            Stream& stream) override {
    return Copy(const_cast<Slot&>(slot), id, 0, 0, bytes.data(), bytes.size(),
                stream, false);
  }
  Completion CopyRowsOut(const Slot& slot, ComponentId id, Rows first, Rows end,
                         std::span<std::byte> bytes, Stream& stream) override {
    return Copy(const_cast<Slot&>(slot), id, first, end, bytes.data(),
                bytes.size(), stream, false);
  }
  std::span<const std::byte> BorrowRowsView(const Slot&, ComponentId id,
                                            Rows first,
                                            Rows end) const override {
    if (trace::kComponents[id.value - 1].part == trace::Part::kTokens)
      return {};
    const auto row = descriptors[id.value - 1].row_bytes;
    return {device_rows.get() + first * row, (end - first) * row};
  }
  Completion CopyRowsIn(Slot& slot, ComponentId id, Rows first, Rows end,
                        std::span<const std::byte> bytes,
                        Stream& stream) override {
    return Copy(slot, id, first, end, bytes.data(), bytes.size(), stream, true);
  }
  Completion LoadPrivate(Slot& slot, ComponentId id,
                         std::span<const std::byte> bytes,
                         Stream& stream) override {
    return Copy(slot, id, 0, 0, bytes.data(), bytes.size(), stream, true);
  }
  bool Validate(Slot& slot,
                std::span<const ComponentPosition> expected) override {
    auto& state = As(slot);
    if (state.failed || state.pending ||
        At(state.target) != std::vector(expected.begin(), expected.end()))
      return false;
    state.restoring = false;
    return true;
  }
  bool Invalidate(Slot& slot) noexcept override {
    auto& state = As(slot);
    if (state.pending || state.failed)
      return false;
    Guard(state, true);
    state.target = 0;
    state.restoring = false;
    state.guard.AfterReset();
    return true;
  }
};

struct RunnerState final : TextRunnerState {
  AsyncAdapter& adapter;
  AsyncAdapter::State& slot;
  RunnerState(AsyncAdapter& adapter, Slot& slot)
      : adapter(adapter), slot(AsyncAdapter::As(slot)) {}
  void Invalidate() noexcept override { (void)adapter.Invalidate(slot); }
};
class Runner final : public TextModelRunner {
public:
  mutable AsyncAdapter* adapter{};
  mutable std::weak_ptr<Backing> pool;
  std::size_t budget;
  RegistrationLayout layout;
  mutable std::size_t capacity_queries{};
  mutable bool registration_limits{};
  Runner(std::size_t budget, RegistrationLayout layout)
      : budget(budget), layout(layout) {}
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "async-native-geometry",
            .state_abi = "async-v1",
            .max_context = 200000,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .prefix_reuse = true}};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.retained_snapshot_capacity_bytes = budget};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{TextExecutionPlanKind::kSerial, 1}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::vector<TextRunnerToken>{};
  }
  std::string Decode(std::span<const TextRunnerToken>) const override {
    return {};
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error("component state required");
  }
  bool UsesComponentCache() const override { return true; }
  template<class Resources>
  void ConfigureCapacity(Resources& resources,
                         const std::shared_ptr<Backing>& backing) const {
    registration_limits = true;
    resources.contiguous_bytes = [backing](const void* pointer) {
      return backing->RegisteredBytes(pointer);
    };
    if constexpr (requires { resources.checkpoint_capacity; }) {
      using Capacity =
          std::invoke_result_t<decltype(resources.checkpoint_capacity),
                               std::span<const ComponentPosition>>;
      resources.checkpoint_capacity = [this, backing](auto positions) {
        ++capacity_queries;
        Capacity result{};
        result.capacity_bytes = backing->capacity;
        const auto rounded = [](std::size_t bytes) {
          return bytes ? (1 + (bytes - 1) / 65536) * 65536 : 0;
        };
        for (std::size_t i = 0; i < trace::kComponents.size(); ++i) {
          const auto& c = trace::kComponents[i];
          auto private_tail = rounded(c.state_bytes);
          if (c.row_bytes) {
            const auto rows = positions[i].valid_rows;
            result.full_payload_bytes +=
                rows / 256 * rounded(256 * c.row_bytes);
            private_tail = rounded(rows % 256 * c.row_bytes);
          }
          result.private_tail_bytes += private_tail;
          result.full_payload_bytes += private_tail;
        }
        return result;
      };
    }
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      ResourceLedger& ledger, std::size_t) const override {
    auto backing = std::make_shared<Backing>(ledger, budget, layout);
    pool = backing;
    auto result = std::make_unique<ComponentCacheResources>();
    result->owner = backing;
    result->adapter = std::make_unique<AsyncAdapter>(backing);
    adapter = static_cast<AsyncAdapter*>(result->adapter.get());
    ConfigureCapacity(*result, backing);
    result->allocate = [pool = backing](const PayloadRequest& r) {
      return pool->Allocate(r);
    };
    result->can_capture = [pool = backing](
                              std::span<const PayloadRequest> requests) {
      pool->Refresh(0, pool->blocks.size());
      std::size_t needed = 0;
      for (const auto& request : requests)
        needed += std::max<std::size_t>(1, (request.bytes + 65535) / 65536);
      return needed <=
             static_cast<std::size_t>(std::ranges::count_if(
                 pool->marks, [](const auto& mark) { return !mark.assigned; }));
    };
    result->stream = [] { return std::make_unique<FakeStream>(true); };
    result->transfer_batch_size = 16;
    if (layout.grow)
      result->prepare_prefill = [backing](std::size_t tokens) {
        backing->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
      };
    return result;
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      Adapter& adapter, Slot& slot) const override {
    return std::make_unique<RunnerState>(static_cast<AsyncAdapter&>(adapter),
                                         slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    Require(dynamic_cast<RunnerState&>(state).slot.target == position,
            "restored frontier mismatch");
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t maximum) const override {
    const auto count = std::min(maximum, prompt.size() - offset);
    dynamic_cast<RunnerState&>(state).slot.target = offset + count;
    return {count, offset + count == prompt.size()};
  }
  TextDecodeSelection SelectNext(TextRunnerState&,
                                 sampling::SamplerState&) const override {
    return {.token = 1, .piece = "x"};
  }
  void Advance(TextRunnerState& state, TextRunnerToken) const override {
    ++dynamic_cast<RunnerState&>(state).slot.target;
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return dynamic_cast<const RunnerState&>(state).slot.target;
  }
};

void Run(RegistrationLayout layout, std::size_t budget) {
  auto runner = std::make_shared<Runner>(budget, layout);
  ComponentTextCache cache(runner, 1, budget, {});
  std::vector<Token> tokens(107031 + 256);
  std::iota(tokens.begin(), tokens.end(), 1);
  const auto request = [&](Rows length, Rows restored,
                           std::initializer_list<Rows> captures) {
    std::cout << "request L=" << length << " expected P=" << restored << '\n';
    const auto prompt = std::span(tokens).first(length);
    const auto before = runner->pool.lock()->restores;
    auto lease = cache.Acquire(prompt, {}, {}, true, length - 7, true);
    Require(lease->cached_tokens() == restored, "wrong restore source");
    std::cout << "restore-transfers P=" << restored
              << " count=" << runner->pool.lock()->restores - before << '\n';
    if (restored)
      Require(lease->restored_snapshot_bytes() ==
                  (restored == 87339 ? 2531356492ULL : 2809281392ULL),
              "native restored byte count differs from operator log");
    auto& state = dynamic_cast<RunnerState&>(lease->state());
    for (auto boundary : captures) {
      Require(state.slot.pending == 0, "execution overlaps a pending capture");
      if (layout.grow)
        while (state.slot.target < boundary) {
          const auto count = std::min<Rows>(4096, boundary - state.slot.target);
          lease->PreparePrefill(count);
          state.slot.target += count;
        }
      state.slot.target = boundary;
      const auto prefix = std::span(tokens).first(boundary);
      Require(lease->TryReserveSnapshot(lease->CaptureBytes(), boundary, false,
                                        boundary == length
                                            ? SnapshotPurpose::kRetry
                                            : SnapshotPurpose::kContinuation,
                                        prefix),
              "snapshot reservation refused");
      auto snapshot = lease->Capture();
      Require(lease->PublishSnapshot({prefix.begin(), prefix.end()},
                                     std::move(snapshot), false) != 0,
              "snapshot publication refused");
    }
    Require(state.slot.pending == 0, "generation overlaps a pending capture");
    const auto generated = length == 87339 ? 78 : length == 97460 ? 161 : 38;
    std::vector<Token> complete(prompt.begin(), prompt.end());
    complete.insert(complete.end(), generated, 1'000'000 + length);
    state.slot.target = complete.size();
    // The generated suffix differs from the next frozen request after P.
    // Keep a coherent live frontier before the following RAM restore.
    (void)lease->Commit({prompt.begin(), prompt.end()}, {},
                        std::move(complete));
    const auto backing = runner->pool.lock();
    backing->Refresh(0, backing->blocks.size());
    const auto retained =
        static_cast<std::size_t>(std::ranges::count_if(
            backing->marks, [](const auto& mark) { return mark.assigned; })) *
        65536;
    Require(retained == (length == 87339 ? 2961047552ULL : 3651076096ULL),
            "retained unique payload differs from operator log");
    std::cout << "request-completed L=" << length
              << " retained_payload_bytes=" << retained
              << " committed_bytes=" << backing->blocks.size() * 65536 << '\n';
  };
  try {
    request(87339, 0, {9059, 20480, 43008, 87339});
    request(97460, 87339, {96256, 97453, 97460});
    request(107031, 97460, {});
  } catch (...) {
    std::cerr << runner->pool.lock()->failure << '\n';
    throw;
  }
  const auto backing = runner->pool.lock();
  Require(backing->submissions == backing->completed,
          "pending transfer survived replay");
  Require(runner->registration_limits, "registration bound was not installed");
  std::cout << "PASS async r2/r3/r4 payload_capacity=" << backing->capacity
            << " submitted=" << backing->submissions
            << " completed=" << backing->completed
            << " released=" << backing->released
            << " reused=" << backing->reused << " pool_full=" << backing->full
            << " capacity_queries=" << runner->capacity_queries
            << " registration_limits=" << runner->registration_limits << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    RegistrationLayout layout;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument == "--separate-aliases")
        layout.check = true;
      else if (argument == "--registration-layout=adjacent")
        layout.check = layout.grow = true;
      else if (argument == "--registration-layout=gapped")
        layout.check = layout.grow = layout.gaps = true;
      else if (argument == "--trace-registrations")
        layout.trace = layout.grow = true;
      else if (argument.starts_with("--adjacent-boundary=")) {
        layout.check = layout.grow = layout.gaps = true;
        layout.adjacent_boundary = std::stoull(argument.substr(20));
      } else
        throw std::invalid_argument("unknown replay argument: " + argument);
    }
    Run(layout, 4ULL << 30);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
