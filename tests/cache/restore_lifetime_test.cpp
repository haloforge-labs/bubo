#include "src/cache/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;

namespace {
constexpr ResourceLimits kLimits{128 << 20, 128 << 20, 128 << 20,
                                  128 << 20};

// Physical buffers remain mapped after release, so a stale span deterministically
// reads poison rather than relying on allocator reuse or an access violation.
// ResourceCharge remains the authority for assignment, as in the real pool.
class PoisonPool {
public:
  struct Allocation {
    ResourceReservation reservation;
    std::shared_ptr<const void> owner;
    std::span<std::byte> bytes;
  };
  explicit PoisonPool(ResourceLedger& ledger) : ledger_(ledger) {}
  ~PoisonPool() {
    assert(acquisitions_ == releases_);
    for (const auto& block : blocks_)
      assert(!block->owned);
  }
  Allocation Acquire(ResourceCategory category, std::size_t bytes) {
    const auto capacity = (bytes + 4095) / 4096 * 4096;
    Block* selected = nullptr;
    for (const auto& block : blocks_) {
      const auto info = block->charge.Info();
      if (info.category == ResourceCategory::kBackingFree && !info.reserved) {
        assert(!block->owned);
        if (block->bytes.size() >= capacity) {
          selected = block.get();
          ++reuses_;
          break;
        }
      } else {
        assert(block->owned);
      }
    }
    if (!selected) {
      auto reservation = ledger_.Reserve(ResourceCategory::kBackingFree,
                                          capacity);
      auto block = std::make_unique<Block>();
      block->bytes.resize(capacity);
      block->charge = reservation.Convert();
      selected = block.get();
      blocks_.push_back(std::move(block));
    }
    assert(!selected->owned);
    auto reservation = selected->charge.ReserveBacking(category);
    selected->owned = true;
    ++acquisitions_;
    ++selected->generation;
    const auto generation = selected->generation;
    std::ranges::fill(selected->bytes, std::byte{0xa5});
    auto owner = std::shared_ptr<const void>(
        selected->bytes.data(), [this, selected, generation](const void*) {
          // A second release or an owner surviving reassignment is a failure.
          assert(selected->owned && selected->generation == generation);
          selected->owned = false;
          std::ranges::fill(selected->bytes, std::byte{0xdd});
          ++releases_;
        });
    return {std::move(reservation), std::move(owner), selected->bytes};
  }
  void Readable(std::span<const std::byte> bytes, bool require_backing) const {
    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    for (const auto& block : blocks_) {
      const auto base =
          reinterpret_cast<std::uintptr_t>(block->bytes.data());
      if (address < base || address - base >= block->bytes.size())
        continue;
      assert(block->owned);
      assert(bytes.size() <= block->bytes.size() - (address - base));
      assert(!std::ranges::all_of(bytes, [](std::byte b) {
        return b == std::byte{0xdd};
      }));
      assert(!std::ranges::all_of(bytes, [](std::byte b) {
        return b == std::byte{0xa5};
      }));
      return;
    }
    // An unmaterialized borrowed pin instead names live adapter storage.
    assert(!require_backing);
  }
  void Churn() {
    // Reassign every currently free block while source transfers are pending.
    // Holding all assignments together prevents repeatedly picking one block.
    std::vector<Allocation> allocated;
    for (const auto& block : blocks_) {
      const auto info = block->charge.Info();
      if (info.category == ResourceCategory::kBackingFree && !info.reserved) {
        auto allocation = Acquire(ResourceCategory::kPrivateTail, 32);
        std::ranges::fill(allocation.bytes, std::byte{0x3c});
        allocated.push_back(std::move(allocation));
      }
    }
  }
  std::size_t Reuses() const { return reuses_; }
  std::size_t Releases() const { return releases_; }

private:
  struct Block {
    ResourceCharge charge;
    std::vector<std::byte> bytes;
    bool owned{};
    std::size_t generation{};
  };
  ResourceLedger& ledger_;
  std::vector<std::unique_ptr<Block>> blocks_;
  std::size_t reuses_{}, releases_{}, acquisitions_{};
};

struct SavedState {
  std::vector<ComponentPosition> positions;
  std::map<std::uint32_t, std::vector<std::byte>> components;
};

struct Fixture {
  ResourceLedger ledger{kLimits};
  PoisonPool pool{ledger};
  FakeAdapter adapter{256};
  FakeStream stream{true};
  LeasedSlot slot{ledger, adapter, stream, SlotId{1}};
  SlotLease lease{slot.Acquire()};
  std::unique_ptr<ExecutionHistory> history;
  std::vector<Token> tokens;
  std::map<std::uint64_t, SavedState> saved;

  Fixture() { Cold(); }

  void Cold() {
    history = std::make_unique<ExecutionHistory>(ExecutionHistory::Cold(
        ledger, adapter.Components(), adapter.CompatibilityIdentity()));
    tokens.clear();
  }

  // Target and draft frontiers deliberately differ: Q=P-1, including at the
  // non-chunk-aligned B checkpoint. Values encode cut/full provenance too.
  static Token RowToken(Rows row, Rows boundary) {
    return static_cast<Token>(17 + row % 97 + (row < boundary ? 200 : 0));
  }
  void AppendTo(Rows end, Rows cut_boundary = 0) {
    const auto positions = adapter.Positions(lease.Execution());
    const auto target_first = positions[0].valid_rows;
    const auto draft_first = positions[1].valid_rows;
    assert(end >= target_first);
    const auto draft_end = end ? end - 1 : 0;
    assert(draft_end >= draft_first);
    std::vector<Token> target, draft;
    for (auto row = target_first; row < end; ++row)
      target.push_back(RowToken(row, cut_boundary));
    for (auto row = draft_first; row < draft_end; ++row)
      draft.push_back(RowToken(row, cut_boundary));
    adapter.Append(lease.Execution(), target, draft);
    tokens.insert(tokens.end(), target.begin(), target.end());
    assert(adapter.Positions(lease.Execution())[1].valid_rows == draft_end);
    CheckCurrent(cut_boundary);
  }
  SavedState ReadState() {
    SavedState result{adapter.Positions(lease.Execution()), {}};
    const auto descriptors = adapter.Components();
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
      const auto& d = descriptors[i];
      auto& bytes = result.components[d.id.value];
      bytes.resize(d.kind == ComponentKind::kPrivateState
                       ? d.state_bytes
                       : result.positions[i].valid_rows * d.row_bytes);
      auto copy = d.kind == ComponentKind::kPrivateState
                      ? adapter.CapturePrivate(lease.Execution(), d.id,
                                                 bytes, stream)
                      : adapter.CopyRowsOut(lease.Execution(), d.id, 0,
                                              result.positions[i].valid_rows,
                                              bytes, stream);
      assert(copy.Wait() == TransferResult::kSucceeded);
    }
    return result;
  }
  void CheckCurrent(Rows cut_boundary) {
    const auto state = ReadState();
    std::array<std::uint64_t, 4> recurrent{
        state.positions[0].valid_rows, state.positions[1].valid_rows, 0, 0};
    for (std::size_t component = 0; component < 2; ++component) {
      std::vector<Token> expected;
      std::uint64_t hash = 0xcbf29ce484222325ULL;
      auto context = hash;
      for (Rows row = 0; row < state.positions[component].valid_rows; ++row) {
        const auto token = RowToken(row, cut_boundary);
        expected.push_back(token);
        hash = (hash ^ context ^ token) * 0x100000001b3ULL;
        context = (context ^ token) * 0x100000001b3ULL;
      }
      recurrent[2 + component] = hash;
      const auto bytes = std::as_bytes(std::span(expected));
      const auto& actual =
          state.components.at(state.positions[component].id.value);
      assert(std::equal(bytes.begin(), bytes.end(), actual.begin(),
                        actual.end()));
    }
    const auto bytes = std::as_bytes(std::span(recurrent));
    const auto& actual = state.components.at(kRecurrent.value);
    assert(std::equal(bytes.begin(), bytes.end(), actual.begin(),
                      actual.end()));
  }
  Payload Save(const PayloadRequest& request) {
    auto allocation = pool.Acquire(request.category, request.bytes);
    const auto bytes = allocation.bytes.first(request.bytes);
    if (request.category != ResourceCategory::kPrivateState) {
      auto view = adapter.BorrowRowsView(lease.Execution(), request.component,
                                           request.first, request.end);
      return Payload::Borrowed(lease.Borrow(
          request.component, request.first, request.end,
          std::move(allocation.reservation), std::move(allocation.owner),
          bytes, view));
    }
    auto copy = adapter.CapturePrivate(lease.Execution(), request.component,
                                         bytes, stream);
    assert(copy.Wait() == TransferResult::kSucceeded);
    return Payload::Committed(allocation.reservation.Convert(),
                              std::move(allocation.owner));
  }
  std::shared_ptr<const Checkpoint> Capture() {
    auto state = ReadState();
    auto checkpoint = history->Capture(
        {tokens, InputIdentity(tokens.size()), state.positions,
         CheckpointPurpose::kPrompt, 1},
        [this](const PayloadRequest& request) { return Save(request); });
    saved.emplace(checkpoint->Id().value, std::move(state));
    return checkpoint;
  }
  void CheckPayload(const Payload& payload, const ComponentDescriptor& d,
                    Rows first, Rows end, const SavedState& expected) {
    const auto bytes = d.kind == ComponentKind::kPrivateState
                           ? d.state_bytes
                           : (end - first) * d.row_bytes;
    auto pin = payload.PinRows();
    const auto view = pin.Bytes().first(bytes);
    pool.Readable(view, !pin.Location());
    const auto& whole = expected.components.at(d.id.value);
    const auto offset = d.kind == ComponentKind::kPrivateState
                            ? 0 : first * d.row_bytes;
    assert(std::equal(view.begin(), view.end(), whole.begin() + offset));
  }
  void CheckCheckpoint(const Checkpoint& checkpoint) {
    assert(checkpoint.IsValid());
    const auto& expected = saved.at(checkpoint.Id().value);
    for (const auto& c : checkpoint.Components()) {
      if (c.private_state)
        CheckPayload(*c.private_state, c.descriptor, 0, 0, expected);
      for (const auto& chunk : c.chunks)
        CheckPayload(chunk.Storage(), c.descriptor, chunk.First(), chunk.End(),
                     expected);
      if (c.tail) {
        const auto end = c.position.valid_rows;
        CheckPayload(*c.tail, c.descriptor,
                     end / c.descriptor.rows_per_chunk *
                         c.descriptor.rows_per_chunk,
                     end, expected);
      }
    }
  }
  void Restore(const Checkpoint& checkpoint) {
    const auto& expected = saved.at(checkpoint.Id().value);
    // This also exercises the 6267 ordering without requiring its new APIs:
    // obsolete weak history is released before replacement history allocation.
    history.reset();
    adapter.BeginRestore(lease.Execution(), expected.positions);
    std::vector<RowPin> pins;
    for (const auto& c : checkpoint.Components()) {
      if (c.private_state)
        pins.push_back(c.private_state->PinRows());
      for (const auto& chunk : c.chunks)
        pins.push_back(chunk.Storage().PinRows());
      if (c.tail)
        pins.push_back(c.tail->PinRows());
    }
    std::vector<Completion> copies;
    std::size_t next = 0;
    for (const auto& c : checkpoint.Components()) {
      auto load = [&](Rows first, Rows end, bool private_state) {
        auto& pin = pins[next++];
        const auto bytes = private_state ? c.descriptor.state_bytes
                                        : (end - first) * c.descriptor.row_bytes;
        const auto view = pin.Bytes().first(bytes);
        pool.Readable(view, !pin.Location());
        copies.push_back(
            private_state
                ? adapter.LoadPrivate(lease.Execution(), c.descriptor.id,
                                         view, stream)
                : adapter.CopyRowsIn(lease.Execution(), c.descriptor.id, first,
                                        end, view, stream));
      };
      if (c.private_state)
        load(0, 0, true);
      for (const auto& chunk : c.chunks)
        load(chunk.First(), chunk.End(), false);
      if (c.tail) {
        const auto end = c.position.valid_rows;
        load(end / c.descriptor.rows_per_chunk * c.descriptor.rows_per_chunk,
             end, false);
      }
    }
    pool.Churn();
    for (auto& copy : copies)
      assert(copy.Wait() == TransferResult::kSucceeded);
    assert(adapter.Validate(lease.Execution(), expected.positions));
    pins.clear();
    pins.shrink_to_fit();
    history = std::make_unique<ExecutionHistory>(
        ExecutionHistory::Restored(ledger, checkpoint));
    tokens.assign(checkpoint.Tokens().begin(), checkpoint.Tokens().end());
    const auto actual = ReadState();
    assert(actual.positions == expected.positions);
    assert(actual.components == expected.components);
  }
  void Reassign() {
    lease.Commit();
    lease = slot.Acquire();
    Cold();
  }
};

void PairAfterPressureBlockReuse() {
  Fixture f;
  f.AppendTo(86708);
  auto older = f.Capture();
  f.AppendTo(105500);
  auto pressure = f.Capture();
  f.Reassign();
  older.reset();
  pressure.reset();
  f.pool.Churn();

  constexpr Rows first_boundary = 18324 - 256;
  f.AppendTo(first_boundary, first_boundary);
  auto boundary = f.Capture();
  f.AppendTo(18324, first_boundary);
  auto final = f.Capture();
  f.CheckCheckpoint(*boundary);
  // The final checkpoint shares all completed chunks below B. Its partial
  // chunk contains different rows above B and must not replace B's tail.
  assert(boundary->Components()[0].chunks.front().SameChunk(
      final->Components()[0].chunks.front()));
  f.Restore(*boundary);
  final.reset();
  f.pool.Churn();
  f.CheckCheckpoint(*boundary);
  constexpr Rows next_boundary = 43218 - 256;
  for (Rows point : {Rows{20480}, Rows{30720}, Rows{34011}, next_boundary}) {
    f.AppendTo(point, next_boundary);
    auto checkpoint = f.Capture();
    f.CheckCheckpoint(*boundary);
  }
  f.AppendTo(43218, next_boundary);
  auto next_final = f.Capture();
  f.CheckCheckpoint(*boundary);
  f.CheckCheckpoint(*next_final);
  assert(f.pool.Reuses() > 0 && f.pool.Releases() > 0);
}

void ForkSiblingEvictionPreservesRestoreSource() {
  Fixture f;
  f.AppendTo(84861);
  auto shallow = f.Capture();
  f.AppendTo(106496);
  auto grid = f.Capture();
  f.AppendTo(107253);
  auto sibling = f.Capture();
  f.AppendTo(107260);
  auto source = f.Capture();
  assert(source->Components()[0].chunks.front().SameChunk(
      sibling->Components()[0].chunks.front()));
  f.Restore(*source);
  shallow.reset();
  sibling.reset();
  grid.reset();
  f.pool.Churn();
  f.CheckCheckpoint(*source);
  f.AppendTo(107541);
  auto stable = f.Capture();
  f.AppendTo(107548);
  auto final = f.Capture();
  f.CheckCheckpoint(*source);
  // r9 restores the still-indexed ancestor after sibling eviction and block
  // reassignment, while complete target/draft chunks remain shared by children.
  f.Restore(*source);
  stable.reset();
  final.reset();
  f.pool.Churn();
  f.CheckCheckpoint(*source);
  f.AppendTo(130018 - 256);
  auto tail_boundary = f.Capture();
  f.AppendTo(130018);
  auto last = f.Capture();
  f.CheckCheckpoint(*source);
  f.CheckCheckpoint(*tail_boundary);
  f.CheckCheckpoint(*last);
  assert(f.pool.Reuses() > 0 && f.pool.Releases() > 0);
}
}  // namespace

int main() {
  PairAfterPressureBlockReuse();
  ForkSiblingEvictionPreservesRestoreSource();
  std::cout << "component restore lifetime: pair and fork passed\n";
}
