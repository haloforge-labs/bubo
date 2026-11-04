// Real operator request/capture order with the native component geometry.
// Payloads are compact ownership markers: this test exercises accounting,
// inheritance, admission and prefix selection, and never reads model bytes.
#include "tests/cli/component_sequence_trace.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/cache/prefix_index.hpp"
#include "src/cache/retention.hpp"

namespace {
using namespace gufo::cache;
namespace trace = gufo::tests::component_sequence_trace;
using Tokens = std::vector<Token>;

void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}

std::vector<ComponentDescriptor> Descriptors() {
  std::vector<ComponentDescriptor> result;
  for (const auto& component : trace::kComponents)
    result.push_back({{static_cast<std::uint32_t>(result.size() + 1)},
                      1,
                      component.row_bytes ? ComponentKind::kAppendRows
                                          : ComponentKind::kPrivateState,
                      component.row_bytes, component.rows_per_chunk,
                      component.state_bytes});
  return result;
}

std::vector<ComponentPosition> Positions(std::uint32_t target) {
  std::vector<ComponentPosition> result;
  const auto draft = target ? target - 1 : 0;
  for (const auto& component : trace::kComponents) {
    auto rows = component.draft ? draft : target;
    if (!component.row_bytes)
      rows = target;
    else if (component.pooled)
      rows = rows > 2048 ? rows / 4 : 0;
    result.push_back({{static_cast<std::uint32_t>(result.size() + 1)}, rows});
  }
  return result;
}

struct MarkerPool : std::enable_shared_from_this<MarkerPool> {
  struct Marker {
    std::shared_ptr<MarkerPool> pool;
    std::size_t first{}, count{};
    std::uint64_t value{};
    ~Marker() {
      std::fill_n(pool->used.begin() + first, count, std::uint8_t{});
    }
  };
  std::vector<ResourceCharge> blocks;
  std::vector<std::uint8_t> used;
  ResourceCharge metadata;
  std::size_t next{}, next_private{};
  std::uint64_t serial{};

  explicit MarkerPool(ResourceLedger& ledger) {
    metadata = ledger.Reserve(ResourceCategory::kMetadata,
                               trace::kBackingMetadataBytes).Convert();
    blocks.reserve(trace::kBackingBlocks);
    used.resize(trace::kBackingBlocks);
    for (std::size_t i = 0; i < trace::kBackingBlocks; ++i)
      blocks.push_back(ledger.Reserve(ResourceCategory::kBackingFree,
                                      trace::kPayloadBlockBytes).Convert());
    next_private = blocks.size();
  }

  Payload Allocate(const PayloadRequest& request) {
    const auto count = std::max<std::size_t>(
        1, (request.bytes + trace::kPayloadBlockBytes - 1) /
               trace::kPayloadBlockBytes);
    if (count > blocks.size())
      throw ResourceExhausted();
    const bool from_end = request.category == ResourceCategory::kPrivateState;
    std::optional<std::size_t> selected;
    const auto free = [&](std::size_t first) {
      return first + count <= used.size() &&
             std::none_of(used.begin() + first,
                          used.begin() + first + count,
                          [](auto value) { return value != 0; });
    };
    if (from_end) {
      for (std::size_t end = std::min(next_private, blocks.size());
           end >= count; --end)
        if (free(end - count)) {
          selected = end - count;
          break;
        }
      if (!selected)
        for (std::size_t end = blocks.size(); end >= count; --end)
          if (free(end - count)) {
            selected = end - count;
            break;
          }
    } else {
      for (std::size_t first = next; first + count <= blocks.size(); ++first)
        if (free(first)) {
          selected = first;
          break;
        }
      if (!selected)
        for (std::size_t first = 0; first + count <= blocks.size(); ++first)
          if (free(first)) {
            selected = first;
            break;
          }
    }
    if (!selected)
      throw ResourceExhausted();
    auto reservation = ResourceCharge::ReserveBacking(
        std::span<const ResourceCharge>(blocks).subspan(*selected, count),
        request.category);
    auto marker = std::make_shared<Marker>();
    marker->pool = shared_from_this();
    marker->first = *selected;
    marker->count = count;
    marker->value = ++serial;
    std::fill_n(used.begin() + *selected, count, std::uint8_t{1});
    if (from_end)
      next_private = *selected;
    else
      next = *selected + count;
    // Only the accounting/ownership graph is under test. PinRows, transfers
    // and serialization are deliberately absent: the marker is not a byte
    // representation of the charged multi-megabyte payload.
    auto owner = std::shared_ptr<const void>(marker, &marker->value);
    return Payload::Committed(reservation.Convert(), std::move(owner));
  }
};

struct Log final : RetentionEventSink {
  std::size_t evictions{};
  void Emit(const RetentionEvent& event) noexcept override {
    evictions += event.action == RetentionAction::kRemoved;
  }
};

struct Replay {
  ResourceLedger ledger{{trace::kCacheCapacity, trace::kCacheCapacity, 0, 0}};
  PrefixIndex index{ledger};
  Log log;
  RetentionPolicy retention{ledger, index, &log};
  std::vector<ComponentDescriptor> descriptors{Descriptors()};
  std::shared_ptr<MarkerPool> pool{std::make_shared<MarkerPool>(ledger)};
  std::unique_ptr<ExecutionHistory> history;
  std::optional<IndexEntryId> live;
  Tokens live_tokens;
  std::array<Tokens, trace::kLineages.size()> lineages;
  std::size_t requests{}, captures{}, refused{}, regressions{};
  std::size_t impossible_refusals{}, fallback_plans{}, fallback_captures{};
  std::array<std::uint32_t, trace::kLineages.size()> published_fallbacks{};
  std::uint32_t row204_last_reuse{};

  Replay() { index.Register({1}, descriptors); }

  Tokens Prompt(const trace::Request& request) {
    auto& previous = lineages[request.lineage];
    if (request.parent_lineage >= 0 && previous.empty())
      previous = lineages[static_cast<std::size_t>(request.parent_lineage)];
    const auto shared = std::min<std::size_t>(
        previous.size(), std::max(request.shared_prefix_tokens,
                                  request.minimum_reuse));
    Tokens prompt(previous.begin(), previous.begin() + shared);
    prompt.reserve(request.length);
    while (prompt.size() < request.length)
      prompt.push_back(static_cast<Token>(
          1'000'000 * (request.lineage + 1) + prompt.size() + requests * 200'000));
    return prompt;
  }

  bool Capture(Tokens& prompt, std::uint32_t boundary,
               RetentionPurpose purpose, CheckpointId source,
               std::uint32_t reused, std::span<const CheckpointId> protected_ids,
               std::size_t protected_private_bytes, std::uint32_t stable) {
    const auto positions = Positions(boundary);
    // Shared complete rows are counted once in the incoming checkpoint. Each
    // protected History checkpoint additionally keeps its own private state
    // and partial tails. Those bytes cannot be spent by reclaiming its record.
    const auto whole = RoundedCheckpointBytes(
        descriptors, positions, trace::kPayloadBlockBytes);
    if (protected_private_bytes > trace::kPayloadCapacity ||
        whole > trace::kPayloadCapacity - protected_private_bytes) {
      ++refused;
      ++impossible_refusals;
      return false;
    }
    const auto tokens = std::span<const Token>(prompt).first(boundary);
    const auto input = InputIdentity(boundary, {});
    RetentionRequest retain{.tokens = tokens,
                            .compatibility = {1},
                            .purpose = purpose,
                            .stable = purpose == RetentionPurpose::kRetry
                                          ? stable : 0,
                            .source = source,
                            .preserve_source = purpose !=
                                               RetentionPurpose::kContinuation,
                            .new_payload_bytes = history->NewPayloadBytes(positions),
                            .protected_checkpoints = protected_ids,
                            .reused = reused != 0,
                            .reused_tokens = reused};
    std::shared_ptr<const Checkpoint> checkpoint;
    try {
      // Mirrors ComponentTextCache::Lease::Capture: the history pins inherited
      // chunks while prepare allocates largest-first and reclaims on refusal.
      const auto inherited = history->PinInheritedChunks();
      (void)inherited;
      std::vector<std::optional<Payload>> prepared;
      std::size_t next = 0;
      checkpoint = history->Capture(
          {tokens, input, positions,
           purpose == RetentionPurpose::kHistory ? CheckpointPurpose::kGrid
                                                 : CheckpointPurpose::kPrompt,
           0},
          [&](const PayloadRequest&) { return std::move(*prepared.at(next++)); },
          [&](std::span<const PayloadRequest> payloads) {
            std::vector<std::size_t> order(payloads.size());
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&](auto a, auto b) {
              return payloads[a].bytes != payloads[b].bytes
                         ? payloads[a].bytes > payloads[b].bytes : a < b;
            });
            while (true) {
              try {
                prepared.resize(payloads.size());
                for (const auto i : order)
                  prepared[i].emplace(pool->Allocate(payloads[i]));
                break;
              } catch (const ResourceExhausted&) {
                prepared.clear();
                if (!retention.Reclaim(retain))
                  throw;
              }
            }
          });
    } catch (const ResourceExhausted&) {
      ++refused;
      return false;
    }
    retain.new_payload_bytes = 0;
    if (!retention.Admit(retain, [&] { return checkpoint; })) {
      ++refused;
      return false;
    }
    ++captures;
    return true;
  }

  void Request(const trace::Request& request, std::span<const trace::Event> events) {
    auto prompt = Prompt(request);
    const auto input = InputIdentity(prompt.size(), {});
    auto found = index.Lookup({{1}, prompt, input});
    const auto reused = found.selected
                            ? static_cast<std::uint32_t>(found.selected->boundary)
                            : 0U;
    CheckpointId source{};
    if (found.selected && found.selected->checkpoint) {
      source = found.selected->checkpoint->Id();
      history = std::make_unique<ExecutionHistory>(
          ExecutionHistory::Restored(ledger, *found.selected->checkpoint));
      retention.Touch(source);
    } else if (!found.selected || !found.selected->live) {
      history = std::make_unique<ExecutionHistory>(
          ExecutionHistory::Cold(ledger, descriptors, {1}));
    }
    if (live) {
      index.Erase(*live);
      live.reset();
    }
    found = {};
    ++requests;
    std::cout << request.point_id << " L=" << request.length << " P=" << reused
              << " baseline=" << request.minimum_reuse << '\n';
    if (reused < request.minimum_reuse)
      ++regressions;
    if (published_fallbacks[request.lineage] &&
        request.shared_prefix_tokens >= published_fallbacks[request.lineage])
      Require(reused >= published_fallbacks[request.lineage],
              std::string(request.point_id) +
                  " did not reuse the preceding feasible checkpoint");
    if (request.point_id == "more-ernie-row204--p05")
      row204_last_reuse = reused;

    struct Attempt {
      std::uint32_t position;
      RetentionPurpose purpose;
      bool fallback{false};
    };
    std::vector<Attempt> attempts;
    for (const auto& event : events)
      if ((event.kind == trace::EventKind::kCapture ||
           event.kind == trace::EventKind::kRefused) &&
          event.tokens > reused && event.tokens <= request.length)
        attempts.push_back({event.tokens,
                            event.tokens == request.length
                                ? RetentionPurpose::kRetry
                            : event.tokens % 2048 == 0
                                ? RetentionPurpose::kHistory
                                : RetentionPurpose::kContinuation});
    std::ranges::sort(attempts, {}, &Attempt::position);
    attempts.erase(std::unique(attempts.begin(), attempts.end(),
                                [](const auto& a, const auto& b) {
                                  return a.position == b.position;
                                }), attempts.end());
    // The deepest off-grid prompt capture is the stable framing boundary.
    // Retry retention needs that frontier to distinguish a redundant final
    // copy from the conversation's useful stable copy.
    std::uint32_t stable = 0;
    for (const auto& attempt : attempts)
      if (attempt.position < request.length && attempt.position % 2048 != 0)
        stable = std::max(stable, attempt.position);
    const auto planned_private = [&](std::uint32_t boundary) {
      std::size_t bytes = 0;
      for (const auto& attempt : attempts)
        if (attempt.purpose == RetentionPurpose::kHistory &&
            attempt.position > reused && attempt.position < boundary)
          bytes += RoundedCheckpointBytes(
              descriptors, Positions(attempt.position),
              trace::kPayloadBlockBytes, false);
      return bytes;
    };
    const auto fits_planned = [&](std::uint32_t boundary) {
      const auto private_bytes = planned_private(boundary);
      return private_bytes <= trace::kPayloadCapacity &&
             RoundedCheckpointBytes(descriptors, Positions(boundary),
                                    trace::kPayloadBlockBytes) <=
                 trace::kPayloadCapacity - private_bytes;
    };
    const bool needs_fallback = std::ranges::any_of(
        attempts, [&](const auto& attempt) {
          return !fits_planned(attempt.position);
        });
    std::uint32_t feasible = 0, maximum = 0;
    if (needs_fallback) {
      // Compute the unrounded maximum independently of the checkpoint grid.
      // The floor then proves the checkpoint is within one 4096-row grid.
      std::uint32_t first = 0, last = request.length;
      while (first < last) {
        const auto middle = first + (last - first + 1) / 2;
        if (fits_planned(middle))
          first = middle;
        else
          last = middle - 1;
      }
      maximum = first;
      feasible = maximum / trace::kPrefillChunkRows * trace::kPrefillChunkRows;
      Require(maximum - feasible < trace::kPrefillChunkRows &&
                  fits_planned(feasible),
              "fallback is not the deepest feasible checkpoint grid");
      if (request.point_id == "more-ernie-row204--p03")
        Require(feasible == 118784,
                "row204 p03 omitted either protected History private state");
      if (request.point_id == "more-ernie-row204--p04")
        Require(feasible == 131072,
                "row204 p04 did not recover the deeper feasible frontier");
      if (feasible > reused &&
          std::ranges::none_of(attempts, [&](const auto& attempt) {
            return attempt.position == feasible;
          })) {
        // A fallback is an ordinary continuation, never another protected
        // History record that would consume the capacity it just planned.
        attempts.push_back({feasible, RetentionPurpose::kContinuation, true});
        ++fallback_plans;
      }
    }
    std::ranges::sort(attempts, {}, &Attempt::position);
    std::vector<CheckpointId> protected_ids;
    std::size_t protected_private_bytes = 0;
    for (const auto& attempt : attempts) {
      const auto whole = RoundedCheckpointBytes(
          descriptors, Positions(attempt.position), trace::kPayloadBlockBytes);
      const bool impossible = protected_private_bytes > trace::kPayloadCapacity ||
                              whole > trace::kPayloadCapacity -
                                          protected_private_bytes;
      const auto evictions_before = log.evictions;
      const auto records_before = retention.Size();
      const auto ledger_before = ledger.Snapshot().total_bytes;
      const bool captured = Capture(
          prompt, attempt.position, attempt.purpose, source, reused,
          protected_ids, protected_private_bytes, stable);
      if (impossible)
        Require(!captured && log.evictions == evictions_before &&
                    retention.Size() == records_before &&
                    ledger.Snapshot().total_bytes == ledger_before,
                "impossible capture changed retention or backing accounting");
      if (captured && attempt.fallback) {
        ++fallback_captures;
        published_fallbacks[request.lineage] = attempt.position;
        Require(attempt.position > 0 && attempt.position == feasible &&
                    maximum - attempt.position < trace::kPrefillChunkRows,
                "published fallback does not match the feasible-depth oracle");
      }
      if (captured && attempt.purpose == RetentionPurpose::kHistory) {
        auto published = index.Lookup(
            {{1}, std::span<const Token>(prompt).first(attempt.position),
             InputIdentity(attempt.position, {})});
        if (published.selected && published.selected->checkpoint) {
          protected_ids.push_back(published.selected->checkpoint->Id());
          protected_private_bytes += RoundedCheckpointBytes(
              descriptors, Positions(attempt.position),
              trace::kPayloadBlockBytes, false);
        }
      }
    }
    lineages[request.lineage] = prompt;
    for (std::uint32_t i = 0; i < request.generated_tokens; ++i)
      lineages[request.lineage].push_back(static_cast<Token>(
          20'000'000 + request.lineage * 100'000 + requests * 512 + i));
    live_tokens = lineages[request.lineage];
    live = index.Insert(LiveFrontier{{{1}, requests}, live_tokens, {1}, {},
                                      Positions(static_cast<std::uint32_t>(
                                          live_tokens.size()))});
    Require(ledger.Snapshot().total_bytes <= trace::kCacheCapacity,
            "ledger exceeded the real 4 GiB cap");
  }
};

template<std::size_t N, std::size_t E>
void ReplayTrace(const std::array<trace::Request, N>& requests,
                 const std::array<trace::Event, E>& events, bool deep) {
  Replay replay;
  for (const auto& request : requests)
    replay.Request(request, std::span(events).subspan(request.event_begin,
                                                     request.event_count));
  std::cout << "requests=" << replay.requests << " captures=" << replay.captures
            << " refused=" << replay.refused << " evictions="
            << replay.log.evictions << " regressions=" << replay.regressions
            << " impossible=" << replay.impossible_refusals
            << " fallback_plans=" << replay.fallback_plans
            << " fallback_captures=" << replay.fallback_captures
            << '\n';
  if (deep) {
    Require(replay.row204_last_reuse > 0,
            "row204 p05 lost the complete lineage after impossible captures");
    Require(replay.impossible_refusals > 0 && replay.fallback_captures > 0,
            "deep sequence did not exercise impossible admission and fallback");
    Require(replay.fallback_captures == replay.fallback_plans,
            "a feasible checkpoint was planned but not published");
  }
  Require(replay.regressions == 0, "reuse fell below the recorded b0 baseline");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string_view(argv[1]) == "--fork")
      ReplayTrace(trace::kForkRequests, trace::kForkEvents, false);
    else
      ReplayTrace(trace::kDeepRequests, trace::kDeepEvents, true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "component sequence failure: " << error.what() << '\n';
    return 1;
  }
}
