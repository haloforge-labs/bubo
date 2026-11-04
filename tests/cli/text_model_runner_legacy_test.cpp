#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/cli/serve/continuation_disk_store.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/prefill_tail_cache.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/npu/prefill_plan.hpp"

using gufo::tokenization::kImEnd;
using gufo::tokenization::kImStart;

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerAdvance;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerDiskCacheOptions;
using gufo::server::TextRunnerMeasuredResources;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeStats {
  bool fail_after_advance{false};
  std::size_t states_created{0};
  std::size_t invalidations{0};
  std::size_t snapshot_restores{0};
  std::size_t snapshot_size_queries{0};
  std::size_t snapshot_captures{0};
  std::size_t cancellation_bindings{0};
  std::size_t cancellation_clears{0};
  std::vector<std::vector<TextRunnerToken>> prepared_prefixes;
  std::vector<std::size_t> prefill_spans;
  std::vector<TextRunnerToken> advanced_tokens;
  std::vector<std::vector<TextRunnerToken>> advanced_batches;
};

class FakeState final : public TextRunnerState {
public:
  FakeState(std::shared_ptr<FakeStats> stats, std::size_t measured_bytes)
      : stats_(std::move(stats)), measured_bytes_(measured_bytes) {}

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    if (is_cancelled) {
      ++stats_->cancellation_bindings;
    } else {
      ++stats_->cancellation_clears;
    }
  }

  void Invalidate() noexcept override {
    ++stats_->invalidations;
    position = 0;
    decode_count = 0;
    frontier.reset();
    policy_provenance.clear();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {
        .per_request_state_bytes = measured_bytes_,
        .temporary_scratch_bytes = 16,
    };
  }

  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;
  // One byte per computed row: 1 = expert cut, 0 = ordinary full-k.
  std::vector<std::uint8_t> policy_provenance;

private:
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected fake runner state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected fake runner state");
  }
  return *fake;
}

class FakeRunner : public TextModelRunner {
public:
  FakeRunner(std::shared_ptr<FakeStats> stats, std::size_t measured_bytes = 64,
             std::size_t state_capacity_bytes = 256,
             std::size_t retained_snapshot_capacity_bytes = 256)
      : stats_(std::move(stats)),
        measured_bytes_(measured_bytes),
        state_capacity_bytes_(state_capacity_bytes),
        retained_snapshot_capacity_bytes_(retained_snapshot_capacity_bytes) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "fake-model",
        .state_abi = "fake-state-v1",
        .max_context = 64,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = state_capacity_bytes_,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 16,
        .retained_snapshot_capacity_bytes = retained_snapshot_capacity_bytes_,
        .retained_snapshot_ceiling_bytes = retained_snapshot_ceiling_bytes,
        .requires_device_runtime_lock = false,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 2,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 4,
        },
    };
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(text.size());
    for (const char value : text) {
      tokens.push_back(static_cast<unsigned char>(value));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    if (request.messages.empty()) {
      return std::nullopt;
    }
    return Tokenize(request.messages.front().content);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      text += std::to_string(token);
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    ++stats_->states_created;
    return std::make_unique<FakeState>(stats_, measured_bytes_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& fake = RequireFakeState(state);
    if (fake.position != prefix.size()) {
      throw std::logic_error("fake retained prefix position mismatch");
    }
    stats_->prepared_prefixes.emplace_back(prefix.begin(), prefix.end());
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& fake = RequireFakeState(state);
    if (offset != fake.position || offset >= prompt.size()) {
      throw std::logic_error("invalid fake prefill position");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    stats_->prefill_spans.push_back(consumed);
    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = 90;
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, gufo::sampling::SamplerState&) const override {
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("fake state has no decode frontier");
    }
    if (fake.decode_count == 2) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    const TextRunnerToken token =
        *fake.frontier + static_cast<TextRunnerToken>(fake.decode_count);
    return {
        .token = token,
        .piece = std::to_string(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& fake = RequireFakeState(state);
    stats_->advanced_tokens.push_back(token);
    ++fake.position;
    ++fake.decode_count;
    if (!fake.policy_provenance.empty())
      fake.policy_provenance.push_back(0);
    if (stats_->fail_after_advance)
      throw std::runtime_error("injected failure after state mutation");
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      tokens.push_back(advance.token);
    }
    stats_->advanced_batches.push_back(std::move(tokens));
    for (const auto& advance : advances) {
      Advance(advance.state.get(), advance.token);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

protected:
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
  std::size_t state_capacity_bytes_;
  std::size_t retained_snapshot_capacity_bytes_;

public:
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;

private:
};

// A runner without engine settings throws the dedicated exception, so server
// startup can leave the "engine" row out instead of failing with no message.
void TestRunnerWithoutDebugSettingsThrowsTheDedicatedException() {
  FakeRunner runner(std::make_shared<FakeStats>());
  bool dedicated = false;
  try {
    (void)runner.DebugSettings();
  } catch (const gufo::server::DebugSettingsUnsupported&) {
    dedicated = true;
  } catch (...) {
  }
  Expect(dedicated, "a runner without debug settings throws DebugSettingsUnsupported");
  bool still_invalid_argument = false;
  try {
    (void)runner.DebugSettings();
  } catch (const std::invalid_argument&) {
    still_invalid_argument = true;
  } catch (...) {
  }
  Expect(still_invalid_argument, "DebugSettingsUnsupported stays an invalid_argument for other callers");
}

void TestBoundedPrefillDecodeAndPrefixReuse() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  Expect(pool.runner().Descriptor().model_id == "fake-model",
         "pool exposes the validated runner");
  Expect(pool.capacity() == 1, "pool exposes its bounded capacity");

  {
    auto request = pool.Acquire({1, 2, 3, 4});
    Expect(static_cast<bool>(request), "cold request acquires state");
    Expect(!request.cache_hit(), "first request is a miss");
    Expect(request.cached_prompt_tokens() == 0,
           "cold request starts at token zero");

    const auto first = request.Prefill(2);
    Expect(first.consumed_tokens == 2 && !first.decode_ready,
           "first prefill obeys its input budget");
    const auto second = request.Prefill(8);
    Expect(second.consumed_tokens == 2 && second.decode_ready,
           "second prefill reaches the decode frontier");

    const auto token_0 = request.SelectNext();
    Expect(!token_0.stop && token_0.token == 90,
           "first frontier token is selected");
    request.Advance();
    const auto token_1 = request.SelectNext();
    Expect(!token_1.stop && token_1.token == 91,
           "second frontier token is selected");
    request.Advance();
    Expect(request.SelectNext().stop,
           "runner reports an explicit stop boundary");
    request.Commit();
  }

  {
    auto extension = pool.Acquire({1, 2, 3, 4, 90, 91, 7, 8});
    Expect(extension.cache_hit(), "exact extension reuses opaque state");
    Expect(extension.cached_prompt_tokens() == 6,
           "cache boundary includes advanced decode tokens");
    Expect(
        stats->prepared_prefixes ==
            std::vector<std::vector<TextRunnerToken>>({{1, 2, 3, 4, 90, 91}}),
        "runner prepares retained metadata before suffix prefill");
    const auto suffix = extension.Prefill(16);
    Expect(suffix.consumed_tokens == 2 && suffix.decode_ready,
           "only the uncached suffix is prefetched");
    extension.Invalidate();
  }

  Expect(stats->prefill_spans == std::vector<std::size_t>({2, 2, 2}),
         "runner receives deterministic bounded prefill work units");
  Expect(stats->advanced_tokens == std::vector<TextRunnerToken>({90, 91}),
         "runner receives one decode advance per emitted token");
}

void TestAbandonedRequestRollsBackState() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto request = pool.Acquire({4, 5, 6});
    (void)request.Prefill(3);
  }
  Expect(stats->invalidations == 1,
         "abandoned request invalidates partially executed state");

  auto retry = pool.Acquire({4, 5, 6, 7});
  Expect(!retry.cache_hit(), "rolled-back state is not reusable");
  retry.Invalidate();
}

void TestRequestBindsAndClearsCancellation() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  auto request = pool.Acquire({4, 5, 6}, [] { return false; });
  Expect(stats->cancellation_bindings == 1,
         "request binds its cancellation check to opaque state");
  request.Invalidate();
  Expect(stats->cancellation_clears == 1,
         "request clears its cancellation check before releasing state");
}

void TestBatchedAdvancePreservesIndependentRequests() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 2);

  auto first = pool.Acquire({1});
  auto second = pool.Acquire({2});
  Expect(first.Prefill(1).decode_ready && second.Prefill(1).decode_ready,
         "independent requests reach their decode frontiers");
  Expect(first.SelectNext().token == 90 && second.SelectNext().token == 90,
         "independent requests select their pending tokens");

  const auto plan = pool.SelectDecodePlan(2);
  Expect(
      plan.kind == TextExecutionPlanKind::kBatched && plan.physical_width == 2,
      "two ready requests select W=2");

  std::array<TextRunnerPool::Request*, 2> requests{&first, &second};
  pool.AdvanceBatch(requests, plan);
  Expect(stats->advanced_batches ==
             std::vector<std::vector<TextRunnerToken>>{{90, 90}},
         "one batched runner call receives both request tokens");

  Expect(first.SelectNext().token == 91 && second.SelectNext().token == 91,
         "batched advance independently updates both request states");
  first.Invalidate();
  second.Invalidate();
}

void TestResourceClaimsAreValidatedBeforeAllocation() {
  auto stats = std::make_shared<FakeStats>();
  bool rejected = false;
  std::string message;
  try {
    auto runner = std::make_shared<FakeRunner>(stats, 64, 159);
    TextRunnerPool pool(runner, 2);
  } catch (const std::invalid_argument& error) {
    rejected = true;
    message = error.what();
  }
  Expect(rejected, "aggregate request-state claim must fit capacity");
  Expect(message.starts_with("2 sessions do not fit: each needs ") &&
             message.find("GiB free after loading the model") !=
                 std::string::npos &&
             message.ends_with("Lower --context or --sessions."),
         "capacity refusal names the sessions, the sizes and the remedy");
  Expect(stats->states_created == 0,
         "invalid resource claim is rejected before state allocation");
  // The same refusal as typed numbers, for serve's exit 19 memory_insufficient (no prose matching).
  const auto fit = gufo::server::LastSessionsFitFailure();
  Expect(fit.has_value() && fit->sessions == 2 && fit->context == 64 && fit->available_bytes == 159 && fit->needed_bytes > 159 &&
             fit->per_session_bytes * 2 <= fit->needed_bytes + 1,
         "the refusal also records sessions, context, needed and available bytes");
}

void TestLoadRefusesOnlyThePhysicalImpossibility() {
  // Operator decision 10-20: a configuration that runs, thin on RAM, must start (with the load_spill warning); only a need beyond the free carve-out
  // plus ALL available RAM is refused (exit 19 in serve). Numbers in bytes: 2 sessions x (64 state + 16 scratch) = 160.
  struct Source final : gufo::platform::MemoryReadingSource {
    gufo::platform::MemoryReadings readings;
    gufo::platform::MemoryReadings Read() noexcept override { return readings; }
  } source;
  source.readings.ram_ok = true;
  source.readings.gpu_ok = true;
  source.readings.gpu_carve_known = true;
  source.readings.gpu_carve_free = 100;
  source.readings.gpu_local_free = 1000;   // the reported free, which overlaps RAM
  source.readings.ram_available = 70;
  gufo::platform::MemoryGuard guard(&source, 20);   // keep-free floor 20: 50 above it
  gufo::platform::SetProcessMemoryGuard(&guard);
  const auto capacity = [&] { return static_cast<std::size_t>(gufo::platform::StateCapacityBytes(1000)); };
  Expect(capacity() == 170, "physical capacity = free carve-out 100 + available RAM 70");
  {   // need 160: 60 beyond the carve-out, 50 above the floor -> eats 10 into the floor: starts
    auto stats = std::make_shared<FakeStats>();
    TextRunnerPool pool(std::make_shared<FakeRunner>(stats, 64, capacity()), 2);
    Expect(stats->states_created == 2, "a start that eats into the floor still creates its sessions");
    const auto spill = gufo::platform::StateSpillFor(160);
    Expect(spill.known && spill.spill_bytes == 60 && spill.floor_eaten_bytes == 10, "the warning's numbers: spill 60, floor eaten 10");
  }
  source.readings.ram_available = 40;   // capacity 140 < 160: physically impossible
  bool refused = false;
  try {
    auto stats = std::make_shared<FakeStats>();
    TextRunnerPool pool(std::make_shared<FakeRunner>(stats, 64, capacity()), 2);
  } catch (const gufo::server::SessionsDoNotFit& error) {
    const auto& f = error.failure();
    refused = f.sessions == 2 && f.available_bytes == 140 && f.needed_bytes == 160 && f.floor_bytes == 20;
  }
  Expect(refused, "beyond the physical capacity: SessionsDoNotFit with the typed numbers");
  gufo::platform::SetProcessMemoryGuard(nullptr);
}

class FakeSnapshot : public TextRunnerSnapshot {
public:
  FakeSnapshot(std::size_t position, std::size_t decode_count,
               std::optional<TextRunnerToken> frontier)
      : position(position), decode_count(decode_count), frontier(frontier) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return sizeof(FakeSnapshot);
  }

  std::size_t position;
  std::size_t decode_count;
  std::optional<TextRunnerToken> frontier;
};

// Provenance is a CPU oracle alongside the modeled fixed-size backend payload.
// Keep ordinary fake payload accounting unchanged for existing budget tests.
class PolicySnapshot : public FakeSnapshot {
public:
  explicit PolicySnapshot(const FakeState& state)
      : FakeSnapshot(state.position, state.decode_count, state.frontier),
        policy_provenance(state.policy_provenance) {}
  PolicySnapshot(std::size_t position, std::vector<std::uint8_t> provenance)
      : FakeSnapshot(position, 0, 90),
        policy_provenance(std::move(provenance)) {}
  std::vector<std::uint8_t> policy_provenance;
};

class SnapshotRunner : public FakeRunner {
public:
  using FakeRunner::FakeRunner;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = FakeRunner::Descriptor();
    descriptor.capabilities.snapshot = true;
    descriptor.capabilities.fork = true;
    return descriptor;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState&) const override {
    ++stats_->snapshot_size_queries;
    return sizeof(FakeSnapshot);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    ++stats_->snapshot_captures;
    const auto& fake = RequireFakeState(state);
    if (!fake.policy_provenance.empty())
      return std::make_unique<PolicySnapshot>(fake);
    return std::make_unique<FakeSnapshot>(fake.position, fake.decode_count,
                                          fake.frontier);
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
    if (fake == nullptr) {
      throw std::invalid_argument("snapshot type mismatch");
    }
    auto& restored = RequireFakeState(state);
    restored.position = fake->position;
    restored.decode_count = fake->decode_count;
    restored.frontier = fake->frontier;
    const auto* policy = dynamic_cast<const PolicySnapshot*>(&snapshot);
    restored.policy_provenance =
        policy ? policy->policy_provenance : std::vector<std::uint8_t>{};
    ++stats_->snapshot_restores;
  }
};

class InPassRunner final : public SnapshotRunner {
public:
  using SnapshotRunner::SnapshotRunner;
  bool fail_after_prefill{false};
  TextRunnerDescriptor Descriptor() const override {
    auto descriptor = SnapshotRunner::Descriptor();
    descriptor.capabilities.in_pass_checkpoint = true;
    return descriptor;
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    if (boundary <= offset ||
        boundary >= std::min(prompt.size(), offset + budget))
      return std::nullopt;
    return sizeof(FakeSnapshot);
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint) const override {
    ++stats_->snapshot_captures;
    *checkpoint = std::make_unique<FakeSnapshot>(boundary, 0, 90);
    auto step = FakeRunner::Prefill(state, prompt, offset, budget);
    if (fail_after_prefill) {
      auto& fake = RequireFakeState(state);
      fake.decode_count = 7;
      fake.frontier = 777;
      throw std::runtime_error("injected in-pass failure after state mutation");
    }
    step.checkpoint_ms = 2;
    return step;
  }
};

void TestInPassStableCheckpoint() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<InPassRunner>(stats), 1);
  auto first = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  const auto step = first.Prefill(64);
  Expect(step.consumed_tokens == 5 && step.decode_ready,
         "in-pass checkpoint completes framing in the same forward");
  Expect(stats->snapshot_captures == 1 &&
             stats->prefill_spans == std::vector<std::size_t>{5},
         "one prefill captures the stable boundary before mutation");
  const auto commit = first.Commit();
  Expect(commit.snapshot_ms >= 2,
         "in-pass capture contributes to snapshot phase timing");
  auto second = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(second.cached_prompt_tokens() == 3,
         "rewritten framing restores the in-pass boundary");
  Expect(second.Prefill(64).decode_ready,
         "warm rewritten turn captures its next boundary in one pass");
  second.Commit();
  auto third = pool.Acquire({1, 2, 3, 50, 51, 7, 60, 61}, {}, {}, {}, true, 6);
  Expect(third.cached_prompt_tokens() == 6,
         "the stable checkpoint advances on warm turns");
  third.Invalidate();
}

void TestInPassFailureRetainsOnlyCompletedCheckpoints() {
  auto stats = std::make_shared<FakeStats>();
  // Both the old fallback and the next stable boundary fit, with no spare
  // reservation. A leaked reservation would force the old fallback out.
  auto runner =
      std::make_shared<InPassRunner>(stats, 64, 256, 2 * sizeof(FakeSnapshot));
  TextRunnerPool pool(runner, 1);
  auto seed = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(seed.Prefill(64).decode_ready, "seed reaches its decode frontier");
  seed.Cancel();  // Retain the stable checkpoint without a full-prompt copy.

  auto failed = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(failed.cached_prompt_tokens() == 3,
         "rewritten framing starts from the immutable fallback");
  runner->fail_after_prefill = true;
  bool saw_failure = false;
  try {
    (void)failed.Prefill(64);
  } catch (const std::runtime_error& error) {
    saw_failure = std::string_view(error.what()) ==
                  "injected in-pass failure after state mutation";
  }
  Expect(saw_failure, "in-pass prefill fails after changing the fake state");
  failed.Cancel();
  runner->fail_after_prefill = false;

  auto retry = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(retry.cached_prompt_tokens() == 3 &&
             retry.cache_restore_bytes() == sizeof(FakeSnapshot),
         "cancel restores the immutable fallback, not the failed frontier");
  const auto step = retry.Prefill(64);
  Expect(step.consumed_tokens == 5 && step.decode_ready,
         "retry recomputes all work from the failed in-pass operation");
  Expect(retry.SelectNext().token == 90,
         "retry does not inherit the failed operation's decode state");
  retry.Advance();
  retry.Commit();

  auto old_branch = pool.Acquire({1, 2, 3, 70, 71}, {}, {}, {}, true, 3);
  Expect(old_branch.cached_prompt_tokens() == 3 &&
             old_branch.cache_restore_bytes() == sizeof(FakeSnapshot),
         "released reservation lets the old fallback survive the healthy turn");
  old_branch.Invalidate();
  auto new_branch =
      pool.Acquire({1, 2, 3, 50, 51, 7, 60, 61}, {}, {}, {}, true, 6);
  Expect(new_branch.cached_prompt_tokens() == 6 &&
             new_branch.cache_restore_bytes() == sizeof(FakeSnapshot),
         "healthy retry also retains its newly completed stable boundary");
  new_branch.Invalidate();
}

class PersistentSnapshotRunner : public SnapshotRunner {
public:
  std::function<void()> before_serialize;
  PersistentSnapshotRunner(std::shared_ptr<FakeStats> stats,
                           std::string identity,
                           std::size_t retained_snapshot_capacity_bytes = 256,
                           std::size_t max_context = 64)
      : SnapshotRunner(std::move(stats), 64, 256,
                       retained_snapshot_capacity_bytes),
        identity_(identity.begin(), identity.end()),
        max_context_(max_context) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = SnapshotRunner::Descriptor();
    descriptor.max_context = max_context_;
    descriptor.persistence = gufo::server::TextRunnerPersistenceDescriptor{
        .compatibility_identity = identity_,
        .payload_version = 1,
    };
    return descriptor;
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    (void)RequireSnapshot(snapshot);
    return 4 * sizeof(std::uint64_t);
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    if (before_serialize)
      before_serialize();
    const auto& saved = RequireSnapshot(snapshot);
    if (destination.size() != 4 * sizeof(std::uint64_t)) {
      throw std::invalid_argument("fake persistent payload size mismatch");
    }
    const std::array<std::uint64_t, 4> fields = {
        saved.position,
        saved.decode_count,
        saved.frontier.has_value() ? 1U : 0U,
        saved.frontier.value_or(0),
    };
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        destination[field * sizeof(std::uint64_t) + byte] =
            static_cast<std::uint8_t>(fields[field] >> (byte * 8U));
      }
    }
    return destination.size();
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != 4 * sizeof(std::uint64_t)) {
      throw std::invalid_argument("fake persistent payload size mismatch");
    }
    std::array<std::uint64_t, 4> fields{};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        fields[field] |= static_cast<std::uint64_t>(
                             payload[field * sizeof(std::uint64_t) + byte])
                         << (byte * 8U);
      }
    }
    auto& restored = RequireFakeState(state);
    restored.position = fields[0];
    restored.decode_count = fields[1];
    restored.frontier = fields[2] != 0
                            ? std::optional<TextRunnerToken>(
                                  static_cast<TextRunnerToken>(fields[3]))
                            : std::nullopt;
    ++stats_->snapshot_restores;
  }

private:
  static const FakeSnapshot& RequireSnapshot(
      const TextRunnerSnapshot& snapshot) {
    const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
    if (fake == nullptr) {
      throw std::invalid_argument("snapshot type mismatch");
    }
    return *fake;
  }

  std::vector<std::uint8_t> identity_;
  std::size_t max_context_;
};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "gufo-runner-disk-cache-XXXXXX")
                              .string();
    const char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create runner cache directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

class PersistentInPassRunner final : public PersistentSnapshotRunner {
public:
  using PersistentSnapshotRunner::PersistentSnapshotRunner;
  bool capable{true};
  bool refuse_claim{false};
  mutable std::size_t in_pass_calls{0};
  mutable std::size_t copying_calls{0};

  TextRunnerDescriptor Descriptor() const override {
    auto descriptor = PersistentSnapshotRunner::Descriptor();
    descriptor.capabilities.in_pass_checkpoint = capable;
    return descriptor;
  }

  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto end = std::min(prompt.size(), offset + budget);
    if (refuse_claim || boundary <= offset + 1 || boundary >= end ||
        end - boundary > 8)
      return std::nullopt;
    return sizeof(FakeSnapshot);
  }

  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint) const override {
    ++in_pass_calls;
    *checkpoint = std::make_unique<FakeSnapshot>(boundary, 0, 90);
    auto step = FakeRunner::Prefill(state, prompt, offset, budget);
    step.checkpoint_ms = 2;
    return step;
  }

  std::unique_ptr<TextRunnerSnapshot> SnapshotForPersistence(
      const TextRunnerState& state) const override {
    ++copying_calls;
    return SnapshotRunner::Snapshot(state);
  }
};

bool DiskInPassExpected() {
  const char* value = std::getenv("GUFO_DISK_INPASS");
  return value == nullptr || std::string_view(value) != "0";
}

void TestDiskDeferRequestLifetime() {
  const char* value = std::getenv("GUFO_DISK_DEFER");
  const bool deferred = value && std::string_view(value) == "1";
  for (const int finish : {0, 1, 2, 3}) {
    TemporaryDirectory directory;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PersistentInPassRunner>(stats, "disk-defer");
    std::binary_semaphore entered(0);
    runner->before_serialize = [&] { entered.release(); };
    {
      TextRunnerPool pool(
          runner, 1,
          TextRunnerDiskCacheOptions{.directory = directory.path(),
                                     .capacity_bytes = 4096,
                                     .staging_capacity_bytes = 4096});
      {
        auto request = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
        while (!request.Prefill(64).decode_ready) {
        }
        if (deferred)
          Expect(!entered.try_acquire_for(std::chrono::milliseconds(100)),
                 "resident request gates its in-pass payload before decode");
        else
          Expect(entered.try_acquire_for(std::chrono::seconds(2)),
                 "default worker serializes during resident generation");
        (void)request.SelectNext();
        request.Advance();
        if (deferred)
          Expect(!entered.try_acquire_for(std::chrono::milliseconds(100)),
                 "request stays active through decode and between steps");
        // Unique Impl ownership follows Request moves; all exits release it.
        auto moved = std::move(request);
        if (finish == 0)
          (void)moved.Commit();
        else if (finish == 1)
          (void)moved.Cancel();
        else if (finish == 2)
          moved.Invalidate();
        // finish == 3 tests ordinary destruction with queued storage retained.
      }
      if (deferred)
        Expect(entered.try_acquire_for(std::chrono::seconds(2)),
               "commit/cancel/invalidate/destruction release the worker gate");
    }  // Store drains after the request has released its activity token.
    gufo::server::ContinuationDiskStore restarted(
        gufo::server::ContinuationDiskStoreOptions{
            .directory = directory.path(),
            .capacity_bytes = 4096,
            .staging_capacity_bytes = 4096});
    auto state = runner->CreateState();
    const std::vector<TextRunnerToken> prompt{1, 2, 3, 50, 51};
    Expect(
        restarted.RestoreLongestPrefix(*runner, *state, prompt).token_count ==
            3,
        "deferred snapshot survives live append and request teardown");
  }
}

std::vector<std::vector<std::uint8_t>> DiskImages(
    const std::filesystem::path& directory) {
  std::vector<std::vector<std::uint8_t>> images;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".kvc")
      continue;
    std::ifstream input(entry.path(), std::ios::binary);
    Expect(input.good(), "persisted checkpoint is readable");
    images.emplace_back(std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>());
  }
  std::ranges::sort(images);
  return images;
}

void TestDiskInPassPayloadAndFallbacks() {
  const bool enabled = DiskInPassExpected();
  const auto replay = [&](bool capable, bool refuse_claim, bool refuse_ram,
                          bool delayed, bool spaced = false) {
    TemporaryDirectory directory;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PersistentInPassRunner>(
        stats, "disk-inpass", refuse_ram ? sizeof(FakeSnapshot) - 1 : 256);
    runner->capable = capable;
    runner->refuse_claim = refuse_claim;
    std::binary_semaphore entered(0), release(0);
    bool first_write = true;  // Accessed only by the single disk worker.
    if (delayed) {
      runner->before_serialize = [&] {
        if (std::exchange(first_write, false)) {
          entered.release();
          release.acquire();
        }
      };
    }
    {
      TextRunnerPool pool(
          runner, 1,
          TextRunnerDiskCacheOptions{
              .directory = directory.path(),
              .capacity_bytes = 4096,
              .staging_capacity_bytes = 4096,
              .min_checkpoint_step_tokens = spaced ? 2048U : 0U});
      auto request = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
      while (!request.Prefill(64).decode_ready) {
      }
      // GUFO_DISK_DEFER=1 deliberately holds the worker back while the request generates, so the immediate-entry check only applies to the default mode.
      const char* defer_env = std::getenv("GUFO_DISK_DEFER");
      if (delayed && !(defer_env && std::string_view(defer_env) == "1"))
        Expect(entered.try_acquire_for(std::chrono::seconds(2)),
               "captured boundary reaches disk worker before live append");
      (void)request.SelectNext();
      request.Advance();
      const auto commit = request.Commit();
      Expect(commit.disk_queued_bytes != 0,
             "every admitted capture still queues disk persistence");
      const bool in_pass = enabled && capable && !refuse_claim && !refuse_ram;
      Expect(runner->in_pass_calls == (in_pass ? 1U : 0U),
             "disk capture honors capability, switch and reservation");
      Expect(stats->prefill_spans == (in_pass ? std::vector<std::size_t>{5}
                                              : std::vector<std::size_t>{3, 2}),
             "refusal/off uses the old split; admission uses one pass");
      Expect(runner->copying_calls == (in_pass ? 0U : 2U),
             "in-pass and its complete prompt use ordinary RAM snapshots");
      if (in_pass)
        Expect(commit.snapshot_ms >= 2, "in-pass capture timing is reported");
      if (delayed) {
        auto rewound = pool.Acquire({1, 2, 3, 50, 51}, {}, {}, {}, true, 3);
        Expect(rewound.cached_prompt_tokens() == 3 &&
                   stats->snapshot_restores != 0,
               "live session rewinds after appending while disk write waits");
        rewound.Invalidate();
        release.release();
      }
    }  // Drain disk work before comparing bytes.
    const auto images = DiskImages(directory.path());
    Expect(
        images.size() == (spaced ? 1U : 2U),
        "stable persists and complete prompt obeys the existing spacing gate");
    // A fresh runner reads the unchanged store and the stable boundary.
    auto reader_stats = std::make_shared<FakeStats>();
    TextRunnerPool restarted(
        std::make_shared<PersistentSnapshotRunner>(reader_stats, "disk-inpass"),
        1,
        TextRunnerDiskCacheOptions{.directory = directory.path(),
                                   .capacity_bytes = 4096,
                                   .staging_capacity_bytes = 4096});
    auto restored = restarted.Acquire({1, 2, 3, 50, 51}, {}, {}, {}, true, 3);
    Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
           "persisted in-pass boundary restores after restart");
    Expect(
        restored.Prefill(64).decode_ready && restored.SelectNext().token == 90,
        "persisted fake frontier reproduces its first token");
    restored.Invalidate();
    return images;
  };
  const auto copying = replay(false, false, false, false);
  Expect(replay(true, false, false, false) == copying,
         "in-pass and copying persist byte-identical store images");
  Expect(replay(true, true, false, false) == copying,
         "engine claim refusal preserves persisted bytes");
  Expect(replay(true, false, true, false) == copying,
         "RAM reservation failure falls back with identical persisted bytes");
  Expect(replay(true, false, false, true) == copying,
         "disk worker retains exact capture through live append and rewind");
  Expect(replay(true, false, false, false, true) ==
             replay(false, false, false, false, true),
         "in-pass capture preserves default min-step persistence policy");
}

void TestDiskInPassCoincidentAlignedLearnedBoundary() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentInPassRunner>(
      stats, "aligned-inpass", 256, 1024);
  std::vector<TextRunnerToken> stored(264, 7), prompt(stored);
  stored[260] = 80;
  prompt[260] = 81;
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "", 1);
  {
    gufo::server::ContinuationDiskStore seed({.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 8192});
    auto state = runner->CreateState();
    (void)runner->Prefill(*state, stored, 0, 1024);
    Expect(seed.Save(*runner, stored, *runner->Snapshot(*state)).stored,
           "seed provides a learned disk divergence");
    Expect(seed.SharedPrefixBoundaries(*runner, prompt, 128, 4) ==
               std::vector<std::size_t>{256},
           "260-token divergence still rounds DOWN to 256");
  }
  stats->prefill_spans.clear();
  {
    TextRunnerPool pool(
        runner, 1,
        TextRunnerDiskCacheOptions{.directory = directory.path(),
                                   .capacity_bytes = 8192,
                                   .staging_capacity_bytes = 8192,
                                   .min_checkpoint_step_tokens = 0});
    auto request = pool.Acquire(prompt, {}, {}, {}, true, 256);
    while (!request.Prefill(1024).decode_ready) {
    }
    Expect(runner->in_pass_calls == (DiskInPassExpected() ? 1U : 0U),
           "coincident aligned learned/stable boundary shares in-pass capture");
    request.Commit();
  }
  gufo::server::ContinuationDiskStore reader({.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 8192});
  prompt[256] = 99;
  auto state = runner->CreateState();
  const auto restored = reader.RestoreLongestPrefix(*runner, *state, prompt);
  Expect(restored.restored && restored.token_count == 256,
         "aligned learned boundary persists at exactly 256 tokens");
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "0", 1);
}

void TestDiskInPassKeepsDistinctLearnedBoundaries() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentInPassRunner>(stats, "two-boundaries");
  const std::vector<TextRunnerToken> prompt{1, 2, 3, 40, 41};
  {
    gufo::server::ContinuationDiskStore seed({.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 8192});
    for (const auto& tokens : {std::vector<TextRunnerToken>{1, 2, 3, 50, 51},
                               std::vector<TextRunnerToken>{1, 2, 3, 40, 60}}) {
      auto state = runner->CreateState();
      (void)runner->Prefill(*state, tokens, 0, 64);
      Expect(seed.Save(*runner, tokens, *runner->Snapshot(*state)).stored,
             "seed stores both divergence histories");
    }
    Expect(seed.SharedPrefixBoundaries(*runner, prompt, 1, 4) ==
               std::vector<std::size_t>({3, 4}),
           "two learned boundaries include a coincident stable target");
  }
  stats->prefill_spans.clear();
  {
    TextRunnerPool pool(
        runner, 1,
        TextRunnerDiskCacheOptions{.directory = directory.path(),
                                   .capacity_bytes = 8192,
                                   .staging_capacity_bytes = 8192,
                                   .min_checkpoint_step_tokens = 0,
                                   .shared_prefix_min_tokens = 1});
    auto request = pool.Acquire(prompt, {}, {}, {}, true, 3);
    while (!request.Prefill(64).decode_ready) {
    }
    Expect(runner->in_pass_calls == 0 &&
               stats->prefill_spans == std::vector<std::size_t>({3, 1, 1}),
           "multiple required boundaries keep every original split");
    request.Commit();
  }
  Expect(DiskImages(directory.path()).size() == 5,
         "stable, distinct learned and complete prompt all still persist");
}

void TestSnapshotForkAndUnsupportedCapabilities() {
  auto stats = std::make_shared<FakeStats>();
  SnapshotRunner runner(stats);
  auto state = runner.CreateState();
  RequireFakeState(*state).position = 7;
  RequireFakeState(*state).frontier = 90;
  auto snapshot = runner.Snapshot(*state);
  Expect(
      snapshot != nullptr && snapshot->PayloadBytes() == sizeof(FakeSnapshot),
      "snapshot reports its opaque payload bytes");
  auto fork = runner.CreateState();
  runner.RestoreOrFork(*fork, *snapshot);
  Expect(RequireFakeState(*fork).position == 7 &&
             RequireFakeState(*fork).frontier == 90,
         "fork restores the exact runner-owned boundary");

  FakeRunner unsupported(stats);
  bool snapshot_rejected = false;
  try {
    (void)unsupported.Snapshot(*state);
  } catch (const std::logic_error&) {
    snapshot_rejected = true;
  }
  Expect(snapshot_rejected, "unsupported snapshots fail explicitly");
}

void TestSnapshotCacheBranchesOnePrefixIntoIndependentStates() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 2);

  {
    auto root = pool.Acquire({1, 2, 3, 4});
    Expect(root.Prefill(4).decode_ready,
           "root prefix reaches its snapshot boundary");
    const auto commit = root.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot) &&
               commit.snapshot_ms >= 0.0,
           "root commit reports retained full-copy snapshot cost");
  }

  auto first = pool.Acquire({1, 2, 3, 4, 5});
  auto second = pool.Acquire({1, 2, 3, 4, 6});
  Expect(first.cache_hit() && second.cache_hit(),
         "two simultaneous requests restore one retained snapshot");
  Expect(
      first.cached_prompt_tokens() == 4 && second.cached_prompt_tokens() == 4,
      "both branches report the same immutable root prefix");
  Expect(
      first.cache_restore_bytes() == 0 &&
          second.cache_restore_bytes() == sizeof(FakeSnapshot) &&
          first.cache_restore_ms() >= 0.0 && second.cache_restore_ms() >= 0.0,
      "one branch uses the live frontier and the other restores the snapshot");
  Expect(stats->snapshot_restores == 1,
         "only the simultaneous branch needs a snapshot copy");
  Expect(stats->states_created == 2,
         "snapshot branching reuses preallocated request states");
  Expect(stats->snapshot_size_queries == 1 && stats->snapshot_captures == 1,
         "root snapshot reserves its exact payload before capture");

  Expect(first.Prefill(1).decode_ready && second.Prefill(1).decode_ready,
         "each branch prefills only its divergent suffix");
  Expect(first.SelectNext().token == 90 && second.SelectNext().token == 90,
         "both restored branches retain an exact frontier");
  first.Advance();
  second.Advance();
  first.Commit();
  second.Commit();

  auto third = pool.Acquire({1, 2, 3, 4, 7});
  Expect(third.cache_hit() && third.cached_prompt_tokens() == 4,
         "branch commits preserve the shared root while capacity permits");
  third.Invalidate();
}

void TestSnapshotRetentionUsesPromptBoundary() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto root = pool.Acquire({1, 2, 3, 4});
    Expect(root.Prefill(4).decode_ready,
           "root prompt reaches its decode frontier");
    Expect(stats->snapshot_captures == 0,
           "prefill returns before optional snapshot capture");
    Expect(root.SelectNext().token == 90,
           "root selects its prompt-boundary frontier");
    root.Advance();
    Expect(stats->snapshot_captures == 1,
           "snapshot is captured before generated tokens mutate the state");
    Expect(root.SelectNext().token == 91,
           "root advances beyond the reusable prompt boundary");
    root.Advance();
    const auto commit = root.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot),
           "root commit publishes the held prompt snapshot");
  }

  {
    auto continuation = pool.Acquire({1, 2, 3, 4, 90, 91, 7});
    Expect(continuation.cache_hit() &&
               continuation.cached_prompt_tokens() == 6 &&
               continuation.cache_restore_bytes() == 0,
           "conversation reuses all executed assistant tokens without "
           "restoration");
    continuation.Invalidate();
  }

  {
    auto repeated = pool.Acquire({1, 2, 3, 4});
    Expect(repeated.cache_hit() && repeated.cached_prompt_tokens() == 4,
           "an identical prompt restores the prompt-boundary snapshot");
    Expect(repeated.prefill_complete(),
           "an exact snapshot hit retains its decode frontier");
    Expect(stats->snapshot_captures == 1,
           "an exact in-memory hit avoids another snapshot copy");
    Expect(repeated.SelectNext().token == 90,
           "exact reuse starts from the prompt rather than post-generation");
    repeated.Invalidate();
  }

  {
    auto extension = pool.Acquire({1, 2, 3, 4, 7, 8});
    Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 4,
           "a prompt that omits generated tokens still reuses the root");
    const auto suffix = extension.Prefill(8);
    Expect(suffix.consumed_tokens == 2 && suffix.decode_ready,
           "only the extension after the stable prompt is prefetched");
    extension.CapturePromptSnapshot();
    Expect(stats->snapshot_captures == 2,
           "the extended prompt captures its own reusable boundary");
    Expect(extension.SelectNext().token == 90,
           "extended reuse preserves the rebuilt decode frontier");
    extension.Invalidate();
  }
}

void TestBehaviourBudgetAndPhaseDoNotLeakThroughPromptSnapshot() {
  // sampling.cpp:447-453 resets request policy state; text_model_runner.cpp:571
  // creates a fresh sampler even when the model frontier comes from a snapshot.
  class PolicySnapshotRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;
    [[nodiscard]] TextDecodeSelection SelectNext(
        TextRunnerState& state,
        gufo::sampling::SamplerState& sampler) const override {
      if (!RequireFakeState(state).frontier)
        throw std::logic_error("policy test needs a restored decode frontier");
      const std::array<float, 3> row{10.0F, 9.0F, 8.0F};
      const auto token = sampler.Sample(row);
      return {.stop = token == 0,
              .token = token,
              .piece = token == 1 ? "word" : token == 2 ? "</think>" : ""};
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PolicySnapshotRunner>(stats);
  TextRunnerPool pool(runner, 1);
  auto policy = std::make_shared<gufo::sampling::BehaviourConfig>();
  policy->initial_thinking = true;
  policy->think_eog_guard = true;
  policy->reasoning_budget_enabled = true;
  policy->reasoning_budget = 1;
  policy->eog_ids = {0};
  policy->think_end = {2};
  // Production preparation appends the encoded think-end to the forced budget
  // sequence.
  policy->budget_forced = {2};
  policy->token_pieces = std::make_shared<const std::vector<std::string>>(
      std::vector<std::string>{"", "word", "</think>"});
  gufo::sampling::SamplingConfig config;
  config.behaviour = policy;
  const std::vector<TextRunnerToken> prompt{1, 1, 1, 1};
  {
    auto first = pool.Acquire(prompt, config);
    Expect(!first.cache_hit() && first.Prefill(4).decode_ready,
           "first policy request creates its prompt frontier");
    const auto word = first.SelectNext();
    Expect(!word.stop && word.token == 1,
           "thinking guard suppresses the higher-logit EOG");
    first.Advance();
    const auto end = first.SelectNext();
    Expect(!end.stop && end.token == 2,
           "one accepted word consumes the request reasoning budget");
    first.Advance();
    Expect(first.Commit().snapshot_bytes == sizeof(FakeSnapshot),
           "first request retains an immutable prompt-boundary snapshot");
  }
  const auto previous_restores = stats->snapshot_restores;
  auto repeated = pool.Acquire(prompt, config);
  Expect(repeated.cache_hit() &&
             repeated.cached_prompt_tokens() == prompt.size() &&
             repeated.cache_restore_bytes() == sizeof(FakeSnapshot) &&
             repeated.prefill_complete() &&
             stats->snapshot_restores == previous_restores + 1,
         "identical prompt restores the immutable frontier without prefill");
  const auto word = repeated.SelectNext();
  Expect(!word.stop && word.token == 1,
         "restored model state inherits neither closed thinking nor spent budget");
  repeated.Advance();
  const auto end = repeated.SelectNext();
  Expect(!end.stop && end.token == 2,
         "restored request receives its own complete one-token reasoning budget");
  repeated.Advance();
  repeated.Invalidate();
}

void TestGeneratedFrontierForksBeforeMutation() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 2);
  {
    auto root = pool.Acquire({1, 2, 3, 4});
    root.Prefill(4);
    Expect(root.SelectNext().token == 90, "generated root token");
    root.Advance();
    root.Commit();
  }
  auto first = pool.Acquire({1, 2, 3, 4, 90, 5});
  auto second = pool.Acquire({1, 2, 3, 4, 90, 6});
  Expect(
      first.cached_prompt_tokens() == 5 && second.cached_prompt_tokens() == 5,
      "concurrent forks must reuse the generated frontier, not re-prefill "
      "its output");
  Expect(first.cache_restore_bytes() == 0 &&
             second.cache_restore_bytes() == sizeof(FakeSnapshot),
         "one live frontier is frozen for its peer");
  Expect(stats->snapshot_captures == 2 && stats->snapshot_restores == 1,
         "the generated frontier is copied once while the prompt is retained");
  auto blocked = pool.Acquire({9}, [] { return true; });
  Expect(!blocked, "publishing a snapshot must not release either live lease");
  Expect(first.Prefill(64).consumed_tokens == 1 &&
             second.Prefill(64).consumed_tokens == 1,
         "neither fork puts generated history into a prefill chunk");
  first.Invalidate();
  second.Invalidate();
  auto root_branch = pool.Acquire({1, 2, 3, 4, 7});
  Expect(root_branch.cached_prompt_tokens() == 4,
         "freezing a generated frontier preserves the original prompt branch");
}

void TestGeneratedFrontierPersistsForForks() {
  // Persist every frontier; spacing is checked below and by the disk store.
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk{.directory = directory.path(),
                                        .capacity_bytes = 4096,
                                        .staging_capacity_bytes = 4096,
                                        .min_checkpoint_step_tokens = 0};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool pool(runner, 2, disk);
    auto root = pool.Acquire({1, 2, 3});
    root.Prefill(3);
    (void)root.SelectNext();
    root.Advance();
    root.Commit();
    auto continuation = pool.Acquire({1, 2, 3, 90, 4});
    Expect(continuation.cached_prompt_tokens() == 4,
           "the live frontier includes generated output");
    continuation.Invalidate();
  }
  TextRunnerPool restarted(runner, 1, disk);
  auto restored = restarted.Acquire({1, 2, 3, 90, 5});
  Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 4,
         "disk forks restore the generated checkpoint before suffix prefill");
  Expect(restored.Prefill(16).consumed_tokens == 1,
         "disk restoration does not re-prefill known generated tokens");

  // With the default step, the generated frontier one token past the prompt
  // is not written; a restart resumes from the prompt and re-prefills it.
  TemporaryDirectory spaced_directory;
  const TextRunnerDiskCacheOptions spaced{.directory = spaced_directory.path(),
                                          .capacity_bytes = 4096,
                                          .staging_capacity_bytes = 4096};
  {
    TextRunnerPool pool(runner, 2, spaced);
    auto root = pool.Acquire({1, 2, 3});
    root.Prefill(3);
    (void)root.SelectNext();
    root.Advance();
    root.Commit();
  }
  TextRunnerPool spaced_restart(runner, 1, spaced);
  auto resumed = spaced_restart.Acquire({1, 2, 3, 90, 5});
  Expect(resumed.cache_disk_hit() && resumed.cached_prompt_tokens() == 3 &&
             resumed.Prefill(16).consumed_tokens == 2,
         "spaced disk checkpoints resume from the nearest stored prefix");
}

void TestCancellationRetainsOnlyCompletedWork() {
  for (const bool pending : {false, true}) {
    auto stats = std::make_shared<FakeStats>();
    TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
    auto request = pool.Acquire({1, 2});
    request.Prefill(2);
    Expect(request.SelectNext().token == 90, "first cancellation token");
    request.Advance();
    if (pending)
      Expect(request.SelectNext().token == 91, "selected but unexecuted token");
    request.Cancel();
    Expect(
        stats->advanced_tokens.size() == 1 && stats->snapshot_captures == 1,
        "cancellation neither advances a pending token nor starts a snapshot");
    auto continuation = pool.Acquire({1, 2, 90, 91, 7});
    Expect(continuation.cached_prompt_tokens() == 3 &&
               continuation.cache_restore_bytes() == 0,
           "cancelled conversation retains the exact completed live frontier");
    continuation.Invalidate();
    auto branch = pool.Acquire({1, 2, 8});
    Expect(branch.cached_prompt_tokens() == 2,
           "the original prompt remains available for branching after "
           "cancellation");
    branch.Invalidate();
  }

  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  auto request = pool.Acquire({1, 2});
  request.Prefill(2);
  (void)request.SelectNext();
  stats->fail_after_advance = true;
  try {
    request.Advance();
    Expect(false, "mutating failure must throw");
  } catch (const std::runtime_error&) {
  }
  request.Cancel();
  auto continuation = pool.Acquire({1, 2, 90, 7});
  Expect(
      continuation.cached_prompt_tokens() == 2 &&
          continuation.cache_restore_bytes() > 0,
      "a failed operation restores the immutable prompt, never mutated state");
  continuation.Invalidate();
}

void TestPersistentSnapshotRestoresAcrossPools() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{
      .directory = directory.path(),
      .capacity_bytes = 4096,
      .staging_capacity_bytes = 4096,
  };

  {
    auto writer_stats = std::make_shared<FakeStats>();
    auto writer = std::make_shared<PersistentSnapshotRunner>(
        writer_stats, "artifact-A", sizeof(FakeSnapshot) - 1);
    TextRunnerPool pool(writer, 1, disk_cache);
    auto request = pool.Acquire({1, 2, 3});
    Expect(request.Prefill(3).decode_ready,
           "writer reaches persistent checkpoint");
    const auto commit = request.Commit();
    Expect(commit.snapshot_bytes == 0 && commit.disk_queued_bytes > 0 &&
               commit.disk_enqueue_ms >= 0.0,
           "disk publication is independent of RAM snapshot admission");
  }

  auto reader_stats = std::make_shared<FakeStats>();
  auto reader =
      std::make_shared<PersistentSnapshotRunner>(reader_stats, "artifact-A");
  TextRunnerPool restarted(reader, 1, disk_cache);
  {
    auto cold = restarted.Acquire({1, 2, 3, 4, 5}, {}, {}, {}, false);
    Expect(!cold.cache_hit() && !cold.cache_disk_hit() &&
               cold.cached_prompt_tokens() == 0 &&
               reader_stats->snapshot_restores == 0,
           "cache_prompt=false bypasses disk restoration");
    cold.Invalidate();
  }
  auto extension = restarted.Acquire({1, 2, 3, 4, 5});
  Expect(extension.cache_hit() && extension.cache_disk_hit(),
         "clean pool restores compatible prefix from disk");
  Expect(extension.cached_prompt_tokens() == 3 &&
             extension.cache_restore_bytes() > 0 &&
             extension.cache_restore_ms() >= 0.0,
         "disk restore reports exact prefix and actual read metrics");
  Expect(extension.Prefill(8).consumed_tokens == 2,
         "restarted request prefills only its unmatched suffix");
  extension.Invalidate();

  auto incompatible_stats = std::make_shared<FakeStats>();
  auto incompatible = std::make_shared<PersistentSnapshotRunner>(
      incompatible_stats, "artifact-B");
  TextRunnerPool incompatible_pool(incompatible, 1, disk_cache);
  auto miss = incompatible_pool.Acquire({1, 2, 3, 4});
  Expect(!miss.cache_hit() && !miss.cache_disk_hit(),
         "changed compatibility identity is a cold miss");
  miss.Invalidate();
}

void TestPromptReuseCanBeDisabledPerRequest() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  {
    auto initial = pool.Acquire({1, 2});
    initial.Prefill(2);
    (void)initial.SelectNext();
    initial.Advance();
    initial.Commit();
  }
  const auto restores = stats->snapshot_restores;
  auto cold = pool.Acquire({1, 2, 90, 7}, {}, {}, {}, false);
  Expect(!cold.cache_hit() && cold.cached_prompt_tokens() == 0 &&
             stats->snapshot_restores == restores,
         "cache_prompt=false bypasses both live and immutable RAM frontiers");
  Expect(cold.Prefill(8).consumed_tokens == 4,
         "disabled reuse processes every prompt token");
  cold.Commit();
  auto retained = pool.Acquire({1, 2, 90, 7, 8});
  Expect(retained.cached_prompt_tokens() == 4,
         "a no-reuse request can populate the cache for later requests");
  retained.Invalidate();
}

void TestStableChatPrefixSurvivesInterruptedFraming() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 4096};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    // {1,2,3} is stable history; {40,41} opens assistant reasoning.
    auto initial = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    auto step = initial.Prefill(64);
    Expect(step.consumed_tokens == 3 && !step.decode_ready,
           "prefill stops before mutable assistant framing");
    Expect(initial.Prefill(64).decode_ready,
           "assistant suffix completes prefill");
    (void)initial.SelectNext();
    initial.Advance();
    initial.Cancel();
    // Client omits the unfinished reasoning and closes the assistant turn.
    auto resumed = pool.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
    Expect(resumed.cached_prompt_tokens() == 3,
           "changed assistant framing retains all stable prompt tokens");
    resumed.Invalidate();
  }
  TextRunnerPool restarted(runner, 1, disk_cache);
  auto restored = restarted.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
         "interrupted stable chat prefix survives server restart");
  restored.Invalidate();

  // Full-prompt entries written by older servers must not prevent creating
  // the shorter, stable checkpoint on the next request.
  TextRunnerPool legacy(runner, 1);
  auto old = legacy.Acquire({1, 2, 3, 40, 41});
  old.Prefill(64);
  old.Commit();
  auto migrated = legacy.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!migrated.cache_hit(),
         "legacy mutable-suffix snapshot cannot bypass the stable boundary");
  Expect(migrated.Prefill(64).consumed_tokens == 3,
         "legacy entry is replaced by a stable chat checkpoint");
  migrated.Cancel();
  auto next = legacy.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(next.cached_prompt_tokens() == 3,
         "migrated checkpoint survives changed assistant framing");
  next.Invalidate();
}

void TestWarmChatCheckpointsStopAtTheStableBoundary() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  Expect(pool.capacity() == 1 && stats->states_created == 1,
         "extra checkpoint entries do not allocate execution sessions");
  auto root = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(root.Prefill(64).consumed_tokens == 3,
         "cold chat retains the stable fallback before assistant framing");
  Expect(root.Prefill(64).decode_ready, "cold chat completes the suffix");
  const auto first = root.SelectNext().token;
  root.Advance();
  root.Commit();

  auto continuation =
      pool.Acquire({1, 2, 3, 40, 41, first, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(continuation.cached_prompt_tokens() == 6 &&
             continuation.cache_restore_bytes() == 0,
         "retained reasoning reuses generated tokens without a restore");
  // The warm turn stops once at its own stable boundary so that boundary is
  // checkpointed. Without that stop the conversation keeps falling back to
  // this turn's frontier for every later turn that rewrites the assistant.
  const auto step = continuation.Prefill(64);
  Expect(step.consumed_tokens == 1 && !step.decode_ready,
         "warm prefill stops at the stable boundary before assistant framing");
  const auto framing = continuation.Prefill(64);
  Expect(framing.consumed_tokens == 2 && framing.decode_ready,
         "assistant framing completes in the following pass");
  const auto second = continuation.SelectNext().token;
  continuation.Advance();
  continuation.Commit();

  auto repeated =
      pool.Acquire({1, 2, 3, 40, 41, first, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(repeated.cached_prompt_tokens() == 9 && repeated.prefill_complete(),
         "identical thinking prompt reuses the full prompt checkpoint");
  Expect(repeated.SelectNext().token == second,
         "the full checkpoint preserves greedy selection");
  repeated.Advance();
  repeated.Commit();

  // Empty reasoning changes 40,41 into 50,51 in the replayed assistant.
  auto dropped = pool.Acquire({1, 2, 3, 40, 41, first, 7, 50, 51, 8, 40, 41},
                              {}, {}, {}, true, 10);
  Expect(dropped.cached_prompt_tokens() == 7,
         "omitted reasoning resumes from the previous turn's stable boundary "
         "rather than the frontier before it");
  Expect(!dropped.Prefill(64).decode_ready,
         "the rewritten turn stops at its own boundary in turn");
  Expect(dropped.Prefill(64).decode_ready,
         "omitted reasoning prefills the remaining suffix");
  dropped.Cancel();
}

// This backend records each row's policy in its snapshots. Restoring a row
// retains its original provenance rather than deriving it from the new request.
class ExpertTailRunner : public PersistentSnapshotRunner {
public:
  using Pass = std::pair<std::size_t, bool>;
  ExpertTailRunner(std::shared_ptr<FakeStats> stats, std::size_t tail)
      : PersistentSnapshotRunner(std::move(stats), "expert-tail-policy-v1",
                                 8192, 65536),
        tail(tail) {}
  std::size_t tail;
  bool snapshots{true};
  bool policy_identity{true};
  bool policy_active{true};
  std::size_t chunk{8192};
  mutable std::vector<std::size_t> snapshot_positions;
  mutable std::size_t identity_calls{0};
  mutable const gufo::server::TextPromptContext* model_context{nullptr};
  mutable std::size_t boundary{0};
  mutable std::size_t full_length{0};
  mutable std::vector<Pass> passes;
  mutable std::vector<std::size_t> request_lengths;
  mutable std::vector<std::uint8_t> observed_policy;

  TextRunnerDescriptor Descriptor() const override {
    auto d = PersistentSnapshotRunner::Descriptor();
    d.capabilities.in_pass_checkpoint = snapshots;
    d.capabilities.snapshot = snapshots;
    d.capabilities.fork = snapshots;
    if (!snapshots)
      d.persistence.reset();
    return d;
  }
  std::shared_ptr<const gufo::server::TextPromptContext> CacheIdentity(
      std::size_t length,
      std::shared_ptr<const gufo::server::TextPromptContext> context)
      const override {
    ++identity_calls;
    if (!policy_identity)
      return context;
    const auto tail_text = std::to_string(tail);
    const auto settings = gufo::models::qwen38_flash_next::ParsePrefillExperts(
        policy_active ? "2" : nullptr, policy_active ? "ref10" : nullptr,
        nullptr, nullptr, nullptr, 10, 48, tail_text.c_str());
    return gufo::server::MakePrefillExpertTailCacheIdentity(settings, length,
                                                            std::move(context));
  }
  void SetPromptContext(TextRunnerState&,
                        std::shared_ptr<const gufo::server::TextPromptContext>
                            context) const override {
    model_context = context.get();
  }
  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    FakeRunner::PreparePrefixReuse(state, prefix);
    observed_policy = RequireFakeState(state).policy_provenance;
  }
  std::size_t PrefillCheckpointGrid() const noexcept override { return chunk; }
  // Also compiles against the baseline interface for the golden comparison.
  bool PrefillTailSnapshots() const noexcept { return policy_active && tail; }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    snapshot_positions.push_back(RequireFakeState(state).position);
    return PersistentSnapshotRunner::Snapshot(state);
  }
  std::size_t PreparePrefill(TextRunnerState&,
                             std::size_t length) const override {
    request_lengths.push_back(length);
    full_length = length;
    boundary = policy_active && tail ? length - std::min(length, tail) : length;
    return boundary;
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    const auto n = gufo::npu::PrefillChunk(
        false, budget, prompt.size() - offset, chunk, chunk + 128);
    Expect(!(offset < boundary && offset + n > boundary), "tail crossing");
    auto& provenance = RequireFakeState(state).policy_provenance;
    Expect(provenance.size() == offset, "tail restored policy row count");
    for (std::size_t row = 0; row < offset; ++row)
      Expect(provenance[row] == (!policy_active || tail == 0 || row < boundary),
             "tail restored row policy provenance changed with request length");
    passes.emplace_back(n, !policy_active || tail == 0 || offset < boundary);
    provenance.resize(offset + n,
                      !policy_active || tail == 0 || offset < boundary);
    observed_policy = provenance;
    return FakeRunner::Prefill(state, prompt, offset, n);
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t checkpoint) const override {
    const auto n = gufo::npu::PrefillChunk(false, budget,
                                           prompt.size() - offset, 8320, 8320);
    // The caller must have clipped before asking about in-pass capture.
    Expect(!(offset < boundary && offset + n > boundary),
           "capture crossed tail");
    if (checkpoint <= offset || checkpoint >= offset + n ||
        offset + n - checkpoint > 8)
      return std::nullopt;
    return sizeof(FakeSnapshot);
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t checkpoint,
      std::unique_ptr<TextRunnerSnapshot>* captured) const override {
    auto result = Prefill(state, prompt, offset, budget);
    const auto& provenance = RequireFakeState(state).policy_provenance;
    *captured = std::make_unique<PolicySnapshot>(
        checkpoint, std::vector<std::uint8_t>(provenance.begin(),
                                              provenance.begin() + checkpoint));
    return result;
  }
  std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* saved = dynamic_cast<const PolicySnapshot*>(&snapshot);
    if (!saved || saved->policy_provenance.size() != saved->position)
      throw std::invalid_argument("tail snapshot lacks row provenance");
    return 4 * sizeof(std::uint64_t) + saved->policy_provenance.size();
  }
  std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    if (destination.size() != PersistentSnapshotPayloadBytes(snapshot))
      throw std::invalid_argument("tail persistent payload size mismatch");
    constexpr auto header = 4 * sizeof(std::uint64_t);
    (void)PersistentSnapshotRunner::SerializePersistentSnapshot(
        snapshot, destination.first(header));
    const auto& saved = dynamic_cast<const PolicySnapshot&>(snapshot);
    std::ranges::copy(saved.policy_provenance, destination.begin() + header);
    return destination.size();
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    constexpr auto header = 4 * sizeof(std::uint64_t);
    if (payload.size() < header)
      throw std::invalid_argument("tail persistent payload is truncated");
    PersistentSnapshotRunner::RestorePersistentSnapshot(state,
                                                        payload.first(header));
    auto& restored = RequireFakeState(state);
    if (payload.size() - header != restored.position ||
        !std::ranges::all_of(payload.subspan(header),
                             [](auto value) { return value <= 1; }))
      throw std::invalid_argument("tail persistent provenance is invalid");
    restored.policy_provenance.assign(payload.begin() + header, payload.end());
    observed_policy = restored.policy_provenance;
  }
};

// The logical payload models the review's 0.9 GiB snapshot without allocating
// it. Row provenance and all pool/cache/disk operations remain real CPU code.
class HostFloorTailRunner final : public ExpertTailRunner {
public:
  static constexpr std::size_t kGiB = std::size_t{1} << 30;
  static constexpr std::size_t kPayload = 9 * kGiB / 10;
  using ExpertTailRunner::ExpertTailRunner;
  bool persistence_enabled{true};
  bool capacity_only{false};
  mutable std::vector<std::size_t> materialized_positions;

  TextRunnerDescriptor Descriptor() const override {
    auto d = ExpertTailRunner::Descriptor();
    if (!persistence_enabled)
      d.persistence.reset();
    return d;
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    auto claim = ExpertTailRunner::ResourceClaim();
    claim.retained_snapshot_capacity_bytes =
        capacity_only ? 3 * kPayload : 8 * kPayload;
    claim.retained_snapshot_ceiling_bytes =
        claim.retained_snapshot_capacity_bytes;
    return claim;
  }
  std::size_t SnapshotPayloadBytes(const TextRunnerState&) const override {
    return kPayload;
  }
  std::uint64_t SnapshotHostBytes(std::uint64_t payload,
                                  std::size_t) const override {
    return capacity_only ? 65536 : payload;
  }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    class SizedSnapshot final : public PolicySnapshot {
    public:
      explicit SizedSnapshot(const FakeState& state) : PolicySnapshot(state) {}
      std::size_t PayloadBytes() const noexcept override { return kPayload; }
    };
    snapshot_positions.push_back(RequireFakeState(state).position);
    ++stats_->snapshot_captures;
    return std::make_unique<SizedSnapshot>(RequireFakeState(state));
  }
  std::unique_ptr<TextRunnerSnapshot> SnapshotForPersistence(
      const TextRunnerState& state) const override {
    materialized_positions.push_back(RequireFakeState(state).position);
    return Snapshot(state);
  }
};

class PressureTailRunner : public ExpertTailRunner {
public:
  using ExpertTailRunner::ExpertTailRunner;
  static constexpr std::size_t kPayload = 966367641;
  std::size_t budget{16 * kPayload / 5};
  std::size_t boundary_extra{0};
  int final_failure{0};
  std::size_t payload{kPayload};
  std::size_t bytes_per_token{0};
  std::function<std::size_t(std::size_t)> logical_bytes;
  std::function<void(const TextRunnerSnapshot&, std::size_t)>
      final_payload_hook;
  std::function<void()> final_capture_hook;
  std::function<void(std::size_t)> snapshot_capture_hook;
  mutable std::vector<std::size_t> materialized_positions;

  TextRunnerDescriptor Descriptor() const override {
    auto d = ExpertTailRunner::Descriptor();
    d.max_context = 300000;
    return d;
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    auto claim = ExpertTailRunner::ResourceClaim();
    claim.retained_snapshot_capacity_bytes = budget;
    claim.retained_snapshot_ceiling_bytes = budget;
    return claim;
  }
  std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    if (logical_bytes)
      return logical_bytes(RequireFakeState(state).position);
    return (bytes_per_token ? bytes_per_token * RequireFakeState(state).position
                            : payload) +
           (RequireFakeState(state).position == boundary ? boundary_extra : 0);
  }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    class SizedSnapshot final : public PolicySnapshot {
    public:
      SizedSnapshot(
          const FakeState& state, std::size_t bytes,
          std::function<void(const TextRunnerSnapshot&, std::size_t)> hook)
          : PolicySnapshot(state), bytes(bytes), hook(std::move(hook)) {}
      std::size_t PayloadBytes() const noexcept override {
        if (hook)
          hook(*this, ++reads);
        return bytes;
      }
      std::size_t bytes;
      std::function<void(const TextRunnerSnapshot&, std::size_t)> hook;
      mutable std::atomic<std::size_t> reads{0};
    };
    snapshot_positions.push_back(RequireFakeState(state).position);
    ++stats_->snapshot_captures;
    if (snapshot_capture_hook)
      snapshot_capture_hook(RequireFakeState(state).position);
    if (RequireFakeState(state).position == full_length && final_failure == 1)
      return {};
    if (RequireFakeState(state).position == full_length && final_failure == 2)
      throw std::runtime_error("injected optional final capture failure");
    if (RequireFakeState(state).position == full_length && final_capture_hook)
      final_capture_hook();
    return std::make_unique<SizedSnapshot>(
        RequireFakeState(state),
        SnapshotPayloadBytes(state) -
            (RequireFakeState(state).position == full_length &&
             final_failure == 3),
        RequireFakeState(state).position == full_length
            ? final_payload_hook
            : decltype(final_payload_hook){});
  }
  std::unique_ptr<TextRunnerSnapshot> SnapshotForPersistence(
      const TextRunnerState& state) const override {
    materialized_positions.push_back(RequireFakeState(state).position);
    return Snapshot(state);
  }
};

TextRunnerPool::Request::CommitMetrics FinishPressureRequest(
    TextRunnerPool::Request& request) {
  while (!request.prefill_complete())
    (void)request.Prefill(32768);
  // Exercise the ordinary post-selection prompt capture/response path.
  (void)request.SelectNext();
  (void)request.Advance();
  return request.Commit();
}

// Real geometry and rewritten assistant framing, in one persistent cache.
// Sizes are logical bytes only; saved row-policy provenance is still real.
gufo::models::qwen38_flash_next::PrefillExpertsSettings RealTailSettings(
    std::size_t tail) {
  using namespace gufo::models::qwen38_flash_next;
  PrefillExpertsSettings settings;
  settings.active = tail != 0;
  settings.keep = 3;
  settings.tail = static_cast<std::uint32_t>(tail);
  settings.mode = PrefillExpertsMode::kFold;
  settings.fold_sha256 =
      "cf6970f9c3809becc5a65733bc2f9aab55e19ba27837efa1bc6d4a1c020ddbd5";
  return settings;
}

void TestPostTailStableIdentity() {
  using namespace gufo::server;
  const auto settings = RealTailSettings(256);
  auto input = std::make_shared<TextPromptContext>();
  input->cache_identity = {0, 255, 17};
  const auto old = MakePrefillExpertTailCacheIdentity(settings, 97460, input);
  const auto retry = MakePrefillExpertTailCacheIdentity(settings, 97460, input);
  Expect(std::ranges::equal(old->CacheIdentity(97453),
                            retry->CacheIdentity(97453)),
         "post-B stable copy can serve a compatible same-length retry");
  for (const auto length : {97461U, 97709U, 107031U}) {
    const auto next =
        MakePrefillExpertTailCacheIdentity(settings, length, input);
    Expect(!std::ranges::equal(old->CacheIdentity(97453),
                               next->CacheIdentity(97453)),
           "every longer request rejects the previous post-B stable identity");
    Expect(std::ranges::equal(old->CacheIdentity(97204),
                              next->CacheIdentity(97204)),
           "the previous all-cut B remains compatible at every extension");
  }
  Expect(!std::ranges::equal(old->CacheIdentity(97453),
                             input->CacheIdentity(97453)),
         "default identity cannot restore a tagged post-B tail copy");
  std::cout
      << "PASS: post-B stable identity is retry-only across prompt lengths\n";
}

class RealSequenceTailRunner final : public PressureTailRunner {
public:
  using PressureTailRunner::PressureTailRunner;
  bool PreferNewestTailSnapshot() const noexcept { return true; }
  std::shared_ptr<const gufo::server::TextPromptContext> CacheIdentity(
      std::size_t length,
      std::shared_ptr<const gufo::server::TextPromptContext> context)
      const override {
    return gufo::server::MakePrefillExpertTailCacheIdentity(
        RealTailSettings(tail), length, std::move(context));
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t) const override {
    // Split at the stable framing boundary. The real engine can capture this
    // seven-row suffix in pass; this fixture keeps its exact logical charge.
    return std::nullopt;
  }
};

template<class Runner>
bool HasNewestTailPolicy(const Runner& runner) {
  if constexpr (requires { runner.PreferNewestTailSnapshot(); })
    return runner.PreferNewestTailSnapshot();
  return false;
}

void TestRealTailSequence(std::size_t gib, bool observe, bool ordinary) {
  TestPostTailStableIdentity();
  const std::vector<std::vector<std::size_t>> sessions{
      {87339, 97460, 107031, 117318, 126714, 136579},
      {17436, 45709, 46920, 143689, 144214, 153742},
      {18302, 37738, 56475, 75381, 94292, 114156},
      {16154, 38688, 61680, 84861, 107260, 130018},
      {17421, 31201, 46417, 63220, 86769, 105756},
      {18324, 43218, 62158, 84117, 104557, 127300},
      {16572, 39016, 61341, 84558, 107222},
      {22167, 44900, 69887, 89827, 116096}};
  const std::vector<std::vector<std::size_t>> stable_prefixes{
      {9059, 97453, 107024, 117311, 126707, 136572},
      {17429, 45702, 46879, 49620, 144154, 153735},
      {18295, 37731, 56128, 75374, 94285, 114149},
      {14566, 38681, 60495, 84734, 107253, 130011},
      {17414, 31117, 46356, 53102, 86708, 105749},
      {17556, 34011, 62108, 84110, 104513, 127293},
      {16565, 38622, 60642, 84551, 107215},
      {22160, 44893, 69880, 89820, 116089}};
  const std::array names{"row1086", "row204", "row304", "row3427",
                         "row343",  "row393", "row872", "nvidia-33102"};
  auto stats = std::make_shared<FakeStats>();
  auto runner =
      std::make_shared<RealSequenceTailRunner>(stats, ordinary ? 0 : 256);
  runner->chunk = 4096;
  runner->budget = gib << 30;
  runner->bytes_per_token = 28300;
  const std::unordered_map<std::size_t, std::size_t> observed_bytes{
      {9059, 368134756ULL},    {15898, 555927872ULL},   {16154, 562957632ULL},
      {16316, 567396296ULL},   {16572, 574385096ULL},   {17165, 590713868ULL},
      {17180, 591121736ULL},   {17421, 597743628ULL},   {17436, 598110536ULL},
      {18046, 614911952ULL},   {18068, 615506216ULL},   {18302, 621941712ULL},
      {18324, 622535976ULL},   {21911, 721050676ULL},   {22167, 728039476ULL},
      {28672, 906692056ULL},   {30945, 969112668ULL},   {32768, 1019168216ULL},
      {37482, 1148624512ULL},  {38432, 1174701656ULL},  {38760, 1183708536ULL},
      {39016, 1190738296ULL},  {40960, 1244120536ULL},  {42962, 1299105312ULL},
      {43218, 1306135072ULL},  {44644, 1345283176ULL},  {45453, 1367502348ULL},
      {45709, 1374532108ULL},  {46161, 1386944028ULL},  {46417, 1393973788ULL},
      {46664, 1400752376ULL},  {49620, 1481924136ULL},  {56219, 1663148356ULL},
      {56475, 1670178116ULL},  {57344, 1694025176ULL},  {61085, 1796757068ULL},
      {61341, 1803786828ULL},  {61424, 1806061976ULL},  {61440, 1806501336ULL},
      {61680, 1813091736ULL},  {61902, 1819197712ULL},  {62158, 1826227472ULL},
      {62964, 1848350376ULL},  {69631, 2031441876ULL},  {69632, 2031207896ULL},
      {69887, 2038430676ULL},  {73728, 2143929816ULL},  {75125, 2182295468ULL},
      {75381, 2189284268ULL},  {81920, 2368882136ULL},  {83861, 2422186028ULL},
      {84117, 2429215788ULL},  {84302, 2434301712ULL},  {84558, 2441331472ULL},
      {84605, 2442616268ULL},  {84861, 2449646028ULL},  {86016, 2481358296ULL},
      {86513, 2495009948ULL},  {86769, 2502039708ULL},  {87083, 2510673796ULL},
      {87339, 2517703556ULL},  {89571, 2578994276ULL},  {89827, 2585983076ULL},
      {94036, 2701587496ULL},  {94208, 2706310616ULL},  {94292, 2708576296ULL},
      {97204, 2788580776ULL},  {97453, 2795422348ULL},  {97460, 2795569576ULL},
      {102400, 2931262936ULL}, {104301, 2983468428ULL}, {104557, 2990498188ULL},
      {105500, 3016388936ULL}, {105756, 3023418696ULL}, {106496, 3043739096ULL},
      {106775, 3051416116ULL}, {107004, 3057688776ULL}, {107024, 3058237976ULL},
      {107031, 3058404916ULL}, {107222, 3063684912ULL}, {107260, 3064677576ULL},
      {110592, 3156215256ULL}, {113900, 3247052936ULL}, {114156, 3254082696ULL},
      {114688, 3268691416ULL}, {116096, 3307314136ULL}, {117062, 3333891312ULL},
      {117311, 3340734676ULL}, {117318, 3340880112ULL}, {122880, 3493643736ULL},
      {126458, 3591905472ULL}, {126707, 3598748836ULL}, {126714, 3598894272ULL},
      {126976, 3606119896ULL}, {127044, 3607987176ULL}, {127300, 3615016936ULL},
      {129762, 3682633312ULL}, {130018, 3689622112ULL}, {135168, 3831072216ULL},
      {136323, 3862804196ULL}, {136572, 3867241068ULL}, {136579, 3869833956ULL},
      {143360, 4056024536ULL}, {143433, 4058033148ULL}, {143689, 4065062908ULL},
      {144214, 4079485232ULL}, {151552, 4280976856ULL}, {153486, 4334094352ULL},
      {153742, 4341124112ULL}};
  runner->logical_bytes = [observed_bytes](std::size_t position) {
    const auto it = observed_bytes.find(position);
    return it == observed_bytes.end()
               ? std::size_t{119358936} + position * std::size_t{27460}
               : it->second;
  };
  TextRunnerPool pool(runner, 1);
  bool correct = true;
  for (std::size_t s = 0; s < sessions.size(); ++s) {
    std::vector<TextRunnerToken> prompt;
    std::size_t previous = 0;
    for (std::size_t turn = 0; turn < sessions[s].size(); ++turn) {
      const auto length = sessions[s][turn];
      // New prompts retain the old prompt only through L-7. Preserve all
      // earlier edits so old full-prompt ancestry genuinely fails.
      if (!prompt.empty())
        std::fill(prompt.end() - 7, prompt.end(), 1000 + turn);
      prompt.resize(length, s + 1);
      std::fill_n(prompt.begin(), 32, 99);
      runner->snapshot_positions.clear();
      stats->prefill_spans.clear();
      const auto stable = stable_prefixes[s][turn];
      auto request = pool.Acquire(prompt, {}, {}, {}, true, stable);
      const auto reused = request.cached_prompt_tokens();
      const auto metrics = FinishPressureRequest(request);
      if (gib == 8 && s == 0 && !ordinary &&
          !HasNewestTailPolicy(static_cast<const TextModelRunner&>(*runner))) {
        constexpr std::array<std::size_t, 6> logged_P{0,     87083, 97204,
                                                      97204, 97204, 97204};
        constexpr std::array<std::size_t, 6> logged_retained{
            8410788720ULL, 8462811676ULL, 5846818752ULL,
            6129315452ULL, 6387329612ULL, 6655821844ULL};
        constexpr std::array<std::size_t, 6> logged_published{
            9092528456ULL, 8290313740ULL, 3058237976ULL,
            3340734676ULL, 3598748836ULL, 3867241068ULL};
        Expect(reused == logged_P[turn], "row1086 reproduces the logged stale-B restore");
        Expect(pool.RetainedSnapshotBytes() == logged_retained[turn],
               "row1086 reproduces the exact logged retained-byte ledger");
        Expect(metrics.snapshot_bytes == logged_published[turn],
               "row1086 reproduces the exact logged publication-byte sum");
      }
      std::size_t rows = 0;
      for (const auto n : stats->prefill_spans)
        rows += n;
      auto extension = prompt;
      std::fill(extension.end() - 7, extension.end(), 1001 + turn);
      extension.resize(length + 1000, s + 1);
      const auto next = pool.CachedPrefixTokens(extension, nullptr);
      const auto expected = ordinary ? length - 7 : length - 256;
      const bool fitting = runner->logical_bytes(expected) <= runner->budget;
      std::cout << "real budget=" << gib << " session=" << names[s]
                << " turn=" << turn << " L=" << length << " P=" << reused
                << " rows=" << rows << " next=" << next
                << " retained=" << pool.RetainedSnapshotBytes()
                << " skipped=" << metrics.snapshots_skipped << " captures=";
      for (auto p : runner->snapshot_positions)
        std::cout << p << ',';
      std::cout << std::endl;
      if (!ordinary && fitting)
        correct &= next == expected;
      if (!ordinary && turn && previous)
        correct &= reused == previous;
      previous = fitting ? expected : 0;
    }
  }
  if (!observe && !ordinary)
    Expect(correct, "real rewritten chat turns retain the newest fitting B");
}

// Decimal 28.3 KB/token is the owner's approximate real-model scale. Only
// logical accounting uses it; snapshots still store actual row provenance.
void TestTailDeepWalk(bool interleaved, bool observe = false,
                      std::size_t slots = 1) {
  const std::vector<std::vector<std::size_t>> sessions{
      {87339, 97460, 107031, 117318, 126714, 136579},
      {17436, 45709, 46920, 143689, 144214, 153742},
      {16154, 38688, 61680, 84861, 107260, 130018},
      {18324, 43218, 62158, 84117, 104557, 127300}};
  const std::array names{"row1086", "row204", "row3427", "row393"};
  bool correct = true;
  const auto walk = [&](std::size_t session, const auto& lengths,
                        TextRunnerPool& pool,
                        const std::shared_ptr<PressureTailRunner>& runner,
                        const std::shared_ptr<FakeStats>& stats,
                        std::size_t& previous) {
    for (const auto length : lengths) {
      std::vector<TextRunnerToken> prompt(length, session + 1);
      // Equal short family hashes must not imply equal session ownership.
      std::fill_n(prompt.begin(), 32, 99);
      stats->prefill_spans.clear();
      runner->snapshot_positions.clear();
      const auto name =
          interleaved ? (session == 0 ? "A" : "B") : names[session];
      std::cerr << "walk_begin session=" << name << " L=" << length
                << " slots=" << slots << '\n';
      auto request = pool.Acquire(prompt);
      const auto reused = request.cached_prompt_tokens();
      const auto metrics = FinishPressureRequest(request);
      std::size_t rows = 0;
      for (const auto n : stats->prefill_spans)
        rows += n;
      auto extension = prompt;
      extension.resize(length + 1000, session + 1);
      const bool published =
          std::ranges::count(runner->snapshot_positions, length - 256) == 1 &&
          pool.CachedPrefixTokens(extension, nullptr) == length - 256;
      std::cout << "deep session=" << name << " L=" << length
                << " B=" << length - 256
                << " bytes=" << (length - 256) * runner->bytes_per_token
                << " P=" << reused << " rows=" << rows
                << " B_published=" << published
                << " retained=" << pool.RetainedSnapshotBytes()
                << " skipped=" << metrics.snapshots_skipped
                << " reason=" << metrics.snapshot_skip_reason << " copies=";
      for (const auto position : runner->snapshot_positions)
        std::cout << position << ',';
      std::cout << std::endl;
      std::cerr << "walk_end session=" << name << " L=" << length << '\n';
      if (!interleaved)
        correct &= published && reused == previous && rows == length - reused;
      // In an interleaved pair the three physical payloads eventually cannot
      // coexist. A refusal must preserve BOTH last boundaries; it may not make
      // room by deleting the other session's snapshots.
      if (interleaved) {
        const auto peer = std::size_t{1} - session;
        const auto peer_length = session == 0 ? length - 10000 : length;
        if (peer_length >= 100000) {
          std::vector<TextRunnerToken> peer_prompt(peer_length + 1000,
                                                   peer + 1);
          std::fill_n(peer_prompt.begin(), 32, 99);
          std::cout << "deep_peer session=" << name << " L=" << length
                    << " peer_P="
                    << pool.CachedPrefixTokens(peer_prompt, nullptr)
                    << std::endl;
        }
        correct &= reused == previous && rows == length - reused;
      }
      if (published)
        previous = length - 256;
    }
  };
  if (!interleaved) {
    for (std::size_t session = 0; session < sessions.size(); ++session) {
      auto stats = std::make_shared<FakeStats>();
      auto runner = std::make_shared<PressureTailRunner>(stats, 256);
      runner->bytes_per_token = 28300;
      runner->budget = std::size_t{8} << 30;
      TextRunnerPool pool(runner, slots);
      std::size_t previous = 0;
      walk(session, sessions[session], pool, runner, stats, previous);
      auto next = pool.Acquire(std::vector<TextRunnerToken>{7});
      next.Invalidate();
    }
  } else {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PressureTailRunner>(stats, 256);
    runner->bytes_per_token = 28300;
    runner->budget = std::size_t{8} << 30;
    TextRunnerPool pool(runner, slots);
    std::array<std::size_t, 2> previous{};
    for (std::size_t length = 100000; length <= 160000; length += 10000)
      for (std::size_t session = 0; session < 2; ++session) {
        const auto source = previous[session];
        const auto peer_source = previous[1 - session];
        walk(session, std::array{length}, pool, runner, stats,
             previous[session]);
        if (source) {
          std::vector<TextRunnerToken> tokens(source + 512, session + 1);
          std::fill_n(tokens.begin(), 32, 99);
          correct &= pool.CachedPrefixTokens(tokens, nullptr) == source;
        }
        if (peer_source) {
          std::vector<TextRunnerToken> tokens(peer_source + 512, 2 - session);
          std::fill_n(tokens.begin(), 32, 99);
          correct &= pool.CachedPrefixTokens(tokens, nullptr) == peer_source;
        }
      }
  }
  if (!observe)
    Expect(correct, interleaved
                        ? "deep sessions keep both restore sources"
                        : "every fitting deep B publishes and is reused");
}

void TestTailDeepDisk() {
  TemporaryDirectory directory;
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  runner->bytes_per_token = 28300;
  runner->budget = std::size_t{8} << 30;
  const TextRunnerDiskCacheOptions options{
      .directory = directory.path(),
      .capacity_bytes = std::size_t{32} << 30,
      .staging_capacity_bytes = std::size_t{8} << 30,
      .min_checkpoint_step_tokens = 0};
  std::size_t previous = 0;
  for (const auto length : {143689U, 144214U, 153742U}) {
    // Each shutdown drains accepted writes. A fake GiB capture is instant;
    // overlapping all turns would test disk staging pressure instead of B's
    // persistence/identity contract. The disk budget remains independent.
    TextRunnerPool pool(runner, 2, options);
    auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
    Expect(request.cached_prompt_tokens() == previous,
           "each deep disk turn restores the previous all-cut B");
    (void)FinishPressureRequest(request);
    previous = length - 256;
  }
  TextRunnerPool restored(runner, 2, options);
  auto request = restored.Acquire(std::vector<TextRunnerToken>(154742, 1));
  std::cout << "deep disk restored_P=" << request.cached_prompt_tokens()
            << " disk=" << request.cache_disk_hit() << std::endl;
  Expect(request.cache_disk_hit() && request.cached_prompt_tokens() == 153486,
         "deep disk restart restores B with original row policy");
  (void)FinishPressureRequest(request);
}

void TestTailDeepByteCeiling() {
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  runner->bytes_per_token = 28300;
  runner->budget = std::size_t{8} << 30;
  TextRunnerPool pool(runner, 2);
  {
    auto first = pool.Acquire(std::vector<TextRunnerToken>(262144, 1));
    (void)FinishPressureRequest(first);
  }
  auto next = pool.Acquire(std::vector<TextRunnerToken>(272144, 1));
  std::cout << "deep ceiling P=" << next.cached_prompt_tokens() << std::endl;
  Expect(next.cached_prompt_tokens() == 261888,
         "cold B beyond 128K still fits its logical byte budget");
  runner->snapshot_positions.clear();
  (void)FinishPressureRequest(next);
  Expect(std::ranges::count(runner->snapshot_positions, 271888) == 0 &&
             pool.CachedPrefixTokens(std::vector<TextRunnerToken>(273144, 1),
                                     nullptr) == 261888,
         "7.7 GB B cannot evict the protected 7.4 GB source to fit");
}

void TestTailPressureLoop(bool observe = false) {
  std::array<std::vector<std::size_t>, 2> prefixes, skips;
  for (const bool tail : {false, true}) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PressureTailRunner>(stats, tail ? 256 : 0);
    TextRunnerPool pool(runner, 1);
    for (const auto length : {30000U, 31200U, 32400U, 33600U, 34800U}) {
      runner->passes.clear();
      auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
      const auto prefix = request.cached_prompt_tokens();
      const auto metrics = FinishPressureRequest(request);
      std::size_t rows = 0;
      for (const auto [n, cut] : runner->passes)
        rows += n;
      prefixes[tail].push_back(prefix);
      skips[tail].push_back(metrics.snapshots_skipped);
      std::cout << "pressure tail=" << tail << " L=" << length
                << " P=" << prefix << " rows=" << rows
                << " published=" << metrics.snapshot_bytes / runner->kPayload
                << " skipped=" << metrics.snapshots_skipped
                << " reason=" << metrics.snapshot_skip_reason;
#ifndef TAIL_SNAPSHOT_POLICY_BASELINE
      std::cout << " policy_skipped=" << metrics.snapshots_skipped_by_policy;
      Expect(metrics.snapshots_published ==
                 metrics.snapshot_bytes / runner->kPayload,
             "published counter counts retained snapshots, not attempts");
      Expect(metrics.snapshots_skipped_by_policy == (tail ? 1U : 0U),
             "intentional final skips are counted separately from capacity "
             "failures");
#endif
      std::cout << std::endl;
      Expect(rows == length - prefix,
             "actual new rows follow the restored prefix");
    }
  }
  if (!observe) {
    Expect(prefixes[1] ==
               std::vector<std::size_t>({0, 29744, 30944, 32144, 33344}),
           "pressure loop retains each previous B");
    // A deliberately unpublished retry is a reported policy skip. Capacity
    // failures must not be added on top of that one optional skip per turn.
    for (std::size_t i = 0; i < skips[1].size(); ++i)
      Expect(skips[1][i] <= skips[0][i] + 1,
             "tail has no extra capacity failures beyond its optional final");
  }
}

void TestTailPressureAdmission(bool depth, bool coincident = false) {
  for (const bool refused : {false, true}) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PressureTailRunner>(stats, 256);
    const auto length = depth ? (refused ? 131073U : 131072U) : 30000U;
    runner->budget = depth ? 16 * runner->kPayload : 3 * runner->kPayload;
    runner->boundary_extra = !depth && refused ? 1 : 0;
    TextRunnerPool pool(runner, 1);
    auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1), {}, {},
                                {}, true, coincident ? length - 256 : 0);
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
    // Cancellation reports B's own reason before final admission can replace
    // it.
    const auto metrics = request.Cancel();
    const auto copies =
        std::ranges::count(runner->snapshot_positions, length - 256);
    std::cout << "B admission depth=" << depth << " coincident=" << coincident
              << " L=" << length << " budget=" << runner->budget
              << " B_copies=" << copies
              << " skipped=" << metrics.snapshots_skipped
              << " reason=" << metrics.snapshot_skip_reason << std::endl;
    Expect(copies == 1, "B uses available own-session bytes at either old cap");
    Expect(metrics.snapshot_skip_reason != "b_depth_cap" &&
               metrics.snapshot_skip_reason != "b_over_third_of_budget",
           "removed policy rules never report a skip");
    if (!depth) {
      auto next = pool.Acquire(std::vector<TextRunnerToken>(31200, 1));
      Expect(next.cached_prompt_tokens() == 29744,
             "B above one third serves the next continuation");
      (void)next.Prefill(32768);
      next.Invalidate();
    }
  }
}

void TestTailFinalAdmission(bool disk) {
  for (const bool fits : {false, true}) {
    TemporaryDirectory directory;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PressureTailRunner>(stats, 256);
    if (fits)
      runner->budget = 8 * runner->kPayload;
    const TextRunnerDiskCacheOptions options{
        .directory = disk ? directory.path() : std::filesystem::path{},
        .capacity_bytes = disk ? 8 * runner->kPayload : 0,
        .staging_capacity_bytes = 4 * runner->kPayload,
        .min_checkpoint_step_tokens = 0};
    {
      TextRunnerPool pool(runner, 1,
                          disk ? std::optional(options) : std::nullopt);
      auto request = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
      const auto metrics = FinishPressureRequest(request);
      const auto copies = std::ranges::count(runner->snapshot_positions, 30000);
      std::cout << "final disk=" << disk << " fits=" << fits
                << " copies=" << copies
                << " published=" << metrics.snapshot_bytes / runner->kPayload
                << " skipped=" << metrics.snapshots_skipped
                << " reason=" << metrics.snapshot_skip_reason << std::endl;
      Expect(copies == (fits ? 1 : 0),
             "final snapshot uses free capacity only");
      if (!fits)
        Expect(metrics.snapshot_skip_reason == "tail_final_not_published",
               "full-prompt pressure reports policy skip");
      if (!disk) {
        auto distract = pool.Acquire({9, 9, 9});
        distract.Invalidate();  // Force immutable retry rather than live reuse.
        auto retry = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
        Expect(retry.cached_prompt_tokens() == (fits ? 30000U : 29744U) &&
                   retry.cache_restore_bytes() == runner->kPayload,
               "free final serves an exact retry; skipped final retains B");
        retry.Invalidate();
      }
    }
    if (disk) {
      TextRunnerPool restarted(runner, 1, options);
      auto retry = restarted.Acquire(std::vector<TextRunnerToken>(30000, 1));
      std::cout << "disk restart fits=" << fits
                << " P=" << retry.cached_prompt_tokens()
                << " disk=" << retry.cache_disk_hit() << std::endl;
      Expect(retry.cache_disk_hit() &&
                 retry.cached_prompt_tokens() == (fits ? 30000U : 29744U),
             "skipped final is absent on disk; admitted B/final persist");
      retry.Invalidate();
      if (!fits)
        Expect(std::ranges::count(runner->materialized_positions, 30000) == 0,
               "skipped final never materializes for disk");
    }
  }
}

void TestTailFailedFinalDoesNotPersist() {
  for (const int failure : {1, 2, 3}) {
    TemporaryDirectory directory;
    auto runner = std::make_shared<PressureTailRunner>(
        std::make_shared<FakeStats>(), 256);
    runner->budget = 8 * runner->kPayload;
    runner->final_failure = failure;
    const TextRunnerDiskCacheOptions options{
        .directory = directory.path(),
        .capacity_bytes = 8 * runner->kPayload,
        .staging_capacity_bytes = 4 * runner->kPayload,
        .min_checkpoint_step_tokens = 0};
    {
      TextRunnerPool pool(runner, 1, options);
      auto request = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
      const auto metrics = FinishPressureRequest(request);
      Expect(metrics.snapshot_skip_reason ==
                 (failure == 3 ? "reservation_mismatch" : "capture_failure"),
             "failed optional final keeps its real capture failure reason");
    }
    TextRunnerPool restarted(runner, 1, options);
    auto retry = restarted.Acquire(std::vector<TextRunnerToken>(30000, 1));
    std::cout << "failed final kind=" << failure
              << " disk_P=" << retry.cached_prompt_tokens() << std::endl;
    Expect(retry.cache_disk_hit() && retry.cached_prompt_tokens() == 29744,
           "failed final publication cannot persist a mismatched snapshot");
    retry.Invalidate();
  }
}

void TestTailCoincidentBoundaryPersists() {
  TemporaryDirectory directory;
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  const TextRunnerDiskCacheOptions options{
      .directory = directory.path(),
      .capacity_bytes = 8 * runner->kPayload,
      .staging_capacity_bytes = 4 * runner->kPayload,
      .min_checkpoint_step_tokens = 8192};
  {
    TextRunnerPool pool(runner, 1, options);
    auto request = pool.Acquire(std::vector<TextRunnerToken>(30000, 1), {}, {},
                                {}, true, 29744);
    (void)FinishPressureRequest(request);
    Expect(std::ranges::count(runner->snapshot_positions, 29744) == 1,
           "coincident stable/B capture is shared");
  }
  TextRunnerPool restarted(runner, 1, options);
  auto next = restarted.Acquire(std::vector<TextRunnerToken>(31200, 1));
  std::cout << "coincident disk P=" << next.cached_prompt_tokens() << std::endl;
  Expect(next.cache_disk_hit() && next.cached_prompt_tokens() == 29744,
         "coincident B persists despite ordinary checkpoint spacing");
  next.Invalidate();
}

void TestTailSupersededEviction(bool rewrite_tail = false,
                                bool new_grid = false) {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PressureTailRunner>(stats, 256);
  runner->budget = 16 * runner->kPayload;
  TextRunnerPool pool(runner, 1);
  // Older different-session entries must survive all five superseded A copies.
  auto other = pool.Acquire(std::vector<TextRunnerToken>(18000, 2));
  (void)FinishPressureRequest(other);
  auto first = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
  (void)FinishPressureRequest(first);
  const std::size_t next_length = new_grid ? 50000 : 31200;
  std::vector<TextRunnerToken> latest(next_length + 1200, 1);
  if (rewrite_tail)
    latest[29990] = 3;  // Rewritten assistant framing after the restored B.
  auto next = pool.Acquire({latest.begin(), latest.begin() + next_length});
  Expect(next.cached_prompt_tokens() == 29744, "demotion source is previous B");
  (void)FinishPressureRequest(next);
  Expect(pool.CachedPrefixTokens(std::vector<TextRunnerToken>(30000, 1),
                                 nullptr) == 30000,
         "superseded old full stays valid for retry before eviction");
  Expect(pool.TrimSnapshots(5 * runner->kPayload) == 5 * runner->kPayload,
         "trim drops five superseded snapshots");
  std::cout << "demotion rewrite=" << rewrite_tail << " old_A="
            << pool.CachedPrefixTokens(std::vector<TextRunnerToken>(30000, 1),
                                       nullptr)
            << " other="
            << pool.CachedPrefixTokens(std::vector<TextRunnerToken>(18000, 2),
                                       nullptr)
            << " latest=" << pool.CachedPrefixTokens(latest, nullptr)
            << std::endl;
  Expect(pool.CachedPrefixTokens(std::vector<TextRunnerToken>(30000, 1),
                                 nullptr) == 0 &&
             pool.CachedPrefixTokens(std::vector<TextRunnerToken>(18000, 2),
                                     nullptr) == 18000 &&
             pool.CachedPrefixTokens(latest, nullptr) == next_length - 256,
         "superseded A grids/B/final evict before older different-session "
         "entries and latest B");
  if (new_grid) {
    latest[49500] = 4;
    Expect(pool.CachedPrefixTokens(latest, nullptr) == 49152,
           "this request's new grid is not demoted with its predecessor");
  }
}

void TestTailSharedSourceDoesNotDemoteOtherSession() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PressureTailRunner>(stats, 256);
  runner->budget = 16 * runner->kPayload;
  TextRunnerPool pool(runner, 1);
  auto seed = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
  (void)FinishPressureRequest(seed);
  std::vector<TextRunnerToken> branch(31200, 1);
  branch[12000] = 2;
  auto other = pool.Acquire(branch);
  Expect(other.cached_prompt_tokens() == 8192,
         "different session shares an early grid only");
  (void)FinishPressureRequest(other);
  (void)pool.TrimSnapshots(runner->kPayload);
  Expect(pool.CachedPrefixTokens(std::vector<TextRunnerToken>(30000, 1),
                                 nullptr) == 29744,
         "shared early grid cannot demote the other session's B or grids");
  std::cout << "PASS: TestTailSharedSourceDoesNotDemoteOtherSession\n";
}

void TestTailCompleteLiveRetryUsesFinalAdmission() {
  for (const bool fits : {false, true}) {
    auto runner = std::make_shared<PressureTailRunner>(
        std::make_shared<FakeStats>(), 256);
    runner->budget = (fits ? 8 : 3) * runner->kPayload;
    TextRunnerPool pool(runner, 2);
    const std::vector<TextRunnerToken> prompt(30000, 1);
    auto seed = pool.Acquire(prompt);
    while (!seed.prefill_complete())
      (void)seed.Prefill(32768);
    (void)seed.Cancel();
    Expect(std::ranges::count(runner->snapshot_positions, 30000) == 0,
           "cancelled complete prefill leaves only the live full state");
    auto retry = pool.Acquire(prompt);
    Expect(retry.cached_prompt_tokens() == 30000 &&
               retry.cache_restore_bytes() == 0,
           "two-slot exact retry selects the complete live frontier");
    const auto metrics = FinishPressureRequest(retry);
    std::cout << "live retry slots=2 tail=256 L=30000 P=30000 fits=" << fits
              << " final_copies="
              << std::ranges::count(runner->snapshot_positions, 30000)
              << " skipped=" << metrics.snapshots_skipped
              << " reason=" << metrics.snapshot_skip_reason << std::endl;
    Expect(
        std::ranges::count(runner->snapshot_positions, 30000) == (fits ? 1 : 0),
        "complete tail live retry uses free-only final admission");
    Expect(
        fits || (metrics.snapshot_skip_reason == "tail_final_not_published" &&
                 metrics.snapshots_skipped == 1),
        "live final refusal goes through the existing prompt policy");
    auto extension = prompt;
    extension.resize(31200, 1);
    Expect(pool.CachedPrefixTokens(extension, nullptr) == 29744,
           "live retry cannot evict the all-cut continuation checkpoint");
  }
  std::cout << "PASS: two-slot complete live retry, full/free budgets\n";
}

void TestTailSharedBoundaryKeepsBranchPurpose() {
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  runner->budget = 3 * runner->kPayload;
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> original(30000, 1);
  auto seed = pool.Acquire(original);
  Expect(seed.ShareCheckpoint(29744) == 29744, "peer explicitly shares B");
  (void)FinishPressureRequest(seed);
  auto independent = pool.Acquire(std::vector<TextRunnerToken>(18000, 2));
  (void)FinishPressureRequest(independent);
  auto extension = pool.Acquire(std::vector<TextRunnerToken>(31200, 1));
  Expect(extension.cached_prompt_tokens() == 29744,
         "linear extension restores shared B");
  (void)FinishPressureRequest(extension);
  (void)pool.TrimSnapshots(runner->kPayload);
  std::cout << "shared B29744 after extension/trim="
            << pool.CachedPrefixTokens(original, nullptr) << std::endl;
  Expect(
      pool.CachedPrefixTokens(original, nullptr) == 29744,
      "explicitly shared B stays a learned branch without divergence evidence");
  std::cout << "PASS: coincident shared B retains explicit branch purpose\n";
}

void TestTailSharedBoundaryGuards() {
  for (const bool depth : {false, true}) {
    auto runner = std::make_shared<PressureTailRunner>(
        std::make_shared<FakeStats>(), 256);
    runner->budget = 3 * runner->kPayload;
    runner->boundary_extra = depth ? 0 : 1;
    TextRunnerPool pool(runner, 1);
    const std::size_t length = depth ? 131073 : 30000;
    auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
    Expect(request.ShareCheckpoint(length - 256) == length - 256,
           "guard fixture explicitly shares the exact B");
    (void)FinishPressureRequest(request);
    Expect(std::ranges::count(runner->snapshot_positions, length - 256) == 1,
           "shared B also uses available own-session bytes above old caps");
  }
  std::cout << "PASS: shared B above old size/depth caps\n";
}

void TestTailCompletedHostCaptureReleasesForecast() {
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  struct Source final : gufo::platform::MemoryReadingSource {
    std::atomic<std::uint64_t> available{std::uint64_t{6} << 30};
    gufo::platform::MemoryReadings Read() noexcept override {
      gufo::platform::MemoryReadings r;
      r.ram_ok = true;
      r.ram_available = available.load();
      return r;
    }
  } source;
  gufo::platform::MemoryGuard guard(&source, 4 * gib);
  gufo::platform::SetProcessMemoryGuard(&guard);
  {
    auto runner = std::make_shared<PressureTailRunner>(
        std::make_shared<FakeStats>(), 256);
    runner->budget = 8 * runner->kPayload;
    TextRunnerPool pool(runner, 2);
    auto first = pool.Acquire(std::vector<TextRunnerToken>(128, 1));
    while (!first.prefill_complete())
      (void)first.Prefill(32768);
    auto second = pool.Acquire(std::vector<TextRunnerToken>(128, 2));
    while (!second.prefill_complete())
      (void)second.Prefill(32768);
    std::binary_semaphore entered{0}, release{0};
    std::atomic<bool> first_copy{true};
    runner->final_capture_hook = [&] {
      if (!first_copy.exchange(false))
        return;
      entered.release();
      release.acquire();
      // Model the physical allocation becoming part of the actual reading.
      source.available.store(5 * gib);
    };
    Expect(!first.PreparePromptSnapshot(),
           "first final copy remains asynchronous");
    Expect(entered.try_acquire_for(std::chrono::seconds(5)),
           "copy hook entered");
    release.release();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (first.SnapshotPending() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Expect(!first.SnapshotPending(),
           "allocation completes before owner publication");
    runner->final_capture_hook = {};
    // The first future is ready but has not been consumed/published. Its
    // forecast must no longer be charged on top of the actual 5 GiB reading.
    second.CapturePromptSnapshot();
    const auto metrics = second.Commit();
    Expect(
        metrics.snapshot_bytes == runner->kPayload &&
            metrics.snapshots_skipped == 0,
        "an allocated, unpublished copy does not double-charge host headroom");
    (void)first.Cancel();
  }
  gufo::platform::SetProcessMemoryGuard(nullptr);
  std::cout << "PASS: async allocation completion discharges host forecast\n";
}

void TestTailGenericLiveFreezeChargesHostForecast() {
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  struct Source final : gufo::platform::MemoryReadingSource {
    std::atomic<std::uint64_t> available{std::uint64_t{64} << 30};
    gufo::platform::MemoryReadings Read() noexcept override {
      gufo::platform::MemoryReadings r;
      r.ram_ok = true;
      r.ram_available = available.load();
      return r;
    }
  } source;
  gufo::platform::MemoryGuard guard(&source, 4 * gib);
  gufo::platform::SetProcessMemoryGuard(&guard);
  {
    auto runner = std::make_shared<PressureTailRunner>(
        std::make_shared<FakeStats>(), 256);
    runner->budget = 2 * runner->kPayload;
    TextRunnerPool pool(runner, 2);
    // Refuse the seed copy at the host floor, leaving a safe all-cut live
    // fallback at 1000. This fixture must not depend on a B size policy.
    source.available.store(4 * gib);
    auto seed = pool.Acquire(std::vector<TextRunnerToken>(1256, 1), {}, {}, {},
                             true, 1000);
    while (seed.prefill_position() < 1000)
      (void)seed.Prefill(1000 - seed.prefill_position());
    (void)seed.Cancel();
    source.available.store(64 * gib);
    Expect(pool.CachedPrefixTokens(std::vector<TextRunnerToken>(30000, 1),
                                   nullptr) == 1000,
           "seed retains the partial live frontier");
    auto peer = pool.Acquire(std::vector<TextRunnerToken>(128, 9));
    while (!peer.prefill_complete())
      (void)peer.Prefill(32768);
    source.available.store(5 * gib);
    std::binary_semaphore entered{0}, resume{0};
    runner->snapshot_capture_hook = [&](std::size_t position) {
      if (position != 1000)
        return;
      entered.release();
      if (!resume.try_acquire_for(std::chrono::seconds(5)))
        throw std::runtime_error("generic live capture gate timed out");
    };
    auto pending = std::async(std::launch::async, [&] {
      return pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
    });
    Expect(entered.try_acquire_for(std::chrono::seconds(5)),
           "generic live freeze is admitted but has not allocated its copy");
    peer.CapturePromptSnapshot();
    const auto metrics = peer.Commit();
    resume.release();
    auto continuation = pending.get();
    continuation.Invalidate();
    runner->snapshot_capture_hook = {};
    std::cout << "generic live P=1000 pending=1 peer_final_bytes="
              << metrics.snapshot_bytes
              << " skipped=" << metrics.snapshots_skipped
              << " reason=" << metrics.snapshot_skip_reason << std::endl;
    Expect(metrics.snapshot_bytes == 0 && metrics.snapshots_skipped == 1 &&
               metrics.snapshot_skip_reason == "host_memory_floor",
           "pending generic live freeze must charge peer final headroom");
  }
  gufo::platform::SetProcessMemoryGuard(nullptr);
  std::cout
      << "PASS: generic live freeze shares outstanding host reservation\n";
}

void TestTailFinalPersistencePublicationOrdering() {
  TemporaryDirectory directory;
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  runner->payload = 1024;
  runner->budget = 8 * runner->kPayload;
  const TextRunnerDiskCacheOptions options{
      .directory = directory.path(),
      .capacity_bytes = 8 * runner->kPayload,
      .staging_capacity_bytes = 4 * runner->kPayload,
      .min_checkpoint_step_tokens = 0};
  const std::vector<TextRunnerToken> prompt(30000, 1);
  bool removed = false, initialized = false;
  const TextRunnerSnapshot* observed = nullptr;
  {
    TextRunnerPool pool(runner, 1, options);
    // Old ordering: the third payload query is in Capture after publication,
    // before bookkeeping is installed or the write-through offer is queued.
    // Prepared ordering: that query is in Commit before publication. This is
    // a deterministic peer-removal interleaving, with no sleeps or data race.
    runner->final_payload_hook = [&](const TextRunnerSnapshot& snapshot,
                                     std::size_t reads) {
      if (reads != 3)
        return;
      observed = &snapshot;
      if (pool.CachedPrefixTokens(prompt, nullptr) == prompt.size()) {
        initialized = bool(snapshot.persistence);
        removed = true;
        (void)pool.TrimSnapshots(8 * runner->kPayload);
      }
    };
    auto request = pool.Acquire(prompt);
    (void)FinishPressureRequest(request);
    Expect(observed != nullptr, "publication ordering hook was reached");
    if (!removed) {
      initialized = bool(observed->persistence);
      (void)pool.TrimSnapshots(8 * runner->kPayload);
      removed = true;
    }
    std::cout << "final peer_removal=" << removed
              << " initialized_at_removal=" << initialized << std::endl;
    Expect(
        initialized,
        "a peer removing a published final must see initialized persistence");
    runner->final_payload_hook = {};
  }
  TextRunnerPool restarted(runner, 1, options);
  auto retry = restarted.Acquire(prompt);
  Expect(retry.cache_disk_hit() && retry.cached_prompt_tokens() == 30000,
         "successful publication/removal cannot lose final persistence");
  retry.Invalidate();
  std::cout << "PASS: final publication/persistence/removal ordering\n";
}

void TestTailFinalPublicationRefusalDoesNotPersist() {
  TemporaryDirectory directory;
  auto runner =
      std::make_shared<PressureTailRunner>(std::make_shared<FakeStats>(), 256);
  runner->payload = 1024;
  runner->budget = 8 * runner->kPayload;
  const TextRunnerDiskCacheOptions options{
      .directory = directory.path(),
      .capacity_bytes = 8 * runner->kPayload,
      .staging_capacity_bytes = 4 * runner->kPayload,
      .min_checkpoint_step_tokens = 0};
  const std::vector<TextRunnerToken> prompt(30000, 1);
  {
    TextRunnerPool pool(runner, 2, options);
    for (std::size_t i = 0; i < 123; ++i) {
      auto seed = pool.Acquire({9, static_cast<TextRunnerToken>(i + 10)});
      (void)FinishPressureRequest(seed);
    }
    bool peer_published = false;
    runner->final_capture_hook = [&] {
      if (peer_published)
        return;
      peer_published = true;
      const auto saved_length = runner->full_length;
      const auto saved_boundary = runner->boundary;
      auto peer = pool.Acquire({8, 8, 8});
      (void)FinishPressureRequest(
          peer);  // consumes record 128 during final copy
      runner->full_length = saved_length;
      runner->boundary = saved_boundary;
    };
    auto request = pool.Acquire(prompt);
    const auto metrics = FinishPressureRequest(request);
    runner->final_capture_hook = {};
    Expect(peer_published &&
               metrics.snapshot_skip_reason == "tail_final_not_published",
           "a peer consumes the last free record after final admission");
    Expect(pool.CachedPrefixTokens(prompt, nullptr) == 29744,
           "publication refusal retains B and excludes full RAM state");
  }
  TextRunnerPool restarted(runner, 1, options);
  auto retry = restarted.Acquire(prompt);
  Expect(retry.cache_disk_hit() && retry.cached_prompt_tokens() == 29744,
         "a final refused at publication is terminal for persistence");
  retry.Invalidate();
  std::cout << "PASS: final record-race refusal cannot persist\n";
}

void TailDefaultGolden() {
  for (const std::size_t slots : {1U, 2U}) {
    for (const bool active : {false, true}) {
      auto stats = std::make_shared<FakeStats>();
      auto runner =
          std::make_shared<PressureTailRunner>(stats, active ? 0 : 256);
      runner->policy_active = active;
      auto context = std::make_shared<gufo::server::TextPromptContext>();
      context->cache_identity = {0, 17, 255, 9};
      const auto identity = runner->CacheIdentity(30000, context);
      std::cout << "golden slots=" << slots << " active=" << active
                << " identity=";
      for (auto b : identity->cache_identity)
        std::cout << static_cast<unsigned>(b) << ',';
      std::cout << " same_owner=" << (identity == context) << '\n';
      TextRunnerPool pool(runner, slots);
      for (const auto length : {30000U, 31200U, 32400U, 33600U, 34800U}) {
        runner->snapshot_positions.clear();
        auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1), {},
                                    {}, context);
        const auto prefix = request.cached_prompt_tokens();
        const auto metrics = FinishPressureRequest(request);
        std::cout << "golden L=" << length << " P=" << prefix
                  << " published=" << metrics.snapshot_bytes / runner->kPayload
                  << " skips=" << metrics.snapshots_skipped << " captures=";
        for (auto position : runner->snapshot_positions)
          std::cout << position << ',';
        std::cout << '\n';
      }
      while (pool.RetainedSnapshotBytes()) {
        (void)pool.TrimSnapshots(1);
        std::cout << "golden retained=" << pool.RetainedSnapshotBytes() << " P="
                  << pool.CachedPrefixTokens(
                         std::vector<TextRunnerToken>(35000, 1), context.get())
                  << '\n';
      }
    }
  }
}

void TestExpertTailHostAdmission() {
  using Runner = HostFloorTailRunner;
  struct Source final : gufo::platform::MemoryReadingSource {
    std::uint64_t available{64 * Runner::kGiB};
    bool recover_after_read{false};
    gufo::platform::MemoryReadings Read() noexcept override {
      gufo::platform::MemoryReadings r;
      r.ram_ok = true;
      r.ram_available = available;
      if (recover_after_read)
        available = 64 * Runner::kGiB;
      return r;
    }
  } source;
  gufo::platform::MemoryGuard guard(&source, 4 * Runner::kGiB);
  gufo::platform::SetProcessMemoryGuard(&guard);
  const auto finish = [](auto& request) {
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
  };
  // Exact review state; last-grid variant; recovery must not undo a current
  // refusal; capacity fallback below/above the materialization floor.
  for (const auto scenario : {0, 1, 2, 3, 4, 5}) {
    TemporaryDirectory directory;
    const TextRunnerDiskCacheOptions options{
        .directory = directory.path(),
        .capacity_bytes = 8 * Runner::kGiB,
        .staging_capacity_bytes = 2 * Runner::kGiB,
        .min_checkpoint_step_tokens = 0};
    source.available = 64 * Runner::kGiB;
    source.recover_after_read = false;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<Runner>(stats, 256);
    runner->capacity_only = scenario == 3 || scenario == 4;
    TextRunnerPool pool(runner, scenario == 5 ? 2 : 1, options);
    // Seed only RAM, so the tested capture starts with empty disk staging.
    // Cancellation keeps the published B without making a full-prompt copy.
    runner->persistence_enabled = false;
    runner->chunk = 32768;
    const std::size_t seed_length = scenario == 1 ? 257 : 30000;
    auto seed = pool.Acquire(std::vector<TextRunnerToken>(seed_length, 1));
    finish(seed);
    (void)seed.Cancel();
    Expect(pool.RetainedSnapshotBytes() == Runner::kPayload,
           "only the protected source snapshot is retained");
    runner->persistence_enabled = true;
    runner->chunk = 8192;
    runner->materialized_positions.clear();
    runner->passes.clear();
    auto request = pool.Acquire(std::vector<TextRunnerToken>(31200, 1));
    const auto source_prefix = seed_length - 256;
    Expect(request.cached_prompt_tokens() == source_prefix &&
               request.cache_restore_bytes() == Runner::kPayload &&
               !request.cache_disk_hit(),
           "extension restores its protected RAM source");
    source.available = scenario == 4 ? 6 * Runner::kGiB : 9 * Runner::kGiB / 2;
    source.recover_after_read = scenario == 2;
    while (request.prefill_position() < 30944)
      (void)request.Prefill(32768);
    finish(
        request);  // Joins an overlapping boundary capture before inspection.
    const auto copies =
        std::ranges::count(runner->materialized_positions, 30944);
    const bool admitted = scenario == 4;
    const char* persist_mode = std::getenv("GUFO_DISK_PERSIST");
    const bool materialized =
        admitted &&
        (!persist_mode || std::string_view(persist_mode) != "evict_idle");
    std::cout << "host admission scenario=" << scenario
              << " P=" << source_prefix << " B=30944 materialized=" << copies
              << " expected=" << materialized
              << " available=" << source.available
              << " floor=" << guard.floor_bytes()
              << " snapshot=" << Runner::kPayload << std::endl;
    Expect(copies == (materialized ? 1 : 0),
           "optional boundary materialization honors host admission");
    if (scenario == 1)
      Expect(std::ranges::count(runner->materialized_positions, 24576) == 0,
             "last all-cut grid also honors host admission");
    const auto retained_prefix = admitted ? 30944U : source_prefix;
    Expect(pool.CachedPrefixTokens(std::vector<TextRunnerToken>(32400, 1),
                                   nullptr) == retained_prefix,
           "optional capture preserves the actual RAM fallback");
    const auto token = request.SelectNext().token;
    if (scenario == 0 || scenario == 5)
      request.Advance();
    const auto metrics =
        scenario == 0 || scenario == 5 ? request.Commit() : request.Cancel();
    Expect(admitted ? metrics.snapshots_skipped == 0
                    : metrics.snapshots_skipped != 0 &&
                          metrics.snapshot_skip_reason == "host_memory_floor",
           "current refusal is counted with its reason");
    auto next = pool.Acquire(std::vector<TextRunnerToken>(32400, 1));
    // Tail final admission cannot evict for the host floor. The source or
    // admitted B remains the real fallback, including after response commit.
    const auto expected_prefix = retained_prefix;
    const auto tier = next.cache_disk_hit() ? "disk"
                      : next.cache_hit()    ? "ram"
                                            : "none";
    std::cout << "host admission response_token=" << token
              << " cache_snapshots_skipped=" << metrics.snapshots_skipped
              << " cache_snapshot_skip_reason=" << metrics.snapshot_skip_reason
              << " next_L=32400 next_P=" << next.cached_prompt_tokens()
              << " next_tier=" << tier << '\n';
    Expect(next.cached_prompt_tokens() == expected_prefix &&
               !next.cache_disk_hit() && next.cache_hit() &&
               next.cache_restore_bytes() == Runner::kPayload,
           "next request reports the real restored prefix, not the ideal B");
    (void)next.Prefill(32768);  // Also checks restored cut/full row provenance.
    (void)next.Cancel();
  }
  gufo::platform::SetProcessMemoryGuard(nullptr);
  std::cout << "PASS: TestExpertTailHostAdmission\n";
}

// Independent acceptance numbers from the review's hand walkthroughs, not a
// second implementation of the planner. No artificial stable prefix seeds disk.
void TestExpertTailBoundaryEconomics(bool disk_only = false) {
  const auto finish = [](auto& request) {
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
    (void)request.Commit();
  };
  for (const auto chunk : {8192U, 4096U})
    for (const bool disk : {false, true}) {
      if (disk_only && !disk)
        continue;
      for (const bool chat : {false, true}) {
        TemporaryDirectory directory;
        const TextRunnerDiskCacheOptions options{
            .directory = directory.path(),
            .capacity_bytes = 2 * 1024 * 1024,
            .staging_capacity_bytes = 1024 * 1024,
            .min_checkpoint_step_tokens = 0};
        auto stats = std::make_shared<FakeStats>();
        auto runner = std::make_shared<ExpertTailRunner>(stats, 256);
        runner->chunk = chunk;
        auto pool = std::make_unique<TextRunnerPool>(
            runner, 1, disk ? std::optional(options) : std::nullopt);
        const std::array<std::size_t, 5> lengths{30000, 31200, 32400, 33600,
                                                 34800};
        const std::array<std::size_t, 5> reuse{0, 29744, 30944, 32144, 33344};
        bool captured_once = true;
        for (std::size_t i = 0; i < lengths.size(); ++i) {
          runner->passes.clear();
          runner->snapshot_positions.clear();
          auto request =
              pool->Acquire(std::vector<TextRunnerToken>(lengths[i], 1), {}, {},
                            {}, true, chat ? lengths[i] - 4 : 0);
          std::cout << "economics chunk=" << chunk << " chat=" << chat
                    << " L=" << lengths[i]
                    << " P=" << request.cached_prompt_tokens() << " tier="
                    << (request.cache_disk_hit() ? "disk" : "ram/live") << '\n';
          Expect(request.cached_prompt_tokens() == reuse[i],
                 "agent loop restores previous B (29744 on first extension)");
          if (disk && i)
            Expect(request.cache_disk_hit(), "boundary restored from disk");
          finish(request);
          std::size_t rows = 0;
          for (const auto [n, cut] : runner->passes) {
            rows += n;
            Expect(n <= chunk + 128, "bounded pass capacity");
          }
          Expect(rows == (i ? 1456U : 30000U), "ideal agent-step prefill rows");
          captured_once &= std::ranges::count(runner->snapshot_positions,
                                              lengths[i] - 256) == 1;
          Expect(pool->RetainedSnapshotBytes() >= sizeof(FakeSnapshot),
                 "boundary is charged to retained snapshot bytes");
          if (disk) {
            pool.reset();
            pool = std::make_unique<TextRunnerPool>(runner, 1, options);
          }
        }
        Expect(captured_once, "boundary gets exactly one normal snapshot");
      }
    }
  if (disk_only)
    return;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ExpertTailRunner>(stats, 512);
  TextRunnerPool pool(runner, 1);
  auto seed = pool.Acquire(std::vector<TextRunnerToken>(24576, 1));
  finish(seed);
  auto next = pool.Acquire(std::vector<TextRunnerToken>(26076, 1));
  Expect(next.cached_prompt_tokens() == 24064,
         "sequence 1 reuses 24064 and prefills 2012 rows");
  finish(next);
  std::cout << "PASS: TestExpertTailBoundaryEconomics (ideal 1456/2012 rows)\n";
}

void TestExpertTailDiskGridFallback() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions options{
      .directory = directory.path(),
      .capacity_bytes = 2 * 1024 * 1024,
      .staging_capacity_bytes = 1024 * 1024,
      .min_checkpoint_step_tokens = 0};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ExpertTailRunner>(stats, 256);
  {
    TextRunnerPool pool(runner, 1, options);
    auto seed = pool.Acquire(std::vector<TextRunnerToken>(30000, 1));
    while (!seed.prefill_complete())
      (void)seed.Prefill(32768);
    (void)seed.Commit();
  }
  TextRunnerPool pool(runner, 1, options);
  std::vector<TextRunnerToken> branch(31200, 1);
  branch[25000] = 2;
  auto request = pool.Acquire(branch);
  std::cout << "grid fallback P=" << request.cached_prompt_tokens()
            << " disk=" << request.cache_disk_hit() << '\n';
  Expect(request.cache_disk_hit() && request.cached_prompt_tokens() == 24576,
         "ordinary last all-cut grid also reaches disk");
  request.Invalidate();
  std::cout << "PASS: TestExpertTailDiskGridFallback\n";
}

void TestExpertTailSnapshotEdges() {
  for (const auto tail : {0U, 256U, 512U}) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<ExpertTailRunner>(stats, tail);
    TextRunnerPool pool(runner, 1);
    const auto length = tail == 256 ? 128U : 8704U;
    auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
    (void)request.Commit();
    const auto count = runner->snapshot_positions.size();
    Expect(count == (tail == 256 ? 1U : 2U),
           "B=0 adds none; grid=B is single; tail0 has legacy grid/full only");
    if (tail == 512)
      Expect(std::ranges::count(runner->snapshot_positions, 8192) == 1,
             "coincident grid/B is captured only once");
    auto exact = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
    Expect(exact.cached_prompt_tokens() == length && exact.prefill_complete(),
           "P>=B uses cached state without a boundary capture");
    (void)exact.Commit();
    Expect(runner->snapshot_positions.size() == count,
           "complete cached prefix adds no snapshots");
  }
  std::cout << "PASS: TestExpertTailSnapshotEdges\n";
}

void TestPromptIdentityOwnership() {
  auto stats = std::make_shared<FakeStats>();
  auto plain = std::make_shared<SnapshotRunner>(stats);
  auto context = std::make_shared<gufo::server::TextPromptContext>();
  context->cache_identity = {1, 2, 3, 4};
  context->cache_prefixes = {{64, {5, 6}}};
  const auto identity = plain->CacheIdentity(30000, context);
  Expect(std::ranges::equal(identity->cache_identity, context->cache_identity),
         "inactive identity remains byte-identical");
  Expect(identity.get() == context.get(),
         "inactive identity references original image bytes without copying");
  namespace qfn = gufo::models::qwen38_flash_next;
  const auto inactive = qfn::ParsePrefillExperts(
      nullptr, nullptr, nullptr, nullptr, nullptr, 10, 48, "512");
  const auto zero_tail = qfn::ParsePrefillExperts(
      "2", "ref10", nullptr, nullptr, nullptr, 10, 48, "0");
  for (const auto& settings : {inactive, zero_tail}) {
    const auto unchanged = gufo::server::MakePrefillExpertTailCacheIdentity(
        settings, 30000, context);
    Expect(
        unchanged.get() == context.get() &&
            unchanged->cache_identity.data() ==
                context->cache_identity.data() &&
            unchanged->cache_prefixes.data() == context->cache_prefixes.data(),
        "inactive/tail0 share original context and every identity vector");
    Expect(
        !gufo::server::MakePrefillExpertTailCacheIdentity(settings, 30000, {}),
        "inactive/tail0 raw prompts allocate no identity context");
  }
  auto runner = std::make_shared<ExpertTailRunner>(stats, 256);
  TextRunnerPool pool(runner, 1);
  auto seed =
      pool.Acquire(std::vector<TextRunnerToken>(30000, 1), {}, {}, context);
  Expect(runner->identity_calls == 1 && runner->model_context == context.get(),
         "one composed identity is shared by lookup and publication");
  while (!seed.prefill_complete())
    (void)seed.Prefill(32768);
  (void)seed.Commit();
  auto next =
      pool.Acquire(std::vector<TextRunnerToken>(31200, 1), {}, {}, context);
  Expect(next.cached_prompt_tokens() == 29744 && runner->identity_calls == 2,
         "vision-composed boundary published in lookup identity");
  next.Invalidate();
  std::cout << "PASS: TestPromptIdentityOwnership\n";
}

// Literal expectations fixed before production changes. This exercises the real
// request planner (including checkpoint-truncated spans), with a CPU backend.
void TestExpertTailRequestBoundary() {
  using Pass = ExpertTailRunner::Pass;
  const std::vector<std::size_t> tails{0, 256, 512, 2048, 4096};
  const std::vector<std::vector<Pass>> expected{
      {{8192, true}, {8192, true}, {8192, true}, {7424, true}},
      {{8192, true}, {8192, true}, {8192, true}, {7168, true}, {256, false}},
      {{8192, true}, {8192, true}, {8192, true}, {6912, true}, {512, false}},
      {{8192, true}, {8192, true}, {8192, true}, {5376, true}, {2048, false}},
      {{8192, true}, {8192, true}, {8192, true}, {3328, true}, {4096, false}}};
  for (std::size_t i = 0; i < tails.size(); ++i)
    for (const auto cached : {0U, 24576U, 31900U}) {
      auto stats = std::make_shared<FakeStats>();
      auto runner = std::make_shared<ExpertTailRunner>(stats, tails[i]);
      TextRunnerPool pool(runner, 1);
      if (cached) {
        // Seed under this same complete request so every cached row has
        // the same policy history, including a frontier inside the tail.
        auto seed = pool.Acquire(std::vector<TextRunnerToken>(32000, 1), {}, {},
                                 {}, true, cached);
        while (seed.prefill_position() < cached)
          (void)seed.Prefill(cached - seed.prefill_position());
        (void)seed.Cancel();
      }
      runner->passes.clear();
      runner->request_lengths.clear();
      auto request = pool.Acquire(std::vector<TextRunnerToken>(32000, 1), {},
                                  {}, {}, true, 32000);
      Expect(request.cached_prompt_tokens() == cached, "tail cached frontier");
      while (!request.prefill_complete())
        (void)request.Prefill(32768);
      const auto want =
          cached == 31900 ? std::vector<Pass>{{100, tails[i] == 0}}
          : cached
              ? std::vector<Pass>(expected[i].begin() + 3, expected[i].end())
              : expected[i];
      Expect(runner->passes == want, "tail real request pass sequence");
      Expect(std::ranges::all_of(runner->request_lengths,
                                 [](auto n) { return n == 32000; }),
             "checkpoint span replaced full request length");
      request.Commit();
    }
  // Protected tail boundary wins over in-pass capture across it. The stable
  // boundary lies two rows after the cut, within the framing capture slack.
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ExpertTailRunner>(stats, 4);
  TextRunnerPool pool(runner, 1);
  auto request = pool.Acquire(std::vector<TextRunnerToken>(8256, 1), {}, {}, {},
                              true, 8254);
  while (!request.prefill_complete())
    (void)request.Prefill(32768);
  Expect(runner->passes == std::vector<Pass>({{8252, true}, {4, false}}),
         "hard tail split and in-pass capture preserve 128-token merge");
  request.Commit();
}

// Failure inventory: stale full-k rows on extension; stale cut rows inside a
// new protected tail; RAM/disk/live lookup disagreement; raw prompt omissions;
// loss of safe all-cut or same-boundary reuse; image identity collisions;
// legacy untagged entries; changes to inactive and zero-tail identities.
void TestExpertTailCacheIdentity() {
  namespace qfn = gufo::models::qwen38_flash_next;
  const auto settings = qfn::ParsePrefillExperts("2", "ref10", nullptr, nullptr,
                                                 nullptr, 10, 48, "512");
  gufo::server::TextPromptContext original;
  original.cache_identity = {19, 20};
  original.cache_prefixes = {{1000, {}}, {25000, {17, 18}}};
  const auto apply = [&](const auto& policy, std::size_t length,
                         const auto& input) {
    auto result = input;
    gufo::server::ApplyPrefillExpertTailCacheIdentity(policy, length, result);
    return result;
  };
  const auto same = [](const auto& a, const auto& b, std::size_t count) {
    return std::ranges::equal(a.CacheIdentity(count), b.CacheIdentity(count));
  };
  auto off = settings;
  off.active = false;
  auto zero = settings;
  zero.tail = 0;
  for (const auto& policy : {off, zero}) {
    const auto unchanged = apply(policy, 32000, original);
    Expect(
        unchanged.cache_identity == original.cache_identity &&
            unchanged.cache_prefixes.size() == original.cache_prefixes.size(),
        "inactive and tail-zero complete identity bytes stay unchanged");
    for (std::size_t i = 0; i < original.cache_prefixes.size(); ++i)
      Expect(unchanged.cache_prefixes[i].token_count ==
                     original.cache_prefixes[i].token_count &&
                 unchanged.cache_prefixes[i].identity ==
                     original.cache_prefixes[i].identity,
             "inactive and tail-zero prefix identity bytes stay unchanged");
  }
  const auto short_request = apply(settings, 25000, original);  // B=24488
  const auto long_request = apply(settings, 32768, original);   // B=32256
  const auto repeat = apply(settings, 25000, original);
  auto other_image = original;
  other_image.cache_identity = {29, 30};
  other_image.cache_prefixes[1].identity = {27, 28};
  const auto changed_image = apply(settings, 25000, other_image);
  for (const auto count : {1U, 1000U, 1001U, 24488U, 24489U, 24576U, 25000U}) {
    Expect(same(short_request, long_request, count) == (count <= 24488),
           "all-cut prefixes share identity; protected prefixes require B");
    Expect(same(short_request, repeat, count),
           "same boundary preserves protected-prefix identity");
    Expect(same(short_request, changed_image, count) == (count <= 1000),
           "tail identity composes with every image prefix identity");
    Expect(!same(short_request, original, count),
           "opt-in tail tag rejects legacy untagged entries");
  }
  TemporaryDirectory directory;
  const auto kpath = directory.path() / "keep.txt";
  const auto read_keep = [&](bool change_last) {
    {
      std::ofstream out(kpath, std::ios::binary);
      for (unsigned layer = 0; layer < 48; ++layer)
        out << (change_last && layer == 47 ? "6\n" : "3\n");
    }
    const auto path = kpath.string();
    return qfn::ParsePrefillExperts(nullptr, "ref10", nullptr, nullptr, nullptr,
                                    10, 48, "512", path.c_str());
  };
  const auto uniform_settings = read_keep(false);
  const auto changed_settings = read_keep(true);
  const auto uniform = apply(uniform_settings, 32000, original);
  const auto changed_keep = apply(changed_settings, 32000, original);
  Expect(uniform_settings.kfile_sha256 != changed_settings.kfile_sha256,
         "different per-layer content changes the canonical policy hash");
  for (const auto count : {1U, 24576U, 31488U, 31489U, 32000U})
    Expect(
        !same(uniform, changed_keep, count),
        "kfile hash automatically separates all-cut and protected identities");
  std::cout << "PASS: TestExpertTailCacheIdentity\n";
}

void TestExpertTailCachePolicyProvenance() {
  enum class Storage { kLive, kRam, kDisk };
  const auto finish = [](auto& request) {
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
  };
  const auto overwrite_live = [&](TextRunnerPool& pool) {
    auto other = pool.Acquire(std::vector<TextRunnerToken>(32, 9));
    finish(other);
    (void)other.Commit();
  };
  const auto verify = [](const ExpertTailRunner& runner, std::size_t length) {
    Expect(runner.observed_policy.size() == length,
           "complete cache test records all restored and executed rows");
    const auto boundary = length - std::min<std::size_t>(length, runner.tail);
    for (std::size_t row = 0; row < length; ++row)
      Expect(runner.observed_policy[row] == (row < boundary),
             "final state retains correct cut/full provenance for every row");
  };
  // Negative controls use the old untagged admission rule in the real cache.
  // The oracle must detect both exact review histories before testing the fix.
  for (const auto seed_length : {24576U, 32768U}) {
    const auto length = seed_length == 24576 ? 32000U : 25000U;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<ExpertTailRunner>(stats, 512);
    runner->policy_identity = false;
    TextRunnerPool pool(runner, 1);
    auto seed = pool.Acquire(std::vector<TextRunnerToken>(seed_length, 1));
    finish(seed);
    (void)seed.Commit();
    auto request = pool.Acquire(std::vector<TextRunnerToken>(length, 1));
    Expect(request.cached_prompt_tokens() == 24576,
           "legacy identity reproduces both unsafe 24576-row cache hits");
    const auto boundary = length - 512;
    std::size_t stale = 0;
    for (std::size_t row = 0; row < runner->observed_policy.size(); ++row)
      stale += runner->observed_policy[row] != (row < boundary);
    Expect(stale == (seed_length == 24576 ? 512U : 88U),
           "provenance oracle detects 512 old full rows or 88 old cut rows");
    request.Invalidate();
  }
  std::cout
      << "PASS: legacy cache controls expose 512 and 88 stale policy rows\n";
  // Both review reproductions are raw completions: no caller-supplied context.
  for (const auto storage : {Storage::kLive, Storage::kRam, Storage::kDisk}) {
    TemporaryDirectory directory;
    const TextRunnerDiskCacheOptions disk{.directory = directory.path(),
                                          .capacity_bytes = 2 * 1024 * 1024,
                                          .staging_capacity_bytes = 1024 * 1024,
                                          .min_checkpoint_step_tokens = 0};
    const bool persistent = storage == Storage::kDisk;
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<ExpertTailRunner>(stats, 512);
    runner->snapshots = storage != Storage::kLive;
    auto pool = std::make_unique<TextRunnerPool>(
        runner, 1, persistent ? std::optional(disk) : std::nullopt);
    // Ordinary raw requests now produce their reusable all-cut disk boundary.
    auto seed = pool->Acquire(std::vector<TextRunnerToken>(24576, 1));
    finish(seed);
    (void)seed.Commit();
    if (storage == Storage::kRam)
      overwrite_live(*pool);
    if (persistent) {
      pool.reset();  // joins the writer, then restores into a clean pool
      pool = std::make_unique<TextRunnerPool>(runner, 1, disk);
    }
    const std::vector<TextRunnerToken> extended(32000, 1);
    Expect(pool->CachedPrefixTokens(extended, nullptr) <= 24064,
           "cache hint excludes prior full-k tail from the new cut");
    auto request = pool->Acquire(extended);
    Expect(request.cached_prompt_tokens() <= 24064,
           "24576/N512 to 32000/N512 replays affected old full-k rows");
    if (storage == Storage::kRam)
      Expect(request.cache_restore_bytes() > 0,
             "extension regression exercises immutable RAM restoration");
    if (persistent)
      Expect(request.cache_disk_hit(),
             "extension regression exercises persistent prefix restoration");
    finish(request);
    verify(*runner, 32000);
    (void)request.Commit();
    if (storage == Storage::kRam)
      overwrite_live(*pool);
    if (persistent) {
      pool.reset();
      pool = std::make_unique<TextRunnerPool>(runner, 1, disk);
    }
    auto exact = pool->Acquire(extended);
    Expect(exact.prefill_complete() && exact.cached_prompt_tokens() == 32000,
           "same-boundary protected complete frontier remains reusable");
    if (storage == Storage::kRam)
      Expect(exact.cache_restore_bytes() > 0,
             "protected rows retain their provenance through RAM restore");
    if (persistent)
      Expect(exact.cache_disk_hit(),
             "protected rows retain their provenance through disk restore");
    verify(*runner, 32000);
    (void)exact.Commit();
  }
  for (const bool persistent : {false, true}) {
    TemporaryDirectory directory;
    const TextRunnerDiskCacheOptions disk{.directory = directory.path(),
                                          .capacity_bytes = 2 * 1024 * 1024,
                                          .staging_capacity_bytes = 1024 * 1024,
                                          .min_checkpoint_step_tokens = 0};
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<ExpertTailRunner>(stats, 512);
    auto pool = std::make_unique<TextRunnerPool>(
        runner, 1, persistent ? std::optional(disk) : std::nullopt);
    auto seed = pool->Acquire(std::vector<TextRunnerToken>(32768, 1), {}, {},
                              {}, true, persistent ? 16384 : 0);
    finish(seed);
    (void)seed.Commit();
    if (persistent) {
      pool.reset();
      pool = std::make_unique<TextRunnerPool>(runner, 1, disk);
      // Prove the old all-cut B exists under a compatible complete length,
      // before demanding rejection of B and 24576 for the shorter request.
      auto compatible = pool->Acquire(std::vector<TextRunnerToken>(33000, 1));
      Expect(compatible.cache_disk_hit() &&
                 compatible.cached_prompt_tokens() == 32256,
             "disk fixture contains the old all-cut boundary checkpoint");
      compatible.Invalidate();
      pool.reset();
      pool = std::make_unique<TextRunnerPool>(runner, 1, disk);
    } else {
      overwrite_live(*pool);
    }
    auto request = pool->Acquire(std::vector<TextRunnerToken>(25000, 1));
    Expect(request.cached_prompt_tokens() == 16384,
           "old all-cut 24576 checkpoint is rejected beyond new B24488");
    Expect(persistent ? request.cache_disk_hit()
                      : request.cache_restore_bytes() > 0,
           "grid regression exercises the requested persistent or RAM tier");
    finish(request);
    verify(*runner, 25000);
    (void)request.Commit();
  }
  // Preserve useful reuse: stop a longer request on an all-cut checkpoint,
  // then stop a same-length request inside its protected tail.
  for (const auto cached : {24576U, 31900U}) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<ExpertTailRunner>(stats, 512);
    TextRunnerPool pool(runner, 1);
    auto seed = pool.Acquire(
        std::vector<TextRunnerToken>(cached == 24576 ? 32768 : 32000, 1), {},
        {}, {}, true, cached);
    while (seed.prefill_position() < cached)
      (void)seed.Prefill(cached - seed.prefill_position());
    (void)seed.Cancel();
    auto request = pool.Acquire(std::vector<TextRunnerToken>(32000, 1));
    Expect(request.cached_prompt_tokens() == cached &&
               request.cache_restore_bytes() == 0,
           "safe all-cut and same-B protected live frontiers stay reusable");
    finish(request);
    verify(*runner, 32000);
    (void)request.Commit();
  }
  std::cout << "PASS: TestExpertTailCachePolicyProvenance\n";
}

void TestNpuPreferredChunkKeepsCheckpointGridAligned() {
  // NPU prefill offloads only exact 4096-token chunks: the intermediate
  // checkpoint grid must follow the preferred chunk, and stay 2048 when off.
  class GridRunner final : public SnapshotRunner {
  public:
    GridRunner(std::shared_ptr<FakeStats> stats, std::size_t preferred)
        : SnapshotRunner(std::move(stats), 64, 256, 4096),
          preferred_(preferred) {}
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = SnapshotRunner::Descriptor();
      descriptor.max_context = 32768;
      return descriptor;
    }
    std::size_t PrefillPreferredChunk() const noexcept override {
      return preferred_;
    }
    std::size_t PrefillBoundaryFallbackChunk() const noexcept override {
      return 2048;
    }

  private:
    std::size_t preferred_;
  };
  const auto spans = [](std::size_t preferred, std::size_t length) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<GridRunner>(stats, preferred);
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(length, 1);
    auto request = pool.Acquire(prompt, {}, {}, {}, true, length - 100);
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
    request.Commit();
    return stats->prefill_spans;
  };
  const auto on_7703 = spans(4096, 7703);
  Expect(!on_7703.empty() && on_7703.front() == 4096,
         "NPU on: a 7,703-token cold prompt starts with a 4096 chunk");
  Expect(on_7703.size() >= 2 && on_7703[1] != 4096,
         "NPU on: the remainder after the 4096 chunk stays on the GPU path");
  const auto on_15384 = spans(4096, 15384);
  Expect(on_15384.size() >= 3 && on_15384[0] == 4096 && on_15384[1] == 4096 &&
             on_15384[2] == 4096,
         "NPU on: a 15,384-token cold prompt gets three 4096 chunks first");
  const auto off_7703 = spans(0, 7703);
  std::size_t widest = 0;
  for (const auto span : off_7703)
    widest = std::max(widest, span);
  Expect(widest <= 2048,
         "NPU off: the original 2048 checkpoint grid is unchanged");

  // GPU alone with a 4096 forward chunk: the runner's checkpoint grid is the
  // forward chunk, so a checkpoint never cuts a 4096 pass short.
  class GpuGridRunner final : public SnapshotRunner {
  public:
    explicit GpuGridRunner(std::shared_ptr<FakeStats> stats)
        : SnapshotRunner(std::move(stats), 64, 256, 4096) {}
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = SnapshotRunner::Descriptor();
      descriptor.max_context = 32768;
      return descriptor;
    }
    std::size_t PrefillCheckpointGrid() const noexcept override { return 4096; }
    std::size_t PrefillBoundaryFallbackChunk() const noexcept override {
      return 4096;
    }
  };
  const auto gpu_spans = [](std::size_t length) {
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<GpuGridRunner>(stats);
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(length, 1);
    auto request = pool.Acquire(prompt, {}, {}, {}, true, length - 100);
    while (!request.prefill_complete())
      (void)request.Prefill(32768);
    request.Commit();
    return stats->prefill_spans;
  };
  const auto gpu_15384 = gpu_spans(15384);
  Expect(gpu_15384.size() >= 3 && gpu_15384[0] == 4096 &&
             gpu_15384[1] == 4096 && gpu_15384[2] == 4096,
         "GPU alone, 4096 grid: a 15,384-token cold prompt gets three 4096 "
         "passes before the checkpoints and tail");
  // Checkpoints at 4096, 8192 and 12288; the last two passes end at the
  // stable boundary (15,284) and the prompt end.
  Expect(gpu_15384 == std::vector<std::size_t>({4096, 4096, 4096, 2996, 100}),
         "GPU alone, 4096 grid: every pass that ends on a checkpoint is a whole "
         "4096-token chunk");
  {
    auto stats = std::make_shared<FakeStats>();
    GridRunner off(stats, 0);
    GridRunner on(stats, 4096);
    Expect(off.PrefillCheckpointGrid() == 2048,
           "the default checkpoint grid without a preferred chunk is 2048");
    Expect(on.PrefillCheckpointGrid() == 4096,
           "the default checkpoint grid follows the preferred chunk");
  }
}

void TestHistoryEditsRestoreIntermediateCheckpoints() {
  class LongSnapshotRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = SnapshotRunner::Descriptor();
      descriptor.max_context = 32768;
      return descriptor;
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<LongSnapshotRunner>(stats, 64, 256, 4096);
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(10000, 1);
  auto original = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  while (!original.prefill_complete())
    (void)original.Prefill(32768);
  original.Commit();
  const auto captures = stats->snapshot_captures;

  auto unchanged = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(unchanged.prefill_complete() &&
             unchanged.cached_prompt_tokens() == prompt.size(),
         "complete prompt retries still reuse every token");
  unchanged.Commit();
  Expect(stats->snapshot_captures == captures,
         "an unchanged retry does not recapture intermediate state");

  prompt[9000] = 2;
  auto late_edit = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(late_edit.cache_hit() && late_edit.cached_prompt_tokens() == 8192,
         "late edits resume from the nearest earlier intermediate checkpoint");
  while (!late_edit.prefill_complete())
    (void)late_edit.Prefill(32768);
  late_edit.Commit();

  prompt[4500] = 3;
  auto earlier_edit = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(
      earlier_edit.cache_hit() && earlier_edit.cached_prompt_tokens() == 4096,
      "committing a late edit preserves earlier usable checkpoints");
  earlier_edit.Invalidate();
  Expect(stats->states_created == 1,
         "intermediate checkpoint entries allocate no extra execution states");

  auto limited_stats = std::make_shared<FakeStats>();
  TextRunnerPool limited_pool(
      std::make_shared<LongSnapshotRunner>(limited_stats, 64, 256,
                                           3 * sizeof(FakeSnapshot)),
      1);
  std::vector<TextRunnerToken> root_prompt(1000, 1);
  auto root = limited_pool.Acquire(root_prompt);
  (void)root.Prefill(32768);
  root.Commit();
  std::vector<TextRunnerToken> extension(10000, 1);
  auto continued = limited_pool.Acquire(extension, {}, {}, {}, true, 9900);
  while (!continued.prefill_complete())
    (void)continued.Prefill(32768);
  continued.Commit();
  extension[1500] = 2;
  auto branched = limited_pool.Acquire(extension, {}, {}, {}, true, 9900);
  Expect(branched.cached_prompt_tokens() == root_prompt.size(),
         "intermediate admission never evicts the reused branching fallback");
  branched.Invalidate();

  auto cancel_stats = std::make_shared<FakeStats>();
  TextRunnerPool cancel_pool(
      std::make_shared<LongSnapshotRunner>(cancel_stats, 64, 256, 4096), 1);
  auto interrupted =
      cancel_pool.Acquire(std::vector<TextRunnerToken>(10000, 1));
  (void)interrupted.Prefill(32768);
  (void)interrupted.Prefill(1024);
  interrupted.Cancel();
  auto resumed = cancel_pool.Acquire(std::vector<TextRunnerToken>(10000, 1));
  Expect(resumed.cached_prompt_tokens() == 2048,
         "cancellation retains completed intermediate state, not partial work");
  resumed.Invalidate();

  auto short_stats = std::make_shared<FakeStats>();
  TextRunnerPool short_pool(
      std::make_shared<LongSnapshotRunner>(short_stats, 64, 256, 4096), 1);
  std::vector<TextRunnerToken> short_prompt(7900, 1);
  auto short_root = short_pool.Acquire(short_prompt);
  while (!short_root.prefill_complete())
    (void)short_root.Prefill(32768);
  short_root.Commit();
  const auto before_short = short_stats->snapshot_captures;
  short_stats->prefill_spans.clear();
  short_prompt.resize(8400, 1);
  auto short_turn = short_pool.Acquire(short_prompt);
  Expect(short_turn.cached_prompt_tokens() == 7900,
         "short continuation starts from the previous frontier");
  Expect(short_turn.Prefill(32768).decode_ready,
         "crossing a nearby grid point does not split a short continuation");
  short_turn.Commit();
  Expect(short_stats->prefill_spans == std::vector<std::size_t>{500} &&
             short_stats->snapshot_captures == before_short + 1,
         "short continuation captures only its completed prompt");
  short_prompt[7800] = 2;
  auto short_edit = short_pool.Acquire(short_prompt);
  Expect(short_edit.cached_prompt_tokens() == 6144,
         "skipping a redundant warm checkpoint retains earlier edit recovery");
  short_edit.Invalidate();

  auto aligned_stats = std::make_shared<FakeStats>();
  TextRunnerPool aligned_pool(
      std::make_shared<LongSnapshotRunner>(aligned_stats, 64, 256, 4096), 1);
  auto aligned_root =
      aligned_pool.Acquire(std::vector<TextRunnerToken>(1000, 1));
  (void)aligned_root.Prefill(32768);
  aligned_root.Commit();
  std::vector<TextRunnerToken> aligned_prompt(5000, 1);
  auto aligned = aligned_pool.Acquire(aligned_prompt, {}, {}, {}, true, 4096);
  while (!aligned.prefill_complete())
    (void)aligned.Prefill(32768);
  aligned.Commit();
  Expect(aligned_stats->snapshot_captures == 3,
         "a nearby grid point is skipped and the stable boundary is captured "
         "only once");
  aligned_prompt[4096] = 2;
  auto aligned_next =
      aligned_pool.Acquire(aligned_prompt, {}, {}, {}, true, 4096);
  Expect(aligned_next.cached_prompt_tokens() == 4096,
         "an aligned stable boundary remains reusable after assistant changes");
  aligned_next.Invalidate();

  auto tail_stats = std::make_shared<FakeStats>();
  TextRunnerPool tail_pool(
      std::make_shared<LongSnapshotRunner>(tail_stats, 64, 256, 4096), 1);
  auto near_end = tail_pool.Acquire(std::vector<TextRunnerToken>(2100, 1));
  Expect(near_end.Prefill(32768).decode_ready &&
             tail_stats->prefill_spans == std::vector<std::size_t>{2100},
         "a grid point near the prompt end does not split the final prefill");
  near_end.Invalidate();
  tail_stats->prefill_spans.clear();
  auto past_tail = tail_pool.Acquire(std::vector<TextRunnerToken>(2300, 2));
  while (!past_tail.prefill_complete())
    (void)past_tail.Prefill(32768);
  Expect(tail_stats->prefill_spans == std::vector<std::size_t>{2048, 252},
         "a grid point farther from the prompt end is still retained");
  past_tail.Invalidate();
}

/// A client that rewrites the assistant turn, as one that drops reasoning
/// does, diverges after the previous turn's stable boundary. Each warm turn
/// must therefore checkpoint its own boundary: if only the reused frontier is
/// retained, every later turn falls back to the same early position and the
/// re-prefilled tail grows for the rest of the conversation.
void TestWarmHitAdvancesTheStableCheckpoint() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);

  // Cold turn. The stable boundary is 3; 40,41 is the assistant framing.
  auto first = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!first.Prefill(64).decode_ready, "cold chat stops at its boundary");
  Expect(first.Prefill(64).decode_ready, "cold chat completes its suffix");
  const auto reply = first.SelectNext().token;
  first.Advance();
  first.Commit();

  // Warm turn replaying the assistant verbatim, then a new user turn. Its
  // stable boundary is 7, past the frontier of 6 that it reuses.
  auto second =
      pool.Acquire({1, 2, 3, 40, 41, reply, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(second.cached_prompt_tokens() == 6,
         "the warm turn resumes from the generated frontier");
  Expect(!second.Prefill(64).decode_ready,
         "the warm turn stops to checkpoint its stable boundary");
  Expect(second.Prefill(64).decode_ready, "the warm turn prefills its suffix");
  (void)second.SelectNext();
  second.Advance();
  second.Commit();

  // The client now drops the reasoning from that assistant turn, so 40,41
  // becomes 50,51 and the prompt diverges at token 7 -- exactly the previous
  // turn's stable boundary.
  auto third = pool.Acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 40, 41}, {},
                            {}, {}, true, 10);
  Expect(third.cached_prompt_tokens() == 7,
         "a warm turn checkpoints its own stable boundary, not only the "
         "frontier it reused");
  Expect(!third.Prefill(64).decode_ready,
         "the rewritten turn stops at its own boundary in turn");
  Expect(third.Prefill(64).decode_ready, "the rewritten turn prefills");
  (void)third.SelectNext();
  third.Advance();
  third.Commit();

  // Dropping reasoning again must keep advancing rather than falling back to
  // the original frontier for a third time.
  auto fourth =
      pool.Acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 60, 61, 11, 40, 41},
                   {}, {}, {}, true, 13);
  Expect(fourth.cached_prompt_tokens() == 10,
         "successive rewritten turns resume from the latest boundary");
  fourth.Cancel();
}

void TestChatFallbackSurvivesSnapshotBudgetPressure() {
  auto stats = std::make_shared<FakeStats>();
  auto runner =
      std::make_shared<SnapshotRunner>(stats, 64, 256, sizeof(FakeSnapshot));
  TextRunnerPool pool(runner, 1);
  auto root = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!root.Prefill(64).decode_ready, "cold chat stops at its fallback");
  Expect(root.Prefill(64).decode_ready, "cold chat completes prefill");
  root.CapturePromptSnapshot();
  Expect(stats->snapshot_captures == 1,
         "a full prompt is not copied when only the fallback fits");
  const auto committed = root.Commit();
  Expect(committed.snapshot_bytes == sizeof(FakeSnapshot),
         "checkpoint retention respects the original one-snapshot budget");
  auto dropped = pool.Acquire({1, 2, 3, 50, 51, 9}, {}, {}, {}, true, 5);
  Expect(dropped.cached_prompt_tokens() == 3,
         "the optional full checkpoint never evicts the required fallback");
  dropped.Invalidate();
}

void TestFullChatCheckpointRestoresWithoutSuffixPrefill() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 4096,
                                              .min_checkpoint_step_tokens = 0};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool writer(runner, 1, disk_cache);
    auto prompt = writer.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    Expect(!prompt.Prefill(64).decode_ready, "writer saves the stable prefix");
    Expect(prompt.Prefill(64).decode_ready, "writer saves the complete prompt");
    (void)prompt.SelectNext();
    prompt.Advance();
    prompt.Commit();
  }
  TextRunnerPool reader(runner, 1, disk_cache);
  auto exact = reader.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(exact.cache_disk_hit() && exact.cached_prompt_tokens() == 5 &&
             exact.prefill_complete(),
         "disk restores the exact prompt instead of recomputing its suffix");
  const auto token = exact.SelectNext().token;
  exact.Advance();
  exact.Commit();
  auto repeated = reader.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!repeated.cache_disk_hit() && repeated.prefill_complete(),
         "a disk-restored full prompt is retained for cheap RAM retries");
  Expect(repeated.SelectNext().token == token,
         "disk and RAM checkpoints select the same token");
  repeated.Advance();
  repeated.Commit();
  auto changed = reader.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(changed.cache_disk_hit() && changed.cached_prompt_tokens() == 3,
         "the earlier disk fallback survives exact full-prompt restoration");
  changed.Invalidate();

  TemporaryDirectory legacy_directory;
  const TextRunnerDiskCacheOptions legacy_disk{
      .directory = legacy_directory.path(),
      .capacity_bytes = 8192,
      .staging_capacity_bytes = 4096,
      .min_checkpoint_step_tokens = 0};
  {
    TextRunnerPool writer(runner, 1, legacy_disk);
    auto old = writer.Acquire({1, 2, 3, 40, 41});
    Expect(old.Prefill(64).decode_ready,
           "legacy writer saves only the full prompt");
    old.Commit();
  }
  TextRunnerPool migrated(runner, 1, legacy_disk);
  auto old = migrated.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!old.cache_hit(),
         "a legacy full-prompt file without a fallback still migrates safely");
  old.Invalidate();
}

void TestNewImageGetsAStableCheckpoint() {
  class ContextRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;
    void SetPromptContext(
        TextRunnerState&,
        std::shared_ptr<const gufo::server::TextPromptContext>) const override {
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ContextRunner>(stats, 64, 256, 1024);
  TextRunnerPool pool(runner, 1);
  {
    auto text = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    while (!text.prefill_complete())
      (void)text.Prefill(64);
    text.Commit();
  }
  auto image = std::make_shared<gufo::server::TextPromptContext>();
  image->cache_identity = {10};
  image->cache_prefixes = {{5, {}}};
  const std::vector<TextRunnerToken> prompt{1, 2, 3, 50, 51, 60, 61, 40, 41};
  {
    auto appended = pool.Acquire(prompt, {}, {}, image, true, 7);
    Expect(appended.cached_prompt_tokens() == 3,
           "new image restores the preceding text checkpoint");
    Expect(!appended.Prefill(64).decode_ready,
           "new image stops before mutable assistant framing");
    Expect(appended.Prefill(64).decode_ready,
           "assistant framing completes after saving the image checkpoint");
    appended.Commit();
  }
  {
    auto retry = pool.Acquire(prompt, {}, {}, image, true, 7);
    Expect(retry.prefill_complete() && retry.cached_prompt_tokens() == 9,
           "exact image retry keeps its complete prompt checkpoint");
    retry.Commit();
  }
  {
    auto followup = pool.Acquire({1, 2, 3, 50, 51, 60, 61, 52, 53, 40, 41}, {},
                                 {}, image, true, 9);
    Expect(followup.cached_prompt_tokens() == 7,
           "rewritten assistant framing does not reprocess the earlier image");
    Expect(!followup.Prefill(64).decode_ready,
           "the warm turn stops to checkpoint its own stable boundary");
    Expect(followup.Prefill(64).decode_ready,
           "assistant framing completes after the boundary is saved");
    followup.Commit();
  }
}

void TestSharedPrefixIsLearnedAndRestoredAcrossConversations() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{
      .directory = directory.path(),
      .capacity_bytes = 8192,
      .staging_capacity_bytes = 4096,
      .shared_prefix_min_tokens = 2,
      .shared_prefix_max_boundaries = 4,
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  // Conversation A: system prefix {7, 7, 7} plus its own turn.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto request = pool.Acquire({7, 7, 7, 1, 2});
    Expect(!request.cache_hit(), "first conversation is cold");
    Expect(request.Prefill(8).consumed_tokens == 5,
           "nothing is shared yet, so prefill runs uninterrupted");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 0,
           "no shared prefix exists after one conversation");
  }

  // Conversation B shares only the system prefix. Prefill stops there so the
  // prefix is persisted for the next conversation.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto request = pool.Acquire({7, 7, 7, 3, 4, 5});
    Expect(!request.cache_hit(), "second conversation still has no prefix");
    const auto first = request.Prefill(8);
    Expect(first.consumed_tokens == 3 && !first.decode_ready,
           "prefill stops at the learned shared prefix");
    const auto second = request.Prefill(8);
    Expect(second.consumed_tokens == 3 && second.decode_ready,
           "prefill resumes after the boundary");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 1 &&
               commit.shared_prefix_bytes > 0 && commit.shared_prefix_ms >= 0.0,
           "the shared prefix was written once during prefill");
  }

  // Conversation C restores the shared prefix and prefills only its turn.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto exact = pool.Acquire({7, 7, 7});
    Expect(exact.cache_disk_hit() && exact.prefill_complete() &&
               exact.SelectNext().token == 90,
           "an exact shared-prefix restore has a current decode frontier");
    exact.Invalidate();
    auto request = pool.Acquire({7, 7, 7, 9});
    Expect(request.cache_hit() && request.cache_disk_hit() &&
               request.cached_prompt_tokens() == 3,
           "third conversation restores the shared prefix from disk");
    const auto step = request.Prefill(8);
    Expect(step.consumed_tokens == 1 && step.decode_ready,
           "only the conversation's own turn is prefilled");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 0,
           "a restored prefix is not written again");
  }
}

void TestRamLearnsDivergenceBoundaries() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "ram-boundary", std::size_t{1} << 20, 4096);
  TextRunnerPool pool(runner, 1);
  // Conversations share a system prompt, then each has its own tail. Below
  // 2048 tokens no grid checkpoint exists, so only learning can help.
  const auto conversation = [](std::size_t shared, TextRunnerToken tail) {
    std::vector<TextRunnerToken> prompt;
    for (std::size_t index = 0; index < shared; ++index)
      prompt.push_back(static_cast<TextRunnerToken>(100 + index % 50));
    for (TextRunnerToken index = 0; index < 600; ++index)
      prompt.push_back(tail + index);
    return prompt;
  };
  const auto run = [&](const std::vector<TextRunnerToken>& prompt) {
    auto request = pool.Acquire(prompt);
    const auto cached = request.cached_prompt_tokens();
    std::vector<std::size_t> steps;
    while (!request.prefill_complete())
      steps.push_back(request.Prefill(4096).consumed_tokens);
    (void)request.Commit();
    return std::pair{cached, steps};
  };

  const auto [first_cached, first_steps] = run(conversation(1000, 10000));
  Expect(first_cached == 0 && first_steps == std::vector<std::size_t>{1600},
         "the first conversation has nothing to share");
  const auto [second_cached, second_steps] = run(conversation(1000, 20000));
  Expect(second_cached == 0 &&
             second_steps == std::vector<std::size_t>({1000, 600}),
         "the second conversation stops at the divergence point to retain it");
  const auto [third_cached, third_steps] = run(conversation(1000, 30000));
  Expect(third_cached == 1000 && third_steps == std::vector<std::size_t>{600},
         "later conversations restore the learned boundary exactly");
  const auto [short_cached, short_steps] = run(conversation(300, 40000));
  Expect(short_cached == 0 && short_steps == std::vector<std::size_t>{900},
         "a short shared prefix is not worth an extra checkpoint");
  // An edit near the end diverges inside this request's own final tokens;
  // its stable checkpoint covers that, so no extra copy is taken.
  auto edited = conversation(1000, 30000);
  edited.back() += 1;
  auto request = pool.Acquire(edited);
  Expect(request.cached_prompt_tokens() == 1000 &&
             request.Prefill(4096).consumed_tokens == 600,
         "a late divergence does not stop prefill for another checkpoint");
  request.Invalidate();
}

void TestLearnedBoundarySurvivesExactRepublication() {
  // A request for exactly the learned prefix publishes the same tokens again.
  // Uncached, it replaces the learned checkpoint, in place or, with a full
  // budget, after admission reclaims the old copy. Either way the copy must
  // stay a branch point, or it ranks as redundant once the older branch is
  // gone and the next new branch retires it.
  for (const auto [full, reuse] :
       {std::pair{false, true}, std::pair{false, false}, std::pair{true, true},
        std::pair{true, false}}) {
    auto stats = std::make_shared<FakeStats>();
    // Room for five snapshots, so unrelated prompts force evictions, or for
    // three, so republication itself must evict.
    auto runner = std::make_shared<PersistentSnapshotRunner>(
        stats, "republished-boundary", (full ? 3 : 5) * sizeof(FakeSnapshot),
        4096);
    TextRunnerPool pool(runner, 1);
    const std::vector<TextRunnerToken> shared(1000, 100);
    const auto run = [&](std::vector<TextRunnerToken> prompt,
                         bool reuse_prompt = true) {
      auto request = pool.Acquire(std::move(prompt), {}, {}, {}, reuse_prompt);
      const auto cached = request.cached_prompt_tokens();
      while (!request.prefill_complete())
        (void)request.Prefill(4096);
      (void)request.Commit();
      return cached;
    };
    const auto branch = [&](TextRunnerToken tail) {
      auto prompt = shared;
      prompt.insert(prompt.end(), 600, tail);
      return prompt;
    };
    (void)run(branch(10000));
    (void)run(branch(20000));  // Learns the branch point after 1000 tokens.
    if (full)
      (void)run({9, 9, 9});  // Fills the budget and evicts the older branch.
    (void)run(shared, reuse);
    if (!full) {
      for (const TextRunnerToken token : {9U, 8U, 7U, 6U})
        (void)run({token, token, token});
    }
    Expect(run(branch(30000)) == 1000,
           "republishing the learned prefix keeps it a branch point");
    Expect(run(branch(40000)) == 1000,
           "a new branch does not retire the republished branch point");
  }
}

void TestCoincidentCacheBoundariesShareOneCopy() {
  for (const bool stable : {false, true}) {
    TemporaryDirectory directory;
    const TextRunnerDiskCacheOptions disk_cache{
        .directory = directory.path(),
        .capacity_bytes = 1024 * 1024,
        .staging_capacity_bytes = 64 * 1024,
        .shared_prefix_min_tokens = 2048,
    };
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PersistentSnapshotRunner>(
        stats, "artifact-A", 1024, 4096);
    std::vector<TextRunnerToken> prompt(2050, 7);
    prompt[2048] = 1;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto source = pool.Acquire(prompt);
      while (!source.prefill_complete())
        (void)source.Prefill(4096);
      source.Commit();
    }
    prompt[2048] = 2;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto branch = pool.Acquire(prompt, {}, {}, {}, true, stable ? 2048 : 0);
      const auto before = stats->snapshot_captures;
      Expect(branch.Prefill(4096).consumed_tokens == 2048,
             "disk boundary coincides with a history or stable checkpoint");
      Expect(branch.Prefill(4096).consumed_tokens == 2 &&
                 stats->snapshot_captures == before + 1,
             "one immutable copy serves both coincident boundaries");
      branch.Commit();
    }
    prompt[2048] = 3;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto restored = pool.Acquire(prompt);
      Expect(restored.cache_disk_hit() &&
                 restored.cached_prompt_tokens() == 2048 &&
                 restored.Prefill(4096).consumed_tokens == 2,
             "the shared copy survives a restart with its exact position");
      restored.Commit();
    }
  }
}

void TestDiskOnlyCaptureReservesBudgetBeforeCommit() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  std::binary_semaphore entered(0), release(0);
  runner->before_serialize = [&] {
    entered.release();
    release.acquire();
  };
  TextRunnerPool pool(runner, 2,
                      TextRunnerDiskCacheOptions{
                          .directory = directory.path(),
                          .capacity_bytes = 4096,
                          .staging_capacity_bytes = 192,
                      });
  auto first = pool.Acquire({1, 2, 3});
  (void)first.Prefill(3);
  first.CapturePromptSnapshot();
  // GUFO_DISK_DEFER=1 holds serialization back while a request is generating (this test keeps `first` open), so the immediate-entry check is for the default mode.
  const char* defer_env = std::getenv("GUFO_DISK_DEFER");
  if (!(defer_env && std::string_view(defer_env) == "1"))
    Expect(entered.try_acquire_for(std::chrono::seconds(2)),
             "disk-only prompt is queued immediately after capture");
  auto second = pool.Acquire({4, 5, 6});
  (void)second.Prefill(3);
  second.CapturePromptSnapshot();
  const auto second_commit = second.Commit();
  Expect(stats->snapshot_captures == 1 && second_commit.disk_queued_bytes == 0,
         "pending disk-only state prevents another unbudgeted capture");
  release.release();
  Expect(first.Commit().disk_queued_bytes != 0,
         "request reports its earlier background persistence admission");
}

void TestDiskPreflightAvoidsUnusableCapture() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  TextRunnerPool pool(runner, 1,
                      TextRunnerDiskCacheOptions{
                          .directory = directory.path(),
                          .capacity_bytes = 4096,
                          .staging_capacity_bytes = 96,
                      });
  auto request = pool.Acquire({1, 2, 3});
  (void)request.Prefill(3);
  const auto commit = request.Commit();
  Expect(stats->snapshot_captures == 0 && commit.disk_queued_bytes == 0,
         "disk staging refusal happens before snapshot capture");
}

void TestMeasuredStateIsReconciledWithClaim() {
  auto stats = std::make_shared<FakeStats>();
  bool rejected = false;
  try {
    auto runner = std::make_shared<FakeRunner>(stats, 65, 256);
    TextRunnerPool pool(runner, 1);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "measured state cannot exceed its proposed allocation");
  Expect(stats->states_created == 1,
         "measured resource check runs immediately after allocation");
}

void TestSnapshotBudgetRefusalDoesNotFailCompletedRequest() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats, 64, 256,
                                                 sizeof(FakeSnapshot) - 1);
  TextRunnerPool pool(runner, 1);

  auto request = pool.Acquire({1, 2, 3});
  Expect(request.Prefill(3).decode_ready,
         "request completes before optional snapshot retention");
  const auto commit = request.Commit();
  Expect(commit.snapshot_bytes == 0 && commit.snapshot_ms == 0.0,
         "budget refusal succeeds without reporting a retained snapshot");
  Expect(stats->snapshot_size_queries == 1 && stats->snapshot_captures == 0,
         "cache admission happens before snapshot allocation");

  auto extension = pool.Acquire({1, 2, 3, 4});
  Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 3 &&
             extension.cache_restore_bytes() == 0,
         "live state remains reusable when snapshot admission is refused");
  extension.Invalidate();
}

class FlakySnapshotRunner final : public SnapshotRunner {
public:
  using SnapshotRunner::SnapshotRunner;

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    ++stats_->snapshot_captures;
    if (stats_->snapshot_captures == 1) {
      throw std::runtime_error("synthetic snapshot capture failure");
    }
    const auto& fake = RequireFakeState(state);
    return std::make_unique<FakeSnapshot>(fake.position, fake.decode_count,
                                          fake.frontier);
  }
};

void TestSnapshotCaptureFailureReleasesReservationAndKeepsRequestSuccessful() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FlakySnapshotRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto first = pool.Acquire({1, 2, 3});
    Expect(first.Prefill(3).decode_ready, "first request reaches checkpoint");
    const auto commit = first.Commit();
    Expect(commit.snapshot_bytes == 0 && commit.snapshot_ms >= 0.0,
           "snapshot exception does not fail the completed request");
  }

  {
    auto second = pool.Acquire({4, 5});
    Expect(!second.cache_hit(), "failed capture retained no partial entry");
    Expect(second.Prefill(2).decode_ready,
           "second request executes normally after capture failure");
    const auto commit = second.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot),
           "released reservation admits a later successful snapshot");
  }

  auto extension = pool.Acquire({4, 5, 6});
  Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 2,
         "later retained snapshot restores after the failed attempt");
  extension.Invalidate();
}

/// Evicting a retained prefix because the entry table is full is not routine:
/// it means the server is configured below its workload and is doing avoidable
/// full re-prefills. It must be visible, unlike exact replacement.
void TestEntryCapacityEvictionIsLogged() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats, 64, 256, 8192);
  // Checkpoint records are bounded independently of execution sessions.
  TextRunnerPool pool(runner, 1);

  for (std::size_t i = 0;
       i <= gufo::server::TextRunnerRamCacheOptions::kMaxEntries; ++i) {
    const std::vector<TextRunnerToken> prefix{
        static_cast<TextRunnerToken>(i + 1), 0};
    auto request = pool.Acquire(prefix);
    Expect(request.Prefill(prefix.size()).decode_ready,
           "each distinct prefix reaches its snapshot boundary");
    (void)request.SelectNext();
    request.Advance();
    Expect(request.Commit().snapshot_bytes == sizeof(FakeSnapshot),
           "each distinct prefix retains a snapshot");
  }

  // The first prefix was evicted, so it can no longer be reused.
  auto evicted = pool.Acquire({1, 0});
  Expect(!evicted.cache_hit(),
         "the oldest retained prefix is gone once the entries are full");
  evicted.Invalidate();
}

void TestSnapshotCacheCapacityIsReportedAtStartup() {
  using gufo::server::TextRunnerRamCacheOptions;
  // Automatic sizing takes the model budget; an explicit limit may use the
  // larger ceiling the model reports, or the budget when it reports none.
  for (const std::optional<std::size_t> ceiling :
       {std::optional<std::size_t>{}, std::optional{std::size_t{128} << 30}}) {
    for (const std::size_t budget :
         {std::size_t{0}, std::size_t{256}, std::size_t{64} << 30}) {
      for (const std::size_t requested :
           {std::size_t{0}, std::size_t{64}, std::size_t{48} << 30,
            std::size_t{96} << 30}) {
        for (const std::size_t sessions : {1U, 2U}) {
          auto stats = std::make_shared<FakeStats>();
          auto runner =
              std::make_shared<SnapshotRunner>(stats, 64, 256, budget);
          runner->retained_snapshot_ceiling_bytes = ceiling;
          std::ostringstream startup_log;
          gufo::server::Logger::FlushSync();
          auto* previous = std::clog.rdbuf(startup_log.rdbuf());
          {
            TextRunnerPool pool(runner, sessions, std::nullopt,
                                {.capacity_bytes = requested});
          }
          gufo::server::Logger::FlushSync();
          std::clog.rdbuf(previous);
          const auto output = startup_log.str();
          const auto automatic =
              std::min(budget, TextRunnerRamCacheOptions::kAutomaticMaxBytes);
          const auto maximum = ceiling.value_or(budget);
          const auto capacity =
              requested == 0 ? automatic : std::min(requested, maximum);
          const std::string expected =
              "event=snapshot_cache_configured sessions=" +
              std::to_string(sessions) +
              " snapshot_entries=128 capacity_bytes=" +
              std::to_string(capacity) +
              " automatic_bytes=" + std::to_string(automatic) +
              " max_bytes=" + std::to_string(maximum) + "\n";
          const auto position = output.find(expected);
          Expect(position != std::string::npos &&
                     output.find(expected, position + expected.size()) ==
                         std::string::npos,
                 "startup reports actual session, entry and byte limits once");
          Expect(output.find("retained_conversations") == std::string::npos,
                 "startup does not present session count as conversation "
                 "capacity");
          Expect(stats->states_created == sessions,
                 "checkpoint record capacity never creates extra execution "
                 "states");
        }
      }
    }
  }
}

void TestMeminfoAvailableExcludesFreeCma() {
  using gufo::server::MeminfoAvailableBytes;
  constexpr std::uint64_t kib = 1024;
  // Ubuntu 26.04's KHO scratch reports free CMA pages with CmaTotal 0.
  Expect(MeminfoAvailableBytes("MemTotal:       131072000 kB\n"
                               "MemAvailable:   100000000 kB\n"
                               "CmaTotal:               0 kB\n"
                               "CmaFree:         13034628 kB\n") ==
             (100000000 - 13034628) * kib,
         "free CMA pages are not available for snapshots");
  Expect(MeminfoAvailableBytes("MemAvailable:   8000000 kB\n") == 8000000 * kib,
         "MemAvailable is used unchanged without a CmaFree line");
  Expect(
      MeminfoAvailableBytes("CmaFree:  9000 kB\nMemAvailable:  4000 kB\n") == 0,
      "free CMA beyond MemAvailable saturates at zero");
  Expect(!MeminfoAvailableBytes("MemTotal:  4000 kB\nCmaFree:  10 kB\n"),
         "missing MemAvailable keeps the caller's fallback");
}

void TestSnapshotStartupReportsSelectedLimits() {
  class ChangingHeadroomRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;

    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = SnapshotRunner::ResourceClaim();
      constexpr std::size_t gib = std::size_t{1} << 30;
      // Synthetic headroom falls during state allocation, then again after
      // cache sizing. No large buffers are allocated for these resource claims.
      // Each claim follows Flash-Next's half-free / free-minus-4-GiB policy.
      const auto available = stats_->states_created == 0 ? 14 * gib
                             : post_state_claims++ == 0  ? 12 * gib
                                                         : 11 * gib;
      claim.retained_snapshot_capacity_bytes = available / 2;
      claim.retained_snapshot_ceiling_bytes =
          available - gufo::server::kHostSnapshotHeadroomBytes;
      return claim;
    }

    mutable std::size_t post_state_claims{0};
  };

  constexpr std::size_t gib = std::size_t{1} << 30;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ChangingHeadroomRunner>(stats);
  std::ostringstream startup_log;
  gufo::server::Logger::FlushSync();
  auto* previous = std::clog.rdbuf(startup_log.rdbuf());
  {
    TextRunnerPool pool(runner, 1, std::nullopt, {.capacity_bytes = 8 * gib});
  }
  gufo::server::Logger::FlushSync();
  std::clog.rdbuf(previous);

  const auto output = startup_log.str();
  const auto event = output.find("event=snapshot_cache_configured ");
  Expect(event != std::string::npos, "startup reports snapshot limits");
  const auto line = output.substr(event, output.find('\n', event) - event);
  const auto field = [&](std::string_view name) {
    const auto start = line.find(name);
    Expect(start != std::string::npos, "startup reports the requested field");
    std::istringstream value(line.substr(start + name.size()));
    std::size_t bytes = 0;
    Expect(static_cast<bool>(value >> bytes), "startup byte field is numeric");
    return bytes;
  };
  const auto capacity = field("capacity_bytes=");
  const auto automatic = field("automatic_bytes=");
  const auto maximum = field("max_bytes=");
  Expect(stats->states_created == 1, "only the requested state was created");
  Expect(capacity == 8 * gib,
         "cache capacity uses the first post-allocation memory observation");
  Expect(capacity <= maximum,
         "startup maximum must not be below the selected cache capacity");
  Expect(automatic == 6 * gib && maximum == 8 * gib,
         "startup limits describe the claim that selected cache capacity");
}

// L-REL-5: capacity from totals, not from free memory.
void TestSnapshotCapacityPlanIsDeterministic() {
  using gufo::server::PlanSnapshotCapacity;
  using gufo::server::SnapshotCapacityInputs;
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  constexpr std::uint64_t mib = std::uint64_t{1} << 20;
  constexpr std::uint64_t reserve = gufo::server::kHostSnapshotHeadroomBytes;
  Expect(reserve == 4 * gib, "the fixed reserve is 4 GiB");

  // Fixed total: the plan has no free-memory input, so a host that has
  // 2, 10 or 30 GiB available gets the same capacity.
  for (const std::uint64_t available : {2 * gib, 10 * gib, 30 * gib}) {
    (void)available;
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 128 * gib,
                                            .model_footprint_bytes = 28 * gib});
    Expect(plan.ceiling_bytes == 96 * gib,
           "ceiling = total - footprint - reserve");
    Expect(plan.automatic_bytes == 32 * gib && plan.capacity_bytes == 32 * gib,
           "automatic = min(32 GiB, ceiling / 2)");
    Expect(plan.clamp.empty(), "no clamp on a roomy machine");
  }
  {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 64 * gib,
                                            .model_footprint_bytes = 30 * gib});
    Expect(plan.ceiling_bytes == 30 * gib && plan.automatic_bytes == 15 * gib,
           "a 64 GiB machine halves its smaller ceiling");
  }
  // The default profile's 6442450944 is exact whenever the machine can hold it.
  {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 64 * gib,
                                            .model_footprint_bytes = 30 * gib,
                                            .explicit_bytes = 6442450944ull});
    Expect(plan.capacity_bytes == 6442450944ull && plan.clamp.empty(),
           "an explicit 6 GiB cache is exact when the total allows");
  }
  {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 37 * gib,
                                            .model_footprint_bytes = 30 * gib,
                                            .explicit_bytes = 6442450944ull});
    Expect(plan.ceiling_bytes == 3 * gib && plan.capacity_bytes == 3 * gib &&
               plan.clamp == "explicit_over_ceiling",
           "an explicit limit above the ceiling clamps and says why");
  }
  // Small and negative ceilings never wrap: the cache is disabled.
  for (const std::uint64_t footprint :
       {34 * gib, 36 * gib, 200 * gib, ~std::uint64_t{0}}) {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 36 * gib,
                                            .model_footprint_bytes = footprint,
                                            .explicit_bytes = 1 * gib});
    Expect(plan.ceiling_bytes == 0 && plan.capacity_bytes == 0 &&
               plan.automatic_bytes == 0 &&
               plan.clamp == "disabled_ceiling_below_minimum",
           "a small or negative ceiling disables the cache without wrapping");
  }
  {
    const auto plan = PlanSnapshotCapacity(
        {.host_total_bytes = 36 * gib,
         .model_footprint_bytes = 31 * gib + 600 * mib});
    Expect(plan.capacity_bytes == 0,
           "a ceiling just under the 512 MiB minimum disables the cache");
    const auto kept = PlanSnapshotCapacity(
        {.host_total_bytes = 36 * gib,
         .model_footprint_bytes = 31 * gib + 400 * mib});
    Expect(kept.ceiling_bytes > gufo::server::kMinSnapshotCacheBytes &&
               kept.capacity_bytes == kept.automatic_bytes,
           "a ceiling over the minimum keeps the cache");
  }
  {
    // Flash-Next: the device total bounds the ceiling too.
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 128 * gib,
                                            .model_footprint_bytes = 28 * gib,
                                            .device_total_bytes = 64 * gib});
    Expect(plan.ceiling_bytes == 32 * gib && plan.limit == "device",
           "device total - footprint - reserve");
  }
  {
    // Production 2026-10-14 (Windows, 96 GiB carve): the 90 GiB model is in
    // the carve, not in the 31.6 GiB Windows sees, so the explicit 6 GiB holds.
    const auto plan =
        PlanSnapshotCapacity({.host_total_bytes = 33982058496ull,
                              .model_footprint_bytes = 96958458708ull,
                              .device_total_bytes = 115510523904ull,
                              .explicit_bytes = 6442450944ull});
    Expect(plan.ceiling_bytes == 107 * gib - 96958458708ull - reserve &&
               plan.limit == "device" && plan.capacity_bytes == 6442450944ull &&
               plan.clamp.empty(),
           "a carve-out APU counts its weights on the device side only");
  }
  {
    // The WDDM budget moved 786432 bytes between two starts (2026-10-14
    // capplan). Whole-GiB device totals keep the automatic plan identical.
    const auto first =
        PlanSnapshotCapacity({.host_total_bytes = 33982058496ull,
                              .model_footprint_bytes = 96958458708ull,
                              .device_total_bytes = 115510523904ull});
    const auto second =
        PlanSnapshotCapacity({.host_total_bytes = 33982058496ull,
                              .model_footprint_bytes = 96958458708ull,
                              .device_total_bytes = 115510523904ull - 786432});
    Expect(first.ceiling_bytes == second.ceiling_bytes &&
               first.automatic_bytes == second.automatic_bytes &&
               first.capacity_bytes == second.capacity_bytes &&
               first.capacity_bytes == first.automatic_bytes &&
               first.automatic_bytes != 0,
           "device-budget jitter does not move the automatic capacity");
  }
  {
    // A3B (Cyber-Tiel): snapshots are host vectors and the weights sit in the
    // carve, so the runner claims no device total and no footprint. The
    // 31.6 GiB Windows sees bounds the cache alone.
    const auto plan =
        PlanSnapshotCapacity({.host_total_bytes = 33982058496ull});
    Expect(plan.ceiling_bytes == 33982058496ull - reserve &&
               plan.automatic_bytes == (33982058496ull - reserve) / 2 &&
               plan.capacity_bytes == plan.automatic_bytes &&
               plan.limit == "host" && plan.clamp.empty(),
           "host-vector snapshots plan from the host total alone");
  }
  {
    // Gemma 4 26B-A4B: device rows, but a small model in the 107 GiB carve,
    // so the host side is the smaller one.
    const auto plan =
        PlanSnapshotCapacity({.host_total_bytes = 33982058496ull,
                              .model_footprint_bytes = 20 * gib,
                              .device_total_bytes = 115510523904ull});
    Expect(plan.ceiling_bytes == 33982058496ull - reserve &&
               plan.limit == "host",
           "a small device-resident model is bounded by the host total");
  }
  {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 32 * gib,
                                            .host_resident_bytes = 26 * gib,
                                            .model_footprint_bytes = 90 * gib,
                                            .device_total_bytes = 108 * gib,
                                            .explicit_bytes = 6442450944ull});
    Expect(plan.ceiling_bytes == 2 * gib && plan.limit == "host" &&
               plan.capacity_bytes == 2 * gib &&
               plan.clamp == "explicit_over_ceiling",
           "host-resident model bytes count against host RAM");
  }
  // Shared RAM (GTT): the device total is at most the host total, so the
  // footprint still comes out of host RAM through the device side.
  {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 124 * gib,
                                            .model_footprint_bytes = 90 * gib,
                                            .device_total_bytes = 110 * gib});
    Expect(plan.ceiling_bytes == 16 * gib && plan.limit == "device",
           "a shared-RAM device keeps the min(host, device) ceiling");
  }
  for (const std::uint64_t footprint : {200 * gib, ~std::uint64_t{0}}) {
    const auto plan = PlanSnapshotCapacity({.host_total_bytes = 124 * gib,
                                            .model_footprint_bytes = footprint,
                                            .device_total_bytes = 110 * gib});
    Expect(plan.ceiling_bytes == 0 && plan.capacity_bytes == 0 &&
               plan.clamp == "disabled_ceiling_below_minimum",
           "a device footprint over the device total disables the cache");
  }
}

void TestHostFloorDefaultAndParsing() {
  using gufo::server::ParseHostFloorBytes;
  constexpr std::uint64_t mib = std::uint64_t{1} << 20;
  Expect(gufo::server::kDefaultHostFloorMiB == 4096, "floor default is 4096");
  Expect(ParseHostFloorBytes(nullptr) == 4096 * mib, "unset selects 4096 MiB");
  Expect(ParseHostFloorBytes("") == 4096 * mib, "empty counts as unset");
  Expect(ParseHostFloorBytes("0") == 0, "0 disables the floor");
  Expect(ParseHostFloorBytes("512") == 512 * mib, "an explicit value is used");
  Expect(ParseHostFloorBytes("-1") == 0, "a negative value disables");
  Expect(ParseHostFloorBytes("12x") == 0 && ParseHostFloorBytes("abc") == 0,
         "a malformed value disables");
}

void TestPoolPlansSnapshotCapacityFromTotals() {
  constexpr std::size_t gib = std::size_t{1} << 30;
  class TotalsRunner final : public SnapshotRunner {
  public:
    TotalsRunner(std::shared_ptr<FakeStats> stats, std::size_t host_total,
                 std::optional<std::size_t> device_total)
        : SnapshotRunner(std::move(stats)),
          host_total_(host_total),
          device_total_(device_total) {}
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = SnapshotRunner::ResourceClaim();
      claim.resident_weights_bytes = 30 * gib;
      // Legacy values that must be ignored.
      claim.retained_snapshot_capacity_bytes = 1;
      claim.retained_snapshot_ceiling_bytes = 1;
      claim.snapshot_capacity_from_totals = true;
      claim.device_total_bytes = device_total_;
      claim.host_total_bytes_override = host_total_;
      return claim;
    }
    std::size_t host_total_;
    std::optional<std::size_t> device_total_;
  };
  struct Case {
    std::size_t host_total, requested;
    std::optional<std::size_t> device_total;
    std::size_t capacity, automatic, maximum;
    const char* limit;
    const char* clamp;
  };
  // Footprint = 30 GiB weights + 1 session * (64 + 16) bytes.
  const Case cases[] = {
      {128 * gib, 0, std::nullopt, 32 * gib, 32 * gib, 94 * gib - 80, "host",
       "none"},
      {128 * gib, 6442450944ull, std::nullopt, 6442450944ull, 32 * gib,
       94 * gib - 80, "host", "none"},
      {40 * gib, 0, std::nullopt, 3 * gib - 40, 3 * gib - 40, 6 * gib - 80,
       "host", "none"},
      {40 * gib, 8 * gib, std::nullopt, 6 * gib - 80, 3 * gib - 40,
       6 * gib - 80, "host", "explicit_over_ceiling"},
      {20 * gib, 6442450944ull, std::nullopt, 0, 0, 0, "host",
       "disabled_ceiling_below_minimum"},
      {128 * gib, 0, 40 * gib, 3 * gib - 40, 3 * gib - 40, 6 * gib - 80,
       "device", "none"},
      // Carve-out APU: the model is larger than host RAM but sits in device
      // memory, so the profile's 6 GiB still holds.
      {20 * gib, 6442450944ull, 108 * gib, 6442450944ull, 8 * gib, 16 * gib,
       "host", "none"},
  };
  for (const auto& c : cases) {
    auto stats = std::make_shared<FakeStats>();
    auto runner =
        std::make_shared<TotalsRunner>(stats, c.host_total, c.device_total);
    std::ostringstream startup_log;
    gufo::server::Logger::FlushSync();
    auto* previous = std::clog.rdbuf(startup_log.rdbuf());
    {
      TextRunnerPool pool(runner, 1, std::nullopt,
                          {.capacity_bytes = c.requested});
    }
    gufo::server::Logger::FlushSync();
    std::clog.rdbuf(previous);
    const auto output = startup_log.str();
    const std::string configured =
        "capacity_bytes=" + std::to_string(c.capacity) +
        " automatic_bytes=" + std::to_string(c.automatic) +
        " max_bytes=" + std::to_string(c.maximum) + "\n";
    Expect(output.find("event=snapshot_cache_configured") !=
                   std::string::npos &&
               output.find(configured) != std::string::npos,
           "pool capacity follows the totals plan and ignores legacy values");
    const std::string plan_line =
        "event=snapshot_capacity_plan host_total_bytes=" +
        std::to_string(c.host_total) +
        " host_resident_bytes=0 model_footprint_bytes=" +
        std::to_string(30 * gib + 80) +
        " reserve_bytes=" + std::to_string(4 * gib) + " device_total_bytes=" +
        (c.device_total ? std::to_string(*c.device_total) : "none") +
        " explicit_bytes=" + std::to_string(c.requested) + " limit=" + c.limit +
        " ceiling_bytes=" + std::to_string(c.maximum) +
        " automatic_bytes=" + std::to_string(c.automatic) +
        " capacity_bytes=" + std::to_string(c.capacity) + " clamp=" + c.clamp +
        "\n";
    Expect(output.find(plan_line) != std::string::npos,
           "startup logs the capacity plan with its inputs");
  }
}

void TestServerInstructionsAreFraming() {
  using namespace gufo::tokenization;
  class ConstraintRunner final : public FakeRunner {
  public:
    explicit ConstraintRunner(gufo::sampling::JsonConstraint::ToolFormat format)
        : FakeRunner(std::make_shared<FakeStats>()), format_(format) {}
    gufo::sampling::JsonConstraint::ToolFormat ToolFormat() const override {
      return format_;
    }
    std::shared_ptr<const gufo::sampling::ConstraintVocabulary>
    BuildConstraintVocabulary() const override {
      return std::make_shared<gufo::sampling::ConstraintVocabulary>(
          256, [](std::uint32_t id) {
            return gufo::sampling::ConstraintVocabulary::Piece{
                std::string(1, static_cast<char>(id)), false};
          });
    }

  private:
    gufo::sampling::JsonConstraint::ToolFormat format_;
  };
  // Only runners without a native call syntax use the JSON envelope and its
  // instruction; a native runner never switches syntax for a schema (#383).
  const ConstraintRunner runner(
      gufo::sampling::JsonConstraint::ToolFormat::kJson);
  const ConstraintRunner native_runner(
      gufo::sampling::JsonConstraint::ToolFormat::kQwen);
  for (const auto format :
       {gufo::sampling::JsonConstraint::ToolFormat::kQwen,
        gufo::sampling::JsonConstraint::ToolFormat::kDeepSeek}) {
    const ConstraintRunner plain_runner(format);
    for (const bool json : {false, true}) {
      ChatRequest request;
      if (json)
        request.response_format = gufo::sampling::JsonConstraint::Compile(
            gufo::json::parse(
                R"({"type":"object","properties":{},"additionalProperties":false})"),
            false);
      gufo::sampling::SamplingConfig sampling;
      std::optional<gufo::sampling::JsonConstraint::ToolFormat> observed;
      const auto constrained = gufo::server::ConstrainChatRequest(
          request, plain_runner, &sampling, &observed);
      Expect(observed == format && constrained.has_value() == json,
             "plain and JSON answers retain the native output dialect");
    }
  }
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  std::unordered_map<std::string, TokenId> specials;
  for (std::string_view token : std::initializer_list<std::string_view>{
           kImStart, kImEnd, "<think>", "</think>", "<tool_call>",
           "</tool_call>"}) {
    specials.emplace(token, vocab.size());
    vocab.emplace_back(token);
  }
  std::string error;
  auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, {}, specials, &error);
  Expect(tokenizer != nullptr, "instruction tokenizer fixture initializes");
  ChatTemplateOptions options;
  options.enable_thinking = false;
  options.require_tool_call = true;
  for (
      const auto schema :
      {R"({"type":"object","properties":{"text":{"type":"string","const":"\n</parameter>\n</｜DSML｜parameter>\\"}},"required":["text"],"additionalProperties":false})",
       R"({"type":"object","properties":{"text":{"type":"string"}},"patternProperties":{"^x_":{"type":"integer"}},"required":["text"]})"}) {
    for (const auto role :
         {ChatRole::kUser, ChatRole::kSystem, ChatRole::kDeveloper}) {
      for (const bool literal_client_tags : {false, true}) {
        ChatRequest request;
        request.reasoning.enabled = false;
        request.tool_choice = ChatRequest::ToolChoice::kRequired;
        request.tools = {{.name = "record", .parameters_json = schema}};
        const std::string client = literal_client_tags
                                       ? "Client <tool_call>example</tool_call>"
                                       : "Be concise.  ";
        if (role != ChatRole::kUser)
          request.messages.emplace_back(role, client);
        request.messages.emplace_back(
            ChatRole::kUser, role == ChatRole::kUser ? client : "Call record.");
        gufo::sampling::SamplingConfig sampling;
        std::optional<gufo::sampling::JsonConstraint::ToolFormat> format;
        const auto constrained = gufo::server::ConstrainChatRequest(
            request, runner, &sampling, &format);
        Expect(
            constrained &&
                format == gufo::sampling::JsonConstraint::ToolFormat::kJson &&
                sampling.constraint,
            "the actual request path selects and binds the JSON fallback");
        Expect(constrained->messages.front().content ==
                   (role == ChatRole::kUser ? "" : client),
               "server instructions do not mutate client system/developer "
               "content");
        Expect(request.messages.front().content == client,
               "constraint construction leaves the original request untouched");
        const auto& instruction = constrained->messages.front().framing_suffix;
        Expect(instruction.find("<tool_call>") != std::string::npos &&
                   instruction.find("</tool_call>") != std::string::npos,
               "JSON fallback uses the server framing field");
        const auto prepared = gufo::models::qwen::vision::Prepare(
            *tokenizer, constrained->messages, constrained->tools, options, {},
            8192);
        auto without = constrained->messages;
        without.front().framing_suffix.clear();
        const auto baseline = QwenChatTemplate::RenderAndTokenize(
            *tokenizer, without, constrained->tools, options);
        Expect(baseline.has_value(), "client-only template encodes");
        for (const auto* tag : {"<tool_call>", "</tool_call>"}) {
          const auto id = *tokenizer->FindSpecialToken(tag);
          Expect(
              std::count(prepared.tokens.begin(), prepared.tokens.end(), id) ==
                  std::count(baseline->begin(), baseline->end(), id) + 1,
              "each server delimiter is exactly one special token, including "
              "beside literal client tags");
        }
        const auto client_only = QwenChatTemplate::RenderAndTokenize(
            *tokenizer, request.messages, options);
        Expect(client_only.has_value(), "client tag control encodes");
        for (const auto* tag : {"<tool_call>", "</tool_call>"})
          Expect(std::count(client_only->begin(), client_only->end(),
                            *tokenizer->FindSpecialToken(tag)) == 0,
                 "client-supplied delimiter spellings remain ordinary text");
        if (!literal_client_tags) {
          auto legacy = constrained->messages;
          legacy.front().content += legacy.front().framing_suffix;
          legacy.front().framing_suffix.clear();
          const auto rendered =
              QwenChatTemplate::Render(legacy, constrained->tools, options);
          TokenizerOptions framing;
          framing.parse_special_tokens = true;
          Expect(rendered &&
                     prepared.tokens == tokenizer->Encode(*rendered, framing),
                 "the full fallback prompt exactly matches main's legacy "
                 "instruction tokenization");
        }
      }
    }
  }
  // JSON-object prompts, schema prompts and their optional descriptions use
  // the same framing field. Native tools add no instruction or prompt work.
  for (const bool schema : {false, true}) {
    ChatRequest request(
        {{ChatRole::kSystem, "Client <tool_call>literal</tool_call>"},
         {ChatRole::kUser, "Return JSON."}});
    request.reasoning.enabled = false;
    request.response_format =
        schema
            ? gufo::sampling::JsonConstraint::Compile(
                  gufo::json::parse(
                      R"({"type":"object","properties":{"text":{"type":"string","const":"<tool_call>"}},"required":["text"],"additionalProperties":false})"),
                  true)
            : gufo::sampling::JsonConstraint::Object();
    request.response_format_description = "Describe <tool_call> in the schema.";
    gufo::sampling::SamplingConfig sampling;
    auto constrained =
        gufo::server::ConstrainChatRequest(request, runner, &sampling);
    Expect(constrained &&
               constrained->messages.front().content ==
                   request.messages.front().content &&
               constrained->messages.front().framing_suffix.find(
                   request.response_format_description + "\n\n" +
                   request.response_format->prompt()) != std::string::npos,
           "response formats and descriptions are separated from client "
           "message content");
    const auto tokens = QwenChatTemplate::RenderAndTokenize(
        *tokenizer, constrained->messages, options);
    Expect(tokens && std::count(tokens->begin(), tokens->end(),
                                *tokenizer->FindSpecialToken("<tool_call>")) ==
                         (schema ? 2 : 1),
           "generated description/schema instruction tags retain framing token "
           "identity");
  }
  ChatRequest native({{ChatRole::kUser, "Call record."}});
  native.reasoning.enabled = false;
  native.tools = {
      {.name = "record",
       .parameters_json =
           R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false})"}};
  gufo::sampling::SamplingConfig sampling;
  const auto constrained =
      gufo::server::ConstrainChatRequest(native, native_runner, &sampling);
  Expect(constrained && constrained->messages.front().framing_suffix.empty(),
         "native constraints add no instruction or change to the prompt");
  // Schemas native tags cannot enforce exactly, beside an ordinary neighbor,
  // under every tool choice: the request stays native as in llama.cpp, so the
  // prompt is the client's own and needs no extra prefill.
  for (
      const auto* schema :
      {R"({"type":"object","properties":{"text":{"type":"string","const":"\n</parameter>\n"}},"required":["text"],"additionalProperties":false})",
       R"({"type":"object","properties":{"text":{"type":"string"}},"patternProperties":{"^x_":{"type":"integer"}},"required":["text"]})",
       R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"text":{"type":"string"}},"required":["text"]})",
       R"({"type":"object","properties":{"date":{"type":"string","pattern":"^[0-9]{4}$"}}})",
       R"({"type":"object","properties":{"v":{"oneOf":[{"type":"string"},{"type":"integer"}]}}})",
       R"({"type":"object","properties":{"v":{"allOf":[{"type":"string"},{"minLength":1}]}}})",
       R"({"type":"object","properties":{"v":{"not":{"type":"null"}}}})",
       R"({"type":"object","properties":{"v":{"type":"string"}},"additionalProperties":true})",
       R"({"type":"object","properties":{"o":{"type":"object","properties":{"x":{"type":"integer"}}}}})"}) {
    for (const auto choice :
         {ChatRequest::ToolChoice::kAuto, ChatRequest::ToolChoice::kRequired}) {
      for (const bool strict : {false, true}) {
        ChatRequest request({{ChatRole::kSystem, "Be concise."},
                             {ChatRole::kUser, "Call record."}});
        request.reasoning.enabled = false;
        request.tool_choice = choice;
        const std::string definition =
            std::string(
                R"({"type":"function","function":{"name":"record","strict":)") +
            (strict ? "true" : "false") + R"(,"parameters":)" + schema + "}}";
        request.tools = {
            {.name = "bash",
             .parameters_json =
                 R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})"},
            {.name = "record",
             .parameters_json = schema,
             .definition_json = definition}};
        gufo::sampling::SamplingConfig sampled;
        std::optional<gufo::sampling::JsonConstraint::ToolFormat> format;
        std::optional<ChatRequest> result;
        try {
          result = gufo::server::ConstrainChatRequest(request, native_runner,
                                                      &sampled, &format);
        } catch (const std::invalid_argument&) {
          // An impossible strict schema is rejected before generation.
          Expect(strict, "only strict schemas may be rejected");
          continue;
        }
        Expect(
            result &&
                format == gufo::sampling::JsonConstraint::ToolFormat::kQwen &&
                sampled.constraint,
            "a native runner keeps native calls for every schema");
        Expect(result->messages.front().framing_suffix.empty() &&
                   result->messages.front().content == "Be concise.",
               "a native runner adds no tool instruction to the prompt");
      }
    }
  }
}

void TestDiskWriterPoolEvictionShutdownAndRestore() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "writer-pool", sizeof(FakeSnapshot));
  const TextRunnerDiskCacheOptions options{.directory = directory.path(),
                                           .capacity_bytes = 4096,
                                           .staging_capacity_bytes = 4096,
                                           .min_checkpoint_step_tokens = 0};
  std::atomic<std::size_t> serialized{0};
  runner->before_serialize = [&] { ++serialized; };
  {
    TextRunnerPool pool(runner, 1, options);
    auto first = pool.Acquire({1, 2, 3});
    Expect(first.Prefill(16).decode_ready, "first pool capture prefills");
    const auto first_metrics = first.Commit();
    Expect(first_metrics.disk_queued_bytes == 0 && serialized == 0,
           "evict_idle capture has no immediate writer offer");
    auto second = pool.Acquire({4, 5, 6});
    Expect(second.Prefill(16).decode_ready, "second pool capture prefills");
    second.Commit();  // Pressure evicts the first; second remains for shutdown.
  }
  Expect(
      serialized == 2 && DiskImages(directory.path()).size() == 2,
      "pool eviction and graceful shutdown each persist one remaining entry");
  {
    TextRunnerPool restarted(runner, 1, options);
    auto restored = restarted.Acquire({4, 5, 6});
    Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
           "shutdown-persisted prompt restores across pools");
    restored.Commit();
  }
  Expect(serialized == 2,
         "disk-restored RAM entry is not rewritten at shutdown");

  TemporaryDirectory rejected_directory;
  auto rejected_stats = std::make_shared<FakeStats>();
  auto rejected_runner = std::make_shared<PersistentSnapshotRunner>(
      rejected_stats, "writer-pool-refused", sizeof(FakeSnapshot) - 1);
  std::atomic<std::size_t> rejected_writes{0};
  rejected_runner->before_serialize = [&] { ++rejected_writes; };
  {
    auto rejected_options = options;
    rejected_options.directory = rejected_directory.path();
    TextRunnerPool rejected(rejected_runner, 1, rejected_options);
    auto request = rejected.Acquire({1, 2, 3});
    Expect(request.Prefill(16).decode_ready,
           "RAM refusal leaves serving functional");
    request.Commit();
  }
  Expect(rejected_stats->snapshot_captures > 0 && rejected_writes == 1 &&
             DiskImages(rejected_directory.path()).size() == 1,
         "RAM refusal falls back to disk-only capture and persistence");
}

void TestServerCacheDisabledHasColdGeometry() {
  class DebugPersistentRunner final : public PersistentSnapshotRunner {
  public:
    using PersistentSnapshotRunner::PersistentSnapshotRunner;
    [[nodiscard]] gufo::json::Value DebugSettings() const override {
      return gufo::json::Value::object();
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<DebugPersistentRunner>(stats, "no-cache-test",
                                                        4096, 32768);
  // A supplied disk directory must never be opened with server cache disabled.
  const auto directory =
      std::filesystem::current_path() / "no-cache-must-not-exist";
  Expect(!std::filesystem::exists(directory),
         "disabled disk fixture starts absent");
  TextRunnerPool pool(
      runner, 1, TextRunnerDiskCacheOptions{.directory = directory},
      gufo::server::TextRunnerRamCacheOptions{.disabled = true});
  const auto settings = pool.DebugSettings();
  Expect(!settings.find("disk_cache")->find("enabled")->as_bool() &&
             settings.find("ram_cache")->member_size("capacity_bytes") == 0,
         "server disables both cache tiers");
  for (const std::size_t length : {4095, 4096, 4097, 10000}) {
    std::vector<TextRunnerToken> prompt(length, 1);
    for (int repeat = 0; repeat < 2; ++repeat) {
      stats->prefill_spans.clear();
      auto request = pool.Acquire(prompt, {}, {}, {}, true, length - 37);
      Expect(!request.cache_hit() && request.cached_prompt_tokens() == 0,
             "server overrides request cache_prompt true with cold prefill");
      Expect(request.ShareCheckpoint(2048) == 0,
             "server disables scheduler shared checkpoints");
      while (!request.prefill_complete())
        (void)request.Prefill(4096);
      std::vector<std::size_t> expected;
      for (std::size_t remaining = length; remaining != 0;) {
        const auto step = std::min<std::size_t>(4096, remaining);
        expected.push_back(step);
        remaining -= step;
      }
      Expect(stats->prefill_spans == expected,
             "cold pass boundaries depend only on prompt length");
      if (repeat == 0)
        request.Commit();
      else
        request.Cancel();
      Expect(pool.CachedPrefixTokens(prompt, nullptr) == 0,
             "completion and cancellation retain no prefix");
    }
  }
  Expect(stats->snapshot_captures == 0 && stats->snapshot_restores == 0,
         "server disables all snapshot captures and restores");
  Expect(!std::filesystem::exists(directory),
         "disabled disk cache performs no filesystem writes");
  std::cout << "PASS: TestServerCacheDisabledHasColdGeometry\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string_view(argv[1]) == "--tail-real-sequence") {
    TestRealTailSequence(argc > 2 ? std::stoull(argv[2]) : 8,
                         argc > 3 && std::string_view(argv[3]) == "observe",
                         argc > 4 && std::string_view(argv[4]) == "default");
    return 0;
  }
  if (argc == 2) {
    const std::string_view test(argv[1]);
    if (test == "--tail-deep-only" || test == "--tail-deep-observe" ||
        test == "--tail-deep-two-slots") {
      TestTailDeepWalk(false, test == "--tail-deep-observe",
                       test == "--tail-deep-two-slots" ? 2 : 1);
      return 0;
    }
    if (test == "--tail-deep-interleaved" ||
        test == "--tail-deep-interleaved-observe") {
      TestTailDeepWalk(true, test == "--tail-deep-interleaved-observe", 2);
      return 0;
    }
    if (test == "--tail-deep-disk") {
      TestTailDeepDisk();
      return 0;
    }
    if (test == "--tail-deep-ceiling") {
      TestTailDeepByteCeiling();
      return 0;
    }
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-generic-host-review") {
    TestTailGenericLiveFreezeChargesHostForecast();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-live-review") {
    TestTailCompleteLiveRetryUsesFinalAdmission();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-shared-review") {
    TestTailSharedBoundaryKeepsBranchPurpose();
    TestTailSharedBoundaryGuards();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-shared-guards-review") {
    TestTailSharedBoundaryGuards();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-persistence-review") {
    TestTailFinalPersistencePublicationOrdering();
    return 0;
  }
  if (argc == 2 &&
      std::string_view(argv[1]) == "--tail-host-completion-review") {
    TestTailCompletedHostCaptureReleasesForecast();
    return 0;
  }
  if (argc == 2 &&
      std::string_view(argv[1]) == "--tail-publication-refusal-review") {
    TestTailFinalPublicationRefusalDoesNotPersist();
    return 0;
  }
  if (argc == 2) {
    const std::string_view test(argv[1]);
    if (test == "--tail-pressure-observe" || test == "--tail-pressure-only") {
      TestTailPressureLoop(test == "--tail-pressure-observe");
      return 0;
    }
    if (test == "--tail-over-third-only" || test == "--tail-depth-only") {
      TestTailPressureAdmission(test == "--tail-depth-only");
      TestTailPressureAdmission(test == "--tail-depth-only", true);
      return 0;
    }
    if (test == "--tail-final-only" || test == "--tail-final-disk-only") {
      TestTailFinalAdmission(test == "--tail-final-disk-only");
      return 0;
    }
    if (test == "--tail-final-failure-only") {
      TestTailFailedFinalDoesNotPersist();
      return 0;
    }
    if (test == "--tail-coincident-disk-only") {
      TestTailCoincidentBoundaryPersists();
      return 0;
    }
    if (test == "--tail-demotion-only") {
      TestTailSupersededEviction();
      TestTailSupersededEviction(true);
      TestTailSupersededEviction(false, true);
      TestTailSharedSourceDoesNotDemoteOtherSession();
      return 0;
    }
    if (test == "--tail-default-golden") {
      TailDefaultGolden();
      return 0;
    }
  }
  // Lead 1 (learned shared-prefix boundaries rounded down to a multiple of 256)
  // is off in these tests: they assert exact tiny boundaries.
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "0", 1);
  if (argc == 2 && std::string_view(argv[1]) == "--tail-host-admission-only") {
    TestExpertTailHostAdmission();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-boundary-only") {
    TestExpertTailBoundaryEconomics();
    TestExpertTailDiskGridFallback();
    TestExpertTailSnapshotEdges();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-disk-boundary-only") {
    TestExpertTailBoundaryEconomics(true);
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--tail-disk-grid-only") {
    TestExpertTailDiskGridFallback();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--identity-ownership-only") {
    TestPromptIdentityOwnership();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--disk-writer-only") {
    ::setenv("GUFO_DISK_PERSIST", "evict_idle", 1);
    ::setenv("GUFO_DISK_IDLE_MS", "600000", 1);
    TestDiskWriterPoolEvictionShutdownAndRestore();
    std::cout << "disk writer pool tests passed\n";
    return 0;
  }
  // Dedicated opt-in invocation preserves the old suite's immediate-worker
  // timing assertions. Run the unchanged full suite with DEFER unset/0 too.
  if (argc == 2 && std::string_view(argv[1]) == "--disk-defer-only") {
    TestDiskDeferRequestLifetime();
    std::cout << "disk defer request tests passed\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--no-cache-only") {
    TestServerCacheDisabledHasColdGeometry();
    return 0;
  }
  TestServerCacheDisabledHasColdGeometry();
  TestRunnerWithoutDebugSettingsThrowsTheDedicatedException();
  TestServerInstructionsAreFraming();
  // The cache warning assertion in this binary matches the plain "[WARN]
  // [cache]" text captured from a redirected sink; a TTY stderr tints it.
  ::setenv("NO_COLOR", "1", 1);
  TestInPassStableCheckpoint();
  TestInPassFailureRetainsOnlyCompletedCheckpoints();
  // Run this binary in separate GUFO_DISK_INPASS=0 and =1/unset processes.
  // The production switch is deliberately read once, never toggled for tests.
  TestDiskInPassPayloadAndFallbacks();
  TestDiskInPassCoincidentAlignedLearnedBoundary();
  TestDiskInPassKeepsDistinctLearnedBoundaries();
  TestBehaviourBudgetAndPhaseDoNotLeakThroughPromptSnapshot();
  TestNewImageGetsAStableCheckpoint();
  TestGeneratedFrontierForksBeforeMutation();
  TestGeneratedFrontierPersistsForForks();
  TestCancellationRetainsOnlyCompletedWork();
  TestPromptReuseCanBeDisabledPerRequest();
  TestStableChatPrefixSurvivesInterruptedFraming();
  TestWarmChatCheckpointsStopAtTheStableBoundary();
  TestWarmHitAdvancesTheStableCheckpoint();
  TestHistoryEditsRestoreIntermediateCheckpoints();
  TestExpertTailRequestBoundary();
  TestExpertTailCacheIdentity();
  TestExpertTailCachePolicyProvenance();
  TestExpertTailBoundaryEconomics();
  TestExpertTailDiskGridFallback();
  TestExpertTailSnapshotEdges();
  TestPromptIdentityOwnership();
  TestExpertTailHostAdmission();
  TestTailDeepWalk(false);
  TestTailDeepWalk(true, false, 2);
  TestTailDeepDisk();
  TestTailDeepByteCeiling();
  TestTailPressureLoop();
  TestTailPressureAdmission(false);
  TestTailPressureAdmission(true);
  TestTailPressureAdmission(false, true);
  TestTailPressureAdmission(true, true);
  TestTailFinalAdmission(false);
  TestTailFinalAdmission(true);
  TestTailFailedFinalDoesNotPersist();
  TestTailCoincidentBoundaryPersists();
  TestTailSupersededEviction();
  TestTailSupersededEviction(true);
  TestTailSupersededEviction(false, true);
  TestTailSharedSourceDoesNotDemoteOtherSession();
  TestTailCompleteLiveRetryUsesFinalAdmission();
  TestTailSharedBoundaryKeepsBranchPurpose();
  TestTailSharedBoundaryGuards();
  TestTailFinalPersistencePublicationOrdering();
  TestTailCompletedHostCaptureReleasesForecast();
  TestTailGenericLiveFreezeChargesHostForecast();
  TestTailFinalPublicationRefusalDoesNotPersist();
  TestNpuPreferredChunkKeepsCheckpointGridAligned();
  TestChatFallbackSurvivesSnapshotBudgetPressure();
  TestFullChatCheckpointRestoresWithoutSuffixPrefill();
  TestDiskOnlyCaptureReservesBudgetBeforeCommit();
  TestDiskPreflightAvoidsUnusableCapture();
  TestBoundedPrefillDecodeAndPrefixReuse();
  TestAbandonedRequestRollsBackState();
  TestRequestBindsAndClearsCancellation();
  TestBatchedAdvancePreservesIndependentRequests();
  TestResourceClaimsAreValidatedBeforeAllocation();
  TestLoadRefusesOnlyThePhysicalImpossibility();
  TestSnapshotForkAndUnsupportedCapabilities();
  TestSnapshotCacheBranchesOnePrefixIntoIndependentStates();
  std::ostringstream normal_log;
  gufo::server::Logger::FlushSync();
  auto* previous = std::clog.rdbuf(normal_log.rdbuf());
  TestSnapshotRetentionUsesPromptBoundary();
  gufo::server::Logger::FlushSync();
  std::clog.rdbuf(previous);
  Expect(normal_log.str().find("action=removed") == std::string::npos &&
             normal_log.str().find("action=skipped") == std::string::npos,
         "routine cache replacement stays quiet");

  std::ostringstream eviction_log;
  gufo::server::Logger::FlushSync();
  previous = std::clog.rdbuf(eviction_log.rdbuf());
  TestEntryCapacityEvictionIsLogged();
  gufo::server::Logger::FlushSync();
  std::clog.rdbuf(previous);
  Expect(
      eviction_log.str().find("action=removed") != std::string::npos &&
          eviction_log.str().find("reason=entry_capacity") != std::string::npos,
      "evicting a retained prefix for entry capacity is reported");

  TestSnapshotCacheCapacityIsReportedAtStartup();
  TestMeminfoAvailableExcludesFreeCma();
  TestSnapshotStartupReportsSelectedLimits();
  TestSnapshotCapacityPlanIsDeterministic();
  TestHostFloorDefaultAndParsing();
  TestPoolPlansSnapshotCapacityFromTotals();
  TestPersistentSnapshotRestoresAcrossPools();
  TestSharedPrefixIsLearnedAndRestoredAcrossConversations();
  TestRamLearnsDivergenceBoundaries();
  TestLearnedBoundarySurvivesExactRepublication();
  TestCoincidentCacheBoundariesShareOneCopy();
  TestMeasuredStateIsReconciledWithClaim();
  TestSnapshotBudgetRefusalDoesNotFailCompletedRequest();
  std::ostringstream failure_log;
  gufo::server::Logger::FlushSync();
  previous = std::clog.rdbuf(failure_log.rdbuf());
  TestSnapshotCaptureFailureReleasesReservationAndKeepsRequestSuccessful();
  gufo::server::Logger::FlushSync();
  std::clog.rdbuf(previous);
  Expect(
      failure_log.str().find("[WARN] [cache]") != std::string::npos &&
          failure_log.str().find("reason=capture_failure") != std::string::npos,
      "cache fallback retains an actionable warning");
  std::cout << "All text model runner tests passed\n";
  return 0;
}
