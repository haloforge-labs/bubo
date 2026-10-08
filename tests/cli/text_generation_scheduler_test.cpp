#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

#include "src/cli/serve/debug_response.hpp"
#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/cli/serve/trace.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationError;
using gufo::server::TextGenerationErrorCode;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillPolicy;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRequestPhase;
using gufo::server::TextRunnerAdvance;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TextSchedulerPolicy;

constexpr auto kTestTimeout = std::chrono::seconds{5};

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestPrefillDiagnosticsHost() {
  using namespace gufo::models::qwen38_flash_next;
  Expect(!PrefillDiagValueEnabled(nullptr) && !PrefillDiagValueEnabled("0") &&
             !PrefillDiagValueEnabled("") && !PrefillDiagValueEnabled("yes") &&
             !PrefillDiagValueEnabled("01") && PrefillDiagValueEnabled("1"),
         "only exactly 1 enables prefill diagnostics");
  PrefillDiagPass pass;
  pass.pass = 3;
  pass.start = 4096;
  pass.totals.passes = 1;
  pass.totals.tokens = 2048;
  TableReadCounters reads;
  reads.table_read_us = 12.25;
  reads.table_reads = 3;
  reads.table_bytes = 72;
  pass.totals.TableReadCounters::Add(reads);
  pass.totals.fetch_elapsed_us = 20;
  pass.totals.fetch_hidden_us = 18;
  pass.totals.fetch_exposed_us = 2;
  pass.totals.fetch_wait_us = 2.5;
  const std::array<std::uint32_t, 4> first{4, 0, 1, 0};
  const std::array<std::uint32_t, 4> last{0, 0, 7, 0};
  pass.RecordLayer(0, 2048, first, 100, 2048);
  pass.RecordLayer(1, 128, last, 200, 2048);   // tail-only layer: listed, but not folded into the pass totals
  Expect(pass.totals.expert_bytes == 200 && pass.totals.layer_samples == 1 &&
             pass.totals.experts_distinct_min == 2 &&
             pass.totals.Mean() == 2.0 && pass.totals.experts_distinct_max == 2,
         "diversity counts nonzero expert buckets of full-width layers only; the tail layer is listed but excluded from the totals");
  const std::string expected =
      "pass=3 tokens=2048 start=4096 table_read_us=12.250 table_reads=3 "
      "table_bytes=72 fetch_elapsed_us=20.000 fetch_hidden_us=18.000 "
      "fetch_exposed_us=2.000 fetch_wait_us=2.500 "
      "experts_distinct_min=2 experts_distinct_mean=2.000 "
      "experts_distinct_max=2 expert_bytes=200 layers=0:2048:2:100,1:128:1:200";
  Expect(PrefillDiagLine(pass) == expected, "structured pass line is exact");
  int takes = 0, emits = 0;
  const auto take = [&] {
    ++takes;
    return std::vector{pass};
  };
  const auto emit = [&](const std::string& line) {
    ++emits;
    Expect(line == expected, "collector emits the pass line");
  };
  const auto off = CollectPrefillDiag(false, take, emit);
  Expect(!off && takes == 0 && emits == 0 && pass.totals.table_reads == 3,
         "flag off never accesses pass counters or emits diagnostics");
  const auto on = CollectPrefillDiag(true, take, emit);
  Expect(on && takes == 1 && emits == 1, "flag on publishes measured passes");
  std::optional<PrefillDiagTotals> totals;
  AddPrefillDiag(totals, off);
  Expect(!totals, "off does not construct request counters");
  AddPrefillDiag(totals, on);
  AddPrefillDiag(totals, on);
  Expect(totals && totals->passes == 2 && totals->tokens == 4096 &&
             totals->table_read_us == 24.5 && totals->table_reads == 6 &&
             totals->table_bytes == 144 && totals->fetch_elapsed_us == 40 &&
             totals->fetch_hidden_us == 36 && totals->fetch_exposed_us == 4 &&
             totals->fetch_wait_us == 5 && totals->expert_bytes == 800 &&
             totals->layer_samples == 4 && totals->experts_distinct_sum == 6 &&
             totals->experts_distinct_min == 1 && totals->Mean() == 1.5 &&
             totals->experts_distinct_max == 2,
         "request totals add every counter and weight mean by layer samples");
  auto metrics = gufo::json::Value::object();
  gufo::server::AddPrefillDiagUsage(metrics, off);
  Expect(metrics.empty(), "off serialization adds no usage fields");
  gufo::server::AddPrefillDiagUsage(metrics, totals);
  const auto* json = metrics.find("prefill_diag");
  Expect(json && json->member_size("passes") == 2 &&
             json->member_size("expert_bytes") == 800 &&
             json->member_double("table_read_us") == 24.5 &&
             json->member_double("experts_distinct_mean") == 1.5 &&
             json->member_size("table_reads") == 6 &&
             json->member_size("table_bytes") == 144 &&
             json->member_double("fetch_elapsed_us") == 40 &&
             json->member_double("fetch_hidden_us") == 36 &&
             json->member_double("fetch_exposed_us") == 4 &&
             json->member_double("fetch_wait_us") == 5,
         "on usage contains measured request totals");
  gufo::server::TextGenerationBackend::Result result;
  Expect(gufo::server::GenerationLogDetails(result).find("table_reads=") ==
             std::string::npos,
         "off completion log has no diagnostic fields");
  result.prefill_diag = totals;
  Expect(gufo::server::GenerationLogDetails(result).find("table_reads=6") !=
             std::string::npos,
         "completion log includes diagnostic totals");
  PrefillDiagTotals unequal;
  unequal.AddLayer(4, 10);
  totals->Add(unequal);
  Expect(totals->layer_samples == 5 && totals->Mean() == 2.0 &&
             totals->experts_distinct_max == 4 && totals->expert_bytes == 840,
         "aggregation weights unequal layer counts correctly");
}

enum class EventKind : std::uint8_t {
  kPrefill,
  kAdvance,
};

struct Event {
  EventKind kind;
  TextRunnerToken label;
  std::size_t index;
  std::size_t count;
};

struct FakeControl {
  void Log(Event event) {
    const std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
    condition.notify_all();
  }

  void WaitForAdvance(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return advance_gate_entered && advance_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked decode advance");
  }

  void WaitForPrefill(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return prefill_gate_entered && prefill_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked prefill");
  }

  void ReleaseAdvance() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_advance = true;
    }
    condition.notify_all();
  }

  void ReleasePrefill() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_prefill = true;
    }
    condition.notify_all();
  }

  void RecordInvalidation() {
    invalidations.fetch_add(1, std::memory_order_relaxed);
    condition.notify_all();
  }

  void WaitForInvalidations(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool reached = condition.wait_for(lock, kTestTimeout, [&] {
      return invalidations.load(std::memory_order_relaxed) >= count;
    });
    Expect(reached, "timed out waiting for state invalidation");
  }

  [[nodiscard]] std::vector<Event> Events() const {
    const std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

  [[nodiscard]] std::vector<std::vector<TextRunnerToken>> AdvanceBatches()
      const {
    const std::lock_guard<std::mutex> lock(mutex);
    return advance_batches;
  }

  mutable std::mutex mutex;
  std::condition_variable condition;
  std::vector<Event> events;
  std::vector<std::vector<TextRunnerToken>> advance_batches;
  std::optional<TextRunnerToken> block_advance_label;
  std::optional<TextRunnerToken> block_prefill_label;
  std::optional<TextRunnerToken> throw_advance_label;
  std::atomic<bool> device_usable{true};
  std::atomic<std::size_t> device_probes{0};
  std::atomic<std::size_t> idle_probes{0};
  std::atomic<TextModelRunner::DeviceProbeStatus> idle_probe_status{
      TextModelRunner::DeviceProbeStatus::kUsable};
  TextRunnerToken advance_gate_label{0};
  TextRunnerToken prefill_gate_label{0};
  bool advance_gate_entered{false};
  bool prefill_gate_entered{false};
  bool release_advance{false};
  bool release_prefill{false};
  std::atomic<std::size_t> invalidations{0};
  std::atomic<std::size_t> states_created{0};
  std::atomic<std::size_t> decode_calls{0};
  std::atomic<std::size_t> batch_preparations{0};
  std::atomic<std::int64_t> first_request_advance_ns{0};
  bool incremental_prefill{true};
  std::size_t prefill_capacity{std::numeric_limits<std::size_t>::max()};
  std::optional<gufo::models::qwen38_flash_next::PrefillDiagTotals> prefill_diag;
  bool supports_batched_advance{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool timeline_mixed_drafts{false};
  bool timeline_multi_cycle_step{false};
  std::atomic<std::size_t> debug_settings_calls{0};
  bool batched_multi_token_decode{false};
  std::size_t actual_batch_width{4};
  bool prefix_reuse{true};
  bool preview_first_token{false};
  std::vector<std::string> output_pieces;
  std::uint32_t max_context{128};
  std::optional<std::size_t> stop_after;
  std::optional<std::size_t> eos_after;
  std::function<void()> snapshot_callback;
  bool in_pass_checkpoint{false};
  bool persistent_snapshots{false};
  bool score_supported{false};
  bool omit_score{false};
  std::atomic<std::size_t> score_calls{0};
  std::function<void()> score_callback;
};

class FakeState final : public TextRunnerState {
public:
  explicit FakeState(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  void Invalidate() noexcept override {
    control_->RecordInvalidation();
    label = 0;
    position = 0;
    decode_count = 0;
    frontier.reset();
  }

  std::shared_ptr<FakeControl> control_;
  TextRunnerToken label{0};
  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

class FakeRunner final : public TextModelRunner {
public:
  explicit FakeRunner(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "scheduler-fake",
        .state_abi = "scheduler-fake-v1",
        .max_context = control_->max_context,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = control_->incremental_prefill,
                .snapshot = bool(control_->snapshot_callback),
                .fork = bool(control_->snapshot_callback),
                .final_token_advance_required =
                    control_->final_token_advance_required,
                .incremental_text_is_exact =
                    control_->incremental_text_is_exact,
                .multi_token_decode = control_->multi_token_decode,
                .batched_multi_token_decode =
                    control_->batched_multi_token_decode,
                .batched_multi_token_decode_max_width = 4,
                .prefix_reuse = control_->prefix_reuse,
                .in_pass_checkpoint = control_->in_pass_checkpoint,
            },
        .persistence = control_->persistent_snapshots
            ? std::optional<gufo::server::TextRunnerPersistenceDescriptor>(
                  {{1, 2, 3}, 1})
            : std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = 0,
        .state_capacity_bytes = 8 * 64,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes =
            control_->snapshot_callback ? 4096U : 0U,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
    if (control_->supports_batched_advance) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 2,
      });
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 4,
      });
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }

  [[nodiscard]] std::vector<std::optional<gufo::sampling::TokenLogprob>>
  ScorePrompt(std::span<const TextRunnerToken> tokens,
              const gufo::sampling::SamplingConfig& config,
              const TextRunnerState::CancellationCheck& cancelled) const override {
    if (!control_->score_supported)
      return TextModelRunner::ScorePrompt(tokens, config, cancelled);
    control_->score_calls.fetch_add(1, std::memory_order_relaxed);
    if (control_->score_callback)
      control_->score_callback();
    gufo::sampling::SamplerState sampler(config, {});
    const std::array<float, 3> row{0.0F, 1.0F, -1.0F};
    std::vector<std::optional<gufo::sampling::TokenLogprob>> scores(tokens.size());
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      if (cancelled && cancelled())
        throw std::runtime_error("scoring cancelled");
      scores[i] = sampler.RawLogprobs(row, tokens[i], static_cast<std::size_t>(config.logprobs));
    }
    return scores;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    control_->decode_calls.fetch_add(1, std::memory_order_relaxed);
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (!text.empty()) {
        text.push_back(',');
      }
      text += std::to_string(token);
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    control_->states_created.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FakeState>(control_);
  }

  void PrepareBatchExecution(TextRunnerState& state) const override {
    (void)RequireFakeState(state);
    control_->batch_preparations.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& fake = RequireFakeState(state);
    if (offset != fake.position || prompt.empty()) {
      throw std::logic_error("invalid scheduler fake prefill");
    }
    fake.label = prompt.front();
    const std::size_t consumed = std::min(
        {max_input_tokens, prompt.size() - offset, control_->prefill_capacity});

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      control_->events.push_back({
          .kind = EventKind::kPrefill,
          .label = fake.label,
          .index = fake.position,
          .count = consumed,
      });
      if (control_->block_prefill_label == fake.label &&
          !control_->prefill_gate_entered) {
        control_->prefill_gate_entered = true;
        control_->prefill_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_prefill; });
        if (!released) {
          throw std::runtime_error("prefill gate timed out");
        }
      }
    }

    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = fake.label * 100;
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = ready,
        .prefill_diag = control_->prefill_diag,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, gufo::sampling::SamplerState& sampler) const override {
    const auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no frontier");
    }
    if (control_->stop_after && fake.decode_count >= *control_->stop_after)
      return {.stop = true};
    if (control_->eos_after && fake.decode_count >= *control_->eos_after &&
        fake.stop_at_eos())
      return {.stop = true};
    if (!control_->output_pieces.empty()) {
      const auto index =
          static_cast<std::size_t>(*fake.frontier - fake.label * 100);
      if (index >= control_->output_pieces.size())
        return {.stop = true};
      return {.token = *fake.frontier, .piece = control_->output_pieces[index]};
    }
    TextDecodeSelection selection{
        .stop = false,
        .token = *fake.frontier,
        .piece = std::to_string(*fake.frontier),
    };
    if (sampler.config().logprobs >= 0 && !control_->omit_score) {
      std::vector<float> row(selection.token + 1, 0.0F);
      row[selection.token] = 1.0F;
      selection.logprob = std::make_shared<const gufo::sampling::TokenLogprob>(
          sampler.RawLogprobs(row, selection.token,
                             static_cast<std::size_t>(sampler.config().logprobs)));
    }
    return selection;
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    if (!control_->preview_first_token)
      return std::nullopt;
    return SelectNext(state, sampler);
  }

  struct SnapshotState final : gufo::server::TextRunnerSnapshot {
    explicit SnapshotState(const FakeState& state)
        : label(state.label),
          position(state.position),
          decode_count(state.decode_count),
          frontier(state.frontier) {}
    TextRunnerToken label;
    std::size_t position, decode_count;
    std::optional<TextRunnerToken> frontier;
    std::size_t PayloadBytes() const noexcept override {
      return sizeof(SnapshotState);
    }
  };
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary) const override {
    if (!control_->in_pass_checkpoint || boundary <= offset ||
        boundary >= offset + std::min(budget, prompt.size() - offset))
      return std::nullopt;
    return sizeof(SnapshotState);
  }

  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<gufo::server::TextRunnerSnapshot>* checkpoint) const override {
    auto step = Prefill(state, prompt, offset, budget);
    const auto started = std::chrono::steady_clock::now();
    auto saved = std::make_unique<SnapshotState>(RequireFakeState(state));
    saved->position = boundary;
    *checkpoint = std::move(saved);
    step.checkpoint_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (state.debug_timeline())
      step.capture_started = started;
    return step;
  }

  std::size_t SnapshotPayloadBytes(const TextRunnerState&) const override {
    return sizeof(SnapshotState);
  }
  std::unique_ptr<gufo::server::TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    if (control_->snapshot_callback)
      control_->snapshot_callback();
    return std::make_unique<SnapshotState>(RequireFakeState(state));
  }
  void RestoreOrFork(
      TextRunnerState& state,
      const gufo::server::TextRunnerSnapshot& snapshot) const override {
    auto& restored = RequireFakeState(state);
    const auto& saved = dynamic_cast<const SnapshotState&>(snapshot);
    restored.label = saved.label;
    restored.position = saved.position;
    restored.decode_count = saved.decode_count;
    restored.frontier = saved.frontier;
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const gufo::server::TextRunnerSnapshot&) const override {
    return 5 * sizeof(std::uint64_t);
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const gufo::server::TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& saved = dynamic_cast<const SnapshotState&>(snapshot);
    if (destination.size() != 5 * sizeof(std::uint64_t))
      throw std::invalid_argument("scheduler snapshot payload size mismatch");
    const std::array<std::uint64_t, 5> fields = {
        saved.label, saved.position, saved.decode_count,
        saved.frontier.has_value() ? 1U : 0U, saved.frontier.value_or(0)};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
        destination[field * sizeof(std::uint64_t) + byte] =
            static_cast<std::uint8_t>(fields[field] >> (byte * 8U));
    }
    return destination.size();
  }

  void RestorePersistentSnapshot(TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != 5 * sizeof(std::uint64_t))
      throw std::invalid_argument("scheduler snapshot payload size mismatch");
    std::array<std::uint64_t, 5> fields{};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
        fields[field] |= static_cast<std::uint64_t>(
            payload[field * sizeof(std::uint64_t) + byte]) << (byte * 8U);
    }
    auto& restored = RequireFakeState(state);
    restored.label = fields[0];
    restored.position = fields[1];
    restored.decode_count = fields[2];
    restored.frontier = fields[3] ? std::optional<TextRunnerToken>(fields[4])
                                  : std::nullopt;
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    const auto started = std::chrono::steady_clock::now();
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value() || token != *fake.frontier) {
      throw std::logic_error("scheduler fake frontier mismatch");
    }

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      if (control_->block_advance_label == fake.label &&
          fake.decode_count == 0 && !control_->advance_gate_entered) {
        control_->advance_gate_entered = true;
        control_->advance_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_advance; });
        if (!released) {
          throw std::runtime_error("advance gate timed out");
        }
      }
      if (control_->throw_advance_label == fake.label) {
        throw std::runtime_error("injected scheduler runner failure");
      }
      control_->events.push_back({
          .kind = EventKind::kAdvance,
          .label = fake.label,
          .index = fake.decode_count,
          .count = 1,
      });
    }

    ++fake.position;
    ++fake.decode_count;
    fake.frontier = token + 1;
    if (fake.label == 1)
      control_->first_request_advance_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count(),
          std::memory_order_relaxed);
  }

  [[nodiscard]] gufo::json::Value DebugSettings() const override {
    ++control_->debug_settings_calls;
    auto settings = gufo::json::Value::object();
    if (control_->timeline_mixed_drafts) {
      auto policy = gufo::json::Value::object();
      policy["enabled"] = true;
      policy["prompt_lookup"] = true;
      policy["survival"] = true;
      policy["runner_max_draft_tokens"] = 4;
      policy["draft_vocabulary"] = "latin_text";
      settings["mtp_policy"] = std::move(policy);
    }
    return settings;
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (!control_->multi_token_decode) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no multi-token frontier");
    }
    const std::size_t count = std::min<std::size_t>(max_tokens, 3);
    TextDecodeStep step;
    step.selections.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto selection = SelectNext(state, sampler);
      if (selection.stop) {
        step.stop = true;
        break;
      }
      Advance(state, selection.token);
      if (selection.logprob)
        step.logprobs.push_back(*selection.logprob);
      step.selections.push_back(std::move(selection));
    }
    step.draft_rounds = 1;
    step.draft_tokens = count + 1;
    step.draft_accepted_tokens = count;
    if (control_->timeline_mixed_drafts) {
      step.policy_chosen_length_sum = 4;
      step.survival_any = sampler.config().uses_random_sampling();
      step.lookup_tokens = fake.decode_count <= 3 ? 2 : 0;
      step.draft_accepted_tokens = count - 1;
      // The second scripted survival cycle stops drafting below its budget.
      step.draft_tokens = fake.decode_count <= 3 ? count + 1 : count;
      step.mtp_offered = fake.decode_count <= 3 ? 2 : 4;
      step.cut = fake.decode_count <= 3 ? 0 : 1;
      if (control_->timeline_multi_cycle_step && fake.decode_count > 3) {
        step.draft_rounds = 3;
        step.policy_chosen_length_sum = 12;
        step.survival_any = true;
        step.mtp_offered = 10;
        step.draft_tokens = 9;
      }
    }
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const gufo::server::TextRunnerDecode> decodes) const override {
    auto steps = TextModelRunner::DecodeBatch(decodes);
    const auto width = std::min(control_->actual_batch_width, decodes.size());
    for (auto& step : steps) {
      step.execution_plan = {.kind = width > 1 ? TextExecutionPlanKind::kBatched
                                               : TextExecutionPlanKind::kSerial,
                             .physical_width = width};
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    std::vector<TextRunnerToken> labels;
    labels.reserve(advances.size());
    for (const auto& advance : advances) {
      labels.push_back(RequireFakeState(advance.state.get()).label);
    }
    {
      const std::lock_guard<std::mutex> lock(control_->mutex);
      control_->advance_batches.push_back(std::move(labels));
    }
    TextModelRunner::AdvanceBatch(advances);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    control_->idle_probes.fetch_add(1, std::memory_order_relaxed);
    control_->condition.notify_all();
    return control_->idle_probe_status.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool DeviceUsable() const override {
    control_->device_probes.fetch_add(1, std::memory_order_relaxed);
    return control_->device_usable.load(std::memory_order_relaxed);
  }

private:
  std::shared_ptr<FakeControl> control_;
};

std::unique_ptr<TextGenerationScheduler> MakeScheduler(
    const std::shared_ptr<FakeControl>& control, std::size_t capacity,
    TextPrefillPolicy prefill_policy = {},
    TextSchedulerPolicy scheduler_policy = {}) {
  auto runner = std::make_shared<FakeRunner>(control);
  auto pool = std::make_shared<TextRunnerPool>(std::move(runner), capacity);
  return std::make_unique<TextGenerationScheduler>(
      std::move(pool), prefill_policy, scheduler_policy);
}

std::size_t EventIndex(std::span<const Event> events, EventKind kind,
                       TextRunnerToken label, std::size_t occurrence = 0) {
  std::size_t seen = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind == kind && events[index].label == label) {
      if (seen == occurrence) {
        return index;
      }
      ++seen;
    }
  }
  return events.size();
}

std::vector<TextRunnerToken> ExpectedTokens(TextRunnerToken label,
                                            std::size_t count) {
  std::vector<TextRunnerToken> tokens;
  tokens.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens.push_back(label * 100 + static_cast<TextRunnerToken>(index));
  }
  return tokens;
}

TextRequestMetadata ClientMetadata(std::string client_id) {
  return {
      .client_id = std::move(client_id),
      .deadline = std::nullopt,
      .request_start = TextGenerationScheduler::Clock::now(),
  };
}

std::vector<TextRunnerToken> SharedPrompt(std::size_t shared,
                                          TextRunnerToken tail_start,
                                          std::size_t tail) {
  std::vector<TextRunnerToken> prompt{1};
  for (std::size_t index = 1; index < shared; ++index)
    prompt.push_back(1000 + static_cast<TextRunnerToken>(index));
  for (std::size_t index = 0; index < tail; ++index)
    prompt.push_back(tail_start + static_cast<TextRunnerToken>(index));
  return prompt;
}

std::shared_ptr<FakeControl> SharedPrefixControl(bool snapshots) {
  auto control = std::make_shared<FakeControl>();
  control->max_context = 4096;
  control->prefill_capacity = 256;
  if (snapshots)
    control->snapshot_callback = [] {};
  return control;
}

void TestConcurrentSharedPrefixesPrefillOnce() {
  auto control = SharedPrefixControl(true);
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 4);
  const auto leader_prompt = SharedPrompt(1200, 7000, 100);
  auto leader = scheduler->Submit(leader_prompt, 2, 0.0F);
  control->WaitForPrefill(1);
  // Arrivals while the leader prefills: an identical prompt, one diverging
  // after the shared system prompt, and one sharing too little to wait for.
  auto identical = scheduler->Submit(leader_prompt, 2, 0.0F);
  auto diverging = scheduler->Submit(SharedPrompt(1200, 8000, 100), 2, 0.0F);
  auto unrelated = scheduler->Submit(SharedPrompt(100, 9000, 900), 2, 0.0F);
  control->ReleasePrefill();
  const auto a = leader.Wait();
  const auto b = identical.Wait();
  const auto d = diverging.Wait();
  const auto e = unrelated.Wait();
  for (const auto* result : {&a, &b, &d, &e})
    Expect(result->tokens == ExpectedTokens(1, 2),
           "shared-prefix waiting preserves every output");
  Expect(a.prefill_tokens == 1300 && a.shared_prefix_wait_ms == 0.0,
         "the leader prefills its prompt without waiting");
  Expect(b.cached_prompt_tokens == 1299 && b.prefill_tokens == 1 &&
             b.shared_prefix_wait_ms > 0.0,
         "an identical arrival restores the leader's last prefill position");
  Expect(
      d.cached_prompt_tokens == 1200 && d.prefill_tokens == 100 &&
          d.shared_prefix_wait_ms > 0.0,
      "a diverging arrival restores the shared prefix and prefills its tail");
  Expect(e.cached_prompt_tokens == 0 && e.prefill_tokens == 1000 &&
             e.shared_prefix_wait_ms == 0.0,
         "a short shared prefix is prefilled without waiting");
}

// The leader is cancelled while the worker is still inside its first prefill
// chunk, before the follower can be admitted, so the follower never waits.
void TestLeaderCancelledBeforeFollowerIsAdmitted() {
  auto control = SharedPrefixControl(true);
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 2);
  const auto prompt = SharedPrompt(1200, 7000, 100);
  auto leader = scheduler->Submit(prompt, 2, 0.0F);
  control->WaitForPrefill(1);
  auto follower = scheduler->Submit(prompt, 2, 0.0F);
  leader.Cancel();
  control->ReleasePrefill();
  const auto a = leader.Wait();
  const auto b = follower.Wait();
  Expect(a.cancelled, "the leader is cancelled");
  Expect(!b.cancelled && b.tokens == ExpectedTokens(1, 2) &&
             b.cached_prompt_tokens + b.prefill_tokens == prompt.size(),
         "a follower admitted after its leader's cancellation prefills alone");
}

void TestParkedFollowerSurvivesLeaderCancellation() {
  for (const bool fail_capture : {false, true}) {
    auto control = SharedPrefixControl(true);
    control->block_prefill_label = 1;
    std::binary_semaphore entered(0), release(0);
    std::atomic<unsigned> captures{0};
    control->snapshot_callback = [&] {
      if (captures.fetch_add(1) == 0) {
        entered.release();
        if (!release.try_acquire_for(kTestTimeout))
          throw std::runtime_error("shared snapshot gate timed out");
        if (fail_capture)
          throw std::runtime_error("injected shared snapshot failure");
      }
    };
    auto scheduler = MakeScheduler(control, 2);
    const auto prompt = SharedPrompt(1200, 7000, 100);
    auto leader = scheduler->Submit(prompt, 2, 0.0F);
    control->WaitForPrefill(1);
    auto follower = scheduler->Submit(prompt, 2, 0.0F);
    control->ReleasePrefill();

    // This fresh short prompt has no grid or fallback snapshot at 1299.
    // Only the parked follower asks the leader to capture that boundary.
    const bool capturing = entered.try_acquire_for(kTestTimeout);
    const auto events = control->Events();
    const bool at_shared_boundary =
        !events.empty() && events.back().kind == EventKind::kPrefill &&
        events.back().index + events.back().count == prompt.size() - 1;
    leader.Cancel();
    release.release();
    Expect(capturing && at_shared_boundary,
           "the follower is parked before its leader is cancelled");

    auto completed = std::async(std::launch::async, [&] {
      return std::pair{leader.Wait(), follower.Wait()};
    });
    Expect(completed.wait_for(kTestTimeout) == std::future_status::ready,
           "a parked follower is released after leader cancellation");
    const auto [a, b] = completed.get();
    Expect(a.cancelled && a.tokens.empty(),
           "the leader is cancelled before completing prefill");
    Expect(!b.cancelled && b.tokens == ExpectedTokens(1, 2) &&
               b.shared_prefix_wait_ms > 0.0 &&
               b.cached_prompt_tokens + b.prefill_tokens == prompt.size(),
           "the parked follower completes with exact prompt accounting");
    Expect(b.cached_prompt_tokens == (fail_capture ? 0 : prompt.size() - 1),
           "only a successfully captured shared checkpoint is reused");

    auto next = std::async(std::launch::async, [&] {
      return scheduler->Submit({2, 20, 21}, 2, 0.0F).Wait();
    });
    Expect(next.wait_for(kTestTimeout) == std::future_status::ready,
           "leader cancellation leaves capacity for an independent request");
    const auto next_result = next.get();
    Expect(!next_result.cancelled && next_result.tokens == ExpectedTokens(2, 2),
           "the independent request completes with its own state");
  }
}

void TestSharedPrefixWaitRequiresSnapshots() {
  auto control = SharedPrefixControl(false);
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 2);
  const auto prompt = SharedPrompt(1200, 7000, 100);
  auto leader = scheduler->Submit(prompt, 2, 0.0F);
  control->WaitForPrefill(1);
  auto follower = scheduler->Submit(prompt, 2, 0.0F);
  control->ReleasePrefill();
  (void)leader.Wait();
  const auto b = follower.Wait();
  Expect(b.tokens == ExpectedTokens(1, 2) && b.shared_prefix_wait_ms == 0.0 &&
             b.prefill_tokens == prompt.size(),
         "runners without snapshots never park concurrent requests");
}

void TestParkedFollowersReserveVisibleSessions() {
  namespace metrics = gufo::server::detail;
  auto control = SharedPrefixControl(true);
  control->block_prefill_label = 1;
  std::binary_semaphore entered(0), release(0);
  std::atomic<unsigned> captures{0};
  control->snapshot_callback = [&] {
    if (captures.fetch_add(1) == 0) {
      entered.release();
      if (!release.try_acquire_for(kTestTimeout))
        throw std::runtime_error("shared snapshot gate timed out");
    }
  };
  auto scheduler = MakeScheduler(control, 2);
  const auto processing_before = metrics::RequestsProcessing().load();
  const auto prompt = SharedPrompt(1200, 7000, 100);
  auto leader = scheduler->Submit(prompt, 2, 0.0F);
  control->WaitForPrefill(1);
  auto follower = scheduler->Submit(prompt, 2, 0.0F);
  control->ReleasePrefill();
  Expect(entered.try_acquire_for(kTestTimeout),
         "shared checkpoint capture starts");
  auto third = scheduler->Submit({2, 20}, 2, 0.0F);
  bool exceeded_capacity = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (std::chrono::steady_clock::now() < deadline) {
    exceeded_capacity |=
        metrics::RequestsProcessing().load() - processing_before > 2;
    std::this_thread::yield();
  }
  const auto states = scheduler->SessionStates();
  const bool reserved = states.size() == 2 && states[0].processing &&
                        states[1].processing &&
                        states[0].request_id == leader.id() &&
                        states[1].request_id == follower.id();
  const bool queued = third.phase() == TextRequestPhase::kQueued;
  release.release();
  (void)leader.Wait();
  const auto reused = follower.Wait();
  const auto independent = third.Wait();
  Expect(!exceeded_capacity && reserved && queued,
         "parked followers reserve sessions and keep newer arrivals queued");
  Expect(reused.cached_prompt_tokens > 0 &&
             independent.tokens == ExpectedTokens(2, 2),
         "reserved followers resume and then release admission capacity");
}

void TestRetainedHistoryBeatsSharedPrefixWait() {
  auto control = SharedPrefixControl(true);
  auto scheduler = MakeScheduler(control, 3);
  const auto history = SharedPrompt(1200, 7000, 100);
  (void)scheduler->Submit(history, 1, 0.0F).Wait();
  {
    const std::lock_guard<std::mutex> lock(control->mutex);
    control->block_prefill_label = 1;
  }
  auto leader = scheduler->Submit(SharedPrompt(1200, 8000, 100), 2, 0.0F);
  control->WaitForPrefill(1);
  // The next turn of the retained conversation shares more with its own
  // checkpoint than with the leader, so it must not wait for the leader.
  auto next_turn = history;
  next_turn.insert(next_turn.end(), {9001, 9002, 9003});
  auto continuation = scheduler->Submit(next_turn, 2, 0.0F);
  control->ReleasePrefill();
  const auto c = continuation.Wait();
  (void)leader.Wait();
  Expect(c.shared_prefix_wait_ms == 0.0 &&
             c.cached_prompt_tokens >= history.size(),
         "a longer retained prefix is reused without waiting");
}

void TestAwaitedCheckpointSurvivesCachePressure() {
  auto control = SharedPrefixControl(true);
  auto scheduler = MakeScheduler(control, 2);
  // Fill every checkpoint record with unrelated conversations, as on a
  // long-running server. Optional copies can no longer be admitted.
  for (TextRunnerToken label = 2; label < 200; ++label)
    (void)scheduler->Submit({label, 1, 2}, 1, 0.0F).Wait();
  {
    const std::lock_guard<std::mutex> lock(control->mutex);
    control->block_prefill_label = 1;
  }
  auto leader = scheduler->Submit(SharedPrompt(1200, 7000, 100), 2, 0.0F);
  control->WaitForPrefill(1);
  auto follower = scheduler->Submit(SharedPrompt(1200, 8000, 100), 2, 0.0F);
  control->ReleasePrefill();
  (void)leader.Wait();
  const auto result = follower.Wait();
  Expect(result.shared_prefix_wait_ms > 0.0 &&
             result.cached_prompt_tokens == 1200 &&
             result.prefill_tokens == 100,
         "a checkpoint peers wait for is retained under cache pressure");
}

void TestPrefillDiagnosticSchedulerTotals() {
  for (const bool enabled : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->prefill_capacity = 2;
    if (enabled) {
      control->prefill_diag.emplace();
      control->prefill_diag->passes = 1;
      control->prefill_diag->table_reads = 3;
      control->prefill_diag->AddLayer(2, 100);
    }
    auto scheduler = MakeScheduler(control, 1, {.decode_active_tokens = 2});
    const auto result = scheduler->Submit({7, 70, 71, 72, 73}, 2, 0.0F).Wait();
    Expect(result.prefill_chunks == 3 && bool(result.prefill_diag) == enabled,
           "scheduler carries opt-in diagnostics over multiple fake chunks");
    if (result.prefill_diag) {
      Expect(result.prefill_diag->passes == 3 &&
                 result.prefill_diag->table_reads == 9 &&
                 result.prefill_diag->expert_bytes == 600 &&
                 result.prefill_diag->layer_samples == 3,
             "scheduler request totals include every prefill step");
    }
  }
}

void TestIdlePrefillUsesBulkWorkUnit() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 1, {.decode_active_tokens = 2});

  const auto result = scheduler->Submit({7, 70, 71, 72, 73}, 2, 0.0F).Wait();

  const auto events = control->Events();
  Expect(events.front().kind == EventKind::kPrefill &&
             events.front().label == 7 && events.front().count == 5,
         "decode-idle prefill consumes the complete prompt");
  Expect(result.prefill_chunks == 1 && result.prefill_tokens == 5,
         "decode-idle prefill metrics report one bulk work unit");
  Expect(result.active_decode_prefill_chunks == 0,
         "decode-idle prefill is not counted as active-decode work");
  Expect(result.requested_logical_concurrency == 1 &&
             result.physical_execution_width == 1 &&
             result.execution_plan == "serial-c1",
         "C=1 telemetry reports immediate serial dispatch");
}

void TestRunnerCanSkipUnusedFinalAdvance() {
  auto control = std::make_shared<FakeControl>();
  control->final_token_advance_required = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.tokens == ExpectedTokens(7, 2),
         "skipped final advance preserves emitted tokens");

  std::size_t advances = 0;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kAdvance && event.label == 7) {
      ++advances;
    }
  }
  Expect(advances == 1,
         "runner skips the unused frontier computation after the final token");
}

void TestRunnerCanReuseExactIncrementalText() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_text_is_exact = true;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.text == "700701",
         "exact incremental pieces form the final response text");
  Expect(control->decode_calls.load(std::memory_order_relaxed) == 0,
         "exact incremental text avoids duplicate final decoding");
}

// An armed `--trace` records each finished request under the HTTP request id
// bound where it was submitted: the prompt as the runner decodes it, the
// effective limits and sampling, the cache outcome and the raw output.
void TestGenerationTraceRecordsPromptAndOutput() {
  namespace fs = std::filesystem;
  using gufo::server::Trace;
  auto control = std::make_shared<FakeControl>();
  control->incremental_text_is_exact = true;
  auto scheduler = MakeScheduler(control, 1);
  (void)scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(control->decode_calls.load(std::memory_order_relaxed) == 0,
         "an unarmed trace decodes no prompt");

  const auto path =
      fs::temp_directory_path() /
      ("gufo-scheduler-trace-" + std::to_string(::getpid()) + ".jsonl");
  fs::remove(path);
  Expect(!Trace::Open(path.string()).has_value(), "trace file opens");
  gufo::sampling::SamplingConfig sampling;
  sampling.temperature = 0.7F;
  sampling.top_k = 4;
  sampling.seed = 9;
  TextGenerationScheduler::Result result;
  {
    const Trace::RequestScope scope("r42");
    result = scheduler->Submit({7, 8}, 2, sampling).Wait();
  }
  Expect(control->decode_calls.load(std::memory_order_relaxed) == 1,
         "an armed trace decodes the prompt once");
  control->throw_advance_label = 9;
  bool failed = false;
  {
    const Trace::RequestScope scope("r43");
    try {
      (void)scheduler->Submit({9}, 2, sampling).Wait();
    } catch (const std::runtime_error&) {
      failed = true;
    }
  }
  Trace::Close();
  Expect(failed, "the injected runner failure reaches the consumer");

  std::vector<gufo::json::Value> records;
  {
    std::ifstream input(path);
    for (std::string line; std::getline(input, line);) {
      records.push_back(gufo::json::parse(line));
    }
  }
  fs::remove(path);
  Expect(records.size() == 2, "one generation record per request");
  const auto& record = records.front();
  Expect(record.member_str("event") == "generation" &&
             record.member_str("request") == "r42",
         "the record names the bound HTTP request");
  Expect(record.member_str("prompt") == "7,8",
         "the prompt is recorded as the runner decodes it");
  Expect(!result.text.empty() && record.member_str("output") == result.text,
         "the output is the generated text before transport parsing");
  Expect(record.member_size("max_tokens") == 2 &&
             record.member_size("prompt_tokens") == 2 &&
             record.member_size("generated_tokens") == result.tokens.size(),
         "the record carries the effective limits and counts");
  const auto* recorded_sampling = record.find("sampling");
  Expect(recorded_sampling != nullptr &&
             recorded_sampling->find("temperature")->as_double() == 0.7 &&
             recorded_sampling->member_size("top_k") == 4 &&
             recorded_sampling->member_size("seed") == 9 &&
             !recorded_sampling->find("constrained")->as_bool(),
         "the record carries the effective sampling");
  Expect(record.member_str("cache") == "miss" &&
             record.member_str("finish") == "length" &&
             record.find("error") == nullptr,
         "the record carries the cache outcome and finish");
  const auto& failure = records.back();
  Expect(failure.member_str("request") == "r43" &&
             failure.member_str("prompt") == "9" &&
             failure.member_str("error") == "injected scheduler runner failure",
         "a failed request still records its prompt and the failure");
}

void TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->multi_token_decode = true;
  control->prefix_reuse = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto first = scheduler->Submit({7}, 5, 0.0F).Wait();
  Expect(first.tokens == ExpectedTokens(7, 5),
         "multi-token decode preserves the generated trajectory");
  Expect(first.draft_rounds == 2 && first.draft_tokens == 7 &&
             first.draft_accepted_tokens == 5,
         "multi-token decode reports accumulated draft statistics");
  Expect(!first.cache_hit,
         "multi-token state starts without continuation reuse");
  control->WaitForInvalidations(1);

  const auto second = scheduler->Submit({7, 70}, 2, 0.0F).Wait();
  Expect(!second.cache_hit,
         "runner-disabled prefix reuse cannot retain speculative state");
}

void TestRequestTotalsDoNotNeedAConsumer() {
  namespace metrics = gufo::server::detail;
  const auto wait_terminal =
      [](const TextGenerationScheduler::Request& request) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (request.phase() != TextRequestPhase::kTerminal) {
          Expect(std::chrono::steady_clock::now() < deadline,
                 "request reaches terminal without a consumer");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      };
  {
    auto control = std::make_shared<FakeControl>();
    auto scheduler = MakeScheduler(control, 1);
    gufo::server::RecordServerMetrics(
        scheduler->Submit({1, 10}, 1, 0.0F).Wait());
    const auto cached_before = metrics::TotalCachedPromptTokens().load();
    // Never waited on, like a stream whose first write fails.
    auto abandoned = scheduler->Submit({1, 10, 100, 11}, 1, 0.0F);
    wait_terminal(abandoned);
    Expect(metrics::TotalCachedPromptTokens().load() - cached_before == 3,
           "a finished request counts before anyone reads it");
    abandoned = {};
    const auto consumed = scheduler->Submit({1, 10, 100, 11}, 1, 0.0F).Wait();
    gufo::server::RecordServerMetrics(consumed);
    Expect(metrics::TotalCachedPromptTokens().load() - cached_before ==
               3 + consumed.cached_prompt_tokens,
           "reading the result does not count it again");
  }
  {
    auto control = std::make_shared<FakeControl>();
    control->incremental_prefill = false;
    control->multi_token_decode = true;
    control->prefix_reuse = false;
    auto scheduler = MakeScheduler(control, 1);
    const auto rounds_before = metrics::TotalDraftRounds().load();
    const auto drafts_before = metrics::TotalDraftTokens().load();
    const auto accepted_before = metrics::TotalDraftAcceptedTokens().load();
    auto request = scheduler->Submit({7}, 5, 0.0F);
    wait_terminal(request);
    Expect(
        metrics::TotalDraftRounds().load() - rounds_before == 2 &&
            metrics::TotalDraftTokens().load() - drafts_before == 7 &&
            metrics::TotalDraftAcceptedTokens().load() - accepted_before == 5,
        "draft totals count at completion");
  }
}

void TestMultiTokenRunnerCanSwitchToBatchedExecution() {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = true;
  control->prefix_reuse = false;
  control->supports_batched_advance = true;
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1, 10}, 4, 0.0F);
  control->WaitForPrefill(1);
  auto request_b = scheduler->Submit({2, 20}, 4, 0.0F);
  control->ReleasePrefill();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 4) &&
             result_b.tokens == ExpectedTokens(2, 4),
         "batched speculative-capable requests preserve both trajectories");
  Expect(result_a.execution_plan == "batched-w2" &&
             result_b.execution_plan == "batched-w2",
         "speculative-capable requests can use the physical W2 plan");
  Expect(result_a.draft_rounds == 0 && result_b.draft_rounds == 0 &&
             result_a.draft_tokens == 0 && result_b.draft_tokens == 0,
         "target batching bypasses per-request draft steps");
  Expect(control->batch_preparations.load(std::memory_order_relaxed) >= 2,
         "both resident states are prepared before target batching");
}

void TestBatchFailureIsolation() {
  for (const bool speculative : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = speculative;
    control->batched_multi_token_decode = speculative;
    control->supports_batched_advance = true;
    control->block_prefill_label = 1;
    control->throw_advance_label = 1;
    auto scheduler = MakeScheduler(control, 2);
    auto failed = scheduler->Submit({1}, 12, 0.0F);
    control->WaitForPrefill(1);
    auto healthy = scheduler->Submit({2}, 12, 0.0F);
    control->ReleasePrefill();
    bool threw = false;
    try {
      (void)failed.Wait();
    } catch (const std::exception&) {
      threw = true;
    }
    Expect(threw, "failed batch member reports its error");
    const auto result = healthy.Wait();
    Expect(result.tokens == ExpectedTokens(2, 12),
           "a failing batch member must not fail or corrupt a healthy peer");
  }
}

void TestBatchDeviceLossFailsSuccessfulPeers() {
  for (const bool speculative : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = speculative;
    control->batched_multi_token_decode = speculative;
    control->supports_batched_advance = true;
    control->block_prefill_label = 1;
    // A later failed lane must be classified before the earlier successful
    // lane commits and releases its state on the same unusable context.
    control->throw_advance_label = 2;
    control->device_usable = false;
    auto scheduler = MakeScheduler(control, 2);
    // AR gives the first decoder one advance before admitting the peer;
    // multi-token batching assembles the initial cohort before decoding.
    const std::size_t max_tokens = speculative ? 1 : 2;
    auto first = scheduler->Submit({1}, max_tokens, 0.0F);
    control->WaitForPrefill(1);
    auto second = scheduler->Submit({2}, max_tokens, 0.0F);
    control->ReleasePrefill();
    for (auto* request : {&first, &second}) {
      bool lost = false;
      try {
        (void)request->Wait();
      } catch (const TextGenerationError& error) {
        lost = error.code() == TextGenerationErrorCode::kDeviceLost;
      }
      Expect(lost,
             "device loss fails the entire batch before committing peers");
    }
    Expect(control->device_probes == 1 && scheduler->device_lost(),
           "batch loss probes once and remains sticky");
  }
}

void TestModelOwnedBatchMetrics() {
  for (const std::size_t actual_width : {1U, 2U}) {
    namespace metrics = gufo::server::detail;
    const auto prompt_before = metrics::TotalPromptTokens().load();
    const auto generated_before = metrics::TotalGenTokens().load();
    const auto rounds_before = metrics::TotalDraftRounds().load();
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = true;
    control->batched_multi_token_decode = true;
    control->supports_batched_advance = true;
    control->actual_batch_width = actual_width;
    control->block_prefill_label = 1;
    auto scheduler = MakeScheduler(control, 2);
    auto first = scheduler->Submit({1}, 12, 0.0F);
    control->WaitForPrefill(1);
    auto second = scheduler->Submit({2}, 12, 0.0F);
    control->ReleasePrefill();
    for (const auto& result : {first.Wait(), second.Wait()}) {
      Expect(
          result.physical_execution_width == actual_width &&
              result.execution_plan ==
                  (actual_width == 1 ? "serial-fallback" : "batched-w2"),
          "scheduler reports the runner's actual subgroup or serial execution");
      Expect(result.draft_rounds == 4,
             "each batched request reports its own verification rounds");
      Expect(result.token_metrics_recorded, "scheduled tokens counted live");
      gufo::server::RecordServerMetrics(result);
    }
    Expect(metrics::TotalPromptTokens().load() - prompt_before == 2 &&
               metrics::TotalGenTokens().load() - generated_before == 24 &&
               metrics::TotalDraftRounds().load() - rounds_before == 8,
           "speculative batches and HTTP completion count each token once");
  }
}

void TestMultiResidentPrefillUsesBoundedWorkUnits() {
  for (const auto capacity : {1U, 2U}) {
    auto control = std::make_shared<FakeControl>();
    control->prefill_capacity = 3;
    auto scheduler =
        MakeScheduler(control, capacity, {.decode_active_tokens = 2});
    const auto result =
        scheduler->Submit({7, 70, 71, 72, 73, 74, 75}, 2, 0.0F).Wait();
    std::vector<std::size_t> chunks;
    for (const auto& event : control->Events()) {
      if (event.kind == EventKind::kPrefill && event.label == 7)
        chunks.push_back(event.count);
    }
    Expect(chunks == std::vector<std::size_t>({3, 3, 1}),
           "idle prefill uses model geometry regardless of spare slots");
    Expect(
        result.prefill_chunks == 3 && result.active_decode_prefill_chunks == 0,
        "model-sized work units still yield for admission and cancellation");
  }
}

void TestDecodeActivePrefillIsBounded(bool multi_token, bool batched) {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = multi_token;
  control->batched_multi_token_decode = batched;
  control->supports_batched_advance = multi_token;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  const std::size_t output_tokens = multi_token ? 30 : 10;
  auto request_a = scheduler->Submit({1}, output_tokens, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23, 24, 25}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, output_tokens),
         "active decoder preserves its isolated trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "chunked prefill preserves the new request trajectory");

  const auto events = control->Events();
  std::size_t previous_b_prefill = events.size();
  std::size_t b_prefill_chunks = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 2) {
      continue;
    }
    Expect(events[index].count <= 2,
           "active-decode prefill respects its token budget");
    if (previous_b_prefill != events.size()) {
      bool a_advanced = false;
      for (std::size_t between = previous_b_prefill + 1; between < index;
           ++between) {
        a_advanced =
            a_advanced || (events[between].kind == EventKind::kAdvance &&
                           events[between].label == 1);
      }
      Expect(a_advanced, "an active decoder advances between prefill chunks");
    }
    previous_b_prefill = index;
    ++b_prefill_chunks;
  }
  Expect(b_prefill_chunks == 4,
         "long active-decode prompt is split into bounded chunks");
  Expect(result_b.prefill_chunks == 4 && result_b.prefill_tokens == 7 &&
             result_b.active_decode_prefill_chunks == 4 &&
             result_b.max_prefill_chunk_tokens == 2 &&
             result_b.max_consecutive_active_prefill_chunks == 1,
         "chunk metrics capture the selected active-decode policy");
}

void TestPrefillYieldsToEveryDueDecoder(bool multi_token, bool batched) {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = multi_token;
  control->batched_multi_token_decode = batched;
  control->supports_batched_advance = multi_token;
  control->block_advance_label = 2;
  auto scheduler = MakeScheduler(control, 3, {.decode_active_tokens = 2});

  const std::size_t output_tokens = multi_token ? 30 : 10;
  auto request_a = scheduler->Submit({1}, output_tokens, 0.0F);
  auto request_b = scheduler->Submit({2}, output_tokens, 0.0F);
  control->WaitForAdvance(2);
  auto request_c = scheduler->Submit({3, 30, 31, 32, 33, 34, 35}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, output_tokens) &&
             result_b.tokens == ExpectedTokens(2, output_tokens) &&
             result_c.tokens == ExpectedTokens(3, 2),
         "all mixed prefill/decode trajectories remain isolated");

  const auto events = control->Events();
  std::size_t previous_c_prefill = events.size();
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 3) {
      continue;
    }
    if (previous_c_prefill != events.size()) {
      bool a_advanced = false;
      bool b_advanced = false;
      for (std::size_t between = previous_c_prefill + 1; between < index;
           ++between) {
        if (events[between].kind == EventKind::kAdvance) {
          a_advanced = a_advanced || events[between].label == 1;
          b_advanced = b_advanced || events[between].label == 2;
        }
      }
      Expect(a_advanced && b_advanced,
             "every due decoder advances before another prefill chunk");
    }
    previous_c_prefill = index;
  }
  Expect(result_c.max_consecutive_active_prefill_chunks == 1,
         "scheduler never runs consecutive chunks while decode is due");
}

void TestNonIncrementalRunnerFallsBackSafely() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23}, 2, 0.0F);
  control->ReleaseAdvance();

  Expect(request_a.Wait().tokens == ExpectedTokens(1, 8),
         "fallback preserves the active decoder trajectory");
  const auto result_b = request_b.Wait();
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "fallback preserves the admitted request trajectory");
  Expect(
      !result_b.incremental_prefill_supported && result_b.prefill_chunks == 1 &&
          result_b.max_prefill_chunk_tokens == 5 &&
          result_b.prefill_fallback_reason == "incremental_prefill_unavailable",
      "non-incremental runners report their full-prefill fallback");
}

void TestPendingLimitsRejectBeforeStateAdmission() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 1,
                                     .max_pending_requests_per_client = 1,
                                 });

  auto active = scheduler->Submit({1}, 4, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto queued = scheduler->Submit({2}, 1, 0.0F, {}, false,
                                  ClientMetadata("queued-client"));

  bool rejected = false;
  try {
    (void)scheduler->Submit({3}, 1, 0.0F, {}, false,
                            ClientMetadata("third-client"));
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kQueueFull;
  }
  Expect(rejected, "full pending queue rejects before admission");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "rejected request cannot allocate another runner state");

  control->ReleaseAdvance();
  Expect(active.Wait().tokens == ExpectedTokens(1, 4) &&
             queued.Wait().tokens == ExpectedTokens(2, 1),
         "accepted work survives a queue rejection");
}

void TestPendingClientsAreRoundRobinAndIndividuallyBounded() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 4,
                                     .max_pending_requests_per_client = 2,
                                 });

  auto active = scheduler->Submit({1}, 3, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto client_a_first =
      scheduler->Submit({2}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_a_second =
      scheduler->Submit({3}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_b =
      scheduler->Submit({4}, 1, 0.0F, {}, false, ClientMetadata("client-b"));

  bool client_rejected = false;
  try {
    (void)scheduler->Submit({5}, 1, 0.0F, {}, false,
                            ClientMetadata("client-a"));
  } catch (const TextGenerationError& error) {
    client_rejected = error.code() == TextGenerationErrorCode::kClientQueueFull;
  }
  Expect(client_rejected, "one client cannot monopolize the pending queue");

  control->ReleaseAdvance();
  (void)active.Wait();
  (void)client_a_first.Wait();
  (void)client_a_second.Wait();
  (void)client_b.Wait();

  const auto events = control->Events();
  const std::size_t a_first = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_second = EventIndex(events, EventKind::kPrefill, 3);
  const std::size_t b_first = EventIndex(events, EventKind::kPrefill, 4);
  Expect(a_first < b_first && b_first < a_second,
         "pending clients rotate before one client receives another admission");
}

void TestExpiredQueuedRequestNeverConsumesState() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto active = scheduler->Submit({1}, 3, 0.0F);
  control->WaitForAdvance(1);
  auto expired = scheduler->Submit(
      {2}, 1, 0.0F, {}, false,
      TextRequestMetadata{
          .client_id = "expired-client",
          .deadline = TextGenerationScheduler::Clock::now() -
                      std::chrono::milliseconds{1},
          .request_start = TextGenerationScheduler::Clock::now(),
      });
  control->ReleaseAdvance();

  bool deadline_reported = false;
  try {
    (void)expired.Wait();
  } catch (const TextGenerationError& error) {
    deadline_reported =
        error.code() == TextGenerationErrorCode::kDeadlineExceeded;
  }
  Expect(deadline_reported, "expired queued work reports a stable deadline");
  Expect(active.Wait().tokens == ExpectedTokens(1, 3),
         "expired queued work does not disturb active generation");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "expired queued work never creates or acquires another state");
}

void TestSlowConsumerOutputIsBoundedAndReclaimed() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_buffered_output_bytes_per_request = 4,
                                     .max_buffered_output_bytes_total = 4,
                                 });

  auto request = scheduler->Submit({5}, 5, 0.0F, {}, true);
  std::mutex callback_mutex;
  std::condition_variable callback_condition;
  bool callback_entered = false;
  bool release_callback = false;
  bool backpressure_reported = false;
  std::jthread consumer([&] {
    try {
      (void)request.Wait([&](std::string_view) {
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_entered = true;
        control->ReleaseAdvance();
        callback_condition.notify_all();
        callback_condition.wait(lock, [&] { return release_callback; });
        return true;
      });
    } catch (const TextGenerationError& error) {
      backpressure_reported =
          error.code() == TextGenerationErrorCode::kOutputBackpressure;
    }
  });

  {
    std::unique_lock<std::mutex> lock(callback_mutex);
    const bool entered = callback_condition.wait_for(
        lock, kTestTimeout, [&] { return callback_entered; });
    Expect(entered, "slow consumer receives its first output piece");
  }
  control->WaitForInvalidations(1);
  {
    const std::lock_guard<std::mutex> lock(callback_mutex);
    release_callback = true;
  }
  callback_condition.notify_all();
  consumer.join();

  Expect(backpressure_reported,
         "bounded output queue fails a persistently slow consumer");
  Expect(scheduler->buffered_output_bytes() == 0,
         "failed slow-consumer output is fully reclaimed");
  Expect(scheduler->max_buffered_output_bytes() <= 4,
         "server-wide buffered output never exceeds its declared limit");

  const auto replacement = scheduler->Submit({6}, 2, 0.0F).Wait();
  Expect(replacement.tokens == ExpectedTokens(6, 2),
         "state is reusable after output backpressure cancellation");
}

void TestGeneratedOutputLimitAppliesWithoutStreaming() {
  auto control = std::make_shared<FakeControl>();
  control->device_usable = false;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_output_bytes_per_request = 4,
                                 });

  bool output_limit_reported = false;
  try {
    (void)scheduler->Submit({7}, 3, 0.0F).Wait();
  } catch (const TextGenerationError& error) {
    output_limit_reported =
        error.code() == TextGenerationErrorCode::kOutputLimit;
  }
  Expect(output_limit_reported,
         "non-streaming generation obeys its output byte limit");
  Expect(control->device_probes == 0 && !scheduler->device_lost(),
         "a scheduler-classified failure never probes the device");
}

void TestCompletedAbandonedStreamReleasesOutputBudget() {
  for (const bool replace_handle : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->block_prefill_label = 2;
    auto scheduler =
        MakeScheduler(control, 1, {},
                      {
                          .max_buffered_output_bytes_per_request = 3,
                          .max_buffered_output_bytes_total = 3,
                      });

    TextGenerationScheduler::Request barrier;
    {
      auto abandoned = scheduler->Submit({1}, 1, 0.0F, {}, true);
      barrier = scheduler->Submit({2}, 1, 0.0F);
      // The next request cannot enter this single runner until the first has
      // completed. Leave its streamed token unread, as when HTTP headers fail.
      control->WaitForPrefill(2);
      Expect(abandoned.phase() == TextRequestPhase::kTerminal &&
                 scheduler->buffered_output_bytes() == 3,
             "completed stream retains its unread output before abandonment");
      if (replace_handle)
        abandoned = {};
    }
    control->ReleasePrefill();
    Expect(barrier.Wait().tokens == ExpectedTokens(2, 1),
           "abandoning completed output leaves the active request usable");

    std::string output;
    TextGenerationScheduler::Result replacement;
    try {
      replacement = scheduler->Submit({3}, 1, 0.0F, {}, true)
                        .Wait([&](std::string_view piece) {
                          output += piece;
                          return true;
                        });
    } catch (const TextGenerationError& error) {
      Expect(false,
             std::string("replacement stream failed after abandonment: ") +
                 error.what());
    }
    Expect(replacement.tokens == ExpectedTokens(3, 1) && output == "300" &&
               !replacement.cancelled,
           "replacement stream can reuse abandoned output capacity");
    Expect(scheduler->buffered_output_bytes() == 0,
           "abandoned and consumed output leave no reserved capacity");
  }
}

void TestCompletedStreamOutlivesScheduler() {
  for (const bool consume : {false, true}) {
    TextGenerationScheduler::Request request;
    {
      auto scheduler = MakeScheduler(std::make_shared<FakeControl>(), 1);
      request = scheduler->Submit({1}, 1, 0.0F, {}, true);
      // Completion of later work proves this unread stream is already done.
      (void)scheduler->Submit({2}, 1, 0.0F).Wait();
      Expect(request.phase() == TextRequestPhase::kTerminal,
             "stream completes before its scheduler is destroyed");
    }
    if (consume) {
      std::string output;
      const auto result = request.Wait([&](std::string_view piece) {
        output += piece;
        return true;
      });
      Expect(result.tokens == ExpectedTokens(1, 1) && output == "100" &&
                 !result.cancelled,
             "completed output remains readable after scheduler destruction");
    }
    // Both unread and consumed requests release their final owner here.
  }
}

void TestMidGenerationAdmissionAndIsolatedTrajectories() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1, 10}, 4, 0.0F, {}, true);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 2, 0.0F, {}, true);
  control->ReleaseAdvance();

  std::vector<std::string> pieces_a;
  const auto result_a = request_a.Wait([&](std::string_view piece) {
    pieces_a.emplace_back(piece);
    return true;
  });
  std::vector<std::string> pieces_b;
  const auto result_b = request_b.Wait([&](std::string_view piece) {
    pieces_b.emplace_back(piece);
    return true;
  });

  Expect(result_a.tokens == ExpectedTokens(1, 4),
         "request A follows its isolated greedy trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "request B follows its isolated greedy trajectory");
  Expect(pieces_a.size() == 4 && pieces_b.size() == 2,
         "each client receives only its own token pieces");

  const auto events = control->Events();
  const std::size_t b_prefill = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_last_advance =
      EventIndex(events, EventKind::kAdvance, 1, 3);
  Expect(b_prefill < a_last_advance,
         "request B begins prefill before request A completes");
}

void TestMidGenerationRequestJoinsNextDecodeBatch() {
  auto control = std::make_shared<FakeControl>();
  control->supports_batched_advance = true;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2}, 4, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 8) &&
             result_b.tokens == ExpectedTokens(2, 4),
         "dynamic batching preserves both isolated trajectories");

  bool joined = false;
  for (const auto& labels : control->AdvanceBatches()) {
    joined = joined || labels == std::vector<TextRunnerToken>({1, 2}) ||
             labels == std::vector<TextRunnerToken>({2, 1});
  }
  Expect(joined,
         "request B joins request A at a decode boundary after prefill");
  Expect(result_a.physical_execution_width == 2 &&
             result_b.physical_execution_width == 2 &&
             result_a.execution_plan == "batched-w2" &&
             result_b.execution_plan == "batched-w2",
         "joined requests report the real W=2 execution plan");
}

void TestFifoReplacementAdmissionWithOneSlot() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto request_a = scheduler->Submit({1, 10}, 2, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 1, 0.0F);
  auto request_c = scheduler->Submit({3, 30}, 1, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(!result_a.cancelled && !result_b.cancelled && !result_c.cancelled,
         "all FIFO requests complete");

  std::vector<TextRunnerToken> prefill_order;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kPrefill) {
      prefill_order.push_back(event.label);
    }
  }
  Expect(prefill_order == std::vector<TextRunnerToken>({1, 2, 3}),
         "replacement admission preserves FIFO order");
}

void TestServerMetricsAreLive() {
  namespace metrics = gufo::server::detail;
  const auto read = [](const auto& value) {
    return value.load(std::memory_order_relaxed);
  };
  const auto prompt_before = read(metrics::TotalPromptTokens());
  const auto generated_before = read(metrics::TotalGenTokens());
  const auto processing_before = read(metrics::RequestsProcessing());
  const auto deferred_before = read(metrics::RequestsDeferred());
  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);

    auto active = scheduler->Submit({1, 10}, 3, 0.0F);
    control->WaitForAdvance(1);
    Expect(read(metrics::TotalPromptTokens()) - prompt_before == 2,
           "prompt tokens are counted when prefill executes");
    Expect(read(metrics::TotalGenTokens()) - generated_before == 1,
           "generated tokens are counted before the request completes");
    Expect(read(metrics::RequestsProcessing()) - processing_before == 1,
           "the running request is reported");

    auto queued = scheduler->Submit({2, 20}, 1, 0.0F);
    Expect(read(metrics::RequestsDeferred()) - deferred_before == 1,
           "the waiting request is reported");
    control->ReleaseAdvance();
    const auto active_result = active.Wait();
    const auto queued_result = queued.Wait();
    Expect(active_result.tokens.size() == 3, "active request completes");
    Expect(queued_result.tokens.size() == 1, "queued request completes");
    Expect(active_result.token_metrics_recorded &&
               queued_result.token_metrics_recorded,
           "scheduler results identify live token accounting");
    gufo::server::RecordServerMetrics(active_result);
    gufo::server::RecordServerMetrics(queued_result);
    Expect(read(metrics::TotalPromptTokens()) - prompt_before == 4,
           "HTTP completion does not count the prompt again");
    Expect(read(metrics::TotalGenTokens()) - generated_before == 4,
           "every generated token is counted once");
  }
  Expect(read(metrics::RequestsProcessing()) == processing_before &&
             read(metrics::RequestsDeferred()) == deferred_before,
         "a stopped scheduler withdraws its load");
}

void TestMetricsDuringCacheAdmission() {
  namespace metrics = gufo::server::detail;
  for (const bool cancel : {false, true}) {
    const auto processing_before = metrics::RequestsProcessing().load();
    const auto deferred_before = metrics::RequestsDeferred().load();
    auto control = std::make_shared<FakeControl>();
    std::binary_semaphore entered(0), release(0);
    std::atomic<bool> arm{false};
    control->snapshot_callback = [&] {
      if (arm.exchange(false)) {
        entered.release();
        release.acquire();
      }
    };
    auto scheduler = MakeScheduler(control, 2);
    const auto first = scheduler->Submit({1, 10}, 1, 0.0F).Wait();
    Expect(first.tokens == ExpectedTokens(1, 1), "first request completes");
    Expect(metrics::RequestsProcessing().load() == processing_before,
           "completed requests retire before Wait returns");

    arm.store(true);
    auto next = scheduler->Submit({1, 10, 100, 11}, 1, 0.0F);
    Expect(entered.try_acquire_for(kTestTimeout),
           "continuation admission captures its live checkpoint");
    Expect(metrics::RequestsProcessing().load() == processing_before + 1 &&
               metrics::RequestsDeferred().load() == deferred_before,
           "cache preparation remains visible as processing");
    auto queued = scheduler->Submit({2, 20}, 1, 0.0F);
    Expect(metrics::RequestsDeferred().load() == deferred_before + 1,
           "requests waiting behind cache preparation remain deferred");
    if (cancel) {
      next.Cancel();
      queued.Cancel();
    }
    release.release();
    const auto result = next.Wait();
    const auto queued_result = queued.Wait();
    Expect(result.cancelled == cancel && queued_result.cancelled == cancel,
           "admission and queued cancellation preserve request outcomes");
    if (!cancel)
      Expect(result.cache_hit && result.cached_prompt_tokens == 3,
             "continuation still reuses the live frontier");
    Expect(metrics::RequestsProcessing().load() == processing_before &&
               metrics::RequestsDeferred().load() == deferred_before,
           "completion and cancellation retire their gauges exactly once");
  }
  auto scheduler = MakeScheduler(std::make_shared<FakeControl>(), 1);
  TextRequestMetadata metadata;
  metadata.prompt_context = std::make_shared<gufo::server::TextPromptContext>();
  bool failed = false;
  try {
    (void)scheduler->Submit({3}, 1, 0.0F, {}, false, metadata).Wait();
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  Expect(failed, "unsupported prompt context fails during admission");
  Expect(metrics::RequestsProcessing().load() == 0 &&
             metrics::RequestsDeferred().load() == 0,
         "failed admission retires its processing count");
  Expect(scheduler->Submit({4}, 1, 0.0F).Wait().tokens == ExpectedTokens(4, 1),
         "failed admission releases the session for replacement work");
}

void TestSessionStatesFollowAdmittedRequests() {
  namespace metrics = gufo::server::detail;
  const auto processing_before = metrics::RequestsProcessing().load();
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  control->block_prefill_label = 2;
  auto scheduler = MakeScheduler(control, 2);
  const auto all_idle = [&] {
    const auto states = scheduler->SessionStates();
    return states.size() == 2 &&
           std::none_of(states.begin(), states.end(), [](const auto& state) {
             return state.processing || state.request_id != 0 ||
                    state.prompt_tokens != 0 || state.generated_tokens != 0;
           });
  };
  Expect(all_idle(), "every session starts idle");

  auto first = scheduler->Submit({1, 10}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto states = scheduler->SessionStates();
  Expect(states[0].processing && states[0].request_id == first.id() &&
             states[0].prompt_tokens == 2 &&
             states[0].cached_prompt_tokens == 0 &&
             states[0].processed_prompt_tokens == 2 &&
             states[0].generated_tokens == 1 &&
             states[0].remaining_tokens == 7 && !states[1].processing,
         "a decoding request reports its session progress");
  auto second = scheduler->Submit({2, 20, 21}, 1, 0.0F);
  Expect(!scheduler->SessionStates()[1].processing,
         "a queued request does not hold a session");
  control->ReleaseAdvance();
  control->WaitForPrefill(2);
  states = scheduler->SessionStates();
  Expect(states[0].processing && states[0].request_id == first.id() &&
             states[1].processing && states[1].request_id == second.id() &&
             states[1].prompt_tokens == 3 &&
             states[1].processed_prompt_tokens == 0 &&
             states[1].generated_tokens == 0 && states[1].remaining_tokens == 1,
         "concurrent requests hold distinct sessions");
  Expect(metrics::RequestsProcessing().load() - processing_before == 2,
         "processing sessions match the processing gauge");
  control->ReleasePrefill();
  Expect(first.Wait().tokens.size() == 8 && second.Wait().tokens.size() == 1,
         "both requests complete");
  Expect(all_idle(), "completed requests release their sessions");

  auto cache_control = std::make_shared<FakeControl>();
  cache_control->snapshot_callback = [] {};
  auto cache_scheduler = MakeScheduler(cache_control, 2);
  (void)cache_scheduler->Submit({1, 10}, 1, 0.0F).Wait();
  {
    const std::lock_guard<std::mutex> lock(cache_control->mutex);
    cache_control->block_prefill_label = 1;
  }
  auto continuation = cache_scheduler->Submit({1, 10, 100, 11}, 2, 0.0F);
  cache_control->WaitForPrefill(1);
  const auto resumed = cache_scheduler->SessionStates().front();
  Expect(resumed.processing && resumed.request_id == continuation.id() &&
             resumed.prompt_tokens == 4 && resumed.cached_prompt_tokens == 3 &&
             resumed.processed_prompt_tokens == 0,
         "a continuation reports its cached prompt tokens");
  continuation.Cancel();
  cache_control->ReleasePrefill();
  Expect(continuation.Wait().cancelled, "continuation is cancelled");
  Expect(!cache_scheduler->SessionStates().front().processing,
         "cancelled requests release their sessions");

  auto speculative = std::make_shared<FakeControl>();
  speculative->multi_token_decode = true;
  Expect(MakeScheduler(speculative, 1)->SessionStates().front().speculative,
         "multi-token decoding is reported as speculative");
}

void TestQueuedAndPrefillCancellation() {
  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);

    auto active = scheduler->Submit({1, 10}, 2, 0.0F);
    control->WaitForAdvance(1);
    auto queued = scheduler->Submit({2, 20}, 1, 0.0F);
    queued.Cancel();
    control->ReleaseAdvance();

    Expect(!active.Wait().cancelled, "active request completes normally");
    Expect(queued.Wait().cancelled, "queued request cancellation is reported");
    const auto events = control->Events();
    Expect(EventIndex(events, EventKind::kPrefill, 2) == events.size(),
           "cancelled queued request never acquires a model state");
  }

  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    control->block_prefill_label = 4;
    auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

    auto active = scheduler->Submit({1}, 4, 0.0F);
    control->WaitForAdvance(1);
    auto request = scheduler->Submit({4, 40, 41, 42}, 2, 0.0F);
    control->ReleaseAdvance();
    control->WaitForPrefill(4);
    request.Cancel();
    control->ReleasePrefill();

    const auto result = request.Wait();
    Expect(!active.Wait().cancelled,
           "active decoder survives another request cancellation");
    Expect(result.cancelled, "active-decode prefill cancellation is reported");
    Expect(EventIndex(control->Events(), EventKind::kAdvance, 4) ==
               control->Events().size(),
           "cancelled prefill never advances decode");
  }
}

void TestDecodeCancellationAndStateReclamation() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1);

  auto cancelled = scheduler->Submit({5, 50}, 5, 0.0F, {}, true);
  std::size_t delivered_pieces = 0;
  const auto cancelled_result = cancelled.Wait([&](std::string_view) {
    ++delivered_pieces;
    control->ReleaseAdvance();
    return false;
  });
  Expect(cancelled_result.cancelled,
         "callback cancellation reaches the scheduler");
  Expect(delivered_pieces == 1,
         "no token is published after callback cancellation");

  auto replacement = scheduler->Submit({6, 60}, 2, 0.0F);
  const auto replacement_result = replacement.Wait();
  Expect(replacement_result.tokens == ExpectedTokens(6, 2),
         "cancelled state slot is immediately reusable");
  Expect(control->invalidations.load(std::memory_order_relaxed) >= 1,
         "decode cancellation invalidates partial model state");
}

void TestFourResidentRequestsMakeProgress() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 4);

  std::vector<TextGenerationScheduler::Request> requests;
  for (TextRunnerToken label = 1; label <= 4; ++label) {
    requests.push_back(scheduler->Submit({label, label + 10}, 3, 0.0F));
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto result = requests[index].Wait();
    Expect(result.tokens ==
               ExpectedTokens(static_cast<TextRunnerToken>(index + 1), 3),
           "C=4 serial fallback preserves isolated output");
    Expect(result.requested_logical_concurrency == 4 &&
               result.physical_execution_width == 1 &&
               result.execution_plan == "serial-fallback",
           "C=4 telemetry exposes the physical serial fallback");
  }
}

void TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement() {
  auto control = std::make_shared<FakeControl>();
  control->throw_advance_label = 9;
  auto scheduler = MakeScheduler(control, 1);

  bool failed = false;
  try {
    auto request = scheduler->Submit({9, 90}, 2, 0.0F);
    (void)request.Wait();
  } catch (const std::runtime_error& exception) {
    failed = std::string_view(exception.what()) ==
             "injected scheduler runner failure";
  }
  Expect(failed, "runner failure reaches the submitting client");
  Expect(!scheduler->SessionStates().front().processing,
         "a failed request releases its visible session");
  // A probe still pending at its bound also reports a usable device.
  Expect(control->device_probes == 1 && !scheduler->device_lost(),
         "a failure on a usable device is probed once and stays recoverable");

  control->throw_advance_label.reset();
  auto replacement = scheduler->Submit({8, 80}, 2, 0.0F);
  Expect(replacement.Wait().tokens == ExpectedTokens(8, 2),
         "replacement request succeeds after runner failure");
  Expect(control->device_probes == 1, "successful work never probes");
}

void TestIdleDeviceProbe() {
  using Status = TextModelRunner::DeviceProbeStatus;
  auto control = std::make_shared<FakeControl>();
  control->idle_probe_status = Status::kPending;
  TextSchedulerPolicy policy{.device_probe_interval =
                                 std::chrono::milliseconds(10)};
  const auto before = gufo::server::detail::DeviceLostTotal().load();
  auto scheduler = MakeScheduler(control, 1, {}, policy);
  {
    std::unique_lock lock(control->mutex);
    Expect(
        control->condition.wait_for(lock, kTestTimeout,
                                    [&] { return control->idle_probes >= 2; }),
        "an idle scheduler submits/polls without generation or HTTP traffic");
  }
  Expect(!scheduler->device_lost(), "pending probe is inconclusive");
  Expect(gufo::server::detail::DeviceLostTotal() == before,
         "pending probes do not count as losses");
  control->block_prefill_label = 1;
  auto request = scheduler->Submit({1}, 2, 0.0F);
  control->WaitForPrefill(1);
  const auto probes = control->idle_probes.load();
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  Expect(control->idle_probes == probes,
         "a pending idle probe never polls during active inference");
  control->ReleasePrefill();
  Expect(request.Wait().tokens == ExpectedTokens(1, 2),
         "arriving inference succeeds with an outstanding probe");
  control->idle_probe_status = Status::kLost;
  const auto deadline = TextGenerationScheduler::Clock::now() + kTestTimeout;
  while (!scheduler->device_lost() &&
         TextGenerationScheduler::Clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  Expect(scheduler->device_lost(), "idle probe loss is fatal and sticky");
  scheduler.reset();
  Expect(gufo::server::detail::DeviceLostTotal() == before + 1,
         "confirmed idle loss increments the counter once");
}

void TestDeviceLossIsStickyAndReported() {
  auto control = std::make_shared<FakeControl>();
  control->throw_advance_label = 9;
  control->block_advance_label = 9;
  control->device_usable = false;
  auto scheduler = MakeScheduler(control, 2);

  const auto expect_device_lost = [](const auto& action,
                                     std::string_view message) {
    bool lost = false;
    try {
      action();
    } catch (const TextGenerationError& error) {
      lost = error.code() == TextGenerationErrorCode::kDeviceLost &&
             error.http_status() == 503 &&
             std::string_view(error.stable_code()) == "device_lost" &&
             std::string_view(error.what()) == gufo::server::kDeviceLostMessage;
    }
    Expect(lost, message);
  };
  auto failing = scheduler->Submit({9, 90}, 2, 0.0F);
  control->WaitForAdvance(9);
  const auto invalidations_before_loss = control->invalidations.load();
  auto queued = scheduler->Submit({8, 80}, 2, 0.0F);
  auto cancelled = scheduler->Submit({7, 70}, 2, 0.0F);
  cancelled.Cancel();
  control->ReleaseAdvance();
  expect_device_lost([&] { (void)failing.Wait(); },
                     "a failure on a lost device reports device_lost");
  Expect(scheduler->device_lost() && control->device_probes == 1,
         "the failing work unit probes the device once");
  expect_device_lost([&] { (void)queued.Wait(); },
                     "queued work fails without touching the lost device");
  expect_device_lost([&] { (void)cancelled.Wait(); },
                     "shutdown never resets a cancelled lost-device request");
  Expect(control->invalidations == invalidations_before_loss,
         "lost-device failure skips state invalidation and its HIP cleanup");
  const auto states = scheduler->SessionStates();
  Expect(std::none_of(states.begin(), states.end(),
                      [](const auto& state) { return state.processing; }),
         "device loss retires visible sessions without HIP cleanup");
  const auto events = control->Events();
  Expect(std::none_of(events.begin(), events.end(),
                      [](const auto& event) { return event.label != 9; }),
         "no peer performs model work after loss");

  control->throw_advance_label.reset();
  control->device_usable = true;
  expect_device_lost([&] { (void)scheduler->Submit({8, 80}, 2, 0.0F); },
                     "submissions after device loss fail before admission");
  Expect(control->device_probes == 1, "device loss is never probed again");
}

void TestStopSequenceChunkBoundaries() {
  using gufo::server::StopSequenceFilter;
  const auto check = [](std::vector<std::string> stops, std::string text,
                        std::string expected, std::string matched) {
    // Include single-byte token pieces and every two-piece boundary.
    for (std::size_t split = 0; split <= text.size() + 1; ++split) {
      StopSequenceFilter filter(stops);
      std::string output;
      if (split > text.size()) {
        for (const char byte : text)
          output += filter.Push(std::string_view(&byte, 1));
      } else {
        output += filter.Push(std::string_view(text).substr(0, split));
        output += filter.Push(std::string_view(text).substr(split));
      }
      output += filter.Finish();
      Expect(output == expected,
             "stop matching is independent of token boundaries");
      Expect(filter.stopped() == !matched.empty(), "correct stop detection");
      if (filter.stopped())
        Expect(filter.matched_sequence() == matched,
               "correct matched sequence");
    }
  };
  check({"<STOP>"}, "before<STOP>after", "before", "<STOP>");
  check({"END", "STOP"}, "stENDlaterSTOP", "st", "END");
  check({"abc", "b"}, "abc", "a", "b");
  check({"aba", "ba"}, "aba", "", "aba");
  check({"abab"}, "aaababtail", "aa", "abab");
  check({"┌終"}, "hi┌終later", "hi", "┌終");
  check({"STOP"}, "SSTOSTxST", "SSTOSTxST", "");
  check({"STOP"}, "STOP", "", "STOP");
}

void TestStopSequencesPreserveExecutedState() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool multi : {false, true}) {
    for (const bool preview : {false, true}) {
      for (const bool first : {false, true}) {
        auto control = std::make_shared<FakeControl>();
        control->incremental_text_is_exact = true;
        control->multi_token_decode = multi;
        control->preview_first_token = preview;
        control->snapshot_callback = [] {};
        control->output_pieces =
            first ? std::vector<std::string>{"<STOP>tail", "later", "last"}
                  : std::vector<std::string>{"ab<ST", "OP>tail", "later"};
        auto scheduler = MakeScheduler(control, 1);
        TextGenerationScheduler::RequestMetadata metadata;
        metadata.stop_sequences = {"<STOP>"};
        std::string streamed;
        auto request = scheduler->Submit({1, 10}, 8, 0.0F, {}, true, metadata);
        const auto result = request.Wait([&](std::string_view piece) {
          streamed += piece;
          return true;
        });
        Expect(
            result.finish_reason == Finish::kStopSequence && !result.cancelled,
            "stop sequence is a successful completion");
        Expect(result.stop_sequence == "<STOP>" &&
                   result.text == (first ? "" : "ab") &&
                   streamed == result.text,
               "neither the stop marker nor speculative tail leaks");
        Expect(result.tokens.size() == (first ? 1U : 2U) &&
                   result.completion_tokens == result.tokens.size(),
               "usage includes the token that completed the stop");
        // Repeating the prompt must not resume from an incorrectly labelled
        // generated frontier, including when preview or a speculative tail ran.
        const auto replay =
            scheduler->Submit({1, 10}, 8, 0.0F, {}, true, metadata).Wait();
        Expect(replay.tokens == result.tokens && replay.text == result.text,
               "a stopped request leaves a safe reusable prompt cache");
      }
    }
  }
}

void TestStopPrefixFlushAndBatchIsolation() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->incremental_text_is_exact = true;
    control->multi_token_decode = multi;
    control->batched_multi_token_decode = multi;
    control->supports_batched_advance = true;
    control->block_prefill_label = 1;
    control->output_pieces = {"hello", "<ST", "OP>tail", "ST"};
    auto scheduler = MakeScheduler(control, 2);
    TextGenerationScheduler::RequestMetadata metadata;
    metadata.stop_sequences = {"<STOP>"};
    auto first = scheduler->Submit({1}, 8, 0.0F, {}, true, metadata);
    control->WaitForPrefill(1);
    auto peer = scheduler->Submit({2}, 4, 0.0F);
    control->ReleasePrefill();
    const auto stopped = first.Wait();
    const auto full = peer.Wait();
    Expect(stopped.text == "hello" &&
               stopped.finish_reason == Finish::kStopSequence,
           "batched request stops on its own marker");
    Expect(full.text == "hello<STOP>tailST" &&
               full.finish_reason == Finish::kLength,
           "peer without stop sequences keeps every accepted token");
    metadata.stop_sequences = {"STXYZ"};
    for (const std::size_t limit : {4U, 8U}) {
      std::string streamed;
      const auto result =
          scheduler->Submit({4}, limit, 0.0F, {}, true, metadata)
              .Wait([&](std::string_view piece) {
                streamed += piece;
                return true;
              });
      Expect(result.text == "hello<STOP>tailST" && streamed == result.text &&
                 result.stop_sequence.empty() &&
                 result.finish_reason ==
                     (limit == 4 ? Finish::kLength : Finish::kStop),
             "unmatched stop prefix is flushed on both EOS and length");
    }
  }
}

void TestCancellationWithBufferedStopPrefix() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->incremental_text_is_exact = true;
    control->multi_token_decode = multi;
    control->block_advance_label = 1;
    control->output_pieces = {"safe<ST", "OP>tail", "end"};
    auto scheduler = MakeScheduler(control, 1);
    TextGenerationScheduler::RequestMetadata metadata;
    metadata.stop_sequences = {"<STOP>"};
    auto request = scheduler->Submit({1}, 8, 0.0F, {}, true, metadata);
    control->WaitForAdvance(1);
    request.Cancel();
    control->ReleaseAdvance();
    std::string streamed;
    const auto result = request.Wait([&](std::string_view piece) {
      streamed += piece;
      return true;
    });
    Expect(
        result.cancelled &&
            result.finish_reason ==
                gufo::server::TextGenerationBackend::FinishReason::kCancelled &&
            streamed.find("<ST") == std::string::npos,
        "cancellation does not flush a buffered stop prefix");
    const auto replacement = scheduler->Submit({2}, 3, 0.0F).Wait();
    Expect(!replacement.cancelled && replacement.text == "safe<STOP>tailend",
           "cancellation with a partial stop releases the runner");
  }
}

}  // namespace

void TestFirstTokenPrecedesSnapshotAndPreservesBudget() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = multi;
    control->preview_first_token = true;
    std::binary_semaphore captured(0), release(0);
    control->snapshot_callback = [&] {
      captured.release();
      release.acquire();
    };
    auto scheduler = MakeScheduler(control, 1);
    auto request = scheduler->Submit({1, 10}, 7, 0.0F, {}, true);
    std::counting_semaphore<16> first_token(0);
    auto result = std::async(std::launch::async, [&] {
      return request.Wait([&](std::string_view) {
        first_token.release();
        return true;
      });
    });
    Expect(captured.try_acquire_for(kTestTimeout), "snapshot capture reached");
    const bool published = first_token.try_acquire_for(std::chrono::seconds(1));
    release.release();
    Expect(published, "first token is delivered while snapshot capture blocks");
    Expect(result.get().tokens == ExpectedTokens(1, 7),
           "preview is emitted once and counts toward the original budget");
  }
  for (const std::size_t limit : {1U, 7U}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = true;
    control->preview_first_token = true;
    control->batched_multi_token_decode = true;
    control->supports_batched_advance = true;
    auto scheduler = MakeScheduler(control, 4);
    std::vector<TextGenerationScheduler::Request> requests;
    for (TextRunnerToken label = 1; label <= 4; ++label)
      requests.push_back(scheduler->Submit({label, 10}, limit, 0.0F));
    for (std::size_t index = 0; index < requests.size(); ++index)
      Expect(requests[index].Wait().tokens ==
                 ExpectedTokens(static_cast<TextRunnerToken>(index + 1), limit),
             "batched previews retain independent token limits");
  }
}

void TestSnapshotDoesNotBlockOtherRequests() {
  for (const auto [multi, history] :
       {std::pair{false, false}, {true, false}, {false, true}, {true, true}}) {
    auto control = std::make_shared<FakeControl>();
    control->max_context = 4096;
    control->multi_token_decode = multi;
    control->preview_first_token = true;
    control->supports_batched_advance = true;
    control->batched_multi_token_decode = multi;
    std::binary_semaphore entered(0), release(0);
    std::atomic<unsigned> captures{0};
    control->snapshot_callback = [&] {
      if (captures.fetch_add(1) == 0) {
        entered.release();
        release.acquire();
      }
    };
    auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});
    const std::vector<TextRunnerToken> prompt =
        history ? std::vector<TextRunnerToken>(2200, 1)
                : std::vector<TextRunnerToken>{1, 10};
    auto first = scheduler->Submit(prompt, 7, 0.0F);
    const bool started = entered.try_acquire_for(kTestTimeout);
    auto second = scheduler->Submit({2, 20, 21, 22, 23, 24, 25}, 7, 0.0F);
    auto result = std::async(std::launch::async, [&] { return second.Wait(); });
    const bool independent =
        result.wait_for(kTestTimeout) == std::future_status::ready;
    release.release();
    Expect(started && independent,
           "neither an intermediate nor a final capture blocks other requests");
    const auto second_result = result.get();
    Expect(second_result.tokens == ExpectedTokens(2, 7),
           "other request stays independent");
    Expect(second_result.max_prefill_chunk_tokens == 2 &&
               second_result.prefill_chunks == 4,
           "prefill stays bounded while an already-published request captures");
    const auto first_result = first.Wait();
    Expect(first_result.tokens == ExpectedTokens(1, 7),
           "captured request resumes exactly");
    const double advance_ms =
        control->first_request_advance_ns.load(std::memory_order_relaxed) / 1e6;
    Expect(advance_ms > 0 && first_result.decode_ms >= advance_ms,
           "async snapshot time must not be subtracted from timed model work");
    auto cached = scheduler->Submit(prompt, 7, 0.0F).Wait();
    Expect(cached.cache_hit && cached.tokens == ExpectedTokens(1, 7),
           "asynchronous capture retains the immutable prompt frontier");
  }
}

void TestCapturedDecoderResumesBeforeMorePrefill() {
  auto control = std::make_shared<FakeControl>();
  control->max_context = 4096;
  control->preview_first_token = true;
  control->block_prefill_label = 2;
  std::binary_semaphore entered(0), release(0), finished(0);
  std::atomic<unsigned> captures{0};
  control->snapshot_callback = [&] {
    if (captures.fetch_add(1) == 0) {
      entered.release();
      release.acquire();
      finished.release();
    }
  };
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});
  auto first = scheduler->Submit({1, 10}, 3, 0.0F);
  Expect(entered.try_acquire_for(kTestTimeout), "first capture starts");
  auto second = scheduler->Submit({2, 20, 21, 22, 23, 24, 25, 26, 27}, 1, 0.0F);
  // Finish the capture while the peer's bounded chunk is still running.
  control->WaitForPrefill(2);
  release.release();
  Expect(finished.try_acquire_for(kTestTimeout), "first capture finishes");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  control->ReleasePrefill();
  Expect(first.Wait().tokens == ExpectedTokens(1, 3),
         "captured request resumes exactly");
  Expect(second.Wait().tokens == ExpectedTokens(2, 1),
         "peer prefill completes");
  const auto events = control->Events();
  const auto position = [&](EventKind kind, TextRunnerToken label,
                            std::size_t occurrence) {
    for (std::size_t index = 0; index < events.size(); ++index)
      if (events[index].kind == kind && events[index].label == label &&
          occurrence-- == 0)
        return index;
    return events.size();
  };
  Expect(
      position(EventKind::kAdvance, 1, 0) < position(EventKind::kPrefill, 2, 1),
      "a request leaving capture decodes before another peer chunk");
}

void TestCapturesAtCapacityAllowQueuedProgress() {
  for (const auto [multi, history] :
       {std::pair{false, false}, {true, false}, {false, true}, {true, true}}) {
    for (const std::size_t capacity : {1U, 2U, 4U, 8U}) {
      auto control = std::make_shared<FakeControl>();
      control->max_context = 4096;
      control->multi_token_decode = multi;
      control->preview_first_token = true;
      control->supports_batched_advance = true;
      control->batched_multi_token_decode = multi;
      std::counting_semaphore<16> entered(0), release(0);
      std::atomic<std::size_t> captures{0};
      control->snapshot_callback = [&] {
        if (captures.fetch_add(1) < capacity) {
          entered.release();
          release.acquire();
        }
      };
      auto scheduler = MakeScheduler(control, capacity);
      std::vector<TextGenerationScheduler::Request> requests;
      for (std::size_t i = 0; i < capacity; ++i) {
        TextGenerationScheduler::RequestMetadata metadata;
        // Exercise captures during prefill as well as after first-token
        // publication, including the single-slot admission deadlock.
        metadata.cache_prefix_tokens = multi && !history ? 1 : 0;
        auto prompt = std::vector<TextRunnerToken>(history ? 2200 : 2, 10);
        prompt.front() = static_cast<TextRunnerToken>(i + 1);
        requests.push_back(
            scheduler->Submit(prompt, 7, 0.0F, {}, false, metadata));
        Expect(entered.try_acquire_for(kTestTimeout),
               "every resident reaches snapshot capture");
      }
      for (std::size_t i = capacity; i < capacity * 2; ++i)
        requests.push_back(scheduler->Submit(
            {static_cast<TextRunnerToken>(i + 1), 10}, 7, 0.0F, {}, false,
            ClientMetadata(std::to_string(i))));
      // Cancellation while a history copy is in flight must release the
      // slot after joining, without publishing an unexecuted prompt suffix.
      if (history)
        requests.front().Cancel();
      release.release(static_cast<std::ptrdiff_t>(capacity));
      auto results = std::async(std::launch::async, [&] {
        for (std::size_t i = 0; i < requests.size(); ++i) {
          const auto result = requests[i].Wait();
          if (history && i == 0)
            Expect(result.cancelled && result.tokens.empty(),
                   "cancellation joins only completed history work");
          else
            Expect(result.tokens ==
                       ExpectedTokens(static_cast<TextRunnerToken>(i + 1), 7),
                   "capture and queued requests retain independent output");
        }
      });
      Expect(results.wait_for(kTestTimeout) == std::future_status::ready,
             "all slots capturing must not deadlock queued admission");
      results.get();
      if (history) {
        auto prompt = std::vector<TextRunnerToken>(2200, 10);
        prompt.front() = 1;
        const auto resumed = scheduler->Submit(prompt, 7, 0.0F).Wait();
        Expect(resumed.cache_hit && resumed.cached_prompt_tokens == 2048 &&
                   resumed.tokens == ExpectedTokens(1, 7),
               "cancelled intermediate capture restores its exact frontier");
      }
    }
  }
}

void TestShutdownCancelsRunnerAcquisition() {
  auto control = std::make_shared<FakeControl>();
  auto pool = std::make_shared<TextRunnerPool>(
      std::make_shared<FakeRunner>(control), 1);
  auto lease = pool->Acquire({1, 10});
  auto scheduler = std::make_unique<TextGenerationScheduler>(pool);
  std::binary_semaphore acquiring(0);
  std::atomic<bool> notified{false};
  auto request = scheduler->Submit({2, 10}, 7, 0.0F, [&] {
    if (!notified.exchange(true))
      acquiring.release();
    return false;
  });
  Expect(acquiring.try_acquire_for(kTestTimeout),
         "request reaches admission with external runner lease");
  auto stopped = std::async(std::launch::async, [&] { scheduler.reset(); });
  const bool completed =
      stopped.wait_for(kTestTimeout) == std::future_status::ready;
  lease = {};
  Expect(completed, "shutdown cancels acquisition of a leased runner");
  stopped.get();
}

void TestProgressLoggingIsOptInAndBounded() {
  auto run = [](bool enabled) {
    auto control = std::make_shared<FakeControl>();
    control->prefill_capacity = 2;
    std::ostringstream output;
    auto* previous = std::clog.rdbuf(output.rdbuf());
    {
      auto scheduler = MakeScheduler(
          control, 1, {}, TextSchedulerPolicy{.log_progress = enabled});
      const auto result =
          scheduler->Submit({7, 70, 71, 72, 73}, 55, 0.0F).Wait();
      Expect(result.tokens.size() == 55,
             "progress logging preserves generated tokens");
    }
    std::clog.rdbuf(previous);
    return output.str();
  };

  Expect(run(false).find("[progress]") == std::string::npos,
         "progress logging is disabled by default");
  const auto output = run(true);
  Expect(output.find("phase=prefill tokens=2/5") != std::string::npos &&
             output.find("phase=prefill tokens=5/5") != std::string::npos,
         "progress logging reports model-owned prefill chunks");
  Expect(output.find("phase=decode tokens=50/55") != std::string::npos &&
             output.find("phase=decode tokens=55/55") != std::string::npos,
         "progress logging reports decode intervals and the final remainder");
}

void TestAdmissionLoggingIsDebugTierOnly() {
  const auto run = [](gufo::server::LogLevel level) {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    const auto previous_level = gufo::server::Logger::Level();
    gufo::server::Logger::SetLevel(level);
    std::ostringstream output;
    auto* previous = std::clog.rdbuf(output.rdbuf());
    bool rejected = false;
    {
      auto scheduler = MakeScheduler(control, 1, {},
                                     TextSchedulerPolicy{
                                         .max_pending_requests = 1,
                                         .max_pending_requests_per_client = 1,
                                     });
      auto active = scheduler->Submit({1}, 4, 0.0F, {}, false,
                                      ClientMetadata("active-client"));
      control->WaitForAdvance(1);
      auto queued = scheduler->Submit({2}, 1, 0.0F, {}, false,
                                      ClientMetadata("queued-client"));
      try {
        (void)scheduler->Submit({3}, 1, 0.0F, {}, false,
                                ClientMetadata("third-client"));
      } catch (const TextGenerationError& error) {
        rejected = error.code() == TextGenerationErrorCode::kQueueFull;
      }
      control->ReleaseAdvance();
      (void)active.Wait();
      (void)queued.Wait();
    }
    std::clog.rdbuf(previous);
    gufo::server::Logger::SetLevel(previous_level);
    Expect(rejected, "admission logging still rejects a full pending queue");
    return output.str();
  };

  Expect(run(gufo::server::LogLevel::kInfo).find("event=admission") ==
             std::string::npos,
         "admission detail is silent at the default level");

  const auto output = run(gufo::server::LogLevel::kDebug);
  Expect(output.find("[DEBUG] [scheduler] event=admitted") != std::string::npos,
         "debug tier records the admission decision");
  Expect(output.find(" client_id=queued-client queued=1") != std::string::npos,
         "admission detail reports the queue depth the request joined");
  Expect(output.find("event=admission_refused reason=queue_full queued=1 "
                     "limit=1 client_id=third-client") != std::string::npos,
         "a refusal names the limit and the rejected client");
}

void TestPromptProgressPrecedesStreamedTokens() {
  using gufo::server::TextGenerationBackend;
  auto control = std::make_shared<FakeControl>();
  control->prefill_capacity = 2;
  auto scheduler = MakeScheduler(control, 1);
  auto run = [&](std::vector<TextRunnerToken> prompt, bool stream,
                 bool report_progress = true) {
    std::vector<TextGenerationBackend::PromptProgress> progress;
    std::size_t tokens = 0;
    TextGenerationScheduler::RequestMetadata metadata;
    metadata.return_progress = report_progress;
    auto request =
        scheduler->Submit(std::move(prompt), 2, 0.0F, {}, stream, metadata);
    (void)request.Wait(
        [&](std::string_view) {
          ++tokens;
          return true;
        },
        [&](const TextGenerationBackend::PromptProgress& value) {
          Expect(tokens == 0, "prompt progress precedes generated tokens");
          Expect(
              progress.empty() || value.processed >= progress.back().processed,
              "prompt progress never moves backwards");
          progress.push_back(value);
          return true;
        });
    return progress;
  };

  const auto cold = run({7, 70, 71, 72, 73}, true);
  Expect(!cold.empty() && cold.back().total == 5 && cold.back().cache == 0 &&
             cold.back().processed == 5 && cold.back().time_ms >= 0,
         "cold prompt progress ends with the complete prompt");

  const auto cached = run({7, 70, 71, 72, 73, 700, 701, 80}, true);
  Expect(!cached.empty() && cached.back().total == 8 &&
             cached.back().cache == 7 && cached.back().processed == 8,
         "cached prompt progress counts reused tokens as processed");

  Expect(run({9, 90, 91}, false).empty(),
         "buffered requests do not publish prompt progress");
  Expect(run({9, 90, 91}, true, false).empty(),
         "streaming alone does not enable progress publication");
}

void TestStreamStartPrecedesPrefill() {
  using namespace std::chrono_literals;
  const auto submit = [](TextGenerationScheduler& scheduler,
                         std::vector<TextRunnerToken> prompt, bool stream) {
    return scheduler.Submit(std::move(prompt), 2, 0.0F, {}, stream,
                            TextGenerationScheduler::RequestMetadata{});
  };
  {
    // Admission starts a stream while its prompt is still being processed.
    auto control = std::make_shared<FakeControl>();
    control->block_prefill_label = 1;
    auto scheduler = MakeScheduler(
        control, 1, {}, {.stream_start_delay = std::chrono::hours(1)});
    auto request = submit(*scheduler, {1, 10, 11}, true);
    std::binary_semaphore started(0);
    std::atomic<int> starts{0};
    std::atomic<int> tokens{0};
    auto consume = std::async(std::launch::async, [&] {
      return request.Wait(
          [&](std::string_view) {
            ++tokens;
            return true;
          },
          {},
          [&] {
            Expect(tokens == 0, "stream start precedes generated tokens");
            ++starts;
            started.release();
            return true;
          });
    });
    control->WaitForPrefill(1);
    Expect(started.try_acquire_for(kTestTimeout),
           "admission starts a stream before prefill completes");
    control->ReleasePrefill();
    Expect(!consume.get().cancelled && tokens > 0 && starts == 1,
           "an admitted stream starts exactly once");
  }
  {
    // A stream waiting behind another request starts after the bound.
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler =
        MakeScheduler(control, 1, {}, {.stream_start_delay = 20ms});
    auto active = submit(*scheduler, {1, 10}, true);
    control->WaitForAdvance(1);
    auto queued = submit(*scheduler, {2, 20}, true);
    std::atomic<int> starts{0};
    std::binary_semaphore started(0);
    auto consume = std::async(std::launch::async, [&] {
      return queued.Wait({}, {}, [&] {
        Expect(queued.phase() == gufo::server::TextRequestPhase::kQueued,
               "a queued stream starts before admission");
        ++starts;
        started.release();
        return true;
      });
    });
    Expect(started.try_acquire_for(kTestTimeout),
           "a queued stream starts after its bound");
    control->ReleaseAdvance();
    Expect(!active.Wait().cancelled && !consume.get().cancelled && starts == 1,
           "admission after a queued start does not start again");
  }
  {
    // Buffered requests and failures before admission never start.
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(
        control, 1, {}, {.stream_start_delay = std::chrono::hours(1)});
    auto active = submit(*scheduler, {1, 10}, false);
    control->WaitForAdvance(1);
    auto queued = submit(*scheduler, {2, 20}, true);
    queued.Cancel();
    control->ReleaseAdvance();
    bool started = false;
    Expect(queued.Wait({}, {}, [&] { return started = true; }).cancelled &&
               !started,
           "a request cancelled while queued never starts");
    Expect(!active.Wait({}, {}, [&] { return started = true; }).cancelled &&
               !started,
           "buffered requests never start");
  }
}

void TestPromptProgressCancellation() {
  for (const bool callback_throws : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->prefill_capacity = 2;
    control->block_prefill_label = 1;
    auto scheduler = MakeScheduler(control, 2);
    TextGenerationScheduler::RequestMetadata metadata;
    metadata.return_progress = true;
    auto request =
        scheduler->Submit({1, 10, 11, 12}, 8, 0.0F, {}, true, metadata);
    control->WaitForPrefill(1);
    std::binary_semaphore received(0);
    auto consume = std::async(std::launch::async, [&] {
      try {
        const auto result = request.Wait({}, [&](const auto&) {
          received.release();
          if (callback_throws)
            throw std::runtime_error("progress callback failed");
          return false;
        });
        Expect(!callback_throws && result.cancelled,
               "progress callback can cancel before the first token");
      } catch (const std::runtime_error& error) {
        Expect(callback_throws &&
                   std::string_view(error.what()) == "progress callback failed",
               "progress callback exceptions propagate");
      }
    });
    Expect(received.try_acquire_for(kTestTimeout), "initial progress is live");
    request.Cancel();
    control->ReleasePrefill();
    consume.get();
    const auto peer = scheduler->Submit({2, 20}, 2, 0.0F).Wait();
    Expect(!peer.cancelled && peer.completion_tokens == 2,
           "progress cancellation leaves peer state usable");
  }
}

void TestIgnoreEosIsRequestScoped() {
  auto control = std::make_shared<FakeControl>();
  control->eos_after = 0;
  auto scheduler = MakeScheduler(control, 1);
  gufo::sampling::SamplingConfig sampling;
  const auto normal = scheduler->Submit({1}, 4, sampling).Wait();
  Expect(normal.tokens.empty() &&
             normal.finish_reason ==
                 gufo::server::TextGenerationBackend::FinishReason::kStop,
         "normal request stops at EOS");
  TextGenerationScheduler::RequestMetadata metadata;
  metadata.stop_at_eos = false;
  const auto ignored =
      scheduler->Submit({2}, 4, sampling, {}, false, metadata).Wait();
  Expect(ignored.tokens.size() == 4 &&
             ignored.finish_reason ==
                 gufo::server::TextGenerationBackend::FinishReason::kLength,
         "ignore-EOS request reaches its exact token limit");
  const auto restored = scheduler->Submit({3}, 4, sampling).Wait();
  Expect(restored.tokens.empty() &&
             restored.finish_reason ==
                 gufo::server::TextGenerationBackend::FinishReason::kStop,
         "reused runner restores EOS stopping for the next request");
}

void TestEmptyTokenIsPublished() {
  auto control = std::make_shared<FakeControl>();
  control->output_pieces = {""};
  auto scheduler = MakeScheduler(control, 1);
  std::size_t events = 0;
  auto request = scheduler->Submit({1}, 1, 0.0F, {}, true);
  const auto result = request.Wait([&](std::string_view piece) {
    Expect(piece.empty(), "empty token has no text bytes");
    ++events;
    return true;
  });
  Expect(result.tokens.size() == 1 && events == 1,
         "one empty decoded token still has one streaming event");
  Expect(result.max_buffered_output_bytes >= 1,
         "an empty streamed piece still charges the output budget");
}

void TestDebugTimelineOrderCountsAndOffPath() {
  auto control = std::make_shared<FakeControl>();
  control->prefill_capacity = 2;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.cache_prompt = false;
  metadata.debug_timeline = true;
  const auto result = scheduler->Submit({1, 2, 3, 4, 5, 6}, 4, 0.0F, {}, false,
                                        metadata).Wait();
  Expect(bool(result.debug_timeline), "opt-in creates a timeline");
  const auto events = gufo::server::SerializeDebugTimeline(result);
  Expect(events.items().front().member_str("event") == "admission",
         "timeline begins at admission");
  Expect(events.items().back().member_str("event") == "finish",
         "timeline ends at finish");
  std::size_t passes = 0, cycles = 0, first = 0, lookup = 0, queue = 0;
  double previous = -1;
  double cycle_ms = 0, prefill_ms = 0, restore_ms = 0, queue_ms = 0;
  for (const auto& event : events.items()) {
    Expect(event.member_double("t_ms") >= previous,
           "timeline is chronological");
    previous = event.member_double("t_ms");
    const auto name = event.member_str("event");
    if (name == "prefill") {
      prefill_ms += event.member_double("ms");
      Expect(event.member_size("index") == passes &&
                 event.member_size("start_depth") == passes * 2 &&
                 event.member_size("tokens") == 2,
             "scripted prefill pass index, tokens and depth");
      ++passes;
    } else if (name == "decode_cycle") {
      cycle_ms += event.member_double("ms");
      Expect(event.member_size("verify_width") == 1 &&
                 event.member_size("drafts") == 0,
             "AR cycles report width one");
      ++cycles;
    } else if (name == "first_token") {
      ++first;
    } else if (name == "cache_lookup") {
      restore_ms += event.member_double("restore_ms");
      ++lookup;
      Expect(event.member_str("source") == "none",
             "cold lookup reports its source");
    } else if (name == "queue") {
      queue_ms += event.member_double("wait_ms");
      ++queue;
    }
  }
  Expect(passes == result.prefill_chunks && passes == 3 && cycles == 4 &&
             first == 1 && lookup == 1 && queue == 1,
         "timeline captures every scripted work unit once");
  const auto* summary = events.items().back().find("summary");
  Expect(summary && summary->find("prefill_passes")->size() == passes,
         "summary lists all actual prefill passes");
  Expect(summary->member_double("cycle_ms") == cycle_ms &&
             summary->member_double("prefill_ms") == prefill_ms &&
             summary->member_double("restore_ms") == restore_ms &&
             summary->member_double("queue_ms") == queue_ms,
         "summary totals reconcile against independently accumulated event measurements");
  const double accounted = summary->member_double("cycle_ms") +
      summary->member_double("prefill_ms") + summary->member_double("restore_ms") +
      summary->member_double("queue_ms") + summary->member_double("residual_ms");
  Expect(std::abs(accounted - summary->member_double("wall_ms")) < 1e-8,
         "raw residual preserves elapsed accounting without hiding overlaps");
  Expect(summary->member_double("unattributed_ms") ==
             std::max(0.0, summary->member_double("residual_ms")) &&
             summary->member_double("overlap_ms") ==
             std::max(0.0, -summary->member_double("residual_ms")),
         "gap and overlap retain the sign of independent phase accounting");
  Expect(!result.debug_timeline->truncated(), "short request is untruncated");
  metadata.debug_timeline = false;
  const auto off = scheduler->Submit({1, 2, 3, 4, 5, 6}, 4, 0.0F, {}, false,
                                     metadata).Wait();
  Expect(!off.debug_timeline && off.tokens == result.tokens &&
             off.text == result.text,
         "timeline OFF has no event storage and identical outputs");
  auto response = gufo::json::Value::object();
  response["text"] = off.text;
  const auto off_bytes = response.dump();
  gufo::server::AddDebugTimeline(response, off);
  // Base fake frontier is prompt label * 100, then increments per token.
  Expect(off_bytes == "{\"text\":\"100101102103\"}" && response.dump() == off_bytes,
         "OFF response keeps base fake-output bytes and emits no attribution fields");
  Expect(!response.find("gufo"), "OFF response contains no timeline or summary");
}

void TestDebugTimelineMixedSourcesAndPolicy() {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = true;
  control->timeline_mixed_drafts = true;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.cache_prompt = false;
  const auto initial_off = scheduler->Submit({1, 2, 3}, 3, 0.0F, {}, false,
                                             metadata).Wait();
  Expect(!initial_off.debug_timeline && control->debug_settings_calls == 0,
         "initial OFF request does not build settings");
  metadata.debug_timeline = true;
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  auto behaviour = std::make_shared<gufo::sampling::BehaviourConfig>();
  behaviour->think_eog_guard = true;
  behaviour->spec_eog_stop = true;
  behaviour->eog_min_prob = 0.25;
  config.behaviour = behaviour;
  const auto result = scheduler->Submit({1, 2, 3}, 6, config, {}, false,
                                        metadata).Wait();
  const auto events = result.debug_timeline->Serialize();
  const auto* bundle = events.items().front().find("bundle");
  Expect(bundle && bundle->find("mtp_policy")->member_str("name") == "survival" &&
             bundle->find("mtp_policy")->member_size("cap") == 4 &&
             bundle->find("behaviour")->find("think_eog_guard")->as_bool() &&
             bundle->find("behaviour")->member_double("eog_min_prob") == 0.25,
         "header projects configured policy and actual request switches");
  std::size_t cycles = 0;
  for (const auto& event : events.items()) {
    if (event.member_str("event") != "decode_cycle")
      continue;
    const auto* sources = event.find("draft_sources");
    Expect(sources && sources->member_size("prompt_lookup") == (cycles == 0 ? 2 : 0) &&
               sources->member_size("mtp_head") == (cycles == 0 ? 2 : 3) &&
               event.member_str("draft_source") == (cycles == 0 ? "mixed" : "mtp_head") &&
               event.member_size("drafts") == (cycles == 0 ? 4 : 3) &&
               event.member_size("verify_width") == (cycles == 0 ? 5 : 4) &&
               event.member_size("accepted_length") == 2 &&
               event.find("partial_accept")->as_bool() &&
               event.member_size("policy_chosen_length_sum") == 4 &&
               !event.contains("policy") &&
               event.member_str("policy_effective") == "survival" &&
               event.member_size("accepted_length") + 1 == 3,
           "scripted mixed and survival-shortened head cycles retain exact budget/source/verify counts");
    Expect(event.member_size("n_cycles") == 1 &&
               event.member_size("mtp_offered") == (cycles == 0 ? 2 : 4) &&
               event.member_size("lookup_offered") == (cycles == 0 ? 2 : 0) &&
               event.member_size("proposed") == 4 &&
               event.member_size("cut") == (cycles == 0 ? 0 : 1) &&
               event.find("survival_any")->as_bool(),
           "offers include the cut proposal and reconcile with verified drafts");
    Expect(!event.find("latin_filtered") && !event.find("latin_rejected") &&
               !event.find("proposed_length") &&
               !event.find("filter_changed") && !event.find("filter_rejected") &&
               !event.find("rollback"),
           "obsolete name and unavailable counterfactual filter/rollback fields are omitted");
    ++cycles;
  }
  Expect(cycles == 2, "scripted multi-token request has two cycles");
  Expect(bundle->find("mtp_policy")->find("prompt_lookup")->as_bool() &&
             bundle->find("mtp_policy")->member_str("draft_vocabulary") == "latin_text" &&
             bundle->find("mtp_policy")->member_str("latin_vocab") == "latin",
         "default stack settings are resolved in the bundle");
  config.temperature = 0.0F;
  const auto deterministic = scheduler->Submit({1, 2, 3}, 3, config, {}, false,
                                                metadata).Wait();
  const auto deterministic_events = deterministic.debug_timeline->Serialize();
  for (const auto& event : deterministic_events.items()) {
    if (event.member_str("event") == "decode_cycle")
      Expect(event.member_str("policy_effective") == "length" &&
                 event.member_size("policy_chosen_length_sum") == 4,
             "scripted deterministic policy decision reports length and initial budget");
  }
  metadata.debug_timeline = false;
  const auto off = scheduler->Submit({1, 2, 3}, 3, config, {}, false, metadata).Wait();
  auto response = gufo::json::Value::object();
  response["text"] = off.text;
  const auto off_bytes = response.dump();
  gufo::server::AddDebugTimeline(response, off);
  // Base fake frontier is prompt label * 100, then increments per token.
  Expect(off_bytes == "{\"text\":\"100101102\"}" && response.dump() == off_bytes,
         "OFF response keeps base fake-output bytes and emits no attribution fields");
  Expect(!response.contains("gufo") && control->debug_settings_calls == 1,
         "settings are built once and all diagnostic fields are absent OFF");
}

void TestDebugTimelineStepCycleSemantics() {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = true;
  control->timeline_mixed_drafts = true;
  control->timeline_multi_cycle_step = true;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.cache_prompt = false;
  metadata.debug_timeline = true;
  const auto result = scheduler->Submit({1, 2, 3}, 6, 1.0F, {}, false,
                                        metadata).Wait();
  const auto events = result.debug_timeline->Serialize();
  std::size_t steps = 0;
  for (const auto& event : events.items()) {
    if (event.member_str("event") != "decode_cycle")
      continue;
    Expect(event.contains("n_cycles") &&
               event.member_size("n_cycles") == (steps == 0 ? 1 : 3) &&
               event.member_size("policy_chosen_length_sum") == (steps == 0 ? 4 : 12) &&
               !event.contains("policy_chosen_length") &&
               event.find("survival_any") && event.find("survival_any")->as_bool() &&
               event.contains("policy_effective") == (steps == 0),
           "one- and three-cycle steps cannot silently label a budget sum as one cycle");
    Expect(event.member_size("mtp_offered") == (steps == 0 ? 2 : 10) &&
               event.member_size("lookup_offered") == (steps == 0 ? 2 : 0) &&
               event.member_size("proposed") == (steps == 0 ? 4 : 10) &&
               event.member_size("cut") == (steps == 0 ? 0 : 1) &&
               event.member_size("drafts") == (steps == 0 ? 4 : 9),
           "scripted step totals preserve offers and cuts across multiple cycles");
    ++steps;
  }
  Expect(steps == 2 && result.draft_rounds == 4,
         "two work units report four speculative cycles");
}

void TestDebugTimelineRamAndLiveRestoreKinds() {
  auto control = std::make_shared<FakeControl>();
  control->snapshot_callback = [] {};
  control->in_pass_checkpoint = true;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.debug_timeline = true;
  metadata.cache_prefix_tokens = 2;
  const auto first = scheduler->Submit({1, 2, 3, 4, 5, 6}, 1, 0.0F, {}, false,
                                       metadata).Wait();
  // Diverge immediately after the in-pass stable checkpoint: the live frontier
  // does not match, so Acquire must restore the retained RAM snapshot.
  const auto restored = scheduler->Submit({1, 2, 9, 10}, 1, 0.0F, {}, false,
                                          metadata).Wait();
  Expect(restored.cache_hit && restored.cached_prompt_tokens == 2 &&
             restored.cache_restore_bytes > 0 && !restored.cache_disk_hit,
         "divergent suffix forces restoration of the in-pass RAM checkpoint");
  const auto events = restored.debug_timeline->Serialize();
  const auto* summary = events.items().back().find("summary");
  const auto& restore = summary->find("restores")->items().front();
  Expect(restore.member_str("kind") == "RAM" &&
             restore.member_size("bytes") == restored.cache_restore_bytes &&
             restore.member_double("ms") == restored.cache_restore_ms &&
             !restore.find("capture_origin"),
         "RAM restore uses measured bytes/ms without fabricating lost capture provenance");
  // Extend the current live token prefix, including its generated token.
  metadata.cache_prefix_tokens = 0;
  const auto live = scheduler->Submit({1, 2, 9, 10, 100, 11}, 1, 0.0F, {}, false,
                                      metadata).Wait();
  const auto live_events = live.debug_timeline->Serialize();
  const auto* live_summary = live_events.items().back().find("summary");
  Expect(live.cache_hit && live.cache_restore_bytes == 0 &&
             live_summary->find("restores")->items().front().member_str("kind") == "live_slot",
         "live slot reuse is distinguished from snapshot restoration");
}

void TestDebugTimelineBatchOmissionsAndAccounting() {
  const auto start = TextGenerationScheduler::Clock::now();
  gufo::server::DebugTimeline timeline(start);
  timeline.Add("admission", start);
  auto bundle = gufo::json::Value::object();
  bundle["mtp_policy"] = gufo::json::Value::object();
  bundle["mtp_policy"]["name"] = "survival";
  bundle["mtp_policy"]["cap"] = 4;
  timeline.Header(std::move(bundle));
  auto queue = gufo::json::Value::object();
  queue["wait_ms"] = 4.0;
  queue["queue_ms"] = 20.0;
  timeline.Add("queue", start, queue);
  auto lookup = gufo::json::Value::object();
  lookup["source"] = "RAM";
  lookup["restore_ms"] = 3.0;
  lookup["bytes"] = 100;
  timeline.Add("cache_lookup", start, lookup);
  auto pass = gufo::json::Value::object();
  pass["tokens"] = 2;
  pass["ms"] = 2.0;
  pass["start_depth"] = 3;
  pass["end_reason"] = "cap";
  timeline.Add("prefill", start, pass);
  timeline.Cycle(start, 5, 5, 4, 2, std::nullopt);
  timeline.Cycle(start + std::chrono::milliseconds(5), 0, 1, 0, 0);
  timeline.Add("finish", start + std::chrono::milliseconds(20));
  const auto events = timeline.Serialize();
  const auto& cycle = events.items()[4];
  Expect(!cycle.find("draft_sources") && !cycle.find("draft_source") &&
             !cycle.contains("policy_effective") && !cycle.contains("policy") &&
             !cycle.contains("policy_chosen_length_sum") &&
             events.items()[0].find("bundle")->find("mtp_policy")->member_size("cap") == 4,
         "batch path omits unavailable source and effective-policy names while retaining the configured cap");
  Expect(events.items()[5].member_str("policy_effective") == "none",
         "non-MTP cycle policy is none");
  const auto* summary = events.items().back().find("summary");
  Expect(summary->member_double("residual_ms") == 6 &&
             summary->member_double("overlap_ms") == 0 &&
             summary->member_double("cycle_ms") == 5 &&
             summary->member_double("prefill_ms") == 2 &&
             summary->member_double("restore_ms") == 3 &&
             summary->member_double("queue_ms") == 4 &&
             summary->member_double("unattributed_ms") == 6 &&
             summary->member_double("wall_ms") == 20,
         "accounting uses queue wait rather than admission duration, excluding duplicate restore/render time");
}

void TestDebugTimelineOverlappingPhases() {
  const auto start = TextGenerationScheduler::Clock::time_point{};
  gufo::server::DebugTimeline timeline(start);
  auto queue = gufo::json::Value::object();
  queue["wait_ms"] = 4.0;
  timeline.Add("queue", start, queue);
  auto restore = gufo::json::Value::object();
  restore["source"] = "RAM";
  restore["restore_ms"] = 3.0;
  restore["bytes"] = 128;
  timeline.Add("cache_lookup", start + std::chrono::milliseconds(4), restore);
  auto pass = gufo::json::Value::object();
  pass["ms"] = 8.0;
  timeline.Add("prefill", start + std::chrono::milliseconds(4), pass);
  timeline.Cycle(start + std::chrono::milliseconds(7), 5, 5, 4, 2,
                 0, 4, true);
  timeline.Add("finish", start + std::chrono::milliseconds(12));
  const auto events = timeline.Serialize();
  const auto& summary = *events.items().back().find("summary");
  Expect(summary.member_double("cycle_ms") == 5 &&
             summary.member_double("prefill_ms") == 8 &&
             summary.member_double("restore_ms") == 3 &&
             summary.member_double("queue_ms") == 4 &&
             summary.member_double("wall_ms") == 12 &&
             summary.member_double("residual_ms") == -8 &&
             summary.member_double("overlap_ms") == 8 &&
             summary.member_double("unattributed_ms") == 0,
         "overlapping script retains raw negative residual independently of clamping");
}

void TestDebugTimelineDiskRestoreKind() {
  std::string pattern = (std::filesystem::temp_directory_path() /
                         "gufo-timeline-disk-XXXXXX").string();
  const char* created = ::mkdtemp(pattern.data());
  Expect(created != nullptr, "fake disk timeline test creates a private directory");
  const std::filesystem::path directory(created);
  const gufo::server::TextRunnerDiskCacheOptions disk{
      .directory = directory, .capacity_bytes = 4096, .staging_capacity_bytes = 4096};
  auto control = std::make_shared<FakeControl>();
  control->snapshot_callback = [] {};
  control->persistent_snapshots = true;
  TextRequestMetadata metadata;
  metadata.debug_timeline = true;
  {
    auto pool = std::make_shared<TextRunnerPool>(
        std::make_shared<FakeRunner>(control), 1, disk);
    TextGenerationScheduler writer(pool);
    const auto written = writer.Submit({1, 2, 3}, 1, 0.0F, {}, false, metadata).Wait();
    Expect(written.cache_disk_queued_bytes > 0, "first turn queues a real disk checkpoint");
  }
  {
    auto pool = std::make_shared<TextRunnerPool>(
        std::make_shared<FakeRunner>(control), 1, disk);
    TextGenerationScheduler reader(pool);
    const auto restored = reader.Submit({1, 2, 3, 9}, 1, 0.0F, {}, false,
                                        metadata).Wait();
    const auto events = restored.debug_timeline->Serialize();
    const auto& restore = events.items().back().find("summary")
        ->find("restores")->items().front();
    Expect(restored.cache_disk_hit && restore.member_str("kind") == "disk" &&
               restore.member_size("bytes") == restored.cache_restore_bytes &&
               restore.member_double("ms") == restored.cache_restore_ms,
           "restarted fake runner reports a measured disk restore in the timeline");
  }
  std::error_code error;
  std::filesystem::remove_all(directory, error);
}

void TestDebugTimelineSnapshotsAndCycleCap() {
  auto control = std::make_shared<FakeControl>();
  control->snapshot_callback = [] {};
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.debug_timeline = true;
  const auto result = scheduler->Submit({1, 2, 3}, 2, 0.0F, {}, false,
                                        metadata).Wait();
  bool snapshot = false;
  const auto events = gufo::server::SerializeDebugTimeline(result);
  for (const auto& event : events.items()) {
    if (event.member_str("event") == "snapshot") {
      snapshot = true;
      Expect(event.member_size("tokens") == 3 &&
                 event.member_size("bytes") > 0 && event.contains("ms") &&
                 event.contains("in_pass") && event.member_str("kind") == "continuation",
             "snapshot timeline has the measured payload and purpose");
    }
  }
  Expect(snapshot, "prompt snapshot capture is included");
  const auto& captures = events.items().back().find("summary")->find("captures")->items();
  Expect(!captures.empty(), "separate capture appears in summary");
  for (const auto& capture : captures)
    Expect(capture.find("in_pass") && !capture.find("in_pass")->as_bool() &&
               capture.member_str("capture_kind") == "separate" &&
               capture.member_size("tokens") == 3 &&
               capture.member_size("bytes") == sizeof(FakeRunner::SnapshotState),
           "summary preserves each separate capture's measured false flag");
  const auto start = TextGenerationScheduler::Clock::now();
  gufo::server::DebugTimeline capped(start);
  for (std::size_t i = 0; i <= gufo::server::DebugTimeline::kMaxDecodeCycles; ++i)
    capped.Cycle(start + std::chrono::milliseconds(i), 1, 3, 2, 1);
  capped.Add("finish", start + std::chrono::seconds(5));
  gufo::server::DebugTimeline failed(start);
  failed.Failed(std::make_exception_ptr(std::runtime_error("diagnostic failed")));
  bool surfaced = false;
  try {
    (void)failed.Serialize();
  } catch (const std::runtime_error& error) {
    surfaced = std::string(error.what()) == "diagnostic failed";
  }
  Expect(surfaced, "diagnostic allocation/log failure cannot return a partial success");
  Expect(capped.truncated() && capped.Serialize().items().size() ==
             gufo::server::DebugTimeline::kMaxDecodeCycles + 1,
         "decode cap truncates cycles while retaining finish");
  const auto capped_events = capped.Serialize();
  const auto* totals = capped_events.items().back().find("summary");
  Expect(totals && totals->member_size("cycle_count") == 4097 &&
             totals->member_double("cycle_ms") == 4097 &&
             totals->member_double("unattributed_ms") == 903,
         "summary includes cycle time and count beyond the retained cap");
}

void TestDebugTimelineRequestTruncation() {
  auto control = std::make_shared<FakeControl>();
  control->max_context = 5000;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.debug_timeline = true;
  std::ostringstream lines;
  auto* old_buffer = std::clog.rdbuf(lines.rdbuf());
  const auto result = scheduler->Submit({1, 2},
      gufo::server::DebugTimeline::kMaxDecodeCycles + 1, 0.0F, {}, false,
      metadata).Wait();
  std::clog.rdbuf(old_buffer);
  std::size_t cycles = 0;
  const auto events = gufo::server::SerializeDebugTimeline(result);
  for (const auto& event : events.items())
    cycles += event.member_str("event") == "decode_cycle";
  Expect(result.debug_timeline->truncated() &&
             cycles == gufo::server::DebugTimeline::kMaxDecodeCycles &&
             events.items().back().member_str("event") == "finish" &&
             events.items().back().find("truncated")->as_bool(),
         "fake request caps cycle events and reports truncation at finish");
}

void TestDebugTimelineInPassCheckpoint() {
  auto control = std::make_shared<FakeControl>();
  control->snapshot_callback = [] {};
  control->in_pass_checkpoint = true;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.debug_timeline = true;
  metadata.cache_prefix_tokens = 2;
  const auto result = scheduler->Submit({1, 2, 3, 4, 5, 6}, 1, 0.0F, {}, false,
                                        metadata).Wait();
  bool in_pass = false, pass_boundary = false;
  const auto events = gufo::server::SerializeDebugTimeline(result);
  for (const auto& event : events.items()) {
    if (event.member_str("event") == "snapshot" &&
        event.find("in_pass") && event.find("in_pass")->as_bool()) {
      in_pass = true;
      Expect(event.member_size("tokens") == 2 && event.member_size("bytes") > 0 &&
                 event.member_double("ms") >= 0,
             "in-pass snapshot exposes its measured stable boundary");
    }
    if (event.member_str("event") == "prefill")
      pass_boundary = event.member_str("end_reason") == "prompt_end" &&
                      event.find("in_pass_checkpoint") &&
                      event.find("in_pass_checkpoint")->as_bool();
  }
  const auto* summary = events.items().back().find("summary");
  const auto* captures = summary ? summary->find("captures") : nullptr;
  bool summary_in_pass = false;
  Expect(captures && !captures->items().empty(), "summary retains capture records");
  for (const auto& capture : captures->items()) {
    if (capture.find("in_pass")->as_bool()) {
      summary_in_pass = true;
      Expect(capture.member_str("capture_kind") == "in_pass_checkpoint" &&
                 capture.member_size("tokens") == 2 &&
                 capture.member_size("bytes") == sizeof(FakeRunner::SnapshotState),
             "summary exposes the actual in-pass capture flag and kind");
    } else {
      Expect(capture.member_str("capture_kind") == "separate",
             "separate captures preserve false in-pass flag");
    }
  }
  Expect(summary_in_pass && in_pass && pass_boundary && result.prefill_chunks == 1,
         "in-pass capture appears without splitting the prefill pass");
}

void TestPromptScoreDedicatedPathAndCpuReference() {
  auto control = std::make_shared<FakeControl>();
  control->score_supported = true;
  auto scheduler = MakeScheduler(control, 1);
  gufo::sampling::SamplingConfig config;
  config.logprobs = 2;
  const auto states = control->states_created.load();
  const auto scored = scheduler->ScorePrompt({0, 1, 2}, config).Wait();
  Expect(scored.prompt_tokens == 3 && scored.completion_tokens == 0 &&
             scored.tokens == std::vector<TextRunnerToken>({0, 1, 2}) &&
             scored.text.empty() &&
             scored.prompt_logprobs.size() == 3 && !scored.prompt_logprobs[0],
         "teacher-forced scoring generates no tokens and first score is null");
  const double normalizer = std::log(1.0 + std::exp(1.0) + std::exp(-1.0));
  Expect(std::abs(scored.prompt_logprobs[1]->logprob - (1.0 - normalizer)) < 1e-12 &&
             std::abs(scored.prompt_logprobs[2]->logprob - (-1.0 - normalizer)) < 1e-12,
         "teacher-forced prompt scores equal independent CPU reference");
  Expect(control->events.empty() && control->states_created.load() == states,
         "scoring never acquires a continuation state or performs generation");
  const auto warm = scheduler->Submit({1, 2}, 1, 0.0F).Wait();
  (void)scheduler->ScorePrompt({0, 1}, config).Wait();
  const auto hit = scheduler->Submit({1, 2, 100}, 1, 0.0F).Wait();
  Expect(!warm.cache_hit && hit.cache_hit && hit.cached_prompt_tokens == 3,
         "scoring preserves the retained continuation");
  control->score_supported = false;
  bool unsupported = false;
  try {
    (void)scheduler->ScorePrompt({0, 1}, config).Wait();
  } catch (const std::invalid_argument& error) {
    unsupported = std::string(error.what()) ==
                  "prompt logprobs are not supported for this model";
  }
  Expect(unsupported, "unsupported scorer produces the explicit error");
}

void TestScoredGenerationPreservesFakeOutputAndAlignment() {
  for (const bool multi_token : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = multi_token;
    auto scheduler = MakeScheduler(control, 1);
    TextRequestMetadata metadata;
    metadata.cache_prompt = false;
    gufo::sampling::SamplingConfig config;
    config.seed = 123;
    const auto plain = scheduler->Submit({1, 2}, 5, config, {}, false, metadata).Wait();
    config.logprobs = 2;
    const auto scored = scheduler->Submit({1, 2}, 5, config, {}, false, metadata).Wait();
    Expect(scored.tokens == plain.tokens && scored.text == plain.text &&
               scored.logprobs.size() == scored.tokens.size(),
           "fixed-seed fake output and per-token score alignment survive opt-in");
    for (std::size_t i = 0; i < scored.tokens.size(); ++i) {
      const auto token = scored.tokens[i];
      const double reference = 1.0 - std::log(double(token) + std::exp(1.0));
      Expect(scored.logprobs[i].token == token &&
                 std::abs(scored.logprobs[i].logprob - reference) < 1e-12,
             "emitted fake token scores match independent raw-row reference");
    }
  }
}

void TestPromptScoreSerializationAndCancellation() {
  auto control = std::make_shared<FakeControl>();
  control->score_supported = true;
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 2);
  auto generated = scheduler->Submit({1, 2}, 4, 0.0F);
  control->WaitForPrefill(1);
  gufo::sampling::SamplingConfig config;
  config.logprobs = 0;
  auto scored = scheduler->ScorePrompt({0, 1}, config);
  Expect(
      control->score_calls.load() == 0 &&
          scored.phase() == gufo::server::TextRequestPhase::kScoreQueued,
      "scorer is explicitly queued behind generation on the inference owner");
  control->score_callback = [&] {
    Expect(generated.phase() == TextRequestPhase::kTerminal,
           "scorer cannot pause a resident decoder: it starts only after "
           "terminal publication");
    Expect(scored.phase() == TextRequestPhase::kScoring,
           "owner exposes the explicit scoring phase");
  };
  control->ReleasePrefill();
  (void)generated.Wait();
  Expect(scored.Wait().prompt_logprobs.size() == 2 && control->score_calls.load() == 1,
         "queued scorer executes after generation");
  const auto calls = control->score_calls.load();
  auto cancelled = scheduler->ScorePrompt({0, 1}, config, [] { return true; });
  Expect(cancelled.Wait().cancelled && control->score_calls.load() == calls,
         "cancelled scoring never invokes its dedicated runner");
}

void TestLiveScoredQueueAndMissingRecords() {
  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);
    gufo::sampling::SamplingConfig config;
    config.logprobs = 0;
    auto live = scheduler->Submit({1, 2}, 3, config, {}, true);
    control->WaitForAdvance(1);
    std::promise<void> first;
    auto first_seen = first.get_future();
    std::size_t count = 0;
    auto consumed = std::async(std::launch::async, [&] {
      return live.Wait(
          {}, {}, {},
          [&](std::string_view, const gufo::sampling::TokenLogprob&) {
            if (++count == 1)
              first.set_value();
            return true;
          });
    });
    Expect(first_seen.wait_for(kTestTimeout) == std::future_status::ready &&
               live.phase() != TextRequestPhase::kTerminal,
           "first scored token reaches consumer while next runner advance is "
           "blocked");
    control->ReleaseAdvance();
    Expect(consumed.get().tokens.size() == 3 && count == 3,
           "live scored stream completes with exactly three callbacks");
  }

  for (const bool multi_token : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = multi_token;
    auto scheduler = MakeScheduler(control, 1);
    gufo::sampling::SamplingConfig config;
    config.seed = 123;
    config.logprobs = 2;
    TextRequestMetadata metadata;
    metadata.cache_prompt = false;
    auto request = scheduler->Submit({1, 2}, 5, config, {}, true, metadata);
    std::vector<gufo::sampling::TokenId> published;
    const auto result = request.Wait(
        {}, {}, {},
        [&](std::string_view piece, const gufo::sampling::TokenLogprob& score) {
          Expect(!piece.empty() && score.top_logprobs.size() == 2,
                 "real queue publishes one scored token at a time");
          published.push_back(score.token);
          return true;
        });
    Expect(published == result.tokens && published.size() == 5,
           "real streaming queue has exactly one scored callback per generated "
           "token");
    auto diagnostic = result;
    diagnostic.debug_timeline = std::make_shared<gufo::server::DebugTimeline>(
        gufo::server::TextGenerationScheduler::Clock::now());
    auto value = gufo::json::Value::object();
    gufo::server::AddDebugTimeline(value, diagnostic);
    const auto& ids = value.find("gufo")->find("token_ids")->items();
    Expect(ids.size() == result.tokens.size(),
           "timeline transports actual published ID count");
    for (std::size_t i = 0; i < ids.size(); ++i)
      Expect(ids[i].as_size() == result.tokens[i],
             "timeline IDs are not inferred from output text");
    control->omit_score = true;
    auto broken = scheduler->Submit({1, 2}, 3, config, {}, true, metadata);
    bool failed = false;
    try {
      (void)broken.Wait();
    } catch (const std::logic_error& error) {
      failed = std::string(error.what()) ==
               "runner omitted a requested token logprob";
    }
    Expect(failed,
           "short/missing score vector fails explicitly on AR and MTP paths");
  }
}

int main(int argc, char** argv) {
  TestPrefillDiagnosticsHost();
  if (argc == 2 && std::string_view(argv[1]) == "--debug-hooks") {
    TestLiveScoredQueueAndMissingRecords();
    TestDebugTimelineOrderCountsAndOffPath();
    TestDebugTimelineMixedSourcesAndPolicy();
    TestDebugTimelineStepCycleSemantics();
    TestDebugTimelineRamAndLiveRestoreKinds();
    TestDebugTimelineDiskRestoreKind();
    TestDebugTimelineBatchOmissionsAndAccounting();
    TestDebugTimelineOverlappingPhases();
    TestDebugTimelineSnapshotsAndCycleCap();
    TestDebugTimelineInPassCheckpoint();
    TestDebugTimelineRequestTruncation();
    TestScoredGenerationPreservesFakeOutputAndAlignment();
    TestPromptScoreDedicatedPathAndCpuReference();
    TestPromptScoreSerializationAndCancellation();
    return 0;
  }

  TestLiveScoredQueueAndMissingRecords();
  TestDebugTimelineOrderCountsAndOffPath();
  TestDebugTimelineMixedSourcesAndPolicy();
  TestDebugTimelineStepCycleSemantics();
  TestDebugTimelineRamAndLiveRestoreKinds();
  TestDebugTimelineDiskRestoreKind();
  TestDebugTimelineBatchOmissionsAndAccounting();
  TestDebugTimelineOverlappingPhases();
  TestDebugTimelineSnapshotsAndCycleCap();
  TestDebugTimelineRequestTruncation();
  TestDebugTimelineInPassCheckpoint();
  TestPromptScoreDedicatedPathAndCpuReference();
  TestPromptScoreSerializationAndCancellation();
  TestScoredGenerationPreservesFakeOutputAndAlignment();
  // Log assertions in this binary match the plain "[LEVEL] [component]" text
  // captured from a redirected sink; a TTY stderr would tint the level tag.
  ::setenv("NO_COLOR", "1", 1);
  TestProgressLoggingIsOptInAndBounded();
  TestAdmissionLoggingIsDebugTierOnly();
  TestPromptProgressPrecedesStreamedTokens();
  TestStreamStartPrecedesPrefill();
  TestPromptProgressCancellation();
  TestIgnoreEosIsRequestScoped();
  TestEmptyTokenIsPublished();
  TestStopSequenceChunkBoundaries();
  TestStopSequencesPreserveExecutedState();
  TestStopPrefixFlushAndBatchIsolation();
  TestCancellationWithBufferedStopPrefix();
  for (const bool speculative : {false, true}) {
    for (const bool preview : {false, true}) {
      auto control = std::make_shared<FakeControl>();
      control->max_context = 256;
      control->multi_token_decode = speculative;
      control->preview_first_token = preview;
      auto scheduler = MakeScheduler(control, 2);
      const auto run = [&](std::size_t limit, std::size_t expected) {
        auto first = scheduler->Submit({1, 10}, limit, 0.0F);
        auto second = scheduler->Submit({2, 20, 30}, limit, 0.0F);
        const auto a = first.Wait();
        const auto b = second.Wait();
        Expect(a.tokens.size() == expected,
               "default/explicit limit respects remaining context");
        Expect(b.tokens.size() == (limit == 0 || limit > 253 ? 253 : limit),
               "concurrent requests have independent context budgets");
        Expect(a.finish_reason ==
                   gufo::server::TextGenerationBackend::FinishReason::kLength,
               "exhausted context is reported as length");
      };
      run(0, 254);
      run(1000, 254);
      for (std::size_t limit = 1; limit <= 8; ++limit)
        run(limit, limit);
      control->stop_after = 150;
      const auto stopped = scheduler->Submit({3}, 0, 0.0F).Wait();
      Expect(stopped.tokens.size() == 150 &&
                 stopped.finish_reason ==
                     gufo::server::TextGenerationBackend::FinishReason::kStop,
             "unlimited default passes 128 tokens and still respects EOS");
      for (const auto size : {256, 257}) {
        bool rejected = false;
        try {
          (void)scheduler->Submit(std::vector<TextRunnerToken>(size, 1), 0,
                                  0.0F);
        } catch (const std::length_error&) {
          rejected = true;
        }
        Expect(rejected, "full context is rejected before generation");
      }
    }
  }
  TestCapturedDecoderResumesBeforeMorePrefill();
  TestCapturesAtCapacityAllowQueuedProgress();
  TestShutdownCancelsRunnerAcquisition();
  TestSnapshotDoesNotBlockOtherRequests();
  TestFirstTokenPrecedesSnapshotAndPreservesBudget();
  TestPrefillDiagnosticSchedulerTotals();
  TestIdlePrefillUsesBulkWorkUnit();
  TestConcurrentSharedPrefixesPrefillOnce();
  TestLeaderCancelledBeforeFollowerIsAdmitted();
  TestParkedFollowerSurvivesLeaderCancellation();
  TestSharedPrefixWaitRequiresSnapshots();
  TestParkedFollowersReserveVisibleSessions();
  TestRetainedHistoryBeatsSharedPrefixWait();
  TestAwaitedCheckpointSurvivesCachePressure();
  TestRunnerCanSkipUnusedFinalAdvance();
  TestRunnerCanReuseExactIncrementalText();
  TestGenerationTraceRecordsPromptAndOutput();
  TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse();
  TestRequestTotalsDoNotNeedAConsumer();
  TestMultiTokenRunnerCanSwitchToBatchedExecution();
  TestModelOwnedBatchMetrics();
  TestBatchFailureIsolation();
  TestBatchDeviceLossFailsSuccessfulPeers();
  TestMultiResidentPrefillUsesBoundedWorkUnits();
  for (const bool multi_token : {false, true}) {
    for (const bool batched : {false, true}) {
      if (batched && !multi_token)
        continue;
      TestDecodeActivePrefillIsBounded(multi_token, batched);
      TestPrefillYieldsToEveryDueDecoder(multi_token, batched);
    }
  }
  TestNonIncrementalRunnerFallsBackSafely();
  TestPendingLimitsRejectBeforeStateAdmission();
  TestPendingClientsAreRoundRobinAndIndividuallyBounded();
  TestExpiredQueuedRequestNeverConsumesState();
  TestSlowConsumerOutputIsBoundedAndReclaimed();
  TestCompletedAbandonedStreamReleasesOutputBudget();
  TestCompletedStreamOutlivesScheduler();
  TestGeneratedOutputLimitAppliesWithoutStreaming();
  TestMidGenerationAdmissionAndIsolatedTrajectories();
  TestMidGenerationRequestJoinsNextDecodeBatch();
  TestFifoReplacementAdmissionWithOneSlot();
  TestQueuedAndPrefillCancellation();
  TestServerMetricsAreLive();
  TestMetricsDuringCacheAdmission();
  TestSessionStatesFollowAdmittedRequests();
  TestDecodeCancellationAndStateReclamation();
  TestFourResidentRequestsMakeProgress();
  TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement();
  TestDeviceLossIsStickyAndReported();
  TestIdleDeviceProbe();
  Expect(
      gufo::server::detail::RequestsProcessing().load() == 0 &&
          gufo::server::detail::RequestsDeferred().load() == 0,
      "all success, failure, cancellation and shutdown paths balance gauges");
  std::cout << "All text generation scheduler tests passed\n";
  return 0;
}
