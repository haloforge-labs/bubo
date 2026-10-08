#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_DIAG_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_DIAG_HPP_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace gufo::models::qwen38_flash_next {

inline bool PrefillDiagValueEnabled(const char* value) {
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool MoeBenchEnabled() {
  static const bool enabled =
      PrefillDiagValueEnabled(std::getenv("GUFO_MOE_BENCH"));
  return enabled;
}

inline double MoeGbps(double bytes, double us) {
  return us > 0 ? bytes / us / 1e3 : 0.0;
}

inline double MoeFlops(std::uint32_t tokens, std::uint32_t used,
                       std::uint32_t hidden, std::uint32_t ff) {
  return static_cast<double>(tokens) * used * 2.0 * 3.0 * hidden * ff;
}

inline double MoeTflops(double flops, double us) {
  return us > 0 ? flops / us / 1e6 : 0.0;
}

inline bool PrefillDiagEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_PREFILL_DIAG");
    return PrefillDiagValueEnabled(value) || MoeBenchEnabled();
  }();
  return enabled;
}

using DiagClock = std::chrono::steady_clock;
inline double DiagMicroseconds(DiagClock::duration elapsed) {
  return std::chrono::duration<double, std::micro>(elapsed).count();
}

struct TableReadCounters {
  double table_read_us{0};  // Sum of worker wall times inside pread loops.
  std::uint64_t table_reads{0};
  std::uint64_t table_bytes{
      0};  // Positive bytes returned, including alignment.
  void Add(const TableReadCounters& other) {
    table_read_us += other.table_read_us;
    table_reads += other.table_reads;
    table_bytes += other.table_bytes;
  }
};

struct PrefillDiagTotals : TableReadCounters {
  std::uint64_t passes{0}, tokens{0}, expert_bytes{0};
  double fetch_elapsed_us{0}, fetch_hidden_us{0}, fetch_exposed_us{0},
      fetch_wait_us{0};
  std::uint64_t layer_samples{0}, experts_distinct_sum{0};
  std::uint32_t experts_distinct_min{0}, experts_distinct_max{0};
  bool moe_bench{false};
  double moe_us{0}, moe_flops{0};
  std::uint64_t moe_expert_bytes{0};  // Includes tail-only layers too.
  // Route attribution (review F1): the timed region is the WMMA route for layers that qualify and the GatedExperts/Experts fallback otherwise;
  // kill-rule numbers are only meaningful when moe_fallback_layers == 0. tiles = sum of routed row tiles of the WMMA layers (re-read factor of the weights).
  double moe_wmma_us{0};
  std::uint64_t moe_wmma_layers{0}, moe_fallback_layers{0}, moe_gated_layers{0}, moe_wide_down_layers{0}, moe_tiles{0};
  std::uint64_t moe_lanerow_layers{0}, moe_lanerow_fallback_layers{0};
  void AddLayer(std::uint32_t distinct, std::uint64_t bytes_per_expert) {
    experts_distinct_min = layer_samples == 0
                               ? distinct
                               : std::min(experts_distinct_min, distinct);
    experts_distinct_max = std::max(experts_distinct_max, distinct);
    ++layer_samples;
    experts_distinct_sum += distinct;
    expert_bytes += distinct * bytes_per_expert;
  }
  double Mean() const {
    return layer_samples ? static_cast<double>(experts_distinct_sum) /
                               static_cast<double>(layer_samples)
                         : 0.0;
  }
  void Add(const PrefillDiagTotals& other) {
    TableReadCounters::Add(other);
    if (other.moe_bench) {
      moe_bench = true;
      moe_us += other.moe_us;
      moe_flops += other.moe_flops;
      moe_expert_bytes += other.moe_expert_bytes;
      moe_wmma_us += other.moe_wmma_us;
      moe_wmma_layers += other.moe_wmma_layers;
      moe_fallback_layers += other.moe_fallback_layers;
      moe_gated_layers += other.moe_gated_layers;
      moe_wide_down_layers += other.moe_wide_down_layers;
      moe_tiles += other.moe_tiles;
    }
    moe_lanerow_layers += other.moe_lanerow_layers;
    moe_lanerow_fallback_layers += other.moe_lanerow_fallback_layers;
    passes += other.passes;
    tokens += other.tokens;
    expert_bytes += other.expert_bytes;
    fetch_elapsed_us += other.fetch_elapsed_us;
    fetch_hidden_us += other.fetch_hidden_us;
    fetch_exposed_us += other.fetch_exposed_us;
    fetch_wait_us += other.fetch_wait_us;
    if (other.layer_samples) {
      experts_distinct_min =
          layer_samples == 0
              ? other.experts_distinct_min
              : std::min(experts_distinct_min, other.experts_distinct_min);
      experts_distinct_max =
          std::max(experts_distinct_max, other.experts_distinct_max);
      layer_samples += other.layer_samples;
      experts_distinct_sum += other.experts_distinct_sum;
    }
  }
};

// The false path neither constructs counters nor formats/emits output.
inline void AddPrefillDiag(std::optional<PrefillDiagTotals>& destination,
                           const std::optional<PrefillDiagTotals>& source) {
  if (source) {
    if (!destination)
      destination.emplace();
    destination->Add(*source);
  }
}

struct PrefillDiagLayer {
  std::uint32_t layer, routed_tokens, distinct;
  std::uint64_t bytes_per_expert;
};

struct PrefillDiagPass {
  std::uint64_t pass{0};
  std::uint32_t start{0};
  PrefillDiagTotals totals;
  std::vector<PrefillDiagLayer> layers;
  // pass_tokens: the pass width. Without MTP the last trunk layer routes only
  // the tail rows; its distinct-expert count is not comparable with the
  // full-width layers, so it stays in `layers` (with its routed_tokens) but is
  // left out of the pass totals (review finding 1).
  void RecordLayer(std::uint32_t layer, std::uint32_t routed_tokens,
                   std::span<const std::uint32_t> counts,
                   std::uint64_t bytes_per_expert, std::uint32_t pass_tokens) {
    const auto distinct = static_cast<std::uint32_t>(std::count_if(
        counts.begin(), counts.end(), [](auto n) { return n != 0; }));
    layers.push_back({layer, routed_tokens, distinct, bytes_per_expert});
    if (routed_tokens == pass_tokens)
      totals.AddLayer(distinct, bytes_per_expert);
  }
};

inline std::string PrefillDiagFields(const PrefillDiagTotals& t) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << "table_read_us=" << t.table_read_us << " table_reads=" << t.table_reads
      << " table_bytes=" << t.table_bytes
      << " fetch_elapsed_us=" << t.fetch_elapsed_us
      << " fetch_hidden_us=" << t.fetch_hidden_us
      << " fetch_exposed_us=" << t.fetch_exposed_us
      << " fetch_wait_us=" << t.fetch_wait_us
      << " experts_distinct_min=" << t.experts_distinct_min
      << " experts_distinct_mean=" << t.Mean()
      << " experts_distinct_max=" << t.experts_distinct_max
      << " expert_bytes=" << t.expert_bytes;
  if (t.moe_bench)
    out << " moe_us=" << t.moe_us << " moe_expert_bytes=" << t.moe_expert_bytes
        << " moe_flops=" << t.moe_flops
        << " moe_gbps=" << MoeGbps(t.moe_expert_bytes, t.moe_us)
        << " moe_tflops=" << MoeTflops(t.moe_flops, t.moe_us)
        << " moe_wmma_layers=" << t.moe_wmma_layers << " moe_fallback_layers=" << t.moe_fallback_layers << " moe_wmma_us=" << t.moe_wmma_us
        << " moe_gated_layers=" << t.moe_gated_layers << " moe_wide_down_layers=" << t.moe_wide_down_layers << " moe_tiles=" << t.moe_tiles;
  if (t.moe_lanerow_layers || t.moe_lanerow_fallback_layers)
    out << " moe_lanerow_layers=" << t.moe_lanerow_layers
        << " moe_lanerow_fallback_layers=" << t.moe_lanerow_fallback_layers;
  return out.str();
}

inline std::string PrefillDiagLine(const PrefillDiagPass& p) {
  std::ostringstream out;
  out << "pass=" << p.pass << " tokens=" << p.totals.tokens
      << " start=" << p.start << ' ' << PrefillDiagFields(p.totals)
      << " layers=";
  // Each tuple is layer:routed_tokens:distinct:bytes_per_expert.
  for (std::size_t i = 0; i < p.layers.size(); ++i) {
    const auto& l = p.layers[i];
    if (i)
      out << ',';
    out << l.layer << ':' << l.routed_tokens << ':' << l.distinct << ':'
        << l.bytes_per_expert;
  }
  return out.str();
}

template<typename Take, typename Emit>
std::optional<PrefillDiagTotals> CollectPrefillDiag(bool enabled, Take take,
                                                    Emit emit) {
  if (!enabled)
    return std::nullopt;
  std::optional<PrefillDiagTotals> totals;
  for (const auto& pass : take()) {
    emit(PrefillDiagLine(pass));
    if (!totals)
      totals.emplace();
    totals->Add(pass.totals);
  }
  return totals;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
