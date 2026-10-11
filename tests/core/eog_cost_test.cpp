#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "tests/core/eog_cost_api.hpp"

namespace {
using Clock = std::chrono::steady_clock;
volatile std::uint64_t sink = 0;
unsigned timing_ms = 10;
bool focus = false;
struct Row {
  std::vector<float> logits, values;
  std::vector<std::uint32_t> ids;
  Row(const EogCostConfig& cfg, unsigned rank, std::uint64_t seed,
      bool varied = false)
      : logits(cfg.vocab) {
    std::mt19937_64 rng(seed);
    for (auto& v : logits)
      v = -16.F - static_cast<float>(rng() % 10000) / 1000;
    const float slope =
        varied ? .01F + static_cast<float>(rng() % 1600) / 1000 : .24F;
    for (unsigned i = 0; i < 96; ++i)
      logits[i + 10] = 8.F - slope * i;
    for (unsigned i = 0; i < cfg.eog.size(); ++i)
      logits[cfg.eog[i]] =
          rank == 0
              ? (i ? -12.F
                   : (varied ? 8.1F + static_cast<float>(rng() % 16000) / 1000
                             : 13.F))
          : rank == 1 ? 8.F - slope * (5 + i)
                      : -20.F - i;
    if (varied) {
      // Ties, missing finite EOG, shuffled peak IDs, and nonfinite tail.
      if (seed % 11 == 0)
        logits[cfg.eog[0]] = -INFINITY;
      if (seed % 17 == 0)
        logits[cfg.eog[1]] = NAN;
      if (seed % 7 == 0)
        logits[14] = logits[13];
      if (seed % 13 == 0)
        std::swap(logits[10], logits[cfg.vocab / 2]);
      if (seed % 19 == 0) {
        // An EOG too small to be sampled can still move a top-p cutoff.
        // Exercise the proof's conservative roundoff/mass rejection.
        std::ranges::fill(logits, -40.F);
        logits[10] = 8.F;
        logits[11] = 8.F + static_cast<float>(std::log((1.0 - .95F) / .95F));
        logits[cfg.eog[0]] = -8.F;
        logits[cfg.eog[1]] = -INFINITY;
      }
    }
    ids.resize(cfg.vocab);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(
        ids.begin(), ids.begin() + 64, ids.end(), [&](auto a, auto b) {
          const float av = std::isfinite(logits[a]) ? logits[a] : -INFINITY;
          const float bv = std::isfinite(logits[b]) ? logits[b] : -INFINITY;
          return av == bv ? a < b : av > bv;
        });
    ids.resize(64);
    for (auto id : ids)
      values.push_back(logits[id]);
  }
};
auto Factory(bool old, const EogCostConfig& c) {
  return old ? EogCostOld(c) : EogCostNew(c);
}
double Measure(bool old, const EogCostConfig& c, const Row& row, unsigned width,
               bool candidates) {
  std::vector<std::unique_ptr<EogCostSampler>> samplers;
  for (unsigned i = 0; i < width; ++i)
    samplers.push_back(Factory(old, c));
  std::vector<float> pinned(c.vocab), host(c.vocab);
  std::vector<std::vector<float>> lane_rows(width, row.logits);
  std::vector<std::uint32_t> tokens(width);
  std::uint64_t sum = 0, rows = 0;
  const auto start = Clock::now();
  do {
    // StepDecodeBatch selects serially. Request::SelectNext accepts each
    // non-stop token before AdvanceBatch runs. No GPU delay is simulated.
    for (unsigned i = 0; i < width; ++i) {
      samplers[i]->Reset(c.state, 100 + rows + i);
      tokens[i] =
          samplers[i]->Row(lane_rows[i], row.values, row.ids, candidates);
      sum += tokens[i];
      if (std::ranges::find(c.eog, tokens[i]) == c.eog.end())
        samplers[i]->Commit(tokens[i]);
    }
    if (!candidates) {
      // CPU copy-size control for the next AR frontier. Actual D2H and HIP
      // synchronization are explicitly unmeasured in this CPU-only harness.
      for (unsigned i = 0; i < width; ++i) {
        if (std::ranges::find(c.eog, tokens[i]) != c.eog.end())
          continue;
        std::memcpy(pinned.data(), lane_rows[i].data(),
                    c.vocab * sizeof(float));
        std::copy(pinned.begin(), pinned.end(), host.begin());
        // The required host copies must survive dead-store elimination.
        asm volatile("" : : "r"(pinned.data()), "r"(host.data()) : "memory");
        sum += static_cast<std::uint64_t>(pinned[10] + 100);
      }
    }
    rows += width;
  } while (Clock::now() - start < std::chrono::milliseconds(timing_ms));
  const auto elapsed =
      std::chrono::duration<double, std::micro>(Clock::now() - start).count();
  sink = sum;
  return elapsed / rows;
}
void Bench(bool old, int top_k, bool candidates, bool fix = false) {
  const std::vector<unsigned> flags =
      fix ? std::vector<unsigned>{31}
          : std::vector<unsigned>{0, 1, 2, 4, 8, 16, 31, 30, 29, 27, 23, 15};
  std::cout << "kind,impl,path,top_k,width,state,rank,flags,repeat,a1,b1,b2,a2,"
               "diff,aa\n";
  for (unsigned width : {1U, 4U, 8U})
    for (unsigned state = 0; state < 4; ++state)
      for (unsigned rank = 0; rank < 3; ++rank) {
        if (focus && (width != 4 || state != 1 || rank != 2))
          continue;
        EogCostConfig c;
        c.top_k = top_k;
        c.temperature = candidates ? 1.F : 0.F;
        c.top_p = candidates ? .95F : 1.F;
        c.state = state;
        const Row row(c, rank, 42 + rank);
        for (auto mask : flags) {
          auto b = c;
          b.flags = mask;
          auto a = c;
          a.flags = fix ? 31 : 0;
          for (unsigned rep = 0; rep < 7; ++rep) {
            const double a1 =
                Measure(fix ? true : old, a, row, width, candidates);
            const double b1 =
                Measure(fix ? false : old, b, row, width, candidates);
            const double b2 =
                Measure(fix ? false : old, b, row, width, candidates);
            const double a2 =
                Measure(fix ? true : old, a, row, width, candidates);
            std::cout << "timing,"
                      << (fix   ? "fix"
                          : old ? "old"
                                : "new")
                      << ',' << (candidates ? "candidate" : "ar") << ','
                      << top_k << ',' << width << ',' << state << ',' << rank
                      << ',' << mask << ',' << rep << ',' << a1 << ',' << b1
                      << ',' << b2 << ',' << a2 << ','
                      << (b1 + b2 - a1 - a2) / 2 << ',' << a2 - a1 << '\n';
          }
          auto counter = Factory(old, b);
          counter->Reset(state, 42);
          counter->Row(row.logits, row.values, row.ids, candidates);
          for (const auto& line : counter->Counters())
            std::cerr << "COUNTER " << (old ? "old" : "new") << ' '
                      << (candidates ? "candidate" : "ar") << " k=" << top_k
                      << " width=" << width << " state=" << state
                      << " rank=" << rank << " flags=" << mask << ' ' << line
                      << '\n';
        }
      }
}
void Diff(unsigned count) {
  std::uint64_t compared = 0, distributions = 0;
  std::mt19937_64 random(0xe09c057);
  for (unsigned group = 0; compared < count; ++group) {
    EogCostConfig c;
    c.flags = group % 32;
    c.permissive_tools = group % 17 == 0;
    c.state = group / 32 % 6;
    c.temperature = group / 192 % 2 ? 0.F : 1.F;
    c.top_k = group / 384 % 3 == 0 ? 0 : (group / 384 % 3 == 1 ? 20 : 1);
    c.top_p = group % 9 == 0 ? 1.F : .95F;
    // Every row retains the production vocabulary size and control IDs.
    const unsigned width = group % 3 == 0 ? 1 : group % 3 == 1 ? 4 : 8;
    for (unsigned lane = 0; lane < width && compared < count; ++lane) {
      const Row row(c, group % 3, random(), true);
      auto old = Factory(true, c), fresh = Factory(false, c);
      const auto seed = random();
      old->Reset(c.state, seed);
      fresh->Reset(c.state, seed);
      if (lane == 0 && group % 7 == 0) {
        if (old->Distribution(row.logits) != fresh->Distribution(row.logits))
          throw std::runtime_error("distribution mismatch group=" +
                                   std::to_string(group));
        ++distributions;
      }
      const bool candidates = group % 2 != 0;
      for (unsigned step = 0; step < 2; ++step) {
        auto a = old->Row(row.logits, row.values, row.ids, candidates);
        auto b = fresh->Row(row.logits, row.values, row.ids, candidates);
        if (a != b || old->Snapshot() != fresh->Snapshot())
          throw std::runtime_error(
              "state/token mismatch group=" + std::to_string(group) +
              " step=" + std::to_string(step));
        old->Commit(a);
        fresh->Commit(b);
        if (old->Snapshot() != fresh->Snapshot())
          throw std::runtime_error("committed state mismatch group=" +
                                   std::to_string(group));
      }
      ++compared;
    }
    if (group % 256 == 0)
      std::cerr << "PROGRESS rows=" << compared << '\n';
  }
  std::cout << "DIFFERENTIAL rows=" << compared << " draws=" << compared * 2
            << " distributions=" << distributions
            << " mismatches=0 seed=0xe09c057\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string_view(argv[1]) == "--diff") {
      Diff(argc > 2 ? std::stoul(argv[2]) : 100000);
      return 0;
    }
    if (argc >= 5 && std::string_view(argv[1]) == "--bench") {
      if (argc >= 6)
        timing_ms = std::stoul(argv[5]);
      focus = argc >= 7 && std::string_view(argv[6]) == "focus";
      Bench(std::string_view(argv[2]) == "old", std::stoi(argv[3]),
            std::string_view(argv[4]) == "candidate",
            std::string_view(argv[2]) == "fix");
      return 0;
    }
    throw std::runtime_error(
        "usage: --diff [rows] | --bench old|new top_k ar|candidate");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
