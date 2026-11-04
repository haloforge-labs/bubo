#include "expert_bench_policy.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/decode_iq_routing.hpp"

namespace b = expert_bench;

void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

template<class F>
void Throws(F call) {
  bool caught = false;
  try {
    call();
  } catch (const std::runtime_error&) {
    caught = true;
  }
  Require(caught, "invalid input accepted");
}

int main() try {
  {
    namespace r = gufo::models::qwen38_flash_next::rocm;
    const auto path = std::filesystem::temp_directory_path() /
                      "bubo-expert-routing-policy-test.txt";
    r::DecodeIqRoutingCapture capture(path.string());
    std::vector<std::int32_t> ids(20);
    for (int i = 0; i < 20; ++i)
      ids[i] = i % 10;
    std::string error;
    Require(!capture.Wants(0, 0) && !capture.Wants(0, 9),
            "invalid routing width");
    Require(!capture.Append(0, 2, std::span(ids).first(19), &error),
            "short routing capture accepted");
    for (unsigned i = 0; i < capture.kMaxSamplesPerLayerWidth; ++i)
      Require(capture.Append(3, 2, ids, &error),
              "routing capture write failed");
    Require(!capture.Wants(3, 2) && capture.Wants(4, 2),
            "routing cap is not per layer");
    Require(capture.Append(3, 2, ids, &error), "capped capture failed");
    std::ifstream input(path);
    std::string line;
    unsigned lines = 0;
    while (std::getline(input, line)) {
      Require(line == "3 2 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9",
              "routing replay format changed");
      ++lines;
    }
    Require(lines == capture.kMaxSamplesPerLayerWidth, "routing cap exceeded");
    input.close();
    std::filesystem::remove(path);
    r::DecodeIqRoutingCapture bad((path / "absent" / "ids.txt").string());
    Require(!bad.Append(0, 2, ids, &error) && !error.empty(),
            "routing file failure hidden");
  }
  Require(b::kEvictBytes >= 8 * b::kCacheBytes,
          "insufficient eviction traffic");
  std::vector<std::string> observed;
  const auto run = [&] {
    return b::Measure(
        3, [&] { observed.push_back("evict"); },
        [&] { observed.push_back("start"); },
        [&](int i) {
          observed.push_back("call" + std::to_string(i));
          return 0;
        },
        [&] { observed.push_back("stop"); },
        [&] {
          observed.push_back("elapsed");
          return 4.0;
        });
  };
  const auto baseline = run();
  const auto candidate = run();
  const std::vector<std::string> expected{
      "evict",   "start",   "call0",   "stop",    "elapsed", "evict",
      "start",   "call1",   "stop",    "elapsed", "evict",   "start",
      "call2",   "stop",    "elapsed", "evict",   "start",   "call0",
      "stop",    "elapsed", "evict",   "start",   "call1",   "stop",
      "elapsed", "evict",   "start",   "call2",   "stop",    "elapsed"};
  Require(observed == expected, "sample or variant boundary bypasses eviction");
  Require(baseline.us == 4 && candidate.us == 4, "mean or units changed");
  const int distinct[]{10, 20, 30};
  Require(b::MeanDistinct(distinct, 2) == 15, "untimed sets counted");
  Require(b::MeanDistinct(distinct, 4) == 17.5, "partial cycle miscounted");
  Require(b::MeanDistinct(distinct, 6) == 20, "full cycles miscounted");
  Throws([&] { b::MeanDistinct(distinct, 0); });
  Throws([] { b::MeanDistinct({}, 1); });
  b::RequireComparable(baseline, candidate);
  Throws(
      [&] { b::RequireComparable(baseline, {4, "legacy", b::kEvictBytes}); });
  Throws([&] { b::RequireComparable(baseline, {4, b::kProtocol, 0}); });
  Require(b::Valid(1000000, baseline), "250 GB/s boundary rejected");
  Require(!b::Valid(1000001, baseline), "above ceiling accepted");
  for (double us : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()}) {
    Require(!b::Valid(1000000, {us}), "invalid time accepted");
    const auto failed = b::Measure(
        1, [] {}, [] {}, [](int) { return 0; }, [] {}, [&] { return us; });
    Require(failed.reason != "ok" && failed.us == 0 && failed.completed == 0,
            "invalid interval became a valid partial mean");
  }
  for (double bytes : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    Require(!b::Valid(bytes, baseline), "invalid byte count accepted");
  Throws([] {
    b::Measure(0, [] {}, [] {}, [](int) { return 0; }, [] {}, [] { return 1; });
  });
  const int cells[]{0, 1, 2, 3, 4, 5, 6};
  std::vector<b::Timing> rows;
  std::vector<int> order;
  b::Sweep(
      cells,
      [&](int cell) {
        if (cell == 5)
          throw b::Failure("cell_preflight", 700, 0, 700, 700);
        int sample = -1;
        return b::Measure(
            3, [] {}, [] {},
            [&](int i) {
              sample = i;
              return cell == 4 && i == 1 ? -3 : 0;
            },
            [] {},
            [&] {
              if (sample == 1) {
                if (cell == 1)
                  return 0.0;
                if (cell == 2)
                  return -2.0;
                if (cell == 3)
                  return std::numeric_limits<double>::quiet_NaN();
              }
              return 4.0;
            });
      },
      [&](int cell, const b::Timing& timing) {
        order.push_back(cell);
        rows.push_back(timing);
      });
  Require(order == std::vector<int>({0, 1, 2, 3, 4, 5, 6}),
          "bad cell aborted or reordered the sweep");
  Require(rows.size() == 7 && b::Valid(1000000, rows.front()) &&
              b::Valid(1000000, rows.back()),
          "valid later cell suppressed");
  for (int i = 1; i <= 4; ++i)
    Require(!b::Valid(1000000, rows[i]) && rows[i].completed == 1 &&
                rows[i].sample == 1 && rows[i].us == 0,
            "partial cell accepted or failure sample lost");
  Require(
      rows[1].raw_us == 0 && rows[2].raw_us == -2 && std::isnan(rows[3].raw_us),
      "raw bad interval lost");
  Require(rows[1].start_recorded && rows[1].stop_recorded,
          "recorded event status lost");
  Require(rows[4].call_rc == -3 && rows[4].reason == "launch_failure" &&
              !rows[4].stop_recorded,
          "launch failure status lost");
  Require(rows[5].hip_error == 700 && rows[5].sync_error == 700 &&
              rows[5].last_error == 700,
          "HIP completion status lost");
  const auto eviction = b::Measure(
      1, [] { throw b::Failure("eviction_launch", 7, 0, 8, 9); }, [] {},
      [](int) { return 0; }, [] {}, [] { return 1; });
  Require(eviction.reason == "eviction_launch" && eviction.hip_error == 7 &&
              eviction.sync_error == 8 && eviction.last_error == 9 &&
              !eviction.start_recorded,
          "eviction status lost");
  Require(b::ExitCode(true, 5) == 2 && b::ExitCode(false, 5) == 0 &&
              b::ExitCode(true, 0) == 0,
          "strict exit status changed");
  std::cout << "EXPERT BENCH POLICY PASS\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "EXPERT BENCH POLICY FAIL: " << e.what() << '\n';
  return 1;
}
