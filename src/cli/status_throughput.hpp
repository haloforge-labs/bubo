#pragma once

#include <chrono>
#include <cmath>
#include <optional>
#include <string>

#include "src/cli/machine_output.hpp"

namespace bubo::cli {
// Counter deltas over monotonic wall time, shared by instance publication and
// watch streams. The owner serializes calls; a reset or new instance warms up.
class ThroughputBaseline {
public:
  void Apply(Value& snapshot, std::chrono::steady_clock::time_point now =
                                  std::chrono::steady_clock::now()) {
    auto rates = Value::object();
    rates["label"] = "aggregate wall throughput";
    rates["unit"] = "tokens_per_second";
    rates["interval_ms"] = Value();
    rates["prefill_tokens_per_second"] = Value();
    rates["decode_tokens_per_second"] = Value();
    const auto* identity = snapshot.find("identity");
    const auto id = identity ? identity->member_str("instance_id") : "";
    const auto* counters = snapshot.find("counters");
    const auto prefill = Counter(counters, "prefill_tokens_total");
    const auto decode = Counter(counters, "decode_tokens_total");
    const double elapsed =
        std::chrono::duration<double>(now - sampled_).count();
    if (!id.empty() && id == instance_id_ && prefill && decode && prefill_ &&
        decode_ && *prefill >= *prefill_ && *decode >= *decode_ &&
        elapsed > 0.0) {
      rates["interval_ms"] = elapsed * 1000.0;
      rates["prefill_tokens_per_second"] = (*prefill - *prefill_) / elapsed;
      rates["decode_tokens_per_second"] = (*decode - *decode_) / elapsed;
    }
    instance_id_ = id;
    sampled_ = now;
    prefill_ = prefill;
    decode_ = decode;
    snapshot["live_throughput"] = std::move(rates);
  }

private:
  static std::optional<double> Counter(const Value* counters,
                                       const std::string& name) {
    const auto* value = counters ? counters->find(name) : nullptr;
    if (!value || !value->is_number())
      return std::nullopt;
    const double number = value->as_double();
    if (!std::isfinite(number) || number < 0.0 || number > 9007199254740991.0 ||
        std::floor(number) != number)
      return std::nullopt;
    return number;
  }
  std::string instance_id_;
  std::chrono::steady_clock::time_point sampled_{};
  std::optional<double> prefill_;
  std::optional<double> decode_;
};
}  // namespace bubo::cli
