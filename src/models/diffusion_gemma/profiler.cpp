#include "src/models/diffusion_gemma/profiler.hpp"

#include <cstdio>
#include <set>

#include "src/models/diffusion_gemma/hip_check.hpp"

namespace gufo::models::diffusion_gemma {

Profiler::Profiler(hipStream_t stream, bool enabled)
    : stream_(stream), enabled_(enabled) {}

Profiler::~Profiler() {
  Release();
}

void Profiler::Phase(const char* name) {
  if (!enabled_)
    return;
  hipEvent_t event{};
  HipCheck(hipEventCreate(&event), "profile event");
  HipCheck(hipEventRecord(event, stream_), "profile event");
  if (open_ != nullptr)
    pending_[open_].emplace_back(last_, event);
  last_ = event;
  open_ = name;
}

void Profiler::Report(const char* label) {
  if (!enabled_)
    return;
  std::fprintf(stderr, "profile --- %s\n", label);
  Phase(nullptr);
  open_ = nullptr;
  HipCheck(hipStreamSynchronize(stream_), "profile sync");
  double total = 0;
  std::map<std::string, double> sums;
  for (auto& [name, list] : pending_) {
    for (auto& [a, b] : list) {
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      sums[name] += ms;
      total += ms;
    }
  }
  for (const auto& [name, ms] : sums)
    std::fprintf(stderr, "profile %-14s %10.1f ms %5.1f%%\n", name.c_str(), ms,
                 100.0 * ms / total);
  Release();
}

void Profiler::Release() {
  // Consecutive phases share their boundary event: destroy each once.
  std::set<hipEvent_t> events;
  for (auto& [name, list] : pending_)
    for (auto& [a, b] : list) {
      events.insert(a);
      events.insert(b);
    }
  if (last_ != nullptr)
    events.insert(last_);
  for (auto event : events)
    (void)hipEventDestroy(event);
  pending_.clear();
  last_ = nullptr;
}

}  // namespace gufo::models::diffusion_gemma
