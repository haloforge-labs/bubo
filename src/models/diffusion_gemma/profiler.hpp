#ifndef GUFO_MODELS_DIFFUSION_GEMMA_PROFILER_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_PROFILER_HPP_

#include <hip/hip_runtime_api.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::diffusion_gemma {

/// Per-phase GPU time from stream events, printed to stderr by Report().
/// Disabled profilers record nothing and cost nothing.
class Profiler {
public:
  Profiler(hipStream_t stream, bool enabled);
  ~Profiler();
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  /// Marks the start of `name`; the previous phase ends here.
  void Phase(const char* name);
  /// Prints the time of each phase since the last report under `label`.
  void Report(const char* label = "");
  [[nodiscard]] bool enabled() const { return enabled_; }

private:
  void Release();

  hipStream_t stream_;
  bool enabled_;
  const char* open_{nullptr};
  hipEvent_t last_{nullptr};
  std::map<std::string, std::vector<std::pair<hipEvent_t, hipEvent_t>>>
      pending_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_PROFILER_HPP_
