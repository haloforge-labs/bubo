#ifndef GUFO_MODELS_DIFFUSION_GEMMA_HIP_CHECK_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_HIP_CHECK_HPP_

#include <hip/hip_runtime_api.h>

#include <stdexcept>
#include <string>

namespace gufo::models::diffusion_gemma {

/// Throws std::runtime_error("DiffusionGemma <what>: <HIP error>") on
/// failure.
inline void HipCheck(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("DiffusionGemma ") + what + ": " +
                             hipGetErrorString(status));
}

/// Checks the last kernel launch.
inline void CheckLaunch(const char* kernel) {
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    throw std::runtime_error(std::string("DiffusionGemma kernel ") + kernel +
                             ": " + hipGetErrorString(status));
}

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_HIP_CHECK_HPP_
