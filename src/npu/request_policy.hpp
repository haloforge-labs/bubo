// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_REQUEST_POLICY_HPP_
#define GUFO_NPU_REQUEST_POLICY_HPP_
#include <stdexcept>
#include <string>

#include "staging.hpp"
namespace gufo::npu {
// OFF callers retain their exact historical identity bytes. ON marks every
// lineage approximate even if this prompt was too short to offload. Artifact
// acceptance pins the binaries; revision identifies the host
// packing/arithmetic.
inline std::string CacheSuffix(const Knobs& k) {
  return "\nprecision=npu-bfp8-bf16-v1\nartifact=4096x2560x1024-pinned-v1\n"
         "linear_slices=" +
         std::to_string(k.slices) +
         "\ngate_slices=" + std::to_string(k.gate_slices) +
         "\nshared_expert=" + std::to_string(k.shared_expert) + "\n";
}
}  // namespace gufo::npu
#endif
