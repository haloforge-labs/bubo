// SPDX-License-Identifier: MIT
#include "server_prefill.hpp"

#include "request_stats.hpp"
namespace gufo::npu {
void ServerPrefill::BeginChunk() noexcept {
  chunk_used_ = false;
  reset_invalid_ = true;
  Stats().last_layer.store(-1, std::memory_order_relaxed);
}
bool ServerPrefill::Start() {
  ready_ = session_.Start().healthy;
  return ready_;
}
void ServerPrefill::Reject(std::string error) {
  ready_ = false;
  session_.RejectResult(std::move(error));
}
void ServerPrefill::Hold(std::string error) noexcept {
  try {
    Reject(std::move(error));
  } catch (...) {
  }
}
ServerPrefill::~ServerPrefill() {
  session_.Shutdown();
}
}  // namespace gufo::npu
