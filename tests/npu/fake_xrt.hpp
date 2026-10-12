// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_FAKE_XRT_HPP_
#define GUFO_NPU_FAKE_XRT_HPP_
#include <functional>
#include <string>
#include <vector>

namespace gufo::npu::fake {
struct Control {
  std::string fail_call;
  std::string power_mode = "Performance";
  int bad_group = -1;
  int state = 4;  // ERT_CMD_STATE_COMPLETED, asserted against pinned headers.
  bool null_map = false;
  bool wrong_size = false;
  bool corrupt_guard = false;
  bool corrupt_quarantine = false;
  bool runtime_mismatch = false;
  unsigned char expected_a = 0x17;
  unsigned char expected_b = 0;
  int devices = 0, allocations = 0, starts = 0, waits = 0, live_bos = 0;
  long long wait_ms = 0;
  std::vector<std::string> calls;
  std::function<void(const std::string&)> on_call;
};
extern Control control;
void Reset();
void Call(const std::string& name);
}  // namespace gufo::npu::fake
#endif
