// SPDX-License-Identifier: MIT
// Fake XRT implementation instrumented for adapter validation.
// Implements the SAME upstream declarations, never replacement XRT layouts.
#include "fake_xrt.hpp"

#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <system_error>

#include "src/npu/private_bo_adapter.hpp"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace gufo::npu::fake {
Control control;
void Reset() {
  control = Control{};
}
void Call(const std::string& name) {
  if (control.on_call)
    control.on_call(name);
  control.calls.push_back(name);
  if (control.fail_call == name)
    throw std::system_error(73, std::generic_category(), "planted " + name);
}
void Check(bool valid, const char* text) {
  if (!valid)
    throw std::runtime_error(text);
}
}  // namespace gufo::npu::fake

namespace gufo::npu {
void EnsureXrtRuntime() {
  fake::Call("runtime.verify");
  fake::Check(!fake::control.runtime_mismatch, "fake runtime version mismatch");
}
}  // namespace gufo::npu

namespace xrt_core {
class device {};
}  // namespace xrt_core
namespace xrt {
using gufo::npu::fake::Call;
using gufo::npu::fake::Check;
using gufo::npu::fake::control;
class xclbin_impl {};
class hw_context_impl {};
class kernel_impl {};
class bo_impl {
public:
  std::vector<std::byte> data;
  bool synced = false;
  int index;
  bo_impl(size_t n, int index) : data(n), index(index) { ++control.live_bos; }
  ~bo_impl() { --control.live_bos; }
};
class run_impl {
public:
  std::map<int, bo> buffers;
  std::map<int, std::uint64_t> scalars;
  bool started = false;
};

device::device(unsigned index) {
  Call("device.create");
  Check(index == 0, "device index");
  ++control.devices;
  handle = std::make_shared<xrt_core::device>();
}
device::~device() {
  Call("device.destroy");
}
uuid device::register_xclbin(const xclbin&) {
  Call("device.register_xclbin");
  return {};
}
std::any device::get_info_std(info::device param,
                              const xrt::detail::abi&) const {
  Call("device.get_info.platform");
  Check(param == info::device::platform, "read-only platform query");
  return std::string(R"({"platforms":[{"status":{"power_mode":")") +
         control.power_mode + R"("}}]})";
}
xclbin::xclbin(const std::string&) {
  Call("xclbin.parse");
  handle = std::make_shared<xclbin_impl>();
}
hw_context::hw_context(const device&, const uuid&, access_mode) {
  Call("hw_context.create");
  handle = std::make_shared<hw_context_impl>();
}
hw_context::~hw_context() {
  Call("hw_context.destroy");
}
kernel::kernel(const hw_context&, const std::string& name) {
  Call("kernel.create");
  Check(name == "MLIR_AIE", "kernel name");
  handle = std::make_shared<kernel_impl>();
}
kernel::~kernel() {
  Call("kernel.destroy");
}
int kernel::group_id(int arg) const {
  Call("kernel.group_id." + std::to_string(arg));
  Check(arg == 1 || (arg >= 3 && arg <= 7), "group slot");
  return arg == control.bad_group ? 99 : arg == 1 ? 1 : 0;
}
bo::bo(const device&, size_t n, flags flags, memory_group group) {
  const int index = control.allocations;
  Call("bo.create." + std::to_string(index));
  Check(flags == (index == 0 ? bo::flags::cacheable : bo::flags::host_only),
        "private BO flags");
  Check(group == (index == 0 ? 1 : 0), "BO group");
  ++control.allocations;
  handle = std::make_shared<bo_impl>(n, index);
}
bo::~bo() = default;
size_t bo::size() const {
  Call("bo.size." + std::to_string(handle->index));
  return handle->data.size() + (control.wrong_size ? 1 : 0);
}
void* bo::map() {
  Call("bo.map." + std::to_string(handle->index));
  return control.null_map ? nullptr : handle->data.data();
}
void bo::sync(xclBOSyncDirection direction, size_t n, size_t offset) {
  Call("bo.sync." + std::to_string(handle->index) +
       (direction == XCL_BO_SYNC_BO_TO_DEVICE ? ".TO_DEVICE" : ".FROM_DEVICE"));
  Check(offset + n <= handle->data.size(), "BO sync range");
  if (direction == XCL_BO_SYNC_BO_TO_DEVICE)
    handle->synced = true;
}
run::run(const kernel&) {
  Call("run.create");
  handle = std::make_shared<run_impl>();
}
run::~run() {
  Call("run.destroy");
}
void run::set_arg_at_index(int index, const bo& buffer) {
  Call("run.set_arg." + std::to_string(index));
  handle->buffers[index] = buffer;
}
void run::set_arg_at_index(int index, const void* p, size_t n) {
  Call("run.set_arg." + std::to_string(index));
  Check(n == (index == 0 ? 8u : 4u), "XML scalar ABI width");
  std::uint64_t value = 0;
  std::memcpy(&value, p, n);
  handle->scalars[index] = value;
}
void run::start() {
  Call("run.start");
  Check(handle->buffers.at(3).get_handle()->data.front() ==
            std::byte{control.expected_a},
        "private input must not alias caller's storage");
  Check(handle->buffers.at(4).get_handle()->data.front() ==
            std::byte{control.expected_b},
        "B must match the current input");
  Check(handle->scalars.at(0) == 3 && handle->scalars.at(2) == 5508,
        "opcode / real instruction word count");
  for (int index : {1, 3, 4, 6, 7})
    Check(handle->buffers.at(index).get_handle()->synced,
          "TO_DEVICE sync missing (including instructions)");
  ++control.starts;
  handle->started = true;
}
ert_cmd_state run::wait(const std::chrono::milliseconds& ms) const {
  Call("run.wait");
  static_assert(ERT_CMD_STATE_COMPLETED == 4);
  Check(ms.count() > 0 && ms.count() <= 8000 && handle->started,
        "bounded wait contract");
  ++control.waits;
  control.wait_ms = ms.count();
  if (control.state != ERT_CMD_STATE_COMPLETED)
    return static_cast<ert_cmd_state>(control.state);
  auto& c = handle->buffers.at(5).get_handle()->data;
  std::memset(c.data(), 0x42, gufo::npu::kCBytes);
  if (control.corrupt_guard)
    c.back() = std::byte{0};
  if (control.corrupt_quarantine)
    handle->buffers.at(7).get_handle()->data.front() = std::byte{1};
  return ERT_CMD_STATE_COMPLETED;
}
}  // namespace xrt
