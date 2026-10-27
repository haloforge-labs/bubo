#ifndef GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_
#define GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace gufo::diagnostics {

struct HostUnameInfo {
  std::string sysname;
  std::string nodename;
  std::string release;
  std::string version;
  std::string machine;
};

struct HostCpuInfo {
  std::string model_name;
  std::string vendor_id;
  std::string architecture{"x86_64"};
  std::uint32_t logical_cores{0};
  std::uint32_t physical_cores{0};
  std::uint32_t numa_nodes{1};
};

struct HostMemInfo {
  std::uint64_t total_bytes{0};
  std::uint64_t available_bytes{0};
  std::uint64_t free_bytes{0};
  bool is_unified_memory{true};
};

struct SysfsGpuDevice {
  std::string card_name;
  std::string pci_id;
  std::string pci_slot;
  std::string driver;
  std::string vendor_id;
  std::string device_id;
  std::string power_mode;
  std::map<std::string, std::string> hwmon_sensors;
};

class LinuxSysfs {
public:
  explicit LinuxSysfs(const std::filesystem::path& sys_root = "/sys",
                      const std::filesystem::path& proc_root = "/proc");

  [[nodiscard]] std::optional<HostUnameInfo> QueryUname() const;
  [[nodiscard]] std::optional<HostCpuInfo> QueryCpuInfo() const;
  [[nodiscard]] std::optional<HostMemInfo> QueryMemInfo() const;
  [[nodiscard]] bool IsNativeHost() const { return native_host_; }
  [[nodiscard]] std::vector<SysfsGpuDevice> QueryDrmGpuDevices() const;

  [[nodiscard]] std::optional<std::string> ReadFile(
      const std::filesystem::path& relative_or_absolute_path) const;

private:
  std::filesystem::path sys_root_;
  std::filesystem::path proc_root_;
  bool native_host_ = false;
};

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_
