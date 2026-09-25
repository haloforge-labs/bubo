#include "src/core/diagnostics/linux_sysfs.h"

#include <sys/utsname.h>
#ifdef _WIN32
#include <cpuid.h>
#include <unistd.h>

#include <cstring>
#endif

#include <charconv>
#include <fstream>
#include <set>
#include <sstream>
#include <string_view>

namespace gufo::diagnostics {

namespace {

std::string Trim(std::string_view str) {
  const auto start = str.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) {
    return {};
  }
  const auto end = str.find_last_not_of(" \t\r\n");
  return std::string(str.substr(start, end - start + 1));
}

std::uint32_t ParseUint(std::string_view str) {
  std::uint32_t val = 0;
  const auto* const begin = str.data();
  const auto* const finish = begin + str.size();
  std::from_chars(begin, finish, val);
  return val;
}

}  // namespace

LinuxSysfs::LinuxSysfs(const std::filesystem::path& sys_root,
                       const std::filesystem::path& proc_root)
    : sys_root_(std::filesystem::absolute(sys_root)),
      proc_root_(std::filesystem::absolute(proc_root)) {}

std::optional<std::string> LinuxSysfs::ReadFile(
    const std::filesystem::path& path) const {
  std::filesystem::path target = path;
  if (!std::filesystem::exists(target)) {
    target = sys_root_ / path;
  }
  std::ifstream file(target);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return Trim(buffer.str());
}

std::optional<HostUnameInfo> LinuxSysfs::QueryUname() const {
  const auto version_opt = ReadFile(proc_root_ / "version");
  struct utsname uts{};
  if (uname(&uts) == 0) {
    HostUnameInfo info;
    info.sysname = uts.sysname;
    info.nodename = uts.nodename;
    info.release = uts.release;
    info.version = version_opt ? *version_opt : uts.version;
    info.machine = uts.machine;
    return info;
  }
  return std::nullopt;
}

#ifdef _WIN32
namespace {
// Windows has no /proc: CPUID supplies the names, the OS the counts.
HostCpuInfo WindowsCpuInfo() {
  HostCpuInfo cpu;
  unsigned regs[4] = {};
  if (__get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3])) {
    char vendor[13] = {};
    std::memcpy(vendor, &regs[1], 4);
    std::memcpy(vendor + 4, &regs[3], 4);
    std::memcpy(vendor + 8, &regs[2], 4);
    cpu.vendor_id = vendor;
  }
  char brand[49] = {};
  for (unsigned leaf = 0; leaf < 3; ++leaf) {
    if (!__get_cpuid(0x80000002U + leaf, &regs[0], &regs[1], &regs[2],
                     &regs[3]))
      break;
    std::memcpy(brand + leaf * 16, regs, 16);
  }
  cpu.model_name = Trim(brand);
  const long logical = sysconf(_SC_NPROCESSORS_ONLN);
  cpu.logical_cores = logical > 0 ? static_cast<std::uint32_t>(logical) : 0;
  cpu.physical_cores = cpu.logical_cores / 2;
  return cpu;
}
}  // namespace
#endif

std::optional<HostCpuInfo> LinuxSysfs::QueryCpuInfo() const {
  const auto content_opt = ReadFile(proc_root_ / "cpuinfo");
  if (!content_opt) {
#ifdef _WIN32
    if (proc_root_ == "/proc")
      return WindowsCpuInfo();
#endif
    return std::nullopt;
  }

  HostCpuInfo cpu_info;
  std::istringstream stream(*content_opt);
  std::string line;
  std::uint32_t processor_count = 0;
  std::set<std::string> physical_ids;

  while (std::getline(stream, line)) {
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, colon_pos));
    const std::string value = Trim(line.substr(colon_pos + 1));

    if (key == "model name" && cpu_info.model_name.empty()) {
      cpu_info.model_name = value;
    } else if (key == "vendor_id" && cpu_info.vendor_id.empty()) {
      cpu_info.vendor_id = value;
    } else if (key == "processor") {
      ++processor_count;
    } else if (key == "siblings" && cpu_info.logical_cores == 0) {
      cpu_info.logical_cores = ParseUint(value);
    } else if (key == "cpu cores" && cpu_info.physical_cores == 0) {
      cpu_info.physical_cores = ParseUint(value);
    } else if (key == "physical id") {
      physical_ids.insert(value);
    }
  }

  if (cpu_info.logical_cores == 0) {
    cpu_info.logical_cores = processor_count;
  }
  if (cpu_info.physical_cores == 0) {
    cpu_info.physical_cores = processor_count > 0 ? (processor_count / 2) : 0;
  }
  cpu_info.numa_nodes = physical_ids.empty()
                            ? 1
                            : static_cast<std::uint32_t>(physical_ids.size());
  return cpu_info;
}

std::optional<HostMemInfo> LinuxSysfs::QueryMemInfo() const {
  const auto content_opt = ReadFile(proc_root_ / "meminfo");
  if (!content_opt) {
#ifdef _WIN32
    if (proc_root_ == "/proc") {
      HostMemInfo windows;
      windows.total_bytes = gufo_total_physical_bytes();
      windows.available_bytes = gufo_available_physical_bytes();
      windows.free_bytes = windows.available_bytes;
      return windows;
    }
#endif
    return std::nullopt;
  }

  HostMemInfo mem_info;
  std::istringstream stream(*content_opt);
  std::string line;

  while (std::getline(stream, line)) {
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, colon_pos));
    const std::string rest = Trim(line.substr(colon_pos + 1));

    std::uint64_t val_kb = 0;
    std::istringstream val_stream(rest);
    if (val_stream >> val_kb) {
      if (key == "MemTotal") {
        mem_info.total_bytes = val_kb * 1024ULL;
      } else if (key == "MemAvailable") {
        mem_info.available_bytes = val_kb * 1024ULL;
      } else if (key == "MemFree") {
        mem_info.free_bytes = val_kb * 1024ULL;
      }
    }
  }
  return mem_info;
}

std::vector<SysfsGpuDevice> LinuxSysfs::QueryDrmGpuDevices() const {
  std::vector<SysfsGpuDevice> devices;
  const auto drm_path = sys_root_ / "class" / "drm";

  std::error_code ec;
  if (!std::filesystem::exists(drm_path, ec)) {
    return devices;
  }

  for (const auto& entry : std::filesystem::directory_iterator(drm_path, ec)) {
    if (ec) {
      break;
    }
    const std::string filename = entry.path().filename().string();
    if (!filename.starts_with("card") ||
        filename.find('-') != std::string::npos) {
      continue;
    }

    const auto dev_dir = entry.path() / "device";
    if (!std::filesystem::exists(dev_dir, ec)) {
      continue;
    }

    SysfsGpuDevice gpu;
    gpu.card_name = filename;
    gpu.vendor_id = ReadFile(dev_dir / "vendor").value_or("");
    gpu.device_id = ReadFile(dev_dir / "device").value_or("");

    const auto uevent_content = ReadFile(dev_dir / "uevent");
    if (uevent_content) {
      std::istringstream stream(*uevent_content);
      std::string line;
      while (std::getline(stream, line)) {
        if (line.starts_with("DRIVER=")) {
          gpu.driver = line.substr(7);
        } else if (line.starts_with("PCI_ID=")) {
          gpu.pci_id = line.substr(7);
        } else if (line.starts_with("PCI_SLOT_NAME=")) {
          gpu.pci_slot = line.substr(14);
        }
      }
    }

    gpu.power_mode = ReadFile(dev_dir / "power_dpm_force_performance_level")
                         .value_or("unavailable (unreadable or not exposed)");

    const auto hwmon_base = dev_dir / "hwmon";
    if (std::filesystem::exists(hwmon_base, ec)) {
      for (const auto& hw_entry :
           std::filesystem::directory_iterator(hwmon_base, ec)) {
        if (ec) {
          break;
        }
        const auto temp1_input = ReadFile(hw_entry.path() / "temp1_input");
        if (temp1_input) {
          const std::uint32_t millicelsius = ParseUint(*temp1_input);
          if (millicelsius > 0) {
            const double temp_c = static_cast<double>(millicelsius) / 1000.0;
            gpu.hwmon_sensors["edge_temperature_c"] =
                std::to_string(temp_c).substr(0, 5);
          }
        }
      }
    }

    devices.push_back(std::move(gpu));
  }

  return devices;
}

}  // namespace gufo::diagnostics
